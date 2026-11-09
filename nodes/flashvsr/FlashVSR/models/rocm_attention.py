"""ROCm execution of the official 128-token LCSA block mask.

Port for the ComfyUI FlashVSR node (gfx1100 / torch 2.10.0+rocm7.0).

Background: upstream FlashVSR applies the draft block-sparse attention mask
(block_sparse_attn_func, NVIDIA-only) as the FIRST dispatch branch, so
self-attention always runs sparse. The ComfyUI node only applies the mask on
its sparse_sageattn branch, so on ROCm (and on NVIDIA without sageattention)
self-attention silently degrades to DENSE attention - a train-test mismatch
that upstream issue #61 and node issue #10 link to flicker/jerky artifacts.

Why not flex_attention here: on torch 2.10+rocm, a hand-built BlockMask via
BlockMask.from_kv_blocks is silently ignored by the compiled flex kernel
(verified: diagonal mask == dense output), and create_block_mask evaluates
mask_mod on the token grid (OOM at real resolutions). Upstream PR #114 works
around this on newer torch; this file instead provides an exact vectorized
gather implementation on top of SDPA:

  gather    (default) per q-block: gather allowed kv blocks, batched SDPA.
            Semantically identical to the FP32 loop reference in PR #114.
  reference the PR's b/h/row loop reference (slow, diagnostic gold standard).
  flex      PR #114's BlockMask.from_kv_blocks path (kept for newer torch;
            WARNING: on torch 2.10 rocm this was measured to silently degrade
            to dense attention).

Env:
  FLASHVSR_ROCM_ATTENTION   gather | reference | flex | drop (ablation)  (default gather)
  FLASHVSR_ROCM_PRECISION   native | fp32               (default native)
  FLASHVSR_ROCM_GATHER_CHUNK  q-blocks per SDPA batch    (default 8)
"""

import importlib.util
import os
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_BLOCK = 128
_announced = [False]
_ACC = [0.0, 0]  # profiling accumulator: total seconds, call count


def _log_once(msg: str):
    if not _announced[0]:
        _announced[0] = True
        print(f"[FlashVSR-ROCm] {msg}")


def reference_attention(q, k, v, mask):
    """Gather all allowed keys per query block before a single softmax.

    Verbatim from upstream PR #114 (diffsynth/models/rocm_attention.py):
    the slow FP32 loop used to validate faster backends.
    """
    out = torch.zeros_like(q)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for row in range(mask.shape[2]):
                blocks = mask[b, h, row].nonzero().flatten()
                if blocks.numel() == 0:
                    continue
                indices = (
                    blocks[:, None] * 128 + torch.arange(128, device=q.device)
                ).flatten()
                indices = indices[indices < k.shape[2]]
                start, end = row * 128, min((row + 1) * 128, q.shape[2])
                out[b, h, start:end] = F.scaled_dot_product_attention(
                    q[b, h, start:end].float(),
                    k[b, h, indices].float(),
                    v[b, h, indices].float(),
                    dropout_p=0.0,
                ).to(q.dtype)
    return out


def gather_attention(q, k, v, mask, chunk=None):
    """Vectorized equivalent of reference_attention.

    q (B,H,Lq,D), k/v (B,H,Lk,D), mask (B,H,QB,KB) bool at 128-token block
    granularity. For each q block, gather its allowed kv blocks (padded to the
    row-wise max) and run one batched SDPA; rows with no allowed keys return
    zeros, matching the PR reference exactly.

    Implementation note: torch advanced indexing (aten::index) with a 3-D
    index tensor crashes the HIP kernel on gfx1100 / torch 2.10 rocm7.0
    (verified: int64/int32 indices, contiguous or not, bf16/fp32 all fail);
    index_select on a flattened view is used instead.
    """
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    QB, KB = mask.shape[2], mask.shape[3]
    assert QB * _BLOCK == Lq and KB * _BLOCK == Lk, "exact block multiples expected"
    if chunk is None:
        chunk = max(1, int(os.environ.get("FLASHVSR_ROCM_GATHER_CHUNK", "8")))
    _prof = os.environ.get("FLASHVSR_PROFILE") == "1"
    _t0 = time.perf_counter() if _prof else 0.0

    counts = mask.sum(-1)  # (B,H,QB)
    order = torch.argsort(
        mask.to(torch.int8), dim=-1, descending=True, stable=True
    )  # allowed blocks first, ascending among allowed
    # Keep max_c static (no .item()): a GPU->CPU sync per call serialized the
    # whole pipeline 4590x per render (measured 19.1ms/call dominated by the
    # sync, not compute — swapping the math for fused SDPA changed nothing).
    # Fast preset runs near-dense (counts ~ KB), so gathering all KB blocks
    # costs little; validity is enforced by the attn mask regardless.
    max_c = KB
    sel = order[..., :max_c].contiguous()  # (B,H,QB,max_c)
    valid = (
        torch.arange(max_c, device=q.device) < counts[..., None]
    )  # (B,H,QB,max_c)
    empty = counts == 0  # (B,H,QB)

    BH = B * H
    kf = k.reshape(BH * KB, _BLOCK * D)
    vf = v.reshape(BH * KB, _BLOCK * D)
    qb = q.reshape(BH, QB, _BLOCK, D)
    sel_f = sel.view(BH, QB, max_c)
    valid_f = valid.view(BH, QB, max_c)
    empty_f = empty.view(BH, QB)
    base = (torch.arange(BH, device=q.device) * KB).view(BH, 1, 1)
    out = torch.empty(BH, QB, _BLOCK, D, device=q.device, dtype=q.dtype)

    for s in range(0, QB, chunk):
        e = min(s + chunk, QB)
        qc = e - s
        n = BH * qc
        idx = (sel_f[:, s:e] + base).reshape(-1)  # flat row indices
        k_g = torch.index_select(kf, 0, idx).view(n, max_c * _BLOCK, D)
        v_g = torch.index_select(vf, 0, idx).view(n, max_c * _BLOCK, D)
        m = (
            valid_f[:, s:e]
            .repeat_interleave(_BLOCK, dim=-1)
            .reshape(n, 1, 1, max_c * _BLOCK)
        )
        # rows with zero allowed keys would produce NaN softmax; let them
        # attend to an arbitrary block, then zero the output below.
        m = m | empty_f[:, s:e].reshape(n, 1, 1, 1)
        # manual attention math: SDPA's mask broadcast semantics proved
        # fragile on this build (gfx1100 / torch 2.10 rocm); plain
        # matmul + masked softmax is unambiguous and matches the PR
        # reference numerics (bf16 GEMMs accumulate in fp32; softmax in fp32).
        q4 = qb[:, s:e].reshape(n, 1, _BLOCK, D)
        k4 = k_g.view(n, 1, max_c * _BLOCK, D)
        v4 = v_g.view(n, 1, max_c * _BLOCK, D)
        att = torch.matmul(q4, k4.transpose(-1, -2)) * (D ** -0.5)
        att = att.float().masked_fill(~m, float("-inf")).softmax(dim=-1)
        o = torch.matmul(att.to(q.dtype), v4).view(n, _BLOCK, D)
        out[:, s:e] = o.view(BH, qc, _BLOCK, D)

    out = torch.where((~empty_f)[..., None, None], out, torch.zeros_like(out))
    if _prof:
        _ACC[0] += time.perf_counter() - _t0
        _ACC[1] += 1
        if _ACC[1] % 100 == 0:
            print(
                f"[PROF] gather_attn calls={_ACC[1]} total={_ACC[0]:.1f}s "
                f"avg={_ACC[0] / _ACC[1] * 1000:.1f}ms",
                flush=True,
            )
    return out.view(B, H, Lq, D)


def gather_attention_sdpa(q, k, v, mask, chunk=None):
    """SDPA-backend variant of gather_attention: same gather/masking scheme,
    but the attention math runs in one fused SDPA call per chunk instead of
    explicit matmul+fp32-softmax (which materializes an (n,1,Lq,W) fp32
    attn tensor — a bandwidth killer). ROCm's memory-efficient SDPA kernel
    supports boolean attn_mask with fp32 accumulation.
    """
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    QB, KB = mask.shape[2], mask.shape[3]
    assert QB * _BLOCK == Lq and KB * _BLOCK == Lk, "exact block multiples expected"
    if chunk is None:
        chunk = max(1, int(os.environ.get("FLASHVSR_ROCM_GATHER_CHUNK", "8")))
    _prof = os.environ.get("FLASHVSR_PROFILE") == "1"
    _t0 = time.perf_counter() if _prof else 0.0

    counts = mask.sum(-1)
    order = torch.argsort(
        mask.to(torch.int8), dim=-1, descending=True, stable=True
    )
    max_c = max(int(counts.max().item()), 1)
    sel = order[..., :max_c].contiguous()
    valid = torch.arange(max_c, device=q.device) < counts[..., None]
    empty = counts == 0

    BH = B * H
    kf = k.reshape(BH * KB, _BLOCK * D)
    vf = v.reshape(BH * KB, _BLOCK * D)
    qb = q.reshape(BH, QB, _BLOCK, D)
    sel_f = sel.view(BH, QB, max_c)
    valid_f = valid.view(BH, QB, max_c)
    empty_f = empty.view(BH, QB)
    base = (torch.arange(BH, device=q.device) * KB).view(BH, 1, 1)
    out = torch.empty(BH, QB, _BLOCK, D, device=q.device, dtype=q.dtype)

    for s in range(0, QB, chunk):
        e = min(s + chunk, QB)
        qc = e - s
        n = BH * qc
        idx = (sel_f[:, s:e] + base).reshape(-1)
        k_g = torch.index_select(kf, 0, idx).view(n, 1, max_c * _BLOCK, D)
        v_g = torch.index_select(vf, 0, idx).view(n, 1, max_c * _BLOCK, D)
        m = (
            valid_f[:, s:e]
            .repeat_interleave(_BLOCK, dim=-1)
            .reshape(n, 1, 1, max_c * _BLOCK)
        )
        m = m | empty_f[:, s:e].reshape(n, 1, 1, 1)
        q4 = qb[:, s:e].reshape(n, 1, _BLOCK, D)
        o = F.scaled_dot_product_attention(q4, k_g, v_g, attn_mask=m)
        out[:, s:e] = o.view(BH, qc, _BLOCK, D)

    out = torch.where((~empty_f)[..., None, None], out, torch.zeros_like(out))
    if _prof:
        _ACC[0] += time.perf_counter() - _t0
        _ACC[1] += 1
        if _ACC[1] % 100 == 0:
            print(
                f"[PROF] gather_attn calls={_ACC[1]} total={_ACC[0]:.1f}s "
                f"avg={_ACC[0] / _ACC[1] * 1000:.1f}ms",
                flush=True,
            )
    return out.view(B, H, Lq, D)


def flex_attention_masked(q, k, v, mask):
    """PR #114 path. On torch 2.10 rocm this silently degrades to dense
    attention (measured); kept for newer torch builds where it is the fastest
    option (PSNR 46.62 dB vs reference on the PR author's build)."""
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

    counts = mask.sum(-1).to(torch.int32)
    indices = torch.argsort(
        mask.to(torch.int32), dim=-1, descending=True, stable=True
    ).to(torch.int32)
    block_mask = BlockMask.from_kv_blocks(
        torch.zeros_like(counts),
        indices,
        counts,
        indices,
        BLOCK_SIZE=128,
        seq_lengths=(q.shape[2], k.shape[2]),
    )
    out = flex_attention(q, k, v, block_mask=block_mask)
    valid = (counts > 0).repeat_interleave(128, dim=-1)[..., : q.shape[2]]
    return torch.where(valid[..., None], out, 0)


def sparse_attention(q, k, v, mask):
    """Entry point used by wan_video_dit.flash_attention on HIP.

    Expects a boolean block mask of shape (B, H, ceil(Lq/128), ceil(Lk/128))
    at 128-token granularity (caller de-interleaves the node's 64-gran mask).
    """
    expected = (
        q.shape[0],
        q.shape[1],
        (q.shape[2] + _BLOCK - 1) // _BLOCK,
        (k.shape[2] + _BLOCK - 1) // _BLOCK,
    )
    if mask.shape != expected or mask.dtype != torch.bool or mask.device != q.device:
        raise ValueError(f"Expected boolean block mask {expected} on {q.device}")
    backend = os.environ.get("FLASHVSR_ROCM_ATTENTION", "gather")
    if backend == "reference":
        _log_once("LCSA reference (FP32 loop) backend active")
        return reference_attention(q, k, v, mask)
    if backend == "drop":
        # ablation only: identical code path and math, but the LCSA mask is
        # replaced by all-True (v1 semantics = mask silently dropped).
        _log_once("LCSA mask DROPPED (ablation; same math, dense mask)")
        return gather_attention(q, k, v, torch.ones_like(mask))
    if backend == "flex":
        _log_once("LCSA flex_attention backend active (CAUTION: see module docstring)")
        fn = flex_attention_masked
    elif backend == "gather":
        _log_once("LCSA gather+SDPA backend active")
        fn = gather_attention
    elif backend == "sdpa":
        _log_once("LCSA gather + fused SDPA backend active")
        fn = gather_attention_sdpa
    else:
        raise ValueError("FLASHVSR_ROCM_ATTENTION must be gather, reference or flex")

    precision = os.environ.get("FLASHVSR_ROCM_PRECISION", "native")
    if precision not in ("native", "fp32"):
        raise ValueError("FLASHVSR_ROCM_PRECISION must be native or fp32")
    if precision == "fp32":
        return fn(q.float(), k.float(), v.float(), mask).to(q.dtype)
    return fn(q, k, v, mask)
