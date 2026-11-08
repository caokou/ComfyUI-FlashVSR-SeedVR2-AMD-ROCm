#!/usr/bin/env python3
"""FlashVSR ROCm LCSA patcher — run ON THE SERVER (python3, stdlib only).

Ports the draft block-sparse attention (LCSA) mask to ROCm, mirroring
upstream OpenImagingLab/FlashVSR dispatch semantics (mask branch is the
FIRST priority) and adapting upstream PR #114's ROCm work:

  1) copies rocm_attention.py (gather/reference/flex backends) next to
     wan_video_dit.py in the live node;
  2) patches flash_attention(): on HIP, when a draft block mask is present,
     route to rocm_attention.sparse_attention() instead of silently running
     dense SDPA (the node only applied the mask on the sparse_sageattn
     branch, which never exists on ROCm);
  3) ports PR #114's grouped HIP path in generate_draft_block_mask() that
     bounds temporary memory of the block-score einsum at high resolutions.

Default behavior on CUDA is 100% unchanged (branch is HIP-gated).
Idempotent; --dry prints anchor matches only. .orig backups preserved.
"""
import os
import py_compile
import shutil
import sys

NODE = os.path.expanduser("~/comfyui-rocm/custom_nodes/flashvsr")
DIT = os.path.join(NODE, "FlashVSR", "models", "wan_video_dit.py")
ROCM_ATTN = os.path.join(NODE, "FlashVSR", "models", "rocm_attention.py")
HERE = os.path.dirname(os.path.abspath(__file__))
DRY = "--dry" in sys.argv


def backup(path):
    if not os.path.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")
        return "backup created"
    return "backup exists"


# ---- 1) dispatch branch in flash_attention (HIP-gated, first priority) ----
DISPATCH_ANCHOR = (
    'def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attention_mask=None, return_KV=False, enable_sageattention=True):\n'
    '    if attention_mask is not None and enable_sageattention and SPARSE_SAGE_AVAILABLE:\n'
)
DISPATCH_NEW = (
    'def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attention_mask=None, return_KV=False, enable_sageattention=True):\n'
    '    if attention_mask is not None and torch.version.hip:\n'
    '        # ROCm LCSA port (mirrors upstream dispatch priority): apply the\n'
    '        # draft block mask instead of silently dropping it. The node mask\n'
    '        # is 64-token interleaved (sparse_sageattn API); de-interleave to\n'
    '        # the 128-token granularity it was computed at.\n'
    '        if not hasattr(flash_attention, "_rocm_sparse"):\n'
    '            import importlib.util as _ilu, os as _os\n'
    '            _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "rocm_attention.py")\n'
    '            _sp = _ilu.spec_from_file_location("flashvsr_rocm_attention", _p)\n'
    '            _m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)\n'
    '            flash_attention._rocm_sparse = _m.sparse_attention\n'
    '        _q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)\n'
    '        _k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)\n'
    '        _v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)\n'
    '        _m2 = attention_mask[..., ::2] if attention_mask.shape[-1] == 2 * ((_k.shape[2] + 127) // 128) else attention_mask\n'
    '        _x = flash_attention._rocm_sparse(_q, _k, _v, _m2)\n'
    '        x = rearrange(_x, "b n s d -> b s (n d)", n=num_heads)\n'
    '        return x\n'
    '    if attention_mask is not None and enable_sageattention and SPARSE_SAGE_AVAILABLE:\n'
)

# ---- 2) grouped HIP path in generate_draft_block_mask (PR #114) -----------
GROUP_ANCHOR = (
    '    q_heads = avgpool_q.permute(1, 0, 2)\n'
    '    k_heads = avgpool_k.permute(1, 0, 2)\n'
    '    D = avgpool_q.shape[-1]\n'
)
GROUP_NEW = (
    '    q_heads = avgpool_q.permute(1, 0, 2)\n'
    '    k_heads = avgpool_k.permute(1, 0, 2)\n'
    '    D = avgpool_q.shape[-1]\n'
    '    if nheads > 1 and (torch.version.hip and q_heads.shape[1] * k_heads.shape[1] > 1000000):\n'
    '        # PR #114: bound temporary memory of the block-score einsum on HIP\n'
    '        masks = []\n'
    '        group_size = q_w.shape[0] // seqlen\n'
    '        for head in range(nheads):\n'
    '            groups = []\n'
    '            for group in range(seqlen):\n'
    '                groups.append(generate_draft_block_mask(\n'
    '                    batch_size, 1, 1,\n'
    '                    q_w[group*group_size:(group+1)*group_size, :, head*D:(head+1)*D],\n'
    '                    k_w[..., head*D:(head+1)*D],\n'
    '                    topk=topk, local_attn_mask=local_attn_mask,\n'
    '                ))\n'
    '            masks.append(torch.cat(groups, dim=2))\n'
    '        return torch.cat(masks, dim=1)\n'
)


def patch(path, anchor, replacement, label):
    src = open(path, encoding="utf-8").read()
    if replacement.strip() in src:
        return f"[skip] {label}: already applied"
    n = src.count(anchor)
    if n != 1:
        return f"[FAIL] {label}: anchor matched {n} times (need exactly 1) — NOT patched"
    if DRY:
        return f"[dry ] {label}: anchor ok"
    backup(path)
    open(path, "w", encoding="utf-8").write(src.replace(anchor, replacement))
    return f"[ ok ] {label}"


def main():
    print(f"== FlashVSR ROCm LCSA patcher {'(DRY)' if DRY else ''} ==")
    if not os.path.exists(DIT):
        print(f"[FAIL] wan_video_dit.py not found at {DIT}")
        sys.exit(1)
    print(patch(DIT, DISPATCH_ANCHOR, DISPATCH_NEW, "HIP mask dispatch branch"))
    print(patch(DIT, GROUP_ANCHOR, GROUP_NEW, "grouped HIP mask generation"))
    if os.path.exists(ROCM_ATTN) and not DRY:
        print("[skip] rocm_attention.py: already present")
    else:
        if not DRY:
            shutil.copy2(os.path.join(HERE, "rocm_attention.py"), ROCM_ATTN)
            print(f"[ ok ] rocm_attention.py -> {ROCM_ATTN}")
        else:
            print("[dry ] rocm_attention.py would be copied")
    if not DRY:
        py_compile.compile(ROCM_ATTN, doraise=True)
        py_compile.compile(DIT, doraise=True)
        print("py_compile: OK")
    print("\nRunbook:")
    print("  1. restart 8189 (no old crutch env; FLASHVSR_ROCM_ATTENTION defaults to gather)")
    print("  2. rerun 10s workflow -> montage + face-curve verification")


if __name__ == "__main__":
    main()
