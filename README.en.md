# AMD ROCm Video Super-Resolution Suite (FlashVSR + SeedVR2) — by lundao1yu0

Making the two hottest AI video upscaling plugins for ComfyUI actually usable
on AMD GPUs (ROCm): flicker fixed, install streamlined, and the first public
benchmarks on a Radeon RX 7900 XT.

## About lundao1yu0

lundao1yu0 is a channel focused on real-machine testing of AI video/image
workflows: run it on real hardware, let the data speak, and document every
pitfall along the way. This project is the complete deliverable of our
AMD ROCm adaptation series, with every fix verified on a Radeon RX 7900 XT.

📮 **Business inquiries: [lundao1yu0@outlook.com](mailto:lundao1yu0@outlook.com)**

## What's in this repo

An umbrella **custom node that works by direct clone** — the patched sources of
both plugins are vendored inside (`nodes/flashvsr/`, `nodes/seedvr2_videoupscaler/`),
so there is nothing to install or patch, and no ROCm setup is required
(NVIDIA users get the LQ-cache fix by cloning too):

- **FlashVSR (flagship)** — 3 model-level fixes bringing AMD quality from
  "unusable" to "production-accepted"
- **SeedVR2** — verified on ROCm, fully Chinese-localized node interface
- `install_rocm.sh` optional helper (installs verified workflows / `--models`
  pre-downloads weights / prints ROCm tuning notes)
- Two ready-to-run verified ComfyUI workflows (`workflows/`)
- `patches/` keeps standalone patch scripts for users who prefer patching
  their own upstream clones

## Core fixes (FlashVSR)

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | The LCSA sparse-attention mask is **silently dropped** on AMD (upstream node only applies it on its sageattention branch, which never runs on ROCm) | Brightness pumping / flicker | Ported the LCSA mask with a dedicated ROCm gather attention backend (`rocm_attention.py`), validated at PSNR 78.8 dB against the upstream reference implementation |
| 2 | The LQ-condition module `Buffer_LQ4x_Proj` **overwrites its causal cache before using it** (**affects ALL platforms**, NVIDIA included — this explains part of upstream issue #10) | Facial elements jump at every 4-frame boundary | Restored the upstream `Causal_LQ4x_Proj` use-then-store order (`fix_lq_cache_order.py`) |
| 3 | One GPU→CPU sync (`.item()`) per attention layer per block | Needless pipeline stalls | Fully async (static max_c) |

The evidence chain for fix #2 (periodicity detection via autocorrelation →
upstream source diff → single-point fix → re-measurement) is documented in
`docs/BENCHMARK.md`. **Fix #2 benefits every hardware platform — NVIDIA users
are welcome to test it.**

## Benchmarks (Radeon RX 7900 XT, 20GB)

| Metric | Value |
|--------|-------|
| 544→1088 ×2 (Tiny Fast, 24fps) | ~25 s per second of footage (~0.96 fps) |
| Full render of a 10s clip | ~4.2 min |
| Quality cost of the fixes | None — the old "faster" behavior was running with bugs |

**ROCm version advice: stay on 6.4.x + a torch rocm7.0 wheel.** We fully
upgraded to ROCm 7.2.4 + torch 2.12.1 and measured: no gain on RDNA3 consumer
cards, the full stack is actually 9% slower (ROCm 7.x optimization focuses on
CDNA data-center GPUs). Details in `docs/BENCHMARK.md`.

## Install

### Option A · Direct clone (recommended, any platform)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/caokou/ComfyUI-FlashVSR-SeedVR2-AMD-ROCm.git
# Restart ComfyUI. Dependencies auto-install from requirements.txt.
# Categories: lundao1yu0/⚡FlashVSR and lundao1yu0/SEEDVR2
```

- NVIDIA / no-ROCm users: cloning gives you all fixes (the LQ-cache fix is
  platform-independent; the ROCm attention backend is gated by
  `torch.version.hip` and never touches the CUDA path)
- First run downloads `FlashVSR1_1.safetensors` (~5.3GB); SeedVR2 weights
  auto-download via the node itself. Set `HF_ENDPOINT=https://hf-mirror.com`
  if GitHub/HuggingFace direct access is slow.

### Option B · Optional helper script

```bash
bash ComfyUI-FlashVSR-SeedVR2-AMD-ROCm/install_rocm.sh           # install 2 verified workflows + ROCm notes
bash ComfyUI-FlashVSR-SeedVR2-AMD-ROCm/install_rocm.sh --models  # also pre-download FlashVSR weights
```

It no longer clones upstream or patches anything (sources are vendored and
already fixed). Users maintaining their own upstream clones can apply
`patches/` standalone.

## ROCm hard rules (read this first)

1. **Never disable tiling** — on gfx1100, running out of VRAM does not raise
   an OOM exception; it **freezes the whole machine** (SSH dies; physical
   reboot required).
2. **tile_size 256 recommended** — larger tiles are slower: attention cost
   grows super-linearly with sequence length (384 measured 52% slower than 256).
3. **`aten::index` advanced indexing (3-D index × 4-D tensor) hard-crashes on
   gfx1100** (HSAIL exception) — our code uses `index_select` everywhere;
   keep this in mind when writing custom nodes.
4. flex_attention on torch 2.10 rocm **silently treats hand-built BlockMasks
   as dense**; revisit after upgrading torch.

## Workflows

- `workflows/ui_flashvsr_e2.json` — FlashVSR ×2 (acceptance-verified config)
- `workflows/ui_seedvr2_e2.json` — SeedVR2 3B Q4 →1088

Both verified to run directly from the ComfyUI UI on the reference machine.

## Acknowledgements & License

This repo vendors patched snapshots of two upstream plugins (provenance and
the full list of changes relative to upstream: [NOTICE.md](NOTICE.md)):

- Upstream model: [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR) (ROCm work inspired by PR #114, Apache-2.0)
- Upstream node: [1038lab/ComfyUI-FlashVSR](https://github.com/1038lab/ComfyUI-FlashVSR) (GPL-3.0 → this repo is GPL-3.0 overall)
- Upstream node: [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (Apache-2.0)

All fixes are being contributed back upstream (the FlashVSR LQ-cache bug is
filed as [issue #20](https://github.com/1038lab/ComfyUI-FlashVSR/issues/20)
with the full evidence chain).

---

中文版：[README.md](README.md)
