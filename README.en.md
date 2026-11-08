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

An umbrella repo with AMD ROCm adaptations of two ComfyUI plugins:

- **FlashVSR (flagship)** — 3 model-level fixes bringing AMD quality from
  "unusable" to "production-accepted"
- **SeedVR2** — verified on ROCm, fully Chinese-localized node interface
- `install_rocm.sh` one-shot installer (clones upstream nodes → applies all
  patches automatically)
- Two ready-to-run verified ComfyUI workflows (`workflows/`)

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

```bash
# Prerequisites: ComfyUI (custom_nodes/ present) + ROCm 6.x + a torch rocm wheel
cd ComfyUI/custom_nodes
bash /path/to/install_rocm.sh
```

The installer: clones both upstream nodes → applies all model-level patches →
applies the Chinese-localization patches → drops the two verified workflows
into `user/default/workflows/`. All patches are idempotent.

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

- Upstream model: [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR) (ROCm work inspired by PR #114)
- Upstream node: [1038lab/ComfyUI-FlashVSR](https://github.com/1038lab/ComfyUI-FlashVSR) (GPL-3.0)
- Upstream node: [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

This repo continues the upstream GPL-3.0 license; all fixes are provided as
patches and are being contributed back upstream. The Chinese-localization
patches (`zh_*.py`) are new content added by this repo.

---

中文版：[README.md](README.md)
