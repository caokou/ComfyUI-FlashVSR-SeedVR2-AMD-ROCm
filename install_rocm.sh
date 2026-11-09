#!/bin/bash
# ComfyUI-FlashVSR-SeedVR2-AMD-ROCm — optional post-clone helper.
#
# The repo works as a custom node by simply cloning it into
# ComfyUI/custom_nodes/ (nodes are vendored & patched already; deps are
# auto-installed from requirements.txt on ComfyUI start). This script is
# NOT required for that. It only:
#   1) copies the two verified workflows into user/default/workflows
#   2) (--models) downloads FlashVSR weights (honors HF_ENDPOINT mirror)
#   3) prints ROCm (gfx1100) tuning notes
# Idempotent: safe to re-run. Repo: https://github.com/caokou/ComfyUI-FlashVSR-SeedVR2-AMD-ROCm
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
WANT_MODELS=0
[ "${1:-}" = "--models" ] && WANT_MODELS=1

echo "== lundao1yu0 · FlashVSR + SeedVR2 AMD ROCm suite — post-clone helper =="

# ---- 1) verified workflows ----------------------------------------------------
WF_DIR="$HERE/../user/default/workflows"
mkdir -p "$WF_DIR"
cp "$HERE/workflows/ui_flashvsr_e2.json" "$HERE/workflows/ui_seedvr2_e2.json" "$WF_DIR/"
echo "[ ok ] workflows installed -> $WF_DIR (ui_flashvsr_e2 / ui_seedvr2_e2)"

# ---- 2) optional: FlashVSR weights -------------------------------------------
# SeedVR2 weights auto-download on first use (its own download_weight util).
if [ "$WANT_MODELS" = 1 ]; then
  echo "== downloading FlashVSR weights (HF_ENDPOINT honored, e.g. https://hf-mirror.com) =="
  "$PY" - <<'EOF'
import os
from huggingface_hub import hf_hub_download
import folder_paths  # ComfyUI runtime; run this script inside the ComfyUI env if missing
dst = os.path.join(folder_paths.models_dir, "FlashVSR")
os.makedirs(dst, exist_ok=True)
for repo, fn in [("JunhaoZhuang/FlashVSR", "FlashVSR1_1.safetensors")]:
    p = hf_hub_download(repo_id=repo, filename=fn, local_dir=dst)
    print("[ ok ]", p)
EOF
else
  echo "[skip] weights: re-run with '--models' to fetch FlashVSR1_1.safetensors (~5.3GB)"
fi

# ---- 3) ROCm tuning notes ------------------------------------------------------
echo "== ROCm (gfx1100 / RX 7900 XT) notes =="
cat <<'EOF'
  * torch: use the rocm wheel matching your driver (tested: 2.10.0+rocm7.0).
  * NEVER disable spatial tiling on 20GB RDNA3 cards — VRAM exhaustion freezes
    the whole machine instead of raising OOM (see docs/BENCHMARK.md).
  * tile 256 is the measured sweet spot (384 is 52% slower; attention is
    super-linear in tile size).
  * TunableOp showed no benefit here (first-run tuning cost); leave it off.
  * Benchmarks: ROCm 6.4.1 + torch 2.10 = 25.8 s per source-second (544->1088 x2).
EOF

echo "== done. Restart ComfyUI if it was running during the copy. =="
