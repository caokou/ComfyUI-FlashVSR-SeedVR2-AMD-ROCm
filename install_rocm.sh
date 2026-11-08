#!/bin/bash
# lundao1yu0 ROCm VSR suite — one-shot installer (run inside ComfyUI/custom_nodes)
# Clones upstream nodes, applies all model-level + localization patches, and
# installs the two verified workflows. Idempotent: safe to re-run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

echo "== lundao1yu0 ROCm VSR suite installer =="

# ---- 1) upstream nodes ------------------------------------------------------
if [ ! -d flashvsr ]; then
  git clone https://github.com/1038lab/ComfyUI-FlashVSR flashvsr
else
  echo "[skip] flashvsr already present"
fi
if [ ! -d seedvr2_videoupscaler ]; then
  git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler seedvr2_videoupscaler
else
  echo "[skip] seedvr2_videoupscaler already present"
fi

# ---- 2) model-level patches (FlashVSR) --------------------------------------
echo "== applying model-level patches =="
cp "$HERE/patches/rocm_attention.py" /tmp/rocm_attention.py
# apply_rocm_lcsa_patch.py expects rocm_attention.py next to itself
( cd "$HERE/patches" && "$PY" apply_rocm_lcsa_patch.py )
"$PY" "$HERE/patches/fix_lq_cache_order.py"

# ---- 3) branding + Chinese localization -------------------------------------
echo "== applying branding + Chinese localization =="
for f in flashvsr/AILab_FlashVSR.py \
         seedvr2_videoupscaler/src/interfaces/video_upscaler.py \
         seedvr2_videoupscaler/src/interfaces/dit_model_loader.py \
         seedvr2_videoupscaler/src/interfaces/vae_model_loader.py; do
  # category branding: lundao1yu0/...
  sed -i.bak 's|CATEGORY = "🧪AILab/⚡FlashVSR"|CATEGORY = "lundao1yu0/⚡FlashVSR"|' "$f" 2>/dev/null || true
  sed -i.bak 's|category="SEEDVR2",|category="lundao1yu0/SEEDVR2",|' "$f" 2>/dev/null || true
  rm -f "$f.bak"
done
"$PY" "$HERE/patches/zh_comments_patch.py" || true
"$PY" "$HERE/patches/zh_node_desc_patch.py" || true
"$PY" "$HERE/patches/zh_full_tooltips_patch.py" || true
"$PY" "$HERE/patches/zh_cleanup_patch.py" || true

# ---- 4) verified workflows ---------------------------------------------------
WF_DIR="../user/default/workflows"
mkdir -p "$WF_DIR"
cp "$HERE/workflows/ui_flashvsr_e2.json" "$HERE/workflows/ui_seedvr2_e2.json" "$WF_DIR/"
echo "[ ok ] workflows installed -> $WF_DIR"

echo "== done. Restart ComfyUI, then open workflows 'ui_flashvsr_e2' / 'ui_seedvr2_e2'. =="
