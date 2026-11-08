#!/usr/bin/env python3
"""Remove machine-specific notes from tooltips; keep usage-only wording.
Run ON THE SERVER."""
import py_compile
import sys

F = "/home/chenjiaxing/comfyui-rocm/custom_nodes/flashvsr/AILab_FlashVSR.py"
D = "/home/chenjiaxing/comfyui-rocm/custom_nodes/seedvr2_videoupscaler/src/interfaces/dit_model_loader.py"
V = "/home/chenjiaxing/comfyui-rocm/custom_nodes/seedvr2_videoupscaler/src/interfaces/vae_model_loader.py"

EDITS = [
    # FlashVSR: drop measured-results phrasing, keep usage guidance
    (F,
     "每个分块的宽高（超分前）。块越大显存越高；实测 7900 XT (20GB) 上 256 最优——块越大注意力开销超线性增长，反而更慢。",
     "每个分块的宽高（超分前）。块越大显存占用越高；块过大时因注意力开销超线性增长反而变慢，建议 256-512。"),
    (F,
     "分块（tile）处理视频。更慢但大幅省显存。AMD/ROCm 上务必保持开启（关闭会导致显存溢出甚至系统冻结）。",
     "分块（tile）处理视频。更慢但大幅节省显存；显存紧张或 AMD/ROCm 环境请保持开启。"),
    (F,
     "计算精度。现代 GPU 上 BF16 通常更快；实测 7900 XT 上两者速度相同。",
     "计算精度。现代 GPU 上 BF16 通常更快，FP16 为备选。"),
    # SeedVR2 DiT: drop "本机已就绪" notes
    (D,
     "用于视频超分的 DiT（扩散 Transformer）模型。首次使用自动下载。\\n\"\n                        \"本机已就绪：seedvr2_ema_3b-Q4_K_M.gguf（3B Q4 量化，20GB 显存实测可用）。",
     "用于视频超分的 DiT（扩散 Transformer）模型。首次使用自动下载；也可手动放入 ComfyUI models 目录。"),
    (D,
     "连接到 Video Upscaler 节点。本机已就绪：3B Q4 量化版（20GB 显存实测可用）。",
     "连接到 Video Upscaler 节点。"),
    # SeedVR2 VAE: drop readiness notes
    (V,
     "支持分块处理（低显存跑高分辨率）、模型缓存、多卡卸载与 torch.compile 加速。\\n\\n\"\n                \"连接到 Video Upscaler 节点。本机已就绪：ema_vae_fp16。",
     "支持分块处理（低显存跑高分辨率）、模型缓存、多卡卸载与 torch.compile 加速。\\n\\n\"\n                \"连接到 Video Upscaler 节点。"),
    (V,
     "用于编/解码的 VAE（变分自编码器）模型。首次使用自动下载。\\n\"\n                        \"本机已就绪：ema_vae_fp16.safetensors。",
     "用于编/解码的 VAE（变分自编码器）模型。首次使用自动下载；也可手动放入 ComfyUI models 目录。"),
]


def main():
    dry = "--dry" in sys.argv
    ok = True
    files = {}
    for path, old, new in EDITS:
        src = files.get(path) or open(path, encoding="utf-8").read()
        n = src.count(old)
        if n != 1:
            print(f"[FAIL] {path.rsplit('/',1)[1]}: anchor x{n}: {old[:40]!r}")
            ok = False
            continue
        files[path] = src.replace(old, new)
        print(f"[{'dry ' if dry else ' ok '}] {path.rsplit('/',1)[1]}: {old[:28]!r}")
    if not ok or dry:
        sys.exit(0 if ok else 1)
    for path, src in files.items():
        open(path, "w", encoding="utf-8").write(src)
        py_compile.compile(path, doraise=True)
        print("wrote+compiled:", path.rsplit('/',1)[1])


if __name__ == "__main__":
    main()
