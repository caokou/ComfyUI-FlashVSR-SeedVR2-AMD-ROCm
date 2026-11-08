#!/usr/bin/env python3
"""Chinese-localize node-level descriptions (hover on node title). Run ON SERVER."""
import py_compile
import sys

B = "/home/chenjiaxing/comfyui-rocm/custom_nodes/seedvr2_videoupscaler/src/interfaces/"
EDITS = [
    (B + "video_upscaler.py",
     '''"SeedVR2 main upscaling node: processes video frames using DiT and VAE models with diffusion-based enhancement. "
                "Handles RGB/RGBA formats, maintains temporal consistency across frames, applies color correction, "
                "and manages VRAM through intelligent tensor offloading. \\n\\n"
                "Requires DiT and VAE model configurations."''',
     '''"SeedVR2 主超分节点：使用 DiT + VAE 做扩散增强式视频超分。"
                "支持 RGB/RGBA，保持帧间时序一致性，自动颜色校正，并通过智能张量卸载管理显存。\\n\\n"
                "需先连接 DiT 与 VAE 模型配置（lundao1yu0 频道 AMD/ROCm 适配版）。"'''),
    (B + "dit_model_loader.py",
     '''"Load and configure SeedVR2 DiT (Diffusion Transformer) model for video upscaling. "
                "Supports BlockSwap memory optimization for low VRAM systems, model caching for batch processing, "
                "multi-GPU offloading, and torch.compile acceleration. \\n\\n"
                "Connect to Video Upscaler node."''',
     '''"加载并配置 SeedVR2 的 DiT（扩散 Transformer）超分模型。"
                "支持 BlockSwap 低显存优化、批量处理时的模型缓存、多卡卸载与 torch.compile 加速。\\n\\n"
                "连接到 Video Upscaler 节点。本机已就绪：3B Q4 量化版（20GB 显存实测可用）。"'''),
    (B + "vae_model_loader.py",
     '''"Supports tiled processing to handle high resolutions on limited VRAM, model caching, "
                "multi-GPU offloading, and torch.compile acceleration. \\n\\n"
                "Connect to Video Upscaler node."''',
     '''"支持分块处理（低显存跑高分辨率）、模型缓存、多卡卸载与 torch.compile 加速。\\n\\n"
                "连接到 Video Upscaler 节点。本机已就绪：ema_vae_fp16。"'''),
    (B + "torch_compile_settings.py",
     '''"Configure SeedVR2 torch.compile optimization for 20-40% DiT speedup and 15-25% VAE speedup. "
                "Trades longer first-run compilation time for faster inference.\\n\\n"
                "Connect to DiT and/or VAE model loaders. Requires PyTorch 2.0+ and Triton for inductor backend."''',
     '''"配置 SeedVR2 的 torch.compile 优化：DiT 提速约 20-40%，VAE 提速约 15-25%。"
                "代价是首次运行需要较长的编译时间。\\n\\n"
                "连接到 DiT / VAE 加载器。需要 PyTorch 2.0+，inductor 后端还需 Triton（AMD/ROCm 上实验性，建议默认关闭）。"'''),
]


def main():
    dry = "--dry" in sys.argv
    ok = True
    for path, old, new in EDITS:
        src = open(path, encoding="utf-8").read()
        if new in src:
            print(f"[skip] {path.rsplit('/',1)[1]}")
            continue
        n = src.count(old)
        if n != 1:
            print(f"[FAIL] {path.rsplit('/',1)[1]}: anchor x{n}")
            ok = False
            continue
        print(f"[{'dry ' if dry else ' ok '}] {path.rsplit('/',1)[1]}")
        if not dry:
            open(path, "w", encoding="utf-8").write(src.replace(old, new))
            py_compile.compile(path, doraise=True)
            print("   wrote+compiled")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
