#!/usr/bin/env python3
"""Add Chinese annotations (docstrings + tooltips) to node interfaces.
Run ON THE SERVER. Replaces English tooltips with Chinese ones in:
  - flashvsr/AILab_FlashVSR.py            (2 nodes)
  - seedvr2 interfaces/*.py               (4 nodes)
Tooltips are user-visible in ComfyUI on hover. Repeats are replaced globally.
"""
import py_compile
import sys

EDITS = []


def add(path, old, new):
    EDITS.append((path, old, new))


F = "/home/chenjiaxing/comfyui-rocm/custom_nodes/flashvsr/AILab_FlashVSR.py"
S = "/home/chenjiaxing/comfyui-rocm/custom_nodes/seedvr2_videoupscaler/src/interfaces/video_upscaler.py"
D = "/home/chenjiaxing/comfyui-rocm/custom_nodes/seedvr2_videoupscaler/src/interfaces/dit_model_loader.py"
V = "/home/chenjiaxing/comfyui-rocm/custom_nodes/seedvr2_videoupscaler/src/interfaces/vae_model_loader.py"

# ---- FlashVSR（两节点通用 tooltip）----
add(F, '"tooltip": "The low-resolution video frames to be upscaled."',
    '"tooltip": "待超分的低分辨率视频帧序列。"')
add(F, "'Long Video (Low VRAM)' is best for long videos as it uses less VRAM. 'High Quality' uses the 'Full' model.\"",
    "'Long Video (Low VRAM)' 显存占用最低，适合长视频；'High Quality' 使用完整版模型，质量最佳。\"")
add(F, '"tooltip": "The upscaling factor. 4x is recommended for best quality."',
    '"tooltip": "放大倍率。推荐 4x 以获得最佳质量。"')
add(F, '"tooltip": "Unload the main model from VRAM before decoding frames with the VAE. Saves VRAM but is slightly slower."',
    '"tooltip": "VAE 解码前先把主模型卸载出显存。省显存但略慢。"')
add(F, '"tooltip": "The random seed for the diffusion process."',
    '"tooltip": "扩散过程的随机种子，同种子+同输入可复现。"')

# ---- FlashVSR Advanced 专属 ----
add(F, '"tooltip": "Tiny: Fast, good quality. Tiny Long: Slower, uses much less VRAM for long videos. Full: Best quality, highest VRAM."',
    '"tooltip": "模型档位：Tiny(Fast)=快速、质量良好（日常推荐）；Tiny Long=长视频省显存；Full=完整模型、质量最高但显存占用最大。"')
add(F, '"tooltip": "Process the video in tiles. Slower, but saves a large amount of VRAM."',
    '"tooltip": "分块（tile）处理视频。更慢但大幅省显存。AMD/ROCm 上务必保持开启（关闭会导致显存溢出甚至系统冻结）。"')
add(F, '"tooltip": "The height/width of each tile (before upscaling). Larger tiles are faster but use more VRAM. Recommended: 384-512 for HD videos."',
    '"tooltip": "每个分块的宽高（超分前）。块越大显存越高；实测 7900 XT (20GB) 上 256 最优——块越大注意力开销超线性增长，反而更慢。"')
add(F, '"tooltip": "The amount of overlap between tiles to prevent visible seams. Recommended: 48-64 for smooth blending. (Must be < Tile Size / 2)"',
    '"tooltip": "相邻分块的重叠像素，用于消除拼接缝。需小于块尺寸的一半。"')
add(F, '"tooltip": "Higher value (e.g., 2.0) is faster but may capture less fine detail. Lower (e.g., 1.5) is slower but more detailed."',
    '"tooltip": "速度优化系数（= LCSA 稀疏注意力的 topk 比例）。2.0 更快；1.5 更慢但细节更丰富。"')
add(F, '"tooltip": "Increases quality and detail preservation at the cost of more VRAM. (Higher = Better Quality)"',
    '"tooltip": "质量增强系数。越大细节保留越好，显存占用越高。"')
add(F, '"tooltip": "Controls temporal stability. 11 is more stable (default). 9 can be sharper but may have less temporal consistency."',
    '"tooltip": "时序稳定性等级。11=更稳定（默认）；9=更锐利但时序一致性略差。"')
add(F, '"tooltip": "Applies a color correction step to prevent potential color shifts from the original video."',
    '"tooltip": "颜色校正：防止输出相对原视频出现色偏。"')
add(F, '"tooltip": "Enables tiling for the VAE decoding step. Further reduces VRAM usage, especially for high-resolution output."',
    '"tooltip": "VAE 解码分块。进一步降低显存占用，高分辨率输出时尤其有效。"')
add(F, '"tooltip": "Enable SageAttention optimization for faster inference (~20-30% speedup). Disable if encountering compatibility issues."',
    '"tooltip": "SageAttention 加速（NVIDIA 专用，约 20-30%）。AMD/ROCm 请保持 disable——本移植已内置 ROCm 专用 LCSA 注意力后端。"')
add(F, '"tooltip": "Select the processing device. \'auto\' will try to pick the best one available (CUDA > MPS > CPU)."',
    '"tooltip": "计算设备。auto 自动选择可用设备。"')
add(F, '"tooltip": "Processing precision. BF16 is often faster on modern GPUs. FP16 is a good alternative."',
    '"tooltip": "计算精度。现代 GPU 上 BF16 通常更快；实测 7900 XT 上两者速度相同。"')

# ---- 类 docstring ----
add(F, '''
class AILab_FlashVSR:
    @classmethod
    def INPUT_TYPES(cls):''',
    '''
class AILab_FlashVSR:
    """FlashVSR 快速视频超分（简版节点）— lundao1yu0 频道 AMD/ROCm 适配版。

    内置 ROCm 专用 LCSA 稀疏注意力后端（自动生效，无需 sageattention）；
    已修复上游 LQ 条件模块的因果缓存顺序 bug（消除每 4 帧一次的元素跳动）。
    """

    @classmethod
    def INPUT_TYPES(cls):''')
add(F, '''
class AILab_FlashVSR_Advanced:
    @classmethod
    def INPUT_TYPES(cls):''',
    '''
class AILab_FlashVSR_Advanced:
    """FlashVSR 快速视频超分（完整参数版）— lundao1yu0 频道 AMD/ROCm 适配版。

    在原 1038lab 节点基础上的三个模型级修复：
      1. LCSA 稀疏注意力掩码 ROCm 移植（原版在 AMD 上被静默丢弃 → 亮度泵动闪烁）
      2. LQ 条件模块因果缓存顺序修复（消除脸部元素周期性跳动，全平台受益）
      3. 注意力后端纯异步化（消除逐层 GPU→CPU 同步）
    高级环境变量：FLASHVSR_ROCM_ATTENTION = gather | reference | flex
    """

    @classmethod
    def INPUT_TYPES(cls):''')

# ---- SeedVR2 upscaler（按源码精确格式）----
add(S, '''"Input video frames as image batch.\\n"
                        "Accepts both RGB (3-channel) and RGBA (4-channel) formats.\\n"
                        "Output format will match input format."''',
    '''"输入视频帧（图像批次）。\\n"
                        "支持 RGB（3 通道）与 RGBA（4 通道），输出格式与输入一致。"''')
add(S, '''"Random seed for reproducible generation (default: 42).\\n"
                        "Same seed with same inputs produces identical output."''',
    '''"随机种子（默认 42）。同种子+同输入=可复现输出。"''')
add(S, '''"Target resolution for the shortest edge in pixels (default: 1080).\\n"
                        "Automatically maintains aspect ratio of input.\\n"
                        "Even values required for optimal processing."''',
    '''"目标分辨率（短边像素，默认 1080）。自动保持输入宽高比；需要偶数值。"''')
add(S, '''"Maximum resolution limit for any dimension (default: 0, no limit).\\n"
                        "If any edge exceeds this after applying resolution,\\n"
                        "both dimensions are scaled down proportionally.\\n"
                        "Useful to prevent excessive VRAM usage on extreme aspect ratios."''',
    '''"任一边的最大分辨率上限（默认 0=不限制）。防止极端宽高比导致显存爆炸。"''')
add(S, '''"Number of frames processed together per batch (default: 5).\\n"
                        "Must follow pattern 4n+1: 1, 5, 9, 13, 17, 21, ...\\n"''',
    '''"每批一起处理的帧数（默认 5）。必须符合 4n+1：1, 5, 9, 13, 17, 21, ...\\n"''')
add(S, '''"• Higher values: Better temporal consistency and faster processing\\n"
                        "• Lower values: Reduced VRAM usage\\n"''',
    '''"批越大：时序一致性越好、吞吐越高；批越小：显存越省。\\n"''')
add(S, '''"Pad final batch to match batch_size (default: False).\\n"
                        "Prevents temporal artifacts caused by small final batch.\\n"
                        "Add extra compute but recommended for optimal quality."''',
    '''"把最后一批补齐到 batch_size（默认关闭）。防止末尾小批次的时序伪影，略增计算但推荐开启。"''')
add(S, '''"Overlapping frames between consecutive batches (default: 0, disabled).\\n"
                        "Improves temporal consistency across batch boundaries through blending.\\n"
                        "Values 1-4 work well for most content."''',
    '''"相邻批次间的重叠帧数（默认 0=关闭）。混合处理批次边界，1-4 对多数内容有效。"''')
add(S, 'tooltip="DiT model configuration from SeedVR2 (Down)Load DiT Model node"',
    'tooltip="来自 SeedVR2 (Down)Load DiT Model 节点的模型配置"')
add(S, 'tooltip="VAE model configuration from SeedVR2 (Down)Load VAE Model node"',
    'tooltip="来自 SeedVR2 (Down)Load VAE Model 节点的模型配置"')

# ---- SeedVR2 loaders ----
add(D, '''"DiT (Diffusion Transformer) model for video upscaling.\\n"
                        "Models automatically download on first use.\\n"
                        "Additional models can be added to the ComfyUI models folder."''',
    '''"用于视频超分的 DiT（扩散 Transformer）模型。首次使用自动下载。\\n"
                        "本机已就绪：seedvr2_ema_3b-Q4_K_M.gguf（3B Q4 量化，20GB 显存实测可用）。"''')
add(D, 'tooltip="GPU device for DiT model inference (upscaling phase)"',
    'tooltip="DiT 模型推理（超分阶段）使用的 GPU 设备"')
add(V, '"Load and configure SeedVR2 VAE (Variational Autoencoder) for encoding/decoding video frames to/from latent space. "',
    '"加载并配置 SeedVR2 VAE（变分自编码器），用于视频帧与隐空间的相互编解码。"')
add(V, 'tooltip="GPU device for VAE model inference (encoding/decoding phases)"',
    'tooltip="VAE 模型推理（编/解码阶段）使用的 GPU 设备"')


def main():
    dry = "--dry" in sys.argv
    files = {}
    ok = True
    for path, old, new in EDITS:
        src = files.get(path) or open(path, encoding="utf-8").read()
        if new in src:
            print(f"[skip] {path.rsplit('/',1)[1]}: already applied")
            continue
        n = src.count(old)
        if n < 1:
            print(f"[FAIL] {path.rsplit('/',1)[1]}: anchor x{n}: {old[:60]!r}")
            ok = False
            continue
        files[path] = src.replace(old, new)
        print(f"[{'dry ' if dry else ' ok '}] {path.rsplit('/',1)[1]} x{n}: {old[:44]!r}...")
    if not ok or dry:
        sys.exit(0 if ok else 1)
    for path, src in files.items():
        open(path, "w", encoding="utf-8").write(src)
        py_compile.compile(path, doraise=True)
        print("wrote+compiled:", path)


if __name__ == "__main__":
    main()
