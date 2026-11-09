# ComfyUI-FlashVSR
# Powerful ComfyUI custom node built on the FlashVSR model,
# facilitating real-time diffusion-based video super-resolution for streaming applications.
#
# Models License Notice:
# - FlashVSR: Apache-2.0 License (https://huggingface.co/JunhaoZhuang/FlashVSR)
# - FlashVSR: Apache-2.0 License (https://github.com/OpenImagingLab/FlashVSR)
# - FlashVSR: Apache-2.0 License (https://huggingface.co/1038lab/FlashVSR)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-FlashVSR

import os
import math
import torch
import threading
import folder_paths
import comfy.utils
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from .FlashVSR import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from .FlashVSR.models.TCDecoder import build_tcdecoder
from .FlashVSR.models.utils import clean_vram, Buffer_LQ4x_Proj
from .FlashVSR.models import wan_video_dit as _wan_video_dit

def get_device_list():
    devs = ["auto"]
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except:
        pass
    try:
        if hasattr(torch, "mps") and torch.mps.is_available():
            devs += [f"mps:{i}" for i in range(torch.mps.device_count())]
    except:
        pass
    return devs

DEVICE_CHOICES = get_device_list()

def log(msg, level='info'):
    colors = {'error': '\033[1;41m', 'warning': '\033[1;31m', 'success': '\033[1;32m', 'info': '\033[1;33m'}
    c = colors.get(level, '')
    print(f"{c}[FlashVSR] {msg}\033[m" if c else f"[FlashVSR] {msg}")

_flashvsr_models_checked = False
_flashvsr_check_lock = threading.Lock()
FLASHVSR_MODEL_DIR = os.path.join(folder_paths.models_dir, "FlashVSR")


def _patch_wan_video_dit():
    if getattr(_wan_video_dit, "_flashvsr_dtype_patch", False):
        return

    def sinusoidal_embedding_1d(dim, position):
        work_dtype = torch.float32
        half_dim = max(dim // 2, 1)
        scale = torch.arange(half_dim, dtype=work_dtype, device=position.device)
        inv_freq = torch.pow(10000.0, -scale / half_dim)
        sinusoid = torch.outer(position.to(work_dtype), inv_freq)
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return x.to(position.dtype)

    def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
        work_dtype = torch.float32
        half_dim = max(dim // 2, 1)
        base = torch.arange(0, dim, 2, dtype=work_dtype)[:half_dim]
        freqs = torch.pow(theta, -base / max(dim, 1))
        steps = torch.arange(end, dtype=work_dtype)
        angles = torch.outer(steps, freqs)
        return torch.polar(torch.ones_like(angles), angles)

    def rope_apply(x, freqs, num_heads):
        x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
        orig_dtype = x.dtype
        work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
        reshaped = x.to(work_dtype).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
        x_complex = torch.view_as_complex(reshaped)
        freqs = freqs.to(dtype=x_complex.dtype, device=x_complex.device)
        x_out = torch.view_as_real(x_complex * freqs).flatten(2)
        return x_out.to(orig_dtype)

    _wan_video_dit.sinusoidal_embedding_1d = sinusoidal_embedding_1d
    _wan_video_dit.precompute_freqs_cis = precompute_freqs_cis
    _wan_video_dit.rope_apply = rope_apply
    _wan_video_dit._flashvsr_dtype_patch = True


_patch_wan_video_dit()

def check_and_download_models():
    global _flashvsr_models_checked
    
    with _flashvsr_check_lock:

        if _flashvsr_models_checked:
            return FLASHVSR_MODEL_DIR
        
        log("First run detected, checking for models...")
        model_dir = FLASHVSR_MODEL_DIR

        required_files = [
            "FlashVSR1_1.safetensors",
            "Wan2.1_VAE.safetensors",
            "LQ_proj_in.safetensors",
            "TCDecoder.safetensors",
            "Prompt.safetensors",
        ]

        if not os.path.exists(model_dir):
            log(f"Creating model directory: {model_dir}")
            os.makedirs(model_dir)

        missing_files = []
        for file_name in required_files:
            file_path = os.path.join(model_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            log(f"Missing models detected: {', '.join(missing_files)}. Downloading from HuggingFace...")
            try:
                snapshot_download(repo_id="1038lab/FlashVSR", local_dir=model_dir, 
                                local_dir_use_symlinks=False, resume_download=True)
                log("Download complete.", 'success')
            except Exception as e:
                log(f"Failed to download models: {e}", 'error')
                raise e
        else:
            log("All required models are present in cache.", 'success')

        final_missing = []
        for file_name in required_files:
            if not os.path.exists(os.path.join(model_dir, file_name)):
                final_missing.append(file_name)
        
        if final_missing:
            error_msg = f"Failed to find required models after download: {', '.join(final_missing)}. Node will not load."
            log(error_msg, 'error')
            raise FileNotFoundError(error_msg)
        
        _flashvsr_models_checked = True
        log("Model check complete.", 'success')
        return model_dir


def compute_dims(w, h, scale, align=128):
    sw, sh = w * scale, h * scale
    tw = math.ceil(sw / align) * align
    th = math.ceil(sh / align) * align
    return sw, sh, tw, th


def align_frames(n):
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def _repeat_last_frame(frames, repeat_count):
    if repeat_count <= 0:
        return frames
    repeats = [repeat_count] + [1 for _ in range(frames.ndim - 1)]
    tail = frames[-1:].repeat(*repeats)
    return torch.cat([frames, tail], dim=0)


def _pad_video_sequence(frames):
    frames = _repeat_last_frame(frames, 2)
    added_frames = 0
    remainder = (frames.shape[0] - 5) % 8
    if remainder != 0:
        added_frames = 8 - remainder
        frames = _repeat_last_frame(frames, added_frames)
    return frames, added_frames


def _restore_video_sequence(result, added_frames, expected_frames):
    if added_frames > 0 and result.shape[0] > added_frames:
        log(f"Removed {added_frames} padded frame(s) from the end.", 'info')
        result = result[:-added_frames]
    if result.shape[0] <= 2:
        log("FlashVSR returned fewer than 3 frames after padding removal; using fallback trimming.", 'warning')
        return _adjust_frame_count(result, expected_frames)
    result = result[2:]
    log("Removed the first 2 frames duplicated internally by FlashVSR.", 'info')
    return _adjust_frame_count(result, expected_frames)


def prepare_video(frames, device, scale, dtype):
    N, H, W, C = frames.shape
    sw, sh, tw, th = compute_dims(W, H, scale)
    
    num_padded = N + 4
    aligned = align_frames(num_padded)
    if aligned == 0:
        raise ValueError(f"Need at least 21 frames, got {N}")
    
    processed = []
    for i in range(aligned):
        if i < 2:
            idx = 0
        elif i > N + 1:
            idx = N - 1
        else:
            idx = i - 2
            
        frame = frames[idx].permute(2, 0, 1).unsqueeze(0)    

        upscaled = F.interpolate(frame, size=(sh, sw), mode='bicubic', align_corners=False)
        
        pad_h, pad_w = th - sh, tw - sw
        if pad_h > 0 or pad_w > 0:
            upscaled = F.pad(upscaled, (0, pad_w, 0, pad_h), mode='replicate')
        
        normalized = upscaled * 2.0 - 1.0
        processed.append(normalized.squeeze(0).cpu().to(dtype))
    
    video = torch.stack(processed, 0).permute(1, 0, 2, 3).unsqueeze(0)
    return video, th, tw, aligned, sh, sw


def to_frames(video):
    v = video.squeeze(0)
    v = rearrange(v, "C F H W -> F H W C")
    return (v.float() + 1.0) / 2.0


def calc_tiles(h, w, size, overlap):
    tiles = []
    stride = size - overlap
    rows = math.ceil((h - overlap) / stride)
    cols = math.ceil((w - overlap) / stride)
    
    for r in range(rows):
        for c in range(cols):
            y1, x1 = r * stride, c * stride
            y2, x2 = min(y1 + size, h), min(x1 + size, w)
            
            if y2 - y1 < size:
                y1 = max(0, y2 - size)
            if x2 - x1 < size:
                x1 = max(0, x2 - size)
            
            tiles.append((x1, y1, x2, y2))
    return tiles


def make_mask(h, w, overlap):
    mask = torch.ones(1, 1, h, w)
    ramp = torch.linspace(0, 1, overlap)
    
    mask[:, :, :, :overlap] *= ramp.view(1, 1, 1, -1)
    mask[:, :, :, -overlap:] *= ramp.flip(0).view(1, 1, 1, -1)
    mask[:, :, :overlap, :] *= ramp.view(1, 1, -1, 1)
    mask[:, :, -overlap:, :] *= ramp.flip(0).view(1, 1, -1, 1)
    return mask 

def init_pipe(mode, device, dtype):
    model_dir = check_and_download_models()
    
    ckpt = os.path.join(model_dir, "FlashVSR1_1.safetensors")
    vae = os.path.join(model_dir, "Wan2.1_VAE.safetensors")
    lq = os.path.join(model_dir, "LQ_proj_in.safetensors")
    tcd = os.path.join(model_dir, "TCDecoder.safetensors")
    prompt = os.path.join(model_dir, "Prompt.safetensors")
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    
    if mode == "full":
        mm.load_models([ckpt, vae])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        mm.load_models([ckpt])
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device) if mode == "tiny" else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        
        pipe.TCDecoder = build_tcdecoder([512, 256, 128, 128], device, dtype, 16 + 768)
        pipe.TCDecoder.load_state_dict(load_file(tcd, device=device), strict=False)
        pipe.TCDecoder.clean_mem()
    
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(3, 1536, 1).to(device, dtype)
    if os.path.exists(lq):
        lq_state_dict = load_file(lq, device="cpu")
        
        cleaned_state_dict = {}
        prefix_to_remove = "LQ_proj_in."
        for k, v in lq_state_dict.items():
            if k.startswith(prefix_to_remove):
                cleaned_state_dict[k[len(prefix_to_remove):]] = v
            else:
                cleaned_state_dict[k] = v
        
        pipe.denoising_model().LQ_proj_in.load_state_dict(cleaned_state_dict, strict=True)
    
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt)
    pipe.load_models_to_device(["dit", "vae"])
    
    return pipe


def _setup_device_and_dtype(device_str, dtype_str):

    dev = device_str
    if device_str == "auto":
        dev = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else None
        if not dev:
            log("No 'auto' device (CUDA or MPS) available, falling back to CPU. This might be very slow.", 'warning')
            dev = "cpu"
    
    if dev.startswith("cuda"):
        torch.cuda.set_device(dev)
    
    dt = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    return dev, dt


def _adjust_frame_count(result, expected_frames):

    if result.shape[0] != expected_frames:
        log(f"Warning: Output frames ({result.shape[0]}) != input frames ({expected_frames}). Truncating/padding output.", 'warning')
        if result.shape[0] > expected_frames:
            result = result[:expected_frames]
        else:
            padding = torch.zeros(expected_frames - result.shape[0], *result.shape[1:], dtype=result.dtype)
            if result.shape[0] > 0:
                padding[...] = result[-1]
            result = torch.cat((result, padding), dim=0)
    return result


class PBar:
    def __init__(self, iterable, total=None):
        self.iterable = iterable
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = 0
        self.total = total
        self.pbar = comfy.utils.ProgressBar(self.total)
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.total:
            self.current += 1
            self.pbar.update(1)
            if isinstance(self.iterable, range):
                return self.iterable[self.current - 1]
            try:
                return self.iterable[self.current - 1]
            except TypeError:
                return next(self.iterable) 
        else:
            raise StopIteration

    def update(self, n=1):
        self.current += n
        self.pbar.update(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _full(frames, pipe, scale, sr, kvr, lr, cf, ud, tv, seed, dev, dt):
    vid, th, tw, nf, sh, sw = prepare_video(frames, dev, scale, dt)
    
    if "long" not in pipe.__class__.__name__.lower():
        vid = vid.to(dev)
    
    out = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, 
                seed=seed, tiled=tv, progress_bar_cmd=lambda t: PBar(t), LQ_video=vid,
                num_frames=nf, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sr * 768 * 1280 / (th * tw), kv_ratio=kvr, local_range=lr,
                color_fix=cf, unload_dit=ud)
    
    res = to_frames(out).cpu()[:frames.shape[0], :sh, :sw, :]
    return res
    
def _tile(frames, pipe, scale, ts, to, sr, kvr, lr, cf, ud, tv, seed, dev, dt):
    N, H, W, C = frames.shape
    nf = align_frames(N + 4) - 4
    oh, ow = H * scale, W * scale
    
    canvas = torch.zeros((nf, oh, ow, C), dtype=torch.float32)
    weights = torch.zeros_like(canvas)
    tiles = calc_tiles(H, W, ts, to)
    
    with PBar(tiles) as pb:
        for i, (x1, y1, x2, y2) in enumerate(pb):
            log(f"Processing tile {i+1}/{len(tiles)}")
            
            tf = frames[:, y1:y2, x1:x2, :]
            tv_tensor, th, tw, tnf, tsh, tsw = prepare_video(tf, dev, scale, dt)
            
            if "long" not in pipe.__class__.__name__.lower():
                tv_tensor = tv_tensor.to(dev)
            
            tout = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1,
                        seed=seed, tiled=tv, LQ_video=tv_tensor, num_frames=tnf, height=th, width=tw,
                        is_full_block=False, if_buffer=True, topk_ratio=sr * 768 * 1280 / (th * tw),
                        kv_ratio=kvr, local_range=lr, color_fix=cf, unload_dit=ud)
            
            tres = to_frames(tout).cpu()[:nf, :tsh, :tsw, :]
            
            ah, aw = tres.shape[1], tres.shape[2]
            mask = make_mask(ah, aw, min(to * scale, ah // 4, aw // 4)).permute(0, 2, 3, 1)
            
            oy1, ox1 = y1 * scale, x1 * scale
            oy2, ox2 = min(oy1 + ah, oh), min(ox1 + aw, ow)
            ath, atw = oy2 - oy1, ox2 - ox1
            
            tres = tres[:, :ath, :atw, :]
            mask = mask[:, :ath, :atw, :]
            
            canvas[:, oy1:oy2, ox1:ox2, :] += tres * mask
            weights[:, oy1:oy2, ox1:ox2, :] += mask
            
            del tv_tensor, tout, tres
            # Clean VRAM every 4 tiles instead of every tile for better performance
            if (i + 1) % 4 == 0 or i == len(tiles) - 1:
                clean_vram()
    
    weights[weights == 0] = 1.0
    return canvas / weights


class AILab_FlashVSR:
    """FlashVSR 快速视频超分（简版节点）— lundao1yu0 频道 AMD/ROCm 适配版。

    内置 ROCm 专用 LCSA 稀疏注意力后端（自动生效，无需 sageattention）；
    已修复上游 LQ 条件模块的因果缓存顺序 bug（消除每 4 帧一次的元素跳动）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "待超分的低分辨率视频帧序列。"}),
                "preset": (["Fast (2x Speed)", "Balanced (2x Quality)", "Long Video (Low VRAM)", "High Quality (Best)"], {"default": "Balanced (2x Quality)", "tooltip": "Select a balance between speed and quality. 'Long Video (Low VRAM)' 显存占用最低，适合长视频；'High Quality' 使用完整版模型，质量最佳。"}),
                "scale": ("INT", {"default": 2, "min": 2, "max": 4, "step": 2, "tooltip": "放大倍率。推荐 4x 以获得最佳质量。"}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "VAE 解码前先把主模型卸载出显存。省显存但略慢。"}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "扩散过程的随机种子，同种子+同输入可复现。"}),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO",)
    RETURN_NAMES = ("FRAMES", "AUDIO",)
    FUNCTION = "upscale"
    CATEGORY = "lundao1yu0/⚡FlashVSR"
    
    def upscale(self, frames, preset, scale, unload_model, seed, audio=None):
        # Use default values for removed UI parameters
        color_fix = True
        device = "auto"
        dtype = "bf16"
        
        original_frame_count = frames.shape[0]
        if original_frame_count < 21:
            raise ValueError(f"FlashVSR needs at least 21 frames to work, but got {frames.shape[0]}")

        frames, added_frames = _pad_video_sequence(frames)
        if frames.shape[0] != original_frame_count:
            log(f"Extended sequence for FlashVSR: {original_frame_count} -> {frames.shape[0]} frames", 'info')

        presets = {
            "Fast (2x Speed)": ("tiny", 1.5, 1.0, 9, True, True, 256, 32),
            "Balanced (2x Quality)": ("tiny", 2.0, 2.0, 11, True, True, 256, 32),
            "Long Video (Low VRAM)": ("tiny-long", 2.0, 2.0, 11, True, True, 256, 32),
            "High Quality (Best)": ("full", 2.0, 3.0, 11, True, True, 256, 32),
        }
        
        mode, sr, kvr, lr, td, tv, ts, to = presets[preset]
        ud = unload_model
        
        if scale != 4:
            log("Note: FlashVSR is optimized for 4x upscaling", 'warning')
        
        # Use helper function for device setup
        dev, dt = _setup_device_and_dtype(device, dtype)
        
        log(f"Starting: {frames.shape[0]} frames, {scale}x, {preset}")
        
        pipe = init_pipe(mode, dev, dt)
        
        if td:
            result = _tile(frames, pipe, scale, ts, to, sr, kvr, lr, color_fix, ud, tv, seed, dev, dt)
        else:
            result = _full(frames, pipe, scale, sr, kvr, lr, color_fix, ud, tv, seed, dev, dt)
        
        del pipe
        clean_vram()
        log("Done", 'success')

        result = _restore_video_sequence(result, added_frames, original_frame_count)

        return (result, audio,)
    

class AILab_FlashVSR_Advanced:
    """FlashVSR 快速视频超分（完整参数版）— lundao1yu0 频道 AMD/ROCm 适配版。

    在原 1038lab 节点基础上的三个模型级修复：
      1. LCSA 稀疏注意力掩码 ROCm 移植（原版在 AMD 上被静默丢弃 → 亮度泵动闪烁）
      2. LQ 条件模块因果缓存顺序修复（消除脸部元素周期性跳动，全平台受益）
      3. 注意力后端纯异步化（消除逐层 GPU→CPU 同步）
    高级环境变量：FLASHVSR_ROCM_ATTENTION = gather | reference | flex
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "model_version": (["Tiny (Fast)", "Tiny Long (Low VRAM)", "Full (Best Quality)"], {"default": "Full (Best Quality)", "tooltip": "模型档位：Tiny(Fast)=快速、质量良好（日常推荐）；Tiny Long=长视频省显存；Full=完整模型、质量最高但显存占用最大。"}),
                "scale": ("INT", {"default": 2, "min": 2, "max": 4, "step": 2, "tooltip": "放大倍率。推荐 4x 以获得最佳质量。"}),
                "enable_tiling": ("BOOLEAN", {"default": True, "tooltip": "分块（tile）处理视频。更慢但大幅节省显存；显存紧张或 AMD/ROCm 环境请保持开启。"}),
                "tile_size": ("INT", {"default": 384, "min": 128, "max": 1024, "step": 32, "tooltip": "每个分块的宽高（超分前）。块越大显存占用越高；块过大时因注意力开销超线性增长反而变慢，建议 256-512。"}),
                "tile_overlap": ("INT", {"default": 24, "min": 8, "max": 256, "step": 8, "tooltip": "相邻分块的重叠像素，用于消除拼接缝。需小于块尺寸的一半。"}),
                "speed_optimization": ("FLOAT", {"default": 2.0, "min": 1.5, "max": 2.0, "step": 0.1, "tooltip": "速度优化系数（= LCSA 稀疏注意力的 topk 比例）。2.0 更快；1.5 更慢但细节更丰富。"}),
                "quality_boost": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "质量增强系数。越大细节保留越好，显存占用越高。"}),
                "stability_level": ([9, 11], {"default": 11, "tooltip": "时序稳定性等级。11=更稳定（默认）；9=更锐利但时序一致性略差。"}),
                "color_fix": ("BOOLEAN", {"default": True, "tooltip": "颜色校正：防止输出相对原视频出现色偏。"}),
                "vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "VAE 解码分块。进一步降低显存占用，高分辨率输出时尤其有效。"}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "VAE 解码前先把主模型卸载出显存。省显存但略慢。"}),
                "sageattention": (["enable", "disable"], {"default": "enable", "tooltip": "SageAttention 加速（NVIDIA 专用，约 20-30%）。AMD/ROCm 请保持 disable——本移植已内置 ROCm 专用 LCSA 注意力后端。"}),
                "device": (DEVICE_CHOICES, {"default": DEVICE_CHOICES[0], "tooltip": "计算设备。auto 自动选择可用设备。"}),
                "precision": (["bf16", "fp16"], {"default": "bf16", "tooltip": "计算精度。现代 GPU 上 BF16 通常更快，FP16 为备选。"}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "扩散过程的随机种子，同种子+同输入可复现。"}), 
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO",)
    RETURN_NAMES = ("frames", "audio",)
    FUNCTION = "upscale"
    CATEGORY = "lundao1yu0/⚡FlashVSR"
    
    def upscale(self, frames, model_version, scale, enable_tiling, tile_size, tile_overlap,
                speed_optimization, quality_boost, stability_level, color_fix, vae_tiling,
                unload_model, sageattention, device, precision, seed, audio=None):
        
        # Convert sageattention string to boolean for internal use
        use_sage = (sageattention == "enable")
        dtype = precision  # Map precision parameter to dtype variable
        
        original_frame_count = frames.shape[0]
        if original_frame_count < 21:
            raise ValueError(f"FlashVSR needs at least 21 frames to work, but got {frames.shape[0]}")

        if enable_tiling and tile_overlap >= tile_size / 2:
            raise ValueError("tile_overlap must be less than half of tile_size")

        frames, added_frames = _pad_video_sequence(frames)
        if frames.shape[0] != original_frame_count:
            log(f"Extended sequence for FlashVSR: {original_frame_count} -> {frames.shape[0]} frames", 'info')
        
        mode_map = {
            "Tiny (Fast)": "tiny",
            "Tiny Long (Low VRAM)": "tiny-long",
            "Full (Best Quality)": "full"
        }
        mode = mode_map[model_version]
        
        if scale != 4:
            log("Note: FlashVSR is optimized for 4x upscaling", 'warning')
        
        # Use helper function for device setup
        dev, dt = _setup_device_and_dtype(device, dtype)
        
        log(f"Starting: {frames.shape[0]} frames, {scale}x, {model_version}")
        
        pipe = init_pipe(mode, dev, dt)
        
        if enable_tiling:
            result = _tile(frames, pipe, scale, tile_size, tile_overlap,
                           speed_optimization, quality_boost, stability_level,
                           color_fix, unload_model, vae_tiling, seed, dev, dt)
        else:
            result = _full(frames, pipe, scale, speed_optimization,
                           quality_boost, stability_level, color_fix,
                           unload_model, vae_tiling, seed, dev, dt)
        
        del pipe
        clean_vram()
        log("Done", 'success')

        result = _restore_video_sequence(result, added_frames, original_frame_count)

        return (result, audio,)


NODE_CLASS_MAPPINGS = {
    "AILab_FlashVSR": AILab_FlashVSR,
    "AILab_FlashVSR_Advanced": AILab_FlashVSR_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_FlashVSR": "FlashVSR ⚡",
    "AILab_FlashVSR_Advanced": "FlashVSR ⚡ Advanced",
}
