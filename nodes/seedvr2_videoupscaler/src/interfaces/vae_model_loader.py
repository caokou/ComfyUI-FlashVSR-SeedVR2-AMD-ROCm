"""
SeedVR2 VAE Model Loader Node
Configure VAE (Variational Autoencoder) model with tiling support
"""

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context
from typing import Dict, Any, Tuple
from ..utils.model_registry import get_available_vae_models, DEFAULT_VAE
from ..optimization.memory_manager import get_device_list


class SeedVR2LoadVAEModel(io.ComfyNode):
    """
    Configure VAE (Variational Autoencoder) model loader with tiling support
    
    Provides configuration for:
    - Model selection and device placement
    - Tiled encoding/decoding for VRAM reduction
    - Tile size and overlap control
    - Model caching between runs
    - Optional torch.compile integration
    
    Returns:
        SEEDVR2_VAE configuration dictionary for main upscaler node
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:        
        devices = get_device_list()
        vae_models = get_available_vae_models()
        
        return io.Schema(
            node_id="SeedVR2LoadVAEModel",
            display_name="SeedVR2 (Down)Load VAE Model",
            category="lundao1yu0/SEEDVR2",
            description=(
                "加载并配置 SeedVR2 VAE（变分自编码器），用于视频帧与隐空间的相互编解码。"
                "支持分块处理（低显存跑高分辨率）、模型缓存、多卡卸载与 torch.compile 加速。\n\n"
                "连接到 Video Upscaler 节点。"
            ),
            inputs=[
                io.Combo.Input("model",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip=(
                        "用于编/解码的 VAE（变分自编码器）模型。首次使用自动下载；也可手动放入 ComfyUI models 目录。\n"
                    )
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="VAE 模型推理（编/解码阶段）使用的 GPU 设备"
                ),
                io.Boolean.Input("encode_tiled",
                    default=False,
                    optional=True,
                    tooltip=(
                        "编码阶段启用分块，降低显存占用。\n"
                    )
                ),
                io.Int.Input("encode_tile_size",
                    default=1024,
                    min=64,
                    step=32,
                    optional=True,
                    tooltip=(
                        "编码分块尺寸（像素，默认 1024，高宽同用）。\n"
                        "越小越省显存但更慢。\n"
                    )
                ),
                io.Int.Input("encode_tile_overlap",
                    default=128,
                    min=0,
                    step=32,
                    optional=True,
                    tooltip=(
                        "编码分块间的重叠像素（默认 128）。\n"
                        "通过混合消除分块拼接缝，越大质量越好但越慢。\n"
                    )
                ),
                io.Boolean.Input("decode_tiled",
                    default=False,
                    optional=True,
                    tooltip=(
                        "解码阶段启用分块，降低显存占用。\n"
                    )
                ),
                io.Int.Input("decode_tile_size",
                    default=1024,
                    min=64,
                    step=32,
                    optional=True,
                    tooltip=(
                        "解码分块尺寸（像素，默认 1024，高宽同用）。\n"
                        "越小越省显存但更慢。\n"
                    )
                ),
                io.Int.Input("decode_tile_overlap",
                    default=128,
                    min=0,
                    step=32,
                    optional=True,
                    tooltip=(
                        "解码分块间的重叠像素（默认 128）。\n"
                        "通过混合消除分块拼接缝，越大质量越好但越慢。\n"
                    )
                ),
                io.Combo.Input("tile_debug",
                    options=["false", "encode", "decode"],
                    default="false",
                    optional=True,
                    tooltip=(
                        "分块调试可视化：\n"
                        "false=关闭（默认）；encode=显示编码分块边界；decode=显示解码分块边界。\n"
                    )
                ),
                io.Combo.Input("offload_device",
                    options=get_device_list(include_none=True, include_cpu=True),
                    default="none",
                    optional=True,
                    tooltip=(
                        "VAE 模型空闲时的卸载设备。\n"
                        "'none'=留在推理设备（默认，最快）；'cpu'=卸载到内存（省显存）。\n"
                    )
                ),
                io.Boolean.Input("cache_model",
                    default=False,
                    optional=True,
                    tooltip=(
                        "两次运行之间把 VAE 模型保留在卸载设备上。\n"
                        "批量处理时可避免反复加载；需设置 offload_device。\n"
                    )
                ),
                io.Custom("TORCH_COMPILE_ARGS").Input("torch_compile_args",
                    optional=True,
                    tooltip=(
                        "可选的 torch.compile 优化设置（来自 Torch Compile Settings 节点）。\n"
                        "兼容环境下 VAE 提速约 15-25%。\n"
                    )
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_VAE").Output(
                    tooltip=(
                        "VAE 模型配置（模型路径、设备、分块参数与编译选项）。连接到 Video Upscaler 节点。\n"
                    )
                )
            ]
        )
    
    @classmethod
    def execute(cls, model: str, device: str, offload_device: str = "none",
                     cache_model: bool = False, encode_tiled: bool = False,
                     encode_tile_size: int = 512, encode_tile_overlap: int = 64,
                     decode_tiled: bool = False, decode_tile_size: int = 512, 
                     decode_tile_overlap: int = 64, tile_debug: str = "false",
                     torch_compile_args: Dict[str, Any] = None
                     ) -> io.NodeOutput:
        """
        Create VAE model configuration for SeedVR2 main node
        
        Args:
            model: Model filename to load
            device: Target device for model execution
            offload_device: Device to offload model to when not in use
            cache_model: Whether to keep model loaded between runs
            encode_tiled: Enable tiled encoding
            encode_tile_size: Tile size for encoding
            encode_tile_overlap: Tile overlap for encoding
            decode_tiled: Enable tiled decoding
            decode_tile_size: Tile size for decoding
            decode_tile_overlap: Tile overlap for decoding
            tile_debug: Tile visualization mode (false/encode/decode)
            torch_compile_args: Optional torch.compile configuration from settings node
            
        Returns:
            NodeOutput containing configuration dictionary for SeedVR2 main node
            
        Raises:
            ValueError: If cache_model is enabled but offload_device is invalid
        """
        # Validate cache_model configuration
        if cache_model and offload_device == "none":
            raise ValueError(
                "Model caching (cache_model=True) requires offload_device to be set. "
                f"Current: offload_device='{offload_device}'. "
                "Please set offload_device to specify where the cached VAE model should be stored "
                "(e.g., 'cpu' or another device). Set cache_model=False if you don't want to cache the model."
            )
        
        config = {
            "model": model,
            "device": device,
            "offload_device": offload_device,
            "cache_model": cache_model,
            "encode_tiled": encode_tiled,
            "encode_tile_size": encode_tile_size,
            "encode_tile_overlap": encode_tile_overlap,
            "decode_tiled": decode_tiled,
            "decode_tile_size": decode_tile_size,
            "decode_tile_overlap": decode_tile_overlap,
            "tile_debug": tile_debug,
            "torch_compile_args": torch_compile_args,
            "node_id": get_executing_context().node_id,
        }
        return io.NodeOutput(config)