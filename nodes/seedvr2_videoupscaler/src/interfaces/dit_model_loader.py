"""
SeedVR2 DiT Model Loader Node
Configure DiT (Diffusion Transformer) model with memory optimization
"""

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context
from typing import Dict, Any, Tuple
from ..utils.model_registry import get_available_dit_models, DEFAULT_DIT
from ..optimization.memory_manager import get_device_list


class SeedVR2LoadDiTModel(io.ComfyNode):
    """
    Configure DiT (Diffusion Transformer) model loader with memory optimization
    
    Provides configuration for:
    - Model selection and device placement
    - BlockSwap memory optimization for limited VRAM
    - Model caching between runs
    - Optional torch.compile integration
    
    Returns:
        SEEDVR2_DIT configuration dictionary for main upscaler node
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:        
        devices = get_device_list()
        dit_models = get_available_dit_models()
        
        return io.Schema(
            node_id="SeedVR2LoadDiTModel",
            display_name="SeedVR2 (Down)Load DiT Model",
            category="lundao1yu0/SEEDVR2",
            description=(
                "加载并配置 SeedVR2 的 DiT（扩散 Transformer）超分模型。"
                "支持 BlockSwap 低显存优化、批量处理时的模型缓存、多卡卸载与 torch.compile 加速。\n\n"
                "连接到 Video Upscaler 节点。"
            ),
            inputs=[
                io.Combo.Input("model",
                    options=dit_models,
                    default=DEFAULT_DIT,
                    tooltip=(
                        "用于视频超分的 DiT（扩散 Transformer）模型。首次使用自动下载；也可手动放入 ComfyUI models 目录。"
                    )
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="DiT 模型推理（超分阶段）使用的 GPU 设备"
                ),
                io.Int.Input("blocks_to_swap",
                    default=0,
                    min=0,
                    max=36,
                    step=1,
                    optional=True,
                    tooltip=(
                        "在设备间交换的 transformer 块数（BlockSwap 显存优化）。\n"
                        "0=关闭（默认）；3B 模型 0-32；7B 模型 0-36。显存不足时逐步调大。\n"
                    )
                ),
                io.Boolean.Input("swap_io_components",
                    default=False,
                    optional=True,
                    tooltip=(
                        "把输入/输出 embedding 与归一化层卸载以省显存。\n"
                        "需设置 offload_device 且与 device 不同；部分模型不可用。\n"
                    )
                ),
                io.Combo.Input("offload_device",
                    options=get_device_list(include_none=True, include_cpu=True),
                    default="none",
                    optional=True,
                    tooltip=(
                        "DiT 模型空闲时的卸载设备。\n"
                        "'none'=留在推理设备（默认，最快）；'cpu'=卸载到内存（省显存）。\n"
                    )
                ),
                io.Boolean.Input("cache_model",
                    default=False,
                    optional=True,
                    tooltip=(
                        "两次运行之间把 DiT 模型保留在卸载设备上。\n"
                        "批量处理时可避免反复加载；需设置 offload_device。\n"
                    )
                ),
                io.Combo.Input("attention_mode",
                    options=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"],
                    default="sdpa",
                    optional=True,
                    tooltip=(
                        "注意力后端：\n"
                        "sdpa=PyTorch 标准注意力（默认，稳定通用）；flash_attn_2=Flash Attention 2（需额外安装，AMD/ROCm 不支持）。\n"
                    )
                ),
                io.Custom("TORCH_COMPILE_ARGS").Input("torch_compile_args",
                    optional=True,
                    tooltip=(
                        "可选的 torch.compile 优化设置（来自 Torch Compile Settings 节点）。\n"
                        "兼容环境下 DiT 提速约 20-40%。\n"
                    )
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_DIT").Output(
                    tooltip=(
                        "DiT 模型配置（模型路径、设备、BlockSwap 与编译选项）。连接到 Video Upscaler 节点。\n"
                    )
                )
            ]
        )
    
    @classmethod
    def execute(cls, model: str, device: str, offload_device: str = "none",
                     cache_model: bool = False, blocks_to_swap: int = 0, 
                     swap_io_components: bool = False, attention_mode: str = "sdpa",
                     torch_compile_args: Dict[str, Any] = None) -> io.NodeOutput:
        """
        Create DiT model configuration for SeedVR2 main node
        
        Args:
            model: Model filename to load
            device: Target device for model execution
            offload_device: Device to offload model to when not in use
            cache_model: Whether to keep model loaded between runs
            blocks_to_swap: Number of transformer blocks to swap (requires offload_device != device)
            swap_io_components: Whether to offload I/O components (requires offload_device != device)
            attention_mode: Attention computation backend ('sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3')
            torch_compile_args: Optional torch.compile configuration from settings node
            
        Returns:
            NodeOutput containing configuration dictionary for SeedVR2 main node
            
        Raises:
            ValueError: If cache_model is enabled but offload_device is not set
        """
        # Validate cache_model configuration
        if cache_model and offload_device == "none":
            raise ValueError(
                "Model caching (cache_model=True) requires offload_device to be set. "
                f"Current: offload_device='{offload_device}'. "
                "Please set offload_device to specify where the cached DiT model should be stored "
                "(e.g., 'cpu' or another device). Set cache_model=False if you don't want to cache the model."
            )
        
        config = {
            "model": model,
            "device": device,
            "offload_device": offload_device,
            "cache_model": cache_model,
            "blocks_to_swap": blocks_to_swap,
            "swap_io_components": swap_io_components,
            "attention_mode": attention_mode,
            "torch_compile_args": torch_compile_args,
            "node_id": get_executing_context().node_id,
        }
        
        return io.NodeOutput(config)