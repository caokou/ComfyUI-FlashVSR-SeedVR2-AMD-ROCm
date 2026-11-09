"""
SeedVR2 Torch Compile Settings Node
Configure torch.compile optimization for DiT and VAE models
"""

from comfy_api.latest import io
from typing import Dict, Any, Tuple


class SeedVR2TorchCompileSettings(io.ComfyNode):
    """Configure torch.compile optimization for DiT and VAE models"""
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeedVR2TorchCompileSettings",
            display_name="SeedVR2 Torch Compile Settings",
            category="lundao1yu0/SEEDVR2",
            description=(
                "配置 SeedVR2 的 torch.compile 优化：DiT 提速约 20-40%，VAE 提速约 15-25%。"
                "代价是首次运行需要较长的编译时间。\n\n"
                "连接到 DiT / VAE 加载器。需要 PyTorch 2.0+，inductor 后端还需 Triton（AMD/ROCm 上实验性，建议默认关闭）。"
            ),
            inputs=[
                io.Combo.Input("backend", 
                    options=["inductor", "cudagraphs"],
                    default="inductor",
                    tooltip=(
                        "编译后端：\n"
                        "inductor=完整优化（Triton 内核生成与融合，推荐）；\n"
                        "cudagraphs=轻量封装（兼容性更好，提升有限）。\n"
                    )
                ),
                io.Combo.Input("mode",
                    options=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                    default="default",
                    tooltip=(
                        "优化等级（编译时间 vs 运行性能）：\n"
                        "default=编译快、提速良好（推荐）；reduce-overhead=更大提速但稳定性下降；max-autotune=最强优化但编译很慢。\n"
                    )
                ),
                io.Boolean.Input("fullgraph",
                    default=False,
                    tooltip=(
                        "整图编译（不允许图断裂）：\n"
                        "False=允许 graph break，兼容性更好（默认）；True=强制无断裂，不兼容时报错。\n"
                    )
                ),
                io.Boolean.Input("dynamic",
                    default=False,
                    tooltip=(
                        "动态形状支持（避免形状变化时重新编译）：\n"
                        "False=按固定形状特化（默认）；True=生成自适应内核。\n"
                    )
                ),
                io.Int.Input("dynamo_cache_size_limit",
                    default=64,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "每个函数缓存的编译版本数上限（默认 64）。\n"
                        "形状多变时适当调大。\n"
                    )
                ),
                io.Int.Input("dynamo_recompile_limit",
                    default=128,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "回退到 eager 模式前的最大重编译次数（默认 128）。\n"
                        "防止无限编译循环的安全阈值，一般不用改。\n"
                    )
                ),
            ],
            outputs=[
                io.Custom("TORCH_COMPILE_ARGS").Output(
                    tooltip=(
                        "torch.compile 优化设置（后端、模式与 Dynamo 配置）。连接到 DiT / VAE 加载器节点。\n"
                    )
                )
            ]
        )
    
    @classmethod
    def execute(cls, backend: str, mode: str, fullgraph: bool, dynamic: bool, 
                   dynamo_cache_size_limit: int, dynamo_recompile_limit: int) -> io.NodeOutput:
        """
        Create torch.compile configuration for model optimization
        
        Args:
            backend: Compilation backend ("inductor" or "cudagraphs")
            mode: Optimization mode ("default", "reduce-overhead", "max-autotune", etc.)
            fullgraph: Whether to compile entire model as single graph
            dynamic: Whether to handle varying input shapes without recompilation
            dynamo_cache_size_limit: Maximum cached compiled versions per function
            dynamo_recompile_limit: Maximum recompilation attempts before fallback
            
        Returns:
            NodeOutput containing torch.compile configuration dictionary
        """
        compile_args = {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "dynamo_recompile_limit": dynamo_recompile_limit,
        }
        return io.NodeOutput(compile_args)