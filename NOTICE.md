# NOTICE — Vendored code & patches

本仓库（umbrella）直接 vendor 了两个上游 ComfyUI 插件的**已修复版本**，clone 即用，
无需运行安装器、无需自行打补丁。

## Vendored upstreams

| 目录 | 上游 | 许可证 | 获取方式 |
|---|---|---|---|
| `nodes/flashvsr/` | [1038lab/ComfyUI-FlashVSR](https://github.com/1038lab/ComfyUI-FlashVSR) | GPL-3.0 | 2026-11 zip 快照（经镜像下载），已应用下列补丁 |
| `nodes/seedvr2_videoupscaler/` | [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) | Apache-2.0 | 2026-11 快照，已应用下列补丁 |

因 vendor 了 GPL-3.0 代码，本 umbrella 仓库整体以 **GPL-3.0** 发布（见 LICENSE）。
FlashVSR 模型本体为 Apache-2.0（OpenImagingLab/FlashVSR）。
`patches/` 目录保留了独立补丁脚本，供希望直接给上游克隆打补丁的用户使用。

## 本仓库新增的修改（相对上游快照）

1. **FlashVSR LQ 因果缓存顺序修复**（`flashvsr/FlashVSR/models/utils.py`）
   `Buffer_LQ4x_Proj` 原「先覆盖缓存再卷积」→ 改为与上游模型一致的「先卷积再存」，
   消除每 4 帧一次的脸部元素跳动（全平台 bug，NVIDIA 同样受益）。
2. **ROCm LCSA 稀疏注意力后端**（`flashvsr/FlashVSR/models/wan_video_dit.py` +
   `rocm_attention.py`，源自上游模型仓库 PR #114 思路重写）
   ROCm 上 flex_attention 会把 BlockMask 静默当 dense 执行；此补丁在
   `torch.version.hip` 为真时启用 gather+SDPA 后端，**NVIDIA 走原路径不受影响**。
3. **SeedVR2 中文提示 + 分类品牌**：节点分类改为 `lundao1yu0/...`，参数 tooltip /
   节点说明全部中文化（仅使用方法说明）。`seedvr2_videoupscaler/src/core/generation_utils.py`
   读取的 `pos_emb.pt` / `neg_emb.pt` 为上游运行时必需文件，随仓库分发。
4. **伞形加载器**（根 `__init__.py`）：FlashVSR 经典 V1 映射 + SeedVR2 V3
   `io.ComfyNode` 注册合并为单一 custom node。

## 维护

- 频道：lundao1yu0（商务合作 lundao1yu0@outlook.com）
- 基准与证据链见 `docs/BENCHMARK.md`；已验证工作流见 `workflows/`。
