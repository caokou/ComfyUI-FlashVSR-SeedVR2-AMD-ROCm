# **ComfyUI-FlashVSR ⚡**

基于 FlashVSR 模型的强大 ComfyUI 自定义节点，实现实时扩散式视频超分辨率处理，适用于流媒体应用。

https://github.com/user-attachments/assets/1d1528c5-e3c1-487f-9c29-267ddb817809

## **新闻与更新**
**2025/11/15**：FlashVSR 1.1 模型更新 + 画面重复修复 ( [update.md](https://github.com/1038lab/ComfyUI-FlashVSR/blob/main/update.md#v110-20251115) )
- 新增模型：Wan2_1-T2V-1.1_3B_FlashVSR_fp32.safetensors
- 大幅提升 T2V → VSR 的清晰度、稳定性与细节表现
- 应用了帧重复修复补丁（Issue #3）
- 同步更新 UPDATE.md 文件

**2025/10/24**: ComfyUI-FlashVSR 首次发布  
- 添加了 **FlashVSR ⚡** 和 **FlashVSR Advanced ⚡** 节点  
- 实现从 Hugging Face 自动下载模型 (1038lab/FlashVSR)  
- 支持 `.safetensors` 模型、音频透传以及低显存 tiling 功能

## **功能特性**

* **高质量视频超分**: 使用先进的 FlashVSR 模型将视频放大至 2 倍或 4 倍分辨率
* **多种模型版本**:  
  * **Full (最高质量)**: 最佳画质,显存占用较高  
  * **Tiny (快速)**: 平衡画质与速度,处理更快  
  * **Tiny Long (低显存)**: 针对显存有限的 GPU 优化,适合长视频  
* **SageAttention 优化** (可选): 安装 SageAttention 后自动获得约 20-30% 的速度提升。如未安装会自动回退到标准模式
* **智能分块处理**: 支持 `enable_tiling`,在低显存 GPU 上高效处理高分辨率视频  
* **自动模型下载**: 首次运行时将自动从 Hugging Face ([1038lab/FlashVSR](https://huggingface.co/1038lab/FlashVSR)) 下载所需的 `.safetensors` 模型  
* **音频透传**: 在视频帧处理过程中保留原始音频,确保同步和质量

## **安装方法**

### **方法 1: 通过 ComfyUI Manager 安装 (推荐)**

1. 启动 ComfyUI  
2. 点击侧边栏的"Manager"按钮 → "Install Custom Nodes"  
3. 搜索 **ComfyUI-FlashVSR**  
4. 点击"Install"按钮  
5. 重启 ComfyUI

### **方法 2: 克隆仓库**

1. 导航到你的 ComfyUI `custom_nodes` 目录  
2. 运行:  
```bash
git clone https://github.com/1038lab/ComfyUI-FlashVSR.git
```

3. 重启 ComfyUI

### **方法 3: 通过 Comfy CLI 安装**

1. 确保已安装 `comfy-cli`:

   ```bash
   pip install comfy-cli
   ```
2. 使用以下命令安装 ComfyUI-FlashVSR:

   ```bash
   comfy node install ComfyUI-FlashVSR
   ```
3. 重启 ComfyUI

### **方法 4: 手动下载模型**

* 模型将在首次运行时自动下载到 `ComfyUI/models/FlashVSR/`
* 如需手动下载模型,请访问 [1038lab/FlashVSR on Hugging Face](https://huggingface.co/1038lab/FlashVSR) 并将 `.safetensors` 文件下载到 `ComfyUI/models/FlashVSR/` 文件夹

| 模型文件 | 用途 |
|----------|------|
| Wan2_1-T2V-1.1_3B_FlashVSR_fp32.safetensors | **全新 FlashVSR 1.1 主模型** |
| Wan2_1-T2V-1_3B_FlashVSR_fp32.safetensors | 旧 FlashVSR 1.0 主模型 |
| Wan2.1_VAE.safetensors | 视频 VAE |
| Wan2_1_FlashVSR_LQ_proj_model_bf16.safetensors | 低质投影模型 |
| Wan2_1_FlashVSR_TCDecoder_fp32.safetensors | Tiny 模型解码器 |


> **📖 可选性能优化指南（约 20-30% 速度提升），请参见 [SageAttention 安装指南](./SAGEATTENTION_INSTALL.md)**

## **使用方法**

此节点处理**图像(帧)序列**。要完成完整的视频工作流,需要与 ComfyUI 中的其他节点配合使用。

* **加载**: 使用视频加载器(例如 **VHS - Video Load**)加载视频帧和音频
* **处理**: 将帧和音频连接到 **FlashVSR 节点**
* **保存**: 使用视频合成器(例如 **VHS - Video Combine**)将输出帧和音频合成为最终的超分视频

### **FlashVSR 节点**

### **可选设置 💡 提示**

| 可选设置                      | 说明                                                                        | 提示                                                                              |
| ---------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **preset** (基础)            | 选择: `Fast` (Tiny 模型)、`Balanced` (Tiny 模型)、`High Quality` (Full 模型) | `High Quality` 需要较大显存。如遇显存不足请考虑使用 Advanced 节点                        |
| **model_version** (高级)     | 选项: `Tiny (Fast)`、`Tiny Long (Low VRAM)`、`Full (Best Quality)`          | `Full` 提供最佳画质,而 `Tiny Long` 针对最小显存优化                                     |
| **enable_tiling** (高级)     | 将视频分成小块处理以节省显存                                                   | 如遇显存不足错误请启用,尤其是使用 Full 模型 4x 放大时                                    |
| **speed_optimization**       | 优化处理速度。数值越高处理越快                                                | 默认值为 `2.0`                                                                     |
| **quality_boost**            | 以显存为代价提升画质。数值越高画质越好                                          | 默认值为 `2.0`。Full 模型可设置 `3.0` 而不会崩溃                                       |
| **Input Frames**             | 要处理的视频帧                                                               | 初始化至少需要 **21 帧**                                                            |
| **4x Upscaling**             | 针对 4x 放大优化                                                             | 支持 2x 放大,但 4x 通常效果更好                                                       |
| **sageattention** (高级)     | 启用/禁用 SageAttention 优化                                                 | 默认启用。如已安装 `sageattention` 包可提供约 20-30% 的速度提升                          |

## **关于 FlashVSR 模型**

**FlashVSR** 是一个实时扩散式视频超分辨率模型。它旨在提供高质量的超分效果,特别适合流媒体应用。包含的 `.safetensors` 版本提供了更好的兼容性和安全性。

## **环境要求**

* **ComfyUI**
* **Python 3.10+**
* **必需包**:
  * `torch >= 2.0.0`
  * `torchvision >= 0.15.0`
  * `safetensors >= 0.4.0`
  * `huggingface_hub >= 0.19.0`
  * `einops >= 0.6.0`
  * `numpy >= 1.24.0`
  * `tqdm >= 4.65.0`
  * `pillow >= 9.5.0`

* **可选包** (用于性能提升):
  * `sageattention >= 1.0.0` - 提供约 20-30% 的速度提升 (参见[可选性能优化](#可选性能优化))
  * `triton >= 2.1.0` - SageAttention 所需

这些包通常已包含在 ComfyUI 环境中。如遇导入错误,请运行:

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 safetensors>=0.4.0 huggingface-hub>=0.19.0 einops>=0.6.0
```

### **可选性能优化**

要获得约 20-30% 的自动性能提升,可以安装 SageAttention:

```bash
pip install sageattention triton
```

**注意**: 
- SageAttention 需要支持 CUDA 的 GPU,且可能与某些 ComfyUI 环境冲突
- **详细的安装说明和故障排除**,请参见 [SageAttention 安装指南](./SAGEATTENTION_INSTALL.md)
- 如果安装 SageAttention 后遇到问题,您可以:
  1. 在 **FlashVSR Advanced ⚡** 节点中将 `sageattention` 设置为 `disable` 来禁用它
  2. 或者卸载它: `pip uninstall sageattention triton`
- 即使不安装 SageAttention,节点也能完美运行 - 它会自动回退到标准 PyTorch 注意力机制

## **故障排除**

* **FileNotFoundError: 缺少 `Wan2.1_VAE.safetensors`**:

  * 此错误通常在模型下载失败或被跳过时发生
  * **解决方法**: 删除 `ComfyUI/models/` 中的 `FlashVSR` 文件夹,然后重启 ComfyUI 以再次触发自动下载

* **显存不足 (OOM) 错误 / CUDAMallocAsyncAllocator.cpp 错误**:

  * 在使用 High Quality 预设或 Full 模型 4x 放大时显存耗尽
  * **解决方法**: 使用 **FlashVSR Advanced ⚡** 节点并启用 `enable_tiling` 以减少显存使用

## **致谢**

* **FlashVSR**: [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
* **原始 HF 模型**: [JunhaoZhuang/FlashVSR](https://huggingface.co/JunhaoZhuang/FlashVSR)
* **Safetensors 模型**: [1038lab/FlashVSR](https://huggingface.co/1038lab/FlashVSR)
* **创建者**: [AILab](https://github.com/1038lab)

## **Star 历史**

如果这个自定义节点对你有帮助,或者你欣赏这项工作,请在此仓库上给一个 ⭐!这是对我努力的极大鼓励!

## **许可证**

[GPL-3.0 License](https://github.com/1038lab/ComfyUI-FlashVSR/blob/main/LICENSE)

