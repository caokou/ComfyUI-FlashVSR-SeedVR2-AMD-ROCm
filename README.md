# AMD ROCm 视频超分套件（FlashVSR + SeedVR2）— lundao1yu0 出品

让 ComfyUI 上最热的两个 AI 视频超分插件在 AMD 显卡（ROCm）上真正可用：
修好闪烁、打通安装、附上 Radeon RX 7900 XT 首份实测数据。

## 关于 lundao1yu0

lundao1yu0 是专注于 AI 视频/图像工作流实测的频道：真机跑通、数据说话、
坑全部排给你看。本项目是频道 AMD ROCm 适配系列的完整产出，
全部修复在 Radeon RX 7900 XT 实机上验证。

📮 **商务合作请联系：[lundao1yu0@outlook.com](mailto:lundao1yu0@outlook.com)**

## 这个仓库是什么

伞形 custom node：**直接 clone 即可用**，已内置两个插件的修复版源码
（`nodes/flashvsr/`、`nodes/seedvr2_videoupscaler/`），无需安装器、无需打补丁、
无需 ROCm 环境（NVIDIA 用户 clone 后同样直接获得 LQ 缓存修复）：

- **FlashVSR（主推）** — 3 个模型级修复，首次在 AMD 上达到"可用"级画质
- **SeedVR2** — ROCm 运行验证 + 全中文注释的节点接口
- `install_rocm.sh` 可选辅助（装验证工作流 / `--models` 下载权重 / ROCm 调优提示）
- 两个已验证可直接运行的 ComfyUI 工作流（`workflows/`）
- `patches/` 保留独立补丁脚本，供偏好「克隆上游自行打补丁」的用户使用

## 核心修复（FlashVSR）

| # | 问题 | 症状 | 修复 |
|---|------|------|------|
| 1 | LCSA 稀疏注意力掩码在 AMD 上被**静默丢弃**（原节点仅在 sageattention 分支应用掩码，ROCm 永远走不到） | 亮度泵动闪烁 | 移植上游 LCSA 掩码：ROCm 专用 gather 注意力后端（`rocm_attention.py`），对拍上游参考实现 PSNR 78.8 dB |
| 2 | LQ 条件模块 `Buffer_LQ4x_Proj` 因果缓存**先覆盖后使用**（**全平台 bug**，NVIDIA 同样中招，上游 issue #10 的部分现象由此而来） | 脸部元素每 4 帧周期性跳动 | 对齐上游 `Causal_LQ4x_Proj` 的"先用后存"顺序（`fix_lq_cache_order.py`） |
| 3 | 注意力后端每层一次 GPU→CPU 同步（`.item()`） | 无谓的流水线停摆 | 纯异步化（静态 max_c） |

修复 2 的完整证据链（周期性自相关检测 → 上游源码 diff → 单点修复 → 复测）
见 `docs/BENCHMARK.md`。**该修复对所有硬件平台有效，欢迎 NVIDIA 用户实测。**

## 性能实测（Radeon RX 7900 XT，20GB）

| 指标 | 数值 |
|------|------|
| 544→1088 ×2（Tiny Fast，24fps） | 每秒素材约 25 s（约 0.96 fps） |
| 10s 片完整出片 | ~4.2 分钟 |
| 修复的画质代价 | 无（修复前"快"是因为带着 bug 在跑） |

**ROCm 版本建议：留在 6.4.x + torch rocm7.0。** 我们完整升级到 ROCm 7.2.4 +
torch 2.12.1 实测：RDNA3 消费卡无收益，全栈组合反而慢 9%（7.x 优化重心在
CDNA 数据卡）。详见 `docs/BENCHMARK.md`。

## 安装

### 方式 A · 直接 clone（推荐，任何平台）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/caokou/ComfyUI-FlashVSR-SeedVR2-AMD-ROCm.git
# 重启 ComfyUI 即可。依赖会从 requirements.txt 自动安装；
# 分类名: lundao1yu0/⚡FlashVSR 与 lundao1yu0/SEEDVR2
```

- NVIDIA / 无 ROCm 用户：clone 即得到全部修复（LQ 缓存修复对全平台有效，
  ROCm 注意力后端自动按 `torch.version.hip` 门控，不影响 CUDA 路径）
- 首次运行时 FlashVSR 需下载 `FlashVSR1_1.safetensors`（~5.3GB），
  SeedVR2 权重由节点自动下载；国内可 `export HF_ENDPOINT=https://hf-mirror.com`

### 方式 B · 可选辅助脚本

```bash
bash ComfyUI-FlashVSR-SeedVR2-AMD-ROCm/install_rocm.sh           # 装 2 个验证工作流 + ROCm 调优提示
bash ComfyUI-FlashVSR-SeedVR2-AMD-ROCm/install_rocm.sh --models  # 另外预下载 FlashVSR 权重
```

不需要 ROCm 环境检测、不克隆上游、不打补丁（源码已 vendor 并修复完毕）。
想自己维护上游克隆的用户，可单独使用 `patches/` 里的补丁脚本。

## ROCm 红线（必读）

1. **不要关闭 tiling** — gfx1100 上显存耗尽不是抛 OOM 异常，是**整机冻结**
  （连 SSH 都会失去响应，需物理重启）。
2. **tile_size 建议 256** — 更大的块反而更慢：注意力开销随序列长度超线性增长
  （实测 384 比 256 慢 52%）。
3. **`aten::index` 高级索引（3-D 索引 × 4-D 张量）在 gfx1100 必崩**（HSAIL
  exception）— 我们的代码已全部改用 `index_select`，写自定义节点时注意。
4. flex_attention 在 torch 2.10 rocm 上会把手工 BlockMask **静默当 dense 执行**，
  更新 torch 后可重新评估。

## 工作流

- `workflows/ui_flashvsr_e2.json` — FlashVSR ×2 超分（Lipsync/AI 素材验收配置）
- `workflows/ui_seedvr2_e2.json` — SeedVR2 3B Q4 →1088

两个工作流均已在本机验证可从 ComfyUI 界面直接运行。

## 致谢与许可

本仓库 vendor 了两个上游插件的修复版快照（来源、快照时间与本仓库相对上游的
全部修改清单见 [NOTICE.md](NOTICE.md)）：

- 上游模型：[OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR)（含 PR #114 的 ROCm 工作启发，Apache-2.0）
- 上游节点：[1038lab/ComfyUI-FlashVSR](https://github.com/1038lab/ComfyUI-FlashVSR)（GPL-3.0 → 本仓库整体 GPL-3.0）
- 上游节点：[numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)（Apache-2.0）

全部修复计划回馈上游（FlashVSR 的 LQ 缓存 bug 已提 [issue #20](https://github.com/1038lab/ComfyUI-FlashVSR/issues/20)，附完整证据链）。

---

English version: [README.en.md](README.en.md)
