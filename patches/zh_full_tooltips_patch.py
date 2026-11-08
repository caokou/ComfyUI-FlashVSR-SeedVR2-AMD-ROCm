#!/usr/bin/env python3
"""Full Chinese localization of ALL remaining SeedVR2 node tooltips.
Locates each io.*.Input("param", ...)/Output tooltip block by parameter name
and replaces the whole block with a Chinese tooltip. Run ON THE SERVER.
"""
import py_compile
import re
import sys

B = "/home/chenjiaxing/comfyui-rocm/custom_nodes/seedvr2_videoupscaler/src/interfaces/"

# (file, kind, name, chinese)  kind: "in" | "out"
ZH = [
    # ---- video_upscaler.py ----
    (B+"video_upscaler.py","in","prepend_frames","在处理前于视频开头 prepend 的帧数（取开头帧倒放，默认 0）。\n可减轻视频开头的伪影；prepend 的帧会被裁掉，不出现在输出里。"),
    (B+"video_upscaler.py","in","color_correction","颜色校正（默认 lab）：修正超分输出相对原始输入的色偏。\n超分过程可能改变颜色，此选项对输出做调色以贴近原片。"),
    (B+"video_upscaler.py","in","input_noise_scale","输入噪声注入强度（默认 0.0=关闭）。\n编码前向输入图像注入受控噪声，范围 0.0（无）到 1.0（最强）。"),
    (B+"video_upscaler.py","in","latent_noise_scale","隐空间噪声注入强度（默认 0.0=关闭）。\n扩散过程中注入受控噪声，范围 0.0（无）到 1.0（最强）。"),
    (B+"video_upscaler.py","in","offload_device","中间张量的存放设备（默认 cpu）。\n'none'=全部留在推理设备（最快、显存最高）；'cpu'=卸载到内存（省显存、稍慢）。"),
    (B+"video_upscaler.py","in","enable_debug","详细调试日志（默认关闭）。\n显示显存占用、耗时与处理细节，排查问题时开启。"),
    (B+"video_upscaler.py","out","IMAGE","超分后的视频帧（已做颜色校正）。格式（RGB/RGBA）与输入一致，取值范围 [0,1]。"),
    # ---- dit_model_loader.py ----
    (B+"dit_model_loader.py","in","blocks_to_swap","在设备间交换的 transformer 块数（BlockSwap 显存优化）。\n0=关闭（默认）；3B 模型 0-32；7B 模型 0-36。显存不足时逐步调大。"),
    (B+"dit_model_loader.py","in","swap_io_components","把输入/输出 embedding 与归一化层卸载以省显存。\n需设置 offload_device 且与 device 不同；部分模型不可用。"),
    (B+"dit_model_loader.py","in","offload_device","DiT 模型空闲时的卸载设备。\n'none'=留在推理设备（默认，最快）；'cpu'=卸载到内存（省显存）。"),
    (B+"dit_model_loader.py","in","cache_model","两次运行之间把 DiT 模型保留在卸载设备上。\n批量处理时可避免反复加载；需设置 offload_device。"),
    (B+"dit_model_loader.py","in","attention_mode","注意力后端：\nsdpa=PyTorch 标准注意力（默认，稳定通用）；flash_attn_2=Flash Attention 2（需额外安装，AMD/ROCm 不支持）。"),
    (B+"dit_model_loader.py","in","torch_compile_args","可选的 torch.compile 优化设置（来自 Torch Compile Settings 节点）。\n兼容环境下 DiT 提速约 20-40%。"),
    (B+"dit_model_loader.py","out","SEEDVR2_DIT","DiT 模型配置（模型路径、设备、BlockSwap 与编译选项）。连接到 Video Upscaler 节点。"),
    # ---- vae_model_loader.py ----
    (B+"vae_model_loader.py","in","model","用于编/解码的 VAE（变分自编码器）模型。首次使用自动下载。\n本机已就绪：ema_vae_fp16.safetensors。"),
    (B+"vae_model_loader.py","in","encode_tiled","编码阶段启用分块，降低显存占用。"),
    (B+"vae_model_loader.py","in","encode_tile_size","编码分块尺寸（像素，默认 1024，高宽同用）。\n越小越省显存但更慢。"),
    (B+"vae_model_loader.py","in","encode_tile_overlap","编码分块间的重叠像素（默认 128）。\n通过混合消除分块拼接缝，越大质量越好但越慢。"),
    (B+"vae_model_loader.py","in","decode_tiled","解码阶段启用分块，降低显存占用。"),
    (B+"vae_model_loader.py","in","decode_tile_size","解码分块尺寸（像素，默认 1024，高宽同用）。\n越小越省显存但更慢。"),
    (B+"vae_model_loader.py","in","decode_tile_overlap","解码分块间的重叠像素（默认 128）。\n通过混合消除分块拼接缝，越大质量越好但越慢。"),
    (B+"vae_model_loader.py","in","tile_debug","分块调试可视化：\nfalse=关闭（默认）；encode=显示编码分块边界；decode=显示解码分块边界。"),
    (B+"vae_model_loader.py","in","offload_device","VAE 模型空闲时的卸载设备。\n'none'=留在推理设备（默认，最快）；'cpu'=卸载到内存（省显存）。"),
    (B+"vae_model_loader.py","in","cache_model","两次运行之间把 VAE 模型保留在卸载设备上。\n批量处理时可避免反复加载；需设置 offload_device。"),
    (B+"vae_model_loader.py","in","torch_compile_args","可选的 torch.compile 优化设置（来自 Torch Compile Settings 节点）。\n兼容环境下 VAE 提速约 15-25%。"),
    (B+"vae_model_loader.py","out","SEEDVR2_VAE","VAE 模型配置（模型路径、设备、分块参数与编译选项）。连接到 Video Upscaler 节点。"),
    # ---- torch_compile_settings.py ----
    (B+"torch_compile_settings.py","in","backend","编译后端：\ninductor=完整优化（Triton 内核生成与融合，推荐）；\ncudagraphs=轻量封装（兼容性更好，提升有限）。"),
    (B+"torch_compile_settings.py","in","mode","优化等级（编译时间 vs 运行性能）：\ndefault=编译快、提速良好（推荐）；reduce-overhead=更大提速但稳定性下降；max-autotune=最强优化但编译很慢。"),
    (B+"torch_compile_settings.py","in","fullgraph","整图编译（不允许图断裂）：\nFalse=允许 graph break，兼容性更好（默认）；True=强制无断裂，不兼容时报错。"),
    (B+"torch_compile_settings.py","in","dynamic","动态形状支持（避免形状变化时重新编译）：\nFalse=按固定形状特化（默认）；True=生成自适应内核。"),
    (B+"torch_compile_settings.py","in","dynamo_cache_size_limit","每个函数缓存的编译版本数上限（默认 64）。\n形状多变时适当调大。"),
    (B+"torch_compile_settings.py","in","dynamo_recompile_limit","回退到 eager 模式前的最大重编译次数（默认 128）。\n防止无限编译循环的安全阈值，一般不用改。"),
    (B+"torch_compile_settings.py","out","SEEDVR2_TORCH_COMPILE","torch.compile 优化设置（后端、模式与 Dynamo 配置）。连接到 DiT / VAE 加载器节点。"),
]

TOOLTIP_BLOCK = re.compile(
    r'tooltip=\(\n(?:\s*"(?:[^"\\]|\\.)*"\s*\n)+\s*\)|tooltip="(?:[^"\\]|\\.)*"'
)


def patch_one(src, kind, name, zh):
    anchor = r'\.Output\(' if kind == "out" else r'\.Input\("' + name + r'"'
    m = re.search(anchor, src)
    if not m:
        return src, "anchor-missing"
    seg_start = m.start()
    seg = src[seg_start:seg_start + 2600]
    tm = TOOLTIP_BLOCK.search(seg)
    if not tm:
        return src, "tooltip-missing"
    # keep indentation of the tooltip=( line
    line_start = src.rfind("\n", 0, seg_start + tm.start()) + 1
    indent = re.match(r"[ \t]*", src[line_start:seg_start + tm.start()]).group(0)
    inner = indent + "    "
    lines = zh.split("\n")
    lit = "\n".join(f'{inner}"{ln}\\n"' for ln in lines)
    block = f'tooltip=(\n{lit}\n{indent})'
    seg2 = seg[:tm.start()] + block + seg[tm.end():]
    return src[:seg_start] + seg2 + src[seg_start + 2600:], "ok"


def main():
    dry = "--dry" in sys.argv
    files = {}
    fails = []
    for path, kind, name, zh in ZH:
        src = files.get(path) or open(path, encoding="utf-8").read()
        if f'"{zh.splitlines()[0][:24]}' in src:
            print(f"[skip] {path.rsplit('/',1)[1]} {name}")
            continue
        new, status = patch_one(src, kind, name, zh)
        if status != "ok":
            fails.append((path.rsplit('/',1)[1], name, status))
            print(f"[FAIL] {path.rsplit('/',1)[1]} {name}: {status}")
            continue
        files[path] = new
        print(f"[{'dry ' if dry else ' ok '}] {path.rsplit('/',1)[1]} {kind}:{name}")
    if fails:
        sys.exit(1)
    if dry:
        return
    for path, src in files.items():
        open(path, "w", encoding="utf-8").write(src)
        py_compile.compile(path, doraise=True)
        print("wrote+compiled:", path.rsplit('/',1)[1])


if __name__ == "__main__":
    main()
