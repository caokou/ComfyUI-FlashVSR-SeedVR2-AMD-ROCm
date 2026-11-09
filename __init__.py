"""lundao1yu0 · AMD ROCm VSR suite (FlashVSR + SeedVR2) — umbrella custom node.

Clone this repo directly into ComfyUI/custom_nodes/ and restart — no installer
needed. Two patched node packs are vendored side by side:

  - nodes are loaded from:
      flashvsr/               <- 1038lab/ComfyUI-FlashVSR (GPL-3.0) + our fixes
      seedvr2_videoupscaler/  <- numz/ComfyUI-SeedVR2_VideoUpscaler (Apache-2.0) + our fixes

Included fixes (see NOTICE.md / docs/BENCHMARK.md):
  1. FlashVSR LQ causal-cache order bug (all-platform facial flicker, every 4 frames)
  2. ROCm LCSA sparse-attention backend (gated by torch.version.hip; NVIDIA takes
     the upstream path unchanged)
  3. Chinese tooltips/descriptions + lundao1yu0 category branding

Licensing: the umbrella repo is GPL-3.0 (forced by the vendored GPL-3.0
FlashVSR node pack); the vendored SeedVR2 pack stays Apache-2.0.
"""
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_NODES = os.path.join(_HERE, "nodes")

# Parent package anchor: relative imports inside the packs (e.g. `from ..configs`)
# walk up to this name, so it must exist in sys.modules with the nodes/ search path.
_PARENT = "lundao1yu0_vsr"
if _PARENT not in sys.modules:
    _anchor = types.ModuleType(_PARENT)
    _anchor.__path__ = [_NODES]
    sys.modules[_PARENT] = _anchor


def _load_pack(rel):
    """Load a vendored pack (under nodes/) as an isolated package."""
    path = os.path.join(_NODES, rel)
    spec = importlib.util.spec_from_file_location(
        "lundao1yu0_vsr." + rel,
        os.path.join(path, "__init__.py"),
        submodule_search_locations=[path],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


print("\n[lundao1yu0 VSR suite] loading vendored node packs ...")

_flashvsr = _load_pack("flashvsr")
_seedvr2 = _load_pack("seedvr2_videoupscaler")

# ---- FlashVSR: classic V1 mappings, exported normally ------------------------
NODE_CLASS_MAPPINGS = dict(getattr(_flashvsr, "NODE_CLASS_MAPPINGS", {}) or {})
NODE_DISPLAY_NAME_MAPPINGS = dict(getattr(_flashvsr, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {})

# ---- SeedVR2: pure V3 (io.ComfyNode) classes; register them the same way
# ComfyUI's own V3 branch does (nodes.py: comfy_entrypoint -> get_node_list ->
# NODE_CLASS_MAPPINGS[schema.node_id] = cls). Done at import time because the
# module-level V1 export above takes precedence in ComfyUI's loader.
def _register_v3_nodes(pack_modname):
    registered = 0
    try:
        import nodes as _comfy_nodes  # ComfyUI runtime registry, safe at custom-node import time
        iface = importlib.import_module(pack_modname + ".src.interfaces")
        node_classes = [getattr(iface, a) for a in
                        ("SeedVR2VideoUpscaler", "SeedVR2LoadDiTModel",
                         "SeedVR2LoadVAEModel", "SeedVR2TorchCompileSettings")]
        for cls in node_classes:
            schema = cls.define_schema()
            cls.RELATIVE_PYTHON_MODULE = "custom_nodes.ComfyUI-FlashVSR-SeedVR2-AMD-ROCm"
            _comfy_nodes.NODE_CLASS_MAPPINGS[schema.node_id] = cls
            if getattr(schema, "display_name", None):
                _comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS[schema.node_id] = schema.display_name
            registered += 1
    except Exception as e:  # never break the other pack
        print(f"[lundao1yu0 VSR suite] WARNING: SeedVR2 V3 registration issue: {e}")
    return registered


_svr2_count = _register_v3_nodes("lundao1yu0_vsr.seedvr2_videoupscaler")

try:
    import torch as _torch
    _hip = bool(getattr(_torch.version, "hip", None))
except Exception:
    _hip = False

print(f"[lundao1yu0 VSR suite] FlashVSR nodes: {len(NODE_CLASS_MAPPINGS)} | "
      f"SeedVR2 nodes: {_svr2_count} | ROCm backend: {'active' if _hip else 'not present (upstream paths)'}\n")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
