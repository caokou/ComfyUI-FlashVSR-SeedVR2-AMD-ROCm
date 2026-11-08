#!/usr/bin/env python3
"""Fix causal cache ordering in Buffer_LQ4x_Proj — run ON THE SERVER.

Root cause of the periodic (every 4 video frames) face-element jumping:
the node's Buffer_LQ4x_Proj updates self.cache BEFORE calling the causal
conv, so the conv's "history" is the CURRENT clip's tail instead of the
PREVIOUS clip's tail. Upstream Causal_LQ4x_Proj (examples/WanVSR/utils/
utils.py) uses the conv first, then stores — restoring temporal continuity
of the LQ condition features that are injected into every DiT block.

CausalConv3d(x, cache) concatenates cache in front of x along time, so
storing-then-using makes the conv see the present as its own past.

3 fixes (all aligned verbatim with upstream order):
  1) forward()      conv1: use-then-store
  2) forward()      conv2: store only at i==0; else use-then-store
  3) stream_forward() else-branch conv1+conv2: use-then-store

Idempotent; .orig backups; py_compile check.
"""
import os
import py_compile
import shutil
import sys

TARGET = os.path.expanduser(
    "~/comfyui-rocm/custom_nodes/flashvsr/FlashVSR/models/utils.py"
)
DRY = "--dry" in sys.argv

# --- fix 1+2: forward() conv1 & conv2 (node order -> upstream order) ---
FWD_OLD = """            x = self.pixel_shuffle(video[:,:,i*4:(i+1)*4,:,:])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
"""
FWD_NEW = """            x = self.pixel_shuffle(video[:,:,i*4:(i+1)*4,:,:])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
"""
FWD2_OLD = """            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            if i == 0:
                continue
            x = self.conv2(x, self.cache['conv2'])
"""
FWD2_NEW = """            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            if i == 0:
                self.cache['conv2'] = cache2_x
                continue
            x = self.conv2(x, self.cache['conv2'])
            self.cache['conv2'] = cache2_x
"""

# --- fix 3: stream_forward() else branch (the path the pipeline actually uses) ---
ST_OLD = """            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            x = self.conv2(x, self.cache['conv2'])
"""
ST_NEW = """            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv2(x, self.cache['conv2'])
            self.cache['conv2'] = cache2_x
"""


def patch(src, anchor, repl, label):
    if repl in src:
        print(f"[skip] {label}: already applied")
        return src, True
    n = src.count(anchor)
    if n != 1:
        print(f"[FAIL] {label}: anchor matched {n} times (need exactly 1)")
        return src, False
    print(f"[{'dry ' if DRY else ' ok '}] {label}")
    return (src if DRY else src.replace(anchor, repl)), True


def main():
    print("== Buffer_LQ4x_Proj causal-cache-order fixer ==")
    src = open(TARGET, encoding="utf-8").read()
    ok = True
    src, r = patch(src, FWD_OLD, FWD_NEW, "forward() conv1 use-then-store")
    ok &= r
    src, r = patch(src, FWD2_OLD, FWD2_NEW, "forward() conv2 use-then-store")
    ok &= r
    src, r = patch(src, ST_OLD, ST_NEW, "stream_forward() conv1+conv2 use-then-store")
    ok &= r
    if not ok:
        sys.exit("anchor mismatch — file NOT modified")
    if not DRY:
        if not os.path.exists(TARGET + ".orig"):
            shutil.copy2(TARGET, TARGET + ".orig")
            print("[ ok ] backup .orig created")
        open(TARGET, "w", encoding="utf-8").write(src)
        py_compile.compile(TARGET, doraise=True)
        print("py_compile: OK")


if __name__ == "__main__":
    main()
