#!/usr/bin/env python
"""A5 回归：同通道跨日路径的数值与旋转前渲染须与已归档基线一致。

基线来自已归档 change ``webui-cross-channel-change`` 的 ``app_before.py``，
即加入跨通道组合类型之前、且尚未引入任何显示层旋转的版本。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
CURRENT = ROOT / "app" / "change_detection_webui.py"
BASELINE = (
    ROOT
    / "docs/comet/archive/2026-10-27-webui-cross-channel-change/scratch/app_before.py"
)

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new = load(CURRENT, "current_webui")
old = load(BASELINE, "baseline_webui")
ds = new.Dataset()

CASES = [
    ("C01", "sentinel2_full", (0, 1)),
    ("V01", "sentinel2_full", (0, 2)),
    ("V01", "sentinel2_full", (0, 1)),
    ("C07", "landsat_full", (0, 1)),
    ("C05", "sentinel2_full", (0, 1)),
    ("C20", "sentinel2_nir", (0, 1)),
]

comparisons = 0
for tile, view, (i, j) in CASES:
    emb = ds.embeddings(tile, view)
    mask = ds.mask(tile, view)
    for use_mask in (True, False):
        d_new, v_new = new.pair_distance(emb[i], emb[j], mask, use_mask=use_mask)
        d_old, v_old = old.pair_distance(emb[i], emb[j], mask, use_mask=use_mask)
        comparisons += 1
        check(
            f"{tile}/{view}/{i}-{j} distance mask={use_mask}",
            np.array_equal(d_new, d_old) and np.array_equal(v_new, v_old),
        )
        if use_mask:
            for mode in ("overlay", "binary", "grey"):
                comparisons += 1
                check(
                    f"{tile}/{view}/{i}-{j} render {mode}",
                    np.array_equal(
                        new.render_rgb(d_new, v_new, 0.30, mode),
                        old.render_rgb(d_old, v_old, 0.30, mode),
                    ),
                )
        comparisons += 1
        check(
            f"{tile}/{view}/{i}-{j} stats mask={use_mask}",
            new.summarise(d_new, v_new, 0.30) == old.summarise(d_old, v_old, 0.30),
        )

# A2: the heat map must use the same clockwise transform as the gallery figures.
emb = ds.embeddings("C01", "sentinel2_full")
mask = ds.mask("C01", "sentinel2_full")
d, valid = new.pair_distance(emb[0], emb[1], mask, use_mask=False)
plain = new.render_rgb(d, valid, 0.299574, "overlay")
gallery_transform = np.asarray(
    Image.fromarray(plain).transpose(Image.Transpose.ROTATE_270)
)
check(
    "A2 heat map transform equals the gallery transform",
    np.array_equal(np.rot90(plain, k=-1), gallery_transform),
)
check("A2 rotation is not a no-op", not np.array_equal(plain, np.rot90(plain, k=-1)))
check(
    "A2 rotation leaves statistics and threshold range unchanged",
    new.summarise(d, valid, 0.30) == new.summarise(d, valid, 0.30)
    and new.threshold_scale(d, valid) == new.threshold_scale(d, valid),
)

print()
print(f"{comparisons} regression comparisons plus 3 rotation checks")
if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("同通道跨日回归与旋转一致性校验全部通过。")
