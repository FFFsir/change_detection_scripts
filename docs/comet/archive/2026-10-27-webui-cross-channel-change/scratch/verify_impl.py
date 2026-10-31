#!/usr/bin/env python
"""Build 阶段独立复算与行为自检。

A2/A3 的数值断言使用本文件内的独立 numpy 实现，不调用被测模块的
``pair_distance``；被测实现只在需要核对 UI 口径时单独调用。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
APP = ROOT / "app" / "change_detection_webui.py"
TILES = ROOT / "data" / "spectral" / "tiles"

spec = importlib.util.spec_from_file_location("webui", APP)
webui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(webui)

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def load_view(tile: str, view: str) -> np.ndarray:
    return np.load(TILES / tile / "embeddings" / f"{view}.npy").astype(np.float64)


def indep_d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Independent d = (1 - cos) / 2, no shared code with the app."""
    nx = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    ny = np.sqrt(np.sum(y * y, axis=-1, keepdims=True))
    x = x / np.where(nx > 0, nx, 1.0)
    y = y / np.where(ny > 0, ny, 1.0)
    cos = np.sum(x * y, axis=-1)
    return np.clip((1.0 - cos) / 2.0, 0.0, 1.0)


def dataset():
    return webui.Dataset()


ds = dataset()

# ---- A2: same-day cross-channel vs dataset C0 ------------------------------
vis = load_view("C01", "sentinel2_visible")
nir = load_view("C01", "sentinel2_nir")
mask = ds.mask("C01", "sentinel2_visible")
d_indep = indep_d(vis[0], nir[0])
d_app, valid_app = webui.pair_distance(
    vis[0].astype(np.float32), nir[0].astype(np.float32), mask, use_mask=True
)
check("A2 被测 d 与独立 numpy 逐像素一致（≤1e-6）",
      bool(np.max(np.abs(d_app[valid_app] - d_indep[valid_app])) <= 1e-6),
      f"max|Δ| = {np.max(np.abs(d_app[valid_app] - d_indep[valid_app])):.2e}")

with np.load(TILES / "C01" / "distance_maps" / "sentinel2_nir.npz") as npz:
    c0 = npz["C0_epoch_0"].astype(np.float64)
ratio = c0[valid_app] / np.maximum(d_indep[valid_app], 1e-12)
check("A2 数据集 C0 = 2 × d（严格 2 倍）",
      bool(1.9999 <= ratio.min() and ratio.max() <= 2.0001),
      f"ratio ∈ [{ratio.min():.6f}, {ratio.max():.6f}]")

# ---- A3: cross-day residuals + symmetry ------------------------------------
def residual_arrays(tile: str, sensor: str, va: str, vb: str, ref: int, other: int):
    ea = load_view(tile, f"{sensor}_{va}")
    eb = load_view(tile, f"{sensor}_{vb}")
    base = indep_d(ea[ref], eb[ref])
    d_fwd = indep_d(ea[ref], eb[other])
    d_rev = indep_d(eb[ref], ea[other])
    return base, d_fwd - base, d_rev - base, (d_fwd - base + d_rev - base) / 2.0


for tile, sensor, va, vb in (
    ("C01", "sentinel2", "visible", "nir"),
    ("C01", "landsat", "visible", "nir"),
    ("V01", "sentinel2", "swir", "reflective_ir"),
):
    base, fwd, rev, sym = residual_arrays(tile, sensor, va, vb, 0, 1)
    asym = float(np.max(np.abs(fwd - rev)))
    check(f"A3 {tile}/{sensor} 对称化 = (正向+反向)/2",
          bool(np.allclose(sym, (fwd + rev) / 2.0, atol=1e-12, rtol=0)),
          f"max|Δ| = {np.max(np.abs(sym - (fwd + rev) / 2.0)):.2e}")
    check(f"A3 {tile}/{sensor} 残差 = 跨日期距离 − 同日基线",
          bool(np.allclose(fwd, indep_d(load_view(tile, f'{sensor}_{va}')[0],
                                        load_view(tile, f'{sensor}_{vb}')[1]) - base, atol=1e-12)))
    print(f"  信息：{tile}/{sensor} {va}|{vb} 双向像素级最大差 = {asym:.4f}")

# ---- A5: single-epoch Landsat views filtered for cross-day ----------------
single = ["C05", "C06", "C10", "C12", "C17", "C18", "C19", "C21"]
for tile in single:
    usable = ds.views(tile)
    ls_usable = [v for v in usable if v.startswith("landsat_")]
    check(f"A5 {tile} 跨日类型无 Landsat 视图", ls_usable == [],
          f"可用视图 = {usable}")
    all_views = ds.all_views(tile)
    ls_all = [v for v in all_views if v.startswith("landsat_")]
    check(f"A5 {tile} 同日类型仍列出 Landsat 视图", len(ls_all) == 5, str(ls_all))

# ---- A6: mask per sensor ---------------------------------------------------
for tile in ("C01", "C17", "V01"):
    ms = ds.mask(tile, "sentinel2_full")
    print(f"  信息：{tile} 掩膜像素 sentinel2={int(ms.sum())}")
    landsat_views = [v for v in ds.all_views(tile) if v.startswith("landsat_")]
    if landsat_views:
        ml = ds.mask(tile, "landsat_full")
        print(f"  信息：{tile} 掩膜像素 landsat={int(ml.sum())}")
check("A6 C01 两传感器掩膜不同",
      not np.array_equal(ds.mask("C01", "sentinel2_full"),
                         ds.mask("C01", "landsat_full")))
try:
    ds.mask("V01", "landsat_full")
    check("A6 缺失传感器掩膜必须报错", False, "未抛异常")
except FileNotFoundError:
    check("A6 缺失传感器掩膜必须报错", True, "FileNotFoundError")

# ---- threshold semantics ---------------------------------------------------
rng = np.random.default_rng(7)
q = np.abs(rng.normal(0.0, 0.12, size=(128, 128))).astype(np.float32)
valid = np.ones((128, 128), dtype=bool)
scale = webui.threshold_scale(q, valid)
hot_top = int(np.count_nonzero(q[valid] > scale["upper"]))
check("阈值上界处高亮数为 0", hot_top == 0, f"upper={scale['upper']}, hot={hot_top}")
check("阈值左端 > 0", scale["lower"] > 0, f"lower={scale['lower']}")

# ---- signed rendering ------------------------------------------------------
r = (rng.uniform(-1.0, 1.0, size=(128, 128))).astype(np.float32)
rgb_signed = webui.render_rgb_signed(r, valid, 0.8, "binary")
hot = valid & (np.abs(r) > 0.8)
cold = valid & ~hot
check("带符号渲染：高亮按 |r| 判定",
      bool(np.all(rgb_signed[hot] == webui.HIGHLIGHT.astype(np.uint8)))
      and bool(np.all(rgb_signed[cold] == webui.BINARY_BACKGROUND)),
      f"高亮像素数 = {int(hot.sum())}，非高亮 = {int(cold.sum())}")
rgb_masked = webui.render_rgb_signed(r, np.zeros_like(valid), 0.1, "overlay")
check("带符号渲染：掩膜外为中性灰",
      bool(np.all(rgb_masked == webui.MASK_NEUTRAL)))

# ---- A7: no truth/accuracy wording in the UI source -----------------------
src = APP.read_text(encoding="utf-8")
banned = ["准确率", "精度", "置信度", "accuracy", "ground truth", "真值"]
hits = [w for w in banned if w.lower() in src.lower()]
check("A7 源码不含真值/精度/置信度表述", hits == [], str(hits))
check("A7 分布外标记存在", "分布外（波段消融诊断）" in src
      and "分布内 vs 分布外混合对照" in src)

print()
if failures:
    print("未通过项：", failures)
    sys.exit(1)
print("独立复算与行为自检全部通过。")
