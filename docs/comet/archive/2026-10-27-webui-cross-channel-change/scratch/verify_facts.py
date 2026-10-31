#!/usr/bin/env python
"""Shape 阶段数据事实自检（只读，不修改 data/ 下任何文件）。

逐项核实需求文档中给出的数据事实：
  F1  瓦片/epoch 规模：S2 33 瓦（32 两期 + V01 三期 = 67 epoch）；Landsat 20 瓦（12 两期 + 8 单期 = 32 epoch）
  F2  视图清单：S2 = full/visible/nir/swir/reflective_ir；Landsat = full/visible/nir/swir/thermal
  F3  同一瓦片同一传感器：各 view 的 date_utc 与 parent_raw_index 完全一致
  F4  common_mask 同一瓦片同一传感器内对所有 view 相同；跨传感器不同
  F5  gallery/index.json 以 (tile, source, date) 为键共 98 条；V01 的 2024-09-01 无图
  F6  数据集自带 C0/C1 用 1-cos（为本实现 d=(1-cos)/2 的 2 倍）
  F7  组合规模（只计四个严格单通道）：同日跨通道 594（S2 402 / LS 192）、
      跨日同通道 188（140/48）、跨日跨通道 282（210/72）；跨日类 45 个多期组
  F8  跨模态残差方向不对称（抽样）；带符号残差的负值占比（为决策 Q1/Q3 提供依据）
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5] / "data" / "spectral"
TILES_ROOT = ROOT / "tiles"

STRICT = {
    "sentinel2": ["visible", "nir", "swir", "reflective_ir"],
    "landsat": ["visible", "nir", "swir", "thermal"],
}

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def frames_of(tile: str, view: str) -> list[dict]:
    meta = json.loads((TILES_ROOT / tile / "embeddings" / f"{view}_index.json").read_text(encoding="utf-8"))
    return meta.get("frames", [])


def main() -> int:
    tiles = sorted(p.name for p in TILES_ROOT.iterdir() if p.is_dir())
    print(f"tile count = {len(tiles)}  ({', '.join(tiles)})")

    # ---- F1 / F2 / F3 ------------------------------------------------------
    s2_tiles: dict[str, list[str]] = {}   # tile -> dates
    ls_tiles: dict[str, list[str]] = {}
    view_by_sensor: dict[str, set[str]] = {"sentinel2": set(), "landsat": set()}
    inconsistent = []
    for tile in tiles:
        views = sorted(p.stem for p in (TILES_ROOT / tile / "embeddings").glob("*.npy"))
        sensors = sorted({v.split("_", 1)[0] for v in views})
        for sensor in sensors:
            sviews = [v for v in views if v.startswith(sensor + "_")]
            view_by_sensor[sensor].update(v.split("_", 1)[1] for v in sviews)
            ref_dates = None
            ref_idx = None
            for v in sviews:
                frames = frames_of(tile, v)
                dates = [f.get("date_utc") for f in frames]
                idxs = [f.get("parent_raw_index") for f in frames]
                if ref_dates is None:
                    ref_dates, ref_idx = dates, idxs
                elif dates != ref_dates or idxs != ref_idx:
                    inconsistent.append(f"{tile}/{sensor}/{v}")
            dates = ref_dates or []
            target = s2_tiles if sensor == "sentinel2" else ls_tiles
            target[tile] = dates
    check("F2 S2 视图集合", view_by_sensor["sentinel2"] == {"full", "visible", "nir", "swir", "reflective_ir"},
          str(sorted(view_by_sensor["sentinel2"])))
    check("F2 Landsat 视图集合", view_by_sensor["landsat"] == {"full", "visible", "nir", "swir", "thermal"},
          str(sorted(view_by_sensor["landsat"])))
    check("F3 同传感器各 view 日期/parent_raw_index 一致", not inconsistent, str(inconsistent[:5]))

    s2_multi = {t: d for t, d in s2_tiles.items() if len(d) >= 2}
    ls_multi = {t: d for t, d in ls_tiles.items() if len(d) >= 2}
    s2_epochs = sum(len(d) for d in s2_tiles.values())
    ls_epochs = sum(len(d) for d in ls_tiles.values())
    n_two = sum(1 for d in s2_tiles.values() if len(d) == 2)
    check("F1 S2 = 33 瓦（32 两期 + V01 三期，67 epoch）",
          len(s2_tiles) == 33 and n_two == 32 and len(s2_tiles.get("V01", [])) == 3 and s2_epochs == 67,
          f"S2 {len(s2_tiles)} 瓦 / 两期 {n_two} / epoch {s2_epochs} / V01 {len(s2_tiles.get('V01', []))} 期")
    check("F1 Landsat = 20 瓦（12 两期 + 8 单期，32 epoch）",
          len(ls_tiles) == 20 and len(ls_multi) == 12 and ls_epochs == 32,
          f"LS {len(ls_tiles)} 瓦 / 多期 {len(ls_multi)} / epoch {ls_epochs}")
    single_ls = sorted(set(ls_tiles) - set(ls_multi))
    check("F1 单期 Landsat 瓦片数 = 8", len(single_ls) == 8, str(single_ls))

    # ---- F4 common_mask ----------------------------------------------------
    mask_bad = []
    identical_cross_sensor = []
    sensor_masks: dict[tuple[str, str], np.ndarray] = {}
    for tile in tiles:
        by_sensor: dict[str, list[np.ndarray]] = {}
        for p in sorted((TILES_ROOT / tile / "distance_maps").glob("*.npz")):
            sensor = p.name.split("_", 1)[0]
            with np.load(p) as data:
                if "common_mask" in data:
                    by_sensor.setdefault(sensor, []).append(data["common_mask"])
        for sensor, masks in by_sensor.items():
            sensor_masks[(tile, sensor)] = masks[0].astype(bool)
            if any(not np.array_equal(m, masks[0]) for m in masks[1:]):
                mask_bad.append(f"{tile}/{sensor}")
        if len(by_sensor) == 2:
            s2m = by_sensor.get("sentinel2", [None])[0]
            lsm = by_sensor.get("landsat", [None])[0]
            if s2m is not None and lsm is not None and np.array_equal(s2m, lsm):
                identical_cross_sensor.append(tile)
    check("F4 common_mask 同传感器内一致", not mask_bad, str(mask_bad[:5]))
    print(f"  信息：跨传感器掩膜数值恰好相同的瓦片 = {identical_cross_sensor or '无'}"
          f"（仍按传感器分别读取，不跨传感器复用）")
    c01s = int(sensor_masks[("C01", "sentinel2")].sum())
    c01l = int(sensor_masks[("C01", "landsat")].sum())
    print(f"  信息：C01 掩膜像素 sentinel2 = {c01s}, landsat = {c01l}")

    # ---- F5 gallery ---------------------------------------------------------
    entries = json.loads((ROOT / "gallery" / "index.json").read_text(encoding="utf-8"))
    keys = {(e["id"], e["source"], e["date"]) for e in entries if e.get("image")}
    check("F5 gallery 唯一键 = 98", len(keys) == 98, str(len(keys)))
    check("F5 V01 sentinel2 2024-09-01 无图",
          ("V01", "sentinel2", "2024-09-01") not in keys,
          "")
    missing_expected = []
    source_of: dict[tuple[str, str], str] = {}
    for tile, dates, sensor in (
        [(t, d, "sentinel2") for t, d in s2_tiles.items()]
        + [(t, d, "landsat") for t, d in ls_tiles.items()]
    ):
        meta = json.loads((TILES_ROOT / tile / "embeddings" / f"{sensor}_full_index.json").read_text(encoding="utf-8"))
        source = meta.get("encoder_source", sensor)
        source_of[(tile, sensor)] = source
        for d in dates:
            if (tile, source, d) not in keys:
                missing_expected.append(f"{tile}/{source}/{d}")
    check("F5 除 V01 2024-09-01 外全部 epoch 均有图", missing_expected == ["V01/sentinel2/2024-09-01"],
          str(missing_expected))
    odd_sources = {f"{t}:{s}" for (t, s), src in source_of.items() if src != s}
    print(f"  信息：encoder_source 与传感器名不同的瓦片 = {sorted(odd_sources) or '无'}")

    # ---- F6 C0/C1 口径 ------------------------------------------------------
    npz = np.load(next((TILES_ROOT / "C01" / "distance_maps").glob("sentinel2_*.npz")))
    print(f"  C01 sentinel2 distance_maps keys = {sorted(npz.files)}")

    def load_view(tile: str, view: str) -> np.ndarray:
        return np.load(TILES_ROOT / tile / "embeddings" / f"{view}.npy").astype(np.float32)

    def dmap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        ua = a / np.where(na > 0, na, 1.0)[..., None]
        ub = b / np.where(nb > 0, nb, 1.0)[..., None]
        return np.clip((1.0 - np.clip(np.sum(ua * ub, axis=-1), -1.0, 1.0)) / 2.0, 0.0, 1.0)

    vis = load_view("C01", "sentinel2_visible")
    ir_views = {n: load_view("C01", f"sentinel2_{n}") for n in ("nir", "swir", "reflective_ir")}
    mask_s2 = sensor_masks[("C01", "sentinel2")]

    def ratio_vs(key: str, d: np.ndarray) -> tuple[float, float]:
        ref = npz[key].astype(np.float64)
        both = mask_s2 & np.isfinite(ref)
        denom = np.where(d[both] > 1e-12, d[both], 1e-12)
        ratio = ref[both] / denom
        return float(np.min(ratio)), float(np.max(ratio))

    for ir_name, ir in ir_views.items():
        d_same0 = dmap(vis[0], ir[0])
        lo, hi = ratio_vs("C0_epoch_0", d_same0)
        print(f"  2×d(visible@0, {ir_name}@0) vs C0_epoch_0: ratio ∈ [{lo:.6f}, {hi:.6f}]")
    d_vis = dmap(vis[0], vis[1])
    lo, hi = ratio_vs("same_modality_visible", d_vis)
    print(f"  2×d(visible@0, visible@1) vs same_modality_visible: ratio ∈ [{lo:.6f}, {hi:.6f}]")
    for ir_name, ir in ir_views.items():
        d_fwd = dmap(vis[0], ir[1])
        lo, hi = ratio_vs("C1_visible_before_ir_after", d_fwd)
        print(f"  2×d(visible@0, {ir_name}@1) vs C1_visible_before_ir_after: ratio ∈ [{lo:.6f}, {hi:.6f}]")

    # ---- F7 组合规模 ---------------------------------------------------------
    same_day = same_ch = cross_day_cross_ch = 0
    per_source = Counter()
    multi_groups = 0
    for tile, dates in list(s2_tiles.items()) + list(ls_tiles.items()):
        sensor = "sentinel2" if tile in s2_tiles else "landsat"
        k = len(STRICT[sensor])
        nd = len(dates)
        per_source[sensor] += 0
        if nd >= 2:
            multi_groups += 1
        same_day += nd * (k * (k - 1) // 2)
        same_ch += (k * nd * (nd - 1) // 2) if nd >= 2 else 0
        cross_day_cross_ch += ((k * (k - 1) // 2) * (nd * (nd - 1) // 2)) if nd >= 2 else 0
    check("F7 同日跨通道 = 594", same_day == 594, f"实际 {same_day}")
    check("F7 跨日同通道 = 188", same_ch == 188, f"实际 {same_ch}")
    check("F7 跨日跨通道 = 282", cross_day_cross_ch == 282, f"实际 {cross_day_cross_ch}")
    check("F7 跨日类多期组 = 45", multi_groups == 45, f"实际 {multi_groups}")

    # ---- F9 含 full 的组合规模（Q2 决策后新口径：5 个视图全部可选） -----------
    same_day5 = same_ch5 = cdc5 = 0
    for tile, dates, sensor in (
        [(t, d, "sentinel2") for t, d in s2_tiles.items()]
        + [(t, d, "landsat") for t, d in ls_tiles.items()]
    ):
        k = 5  # full + 4 严格单通道
        nd = len(dates)
        same_day5 += nd * (k * (k - 1) // 2)
        if nd >= 2:
            same_ch5 += k * (nd * (nd - 1) // 2)
            cdc5 += (k * (k - 1) // 2) * (nd * (nd - 1) // 2)
    print(f"F9 含 full 组合规模：同日跨通道 = {same_day5}，跨日同通道 = {same_ch5}，跨日跨通道 = {cdc5}")
    v01_views = sorted(p.stem for p in (TILES_ROOT / "V01" / "embeddings").glob("*.npy"))
    print(f"F9 V01 视图 = {v01_views}")

    # ---- F8 残差不对称与符号（抽样 3 个双期组） --------------------------------
    samples = [("C01", "sentinel2", "visible", "nir"), ("C01", "landsat", "visible", "nir"),
               ("V01", "sentinel2", "swir", "reflective_ir")]
    print("\nF8 抽样（带符号残差 = 跨日期距离 − 同日基线，掩膜内）：")
    for tile, sensor, va, vb in samples:
        dates = (s2_tiles if sensor == "sentinel2" else ls_tiles)[tile]
        if len(dates) < 2:
            continue
        ema = np.load(TILES_ROOT / tile / "embeddings" / f"{sensor}_{va}.npy").astype(np.float32)
        emb_ = np.load(TILES_ROOT / tile / "embeddings" / f"{sensor}_{vb}.npy").astype(np.float32)
        m = sensor_masks[(tile, sensor)]

        def dist(a, b):
            na = np.linalg.norm(a, axis=-1)
            nb = np.linalg.norm(b, axis=-1)
            ua = a / np.where(na > 0, na, 1.0)[..., None]
            ub = b / np.where(nb > 0, nb, 1.0)[..., None]
            return np.clip((1.0 - np.clip(np.sum(ua * ub, axis=-1), -1.0, 1.0)) / 2.0, 0.0, 1.0)

        base = dist(ema[0], emb_[0])[m]
        d_fwd = dist(ema[0], emb_[1])[m]   # A@t0 vs B@t1
        d_rev = dist(ema[1], emb_[0])[m]   # A@t1 vs B@t0
        resid_fwd = d_fwd - base
        resid_rev = d_rev - base
        print(f"  {tile}/{sensor} {va}|{vb}  基线均值 {base.mean():.4f}  "
              f"d(A0,B1) 均值 {d_fwd.mean():.4f}  d(A1,B0) 均值 {d_rev.mean():.4f}  "
              f"|方向差| 最大 {np.abs(d_fwd - d_rev).max():.4f}")
        for tag, r in (("正向", resid_fwd), ("反向", resid_rev)):
            neg = float((r < 0).mean()) * 100
            print(f"    {tag}残差: 均值 {r.mean():+.4f}  最小 {r.min():+.4f}  最大 {r.max():+.4f}  负值占比 {neg:.1f}%")
        del ema, emb_

    print()
    if failures:
        print("未通过项：", failures)
        return 1
    print("全部事实自检通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
