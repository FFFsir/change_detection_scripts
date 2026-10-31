#!/usr/bin/env python
"""AEF unsupervised change-detection viewer.

Implements the unsupervised branch of AlphaEarth Foundations (supplement S4.1,
equation 8): two per-pixel 64-d embeddings are L2 normalized and compared with

    d = (1 - cos) / 2,   d in [0, 1]

The result is rendered as a 128x128 heat map beside the original gallery figures
of the two frames, with a threshold slider whose range adapts to each image pair
as ``(0, x]`` (the top of the range yields zero highlighted pixels).

Run with the project virtual environment:

    .venv/Scripts/python.exe app/change_detection_webui.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from nicegui import ui

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "spectral"
TILES_ROOT = DATA_ROOT / "tiles"
GALLERY_INDEX = DATA_ROOT / "gallery" / "index.json"

MASK_NEUTRAL = np.array([110, 112, 118], dtype=np.uint8)
HIGHLIGHT = np.array([235, 45, 45], dtype=np.float32)
BINARY_BACKGROUND = np.array([26, 28, 36], dtype=np.uint8)

# One embedding view is (frames, 128, 128, 64) float32 = 8 MiB per frame.
EMBEDDING_CACHE_SIZE = 8

# Inferno-like colour ramp, 11 stops.
_RAMP = np.array(
    [
        [0, 0, 4],
        [22, 11, 57],
        [66, 10, 104],
        [106, 23, 110],
        [147, 38, 103],
        [188, 55, 84],
        [221, 81, 58],
        [243, 120, 25],
        [252, 165, 10],
        [246, 215, 70],
        [252, 255, 164],
    ],
    dtype=np.float32,
)

MODES = {
    "overlay": "半透明红色叠加",
    "binary": "二值掩膜",
    "grey": "灰底彩色高亮",
}


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #
class Dataset:
    """Read-only access to the embedding package."""

    def __init__(self, root: Path = TILES_ROOT) -> None:
        self.root = root
        self._emb_cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._meta_cache: dict[tuple[str, str], dict] = {}
        self._mask_cache: dict[tuple[str, str], np.ndarray] = {}
        self._manifest_cache: dict[str, dict] = {}
        self._views_cache: dict[str, tuple[str, ...]] = {}
        self.gallery = self._load_gallery()

    @staticmethod
    def _load_gallery() -> dict[tuple[str, str, str], str]:
        if not GALLERY_INDEX.exists():
            return {}
        entries = json.loads(GALLERY_INDEX.read_text(encoding="utf-8"))
        out: dict[tuple[str, str, str], str] = {}
        for entry in entries:
            image = entry.get("image")
            if not image:
                continue
            out[(entry["id"], entry["source"], entry["date"])] = str(DATA_ROOT.parent / image)
        return out

    def tiles(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def manifest(self, tile: str) -> dict:
        if tile not in self._manifest_cache:
            path = self.root / tile / "tile_manifest.json"
            self._manifest_cache[tile] = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            )
        return self._manifest_cache[tile]

    def all_views(self, tile: str) -> list[str]:
        return sorted(p.stem for p in (self.root / tile / "embeddings").glob("*.npy"))

    def frame_total(self, tile: str, view: str) -> int:
        """Frame count from the index JSON, without loading the array."""
        try:
            return len(self.frame_meta(tile, view).get("frames", []))
        except (OSError, ValueError, json.JSONDecodeError):
            return 0

    def views(self, tile: str) -> list[str]:
        """Views usable for a two-frame comparison.

        Some Landsat views only carry a single frame; comparing a frame with
        itself would silently produce an all-zero "no change" map, so they are
        excluded from selection.
        """
        cached = self._views_cache.get(tile)
        if cached is not None:
            return list(cached)
        usable = [n for n in self.all_views(tile) if self.frame_total(tile, n) >= 2]
        self._views_cache[tile] = tuple(usable)
        return list(usable)

    def skipped_views(self, tile: str) -> list[str]:
        return sorted(set(self.all_views(tile)) - set(self.views(tile)))

    def frame_meta(self, tile: str, view: str) -> dict:
        key = (tile, view)
        if key not in self._meta_cache:
            path = self.root / tile / "embeddings" / f"{view}_index.json"
            self._meta_cache[key] = json.loads(path.read_text(encoding="utf-8"))
        return self._meta_cache[key]

    def embeddings(self, tile: str, view: str) -> np.ndarray:
        key = (tile, view)
        if key in self._emb_cache:
            self._emb_cache.move_to_end(key)
            return self._emb_cache[key]
        path = self.root / tile / "embeddings" / f"{view}.npy"
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 4 or arr.shape[-1] != 64:
            raise ValueError(f"unexpected embedding shape {arr.shape} in {path}")
        self._emb_cache[key] = arr
        while len(self._emb_cache) > EMBEDDING_CACHE_SIZE:
            self._emb_cache.popitem(last=False)
        return arr

    def mask(self, tile: str, view: str) -> np.ndarray:
        """Common mask of the sensor that produced ``view``.

        Each sensor keeps its own mask under distance_maps and the masks differ
        (C01: 15971 vs 16384 pixels), so selection follows the view's sensor
        prefix. There is no fallback to another sensor's mask and no all-true
        fallback: either the right mask exists or this raises.
        """
        sensor = view.split("_", 1)[0]
        key = (tile, sensor)
        if key in self._mask_cache:
            return self._mask_cache[key]
        for path in sorted((self.root / tile / "distance_maps").glob(f"{sensor}_*.npz")):
            with np.load(path) as data:
                if "common_mask" in data:
                    mask = data["common_mask"].astype(bool)
                    self._mask_cache[key] = mask
                    return mask
        raise FileNotFoundError(
            f"没有为 {tile} 的 {sensor} 传感器找到 common_mask"
            f"（期望 {self.root / tile / 'distance_maps' / (sensor + '_*.npz')}）"
        )

    def gallery_candidates(self, tile: str, view: str) -> list[str]:
        """Existing gallery files for this tile and sensor, for diagnostics."""
        source = self.frame_meta(tile, view).get("encoder_source", view.split("_", 1)[0])
        return sorted(
            p.name for p in (DATA_ROOT / "gallery").glob(f"{tile}_{source}_*.png")
        )

    def image_entry(self, tile: str, view: str, date: str) -> tuple[str | None, str]:
        """Return ``(path_or_None, detail)`` where detail describes what is missing.

        Gallery files are named ``<tile>_<source>_<frame_index>.png``, so a date
        alone cannot be turned into a file name; ``detail`` reports the naming
        pattern and the files that do exist.
        """
        source = self.frame_meta(tile, view).get("encoder_source", view.split("_", 1)[0])
        path = self.gallery.get((tile, source, date))
        if path is not None and Path(path).exists():
            return path, ""
        candidates = self.gallery_candidates(tile, view)
        names = ", ".join(candidates) if candidates else "无"
        if path is None:
            detail = (
                f"gallery/index.json 中没有 ({tile}, {source}, {date}) 的记录\n"
                f"gallery 命名模式 = {tile}_{source}_<帧序号>.png\n"
                f"该瓦片现有图像 = {names}"
            )
        else:
            detail = (
                f"index.json 有该日期的记录，但文件不存在\n"
                f"记录路径 = {path}\n"
                f"gallery 命名模式 = {tile}_{source}_<帧序号>.png\n"
                f"该瓦片现有图像 = {names}"
            )
        return None, detail


DATASET = Dataset()


# --------------------------------------------------------------------------- #
# computation
# --------------------------------------------------------------------------- #
def pair_distance(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, use_mask: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(d, valid)`` for two (128, 128, 64) embedding frames.

    ``d = (1 - cos) / 2`` follows AEF equation (8), so 0 means identical
    directions and 1 means opposite directions. Pixels failing the mask, with a
    zero-norm vector, or with a non-finite component are marked invalid and set
    to 0.

    ``use_mask=False`` ignores ``common_mask`` and only excludes pixels whose
    embedding is unusable. The embeddings outside the mask are ordinary vectors
    (unit norm, finite), so they colourise normally.
    """
    finite_x = np.isfinite(x).all(axis=-1)
    finite_y = np.isfinite(y).all(axis=-1)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    usable = finite_x & finite_y & (norm_x > 0) & (norm_y > 0)
    valid = (mask & usable) if use_mask else usable
    unit_x = x / np.where(norm_x > 0, norm_x, 1.0)[..., None]
    unit_y = y / np.where(norm_y > 0, norm_y, 1.0)[..., None]
    cos = np.clip(np.sum(unit_x * unit_y, axis=-1), -1.0, 1.0)
    d = np.clip((1.0 - cos) / 2.0, 0.0, 1.0).astype(np.float32)
    return np.where(valid, d, 0.0), valid


def colourise(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    pos = t * (len(_RAMP) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.minimum(lo + 1, len(_RAMP) - 1)
    frac = (pos - lo)[..., None]
    return (_RAMP[lo] * (1.0 - frac) + _RAMP[hi] * frac).astype(np.uint8)


def render_rgb(d: np.ndarray, valid: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    """Build the RGB image for one display mode."""
    base = colourise(d)
    hot = valid & (d > threshold)

    if mode == "binary":
        rgb = np.broadcast_to(BINARY_BACKGROUND, d.shape + (3,)).copy()
        rgb[hot] = HIGHLIGHT.astype(np.uint8)
    elif mode == "grey":
        grey = (np.clip(d, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = np.stack([grey, grey, grey], axis=-1)
        rgb[hot] = base[hot]
    else:  # overlay
        rgb = base.copy()
        blend = base[hot].astype(np.float32) * 0.45 + HIGHLIGHT * 0.55
        rgb[hot] = blend.astype(np.uint8)

    rgb[~valid] = MASK_NEUTRAL
    return rgb


def png_data_url(rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def threshold_scale(d: np.ndarray, valid: np.ndarray) -> dict:
    """Compute the adaptive, left-open threshold range ``(0, x]``.

    ``x`` is the smallest multiple of the step that is strictly greater than the
    largest distance, so the top of the range yields zero highlighted pixels.
    The left end is excluded: the smallest selectable value is one step, so the
    threshold can never be 0.
    """
    values = d[valid]
    maximum = float(values.max()) if values.size else 0.0

    step = maximum / 100.0 if maximum > 0 else 0.01
    # Keep the step readable; never let it drop below 1e-4.
    step = max(round(step, 6), 1e-4)

    if maximum <= 0:
        upper = step
    else:
        upper = np.ceil(maximum / step + 1e-9) * step
        upper = min(round(float(upper), 6), 1.0)
        # Guarantee count(d > upper) == 0 even after rounding.
        while upper <= maximum and upper < 1.0:
            upper = min(round(upper + step, 6), 1.0)
        if upper <= maximum:
            upper = 1.0

    lower = step
    if upper < lower:
        upper = lower

    # Precision must follow the step: a fixed 4 decimals would snap values off the
    # step grid whenever the step itself needs more digits (e.g. 0.003366).
    text = f"{step:.6f}".rstrip("0")
    decimals = max(2, min(6, len(text.split(".")[1]) if "." in text else 2))
    default = 0.30 if upper >= 0.30 else upper / 2.0
    default = float(np.clip(round(default / step) * step, lower, upper))
    default = round(default, decimals)

    return {
        "max": maximum,
        "step": step,
        "lower": lower,
        "upper": upper,
        "default": default,
        "decimals": decimals,
        "fmt": f"%.{decimals}f",
    }


def summarise(d: np.ndarray, valid: np.ndarray, threshold: float) -> dict:
    values = d[valid]
    total = int(values.size)
    hot = int(np.count_nonzero(values > threshold))
    return {
        "valid": total,
        "hot": hot,
        "ratio": (hot / total) if total else 0.0,
        "median": float(np.median(values)) if total else 0.0,
        "p95": float(np.percentile(values, 95)) if total else 0.0,
        "maximum": float(values.max()) if total else 0.0,
    }


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def frame_label(meta: dict, index: int) -> str:
    frames = meta.get("frames", [])
    date = frames[index].get("date_utc", "?") if index < len(frames) else "?"
    return f"帧 {index} · {date}"


def build_page(dataset: Dataset) -> None:
    tiles = dataset.tiles()
    if not tiles:
        ui.label(f"未在 {TILES_ROOT} 找到任何瓦片").classes("text-red-600")
        return

    default_tile = "C01" if "C01" in tiles else tiles[0]
    tile_options = {}
    for name in tiles:
        manifest = dataset.manifest(name)
        category = manifest.get("category") or "未分类"
        tile_options[name] = f"{name} · {category}"

    default_views = dataset.views(default_tile)
    if "sentinel2_full" in default_views:
        default_view = "sentinel2_full"
    elif default_views:
        default_view = default_views[0]
    else:
        default_view = ""

    state = {
        "tile": default_tile,
        "view": default_view,
        "pair": (0, 1),
        "threshold": 0.30,
        "mode": "overlay",
        "use_mask": True,
        "rescale": True,
        "scale": {
            "max": 0.0, "step": 0.01, "lower": 0.01, "upper": 1.0,
            "default": 0.30, "decimals": 2, "fmt": "%.2f",
        },
    }

    def current_meta() -> dict:
        return dataset.frame_meta(state["tile"], state["view"])

    def frame_count() -> int:
        return int(dataset.embeddings(state["tile"], state["view"]).shape[0])

    def compute() -> tuple[np.ndarray, np.ndarray]:
        emb = dataset.embeddings(state["tile"], state["view"])
        mask = dataset.mask(state["tile"], state["view"])
        i, j = state["pair"]
        return pair_distance(emb[i], emb[j], mask, use_mask=state["use_mask"])

    def rescale(d: np.ndarray, valid: np.ndarray) -> None:
        """Fit the threshold controls to the current image pair, range ``(0, x]``."""
        scale = threshold_scale(d, valid)
        state["scale"] = scale
        decimals = scale["decimals"]

        # The whole update runs under the guard: changing min/max makes NiceGUI's
        # Number sanitize the current value, and that write fires on_number which
        # would push the clamped old threshold back into the state. Holding the
        # guard here keeps the freshly computed default authoritative.
        syncing["active"] = True
        try:
            state["threshold"] = scale["default"]
            # Numeric props matter: a string min/max makes Quasar's slider compute
            # "min + ratio * (max - min)" as string concatenation, which throws
            # ``toFixed is not a function`` on drag and leaves the page inert.
            threshold_slider._props["min"] = scale["lower"]  # noqa: SLF001
            threshold_slider._props["max"] = scale["upper"]  # noqa: SLF001
            threshold_slider._props["step"] = scale["step"]  # noqa: SLF001
            threshold_slider.update()
            threshold_number.min = scale["lower"]
            threshold_number.max = scale["upper"]
            # ``step`` has no property setter in NiceGUI's Number: it is a plain
            # instance attribute, so the prop must be written directly to reach Vue.
            threshold_number._props["step"] = scale["step"]  # noqa: SLF001
            threshold_number.format = f"%.{decimals}f"
            threshold_slider.value = scale["default"]
            threshold_number.value = round(scale["default"], decimals)
            threshold_number.update()
        finally:
            syncing["active"] = False
        scale_label.set_text(
            f"当前量程 (0, {scale['upper']:.{decimals}f}]   步长 {scale['step']:.{decimals}f}   "
            f"最大距离 {scale['max']:.{decimals}f}"
        )

    heat_image = None
    colorbar = None
    stats_label = None
    meta_label = None
    image_0 = None
    image_1 = None
    caption_0 = None
    caption_1 = None
    missing_0 = None
    missing_1 = None
    tile_select = None
    view_select = None
    pair_select = None
    threshold_slider = None
    threshold_number = None
    mode_select = None
    mask_switch = None
    scale_label = None
    colorbar_cache = {"set": False}
    def refresh() -> None:
        if not state["view"]:
            skipped = dataset.skipped_views(state["tile"])
            skipped_note = f"\n被排除的视图 = {', '.join(skipped)}" if skipped else ""
            meta_label.set_text(
                f"瓦片 = {state['tile']}\n"
                f"该瓦片没有可用于两帧比较的视图（所有视图均不足 2 帧）{skipped_note}"
            )
            stats_label.set_text("暂无可比较的数据")
            heat_image.set_source("")
            for label, caption, missing in (
                (image_0, caption_0, missing_0),
                (image_1, caption_1, missing_1),
            ):
                label.set_visibility(False)
                missing.set_visibility(False)
                caption.set_text("")
            return
        meta = current_meta()
        d, valid = compute()
        if state.get("rescale"):
            state["rescale"] = False
            rescale(d, valid)
        rgb = render_rgb(d, valid, state["threshold"], state["mode"])
        heat_image.set_source(png_data_url(rgb))

        if colorbar is not None and not colorbar_cache["set"]:
            bar = np.linspace(0.0, 1.0, 512)[None, :]
            colorbar.set_source(png_data_url(colourise(bar)))
            colorbar_cache["set"] = True

        summary = summarise(d, valid, state["threshold"])
        scale = state["scale"]
        digits = scale["decimals"]
        scope = "common_mask 内有效像素" if state["use_mask"] else "全部像素（未应用 common_mask）"
        stats_label.set_text(
            f"阈值 = {state['threshold']:.{digits}f}   |   统计范围 = {scope}\n"
            f"超阈值像素 = {summary['hot']} / {summary['valid']}  ({summary['ratio'] * 100:.2f}%)\n"
            f"距离 中位数 = {summary['median']:.4f}   p95 = {summary['p95']:.4f}   最大 = {summary['maximum']:.4f}"
        )

        i, j = state["pair"]
        dates = [meta["frames"][k].get("date_utc", "?") for k in (i, j)]
        manifest = dataset.manifest(state["tile"])
        skipped = dataset.skipped_views(state["tile"])
        skipped_note = (
            f"\n已排除单帧视图 = {', '.join(skipped)}（无法构成两帧比较）" if skipped else ""
        )
        if state["use_mask"]:
            excluded = int((~valid).sum())
            mask_note = f"\ncommon_mask = 开启   掩膜外像素 = {excluded}（灰色，不参与统计与高亮）"
        else:
            mask_note = "\ncommon_mask = 关闭   全部像素参与着色与统计"
        meta_label.set_text(
            f"瓦片 = {state['tile']}   split = {manifest.get('split') or '未标注'}   "
            f"类别 = {manifest.get('category') or '未分类'}\n"
            f"视图 = {state['view']}   帧数 = {frame_count()}   比较帧 = {i} → {j}   "
            f"日期 = {dates[0]} → {dates[1]}{skipped_note}{mask_note}"
        )

        for slot, label, caption, missing, date in (
            (0, image_0, caption_0, missing_0, dates[0]),
            (1, image_1, caption_1, missing_1, dates[1]),
        ):
            path, detail = dataset.image_entry(state["tile"], state["view"], date)
            caption.set_text(f"帧 {state['pair'][slot]} · {date} · {state['view']}")
            if path is not None:
                label.set_source(path)
                label.set_visibility(True)
                missing.set_visibility(False)
            else:
                label.set_visibility(False)
                missing.set_text(f"缺少图像：{date}\n{detail}")
                missing.set_visibility(True)

    def rebuild_view_and_pair() -> None:
        views = dataset.views(state["tile"])
        if not views:
            state["view"] = ""
            view_select.set_options({}, value=None)
            pair_select.set_options({})
            pair_select.set_visibility(False)
            return
        if state["view"] not in views:
            state["view"] = views[0]
        view_select.set_options({v: v for v in views}, value=state["view"])
        rebuild_pairs()

    def rebuild_pairs() -> None:
        n = frame_count()
        if n < 2:
            state["pair"] = (0, 0)
            pair_select.set_options({})
            pair_select.set_visibility(False)
            return
        if n == 2:
            state["pair"] = (0, 1)
            pair_select.set_options({"0-1": "帧 0 → 1"})
            pair_select.set_visibility(False)
            return
        meta = current_meta()
        options = {}
        for i in range(n):
            for j in range(i + 1, n):
                options[f"{i}-{j}"] = f"{frame_label(meta, i)} → {frame_label(meta, j)}"
        default = f"0-{n - 1}"
        pair_select.set_options(options, value=default)
        pair_select.set_visibility(True)
        state["pair"] = (0, n - 1)

    def on_tile(event) -> None:
        if not event.value:
            return
        state["tile"] = event.value
        state["rescale"] = True
        rebuild_view_and_pair()
        refresh()

    def on_view(event) -> None:
        if not event.value:
            return
        state["view"] = event.value
        state["rescale"] = True
        rebuild_pairs()
        refresh()

    def on_pair(event) -> None:
        if not event.value:
            return
        state["pair"] = tuple(int(x) for x in str(event.value).split("-"))
        state["rescale"] = True
        refresh()

    syncing = {"active": False}

    def sync_number_text() -> None:
        """Push the clamped threshold into the numeric field with range precision.

        Assigning the same value is a no-op for NiceGUI's change detection, so the
        field is re-sent explicitly; otherwise the box keeps the raw text the user
        typed (``0``, ``-2``, ``999``) while the state is already clamped.
        """
        decimals = state["scale"]["decimals"]
        threshold_number.value = round(state["threshold"], decimals)
        threshold_number.update()

    def apply_threshold(value: float) -> None:
        """Clamp to the current left-open range, snap to the step, then sync both controls."""
        if syncing["active"]:
            return
        syncing["active"] = True
        scale = state["scale"]
        step = max(scale["step"], 1e-9)
        snapped = round(round(value / step) * step, scale["decimals"])
        clamped = float(np.clip(snapped, scale["lower"], scale["upper"]))
        state["threshold"] = clamped
        threshold_slider.value = clamped
        sync_number_text()
        syncing["active"] = False
        refresh()

    def on_slider(event) -> None:
        if event.value is not None:
            apply_threshold(float(event.value))

    def on_number(event) -> None:
        if event.value is None:
            syncing["active"] = True
            sync_number_text()
            syncing["active"] = False
            return
        try:
            value = float(event.value)
        except (TypeError, ValueError):
            syncing["active"] = True
            sync_number_text()
            syncing["active"] = False
            return
        apply_threshold(value)

    def on_mode(event) -> None:
        state["mode"] = event.value
        refresh()

    def on_mask_toggle(event) -> None:
        state["use_mask"] = bool(event.value)
        state["rescale"] = True
        refresh()

    with ui.row().classes("w-full items-start no-wrap gap-6 p-4"):
        with ui.column().classes("w-[360px] shrink-0 gap-3"):
            ui.label("AEF 无监督变化检测").classes("text-xl font-semibold")
            tile_select = ui.select(tile_options, value=state["tile"], label="瓦片").classes("w-full")
            tile_select.on_value_change(on_tile)

            view_select = ui.select({}, label="视图").classes("w-full")
            view_select.on_value_change(on_view)

            pair_select = ui.select({}, label="帧对").classes("w-full")
            pair_select.on_value_change(on_pair)

            ui.label("阈值（d = (1 − cos) / 2，量程随图像对自适应，左端不可取 0）").classes("text-sm text-gray-600")
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                threshold_slider = ui.slider(
                    min=0.01, max=1.0, step=0.01, value=state["threshold"]
                ).classes("flex-1")
                threshold_number = ui.number(
                    value=state["threshold"], step=0.01, format="%.4f"
                ).classes("w-28")
            threshold_slider.on_value_change(on_slider)
            threshold_number.on_value_change(on_number)
            scale_label = ui.label().classes("text-xs text-gray-500")

            mode_select = ui.select(MODES, value=state["mode"], label="高亮显示模式").classes("w-full")
            mode_select.on_value_change(on_mode)

            mask_switch = ui.switch(
                "应用 common_mask（关闭后全部像素参与着色与统计）", value=state["use_mask"]
            ).classes("text-sm")
            mask_switch.on_value_change(on_mask_toggle)

            ui.separator()
            meta_label = ui.label().classes("text-sm text-gray-600 whitespace-pre-line")
            stats_label = ui.label().classes("text-sm whitespace-pre-line")
            ui.separator()
            ui.label("色标：0 → 1（d 越大表示两帧差异越大）").classes("text-xs text-gray-500")
            colorbar = ui.image().classes("w-full h-[14px]").style("image-rendering: pixelated")
            ui.label("灰色区域 = common_mask 排除的像素（仅在开关开启时出现）").classes("text-xs text-gray-500")

        with ui.column().classes("w-[540px] shrink-0 items-center gap-2"):
            ui.label("变化程度热力图（128 × 128）").classes("text-sm font-medium")
            heat_image = ui.image().classes("w-[520px] h-[520px]").style("image-rendering: pixelated")

        with ui.column().classes("flex-1 min-w-0 gap-3"):
            ui.label("gallery 原始图像").classes("text-sm font-medium")
            with ui.column().classes("gap-1 w-full"):
                caption_0 = ui.label().classes("text-sm font-medium")
                image_0 = ui.image().classes("w-full")
                missing_0 = ui.label().classes("text-sm text-red-600")
            with ui.column().classes("gap-1 w-full"):
                caption_1 = ui.label().classes("text-sm font-medium")
                image_1 = ui.image().classes("w-full")
                missing_1 = ui.label().classes("text-sm text-red-600")

    rebuild_view_and_pair()
    refresh()


@ui.page("/")
def index() -> None:
    ui.dark_mode(False)
    ui.add_head_html(
        "<style>.nicegui-content{padding:0;align-items:stretch;justify-content:flex-start;}"
        "body{background:#f5f6f8;}</style>"
    )
    build_page(DATASET)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--show", action="store_true", help="open a browser window")
    args = parser.parse_args()
    ui.run(host=args.host, port=args.port, title="AEF 无监督变化检测", reload=False, show=args.show)


if __name__ in {"__main__", "__mp_main__"}:
    main()
