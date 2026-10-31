#!/usr/bin/env python
"""AEF unsupervised change-detection viewer.

Implements the unsupervised branch of AlphaEarth Foundations (supplement S4.1,
equation 8): two per-pixel 64-d embeddings are L2 normalized and compared with

    d = (1 - cos) / 2,   d in [0, 1]

Three combination types are offered:

* same-channel cross-day: the original same-view two-frame comparison;
* same-day cross-channel: two views of one sensor on one date, a cross-modal
  baseline whose high values are channel differences, not change;
* cross-day cross-channel: two views of one sensor on two dates, rendered as
  the residual between the cross-date distance and the same-day baseline.

The result is rendered as a 128x128 heat map beside the original gallery figures,
with a threshold slider whose range adapts to the current non-negative quantity
as ``(0, x]`` (the top of the range yields zero highlighted pixels). Signed
residuals use a diverging colour ramp; thresholding and highlighting always act
on ``|residual|``.

Run with the project virtual environment:

    .venv/Scripts/python.exe app/change_detection_webui.py
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import logging
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from nicegui import ui

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "spectral"
# Set once in ``main`` from ``--data-root``: the resolved package root and the
# tile container inside it. Both are kept because a container may not be named
# ``tiles``, and ``Dataset`` needs to know which directory actually holds tiles.
_initial_paths = {
    "data_root": DEFAULT_DATA_ROOT,
    "tiles_dir": DEFAULT_DATA_ROOT / "tiles",
}

MASK_NEUTRAL = np.array([110, 112, 118], dtype=np.uint8)
HIGHLIGHT = np.array([235, 45, 45], dtype=np.float32)
BINARY_BACKGROUND = np.array([26, 28, 36], dtype=np.uint8)

# One embedding view is (frames, 128, 128, 64) float32 = 8 MiB per frame.
EMBEDDING_CACHE_SIZE = 8

# Rotated gallery figures are kept as base64 data URLs (~300 KiB each); the
# cache only exists so threshold changes do not re-encode the same strip.
ROTATED_IMAGE_CACHE_SIZE = 12

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
def _looks_like_tiles_dir(path: Path) -> bool:
    """True when ``path`` directly contains tile packages."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and (
            (child / "embeddings").is_dir() or (child / "tile_manifest.json").exists()
        ):
            return True
    return False


def resolve_data_paths(
    raw: str | Path, repo_root: Path = REPO_ROOT
) -> tuple[Path, Path]:
    """Resolve a user-supplied path to ``(data_root, tiles_dir)``.

    Accepts either the package root (the directory that holds a tile container
    and ``gallery/``) or the tile container itself, whatever its name. Raises
    ``ValueError`` with a readable reason otherwise.
    """
    text = str(raw).strip()
    if not text:
        raise ValueError("数据根路径为空")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"目录不存在：{path}")

    # Collect every tile container so an ambiguous package is reported instead
    # of silently preferring one of them.
    containers = [
        child for child in sorted(path.iterdir())
        if child.is_dir() and _looks_like_tiles_dir(child)
    ]
    if len(containers) == 1:
        return path, containers[0]
    if len(containers) > 1:
        names = ", ".join(child.name for child in containers)
        raise ValueError(
            f"{path} 下有多个瓦片目录（{names}），请直接指定其中一个"
        )
    # The directory itself may be the tile container.
    if _looks_like_tiles_dir(path):
        return path.parent, path
    raise ValueError(
        f"{path} 既不是数据包根（没有含瓦片的 tiles/ 或其他瓦片目录），"
        "也不是瓦片目录（其下没有瓦片子目录）"
    )


def resolve_data_root(raw: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Convenience wrapper returning only the package root."""
    return resolve_data_paths(raw, repo_root)[0]


class Dataset:
    """Read-only access to one embedding package.

    The data root may differ between instances; every path (including the
    gallery index) is derived from ``self.data_root`` so switching roots never
    leaks paths or caches from the previous package.
    """

    def __init__(
        self, data_root: Path = DEFAULT_DATA_ROOT, tiles_dir: Path | None = None
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.root = Path(tiles_dir).resolve() if tiles_dir else self.data_root / "tiles"
        self.gallery_dir = self.data_root / "gallery"
        self.gallery_index = self.gallery_dir / "index.json"
        self._emb_cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._meta_cache: dict[tuple[str, str], dict] = {}
        self._mask_cache: dict[tuple[str, str], np.ndarray] = {}
        self._manifest_cache: dict[str, dict] = {}
        self._views_cache: dict[str, tuple[str, ...]] = {}
        self._rotated_cache: OrderedDict[
            tuple[str, str, str], tuple[str, tuple[int, int]]
        ] = OrderedDict()
        self.gallery = self._load_gallery()

    def _gallery_path(self, image: str) -> Path | None:
        """Resolve an ``index.json`` image path inside this data root.

        The packaged index stores paths such as ``spectral/gallery/C01_sentinel2_0.png``,
        whose leading component is the *packaged* root's directory name. A copy
        under a different name must not resolve back into the original package,
        so the leading component is stripped and tried inside this root first;
        the sibling candidate is only used when the path really names this root.

        The first existing candidate wins, but a syntactically valid path that
        does not exist is still returned (as its first in-root candidate) so the
        UI can distinguish "no index record" from "record present, file gone".
        Returns ``None`` only for paths that escape the package or are not
        strings, so a malformed index can never read another package's images.
        """
        if not isinstance(image, str):
            return None
        relative = Path(image)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        # The record's own path (relative to the root) is the diagnostic value
        # shown when the file is missing, so it is kept intact.
        direct = (self.data_root / relative).resolve()
        candidates: list[Path] = []
        if len(relative.parts) > 1:
            candidates.append(self.data_root / Path(*relative.parts[1:]))
        candidates.append(self.data_root / relative)
        if relative.parts and relative.parts[0] == self.data_root.name:
            candidates.append(self.data_root.parent / relative)
        in_root: list[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.is_relative_to(self.data_root):
                continue
            if resolved.exists():
                return resolved
            in_root.append(resolved)
        # Nothing exists: report the record's own in-root path when it is legal,
        # so the message points at the file the index actually named.
        if direct.is_relative_to(self.data_root):
            return direct
        return in_root[0] if in_root else None

    def _load_gallery(self) -> dict[tuple[str, str, str], str]:
        if not self.gallery_index.exists():
            return {}
        entries = json.loads(self.gallery_index.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"{self.gallery_index} 不是索引数组")
        out: dict[tuple[str, str, str], str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            image = entry.get("image")
            if not image:
                continue
            key = (entry.get("id"), entry.get("source"), entry.get("date"))
            if any(not isinstance(part, str) or not part for part in key):
                continue
            path = self._gallery_path(image)
            if path is not None:
                out[key] = str(path)
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
            p.name for p in self.gallery_dir.glob(f"{tile}_{source}_*.png")
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

    def rotated_image_url(
        self, tile: str, view: str, date: str
    ) -> tuple[str | None, str, tuple[int, int] | None]:
        """Return ``(data_url_or_None, detail, rotated_size)`` for the gallery图 rotated 90° clockwise.

        The gallery figures are wide multi-panel strips. Two of them stacked
        vertically push the second one out of view, so each strip is turned into
        a portrait image and the pair is shown side by side. ``data/`` is never
        written: the rotation happens in memory and the PNG data URL is cached
        together with its size, which the layout needs to keep the aspect ratio.
        """
        path, detail = self.image_entry(tile, view, date)
        if path is None:
            return None, detail, None
        key = (tile, view, date)
        if key in self._rotated_cache:
            url, size = self._rotated_cache[key]
            self._rotated_cache.move_to_end(key)
            return url, "", size
        with Image.open(path) as image:
            # ROTATE_270 is 90° clockwise (Pillow counts counter-clockwise).
            rotated = image.convert("RGB").transpose(Image.Transpose.ROTATE_270)
            size = rotated.size
            buffer = io.BytesIO()
            rotated.save(buffer, format="PNG", compress_level=1)
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        self._rotated_cache[key] = (url, size)
        while len(self._rotated_cache) > ROTATED_IMAGE_CACHE_SIZE:
            self._rotated_cache.popitem(last=False)
        return url, "", size


# --------------------------------------------------------------------------- #
# computation
# --------------------------------------------------------------------------- #
def pair_distance(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, use_mask: bool = False
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


def sample_ramp(ramp: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Interpolate ``ramp`` at normalised positions in ``[0, 1]``."""
    pos = np.clip(position, 0.0, 1.0) * (len(ramp) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.minimum(lo + 1, len(ramp) - 1)
    frac = (pos - lo)[..., None]
    return (ramp[lo] * (1.0 - frac) + ramp[hi] * frac).astype(np.uint8)


def colourise(t: np.ndarray) -> np.ndarray:
    return sample_ramp(_RAMP, t)


def colourise_signed(r: np.ndarray) -> np.ndarray:
    """Blue → near-white → red for signed residuals in ``[-1, 1]``."""
    return sample_ramp(_DIVERGING_RAMP, (np.clip(r, -1.0, 1.0) + 1.0) * 0.5)


def ramp_samples(ramp: np.ndarray, levels: int) -> np.ndarray:
    """``levels`` colours sampled at band centres of ``ramp``.

    Band ``k`` takes the ramp colour at ``(k + 0.5) / levels`` so each band is
    represented by its own midpoint hue. The first and last bands therefore sit
    one half-band inside the ramp, keeping the full colour span visible while
    staying symmetric about ``0.5`` (the diverging ramp's neutral white).
    """
    levels = max(1, int(levels))
    positions = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    return sample_ramp(ramp, positions)


def band_index(values: np.ndarray, low: float, high: float, levels: int) -> np.ndarray:
    """Quantise ``values`` into ``levels`` equal bands over ``[low, high]``.

    ``high == low`` collapses every pixel into band 0 instead of dividing by
    zero.
    """
    levels = max(1, int(levels))
    span = float(high) - float(low)
    if span <= 0.0:
        return np.zeros(values.shape, dtype=np.int32)
    t = (values.astype(np.float32) - float(low)) / span
    # Non-finite values would otherwise survive the int cast as INT_MIN and
    # index the palette out of range.
    t = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.minimum((t * levels).astype(np.int32), levels - 1)


def band_boundaries(index: np.ndarray) -> np.ndarray:
    """Pixels sitting on a band edge, from right/down neighbours.

    The mask follows the data: a pixel is an edge pixel where the band to its
    right or below differs, which draws the quantised iso-lines rather than a
    fixed grid.
    """
    edges = np.zeros(index.shape, dtype=bool)
    edges[:, :-1] |= index[:, :-1] != index[:, 1:]
    edges[:-1, :] |= index[:-1, :] != index[1:, :]
    return edges


def _mix_with_edge(base: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Blend edge pixels with black or white according to their own brightness."""
    if not edges.any():
        return base
    out = base.copy()
    # Perceptual brightness (ITU-R BT.601 weights).
    luma = (
        0.299 * out[..., 0].astype(np.float32)
        + 0.587 * out[..., 1].astype(np.float32)
        + 0.114 * out[..., 2].astype(np.float32)
    )
    dark = luma < 128.0
    for colour, selector in (
        (np.array([255, 255, 255], dtype=np.float32), dark),
        (np.array([0, 0, 0], dtype=np.float32), ~dark),
    ):
        target = edges & selector
        if target.any():
            mixed = out[target].astype(np.float32) * 0.35 + colour * 0.65
            out[target] = mixed.astype(np.uint8)
    return out


def render_bands(
    values: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    mode: str,
    *,
    low: float,
    high: float,
    levels: int,
    edges: bool,
    signed: bool,
) -> np.ndarray:
    """Render one frame with banded colouring plus threshold highlighting.

    Colouring and highlighting are independent: ``threshold`` always acts on
    ``|value|`` for signed residuals and on the value itself otherwise, whereas
    the bands only decide the base colour. Statistics never see the bands.
    """
    index = band_index(values, low, high, levels)
    palette = ramp_samples(_DIVERGING_RAMP if signed else _RAMP, levels)
    base = palette[index]
    hot = valid & (np.abs(values) > threshold)

    if mode == "binary":
        rgb = np.broadcast_to(BINARY_BACKGROUND, values.shape + (3,)).copy()
        rgb[hot] = HIGHLIGHT.astype(np.uint8)
    elif mode == "grey":
        magnitude = np.abs(values) if signed else values
        grey = (np.clip(magnitude, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = np.stack([grey, grey, grey], axis=-1)
        rgb[hot] = base[hot]
    else:  # overlay
        rgb = base.copy()
        blend = base[hot].astype(np.float32) * 0.45 + HIGHLIGHT * 0.55
        rgb[hot] = blend.astype(np.uint8)

    # Band edges trace the quantised palette. The binary mode renders no
    # palette at all, so edges there would draw a contour map over the red mask
    # and hide the highlight; that mode keeps its edges off by design.
    if edges and mode != "binary":
        rgb = _mix_with_edge(rgb, band_boundaries(index))

    rgb[~valid] = MASK_NEUTRAL
    return rgb


def render_band_strip(
    *,
    low: float,
    high: float,
    levels: int,
    edges: bool,
    signed: bool,
    width: int = 512,
    height: int = 14,
) -> np.ndarray:
    """The colour bar: the same discrete palette as the map, edge lines included."""
    palette = ramp_samples(_DIVERGING_RAMP if signed else _RAMP, levels)
    index = band_index(
        np.linspace(low, high, width, dtype=np.float32)[None, :],
        low,
        high,
        levels,
    )
    strip = np.repeat(palette[index], height, axis=0)
    if edges:
        strip = _mix_with_edge(strip, band_boundaries(index))
    return strip


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


_DIVERGING_RAMP = np.array(
    [
        [33, 102, 172],
        [67, 147, 195],
        [146, 197, 222],
        [209, 229, 240],
        [247, 247, 247],
        [253, 219, 199],
        [244, 165, 130],
        [214, 96, 77],
        [178, 24, 43],
    ],
    dtype=np.float32,
)

COMBOS = {
    "same_channel": "同通道跨日（同模态对照）",
    "same_day": "同日跨通道（跨模态基线）",
    "cross_day": "跨日跨通道（跨模态变化）",
}

RESIDUAL_DIRECTIONS = {"sym": "对称化", "fwd": "正向", "rev": "反向"}
RESIDUAL_MODES = {"signed": "带符号", "abs": "绝对值"}

BAND_MODES = {"fixed": "固定分段", "dynamic": "动态分段"}
LEVELS_MIN = 2
LEVELS_MAX = 64
LEVELS_DEFAULT = 12


def sensor_of(view: str) -> str:
    return view.split("_", 1)[0]


def band_of(view: str) -> str:
    return view.split("_", 1)[1] if "_" in view else view


def distribution_label(view: str) -> str:
    return "分布内" if band_of(view) == "full" else "分布外（波段消融诊断）"


def render_rgb_signed(
    r: np.ndarray, valid: np.ndarray, threshold: float, mode: str
) -> np.ndarray:
    """RGB image of a signed residual; threshold and highlight use ``|r|``."""
    q = np.abs(r)
    base = colourise_signed(r)
    hot = valid & (q > threshold)

    if mode == "binary":
        rgb = np.broadcast_to(BINARY_BACKGROUND, r.shape + (3,)).copy()
        rgb[hot] = HIGHLIGHT.astype(np.uint8)
    elif mode == "grey":
        grey = (np.clip(q, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = np.stack([grey, grey, grey], axis=-1)
        rgb[hot] = base[hot]
    else:  # overlay
        rgb = base.copy()
        blend = base[hot].astype(np.float32) * 0.45 + HIGHLIGHT * 0.55
        rgb[hot] = blend.astype(np.uint8)

    rgb[~valid] = MASK_NEUTRAL
    return rgb


def summarise_signed(r: np.ndarray, valid: np.ndarray, threshold: float) -> dict:
    """Statistics of a signed residual; hot count still uses ``|r| > threshold``."""
    values = r[valid]
    total = int(values.size)
    hot = int(np.count_nonzero(np.abs(values) > threshold))
    return {
        "valid": total,
        "hot": hot,
        "ratio": (hot / total) if total else 0.0,
        "median": float(np.median(values)) if total else 0.0,
        "p95": float(np.percentile(values, 95)) if total else 0.0,
        "maximum": float(values.max()) if total else 0.0,
        "minimum": float(values.min()) if total else 0.0,
    }


_TILE_COORD_RE = re.compile(r"_(?P<lon>[EW])(?P<lonv>\d+(?:\.\d+)?)_(?P<lat>[NS])(?P<latv>\d+(?:\.\d+)?)")


def tile_coordinates(manifest: dict) -> tuple[float, float] | None:
    """Centre coordinates parsed from ``tile_manifest.json``'s ``parent``.

    The packaged tiles encode the location in the parent directory name, e.g.
    ``registered/tiles/C01_E121.4790_N25.1724_20240101_20241231``. Anything
    that does not match that shape (``controls/tiles/V01``) is reported as
    unknown instead of guessed.
    """
    parent = manifest.get("parent")
    if not isinstance(parent, str):
        return None
    match = _TILE_COORD_RE.search(parent)
    if match is None:
        return None
    lon = float(match.group("lonv"))
    lat = float(match.group("latv"))
    if match.group("lon") == "W":
        lon = -lon
    if match.group("lat") == "S":
        lat = -lat
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return lon, lat


def coordinate_text(manifest: dict) -> tuple[str, str]:
    """Return ``(centre_text, parent_text)`` for the metadata panel."""
    parent = manifest.get("parent")
    parent_text = parent.strip() if isinstance(parent, str) else ""
    coords = tile_coordinates(manifest)
    if coords is None:
        centre = "无坐标信息"
    else:
        lon, lat = coords
        lon_hemisphere = "E" if lon >= 0 else "W"
        lat_hemisphere = "N" if lat >= 0 else "S"
        centre = (
            f"{abs(lon):.4f}°{lon_hemisphere} / {abs(lat):.4f}°{lat_hemisphere}"
        )
    return centre, parent_text or "无"


def build_page(initial_root: Path, initial_tiles: Path | None = None) -> None:
    dataset = Dataset(initial_root, initial_tiles)
    tiles = dataset.tiles()
    if not tiles:
        ui.label(f"未在 {dataset.root} 找到任何瓦片").classes("text-red-600")
        return

    def tile_label(name: str, source: "Dataset | None" = None) -> str:
        manifest = (source or dataset).manifest(name)
        category = manifest.get("category") or "未分类"
        return f"{name} · {category} · {manifest.get('split') or '未标注'}"

    def preflight_labels(source: Dataset, names: list[str]) -> dict[str, str]:
        """Build every tile label up front so a corrupt manifest fails early."""
        labels: dict[str, str] = {}
        for name in names:
            try:
                labels[name] = tile_label(name, source)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"瓦片 {name} 的 tile_manifest.json 无法读取：{error}") from error
        return labels

    default_tile = "C01" if "C01" in tiles else tiles[0]
    tile_options = {name: tile_label(name) for name in tiles}

    default_views = dataset.views(default_tile)
    if "sentinel2_full" in default_views:
        default_view = "sentinel2_full"
    elif default_views:
        default_view = default_views[0]
    else:
        default_view = ""

    all_default = dataset.all_views(default_tile)

    def pick_channel(views: list[str], preferred: str, exclude: str = "") -> str:
        if preferred in views and preferred != exclude:
            return preferred
        for name in views:
            if name != exclude:
                return name
        return ""

    default_a = pick_channel(all_default, "sentinel2_visible")
    default_b = pick_channel(
        [v for v in all_default if sensor_of(v) == sensor_of(default_a)],
        "sentinel2_nir",
        exclude=default_a,
    )

    state = {
        "combo": "same_channel",
        "tile": default_tile,
        "view": default_view,
        "pair": (0, 1),
        "view_a": default_a,
        "view_b": default_b,
        "frame_shared": 0,
        "frame_a": 0,
        "frame_b": max(dataset.frame_total(default_tile, default_b) - 1, 0),
        "residual_dir": "fwd",
        "residual_mode": "abs",
        "threshold": 0.30,
        "mode": "overlay",
        "band_mode": "dynamic",
        "levels": 12,
        "edges": True,
        "use_mask": False,
        "rescale": True,
        "scale": {
            "max": 0.0, "step": 0.01, "lower": 0.01, "upper": 1.0,
            "default": 0.30, "decimals": 2, "fmt": "%.2f",
        },
    }

    def available_views(for_combo: str | None = None) -> list[str]:
        """Views selectable for the given combination type.

        Cross-channel same-day allows single-frame views because no two dates
        are compared; every other type needs at least two frames so that a
        tile never offers an uncomputable combination.
        """
        name = for_combo or state["combo"]
        if name == "same_day":
            return dataset.all_views(state["tile"])
        return dataset.views(state["tile"])

    def ensure_channels() -> None:
        views = available_views()
        if not views:
            state["view_a"] = ""
            state["view_b"] = ""
            return
        if state["view_a"] not in views:
            state["view_a"] = pick_channel(views, "sentinel2_visible")
        same_sensor = [
            v for v in views
            if sensor_of(v) == sensor_of(state["view_a"]) and v != state["view_a"]
        ]
        if state["view_b"] not in same_sensor:
            state["view_b"] = pick_channel(same_sensor, "sentinel2_nir")

    def compute() -> dict:
        tile = state["tile"]
        use_mask = state["use_mask"]
        kind = state["combo"]

        if kind == "same_channel":
            view = state["view"]
            emb = dataset.embeddings(tile, view)
            mask = dataset.mask(tile, view)
            i, j = state["pair"]
            i = max(0, min(i, emb.shape[0] - 1))
            j = max(0, min(j, emb.shape[0] - 1))
            d, valid = pair_distance(emb[i], emb[j], mask, use_mask=use_mask)
            return {
                "kind": kind, "q": d, "r": None, "valid": valid, "mask": mask,
                "view_a": view, "view_b": "", "frame_a": i, "frame_b": j,
                "frame_shared": None, "base": None, "fwd": None, "rev": None, "sym": None,
            }

        view_a = state["view_a"]
        view_b = state["view_b"]
        emb_a = dataset.embeddings(tile, view_a)
        emb_b = dataset.embeddings(tile, view_b)
        # Same tile + same sensor share one common_mask; the sensor comes from
        # the view prefix, never from the distance_maps directory order.
        mask = dataset.mask(tile, view_a)

        if kind == "same_day":
            t = int(state["frame_shared"])
            t = max(0, min(t, min(emb_a.shape[0], emb_b.shape[0]) - 1))
            base, valid = pair_distance(emb_a[t], emb_b[t], mask, use_mask=use_mask)
            return {
                "kind": kind, "q": base, "r": None, "valid": valid, "mask": mask,
                "view_a": view_a, "view_b": view_b, "frame_a": t, "frame_b": t,
                "frame_shared": t, "base": None, "fwd": None, "rev": None, "sym": None,
            }

        ref = int(state["frame_a"])
        other = int(state["frame_b"])
        # Both channels come from one imaging session, so their frame lists
        # normally agree; clamping against both lengths keeps a mismatched
        # package from indexing out of range.
        limit = min(emb_a.shape[0], emb_b.shape[0]) - 1
        ref = max(0, min(ref, limit))
        other = max(0, min(other, limit))

        base, valid_base = pair_distance(emb_a[ref], emb_b[ref], mask, use_mask=use_mask)
        d_fwd, valid_fwd = pair_distance(emb_a[ref], emb_b[other], mask, use_mask=use_mask)
        d_rev, valid_rev = pair_distance(emb_b[ref], emb_a[other], mask, use_mask=use_mask)
        valid = valid_base & valid_fwd & valid_rev

        base = np.where(valid, base, 0.0).astype(np.float32)
        fwd = np.clip(d_fwd - base, -1.0, 1.0).astype(np.float32)
        rev = np.clip(d_rev - base, -1.0, 1.0).astype(np.float32)
        sym = ((fwd + rev) / 2.0).astype(np.float32)
        chosen = {"fwd": fwd, "rev": rev, "sym": sym}[state["residual_dir"]]
        chosen = np.where(valid, chosen, 0.0).astype(np.float32)
        return {
            "kind": kind, "q": np.abs(chosen), "r": chosen, "valid": valid, "mask": mask,
            "view_a": view_a, "view_b": view_b, "frame_a": ref, "frame_b": other,
            "frame_shared": None, "base": base, "fwd": fwd, "rev": rev, "sym": sym,
        }

    heat_image = None
    colorbar = None
    colorbar_label = None
    heat_title = None
    semantics_label = None
    threshold_caption = None
    stats_label = None
    meta_label = None
    image_col_0 = None
    image_col_1 = None
    image_0 = None
    image_1 = None
    caption_0 = None
    caption_1 = None
    missing_0 = None
    missing_1 = None
    tile_select = None
    combo_select = None
    view_select = None
    view_a_select = None
    view_b_select = None
    pair_select = None
    frame_shared_select = None
    frame_a_select = None
    frame_b_select = None
    residual_dir_select = None
    residual_mode_select = None
    residual_formula_label = None
    threshold_slider = None
    threshold_number = None
    mode_select = None
    mask_switch = None
    scale_label = None
    colorbar_cache = {"key": None}
    syncing = {"active": False, "levels": False}

    def rescale(
        q: np.ndarray, valid: np.ndarray, keep_threshold: float | None = None
    ) -> None:
        """Fit the threshold controls to the current visualised non-negative value.

        ``keep_threshold`` re-applies a threshold the user had already chosen
        instead of adopting the fresh default. It is used when a failed package
        switch has to restore the previous display without silently resetting
        the user's threshold.
        """
        scale = threshold_scale(q, valid)
        state["scale"] = scale
        decimals = scale["decimals"]

        if keep_threshold is None:
            state["threshold"] = scale["default"]
        else:
            step = max(scale["step"], 1e-9)
            snapped = round(round(float(keep_threshold) / step) * step, decimals)
            state["threshold"] = float(
                np.clip(snapped, scale["lower"], scale["upper"])
            )

        # The whole update runs under the guard: changing min/max makes NiceGUI's
        # Number sanitize the current value, and that write fires on_number which
        # would push the clamped old threshold back into the state. Holding the
        # guard here keeps the freshly computed value authoritative.
        syncing["active"] = True
        try:
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
            threshold_slider.value = state["threshold"]
            threshold_number.value = round(state["threshold"], decimals)
            threshold_number.update()
        finally:
            syncing["active"] = False
        scale_label.set_text(
            f"当前量程 (0, {scale['upper']:.{decimals}f}]   步长 {scale['step']:.{decimals}f}   "
            f"最大值（当前可视化量） {scale['max']:.{decimals}f}"
        )
        threshold_value_label.set_text(
            f"当前阈值 = {state['threshold']:.{decimals}f}"
        )

    def band_range(signed: bool) -> tuple[float, float]:
        """Colour-domain bounds for the current banding mode.

        ``fixed`` always spans the full theoretical range; ``dynamic`` follows
        the threshold scale's upper bound ``x`` so the bands cover exactly the
        interval the slider exposes.
        """
        if state["band_mode"] == "fixed":
            return (-1.0, 1.0) if signed else (0.0, 1.0)
        upper = float(state["scale"]["upper"])
        return (-upper, upper) if signed else (0.0, upper)

    def apply_image(label, caption, missing, caption_text: str, date: str,
                    view: str) -> None:
        caption.set_text(caption_text)
        # Narrow columns can clip the last line; the title keeps the full text
        # reachable as a tooltip.
        caption.props(f'title="{caption_text}"')
        url, detail, size = dataset.rotated_image_url(state["tile"], view, date)
        if url is not None:
            # The rotated strip is portrait; fixing height and aspect ratio keeps
            # it from stretching when its column is wider than the image.
            width, height = size if size else (1, 1)
            label.style(
                f"height: 72vh; width: auto; aspect-ratio: {width} / {height};"
                " max-width: 100%"
            )
            label.set_source(url)
            label.set_visibility(True)
            missing.set_visibility(False)
        else:
            label.set_visibility(False)
            missing.set_text(f"缺少图像：{date}\n{detail}")
            missing.set_visibility(True)

    def clear_threshold_controls() -> None:
        """Disable the threshold row when there is nothing to compare.

        Without this the widgets would keep the previous (or candidate)
        package's range while the stats area says there is no data, and the
        pending ``rescale_keep`` flag would leak into the next successful
        selection.
        """
        state.pop("rescale_keep", None)
        state["rescale"] = False
        syncing["active"] = True
        try:
            threshold_slider._props["min"] = 0.0  # noqa: SLF001
            threshold_slider._props["max"] = 0.0  # noqa: SLF001
            threshold_slider._props["step"] = 1.0  # noqa: SLF001
            threshold_slider.value = 0.0
            threshold_slider.update()
            threshold_number.min = 0.0
            threshold_number.max = 0.0
            threshold_number.value = 0.0
            threshold_number.update()
        finally:
            syncing["active"] = False
        threshold_value_label.set_text("当前阈值 = —（暂无可比较的数据）")
        scale_label.set_text("当前量程不可用：请选择其他瓦片、视图或帧")
        colorbar.set_source("")
        colorbar_label.set_text("")
        colorbar_cache["key"] = None

    def clear_image_area() -> None:
        """Hide both gallery columns and drop their captions.

        Without clearing the captions the previous tile's frame labels stay
        visible while the heat map and stats say there is nothing to compare.
        """
        for label, caption, missing in (
            (image_0, caption_0, missing_0),
            (image_1, caption_1, missing_1),
        ):
            caption.set_text("")
            caption.props('title=""')
            label.set_source("")
            label.set_visibility(False)
            missing.set_text("")
            missing.set_visibility(False)
        for col in (image_col_0, image_col_1):
            col.set_visibility(False)

    def refresh() -> None:
        kind = state["combo"]
        tile = state["tile"]

        if kind == "same_channel":
            if not state["view"]:
                skipped = dataset.skipped_views(tile)
                skipped_note = f"\n被排除的视图 = {', '.join(skipped)}" if skipped else ""
                meta_label.set_text(
                    f"瓦片 = {tile}\n"
                    f"该瓦片没有可用于两帧比较的视图（所有视图均不足 2 帧）{skipped_note}"
                )
                stats_label.set_text("暂无可比较的数据")
                heat_image.set_source("")
                clear_threshold_controls()
                clear_image_area()
                return
        elif not state["view_a"] or not state["view_b"]:
            meta_label.set_text(
                f"瓦片 = {tile}\n"
                f"当前组合类型没有可用的通道组合（需要同一传感器下的两个视图）"
            )
            stats_label.set_text("暂无可比较的数据")
            heat_image.set_source("")
            clear_threshold_controls()
            clear_image_area()
            return

        result = compute()
        q = result["q"]
        r = result["r"]
        valid = result["valid"]

        if state.get("rescale"):
            state["rescale"] = False
            keep = state.pop("rescale_keep", False)
            rescale(q, valid, keep_threshold=state["threshold"] if keep else None)

        signed = kind == "cross_day" and state["residual_mode"] == "signed" and r is not None
        band_low, band_high = band_range(signed)
        values = r if signed else q
        rgb = render_bands(
            values,
            valid,
            state["threshold"],
            state["mode"],
            low=band_low,
            high=band_high,
            levels=state["levels"],
            edges=state["edges"],
            signed=signed,
        )
        # The gallery strips are displayed rotated 90° clockwise (see
        # rotated_image_url). Both figures come from the same 128x128 grid, so
        # the heat map must receive the identical clockwise rotation; otherwise
        # the side-by-side view would be 90° out of step. np.rot90(k=-1) and
        # Pillow's ROTATE_270 are the same clockwise transform.
        heat_image.set_source(png_data_url(np.rot90(rgb, k=-1)))

        bar_key = (
            "signed" if signed else "sequential",
            state["band_mode"],
            state["levels"],
            state["edges"],
            round(band_low, 6),
            round(band_high, 6),
        )
        if colorbar_cache["key"] != bar_key:
            bar = render_band_strip(
                low=band_low,
                high=band_high,
                levels=state["levels"],
                edges=state["edges"],
                signed=signed,
            )
            colorbar.set_source(png_data_url(bar))
            mode_text = "固定分段" if state["band_mode"] == "fixed" else "动态分段"
            if signed:
                colorbar_label.set_text(
                    f"色标（{mode_text} · {state['levels']} 档）："
                    f"{band_low:+.3f} → 0 → {band_high:+.3f}，"
                    "负 = 残差减小，正 = 残差增大；阈值与高亮作用于 |残差|"
                )
            else:
                colorbar_label.set_text(
                    f"色标（{mode_text} · {state['levels']} 档）："
                    f"{max(band_low, 0.0):.3f} → {band_high:.3f}，"
                    "当前可视化量越大表示差异越大"
                )
            colorbar_cache["key"] = bar_key

        if kind == "same_channel":
            threshold_caption.set_text("阈值（作用于 d = (1 − cos) / 2，量程自适应，左端不可取 0）")
            heat_title.set_text("变化程度热力图（128 × 128，顺时针旋转 90°）")
            semantics_label.set_text("同模态对照：同一视图不同日期，高值表示变化程度。")
        elif kind == "same_day":
            threshold_caption.set_text("阈值（作用于通道差异基线 d，量程自适应，左端不可取 0）")
            heat_title.set_text("通道差异基线热力图（128 × 128，顺时针旋转 90°，高值不代表变化）")
            semantics_label.set_text("跨模态基线：同一日期的通道差异，高值不代表变化。")
        else:
            threshold_caption.set_text("阈值（作用于 |残差|，量程自适应，左端不可取 0）")
            heat_title.set_text("跨模态残差热力图（128 × 128，顺时针旋转 90°）")
            semantics_label.set_text(
                "跨模态变化（探索性）：残差 = 跨日期距离 − 同日基线；"
                "数据不含变化参考标注，仅作探索性对照。"
            )

        summary = summarise(q, valid, state["threshold"])
        digits = state["scale"]["decimals"]
        scope = (
            "common_mask 内有效像素"
            if state["use_mask"]
            else "全部像素（未应用 common_mask）"
        )

        if kind == "same_channel":
            stats_label.set_text(
                f"阈值 = {state['threshold']:.{digits}f}   |   统计范围 = {scope}\n"
                f"超阈值像素 = {summary['hot']} / {summary['valid']}  "
                f"({summary['ratio'] * 100:.2f}%)\n"
                f"距离 中位数 = {summary['median']:.4f}   "
                f"p95 = {summary['p95']:.4f}   最大 = {summary['maximum']:.4f}"
            )
        elif kind == "same_day":
            stats_label.set_text(
                f"阈值 = {state['threshold']:.{digits}f}   |   统计范围 = {scope}\n"
                f"超阈值像素（基线 > 阈值）= {summary['hot']} / {summary['valid']}  "
                f"({summary['ratio'] * 100:.2f}%)\n"
                f"通道差异基线 中位数 = {summary['median']:.4f}   "
                f"p95 = {summary['p95']:.4f}   最大 = {summary['maximum']:.4f}"
            )
        else:
            any_valid = bool(valid.any())
            base_mean = float(np.mean(result["base"][valid])) if any_valid else 0.0
            fwd_mean = float(np.mean(result["fwd"][valid])) if any_valid else 0.0
            rev_mean = float(np.mean(result["rev"][valid])) if any_valid else 0.0
            sym_mean = float(np.mean(result["sym"][valid])) if any_valid else 0.0
            if signed:
                signed_summary = summarise_signed(r, valid, state["threshold"])
                head = (
                    f"阈值 = {state['threshold']:.{digits}f}（作用于 |残差|）   |   "
                    f"统计范围 = {scope}\n"
                    f"超阈值像素（|残差| > 阈值）= {signed_summary['hot']} / "
                    f"{signed_summary['valid']}  ({signed_summary['ratio'] * 100:.2f}%)\n"
                    f"残差（带符号）中位数 = {signed_summary['median']:.4f}   "
                    f"p95 = {signed_summary['p95']:.4f}   "
                    f"最大 = {signed_summary['maximum']:.4f}   "
                    f"最小 = {signed_summary['minimum']:.4f}\n"
                )
            else:
                head = (
                    f"阈值 = {state['threshold']:.{digits}f}（作用于 |残差|）   |   "
                    f"统计范围 = {scope}\n"
                    f"超阈值像素（|残差| > 阈值）= {summary['hot']} / {summary['valid']}  "
                    f"({summary['ratio'] * 100:.2f}%)\n"
                    f"|残差| 中位数 = {summary['median']:.4f}   "
                    f"p95 = {summary['p95']:.4f}   最大 = {summary['maximum']:.4f}\n"
                )
            stats_label.set_text(
                head
                + f"同日基线均值 = {base_mean:.4f}   正向残差均值 = {fwd_mean:+.4f}   "
                f"反向残差均值 = {rev_mean:+.4f}\n"
                + f"双向均值之差（正向 − 反向） = {fwd_mean - rev_mean:+.4f}   "
                f"对称化残差均值 = {sym_mean:+.4f}"
            )

        manifest = dataset.manifest(tile)
        mask_pixels = int(result["mask"].sum())
        if state["use_mask"]:
            excluded = int((~valid).sum())
            mask_note = (
                f"common_mask = 开启   掩膜像素 = {mask_pixels}   "
                f"掩膜外像素 = {excluded}（灰色，不参与统计与高亮）"
            )
        else:
            mask_note = (
                f"common_mask = 关闭   掩膜像素 = {mask_pixels}（未应用）   "
                f"全部 16384 像素参与着色与统计"
            )

        head_text = (
            f"瓦片 = {tile}   split = {manifest.get('split') or '未标注'}   "
            f"类别 = {manifest.get('category') or '未分类'}\n"
            f"组合类型 = {COMBOS[kind]}\n"
        )

        if kind == "same_channel":
            frames = dataset.frame_meta(tile, state["view"])["frames"]
            i, j = result["frame_a"], result["frame_b"]
            date_i = frames[i].get("date_utc", "?")
            date_j = frames[j].get("date_utc", "?")
            skipped = dataset.skipped_views(tile)
            skipped_note = (
                f"\n已排除单帧视图 = {', '.join(skipped)}（无法构成两帧比较）"
                if skipped else ""
            )
            meta_label.set_text(
                head_text
                + f"视图 = {state['view']}（{distribution_label(state['view'])}）   "
                f"帧数 = {len(frames)}   比较帧 = {i} → {j}   日期 = {date_i} → {date_j}"
                + f"{skipped_note}\n{mask_note}"
            )
        else:
            view_a = result["view_a"]
            view_b = result["view_b"]
            frames_a = dataset.frame_meta(tile, view_a)["frames"]
            frames_b = dataset.frame_meta(tile, view_b)["frames"]
            mixed = ""
            if (band_of(view_a) == "full") != (band_of(view_b) == "full"):
                mixed = "\n分布类别 = 分布内 vs 分布外混合对照"
            if kind == "same_day":
                t = result["frame_shared"]
                when = f"共同帧 = {t}   日期 = {frames_a[t].get('date_utc', '?')}"
                skipped_note = ""
            else:
                when = (
                    f"A 帧 {result['frame_a']} · "
                    f"{frames_a[result['frame_a']].get('date_utc', '?')}   →   "
                    f"B 帧 {result['frame_b']} · "
                    f"{frames_b[result['frame_b']].get('date_utc', '?')}"
                )
                skipped = dataset.skipped_views(tile)
                skipped_note = (
                    f"\n已排除单帧视图（跨日类型不可用）= {', '.join(skipped)}"
                    if skipped else ""
                )
            meta_label.set_text(
                head_text
                + f"通道 A = {view_a}（传感器 {sensor_of(view_a)}，"
                f"波段 {band_of(view_a)}，{distribution_label(view_a)}）\n"
                + f"通道 B = {view_b}（传感器 {sensor_of(view_b)}，"
                f"波段 {band_of(view_b)}，{distribution_label(view_b)}）{mixed}\n"
                + f"帧数 = A {len(frames_a)} / B {len(frames_b)}   {when}"
                + f"{skipped_note}\n{mask_note}"
            )

        if kind == "same_channel":
            image_col_0.set_visibility(True)
            image_col_1.set_visibility(True)
            frames = dataset.frame_meta(tile, state["view"])["frames"]
            for slot, label, caption, missing, frame_idx in (
                (0, image_0, caption_0, missing_0, result["frame_a"]),
                (1, image_1, caption_1, missing_1, result["frame_b"]),
            ):
                date = frames[frame_idx].get("date_utc", "?")
                apply_image(
                    label, caption, missing,
                    f"帧 {frame_idx} · {date} · {state['view']}",
                    date, state["view"],
                )
        elif kind == "same_day":
            image_col_0.set_visibility(True)
            image_col_1.set_visibility(False)
            t = result["frame_shared"]
            date = dataset.frame_meta(tile, result["view_a"])["frames"][t].get("date_utc", "?")
            apply_image(
                image_0, caption_0, missing_0,
                f"同一观测，两通道 · {date} · A={result['view_a']} / B={result['view_b']}",
                date, result["view_a"],
            )
        else:
            image_col_0.set_visibility(True)
            image_col_1.set_visibility(True)
            for slot, label, caption, missing, view, frame_idx in (
                (0, image_0, caption_0, missing_0, result["view_a"], result["frame_a"]),
                (1, image_1, caption_1, missing_1, result["view_b"], result["frame_b"]),
            ):
                date = dataset.frame_meta(tile, view)["frames"][frame_idx].get("date_utc", "?")
                tag = "A" if slot == 0 else "B"
                # Date precedes the view name so a clipped caption still shows
                # the date; the full text is available as the title tooltip.
                apply_image(
                    label, caption, missing,
                    f"{tag} 通道 · 帧 {frame_idx} · {date} · {view}",
                    date, view,
                )

    def rebuild_pairs() -> None:
        n = dataset.frame_total(state["tile"], state["view"])
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
        meta = dataset.frame_meta(state["tile"], state["view"])
        options = {}
        for i in range(n):
            for j in range(i + 1, n):
                options[f"{i}-{j}"] = f"{frame_label(meta, i)} → {frame_label(meta, j)}"
        default = f"0-{n - 1}"
        pair_select.set_options(options, value=default)
        pair_select.set_visibility(True)
        state["pair"] = (0, n - 1)

    def rebuild_shared_frames() -> None:
        n = dataset.frame_total(state["tile"], state["view_a"])
        if n <= 0:
            state["frame_shared"] = 0
            frame_shared_select.set_options({}, value=None)
            return
        if state["frame_shared"] >= n:
            state["frame_shared"] = 0
        meta = dataset.frame_meta(state["tile"], state["view_a"])
        options = {str(i): frame_label(meta, i) for i in range(n)}
        frame_shared_select.set_options(options, value=str(state["frame_shared"]))

    def rebuild_cross_frames() -> None:
        view_a = state["view_a"]
        view_b = state["view_b"]
        if not view_a or not view_b:
            frame_a_select.set_options({}, value=None)
            frame_b_select.set_options({}, value=None)
            return
        meta_a = dataset.frame_meta(state["tile"], view_a)
        frames_a = meta_a.get("frames", [])
        if not frames_a:
            frame_a_select.set_options({}, value=None)
            frame_b_select.set_options({}, value=None)
            return
        state["frame_a"] = max(0, min(int(state["frame_a"]), len(frames_a) - 1))
        frame_a_select.set_options(
            {str(i): frame_label(meta_a, i) for i in range(len(frames_a))},
            value=str(state["frame_a"]),
        )

        meta_b = dataset.frame_meta(state["tile"], view_b)
        frames_b = meta_b.get("frames", [])
        ref_date = frames_a[state["frame_a"]].get("date_utc")
        options_b = {
            str(j): frame_label(meta_b, j)
            for j in range(len(frames_b))
            if frames_b[j].get("date_utc") != ref_date
        }
        if str(state["frame_b"]) not in options_b:
            state["frame_b"] = int(next(iter(options_b))) if options_b else 0
        frame_b_select.set_options(
            options_b, value=str(state["frame_b"]) if options_b else None
        )

    def rebuild_controls() -> None:
        kind = state["combo"]
        views = available_views()

        if kind == "same_channel":
            if state["view"] not in views:
                state["view"] = views[0] if views else ""
            view_select.set_options(
                {v: v for v in views}, value=state["view"] or None
            )
            rebuild_pairs()
        else:
            ensure_channels()
            view_a_select.set_options(
                {v: v for v in views}, value=state["view_a"] or None
            )
            b_options = [
                v for v in views
                if sensor_of(v) == sensor_of(state["view_a"]) and v != state["view_a"]
            ]
            view_b_select.set_options(
                {v: v for v in b_options}, value=state["view_b"] or None
            )
            if kind == "same_day":
                rebuild_shared_frames()
            else:
                rebuild_cross_frames()

        show_single = kind == "same_channel"
        show_cross = not show_single
        view_select.set_visibility(show_single)
        if not show_single:
            pair_select.set_visibility(False)
        view_a_select.set_visibility(show_cross)
        view_b_select.set_visibility(show_cross)
        frame_shared_select.set_visibility(kind == "same_day")
        frame_a_select.set_visibility(kind == "cross_day")
        frame_b_select.set_visibility(kind == "cross_day")
        residual_dir_select.set_visibility(kind == "cross_day")
        residual_mode_select.set_visibility(kind == "cross_day")
        residual_formula_label.set_visibility(kind == "cross_day")
        update_residual_formula()

    def reset_cross_defaults() -> None:
        """Apply the documented cross-day default: A first frame, B last frame.

        B takes the last frame whose date differs from A's reference date. Both
        entering the cross-day type and changing the tile go through here, so
        the default never lands on an arbitrary index (e.g. V01 frame 1, whose
        gallery image is missing).
        """
        ensure_channels()
        if not state["view_a"] or not state["view_b"]:
            return
        frames_a = dataset.frame_meta(state["tile"], state["view_a"]).get("frames", [])
        frames_b = dataset.frame_meta(state["tile"], state["view_b"]).get("frames", [])
        if not frames_a or not frames_b:
            return
        state["frame_a"] = 0
        ref_date = frames_a[0].get("date_utc")
        state["frame_b"] = next(
            (k for k in range(len(frames_b) - 1, -1, -1)
             if frames_b[k].get("date_utc") != ref_date),
            0,
        )

    def on_combo(event) -> None:
        if not event.value:
            return
        entering_cross_day = event.value == "cross_day" and state["combo"] != "cross_day"
        state["combo"] = event.value
        state["rescale"] = True
        if state["combo"] == "cross_day":
            reset_cross_defaults()
            if entering_cross_day:
                # Documented default whenever this type is entered: positive
                # direction on the absolute residual. Changes inside the type
                # (tile/channel/frame) keep whatever the user picked.
                state["residual_dir"] = "fwd"
                state["residual_mode"] = "abs"
                residual_dir_select.value = "fwd"
                residual_mode_select.value = "abs"
        rebuild_controls()
        refresh()

    def on_tile(event) -> None:
        if not event.value:
            return
        state["tile"] = event.value
        state["rescale"] = True
        if state["combo"] == "cross_day":
            reset_cross_defaults()
        rebuild_controls()
        update_coordinate_label()
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

    def on_view_a(event) -> None:
        if not event.value:
            return
        state["view_a"] = event.value
        state["rescale"] = True
        rebuild_controls()
        refresh()

    def on_view_b(event) -> None:
        if not event.value:
            return
        state["view_b"] = event.value
        state["rescale"] = True
        rebuild_controls()
        refresh()

    def on_frame_shared(event) -> None:
        if event.value is None:
            return
        state["frame_shared"] = int(event.value)
        state["rescale"] = True
        refresh()

    def on_frame_a(event) -> None:
        if event.value is None:
            return
        state["frame_a"] = int(event.value)
        state["rescale"] = True
        # A date change may collide with B's date, so B's options are rebuilt
        # to exclude the reference date.
        rebuild_controls()
        refresh()

    def on_frame_b(event) -> None:
        if event.value is None:
            return
        state["frame_b"] = int(event.value)
        state["rescale"] = True
        refresh()

    def on_residual_dir(event) -> None:
        if not event.value:
            return
        state["residual_dir"] = event.value
        state["rescale"] = True
        update_residual_formula()
        refresh()

    def on_residual_mode(event) -> None:
        if not event.value:
            return
        state["residual_mode"] = event.value
        state["rescale"] = True
        update_residual_formula()
        refresh()

    def update_residual_formula() -> None:
        if state["combo"] != "cross_day":
            residual_formula_label.set_text("")
            return
        direction = state["residual_dir"]
        formulas = {
            "fwd": "正向 fwd = d(A_ref, B_other) − base",
            "rev": "反向 rev = d(B_ref, A_other) − base",
            "sym": "对称化 sym = (正向 + 反向) / 2",
        }
        value_mode = (
            "带符号：蓝 = 残差减小，白 ≈ 0，红 = 残差增大"
            if state["residual_mode"] == "signed"
            else "绝对值：|残差| 的分档色阶"
        )
        residual_formula_label.set_text(
            f"当前显示：{formulas[direction]}\n"
            f"取值：{value_mode}\n"
            f"阈值 / 高亮 / 超阈值统计作用于 |残差|"
        )

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
        threshold_value_label.set_text(
            f"当前阈值 = {clamped:.{scale['decimals']}f}"
        )
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

    def on_band_mode(event) -> None:
        if not event.value:
            return
        state["band_mode"] = event.value
        refresh()

    def on_levels(event) -> None:
        """Clamp the band count; the field is re-sent so raw text cannot linger."""
        if syncing["levels"]:
            return
        if event.value is None:
            # Clearing the field is not a request to change the band count.
            syncing["levels"] = True
            try:
                levels_number.value = state["levels"]
                levels_number.update()
            finally:
                syncing["levels"] = False
            return
        try:
            value = int(float(event.value))
        except (TypeError, ValueError):
            value = state["levels"]
        value = int(np.clip(value, LEVELS_MIN, LEVELS_MAX))
        state["levels"] = value
        if event.value != value:
            syncing["levels"] = True
            try:
                levels_number.value = value
                levels_number.update()
            finally:
                syncing["levels"] = False
        refresh()

    def on_edges(event) -> None:
        state["edges"] = bool(event.value)
        refresh()

    def on_mask_toggle(event) -> None:
        state["use_mask"] = bool(event.value)
        state["rescale"] = True
        refresh()

    def update_coordinate_label() -> None:
        centre, parent = coordinate_text(dataset.manifest(state["tile"]))
        coordinate_label.set_text(
            f"瓦片 {state['tile']}\n中心坐标 = {centre}\n原始父目录 = {parent}"
        )

    def root_input_text() -> str:
        """The path to put in the field: a value that resolves back identically.

        A package whose tile container is not named ``tiles`` round-trips more
        reliably through the container path, because the root alone would need
        the resolver to guess which child holds the tiles.
        """
        if dataset.root == dataset.data_root / "tiles":
            return str(dataset.data_root)
        return str(dataset.root)

    def show_root_error(message: object) -> None:
        data_root_status.set_text(f"路径未应用：{message}")
        data_root_status.classes(
            replace="text-xs text-red-600 whitespace-pre-line break-all"
        )

    def apply_data_root() -> None:
        """Switch the whole page to another package, or keep the current one.

        A rejected path never touches the live dataset, so a typo cannot blank
        the page. Everything that can fail on a malformed package (manifest
        parsing, tile labelling, the first view lookup) is exercised on the
        candidate *before* it is committed; if the post-commit rebuild still
        fails, the previous dataset, tile list and state are restored.
        """
        nonlocal dataset, tiles
        previous = (dataset, tiles, copy.deepcopy(state), state["tile"])
        try:
            new_root, tiles_dir = resolve_data_paths(data_root_input.value)
            candidate = Dataset(new_root, tiles_dir)
            new_tiles = candidate.tiles()
            if not new_tiles:
                raise ValueError(f"{new_root} 下没有找到任何瓦片")
            # Labels load every manifest, so a broken manifest is reported here
            # rather than halfway through the UI rebuild.
            new_labels = preflight_labels(candidate, new_tiles)
            target = state["tile"] if state["tile"] in new_tiles else new_tiles[0]
            try:
                views = candidate.all_views(target)
            except OSError as error:
                raise ValueError(f"{target} 的嵌入目录无法读取：{error}") from error
            if not views:
                raise ValueError(f"{target} 下没有可用的嵌入视图")
        except (ValueError, OSError, KeyError, AttributeError, TypeError,
                json.JSONDecodeError) as error:
            show_root_error(error)
            return

        dataset = candidate
        tiles = new_tiles
        try:
            data_root_input.value = root_input_text()
            data_root_status.set_text(
                f"当前数据根 = {dataset.data_root}（{len(tiles)} 个瓦片）"
            )
            data_root_status.classes(
                replace="text-xs text-gray-600 whitespace-pre-line break-all"
            )
            state["tile"] = target
            tile_select.set_options(new_labels, value=state["tile"])
            if state["combo"] == "cross_day":
                reset_cross_defaults()
            else:
                ensure_channels()
            state["rescale"] = True
            rebuild_controls()
            update_coordinate_label()
            refresh()
        except Exception as error:  # noqa: BLE001 - any failure must roll back
            # Unexpected exception types are still rolled back, but they are
            # logged so a programming error is not silently presented as a bad
            # path. The rollback body is guarded separately: if restoration
            # itself fails, the page state stays diagnosable instead of blank.
            logging.exception("数据根切换失败，回滚到 %s", previous[0].data_root)
            restore_previous(previous, error)

    def restore_previous(previous, error: Exception) -> None:
        """Undo a failed data-root switch and redraw the previous package."""
        nonlocal dataset, tiles
        dataset, tiles, saved_state, saved_tile = previous
        state.clear()
        state.update(saved_state)
        state["tile"] = saved_tile
        # Both flags must be in place *before* set_options: when the failed
        # attempt had switched the tile, setting the old value fires on_tile,
        # whose refresh would otherwise rescale to the default threshold and
        # drop the user's own value.
        state["rescale"] = True
        state["rescale_keep"] = True
        try:
            data_root_input.value = root_input_text()
            data_root_status.classes(
                replace="text-xs text-red-600 whitespace-pre-line break-all"
            )
            show_root_error(f"{error}（已保留原来的数据根）")
            tile_select.set_options(
                {name: tile_label(name) for name in tiles}, value=state["tile"]
            )
            rebuild_controls()
            update_coordinate_label()
            # Push the restored scale back to the widgets in case the failed
            # attempt had already re-scaled them for the other package. Both
            # flags are set again here: setting the tile above may fire on_tile,
            # whose refresh consumes them, and a second refresh with no flag
            # would replace the user's threshold with the default.
            state["rescale"] = True
            state["rescale_keep"] = True
            refresh()
        except Exception:  # noqa: BLE001 - keep the original cause visible
            logging.exception("回滚界面重建失败，数据根已恢复为 %s", dataset.data_root)
            state.pop("rescale_keep", None)
            state["rescale"] = False

    # The page itself never scrolls: the row is pinned to the viewport height and
    # each column scrolls inside its own box, so reading the long control/stat
    # column cannot push the heat map or gallery figures out of view.
    with ui.row().classes(
        "w-full h-screen items-stretch no-wrap gap-6 p-4 overflow-x-auto overflow-y-hidden"
    ):
        with ui.column().classes(
            "w-[360px] shrink-0 gap-3 h-full min-h-0 overflow-y-auto pr-2 panel-scroll"
        ):
            ui.label("AEF 无监督变化检测").classes("text-xl font-semibold")
            semantics_label = ui.label().classes("text-xs text-gray-600 whitespace-pre-line")

            ui.label("数据根目录").classes("text-sm font-medium")
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                data_root_input = ui.input(value=root_input_text()).classes("flex-1")
                ui.button("应用", on_click=apply_data_root).props("dense")
            ui.label(
                "可填数据包根（含 tiles/ 与 gallery/）或瓦片目录（其下为瓦片子目录）；"
                "相对路径按项目根解析"
            ).classes("text-xs text-gray-500")
            data_root_status = ui.label(
                f"当前数据根 = {dataset.data_root}（{len(tiles)} 个瓦片）"
            ).classes("text-xs text-gray-600 whitespace-pre-line break-all")

            combo_select = ui.select(COMBOS, value=state["combo"], label="组合类型").classes("w-full")
            combo_select.on_value_change(on_combo)

            tile_select = ui.select(tile_options, value=state["tile"], label="瓦片").classes("w-full")
            tile_select.on_value_change(on_tile)

            coordinate_label = ui.label().classes(
                "text-xs text-gray-600 whitespace-pre-line break-all"
            )
            update_coordinate_label()

            view_select = ui.select({}, label="视图").classes("w-full")
            view_select.on_value_change(on_view)

            view_a_select = ui.select({}, label="通道 A").classes("w-full")
            view_a_select.on_value_change(on_view_a)

            view_b_select = ui.select({}, label="通道 B（同传感器、不同于 A）").classes("w-full")
            view_b_select.on_value_change(on_view_b)

            pair_select = ui.select({}, label="帧对").classes("w-full")
            pair_select.on_value_change(on_pair)

            frame_shared_select = ui.select({}, label="帧（两通道同日期）").classes("w-full")
            frame_shared_select.on_value_change(on_frame_shared)

            frame_a_select = ui.select({}, label="A 通道帧").classes("w-full")
            frame_a_select.on_value_change(on_frame_a)

            frame_b_select = ui.select({}, label="B 通道帧（日期不同于 A）").classes("w-full")
            frame_b_select.on_value_change(on_frame_b)

            residual_dir_select = ui.select(
                RESIDUAL_DIRECTIONS, value=state["residual_dir"], label="残差方向"
            ).classes("w-full")
            residual_dir_select.on_value_change(on_residual_dir)

            residual_mode_select = ui.select(
                RESIDUAL_MODES, value=state["residual_mode"], label="残差取值"
            ).classes("w-full")
            residual_mode_select.on_value_change(on_residual_mode)

            residual_formula_label = ui.label().classes("text-xs text-gray-600 whitespace-pre-line")

            threshold_caption = ui.label().classes("text-sm text-gray-600")
            threshold_value_label = ui.label().classes("text-xs text-gray-600")
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

            ui.separator()
            ui.label("热力图色阶分层").classes("text-sm font-medium")
            band_mode_select = ui.select(
                BAND_MODES, value=state["band_mode"], label="色阶模式"
            ).classes("w-full")
            band_mode_select.on_value_change(on_band_mode)
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                levels_number = ui.number(
                    label="档数（2–64）",
                    value=state["levels"],
                    min=LEVELS_MIN,
                    max=LEVELS_MAX,
                    step=1,
                    format="%d",
                    precision=0,
                ).classes("w-40")
                edges_switch = ui.switch("档间边界线", value=state["edges"]).classes("text-sm")
            levels_number.on_value_change(on_levels)
            edges_switch.on_value_change(on_edges)
            ui.label(
                "固定分段覆盖全量程；动态分段把当前数据范围拉伸到整条色带，"
                "上界与阈值量程一致"
            ).classes("text-xs text-gray-500")

            mask_switch = ui.switch(
                "应用 common_mask（默认关闭：全部像素参与着色与统计；"
                "开启后掩膜外显示中性灰且不计入统计）",
                value=state["use_mask"],
            ).classes("text-sm")
            mask_switch.on_value_change(on_mask_toggle)

            ui.separator()
            meta_label = ui.label().classes("text-sm text-gray-600 whitespace-pre-line")
            stats_label = ui.label().classes("text-sm whitespace-pre-line")
            ui.separator()
            colorbar_label = ui.label().classes("text-xs text-gray-500")
            # shrink-0 keeps the bar at its 14px height: as a flex child of the
            # scrolling left column it would otherwise be squeezed to 0.
            colorbar = (
                ui.image()
                .classes("w-full h-[14px] shrink-0")
                .style("image-rendering: pixelated")
            )
            ui.label("灰色区域 = common_mask 排除的像素（仅在开关开启时出现）").classes("text-xs text-gray-500")

        with ui.column().classes(
            "w-[540px] shrink-0 items-center gap-2 h-full min-h-0 overflow-y-auto panel-scroll"
        ):
            heat_title = ui.label("变化程度热力图（128 × 128，顺时针旋转 90°）").classes("text-sm font-medium")
            # shrink-0 keeps the square from being squeezed by the flex column on a
            # short viewport; the column then scrolls instead of cropping the image.
            heat_image = (
                ui.image()
                .classes("w-[520px] h-[520px] shrink-0")
                .props("fit=contain")
                .style("image-rendering: pixelated")
            )

        with ui.column().classes(
            "flex-1 min-w-[360px] gap-3 h-full min-h-0 overflow-y-auto panel-scroll"
        ):
            ui.label("gallery 原始图像（顺时针旋转 90°，两帧并排）").classes("text-sm font-medium")
            with ui.row().classes("w-full items-start no-wrap gap-4"):
                image_col_0 = ui.column().classes("flex-1 min-w-0 gap-1 self-start")
                with image_col_0:
                    caption_0 = ui.label().classes(
                        "text-sm font-medium leading-snug h-[2.75rem] overflow-hidden"
                    )
                    image_0 = ui.image().classes("block").props("fit=contain")
                    missing_0 = ui.label().classes(
                        "text-sm text-red-600 whitespace-pre-line"
                    )
                image_col_1 = ui.column().classes("flex-1 min-w-0 gap-1 self-start")
                with image_col_1:
                    caption_1 = ui.label().classes(
                        "text-sm font-medium leading-snug h-[2.75rem] overflow-hidden"
                    )
                    image_1 = ui.image().classes("block").props("fit=contain")
                    missing_1 = ui.label().classes(
                        "text-sm text-red-600 whitespace-pre-line"
                    )

    rebuild_controls()
    update_residual_formula()
    refresh()


@ui.page("/")
def index() -> None:
    ui.dark_mode(False)
    ui.add_head_html(
        "<style>"
        ".nicegui-content{padding:0;align-items:stretch;justify-content:flex-start;}"
        "html,body{height:100%;overflow:hidden;}"
        "body{background:#f5f6f8;}"
        # Thin, always-visible scrollbar so each column's own scroll position is
        # distinguishable from the (disabled) page scroll.
        ".panel-scroll{scrollbar-width:thin;scrollbar-color:#b8bdc7 #eceef2;}"
        ".panel-scroll::-webkit-scrollbar{width:10px;}"
        ".panel-scroll::-webkit-scrollbar-track{background:#eceef2;border-radius:5px;}"
        ".panel-scroll::-webkit-scrollbar-thumb{background:#b8bdc7;border-radius:5px;"
        "border:2px solid #eceef2;}"
        ".panel-scroll::-webkit-scrollbar-thumb:hover{background:#97a0ad;}"
        "</style>"
    )
    build_page(_initial_paths["data_root"], _initial_paths["tiles_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="initial data package root (default: data/spectral)",
    )
    parser.add_argument("--show", action="store_true", help="open a browser window")
    args = parser.parse_args()
    data_root, tiles_dir = resolve_data_paths(args.data_root)
    _initial_paths["data_root"] = data_root
    _initial_paths["tiles_dir"] = tiles_dir
    ui.run(host=args.host, port=args.port, title="AEF 无监督变化检测", reload=False, show=args.show)


if __name__ in {"__main__", "__mp_main__"}:
    main()
