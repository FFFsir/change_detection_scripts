"""Read-only access to embedding packages.

One ``Dataset`` instance points at exactly one package root plus the directory
that actually holds the tiles. Both the legacy ``data/spectral`` packages and
the Dynamic World packages are read through this class; the differences
(view naming, masks, gallery) are resolved here with a documented fallback
order, never by guessing in the pages.
"""

from __future__ import annotations

import base64
import io
import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DW_PACKAGE_NAME = "AEF_DW_change_data_20261031"


def default_data_root(repo_root: Path = REPO_ROOT) -> Path:
    """Default AEF package root: ``data/spectral`` of the checkout we run in.

    ``data/`` is git-ignored, so a linked worktree has none of its own; walking
    up finds the main checkout's copy instead of reporting an empty dataset.
    The nominal path is returned when nothing exists, which keeps the
    "directory does not exist" message pointing at the documented place.
    """
    for base in (repo_root, *repo_root.parents[:4]):
        candidate = base / "data" / "spectral"
        if candidate.is_dir():
            return candidate
    return repo_root / "data" / "spectral"


DEFAULT_DATA_ROOT = default_data_root()


def default_dw_data_root(repo_root: Path = REPO_ROOT) -> Path:
    """Locate the reference DW package without assuming the checkout layout.

    The package normally sits next to the repository, but a linked worktree
    (``.worktrees/<change>``) puts it two levels up, so both places are
    checked before falling back to the nominal next-to-repo path.
    """
    for base in (*repo_root.parents[:4], repo_root):
        candidate = base / DW_PACKAGE_NAME
        if candidate.is_dir():
            return candidate
    return repo_root.parent / DW_PACKAGE_NAME


DEFAULT_DW_DATA_ROOT = default_dw_data_root()

EMBEDDING_CACHE_SIZE = 8
# Rotated gallery figures are kept as base64 data URLs (~300 KiB each); the
# cache only exists so threshold changes do not re-encode the same strip.
ROTATED_IMAGE_CACHE_SIZE = 12


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
        self._mask_source_cache: dict[tuple[str, str], str] = {}
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

    def tile_category(self, tile: str) -> str:
        """Human-readable tile kind, tolerating both manifest dialects."""
        manifest = self.manifest(tile)
        return str(
            manifest.get("category")
            or manifest.get("region_id")
            or manifest.get("tile_id")
            or "未分类"
        )

    def tile_label(self, tile: str) -> str:
        """One-line tile description, tolerant of both manifest dialects.

        Legacy packages carry ``category``; Dynamic World packages carry
        ``region_id``/``tile_id`` instead. Falling back in that order keeps the
        selector readable without inventing metadata.
        """
        manifest = self.manifest(tile)
        return (
            f"{tile} · {self.tile_category(tile)}"
            f" · {manifest.get('split') or '未标注'}"
        )

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

    def sensor(self, view: str) -> str:
        """Sensor prefix of a view name (``sentinel2_full`` → ``sentinel2``)."""
        return view.split("_", 1)[0]

    def bandwidth(self, view: str) -> str:
        """Band group of a view name; empty for packages without band groups."""
        return view.split("_", 1)[1] if "_" in view else ""

    def mask_source(self, tile: str, view: str) -> tuple[np.ndarray, str]:
        """Mask for ``view`` plus a human-readable description of its origin.

        Order (documented in the spec, never extended ad hoc):
        ``distance_maps`` → the Dynamic World ``valid_mask`` of both periods →
        the observation ``valid_mask`` of both periods → all True.

        A ``distance_maps`` directory that exists but lacks this sensor is an
        error: silently falling back there would change the archived legacy
        semantics for a package that was supposed to have per-sensor masks.
        """
        sensor = self.sensor(view)
        key = (tile, sensor)
        if key in self._mask_cache:
            return self._mask_cache[key], self._mask_source_cache[key]

        distance_dir = self.root / tile / "distance_maps"
        if distance_dir.is_dir():
            for path in sorted(distance_dir.glob(f"{sensor}_*.npz")):
                with np.load(path) as data:
                    if "common_mask" in data:
                        mask = data["common_mask"].astype(bool)
                        self._store_mask(key, mask, f"distance_maps/{path.name} 的 common_mask")
                        return mask, self._mask_source_cache[key]
            raise FileNotFoundError(
                f"没有为 {tile} 的 {sensor} 传感器找到 common_mask"
                f"（期望 {distance_dir / (sensor + '_*.npz')}）"
            )

        dw_path = self.root / tile / "land_cover" / "dynamic_world.npz"
        if dw_path.exists():
            with np.load(dw_path, allow_pickle=False) as data:
                if "valid_mask" in data:
                    valid = data["valid_mask"].astype(bool)
                    mask = valid.all(axis=0) if valid.ndim == 3 else valid
                    self._store_mask(
                        key, mask, "DW valid_mask（两期均为有效）"
                    )
                    return mask, self._mask_source_cache[key]

        obs_path = self.root / tile / "observations" / f"{sensor}.npz"
        if obs_path.exists():
            with np.load(obs_path, allow_pickle=False) as data:
                if "valid_mask" in data:
                    valid = data["valid_mask"].astype(bool)
                    mask = valid.all(axis=0) if valid.ndim == 3 else valid
                    self._store_mask(
                        key, mask, f"observations/{sensor}.npz 的 valid_mask（两帧取与）"
                    )
                    return mask, self._mask_source_cache[key]

        shape = self.embeddings(tile, view).shape[1:3]
        mask = np.ones(shape, dtype=bool)
        self._store_mask(key, mask, "全 True（数据包未提供任何掩膜）")
        return mask, self._mask_source_cache[key]

    def _store_mask(self, key: tuple[str, str], mask: np.ndarray, source: str) -> None:
        self._mask_cache[key] = mask
        self._mask_source_cache[key] = source

    def mask(self, tile: str, view: str) -> np.ndarray:
        """Common mask of the sensor that produced ``view`` (see ``mask_source``)."""
        return self.mask_source(tile, view)[0]

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
            if not self.gallery_available():
                return None, (
                    "该数据包没有 gallery/index.json"
                    "（本瓦片可用 preview.png / review.png）"
                    + "\n请求的帧 = "
                    + f"({tile}, {source}, {date})"
                )
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

    def gallery_available(self) -> bool:
        """False for packages that ship per-tile preview figures instead."""
        return self.gallery_index.exists()

    def preview_images(self, tile: str) -> list[tuple[str, Path | None]]:
        """``preview.png`` / ``review.png`` for packages without a gallery index."""
        out: list[tuple[str, Path | None]] = []
        for name in ("preview.png", "review.png"):
            path = self.root / tile / name
            out.append((name, path if path.exists() else None))
        return out

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



# The coordinate block follows a separator in legacy parents
# (``C01_E121.4790_N25.1724_...``) but starts the string in Dynamic World
# ``region_id`` values (``E121.4172_N25.0446``), so both are accepted.
_TILE_COORD_RE = re.compile(
    r"(?:^|_)(?P<lon>[EW])(?P<lonv>\d+(?:\.\d+)?)"
    r"_(?P<lat>[NS])(?P<latv>\d+(?:\.\d+)?)"
)


def tile_coordinates(manifest: dict) -> tuple[float, float] | None:
    """Centre coordinates parsed from ``tile_manifest.json``'s ``parent``.

    The packaged tiles encode the location in the parent directory name, e.g.
    ``registered/tiles/C01_E121.4790_N25.1724_20240101_20241231``. Anything
    that does not match that shape (``controls/tiles/V01``) is reported as
    unknown instead of guessed.
    """
    # ``parent`` is the legacy field; Dynamic World manifests carry the same
    # ``E121.4172_N25.0446`` shape in ``region_id`` and have no ``parent``.
    source = manifest.get("parent") or manifest.get("region_id")
    if not isinstance(source, str):
        return None
    match = _TILE_COORD_RE.search(source)
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
    source = manifest.get("parent") or manifest.get("region_id")
    parent_text = source.strip() if isinstance(source, str) else ""
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

