"""Dynamic World label and change-transition access.

The DW package keeps its labels under ``<tile>/land_cover/`` and its change
core under ``<tile>/masks/``. Reading is deliberately separate from rendering:
:class:`DwTile` turns the npz files into plain arrays plus a manifest, so tests
can check shapes, labels and fallbacks without starting NiceGUI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Neutral colour for ``255``/invalid labels; same family as the AEF mask grey.
INVALID_COLOR = (110, 112, 118)
# Overlay colours chosen to stay distinguishable from all nine DW class colours.
CORE_OUTLINE_COLOR = (0, 255, 255)
BBOX_COLOR = (255, 0, 255)
# Used only when the package ships no legend file at all.
_FALLBACK_PALETTE = (
    (0, 158, 117),
    (56, 125, 73),
    (136, 176, 83),
    (122, 135, 198),
    (228, 150, 53),
    (223, 195, 90),
    (196, 40, 27),
    (165, 155, 143),
    (179, 159, 225),
)


@dataclass(frozen=True)
class DwLegend:
    classes: tuple[dict, ...]
    nodata: int
    source: str

    def palette(self) -> dict[int, tuple[int, int, int]]:
        return {int(c["code"]): tuple(c["rgb"]) for c in self.classes}

    def names(self) -> dict[int, str]:
        return {int(c["code"]): str(c["name"]) for c in self.classes}

    def class_name(self, code: int) -> str:
        return self.names().get(int(code), f"未知类别 {int(code)}")

    def label(self, code: int) -> str:
        if int(code) == self.nodata:
            return f"{int(code)} 无效"
        return f"{int(code)} {self.class_name(code)}"


def _hex_rgb(text: str) -> tuple[int, int, int]:
    value = text.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"无效的十六进制颜色：{text}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def load_legend(data_root: Path, class_names: np.ndarray | None = None) -> DwLegend:
    """Read ``dynamic_world_legend.json``, falling back to the npz class names.

    The legend is the authority for both names and colours, so the page never
    hardcodes a palette that could drift from the package it is displaying.
    """
    path = Path(data_root) / "dynamic_world_legend.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        classes = []
        for entry in raw.get("classes", []):
            classes.append(
                {
                    "code": int(entry["code"]),
                    "name": str(entry.get("name", f"类别 {entry['code']}")),
                    "color": str(entry.get("color", "#808080")),
                    "rgb": _hex_rgb(str(entry.get("color", "#808080"))),
                }
            )
        if classes:
            return DwLegend(
                classes=tuple(classes),
                nodata=int(raw.get("nodata", 255)),
                source=path.name,
            )
    names = [str(n) for n in (class_names if class_names is not None else [])]
    classes = tuple(
        {
            "code": code,
            "name": names[code] if code < len(names) else f"类别 {code}",
            "color": "#%02x%02x%02x" % _FALLBACK_PALETTE[code % len(_FALLBACK_PALETTE)],
            "rgb": _FALLBACK_PALETTE[code % len(_FALLBACK_PALETTE)],
        }
        for code in range(max(len(names), 9))
    )
    return DwLegend(classes=classes, nodata=255, source="内置占位配色（未找到图例文件）")


@dataclass
class DwTile:
    """One tile's two-period DW labels, probabilities and change transition."""

    name: str
    manifest: dict
    labels: np.ndarray            # (2, 128, 128) uint8, 255 = nodata
    probabilities: np.ndarray | None  # (2, 128, 128, 9) float32
    valid_mask: np.ndarray        # (2, 128, 128) bool
    pmax: np.ndarray | None       # (2, 128, 128) float32
    class_names: np.ndarray | None
    candidate_mask: np.ndarray | None   # (128, 128) bool
    bbox: tuple[int, int, int, int] | None  # (y0, x0, y1, x1), end-exclusive
    before_label: int | None
    after_label: int | None
    transition_note: str

    @property
    def periods(self) -> int:
        """Number of DW periods actually present (2 in every shipped package)."""
        return int(self.labels.shape[0])

    @property
    def last_period(self) -> int:
        """Index of the last period, which is what "after" always means."""
        return self.periods - 1

    @property
    def dates(self) -> tuple[str, str]:
        before = str(self.manifest.get("before_date") or "?")
        after = str(self.manifest.get("after_date") or "?")
        return before, after

    @property
    def core_pixels(self) -> int | None:
        """Pixel count of ``candidate_mask`` (the value the page must report)."""
        if self.candidate_mask is None:
            return None
        return int(self.candidate_mask.sum())

    @property
    def manifest_core_pixels(self) -> int | None:
        value = self.manifest.get("core_pixels")
        return int(value) if isinstance(value, (int, float)) else None

    def period_valid(self, index: int) -> np.ndarray:
        """Valid pixels of one period: DW valid flag and a real class code."""
        labels = self.labels[index]
        return (self.valid_mask[index]) & (labels != 255)


def dw_label_path(tile_dir: Path) -> Path:
    return Path(tile_dir) / "land_cover" / "dynamic_world.npz"


def has_dw_labels(tile_dir: Path) -> bool:
    return dw_label_path(tile_dir).exists()


def load_dw_tile(tile_dir: Path, name: str | None = None) -> DwTile:
    """Load one tile; raises ``ValueError`` with a readable reason on mismatch."""
    tile_dir = Path(tile_dir)
    labels_path = dw_label_path(tile_dir)
    if not labels_path.exists():
        raise ValueError(f"{tile_dir.name} 没有 land_cover/dynamic_world.npz")
    with np.load(labels_path, allow_pickle=False) as data:
        labels = data["label"]
        if labels.ndim != 3 or labels.shape[0] < 2 or labels.shape[-2:] != (128, 128):
            raise ValueError(f"{labels_path} 的 label 形状异常：{labels.shape}")
        valid = (
            data["valid_mask"].astype(bool)
            if "valid_mask" in data
            else np.ones(labels.shape, dtype=bool)
        )
        probabilities = (
            data["probabilities"].astype(np.float32)
            if "probabilities" in data
            else None
        )
        pmax = data["pmax"].astype(np.float32) if "pmax" in data else None
        class_names = data["class_names"] if "class_names" in data else None
    labels = labels.astype(np.uint8)

    # Shapes are checked up front: a per-period mask stored without the period
    # axis would otherwise be sliced as rows and silently produce nonsense.
    if valid.shape != labels.shape:
        raise ValueError(
            f"{labels_path} 的 valid_mask 形状 {valid.shape} 与 label {labels.shape} 不符"
        )
    if pmax is not None and pmax.shape != labels.shape:
        raise ValueError(
            f"{labels_path} 的 pmax 形状 {pmax.shape} 与 label {labels.shape} 不符"
        )
    if probabilities is not None and (
        probabilities.ndim != 4 or probabilities.shape[:3] != labels.shape
    ):
        raise ValueError(
            f"{labels_path} 的 probabilities 形状 {probabilities.shape} 与 label {labels.shape} 不符"
        )

    manifest_path = tile_dir / "tile_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )

    candidate_mask = None
    bbox = None
    before_label = None
    after_label = None
    note = ""
    transition_path = tile_dir / "masks" / "label_transition.npz"
    if transition_path.exists():
        with np.load(transition_path, allow_pickle=False) as transition:
            if "candidate_mask" in transition:
                candidate_mask = transition["candidate_mask"].astype(bool)
            if "bbox_yxyx" in transition:
                values = transition["bbox_yxyx"].astype(int).tolist()
                if len(values) == 4:
                    bbox = tuple(values)  # type: ignore[assignment]
            if "before_label" in transition:
                before_label = int(transition["before_label"])
            if "after_label" in transition:
                after_label = int(transition["after_label"])
    else:
        note = f"缺少 {transition_path.relative_to(tile_dir)}（无变化核心叠加）"

    if candidate_mask is not None and candidate_mask.shape != labels.shape[-2:]:
        raise ValueError(
            f"{transition_path} 的 candidate_mask 形状 {candidate_mask.shape} 与标签不符"
        )

    return DwTile(
        name=name or tile_dir.name,
        manifest=manifest,
        labels=labels,
        probabilities=probabilities,
        valid_mask=valid,
        pmax=pmax,
        class_names=class_names,
        candidate_mask=candidate_mask,
        bbox=bbox,
        before_label=before_label,
        after_label=after_label,
        transition_note=note,
    )


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Inner boundary of a boolean mask: pixels touching a ``False`` neighbour.

    The mask is padded with ``False`` so a region that touches the tile edge is
    outlined on that side too; otherwise the outline would run off the image.
    """
    inside = mask.astype(bool)
    padded = np.zeros((inside.shape[0] + 2, inside.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = inside
    eroded = np.ones_like(inside)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = padded[1 + dy : 1 + dy + inside.shape[0],
                         1 + dx : 1 + dx + inside.shape[1]]
        eroded &= shifted
    return inside & ~eroded


def draw_bbox(rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Draw a 1-pixel rectangle; ``bbox`` uses ``[y0, x0, y1, x1]`` (end-exclusive)."""
    out = rgb.copy()
    height, width = out.shape[:2]
    y0, x0, y1, x1 = (int(v) for v in bbox)
    y0, y1 = max(0, min(y0, height - 1)), max(0, min(y1, height))
    x0, x1 = max(0, min(x0, width - 1)), max(0, min(x1, width))
    if y1 <= y0 or x1 <= x0:
        return out
    out[y0, x0:x1] = BBOX_COLOR
    out[y1 - 1, x0:x1] = BBOX_COLOR
    out[y0:y1, x0] = BBOX_COLOR
    out[y0:y1, x1 - 1] = BBOX_COLOR
    return out


def label_rgb(
    labels: np.ndarray,
    valid: np.ndarray,
    legend: DwLegend,
    *,
    candidate_mask: np.ndarray | None = None,
    bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """One period's label map in the official colours, with optional overlays."""
    palette = legend.palette()
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    for code, colour in palette.items():
        rgb[labels == code] = colour
    rgb[(labels == legend.nodata) | (~valid)] = INVALID_COLOR
    if candidate_mask is not None:
        rgb[mask_boundary(candidate_mask)] = CORE_OUTLINE_COLOR
    if bbox is not None:
        rgb = draw_bbox(rgb, bbox)
    return rgb


def upscale(rgb: np.ndarray, factor: int = 4) -> np.ndarray:
    """Nearest-neighbour magnification keeps class boundaries crisp."""
    return np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)


def region_stats(
    d: np.ndarray, valid: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Distribution of ``d`` over every pixel in ``valid``.

    The DW page deliberately reports one scope only: the whole tile. No
    ``candidate_mask`` or ``bbox`` filtering happens here, and none may be
    added by a caller.
    """
    values = d[valid]
    total = int(values.size)
    if total == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "hot": 0,
            "ratio": 0.0,
        }
    hot = int(np.count_nonzero(values > threshold))
    return {
        "count": total,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "hot": hot,
        "ratio": hot / total,
    }
