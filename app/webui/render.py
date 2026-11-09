"""Pixel maths, colour ramps and PNG encoding shared by every page.

Nothing in this module touches the filesystem: callers hand in numpy arrays and
get arrays or data URLs back, so the exact same distance, banding and threshold
code drives both the AEF page and the DW comparison page.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


MASK_NEUTRAL = np.array([110, 112, 118], dtype=np.uint8)
HIGHLIGHT = np.array([235, 45, 45], dtype=np.float32)
BINARY_BACKGROUND = np.array([26, 28, 36], dtype=np.uint8)

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


def has_band_group(view: str) -> bool:
    """False for Dynamic World packages, whose views are plain sensors."""
    return "_" in view


def band_text(view: str) -> str:
    """Band group for the metadata panel; honest about packages without one."""
    return band_of(view) if has_band_group(view) else "无波段组"


def distribution_label(view: str) -> str:
    """In/out-of-distribution marker.

    The ``full``/ablation distinction only exists for packages that encode band
    groups. A view named after a bare sensor is a single view, and saying
    "out-of-distribution" there would be wrong.
    """
    if not has_band_group(view):
        return "单视图（无波段组）"
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


def band_edges_drawn(edges: bool, mode: str) -> bool:
    """Whether band boundary lines appear, given the toggle and display mode.

    Binary mode renders no palette at all, so drawing contour lines there
    would overlay the red mask; the toggle is therefore reported as inactive.
    """
    return bool(edges) and mode != "binary"


def band_range(mode: str, upper: float, signed: bool) -> tuple[float, float]:
    """Colour-domain bounds for a banding mode.

    ``fixed`` always spans the full theoretical range; ``dynamic`` follows the
    threshold scale's upper bound ``x`` so the bands cover exactly the interval
    the slider exposes. Both pages call this, so the two cannot drift.
    """
    if mode == "fixed":
        return (-1.0, 1.0) if signed else (0.0, 1.0)
    bound = float(upper)
    return (-bound, bound) if signed else (0.0, bound)


def clamp_levels(raw: object, current: int) -> int:
    """Band count from a widget value, clamped to the supported range.

    Unparseable input falls back to ``current`` so a stray keystroke cannot
    empty the palette. Callers re-send the result to the field themselves.
    """
    try:
        value = int(float(raw))  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        # ``1e400``/``inf`` parse as floats but overflow on the int cast, so
        # that counts as unparseable input rather than escaping to the caller.
        return int(current)
    return int(np.clip(value, LEVELS_MIN, LEVELS_MAX))


def residual_maps(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    ref: int,
    other: int,
    mask: np.ndarray,
    use_mask: bool = False,
) -> dict[str, np.ndarray]:
    """Cross-day cross-channel residuals for one pair of frames.

    ``base`` is the same-day baseline, ``fwd``/``rev`` the two directions and
    ``sym`` their mean. All four are clipped to ``[-1, 1]``; ``valid`` requires
    every intermediate distance to be usable. This is the single definition
    used by the page and by the tests, so a formula change cannot slip through
    a copy.
    """
    base, valid_base = pair_distance(emb_a[ref], emb_b[ref], mask, use_mask=use_mask)
    d_fwd, valid_fwd = pair_distance(emb_a[ref], emb_b[other], mask, use_mask=use_mask)
    d_rev, valid_rev = pair_distance(emb_b[ref], emb_a[other], mask, use_mask=use_mask)
    valid = valid_base & valid_fwd & valid_rev

    base = np.where(valid, base, 0.0).astype(np.float32)
    fwd = np.clip(d_fwd - base, -1.0, 1.0).astype(np.float32)
    rev = np.clip(d_rev - base, -1.0, 1.0).astype(np.float32)
    sym = ((fwd + rev) / 2.0).astype(np.float32)
    return {"base": base, "fwd": fwd, "rev": rev, "sym": sym, "valid": valid}


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

