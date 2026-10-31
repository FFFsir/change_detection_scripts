"""Independent checks for banded colouring, coordinates and data-root parsing.

Runs against the module's pure functions only (no browser), and asserts the
properties the spec promises:

* banding changes the palette but never the statistics or threshold scale;
* fixed and dynamic modes differ in their colour domain while both stay
  quantised to ``levels`` colours;
* edge lines are a display-only overlay;
* tile coordinates parse from the manifest parent and degrade to ``None``;
* data-root resolution accepts a package root and a tiles directory.

Usage:  .venv/Scripts/python.exe scratch/check_bands.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
APP = ROOT / "app" / "change_detection_webui.py"
WORK = Path(tempfile.mkdtemp(prefix="webui_bands_"))


def make_copy(source: Path, name: str, **overrides: Path) -> Path:
    """Copy the package skeleton into the temp workspace.

    Only the files these checks read are copied, so the 89 MiB embeddings stay
    in place. Copies live outside ``data/`` so a crashed run cannot leave
    sibling packages next to the real one.
    """
    target = WORK / name
    shutil.rmtree(target, ignore_errors=True)
    (target / "gallery").mkdir(parents=True)
    shutil.copy2(source / "gallery" / "index.json", target / "gallery" / "index.json")
    # One real image per source is enough to exercise path resolution without
    # copying the whole 12 MiB gallery into every temporary package.
    for name in ("C01_sentinel2_0.png", "C01_landsat_0.png"):
        source_image = source / "gallery" / name
        if source_image.exists():
            shutil.copy2(source_image, target / "gallery" / name)
    (target / "tiles").mkdir()
    for tile in sorted((source / "tiles").iterdir()):
        if not tile.is_dir():
            continue
        dest = target / "tiles" / tile.name
        dest.mkdir()
        manifest = tile / "tile_manifest.json"
        if manifest.exists():
            shutil.copy2(manifest, dest / "tile_manifest.json")
        (dest / "embeddings").mkdir()
    return target

spec = importlib.util.spec_from_file_location("webui_check", APP)
webui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(webui)

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"[PASS] {message}")
    else:
        failures.append(message)
        print(f"[FAIL] {message}")


data_root = ROOT / "data" / "spectral"
dataset = webui.Dataset(data_root)
data_root_tile_names = dataset.tiles()

# --- banding is display-only ------------------------------------------------
view = "sentinel2_full"
emb = dataset.embeddings("C01", view)
mask = dataset.mask("C01", view)
q, valid = webui.pair_distance(emb[0], emb[1], mask, use_mask=False)
scale = webui.threshold_scale(q, valid)
threshold = scale["default"]

flat = webui.render_bands(
    q, valid, threshold, "overlay",
    low=0.0, high=scale["upper"], levels=12, edges=False, signed=False,
)
edged = webui.render_bands(
    q, valid, threshold, "overlay",
    low=0.0, high=scale["upper"], levels=12, edges=True, signed=False,
)
_, usable = webui.pair_distance(emb[0], emb[1], mask, use_mask=False)
masked_valid = mask & usable
masked = webui.render_bands(
    q, masked_valid, threshold, "overlay",
    low=0.0, high=scale["upper"], levels=12, edges=False, signed=False,
)
check(
    np.all(masked[~masked_valid] == webui.MASK_NEUTRAL)
    and np.count_nonzero(~masked_valid) == int((~mask).sum()),
    f"掩膜外 {int((~mask).sum())} 个像素全部为中性灰",
)
check(
    not np.array_equal(flat, edged),
    "开启边界线确实改变了画面（等值线被绘制）",
)
hot_flat = np.count_nonzero(
    np.all(flat == webui.HIGHLIGHT.astype(np.uint8), axis=-1)
)
stats = webui.summarise(q, valid, threshold)
check(stats["hot"] > 0, f"超阈值像素存在（{stats['hot']}）")

# Quantisation: at most `levels` palette colours (plus highlight and mask grey).
palette = webui.ramp_samples(webui._RAMP, 12)
unique_flat = np.unique(flat.reshape(-1, 3), axis=0)
check(
    len(unique_flat) <= 12 + 2,
    f"12 档渲染的不同颜色数 {len(unique_flat)} ≤ 14（12 档 + 高亮 + 掩膜灰）",
)

# --- statistics and scale are unaffected by banding ------------------------
recomputed = webui.summarise(q, valid, threshold)
check(recomputed == stats, "分档渲染前后统计完全一致")
check(
    webui.threshold_scale(q, valid) == scale,
    "分档渲染前后阈值量程完全一致",
)

# --- fixed vs dynamic domains ----------------------------------------------
fixed_low, fixed_high = 0.0, 1.0
dyn_low, dyn_high = 0.0, scale["upper"]
check(
    np.isclose(fixed_high, 1.0) and np.isclose(dyn_high, scale["upper"]),
    f"固定色域 [0,1] 与动态色域 [0,{scale['upper']:.6f}] 分别取到各自上界",
)

idx_fixed = webui.band_index(q, fixed_low, fixed_high, 12)
idx_dyn = webui.band_index(q, dyn_low, dyn_high, 12)
check(
    idx_dyn.max() == 11,
    f"动态模式下最大档位达到末档（{idx_dyn.max()}）",
)
check(
    idx_fixed.max() < 11,
    f"固定模式下同一数据不占满全档（最大档位 {idx_fixed.max()}）",
)

# --- levels extremes -------------------------------------------------------
for levels in (2, 64):
    out = webui.render_bands(
        q, valid, threshold, "overlay",
        low=0.0, high=scale["upper"], levels=levels, edges=False, signed=False,
    )
    colours = len(np.unique(out.reshape(-1, 3), axis=0))
    check(colours <= levels + 2, f"{levels} 档渲染颜色数 {colours} ≤ {levels + 2}")

zero = webui.band_index(np.zeros((4, 4), dtype=np.float32), 0.0, 0.0, 12)
check(np.array_equal(zero, np.zeros((4, 4), dtype=np.int32)), "色域跨度为 0 时全部落入第 0 档")

# --- signed banding --------------------------------------------------------
r = q - 0.05
signed_rgb = webui.render_bands(
    r, valid, threshold, "overlay",
    low=-scale["upper"], high=scale["upper"], levels=12, edges=False, signed=True,
)
check(signed_rgb.shape == r.shape + (3,), "带符号分档渲染输出 RGB 形状正确")
signed_palette = webui.ramp_samples(webui._DIVERGING_RAMP, 12)
check(
    np.array_equal(signed_palette[5], signed_palette[6]) is False,
    "偶数档时中性白两侧由两档不同颜色承担",
)

# --- colour bar matches the palette ---------------------------------------
strip = webui.render_band_strip(
    low=0.0, high=scale["upper"], levels=12, edges=False, signed=False,
)
check(strip.shape == (14, 512, 3), f"色条形状 {strip.shape} == (14, 512, 3)")
strip_colours = np.unique(strip.reshape(-1, 3), axis=0)
check(
    len(strip_colours) <= 12,
    f"色条只含 {len(strip_colours)} 种颜色，与 12 档一致",
)
check(
    np.array_equal(np.unique(strip[0].reshape(-1, 3), axis=0), strip_colours),
    "色条每一行都是同一组档位颜色",
)

# --- coordinates -----------------------------------------------------------
lon, lat = webui.tile_coordinates(dataset.manifest("C01"))
check(
    abs(lon - 121.4790) < 1e-6 and abs(lat - 25.1724) < 1e-6,
    f"C01 中心坐标解析为 ({lon:.4f}, {lat:.4f})",
)
centre, parent = webui.coordinate_text(dataset.manifest("V01"))
check(centre == "无坐标信息", f"V01 无坐标时显示「无坐标信息」（父目录 {parent}）")
check(
    webui.tile_coordinates({"parent": "garbage"}) is None,
    "无法解析的 parent 返回 None 而不是抛错",
)
check(
    webui.tile_coordinates({"parent": "x_E999.0_N25.0"}) is None,
    "越界经度返回 None",
)
south, west = webui.tile_coordinates(
    {"parent": "tiles/T01_W100.5000_S10.2500_20240101"}
)
check(
    abs(south + 100.5) < 1e-9 and abs(west + 10.25) < 1e-9,
    "W/S 半球解析为负值",
)

# --- data root resolution --------------------------------------------------
check(
    webui.resolve_data_root("data/spectral") == data_root.resolve(),
    "相对路径 data/spectral 解析为数据包根",
)
check(
    webui.resolve_data_root("data/spectral/tiles") == data_root.resolve(),
    "瓦片目录被提升为父级数据包根",
)
check(
    webui.resolve_data_root(str(data_root)) == data_root.resolve(),
    "绝对路径数据包根解析正确",
)
for bad in ("docs", "data/spectral/gallery", "no_such_path"):
    try:
        webui.resolve_data_root(bad)
    except ValueError:
        print(f"[PASS] 非法路径被拒绝：{bad}")
    else:
        failures.append(f"非法路径未被拒绝：{bad}")
        print(f"[FAIL] 非法路径未被拒绝：{bad}")

# --- gallery path resolution tolerates a renamed package ------------------
renamed = webui.Dataset.__new__(webui.Dataset)
renamed.data_root = data_root
renamed.gallery_dir = data_root / "gallery"
resolved = webui.Dataset._gallery_path(
    renamed, "spectral/gallery/C01_sentinel2_0.png"
)
check(resolved.exists(), f"gallery 路径可解析到真实文件：{resolved.name}")

# A renamed sibling package must resolve inside itself, never back into the
# original `spectral/` package sitting next to it.
sibling = make_copy(data_root, "spectral_2025")
copy_ds = webui.Dataset(sibling)
copy_path = copy_ds.gallery[("C01", "sentinel2", "2024-02-29")]
check(
    Path(copy_path).resolve().is_relative_to(sibling.resolve()),
    f"改名后的同级数据包只解析自身 gallery：{Path(copy_path).parent.parent.name}",
)

# A package whose gallery image is missing must report a miss instead of
# borrowing the sibling package's file. The record stays registered (so the UI
# can distinguish "no record" from "record present, file gone"), but its path
# must point inside the package that owns the index.
sibling_no_image = make_copy(data_root, "spectral_noimage")
(sibling_no_image / "gallery" / "C01_sentinel2_0.png").unlink(missing_ok=True)
no_image_ds = webui.Dataset(sibling_no_image)
missing_path = no_image_ds.gallery.get(("C01", "sentinel2", "2024-02-29"))
check(
    missing_path is not None
    and Path(missing_path).resolve().is_relative_to(sibling_no_image.resolve())
    and not Path(missing_path).exists(),
    "缺图的同级数据包只登记本包内不存在的路径，不回落到原包图像",
)

# Escaping paths are rejected instead of reading outside the package.
escape = make_copy(data_root, "spectral_escape")
escape_ds = webui.Dataset(escape)
check(
    escape_ds._gallery_path("../spectral/gallery/C01_sentinel2_0.png") is None,
    "含 .. 的 gallery 路径被拒绝",
)
check(
    escape_ds._gallery_path(str(data_root / "gallery" / "C01_sentinel2_0.png")) is None,
    "绝对路径 gallery 记录被拒绝",
)

# Non-string / unhashable index fields must not crash the loader.
bad_types = make_copy(data_root, "spectral_badtypes")
(bad_types / "gallery" / "index.json").write_text(
    json.dumps([
        {"id": ["x"], "source": "s", "date": "d", "image": "a.png"},
        {"id": "C01", "source": "sentinel2", "date": "2024-02-29", "image": 123},
        {"id": "C01", "source": "sentinel2", "date": "2024-02-30", "image": ["a"]},
    ]),
    encoding="utf-8",
)
bad_type_ds = webui.Dataset(bad_types)
check(bad_type_ds.gallery == {}, "非法类型的索引记录被跳过且不抛异常")

# A malformed gallery index must fail loudly instead of silently yielding an
# empty mapping (the switch handler turns it into a user-visible error).
bad = make_copy(data_root, "spectral_badindex")
bad_index = bad / "gallery" / "index.json"
bad_index.write_text('{"id": "C01"}', encoding="utf-8")
try:
    webui.Dataset(bad)
except ValueError:
    print("[PASS] 非数组 gallery/index.json 被拒绝")
else:
    failures.append("非数组 gallery/index.json 未被拒绝")
    print("[FAIL] 非数组 gallery/index.json 未被拒绝")

# A tile whose manifest is broken must break preflight, not commit halfway.
broken = make_copy(data_root, "spectral_badmanifest")
(broken / "tiles" / "C01" / "tile_manifest.json").write_text("{", encoding="utf-8")
broken_ds = webui.Dataset(broken)
try:
    for name in broken_ds.tiles():
        broken_ds.manifest(name)
except (json.JSONDecodeError, ValueError):
    print("[PASS] 损坏的 tile_manifest.json 会被预检发现")
else:
    failures.append("损坏的 tile_manifest.json 未被预检发现")
    print("[FAIL] 损坏的 tile_manifest.json 未被预检发现")

# A tile directory not named "tiles" is still accepted (A2 keeps both shapes).
renamed_tiles = make_copy(data_root, "spectral_renamed")
shutil.move(str(renamed_tiles / "tiles"), str(renamed_tiles / "tiles_v2"))
root, tiles_dir = webui.resolve_data_paths(renamed_tiles / "tiles_v2")
check(
    root == renamed_tiles.resolve() and tiles_dir == (renamed_tiles / "tiles_v2").resolve(),
    "任意命名的瓦片目录都会被接受并绑定为 tiles 目录",
)
renamed_ds = webui.Dataset(root, tiles_dir)
check(renamed_ds.tiles() == data_root_tile_names, "自定义 tiles 目录能枚举出同样的瓦片")

# Pointing at the package root of a renamed container must also resolve, and
# the displayed value must round-trip through the resolver.
root_via_pkg, tiles_via_pkg = webui.resolve_data_paths(renamed_tiles)
check(
    root_via_pkg == root and tiles_via_pkg == tiles_dir,
    "只给数据根也能定位到非 tiles 命名的唯一瓦片容器",
)
round_trip = renamed_tiles / "tiles_v2" if tiles_dir.name != "tiles" else root
check(
    webui.resolve_data_paths(round_trip) == (root, tiles_dir),
    "输入框回填值再次解析得到同一数据根与瓦片目录",
)

# Two candidate containers are ambiguous and must be rejected.
ambiguous = make_copy(data_root, "spectral_ambiguous")
shutil.copytree(ambiguous / "tiles", ambiguous / "tiles_v2")
try:
    webui.resolve_data_paths(ambiguous)
except ValueError as error:
    print(f"[PASS] 多个瓦片容器被拒绝并要求直接指定：{str(error)[-30:]}")
else:
    failures.append("多个瓦片容器未被拒绝")
    print("[FAIL] 多个瓦片容器未被拒绝")

# Non-finite values must land in a band instead of crashing the palette lookup.
nan_bands = webui.band_index(
    np.array([[np.nan, np.inf, -np.inf, 0.5]], dtype=np.float32), 0.0, 1.0, 12
)
check(
    nan_bands.min() >= 0 and nan_bands.max() < 12,
    f"NaN/Inf 落在合法档位内：{nan_bands.tolist()}",
)

shutil.rmtree(WORK, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} 项检查未通过")
    sys.exit(1)
print("全部分档、坐标与数据根检查通过。")
