"""The DW label comparison page.

Shows two-period Dynamic World labels in their official colours with the change
transition overlaid, plus the embedding distance over the **whole tile**. The
statistics have exactly one scope: every valid pixel of the tile. The change
core is drawn, never used as a filter, so no "inside vs outside" numbers exist.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import numpy as np
from nicegui import ui
from PIL import Image

from .dataset import Dataset, resolve_data_paths
from .dw_data import (
    CORE_OUTLINE_COLOR,
    DwLegend,
    draw_bbox,
    has_dw_labels,
    label_rgb,
    load_dw_tile,
    load_legend,
    mask_boundary,
    region_stats,
    upscale,
)
from .layout import page_shell
from .render import (
    BAND_MODES,
    LEVELS_DEFAULT,
    LEVELS_MAX,
    LEVELS_MIN,
    MODES,
    band_edges_drawn,
    band_range as band_range_for,
    band_text,
    clamp_levels,
    pair_distance,
    png_data_url,
    render_band_strip,
    render_bands,
    threshold_scale,
)

# Label maps are magnified with nearest-neighbour so class edges stay crisp.
LABEL_ZOOM = 4
HEAT_ZOOM = 4


def dw_tiles(dataset: Dataset) -> list[str]:
    """Tiles that actually carry Dynamic World labels."""
    return [
        name
        for name in dataset.tiles()
        if has_dw_labels(dataset.root / name)
    ]


def _image_url(rgb: np.ndarray, compress_level: int = 1) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buffer, format="PNG", compress_level=compress_level
    )
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def _file_image_url(path: Path, cache: dict[str, str]) -> tuple[str | None, str]:
    """Data URL for a preview figure, plus a readable reason when missing."""
    key = str(path)
    if key in cache:
        return cache[key], ""
    if not path.exists():
        return None, f"缺少图像文件：{path}"
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        url = _image_url(np.asarray(rgb), compress_level=6)
    except OSError as error:
        return None, f"图像无法读取：{path}（{error}）"
    cache[key] = url
    return url, ""


def _view_dates(dataset: Dataset, tile: str, view: str) -> tuple[str, str]:
    frames = dataset.frame_meta(tile, view).get("frames", [])
    if len(frames) < 2:
        return "?", "?"
    return (
        str(frames[0].get("date_utc", "?")),
        str(frames[-1].get("date_utc", "?")),
    )


def _default_view(dataset: Dataset, tile: str, views: list[str]) -> str:
    """Prefer the optical sensor; DW packages name views after the sensor."""
    for candidate in ("sentinel2", "sentinel2_full"):
        if candidate in views:
            return candidate
    for view in views:
        if view.startswith("sentinel2"):
            return view
    return views[0] if views else ""


def build_dw_page(
    initial_root: Path, initial_tiles: Path | None = None
) -> None:
    page_shell("dw", subtitle="Dynamic World 两期标签 · 变化过渡叠加 · 整瓦片嵌入距离")
    state = {
        "tile": "",
        "view": "",
        "threshold": 0.30,
        "scale": {
            "max": 0.0, "step": 0.01, "lower": 0.01, "upper": 1.0,
            "default": 0.30, "decimals": 2, "fmt": "%.2f",
        },
        "rescale": True,
        # Display-only settings, mirroring the AEF page's defaults.
        "band_mode": "dynamic",
        "levels": LEVELS_DEFAULT,
        "edges": True,
        "mode": "overlay",
    }
    holder: dict = {"dataset": None, "tiles": [], "tile_data": None, "legend": None}
    image_cache: dict[str, str] = {}
    syncing = {"active": False, "levels": False}

    # ------------------------------------------------------------------ #
    # widget placeholders
    # ------------------------------------------------------------------ #
    data_root_input = None
    data_root_status = None
    tile_select = None
    view_select = None
    threshold_slider = None
    threshold_number = None
    threshold_value_label = None
    scale_label = None
    meta_label = None
    core_label = None
    pmax_label = None
    legend_container = None
    stats_label = None
    note_label = None
    label_title = None
    before_caption = None
    before_image = None
    after_caption = None
    after_image = None
    heat_title = None
    heat_image = None
    colorbar = None
    colorbar_label = None
    band_mode_select = None
    levels_number = None
    edges_switch = None
    mode_select = None
    preview_title = None
    preview_caption = None
    preview_image = None
    preview_missing = None
    review_caption = None
    review_image = None
    review_missing = None

    def root_text() -> str:
        dataset = holder["dataset"]
        if dataset is None:
            return str(initial_root)
        if dataset.root == dataset.data_root / "tiles":
            return str(dataset.data_root)
        return str(dataset.root)

    def set_status(text: str, error: bool = False) -> None:
        data_root_status.set_text(text)
        data_root_status.classes(
            replace="text-xs whitespace-pre-line break-all "
            + ("text-red-600" if error else "text-gray-600")
        )

    def clear_visuals(message: str) -> None:
        """Blank every region that shows per-tile content.

        Without this the previous tile's metadata, pmax and threshold row keep
        sitting next to the "no data" message and read as if they still applied.
        """
        for image, caption in (
            (before_image, before_caption),
            (after_image, after_caption),
        ):
            image.set_source("")
            caption.set_text("")
        heat_image.set_source("")
        heat_title.set_text("嵌入距离热力图")
        colorbar.set_source("")
        colorbar_label.set_text("")
        for image, caption, missing in (
            (preview_image, preview_caption, preview_missing),
            (review_image, review_caption, review_missing),
        ):
            image.set_source("")
            caption.set_text("")
            missing.set_text("")
            missing.set_visibility(False)
        meta_label.set_text("")
        core_label.set_text("")
        pmax_label.set_text("")
        # Without resetting these the previous tile's range and threshold keep
        # sitting next to the "no data" message and look like they still apply.
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
        scale_label.set_text("当前量程不可用：请选择其他瓦片或传感器")
        state["rescale"] = True
        note_label.set_text(
            "DW 标签是 Dynamic World 模型分类结果，不是人工核实的地面真值；"
            "pmax 是模型对该类别给出的概率。"
        )
        stats_label.set_text(message)

    def draw_legend(legend: DwLegend) -> None:
        legend_container.clear()
        with legend_container:
            ui.label(f"DW 类别图例（来源：{legend.source}）").classes(
                "text-sm font-medium"
            )
            for entry in legend.classes:
                with ui.row().classes("items-center no-wrap gap-2"):
                    ui.element("div").style(
                        "width:16px;height:16px;border-radius:3px;flex:0 0 auto;"
                        f"background:{entry['color']};border:1px solid #9ca3af"
                    )
                    ui.label(f"{entry['code']} · {entry['name']}").classes(
                        "text-xs text-gray-700"
                    )
            with ui.row().classes("items-center no-wrap gap-2"):
                ui.element("div").style(
                    "width:16px;height:16px;border-radius:3px;flex:0 0 auto;"
                    "background:#6e7076;border:1px solid #9ca3af"
                )
                ui.label(f"{legend.nodata} · 无效（nodata）").classes(
                    "text-xs text-gray-700"
                )
            with ui.row().classes("items-center no-wrap gap-2"):
                ui.element("div").style(
                    "width:16px;height:16px;flex:0 0 auto;border:2px solid "
                    "rgb(0,255,255)"
                )
                ui.label("candidate_mask 轮廓（仅叠加显示）").classes(
                    "text-xs text-gray-700"
                )
            with ui.row().classes("items-center no-wrap gap-2"):
                ui.element("div").style(
                    "width:16px;height:16px;flex:0 0 auto;border:2px solid "
                    "rgb(255,0,255)"
                )
                ui.label("bbox_yxyx 包围框").classes("text-xs text-gray-700")

    # ------------------------------------------------------------------ #
    # computation
    # ------------------------------------------------------------------ #
    def band_range() -> tuple[float, float]:
        """Colour-domain bounds for the current banding mode (non-negative d)."""
        return band_range_for(
            state["band_mode"], float(state["scale"]["upper"]), False
        )

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
        value = clamp_levels(event.value, state["levels"])
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

    def on_mode(event) -> None:
        if not event.value:
            return
        state["mode"] = event.value
        refresh()

    def rescale(q: np.ndarray, valid: np.ndarray) -> None:
        scale = threshold_scale(q, valid)
        state["scale"] = scale
        state["threshold"] = scale["default"]
        decimals = scale["decimals"]
        syncing["active"] = True
        try:
            threshold_slider._props["min"] = scale["lower"]  # noqa: SLF001
            threshold_slider._props["max"] = scale["upper"]  # noqa: SLF001
            threshold_slider._props["step"] = scale["step"]  # noqa: SLF001
            threshold_slider.update()
            threshold_number.min = scale["lower"]
            threshold_number.max = scale["upper"]
            threshold_number._props["step"] = scale["step"]  # noqa: SLF001
            threshold_number.format = f"%.{decimals}f"
            threshold_slider.value = state["threshold"]
            threshold_number.value = round(state["threshold"], decimals)
            threshold_number.update()
        finally:
            syncing["active"] = False
        scale_label.set_text(
            f"当前量程 (0, {scale['upper']:.{decimals}f}]   "
            f"步长 {scale['step']:.{decimals}f}   "
            f"最大值（整瓦片 d） {scale['max']:.{decimals}f}"
        )
        threshold_value_label.set_text(
            f"当前阈值 = {state['threshold']:.{decimals}f}"
        )

    def apply_threshold(value: float) -> None:
        if syncing["active"]:
            return
        syncing["active"] = True
        scale = state["scale"]
        step = max(scale["step"], 1e-9)
        snapped = round(round(value / step) * step, scale["decimals"])
        clamped = float(np.clip(snapped, scale["lower"], scale["upper"]))
        state["threshold"] = clamped
        threshold_slider.value = clamped
        threshold_number.value = round(clamped, scale["decimals"])
        threshold_number.update()
        syncing["active"] = False
        threshold_value_label.set_text(
            f"当前阈值 = {clamped:.{scale['decimals']}f}"
        )
        refresh()

    def refresh() -> None:
        dataset: Dataset | None = holder["dataset"]
        if dataset is None or not holder["tiles"]:
            clear_visuals("当前数据根没有可分析的 DW 标签瓦片")
            return

        tile = state["tile"]
        view = state["view"]
        dw = holder["tile_data"]
        legend = holder["legend"]
        if dw is None or legend is None or not view:
            clear_visuals("当前瓦片缺少 DW 标签或可用嵌入视图")
            return

        embeddings = dataset.embeddings(tile, view)
        if embeddings.shape[0] < 2:
            clear_visuals(f"视图 {view} 只有 {embeddings.shape[0]} 帧，无法比较两期")
            return
        first, last = 0, embeddings.shape[0] - 1
        # Label periods must describe the same two frames the distance uses;
        # a package with more periods than frames (or vice versa) is ambiguous,
        # so it is reported instead of quietly pairing 0→1 labels with 0→N
        # embeddings.
        if dw.periods != embeddings.shape[0]:
            clear_visuals(
                f"DW 标签期数（{dw.periods}）与视图 {view} 的帧数"
                f"（{embeddings.shape[0]}）不一致，无法一一对应，请换视图或数据包"
            )
            return
        mask, mask_source = dataset.mask_source(tile, view)

        d, usable = pair_distance(embeddings[first], embeddings[last], mask, use_mask=True)
        # DW validity is an additional constraint: a period with a nodata label
        # contributes nothing to the distance distribution.
        valid = usable & dw.period_valid(0) & dw.period_valid(1)

        if state["rescale"]:
            state["rescale"] = False
            rescale(d, valid)

        band_low, band_high = band_range()
        heat = render_bands(
            d,
            valid,
            state["threshold"],
            state["mode"],
            low=band_low,
            high=band_high,
            levels=state["levels"],
            edges=state["edges"],
            signed=False,
        )
        if dw.candidate_mask is not None:
            heat[mask_boundary(dw.candidate_mask)] = CORE_OUTLINE_COLOR
        if dw.bbox is not None:
            heat = draw_bbox(heat, dw.bbox)
        heat_image.set_source(png_data_url(upscale(heat, HEAT_ZOOM)))

        # The bar must draw exactly what its caption reports: in binary mode the
        # map draws no contour lines, so the bar must not draw them either.
        draw_edges = band_edges_drawn(state["edges"], state["mode"])
        bar = render_band_strip(
            low=band_low,
            high=band_high,
            levels=state["levels"],
            edges=draw_edges,
            signed=False,
        )
        colorbar.set_source(png_data_url(bar))
        mode_text = "固定分段" if state["band_mode"] == "fixed" else "动态分段"
        edges_note = "档间边界线 = 开" if draw_edges else "档间边界线 = 关"
        colorbar_label.set_text(
            f"色标（{mode_text} · {state['levels']} 档 · {edges_note}）："
            f"{band_low:.3f} → {band_high:.3f}；"
            "阈值作用于 d，高亮 = d > 阈值；青色轮廓 = candidate_mask，"
            "品红矩形 = bbox_yxyx"
        )

        # DW label dates come from the package manifest; the embedding dates
        # belong to the selected view and can differ (S1 is up to 5 days off
        # the S2/DW acquisition), so the two are never presented as one date.
        before_date, after_date = dw.dates
        emb_before, emb_after = _view_dates(dataset, tile, view)
        before_caption.set_text(
            f"前期 DW 标签 · {before_date} · {dw.name}"
        )
        after_caption.set_text(
            f"后期 DW 标签 · {after_date} · {dw.name}"
        )
        before_image.set_source(
            png_data_url(
                upscale(
                    label_rgb(
                        dw.labels[0],
                        dw.valid_mask[0],
                        legend,
                        candidate_mask=dw.candidate_mask,
                        bbox=dw.bbox,
                    ),
                    LABEL_ZOOM,
                )
            )
        )
        after_image.set_source(
            png_data_url(
                upscale(
                    label_rgb(
                        dw.labels[dw.last_period],
                        dw.valid_mask[dw.last_period],
                        legend,
                        candidate_mask=dw.candidate_mask,
                        bbox=dw.bbox,
                    ),
                    LABEL_ZOOM,
                )
            )
        )
        heat_title.set_text(
            f"嵌入距离热力图 · {view} · {emb_before} → {emb_after}"
            "（128 × 128，原生朝向，未旋转）"
        )

        stats = region_stats(d, valid, state["threshold"])
        digits = state["scale"]["decimals"]
        stats_label.set_text(
            f"统计范围 = 整瓦片全部有效像素（不按 candidate_mask / bbox 筛选）\n"
            f"有效像素 = {stats['count']} / {int(valid.size)}\n"
            f"阈值 = {state['threshold']:.{digits}f}   "
            f"超阈值像素 = {stats['hot']} / {stats['count']}  "
            f"({stats['ratio'] * 100:.2f}%)\n"
            f"d 均值 = {stats['mean']:.4f}   中位数 = {stats['median']:.4f}\n"
            f"p90 = {stats['p90']:.4f}   p95 = {stats['p95']:.4f}   "
            f"最大 = {stats['max']:.4f}\n"
            f"掩膜来源 = {mask_source}"
        )

        manifest = dw.manifest
        meta_label.set_text(
            f"tile_id = {manifest.get('tile_id') or dw.name}   "
            f"region_id = {manifest.get('region_id') or '未标注'}   "
            f"split = {manifest.get('split') or '未标注'}\n"
            f"前后类别 = {manifest.get('before_class') or '?'} → "
            f"{manifest.get('after_class') or '?'}\n"
            f"DW 标签日期 = {before_date} → {after_date}\n"
            f"视图 = {view}（{band_text(view)}）   "
            f"嵌入日期 = {emb_before} → {emb_after}   "
            f"帧 = {first} → {last}"
        )
        core = dw.core_pixels
        manifest_core = dw.manifest_core_pixels
        core_bits = [f"candidate_mask 像素 = {core}" if core is not None else "candidate_mask = 缺失"]
        if manifest_core is not None:
            core_bits.append(f"manifest core_pixels = {manifest_core}")
        if core is not None and manifest_core is not None and core != manifest_core:
            core_bits.append("两者不一致，请核对数据包")
        if dw.bbox is not None:
            core_bits.append(f"bbox_yxyx = {list(dw.bbox)}")
        if dw.before_label is not None and dw.after_label is not None:
            core_bits.append(
                f"before_label/after_label = "
                f"{dw.before_label} {legend.class_name(dw.before_label)} → "
                f"{dw.after_label} {legend.class_name(dw.after_label)}"
            )
        core_label.set_text("\n".join(core_bits))

        # The manifest values come from the package and were averaged over its
        # own labelled subset; the npz values are recomputed over the whole
        # tile. Both are labelled so the two numbers cannot be confused.
        pmax_lines = []
        for key, label in (
            ("before_mean_confidence", "前期"),
            ("after_mean_confidence", "后期"),
        ):
            value = manifest.get(key)
            if isinstance(value, (int, float)):
                pmax_lines.append(
                    f"{label}平均 pmax（manifest 记录值） = {float(value):.3f}"
                )
        if dw.pmax is not None:
            # Same definition of "valid" as the distance statistics: DW valid
            # flag and a real class code, so nodata labels cannot shift the mean.
            for index, label in ((0, "前期"), (dw.last_period, "后期")):
                period_valid = dw.period_valid(index)
                if period_valid.any():
                    pmax_lines.append(
                        f"{label} pmax 均值（整瓦片） = "
                        f"{float(dw.pmax[index][period_valid].mean()):.3f}"
                    )
        pmax_label.set_text(
            "\n".join(pmax_lines)
            + "\n以上均为模型概率（pmax），不是准确率或精度指标。"
        )

        preview_title.set_text("预览图（原生朝向，不旋转）")
        preview_caption.set_text("preview.png：前后期 S2/S1 影像与 DW 标签")
        review_caption.set_text("review.png：支持日期浏览与 DW 有效覆盖")
        for image, caption, missing, name in (
            (preview_image, preview_caption, preview_missing, "preview.png"),
            (review_image, review_caption, review_missing, "review.png"),
        ):
            url, detail = _file_image_url(dataset.root / tile / name, image_cache)
            if url is not None:
                image.set_source(url)
                image.set_visibility(True)
                missing.set_text("")
                missing.set_visibility(False)
            else:
                image.set_source("")
                image.set_visibility(False)
                missing.set_text(detail)
                missing.set_visibility(True)

        note_label.set_text(
            "DW 标签是 Dynamic World 模型分类结果，不是人工核实的地面真值；"
            "pmax 是模型对该类别给出的概率。\n"
            "本页统计只描述嵌入距离在整瓦片上的分布，不评估标签正确性；"
            "本页不做 S2 ↔ S1 跨传感器比较。"
        )

    # ------------------------------------------------------------------ #
    # data root / selectors
    # ------------------------------------------------------------------ #
    def load_package(new_root: Path, tiles_dir: Path) -> Dataset:
        candidate = Dataset(new_root, tiles_dir)
        names = dw_tiles(candidate)
        if not names:
            raise ValueError(
                "该数据包不含 DW 标签"
                "（未找到 tiles/*/land_cover/dynamic_world.npz）"
            )
        # Read one tile up front so a broken npz/manifest fails before commit.
        first = names[0]
        preflight = load_dw_tile(candidate.root / first, first)
        views = candidate.views(first)
        if not views:
            raise ValueError(f"{first} 没有可比较两帧的嵌入视图")
        holder["dataset"] = candidate
        holder["tiles"] = names
        holder["preflight"] = (first, preflight)
        return candidate

    def commit_selection(target: str | None = None) -> None:
        dataset: Dataset | None = holder["dataset"]
        if dataset is None or not holder["tiles"]:
            clear_visuals("当前数据根没有可分析的 DW 标签瓦片")
            return
        tile = target if target in holder["tiles"] else holder["tiles"][0]
        state["tile"] = tile
        preflight = holder.pop("preflight", None)
        if preflight and preflight[0] == tile:
            dw = preflight[1]
        else:
            dw = load_dw_tile(dataset.root / tile, tile)
        holder["tile_data"] = dw
        holder["legend"] = load_legend(dataset.data_root, dw.class_names)
        draw_legend(holder["legend"])

        views = dataset.views(tile)
        state["view"] = _default_view(dataset, tile, views)
        view_select.set_options({v: v for v in views}, value=state["view"] or None)
        tile_select.set_options(
            {name: dataset.tile_label(name) for name in holder["tiles"]},
            value=tile,
        )
        state["rescale"] = True
        refresh()

    def apply_data_root() -> None:
        previous = (
            holder["dataset"],
            list(holder["tiles"]),
            dict(state),
            holder.get("tile_data"),
            holder.get("legend"),
        )
        try:
            new_root, tiles_dir = resolve_data_paths(data_root_input.value)
            load_package(new_root, tiles_dir)
        except (ValueError, OSError, KeyError, AttributeError, TypeError,
                json.JSONDecodeError) as error:
            set_status(f"路径未应用：{error}", error=True)
            return
        try:
            data_root_input.value = root_text()
            dataset = holder["dataset"]
            skipped = [
                name for name in dataset.tiles()
                if name not in holder["tiles"]
            ]
            skipped_note = (
                f"；跳过不含 DW 标签的瓦片 = {', '.join(skipped)}" if skipped else ""
            )
            set_status(
                f"当前数据根 = {dataset.data_root}"
                f"（{len(holder['tiles'])} 个含 DW 标签的瓦片{skipped_note}）"
            )
            # Keep the current tile when the new package still has it, exactly
            # like the AEF page; otherwise fall back to the first DW tile.
            commit_selection(state["tile"])
        except Exception as error:  # noqa: BLE001 - any failure must roll back
            logging.exception("DW 页数据根切换失败，回滚")
            holder["dataset"], holder["tiles"], state_saved, dw_saved, legend_saved = previous
            holder["tile_data"] = dw_saved
            holder["legend"] = legend_saved
            state.clear()
            state.update(state_saved)
            try:
                data_root_input.value = root_text()
                set_status(f"路径未应用：{error}（已保留原来的数据根）", error=True)
                if holder["dataset"] is not None and holder["tiles"]:
                    commit_selection(state["tile"])
            except Exception:  # noqa: BLE001 - keep the original cause visible
                logging.exception("DW 页回滚失败")

    def on_tile(event) -> None:
        if not event.value:
            return
        previous_tile = state["tile"]
        state["tile"] = event.value
        try:
            commit_selection(event.value)
        except (ValueError, OSError, KeyError, AttributeError, TypeError,
                json.JSONDecodeError) as error:
            logging.exception("DW 瓦片读取失败：%s", event.value)
            state["tile"] = previous_tile
            set_status(f"瓦片 {event.value} 读取失败：{error}", error=True)
            try:
                commit_selection(previous_tile)
            except Exception:  # noqa: BLE001 - keep the cause visible
                logging.exception("DW 页恢复上一瓦片失败")

    def on_view(event) -> None:
        if not event.value:
            return
        state["view"] = event.value
        state["rescale"] = True
        refresh()

    # ------------------------------------------------------------------ #
    # layout
    # ------------------------------------------------------------------ #
    with ui.row().classes(
        "w-full h-[calc(100vh-3.5rem)] items-stretch no-wrap gap-6 p-4 "
        "overflow-x-auto overflow-y-hidden"
    ):
        with ui.column().classes(
            "w-[360px] shrink-0 gap-3 h-full min-h-0 overflow-y-auto pr-2 panel-scroll"
        ):
            ui.label("AEF 变化检测与 DW 标签对比").classes("text-xl font-semibold")
            ui.label(
                "传感器内前后两期嵌入距离（d = (1 − cos) / 2），统计范围为整个瓦片；"
                "DW 标签为模型分类，非人工核实真值。"
            ).classes("text-xs text-gray-600 whitespace-pre-line")

            ui.label("数据根目录").classes("text-sm font-medium")
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                data_root_input = ui.input(value=str(initial_root)).classes("flex-1")
                ui.button("应用", on_click=apply_data_root).props("dense")
            ui.label(
                "可填数据包根（含 tiles/）或瓦片目录；相对路径按项目根解析"
            ).classes("text-xs text-gray-500")
            data_root_status = ui.label("尚未加载").classes(
                "text-xs text-gray-600 whitespace-pre-line break-all"
            )

            tile_select = ui.select({}, label="瓦片（仅含 DW 标签）").classes("w-full")
            tile_select.on_value_change(on_tile)

            view_select = ui.select({}, label="传感器 / 视图").classes("w-full")
            view_select.on_value_change(on_view)

            ui.label("阈值（作用于 d，量程自适应）").classes("text-sm text-gray-600")
            threshold_value_label = ui.label("当前阈值 = —").classes("text-xs text-gray-600")
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                threshold_slider = ui.slider(
                    min=0.01, max=1.0, step=0.01, value=state["threshold"]
                ).classes("flex-1")
                threshold_number = ui.number(
                    value=state["threshold"], step=0.01, format="%.4f"
                ).classes("w-28")
            threshold_slider.on_value_change(
                lambda event: apply_threshold(float(event.value))
                if event.value is not None else None
            )
            threshold_number.on_value_change(
                lambda event: apply_threshold(float(event.value))
                if event.value is not None else None
            )
            scale_label = ui.label().classes("text-xs text-gray-500")

            ui.separator()
            ui.label("热力图色阶分层").classes("text-sm font-medium")
            mode_select = ui.select(
                MODES, value=state["mode"], label="高亮显示模式"
            ).classes("w-full")
            mode_select.on_value_change(on_mode)
            band_mode_select = ui.select(
                BAND_MODES, value=state["band_mode"], label="色阶模式"
            ).classes("w-full")
            band_mode_select.on_value_change(on_band_mode)
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                levels_number = ui.number(
                    label=f"档数（{LEVELS_MIN}–{LEVELS_MAX}）",
                    value=state["levels"],
                    min=LEVELS_MIN,
                    max=LEVELS_MAX,
                    step=1,
                    format="%d",
                    precision=0,
                ).classes("w-40")
                edges_switch = ui.switch(
                    "档间边界线", value=state["edges"]
                ).classes("text-sm")
            levels_number.on_value_change(on_levels)
            edges_switch.on_value_change(on_edges)
            ui.label(
                "固定分段覆盖全量程 [0, 1]；动态分段把当前阈值量程的上界 x "
                "作为上限 [0, x]。色阶只影响热力图与色条，不改变距离、阈值与统计。"
            ).classes("text-xs text-gray-500")

            ui.separator()
            meta_label = ui.label().classes("text-sm text-gray-600 whitespace-pre-line")
            core_label = ui.label().classes("text-sm text-gray-600 whitespace-pre-line")
            pmax_label = ui.label().classes("text-xs text-gray-600 whitespace-pre-line")

            ui.separator()
            legend_container = ui.column().classes("w-full gap-1")

            ui.separator()
            stats_label = ui.label().classes("text-sm whitespace-pre-line")
            note_label = ui.label().classes("text-xs text-gray-500 whitespace-pre-line")

        with ui.column().classes(
            "w-[560px] shrink-0 items-center gap-2 h-full min-h-0 overflow-y-auto panel-scroll"
        ):
            label_title = ui.label(
                "DW 标签（官方配色 · 原生朝向 · 青色轮廓 = candidate_mask，品红矩形 = bbox）"
            ).classes("text-sm font-medium text-center")
            with ui.row().classes("w-full items-start no-wrap gap-2 justify-center"):
                with ui.column().classes("gap-1"):
                    before_caption = ui.label().classes(
                        "text-xs text-gray-600 text-center w-[270px] truncate"
                    )
                    before_image = (
                        ui.image()
                        .classes("w-[270px] h-[270px] shrink-0")
                        .props("fit=contain")
                        .style("image-rendering: pixelated")
                    )
                with ui.column().classes("gap-1"):
                    after_caption = ui.label().classes(
                        "text-xs text-gray-600 text-center w-[270px] truncate"
                    )
                    after_image = (
                        ui.image()
                        .classes("w-[270px] h-[270px] shrink-0")
                        .props("fit=contain")
                        .style("image-rendering: pixelated")
                    )
            heat_title = ui.label("嵌入距离热力图").classes("text-sm font-medium text-center")
            heat_image = (
                ui.image()
                .classes("w-[520px] h-[520px] shrink-0")
                .props("fit=contain")
                .style("image-rendering: pixelated")
            )
            colorbar = (
                ui.image()
                .classes("w-[520px] h-[14px] shrink-0")
                .style("image-rendering: pixelated")
            )
            colorbar_label = ui.label().classes("text-xs text-gray-500 text-center")

        with ui.column().classes(
            "flex-1 min-w-[380px] gap-3 h-full min-h-0 overflow-y-auto panel-scroll"
        ):
            preview_title = ui.label("预览图").classes("text-sm font-medium")
            preview_caption = ui.label().classes("text-xs text-gray-600")
            preview_image = ui.image().classes("w-full block").props("fit=contain")
            preview_missing = ui.label().classes("text-sm text-red-600 whitespace-pre-line")
            ui.separator()
            review_caption = ui.label().classes("text-xs text-gray-600")
            review_image = ui.image().classes("w-full block").props("fit=contain")
            review_missing = ui.label().classes("text-sm text-red-600 whitespace-pre-line")

    # Default package: the configured DW root, else the AEF root. The tile
    # container resolved by the entry script is reused when it belongs to the
    # same root, so a container not named ``tiles`` stays addressable.
    try:
        if initial_tiles is not None and Path(initial_tiles).is_dir():
            resolved_root, resolved_tiles = Path(initial_root).resolve(), Path(initial_tiles)
        else:
            resolved_root, resolved_tiles = resolve_data_paths(initial_root)
        load_package(resolved_root, resolved_tiles)
        data_root_input.value = root_text()
        dataset = holder["dataset"]
        set_status(
            f"当前数据根 = {dataset.data_root}"
            f"（{len(holder['tiles'])} 个含 DW 标签的瓦片）"
        )
        commit_selection()
    except (ValueError, OSError, KeyError, AttributeError, TypeError,
            json.JSONDecodeError) as error:
        holder["dataset"] = None
        holder["tiles"] = []
        set_status(f"默认数据根不可用：{error}", error=True)
        data_root_input.value = str(initial_root)
        clear_visuals("当前数据根没有可分析的 DW 标签瓦片")
        try:
            draw_legend(load_legend(Path(initial_root)))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            # A malformed legend must not turn the error page into a crash;
            # the placeholder palette keeps the class codes readable.
            draw_legend(load_legend(Path(initial_root) / "missing"))
