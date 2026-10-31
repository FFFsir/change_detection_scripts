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


def sensor_of(view: str) -> str:
    return view.split("_", 1)[0]


def band_of(view: str) -> str:
    return view.split("_", 1)[1] if "_" in view else view


def distribution_label(view: str) -> str:
    return "分布内" if band_of(view) == "full" else "分布外（波段消融诊断）"


def colourise_signed(r: np.ndarray) -> np.ndarray:
    """Blue → near-white → red for signed residuals in ``[-1, 1]``."""
    t = np.clip(r, -1.0, 1.0)
    pos = (t + 1.0) * 0.5 * (len(_DIVERGING_RAMP) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.minimum(lo + 1, len(_DIVERGING_RAMP) - 1)
    frac = (pos - lo)[..., None]
    return (_DIVERGING_RAMP[lo] * (1.0 - frac) + _DIVERGING_RAMP[hi] * frac).astype(np.uint8)


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
        "residual_dir": "sym",
        "residual_mode": "signed",
        "threshold": 0.30,
        "mode": "overlay",
        "use_mask": True,
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
        ref = max(0, min(ref, emb_a.shape[0] - 1))
        other = max(0, min(other, emb_b.shape[0] - 1))

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
    syncing = {"active": False}

    def rescale(q: np.ndarray, valid: np.ndarray) -> None:
        """Fit the threshold controls to the current visualised non-negative量."""
        scale = threshold_scale(q, valid)
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
            f"最大值（当前可视化量） {scale['max']:.{decimals}f}"
        )

    def apply_image(label, caption, missing, caption_text: str, date: str,
                    view: str) -> None:
        caption.set_text(caption_text)
        path, detail = dataset.image_entry(state["tile"], view, date)
        if path is not None:
            label.set_source(path)
            label.set_visibility(True)
            missing.set_visibility(False)
        else:
            label.set_visibility(False)
            missing.set_text(f"缺少图像：{date}\n{detail}")
            missing.set_visibility(True)

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
                for col in (image_col_0, image_col_1):
                    col.set_visibility(False)
                return
        elif not state["view_a"] or not state["view_b"]:
            meta_label.set_text(
                f"瓦片 = {tile}\n"
                f"当前组合类型没有可用的通道组合（需要同一传感器下的两个视图）"
            )
            stats_label.set_text("暂无可比较的数据")
            heat_image.set_source("")
            for col in (image_col_0, image_col_1):
                col.set_visibility(False)
            return

        result = compute()
        q = result["q"]
        r = result["r"]
        valid = result["valid"]

        if state.get("rescale"):
            state["rescale"] = False
            rescale(q, valid)

        signed = kind == "cross_day" and state["residual_mode"] == "signed" and r is not None
        if signed:
            rgb = render_rgb_signed(r, valid, state["threshold"], state["mode"])
        else:
            rgb = render_rgb(q, valid, state["threshold"], state["mode"])
        heat_image.set_source(png_data_url(rgb))

        bar_key = "signed" if signed else "sequential"
        if colorbar_cache["key"] != bar_key:
            if signed:
                bar = np.linspace(-1.0, 1.0, 512)[None, :]
                colorbar.set_source(png_data_url(colourise_signed(bar)))
                colorbar_label.set_text(
                    "色标：−1 → 0 → +1（负 = 残差减小，正 = 残差增大；阈值与高亮作用于 |残差|）"
                )
            else:
                bar = np.linspace(0.0, 1.0, 512)[None, :]
                colorbar.set_source(png_data_url(colourise(bar)))
                colorbar_label.set_text("色标：0 → 1（当前可视化量越大表示差异越大）")
            colorbar_cache["key"] = bar_key

        if kind == "same_channel":
            threshold_caption.set_text("阈值（作用于 d = (1 − cos) / 2，量程自适应，左端不可取 0）")
            heat_title.set_text("变化程度热力图（128 × 128）")
            semantics_label.set_text("同模态对照：同一视图不同日期，高值表示变化程度。")
        elif kind == "same_day":
            threshold_caption.set_text("阈值（作用于通道差异基线 d，量程自适应，左端不可取 0）")
            heat_title.set_text("通道差异基线热力图（128 × 128，高值不代表变化）")
            semantics_label.set_text("跨模态基线：同一日期的通道差异，高值不代表变化。")
        else:
            threshold_caption.set_text("阈值（作用于 |残差|，量程自适应，左端不可取 0）")
            heat_title.set_text("跨模态残差热力图（128 × 128）")
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
                apply_image(
                    label, caption, missing,
                    f"{tag} 通道 {view} · 帧 {frame_idx} · {date}",
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

    def on_combo(event) -> None:
        if not event.value:
            return
        state["combo"] = event.value
        state["rescale"] = True
        rebuild_controls()
        refresh()

    def on_tile(event) -> None:
        if not event.value:
            return
        state["tile"] = event.value
        state["rescale"] = True
        rebuild_controls()
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
            else "绝对值：0 → 1 顺序色阶"
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
            semantics_label = ui.label().classes("text-xs text-gray-600 whitespace-pre-line")

            combo_select = ui.select(COMBOS, value=state["combo"], label="组合类型").classes("w-full")
            combo_select.on_value_change(on_combo)

            tile_select = ui.select(tile_options, value=state["tile"], label="瓦片").classes("w-full")
            tile_select.on_value_change(on_tile)

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
            colorbar_label = ui.label().classes("text-xs text-gray-500")
            colorbar = ui.image().classes("w-full h-[14px]").style("image-rendering: pixelated")
            ui.label("灰色区域 = common_mask 排除的像素（仅在开关开启时出现）").classes("text-xs text-gray-500")

        with ui.column().classes("w-[540px] shrink-0 items-center gap-2"):
            heat_title = ui.label("变化程度热力图（128 × 128）").classes("text-sm font-medium")
            heat_image = ui.image().classes("w-[520px] h-[520px]").style("image-rendering: pixelated")

        with ui.column().classes("flex-1 min-w-0 gap-3"):
            ui.label("gallery 原始图像").classes("text-sm font-medium")
            image_col_0 = ui.column().classes("gap-1 w-full")
            with image_col_0:
                caption_0 = ui.label().classes("text-sm font-medium")
                image_0 = ui.image().classes("w-full")
                missing_0 = ui.label().classes("text-sm text-red-600")
            image_col_1 = ui.column().classes("gap-1 w-full")
            with image_col_1:
                caption_1 = ui.label().classes("text-sm font-medium")
                image_1 = ui.image().classes("w-full")
                missing_1 = ui.label().classes("text-sm text-red-600")

    rebuild_controls()
    update_residual_formula()
    refresh()
