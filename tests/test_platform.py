"""Tests for the platform modules that need no browser.

Run with the project virtual environment:

    .venv/Scripts/python.exe -m unittest discover -s tests -v

Cases that need a data package are skipped when the package is absent, so the
suite is safe to run on a machine without the (git-ignored) data.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))

from webui.dataset import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_DW_DATA_ROOT,
    Dataset,
    resolve_data_paths,
    tile_coordinates,
)
from webui.dw_data import (  # noqa: E402
    DwLegend,
    label_rgb,
    load_dw_tile,
    load_legend,
    mask_boundary,
    region_stats,
)
from webui.render import (  # noqa: E402
    LEVELS_MAX,
    LEVELS_MIN,
    band_index,
    band_range,
    band_text,
    clamp_levels,
    distribution_label,
    pair_distance,
    render_band_strip,
    render_bands,
    summarise,
    summarise_signed,
    threshold_scale,
)

DW_PACKAGE = DEFAULT_DW_DATA_ROOT
LEGACY_PACKAGE = DEFAULT_DATA_ROOT


def _unit_vectors(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    vectors = rng.normal(size=shape).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / norms).astype(np.float32)


class DistanceTest(unittest.TestCase):
    """``d = (1 - cos) / 2`` must stay bit-for-bit what it was."""

    def test_matches_direct_formula(self) -> None:
        rng = np.random.default_rng(7)
        x = _unit_vectors(rng, (8, 8, 64))
        y = _unit_vectors(rng, (8, 8, 64))
        mask = np.ones((8, 8), dtype=bool)
        d, valid = pair_distance(x, y, mask, use_mask=False)
        cos = np.sum(x * y, axis=-1)
        expected = np.clip((1.0 - cos) / 2.0, 0.0, 1.0)
        np.testing.assert_allclose(d, expected, rtol=0, atol=1e-6)
        self.assertTrue(valid.all())

    def test_zero_norm_is_invalid(self) -> None:
        x = np.zeros((2, 2, 64), dtype=np.float32)
        y = _unit_vectors(np.random.default_rng(1), (2, 2, 64))
        mask = np.ones((2, 2), dtype=bool)
        d, valid = pair_distance(x, y, mask, use_mask=False)
        self.assertFalse(valid.any())
        np.testing.assert_array_equal(d, np.zeros((2, 2), dtype=np.float32))

    def test_masked_pixels_are_excluded(self) -> None:
        rng = np.random.default_rng(3)
        x = _unit_vectors(rng, (4, 4, 64))
        y = _unit_vectors(rng, (4, 4, 64))
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        _, valid_on = pair_distance(x, y, mask, use_mask=True)
        _, valid_off = pair_distance(x, y, mask, use_mask=False)
        np.testing.assert_array_equal(valid_on, mask)
        self.assertEqual(int(valid_off.sum()), 16)


class ResidualTest(unittest.TestCase):
    """fwd / rev / sym keep their documented definitions."""

    def test_residuals_match_independent_recomputation(self) -> None:
        """Every residual is checked against numpy computed from the formula."""
        from webui.render import residual_maps

        rng = np.random.default_rng(11)
        # Views are (frames, H, W, 64); the residual compares frame 0 with 1.
        view_a = _unit_vectors(rng, (2, 4, 4, 64))
        view_b = _unit_vectors(rng, (2, 4, 4, 64))
        a0, a1 = view_a[0], view_a[1]
        b0, b1 = view_b[0], view_b[1]
        mask = np.ones((4, 4), dtype=bool)
        maps = residual_maps(view_a, view_b, 0, 1, mask)

        def cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
            return np.clip(
                np.sum(u / np.linalg.norm(u, axis=-1, keepdims=True)
                       * (v / np.linalg.norm(v, axis=-1, keepdims=True)), axis=-1),
                -1.0, 1.0,
            )

        def dist(u: np.ndarray, v: np.ndarray) -> np.ndarray:
            return np.clip((1.0 - cos(u, v)) / 2.0, 0.0, 1.0)

        base = dist(a0, b0)
        fwd_expected = np.clip(dist(a0, b1) - base, -1.0, 1.0)
        rev_expected = np.clip(dist(b0, a1) - base, -1.0, 1.0)
        np.testing.assert_allclose(maps["base"], base, atol=1e-6)
        np.testing.assert_allclose(maps["fwd"], fwd_expected, atol=1e-6)
        np.testing.assert_allclose(maps["rev"], rev_expected, atol=1e-6)
        np.testing.assert_allclose(
            maps["sym"], (fwd_expected + rev_expected) / 2.0, atol=1e-6
        )
        self.assertTrue(maps["valid"].all())

    def test_residuals_are_clipped_and_invalid_pixels_zeroed(self) -> None:
        from webui.render import residual_maps

        rng = np.random.default_rng(13)
        view_a = _unit_vectors(rng, (2, 2, 2, 64))
        view_b = _unit_vectors(rng, (2, 2, 2, 64))
        # A zero-norm pixel must poison every residual that touches it.
        view_a[0, 0, 0] = 0.0
        mask = np.ones((2, 2), dtype=bool)
        maps = residual_maps(view_a, view_b, 0, 1, mask)
        self.assertFalse(maps["valid"][0, 0])
        for key in ("base", "fwd", "rev", "sym"):
            self.assertTrue(np.all(maps[key] >= -1.0))
            self.assertTrue(np.all(maps[key] <= 1.0))
        # ``base`` is zeroed where the pair is invalid; ``fwd``/``rev``/``sym``
        # keep their arithmetic value and are masked by ``valid`` downstream,
        # exactly as the page has always done.
        self.assertEqual(float(maps["base"][0, 0]), 0.0)


class ThresholdScaleTest(unittest.TestCase):
    def test_top_of_range_has_no_hits(self) -> None:
        rng = np.random.default_rng(5)
        values = rng.random((32, 32)).astype(np.float32) * 0.42
        valid = np.ones((32, 32), dtype=bool)
        scale = threshold_scale(values, valid)
        self.assertEqual(int(np.count_nonzero(values > scale["upper"])), 0)
        self.assertGreaterEqual(scale["lower"], scale["step"])
        self.assertLessEqual(scale["upper"], 1.0)

    def test_zero_maximum_still_has_a_range(self) -> None:
        values = np.zeros((8, 8), dtype=np.float32)
        valid = np.ones((8, 8), dtype=bool)
        scale = threshold_scale(values, valid)
        self.assertEqual(scale["upper"], scale["step"])
        self.assertEqual(scale["lower"], scale["step"])

    def test_default_threshold_rule(self) -> None:
        values = np.full((4, 4), 0.9, dtype=np.float32)
        valid = np.ones((4, 4), dtype=bool)
        scale = threshold_scale(values, valid)
        # The default is snapped to the step grid, so it lands within one step.
        self.assertLessEqual(abs(scale["default"] - 0.30), scale["step"] + 1e-9)


class DisplayControlTest(unittest.TestCase):
    """The DW page exposes the same display controls as the AEF page."""

    def test_fixed_mode_spans_the_full_range(self) -> None:
        self.assertEqual(band_range("fixed", 0.42, signed=False), (0.0, 1.0))
        self.assertEqual(band_range("fixed", 0.42, signed=True), (-1.0, 1.0))

    def test_dynamic_mode_follows_the_threshold_upper_bound(self) -> None:
        self.assertEqual(band_range("dynamic", 0.42, signed=False), (0.0, 0.42))
        self.assertEqual(band_range("dynamic", 0.42, signed=True), (-0.42, 0.42))

    def test_levels_are_clamped_to_the_supported_range(self) -> None:
        self.assertEqual(clamp_levels(1, 12), LEVELS_MIN)
        self.assertEqual(clamp_levels(100, 12), LEVELS_MAX)
        self.assertEqual(clamp_levels(7, 12), 7)
        self.assertEqual(clamp_levels("24", 12), 24)
        self.assertEqual(clamp_levels("7.9", 12), 7)

    def test_unparseable_levels_keep_the_previous_value(self) -> None:
        self.assertEqual(clamp_levels(None, 12), 12)
        self.assertEqual(clamp_levels("abc", 9), 9)
        self.assertEqual(clamp_levels("", 5), 5)

    def test_banding_changes_pixels_but_not_the_statistics(self) -> None:
        """Only displayed colours may change; reported statistics may not."""
        rng = np.random.default_rng(21)
        values = rng.random((16, 16)).astype(np.float32) * 0.4
        valid = np.ones((16, 16), dtype=bool)
        threshold = 0.2
        # A copy proves rendering never mutates the input the stats consume.
        snapshot = values.copy()

        dyn_low, dyn_high = band_range("dynamic", 0.4, signed=False)
        dynamic = render_bands(
            values, valid, threshold, "overlay",
            low=dyn_low, high=dyn_high, levels=12, edges=True, signed=False,
        )
        fixed = render_bands(
            values, valid, threshold, "overlay",
            low=0.0, high=1.0, levels=12, edges=True, signed=False,
        )
        self.assertFalse(np.array_equal(dynamic, fixed))

        np.testing.assert_array_equal(values, snapshot)
        self.assertEqual(
            summarise(values, valid, threshold),
            summarise(snapshot, valid, threshold),
        )
        # Re-running with the other banding settings must not move the stats.
        grey = render_bands(
            values, valid, threshold, "grey",
            low=0.0, high=1.0, levels=2, edges=False, signed=False,
        )
        self.assertEqual(grey.shape, values.shape + (3,))
        self.assertEqual(
            int(np.count_nonzero(values > threshold)),
            summarise(values, valid, threshold)["hot"],
        )

    def test_more_levels_produces_more_colours(self) -> None:
        low, high = band_range("dynamic", 0.4, signed=False)
        four = render_band_strip(low=low, high=high, levels=4, edges=False, signed=False)
        sixteen = render_band_strip(
            low=low, high=high, levels=16, edges=False, signed=False
        )
        four_colours = len(np.unique(four.reshape(-1, 3), axis=0))
        sixteen_colours = len(np.unique(sixteen.reshape(-1, 3), axis=0))
        self.assertEqual(four_colours, 4)
        self.assertEqual(sixteen_colours, 16)

    def test_band_edges_reported_status_matches_rendering(self) -> None:
        """The caption's edge state must match what the bar actually draws."""
        from webui.render import band_edges_drawn

        self.assertTrue(band_edges_drawn(True, "overlay"))
        self.assertFalse(band_edges_drawn(False, "overlay"))
        self.assertFalse(band_edges_drawn(True, "binary"))
        self.assertFalse(band_edges_drawn(False, "grey"))

        values = np.tile(np.linspace(0.0, 0.4, 32, dtype=np.float32), (32, 1))
        valid = np.ones((32, 32), dtype=bool)
        bar_with = render_band_strip(
            low=0.0, high=0.4, levels=8,
            edges=band_edges_drawn(True, "binary"), signed=False,
        )
        bar_plain = render_band_strip(
            low=0.0, high=0.4, levels=8, edges=False, signed=False,
        )
        self.assertTrue(np.array_equal(bar_with, bar_plain))
        map_binary = render_bands(
            values, valid, 0.2, "binary", low=0.0, high=0.4, levels=8,
            edges=band_edges_drawn(True, "binary"), signed=False,
        )
        # Binary rendering uses the dark background plus the single highlight
        # colour, so contour lines would show up as extra colours here.
        self.assertLessEqual(len(np.unique(map_binary.reshape(-1, 3), axis=0)), 3)

    def test_levels_overflow_falls_back(self) -> None:
        self.assertEqual(clamp_levels("1e400", 12), 12)
        self.assertEqual(clamp_levels(float("inf"), 9), 9)

    def test_binary_mode_draws_no_band_edges(self) -> None:
        values = np.tile(np.linspace(0.0, 0.4, 32, dtype=np.float32), (32, 1))
        valid = np.ones((32, 32), dtype=bool)
        with_edges = render_bands(
            values, valid, 0.2, "binary",
            low=0.0, high=0.4, levels=8, edges=True, signed=False,
        )
        without_edges = render_bands(
            values, valid, 0.2, "binary",
            low=0.0, high=0.4, levels=8, edges=False, signed=False,
        )
        self.assertTrue(np.array_equal(with_edges, without_edges))

    def test_edges_toggle_changes_overlay_rendering(self) -> None:
        values = np.tile(np.linspace(0.0, 0.4, 32, dtype=np.float32), (32, 1))
        valid = np.ones((32, 32), dtype=bool)
        on = render_bands(
            values, valid, 0.2, "overlay",
            low=0.0, high=0.4, levels=8, edges=True, signed=False,
        )
        off = render_bands(
            values, valid, 0.2, "overlay",
            low=0.0, high=0.4, levels=8, edges=False, signed=False,
        )
        self.assertFalse(np.array_equal(on, off))


class SummaryTest(unittest.TestCase):
    def test_summary_counts_use_strict_greater(self) -> None:
        values = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=bool)
        stats = summarise(values, valid, 0.2)
        self.assertEqual(stats["hot"], 2)
        self.assertEqual(stats["valid"], 4)

    def test_signed_summary_uses_absolute_residual(self) -> None:
        residual = np.array([[-0.5, 0.5], [0.1, -0.2]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=bool)
        stats = summarise_signed(residual, valid, 0.3)
        self.assertEqual(stats["hot"], 2)


class BandTest(unittest.TestCase):
    def test_band_index_clamps(self) -> None:
        values = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0]], dtype=np.float32)
        index = band_index(values, 0.0, 1.0, 5)
        np.testing.assert_array_equal(index, np.array([[0, 0, 2, 4, 4]]))


class ViewLabelTest(unittest.TestCase):
    def test_legacy_views_keep_distribution_labels(self) -> None:
        self.assertEqual(distribution_label("sentinel2_full"), "分布内")
        self.assertEqual(
            distribution_label("sentinel2_nir"), "分布外（波段消融诊断）"
        )
        self.assertEqual(band_text("sentinel2_nir"), "nir")

    def test_bare_sensor_views_are_single_views(self) -> None:
        self.assertEqual(distribution_label("sentinel2"), "单视图（无波段组）")
        self.assertEqual(distribution_label("sentinel1"), "单视图（无波段组）")
        self.assertEqual(band_text("sentinel2"), "无波段组")


class CoordinateTest(unittest.TestCase):
    def test_parent_coordinates(self) -> None:
        manifest = {
            "parent": "registered/tiles/C01_E121.4790_N25.1724_20240101_20241231"
        }
        lon, lat = tile_coordinates(manifest)
        self.assertAlmostEqual(lon, 121.4790, places=4)
        self.assertAlmostEqual(lat, 25.1724, places=4)

    def test_region_id_fallback(self) -> None:
        lon, lat = tile_coordinates({"region_id": "E121.4172_N25.0446"})
        self.assertAlmostEqual(lon, 121.4172, places=4)
        self.assertAlmostEqual(lat, 25.0446, places=4)

    def test_controls_tile_has_no_coordinates(self) -> None:
        self.assertIsNone(tile_coordinates({"parent": "controls/tiles/V01"}))
        self.assertIsNone(tile_coordinates({}))


@unittest.skipUnless(DW_PACKAGE.is_dir(), f"DW package not found: {DW_PACKAGE}")
class DwPackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.data_root, cls.tiles_dir = resolve_data_paths(DW_PACKAGE)
        except (ValueError, OSError) as error:
            raise unittest.SkipTest(f"DW package is not usable: {error}") from error
        cls.dataset = Dataset(cls.data_root, cls.tiles_dir)
        cls.tiles = [
            name
            for name in cls.dataset.tiles()
            if (cls.dataset.root / name / "land_cover" / "dynamic_world.npz").exists()
        ]

    def test_tiles_found(self) -> None:
        self.assertGreaterEqual(len(self.tiles), 1)
        for name in self.tiles:
            self.assertTrue(name.startswith("D"))

    def test_labels_shapes_and_codes(self) -> None:
        for name in self.tiles:
            tile = load_dw_tile(self.dataset.root / name)
            self.assertEqual(tile.labels.shape, (2, 128, 128))
            self.assertEqual(tile.labels.dtype, np.uint8)
            codes = set(np.unique(tile.labels).tolist())
            self.assertTrue(codes <= set(range(9)) | {255}, codes)
            self.assertEqual(tile.valid_mask.shape, (2, 128, 128))

    def test_core_pixels_match_mask_and_manifest(self) -> None:
        for name in self.tiles:
            tile = load_dw_tile(self.dataset.root / name)
            self.assertIsNotNone(tile.candidate_mask)
            self.assertEqual(tile.core_pixels, tile.manifest_core_pixels, name)

    def test_bbox_is_end_exclusive_and_inside(self) -> None:
        for name in self.tiles:
            tile = load_dw_tile(self.dataset.root / name)
            y0, x0, y1, x1 = tile.bbox
            self.assertTrue(0 <= y0 < y1 <= 128, name)
            self.assertTrue(0 <= x0 < x1 <= 128, name)
            rows, cols = np.nonzero(tile.candidate_mask)
            self.assertGreaterEqual(int(rows.min()), y0)
            self.assertLess(int(rows.max()), y1)
            self.assertGreaterEqual(int(cols.min()), x0)
            self.assertLess(int(cols.max()), x1)

    def test_before_after_labels_match_manifest(self) -> None:
        for name in self.tiles:
            tile = load_dw_tile(self.dataset.root / name)
            self.assertEqual(tile.before_label, tile.manifest.get("before_label"))
            self.assertEqual(tile.after_label, tile.manifest.get("after_label"))

    def test_legend_has_nine_official_colours(self) -> None:
        legend = load_legend(self.data_root)
        self.assertEqual(len(legend.classes), 9)
        self.assertEqual(legend.nodata, 255)
        codes = [entry["code"] for entry in legend.classes]
        self.assertEqual(codes, list(range(9)))
        colours = {entry["color"].lower() for entry in legend.classes}
        self.assertEqual(len(colours), 9)
        self.assertEqual(legend.class_name(7), "bare")

    def test_label_rgb_uses_legend_colours_and_invalid_grey(self) -> None:
        legend = load_legend(self.data_root)
        tile = load_dw_tile(self.dataset.root / self.tiles[0], self.tiles[0])
        rgb = label_rgb(tile.labels[0], tile.valid_mask[0], legend)
        for entry in legend.classes:
            selected = (tile.labels[0] == entry["code"]) & (tile.labels[0] != 255)
            if selected.any():
                self.assertTrue((rgb[selected] == np.array(entry["rgb"])).all())

    def test_invalid_label_pixels_use_neutral_colour(self) -> None:
        from webui.dw_data import INVALID_COLOR

        legend = load_legend(self.data_root)
        tile = load_dw_tile(self.dataset.root / self.tiles[0], self.tiles[0])
        # Render a synthetic nodata pixel plus one masked-out pixel.
        labels = tile.labels[0].copy()
        valid = tile.valid_mask[0].copy()
        labels[0, 0] = 255
        valid[0, 1] = False
        rgb = label_rgb(labels, valid, legend)
        self.assertTrue((rgb[0, 0] == np.array(INVALID_COLOR)).all())
        self.assertTrue((rgb[0, 1] == np.array(INVALID_COLOR)).all())

    def test_mask_falls_back_to_dw_valid_mask(self) -> None:
        name = self.tiles[0]
        mask, source = self.dataset.mask_source(name, "sentinel2")
        self.assertEqual(mask.shape, (128, 128))
        self.assertIn("DW valid_mask", source)
        self.assertFalse((self.dataset.root / name / "distance_maps").exists())

    def test_mask_source_mentions_dw_not_distance_maps(self) -> None:
        for name in self.tiles:
            _, source = self.dataset.mask_source(name, "sentinel1")
            self.assertNotIn("distance_maps", source)

    def test_whole_tile_stats_match_numpy(self) -> None:
        name = self.tiles[0]
        tile = load_dw_tile(self.dataset.root / name, name)
        emb = self.dataset.embeddings(name, "sentinel2")
        mask, _ = self.dataset.mask_source(name, "sentinel2")
        d, usable = pair_distance(emb[0], emb[-1], mask, use_mask=True)
        valid = usable & tile.period_valid(0) & tile.period_valid(1)
        threshold = 0.2
        stats = region_stats(d, valid, threshold)
        values = d[valid]
        self.assertEqual(stats["count"], int(values.size))
        self.assertAlmostEqual(stats["mean"], float(values.mean()), places=6)
        self.assertAlmostEqual(stats["median"], float(np.median(values)), places=6)
        self.assertAlmostEqual(stats["p90"], float(np.percentile(values, 90)), places=6)
        self.assertAlmostEqual(stats["p95"], float(np.percentile(values, 95)), places=6)
        self.assertAlmostEqual(stats["max"], float(values.max()), places=6)
        self.assertEqual(stats["hot"], int(np.count_nonzero(values > threshold)))

    def test_stats_cover_the_whole_tile_not_the_candidate_mask(self) -> None:
        """The reported numbers must describe every valid pixel.

        A grouped implementation would return the core-only mean here, so the
        two candidate values are compared against the reported one.
        """
        name = self.tiles[0]
        tile = load_dw_tile(self.dataset.root / name, name)
        emb = self.dataset.embeddings(name, "sentinel2")
        mask, _ = self.dataset.mask_source(name, "sentinel2")
        d, usable = pair_distance(emb[0], emb[-1], mask, use_mask=True)
        valid = usable & tile.period_valid(0) & tile.period_valid(tile.last_period)
        stats = region_stats(d, valid, 0.2)

        core = valid & tile.candidate_mask
        whole_mean = float(d[valid].mean())
        core_mean = float(d[core].mean())
        self.assertEqual(stats["count"], int(valid.sum()))
        self.assertAlmostEqual(stats["mean"], whole_mean, places=6)
        # The two scopes really differ, otherwise this test would be vacuous.
        self.assertGreater(abs(whole_mean - core_mean), 1e-4)
        self.assertNotAlmostEqual(stats["mean"], core_mean, places=4)

    def test_view_dates_are_not_manifest_dates(self) -> None:
        """S1 embeddings are dated independently of the S2/DW acquisition."""
        from webui.dw_page import _view_dates

        name = self.tiles[0]
        tile = load_dw_tile(self.dataset.root / name, name)
        dw_before, dw_after = tile.dates
        s1_before, s1_after = _view_dates(self.dataset, name, "sentinel1")
        self.assertNotEqual((dw_before, dw_after), (s1_before, s1_after))
        self.assertNotEqual(s1_before, "?")

    def test_preview_images_exist(self) -> None:
        for name in self.tiles:
            found = dict(self.dataset.preview_images(name))
            self.assertTrue(found["preview.png"], name)
            self.assertTrue(found["review.png"], name)

    def test_no_gallery_index(self) -> None:
        self.assertFalse(self.dataset.gallery_available())

    def test_view_names_have_no_band_group(self) -> None:
        for name in self.tiles:
            for view in self.dataset.all_views(name):
                self.assertEqual(band_text(view), "无波段组")


@unittest.skipUnless(LEGACY_PACKAGE.is_dir(), f"legacy package not found: {LEGACY_PACKAGE}")
class LegacyPackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.data_root, cls.tiles_dir = resolve_data_paths(LEGACY_PACKAGE)
        except (ValueError, OSError) as error:
            raise unittest.SkipTest(f"legacy package is not usable: {error}") from error
        cls.dataset = Dataset(cls.data_root, cls.tiles_dir)

    def test_gallery_index_is_used(self) -> None:
        self.assertTrue(self.dataset.gallery_available())
        self.assertGreater(len(self.dataset.gallery), 0)

    def test_mask_comes_from_distance_maps(self) -> None:
        mask, source = self.dataset.mask_source("C01", "sentinel2_full")
        self.assertIn("distance_maps", source)
        self.assertEqual(int(mask.sum()), 15971)

    def test_packaged_maps_are_twice_the_computed_distance(self) -> None:
        """The archived 2x relation to the packaged maps must survive.

        The package stores ``1 - cos`` (``same_modality_*``), while the UI
        computes ``d = (1 - cos) / 2``, so the packaged map equals ``2 * d``.
        """
        archive = self.dataset.root / "C01" / "distance_maps" / "sentinel2_nir.npz"
        if not archive.exists():
            self.skipTest("packaged distance map not present")
        with np.load(archive) as data:
            common_mask = data["common_mask"]
            pairs = [
                ("sentinel2_visible", "same_modality_visible"),
                ("sentinel2_nir", "same_modality_ir"),
            ]
            checked = 0
            for view, key in pairs:
                if key not in data:
                    continue
                emb = self.dataset.embeddings("C01", view)
                d, valid = pair_distance(
                    emb[0], emb[-1], common_mask, use_mask=True
                )
                np.testing.assert_allclose(
                    data[key][valid], 2.0 * d[valid], rtol=1e-3, atol=1e-5
                )
                checked += 1
        self.assertGreater(checked, 0, "no packaged same-modality map to compare")

    def test_single_period_landsat_tiles_offer_no_cross_day_landsat_view(self) -> None:
        single = {"C05", "C06", "C10", "C12", "C17", "C18", "C19", "C21"}
        present = [name for name in sorted(single) if name in self.dataset.tiles()]
        if not present:
            self.skipTest("none of the single-period Landsat tiles are present")
        for name in present:
            views = self.dataset.views(name)
            self.assertFalse(
                any(v.startswith("landsat") for v in views),
                f"{name} should have no two-frame Landsat view",
            )

    def test_missing_mask_for_sensor_is_an_error(self) -> None:
        """V01 has Sentinel-2 maps only, so asking for Landsat must raise."""
        with self.assertRaises(FileNotFoundError):
            self.dataset.mask_source("V01", "landsat_full")


class MaskFallbackTest(unittest.TestCase):
    """The documented fallback chain is exercised on synthetic packages."""

    def _package(self, root: Path, *, dw_valid: bool, obs_valid: bool) -> Dataset:
        tile_dir = root / "tiles" / "T01"
        (tile_dir / "embeddings").mkdir(parents=True)
        np.save(
            tile_dir / "embeddings" / "sentinel2.npy",
            np.zeros((2, 128, 128, 64), dtype=np.float32),
        )
        (tile_dir / "tile_manifest.json").write_text("{}", encoding="utf-8")
        if dw_valid:
            (tile_dir / "land_cover").mkdir(exist_ok=True)
            mask = np.ones((2, 128, 128), dtype=bool)
            mask[0, 0, 0] = False
            np.savez(
                tile_dir / "land_cover" / "dynamic_world.npz",
                label=np.zeros((2, 128, 128), dtype=np.uint8),
                valid_mask=mask,
            )
        if obs_valid:
            (tile_dir / "observations").mkdir(exist_ok=True)
            np.savez(
                tile_dir / "observations" / "sentinel2.npz",
                valid_mask=np.zeros((2, 128, 128), dtype=bool),
            )
        return Dataset(root, root / "tiles")

    def test_dw_valid_mask_is_used_before_observations(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._package(Path(tmp), dw_valid=True, obs_valid=True)
            mask, source = dataset.mask_source("T01", "sentinel2")
            self.assertIn("DW valid_mask", source)
            # Period axis is reduced with "and", so one bad period pixel
            # removes exactly one pixel from the common mask.
            self.assertEqual(mask.shape, (128, 128))
            self.assertEqual(int(mask.sum()), 128 * 128 - 1)

    def test_observation_valid_mask_is_used_when_no_dw_package(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._package(Path(tmp), dw_valid=False, obs_valid=True)
            mask, source = dataset.mask_source("T01", "sentinel2")
            self.assertIn("observations/sentinel2.npz", source)
            self.assertEqual(int(mask.sum()), 0)

    def test_all_true_only_when_nothing_is_available(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._package(Path(tmp), dw_valid=False, obs_valid=False)
            mask, source = dataset.mask_source("T01", "sentinel2")
            self.assertIn("全 True", source)
            self.assertTrue(mask.all())
            self.assertEqual(mask.shape, (128, 128))

    def test_existing_distance_maps_dir_without_sensor_still_raises(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._package(Path(tmp), dw_valid=True, obs_valid=False)
            (Path(tmp) / "tiles" / "T01" / "distance_maps").mkdir()
            with self.assertRaises(FileNotFoundError):
                dataset.mask_source("T01", "sentinel2")

    def test_mask_shape_mismatch_is_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tile_dir = root / "tiles" / "T01"
            (tile_dir / "land_cover").mkdir(parents=True)
            (tile_dir / "embeddings").mkdir()
            np.save(
                tile_dir / "embeddings" / "sentinel2.npy",
                np.zeros((2, 128, 128, 64), dtype=np.float32),
            )
            np.savez(
                tile_dir / "land_cover" / "dynamic_world.npz",
                label=np.zeros((2, 128, 128), dtype=np.uint8),
                valid_mask=np.ones((128, 128), dtype=bool),
            )
            with self.assertRaises(ValueError):
                load_dw_tile(tile_dir)


class ManifestDialectTest(unittest.TestCase):
    def test_tile_label_falls_back_through_manifest_fields(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            tile_dir = root / "tiles" / "D99_region"
            (tile_dir / "embeddings").mkdir(parents=True)
            (tile_dir / "tile_manifest.json").write_text(
                json.dumps(
                    {"tile_id": "D99", "region_id": "E121.0_N25.0", "split": "test"}
                ),
                encoding="utf-8",
            )
            np.save(
                tile_dir / "embeddings" / "sentinel2.npy",
                np.zeros((2, 128, 128, 64), dtype=np.float32),
            )
            dataset = Dataset(root, root / "tiles")
            label = dataset.tile_label("D99_region")
            self.assertIn("E121.0_N25.0", label)
            self.assertIn("test", label)
            self.assertEqual(dataset.tile_category("D99_region"), "E121.0_N25.0")


class LegendFallbackTest(unittest.TestCase):
    def test_placeholder_palette_when_legend_is_missing(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            legend = load_legend(Path(tmp), np.array(["a", "b"]))
            self.assertIsInstance(legend, DwLegend)
            self.assertIn("占位", legend.source)
            self.assertEqual(len(legend.classes), 9)


class MaskBoundaryTest(unittest.TestCase):
    def test_boundary_is_inner_outline(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        boundary = mask_boundary(mask)
        self.assertEqual(int(boundary.sum()), 8)
        self.assertTrue(boundary[1, 1])
        self.assertFalse(boundary[2, 2])

    def test_empty_mask_has_no_boundary(self) -> None:
        self.assertFalse(mask_boundary(np.zeros((4, 4), dtype=bool)).any())


if __name__ == "__main__":
    unittest.main()
