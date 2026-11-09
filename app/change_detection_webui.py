#!/usr/bin/env python
"""AEF change-detection platform.

Two pages share one entry script:

* ``/`` and ``/aef`` — the AEF unsupervised change-detection viewer, which
  implements the unsupervised branch of AlphaEarth Foundations (supplement
  S4.1, equation 8): two per-pixel 64-d embeddings are L2 normalized and
  compared with ``d = (1 - cos) / 2``, ``d in [0, 1]``. Three combination types
  are offered: same-channel cross-day, same-day cross-channel (a cross-modal
  baseline whose high values are channel differences, not change) and
  cross-day cross-channel (the residual between the cross-date distance and
  the same-day baseline).
* ``/dw`` — the Dynamic World label comparison page. It shows the two-period
  DW labels in their official colours with the change transition overlaid, and
  the embedding distance over the whole tile. DW labels are a model
  classification and ``pmax`` is a model probability, so the page never reports
  a model-quality score.

Run with the project virtual environment:

    .venv/Scripts/python.exe app/change_detection_webui.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``importlib``-style loading (the archived verification scripts load this
# file directly by path) as well as ``python app/change_detection_webui.py``.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from nicegui import ui

from webui.aef_page import build_aef_page
from webui.dataset import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DW_DATA_ROOT,
    Dataset,
    resolve_data_paths,
)
from webui.dw_page import build_dw_page
# Re-exported so the archived regression scripts, which load this module and
# call these names directly, keep working unchanged.
from webui.render import (
    BINARY_BACKGROUND,
    HIGHLIGHT,
    MASK_NEUTRAL,
    colourise,
    pair_distance,
    render_band_strip,
    render_bands,
    render_rgb,
    render_rgb_signed,
    sample_ramp,
    summarise,
    threshold_scale,
)

# Set once in ``main`` from ``--data-root`` / ``--dw-data-root``: the resolved
# package root and the tile container inside it. Both are kept because a
# container may not be named ``tiles``, and the pages need to know which
# directory actually holds tiles.
_initial_paths = {
    "data_root": DEFAULT_DATA_ROOT,
    "tiles_dir": DEFAULT_DATA_ROOT / "tiles",
    "dw_data_root": DEFAULT_DW_DATA_ROOT,
    "dw_tiles_dir": None,
}


@ui.page("/")
def index() -> None:
    build_aef_page(_initial_paths["data_root"], _initial_paths["tiles_dir"])


@ui.page("/aef")
def aef() -> None:
    build_aef_page(_initial_paths["data_root"], _initial_paths["tiles_dir"])


@ui.page("/dw")
def dw() -> None:
    build_dw_page(_initial_paths["dw_data_root"], _initial_paths["dw_tiles_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="initial AEF data package root (default: data/spectral)",
    )
    parser.add_argument(
        "--dw-data-root",
        default=str(DEFAULT_DW_DATA_ROOT),
        help=(
            "initial Dynamic World data package root "
            "(default: the AEF_DW_change_data_20261031 package next to this repo)"
        ),
    )
    parser.add_argument("--show", action="store_true", help="open a browser window")
    args = parser.parse_args()

    data_root, tiles_dir = resolve_data_paths(args.data_root)
    _initial_paths["data_root"] = data_root
    _initial_paths["tiles_dir"] = tiles_dir

    # The DW root is optional. When it cannot be resolved the page falls back
    # to the AEF root, which may itself be a DW package; if neither works the
    # page reports it in its status line instead of blocking startup.
    try:
        dw_root, dw_tiles = resolve_data_paths(args.dw_data_root)
    except (ValueError, OSError):
        resolved_dw_root = Path(args.dw_data_root)
        if args.dw_data_root != args.data_root:
            resolved_dw_root = data_root
            dw_tiles = tiles_dir
        else:
            dw_tiles = None
        dw_root = resolved_dw_root
    _initial_paths["dw_data_root"] = dw_root
    _initial_paths["dw_tiles_dir"] = dw_tiles

    ui.run(
        host=args.host,
        port=args.port,
        title="AEF 变化检测平台",
        reload=False,
        show=args.show,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
