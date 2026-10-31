"""Build a tiny synthetic data package for the data-root switch check.

The package contains one tile whose embeddings, mask, manifest and gallery
image are generated here, so switching roots in the browser can be verified
without touching (or copying) the real 89 MiB tiles under ``data/spectral``.

Usage:  .venv/Scripts/python.exe scratch/make_test_package.py <output-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def build(root: Path) -> Path:
    tiles = root / "tiles"
    tile = tiles / "T99"
    embeddings = tile / "embeddings"
    embeddings.mkdir(parents=True, exist_ok=True)
    (root / "gallery").mkdir(parents=True, exist_ok=True)
    (tile / "distance_maps").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    base = rng.normal(size=(128, 128, 64)).astype(np.float32)
    base /= np.linalg.norm(base, axis=-1, keepdims=True)
    changed = base.copy()
    # A deliberately changed block so the synthetic tile has a visible signal.
    changed[40:80, 40:80] = -changed[40:80, 40:80]
    frames = np.stack([base, changed], axis=0)
    np.save(embeddings / "demo_full.npy", frames)

    index = {
        "name": "demo_full",
        "encoder_source": "demo",
        "view": "full",
        "frames": [
            {"raw_index": 0, "date_utc": "2024-01-01", "timestamp_ms": 0.0},
            {"raw_index": 1, "date_utc": "2024-06-01", "timestamp_ms": 1.0},
        ],
        "shape": list(frames.shape),
        "dtype": "float32",
    }
    (embeddings / "demo_full_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mask = np.ones((128, 128), dtype=bool)
    mask[:4, :4] = False
    np.savez_compressed(tile / "distance_maps" / "demo_full.npz", common_mask=mask)

    manifest = {
        "id": "T99",
        "category": "合成测试瓦片",
        "split": "test",
        "parent": "synthetic/tiles/T99_E100.0000_S10.0000_20240101_20241231",
        "views": [{"name": "demo_full", "encoder_source": "demo", "frames": 2}],
    }
    (tile / "tile_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :, 0] = 200
    Image.fromarray(image).save(root / "gallery" / "T99_demo_0.png")

    gallery = [
        {
            "id": "T99",
            "source": "demo",
            "date": "2024-01-01",
            "image": "spectral/gallery/T99_demo_0.png",
        }
    ]
    (root / "gallery" / "index.json").write_text(
        json.dumps(gallery, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve()
    print(build(target))
