"""JPEG and PNG chunk fixtures: images shaped like precomputed chunks
(width = chunk x, height = chunk y * z), written by Pillow.

    python3 scripts/golden/images.py
    node --import ./scripts/golden/ng_register.mjs scripts/golden/jpeg_decode.mts

The second command decodes the JPEGs with neuroglancer's own decoder
(third_party/jpgjs) to `*.jpg.raw`: what neuroglancer shows for them.
"""
import pathlib

import numpy as np
from PIL import Image

OUT = pathlib.Path(__file__).resolve().parents[2] / "crates/ng-datasource/tests/fixtures/images"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)


def texture(h, w, channels):
    y, x = np.mgrid[0:h, 0:w]
    base = (128 + 100 * np.sin(x / 7.0) * np.cos(y / 5.0) + rng.normal(0, 12, (h, w))).clip(0, 255)
    if channels == 1:
        return base.astype(np.uint8)
    return np.stack([base, 255 - base, (base * 0.5 + x % 64)], axis=-1).clip(0, 255).astype(np.uint8)


# Chunk 32×16×8: a 32 × 128 image.
# Baseline and progressive scans, restart intervals, 4:4:4 / 4:2:2 / 4:2:0.
for name, channels, quality, extra in [
    ("gray_q75", 1, 75, {}),
    ("gray_q95", 1, 95, {}),
    ("rgb_q85_444", 3, 85, {"subsampling": 0}),
    ("rgb_q85_420", 3, 85, {"subsampling": 2}),
    ("rgb_q60_422", 3, 60, {"subsampling": 1}),
    ("gray_q80_progressive", 1, 80, {"progressive": True}),
    ("rgb_q80_420_progressive", 3, 80, {"subsampling": 2, "progressive": True}),
    ("rgb_q90_restart", 3, 90, {"subsampling": 2, "restart_marker_blocks": 3}),
    ("gray_q50_restart_progressive", 1, 50, {"progressive": True, "restart_marker_rows": 1}),
]:
    img = Image.fromarray(texture(128, 32, channels))
    kw = {"quality": quality, **extra}
    img.save(OUT / f"{name}.jpg", **kw)

# PNG: 8-bit gray, 16-bit gray (exact; big-endian in the file).
g8 = texture(128, 32, 1)
Image.fromarray(g8).save(OUT / "gray8.png")
(OUT / "gray8.png.raw").write_bytes(g8.tobytes())
g16 = (g8.astype(np.uint16) * 257 + np.arange(32, dtype=np.uint16)).astype(np.uint16)
Image.fromarray(g16).save(OUT / "gray16.png")
(OUT / "gray16.png.raw").write_bytes(g16.astype("<u2").tobytes())
print("written")
