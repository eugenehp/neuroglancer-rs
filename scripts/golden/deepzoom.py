"""Two Deep Zoom images (300 × 200, 64 px tiles with a 1 px overlap) of the
same RGB pattern, one tiled as PNG and one as JPEG, as Deep Zoom lays them
out: level L (0 the 1 × 1 one) the image shrunk by 2^(max − L), rounding
up; tile (x, y) its 64 × 64 cell plus the overlap on each side that has a
neighbour, in `<name>_files/<L>/<x>_<y>.<format>`.

Writes crates/ng-viewer/tests/fixtures/deepzoom/.
"""

import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[2] / "crates/ng-viewer/tests/fixtures/deepzoom"
W, H, TILE, OVERLAP = 300, 200, 64, 1

x, y = np.meshgrid(np.arange(W), np.arange(H))
rgb = np.stack([(x * 255 // (W - 1)), (y * 255 // (H - 1)), ((x // 25 + y // 25) % 2) * 200 + 30], axis=-1).astype(np.uint8)
image = Image.fromarray(rgb, "RGB")

shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
max_level = math.ceil(math.log2(max(W, H)))
tiles = 0
for fmt in ["png", "jpg"]:
    name = f"pattern_{fmt}"
    (OUT / f"{name}.dzi").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<Image xmlns="http://schemas.microsoft.com/deepzoom/2009" Format="{fmt}" Overlap="{OVERLAP}" TileSize="{TILE}">\n  <Size Height="{H}" Width="{W}"/>\n</Image>\n'
    )
    for level in range(max_level + 1):
        scale = 2 ** (max_level - level)
        lw, lh = math.ceil(W / scale), math.ceil(H / scale)
        level_image = image.resize((lw, lh), Image.LANCZOS) if scale > 1 else image
        d = OUT / f"{name}_files" / str(level)
        d.mkdir(parents=True)
        for tx in range(math.ceil(lw / TILE)):
            for ty in range(math.ceil(lh / TILE)):
                x0 = tx * TILE - (OVERLAP if tx > 0 else 0)
                y0 = ty * TILE - (OVERLAP if ty > 0 else 0)
                x1 = min((tx + 1) * TILE + OVERLAP, lw)
                y1 = min((ty + 1) * TILE + OVERLAP, lh)
                tile = level_image.crop((x0, y0, x1, y1))
                if fmt == "png":
                    tile.save(d / f"{tx}_{ty}.png")
                else:
                    tile.save(d / f"{tx}_{ty}.jpg", quality=90)
                tiles += 1
print("written", tiles, "tiles, levels 0 to", max_level)
