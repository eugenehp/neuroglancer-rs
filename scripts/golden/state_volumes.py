"""Two precomputed volumes for the viewer-state test, written by tensorstore:
an 8-bit image at 4×4×40 nm and a uint32 segmentation at half that
resolution, over the same region.

Writes crates/ng-viewer/tests/fixtures/{image,seg}/.
"""

import shutil
from pathlib import Path

import numpy as np
import tensorstore as ts

OUT = Path(__file__).resolve().parents[2] / "crates/ng-viewer/tests/fixtures"


def write(name, dtype, volume_type, size, resolution, chunk, data, encoding="raw"):
    path = OUT / name
    shutil.rmtree(path, ignore_errors=True)
    store = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": str(path)},
            "multiscale_metadata": {"type": volume_type, "data_type": dtype, "num_channels": 1},
            "scale_metadata": {"size": size, "resolution": resolution, "chunk_size": chunk, "encoding": encoding},
        },
        create=True,
    ).result()
    store[...] = data[..., None]


x, y, z = np.meshgrid(np.arange(40), np.arange(30), np.arange(20), indexing="ij")
image = ((x * 7 + y * 3 + z * 11) % 256).astype(np.uint8)
write("image", "uint8", "image", [40, 30, 20], [4, 4, 40], [16, 16, 8], image)

sx, sy, sz = np.meshgrid(np.arange(20), np.arange(15), np.arange(10), indexing="ij")
seg = ((sx // 3) + 7 * (sy // 4) + 31 * (sz // 2)).astype(np.uint32)
seg[(sx + sy) % 5 == 0] = 0
write("seg", "uint32", "segmentation", [20, 15, 10], [8, 8, 80], [8, 8, 4], seg)
print("written")
