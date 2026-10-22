"""A three-channel 8-bit precomputed image over the image fixture's region
(40 × 30 × 20 voxels at 4 × 4 × 40 nm, chunks of 16 × 16 × 8 holding every
channel), raw-encoded by hand: each chunk file `x0-x1_y0-y1_z0-z1` holds
its voxels x fastest, then y, z and the channel.

Channel 0 rises along x, 1 along y, 2 along z, so a shader mapping them to
red, green and blue shows each channel apart.

Writes crates/ng-viewer/tests/fixtures/channels/.
"""

import json
import shutil
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "crates/ng-viewer/tests/fixtures/channels"
SIZE = (40, 30, 20)
CHUNK = (16, 16, 8)
CHANNELS = 3

x, y, z = np.meshgrid(*(np.arange(n) for n in SIZE), indexing="ij")
data = np.stack([(x * 6) % 256, (y * 8) % 256, (z * 12) % 256], axis=-1).astype(np.uint8)

shutil.rmtree(OUT, ignore_errors=True)
key = "4_4_40"
(OUT / key).mkdir(parents=True)
(OUT / "info").write_text(
    json.dumps(
        {
            "@type": "neuroglancer_multiscale_volume",
            "data_type": "uint8",
            "num_channels": CHANNELS,
            "type": "image",
            "scales": [{"chunk_sizes": [list(CHUNK)], "encoding": "raw", "key": key, "resolution": [4.0, 4.0, 40.0], "size": list(SIZE), "voxel_offset": [0, 0, 0]}],
        }
    )
)
count = 0
for x0 in range(0, SIZE[0], CHUNK[0]):
    for y0 in range(0, SIZE[1], CHUNK[1]):
        for z0 in range(0, SIZE[2], CHUNK[2]):
            x1, y1, z1 = min(x0 + CHUNK[0], SIZE[0]), min(y0 + CHUNK[1], SIZE[1]), min(z0 + CHUNK[2], SIZE[2])
            block = data[x0:x1, y0:y1, z0:z1, :]
            # Fortran order over (x, y, z, c): x fastest, channel slowest.
            (OUT / key / f"{x0}-{x1}_{y0}-{y1}_{z0}-{z1}").write_bytes(block.tobytes(order="F"))
            count += 1
print("written", count, "chunks")
