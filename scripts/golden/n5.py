"""A three-scale uint16 N5 volume over the image fixture's region (40 × 30
× 20 voxels at 4 × 4 × 40 nm, axes x, y, z), written by hand: each scale an
`s<i>` dataset (blocks of 16 × 16 × 8, its `downsamplingFactors`) whose
blocks are `<x>/<y>/<z>` files — a big-endian header (mode 0, rank, the
block's shape, truncated at the edges) and big-endian elements, x fastest —
compressed with gzip (s0), blosc lz4 with byte shuffle (s1) and zstd (s2).

Writes crates/ng-viewer/tests/fixtures/n5_image/.
"""

import gzip
import json
import shutil
import struct
from pathlib import Path

import numcodecs
import numpy as np
import zstandard

OUT = Path(__file__).resolve().parents[2] / "crates/ng-viewer/tests/fixtures/n5_image"
SIZE = (40, 30, 20)
BLOCK = (16, 16, 8)

x, y, z = np.meshgrid(*(np.arange(n) for n in SIZE), indexing="ij")
base = ((x * 37 + y * 23 + z * 101) % 997 + 3 * x * y).astype(np.uint16)

compressors = [
    ({"type": "gzip", "level": 6}, lambda b: gzip.compress(b, mtime=0)),
    ({"type": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}, lambda b: numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE).encode(b)),
    ({"type": "zstd", "level": 3}, lambda b: zstandard.ZstdCompressor(level=3).compress(b)),
]

shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
(OUT / "attributes.json").write_text(json.dumps({"axes": ["x", "y", "z"], "units": ["nm", "nm", "nm"], "resolution": [4, 4, 40]}))
data = base
blocks = 0
for scale, (compression, compress) in enumerate(compressors):
    factors = [2**scale, 2**scale, 1]
    if scale > 0:
        # Every other voxel in x and y (and the edge one where odd).
        data = base[:: 2**scale, :: 2**scale, :]
    shape = data.shape
    d = OUT / f"s{scale}"
    d.mkdir()
    (d / "attributes.json").write_text(json.dumps({"dataType": "uint16", "dimensions": list(shape), "blockSize": list(BLOCK), "compression": compression, "downsamplingFactors": factors}))
    for bx in range(0, shape[0], BLOCK[0]):
        for by in range(0, shape[1], BLOCK[1]):
            for bz in range(0, shape[2], BLOCK[2]):
                block = data[bx : bx + BLOCK[0], by : by + BLOCK[1], bz : bz + BLOCK[2]]
                header = struct.pack(">HH", 0, 3) + struct.pack(">3I", *block.shape)
                payload = block.astype(">u2").tobytes(order="F")
                path = d / str(bx // BLOCK[0]) / str(by // BLOCK[1]) / str(bz // BLOCK[2])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(header + compress(payload))
                blocks += 1
print("written", blocks, "blocks")
