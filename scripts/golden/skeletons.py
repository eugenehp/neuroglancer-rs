"""Precomputed skeletons for the segmentation fixture: a copy of
crates/ng-viewer/tests/fixtures/seg as seg_skeleton, its info naming a
`skeletons` directory (`neuroglancer_skeletons`, with a float32 `radius`
vertex attribute), and for each segment a small tree in nanometres: from the
centroid of its voxels to two opposite corners of its bounding box, with a
branch to a third, so the edges run diagonally through the slices.

Writes crates/ng-viewer/tests/fixtures/seg_skeleton/.
"""

import json
import shutil
import struct
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[2] / "crates/ng-viewer/tests/fixtures"
SRC = FIXTURES / "seg"
OUT = FIXTURES / "seg_skeleton"

# The segmentation as state_volumes.py writes it.
sx, sy, sz = np.meshgrid(np.arange(20), np.arange(15), np.arange(10), indexing="ij")
seg = ((sx // 3) + 7 * (sy // 4) + 31 * (sz // 2)).astype(np.uint32)
seg[(sx + sy) % 5 == 0] = 0
resolution = np.array([8.0, 8.0, 80.0])

shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT)
info = json.loads((OUT / "info").read_text())
info["skeletons"] = "skeletons"
(OUT / "info").write_text(json.dumps(info))
(OUT / "skeletons").mkdir()
(OUT / "skeletons" / "info").write_text(
    json.dumps(
        {
            "@type": "neuroglancer_skeletons",
            "vertex_attributes": [{"id": "radius", "data_type": "float32", "num_components": 1}],
        }
    )
)

count = 0
for sid in sorted(set(np.unique(seg).tolist()) - {0}):
    voxels = np.argwhere(seg == sid).astype(np.float64) + 0.5
    lower, upper = voxels.min(axis=0), voxels.max(axis=0)
    centre = voxels.mean(axis=0)
    vertices = [centre, lower, upper, np.array([lower[0], upper[1], centre[2]])]
    edges = [(0, 1), (0, 2), (0, 3)]
    radii = [2.0, 1.0, 1.0, 0.5]
    data = struct.pack("<II", len(vertices), len(edges))
    data += b"".join(struct.pack("<3f", *(v * resolution)) for v in vertices)
    data += b"".join(struct.pack("<2I", *e) for e in edges)
    data += b"".join(struct.pack("<f", r) for r in radii)
    (OUT / "skeletons" / str(sid)).write_bytes(data)
    count += 1
print("written", count, "skeletons")
