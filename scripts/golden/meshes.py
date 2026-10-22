"""Legacy precomputed meshes for the segmentation fixture: a copy of
crates/ng-viewer/tests/fixtures/seg as seg_mesh, its info naming a `mesh`
directory, and for each segment its voxels' boundary faces (two triangles
each, four vertices of their own, so flat-shaded), every face moved 0.1 voxel
into the segment so neighbouring segments never share a plane.

Writes crates/ng-viewer/tests/fixtures/seg_mesh/.
"""

import json
import shutil
import struct
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[2] / "crates/ng-viewer/tests/fixtures"
SRC = FIXTURES / "seg"
OUT = FIXTURES / "seg_mesh"
INSET = 0.1

# The segmentation as state_volumes.py writes it.
sx, sy, sz = np.meshgrid(np.arange(20), np.arange(15), np.arange(10), indexing="ij")
seg = ((sx // 3) + 7 * (sy // 4) + 31 * (sz // 2)).astype(np.uint32)
seg[(sx + sy) % 5 == 0] = 0
resolution = np.array([8.0, 8.0, 80.0])

shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT)
info = json.loads((OUT / "info").read_text())
info["mesh"] = "mesh"
(OUT / "info").write_text(json.dumps(info))
(OUT / "mesh").mkdir()

shape = seg.shape
for sid in sorted(set(np.unique(seg).tolist()) - {0}):
    vertices = []
    triangles = []
    for x, y, z in zip(*np.nonzero(seg == sid)):
        for axis in range(3):
            for side in (0, 1):
                n = [x, y, z]
                n[axis] += 1 if side else -1
                inside = 0 <= n[axis] < shape[axis] and seg[tuple(n)] == sid
                if inside:
                    continue
                # The face at the voxel's `side` along `axis`, moved inward.
                plane = (1 - INSET) if side else INSET
                u, v = [a for a in range(3) if a != axis]
                base = len(vertices)
                for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = [float(x), float(y), float(z)]
                    p[axis] += plane
                    p[u] += du
                    p[v] += dv
                    vertices.append(np.array(p) * resolution)
                triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    data = struct.pack("<I", len(vertices))
    data += b"".join(struct.pack("<3f", *v) for v in vertices)
    data += b"".join(struct.pack("<3I", *t) for t in triangles)
    (OUT / "mesh" / f"{sid}_0").write_bytes(data)
    (OUT / "mesh" / f"{sid}:0").write_text(json.dumps({"fragments": [f"{sid}_0"]}))
print("written", len(list((OUT / "mesh").iterdir())) // 2, "meshes")
