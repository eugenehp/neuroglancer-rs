"""Multi-resolution (Draco) precomputed meshes for the segmentation fixture:
a copy of crates/ng-viewer/tests/fixtures/seg as seg_multires, its info
naming a `multires` directory in the `neuroglancer_multilod_draco` format.

Each segment has two levels of detail over an octree of 4-voxel nodes:
level 0 its voxels' boundary faces, level 1 those of the volume downsampled
2×2×2 (a block in the segment if any of its voxels is), each face moved 10%
of a (level's) voxel into the segment. A fragment is one octree node's faces,
positions normalized to the node and quantized to 10 bits; encoded with
Draco quantization over that grid, so the stored integers are what
neuroglancer reads (it skips Draco's dequantization). Level 1 faces never
cross a node's 2×2×2 partition planes, as the format requires.

Needs DracoPy (`pip install DracoPy`). Writes
crates/ng-viewer/tests/fixtures/seg_multires/.
"""

import json
import shutil
import struct
from pathlib import Path

import DracoPy
import numpy as np

FIXTURES = Path(__file__).resolve().parents[2] / "crates/ng-viewer/tests/fixtures"
SRC = FIXTURES / "seg"
OUT = FIXTURES / "seg_multires"
BITS = 10
MAXQ = (1 << BITS) - 1
NODE = 4  # voxels per level-0 node
INSET = 0.1

sx, sy, sz = np.meshgrid(np.arange(20), np.arange(15), np.arange(10), indexing="ij")
seg = ((sx // 3) + 7 * (sy // 4) + 31 * (sz // 2)).astype(np.uint32)
seg[(sx + sy) % 5 == 0] = 0
resolution = np.array([8.0, 8.0, 80.0])

shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT)
info = json.loads((OUT / "info").read_text())
info["mesh"] = "multires"
(OUT / "info").write_text(json.dumps(info))
mesh_dir = OUT / "multires"
mesh_dir.mkdir()
(mesh_dir / "info").write_text(
    json.dumps(
        {
            "@type": "neuroglancer_multilod_draco",
            "vertex_quantization_bits": BITS,
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "lod_scale_multiplier": 1.0,
        }
    )
)


def faces(mask, scale):
    """Boundary faces of `mask` (voxels `scale` base voxels wide), in base
    voxel units: each a list of 4 corners."""
    out = []
    shape = mask.shape
    for x, y, z in zip(*np.nonzero(mask)):
        for axis in range(3):
            for side in (0, 1):
                n = [x, y, z]
                n[axis] += 1 if side else -1
                if 0 <= n[axis] < shape[axis] and mask[tuple(n)]:
                    continue
                plane = (1 - INSET) if side else INSET
                u, v = [a for a in range(3) if a != axis]
                quad = []
                for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = [float(x), float(y), float(z)]
                    p[axis] += plane
                    p[u] += du
                    p[v] += dv
                    quad.append(np.array(p) * scale)
                out.append(quad)
    return out


def zorder_key(p):
    key = 0
    for bit in range(10):
        for axis in range(3):
            key |= ((p[axis] >> bit) & 1) << (3 * bit + axis)
    return key


chunk_shape = NODE * resolution  # level-0 node, in nanometres
for sid in sorted(set(np.unique(seg).tolist()) - {0}):
    mask0 = seg == sid
    padded = np.zeros((20, 16, 10), bool)
    padded[:, :15, :] = mask0
    mask1 = padded.reshape(10, 2, 8, 2, 5, 2).any(axis=(1, 3, 5))
    lods = []
    for lod, (mask, scale) in enumerate(((mask0, 1), (mask1, 2))):
        node = NODE * (1 << lod)  # node size in base voxels
        nodes = {}
        for quad in faces(mask, scale):
            centre = sum(quad) / 4
            key = tuple(int(c // node) for c in centre)
            nodes.setdefault(key, []).append(quad)
        frags = []
        for key in sorted(nodes, key=zorder_key):
            origin = np.array(key) * node
            points, triangles = [], []
            for quad in nodes[key]:
                base = len(points)
                for p in quad:
                    # Normalized to the node, then on the 10-bit grid.
                    points.append((p - origin) / node * MAXQ)
                triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
            data = DracoPy.encode(
                np.array(points, dtype=np.float32),
                np.array(triangles, dtype=np.uint32),
                quantization_bits=BITS,
                quantization_range=MAXQ,
                quantization_origin=np.zeros(3),
                compression_level=7,
            )
            frags.append((key, data))
        lods.append(frags)
    manifest = struct.pack("<3f", *chunk_shape) + struct.pack("<3f", 0, 0, 0)
    manifest += struct.pack("<I", len(lods))
    manifest += struct.pack(f"<{len(lods)}f", *[resolution[0] * (1 << lod) for lod in range(len(lods))])
    manifest += struct.pack(f"<{3 * len(lods)}f", *([0.0] * 3 * len(lods)))
    manifest += struct.pack(f"<{len(lods)}I", *[len(f) for f in lods])
    blob = b""
    for frags in lods:
        for axis in range(3):
            manifest += struct.pack(f"<{len(frags)}I", *[k[axis] for k, _ in frags])
        manifest += struct.pack(f"<{len(frags)}I", *[len(d) for _, d in frags])
        blob += b"".join(d for _, d in frags)
    (mesh_dir / f"{sid}.index").write_bytes(manifest)
    (mesh_dir / f"{sid}").write_bytes(blob)
print("written", len(list(mesh_dir.glob("*.index"))), "meshes")
