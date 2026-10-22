"""A precomputed point annotation collection (`neuroglancer_annotations_v1`)
over the image fixture's region (40 × 30 × 20 at 4 × 4 × 40 nm), written by
hand: 120 points with an `rgb` colour and a `float32` size, in a two-level
spatial index as neuroglancer's writer lays one out — `spatial0` one cell
holding a random 12 (its limit), `spatial1` a 2 × 2 × 2 grid holding the
rest by cell — each cell's points shuffled, and `by_id` with each point on
its own.

Also the same collection with its spatial levels and `by_id` in shards
(`neuroglancer_uint64_sharded_v1`: identity hash, 1 minishard bit, 1 shard
bit, raw; cells keyed by their compressed Morton code).

Writes crates/ng-viewer/tests/fixtures/annotations_points/ and
annotations_points_sharded/.
"""

import json
import random
import shutil
import struct
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / "crates/ng-viewer/tests/fixtures"
OUT = FIXTURES / "annotations_points"
SHARDED = FIXTURES / "annotations_points_sharded"
SHARDING = {
    "@type": "neuroglancer_uint64_sharded_v1",
    "hash": "identity",
    "preshift_bits": 0,
    "minishard_bits": 1,
    "shard_bits": 1,
    "minishard_index_encoding": "raw",
    "data_encoding": "raw",
}
SIZE = (40, 30, 20)
COUNT = 120
LIMIT0 = 12

rng = random.Random(7)
points = []
for i in range(COUNT):
    p = tuple(round(rng.uniform(0, s), 2) for s in SIZE)
    color = tuple(rng.randrange(256) for _ in range(3))
    size = round(rng.uniform(1, 10), 1)
    points.append((1000 + i, p, color, size))


def record(p):
    # Geometry, then the float32 (alignment 4) before the rgb, padded to 4.
    _, xyz, color, size = p
    return struct.pack("<3f", *xyz) + struct.pack("<f", size) + bytes(color) + b"\0"


def encode(ps):
    out = struct.pack("<Q", len(ps)) + b"".join(record(p) for p in ps)
    return out + b"".join(struct.pack("<Q", p[0]) for p in ps)


shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
order = points[:]
rng.shuffle(order)
top, rest = order[:LIMIT0], order[LIMIT0:]
(OUT / "spatial0").mkdir()
(OUT / "spatial0" / "0_0_0").write_bytes(encode(top))
grid = (2, 2, 2)
chunk = tuple(s / g for s, g in zip(SIZE, grid))
cells = {}
for p in rest:
    cell = tuple(min(int(p[1][k] // chunk[k]), grid[k] - 1) for k in range(3))
    cells.setdefault(cell, []).append(p)
(OUT / "spatial1").mkdir()
for cell, ps in cells.items():
    (OUT / "spatial1" / "_".join(map(str, cell))).write_bytes(encode(ps))
(OUT / "by_id").mkdir()
for p in points:
    (OUT / "by_id" / str(p[0])).write_bytes(record(p))
info = {
    "@type": "neuroglancer_annotations_v1",
    "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]},
    "lower_bound": [0, 0, 0],
    "upper_bound": list(SIZE),
    "annotation_type": "point",
    "properties": [{"id": "color", "type": "rgb"}, {"id": "size", "type": "float32"}],
    "relationships": [],
    "by_id": {"key": "by_id"},
    "spatial": [
        {"key": "spatial0", "grid_shape": [1, 1, 1], "chunk_size": list(SIZE), "limit": LIMIT0},
        {"key": "spatial1", "grid_shape": list(grid), "chunk_size": list(chunk), "limit": max(len(v) for v in cells.values())},
    ],
}
(OUT / "info").write_text(json.dumps(info))


def compressed_morton(position, shape):
    """`encodeZIndexCompressed3d`: the bits of each coordinate interleaved,
    each dimension only as many as its grid size needs."""
    bits = [max(0, (s - 1).bit_length()) for s in shape]
    out, out_bit = 0, 0
    for bit in range(max(bits)):
        for d in range(3):
            if bit < bits[d]:
                out |= ((position[d] >> bit) & 1) << out_bit
                out_bit += 1
    return out


def write_shards(directory, entries):
    """`entries` (key → bytes) as sharded, raw, identity-hashed."""
    directory.mkdir(parents=True)
    mb, sb = SHARDING["minishard_bits"], SHARDING["shard_bits"]
    shards = {}
    for key, blob in entries.items():
        shards.setdefault((key >> mb) & ((1 << sb) - 1), {}).setdefault(key & ((1 << mb) - 1), []).append((key, blob))
    digits = (sb + 3) // 4
    for shard, minishards in shards.items():
        data, index, tables = bytearray(), bytearray(), bytearray()
        rows = {}
        for m in range(1 << mb):
            rows[m] = []
            for key, blob in sorted(minishards.get(m, [])):
                rows[m].append((key, len(data), len(blob)))
                data += blob
        for m in range(1 << mb):
            begin = len(data) + len(tables)
            keys, starts, sizes = [], [], []
            prev_key, prev_end = 0, 0
            for key, start, size in rows[m]:
                keys.append(key - prev_key)
                starts.append(start - prev_end)
                sizes.append(size)
                prev_key, prev_end = key, start + size
            tables += struct.pack(f"<{3 * len(rows[m])}Q", *(keys + starts + sizes))
            index += struct.pack("<QQ", begin, len(data) + len(tables))
        (directory / f"{shard:0{digits}x}.shard").write_bytes(bytes(index) + bytes(data) + bytes(tables))


shutil.rmtree(SHARDED, ignore_errors=True)
SHARDED.mkdir(parents=True)
write_shards(SHARDED / "spatial0", {0: encode(top)})
write_shards(SHARDED / "spatial1", {compressed_morton(cell, grid): encode(ps) for cell, ps in cells.items()})
write_shards(SHARDED / "by_id", {p[0]: record(p) for p in points})
sharded_info = json.loads(json.dumps(info))
for level in sharded_info["spatial"]:
    level["sharding"] = SHARDING
sharded_info["by_id"]["sharding"] = SHARDING
(SHARDED / "info").write_text(json.dumps(sharded_info))
print("written", COUNT, "points:", LIMIT0, "at level 0,", {k: len(v) for k, v in sorted(cells.items())})
