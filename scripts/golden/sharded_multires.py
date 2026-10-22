"""The multi-resolution mesh fixture in shards: a copy of
crates/ng-viewer/tests/fixtures/seg_multires as seg_multires_sharded, its
mesh `info` given `neuroglancer_uint64_sharded_v1` parameters (identity
hash, 1 minishard bit, 1 shard bit, raw encodings) and each segment's
fragments then manifest written as one shard entry, the entry being the
manifest (the fragments end where it begins, as neuroglancer reads them).

Reads seg_multires (scripts/golden/multires_meshes.py), so needs no Draco.
Writes crates/ng-viewer/tests/fixtures/seg_multires_sharded/.
"""

import json
import shutil
import struct
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / "crates/ng-viewer/tests/fixtures"
SRC = FIXTURES / "seg_multires"
OUT = FIXTURES / "seg_multires_sharded"
MINISHARD_BITS = 1
SHARD_BITS = 1

shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT)
mesh_dir = OUT / "multires"
info = json.loads((mesh_dir / "info").read_text())
info["sharding"] = {
    "@type": "neuroglancer_uint64_sharded_v1",
    "hash": "identity",
    "preshift_bits": 0,
    "minishard_bits": MINISHARD_BITS,
    "shard_bits": SHARD_BITS,
    "minishard_index_encoding": "raw",
    "data_encoding": "raw",
}

segments = sorted(int(p.name[: -len(".index")]) for p in mesh_dir.glob("*.index"))
# shard → minishard → [(key, fragments + manifest, manifest length)]
shards: dict[int, dict[int, list]] = {}
for sid in segments:
    manifest = (mesh_dir / f"{sid}.index").read_bytes()
    fragments = (mesh_dir / f"{sid}").read_bytes()
    minishard = sid & ((1 << MINISHARD_BITS) - 1)
    shard = (sid >> MINISHARD_BITS) & ((1 << SHARD_BITS) - 1)
    shards.setdefault(shard, {}).setdefault(minishard, []).append((sid, fragments + manifest, len(manifest)))
    (mesh_dir / f"{sid}.index").unlink()
    (mesh_dir / f"{sid}").unlink()
(mesh_dir / "info").write_text(json.dumps(info))

index_size = 16 << MINISHARD_BITS
digits = (SHARD_BITS + 3) // 4
for shard, minishards in shards.items():
    data = bytearray()  # after the shard index
    indexes = {}
    for minishard in range(1 << MINISHARD_BITS):
        entries = sorted(minishards.get(minishard, []))
        rows = []
        for key, blob, manifest_len in entries:
            start = len(data) + len(blob) - manifest_len
            data += blob
            rows.append((key, start, manifest_len))
        indexes[minishard] = rows
    shard_index = bytearray()
    minishard_data = bytearray()
    for minishard in range(1 << MINISHARD_BITS):
        rows = indexes[minishard]
        begin = len(data) + len(minishard_data)
        # Delta-encoded keys, starts (from the previous end) and sizes.
        keys, starts, sizes = [], [], []
        prev_key, prev_end = 0, 0
        for key, start, size in rows:
            keys.append(key - prev_key)
            starts.append(start - prev_end)
            sizes.append(size)
            prev_key, prev_end = key, start + size
        minishard_data += struct.pack(f"<{3 * len(rows)}Q", *(keys + starts + sizes))
        end = len(data) + len(minishard_data)
        shard_index += struct.pack("<QQ", begin, end)
    assert len(shard_index) == index_size
    (mesh_dir / f"{shard:0{digits}x}.shard").write_bytes(bytes(shard_index) + bytes(data) + bytes(minishard_data))
print("written", len(segments), "segments in", len(shards), "shards")
