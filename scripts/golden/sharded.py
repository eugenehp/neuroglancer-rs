"""Sharded precomputed volumes (`neuroglancer_uint64_sharded_v1`) written by
tensorstore, with the voxels each should read back as.

Writes crates/ng-datasource/tests/fixtures/sharded/<case>/ (a precomputed
directory) and <case>.raw: the whole volume, x fastest then y, z, channel,
little-endian, zero where nothing was written.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import tensorstore as ts

OUT = Path(__file__).resolve().parents[2] / "crates/ng-datasource/tests/fixtures/sharded"
OUT.mkdir(parents=True, exist_ok=True)


def sharding(hash, preshift, minishard, shard, index_enc, data_enc):
    return {
        "@type": "neuroglancer_uint64_sharded_v1",
        "hash": hash,
        "preshift_bits": preshift,
        "minishard_bits": minishard,
        "shard_bits": shard,
        "minishard_index_encoding": index_enc,
        "data_encoding": data_enc,
    }


def write(name, dtype, size, channels, chunk, shard_spec, region, encoding="raw", volume_type="image", block=None):
    path = OUT / name
    shutil.rmtree(path, ignore_errors=True)
    scale = {
        "size": size,
        "resolution": [4, 4, 40],
        "chunk_size": chunk,
        "encoding": encoding,
        "sharding": shard_spec,
        "voxel_offset": [0, 0, 0],
    }
    if block:
        scale["compressed_segmentation_block_size"] = block
    store = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": str(path)},
            "multiscale_metadata": {"type": volume_type, "data_type": dtype, "num_channels": channels},
            "scale_metadata": scale,
        },
        create=True,
    ).result()
    shape = tuple(size) + (channels,)
    rng = np.random.default_rng(len(name))
    full = np.zeros(shape, dtype=dtype)
    values = rng.integers(1, np.iinfo(dtype).max, size=shape, dtype=dtype)
    if volume_type == "segmentation":
        values = (values % 7).astype(dtype) + np.array(2**40 if dtype == "uint64" else 9, dtype=dtype)
    sl = tuple(slice(a, b) for a, b in region) + (slice(None),)
    full[sl] = values[sl]
    store[sl] = values[sl]
    # x fastest: Fortran order of (x, y, z, c).
    (OUT / f"{name}.raw").write_bytes(full.astype(np.dtype(dtype).newbyteorder("<")).tobytes(order="F"))
    shards = sorted(p.name for p in path.rglob("*.shard"))
    print(name, json.loads((path / "info").read_text())["scales"][0]["key"], shards)


# murmurhash, gzip index and data; part of the volume written, so some
# shards, minishards and chunks are missing. Grid 7 × 3 × 8: 3, 2, 3 bits.
write("murmur_gzip_u16", "uint16", [50, 40, 30], 1, [8, 16, 4],
      sharding("murmurhash3_x86_128", 1, 2, 3, "gzip", "gzip"), [(0, 30), (5, 40), (3, 17)])
# identity hash, raw, two channels, edge chunks on every side.
write("identity_raw_u8x2", "uint8", [33, 17, 9], 2, [4, 4, 4],
      sharding("identity", 0, 3, 1, "raw", "raw"), [(0, 33), (0, 17), (0, 9)])
# One shard (shard_bits 0, "0.shard"), compressed segmentation of uint64.
write("one_shard_cseg_u64", "uint64", [20, 20, 10], 1, [8, 8, 8],
      sharding("murmurhash3_x86_128", 0, 1, 0, "gzip", "raw"), [(0, 20), (0, 13), (0, 10)],
      encoding="compressed_segmentation", volume_type="segmentation", block=[4, 4, 2])
