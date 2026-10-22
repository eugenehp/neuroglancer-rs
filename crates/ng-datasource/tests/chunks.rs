//! Chunk decoding against neuroglancer's encoder and zarr-python's writer.

use std::path::PathBuf;

use futures::executor::block_on;
use ng_datasource::chunk::{decode_compressed_segmentation, decode_jpeg, decode_png, read_chunk};
use ng_kvstore::{FileStore, KvStoreLocation};
use serde_json::Value;

fn here() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Fixtures encoded by neuroglancer's `encodeChannels`
/// (`scripts/golden/compressed_segmentation.mts`) decode back to the raw
/// values they were made from.
#[test]
fn compressed_segmentation_round_trips_neuroglancers_encoder() {
    let raw = std::fs::read(here().join("../../../neuroglancer/testdata/codec/compressed_segmentation/64x64x64-raw-uint64-segmentation.dat")).unwrap();
    let all: Vec<u64> = raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    let dir = here().join("tests/fixtures/compressed_segmentation");
    let index: Value = serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
    for (name, case) in index.as_object().unwrap() {
        let size: Vec<u32> = serde_json::from_value(case["size"].clone()).unwrap();
        let block: Vec<u32> = serde_json::from_value(case["block"].clone()).unwrap();
        let channels = case["channels"].as_u64().unwrap() as u32;
        let uint32 = case["uint32"].as_bool().unwrap();
        let encoded = std::fs::read(dir.join(format!("{name}.bin"))).unwrap();
        let decoded = decode_compressed_segmentation(
            &encoded,
            [size[0], size[1], size[2], channels],
            [block[0], block[1], block[2]],
            !uint32,
        )
        .unwrap();
        let elem = if uint32 { 4 } else { 8 };
        let n = (size[0] * size[1] * size[2]) as usize;
        let mut i = 0;
        for c in 0..channels as u64 {
            for z in 0..size[2] as usize {
                for y in 0..size[1] as usize {
                    for x in 0..size[0] as usize {
                        let want = all[x + 64 * (y + 64 * z)] + c;
                        let got = if uint32 {
                            u32::from_le_bytes(decoded[i * elem..i * elem + 4].try_into().unwrap()) as u64
                        } else {
                            u64::from_le_bytes(decoded[i * elem..i * elem + 8].try_into().unwrap())
                        };
                        let want = if uint32 { want & 0xffff_ffff } else { want };
                        assert_eq!(got, want, "{name} channel {c} at ({x},{y},{z})");
                        i += 1;
                    }
                }
            }
        }
        assert_eq!(i, n * channels as usize);
        // And the encoding's size, as neuroglancer's encoder made it.
        let bytes = ng_datasource::chunk::compressed_segmentation_encoded_bytes(&decoded, [size[0] as usize, size[1] as usize, size[2] as usize, channels as usize], [block[0] as usize, block[1] as usize, block[2] as usize], !uint32);
        assert_eq!(bytes, encoded.len(), "{name}: encoded size");
    }
}

/// Zarr chunks through the data source, assembled into whole arrays and
/// compared with zarr-python's values (exg-zarr's fixtures, written by
/// zarr-python), for the arrays neuroglancer supports: plain, F order (a
/// non-identity physical layout), and sharded with the index at either end.
#[test]
fn zarr_chunks_assemble_to_zarr_pythons_arrays() {
    // exg's own fixtures, from a checkout of it beside this one: skipped
    // without it (exg-zarr itself comes from crates.io).
    let fixtures = here().join("../../../exg/crates/exg-zarr/tests/fixtures/codecs");
    if !fixtures.join("golden.json").exists() {
        eprintln!("skipped: no exg checkout at {}", fixtures.display());
        return;
    }
    let golden: Value = serde_json::from_slice(&std::fs::read(fixtures.join("golden.json")).unwrap()).unwrap();
    let store = std::sync::Arc::new(FileStore::new(&fixtures));
    let mut checked = 0;
    for name in [
        "v2_raw_u16",
        "v2_zlib_i16",
        "v2_order_f_i32",
        "v2_big_endian_u16",
        "v2_fill_u64",
        "v2_blosc_lz4_shuffle_u32",
        "v3_transpose_f32",
        "v3_shard_end_u16",
        "v3_shard_start_u64",
        "v3_shard_transposed_i16",
    ] {
        let location = KvStoreLocation { store: store.clone(), path: format!("{name}.zarr") };
        let source = block_on(ng_datasource::zarr::get(&location)).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let volume = source.volume().unwrap();
        let s = &volume.sources[0][0];
        let elem = s.spec.data_type.bytes();
        // The spec is in physical order, x fastest; map back to logical C
        // order through the chunk → model transform (a permutation here).
        let rank = s.spec.rank;
        let n = rank + 1;
        let t = &s.chunk_to_multiscale_transform;
        let model_dim: Vec<usize> = (0..rank).map(|i| (0..rank).find(|&m| t[i * n + m] == 1.0).unwrap()).collect();
        let shape: Vec<usize> = serde_json::from_value(golden[name]["shape"].clone()).unwrap();
        let values = golden[name]["values"].as_array().unwrap();
        let grid = s.spec.grid_shape();
        let mut pos = vec![0u64; rank];
        for _ in 0..grid.iter().product::<u64>() {
            let chunk = block_on(read_chunk(s, &pos)).unwrap_or_else(|e| panic!("{name} {pos:?}: {e:#}"));
            let mut local = vec![0usize; rank];
            let csize: Vec<usize> = chunk.size.iter().map(|&x| x as usize).collect();
            for k in 0..csize.iter().product::<usize>() {
                // Spec voxel → model (logical) index.
                let mut logical = vec![0usize; rank];
                let mut inside = true;
                for i in 0..rank {
                    let v = pos[i] as usize * s.spec.chunk_data_size[i] as usize + local[i];
                    if v >= s.spec.upper_voxel_bound[i] as usize {
                        inside = false;
                    }
                    logical[model_dim[i]] = v;
                }
                if inside {
                    let flat = logical.iter().zip(&shape).fold(0, |acc, (&i, &d)| acc * d + i);
                    let got = &chunk.data[k * elem..(k + 1) * elem];
                    let want = &values[flat];
                    let ok = match s.spec.data_type {
                        ng_core::DataType::Uint64 => u64::from_le_bytes(got.try_into().unwrap()).to_string() == want.as_str().unwrap(),
                        ng_core::DataType::Float32 => {
                            let g = f32::from_le_bytes(got.try_into().unwrap());
                            let w = want.as_str().unwrap();
                            if w == "NaN" { g.is_nan() } else { g == w.parse::<f64>().unwrap() as f32 }
                        }
                        ng_core::DataType::Int16 => i16::from_le_bytes(got.try_into().unwrap()) as i64 == want.as_i64().unwrap(),
                        ng_core::DataType::Int32 => i32::from_le_bytes(got.try_into().unwrap()) as i64 == want.as_i64().unwrap(),
                        ng_core::DataType::Uint16 => u16::from_le_bytes(got.try_into().unwrap()) as i64 == want.as_i64().unwrap(),
                        ng_core::DataType::Uint32 => u32::from_le_bytes(got.try_into().unwrap()) as i64 == want.as_i64().unwrap(),
                        t => panic!("{t:?}"),
                    };
                    assert!(ok, "{name} at {logical:?}: got {got:?}, expected {want}");
                    checked += 1;
                }
                for d in 0..rank {
                    local[d] += 1;
                    if local[d] < csize[d] {
                        break;
                    }
                    local[d] = 0;
                }
            }
            for d in 0..rank {
                pos[d] += 1;
                if pos[d] < grid[d] {
                    break;
                }
                pos[d] = 0;
            }
        }
    }
    assert!(checked > 1000, "{checked}");
}

/// JPEG chunks decode as neuroglancer's jpgjs does, sample for sample
/// (`scripts/golden/images.py` writes the images — baseline, progressive,
/// restart intervals, 4:4:4 / 4:2:2 / 4:2:0 — and `jpeg_decode.mts` decodes
/// them with neuroglancer). Images jpgjs rejects are rejected too.
#[test]
fn jpeg_matches_neuroglancers_decoder() {
    let dir = here().join("tests/fixtures/images");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.unwrap().file_name().into_string().ok())
        .filter(|n| n.ends_with(".jpg"))
        .collect();
    names.sort();
    assert!(names.len() >= 9);
    for name in &names {
        let jpg = std::fs::read(dir.join(name)).unwrap();
        let channels = if name.starts_with("rgb") { 3 } else { 1 };
        // A 32×16×8 chunk stored as a 32×128 image.
        let got = decode_jpeg(&jpg, 32 * 16 * 8, channels);
        if let Ok(error) = std::fs::read_to_string(dir.join(format!("{name}.error"))) {
            let e = format!("{:#}", got.expect_err(name));
            assert!(e.contains(&error), "{name}: {e} vs {error}");
            continue;
        }
        let expected = std::fs::read(dir.join(format!("{name}.raw"))).unwrap();
        assert!(got.unwrap() == expected, "{name}");
    }
    let gray = std::fs::read(dir.join("gray_q75.jpg")).unwrap();
    assert!(decode_jpeg(&gray, 100, 1).is_err(), "area is checked");
    assert!(decode_jpeg(&gray, 32 * 128, 3).is_err(), "components are checked");
}

/// PNG chunks: samples as stored, 16-bit little-endian.
#[test]
fn png_decodes_samples_as_stored() {
    let dir = here().join("tests/fixtures/images");
    for (name, bytes) in [("gray8", 1), ("gray16", 2)] {
        let png = std::fs::read(dir.join(format!("{name}.png"))).unwrap();
        let expected = std::fs::read(dir.join(format!("{name}.png.raw"))).unwrap();
        assert_eq!(decode_png(&png, 32 * 128, 1, bytes).unwrap(), expected, "{name}");
        assert!(decode_png(&png, 32 * 128, 1, 3 - bytes).is_err(), "{name}: sample size is checked");
    }
}

/// Sharded precomputed volumes written by tensorstore
/// (`scripts/golden/sharded.py`) read back chunk by chunk: murmurhash and
/// identity hashes, gzip and raw indexes and data, missing shards, minishards
/// and chunks (zeros), compressed segmentation inside shards.
#[test]
fn sharded_precomputed_reads_tensorstores_volumes() {
    let dir = here().join("tests/fixtures/sharded");
    for name in ["murmur_gzip_u16", "identity_raw_u8x2", "one_shard_cseg_u64"] {
        let url = format!("precomputed://file://{}", dir.join(name).display());
        let source = block_on(ng_datasource::url::get(&url)).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let scale = &source.volume().unwrap().sources[0][0];
        let spec = &scale.spec;
        let bytes = spec.data_type.bytes();
        let expected = std::fs::read(dir.join(format!("{name}.raw"))).unwrap();
        // Single-channel specs have rank 3; pad to (x, y, z, channel).
        let pad = |v: Vec<usize>, c: usize| if v.len() == 3 { [v, vec![c]].concat() } else { v };
        let full = pad(spec.upper_voxel_bound.iter().map(|&u| u as usize).collect(), 1);
        let mut got = vec![0xAAu8; expected.len()];
        let shape = spec.grid_shape();
        let mut chunks = 0;
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    let grid = [x, y, z, 0];
                    let chunk = block_on(read_chunk(scale, &grid[..spec.rank])).unwrap_or_else(|e| panic!("{name}: {e:#}"));
                    chunks += 1;
                    let s = pad(chunk.size.iter().map(|&v| v as usize).collect(), 1);
                    let o = pad(chunk.origin.iter().map(|&v| v as usize).collect(), 0);
                    for c in 0..s[3] {
                        for k in 0..s[2] {
                            for j in 0..s[1] {
                                let src = ((c * s[2] + k) * s[1] + j) * s[0];
                                let dst = (((o[3] + c) * full[2] + o[2] + k) * full[1] + o[1] + j) * full[0] + o[0];
                                got[dst * bytes..(dst + s[0]) * bytes].copy_from_slice(&chunk.data[src * bytes..(src + s[0]) * bytes]);
                            }
                        }
                    }
                }
            }
        }
        assert!(got == expected, "{name}: {} of {} bytes differ", got.iter().zip(&expected).filter(|(a, b)| a != b).count(), got.len());
        eprintln!("{name}: {chunks} chunks");
    }
}

/// The sharded chunk key and hash match neuroglancer's
/// (`scripts/golden/sharding_keys.mts`).
#[test]
fn sharding_keys_match_neuroglancer() {
    use ng_datasource::sharded::{encode_z_index_compressed_3d, murmurhash3_x86_128_hash64};
    let golden: Value = serde_json::from_slice(&std::fs::read(here().join("tests/fixtures/sharding_keys.json")).unwrap()).unwrap();
    for case in golden["murmur"].as_array().unwrap() {
        let x: u64 = case[0].as_str().unwrap().parse().unwrap();
        let h: u64 = case[1].as_str().unwrap().parse().unwrap();
        assert_eq!(murmurhash3_x86_128_hash64(0, x), h, "murmur({x})");
    }
    for case in golden["zindex"].as_array().unwrap() {
        let bits: [u32; 3] = serde_json::from_value(case[0].clone()).unwrap();
        let pos: [u64; 3] = serde_json::from_value(case[1].clone()).unwrap();
        let z: u128 = case[2].as_str().unwrap().parse().unwrap();
        assert_eq!(encode_z_index_compressed_3d(bits, pos), z, "{bits:?} {pos:?}");
    }
}

/// A precomputed point collection (`scripts/golden/precomputed_annotations.py`)
/// as directories and in shards: the same cells (by grid position, or by
/// compressed Morton code) and `by_id` entries.
#[test]
fn precomputed_annotations_read_alike_sharded_or_not() {
    let fixtures = here().join("../ng-viewer/tests/fixtures");
    let info = |name: &str| {
        let url = format!("precomputed://file://{}", fixtures.join(name).display());
        let source = block_on(ng_datasource::url::get(&url)).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        match &source.subsources[0].subsource {
            ng_datasource::model::Subsource::Annotations(info) => (**info).clone(),
            _ => panic!("{name}: not annotations"),
        }
    };
    let (plain, sharded) = (info("annotations_points"), info("annotations_points_sharded"));
    assert!(sharded.spatial.iter().all(|l| l.index.sharding.is_some()));
    let mut total = 0;
    for (level, cells) in [(0, vec![[0, 0, 0]]), (1, (0..8).map(|k| [k & 1, k >> 1 & 1, k >> 2]).collect())] {
        for cell in cells {
            let a = block_on(ng_datasource::annotation::load_cell(&plain, level, &cell)).unwrap();
            let b = block_on(ng_datasource::annotation::load_cell(&sharded, level, &cell)).unwrap();
            assert_eq!(a, b, "level {level} cell {cell:?}");
            total += a.len();
        }
    }
    assert_eq!(total, 120);
    for id in [1000, 1062, 1119] {
        let a = block_on(ng_datasource::annotation::load_by_id(&plain, id)).unwrap().unwrap();
        let b = block_on(ng_datasource::annotation::load_by_id(&sharded, id)).unwrap().unwrap();
        assert_eq!(a, b);
        assert_eq!((a.id, a.properties.len()), (id, 2));
    }
    assert!(block_on(ng_datasource::annotation::load_by_id(&sharded, 5)).unwrap().is_none());
}

/// A three-scale N5 volume (`scripts/golden/n5.py`: gzip, blosc and zstd
/// blocks, truncated at the edges) read back block by block.
#[test]
fn n5_blocks_read_back() {
    let dir = here().join("../ng-viewer/tests/fixtures/n5_image");
    let url = format!("n5://file://{}", dir.display());
    let source = block_on(ng_datasource::url::get(&url)).unwrap_or_else(|e| panic!("{e:#}"));
    let base = |x: u64, y: u64, z: u64| ((x * 37 + y * 23 + z * 101) % 997 + 3 * x * y) as u16;
    let scales = &source.volume().unwrap().sources[0];
    assert_eq!(scales.len(), 3);
    for (s, scale) in scales.iter().enumerate() {
        let spec = &scale.spec;
        let shape = spec.grid_shape();
        for gx in 0..shape[0] {
            for gy in 0..shape[1] {
                for gz in 0..shape[2] {
                    let chunk = block_on(read_chunk(scale, &[gx, gy, gz])).unwrap_or_else(|e| panic!("s{s} {gx}/{gy}/{gz}: {e:#}"));
                    let (sx, sy) = (chunk.size[0] as u64, chunk.size[1] as u64);
                    for (k, v) in chunk.data.chunks_exact(2).enumerate() {
                        let k = k as u64;
                        let (x, y, z) = (chunk.origin[0] as u64 + k % sx, chunk.origin[1] as u64 + k / sx % sy, chunk.origin[2] as u64 + k / (sx * sy));
                        assert_eq!(u16::from_le_bytes([v[0], v[1]]), base(x << s, y << s, z), "s{s} ({x}, {y}, {z})");
                    }
                }
            }
        }
    }
}

/// Deep Zoom tiles (`scripts/golden/deepzoom.py`): an interior JPEG tile of
/// the full-resolution level, without its overlap, planar, close to the
/// pattern on average (PNG tiles are transposed twice, as neuroglancer
/// does, so are not the pattern) (its chroma is subsampled across the pattern's
/// sharp edges: libjpeg decodes it 6 off on average too).
#[test]
fn deepzoom_tiles_read_back() {
    let dir = here().join("../ng-viewer/tests/fixtures/deepzoom");
    let pattern = |x: u32, y: u32| [x * 255 / 299, y * 255 / 199, ((x / 25 + y / 25) % 2) * 200 + 30];
    for (name, tolerance) in [("pattern_jpg.dzi", 7.0)] {
        let url = format!("file://{}", dir.join(name).display());
        let source = block_on(ng_datasource::url::get(&url)).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert!(source.canonical_url.ends_with("|deepzoom:"));
        let finest = &source.volume().unwrap().sources[0][0];
        let chunk = block_on(read_chunk(finest, &[1, 1, 0])).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert_eq!(chunk.size, vec![64, 64, 3]);
        let mut total = 0;
        for k in 0..3usize {
            for j in 0..64u32 {
                for i in 0..64u32 {
                    let got = chunk.data[i as usize + j as usize * 64 + k * 4096] as i32;
                    total += (got - pattern(64 + i, 64 + j)[k] as i32).abs();
                }
            }
        }
        let mean = total as f64 / (3 * 64 * 64) as f64;
        assert!(mean <= tolerance, "{name}: off by {mean} on average");
    }
}

/// neuroglancer's OME-Zarr test data inside a zip archive: found by its
/// signature (`…zip|zip:|zarr3:`), its chunks those of the unzipped copy.
#[test]
fn zip_archives_read_as_their_contents() {
    let dir = here().join("../../../neuroglancer/testdata/datasource/zarr/ome_zarr");
    let zipped = block_on(ng_datasource::url::get(&format!("file://{}", dir.join("simple_0.5.zip").display()))).unwrap_or_else(|e| panic!("{e:#}"));
    assert!(zipped.canonical_url.ends_with("simple_0.5.zip|zip:|zarr3:"), "{}", zipped.canonical_url);
    let plain = block_on(ng_datasource::url::get(&format!("file://{}/", dir.join("simple_0.5").display()))).unwrap();
    let (a, b) = (&zipped.volume().unwrap().sources[0], &plain.volume().unwrap().sources[0]);
    assert_eq!(a.len(), b.len());
    for (sa, sb) in a.iter().zip(b) {
        let grid = vec![0; sa.spec.rank];
        assert_eq!(block_on(read_chunk(sa, &grid)).unwrap(), block_on(read_chunk(sb, &grid)).unwrap());
    }
}
