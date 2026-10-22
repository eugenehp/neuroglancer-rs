//! Data source metadata against neuroglancer's own:
//!
//! - its metadata snapshots (`tests/datasource/metadata_snapshots`, converted
//!   by `scripts/golden/snapshots.py`), for NIfTI, precomputed, Zarr, N5 and
//!   Deep Zoom;
//! - its Zarr and OME-NGFF parsers run on its test data
//!   (`scripts/golden/zarr_metadata.mts`): every array accepted or rejected as
//!   neuroglancer does, with the same metadata; every OME transform bit for bit.

use std::path::PathBuf;
use std::sync::Arc;

use futures::executor::block_on;
use ng_datasource::zarr::{self, Node};
use ng_kvstore::{ByteRange, FileStore, KvStore, KvStoreLocation, ReadFuture};
use serde_json::Value;

fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../neuroglancer/testdata/datasource")
}

/// neuroglancer's test data, under the URLs its snapshots were taken with.
struct TestServer(FileStore);

impl KvStore for TestServer {
    fn read(&self, key: &str, range: ByteRange) -> ReadFuture {
        self.0.read(key, range)
    }
    fn url(&self, key: &str) -> String {
        format!("http://localhost:*/datasource/{key}")
    }
}

fn location(path: &str) -> KvStoreLocation {
    KvStoreLocation {
        store: Arc::new(TestServer(FileStore::new(testdata()))),
        path: path.to_owned(),
    }
}

fn snapshot(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/snapshots").join(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

/// The first place two JSON values differ, or `None`.
fn first_difference(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys()) {
                let (va, vb) = (x.get(k).unwrap_or(&Value::Null), y.get(k).unwrap_or(&Value::Null));
                if let Some(d) = first_difference(va, vb, &format!("{path}.{k}")) {
                    return Some(d);
                }
            }
            None
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Some(format!("{path}: length {} vs {}", x.len(), y.len()));
            }
            x.iter().zip(y).enumerate().find_map(|(i, (va, vb))| first_difference(va, vb, &format!("{path}[{i}]")))
        }
        (Value::Number(x), Value::Number(y)) if x.as_f64() == y.as_f64() => None,
        _ if a == b => None,
        _ => Some(format!("{path}: ours {a} vs neuroglancer {b}")),
    }
}

fn assert_snapshot(name: &str, ours: Value) {
    if let Some(d) = first_difference(&ours, &snapshot(name), "") {
        panic!("{name}: {d}");
    }
}

#[test]
fn nifti_matches_snapshots() {
    for file in ["standard.nii", "standard.nii.gz", "example_nifti2.nii.gz"] {
        let bytes = std::fs::read(testdata().join("nifti").join(file)).unwrap();
        let info = ng_datasource::nifti::load(&bytes).unwrap().info;
        // neuroglancer inserts a gzip adapter for `.gz` files.
        let mut url = format!("http://localhost:*/datasource/nifti/{file}");
        if file.ends_with(".gz") {
            url.push_str("|gzip:");
        }
        assert_snapshot(&format!("nifti__{file}"), info.data_source(&url).snapshot());
    }
}

#[test]
fn precomputed_matches_snapshots() {
    for name in ["one_channel", "two_channels"] {
        let source = block_on(ng_datasource::precomputed::get(&location(&format!("precomputed/{name}")))).unwrap();
        assert_snapshot(&format!("precomputed__{name}"), source.snapshot());
    }
}

#[test]
fn n5_matches_snapshots() {
    // n5-viewer multiscale groups: `scales` at the top (deprecated), or
    // `downsamplingFactors` in each `s<i>` (modern).
    for name in ["n5_viewer_multiscale_deprecated", "n5_viewer_multiscale_modern"] {
        let source = block_on(ng_datasource::n5::get(&location(&format!("n5/{name}")))).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert_snapshot(&format!("n5__{name}"), source.snapshot());
    }
}

#[test]
fn deepzoom_matches_snapshot() {
    let source = block_on(ng_datasource::deepzoom::get(&location("deepzoom/14122_mPPC_BDA_s186.dzi"))).unwrap_or_else(|e| panic!("{e:#}"));
    assert_snapshot("deepzoom__14122_mPPC_BDA_s186.dzi", source.snapshot());
}

#[test]
fn zarr_matches_snapshots() {
    for (path, name) in [
        ("zarr/zarr_v3/examples/single_res", "zarr_v3_examples_single_res"),
        ("zarr/ome_zarr/simple_0.4", "ome_zarr_simple_0.4"),
        ("zarr/ome_zarr/simple_0.5", "ome_zarr_simple_0.5"),
    ] {
        let source = block_on(zarr::get(&location(path))).unwrap_or_else(|e| panic!("{path}: {e:#}"));
        assert_snapshot(&format!("zarr__{name}"), source.snapshot());
    }
    // The same OME-Zarr inside a zip archive (`|zip:`).
    let zip = KvStoreLocation { store: Arc::new(ng_kvstore::ZipStore::new(location("zarr/ome_zarr/simple_0.5.zip"))), path: String::new() };
    let source = block_on(zarr::get(&zip)).unwrap_or_else(|e| panic!("zip: {e:#}"));
    assert_snapshot("zarr__ome_zarr_simple_0.5.zip", source.snapshot());
}

fn golden() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/zarr_metadata.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn read(dir: &std::path::Path, name: &str) -> Option<Vec<u8>> {
    std::fs::read(dir.join(name)).ok()
}

#[test]
fn zarr_arrays_parse_as_neuroglancer_parses_them() {
    let golden = golden();
    let arrays = golden["arrays"].as_object().unwrap();
    assert_eq!(arrays.len(), 74);
    let root = testdata().join("zarr");
    for (rel, expected) in arrays {
        let dir = root.join(rel);
        let parsed = match (read(&dir, ".zarray"), read(&dir, "zarr.json")) {
            (Some(zarray), _) => zarr::parse_v2(&zarray, read(&dir, ".zattrs").as_deref()).map(|a| Node::Array(Box::new(a))),
            (None, Some(json)) => zarr::parse_v3(&json),
            _ => panic!("{rel}: no metadata"),
        };
        if let Some(err) = expected.get("error") {
            assert!(parsed.is_err(), "{rel}: neuroglancer rejects it ({err}) but we accept it");
            continue;
        }
        let a = match parsed {
            Ok(Node::Array(a)) => a,
            Ok(Node::Group(_)) => panic!("{rel}: parsed as a group"),
            Err(e) => panic!("{rel}: we reject it ({e:#}) but neuroglancer accepts it"),
        };
        let shape: Vec<u64> = serde_json::from_value(expected["shape"].clone()).unwrap();
        let chunks: Vec<usize> = serde_json::from_value(expected["chunkShape"].clone()).unwrap();
        assert_eq!(a.codec.shape, shape, "{rel} shape");
        assert_eq!(a.codec.chunk_shape, chunks, "{rel} chunkShape");
        assert_eq!(a.data_type.name(), expected["dataType"].as_str().unwrap(), "{rel} dataType");
        let names: Vec<Option<String>> = serde_json::from_value(expected["dimensionNames"].clone()).unwrap();
        assert_eq!(a.dimension_names, names, "{rel} dimensionNames");
        let units: Vec<Option<String>> = serde_json::from_value(expected["dimensionUnits"].clone()).unwrap();
        assert_eq!(a.dimension_units, units, "{rel} dimensionUnits");
        let p2l: Vec<usize> = serde_json::from_value(expected["physicalToLogicalDimension"].clone()).unwrap();
        assert_eq!(a.physical_to_logical, p2l, "{rel} physicalToLogicalDimension");
        let read_shape: Vec<usize> = serde_json::from_value(expected["readChunkShape"].clone()).unwrap();
        assert_eq!(a.read_chunk_shape, read_shape, "{rel} readChunkShape");
        // Fill value, as neuroglancer holds it (uint64 as a decimal string).
        let fill = &a.codec.fill_value;
        let ours: Value = match a.data_type {
            ng_core::DataType::Uint64 => u64::from_le_bytes(fill[..8].try_into().unwrap()).to_string().into(),
            ng_core::DataType::Float32 => (f32::from_le_bytes(fill[..4].try_into().unwrap()) as f64).into(),
            _ => {
                let mut b = [0u8; 8];
                b[..fill.len()].copy_from_slice(fill);
                let signed = a.data_type.signed() == Some(true);
                let shift = 64 - 8 * fill.len() as u32;
                let v = if signed { ((i64::from_le_bytes(b) << shift) >> shift) as f64 } else { u64::from_le_bytes(b) as f64 };
                ng_core::js::number(v)
            }
        };
        let expected_fill = &expected["fillValue"];
        let same = match (&ours, expected_fill) {
            (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
            _ => &ours == expected_fill,
        };
        assert!(same, "{rel} fillValue: ours {ours} vs {expected_fill}");
    }
}

#[test]
fn ome_parses_as_neuroglancer_parses_it() {
    let golden = golden();
    let groups = golden["ome"].as_object().unwrap();
    assert_eq!(groups.len(), 10);
    let root = testdata().join("zarr");
    for (rel, expected) in groups {
        let dir = root.join(rel);
        let (attrs, version) = match read(&dir, "zarr.json") {
            Some(j) => {
                let v: Value = serde_json::from_slice(&j).unwrap();
                (v["attributes"].clone(), 3)
            }
            None => (serde_json::from_slice(&read(&dir, ".zattrs").unwrap()).unwrap(), 2),
        };
        let url = format!("http://test/{rel}/");
        let result = zarr::ome::parse(&url, &attrs, version);
        if let Some(err) = expected.get("error") {
            assert!(result.is_err(), "{rel}: neuroglancer rejects it ({err})");
            continue;
        }
        let m = result.unwrap_or_else(|e| panic!("{rel}: {e:#}")).unwrap_or_else(|| panic!("{rel}: no OME metadata"));
        let f = |v: &Value| -> Vec<f64> { serde_json::from_value(v.clone()).unwrap() };
        assert_eq!(m.coordinate_space.names, serde_json::from_value::<Vec<String>>(expected["names"].clone()).unwrap(), "{rel}");
        assert_eq!(m.coordinate_space.units, serde_json::from_value::<Vec<String>>(expected["units"].clone()).unwrap(), "{rel}");
        assert_eq!(m.coordinate_space.scales, f(&expected["scales"]), "{rel} scales");
        assert_eq!(m.base_scales, f(&expected["baseScales"]), "{rel} baseScales");
        assert_eq!(m.base_transform, f(&expected["baseTransform"]), "{rel} baseTransform");
        let datasets = expected["datasets"].as_array().unwrap();
        assert_eq!(m.scales.len(), datasets.len(), "{rel}");
        for (s, d) in m.scales.iter().zip(datasets) {
            assert_eq!(s.url, d["url"].as_str().unwrap(), "{rel}");
            assert_eq!(s.transform, f(&d["transform"]), "{rel} {} transform", s.url);
        }
    }
}

/// Every way of naming neuroglancer's test data opens the same source.
#[test]
fn urls_open_sources() {
    let td = testdata();
    for (url, rank) in [
        (format!("nifti://file://{}/nifti/standard.nii", td.display()), 3),
        (format!("{}/nifti/example_nifti2.nii.gz", td.display()), 4),
        (format!("file://{}/precomputed/one_channel|neuroglancer-precomputed:", td.display()), 3),
        (format!("{}/precomputed/two_channels", td.display()), 4),
        (format!("zarr://file://{}/zarr/ome_zarr/simple_0.5", td.display()), 2),
        (format!("{}/zarr/ome_zarr/simple_0.4", td.display()), 2),
    ] {
        let source = block_on(ng_datasource::url::get(&url)).unwrap_or_else(|e| panic!("{url}: {e:#}"));
        assert_eq!(source.volume().unwrap().rank, rank, "{url}");
    }
}
