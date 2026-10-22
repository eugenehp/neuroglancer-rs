//! NIfTI volume info and voxels against values computed with neuroglancer's
//! own libraries (nifti-reader-js, gl-matrix) and logic, on neuroglancer's test
//! files. Regenerate with `node scripts/golden/nifti_info.mjs`.

use ng_core::DataType;
use ng_datasource::nifti;
use serde_json::Value;

const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../neuroglancer/testdata/datasource/nifti/");

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_owned()).collect()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

/// Voxel `i` as JSON would hold it: a number, or a decimal string for uint64.
fn voxel(data: &[u8], t: DataType, i: usize) -> (f64, Value) {
    let b = &data[i * t.bytes()..(i + 1) * t.bytes()];
    match t {
        DataType::Uint8 => (b[0] as f64, b[0].into()),
        DataType::Int8 => (b[0] as i8 as f64, (b[0] as i8).into()),
        DataType::Uint16 => {
            let v = u16::from_le_bytes([b[0], b[1]]);
            (v as f64, v.into())
        }
        DataType::Int16 => {
            let v = i16::from_le_bytes([b[0], b[1]]);
            (v as f64, v.into())
        }
        DataType::Uint32 => {
            let v = u32::from_le_bytes(b.try_into().unwrap());
            (v as f64, v.into())
        }
        DataType::Int32 => {
            let v = i32::from_le_bytes(b.try_into().unwrap());
            (v as f64, v.into())
        }
        DataType::Uint64 => {
            let v = u64::from_le_bytes(b.try_into().unwrap());
            (v as f64, v.to_string().into())
        }
        DataType::Float32 => {
            let v = f32::from_le_bytes(b.try_into().unwrap());
            (v as f64, (v as f64).into())
        }
    }
}

#[test]
fn matches_neuroglancer_on_its_test_files() {
    let golden: Value =
        serde_json::from_str(include_str!("fixtures/nifti_info.json")).unwrap();
    for (name, expected) in golden.as_object().unwrap() {
        let bytes = std::fs::read(format!("{TESTDATA}{name}")).unwrap();
        let volume = nifti::load(&bytes).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let (info, e) = (&volume.info, &expected["info"]);

        assert_eq!(info.rank as u64, e["rank"].as_u64().unwrap(), "{name} rank");
        assert_eq!(info.source_names, strings(&e["sourceNames"]), "{name}");
        assert_eq!(info.view_names, strings(&e["viewNames"]), "{name}");
        assert_eq!(info.units, strings(&e["units"]), "{name}");
        assert_eq!(info.source_scales, floats(&e["sourceScales"]), "{name} sourceScales");
        assert_eq!(info.view_scales, floats(&e["viewScales"]), "{name} viewScales");
        // Bit for bit: the f32 rounding follows gl-matrix's.
        assert_eq!(info.transform, floats(&e["transform"]), "{name} transform");
        assert_eq!(info.data_type.name(), e["dataType"].as_str().unwrap(), "{name}");
        let size: Vec<u32> = floats(&e["volumeSize"]).iter().map(|&v| v as u32).collect();
        assert_eq!(info.volume_size, size, "{name} volumeSize");
        assert_eq!(info.description, e["description"].as_str().unwrap(), "{name}");

        let v = &expected["voxels"];
        let count = v["count"].as_u64().unwrap() as usize;
        assert_eq!(volume.data.len(), count * info.data_type.bytes(), "{name} payload");
        let mut sum = 0.0;
        for i in 0..count {
            let (x, json) = voxel(&volume.data, info.data_type, i);
            if let Some(first) = v["first"].as_array().unwrap().get(i) {
                assert_eq!(&json, first, "{name} voxel {i}");
            }
            sum += x;
        }
        assert_eq!(sum, v["sum"].as_f64().unwrap(), "{name} voxel sum");
    }
}
