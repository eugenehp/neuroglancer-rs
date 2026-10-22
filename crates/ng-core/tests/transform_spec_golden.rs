//! A data source's `transform` from viewer state over its default transform,
//! against neuroglancer's `WatchableCoordinateSpaceTransform.restoreState`
//! (`scripts/golden/transform_spec.mts`).

use std::sync::Arc;

use ng_core::coordinate_space::{BoundingBox, TransformedBoundingBox};
use ng_core::transform::{legacy_transform_spec, CoordinateSpaceTransform, TransformSpec};
use ng_core::CoordinateSpace;
use serde_json::{json, Value};

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn strs(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_owned()).collect()
}
fn norm(v: &Value) -> Value {
    match v {
        Value::Number(n) => json!(n.as_f64().unwrap()),
        Value::Array(a) => Value::Array(a.iter().map(norm).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), norm(v))).collect()),
        v => v.clone(),
    }
}
fn js(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { json!(if x > 0.0 { "Infinity" } else { "-Infinity" }) }
}
fn space(s: &CoordinateSpace) -> Value {
    let f = |v: &[f64]| v.iter().map(|&x| js(x)).collect::<Vec<_>>();
    json!({ "names": s.names, "units": s.units, "scales": f(&s.scales), "lower": f(&s.bounds.lower_bounds),
        "upper": f(&s.bounds.upper_bounds), "voxelCenter": s.bounds.voxel_center_at_integer_coordinates })
}

#[test]
fn transform_specs_apply_as_in_neuroglancer() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/transform_spec.json");
    let cases: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let d = &case["default"];
        let names = strs(&d["names"]);
        let rank = names.len();
        let input = CoordinateSpace::new(true, names.clone(), strs(&d["units"]), nums(&d["scales"])).with_bounding_boxes(vec![
            TransformedBoundingBox::identity(BoundingBox { lower_bounds: nums(&d["lower"]), upper_bounds: nums(&d["upper"]) }),
        ]);
        let output = CoordinateSpace::new(true, names, strs(&d["units"]), nums(&d["scales"]));
        let transform = d.get("transform").map(nums).unwrap_or_else(|| {
            let mut t = vec![0.0; (rank + 1) * (rank + 1)];
            for i in 0..=rank {
                t[i * (rank + 1) + i] = 1.0;
            }
            t
        });
        let default = CoordinateSpaceTransform { rank, source_rank: rank, input_space: Arc::new(input), output_space: Arc::new(output), transform }.normalized();
        let spec = if name.starts_with("legacy_") {
            legacy_transform_spec(Some(&case["spec"]))
        } else {
            TransformSpec::from_json(Some(&case["spec"]))
        }
        .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let v = default.with_spec(spec.as_ref());
        // Written back (`toJSON`): the spec relative to the default.
        let json = v.spec_over(&default).map_or(Value::Null, |s| s.to_json());
        assert_eq!(norm(&json), norm(&case["json"]), "{name}: toJSON");
        let got = json!({ "rank": v.rank, "sourceRank": v.source_rank, "transform": v.transform.iter().map(|&x| js(x)).collect::<Vec<_>>(),
            "input": space(&v.input_space), "output": space(&v.output_space) });
        assert_eq!(norm(&got), norm(&case["result"]), "{name}");
    }
}
