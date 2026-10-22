//! `CoordinateSpaceCombiner` against neuroglancer's own
//! (`scripts/golden/combiner.mts`): the same steps, the same global space and
//! layer transforms after each, bit for bit. Dimension IDs are not compared
//! (opaque; see `ng_core::combiner`).

use std::sync::Arc;

use ng_core::combiner::{Binding, BoundSpace, CoordinateSpaceCombiner};
use ng_core::coordinate_space::{is_global_dimension, BoundingBox, TransformedBoundingBox};
use ng_core::transform::CoordinateSpaceTransform;
use ng_core::CoordinateSpace;
use serde_json::{json, Value};

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn strs(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_owned()).collect()
}
/// Numbers as the golden writes them (non-finite as strings).
fn js(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else {
        json!(if x.is_nan() { "NaN".to_owned() } else if x > 0.0 { "Infinity".into() } else { "-Infinity".into() })
    }
}
/// Every number as an `f64`, so `0` and `0.0` compare equal.
fn norm(v: &Value) -> Value {
    match v {
        Value::Number(n) => json!(n.as_f64().unwrap()),
        Value::Array(a) => Value::Array(a.iter().map(norm).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), norm(v))).collect()),
        v => v.clone(),
    }
}
fn space_json(s: &CoordinateSpace) -> Value {
    json!({
        "valid": s.valid,
        "names": s.names,
        "units": s.units,
        "scales": s.scales.iter().map(|&x| js(x)).collect::<Vec<_>>(),
        "timestamps": s.timestamps.iter().map(|&x| js(x)).collect::<Vec<_>>(),
        "lower": s.bounds.lower_bounds.iter().map(|&x| js(x)).collect::<Vec<_>>(),
        "upper": s.bounds.upper_bounds.iter().map(|&x| js(x)).collect::<Vec<_>>(),
        "voxelCenter": s.bounds.voxel_center_at_integer_coordinates,
    })
}

fn make_layer(spec: &Value) -> CoordinateSpaceTransform {
    let names = strs(&spec["names"]);
    let rank = names.len();
    let input = CoordinateSpace::new(true, names.clone(), strs(&spec["units"]), nums(&spec["scales"])).with_bounding_boxes(vec![
        TransformedBoundingBox::identity(BoundingBox {
            lower_bounds: nums(&spec["lower"]),
            upper_bounds: nums(&spec["upper"]),
        }),
    ]);
    let or = |k: &str, d: &str| if spec.get(k).is_some() { k.to_owned() } else { d.to_owned() };
    let output = CoordinateSpace::new(true, strs(&spec[or("outNames", "names")]), strs(&spec[or("outUnits", "units")]), nums(&spec[or("outScales", "scales")]));
    let transform = match spec.get("transform") {
        Some(t) => nums(t),
        None => {
            let mut t = vec![0.0; (rank + 1) * (rank + 1)];
            for i in 0..=rank {
                t[i * (rank + 1) + i] = 1.0;
            }
            t
        }
    };
    CoordinateSpaceTransform {
        rank,
        source_rank: rank,
        input_space: Arc::new(input),
        output_space: Arc::new(output),
        transform,
    }
    .normalized()
}

#[test]
fn combiner_matches_neuroglancer() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/combiner.json");
    let golden: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for (name, scenario) in golden.as_object().unwrap() {
        let mut global = CoordinateSpace::empty_invalid();
        let mut combiner = CoordinateSpaceCombiner::new(is_global_dimension);
        let mut layers: Vec<(String, Binding, CoordinateSpaceTransform)> = vec![];
        let mut defaults: Vec<(String, CoordinateSpaceTransform)> = vec![];
        let update = |combiner: &mut CoordinateSpaceCombiner, global: &mut CoordinateSpace, layers: &mut Vec<(String, Binding, CoordinateSpaceTransform)>| {
            let mut bound: Vec<(&mut Binding, &mut dyn BoundSpace)> =
                layers.iter_mut().map(|(_, b, t)| (b, t as &mut dyn BoundSpace)).collect();
            combiner.update(global, &mut bound);
        };
        let steps = scenario["steps"].as_array().unwrap();
        let results = scenario["results"].as_array().unwrap();
        for (i, (step, want)) in steps.iter().zip(results).enumerate() {
            match step["op"].as_str().unwrap() {
                "restore" => {
                    global = CoordinateSpace::from_json(Some(&step["dims"]), false).unwrap();
                    // Without bindings nothing listens for the change.
                    if !layers.is_empty() {
                        update(&mut combiner, &mut global, &mut layers);
                    }
                }
                "retain" => combiner.retain(),
                "release" => {
                    combiner.release();
                    update(&mut combiner, &mut global, &mut layers);
                }
                "bind" => {
                    let default = make_layer(&step["spec"]);
                    let spec = ng_core::transform::TransformSpec::from_json(step.get("transformSpec")).unwrap();
                    let t = default.with_spec(spec.as_ref());
                    defaults.push((step["layer"].as_str().unwrap().to_owned(), default));
                    layers.push((step["layer"].as_str().unwrap().to_owned(), Binding::default(), t));
                    update(&mut combiner, &mut global, &mut layers);
                }
                "unbind" => {
                    layers.retain(|(id, _, _)| id != step["layer"].as_str().unwrap());
                    update(&mut combiner, &mut global, &mut layers);
                }
                "editGlobal" => {
                    let d = step["dim"].as_u64().unwrap() as usize;
                    global.scales[d] = step["scale"].as_f64().unwrap();
                    global.timestamps[d] = step["timestamp"].as_f64().unwrap();
                    update(&mut combiner, &mut global, &mut layers);
                }
                op => panic!("unknown op {op}"),
            }
            assert_eq!(norm(&space_json(&global)), norm(&want["global"]), "{name} step {i}: global");
            let got_layers: serde_json::Map<String, Value> = layers
                .iter()
                .map(|(id, _, t)| {
                    let default = &defaults.iter().find(|(d, _)| d == id).unwrap().1;
                    (id.clone(), json!({
                        "transform": t.transform.iter().map(|&x| js(x)).collect::<Vec<_>>(),
                        "output": space_json(&t.output_space),
                        "json": t.spec_over(default).map_or(Value::Null, |s| s.to_json()),
                    }))
                })
                .collect();
            assert_eq!(norm(&Value::Object(got_layers)), norm(&want["layers"]), "{name} step {i}: layers");
        }
    }
}
