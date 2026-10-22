//! Navigation against neuroglancer's own navigation_state.ts, running the
//! scenario in scripts/golden/navigation.mts step for step. Every value is
//! compared exactly: positions and matrices are f32 in both, and the rest is
//! the same f64 arithmetic.

use std::sync::Arc;

use ng_core::coordinate_space::{BoundingBox, TransformedBoundingBox};
use ng_core::navigation::{Navigation, View};
use ng_core::panel::{wheel_zoom_amount, NamedAxes, PanelPointer, SlicePanel};
use ng_core::CoordinateSpace;
use serde_json::{json, Value};

fn space(scales: [f64; 3]) -> Arc<CoordinateSpace> {
    let names = ["x", "y", "z"].map(String::from).to_vec();
    let units = vec!["m".to_string(); 3];
    Arc::new(
        CoordinateSpace::new(true, names, units, scales.to_vec()).with_bounding_boxes(vec![
            TransformedBoundingBox::identity(BoundingBox {
                lower_bounds: vec![0.0; 3],
                upper_bounds: vec![100.0, 51.0, 7.0],
            }),
        ]),
    )
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn check(nav: &mut Navigation, expected: &Value) {
    let name = expected["name"].as_str().unwrap();
    let pos: Vec<f64> = nav.position().iter().map(|&c| c as f64).collect();
    assert_eq!(pos, floats(&expected["position"]), "{name}: position");
    assert_eq!(nav.zoom(View::CrossSection), expected["crossSectionScale"].as_f64().unwrap(), "{name}: crossSectionScale");
    assert_eq!(nav.zoom(View::Projection), expected["projectionScale"].as_f64().unwrap(), "{name}: projectionScale");
    assert_eq!(nav.depth_range(View::CrossSection), expected["crossSectionDepth"].as_f64().unwrap(), "{name}: crossSectionDepth");
    assert_eq!(nav.depth_range(View::Projection), expected["projectionDepth"].as_f64().unwrap(), "{name}: projectionDepth");
    assert_eq!(
        vec![nav.relative_depth_range(View::CrossSection), nav.relative_depth_range(View::Projection)],
        floats(&expected["relativeDepth"]),
        "{name}: relativeDepthRange"
    );
    for (view, key) in [(View::CrossSection, "crossMat"), (View::Projection, "projMat")] {
        let m: Vec<f64> = nav.to_mat4(view).iter().map(|&x| x as f64).collect();
        assert_eq!(m, floats(&expected[key]), "{name}: {key}");
    }
    let info = nav.render_info();
    assert_eq!(info.canonical_voxel_physical_size, expected["canonicalVoxelPhysicalSize"].as_f64().unwrap(), "{name}");
    assert_eq!(info.canonical_voxel_factors.to_vec(), floats(&expected["canonicalVoxelFactors"]), "{name}");
    let indices: Vec<f64> = info.display_dimension_indices.iter().map(|&i| i as f64).collect();
    assert_eq!(indices, floats(&expected["displayDimensionIndices"]), "{name}");
    assert_eq!(Value::Object(nav.to_json()), expected["json"], "{name}: state JSON");
}

#[test]
fn matches_neuroglancer_step_for_step() {
    let golden: Value = serde_json::from_str(include_str!("fixtures/navigation.json")).unwrap();
    let steps = golden.as_array().unwrap();
    let mut steps = steps.iter();
    let mut next = |nav: &mut Navigation, name: &str| {
        let expected = steps.next().unwrap();
        assert_eq!(expected["name"], name);
        check(nav, expected);
    };

    let mut nav = Navigation::new(Arc::new(CoordinateSpace::empty_invalid()));
    nav.set_coordinate_space(space([4e-9, 4e-9, 40e-9]));
    next(&mut nav, "initial");

    nav.translate_dimension_relative(2, 1.0);
    next(&mut nav, "z step");

    nav.update_display_position(|pos| {
        // vec3.add into a Float32Array.
        for (p, d) in pos.iter_mut().zip([-3.25f32, 7.5, 0.0]) {
            *p = (*p as f64 + d as f64) as f32;
        }
        true
    });
    next(&mut nav, "pan");

    let mut fixed = nav.position().to_vec();
    fixed[0] = (fixed[0] as f64 + 10.0) as f32;
    fixed[1] = (fixed[1] as f64 - 3.0) as f32;
    nav.rotate_absolute(View::CrossSection, [0.0, 1.0, 0.0], 0.3, &fixed);
    next(&mut nav, "rotate absolute");

    nav.rotate_relative(View::Projection, [1.0, 0.0, 0.0], -0.7);
    next(&mut nav, "rotate relative");

    nav.zoom_by(View::CrossSection, 1.1);
    nav.zoom_by(View::CrossSection, 1.1);
    nav.zoom_by(View::Projection, 0.5);
    next(&mut nav, "zoom");

    nav.translate_voxels_relative(View::CrossSection, [0.0, 0.0, 1.0]);
    next(&mut nav, "translate voxels");

    nav.set_depth_range(View::CrossSection, 5.0);
    next(&mut nav, "depth");

    nav.snap(View::CrossSection);
    next(&mut nav, "snap");

    let state = json!({"displayDimensions": ["z", "x"], "relativeDisplayScales": {"y": 2}});
    nav.restore_state(state.as_object().unwrap()).unwrap();
    next(&mut nav, "display dimensions");

    nav.set_coordinate_space(space([8e-9, 4e-9, 40e-9]));
    next(&mut nav, "rescaled space");

    let state = json!({
        "position": ["10.5", 20, "3.25px"], "crossSectionOrientation": [0, 0, 1, 1], "crossSectionScale": "2.5",
        "projectionOrientation": "bad", "projectionScale": 300, "crossSectionDepth": 0, "projectionDepth": 12,
        "displayDimensions": ["x", "y", "z"], "relativeDisplayScales": {},
    });
    nav.restore_state(state.as_object().unwrap()).unwrap();
    next(&mut nav, "restore");

    // Slice panels: same scenario as the TS, through `ng_core::panel`.
    let mut xy = SlicePanel::new(NamedAxes::Xy);
    let mut xz = SlicePanel::new(NamedAxes::Xz);
    let mut yz = SlicePanel::new(NamedAxes::Yz);
    let pointer = |x, y| PanelPointer { width: 400.0, height: 300.0, x, y };
    let mut next_panels = |nav: &mut Navigation, panels: [&mut SlicePanel; 3], name: &str| {
        let expected = steps.next().unwrap();
        assert_eq!(expected["name"], name);
        check(nav, expected);
        for (panel, key) in panels.into_iter().zip(["xy", "xz", "yz"]) {
            let e = &expected["panels"][key];
            let q: Vec<f64> = panel.orientation(nav).iter().map(|&x| x as f64).collect();
            assert_eq!(q, floats(&e["orientation"]), "{name}: {key} orientation");
            let m: Vec<f64> = panel.inv_view_matrix(nav).iter().map(|&x| x as f64).collect();
            assert_eq!(m, floats(&e["invView"]), "{name}: {key} invViewMatrix");
        }
    };

    next_panels(&mut nav, [&mut xy, &mut xz, &mut yz], "panels initial");
    xz.translate_by_viewport_pixels(&mut nav, 13.0, -7.0);
    next_panels(&mut nav, [&mut xy, &mut xz, &mut yz], "xz drag");
    yz.z_via_wheel(&mut nav, 120.0, 0.0, 1.0);
    yz.z_via_wheel(&mut nav, -3.0, 0.0, 10.0);
    xy.z_via_wheel(&mut nav, 0.0, 5.0, 1.0);
    next_panels(&mut nav, [&mut xy, &mut xz, &mut yz], "wheel steps");
    xy.zoom_by_mouse(&mut nav, wheel_zoom_amount(-240.0, 0), pointer(310.0, 77.0));
    xz.zoom_by_mouse(&mut nav, wheel_zoom_amount(3.0, 1), pointer(20.0, 260.0));
    next_panels(&mut nav, [&mut xy, &mut xz, &mut yz], "zoom by mouse");
    let initial = xz.mouse_position(&mut nav, pointer(250.0, 120.0));
    xz.rotate_via_mouse_drag(&mut nav, 8.0, -5.0, &initial);
    xz.rotate_via_mouse_drag(&mut nav, 3.0, 11.0, &initial);
    next_panels(&mut nav, [&mut xy, &mut xz, &mut yz], "xz rotate drag");
    yz.rotate_relative_axis(&mut nav, 0, 1.0);
    xz.rotate_relative_axis(&mut nav, 2, -1.0);
    next_panels(&mut nav, [&mut xy, &mut xz, &mut yz], "rotate relative keys");
    xz.translate_axis(&mut nav, 0, 1.0);
    yz.translate_axis(&mut nav, 1, -1.0);
    next_panels(&mut nav, [&mut xy, &mut xz, &mut yz], "arrow keys");
    yz.snap(&mut nav);
    next_panels(&mut nav, [&mut xy, &mut xz, &mut yz], "snap from yz");
    assert!(steps.next().is_none(), "golden has steps this test does not run");
}

#[test]
fn null_members_are_errors_or_resets_as_in_neuroglancer() {
    let mut nav = Navigation::new(space([1.0, 1.0, 1.0]));
    assert!(nav.restore_state(json!({"position": null}).as_object().unwrap()).is_err());
    nav.rotate_relative(View::CrossSection, [0.0, 0.0, 1.0], 1.0);
    nav.restore_state(json!({"crossSectionOrientation": null}).as_object().unwrap()).unwrap();
    assert_eq!(nav.orientation(View::CrossSection), [0.0, 0.0, 0.0, 1.0]);
}

/// Restore as the whole viewer does it: with the layer group viewer's linked
/// copies of the navigation state (`navigation_linked.mts`), which record a
/// restored zoom against the canonical voxel size of the space at hand — 1 m
/// when the state has no `dimensions` — and rescale it when the space
/// arrives or changes.
#[test]
fn restore_then_space_matches_the_linked_viewer() {
    use ng_core::navigation::{Navigation, View};
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/navigation_linked.json");
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for (name, case) in golden.as_object().unwrap() {
        let state = case["state"].as_object().unwrap();
        let mut nav = Navigation::new(std::sync::Arc::new(ng_core::CoordinateSpace::empty_invalid()));
        if let Some(dims) = state.get("dimensions") {
            nav.set_coordinate_space(std::sync::Arc::new(ng_core::CoordinateSpace::from_json(Some(dims), false).unwrap()));
        }
        nav.restore_state(state).unwrap();
        for (space, want) in case["spaces"].as_array().unwrap().iter().zip(case["steps"].as_array().unwrap()) {
            let names: Vec<String> = space["names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect();
            let rank = names.len();
            let scales: Vec<f64> = space["scales"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let upper: Vec<f64> = space["upper"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let s = ng_core::CoordinateSpace::new(true, names, vec!["m".into(); rank], scales).with_bounding_boxes(vec![
                ng_core::coordinate_space::TransformedBoundingBox::identity(ng_core::coordinate_space::BoundingBox {
                    lower_bounds: vec![0.0; rank],
                    upper_bounds: upper,
                }),
            ]);
            nav.set_coordinate_space(std::sync::Arc::new(s));
            assert_eq!(nav.zoom(View::CrossSection).to_bits(), want["crossSectionScale"].as_f64().unwrap().to_bits(), "{name}: crossSectionScale");
            assert_eq!(nav.zoom(View::Projection).to_bits(), want["projectionScale"].as_f64().unwrap().to_bits(), "{name}: projectionScale");
            let json = serde_json::Value::Object(nav.to_json());
            // Numbers compared as numbers (`0` and `0.0` alike).
            fn same(a: &serde_json::Value, b: &serde_json::Value) -> bool {
                match (a, b) {
                    (serde_json::Value::Number(x), serde_json::Value::Number(y)) => x.as_f64() == y.as_f64(),
                    (serde_json::Value::Array(x), serde_json::Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q)),
                    _ => a == b,
                }
            }
            for key in ["crossSectionScale", "projectionScale", "position"] {
                let (a, b) = (json.get(key), want["json"].get(key));
                let close = match (a, b) {
                    (Some(a), Some(b)) => same(a, b),
                    _ => a == b,
                };
                assert!(close, "{name}: {key}: {a:?} vs {b:?}");
            }
        }
    }
}
