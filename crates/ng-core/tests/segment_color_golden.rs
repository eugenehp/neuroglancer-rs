//! Segment colors against values computed by neuroglancer's own TS sources.
//! Regenerate with `node scripts/golden/segment_colors.mts > \
//! crates/ng-core/tests/fixtures/segment_colors.json`.

use ng_core::segment_color::hash_combine;
use ng_core::SegmentColorHash;
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("fixtures/segment_colors.json")).unwrap()
}

#[test]
fn hash_combine_matches_typescript() {
    for row in golden()["hashCombine"].as_array().unwrap() {
        let [state, value, expected] = [0, 1, 2].map(|i| row[i].as_u64().unwrap() as u32);
        assert_eq!(hash_combine(state, value), expected, "hashCombine({state}, {value})");
    }
}

#[test]
fn colors_match_typescript_exactly() {
    let rows = golden()["colors"].as_array().unwrap().clone();
    assert!(rows.len() >= 40);
    for row in rows {
        let seed = row[0].as_u64().unwrap() as u32;
        // IDs are strings: uint64 doesn't survive a JSON number.
        let id: u64 = row[1].as_str().unwrap().parse().unwrap();
        let expected: Vec<f32> = row[2]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_f64().unwrap() as f32)
            .collect();
        let actual = SegmentColorHash::new(seed).compute(id);
        assert_eq!(actual.as_slice(), expected.as_slice(), "seed {seed}, id {id}");
    }
}
