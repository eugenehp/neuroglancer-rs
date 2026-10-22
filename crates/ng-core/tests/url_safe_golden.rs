//! `urlSafeToJSON` against neuroglancer's (`scripts/golden/url_safe.mts`).

use ng_core::json::{parse_fragment, url_safe_to_json, Fragment};

#[test]
fn url_safe_to_json_matches_neuroglancer() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/url_safe.json");
    let cases: Vec<(String, String)> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for (input, want) in cases {
        assert_eq!(url_safe_to_json(&input), want, "{input:?}");
    }
}

#[test]
fn fragments() {
    assert_eq!(parse_fragment("").unwrap(), Fragment::Empty);
    assert_eq!(parse_fragment("#!").unwrap(), Fragment::Empty);
    assert_eq!(
        parse_fragment("#!%7B'a'%3A%5B1_2%5D%7D").unwrap(),
        Fragment::State(serde_json::json!({"a": [1, 2]}))
    );
    assert_eq!(parse_fragment("#!+{'b':true}").unwrap(), Fragment::Update(serde_json::json!({"b": true})));
    assert_eq!(parse_fragment("#!gs://bucket/state.json").unwrap(), Fragment::Remote("gs://bucket/state.json".into()));
    assert!(parse_fragment("#![1]").is_err());
    assert!(parse_fragment("#x").is_err());
}

/// Share-link encoding (`encodeStateAsFragment`) as neuroglancer's
/// (`scripts/golden/fragment.mts`): JavaScript's number formatting, string
/// escapes, and `encodeURI` plus `!'()*;,`.
#[test]
fn state_fragments_match_neuroglancer() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fragment.json");
    let cases: Vec<(serde_json::Value, String)> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for (state, want) in cases {
        assert_eq!(ng_core::json::encode_state_as_fragment(&state), want, "{state}");
        // And it reads back.
        assert_eq!(parse_fragment(&format!("#!{want}")).unwrap(), Fragment::State(state.clone()));
    }
}
