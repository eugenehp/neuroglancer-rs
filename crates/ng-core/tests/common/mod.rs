//! Readers for the golden fixtures' JSON, shared by the golden tests.
#![allow(dead_code)]

use serde_json::Value;

/// An array of numbers.
pub fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

/// An array of strings.
pub fn strs(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_owned()).collect()
}
