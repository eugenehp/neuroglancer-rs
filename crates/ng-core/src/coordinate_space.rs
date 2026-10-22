//! Named, unit-carrying coordinate spaces. Port of the `CoordinateSpace` part
//! of `src/coordinate_transform.ts`.
//!
//! The combiner that merges layer spaces into the global one is in
//! [`crate::combiner`].

use anyhow::{anyhow, bail, ensure, Context, Result};
use serde_json::{Map, Value};

use crate::si_units::{parse_unit, scale_by_exp10};

/// Identifies a dimension across renames. Freshly parsed spaces use `-i`, as
/// in neuroglancer, so they never collide with IDs handed out later.
pub type DimensionId = i64;

/// Labels for the integer coordinates of a categorical dimension, e.g. channel
/// names.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateArray {
    /// Set when the array came from JSON, in which case it is written back out.
    pub explicit: bool,
    /// Strictly increasing.
    pub coordinates: Vec<i64>,
    pub labels: Vec<String>,
}

impl CoordinateArray {
    /// Sorts by coordinate. When a coordinate repeats, the last label wins,
    /// matching `normalizeCoordinateArray`.
    pub fn normalized(explicit: bool, coordinates: &[i64], labels: &[String]) -> Self {
        // Keep each coordinate's last label, at the position it first appeared
        // (a JS `Map`'s semantics), then sort.
        let mut entries: Vec<(i64, String)> = Vec::with_capacity(coordinates.len());
        for (&c, l) in coordinates.iter().zip(labels) {
            match entries.iter_mut().find(|(k, _)| *k == c) {
                Some(entry) => entry.1 = l.clone(),
                None => entries.push((c, l.clone())),
            }
        }
        entries.sort_by_key(|&(c, _)| c);
        let (coordinates, labels) = entries.into_iter().unzip();
        Self {
            explicit,
            coordinates,
            labels,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateSpace {
    /// Whether the space is fully initialized, i.e. based on at least one data
    /// source. Spaces parsed from JSON start out invalid.
    pub valid: bool,
    pub names: Vec<String>,
    pub ids: Vec<DimensionId>,
    /// Time of the last user edit to each dimension; `-inf` if never edited.
    pub timestamps: Vec<f64>,
    /// Base unit per dimension (`"m"`, `"s"`, `"Hz"`, `"rad/s"` or `""`).
    pub units: Vec<String>,
    /// Size of one coordinate step, in base units (so `4nm` is `4e-9`).
    pub scales: Vec<f64>,
    pub coordinate_arrays: Vec<Option<CoordinateArray>>,
    /// The data's extent, from each source's box mapped into this space.
    pub bounding_boxes: Vec<TransformedBoundingBox>,
    /// [`compute_combined_bounds`] of `bounding_boxes`.
    pub bounds: CoordinateSpaceBounds,
}

impl CoordinateSpace {
    /// Port of `makeCoordinateSpace` with its defaults.
    pub fn new(valid: bool, names: Vec<String>, units: Vec<String>, scales: Vec<f64>) -> Self {
        let rank = names.len();
        Self {
            valid,
            ids: (0..rank as i64).map(|i| -i).collect(),
            timestamps: vec![f64::NEG_INFINITY; rank],
            coordinate_arrays: vec![None; rank],
            bounds: compute_combined_bounds(&[], rank),
            bounding_boxes: vec![],
            names,
            units,
            scales,
        }
    }

    /// Sets the bounding boxes and recomputes `bounds` from them.
    pub fn with_bounding_boxes(mut self, boxes: Vec<TransformedBoundingBox>) -> Self {
        self.bounds = compute_combined_bounds(&boxes, self.rank());
        self.bounding_boxes = boxes;
        self
    }

    pub fn empty_invalid() -> Self {
        Self::new(false, vec![], vec![], vec![])
    }

    pub fn rank(&self) -> usize {
        self.names.len()
    }

    /// Port of `coordinateSpaceFromJson`. `None` (the key was absent) gives
    /// the empty invalid space.
    pub fn from_json(json: Option<&Value>, allow_numerical_dimensions: bool) -> Result<Self> {
        let Some(json) = json else {
            return Ok(Self::empty_invalid());
        };
        let obj = json
            .as_object()
            .ok_or_else(|| anyhow!("Expected object, but received: {json}"))?;
        let names = js_object_keys(obj);
        ensure!(
            validate_dimension_names(&names, allow_numerical_dimensions),
            "Invalid dimensions: {names:?}"
        );
        let rank = names.len();
        let mut space = Self::new(false, names, vec![String::new(); rank], vec![0.0; rank]);
        for i in 0..rank {
            let name = &space.names[i];
            let member = &obj[name.as_str()];
            let parsed = (|| -> Result<_> {
                if let Some(pair) = member.as_array() {
                    // Normal unit-scale dimension.
                    let (unit, scale) = unit_and_scale_from_json(pair)?;
                    Ok((unit.to_string(), scale, None))
                } else {
                    // Coordinate array dimension.
                    let m = member
                        .as_object()
                        .ok_or_else(|| anyhow!("Expected object, but received: {member}"))?;
                    let coordinates = integer_array(
                        m.get("coordinates").context("Missing \"coordinates\"")?,
                    )
                    .context("Error parsing \"coordinates\" property")?;
                    let labels = string_array(m.get("labels").context("Missing \"labels\"")?)
                        .context("Error parsing \"labels\" property")?;
                    ensure!(
                        coordinates.len() == labels.len(),
                        "Length of coordinates array ({}) does not match length of labels array ({})",
                        coordinates.len(),
                        labels.len()
                    );
                    let array = CoordinateArray::normalized(true, &coordinates, &labels);
                    Ok((String::new(), 1.0, Some(array)))
                }
            })()
            .with_context(|| format!("Error parsing {name:?} property"))?;
            space.units[i] = parsed.0;
            space.scales[i] = parsed.1;
            space.coordinate_arrays[i] = parsed.2;
        }
        Ok(space)
    }

    /// Port of `coordinateSpaceToJson`. Rank 0 serializes to nothing.
    pub fn to_json(&self) -> Option<Value> {
        if self.rank() == 0 {
            return None;
        }
        let mut json = Map::new();
        for i in 0..self.rank() {
            let value = match &self.coordinate_arrays[i] {
                Some(array) if array.explicit => serde_json::json!({
                    "coordinates": array.coordinates,
                    "labels": array.labels,
                }),
                _ => Value::Array(vec![crate::js::number(self.scales[i]), self.units[i].clone().into()]),
            };
            json.insert(self.names[i].clone(), value);
        }
        Some(Value::Object(json))
    }
}

/// An axis-aligned box: `[lower, upper)` per dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
}

/// A box in some source's coordinates and the affine map into the space.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformedBoundingBox {
    pub bounding_box: BoundingBox,
    /// `output_rank × (input_rank + 1)`, column-major: entry `(out, in)` is at
    /// `in * output_rank + out`; the last column is the translation.
    pub transform: Vec<f64>,
}

impl TransformedBoundingBox {
    /// Port of `makeIdentityTransformedBoundingBox`.
    pub fn identity(bounding_box: BoundingBox) -> Self {
        let rank = bounding_box.lower_bounds.len();
        let mut transform = vec![0.0; rank * (rank + 1)];
        for i in 0..rank {
            transform[i * rank + i] = 1.0;
        }
        Self {
            bounding_box,
            transform,
        }
    }
}

/// Port of `CoordinateSpaceBounds`.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateSpaceBounds {
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    /// Per dimension: whether voxel centers sit at integer coordinates (rather
    /// than half-integers).
    pub voxel_center_at_integer_coordinates: Vec<bool>,
}

impl CoordinateSpaceBounds {
    /// Port of `roundCoordinateToVoxelCenter`.
    pub fn round_to_voxel_center(&self, dim: usize, coordinate: f64) -> f64 {
        if self.voxel_center_at_integer_coordinates[dim] {
            crate::js::round(coordinate)
        } else {
            coordinate.floor() + 0.5
        }
    }

    /// Port of `clampCoordinateToBounds`: to `[lower, upper - 1]`, ignoring
    /// infinite bounds.
    pub fn clamp(&self, dim: usize, mut coordinate: f64) -> f64 {
        let upper = self.upper_bounds[dim];
        if upper.is_finite() {
            coordinate = coordinate.min(upper - 1.0);
        }
        let lower = self.lower_bounds[dim];
        if lower.is_finite() {
            coordinate = coordinate.max(lower);
        }
        coordinate
    }

    /// Port of `clampAndRoundCoordinateToVoxelCenter`.
    pub fn clamp_and_round_to_voxel_center(&self, dim: usize, coordinate: f64) -> f64 {
        self.round_to_voxel_center(dim, self.clamp(dim, coordinate))
    }
}

/// Port of `getCenterBound`: the midpoint, or for a half-infinite range the
/// finite end nearest 0.
pub fn get_center_bound(lower: f64, upper: f64) -> f64 {
    let x = (lower + upper) / 2.0;
    if x.is_finite() {
        x
    } else {
        js_min(js_max(0.0, lower), upper)
    }
}

/// `Math.min`/`Math.max`: NaN if either argument is NaN (`f64::min` would
/// return the other one).
fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}
fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Port of `computeCombinedLowerUpperBound`: the extent of `bbox` along
/// `output_dim`, or `None` if no input dimension maps there.
pub fn compute_combined_lower_upper_bound(
    bbox: &TransformedBoundingBox,
    output_dim: usize,
    output_rank: usize,
) -> Option<(f64, f64)> {
    let lower_in = &bbox.bounding_box.lower_bounds;
    let upper_in = &bbox.bounding_box.upper_bounds;
    let input_rank = lower_in.len();
    let t = &bbox.transform;
    let offset = t[output_rank * input_rank + output_dim];
    let (mut lower, mut upper) = (offset, offset);
    let mut has_coefficient = false;
    for input_dim in 0..input_rank {
        let c = t[output_rank * input_dim + output_dim];
        if c == 0.0 {
            continue;
        }
        let (a, b) = (c * lower_in[input_dim], c * upper_in[input_dim]);
        lower += js_min(a, b);
        upper += js_max(a, b);
        has_coefficient = true;
    }
    has_coefficient.then_some((lower, upper))
}

const INTEGER_BOUNDS_EPSILON: f64 = 1e-3;

/// Port of `computeCombinedBounds`: the union of the boxes, snapped to
/// integers or half-integers when within 1e-3, and per dimension whether voxel
/// centers are at integers (only when every box there has half-integer bounds).
pub fn compute_combined_bounds(
    boxes: &[TransformedBoundingBox],
    output_rank: usize,
) -> CoordinateSpaceBounds {
    let mut lower_bounds = vec![f64::NEG_INFINITY; output_rank];
    let mut upper_bounds = vec![f64::INFINITY; output_rank];
    let mut half_integer_bounds = vec![0usize; output_rank];
    let mut integer_bounds = vec![0usize; output_rank];
    for bbox in boxes {
        for dim in 0..output_rank {
            let Some((mut lower, mut upper)) = compute_combined_lower_upper_bound(bbox, dim, output_rank)
            else {
                continue;
            };
            if lower.is_finite() && upper.is_finite() {
                let (lower_round, upper_round) = (crate::js::round(lower), crate::js::round(upper));
                let (lower_floor, upper_floor) = (lower.floor(), upper.floor());
                if (lower - lower_round).abs() < INTEGER_BOUNDS_EPSILON
                    && (upper - upper_round).abs() < INTEGER_BOUNDS_EPSILON
                {
                    integer_bounds[dim] += 1;
                    lower = lower_round;
                    upper = upper_round;
                } else if (lower - lower_floor - 0.5).abs() < INTEGER_BOUNDS_EPSILON
                    && (upper - upper_floor - 0.5).abs() < INTEGER_BOUNDS_EPSILON
                {
                    half_integer_bounds[dim] += 1;
                    lower = lower_floor + 0.5;
                    upper = upper_floor + 0.5;
                }
            }
            lower_bounds[dim] = if lower_bounds[dim] == f64::NEG_INFINITY {
                lower
            } else {
                js_min(lower_bounds[dim], lower)
            };
            upper_bounds[dim] = if upper_bounds[dim] == f64::INFINITY {
                upper
            } else {
                js_max(upper_bounds[dim], upper)
            };
        }
    }
    let voxel_center_at_integer_coordinates = (0..output_rank)
        .map(|i| half_integer_bounds[i] > 0 && integer_bounds[i] == 0)
        .collect();
    CoordinateSpaceBounds {
        lower_bounds,
        upper_bounds,
        voxel_center_at_integer_coordinates,
    }
}

/// `[scale, unit]`, with the unit's prefix folded into the scale.
fn unit_and_scale_from_json(pair: &[Value]) -> Result<(&'static str, f64)> {
    ensure!(pair.len() == 2, "Expected array of length 2, but received: {pair:?}");
    let scale = pair[0]
        .as_f64()
        .filter(|s| s.is_finite() && *s > 0.0)
        .ok_or_else(|| anyhow!("Expected positive number, but received: {}", pair[0]))?;
    let unit_string = pair[1]
        .as_str()
        .ok_or_else(|| anyhow!("Expected string, but received: {}", pair[1]))?;
    let unit = parse_unit(unit_string).ok_or_else(|| anyhow!("Invalid unit: {unit_string:?}"))?;
    Ok((unit.unit, scale_by_exp10(scale, unit.exponent)))
}

/// Port of `isValidDimensionName`: `[a-zA-Z][a-zA-Z_0-9]*` with an optional
/// trailing `'` (local) or `^` (channel). Non-negative integers are allowed
/// when `allow_numerical_names` is set.
pub fn is_valid_dimension_name(name: &str, allow_numerical_names: bool) -> bool {
    if allow_numerical_names && is_js_nonnegative_integer(name) {
        return true;
    }
    let body = name
        .strip_suffix('\'')
        .or_else(|| name.strip_suffix('^'))
        .unwrap_or(name);
    let mut chars = body.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Port of `validateDimensionNames`: every name valid and none repeated.
pub fn validate_dimension_names(names: &[String], allow_numerical_names: bool) -> bool {
    names.iter().enumerate().all(|(i, name)| {
        is_valid_dimension_name(name, allow_numerical_names) && !names[..i].contains(name)
    })
}

pub fn is_local_dimension(name: &str) -> bool {
    name.ends_with('\'')
}

pub fn is_channel_dimension(name: &str) -> bool {
    name.ends_with('^')
}

pub fn is_local_or_channel_dimension(name: &str) -> bool {
    is_local_dimension(name) || is_channel_dimension(name)
}

pub fn is_global_dimension(name: &str) -> bool {
    !is_local_or_channel_dimension(name)
}

/// `Number(name)` is a non-negative integer. `Number` trims whitespace and
/// accepts forms like `"1e3"` and `"0x10"`; this covers the forms dimension
/// names actually use (decimal digits) and `""`, which is `0` in JS but fails
/// the name regex anyway, so it is rejected here too.
fn is_js_nonnegative_integer(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_digit())
}

/// The order `Object.keys` returns: keys that are canonical array indices
/// (`"0"`, `"17"`, not `"07"`) first in ascending numeric order, then the rest
/// in insertion order. Dimension order comes from key order, so this matters.
pub fn js_object_keys(obj: &Map<String, Value>) -> Vec<String> {
    let is_index = |k: &str| {
        k.parse::<u32>()
            .is_ok_and(|n| n != u32::MAX && n.to_string() == k)
    };
    let mut indices: Vec<&String> = obj.keys().filter(|k| is_index(k)).collect();
    indices.sort_by_key(|k| k.parse::<u32>().unwrap());
    indices
        .into_iter()
        .chain(obj.keys().filter(|k| !is_index(k)))
        .cloned()
        .collect()
}

fn integer_array(v: &Value) -> Result<Vec<i64>> {
    let arr = v
        .as_array()
        .ok_or_else(|| anyhow!("Expected array, but received: {v}"))?;
    arr.iter()
        .map(|x| match x.as_f64() {
            Some(f) if f.fract() == 0.0 && f.is_finite() => Ok(f as i64),
            _ => bail!("Expected integer, but received: {x}"),
        })
        .collect()
}

fn string_array(v: &Value) -> Result<Vec<String>> {
    let arr = v
        .as_array()
        .ok_or_else(|| anyhow!("Expected array, but received: {v}"))?;
    arr.iter()
        .map(|x| {
            x.as_str()
                .map(str::to_owned)
                .ok_or_else(|| anyhow!("Expected string, but received: {x}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(v: Value) -> Result<CoordinateSpace> {
        CoordinateSpace::from_json(Some(&v), false)
    }

    #[test]
    fn parses_unit_scale_dimensions() {
        let space = parse(json!({"x": [4, "nm"], "y": [4, "nm"], "z": [40, "nm"]})).unwrap();
        assert!(!space.valid);
        assert_eq!(space.names, ["x", "y", "z"]);
        assert_eq!(space.units, ["m", "m", "m"]);
        assert_eq!(space.scales, [4e-9, 4e-9, 4e-8]);
        assert_eq!(space.ids, [0, -1, -2]);
    }

    #[test]
    fn round_trips_through_json() {
        let input = json!({
            "x": [4, "nm"],
            "t": [0.5, "s"],
            "c^": {"coordinates": [2, 0, 1], "labels": ["b", "r", "g"]},
        });
        let out = parse(input).unwrap().to_json().unwrap();
        // The prefix is folded into the scale, so "nm" comes back as "m".
        assert_eq!(
            out,
            json!({
                "x": [4e-9, "m"],
                "t": [0.5, "s"],
                "c^": {"coordinates": [0, 1, 2], "labels": ["r", "g", "b"]},
            })
        );
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse(json!({"x": [0, "nm"]})).is_err());
        assert!(parse(json!({"x": [1, "parsec"]})).is_err());
        assert!(parse(json!({"1x": [1, "m"]})).is_err());
        assert!(parse(json!({"x": {"coordinates": [0], "labels": []}})).is_err());
        assert!(parse(json!([1, 2])).is_err());
    }

    #[test]
    fn numerical_names_sort_first_like_object_keys() {
        let v = json!({"b": [1, ""], "10": [1, ""], "2": [1, ""], "a": [1, ""]});
        let space = CoordinateSpace::from_json(Some(&v), true).unwrap();
        assert_eq!(space.names, ["2", "10", "b", "a"]);
        assert!(CoordinateSpace::from_json(Some(&v), false).is_err());
    }

    #[test]
    fn dimension_name_rules() {
        assert!(is_valid_dimension_name("x", false));
        assert!(is_valid_dimension_name("c'", false));
        assert!(is_valid_dimension_name("c^", false));
        assert!(is_valid_dimension_name("a_1", false));
        assert!(!is_valid_dimension_name("_a", false));
        assert!(!is_valid_dimension_name("x''", false));
        assert!(!is_valid_dimension_name("", false));
        assert!(is_valid_dimension_name("3", true));
        assert!(!validate_dimension_names(&["x".into(), "x".into()], false));
    }

    #[test]
    fn absent_space_is_empty_and_invalid() {
        let space = CoordinateSpace::from_json(None, false).unwrap();
        assert_eq!(space, CoordinateSpace::empty_invalid());
        assert_eq!(space.to_json(), None);
    }
}
