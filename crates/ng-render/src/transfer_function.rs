//! Transfer functions (`widget/transfer_function.ts`): control points, each a
//! data value and an RGBA colour (bytes), sorted over the data; the lookup
//! table interpolated between them, which shaders sample.

use ng_core::DataType;
use serde_json::Value;

use crate::image::{default_data_type_range, Interval};

/// A data value: a number, or (uint64) exact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataValue {
    Number(f64),
    Uint64(u64),
}

impl DataValue {
    pub fn as_f64(self) -> f64 {
        match self {
            DataValue::Number(v) => v,
            DataValue::Uint64(v) => v as f64,
        }
    }

    /// As JSON: uint64 as a decimal string.
    pub fn json(self) -> Value {
        match self {
            DataValue::Uint64(v) => Value::String(v.to_string()),
            DataValue::Number(v) if v.fract() == 0.0 && v.abs() < 9.0e15 => Value::from(v as i64),
            DataValue::Number(v) => serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number),
        }
    }

    /// `computeInvlerp`: where the value falls in `range`.
    pub fn invlerp(self, range: Interval) -> f64 {
        match (self, range) {
            (DataValue::Uint64(v), Interval::Uint64([lo, hi])) => (v as i128 - lo as i128) as f64 / (hi as i128 - lo as i128) as f64,
            (v, r) => {
                let [lo, hi] = r.as_f64();
                (v.as_f64() - lo) / (hi - lo)
            }
        }
    }

    /// `computeLerp` over `range` at `x`: integer types rounded and clamped.
    pub fn lerp(range: Interval, t: DataType, x: f64) -> DataValue {
        match range {
            Interval::Uint64([lo, hi]) => {
                let span = (hi as i128 - lo as i128) as f64;
                let v = if x >= 1.0 { hi as i128 + ng_core::js::round(span * (x - 1.0)) as i128 } else { lo as i128 + ng_core::js::round(span * x) as i128 };
                DataValue::Uint64(v.clamp(0, u64::MAX as i128) as u64)
            }
            Interval::Number([lo, hi]) => {
                let mut v = lo * (1.0 - x) + hi * x;
                if t != DataType::Float32 {
                    let [a, b] = default_data_type_range(t).as_f64();
                    v = ng_core::js::round(v).max(a).min(b);
                }
                DataValue::Number(v)
            }
        }
    }
}

/// `ControlPoint`: a data value and its colour, RGBA bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlPoint {
    pub input: DataValue,
    pub color: [u8; 4],
}

impl ControlPoint {
    /// `transferFunctionIndex`.
    fn index(&self, range: Interval, size: usize) -> i64 {
        (self.input.invlerp(range) * (size as f64 - 1.0)).floor() as i64
    }

    /// `interpolateColor`: each channel by `computeLerp` over uint8.
    fn interpolate(&self, other: &ControlPoint, t: f64) -> [u8; 4] {
        std::array::from_fn(|i| {
            let v = self.color[i] as f64 * (1.0 - t) + other.color[i] as f64 * t;
            ng_core::js::round(v).clamp(0.0, 255.0) as u8
        })
    }
}

/// `SortedControlPoints`: the points in order of their data value, and the
/// range they span (the type's whole range for none; from a single point to
/// the type's end).
#[derive(Debug, Clone, PartialEq)]
pub struct SortedControlPoints {
    pub points: Vec<ControlPoint>,
    pub range: Interval,
    pub data_type: DataType,
}

impl SortedControlPoints {
    pub fn new(points: Vec<ControlPoint>, data_type: DataType) -> Self {
        let mut s = SortedControlPoints { points, range: default_data_type_range(data_type), data_type };
        s.sort_and_compute_range();
        s
    }

    /// `addPoint`: a point at an existing value is added beside it.
    pub fn add(&mut self, point: ControlPoint) {
        self.points.push(point);
        self.sort_and_compute_range();
    }

    /// `removePoint`.
    pub fn remove(&mut self, index: usize) {
        if index < self.points.len() {
            self.points.remove(index);
            self.compute_range();
        }
    }

    /// `updatePoint`: the point's index once sorted.
    pub fn update(&mut self, index: usize, point: ControlPoint) -> Option<usize> {
        *self.points.get_mut(index)? = point;
        self.sort_and_compute_range();
        self.points.iter().position(|p| *p == point)
    }

    /// `findNearestControlPointIndexByNormalizedInput`.
    pub fn nearest(&self, normalized: f64) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (i, p) in self.points.iter().enumerate() {
            let d = (p.input.invlerp(self.range) - normalized).abs();
            if best.is_none_or(|(_, b)| d < b) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }

    fn sort_and_compute_range(&mut self) {
        let range = self.range;
        // A stable sort, as `Array.prototype.sort` is.
        self.points.sort_by(|a, b| a.input.invlerp(range).partial_cmp(&b.input.invlerp(range)).unwrap_or(std::cmp::Ordering::Equal));
        self.compute_range();
    }

    fn compute_range(&mut self) {
        let t = self.data_type;
        let default = default_data_type_range(t);
        self.range = match self.points.as_slice() {
            [] => default,
            [only] => match (only.input, default) {
                (DataValue::Uint64(v), Interval::Uint64([_, hi])) => Interval::Uint64([v, hi]),
                (DataValue::Uint64(v), Interval::Number([_, hi])) => Interval::Uint64([v, hi as u64]),
                (DataValue::Number(v), d) => {
                    let mut end = d.as_f64()[1];
                    if t == DataType::Float32 && end <= v {
                        end = v + 1.0;
                    }
                    Interval::Number([v, end])
                }
            },
            [first, .., last] => match (first.input, last.input) {
                (DataValue::Uint64(a), DataValue::Uint64(b)) => Interval::Uint64([a, b]),
                (a, b) => Interval::Number([a.as_f64(), b.as_f64()]),
            },
        };
        let [lo, hi] = self.range.as_f64();
        let equal = match self.range {
            Interval::Uint64([a, b]) => a == b,
            Interval::Number(_) => lo == hi,
        };
        if equal {
            self.range = default;
        }
    }

    /// `LookupTable.updateFromControlPoints` over the points' range: `size`
    /// colours, transparent before the first point, the last one's after.
    pub fn lookup_table(&self, size: usize) -> Vec<[u8; 4]> {
        let mut out = vec![[0u8; 4]; size];
        let points = &self.points;
        if points.is_empty() {
            return out;
        }
        let range = self.range;
        let first = points[0].index(range, size);
        let mut k = 0;
        let mut i = first;
        while i < size as i64 {
            let current = &points[k];
            let color = if k == points.len() - 1 {
                current.color
            } else {
                let next = &points[k + 1];
                let (a, b) = (current.index(range, size), next.index(range, size));
                let color = current.interpolate(next, (i - a) as f64 / (b - a) as f64);
                if i >= b {
                    k += 1;
                }
                color
            };
            if i >= 0 {
                out[i as usize] = color;
            }
            i += 1;
        }
        out
    }
}

/// `defaultTransferFunctionSizes`.
pub fn lookup_table_size(t: DataType) -> usize {
    match t {
        DataType::Uint8 | DataType::Int8 => 256,
        _ => 8192,
    }
}

/// `TransferFunctionParameters`.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferFunction {
    pub points: SortedControlPoints,
    pub window: Interval,
    pub channel: Vec<u32>,
    pub default_color: [f32; 3],
}

/// `parseDataTypeValue`: a number in the type's range (integers whole), or
/// for uint64 a decimal string or number.
fn parse_value(t: DataType, x: &Value) -> Result<DataValue, String> {
    if t == DataType::Uint64 {
        let s = match x {
            Value::String(s) => s.clone(),
            v => v.to_string(),
        };
        return s.parse::<u64>().map(DataValue::Uint64).map_err(|_| format!("Invalid uint64 value: {}", serde_json::to_string(&s).unwrap()));
    }
    let v = x.as_f64().ok_or_else(|| format!("Invalid {} value: {x}", t.name()))?;
    if t != DataType::Float32 {
        let [lo, hi] = default_data_type_range(t).as_f64();
        if v.fract() != 0.0 || v < lo || v > hi {
            return Err(format!("Invalid {} value: {}", t.name(), serde_json::to_string(&v.to_string()).unwrap()));
        }
    }
    Ok(DataValue::Number(v))
}

/// `parseTransferFunctionControlPoints`: `[value, "#rrggbb", opacity]`s.
pub fn parse_control_points(v: &Value, t: DataType) -> Result<SortedControlPoints, String> {
    let items = v.as_array().ok_or_else(|| format!("Expected array, but received: {v}"))?;
    let mut points = vec![];
    for x in items {
        let a = x.as_array().filter(|a| a.len() == 3);
        let allowed = |v: &Value| if t == DataType::Uint64 { v.is_string() || v.is_number() } else { v.is_number() };
        let Some(a) = a.filter(|a| allowed(&a[0]) && a[1].is_string() && a[2].is_number()) else {
            return Err(format!("Expected array of length 3 (x, \"#RRGGBB\", A), but received: {x}"));
        };
        let input = parse_value(t, &a[0])?;
        let hex = a[1].as_str().unwrap();
        if hex.len() != 7 || !hex.starts_with('#') {
            return Err(format!("Expected #RRGGBB, but received: {}", a[1]));
        }
        let opacity = a[2].as_f64().unwrap();
        if !(0.0..=1.0).contains(&opacity) {
            return Err(format!("Expected opacity in range [0, 1], but received: {}", a[2]));
        }
        let color = crate::user_shader::parse_color(hex).ok_or_else(|| format!("Invalid color: {hex}"))?;
        let byte = |f: f32| ng_core::js::round(f as f64 * 255.0).clamp(0.0, 255.0) as u8;
        points.push(ControlPoint { input, color: [byte(color[0]), byte(color[1]), byte(color[2]), byte(opacity as f32)] });
    }
    Ok(SortedControlPoints::new(points, t))
}

/// `controlPointsToJson`.
pub fn control_points_json(points: &[ControlPoint]) -> Value {
    Value::Array(
        points
            .iter()
            .map(|p| {
                let hex = format!("#{:02x}{:02x}{:02x}", p.color[0], p.color[1], p.color[2]);
                serde_json::json!([p.input.json(), hex, ng_core::js::number(p.color[3] as f64 / 255.0)])
            })
            .collect(),
    )
}

/// `serializeColor` of an opaque colour.
pub fn serialize_color(c: [f32; 3]) -> String {
    let byte = |f: f32| ng_core::js::round(f as f64 * 255.0).clamp(0.0, 255.0) as u8;
    format!("#{:02x}{:02x}{:02x}", byte(c[0]), byte(c[1]), byte(c[2]))
}

fn interval_json(i: Interval) -> Value {
    match i {
        Interval::Uint64([a, b]) => serde_json::json!([a.to_string(), b.to_string()]),
        Interval::Number([a, b]) => serde_json::json!([DataValue::Number(a).json(), DataValue::Number(b).json()]),
    }
}

impl TransferFunction {
    /// `parseTransferFunctionParameters`: what `obj` sets over `default`.
    pub fn from_json(obj: Option<&Value>, t: DataType, default: &TransferFunction) -> TransferFunction {
        let Some(o) = obj.and_then(Value::as_object) else { return default.clone() };
        let points = o.get("controlPoints").and_then(|x| parse_control_points(x, t).ok()).unwrap_or_else(|| default.points.clone());
        let window = o.get("window").and_then(|x| crate::user_shader::parse_interval(x, t)).unwrap_or(default.window);
        let default_color = o.get("defaultColor").and_then(Value::as_str).and_then(crate::user_shader::parse_color).unwrap_or(default.default_color);
        let channel = o.get("channel").and_then(|c| crate::user_shader::parse_channel(c, default.channel.len()).ok()).unwrap_or_else(|| default.channel.clone());
        TransferFunction { points, window, channel, default_color }
    }

    /// `TrackableTransferFunctionParameters.toJSON`: what differs from
    /// `default`, or nothing.
    pub fn to_json(&self, default: &TransferFunction) -> Option<Value> {
        let mut o = serde_json::Map::new();
        if serialize_color(self.default_color) != serialize_color(default.default_color) {
            o.insert("defaultColor".into(), serialize_color(self.default_color).into());
        }
        if self.window != default.window {
            o.insert("window".into(), interval_json(self.window));
        }
        if self.points.points != default.points.points {
            o.insert("controlPoints".into(), control_points_json(&self.points.points));
        }
        if self.channel != default.channel {
            o.insert("channel".into(), self.channel.iter().map(|&c| Value::from(c)).collect());
        }
        (!o.is_empty()).then_some(Value::Object(o))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn control_points_sort_and_span_as_neuroglancer() {
        let t = DataType::Uint8;
        let p = parse_control_points(&json!([[200, "#ff0000", 1], [50, "#00ff00", 0.5]]), t).unwrap();
        assert_eq!(p.points.iter().map(|p| p.input).collect::<Vec<_>>(), vec![DataValue::Number(50.0), DataValue::Number(200.0)]);
        assert_eq!(p.points[0].color, [0, 255, 0, 128]);
        assert_eq!(p.range, Interval::Number([50.0, 200.0]));
        // One point: to the type's end; none: the type's range.
        assert_eq!(parse_control_points(&json!([[10, "#ffffff", 1]]), t).unwrap().range, Interval::Number([10.0, 255.0]));
        assert_eq!(parse_control_points(&json!([]), t).unwrap().range, Interval::Number([0.0, 255.0]));
        assert_eq!(parse_control_points(&json!([[1.5, "#ffffff", 1]]), DataType::Float32).unwrap().range, Interval::Number([1.5, 2.5]));
        assert!(parse_control_points(&json!([[10, "#fff", 1]]), t).unwrap_err().starts_with("Expected #RRGGBB"));
        assert!(parse_control_points(&json!([[10, "#ffffff", 2]]), t).unwrap_err().starts_with("Expected opacity"));
        assert!(parse_control_points(&json!([[300, "#ffffff", 1]]), t).is_err());
        assert_eq!(control_points_json(&p.points), json!([[50, "#00ff00", 0.5019607843137255], [200, "#ff0000", 1]]));
    }

    #[test]
    fn lookup_tables_interpolate_as_neuroglancer() {
        let t = DataType::Uint8;
        let p = parse_control_points(&json!([[0, "#000000", 0], [255, "#ffffff", 1]]), t).unwrap();
        let lut = p.lookup_table(256);
        assert_eq!(lut[0], [0, 0, 0, 0]);
        assert_eq!(lut[128], [128, 128, 128, 128]);
        assert_eq!(lut[255], [255, 255, 255, 255]);
        // Before the first point transparent, after the last its colour.
        let p = SortedControlPoints {
            points: vec![ControlPoint { input: DataValue::Number(64.0), color: [10, 20, 30, 40] }, ControlPoint { input: DataValue::Number(128.0), color: [50, 60, 70, 80] }],
            range: Interval::Number([0.0, 255.0]),
            data_type: t,
        };
        let lut = p.lookup_table(256);
        assert_eq!(lut[63], [0; 4]);
        assert_eq!(lut[64], [10, 20, 30, 40]);
        assert_eq!(lut[200], [50, 60, 70, 80]);
    }
}
