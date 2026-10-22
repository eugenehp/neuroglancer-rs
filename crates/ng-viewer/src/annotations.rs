//! Annotation layers' local annotations (`annotation/index.ts`
//! `restoreAnnotation`, `LocalAnnotationSource`): each a point, line,
//! axis-aligned bounding box, ellipsoid or polyline in the source's
//! coordinates (float32, as neuroglancer keeps them), with its ID and
//! description.

use serde_json::Value;

pub use ng_datasource::annotation::PropertyType;

/// `AnnotationType` and its geometry.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    Point(Vec<f64>),
    Line(Vec<f64>, Vec<f64>),
    AxisAlignedBoundingBox(Vec<f64>, Vec<f64>),
    Ellipsoid { center: Vec<f64>, radii: Vec<f64> },
    Polyline(Vec<Vec<f64>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    pub id: String,
    pub description: Option<String>,
    pub geometry: Geometry,
    /// Its property values, in the source's property order.
    pub properties: Vec<f64>,
}

/// A point of `rank` finite numbers, as float32.
fn point(v: Option<&Value>, rank: usize) -> Option<Vec<f64>> {
    let a = v?.as_array().filter(|a| a.len() == rank)?;
    a.iter().map(|x| x.as_f64().filter(|f| f.is_finite()).map(|f| f as f32 as f64)).collect()
}

/// A local annotation property (`annotationProperties`): its id, type and
/// default value (as [`Annotation::properties`] hold values).
#[derive(Debug, Clone, PartialEq)]
pub struct PropertySpec {
    pub id: String,
    pub kind: PropertyType,
    pub default: f64,
    pub description: Option<String>,
}

/// A property value from JSON (`deserializeJson`): colours as CSS colours,
/// packed little-endian; numbers as they are.
fn property_value(kind: PropertyType, v: &Value) -> Option<f64> {
    match kind {
        PropertyType::Rgb => {
            let c = crate::state::parse_color(v.as_str()?)?;
            let b = |x: f32| (x * 255.0).round() as u32;
            Some((b(c[0]) | b(c[1]) << 8 | b(c[2]) << 16) as f64)
        }
        PropertyType::Rgba => {
            let s = v.as_str()?;
            let (rgb, a) = match s.strip_prefix('#').filter(|h| h.len() == 8) {
                Some(h) => (format!("#{}", &h[..6]), u32::from_str_radix(&h[6..], 16).ok()?),
                None => (s.to_owned(), 255),
            };
            let c = crate::state::parse_color(&rgb)?;
            let b = |x: f32| (x * 255.0).round() as u32;
            Some((b(c[0]) | b(c[1]) << 8 | b(c[2]) << 16 | a << 24) as f64)
        }
        _ => v.as_f64().filter(|f| f.is_finite()),
    }
}

/// `parseAnnotationPropertySpecs` of a layer's `annotationProperties`;
/// none if malformed.
pub fn parse_property_specs(json: Option<&Value>) -> Vec<PropertySpec> {
    let Some(a) = json.and_then(Value::as_array) else { return vec![] };
    a.iter()
        .map(|p| {
            let id = p.get("id")?.as_str()?.to_owned();
            let kind = PropertyType::parse(p.get("type")?.as_str()?)?;
            let default = match p.get("default") {
                Some(d) => property_value(kind, d)?,
                None => 0.0,
            };
            Some(PropertySpec { id, kind, default, description: p.get("description").and_then(Value::as_str).map(str::to_owned) })
        })
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

/// `restoreAnnotation` for one annotation of `rank` dimensions; `None` if
/// it is malformed.
pub fn parse_annotation(v: &Value, rank: usize) -> Option<Annotation> {
    parse_annotation_with(v, rank, &[])
}

/// [`parse_annotation`] with the layer's properties: its `props` (each
/// property's default where absent).
pub fn parse_annotation_with(v: &Value, rank: usize, specs: &[PropertySpec]) -> Option<Annotation> {
    let o = v.as_object()?;
    let geometry = match o.get("type")?.as_str()? {
        "point" => Geometry::Point(point(o.get("point"), rank)?),
        "line" => Geometry::Line(point(o.get("pointA"), rank)?, point(o.get("pointB"), rank)?),
        "axis_aligned_bounding_box" => Geometry::AxisAlignedBoundingBox(point(o.get("pointA"), rank)?, point(o.get("pointB"), rank)?),
        "ellipsoid" => Geometry::Ellipsoid { center: point(o.get("center"), rank)?, radii: point(o.get("radii"), rank)? },
        "polyline" => Geometry::Polyline(o.get("points")?.as_array()?.iter().map(|p| point(Some(p), rank)).collect::<Option<Vec<_>>>()?),
        _ => return None,
    };
    Some(Annotation {
        id: o.get("id").and_then(Value::as_str).unwrap_or_default().to_owned(),
        description: o.get("description").and_then(Value::as_str).map(str::to_owned),
        geometry,
        properties: match o.get("props") {
            None => specs.iter().map(|p| p.default).collect(),
            Some(props) => {
                let a = props.as_array().filter(|a| a.len() == specs.len())?;
                a.iter().zip(specs).map(|(v, p)| property_value(p.kind, v)).collect::<Option<Vec<_>>>()?
            }
        },
    })
}

/// `formatAnnotationPropertyValue`: colours as CSS, `float32` to 6
/// significant digits, other numbers as they are.
pub fn format_property_value(kind: ng_datasource::annotation::PropertyType, value: f64) -> String {
    use ng_datasource::annotation::PropertyType as P;
    let channel = |k: u32| ((value as u32) >> (8 * k) & 0xff) as f32 / 255.0;
    match kind {
        P::Rgb => crate::state::serialize_color([channel(0), channel(1), channel(2)]),
        P::Rgba if channel(3) == 1.0 => crate::state::serialize_color([channel(0), channel(1), channel(2)]),
        P::Rgba => format!("rgba({}, {}, {}, {})", (value as u32) & 0xff, (value as u32) >> 8 & 0xff, (value as u32) >> 16 & 0xff, channel(3)),
        P::Float32 => ng_core::js::to_precision(value, 6),
        _ => ng_core::js::number_to_string(value),
    }
}

/// A precomputed source's annotation: its ID as a decimal string, as
/// neuroglancer keys them.
pub fn from_precomputed(t: ng_datasource::annotation::AnnotationType, a: &ng_datasource::annotation::PrecomputedAnnotation) -> Annotation {
    use ng_datasource::annotation::AnnotationType as T;
    let v = |k: usize| a.vectors.get(k).cloned().unwrap_or_default();
    let geometry = match t {
        T::Point => Geometry::Point(v(0)),
        T::Line => Geometry::Line(v(0), v(1)),
        T::AxisAlignedBoundingBox => Geometry::AxisAlignedBoundingBox(v(0), v(1)),
        T::Ellipsoid => Geometry::Ellipsoid { center: v(0), radii: v(1) },
        T::Polyline => Geometry::Polyline(a.vectors.clone()),
    };
    Annotation { id: a.id.to_string(), description: None, geometry, properties: a.properties.clone() }
}

/// `LocalAnnotationSource.restoreState`: the layer's `annotations`
/// (malformed ones left out, as neuroglancer ignores the errors).
pub fn parse_annotations(json: Option<&Value>, rank: usize) -> Vec<Annotation> {
    parse_annotations_with(json, rank, &[])
}

/// [`parse_annotations`] with the layer's properties.
pub fn parse_annotations_with(json: Option<&Value>, rank: usize, specs: &[PropertySpec]) -> Vec<Annotation> {
    json.and_then(Value::as_array).map_or(vec![], |a| a.iter().filter_map(|v| parse_annotation_with(v, rank, specs)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn annotations_parse_as_neuroglancer() {
        let a = parse_annotations(
            Some(&json!([
                { "type": "point", "point": [1, 2.1, 3], "id": "a" },
                { "type": "line", "pointA": [0, 0, 0], "pointB": [1, 1, 1], "id": "b", "description": "d" },
                { "type": "axis_aligned_bounding_box", "pointA": [0, 0, 0], "pointB": [1, 1, 1], "id": "c" },
                { "type": "ellipsoid", "center": [0, 0, 0], "radii": [1, 2, 3], "id": "e" },
                { "type": "point", "point": [1, 2], "id": "short" },
                { "type": "nothing", "id": "x" }
            ])),
            3,
        );
        assert_eq!(a.len(), 4);
        assert_eq!(a[0].geometry, Geometry::Point(vec![1.0, 2.1f32 as f64, 3.0]));
        assert_eq!(a[1].description.as_deref(), Some("d"));
        assert!(matches!(a[3].geometry, Geometry::Ellipsoid { .. }));
    }
}
