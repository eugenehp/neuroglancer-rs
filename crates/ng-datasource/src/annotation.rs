//! Precomputed annotation collections (`neuroglancer_annotations_v1`;
//! `datasource/precomputed/annotations.md`, `AnnotationMetadata`,
//! `parseAnnotations`): the `info` (the coordinate space and its bounds,
//! the geometry type, properties, relationships, the id index and the
//! spatial index levels) and the multiple-annotation encoding the spatial
//! index cells hold.

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use crate::precomputed::ShardingParameters;

/// `AnnotationType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationType {
    Point,
    Line,
    AxisAlignedBoundingBox,
    Ellipsoid,
    Polyline,
}

impl AnnotationType {
    fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_uppercase().as_str() {
            "POINT" => Self::Point,
            "LINE" => Self::Line,
            "AXIS_ALIGNED_BOUNDING_BOX" => Self::AxisAlignedBoundingBox,
            "ELLIPSOID" => Self::Ellipsoid,
            "POLYLINE" => Self::Polyline,
            _ => return None,
        })
    }

    /// How many `rank`-vectors the geometry holds (polylines: per point).
    fn vectors(self) -> usize {
        match self {
            Self::Point | Self::Polyline => 1,
            _ => 2,
        }
    }
}

/// An annotation property's type (`AnnotationPropertySpec.type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyType {
    Rgb,
    Rgba,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
}

impl PropertyType {
    /// From its JSON name (`rgb`, `uint8`, …).
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "rgb" => Self::Rgb,
            "rgba" => Self::Rgba,
            "uint8" => Self::Uint8,
            "int8" => Self::Int8,
            "uint16" => Self::Uint16,
            "int16" => Self::Int16,
            "uint32" => Self::Uint32,
            "int32" => Self::Int32,
            "float32" => Self::Float32,
            _ => return None,
        })
    }

    /// Its JSON name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rgb => "rgb",
            Self::Rgba => "rgba",
            Self::Uint8 => "uint8",
            Self::Int8 => "int8",
            Self::Uint16 => "uint16",
            Self::Int16 => "int16",
            Self::Uint32 => "uint32",
            Self::Int32 => "int32",
            Self::Float32 => "float32",
        }
    }

    /// `serializedBytes`, `alignment`.
    pub fn bytes(self) -> usize {
        match self {
            Self::Rgb => 3,
            Self::Rgba | Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint16 | Self::Int16 => 2,
            Self::Uint8 | Self::Int8 => 1,
        }
    }

    fn alignment(self) -> usize {
        match self {
            Self::Rgb | Self::Rgba | Self::Uint8 | Self::Int8 => 1,
            Self::Uint16 | Self::Int16 => 2,
            _ => 4,
        }
    }
}

/// A property (`id`, `type`, optional description).
#[derive(Debug, Clone, PartialEq)]
pub struct PropertySpec {
    pub id: String,
    pub kind: PropertyType,
    pub description: Option<String>,
}

/// An index directory: its location and sharding.
#[derive(Debug, Clone)]
pub struct IndexSpec {
    pub location: ng_kvstore::KvStoreLocation,
    pub sharding: Option<ShardingParameters>,
}

/// A spatial index level: its directory, grid shape, cell size (in the
/// coordinate space's units) and per-cell limit.
#[derive(Debug, Clone)]
pub struct SpatialLevel {
    pub index: IndexSpec,
    pub grid_shape: Vec<u64>,
    pub chunk_size: Vec<f64>,
    pub limit: u64,
}

/// `AnnotationMetadata`.
#[derive(Debug, Clone)]
pub struct AnnotationInfo {
    pub annotation_type: AnnotationType,
    pub rank: usize,
    pub names: Vec<String>,
    pub units: Vec<String>,
    pub scales: Vec<f64>,
    pub lower_bound: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub properties: Vec<PropertySpec>,
    pub relationships: Vec<(String, IndexSpec)>,
    pub by_id: Option<IndexSpec>,
    /// Coarsest first, as the info lists them.
    pub spatial: Vec<SpatialLevel>,
}

/// One decoded annotation: its id, geometry (float32) and property values
/// (as numbers; colours packed little-endian).
#[derive(Debug, Clone, PartialEq)]
pub struct PrecomputedAnnotation {
    pub id: u64,
    /// Point: one vector; line and box: two; ellipsoid: centre, radii;
    /// polyline: its points.
    pub vectors: Vec<Vec<f64>>,
    pub properties: Vec<f64>,
    /// Each relationship's segments (`by_id` only).
    pub related_segments: Vec<Vec<u64>>,
}

fn index_spec(dir: &ng_kvstore::KvStoreLocation, v: &Value) -> Result<IndexSpec> {
    let key = v.get("key").and_then(Value::as_str).context("key")?;
    Ok(IndexSpec { location: dir.child(key).as_directory(), sharding: crate::precomputed::parse_sharding(v.get("sharding"))? })
}

/// `AnnotationMetadata`'s parsing of `info` in directory `dir`.
pub fn parse_info(dir: &ng_kvstore::KvStoreLocation, v: &Value) -> Result<AnnotationInfo> {
    let o = v.as_object().context("annotation info is not an object")?;
    let dims = o.get("dimensions").and_then(Value::as_object).context("dimensions")?;
    let mut names = vec![];
    let mut units = vec![];
    let mut scales = vec![];
    for (name, d) in dims {
        let a = d.as_array().filter(|a| a.len() == 2).with_context(|| format!("dimension {name}"))?;
        names.push(name.clone());
        scales.push(a[0].as_f64().context("scale")?);
        units.push(a[1].as_str().context("unit")?.to_owned());
    }
    let rank = names.len();
    let bound = |k: &str| -> Result<Vec<f64>> {
        let a = o.get(k).and_then(Value::as_array).filter(|a| a.len() == rank).with_context(|| k.to_owned())?;
        a.iter().map(|x| x.as_f64().filter(|f| f.is_finite()).with_context(|| k.to_owned())).collect()
    };
    let (lower_bound, upper_bound) = (bound("lower_bound")?, bound("upper_bound")?);
    let t = o.get("annotation_type").and_then(Value::as_str).context("annotation_type")?;
    let annotation_type = AnnotationType::parse(t).with_context(|| format!("Invalid enum value: {t:?}"))?;
    let mut properties = vec![];
    for p in o.get("properties").and_then(Value::as_array).into_iter().flatten() {
        let id = p.get("id").and_then(Value::as_str).context("property id")?.to_owned();
        let k = p.get("type").and_then(Value::as_str).context("property type")?;
        properties.push(PropertySpec { id, kind: PropertyType::parse(k).with_context(|| format!("Invalid property type: {k:?}"))?, description: p.get("description").and_then(Value::as_str).map(str::to_owned) });
    }
    let mut relationships = vec![];
    for r in o.get("relationships").and_then(Value::as_array).into_iter().flatten() {
        relationships.push((r.get("id").and_then(Value::as_str).context("relationship id")?.to_owned(), index_spec(dir, r)?));
    }
    let by_id = o.get("by_id").map(|b| index_spec(dir, b)).transpose()?;
    let mut spatial = vec![];
    for l in o.get("spatial").and_then(Value::as_array).into_iter().flatten() {
        let grid_shape: Vec<u64> = l.get("grid_shape").and_then(Value::as_array).filter(|a| a.len() == rank).context("grid_shape")?.iter().map(|x| x.as_u64().filter(|&n| n > 0).context("grid_shape")).collect::<Result<_>>()?;
        let chunk_size: Vec<f64> = l.get("chunk_size").and_then(Value::as_array).filter(|a| a.len() == rank).context("chunk_size")?.iter().map(|x| x.as_f64().filter(|&f| f > 0.0 && f.is_finite()).context("chunk_size")).collect::<Result<_>>()?;
        let limit = l.get("limit").and_then(Value::as_u64).filter(|&n| n > 0).context("limit")?;
        spatial.push(SpatialLevel { index: index_spec(dir, l)?, grid_shape, chunk_size, limit });
    }
    Ok(AnnotationInfo { annotation_type, rank, names, units, scales, lower_bound, upper_bound, properties, relationships, by_id, spatial })
}

impl AnnotationInfo {
    /// `getPropertyOffsets`: each property's byte offset after the
    /// geometry (sorted by alignment, largest first), and the record's
    /// padded size beyond the geometry.
    fn property_layout(&self, geometry_bytes: usize) -> (Vec<usize>, usize) {
        let mut order: Vec<usize> = (0..self.properties.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.properties[i].kind.alignment()));
        let mut offsets = vec![0; self.properties.len()];
        let mut offset = geometry_bytes;
        for i in order {
            let k = self.properties[i].kind;
            offset += (k.alignment() - offset % k.alignment()) % k.alignment();
            offsets[i] = offset;
            offset += k.bytes();
        }
        offset += (4 - offset % 4) % 4;
        (offsets, offset)
    }
}

/// The multiple-annotation encoding (`parseAnnotations`): the count, each
/// annotation's geometry and properties, then the ids.
pub fn decode_annotations(info: &AnnotationInfo, bytes: &[u8]) -> Result<Vec<PrecomputedAnnotation>> {
    ensure!(bytes.len() >= 8, "Expected at least 8 bytes");
    let u32_at = |o: usize| -> Result<u32> { Ok(u32::from_le_bytes(bytes.get(o..o + 4).context("annotation data truncated")?.try_into().unwrap())) };
    ensure!(u32_at(4)? == 0, "Annotation count too high");
    let count = u32_at(0)? as usize;
    let mut offset = 8;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let (a, end) = decode_record(info, bytes, offset)?;
        out.push(a);
        offset = end;
    }
    for a in &mut out {
        let b = bytes.get(offset..offset + 8).context("annotation ids truncated")?;
        a.id = u64::from_le_bytes(b.try_into().unwrap());
        offset += 8;
    }
    if offset != bytes.len() {
        bail!("Expected {offset} bytes, but received: {} bytes", bytes.len());
    }
    Ok(out)
}

/// One annotation's geometry and properties at `offset` (its id unset);
/// where the next begins.
fn decode_record(info: &AnnotationInfo, bytes: &[u8], mut offset: usize) -> Result<(PrecomputedAnnotation, usize)> {
    let u32_at = |o: usize| -> Result<u32> { Ok(u32::from_le_bytes(bytes.get(o..o + 4).context("annotation data truncated")?.try_into().unwrap())) };
    let rank = info.rank;
    let points = if info.annotation_type == AnnotationType::Polyline {
        let n = (u32_at(offset)? & 0x7fff_ffff) as usize;
        offset += 4;
        n
    } else {
        info.annotation_type.vectors()
    };
    let geometry_start = offset;
    let mut vectors = Vec::with_capacity(points);
    for _ in 0..points {
        let v: Vec<f64> = (0..rank).map(|k| Ok(f32::from_le_bytes(bytes.get(offset + 4 * k..offset + 4 * k + 4).context("annotation data truncated")?.try_into().unwrap()) as f64)).collect::<Result<_>>()?;
        offset += 4 * rank;
        vectors.push(v);
    }
    let geometry_bytes = offset - geometry_start + if info.annotation_type == AnnotationType::Polyline { 4 } else { 0 };
    let (offsets, size) = info.property_layout(geometry_bytes);
    let record = geometry_start - if info.annotation_type == AnnotationType::Polyline { 4 } else { 0 };
    let mut properties = Vec::with_capacity(info.properties.len());
    for (p, &po) in info.properties.iter().zip(&offsets) {
        let at = record + po;
        let b = bytes.get(at..at + p.kind.bytes()).context("annotation data truncated")?;
        properties.push(match p.kind {
            PropertyType::Rgb => (b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16) as f64,
            PropertyType::Rgba | PropertyType::Uint32 => u32::from_le_bytes(b.try_into().unwrap()) as f64,
            PropertyType::Int32 => i32::from_le_bytes(b.try_into().unwrap()) as f64,
            PropertyType::Float32 => f32::from_le_bytes(b.try_into().unwrap()) as f64,
            PropertyType::Uint16 => u16::from_le_bytes(b.try_into().unwrap()) as f64,
            PropertyType::Int16 => i16::from_le_bytes(b.try_into().unwrap()) as f64,
            PropertyType::Uint8 => b[0] as f64,
            PropertyType::Int8 => b[0] as i8 as f64,
        });
    }
    Ok((PrecomputedAnnotation { id: 0, vectors, properties, related_segments: vec![] }, record + size))
}

/// The single-annotation encoding of `by_id` (`parseSingleAnnotation`):
/// the record, then each relationship's segment count and IDs.
pub fn decode_single(info: &AnnotationInfo, bytes: &[u8], id: u64) -> Result<PrecomputedAnnotation> {
    let (mut a, mut offset) = decode_record(info, bytes, 0)?;
    a.id = id;
    for _ in &info.relationships {
        let count = u32::from_le_bytes(bytes.get(offset..offset + 4).with_context(|| format!("Expected at least {} bytes, but received: {}", offset + 4, bytes.len()))?.try_into().unwrap()) as usize;
        offset += 4;
        let ids = bytes.get(offset..offset + 8 * count).with_context(|| format!("Expected at least {} bytes, but received: {}", offset + 8 * count, bytes.len()))?;
        a.related_segments.push(ids.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect());
        offset += 8 * count;
    }
    if offset != bytes.len() {
        bail!("Expected {offset} bytes, but received: {}", bytes.len());
    }
    Ok(a)
}

/// Annotation `id` from the `by_id` index (`downloadMetadata`); `None` if
/// there is no such annotation.
pub async fn load_by_id(info: &AnnotationInfo, id: u64) -> Result<Option<PrecomputedAnnotation>> {
    let index = info.by_id.as_ref().context("the collection has no by_id index")?;
    let bytes = match &index.sharding {
        None => index.location.read(&id.to_string(), ng_kvstore::ByteRange::Full).await?,
        Some(sharding) => crate::sharded::read(&index.location, sharding, id as u128).await?,
    };
    bytes.map(|b| decode_single(info, &b, id)).transpose()
}

/// A spatial index cell's annotations (`<cell>` joined by `_`, or its
/// compressed Morton code in the shards); none where the cell was never
/// written.
pub async fn load_cell(info: &AnnotationInfo, level: usize, cell: &[u64]) -> Result<Vec<PrecomputedAnnotation>> {
    let l = info.spatial.get(level).context("no such spatial level")?;
    let bytes = match &l.index.sharding {
        None => l.index.location.read(&cell.iter().map(u64::to_string).collect::<Vec<_>>().join("_"), ng_kvstore::ByteRange::Full).await?,
        Some(sharding) => {
            ensure!(info.rank == 3, "sharded spatial indices of rank {} are not supported", info.rank);
            let bits: [u32; 3] = std::array::from_fn(|k| crate::sharded::ceil_log2(l.grid_shape[k]));
            let key = crate::sharded::encode_z_index_compressed_3d(bits, [cell[0], cell[1], cell[2]]);
            crate::sharded::read(&l.index.location, sharding, key).await?
        }
    };
    match bytes {
        Some(b) => decode_annotations(info, &b),
        None => Ok(vec![]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn annotations_decode_as_neuroglancer() {
        let dir = ng_kvstore::parse_url("file:///tmp/a/").unwrap();
        let info = parse_info(
            &dir,
            &json!({
                "@type": "neuroglancer_annotations_v1",
                "dimensions": { "x": [1e-9, "m"], "y": [1e-9, "m"], "z": [1e-9, "m"] },
                "lower_bound": [0, 0, 0], "upper_bound": [10, 10, 10],
                "annotation_type": "point",
                "properties": [{ "id": "color", "type": "rgb" }, { "id": "size", "type": "float32" }],
                "relationships": [],
                "by_id": { "key": "by_id" },
                "spatial": [{ "key": "spatial0", "grid_shape": [1, 1, 1], "chunk_size": [10, 10, 10], "limit": 100 }]
            }),
        )
        .unwrap();
        // The float (alignment 4) before the colour: 12 + 4 + 3, padded to 20.
        let mut b = 2u64.to_le_bytes().to_vec();
        for (p, size, c) in [([1f32, 2.0, 3.0], 1.5f32, [255u8, 0, 0]), ([4.0, 5.0, 6.0], 2.5, [0, 255, 0])] {
            for v in p {
                b.extend(v.to_le_bytes());
            }
            b.extend(size.to_le_bytes());
            b.extend(c);
            b.push(0);
        }
        b.extend(7u64.to_le_bytes());
        b.extend(9u64.to_le_bytes());
        let a = decode_annotations(&info, &b).unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!((a[0].id, a[1].id), (7, 9));
        assert_eq!(a[1].vectors, vec![vec![4.0, 5.0, 6.0]]);
        assert_eq!(a[0].properties, vec![255.0, 1.5]);
        assert_eq!(a[1].properties, vec![(255 << 8) as f64, 2.5]);
        assert!(decode_annotations(&info, &b[..b.len() - 1]).is_err());
        // A `by_id` entry: the record alone (no relationships here).
        let single = decode_single(&info, &b[8..28], 7).unwrap();
        assert_eq!((single.id, single.vectors.clone(), single.properties.clone()), (7, vec![vec![1.0, 2.0, 3.0]], vec![255.0, 1.5]));
        assert!(decode_single(&info, &b[8..29], 7).is_err());
    }
}
