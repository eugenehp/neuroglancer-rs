//! Precomputed skeletons (`datasource/precomputed`, `skeleton/
//! decode_precomputed_skeleton.ts`): a `neuroglancer_skeletons` directory
//! whose `info` names the vertex attributes and optional sharding, and per
//! segment (keyed by its ID, or its uint64 in the shards) a vertex and edge
//! count, `[x, y, z]` float32 positions, `u32` edge index pairs and each
//! attribute's values (little-endian).

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use crate::precomputed::ShardingParameters;

/// A vertex attribute (`VertexAttributeInfo`).
#[derive(Debug, Clone, PartialEq)]
pub struct VertexAttribute {
    pub id: String,
    pub data_type: ng_core::DataType,
    pub num_components: usize,
}

/// `SkeletonMetadata`: the transform from the vertex coordinates (column-
/// major 4×4, `parseTransform`), attributes and sharding.
#[derive(Debug, Clone)]
pub struct SkeletonInfo {
    pub transform: [f64; 16],
    pub vertex_attributes: Vec<VertexAttribute>,
    pub sharding: Option<ShardingParameters>,
    pub segment_properties: Option<String>,
}

/// A skeleton source: its directory, metadata and the transform from its
/// coordinates to the volume's model space (`subsourceToModelSubspaceTransform`).
#[derive(Debug, Clone)]
pub struct SkeletonSource {
    pub location: ng_kvstore::KvStoreLocation,
    pub info: SkeletonInfo,
    pub to_model: [f64; 16],
}

/// A segment's skeleton.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SkeletonData {
    pub positions: Vec<f32>,
    /// Edges as vertex index pairs.
    pub indices: Vec<u32>,
    /// Each attribute's values, as little-endian bytes.
    pub attributes: Vec<Vec<u8>>,
}

impl SkeletonData {
    pub fn num_vertices(&self) -> usize {
        self.positions.len() / 3
    }

    pub fn bytes(&self) -> usize {
        self.positions.len() * 4 + self.indices.len() * 4 + self.attributes.iter().map(Vec::len).sum::<usize>()
    }
}

/// `parseTransform`: 12 numbers (a row-major 3×4 matrix) or none (the
/// identity), as a column-major 4×4.
pub fn parse_transform(v: Option<&Value>) -> Result<[f64; 16]> {
    let mut m = [0.0; 16];
    for i in 0..4 {
        m[5 * i] = 1.0;
    }
    let Some(v) = v.filter(|v| !v.is_null()) else { return Ok(m) };
    let a = v.as_array().filter(|a| a.len() == 12).with_context(|| format!("Expected array of length 12, but received: {v}"))?;
    for r in 0..3 {
        for c in 0..4 {
            m[c * 4 + r] = a[r * 4 + c].as_f64().filter(|x| x.is_finite()).with_context(|| format!("Expected finite floating-point number, but received: {}", a[r * 4 + c]))?;
        }
    }
    Ok(m)
}

/// `parseSkeletonMetadata`.
pub fn parse_info(v: &Value) -> Result<SkeletonInfo> {
    let o = v.as_object().context("skeleton info is not an object")?;
    let t = o.get("@type").and_then(Value::as_str).unwrap_or_default();
    ensure!(t == "neuroglancer_skeletons", "Unsupported skeleton type: {}", o.get("@type").unwrap_or(&Value::Null));
    let transform = parse_transform(o.get("transform"))?;
    let mut vertex_attributes: Vec<VertexAttribute> = vec![];
    if let Some(attrs) = o.get("vertex_attributes").filter(|v| !v.is_null()) {
        for a in attrs.as_array().context("vertex_attributes is not an array")? {
            let id = a.get("id").and_then(Value::as_str).context("vertex attribute id")?.to_owned();
            ensure!(!id.is_empty(), "vertex attribute id must not be empty");
            ensure!(!vertex_attributes.iter().any(|x| x.id == id), "duplicate vertex attribute id {id:?}");
            let dt = a.get("data_type").and_then(Value::as_str).context("data_type")?;
            let data_type = ng_core::DataType::from_name(dt).with_context(|| format!("Invalid enum value: {dt:?}."))?;
            let num_components = a.get("num_components").and_then(Value::as_u64).filter(|&n| n > 0).context("num_components must be a positive integer")? as usize;
            vertex_attributes.push(VertexAttribute { id, data_type, num_components });
        }
    }
    Ok(SkeletonInfo {
        transform,
        vertex_attributes,
        sharding: crate::precomputed::parse_sharding(o.get("sharding"))?,
        segment_properties: o.get("segment_properties").and_then(Value::as_str).map(str::to_owned),
    })
}

/// `decodeSkeletonChunk`.
pub fn decode(bytes: &[u8], attributes: &[VertexAttribute]) -> Result<SkeletonData> {
    ensure!(bytes.len() >= 8, "skeleton too short");
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let (nv, ne) = (u32_at(0) as usize, u32_at(4) as usize);
    let mut offset = 8 + nv * 12;
    let edges_end = offset + ne * 8;
    if bytes.len() < edges_end {
        bail!("skeleton of {} bytes does not hold {nv} vertices and {ne} edges", bytes.len());
    }
    let positions = bytes[8..offset].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let indices: Vec<u32> = bytes[offset..edges_end].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    offset = edges_end;
    let mut values = vec![];
    for a in attributes {
        let n = a.data_type.bytes() * a.num_components * nv;
        let data = bytes.get(offset..offset + n).with_context(|| format!("skeleton attribute {:?} truncated", a.id))?;
        values.push(data.to_vec());
        offset += n;
    }
    Ok(SkeletonData { positions, indices, attributes: values })
}

/// `PrecomputedSkeletonSource.download` (`fetchByUint64`): `None` where the
/// segment has no skeleton.
pub async fn load(source: &SkeletonSource, id: u64) -> Result<Option<SkeletonData>> {
    let bytes = match &source.info.sharding {
        None => source.location.read(&id.to_string(), ng_kvstore::ByteRange::Full).await?,
        Some(sharding) => crate::sharded::read(&source.location, sharding, id as u128).await?,
    };
    bytes.map(|b| decode(&b, &source.info.vertex_attributes)).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn skeletons_decode_as_neuroglancer() {
        let info = parse_info(&json!({
            "@type": "neuroglancer_skeletons",
            "transform": [1, 0, 0, 5, 0, 2, 0, 0, 0, 0, 1, 0],
            "vertex_attributes": [{"id": "radius", "data_type": "float32", "num_components": 1}]
        }))
        .unwrap();
        assert_eq!((info.transform[0], info.transform[5], info.transform[12]), (1.0, 2.0, 5.0));
        let mut b = vec![];
        b.extend(2u32.to_le_bytes());
        b.extend(1u32.to_le_bytes());
        for v in [0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 7.5, 8.5] {
            b.extend(v.to_le_bytes());
        }
        // Edge (0, 1) as above: vertex positions, then indices, then radius.
        let mut bytes = b[..8].to_vec();
        bytes.extend(&b[8..32]);
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(&b[40..48]);
        let s = decode(&bytes, &info.vertex_attributes).unwrap();
        assert_eq!(s.positions, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(s.indices, vec![0, 1]);
        assert_eq!(s.attributes[0].len(), 8);
        assert!(decode(&bytes[..20], &[]).is_err());
        assert!(parse_info(&json!({"@type": "neuroglancer_legacy_mesh"})).is_err());
    }
}
