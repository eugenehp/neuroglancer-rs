//! Precomputed legacy (single-resolution) meshes (`datasource/precomputed`,
//! `mesh/backend.ts`): per segment a JSON manifest `<id>:0` listing fragment
//! files, each a vertex count, `[x, y, z]` float32 positions (nanometres)
//! and `u32` triangle indices; the vertex normals as neuroglancer computes
//! and encodes them.

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// A mesh source (`MeshSource`): the directory holding the manifests and
/// fragments, and the transform from its coordinates to the volume's model
/// space (`subsourceToModelSubspaceTransform`, column-major 4×4).
#[derive(Debug, Clone)]
pub struct MeshSource {
    pub location: ng_kvstore::KvStoreLocation,
    pub to_model: [f64; 16],
    /// `None` for single-resolution meshes.
    pub multiscale: Option<crate::multiscale_mesh::MultiscaleInfo>,
}

/// A segment's mesh: its fragments concatenated.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshData {
    pub positions: Vec<f32>,
    pub indices: Vec<u32>,
    /// Per vertex, the normal as two snorm8 octahedron components
    /// (`encodeNormals32fx3ToOctahedron8x2`).
    pub normals: Vec<[i8; 2]>,
    /// The manifest's fragments, and those read (neuroglancer's chunks, as
    /// the render scale histogram counts them).
    pub fragments: u32,
    pub fragments_loaded: u32,
}

impl MeshData {
    pub fn bytes(&self) -> usize {
        self.positions.len() * 4 + self.indices.len() * 4 + self.normals.len() * 2
    }
}

/// The manifest's fragment keys (`decodeManifestChunk`).
pub fn parse_manifest(bytes: &[u8]) -> Result<Vec<String>> {
    let v: Value = serde_json::from_slice(bytes).context("parsing mesh manifest")?;
    let fragments = v.get("fragments").and_then(Value::as_array).context("Expected \"fragments\" array")?;
    fragments.iter().map(|f| f.as_str().map(str::to_owned).context("Expected string fragment key")).collect()
}

/// A fragment (`decodeTriangleVertexPositionsAndIndices`, little-endian):
/// its positions and indices.
pub fn parse_fragment(bytes: &[u8]) -> Result<(Vec<f32>, Vec<u32>)> {
    if bytes.len() < 4 {
        bail!("mesh fragment too short");
    }
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let end = 4 + 12 * n;
    if bytes.len() < end || !(bytes.len() - end).is_multiple_of(12) {
        bail!("mesh fragment of {} bytes does not hold {n} vertices and whole triangles", bytes.len());
    }
    let f32s = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
    let positions = f32s(&bytes[4..end]);
    let indices: Vec<u32> = bytes[end..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    if indices.iter().any(|&i| i as usize >= n) {
        bail!("mesh fragment index out of range");
    }
    Ok((positions, indices))
}

/// `computeVertexNormals`: each triangle's normalized `(v1 − v0) × (v2 − v1)`
/// summed at its vertices, then normalized (in float32, as the typed arrays
/// hold them).
pub fn vertex_normals(positions: &[f32], indices: &[u32]) -> Vec<f32> {
    let mut normals = vec![0f32; positions.len()];
    for t in indices.chunks_exact(3) {
        let p = |i: u32| {
            let o = i as usize * 3;
            [positions[o], positions[o + 1], positions[o + 2]]
        };
        let (v0, v1, v2) = (p(t[0]), p(t[1]), p(t[2]));
        let a = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let b = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
        // `vec3.cross` in doubles, stored as float32.
        let (a, b) = (a.map(f64::from), b.map(f64::from));
        let mut n = [(a[1] * b[2] - a[2] * b[1]) as f32, (a[2] * b[0] - a[0] * b[2]) as f32, (a[0] * b[1] - a[1] * b[0]) as f32];
        normalize(&mut n);
        for &i in t {
            let o = i as usize * 3;
            for j in 0..3 {
                normals[o + j] += n[j];
            }
        }
    }
    for n in normals.chunks_exact_mut(3) {
        let mut v = [n[0], n[1], n[2]];
        normalize(&mut v);
        n.copy_from_slice(&v);
    }
    normals
}

/// gl-matrix's `vec3.normalize` (zero stays zero), on float32 storage.
fn normalize(v: &mut [f32; 3]) {
    let len = v[0] as f64 * v[0] as f64 + v[1] as f64 * v[1] as f64 + v[2] as f64 * v[2] as f64;
    if len > 0.0 {
        let inv = 1.0 / len.sqrt();
        for x in v.iter_mut() {
            *x = (*x as f64 * inv) as f32;
        }
    }
}

/// `snorm8`: `min(max(-127, x·127 + 0.5), 127) >>> 0`, stored in a byte —
/// truncated toward zero.
fn snorm8(x: f64) -> i8 {
    (x * 127.0 + 0.5).clamp(-127.0, 127.0).trunc() as i8
}

/// `encodeNormals32fx3ToOctahedron8x2`.
pub fn encode_normals(normals: &[f32]) -> Vec<[i8; 2]> {
    let sign = |x: f64| if x < 0.0 { -1.0 } else { 1.0 };
    normals
        .chunks_exact(3)
        .map(|n| {
            let (x, y, z) = (n[0] as f64, n[1] as f64, n[2] as f64);
            let inv = 1.0 / (x.abs() + y.abs() + z.abs());
            if z < 0.0 {
                [snorm8((1.0 - (y * inv).abs()) * sign(x)), snorm8((1.0 - (x * inv).abs()) * sign(y))]
            } else {
                [snorm8(x * inv), snorm8(y * inv)]
            }
        })
        .collect()
}

/// A segment's mesh from `source`: its manifest, then each fragment; none
/// if it has no manifest (`<id>:0` missing).
pub async fn load(source: &MeshSource, id: u64) -> Result<MeshData> {
    let manifest = source.location.read(&format!("{id}:0"), ng_kvstore::ByteRange::Full).await?;
    let Some(manifest) = manifest else { return Ok(MeshData::default()) };
    let mut mesh = MeshData::default();
    let keys = parse_manifest(&manifest)?;
    mesh.fragments = keys.len() as u32;
    for key in keys {
        let Some(bytes) = source.location.read(&key, ng_kvstore::ByteRange::Full).await? else { continue };
        mesh.fragments_loaded += 1;
        let (positions, indices) = parse_fragment(&bytes)?;
        let base = (mesh.positions.len() / 3) as u32;
        mesh.positions.extend_from_slice(&positions);
        mesh.indices.extend(indices.iter().map(|i| i + base));
        // Normals per fragment (each a separate chunk in neuroglancer).
        mesh.normals.extend(encode_normals(&vertex_normals(&positions, &indices)));
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragments_and_normals_as_neuroglancer() {
        let mut bytes = 3u32.to_le_bytes().to_vec();
        for v in [0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for i in [0u32, 1, 2] {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        let (p, i) = parse_fragment(&bytes).unwrap();
        assert_eq!((p.len(), i.clone()), (9, vec![0, 1, 2]));
        let n = vertex_normals(&p, &i);
        assert_eq!(&n[0..3], &[0.0, 0.0, 1.0]);
        // +z: (0, 0); −z: the folded octant; snorm8 truncates toward zero.
        assert_eq!(encode_normals(&[0.0, 0.0, 1.0]), vec![[0, 0]]);
        assert_eq!(encode_normals(&[0.0, 0.0, -1.0]), vec![[127, 127]]);
        assert_eq!(encode_normals(&[-1.0, 0.0, 0.0]), vec![[-126, 0]]);
        assert_eq!(encode_normals(&[1.0, 0.0, 0.0]), vec![[127, 0]]);
        assert!(parse_fragment(&bytes[..10]).is_err());
        assert_eq!(parse_manifest(br#"{"fragments": ["1_0", "1_1"]}"#).unwrap(), vec!["1_0", "1_1"]);
    }
}
