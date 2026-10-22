//! Multi-resolution meshes (`neuroglancer_multilod_draco`;
//! `datasource/precomputed/backend.ts`, `mesh/multiscale.ts`,
//! `mesh/draco/neuroglancer_draco.cc`): per segment a manifest (an octree of
//! fragments over levels of detail) and the fragments, Draco meshes whose
//! integer positions are fractions of their octree node.

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// The mesh directory's `info`: the format's parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiscaleInfo {
    pub vertex_quantization_bits: u32,
    /// Stored model → model coordinates (column-major 4×4).
    pub transform: [f64; 16],
    pub lod_scale_multiplier: f64,
    /// Manifests (and the fragments before them) in shards.
    pub sharding: Option<crate::precomputed::ShardingParameters>,
}

/// `parseMeshMetadata` for `neuroglancer_multilod_draco`.
pub fn parse_info(v: &Value) -> Result<MultiscaleInfo> {
    let bits = v.get("vertex_quantization_bits").and_then(Value::as_u64).context("vertex_quantization_bits")? as u32;
    if bits != 10 && bits != 16 {
        bail!("vertex_quantization_bits must be 10 or 16");
    }
    let multiplier = v.get("lod_scale_multiplier").and_then(Value::as_f64).filter(|x| *x > 0.0 && x.is_finite()).context("lod_scale_multiplier")?;
    // `parseTransform`: 12 numbers, a 3×4 row-major matrix.
    let mut transform = [0.0; 16];
    transform[15] = 1.0;
    let t = v.get("transform").and_then(Value::as_array).filter(|a| a.len() == 12).context("transform")?;
    for r in 0..3 {
        for c in 0..4 {
            transform[c * 4 + r] = t[r * 4 + c].as_f64().context("transform")?;
        }
    }
    let sharding = crate::precomputed::parse_sharding(v.get("sharding"))?;
    Ok(MultiscaleInfo { vertex_quantization_bits: bits, transform, lod_scale_multiplier: multiplier, sharding })
}

/// `MultiscaleMeshManifest`, with the fragments' byte offsets.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Manifest {
    pub chunk_shape: [f32; 3],
    pub grid_origin: [f32; 3],
    pub clip_lower: [f32; 3],
    pub clip_upper: [f32; 3],
    pub lod_scales: Vec<f32>,
    pub vertex_offsets: Vec<f32>,
    /// Rows of `[x, y, z, child begin, child end | empty << 31]`.
    pub octree: Vec<u32>,
    /// Each row's fragment's bytes: `offsets[row]..offsets[row + 1]`.
    pub offsets: Vec<f64>,
    /// Where a sharded manifest lies: its fragments end where it begins.
    pub shard: Option<crate::sharded::ShardInfo>,
}

impl Manifest {
    /// `downloadFragment`'s request: the file and byte range of `row`'s
    /// fragment, of segment `segment` (sharded: in the shard, before the
    /// manifest).
    pub fn fragment_range(&self, segment: u64, row: usize) -> (String, u64, u64) {
        let (start, end) = (self.offsets[row], self.offsets[row + 1]);
        match &self.shard {
            Some(shard) => {
                let full = *self.offsets.last().unwrap_or(&0.0);
                let begin = shard.offset as f64 - full + start;
                (shard.shard_path.clone(), begin as u64, (end - start) as u64)
            }
            None => (segment.to_string(), start as u64, (end - start) as u64),
        }
    }
}

/// `lessMsb`.
fn less_msb(a: u32, b: u32) -> bool {
    a < b && a < (a ^ b)
}

/// `zorder3LessThan`.
pub fn zorder3_less_than(x0: u32, y0: u32, z0: u32, x1: u32, y1: u32, z1: u32) -> bool {
    let (mut m0, mut m1) = (z0, z1);
    if less_msb(m0 ^ m1, y0 ^ y1) {
        m0 = y0;
        m1 = y1;
    }
    if less_msb(m0 ^ m1, x0 ^ x1) {
        m0 = x0;
        m1 = x1;
    }
    m0 < m1
}

/// `generateHigherOctreeLevel`.
fn generate_higher_level(octree: &mut [u32], prior_start: usize, prior_end: usize) -> usize {
    let mut cur = prior_end;
    for i in 0..3 {
        octree[cur * 5 + i] = octree[prior_start * 5 + i] >> 1;
    }
    octree[cur * 5 + 3] = prior_start as u32;
    for i in prior_start + 1..prior_end {
        let (x, y, z) = (octree[i * 5] >> 1, octree[i * 5 + 1] >> 1, octree[i * 5 + 2] >> 1);
        if x != octree[cur * 5] || y != octree[cur * 5 + 1] || z != octree[cur * 5 + 2] {
            octree[cur * 5 + 4] = i as u32;
            cur += 1;
            octree[cur * 5] = x;
            octree[cur * 5 + 1] = y;
            octree[cur * 5 + 2] = z;
            octree[cur * 5 + 3] = i as u32;
        }
    }
    octree[cur * 5 + 4] = prior_end as u32;
    cur + 1
}

/// `computeOctreeChildOffsets`.
fn compute_child_offsets(octree: &mut [u32], child_start: usize, child_end: usize, parent_end: usize) {
    let mut child = child_start;
    for parent in child_end..parent_end {
        let (px, py, pz) = (octree[parent * 5], octree[parent * 5 + 1], octree[parent * 5 + 2]);
        while child < child_end {
            let (cx, cy, cz) = (octree[child * 5] >> 1, octree[child * 5 + 1] >> 1, octree[child * 5 + 2] >> 1);
            if !zorder3_less_than(cx, cy, cz, px, py, pz) {
                break;
            }
            child += 1;
        }
        octree[parent * 5 + 3] = child as u32;
        while child < child_end {
            let (cx, cy, cz) = (octree[child * 5] >> 1, octree[child * 5 + 1] >> 1, octree[child * 5 + 2] >> 1);
            if cx != px || cy != py || cz != pz {
                break;
            }
            child += 1;
        }
        octree[parent * 5 + 4] = octree[parent * 5 + 4].wrapping_add(child as u32);
    }
}

/// `decodeMultiscaleManifestChunk`.
pub fn parse_manifest(bytes: &[u8], lod_scale_multiplier: f64) -> Result<Manifest> {
    if bytes.len() < 28 || !bytes.len().is_multiple_of(4) {
        bail!("Invalid index file size: {}", bytes.len());
    }
    let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let chunk_shape = [f32_at(0), f32_at(4), f32_at(8)];
    let grid_origin = [f32_at(12), f32_at(16), f32_at(20)];
    let num_stored = u32_at(24) as usize;
    let mut offset = 28;
    if bytes.len() < offset + 20 * num_stored {
        bail!("Invalid index file size for {num_stored} lods: {}", bytes.len());
    }
    let stored_scales: Vec<f32> = (0..num_stored).map(|i| f32_at(offset + 4 * i)).collect();
    offset += 4 * num_stored;
    let vertex_offsets: Vec<f32> = (0..3 * num_stored).map(|i| f32_at(offset + 4 * i)).collect();
    offset += 12 * num_stored;
    let per_lod: Vec<usize> = (0..num_stored).map(|i| u32_at(offset + 4 * i) as usize).collect();
    offset += 4 * num_stored;
    let total: usize = per_lod.iter().sum();
    if bytes.len() != offset + 16 * total {
        bail!("Invalid index file size for {num_stored} lods and {total} total fragments: {}", bytes.len());
    }
    let info: Vec<u32> = (0..4 * total).map(|i| u32_at(offset + 4 * i)).collect();
    let mut clip_lower = [f64::INFINITY; 3];
    let mut clip_upper = [f64::NEG_INFINITY; 3];
    let mut num_lods = num_stored.max(1);
    let mut base = 0;
    for (lod, &n) in per_lod.iter().enumerate() {
        for i in 0..3 {
            let values = &info[base + n * i..base + n * (i + 1)];
            let (lo, hi) = (values.iter().copied().min(), values.iter().copied().max());
            if let (Some(lo), Some(hi)) = (lo, hi) {
                // `>>>` takes its count modulo 32.
                while (hi >> ((num_lods - lod - 1) % 32)) != (lo >> ((num_lods - lod - 1) % 32)) {
                    num_lods += 1;
                }
                if lod == 0 {
                    clip_lower[i] = clip_lower[i].min(lo as f64);
                    clip_upper[i] = clip_upper[i].max(hi as f64 + 1.0);
                }
            }
        }
        base += n * 4;
    }
    // An upper bound on the octree's rows.
    let mut max_fragments = 0;
    {
        let (mut prev_n, mut prev_lod) = (0, 0);
        for (lod, &n) in per_lod.iter().enumerate() {
            max_fragments += prev_n * (lod - prev_lod);
            prev_lod = lod;
            prev_n = n;
            max_fragments += n;
        }
        max_fragments += (num_lods - 1 - prev_lod) * prev_n;
    }
    let mut octree = vec![0u32; 5 * max_fragments.max(1)];
    let mut offsets = vec![0f64; max_fragments + 1];
    let (mut prior_start, mut base_row, mut data_offset, mut fragment_base) = (0usize, 0usize, 0f64, 0usize);
    let mut lod = 0;
    while lod < num_stored {
        let n = per_lod[lod];
        for j in 0..n {
            for i in 0..3 {
                octree[5 * (base_row + j) + i] = info[fragment_base + j + i * n];
            }
            let size = info[fragment_base + j + 3 * n];
            data_offset += size as f64;
            offsets[base_row + j + 1] = data_offset;
            if size == 0 {
                octree[5 * (base_row + j) + 4] = 0x8000_0000;
            }
        }
        fragment_base += 4 * n;
        if lod != 0 {
            compute_child_offsets(&mut octree, prior_start, base_row, base_row + n);
        }
        prior_start = base_row;
        base_row += n;
        while lod + 1 < num_lods && (lod + 1 >= stored_scales.len() || stored_scales[lod + 1] == 0.0) {
            let end = generate_higher_level(&mut octree, prior_start, base_row);
            for o in &mut offsets[base_row + 1..end + 1] {
                *o = data_offset;
            }
            prior_start = base_row;
            base_row = end;
            lod += 1;
        }
        lod += 1;
    }
    octree.truncate(5 * base_row);
    offsets.truncate(base_row + 1);
    let mut lod_scales = vec![0f32; num_lods];
    for (i, s) in stored_scales.iter().enumerate() {
        lod_scales[i] = (*s as f64 * lod_scale_multiplier) as f32;
    }
    // `vec3.multiply` then `vec3.add`, each stored as float32.
    let f = |v: [f64; 3], _: bool| -> [f32; 3] { std::array::from_fn(|i| (grid_origin[i] as f64 + (v[i] as f32 as f64 * chunk_shape[i] as f64) as f32 as f64) as f32) };
    Ok(Manifest {
        chunk_shape,
        grid_origin,
        clip_lower: f(clip_lower, true),
        clip_upper: f(clip_upper, true),
        lod_scales,
        vertex_offsets,
        octree,
        offsets,
        shard: None,
    })
}

/// `getFrustumPlanes` (as float32): left, right, bottom, top, then the
/// normalized near and far planes.
pub fn frustum_planes(m: &[f32; 16]) -> [f32; 24] {
    let g = |i: usize| m[i] as f64;
    let (m00, m10, m20, m30) = (g(0), g(1), g(2), g(3));
    let (m01, m11, m21, m31) = (g(4), g(5), g(6), g(7));
    let (m02, m12, m22, m32) = (g(8), g(9), g(10), g(11));
    let (m03, m13, m23, m33) = (g(12), g(13), g(14), g(15));
    let mut out = [0f64; 24];
    out[..16].copy_from_slice(&[
        m30 + m00, m31 + m01, m32 + m02, m33 + m03,
        m30 - m00, m31 - m01, m32 - m02, m33 - m03,
        m30 + m10, m31 + m11, m32 + m12, m33 + m13,
        m30 - m10, m31 - m11, m32 - m12, m33 - m13,
    ]);
    // `out` holds float32: the plane coefficients are rounded before use.
    for v in &mut out[..16] {
        *v = *v as f32 as f64;
    }
    let near = [m30 + m20, m31 + m21, m32 + m22, m33 + m23];
    let far = [m30 - m20, m31 - m21, m32 - m22, m33 - m23];
    for (k, p) in [near, far].iter().enumerate() {
        let norm = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        for i in 0..4 {
            out[16 + 4 * k + i] = p[i] / norm;
        }
    }
    out.map(|v| v as f32)
}

/// `isAABBVisible`.
pub fn is_aabb_visible(lower: [f64; 3], upper: [f64; 3], planes: &[f32; 24]) -> bool {
    (0..6).all(|i| {
        let [a, b, c, d] = [0, 1, 2, 3].map(|k| planes[i * 4 + k] as f64);
        (a * lower[0]).max(a * upper[0]) + (b * lower[1]).max(b * upper[1]) + (c * lower[2]).max(c * upper[2]) + d >= 0.0
    })
}

/// `getDesiredMultiscaleMeshChunks`: each chunk wanted, coarsest first, as
/// `(lod, row, renderScale, empty)`.
pub fn desired_chunks(manifest: &Manifest, mvp: &[f32; 16], planes: &[f32; 24], detail_cutoff: f64, width: f64, height: f64, callback: &mut dyn FnMut(usize, usize, f64, bool)) {
    let octree = &manifest.octree;
    if octree.is_empty() {
        return;
    }
    let max_lod = manifest.lod_scales.len() - 1;
    let g = |i: usize| mvp[i] as f64;
    let (m00, m01, m02) = (g(0), g(4), g(8));
    let (m10, m11, m12) = (g(1), g(5), g(9));
    let (m30, m31, m32, m33) = (g(3), g(7), g(11), g(15));
    let coeff = |m: f64| if m > 0.0 { 0.0 } else { 1.0 };
    let (cx, cy, cz) = (coeff(m30), coeff(m31), coeff(m32));
    let point_w = |x: f64, y: f64, z: f64| m30 * x + m31 * y + m32 * z + m33;
    let near = [planes[16] as f64, planes[17] as f64, planes[18] as f64, planes[19] as f64];
    let min_w_clip = point_w(-near[3] * near[0], -near[3] * near[1], -near[3] * near[2]);
    let lower = manifest.clip_lower.map(f64::from);
    let upper = manifest.clip_upper.map(f64::from);
    let x_scale = ((m00 * width).powi(2) + (m10 * height).powi(2)).sqrt();
    let y_scale = ((m01 * width).powi(2) + (m11 * height).powi(2)).sqrt();
    let z_scale = ((m02 * width).powi(2) + (m12 * height).powi(2)).sqrt();
    let scale_factor = x_scale.max(y_scale).max(z_scale);
    let shape = manifest.chunk_shape.map(f64::from);
    let origin = manifest.grid_origin.map(f64::from);

    fn handle(
        lod: usize,
        row: usize,
        prior_lod_scale: f64,
        ctx: &dyn Fn(usize, usize) -> Option<f64>,
        manifest: &Manifest,
        detail_cutoff: f64,
        callback: &mut dyn FnMut(usize, usize, f64, bool),
    ) {
        let Some(pixel_size) = ctx(lod, row) else { return };
        if !(prior_lod_scale == 0.0 || pixel_size * detail_cutoff < prior_lod_scale) {
            return;
        }
        let lod_scale = manifest.lod_scales[lod] as f64;
        let end_and_empty = manifest.octree[row * 5 + 4];
        if lod_scale != 0.0 {
            callback(lod, row, lod_scale / pixel_size, end_and_empty >> 31 != 0);
        }
        if lod > 0 && (lod_scale == 0.0 || pixel_size * detail_cutoff < lod_scale) {
            let next = if lod_scale == 0.0 { prior_lod_scale } else { lod_scale };
            let begin = manifest.octree[row * 5 + 3] as usize;
            let end = (end_and_empty & 0x7fff_ffff) as usize;
            for child in begin..end {
                handle(lod - 1, child, next, ctx, manifest, detail_cutoff, callback);
            }
        }
    }
    // A node's visible pixel size, if it is visible.
    let ctx = |lod: usize, row: usize| -> Option<f64> {
        let size = (1u64 << lod) as f64;
        let mut lo = [0.0; 3];
        let mut hi = [0.0; 3];
        for i in 0..3 {
            let l = octree[row * 5 + i] as f64 * size * shape[i] + origin[i];
            lo[i] = l.max(lower[i]);
            hi[i] = (l + size * shape[i]).min(upper[i]);
        }
        if !is_aabb_visible(lo, hi, planes) {
            return None;
        }
        let box_w = point_w(lo[0] + cx * (hi[0] - lo[0]), lo[1] + cy * (hi[1] - lo[1]), lo[2] + cz * (hi[2] - lo[2]));
        Some(min_w_clip.max(box_w) / scale_factor)
    };
    handle(max_lod, octree.len() / 5 - 1, 0.0, &ctx, manifest, detail_cutoff, callback);
}

/// `getOctreeChildIndex`.
fn child_index(x: u32, y: u32, z: u32) -> usize {
    ((x & 1) | ((y << 1) & 2) | ((z << 2) & 4)) as usize
}

/// `getMultiscaleChunksToDraw`: what to draw, `(lod, row, sub-chunk begin,
/// end)`, given which chunks are loaded (`has_chunk`).
#[allow(clippy::too_many_arguments)]
pub fn chunks_to_draw(
    manifest: &Manifest,
    mvp: &[f32; 16],
    planes: &[f32; 24],
    detail_cutoff: f64,
    width: f64,
    height: f64,
    has_chunk: &mut dyn FnMut(usize, usize, f64) -> bool,
    callback: &mut dyn FnMut(usize, usize, usize, usize, f64),
) {
    let lod_scales = &manifest.lod_scales;
    let mut max_lod = 0;
    while max_lod + 1 < lod_scales.len() && lod_scales[max_lod + 1] != 0.0 {
        max_lod += 1;
    }
    // [row (-1: empty), parent sub-chunk index, render scale] per stack entry.
    let mut stack: Vec<(i64, usize, f64)> = vec![(0, 0, 0.0); lod_scales.len() + 1];
    let mut depth = 0usize;
    let mut prior_sub = 0usize;
    let emit = |target: usize, sub: usize, depth: &mut usize, prior_sub: &mut usize, stack: &[(i64, usize, f64)], callback: &mut dyn FnMut(usize, usize, usize, usize, f64)| loop {
        if *depth == 0 {
            return;
        }
        let index = *depth - 1;
        let entry_lod = max_lod.wrapping_sub(index);
        let (row, entry_sub, scale) = stack[index];
        let num_sub = if entry_lod == 0 { 1 } else { 8 };
        if target == *depth {
            let end_sub = sub & (num_sub - 1);
            if *prior_sub != end_sub && row != -1 {
                callback(entry_lod, row as usize, *prior_sub, end_sub, scale);
            }
            *prior_sub = end_sub + 1;
            return;
        }
        if *prior_sub != num_sub && row != -1 {
            callback(entry_lod, row as usize, *prior_sub, num_sub, scale);
        }
        *prior_sub = entry_sub + 1;
        *depth -= 1;
    };
    let mut prior_missing = 0usize;
    let octree = &manifest.octree;
    desired_chunks(manifest, mvp, planes, detail_cutoff, width, height, &mut |lod, row, scale, empty| {
        if !empty && !has_chunk(lod, row, scale) {
            prior_missing = prior_missing.max(lod);
            return;
        }
        if lod < prior_missing {
            return;
        }
        prior_missing = 0;
        let sub = child_index(octree[row * 5], octree[row * 5 + 1], octree[row * 5 + 2]);
        let index = max_lod - lod;
        emit(index, sub, &mut depth, &mut prior_sub, &stack, callback);
        stack[index] = (if empty { -1 } else { row as i64 }, sub, scale);
        prior_sub = 0;
        depth = index + 1;
    });
    emit(0, 0, &mut depth, &mut prior_sub, &stack, callback);
}

/// A decoded fragment (`RawPartitionedMeshData`, converted): positions as
/// fractions of the node, normals, indices partitioned into the node's
/// eight sub-chunks (one for level 0) at `sub_chunk_offsets`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fragment {
    pub positions: Vec<f32>,
    pub normals: Vec<[i8; 2]>,
    pub indices: Vec<u32>,
    pub sub_chunk_offsets: Vec<u32>,
}

impl Fragment {
    pub fn bytes(&self) -> usize {
        self.positions.len() * 4 + self.normals.len() * 2 + self.indices.len() * 4
    }
}

/// `neuroglancer_draco_decode` then `convertMeshData`: the Draco mesh's
/// quantized positions (not dequantized), its faces, partitioned by the
/// node's midpoints when `partition` (levels above 0).
pub fn decode_fragment(bytes: &[u8], bits: u32, partition: bool) -> Result<Fragment> {
    use threers::loaders::draco::{attributes::attribute_type, attributes::AttributeValues, decode_mesh_with, DecodeOptions};
    let options = DecodeOptions { skip_attribute_transform: vec![attribute_type::POSITION] };
    let mesh = decode_mesh_with(bytes, &options).map_err(|e| anyhow::anyhow!("Failed to decode draco mesh: {e:?}"))?;
    let att = mesh.attribute(attribute_type::POSITION).context("Failed to decode draco mesh: no positions")?;
    if att.num_components() != 3 {
        bail!("Failed to decode draco mesh: positions of {} components", att.num_components());
    }
    let q: Vec<u32> = match &att.values {
        AttributeValues::Integer(v) => v.iter().map(|&x| x as u32).collect(),
        AttributeValues::Float(v) => v.iter().map(|&x| x as u32).collect(),
    };
    let indices = mesh.indices();
    let num_faces = indices.len() / 3;
    let (indices, sub_chunk_offsets) = if partition {
        let point = (u32::MAX >> (32 - bits)) / 2 + 1;
        let mask = |v: usize| -> u32 {
            let p = &q[v * 3..v * 3 + 3];
            let mut m = 0xFFu32;
            let sides = [(0b0101_0101, 0b1010_1010), (0b0011_0011, 0b1100_1100), (0b0000_1111, 0b1111_0000)];
            for (k, (low, high)) in sides.iter().enumerate() {
                if p[k] < point {
                    m &= low;
                } else if p[k] > point {
                    m &= high;
                }
            }
            m
        };
        let first_bit = |m: u32| if m == 0 { 0 } else { m.trailing_zeros() as usize };
        let parts: Vec<usize> = (0..num_faces).map(|f| first_bit((0..3).fold(0xFF, |m, j| m & mask(indices[f * 3 + j] as usize)))).collect();
        let mut offsets = [0u32; 9];
        for &p in &parts {
            offsets[p + 1] += 3;
        }
        let mut sum = 0;
        for o in offsets.iter_mut().skip(1) {
            let count = *o;
            *o = sum;
            sum += count;
        }
        let mut out = vec![0u32; num_faces * 3];
        for (f, &p) in parts.iter().enumerate() {
            let at = offsets[p + 1] as usize;
            offsets[p + 1] += 3;
            out[at..at + 3].copy_from_slice(&indices[f * 3..f * 3 + 3]);
        }
        (out, offsets.to_vec())
    } else {
        (indices, vec![0, num_faces as u32 * 3])
    };
    let raw: Vec<f32> = q.iter().map(|&x| x as f32).collect();
    let normals = crate::mesh::encode_normals(&crate::mesh::vertex_normals(&raw, &indices));
    // The vertex position format's normalization (`/ 1023.0`, or unorm16).
    let max = ((1u64 << bits) - 1) as f32;
    let positions = raw.iter().map(|&x| x / max).collect();
    Ok(Fragment { positions, normals, indices, sub_chunk_offsets })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zorder_and_octrees_as_neuroglancer() {
        assert!(zorder3_less_than(0, 0, 0, 1, 0, 0));
        assert!(zorder3_less_than(1, 0, 0, 0, 1, 0));
        assert!(zorder3_less_than(1, 1, 1, 2, 0, 0));
        assert!(!zorder3_less_than(0, 0, 1, 1, 1, 0));
        // Two level-0 fragments under one root: the root's children span both.
        let mut b = vec![];
        for v in [1f32, 1.0, 1.0, 0.0, 0.0, 0.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&1f32.to_le_bytes());
        for _ in 0..3 {
            b.extend_from_slice(&0f32.to_le_bytes());
        }
        b.extend_from_slice(&2u32.to_le_bytes());
        for v in [0u32, 1, 0, 0, 0, 0, 10, 20] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let m = parse_manifest(&b, 2.0).unwrap();
        assert_eq!(m.lod_scales, vec![2.0, 0.0]);
        assert_eq!(m.octree, vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2]);
        assert_eq!(m.offsets, vec![0.0, 10.0, 30.0, 30.0]);
        assert_eq!((m.clip_lower, m.clip_upper), ([0.0; 3], [2.0, 1.0, 1.0]));
    }

    /// `mesh/multiscale.spec.ts`'s cases.
    fn spec_manifest(eight: bool) -> Manifest {
        let mut octree = vec![];
        if eight {
            for (x, y, z) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)] {
                octree.extend_from_slice(&[x, y, z, 0, 0]);
            }
            octree.extend_from_slice(&[0, 0, 0, 0, 8]);
        } else {
            octree.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        }
        Manifest {
            chunk_shape: [10.0, 20.0, 30.0],
            grid_origin: [5.0, 6.0, -50.0],
            clip_lower: if eight { [5.0, 6.0, -50.0] } else { [20.0, 23.0, -50.0] },
            clip_upper: if eight { [100.0, 200.0, 10.0] } else { [40.0, 45.0, -20.0] },
            lod_scales: vec![20.0, 40.0],
            vertex_offsets: vec![0.0; 6],
            octree,
            offsets: vec![],
            shard: None,
        }
    }

    fn desired(m: &Manifest, cutoff: f64) -> Vec<(usize, usize, f64, bool)> {
        let mvp = ng_core::glm::mat4_perspective(std::f64::consts::PI / 2.0, 640.0 / 480.0, 5.0, 100.0);
        let planes = frustum_planes(&mvp);
        let mut out = vec![];
        desired_chunks(m, &mvp, &planes, cutoff, 640.0, 480.0, &mut |lod, row, scale, empty| out.push((lod, row, scale, empty)));
        out
    }

    fn draw(m: &Manifest, cutoff: f64, has: impl Fn(usize) -> bool) -> Vec<(usize, usize, usize, usize, f64)> {
        let mvp = ng_core::glm::mat4_perspective(std::f64::consts::PI / 2.0, 640.0 / 480.0, 5.0, 100.0);
        let planes = frustum_planes(&mvp);
        let mut out = vec![];
        chunks_to_draw(m, &mvp, &planes, cutoff, 640.0, 480.0, &mut |_, row, _| has(row), &mut |lod, row, b, e, scale| out.push((lod, row, b, e, scale)));
        out
    }

    #[test]
    fn desired_chunks_as_the_spec() {
        let simple = spec_manifest(false);
        assert_eq!(desired(&simple, 1000.0), vec![(1, 1, 960.0, false)]);
        assert_eq!(desired(&simple, 800.0), vec![(1, 1, 960.0, false), (0, 0, 480.0, false)]);
        let m = spec_manifest(true);
        assert_eq!(desired(&m, 4000.0), vec![(1, 8, 3840.0, false)]);
        assert_eq!(desired(&m, 1000.0), vec![(1, 8, 3840.0, false), (0, 4, 1920.0, false), (0, 5, 1920.0, false)]);
        assert_eq!(
            desired(&m, 800.0),
            vec![(1, 8, 3840.0, false), (0, 0, 480.0, false), (0, 1, 480.0, false), (0, 2, 480.0, false), (0, 3, 480.0, false), (0, 4, 1920.0, false), (0, 5, 1920.0, false)]
        );
    }

    #[test]
    fn chunks_to_draw_as_the_spec() {
        let m = spec_manifest(true);
        assert_eq!(draw(&m, 4000.0, |_| true), vec![(1, 8, 0, 8, 3840.0)]);
        assert_eq!(draw(&m, 1000.0, |row| row != 4), vec![(1, 8, 0, 5, 3840.0), (0, 5, 0, 1, 1920.0), (1, 8, 6, 8, 3840.0)]);
        assert_eq!(draw(&m, 1000.0, |row| row != 8), vec![]);
    }
}
