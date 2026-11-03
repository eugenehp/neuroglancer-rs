//! Which chunks a slice panel shows, and how to draw them. Ports of the
//! transformed-source setup (`getVolumetricTransformedSources`), scale choice
//! (`filterVisibleSources`) and chunk enumeration
//! (`forEachPlaneIntersectingVolumetricChunk`) from `src/sliceview/`.

use ng_core::glm::Mat4;
use ng_core::matrix::{self, inverse4, multiply4, transform_point4};
use ng_core::navigation::DisplayDimensionRenderInfo;

use crate::layer::VolumeLayer;

/// A source (one scale) as one panel sees it.
#[derive(Debug, Clone)]
pub struct PanelSource {
    pub scale: usize,
    /// Chunk dimensions shown along display dimensions 0..3 (`None` where the
    /// display has fewer dimensions than 3, or the chunk does not vary).
    pub chunk_display_dims: [Option<usize>; 3],
    /// Chunk dimensions that are not displayed, with the position's chunk
    /// index and offset within that chunk.
    pub fixed: Vec<(usize, u64, u32)>,
    /// 4×4, column-major: chunk display voxels → display coordinates.
    pub chunk_to_display: [f64; 16],
    /// Physical size of one chunk voxel along each chunk display dimension
    /// (`effectiveVoxelSize`).
    pub effective_voxel_size: [f64; 3],
    /// Per chunk dimension, its coefficient on each global dimension (how
    /// the position moves through chunks, for prefetching).
    pub global_to_chunk: Vec<Vec<f64>>,
    /// The position in chunk voxels (floored), per chunk dimension.
    pub position_voxel: Vec<f64>,
}

/// What to draw for one chunk in one panel.
#[derive(Debug, Clone)]
pub struct ChunkDraw {
    pub scale: usize,
    /// Full chunk grid position (all chunk dimensions).
    pub grid: Vec<u64>,
    pub chunk_display_dims: [Option<usize>; 3],
    pub fixed: Vec<(usize, u64, u32)>,
    /// The chunk's region of the panel, in panel pixels (top-left origin),
    /// and NDC of that region → texture voxels.
    pub rect: [f64; 4],
    pub ndc_to_texture: [f32; 16],
    /// The clip bounds in the texture's voxels (lower, upper): what of the
    /// chunk is drawn. `rect` is the box's outline on the panel, so on a
    /// plane at an angle to the box a pixel inside it can still be outside
    /// the bounds; the shader checks each (`uLowerClipBound`, …).
    pub clip: [[f32; 3]; 2],
}

impl ChunkDraw {
    /// No clip bounds beyond the texture's own.
    pub const UNCLIPPED: [[f32; 3]; 2] = [[-1e9; 3], [1e9; 3]];
}

/// The panel's camera: view (display coordinates) and orthographic
/// projection, as the slice view sets them up.
pub struct PanelCamera {
    /// View pixels (x right, y down, centered) → display coordinates.
    pub inv_view: [f64; 16],
    /// Panel size, logical pixels.
    pub width: f64,
    pub height: f64,
}

impl PanelCamera {
    pub fn new(inv_view: Mat4, width: f64, height: f64) -> Self {
        Self {
            inv_view: inv_view.map(|x| x as f64),
            width,
            height,
        }
    }

    /// Display coordinates → NDC (z: along the view direction, 1 view pixel
    /// per unit).
    pub fn view_projection(&self) -> [f64; 16] {
        let view = inverse4(&self.inv_view);
        // `mat4.ortho(-w/2, w/2, h/2, -h/2, -1, 1)` with the depth scaled to
        // 1 per view unit (only the sign of z is used).
        let mut proj = [0.0; 16];
        proj[0] = 2.0 / self.width;
        proj[5] = -2.0 / self.height;
        proj[10] = 1.0;
        proj[15] = 1.0;
        multiply4(&proj, &view)
    }

    /// `pixelSize`: physical length of one view pixel along view x.
    pub fn pixel_size(&self, info: &DisplayDimensionRenderInfo) -> f64 {
        (0..3)
            .map(|i| (info.voxel_physical_scales[i] * self.inv_view[i]).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Every scale of `layer` as the panel sees it; `None` for a scale whose
/// displayed dimensions depend on more than three chunk dimensions, or that
/// does not contain the position along a non-displayed dimension.
pub fn panel_sources(layer: &VolumeLayer, info: &DisplayDimensionRenderInfo, position: &[f32]) -> Vec<Option<PanelSource>> {
    let rank = layer.rank;
    let n = rank + 1;
    (0..layer.scales.len())
        .map(|s| {
            let src = &layer.scales[s];
            let c2l = layer.chunk_to_layer(s);
            // Display rows of chunk → global.
            let mut rows = vec![vec![0.0f64; n]; 3];
            for (row, &g) in rows.iter_mut().zip(&info.display_dimension_indices).take(info.display_rank) {
                if let Some(layer_dim) = layer.layer_to_global.iter().position(|&x| x == Some(g as usize)) {
                    for (col, r) in row.iter_mut().enumerate() {
                        *r = c2l[col * n + layer_dim];
                    }
                }
            }
            let shown: Vec<usize> = (0..rank).filter(|&c| (0..info.display_rank).any(|i| rows[i][c] != 0.0)).collect();
            // A displayed dimension that only a transform spec added (no
            // chunk dimension behind it) is not drawn.
            if shown.len() > 3 || shown.iter().any(|&c| c >= layer.source_rank) {
                return None;
            }
            let mut chunk_display_dims = [None; 3];
            // Pair chunk dimensions with display dimensions where one maps
            // straight to one; the rest in order.
            let mut remaining = shown.clone();
            for i in 0..info.display_rank {
                if let Some(k) = remaining.iter().position(|&c| rows[i][c] != 0.0 && (0..info.display_rank).all(|j| j == i || rows[j][c] == 0.0)) {
                    chunk_display_dims[i] = Some(remaining.remove(k));
                }
            }
            for slot in chunk_display_dims.iter_mut() {
                if slot.is_none() && !remaining.is_empty() {
                    *slot = Some(remaining.remove(0));
                }
            }
            let mut m = [0.0; 16];
            m[15] = 1.0;
            for (k, cd) in chunk_display_dims.iter().enumerate() {
                match cd {
                    Some(c) => {
                        for i in 0..3 {
                            m[k * 4 + i] = rows[i][*c];
                        }
                    }
                    None => m[k * 4 + k] = 1.0,
                }
            }
            for i in 0..3 {
                m[12 + i] = rows[i][rank];
            }
            let mut effective_voxel_size = [1.0; 3];
            for (k, e) in effective_voxel_size.iter_mut().enumerate().take(info.display_rank) {
                *e = (0..info.display_rank)
                    .map(|i| (m[k * 4 + i] * info.voxel_physical_scales[i]).powi(2))
                    .sum::<f64>()
                    .sqrt();
            }
            // Non-displayed chunk dimensions, from the position.
            let (l2c, _) = matrix::inverse(&c2l, n);
            let mut layer_pos = vec![0.0; n];
            layer_pos[rank] = 1.0;
            for (layer_dim, g) in layer.layer_to_global.iter().enumerate() {
                layer_pos[layer_dim] = match g {
                    Some(g) => position[*g] as f64,
                    // Channel dimensions: the first channel (the default
                    // image shader's `invlerp` reads channel 0).
                    None => 0.5,
                };
            }
            let global_rank = position.len();
            let global_to_chunk: Vec<Vec<f64>> = (0..rank)
                .map(|c| {
                    let mut row = vec![0.0; global_rank];
                    for (layer_dim, g) in layer.layer_to_global.iter().enumerate() {
                        if let Some(g) = g {
                            row[*g] = l2c[layer_dim * n + c];
                        }
                    }
                    row
                })
                .collect();
            let position_voxel: Vec<f64> = (0..rank).map(|c| (0..n).map(|k| l2c[k * n + c] * layer_pos[k]).sum::<f64>().floor()).collect();
            let mut fixed = vec![];
            for c in 0..rank {
                if shown.contains(&c) {
                    continue;
                }
                let v: f64 = (0..n).map(|k| l2c[k * n + c] * layer_pos[k]).sum();
                let v = v.floor();
                if c >= layer.source_rank {
                    // An added dimension: the singleton extent [0, 1).
                    if v != 0.0 {
                        return None;
                    }
                    continue;
                }
                let lower = src.lower_clip_bound.as_ref().map_or(src.spec.lower_voxel_bound[c] as f64, |b| b[c] as f64);
                let upper = src.upper_clip_bound.as_ref().map_or(src.spec.upper_voxel_bound[c] as f64, |b| b[c] as f64);
                if !(v >= lower && v < upper) {
                    return None;
                }
                let size = src.spec.chunk_data_size[c] as f64;
                let chunk = (v / size).floor();
                fixed.push((c, chunk as u64, (v - chunk * size) as u32));
            }
            Some(PanelSource {
                scale: s,
                chunk_display_dims,
                fixed,
                chunk_to_display: m,
                effective_voxel_size,
                global_to_chunk,
                position_voxel,
            })
        })
        .collect()
}

/// `filterVisibleSources`: scale indices to draw, coarsest first (finer ones
/// draw over coarser ones). `render_scale_target` is the layer's
/// `renderScaleTarget` (default 1).
pub fn visible_scales(sources: &[Option<PanelSource>], pixel_size: f64, render_scale_target: f64) -> Vec<usize> {
    let available: Vec<&PanelSource> = sources.iter().flatten().collect();
    let Some(finest) = available.first() else { return vec![] };
    let pixel_size = pixel_size * 1.1;
    let smallest = finest.effective_voxel_size;
    let target = pixel_size * render_scale_target;
    let can_improve = |v: &[f64; 3]| (0..3).any(|i| v[i] > target && v[i] > 1.01 * smallest[i]);
    let improves = |v: &[f64; 3], prev: &[f64; 3]| {
        (0..3).any(|i| (target - v[i]).abs() < (target - prev[i]).abs() && v[i] < 1.01 * prev[i])
    };
    let mut out = vec![];
    let mut prev: Option<[f64; 3]> = None;
    for (index, s) in available.iter().enumerate().rev() {
        if let Some(p) = &prev {
            if !improves(&s.effective_voxel_size, p) {
                break;
            }
        }
        out.push(s.scale);
        if index == 0 || !can_improve(&s.effective_voxel_size) {
            break;
        }
        prev = Some(s.effective_voxel_size);
    }
    out
}

/// The chunks of `ps` the panel's slice plane crosses, with how to draw each.
pub fn visible_chunks(layer: &VolumeLayer, ps: &PanelSource, camera: &PanelCamera) -> Vec<ChunkDraw> {
    let src = &layer.scales[ps.scale];
    let spec = &src.spec;
    let grid_shape = spec.grid_shape();
    let mut size = [1.0f64; 3];
    let mut upper_grid = [1u64; 3];
    // The clip bounds (`lowerClipDisplayBound`, `upperClipDisplayBound`),
    // in chunk grid units: the chunks' boxes are cut to them.
    let mut clip_lower = [f64::NEG_INFINITY; 3];
    let mut clip_upper = [f64::INFINITY; 3];
    for (k, cd) in ps.chunk_display_dims.iter().enumerate() {
        if let Some(c) = cd {
            size[k] = spec.chunk_data_size[*c] as f64;
            upper_grid[k] = grid_shape[*c];
            clip_lower[k] = src.lower_clip_bound.as_ref().map_or(spec.lower_voxel_bound[*c] as f64, |b| b[*c] as f64) / size[k];
            clip_upper[k] = src.upper_clip_bound.as_ref().map_or(spec.upper_voxel_bound[*c] as f64, |b| b[*c] as f64) / size[k];
        }
    }
    let mut scale = [0.0; 16];
    scale[0] = size[0];
    scale[5] = size[1];
    scale[10] = size[2];
    scale[15] = 1.0;
    // Chunk grid units → NDC.
    let mvp = multiply4(&camera.view_projection(), &multiply4(&ps.chunk_to_display, &scale));
    let inv = inverse4(&mvp);
    // The viewport rectangle on the plane, in chunk grid units.
    let mut lower = [0i64; 3];
    let mut upper = [0i64; 3];
    for i in 0..3 {
        let c = inv[12 + i] + 1e-4 / size[i];
        let (x, y) = (inv[i].abs(), inv[4 + i].abs());
        lower[i] = (c - x - y).floor().max(0.0) as i64;
        upper[i] = ((c + x + y + 1.0).floor() as i64).min(upper_grid[i] as i64);
    }
    // Panel NDC → chunk display voxels.
    let display_to_voxel_ndc = inverse4(&multiply4(&camera.view_projection(), &ps.chunk_to_display));
    let rank = layer.source_rank;
    let mut out = vec![];
    for gz in lower[2]..upper[2] {
        for gy in lower[1]..upper[1] {
            for gx in lower[0]..upper[0] {
                let g = [gx as f64, gy as f64, gz as f64];
                let lo: [f64; 3] = std::array::from_fn(|k| g[k].max(clip_lower[k]));
                let hi: [f64; 3] = std::array::from_fn(|k| (g[k] + 1.0).min(clip_upper[k]));
                if (0..3).any(|k| hi[k] < lo[k]) {
                    continue;
                }
                // Whether the plane cuts the chunk (`isAABBIntersectingPlane`,
                // on the whole chunk); what is drawn is the box cut to the
                // clip bounds (the vertex shader's `uLowerClipBound`, …).
                let whole: Vec<[f64; 3]> = (0..8).map(|b| transform_point4(&mvp, [g[0] + (b & 1) as f64, g[1] + ((b >> 1) & 1) as f64, g[2] + ((b >> 2) & 1) as f64])).collect();
                let (zmin, zmax) = whole.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), c| (a.min(c[2]), b.max(c[2])));
                if zmin > 0.0 || zmax < 0.0 {
                    continue; // the plane misses the chunk
                }
                let corners: Vec<[f64; 3]> = (0..8)
                    .map(|b| transform_point4(&mvp, [if b & 1 == 1 { hi[0] } else { lo[0] }, if (b >> 1) & 1 == 1 { hi[1] } else { lo[1] }, if (b >> 2) & 1 == 1 { hi[2] } else { lo[2] }]))
                    .collect();
                let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
                for c in &corners {
                    x0 = x0.min(c[0]);
                    x1 = x1.max(c[0]);
                    y0 = y0.min(c[1]);
                    y1 = y1.max(c[1]);
                }
                let (x0, x1, y0, y1) = (x0.max(-1.0), x1.min(1.0), y0.max(-1.0), y1.min(1.0));
                if x0 >= x1 || y0 >= y1 {
                    continue;
                }
                // NDC rectangle → panel pixels (y down), snapped outward.
                let px0 = ((x0 + 1.0) / 2.0 * camera.width).floor();
                let px1 = ((x1 + 1.0) / 2.0 * camera.width).ceil();
                let py0 = ((1.0 - y1) / 2.0 * camera.height).floor();
                let py1 = ((1.0 - y0) / 2.0 * camera.height).ceil();
                let rect = [px0, py0, px1 - px0, py1 - py0];
                // Sub-rectangle NDC → panel NDC.
                let mut sub = [0.0; 16];
                sub[0] = rect[2] / camera.width;
                sub[5] = rect[3] / camera.height;
                sub[10] = 1.0;
                sub[15] = 1.0;
                sub[12] = (px0 + rect[2] / 2.0) / camera.width * 2.0 - 1.0;
                sub[13] = 1.0 - (py0 + rect[3] / 2.0) / camera.height * 2.0;
                // Panel NDC → chunk display voxels, then relative to the chunk.
                let mut origin = [0.0; 16];
                origin[0] = 1.0;
                origin[5] = 1.0;
                origin[10] = 1.0;
                origin[15] = 1.0;
                // Relative to the chunk, biased 1e-4 voxels toward the higher
                // voxel (neuroglancer's `BIAS_EPSILON`): a pixel on the
                // boundary between two chunks then belongs to exactly one,
                // rather than rounding out of both and leaving a seam.
                let ndc_to_voxel = multiply4(&display_to_voxel_ndc, &sub);
                for k in 0..3 {
                    // A plane exactly on the volume's upper face (parallel
                    // to it): neuroglancer's chunk box is closed and clipped
                    // there, and `getDataValue` clamps to the last voxel
                    // (`min(floor(p), uChunkDataSize − 1)`); biased the other
                    // way, the last voxel shows.
                    let on_upper_face = ps.chunk_display_dims[k].is_some_and(|c| {
                        // Within float32 error of the panel's matrices.
                        let flat = ndc_to_voxel[k].abs() < 1e-5 && ndc_to_voxel[4 + k].abs() < 1e-5;
                        flat && (ndc_to_voxel[12 + k] - spec.upper_voxel_bound[c] as f64).abs() < 1e-5 && g[k] * size[k] < ndc_to_voxel[12 + k]
                    });
                    origin[12 + k] = -g[k] * size[k] + if on_upper_face { -1e-4 } else { 1e-4 };
                }
                let ndc_to_texture = multiply4(&origin, &ndc_to_voxel).map(|x| x as f32);
                let mut grid = vec![0u64; rank];
                for (k, cd) in ps.chunk_display_dims.iter().enumerate() {
                    if let Some(c) = cd {
                        grid[*c] = g[k] as u64;
                    }
                }
                for &(c, chunk, _) in &ps.fixed {
                    grid[c] = chunk;
                }
                let mut clip = ChunkDraw::UNCLIPPED;
                for k in 0..3 {
                    if ps.chunk_display_dims[k].is_some() {
                        clip[0][k] = ((clip_lower[k] - g[k]) * size[k]) as f32;
                        clip[1][k] = ((clip_upper[k] - g[k]) * size[k]) as f32;
                    }
                }
                out.push(ChunkDraw {
                    scale: ps.scale,
                    grid,
                    chunk_display_dims: ps.chunk_display_dims,
                    fixed: ps.fixed.clone(),
                    rect,
                    ndc_to_texture,
                    clip,
                });
            }
        }
    }
    out
}

/// `modelClipBounds`: how far the position is inside a box (model `lower`
/// to `upper`) along the model dimensions not `shown`: 1 inside, falling to 0
/// a voxel outside; and the position in model coordinates. `None` if
/// `position` is shorter than the global rank.
fn position_clip(layer: &VolumeLayer, position: &[f32], lower: &[f64], upper: &[f64], shown: &[usize]) -> Option<(f64, Vec<f64>)> {
    let rank = layer.rank;
    let n = rank + 1;
    let model_rank = lower.len().min(rank);
    // The position in model coordinates.
    let layer_pos = layer.layer_position(position, 0.5)?;
    let (l2m, _) = matrix::inverse(&layer.model_to_layer, n);
    let model_pos: Vec<f64> = (0..rank).map(|c| (0..n).map(|k| l2m[k * n + c] * layer_pos[k]).sum()).collect();
    let mut clip = 1.0;
    for d in (0..model_rank).filter(|d| !shown.contains(d)) {
        let (a, b) = (lower[d].min(upper[d]), lower[d].max(upper[d]));
        let c = model_pos[d];
        clip *= (1.0 - (c.clamp(a, b) - c).abs()).max(0.0);
    }
    Some((clip, model_pos))
}

/// A bounding box annotation (`AxisAlignedBoundingBox`, in model
/// coordinates) as a panel shows it: the polygon where the slice plane cuts
/// the box, in panel pixels (logical, top-left origin), in order, and the
/// clip coefficient (`getMaxSubspaceClipCoefficient`: how far the position
/// lies outside the box along model dimensions the panel does not display,
/// fading over one model unit). `None` if the plane misses it.
///
/// Port of the cross-section face rendering of `src/annotation/bounding_box.ts`
/// (`getBoundingBoxPlaneIntersectionVertexPosition`) for boxes whose model
/// dimensions map to display dimensions (`projectModelVectorToSubspace`).
pub fn box_cross_section(
    layer: &VolumeLayer,
    lower: &[f64],
    upper: &[f64],
    info: &DisplayDimensionRenderInfo,
    position: &[f32],
    camera: &PanelCamera,
) -> Option<(Vec<[f64; 2]>, f64)> {
    let rank = layer.rank;
    let n = rank + 1;
    let m2l = &layer.model_to_layer;
    let model_rank = lower.len().min(rank);
    // Model dimensions shown along display dimensions.
    let display_of = |model_dim: usize| -> bool {
        (0..info.display_rank).any(|i| {
            let g = info.display_dimension_indices[i];
            layer
                .layer_to_global
                .iter()
                .position(|&x| x == Some(g as usize))
                .is_some_and(|ld| m2l[model_dim * n + ld] != 0.0)
        })
    };
    let shown: Vec<usize> = (0..model_rank).filter(|&d| display_of(d)).collect();
    let (clip, model_pos) = position_clip(layer, position, lower, upper, &shown)?;
    if clip == 0.0 {
        return None;
    }
    // Corners over the shown dimensions; the others at the position.
    let view_projection = camera.view_projection();
    let corner = |bits: usize| -> [f64; 3] {
        let mut model = model_pos.clone();
        for (k, &d) in shown.iter().enumerate() {
            model[d] = if bits >> k & 1 == 1 { lower[d].max(upper[d]) } else { lower[d].min(upper[d]) };
        }
        let mut display = [0.0; 3];
        for (i, slot) in display.iter_mut().enumerate().take(info.display_rank) {
            let g = info.display_dimension_indices[i] as usize;
            if let Some(ld) = layer.layer_to_global.iter().position(|&x| x == Some(g)) {
                *slot = (0..rank).map(|c| m2l[c * n + ld] * model[c]).sum::<f64>() + m2l[rank * n + ld];
            }
        }
        transform_point4(&view_projection, display)
    };
    let count = 1usize << shown.len();
    let mut corners: Vec<[f64; 3]> = (0..count).map(corner).collect();
    // Corners within float error of the plane are on it (a plane on a face
    // otherwise cuts it slightly askew).
    let depth = corners.iter().fold(0f64, |m, c| m.max(c[2].abs()));
    for c in &mut corners {
        if c[2].abs() < 1e-6 * depth {
            c[2] = 0.0;
        }
    }
    let mut points: Vec<[f64; 2]> = vec![];
    // Fewer than three shown dimensions: neuroglancer stretches the box over
    // the view's depth along the missing ones (`uModelSpaceBoundOffsets`), so
    // the plane cuts it in its outline — its corners, projected.
    let flat = shown.len() < 3;
    if flat {
        for c in &corners {
            if !points.iter().any(|o| (o[0] - c[0]).abs() < 1e-12 && (o[1] - c[1]).abs() < 1e-12) {
                points.push([c[0], c[1]]);
            }
        }
    }
    // Edges join corners differing in one shown dimension; keep where the
    // plane (NDC z = 0) crosses them.
    for i in (0..count).filter(|_| !flat) {
        for k in 0..shown.len() {
            let j = i | (1 << k);
            if j == i {
                continue;
            }
            let (p, q) = (corners[i], corners[j]);
            if (p[2] > 0.0 && q[2] > 0.0) || (p[2] < 0.0 && q[2] < 0.0) {
                continue;
            }
            let t = if p[2] == q[2] { 0.0 } else { p[2] / (p[2] - q[2]) };
            let x = p[0] + (q[0] - p[0]) * t;
            let y = p[1] + (q[1] - p[1]) * t;
            // Within float error of a point already found (a plane on a
            // face meets its corners along several edges).
            if !points.iter().any(|o| (o[0] - x).abs() < 1e-6 && (o[1] - y).abs() < 1e-6) {
                points.push([x, y]);
            }
        }
    }
    if points.len() < 2 {
        return None;
    }
    // A convex polygon: order around the centroid.
    let cx = points.iter().map(|p| p[0]).sum::<f64>() / points.len() as f64;
    let cy = points.iter().map(|p| p[1]).sum::<f64>() / points.len() as f64;
    points.sort_by(|a, b| (a[1] - cy).atan2(a[0] - cx).total_cmp(&(b[1] - cy).atan2(b[0] - cx)));
    let pixels = points
        .iter()
        .map(|p| [(p[0] + 1.0) / 2.0 * camera.width, (1.0 - p[1]) / 2.0 * camera.height])
        .collect();
    Some((pixels, clip))
}

/// A line segment in display coordinates.
pub type Edge3 = ([f64; 3], [f64; 3]);

/// The 12 edges of a bounding box annotation in display coordinates, as the
/// 3D panel draws them (`PerspectiveViewRenderHelper.drawEdges`), and the clip
/// coefficient (`getMaxSubspaceClipCoefficient` over model dimensions the
/// view does not display). Display dimensions the box has no model dimension
/// for span the view frustum (`getViewFrustumWorldBounds`, widened by its own
/// extent): `inv_view_projection` is the 3D camera's.
pub fn box_edges_3d(
    layer: &VolumeLayer,
    lower: &[f64],
    upper: &[f64],
    info: &DisplayDimensionRenderInfo,
    position: &[f32],
    inv_view_projection: &[f32; 16],
) -> Option<(Vec<Edge3>, f64)> {
    let rank = layer.rank;
    let n = rank + 1;
    let m2l = &layer.model_to_layer;
    let model_rank = lower.len().min(rank);
    // Model dimensions the display depends on (the chunk display
    // dimensions), and display dimensions none of them reach.
    let feeds = |d: usize, i: usize| -> bool {
        let g = info.display_dimension_indices[i] as usize;
        layer.layer_to_global.iter().position(|&x| x == Some(g)).is_some_and(|ld| m2l[d * n + ld] != 0.0)
    };
    let shown: Vec<usize> = (0..model_rank).filter(|&d| (0..info.display_rank).any(|i| feeds(d, i))).take(3).collect();
    let missing: Vec<usize> = (0..3).filter(|&i| i >= info.display_rank || !shown.iter().any(|&d| feeds(d, i))).collect();
    // The box's three axes: shown model dimensions, then missing display
    // dimensions (spanning the frustum).
    #[derive(Clone, Copy)]
    enum Axis {
        Model(usize),
        Display(usize),
    }
    let axes: Vec<Axis> = shown.iter().map(|&d| Axis::Model(d)).chain(missing.iter().map(|&i| Axis::Display(i))).take(3).collect();
    let (clip, model_pos) = position_clip(layer, position, lower, upper, &shown)?;
    if clip == 0.0 {
        return None;
    }
    // `getViewFrustumWorldBounds`.
    let mut world = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for bits in 0..8 {
        let ndc = [if bits & 1 == 1 { 1.0 } else { -1.0 }, if bits & 2 == 2 { 1.0 } else { -1.0 }, if bits & 4 == 4 { 1.0 } else { -1.0 }, 1.0];
        let p = ng_core::glm::vec4_transform_mat4(ndc, inv_view_projection);
        for k in 0..3 {
            let v = p[k] / p[3];
            world[k] = world[k].min(v);
            world[k + 3] = world[k + 3].max(v);
        }
    }
    // A display coordinate of a model point, with the box's display extent.
    let display_of = |model: &[f64]| -> [f64; 3] {
        let mut out = [0.0; 3];
        for (i, slot) in out.iter_mut().enumerate().take(info.display_rank) {
            let g = info.display_dimension_indices[i] as usize;
            if let Some(ld) = layer.layer_to_global.iter().position(|&x| x == Some(g)) {
                *slot = (0..rank).map(|c| m2l[c * n + ld] * model[c]).sum::<f64>() + m2l[rank * n + ld];
            }
        }
        out
    };
    let corner = |bits: usize| -> [f64; 3] {
        let mut model = model_pos.clone();
        for (k, axis) in axes.iter().enumerate() {
            if let Axis::Model(d) = *axis {
                model[d] = if bits >> k & 1 == 1 { lower[d].max(upper[d]) } else { lower[d].min(upper[d]) };
            }
        }
        let mut p = display_of(&model);
        for (k, axis) in axes.iter().enumerate() {
            if let Axis::Display(i) = *axis {
                let delta = (world[i + 3] - world[i]).abs();
                p[i] = if bits >> k & 1 == 1 { world[i + 3] + delta } else { world[i] - delta };
            }
        }
        p
    };
    // The edge order of `edgeBoxCornerOffsetData`: along z, then y, then x.
    let edges = [
        (0b000, 0b100), (0b001, 0b101), (0b010, 0b110), (0b011, 0b111),
        (0b000, 0b010), (0b100, 0b110), (0b001, 0b011), (0b101, 0b111),
        (0b000, 0b001), (0b100, 0b101), (0b010, 0b011), (0b110, 0b111),
    ];
    Some((edges.iter().map(|&(a, b)| (corner(a), corner(b))).collect(), clip))
}

/// What volume rendering draws of a layer (`forEachVisibleVolumeRenderingChunk`):
/// the scale chosen for the depth samples, how far apart they are
/// physically and how many the scale would want, the source's clip bounds
/// and chunk size in chunk display voxels, and the chunks within the
/// frustum (full grid positions).
#[derive(Debug, Clone)]
pub struct VolumeRenderingChunks {
    pub source: PanelSource,
    pub physical_spacing: f64,
    pub optimal_samples: f64,
    pub lower_clip: [f64; 3],
    pub upper_clip: [f64; 3],
    pub chunk_size: [f64; 3],
    pub chunks: Vec<Vec<u64>>,
    /// `histogramInformation`: each scale's physical spacing and optimal
    /// samples, coarsest first, and the index of the one chosen.
    pub scales: Vec<(f64, f64)>,
    pub active: usize,
}

/// `getViewFrustumDepthRange`.
pub fn view_frustum_depth_range(projection: &Mat4) -> f64 {
    let p = |i: usize| projection[i] as f64;
    if p(15) == 1.0 {
        return 2.0 / p(10).abs();
    }
    let (a, b) = (p(10), p(14));
    let near = (2.0 * b) / (2.0 * a - 2.0);
    let far = ((a - 1.0) * near) / (a + 1.0);
    (far - near).abs()
}

fn det3(m: &[f64; 16]) -> f64 {
    let g = |c: usize, r: usize| m[c * 4 + r];
    g(0, 0) * (g(1, 1) * g(2, 2) - g(2, 1) * g(1, 2)) - g(1, 0) * (g(0, 1) * g(2, 2) - g(2, 1) * g(0, 2)) + g(2, 0) * (g(0, 1) * g(1, 2) - g(1, 1) * g(0, 2))
}

/// `forEachVisibleVolumeRenderingChunk` over `sources` (`panel_sources`):
/// the finest scale whose voxels, seen from the view, are no smaller than
/// the depth range over `depth_samples` cubed; its chunks within the frustum
/// of `view_projection` (`forEachVolumetricChunkWithinFrustum`).
pub fn volume_rendering_chunks(layer: &VolumeLayer, sources: &[Option<PanelSource>], info: &DisplayDimensionRenderInfo, projection: &Mat4, inv_view: &Mat4, view_projection: &Mat4, depth_samples: f64) -> Option<VolumeRenderingChunks> {
    let available: Vec<&PanelSource> = sources.iter().flatten().collect();
    if available.is_empty() {
        return None;
    }
    let depth_range = view_frustum_depth_range(projection);
    let target = (depth_range / depth_samples).powi(3);
    let view = inverse4(&inv_view.map(|x| x as f64));
    let view_det = det3(&view);
    let canonical_to_physical: f64 = info.voxel_physical_scales.iter().product();
    let view_volume = |ps: &PanelSource| (det3(&ps.chunk_to_display) * view_det).abs();
    let mut best = available.len() - 1;
    let mut scales = vec![];
    for k in (0..available.len()).rev() {
        let volume = view_volume(available[k]);
        scales.push(((volume * canonical_to_physical / view_det).cbrt(), depth_range / volume.cbrt()));
        if volume - target >= 0.0 {
            best = k;
        }
    }
    let ps = available[best].clone();
    let best_volume = view_volume(&ps);
    let physical_spacing = (best_volume * canonical_to_physical / view_det).cbrt();
    let optimal_samples = depth_range / best_volume.cbrt();
    // Bounds along the displayed chunk dimensions, in voxels and chunks.
    let src = &layer.scales[ps.scale];
    let mut lower_clip = [0.0; 3];
    let mut upper_clip = [1.0; 3];
    let mut chunk_size = [1.0; 3];
    for (i, cd) in ps.chunk_display_dims.iter().enumerate() {
        let Some(c) = *cd else { continue };
        lower_clip[i] = src.lower_clip_bound.as_ref().map_or(src.spec.lower_voxel_bound[c] as f64, |b| b[c] as f64);
        upper_clip[i] = src.upper_clip_bound.as_ref().map_or(src.spec.upper_voxel_bound[c] as f64, |b| b[c] as f64);
        chunk_size[i] = src.spec.chunk_data_size[c] as f64;
    }
    let lower_chunk: [f64; 3] = std::array::from_fn(|i| (lower_clip[i] / chunk_size[i]).floor());
    let upper_chunk: [f64; 3] = std::array::from_fn(|i| (upper_clip[i] / chunk_size[i]).ceil());
    // The view-projection of chunk-grid coordinates.
    let mut scaled = ps.chunk_to_display;
    for (i, s) in chunk_size.iter().enumerate() {
        for j in 0..4 {
            scaled[4 * i + j] *= s;
        }
    }
    let vp = view_projection.map(|x| x as f64);
    let mvp = multiply4(&vp, &scaled).map(|x| x as f32);
    let planes = ng_datasource::multiscale_mesh::frustum_planes(&mvp);
    let base: Vec<u64> = {
        let mut g = vec![0u64; layer.source_rank];
        for &(c, chunk, _) in &ps.fixed {
            if c < g.len() {
                g[c] = chunk;
            }
        }
        g
    };
    let chunks = cells_within_frustum(lower_chunk, upper_chunk, &planes, &ps.chunk_display_dims, &base);
    Some(VolumeRenderingChunks { source: ps, physical_spacing, optimal_samples, lower_clip, upper_clip, chunk_size, chunks, scales, active: best })
}

/// `forEachVolumetricChunkWithinFrustum`: the cells of the grid box
/// `[lower, upper)` (chunk display dimensions) within `planes`, as full grid
/// positions (`base` with the displayed dimensions `dims` set), in the
/// order neuroglancer visits them.
pub fn cells_within_frustum(lower: [f64; 3], upper: [f64; 3], planes: &[f32; 24], dims: &[Option<usize>; 3], base: &[u64]) -> Vec<Vec<u64>> {
    fn recurse(lower: [f64; 3], upper: [f64; 3], planes: &[f32; 24], dims: &[Option<usize>; 3], base: &[u64], out: &mut Vec<Vec<u64>>) {
        if !ng_datasource::multiscale_mesh::is_aabb_visible(lower, upper, planes) {
            return;
        }
        let mut split = 0;
        let mut split_size = (upper[0] - lower[0]).max(0.0);
        let mut volume = split_size;
        for i in 1..3 {
            let size = (upper[i] - lower[i]).max(0.0);
            volume *= size;
            if size > split_size {
                split_size = size;
                split = i;
            }
        }
        if volume == 0.0 {
            return;
        }
        if volume == 1.0 {
            let mut g = base.to_vec();
            for (i, d) in dims.iter().enumerate() {
                if let Some(c) = d {
                    g[*c] = lower[i] as u64;
                }
            }
            out.push(g);
            return;
        }
        let mid = lower[split] + (split_size / 2.0).floor();
        let (mut a_upper, mut b_lower) = (upper, lower);
        a_upper[split] = mid;
        b_lower[split] = mid;
        recurse(lower, a_upper, planes, dims, base, out);
        recurse(b_lower, upper, planes, dims, base, out);
    }
    let mut out = vec![];
    recurse(lower, upper, planes, dims, base, &mut out);
    out
}

/// `getViewFrustumVolume`.
pub fn view_frustum_volume(projection: &[f64; 16]) -> f64 {
    let p = projection;
    if p[15] == 1.0 {
        return (2.0 / p[10].abs()) * (2.0 / p[0].abs()) * (2.0 / p[5].abs());
    }
    let (a, b) = (p[10], p[14]);
    let near = (2.0 * b) / (2.0 * a - 2.0);
    let far = ((a - 1.0) * near) / (a + 1.0);
    let base_area = 4.0 / (p[0] * p[5]);
    (base_area / 3.0) * (far.abs().powi(3) - near.abs().powi(3))
}

/// A spatial index level of an annotation source as a chunk layout: the
/// map from its model units (along the chunk display dimensions `dims`) to
/// display coordinates, its cell size and grid there, the grid position of
/// the other dimensions at the current position (`None` if outside the
/// grid), its per-cell limit and the fraction of it in one slice of the
/// non-displayed dimensions.
#[derive(Debug, Clone)]
pub struct AnnotationLevelLayout {
    pub to_display: [f64; 16],
    pub dims: [Option<usize>; 3],
    pub chunk_size: [f64; 3],
    pub grid: [f64; 3],
    pub base: Option<Vec<u64>>,
    pub limit: f64,
    pub slice_fraction: f64,
}

/// A cell to draw: its level (finest first), grid position, the fraction
/// of its annotations to draw and the spacing the histogram records.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnotationCell {
    pub level: usize,
    pub cell: Vec<u64>,
    pub draw_fraction: f64,
    pub physical_spacing: f64,
    pub pixel_spacing: f64,
}

/// `forEachVisibleAnnotationChunk`: from the coarsest level, as long as the
/// annotations drawn are sparser than `render_scale_target` pixels apart,
/// each level's cells within the frustum of `projection · view` and the
/// fraction of each to draw. `levels` finest first; `size` the viewport.
pub fn visible_annotation_cells(levels: &[AnnotationLevelLayout], projection: &[f64; 16], view: &[f64; 16], size: (f64, f64), canonical_to_physical: f64, render_scale_target: f64) -> Vec<AnnotationCell> {
    let mut out = vec![];
    let Some(base_level) = levels.first() else { return out };
    let view_det = det3(view).abs();
    let view_frustum = view_frustum_volume(projection) / view_det * canonical_to_physical;
    let mut source_volume = det3(&base_level.to_display).abs() * canonical_to_physical;
    for i in 0..3 {
        source_volume *= base_level.grid[i] * base_level.chunk_size[i];
    }
    let effective_volume = source_volume.min(view_frustum);
    let viewport_area = size.0 * size.1;
    let target_num = viewport_area / render_scale_target.powi(2);
    let density_target = target_num / effective_volume;
    let view_projection = multiply4(projection, view);
    let mut total_density = 0.0;
    for (index, level) in levels.iter().enumerate().rev() {
        if total_density >= density_target {
            break;
        }
        let physical_volume = level.chunk_size.iter().product::<f64>() * det3(&level.to_display).abs() * canonical_to_physical;
        let density = level.limit * level.slice_fraction / physical_volume;
        let new_total = total_density + density;
        let physical_spacing = (1.0 / new_total).cbrt();
        let pixel_spacing = (viewport_area / (new_total * effective_volume)).sqrt();
        let desired = (density_target - total_density) * physical_volume / level.slice_fraction;
        let draw_fraction = (desired / level.limit).min(1.0);
        if let Some(base) = &level.base {
            let mut scaled = level.to_display;
            for (i, s) in level.chunk_size.iter().enumerate() {
                for j in 0..4 {
                    scaled[4 * i + j] *= s;
                }
            }
            let mvp = multiply4(&view_projection, &scaled).map(|x| x as f32);
            let planes = ng_datasource::multiscale_mesh::frustum_planes(&mvp);
            for cell in cells_within_frustum([0.0; 3], level.grid, &planes, &level.dims, base) {
                out.push(AnnotationCell { level: index, cell, draw_fraction, physical_spacing, pixel_spacing });
            }
        }
        total_density = new_total;
    }
    out
}

/// `getNearIsotropicBlockSize`: from 1 voxel each, doubling the
/// dimension that is smallest in view units (`effective`: each chunk
/// dimension's length there; 0 for one not shown) while the block stays
/// within `2^max_log2` voxels, the volume's extent (rounded down to a power
/// of two) and `max_block`.
pub fn near_isotropic_block_size(effective: &[f64], lower: &[f64], upper: &[f64], max_block: &[u32], max_log2: u32) -> Vec<u32> {
    let rank = effective.len();
    let mut size = vec![1u32; rank];
    let bound: Vec<f64> = (0..rank)
        .map(|c| {
            if effective[c] == 0.0 {
                return size[c] as f64;
            }
            let extent = 2f64.powf((upper[c] - lower[c]).log2().floor());
            extent.min(max_block[c] as f64)
        })
        .collect();
    for _ in 0..max_log2 {
        let next = (0..rank).filter(|&c| (size[c] as f64) < bound[c]).min_by(|&a, &b| (size[a] as f64 * effective[a]).total_cmp(&(size[b] as f64 * effective[b])));
        match next {
            Some(c) => size[c] *= 2,
            None => break,
        }
    }
    size
}

/// The compressed segmentation block neuroglancer transcodes scale `s` of
/// a segmentation layer into (`shouldTranscodeToCompressedSegmentation`:
/// `uint32`/`uint64` chunks not already so encoded, 3-d or with one
/// channel; `makeVolumeChunkSpecificationWithDefaultCompression`'s 512
/// voxels near-isotropic in the display).
pub fn transcode_block(layer: &VolumeLayer, s: usize, info: &DisplayDimensionRenderInfo) -> Option<[usize; 3]> {
    use ng_core::DataType;
    let src = layer.scales.get(s)?;
    let spec = &src.spec;
    if layer.volume_type != ng_datasource::model::VolumeType::Segmentation || !matches!(spec.data_type, DataType::Uint32 | DataType::Uint64) || spec.compressed_segmentation_block_size.is_some() {
        return None;
    }
    let rank = spec.rank;
    if rank != 3 && !(rank == 4 && spec.chunk_data_size[3] == 1) {
        return None;
    }
    let n = layer.rank + 1;
    let c2l = layer.chunk_to_layer(s);
    let effective: Vec<f64> = (0..rank)
        .map(|c| {
            (0..info.display_rank)
                .filter_map(|i| {
                    let g = info.display_dimension_indices[i];
                    let ld = layer.layer_to_global.iter().position(|&x| x == Some(g as usize))?;
                    Some(c2l[c * n + ld].powi(2))
                })
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let lower: Vec<f64> = spec.lower_voxel_bound.iter().map(|&v| v as f64).collect();
    let upper: Vec<f64> = spec.upper_voxel_bound.iter().map(|&v| v as f64).collect();
    let b = near_isotropic_block_size(&effective, &lower, &upper, &spec.chunk_data_size, 9);
    Some([b[0] as usize, b[1] as usize, b[2] as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annotation_cells_follow_the_density_target() {
        // A 100 × 100 px orthographic view ±10 deep over a 100 × 100 × 20
        // box: a coarse level of one cell (limit 4), a fine one of 2 × 2
        // (limit 10 each).
        let mut projection = [0.0; 16];
        projection[0] = 2.0 / 100.0;
        projection[5] = 2.0 / 100.0;
        projection[10] = -1.0 / 10.0;
        projection[15] = 1.0;
        let mut view = [0.0; 16];
        for i in 0..4 {
            view[5 * i] = 1.0;
        }
        let mut to_display = view;
        to_display[12..15].copy_from_slice(&[-50.0, -50.0, -10.0]);
        let level = |chunk: [f64; 3], grid: [f64; 3], limit: f64| AnnotationLevelLayout { to_display, dims: [Some(0), Some(1), Some(2)], chunk_size: chunk, grid, base: Some(vec![0; 3]), limit, slice_fraction: 1.0 };
        let levels = [level([50.0, 50.0, 20.0], [2.0, 2.0, 1.0], 10.0), level([100.0, 100.0, 20.0], [1.0, 1.0, 1.0], 4.0)];
        // 8 px apart wants 156 here: both levels in full, the coarse first.
        let cells = visible_annotation_cells(&levels, &projection, &view, (100.0, 100.0), 1.0, 8.0);
        assert_eq!(cells.len(), 5);
        assert_eq!((cells[0].level, cells[0].draw_fraction), (1, 1.0));
        assert!(cells[1..].iter().all(|c| c.level == 0 && c.draw_fraction == 1.0));
        // 80 px apart wants 1.5625: part of the coarse cell, nothing finer.
        let cells = visible_annotation_cells(&levels, &projection, &view, (100.0, 100.0), 1.0, 80.0);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].draw_fraction, 1.5625 / 4.0);
    }
}
