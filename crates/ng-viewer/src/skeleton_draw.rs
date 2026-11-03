//! Skeletons as the 3D and slice panels draw them: line segments, thinned
//! where they would be denser than a pixel.

use super::*;

impl Viewer {
    pub(crate) fn skeleton_draws(&mut self, hovered: &[Option<u64>], info: &ng_core::navigation::DisplayDimensionRenderInfo, view_2d: bool) -> Vec<SkeletonDraw> {
        let mut out = vec![];
        let mut wanted = vec![];
        // the segments drawn as meshes (loaded, within the cap)
        let meshed = if self.objects.draw_skeletons_with_meshes { Default::default() } else { self.scene_meshed_segments() };
        for (li, managed) in self.layers.iter().enumerate() {
            if !managed.shown() {
                continue;
            }
            let (Some(state), Some(LayerKind::Segmentation(display))) = (managed.render.as_ref(), managed.kind()) else { continue };
            let sources = managed.skeleton_sources();
            if sources.is_empty() {
                continue;
            }
            let alpha = managed.spec.json.get("objectAlpha").and_then(Value::as_f64).unwrap_or(1.0).min(1.0) as f32;
            if alpha <= 0.0 {
                continue;
            }
            let (points, width) = skeleton_options(&managed.spec.json, view_2d);
            // loaded, not drawn, where its mesh is (decided per segment below)
            let hide_under_mesh = !self.objects.draw_skeletons_with_meshes && !managed.mesh_sources().is_empty();
            let segments = managed.segments();
            for (k, source) in sources.iter().enumerate() {
                let Some(model) = mesh_model_matrix(&state.layer, &source.to_model, info) else { continue };
                for &(id, visible) in segments.iter() {
                    if visible {
                        let color = object_color(display, hovered.get(li).copied().flatten(), id, alpha);
                        // under its mesh (one loaded and drawn), it adds nothing to see
                        let hidden = hide_under_mesh && meshed.contains(&(li, id));
                        wanted.push(((managed.id, k, id), model, color, points, width, hidden));
                    }
                }
            }
        }
        let count = std::mem::replace(&mut self.skeletons_counted, self.frame) != self.frame;
        // each drawn skeleton's share of the budget, a power of two
        let drawn = wanted.iter().filter(|w| !w.5).count().max(1);
        let mut cap = self.objects.skeleton_draw_edges;
        if self.objects.skeleton_total_edges > 0 {
            let share = (self.objects.skeleton_total_edges / drawn).max(256);
            let share = 1usize << (usize::BITS - 1 - share.leading_zeros());
            cap = if cap == 0 { share } else { cap.min(share) };
        }
        for (key, model, color, points, width, hidden) in wanted {
            self.skeletons.request(&key, 0.0);
            let ready = self.skeletons.get(&key);
            if let Some(l) = self.layers.iter_mut().find(|l| l.id == key.0).filter(|_| count) {
                l.progress.0 += 1;
                l.progress.1 += u32::from(ready.is_some());
            }
            if hidden || !self.objects.skeletons_drawn {
                continue;
            }
            if let Some(mut data) = ready.filter(|d| d.num_vertices() > 0) {
                // a detailed skeleton drawn thinned (kept while the skeleton is the same)
                if cap > 0 && data.indices.len() / 2 > cap {
                    let fresh = match self.thinned_skeletons.get(&key) {
                        Some((from, at, thin)) if Arc::ptr_eq(from, &data) && *at == cap => Some(thin.clone()),
                        _ => None,
                    };
                    let thin = fresh.unwrap_or_else(|| {
                        let t = Arc::new(thin_skeleton(&data, cap));
                        self.thinned_skeletons.insert(key, (data.clone(), cap, t.clone()));
                        t
                    });
                    data = thin;
                }
                out.push(SkeletonDraw { segment: (key.0, key.2), model, color, data, points, width });
            }
        }
        out
    }
}

/// `SkeletonRenderingOptions`' view-specific options (`mode2d` and
/// `lineWidth2d`, or the 3D ones): whether nodes are drawn as points, and
/// the line width in pixels. Invalid values are the defaults, as
/// `TrackableValue.restoreState` resets them.
fn skeleton_options(json: &serde_json::Map<String, Value>, view_2d: bool) -> (bool, f64) {
    let o = json.get("skeletonRendering").and_then(Value::as_object);
    let (mode_key, width_key, default_points, default_width) = if view_2d { ("mode2d", "lineWidth2d", true, 2.0) } else { ("mode3d", "lineWidth3d", false, 1.0) };
    // `verifyEnumString`: case-insensitive.
    let points = match o.and_then(|o| o.get(mode_key)).and_then(Value::as_str).map(str::to_ascii_uppercase).as_deref() {
        Some("LINES") => false,
        Some("LINES_AND_POINTS") => true,
        _ => default_points,
    };
    let width = o.and_then(|o| o.get(width_key)).and_then(Value::as_f64).filter(|w| w.is_finite() && *w > 0.0).unwrap_or(default_width);
    (points, width)
}

/// A segment's skeleton to draw: its model matrix to the display
/// dimensions, colour, data, and the view's options.
pub(crate) struct SkeletonDraw {
    /// The layer and segment (for picking).
    pub(crate) segment: (LayerId, u64),
    model: [f32; 16],
    pub(crate) color: [f32; 4],
    data: Arc<ng_datasource::skeleton::SkeletonData>,
    points: bool,
    width: f64,
}

impl SkeletonDraw {
    /// `drawSkeleton`: each edge a line, each vertex a circle (`max(5, 2 ·
    /// width)` across with points, else `width`), without border; feathered
    /// 1 px in cross-sections, not in 3D. `to_clip` maps display coordinates
    /// to GL clip coordinates.
    pub(crate) fn marks(&self, to_clip: &dyn Fn([f64; 3]) -> [f64; 4], feather: f64, lines: &mut Vec<ng_render::annotations::Line>, circles: &mut Vec<ng_render::annotations::Circle>) {
        let m = &self.model;
        let clips: Vec<[f64; 4]> = self
            .data
            .positions
            .chunks_exact(3)
            .map(|p| {
                let d = ng_core::glm::vec4_transform_mat4([p[0] as f64, p[1] as f64, p[2] as f64, 1.0], m);
                to_clip([d[0], d[1], d[2]])
            })
            .collect();
        for e in self.data.indices.chunks_exact(2) {
            let (Some(&a), Some(&b)) = (clips.get(e[0] as usize), clips.get(e[1] as usize)) else { continue };
            lines.push(ng_render::annotations::Line { a, b, width: self.width, feather, color: self.color, color_b: self.color, pick: 0 });
        }
        // the vertices as points only when asked for (`linesAndPoints`): one
        // per vertex otherwise doubles the work for nothing to see
        if !self.points {
            return;
        }
        let diameter = (self.width * 2.0).max(5.0);
        for &center in &clips {
            circles.push(ng_render::annotations::Circle { center, diameter, border_width: 0.0, feather, interior: self.color, border: self.color, pick: 0 });
        }
    }
}

/// A skeleton thinned for drawing to about `max_edges` edges: vertices with
/// other than two neighbours (branch points, tips) kept, and every k-th along
/// the unbranched runs between them, each joined to the previous kept one
/// along its run. Vertex attributes kept with their vertices.
pub(crate) fn thin_skeleton(d: &ng_datasource::skeleton::SkeletonData, max_edges: usize) -> ng_datasource::skeleton::SkeletonData {
    let n = d.positions.len() / 3;
    let edges = d.indices.len() / 2;
    let every = edges.div_ceil(max_edges.max(1)).max(1);
    let mut adj: Vec<Vec<u32>> = vec![vec![]; n];
    for e in d.indices.chunks_exact(2) {
        let (a, b) = (e[0] as usize, e[1] as usize);
        if a < n && b < n && a != b {
            adj[a].push(b as u32);
            adj[b].push(a as u32);
        }
    }
    let mut keep: Vec<bool> = adj.iter().map(|a| a.len() != 2).collect();
    let mut walked = vec![false; n];
    let mut new_edges: Vec<(u32, u32)> = vec![];
    // runs from each kept vertex out along each neighbour
    let starts: Vec<usize> = (0..n).filter(|&v| keep[v]).collect();
    let run = |start: usize, keep: &mut Vec<bool>, walked: &mut Vec<bool>, new_edges: &mut Vec<(u32, u32)>| {
        for &first in &adj[start].clone() {
            let (mut prev, mut cur, mut last, mut steps) = (start, first as usize, start, 0usize);
            if walked[cur] && !keep[cur] {
                continue;
            }
            loop {
                steps += 1;
                if keep[cur] {
                    // a run between two kept vertices may be walked from both
                    // ends: its last edge then twice, which only draws a line twice
                    new_edges.push((last as u32, cur as u32));
                    break;
                }
                walked[cur] = true;
                if steps >= every {
                    keep[cur] = true;
                    new_edges.push((last as u32, cur as u32));
                    last = cur;
                    steps = 0;
                }
                let next = adj[cur].iter().map(|&x| x as usize).find(|&x| x != prev);
                match next {
                    Some(x) if !(walked[x] && !keep[x]) => {
                        prev = cur;
                        cur = x;
                    }
                    _ => break,
                }
            }
        }
    };
    for s in starts {
        run(s, &mut keep, &mut walked, &mut new_edges);
    }
    // a run that closes on itself (no vertex but degree 2): from its first vertex
    for v in 0..n {
        if !walked[v] && !keep[v] && !adj[v].is_empty() {
            keep[v] = true;
            run(v, &mut keep, &mut walked, &mut new_edges);
        }
    }
    let mut index = vec![u32::MAX; n];
    let mut positions = vec![];
    let mut kept = vec![];
    for v in 0..n {
        if keep[v] {
            index[v] = kept.len() as u32;
            kept.push(v);
            positions.extend_from_slice(&d.positions[3 * v..3 * v + 3]);
        }
    }
    let indices = new_edges.iter().filter(|(a, b)| a != b).flat_map(|&(a, b)| [index[a as usize], index[b as usize]]).collect();
    let attributes = d
        .attributes
        .iter()
        .map(|bytes| {
            let stride = bytes.len().checked_div(n).unwrap_or(0);
            kept.iter().flat_map(|&v| bytes[v * stride..(v + 1) * stride].iter().copied()).collect()
        })
        .collect();
    ng_datasource::skeleton::SkeletonData { positions, indices, attributes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_skeleton_keeps_branches_and_tips() {
        // a chain 0..=10 with a branch 5-11-12
        let mut indices = vec![];
        for i in 0..10u32 {
            indices.extend_from_slice(&[i, i + 1]);
        }
        indices.extend_from_slice(&[5, 11, 11, 12]);
        let positions: Vec<f32> = (0..13).flat_map(|i| [i as f32, 0.0, 0.0]).collect();
        let d = ng_datasource::skeleton::SkeletonData { positions, indices, attributes: vec![(0..13u8).collect()] };
        let t = thin_skeleton(&d, 4);
        let kept: Vec<f32> = t.positions.chunks(3).map(|p| p[0]).collect();
        for must in [0.0, 5.0, 10.0, 12.0] {
            assert!(kept.contains(&must), "{must} kept: {kept:?}");
        }
        assert!(kept.len() < 13, "thinned: {kept:?}");
        // every kept vertex joined to something, and the attribute travels with it
        assert!(t.indices.len() / 2 >= kept.len() - 1);
        assert_eq!(t.attributes[0].len(), kept.len());
    }
}
