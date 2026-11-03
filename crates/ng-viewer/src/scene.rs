//! Scene mode: the viewer's 3D content drawn from a host renderer's camera,
//! for compositing into the host's own frame.
//!
//! An embedder with its own 3D scene (a brain atlas, a game engine) wants
//! neuroglancer's volumes inside that scene, not in neuroglancer's panels.
//! [`Viewer::render_scene`](crate::Viewer::render_scene) runs the normal
//! frame with three changes:
//!
//! - the cross-section panels are drawn offscreen at a fixed size
//!   ([`SceneView::slice_size`]) and shown as planes in 3D, as the 3D panel
//!   of the `4panel` layout shows them — all three, or the one a layout of
//!   one slice names (`xy-3d`, `xz`, …);
//! - the 3D panel is the host's size and uses the host's camera
//!   ([`HostCamera`]): its chunks, meshes and volume rendering are chosen and
//!   drawn for that view;
//! - nothing is composited into a layout: [`Viewer::composite_scene`](crate::Viewer::composite_scene)
//!   draws the 3D panel over the host's frame where it is nearer than the
//!   host's depth (or the host reads the colour and depth itself,
//!   [`SceneOutput`]).
//!
//! The slices are drawn again only when what they show changes (the
//! position, the scale, a chunk arriving, a shader): a host camera moving
//! around them costs only the 3D panel.
//!
//! Navigation (the position the slices cut through, the cross-section
//! scale, layers and shaders) stays the viewer's: set it with the state as
//! usual.
use ng_core::glm::Mat4;
use ng_core::matrix::{multiply4, try_inverse4, IDENTITY4};
use ng_core::navigation::DisplayDimensionRenderInfo;
use ng_core::panel::{NamedAxes, PerspectiveCamera};
use threers::renderer::PixelRect;

use crate::Panel;

/// The host's camera, column-major `[f64; 16]` matrices
/// ([`webgpu`](Self::webgpu) for a projection with WebGPU's depth range).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostCamera {
    /// Host world → eye (`camera.matrixWorldInverse`).
    pub view: [f64; 16],
    /// Eye → GL clip space, z from −w to w (`camera.projectionMatrix`).
    pub projection: [f64; 16],
    /// Host world → physical metres of the viewer's displayed dimensions,
    /// in display order (an affine: where the data sits in the host's world).
    pub physical_from_world: [f64; 16],
}

impl HostCamera {
    /// From a projection whose clip z runs 0..w (WebGPU's, threers'): taken
    /// to the GL form with the same window depth (`z_gl = 2 z − w`, so
    /// `(z_gl / w + 1) / 2 = z / w`) — the depth neuroglancer writes is then
    /// the host's, compared directly.
    pub fn webgpu(view: [f64; 16], projection: [f64; 16], physical_from_world: [f64; 16]) -> Self {
        let mut gl = projection;
        for c in 0..4 {
            gl[c * 4 + 2] = 2.0 * projection[c * 4 + 2] - projection[c * 4 + 3];
        }
        Self { view, projection: gl, physical_from_world }
    }

    /// An orthographic projection (w = 1).
    pub fn orthographic(&self) -> bool {
        self.projection[11] == 0.0 && self.projection[15] == 1.0
    }

    /// A perspective projection's near and far planes (eye distances).
    fn near_far(&self) -> Option<(f64, f64)> {
        let (a, b) = (self.projection[10], self.projection[14]);
        let (n, f) = (b / (a - 1.0), b / (a + 1.0));
        (!self.orthographic() && n.is_finite() && f.is_finite() && n > 0.0 && f > n).then_some((n, f))
    }

    /// The camera's position in the host's world (the view's inverse
    /// applied to the origin; the view rigid).
    fn eye(&self) -> [f64; 3] {
        let v = &self.view;
        let t = [v[12], v[13], v[14]];
        std::array::from_fn(|c| -(v[c * 4] * t[0] + v[c * 4 + 1] * t[1] + v[c * 4 + 2] * t[2]))
    }
}

/// The near and far planes of the host's projection and of the one the
/// viewer drew with (fitted to [`SceneView::bounds`]), eye distances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthRanges {
    pub host: (f64, f64),
    pub viewer: (f64, f64),
}

/// What [`render_scene`](crate::Viewer::render_scene) draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneView {
    /// The host's frame, physical pixels: the 3D output's size.
    pub width: u32,
    pub height: u32,
    /// Each cross-section's offscreen size, physical pixels (square). Its
    /// field of view is the cross-section scale's, as in a panel this size.
    pub slice_size: u32,
    pub camera: HostCamera,
    /// Where the content is, the host's world (centre, radius): a
    /// perspective camera's near and far planes are fitted to it.
    /// Volume rendering samples the whole near..far range, so with a host's
    /// own planes (near 1, far 5000 around a brain a few units across) the
    /// data is a sliver of it and the ray-march all but misses it.
    /// [`composite_scene`](crate::Viewer::composite_scene) brings the depth
    /// back to the host's range.
    pub bounds: Option<([f64; 3], f64)>,
}

impl SceneView {
    /// The camera drawn with: near/far fitted to `bounds` (perspective
    /// only), and the ranges before and after.
    pub(crate) fn fitted(&self) -> (HostCamera, Option<DepthRanges>) {
        let mut camera = self.camera;
        let Some(host) = camera.near_far() else { return (camera, None) };
        let mut viewer = host;
        if let Some((centre, radius)) = self.bounds {
            let eye = camera.eye();
            let dist = ((centre[0] - eye[0]).powi(2) + (centre[1] - eye[1]).powi(2) + (centre[2] - eye[2]).powi(2)).sqrt();
            // the near plane may come close: a camera inside the data looks
            // at tissue micrometres away (EM). A floor of far × 1e-6 keeps the
            // depth (Depth32Float) well-conditioned without clipping it.
            let f = (dist + radius).min(host.1);
            let n = (dist - radius).max(host.0).max(f * 1e-6);
            let f = f.max(n * 2.0);
            if f > n {
                camera.projection[10] = (f + n) / (n - f);
                camera.projection[14] = 2.0 * f * n / (n - f);
                viewer = (n, f);
            }
        }
        (camera, Some(DepthRanges { host, viewer }))
    }

    /// The panels drawn, in physical pixels: the slices (if shown: those of
    /// `axes`) then the 3D panel, each at the origin of its own offscreen
    /// target.
    pub(crate) fn panel_rects(&self, show_slices: bool, axes: &[NamedAxes]) -> Vec<(Panel, PixelRect)> {
        let s = self.slice_size.max(1);
        let mut out: Vec<(Panel, PixelRect)> = if show_slices {
            axes.iter().map(|&a| (Panel::Slice(a), PixelRect { x: 0, y: 0, width: s, height: s })).collect()
        } else {
            vec![]
        };
        out.push((Panel::Perspective, PixelRect { x: 0, y: 0, width: self.width.max(1), height: self.height.max(1) }));
        out
    }

    /// A panel's rectangle in logical pixels (scene mode draws at scale 1).
    pub(crate) fn logical_rect(&self, panel: Panel) -> PixelRect {
        match panel {
            Panel::Slice(_) => PixelRect { x: 0, y: 0, width: self.slice_size.max(1), height: self.slice_size.max(1) },
            Panel::Perspective => PixelRect { x: 0, y: 0, width: self.width.max(1), height: self.height.max(1) },
        }
    }

    /// The 3D panel's camera in display coordinates: the host's view and
    /// projection, with display coordinates (global units of the displayed
    /// dimensions) taken to the host's world.
    pub(crate) fn camera(&self, info: &DisplayDimensionRenderInfo) -> Option<PerspectiveCamera> {
        let c = &self.camera;
        // display → physical: each displayed dimension's metres per unit.
        let mut physical_from_display = IDENTITY4;
        for (k, scale) in info.voxel_physical_scales.iter().enumerate() {
            physical_from_display[k * 5] = *scale;
        }
        let world_from_physical = try_inverse4(&c.physical_from_world)?;
        let world_from_display = multiply4(&world_from_physical, &physical_from_display);
        let view = multiply4(&c.view, &world_from_display);
        let inv_view = try_inverse4(&view)?;
        let view_projection = multiply4(&c.projection, &view);
        Some(PerspectiveCamera { projection: to_f32(&c.projection), inv_view: to_f32(&inv_view), view_projection: to_f32(&view_projection) })
    }
}

/// The 3D output of [`render_scene`](crate::Viewer::render_scene), valid
/// until the next frame.
///
/// Both textures are **upside down** (row 0 at the bottom, as neuroglancer's
/// GL framebuffer): sample them at `(u, 1 − v)`.
pub struct SceneOutput<'a> {
    /// `Rgba8Unorm`, premultiplied alpha; 0 where nothing was drawn.
    pub color: &'a wgpu::TextureView,
    /// `Depth32Float`: window depth of the host's projection (GL clip z
    /// mapped to `(z / w + 1) / 2`), 1 where nothing was drawn. Bindable as a
    /// depth texture.
    pub depth: &'a wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

/// What [`Viewer::scene_probe`](crate::Viewer::scene_probe) found along a ray.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenePick {
    /// Where the ray meets the cross-section, in the host's world.
    pub world: [f64; 3],
    /// The same point, physical metres of the displayed dimensions.
    pub physical: [f64; 3],
    /// Which cross-section: 0 xy, 1 xz, 2 yz (`NamedAxes::ALL` order).
    pub plane: usize,
    /// Each layer's value there (name, value as neuroglancer's layer panel
    /// shows it: a segment ID, an intensity), for layers with data loaded.
    pub values: Vec<(String, String)>,
}

impl crate::Viewer {
    /// The cross-sections' crossing point (the navigation position), physical
    /// metres of the displayed dimensions.
    pub fn scene_position(&self) -> Option<[f64; 3]> {
        let info = self.nav.render_info();
        let pos = self.nav.position();
        (info.display_rank == 3).then(|| std::array::from_fn(|k| pos[info.display_dimension_indices[k] as usize] as f64 * info.voxel_physical_scales[k]))
    }

    /// Moves the cross-sections' crossing point to `physical` (metres of the
    /// displayed dimensions); the slices are drawn through it next frame.
    pub fn scene_set_position(&mut self, physical: [f64; 3]) -> bool {
        let scales = self.nav.render_info().voxel_physical_scales;
        if self.nav.render_info().display_rank != 3 {
            return false;
        }
        self.nav.update_display_position(|p| {
            for k in 0..3 {
                p[k] = (physical[k] / scales[k]) as f32;
            }
            true
        })
    }

    /// The cross-sections' scale: global units (canonical voxels) per slice
    /// pixel — neuroglancer's `crossSectionScale`.
    pub fn scene_zoom(&self) -> f64 {
        self.nav.zoom(ng_core::navigation::View::CrossSection)
    }

    /// Sets the cross-sections' scale (`crossSectionScale`: global units per
    /// slice pixel). A host whose camera zooms in on the slices sets it
    /// smaller, so the data streams in at the resolution the screen shows.
    pub fn scene_set_zoom(&mut self, units_per_pixel: f64) -> bool {
        if !(units_per_pixel.is_finite() && units_per_pixel > 0.0) {
            return false;
        }
        let before = self.scene_zoom();
        self.nav.set_zoom(ng_core::navigation::View::CrossSection, units_per_pixel);
        let gated = self.update_zoom_gates();
        (self.scene_zoom() - before).abs() > before * 1e-9 || gated
    }

    /// The nearest point where a ray from the host's camera (`origin`,
    /// `direction`, host world) meets one of the scene's cross-sections, within
    /// its field of view (`slice_size` pixels across, as drawn), and the
    /// layers' values there: what a hover or a click on the slices means.
    /// Each cross-section's normal in physical space, turned as drawn (the
    /// cross-section orientation), unit length: xy, xz, yz (`NamedAxes::ALL`
    /// order). A host moves a plane along its normal by moving the crossing
    /// point (`scene_set_position`) along it.
    pub fn scene_plane_normals(&self) -> Option<[[f64; 3]; 3]> {
        let info = self.nav.render_info();
        if info.display_rank != 3 {
            return None;
        }
        let q = self.nav.orientation(ng_core::navigation::View::CrossSection);
        let axis = |k: usize| -> [f64; 3] {
            let mut e = [0f32; 3];
            e[k] = 1.0;
            let v = ng_core::glm::vec3_transform_quat(e, q);
            let v = [v[0] as f64 * info.canonical_voxel_factors[0], v[1] as f64 * info.canonical_voxel_factors[1], v[2] as f64 * info.canonical_voxel_factors[2]];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-300);
            v.map(|x| x / n)
        };
        // xy's normal is the z axis; xz's, y; yz's, x
        Some([axis(2), axis(1), axis(0)])
    }

    pub fn scene_probe(&self, physical_from_world: &[f64; 16], origin: [f64; 3], direction: [f64; 3], slice_size: u32) -> Option<ScenePick> {
        let info = self.nav.render_info().clone();
        if info.display_rank != 3 {
            return None;
        }
        let m = physical_from_world;
        let to_phys = |p: [f64; 3], w: f64| -> [f64; 3] { std::array::from_fn(|r| m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r] * w) };
        let (o, d) = (to_phys(origin, 1.0), to_phys(direction, 0.0));
        let centre = self.scene_position()?;
        // Each cross-section's axes in physical space: the cross-section
        // orientation turns the display axes (scaled to physical lengths).
        let q = self.nav.orientation(ng_core::navigation::View::CrossSection);
        let axis = |k: usize| -> [f64; 3] {
            let mut e = [0f32; 3];
            e[k] = 1.0;
            let v = ng_core::glm::vec3_transform_quat(e, q);
            let v = [v[0] as f64 * info.canonical_voxel_factors[0], v[1] as f64 * info.canonical_voxel_factors[1], v[2] as f64 * info.canonical_voxel_factors[2]];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-300);
            v.map(|x| x / n)
        };
        let (x, y, z) = (axis(0), axis(1), axis(2));
        // xy, xz, yz: the plane's two in-plane axes and its normal
        let planes = [(x, y, z), (x, z, y), (y, z, x)];
        let half = 0.5 * slice_size as f64 * self.nav.zoom(ng_core::navigation::View::CrossSection) * info.canonical_voxel_physical_size;
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let world_from_physical = try_inverse4(m)?;
        // only the cross-sections drawn (the layout may name one; none
        // when the slices are hidden; the host may hide any: `scene_planes`)
        let drawn: Vec<NamedAxes> = self.panel_rects().into_iter().filter_map(|(p, _)| if let Panel::Slice(a) = p { Some(a) } else { None }).collect();
        let mut best: Option<(f64, ScenePick)> = None;
        for (k, (u, v, n)) in planes.into_iter().enumerate() {
            if !drawn.contains(&NamedAxes::ALL[k]) || !self.scene_planes[k] {
                continue;
            }
            let denom = dot(n, d);
            if denom.abs() < 1e-30 {
                continue;
            }
            let t = dot(n, [centre[0] - o[0], centre[1] - o[1], centre[2] - o[2]]) / denom;
            if t <= 0.0 || best.as_ref().is_some_and(|(bt, _)| *bt <= t) {
                continue;
            }
            let p: [f64; 3] = std::array::from_fn(|i| o[i] + t * d[i]);
            let rel = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
            if dot(rel, u).abs() > half || dot(rel, v).abs() > half {
                continue;
            }
            // the point as a global position: the display dimensions moved
            let mut global = self.nav.position().to_vec();
            for i in 0..3 {
                global[info.display_dimension_indices[i] as usize] = (p[i] / info.voxel_physical_scales[i]) as f32;
            }
            let values: Vec<(String, String)> = (0..self.layers.len())
                .filter(|&i| self.layers[i].shown())
                .filter_map(|i| Some((self.layers[i].name().to_string(), self.layer_value_at(i, &global)?)))
                .collect();
            if values.is_empty() {
                continue;
            }
            let w: [f64; 3] = std::array::from_fn(|r| world_from_physical[r] * p[0] + world_from_physical[4 + r] * p[1] + world_from_physical[8 + r] * p[2] + world_from_physical[12 + r]);
            best = Some((t, ScenePick { world: w, physical: p, plane: k, values }));
        }
        best.map(|(_, pick)| pick)
    }
}

/// A shown segment's shape, in the scene's physical space (`ScenePick`'s):
/// its skeleton's vertices and edges, or — without a skeleton loaded — its
/// loaded mesh's vertices alone (`edges` empty).
#[derive(Debug, Clone, Default)]
pub struct SegmentGraph {
    /// x y z per vertex, physical units (nm)
    pub positions: Vec<f64>,
    /// pairs of vertex indices; empty for a mesh's vertices
    pub edges: Vec<u32>,
    pub skeleton: bool,
    /// a skeleton's radius per vertex (its `radius` attribute, the
    /// skeleton's units), where it has one; else empty
    pub radius: Vec<f32>,
}

impl crate::Viewer {
    /// The segments shown as objects (meshes, skeletons) in the scene: each
    /// visible segmentation layer's index, name and chosen segment ids.
    pub fn scene_shown_segments(&self) -> Vec<(usize, String, u64)> {
        let mut out = vec![];
        for (li, managed) in self.layers.iter().enumerate() {
            if !managed.shown() || !matches!(managed.kind(), Some(crate::LayerKind::Segmentation(_))) {
                continue;
            }
            let segments = managed.segments();
            for &(id, visible) in segments.iter() {
                if visible {
                    out.push((li, managed.name().to_string(), id));
                }
            }
        }
        out
    }

    /// Segment `id` of layer `layer`'s skeleton, if loaded, in physical space
    /// and thinned to about `max_edges` edges (`thin_skeleton`: branch points
    /// and tips kept) — what a host draws for it, batched with the others.
    pub fn scene_segment_skeleton(&self, layer: usize, id: u64, max_edges: usize) -> Option<SegmentGraph> {
        let info = self.nav.render_info().clone();
        if info.display_rank != 3 {
            return None;
        }
        let managed = self.layers.get(layer)?;
        let state = managed.render.as_ref()?;
        let scales = [info.voxel_physical_scales[0], info.voxel_physical_scales[1], info.voxel_physical_scales[2]];
        for (k, source) in managed.skeleton_sources().iter().enumerate() {
            let Some(model) = crate::render::mesh_model_matrix(&state.layer, &source.to_model, &info) else { continue };
            let Some(data) = self.skeletons.get(&(managed.id, k, id)) else { continue };
            if data.num_vertices() == 0 {
                continue;
            }
            let thin;
            let d = if data.indices.len() / 2 > max_edges.max(1) {
                thin = crate::skeleton_draw::thin_skeleton(&data, max_edges);
                &thin
            } else {
                &*data
            };
            let positions = d
                .positions
                .chunks_exact(3)
                .flat_map(|v| -> [f64; 3] {
                    let p = [v[0] as f64, v[1] as f64, v[2] as f64];
                    std::array::from_fn(|r| (model[r] as f64 * p[0] + model[4 + r] as f64 * p[1] + model[8 + r] as f64 * p[2] + model[12 + r] as f64) * scales[r])
                })
                .collect();
            return Some(SegmentGraph { positions, edges: d.indices.clone(), skeleton: true, radius: vec![] });
        }
        None
    }

    /// [`scene_segment_skeleton`](Self::scene_segment_skeleton), its
    /// vertices taken straight to the host's frame: `out_from_physical`
    /// (an affine, column-major) composed with the model matrix once, and
    /// each vertex through it — float32 xyz, ready for a vertex buffer.
    pub fn scene_segment_skeleton_into(&self, layer: usize, id: u64, max_edges: usize, out_from_physical: &[f64; 16]) -> Option<(Vec<f32>, Vec<u32>)> {
        let info = self.nav.render_info();
        if info.display_rank != 3 {
            return None;
        }
        let managed = self.layers.get(layer)?;
        let state = managed.render.as_ref()?;
        let scales = info.voxel_physical_scales;
        for (k, source) in managed.skeleton_sources().iter().enumerate() {
            let Some(model) = crate::render::mesh_model_matrix(&state.layer, &source.to_model, info) else { continue };
            let Some(data) = self.skeletons.get(&(managed.id, k, id)) else { continue };
            if data.num_vertices() == 0 {
                continue;
            }
            // physical_from_model: the model matrix, then each display
            // axis's metres per unit
            let mut physical = [0f64; 16];
            for c in 0..4 {
                for r in 0..4 {
                    physical[c * 4 + r] = model[c * 4 + r] as f64 * if r < 3 { scales[r] } else { 1.0 };
                }
            }
            let m = multiply4(out_from_physical, &physical);
            let thin;
            let d = if data.indices.len() / 2 > max_edges.max(1) {
                thin = crate::skeleton_draw::thin_skeleton(&data, max_edges);
                &thin
            } else {
                &*data
            };
            let positions = d
                .positions
                .chunks_exact(3)
                .flat_map(|v| -> [f32; 3] {
                    let p = [v[0] as f64, v[1] as f64, v[2] as f64];
                    std::array::from_fn(|r| (m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]) as f32)
                })
                .collect();
            return Some((positions, d.indices.clone()));
        }
        None
    }

    /// Segment `id` of layer `layer`'s colour as the layer draws it (rgb).
    pub fn scene_segment_color(&self, layer: usize, id: u64) -> Option<[f32; 3]> {
        let managed = self.layers.get(layer)?;
        let Some(crate::LayerKind::Segmentation(display)) = managed.kind() else { return None };
        let c = crate::render::object_color(display, None, id, 1.0);
        Some([c[0], c[1], c[2]])
    }

    /// Segment `id` of layer `layer` as a graph in physical space: its
    /// skeleton where its layer has one and it is loaded; else the vertices
    /// of its loaded mesh (single-resolution, or the multi-resolution
    /// fragments loaded so far, each at its level), at most `max_points` of
    /// them (evenly kept). `None` while nothing of it is loaded.
    pub fn scene_segment_graph(&self, layer: usize, id: u64, max_points: usize) -> Option<SegmentGraph> {
        let info = self.nav.render_info().clone();
        if info.display_rank != 3 {
            return None;
        }
        let managed = self.layers.get(layer)?;
        let state = managed.render.as_ref()?;
        let scales = [info.voxel_physical_scales[0], info.voxel_physical_scales[1], info.voxel_physical_scales[2]];
        // a model point through the model matrix (display dimensions) to physical
        let phys = |m: &[f32; 16], v: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|r| (m[r] as f64 * v[0] + m[4 + r] as f64 * v[1] + m[8 + r] as f64 * v[2] + m[12 + r] as f64) * scales[r])
        };
        for (k, source) in managed.skeleton_sources().iter().enumerate() {
            let Some(model) = crate::render::mesh_model_matrix(&state.layer, &source.to_model, &info) else { continue };
            let Some(data) = self.skeletons.get(&(managed.id, k, id)) else { continue };
            if data.num_vertices() == 0 {
                continue;
            }
            let positions = data.positions.chunks_exact(3).flat_map(|v| phys(&model, [v[0] as f64, v[1] as f64, v[2] as f64])).collect();
            let radius = source
                .info
                .vertex_attributes
                .iter()
                .position(|a| a.id == "radius" && a.num_components == 1 && a.data_type == ng_core::DataType::Float32)
                .and_then(|i| data.attributes.get(i))
                .map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>())
                .filter(|r| r.len() == data.num_vertices())
                .unwrap_or_default();
            return Some(SegmentGraph { positions, edges: data.indices.clone(), skeleton: true, radius });
        }
        self.scene_segment_surface(layer, id, max_points)
    }

    /// Segment `id`'s mesh vertices (physical units), whatever of it is
    /// loaded — every level of a multi-resolution one — at most about
    /// `max_points`, as a graph without edges. None: no mesh loaded.
    pub fn scene_segment_surface(&self, layer: usize, id: u64, max_points: usize) -> Option<SegmentGraph> {
        let info = self.nav.render_info().clone();
        if info.display_rank != 3 {
            return None;
        }
        let managed = self.layers.get(layer)?;
        let state = managed.render.as_ref()?;
        let scales = [info.voxel_physical_scales[0], info.voxel_physical_scales[1], info.voxel_physical_scales[2]];
        let phys = |m: &[f32; 16], v: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|r| (m[r] as f64 * v[0] + m[4 + r] as f64 * v[1] + m[8 + r] as f64 * v[2] + m[12 + r] as f64) * scales[r])
        };
        let mut points: Vec<f64> = vec![];
        for (k, source) in managed.mesh_sources().iter().enumerate() {
            let Some(model) = crate::render::mesh_model_matrix(&state.layer, &source.to_model, &info) else { continue };
            if source.multiscale.is_none() {
                if let Some(mesh) = self.meshes.get(&(managed.id, k, id)) {
                    for v in mesh.positions.chunks_exact(3) {
                        points.extend(phys(&model, [v[0] as f64, v[1] as f64, v[2] as f64]));
                    }
                }
                continue;
            }
            let Some(manifest) = self.manifests.get(&(managed.id, k, id)) else { continue };
            let rows = manifest.octree.len() / 5;
            if rows == 0 {
                continue;
            }
            // each row's level: the root (the last row) at the coarsest, its
            // children (`[begin, end)`) a level finer
            let mut lod = vec![u32::MAX; rows];
            let mut stack = vec![(rows - 1, manifest.lod_scales.len().saturating_sub(1) as u32)];
            while let Some((r, l)) = stack.pop() {
                if lod[r] != u32::MAX {
                    continue;
                }
                lod[r] = l;
                let (begin, end) = (manifest.octree[r * 5 + 3] as usize, (manifest.octree[r * 5 + 4] & 0x7fff_ffff) as usize);
                if l > 0 {
                    for c in begin..end.min(rows) {
                        stack.push((c, l - 1));
                    }
                }
            }
            for (row, &l) in lod.iter().enumerate() {
                if l == u32::MAX {
                    continue;
                }
                let Some(fragment) = self.fragments.get(&(managed.id, k, id, row as u32)) else { continue };
                // `uFragmentOrigin` + `uFragmentShape` × the stored fraction
                let scale = (1u64 << l) as f64;
                let o = &manifest.octree[row * 5..row * 5 + 3];
                let origin: [f64; 3] = std::array::from_fn(|i| manifest.grid_origin[i] as f64 + o[i] as f64 * manifest.chunk_shape[i] as f64 * scale + manifest.vertex_offsets[l as usize * 3 + i] as f64);
                let shape: [f64; 3] = std::array::from_fn(|i| manifest.chunk_shape[i] as f64 * scale);
                for v in fragment.positions.chunks_exact(3) {
                    points.extend(phys(&model, std::array::from_fn(|i| origin[i] + shape[i] * v[i] as f64)));
                }
            }
        }
        if points.is_empty() {
            return None;
        }
        let n = points.len() / 3;
        let step = n.div_ceil(max_points.max(1));
        let positions = if step > 1 { points.chunks_exact(3).step_by(step).flatten().copied().collect() } else { points };
        Some(SegmentGraph { positions, edges: vec![], skeleton: false, radius: vec![] })
    }
}

/// Draws the viewer's 3D colour over a host's frame where its depth is
/// nearer than the host's: one full-screen triangle, premultiplied-alpha
/// blending ([`Viewer::composite_scene`](crate::Viewer::composite_scene)).
pub(crate) struct Composite {
    format: wgpu::TextureFormat,
    samples: u32,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
}

const COMPOSITE_SHADER: &str = r#"
struct U { size: vec2<f32>, srgb_target: f32, on_top: f32, ng_nf: vec2<f32>, host_nf: vec2<f32>, convert: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var ng_color: texture_2d<f32>;
@group(0) @binding(1) var ng_depth: NG_DEPTH_TYPE;
@group(0) @binding(2) var scene_depth: SCENE_DEPTH_TYPE;
@group(0) @binding(3) var<uniform> u: U;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@fragment fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(frag.xy);
    // the viewer's output is upside down (row 0 at the bottom)
    let q = vec2<i32>(p.x, i32(u.size.y) - 1 - p.y);
    let d = NG_DEPTH_LOAD;
    let c = textureLoad(ng_color, q, 0);
    if (c.a <= 0.0) { discard; }
    // volume rendering is blended in without a depth (its rays stop at the
    // viewer's own surfaces, not at a depth they write): colour where the
    // depth is still clear is a volume, drawn over the frame
    if (d >= 1.0) { return out_colour(c); }
    let s = SCENE_DEPTH_LOAD;
    // the viewer's depth (near/far fitted to the bounds) at the host's range:
    // window depth d(z) = f (z − n) / (z (f − n)) for both, through eye distance z
    var d_host = d;
    if (u.convert > 0.5) {
        let z = u.ng_nf.y * u.ng_nf.x / (u.ng_nf.y - d * (u.ng_nf.y - u.ng_nf.x));
        d_host = u.host_nf.y * (z - u.host_nf.x) / (z * (u.host_nf.y - u.host_nf.x));
    }
    // behind the host's surfaces: hidden (a hair of slack for equal depths)
    if (u.on_top < 0.5 && d_host > s + 1e-6) { discard; }
    return out_colour(c);
}

// the viewer's colours are display (sRGB) values, premultiplied; volume
// rendering's sample-count compensation can push them past alpha, which the
// conversion below would blow out to white
fn out_colour(c: vec4<f32>) -> vec4<f32> {
    var rgb = min(c.rgb, vec3<f32>(c.a));
    if (u.srgb_target > 0.5) { rgb = to_linear(rgb / c.a) * c.a; }
    return vec4<f32>(rgb, c.a);
}
"#;

impl Composite {
    pub(crate) fn matches(&self, format: wgpu::TextureFormat, samples: u32) -> bool {
        self.format == format && self.samples == samples
    }

    /// `webgl`: wgpu's GL backend, where naga cannot translate `textureLoad`
    /// from a depth texture to GLSL — the depths are bound as plain float
    /// textures there (`texelFetch`, the depth in .x).
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat, samples: u32, webgl: bool) -> Self {
        let multisampled = samples > 1;
        let (ng_type, ng_load) = if webgl { ("texture_2d<f32>", "textureLoad(ng_depth, q, 0).x") } else { ("texture_depth_2d", "textureLoad(ng_depth, q, 0)") };
        let scene_type = match (webgl, multisampled) {
            (true, true) => "texture_multisampled_2d<f32>",
            (true, false) => "texture_2d<f32>",
            (false, true) => "texture_depth_multisampled_2d",
            (false, false) => "texture_depth_2d",
        };
        let scene_load = if webgl { "textureLoad(scene_depth, p, 0).x" } else { "textureLoad(scene_depth, p, 0)" };
        let source = COMPOSITE_SHADER
            .replace("NG_DEPTH_TYPE", ng_type)
            .replace("NG_DEPTH_LOAD", ng_load)
            .replace("SCENE_DEPTH_TYPE", scene_type)
            .replace("SCENE_DEPTH_LOAD", scene_load);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ng scene composite"), source: wgpu::ShaderSource::Wgsl(source.into()) });
        let depth_sample = if webgl { wgpu::TextureSampleType::Float { filterable: false } } else { wgpu::TextureSampleType::Depth };
        let texture = |binding, sample_type, multisampled| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng scene composite"),
            entries: &[
                texture(0, wgpu::TextureSampleType::Float { filterable: false }, false),
                texture(1, depth_sample, false),
                texture(2, depth_sample, multisampled),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ng scene composite"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ng scene composite"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ng scene composite"), size: 48, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Self { format, samples, pipeline, layout, uniforms }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw(&self, device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView, output: &SceneOutput<'_>, scene_depth: &wgpu::TextureView, on_top: bool, depth: Option<DepthRanges>) {
        // without ranges (orthographic) the depths are the host's already
        let (ng_nf, host_nf, convert) = match depth {
            Some(d) => (d.viewer, d.host, d.viewer != d.host),
            None => ((0.0, 1.0), (0.0, 1.0), false),
        };
        let flag = |b: bool| if b { 1.0f32 } else { 0.0 };
        let u: [f32; 12] = [output.width as f32, output.height as f32, flag(self.format.is_srgb()), flag(on_top), ng_nf.0 as f32, ng_nf.1 as f32, host_nf.0 as f32, host_nf.1 as f32, flag(convert), 0.0, 0.0, 0.0];
        let bytes: Vec<u8> = u.iter().flat_map(|x| x.to_le_bytes()).collect();
        queue.write_buffer(&self.uniforms, 0, &bytes);
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ng scene composite"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(output.color) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(output.depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(scene_depth) },
                wgpu::BindGroupEntry { binding: 3, resource: self.uniforms.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ng scene composite") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ng scene composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}

fn to_f32(m: &[f64; 16]) -> Mat4 {
    m.map(|x| x as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_maps_display_through_physical_to_world() {
        // 10 µm voxels; the host's world is millimetres, data offset by 1 mm in x.
        let info = DisplayDimensionRenderInfo {
            global_rank: 3,
            global_dimension_names: vec!["x".into(), "y".into(), "z".into()],
            display_rank: 3,
            display_dimension_indices: [0, 1, 2],
            voxel_physical_scales: [1e-5; 3],
            display_dimension_units: ["m".into(), "m".into(), "m".into()],
            display_dimension_scales: [1e-5; 3],
            canonical_voxel_physical_size: 1e-5,
            canonical_voxel_factors: [1.0; 3],
        };
        let mut physical_from_world = IDENTITY4;
        for k in 0..3 {
            physical_from_world[k * 5] = 1e-3;
        }
        physical_from_world[12] = -1e-3; // world x = 1 mm is physical 0
        let view = SceneView { width: 100, height: 100, slice_size: 64, camera: HostCamera { view: IDENTITY4, projection: IDENTITY4, physical_from_world }, bounds: None };
        let cam = view.camera(&info).unwrap();
        // display voxel 100 in x = 1 mm physical = 2 mm world, through identity view/projection
        let x = cam.view_projection[0] as f64 * 100.0 + cam.view_projection[12] as f64;
        assert!((x - 2.0).abs() < 1e-5, "{x}");
    }
}
