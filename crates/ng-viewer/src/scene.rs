//! Scene mode: the viewer's 3D content drawn from a host renderer's camera,
//! for compositing into the host's own frame.
//!
//! An embedder with its own 3D scene (a brain atlas, a game engine) wants
//! neuroglancer's volumes inside that scene, not in neuroglancer's panels.
//! [`Viewer::render_scene`](crate::Viewer::render_scene) runs the normal
//! frame with three changes:
//!
//! - the three cross-section panels are drawn offscreen at a fixed size
//!   ([`SceneView::slice_size`]) and shown as planes in 3D, as the 3D panel
//!   of the `4panel` layout shows them;
//! - the 3D panel is the host's size and uses the host's camera
//!   ([`HostCamera`]): its chunks, meshes and volume rendering are chosen and
//!   drawn for that view;
//! - nothing is composited into a layout: the host reads the 3D panel's
//!   colour and depth ([`SceneOutput`]) and composites them by depth.
//!
//! Navigation (the position the slices cut through, the cross-section
//! scale, layers and shaders) stays the viewer's: set it with the state as
//! usual.
use ng_core::glm::Mat4;
use ng_core::navigation::DisplayDimensionRenderInfo;
use ng_core::panel::{NamedAxes, PerspectiveCamera};
use threers::renderer::PixelRect;

use crate::Panel;

/// The host's camera, column-major `[f64; 16]` matrices.
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
}

impl SceneView {
    /// The panels drawn, in physical pixels: the slices (if shown) then the
    /// 3D panel, each at the origin of its own offscreen target.
    pub(crate) fn panel_rects(&self, show_slices: bool) -> Vec<(Panel, PixelRect)> {
        let s = self.slice_size.max(1);
        let mut out: Vec<(Panel, PixelRect)> = if show_slices {
            NamedAxes::ALL.iter().map(|&a| (Panel::Slice(a), PixelRect { x: 0, y: 0, width: s, height: s })).collect()
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
        let mut physical_from_display = identity();
        for (k, scale) in info.voxel_physical_scales.iter().enumerate() {
            physical_from_display[k * 5] = *scale;
        }
        let world_from_physical = invert(&c.physical_from_world)?;
        let world_from_display = multiply(&world_from_physical, &physical_from_display);
        let view = multiply(&c.view, &world_from_display);
        let inv_view = invert(&view)?;
        let view_projection = multiply(&c.projection, &view);
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

    /// The nearest point where a ray from the host's camera (`origin`,
    /// `direction`, host world) meets one of the scene's cross-sections, within
    /// its field of view (`slice_size` pixels across, as drawn), and the
    /// layers' values there: what a hover or a click on the slices means.
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
        let world_from_physical = invert(m)?;
        let mut best: Option<(f64, ScenePick)> = None;
        for (k, (u, v, n)) in planes.into_iter().enumerate() {
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
                .filter(|&i| self.layers[i].spec.visible)
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

/// The target `render` composites the panels into, unused in scene mode
/// but required by the frame: kept so it is not reallocated every frame.
pub(crate) struct Scratch {
    pub width: u32,
    pub height: u32,
    pub view: wgpu::TextureView,
}

impl Scratch {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ng scene scratch"),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self { width, height, view: texture.create_view(&Default::default()) }
    }
}

fn identity() -> [f64; 16] {
    let mut m = [0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m
}

fn to_f32(m: &[f64; 16]) -> Mat4 {
    m.map(|x| x as f32)
}

/// `a · b`, column-major.
pub(crate) fn multiply(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    out
}

/// The inverse of a column-major 4×4, if it has one.
pub(crate) fn invert(m: &[f64; 16]) -> Option<[f64; 16]> {
    // Gauss–Jordan on rows of the transpose is the same as on columns.
    let mut a = [[0.0f64; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            a[r][c] = m[c * 4 + r];
        }
        a[r][4 + r] = 1.0;
    }
    for col in 0..4 {
        let pivot = (col..4).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        let p = a[col][col];
        for x in a[col].iter_mut() {
            *x /= p;
        }
        for r in 0..4 {
            if r != col {
                let f = a[r][col];
                if f != 0.0 {
                    let row = a[col];
                    for (x, y) in a[r].iter_mut().zip(row) {
                        *x -= f * y;
                    }
                }
            }
        }
    }
    let mut out = [0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            out[c * 4 + r] = a[r][4 + c];
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_round_trips() {
        let m = [2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 5.0, 6.0, 7.0, 1.0];
        let p = multiply(&m, &invert(&m).unwrap());
        for (k, x) in p.iter().enumerate() {
            assert!((x - identity()[k]).abs() < 1e-12, "{p:?}");
        }
    }

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
        let mut physical_from_world = identity();
        for k in 0..3 {
            physical_from_world[k * 5] = 1e-3;
        }
        physical_from_world[12] = -1e-3; // world x = 1 mm is physical 0
        let view = SceneView { width: 100, height: 100, slice_size: 64, camera: HostCamera { view: identity(), projection: identity(), physical_from_world } };
        let cam = view.camera(&info).unwrap();
        // display voxel 100 in x = 1 mm physical = 2 mm world, through identity view/projection
        let x = cam.view_projection[0] as f64 * 100.0 + cam.view_projection[12] as f64;
        assert!((x - 2.0).abs() < 1e-5, "{x}");
    }
}
