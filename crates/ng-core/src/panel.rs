//! What mouse and keyboard actions in a slice panel do to the navigation.
//! Ports of the action handlers in `src/rendered_data_panel.ts` and
//! `src/sliceview/panel.ts`, with the panel geometry passed in instead of read
//! from a DOM element. GPU-free.

use std::f64::consts::PI;

use crate::glm::{self, Mat4, Quat};
use crate::navigation::{Navigation, RelativeOrientation, View};

/// The three named cross-section panels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NamedAxes {
    Xy,
    Xz,
    Yz,
}

impl NamedAxes {
    pub const ALL: [NamedAxes; 3] = [NamedAxes::Xy, NamedAxes::Xz, NamedAxes::Yz];

    /// `AXES_RELATIVE_ORIENTATION` (`src/data_panel_layout.ts`), computed as
    /// gl-matrix computes it. `None` for xy, which uses the viewer's
    /// navigation directly.
    pub fn relative_orientation(self) -> Option<Quat> {
        let half_pi = PI / 2.0;
        match self {
            NamedAxes::Xy => None,
            NamedAxes::Xz => Some(glm::quat_rotate_x(glm::QUAT_IDENTITY, half_pi)),
            NamedAxes::Yz => Some(glm::quat_rotate_y(glm::QUAT_IDENTITY, half_pi)),
        }
    }
}

/// A slice panel's view of the navigation: the viewer's own (xy), or one with
/// a linked relative orientation (xz, yz), as `makeSliceView` builds them.
#[derive(Debug, Clone)]
pub struct SlicePanel {
    relative: Option<RelativeOrientation>,
}

/// A panel's size and the pointer within it, in the panel's pixels from its
/// top-left corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelPointer {
    pub width: f64,
    pub height: f64,
    pub x: f64,
    pub y: f64,
}

impl SlicePanel {
    pub fn new(axes: NamedAxes) -> Self {
        Self {
            relative: axes.relative_orientation().map(RelativeOrientation::new),
        }
    }

    /// Runs `f` with `View::CrossSection` as this panel's navigation.
    pub fn with<R>(&mut self, nav: &mut Navigation, f: impl FnOnce(&mut Navigation) -> R) -> R {
        match &mut self.relative {
            None => f(nav),
            Some(relative) => nav.with_relative_cross_section(relative, f),
        }
    }

    pub fn orientation(&mut self, nav: &mut Navigation) -> Quat {
        self.with(nav, |n| n.orientation(View::CrossSection))
    }

    /// The slice view's `invViewMatrix`: view pixels (x right, y down, origin
    /// at the panel center) to global voxels. `navigationState.toMat4`.
    pub fn inv_view_matrix(&mut self, nav: &mut Navigation) -> Mat4 {
        self.with(nav, |n| n.to_mat4(View::CrossSection))
    }

    /// The global position under the pointer: the displayed dimensions from
    /// the slice plane, the rest from the current position. Equivalent to
    /// neuroglancer's picking (`completePickRequest`) under its orthographic
    /// projection.
    pub fn mouse_position(&mut self, nav: &mut Navigation, p: PanelPointer) -> Vec<f32> {
        let inv = self.inv_view_matrix(nav);
        let view = [(p.x - p.width / 2.0) as f32, (p.y - p.height / 2.0) as f32, 0.0];
        let plane = glm::vec3_transform_mat4(view, inv);
        let mut position = nav.position().to_vec();
        let info = nav.render_info();
        for i in 0..info.display_rank {
            position[info.display_dimension_indices[i] as usize] = plane[i];
        }
        position
    }

    /// `translate-via-mouse-drag`: `SliceViewPanel.translateByViewportPixels`.
    /// The content follows the pointer.
    pub fn translate_by_viewport_pixels(&mut self, nav: &mut Navigation, dx: f64, dy: f64) {
        let inv = self.inv_view_matrix(nav);
        nav.update_display_position(|pos| {
            *pos = glm::vec3_transform_mat4([-dx as f32, -dy as f32, 0.0], inv);
            true
        });
    }

    /// One step of `rotate-via-mouse-drag` in a slice panel: rotate the
    /// *viewer's* orientation about this panel's x and y axes, through
    /// `initial_position` (the point under the pointer when the drag began),
    /// a quarter degree per pixel.
    pub fn rotate_via_mouse_drag(&mut self, nav: &mut Navigation, dx: f64, dy: f64, initial_position: &[f32]) {
        let orientation = self.orientation(nav);
        let x_axis = glm::vec3_transform_quat(glm::AXES[0], orientation);
        let y_axis = glm::vec3_transform_quat(glm::AXES[1], orientation);
        nav.rotate_absolute(View::CrossSection, y_axis, ((-dx / 4.0) * PI) / 180.0, initial_position);
        nav.rotate_absolute(View::CrossSection, x_axis, ((-dy / 4.0) * PI) / 180.0, initial_position);
    }

    /// `zoomByMouse`: zoom by `factor`, keeping the point under the pointer
    /// in place.
    pub fn zoom_by_mouse(&mut self, nav: &mut Navigation, factor: f64, p: PanelPointer) {
        if !nav.valid() {
            return;
        }
        let inv = self.inv_view_matrix(nav);
        let (mx, my) = (p.x - p.width / 2.0, p.y - p.height / 2.0);
        nav.update_display_position(|pos| {
            for (i, c) in pos.iter_mut().enumerate() {
                let f = inv[i] as f64 * mx + inv[4 + i] as f64 * my;
                *c = (*c as f64 + f * (1.0 - factor)) as f32;
            }
            true
        });
        self.with(nav, |n| n.zoom_by(View::CrossSection, factor));
    }

    /// `z+1-via-wheel` / `z+10-via-wheel`: step through slices; scrolling
    /// away from the user (`delta < 0`) moves forward.
    pub fn z_via_wheel(&mut self, nav: &mut Navigation, delta_y: f64, delta_x: f64, amount: f64) {
        let delta = if delta_y != 0.0 { delta_y } else { delta_x };
        let step = if delta > 0.0 { -1.0 } else { 1.0 } * amount;
        self.with(nav, |n| n.translate_voxels_relative(View::CrossSection, [0.0, 0.0, step as f32]));
    }

    /// `x-`/`x+`/`y-`/`y+`/`z-`/`z+`: one voxel along a panel axis.
    pub fn translate_axis(&mut self, nav: &mut Navigation, axis: usize, sign: f32) {
        let mut offset = [0f32; 3];
        offset[axis] = sign;
        self.with(nav, |n| n.translate_voxels_relative(View::CrossSection, offset));
    }

    /// `rotate-relative-{x,y,z}{-,+}`: 0.1 radian about a panel axis.
    pub fn rotate_relative_axis(&mut self, nav: &mut Navigation, axis: usize, sign: f64) {
        self.with(nav, |n| n.rotate_relative(View::CrossSection, glm::AXES[axis], sign * 0.1));
    }

    /// `snap`.
    pub fn snap(&mut self, nav: &mut Navigation) {
        self.with(nav, |n| n.snap(View::CrossSection));
    }
}

/// `getWheelZoomAmount` (`src/util/wheel_zoom.ts`). `delta_mode` is the DOM
/// `WheelEvent.deltaMode`: 0 pixels, 1 lines, 2 pages.
pub fn wheel_zoom_amount(delta_y: f64, delta_mode: u32) -> f64 {
    let multiplier = match delta_mode {
        0 => 1.0 / 200.0,
        1 => 1.0 / 10.0,
        2 => 2.0,
        _ => 0.0,
    };
    crate::js::exp(delta_y * multiplier)
}

/// The 3D panel's camera (`PerspectivePanel`'s projection parameters).
#[derive(Debug, Clone, PartialEq)]
pub struct PerspectiveCamera {
    pub projection: Mat4,
    pub inv_view: Mat4,
    pub view_projection: Mat4,
}

/// `PerspectivePanel`'s `DerivedProjectionParameters.update`: a 45° perspective
/// (or the orthographic projection matching it at the position's plane) of
/// the projection pose, `logical_width × logical_height` pixels.
pub fn perspective_camera(nav: &Navigation, logical_width: f64, logical_height: f64, orthographic: bool) -> PerspectiveCamera {
    use crate::glm::{mat4_invert, mat4_multiply, mat4_ortho, mat4_perspective, mat4_scale, mat4_translate};
    let width_over_height = logical_width / logical_height;
    let fovy = std::f64::consts::PI / 4.0;
    let mut relative_depth_range = nav.relative_depth_range(View::Projection);
    let mut zoom = nav.zoom(View::Projection) / 2.0;
    let projection = if orthographic {
        let near = (1.0 - relative_depth_range).max(0.1);
        let far = 1.0 + relative_depth_range;
        mat4_ortho(-width_over_height, width_over_height, -1.0, 1.0, near, far)
    } else {
        let f = 1.0 / crate::js::tan(fovy / 2.0);
        relative_depth_range /= f;
        let near = (1.0 - relative_depth_range).max(0.1);
        let far = 1.0 + relative_depth_range;
        zoom *= f;
        mat4_perspective(fovy, width_over_height, near, far)
    };
    let inv_view = nav.to_mat4_zoom(View::Projection, zoom);
    let inv_view = mat4_scale(&inv_view, [1.0, -1.0, -1.0]);
    let inv_view = mat4_translate(&inv_view, [0.0, 0.0, 1.0]);
    let view = mat4_invert(&inv_view).unwrap_or(inv_view);
    let view_projection = mat4_multiply(&projection, &view);
    PerspectiveCamera { projection, inv_view, view_projection }
}
