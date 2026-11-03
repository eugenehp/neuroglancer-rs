//! Pointer, wheel and key input: which panel is under the pointer, drags,
//! and the bound actions they run.

use super::*;

impl Viewer {
    /// The panel under the pointer, and the pointer in that panel's logical
    /// pixels.
    pub(crate) fn pointer_panel(&self) -> Option<(Panel, PanelPointer)> {
        // the host routes its own input in scene mode
        if self.scene.is_some() {
            return None;
        }
        let (w, h) = self.logical_size();
        let [ox, oy, _, _] = self.logical_region();
        let (x, y) = self.pointer?;
        let (x, y) = (x - ox, y - oy);
        let panel = self.layout.panel_at(x, y, w, h)?;
        let r = self.layout.rect(panel, w, h)?;
        let p = PanelPointer {
            width: r.width as f64,
            height: r.height as f64,
            x: x - r.x as f64,
            y: y - r.y as f64,
        };
        Some((panel, p))
    }

    pub(crate) fn panel_mut(&mut self, axes: NamedAxes) -> SlicePanel {
        // Panels are small; take a copy to sidestep borrowing `self.nav` and
        // `self.panels` together, then put it back.
        self.panels.iter().find(|(a, _)| *a == axes).unwrap().1.clone()
    }

    pub(crate) fn put_panel(&mut self, axes: NamedAxes, panel: SlicePanel) {
        self.panels.iter_mut().find(|(a, _)| *a == axes).unwrap().1 = panel;
    }

    /// The global position under the pointer, if it is over a slice panel.
    pub fn mouse_position(&mut self) -> Option<Vec<f32>> {
        let (panel, p) = self.pointer_panel()?;
        let pos = if panel == Panel::Perspective {
            // The last pick's (`mouseState` from the pick buffers).
            self.perspective_pick.as_ref()?.position.clone()
        } else {
            let Panel::Slice(axes) = panel else { return None };
            if !self.nav.valid() {
                return None;
            }
            let mut sp = self.panel_mut(axes);
            let pos = sp.mouse_position(&mut self.nav, p);
            self.put_panel(axes, sp);
            pos
        };
        Some(self.snap_to_picked(pos))
    }

    /// `updateMouseState` for a picked annotation (`snapPosition`): the
    /// position moved onto the point, the line (or polyline's first
    /// segment), the endpoint or the box corner picked; boxes' and
    /// ellipsoids' bodies leave it.
    fn snap_to_picked(&self, mut position: Vec<f32>) -> Vec<f32> {
        let Some(PickTarget::Annotation(layer_id, id, part)) = self.picked() else { return position };
        let Some(li) = self.layers.iter().position(|l| l.id == layer_id) else { return position };
        let Some(a) = self.find_annotation(li, &id) else { return position };
        let Some(p) = self.global_to_annotation_model(li, &position) else { return position };
        use annotations::Geometry as G;
        // `projectPointToLineSegment`.
        let project = |a: &[f64], b: &[f64]| -> Vec<f64> {
            let den: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
            let num: f64 = a.iter().zip(b).zip(&p).map(|((x, y), q)| -(x - q) * (y - x)).sum();
            let t = (num / den.max(1e-6)).clamp(0.0, 1.0);
            a.iter().zip(b).map(|(x, y)| x + t * (y - x)).collect()
        };
        let snapped: Option<Vec<f64>> = match (&a.geometry, part) {
            (G::Point(q), _) => Some(q.clone()),
            (G::Line(pa, pb), AnnotationPart::Body) => Some(project(pa, pb)),
            (G::Line(pa, pb), AnnotationPart::Endpoint(k)) => Some(if k == 0 { pa.clone() } else { pb.clone() }),
            (G::Polyline(points), AnnotationPart::Body) if points.len() >= 2 => Some(project(&points[0], &points[1])),
            (G::Polyline(points), AnnotationPart::Endpoint(k)) => points.get(k).cloned(),
            // `snapPositionToCorner`: per dimension the nearer bound.
            (G::AxisAlignedBoundingBox(pa, pb), AnnotationPart::Corner(_)) => Some(pa.iter().zip(pb).zip(&p).map(|((a, b), x)| if (a - x).abs() < (b - x).abs() { *a } else { *b }).collect()),
            _ => None,
        };
        if let Some(model) = snapped {
            let l = &self.layers[li].annotation.as_ref().unwrap().layer;
            let n = l.rank + 1;
            let m2l = &l.model_to_layer;
            for (ld, g) in l.layer_to_global.iter().enumerate() {
                let Some(g) = *g else { continue };
                let mut v = m2l[l.rank * n + ld];
                for (c, x) in model.iter().enumerate().take(l.source_rank) {
                    v += m2l[c * n + ld] * x;
                }
                if v.is_finite() {
                    if let Some(slot) = position.get_mut(g) {
                        *slot = v as f32;
                    }
                }
            }
        }
        position
    }

    /// The pointer moved to `(x, y)`, physical pixels from the top-left.
    /// Returns whether the view changed.
    /// Whether a drag (pan, rotate) is under way.
    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// The pointer left the viewer: nothing is hovered (`mouseState.setActive(false)`).
    pub fn pointer_leave(&mut self) -> bool {
        self.pointer.take().is_some()
    }

    pub fn pointer_move(&mut self, x: f64, y: f64) -> bool {
        let (x, y) = (x / self.scale_factor, y / self.scale_factor);
        let (dx, dy) = self.pointer.map_or((0.0, 0.0), |(px, py)| (x - px, y - py));
        self.pointer = Some((x, y));
        let Some(drag) = self.drag.clone() else {
            // The status line follows the pointer.
            return true;
        };
        match (drag.panel, drag.action) {
            (Panel::Slice(axes), DragAction::Translate) => {
                let mut sp = self.panel_mut(axes);
                sp.translate_by_viewport_pixels(&mut self.nav, dx, dy);
                self.put_panel(axes, sp);
            }
            (Panel::Slice(axes), DragAction::Rotate { initial_position }) => {
                let mut sp = self.panel_mut(axes);
                sp.rotate_via_mouse_drag(&mut self.nav, dx, dy, &initial_position);
                self.put_panel(axes, sp);
            }
            // `PerspectivePanel`'s "rotate-via-mouse-drag".
            (Panel::Perspective, DragAction::PerspectiveRotate) => {
                self.nav.rotate_relative(View::Projection, ng_core::glm::AXES[1], dx / 4.0 * std::f64::consts::PI / 180.0);
                self.nav.rotate_relative(View::Projection, ng_core::glm::AXES[0], -dy / 4.0 * std::f64::consts::PI / 180.0);
            }
            (Panel::Perspective, DragAction::PerspectiveTranslate) => self.perspective_translate(dx, dy),
            (panel, DragAction::MoveAnnotation { layer, id, part, point, geometry, total }) => {
                let total = (total.0 + dx, total.1 + dy);
                if let Some(d) = self.drag.as_mut() {
                    if let DragAction::MoveAnnotation { total: t, .. } = &mut d.action {
                        *t = total;
                    }
                }
                self.move_annotation(panel, layer, &id, part, &point, &geometry, total);
            }
            _ => {}
        }
        true
    }

    pub fn pointer_down(&mut self, button: Button, modifiers: Modifiers) -> bool {
        let Some((panel, _)) = self.pointer_panel() else { return false };
        let kind = match panel {
            Panel::Slice(_) => PanelKind::Slice,
            Panel::Perspective => PanelKind::Perspective,
        };
        let key = format!("mousedown{}", button as u8);
        let Some(action) = lookup(kind, &key, modifiers, self.mac) else {
            return false;
        };
        match (panel, action) {
            (Panel::Slice(_), "translate-via-mouse-drag") => {
                self.drag = Some(Drag { panel, action: DragAction::Translate });
                false
            }
            (Panel::Slice(_), "rotate-via-mouse-drag") => {
                let Some(initial_position) = self.mouse_position() else { return false };
                self.drag = Some(Drag { panel, action: DragAction::Rotate { initial_position } });
                false
            }
            (Panel::Slice(_), "move-to-mouse-position") => match self.mouse_position() {
                Some(p) => {
                    self.nav.set_position(&p);
                    true
                }
                None => false,
            },
            (Panel::Perspective, "rotate-via-mouse-drag") => {
                self.drag = Some(Drag { panel, action: DragAction::PerspectiveRotate });
                false
            }
            (Panel::Perspective, "translate-via-mouse-drag") => {
                self.drag = Some(Drag { panel, action: DragAction::PerspectiveTranslate });
                false
            }
            (_, "move-annotation") => {
                let Some(PickTarget::Annotation(layer, id, part)) = self.picked() else { return false };
                let Some(a) = self.layers.iter().find(|l| l.id == layer).and_then(|l| l.annotation.as_ref()?.annotations.iter().find(|a| a.id == id)).cloned() else { return false };
                let point = representative_point(&a.geometry, part);
                self.drag = Some(Drag { panel, action: DragAction::MoveAnnotation { layer, id, part, point, geometry: a.geometry, total: (0.0, 0.0) } });
                false
            }
            (_, "delete-annotation") => match self.picked_annotation() {
                Some((layer, id)) => {
                    self.delete_annotation(layer, &id);
                    true
                }
                None => false,
            },
            _ => false,
        }
    }

    /// `move-annotation`'s drag: the representative point moved `total`
    /// logical pixels in the panel (`translateDataPointByViewportPixels`),
    /// the annotation updated from it (`updateViaRepresentativePoint`).
    #[allow(clippy::too_many_arguments)]
    fn move_annotation(&mut self, panel: Panel, layer: LayerId, id: &str, part: AnnotationPart, point: &[f64], geometry: &annotations::Geometry, total: (f64, f64)) {
        let Some(li) = self.layers.iter().position(|l| l.id == layer) else { return };
        let info = self.nav.render_info().clone();
        let slice_inv_view = match panel {
            Panel::Slice(axes) => {
                let mut sp = self.panel_mut(axes);
                let inv_view = sp.inv_view_matrix(&mut self.nav);
                self.put_panel(axes, sp);
                Some(inv_view)
            }
            Panel::Perspective => None,
        };
        let Some(ann) = self.layers[li].annotation.as_ref() else { return };
        let al = &ann.layer;
        let display = model_to_display(al, &info, point);
        let moved = match panel {
            Panel::Slice(_) => {
                let Some(inv_view) = slice_inv_view else { return };
                // View pixels (x right, y down) through the view's linear part.
                std::array::from_fn::<f64, 3, _>(|r| display[r] + inv_view[r] as f64 * total.0 + inv_view[4 + r] as f64 * total.1)
            }
            Panel::Perspective => {
                let Some(camera) = self.perspective_camera() else { return };
                let vp = camera.view_projection;
                let Some(inv) = ng_core::glm::mat4_invert(&vp) else { return };
                let (lw, lh) = self.logical_size();
                let Some(r) = self.layout.rect(Panel::Perspective, lw, lh) else { return };
                let mut c = ng_core::glm::vec4_transform_mat4([display[0], display[1], display[2], 1.0], &vp);
                c[0] += 2.0 * total.0 / r.width as f64 * c[3];
                c[1] -= 2.0 * total.1 / r.height as f64 * c[3];
                let d = ng_core::glm::vec4_transform_mat4(c, &inv);
                [d[0] / d[3], d[1] / d[3], d[2] / d[3]]
            }
        };
        // The point's global position with the display dimensions moved,
        // back into the source's coordinates.
        let mut global: Vec<f32> = self.nav.position().to_vec();
        {
            let l = al;
            let n = l.rank + 1;
            for (ld, g) in l.layer_to_global.iter().enumerate() {
                let Some(g) = *g else { continue };
                let mut v = l.model_to_layer[l.rank * n + ld];
                for (c, x) in point.iter().enumerate().take(l.source_rank) {
                    v += l.model_to_layer[c * n + ld] * x;
                }
                if let Some(slot) = global.get_mut(g) {
                    *slot = v as f32;
                }
            }
        }
        for (k, &g) in info.display_dimension_indices.iter().enumerate() {
            if g >= 0 {
                if let Some(slot) = global.get_mut(g as usize) {
                    *slot = moved[k] as f32;
                }
            }
        }
        let Some(new_point) = self.global_to_annotation_model(li, &global) else { return };
        let updated = update_via_representative_point(geometry, &new_point, part);
        self.update_annotation(li, id, |v| write_geometry(v, &updated));
    }

    pub fn pointer_up(&mut self, _button: Button) -> bool {
        self.drag = None;
        false
    }

    /// A wheel event with DOM semantics: `delta_y` positive toward the user,
    /// `delta_mode` 0 pixels / 1 lines / 2 pages.
    pub fn wheel(&mut self, delta_x: f64, delta_y: f64, delta_mode: u32, modifiers: Modifiers) -> bool {
        let Some((panel, p)) = self.pointer_panel() else { return false };
        match panel {
            Panel::Slice(axes) => {
                let Some(action) = lookup(PanelKind::Slice, "wheel", modifiers, self.mac) else {
                    return false;
                };
                let mut sp = self.panel_mut(axes);
                match action {
                    "z+1-via-wheel" => sp.z_via_wheel(&mut self.nav, delta_y, delta_x, 1.0),
                    "z+10-via-wheel" => sp.z_via_wheel(&mut self.nav, delta_y, delta_x, 10.0),
                    "zoom-via-wheel" => sp.zoom_by_mouse(&mut self.nav, wheel_zoom_amount(delta_y, delta_mode), p),
                    "adjust-depth-range-via-wheel" => {
                        let d = self.nav.depth_range(View::CrossSection) * wheel_zoom_amount(delta_y, delta_mode);
                        self.nav.set_depth_range(View::CrossSection, d);
                    }
                    _ => {}
                }
                self.put_panel(axes, sp);
                true
            }
            Panel::Perspective => match lookup(PanelKind::Perspective, "wheel", modifiers, self.mac) {
                // `zoomByMouse`: the projection zoom.
                Some("zoom-via-wheel") => {
                    self.nav.zoom_by(View::Projection, wheel_zoom_amount(delta_y, delta_mode));
                    true
                }
                Some("adjust-depth-range-via-wheel") => {
                    let d = self.nav.depth_range(View::Projection) * wheel_zoom_amount(delta_y, delta_mode);
                    self.nav.set_depth_range(View::Projection, d);
                    true
                }
                _ => false,
            },
        }
    }

    /// A key press: `code` is DOM `KeyboardEvent.code` (any case). Acts on the
    /// panel under the pointer, as neuroglancer's focused panel does.
    pub fn key_down(&mut self, code: &str, modifiers: Modifiers) -> bool {
        let Some((panel, _)) = self.pointer_panel() else { return false };
        let Panel::Slice(axes) = panel else { return false };
        let key = code.to_ascii_lowercase();
        let Some(action) = lookup(PanelKind::Slice, &key, modifiers, self.mac) else {
            return false;
        };
        self.slice_action(axes, action)
    }

    /// A data panel action by name (`snap`, `zoom-in`, `x+`, …), as the
    /// command palette runs it: in the slice panel under the pointer, else
    /// the layout's first. Returns whether it was one.
    pub fn panel_action(&mut self, action: &str) -> bool {
        let axes = match self.pointer_panel() {
            Some((Panel::Slice(axes), _)) => axes,
            _ => match self.layout.cells().into_iter().find_map(|(p, _)| if let Panel::Slice(a) = p { Some(a) } else { None }) {
                Some(a) => a,
                None => return false,
            },
        };
        self.slice_action(axes, action)
    }

    fn slice_action(&mut self, axes: NamedAxes, action: &str) -> bool {
        let mut sp = self.panel_mut(axes);
        let nav = &mut self.nav;
        let handled = match action {
            "snap" => {
                sp.snap(nav);
                true
            }
            "zoom-in" => {
                sp.with(nav, |n| n.zoom_by(View::CrossSection, 0.5));
                true
            }
            "zoom-out" => {
                sp.with(nav, |n| n.zoom_by(View::CrossSection, 2.0));
                true
            }
            "depth-range-decrease" | "depth-range-increase" => {
                let f = if action == "depth-range-decrease" { 0.5 } else { 2.0 };
                let d = nav.depth_range(View::CrossSection) * f;
                nav.set_depth_range(View::CrossSection, d);
                true
            }
            a => {
                let axis_sign = |s: &str| -> Option<(usize, f32)> {
                    let axis = match s.as_bytes().first()? {
                        b'x' => 0,
                        b'y' => 1,
                        b'z' => 2,
                        _ => return None,
                    };
                    let sign = match &s[1..] {
                        "+" => 1.0,
                        "-" => -1.0,
                        _ => return None,
                    };
                    Some((axis, sign))
                };
                if let Some(rest) = a.strip_prefix("rotate-relative-") {
                    axis_sign(rest).map(|(axis, sign)| sp.rotate_relative_axis(nav, axis, sign as f64)).is_some()
                } else if let Some((axis, sign)) = axis_sign(a) {
                    sp.translate_axis(nav, axis, sign);
                    true
                } else {
                    false
                }
            }
        };
        self.put_panel(axes, sp);
        handled
    }

    /// The 3D panel's camera, for its logical size.
    pub(crate) fn perspective_camera(&self) -> Option<ng_core::panel::PerspectiveCamera> {
        if let Some(scene) = &self.scene {
            return scene.camera(self.nav.render_info());
        }
        let (lw, lh) = self.logical_size();
        let r = self.layout.rect(Panel::Perspective, lw, lh)?;
        Some(ng_core::panel::perspective_camera(&self.nav, r.width as f64, r.height as f64, self.orthographic_projection))
    }

    /// `PerspectivePanel.translateByViewportPixels`.
    fn perspective_translate(&mut self, dx: f64, dy: f64) {
        let (lw, lh) = self.logical_size();
        let Some(r) = self.layout.rect(Panel::Perspective, lw, lh) else { return };
        let Some(camera) = self.perspective_camera() else { return };
        let Some(inv) = ng_core::glm::mat4_invert(&camera.view_projection) else { return };
        let (w, h) = (r.width as f64, r.height as f64);
        self.nav.update_display_position(|pos| {
            let t = ng_core::glm::vec3_transform_mat4(*pos, camera.view_projection);
            let t = [t[0] + (-2.0 * dx / w) as f32, t[1] + (2.0 * dy / h) as f32, t[2]];
            *pos = ng_core::glm::vec3_transform_mat4(t, inv);
            true
        });
    }
}
