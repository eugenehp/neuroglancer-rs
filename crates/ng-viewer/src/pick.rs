//! Picking: what is under the pointer in each panel, and the annotation
//! handles that can be grabbed and dragged.

use super::*;

/// A pick window being read: the frame's inverse view-projection, position,
/// display dimensions and pick IDs, and whether its buffer is mapping.
pub(crate) struct PendingPick {
    pub(crate) readback: ng_render::pick::PickReadback,
    pub(crate) inv_vp: [f32; 16],
    pub(crate) position: Vec<f32>,
    pub(crate) display: [i32; 3],
    pub(crate) ids: Vec<PickTarget>,
    pub(crate) mapped: bool,
}

/// What a pick ID stands for (`PickIDManager`'s registrations): a
/// layer's segment (a mesh, a skeleton) or annotation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PickTarget {
    Segment(LayerId, u64),
    Annotation(LayerId, String, AnnotationPart),
}

/// The part of an annotation picked (its `partIndex`), for snapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum AnnotationPart {
    Body,
    /// A line's endpoint marker (`ENDPOINTS_PICK_OFFSET + k`).
    Endpoint(usize),
    /// A box's corner (3D): which (`CORNERS_PICK_OFFSET + k`).
    Corner(usize),
}

impl AnnotationPart {
    /// `annotationPart` (`pickedOffset`) as neuroglancer numbers it: 0 for
    /// the object, then the endpoints or corners.
    pub(crate) fn index(self) -> u32 {
        match self {
            AnnotationPart::Body => 0,
            AnnotationPart::Endpoint(k) | AnnotationPart::Corner(k) => 1 + k as u32,
        }
    }
}

/// A resolved 3D pick: the global position under the pointer, and what
/// was drawn there.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PerspectivePick {
    pub(crate) position: Vec<f32>,
    pub(crate) picked: Option<PickTarget>,
}

/// A pickable mark in a slice panel, in its pixels (from the top left),
/// as its quad covers them.
#[derive(Debug, Clone, PartialEq)]
enum PickShapeKind {
    /// A line's quad: from `a` to `b`, `half` pixels to each side.
    Segment { a: [f64; 2], b: [f64; 2], half: f64 },
    Disc { center: [f64; 2], radius: f64 },
    /// `center + s·u + t·v` for `s² + t² ≤ 1`.
    Ellipse { center: [f64; 2], u: [f64; 2], v: [f64; 2] },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PickShape {
    kind: PickShapeKind,
    pick: u32,
}

/// GL clip coordinates to pixels from the top left of a `w` × `h` panel;
/// `None` beyond the depth range.
fn clip_to_pixels(c: [f64; 4], w: f64, h: f64) -> Option<[f64; 2]> {
    (c[3] > 0.0 && (c[2] / c[3]).abs() <= 1.0).then(|| [(c[0] / c[3] + 1.0) / 2.0 * w, (1.0 - c[1] / c[3]) / 2.0 * h])
}

impl PickShape {
    pub(crate) fn segment(a: [f64; 2], b: [f64; 2], width: f64, feather: f64, pick: u32) -> Self {
        let half = if width == 0.0 { 0.0 } else { width / 2.0 + feather };
        PickShape { kind: PickShapeKind::Segment { a, b, half }, pick }
    }

    pub(crate) fn from_line(l: &ng_render::annotations::Line, w: f64, h: f64) -> Option<Self> {
        let (a, b) = ng_render::annotations::clip_to_depth_range(l.a, l.b)?;
        let px = |c: [f64; 4]| [(c[0] / c[3] + 1.0) / 2.0 * w, (1.0 - c[1] / c[3]) / 2.0 * h];
        (l.pick != 0).then(|| Self::segment(px(a), px(b), l.width, l.feather, l.pick))
    }

    pub(crate) fn from_circle(c: &ng_render::annotations::Circle, w: f64, h: f64) -> Option<Self> {
        let center = clip_to_pixels(c.center, w, h)?;
        let total = if c.diameter == 0.0 { 0.0 } else { c.diameter + 2.0 * (c.border_width + c.feather) };
        (c.pick != 0).then_some(PickShape { kind: PickShapeKind::Disc { center, radius: total / 2.0 }, pick: c.pick })
    }

    pub(crate) fn from_ellipse(e: &ng_render::annotations::Ellipse, w: f64, h: f64) -> Option<Self> {
        let p: Vec<[f64; 2]> = e.corners.iter().map(|&c| clip_to_pixels(c, w, h)).collect::<Option<_>>()?;
        let center = [(p[0][0] + p[2][0]) / 2.0, (p[0][1] + p[2][1]) / 2.0];
        let u = [(p[1][0] - p[0][0]) / 2.0, (p[1][1] - p[0][1]) / 2.0];
        let v = [(p[3][0] - p[0][0]) / 2.0, (p[3][1] - p[0][1]) / 2.0];
        (e.pick != 0).then_some(PickShape { kind: PickShapeKind::Ellipse { center, u, v }, pick: e.pick })
    }

    /// Whether the quad covers the pixel centred at `p`.
    fn covers(&self, p: [f64; 2]) -> bool {
        match self.kind {
            PickShapeKind::Segment { a, b, half } => {
                let d = [b[0] - a[0], b[1] - a[1]];
                let len2 = d[0] * d[0] + d[1] * d[1];
                if len2 < 1e-12 {
                    return (p[0] - a[0]).abs() <= half && (p[1] - a[1]).abs() <= half;
                }
                let r = [p[0] - a[0], p[1] - a[1]];
                let t = (r[0] * d[0] + r[1] * d[1]) / len2;
                let perp = (r[0] * d[1] - r[1] * d[0]).abs() / len2.sqrt();
                (0.0..=1.0).contains(&t) && perp <= half
            }
            PickShapeKind::Disc { center, radius } => (p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2) <= radius * radius,
            PickShapeKind::Ellipse { center, u, v } => {
                let det = u[0] * v[1] - u[1] * v[0];
                if det.abs() < 1e-12 {
                    return false;
                }
                let r = [p[0] - center[0], p[1] - center[1]];
                let s = (r[0] * v[1] - r[1] * v[0]) / det;
                let t = (u[0] * r[1] - u[1] * r[0]) / det;
                s * s + t * t <= 1.0
            }
        }
    }
}

/// A slice panel's pickable marks (`pickIDs` and what the pick buffer
/// would hold): its size, the targets and the shapes in drawing order.
#[derive(Debug, Clone, Default)]
pub(crate) struct SlicePicks {
    pub(crate) size: (u32, u32),
    pub(crate) targets: Vec<PickTarget>,
    pub(crate) shapes: Vec<PickShape>,
}

impl SlicePicks {
    /// `completePickRequest` in a slice panel: from the pixel under `(x,
    /// y)` (physical, from the panel's top left) outwards within the pick
    /// radius, the first covered by a mark — the last one drawn there.
    pub(crate) fn pick(&self, x: f64, y: f64) -> Option<PickTarget> {
        if self.shapes.is_empty() {
            return None;
        }
        let (w, h) = (self.size.0 as i64, self.size.1 as i64);
        let gx = x.floor() as i64;
        let gy = (h as f64 - y).floor() as i64;
        let r = PICK_RADIUS as i64;
        let d = ng_render::pick::pick_diameter(PICK_RADIUS) as i64;
        for o in ng_render::pick::pick_offset_sequence(PICK_RADIUS) {
            let (px, py) = (gx + o as i64 % d - r, gy + o as i64 / d - r);
            if px < 0 || py < 0 || px >= w || py >= h {
                continue;
            }
            let centre = [px as f64 + 0.5, h as f64 - (py as f64 + 0.5)];
            if let Some(s) = self.shapes.iter().rev().find(|s| s.covers(centre)) {
                return self.targets.get(s.pick as usize - 1).cloned();
            }
        }
        None
    }
}

/// Annotations as marks in clip coordinates, a batch per layer and
/// annotation type in drawing order (`annotationTypes`: points, lines,
/// boxes, ellipsoids, polylines).
#[derive(Default)]
pub(crate) struct AnnotationMarks {
    pub(crate) batches: Vec<ng_render::annotations::MarkBatch>,
}

impl AnnotationMarks {
    /// The marks of `annotations` by type (`annotationTypes`), each in its
    /// style (the shader's): points as circles with their borders, lines
    /// with their endpoint markers, polylines' segments and markers,
    /// ellipsoids' cross-sections (`ellipses`, by annotation); boxes go to
    /// `boxes` (model-space corners, border colour and width). The hovered
    /// annotation's colours three quarters of the way to white
    /// (`setPartIndex`); discarded ones are left out.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add(&mut self, annotations: &[annotations::Annotation], ellipses: &[Option<ng_render::annotations::Ellipse>], styles: &[ng_render::annotation_shader::AnnotationStyle], hovered: Option<&str>, register: &mut dyn FnMut(&str, AnnotationPart) -> u32, to_clip: &dyn Fn(&[f64]) -> [f64; 4], boxes: &mut BoxSink) {
        use annotations::Geometry as G;
        type Kind = fn(&G) -> bool;
        let kinds: [Kind; 5] = [|g| matches!(g, G::Point(_)), |g| matches!(g, G::Line(..)), |g| matches!(g, G::AxisAlignedBoundingBox(..)), |g| matches!(g, G::Ellipsoid { .. }), |g| matches!(g, G::Polyline(_))];
        for kind in kinds {
            let mut batch = ng_render::annotations::MarkBatch::default();
            for (k, a) in annotations.iter().enumerate().filter(|(_, a)| kind(&a.geometry)) {
                let Some(s) = styles.get(k).filter(|s| !s.discard) else { continue };
                let lit = |c: [f32; 4]| highlight(c, hovered == Some(a.id.as_str()));
                let pick = register(&a.id, AnnotationPart::Body);
                let marker = |center: [f64; 4], end: usize, pick: u32| ng_render::annotations::Circle {
                    center,
                    diameter: s.marker_size[end] as f64,
                    border_width: s.marker_border_width[end] as f64,
                    feather: 0.5,
                    interior: lit(s.marker_color[end]),
                    border: s.marker_border_color[end],
                    pick,
                };
                match &a.geometry {
                    G::Point(p) => batch.circles.push(ng_render::annotations::Circle {
                        center: to_clip(p),
                        diameter: s.point_size as f64,
                        border_width: s.point_border_width as f64,
                        feather: 1.0,
                        interior: lit(s.point_color),
                        border: s.point_border_color,
                        pick,
                    }),
                    G::Line(pa, pb) => {
                        let (ca, cb) = (to_clip(pa), to_clip(pb));
                        batch.lines.push(ng_render::annotations::Line { a: ca, b: cb, width: s.line_width as f64, feather: 1.0, color: lit(s.line_color[0]), color_b: lit(s.line_color[1]), pick });
                        batch.circles.push(marker(ca, 0, register(&a.id, AnnotationPart::Endpoint(0))));
                        batch.circles.push(marker(cb, 1, register(&a.id, AnnotationPart::Endpoint(1))));
                    }
                    G::Polyline(points) => {
                        // Each segment a line from the start colour to the
                        // end one; a marker at each segment's start (the
                        // start's style) and at the last one's end.
                        let clips: Vec<[f64; 4]> = points.iter().map(|p| to_clip(p)).collect();
                        for w in clips.windows(2) {
                            batch.lines.push(ng_render::annotations::Line { a: w[0], b: w[1], width: s.poly_line_width as f64, feather: 1.0, color: lit(s.poly_line_color[0]), color_b: lit(s.poly_line_color[1]), pick });
                        }
                        if clips.len() >= 2 {
                            // `snapPosition`'s endpoints: each marker the start of
                            // its segment (the first point), the last the end.
                            let (start, end) = (register(&a.id, AnnotationPart::Endpoint(0)), register(&a.id, AnnotationPart::Endpoint(1)));
                            for (k, &c) in clips.iter().enumerate() {
                                let e = usize::from(k + 1 == clips.len());
                                batch.circles.push(ng_render::annotations::Circle {
                                    center: c,
                                    diameter: s.poly_marker_size[e] as f64,
                                    border_width: s.poly_marker_border_width[e] as f64,
                                    feather: 0.5,
                                    interior: lit(s.poly_marker_color[e]),
                                    border: s.poly_marker_border_color[e],
                                    pick: if e == 1 { end } else { start },
                                });
                            }
                        }
                    }
                    G::AxisAlignedBoundingBox(pa, pb) => {
                        let lower: Vec<f64> = pa.iter().zip(pb).map(|(a, b)| a.min(*b)).collect();
                        let upper: Vec<f64> = pa.iter().zip(pb).map(|(a, b)| a.max(*b)).collect();
                        boxes(&lower, &upper, lit(s.box_border_color), s.box_border_width as f64, pick);
                    }
                    G::Ellipsoid { .. } => {
                        if let Some(Some(e)) = ellipses.get(k) {
                            batch.ellipses.push(ng_render::annotations::Ellipse { pick, ..*e });
                        }
                    }
                }
            }
            self.batches.push(batch);
        }
    }
}

/// `setPartIndex`'s highlight: the hovered annotation's colour three
/// quarters of the way to white, its alpha kept.
pub(crate) fn highlight(c: [f32; 4], hovered: bool) -> [f32; 4] {
    if hovered {
        [c[0] + (1.0 - c[0]) * 0.75, c[1] + (1.0 - c[1]) * 0.75, c[2] + (1.0 - c[2]) * 0.75, c[3]]
    } else {
        c
    }
}

/// `getRepresentativePoint`: what a drag moves (a line's or box's first
/// point unless an endpoint is picked; a polyline's first point or the
/// endpoint; an ellipsoid's centre).
pub(crate) fn representative_point(g: &annotations::Geometry, part: AnnotationPart) -> Vec<f64> {
    use annotations::Geometry as G;
    match (g, part) {
        (G::Point(p), _) => p.clone(),
        (G::Line(_, pb), AnnotationPart::Endpoint(1)) => pb.clone(),
        (G::Line(pa, _), _) | (G::AxisAlignedBoundingBox(pa, _), _) => pa.clone(),
        (G::Ellipsoid { center, .. }, _) => center.clone(),
        (G::Polyline(points), AnnotationPart::Endpoint(k)) => points.get(k).or(points.first()).cloned().unwrap_or_default(),
        (G::Polyline(points), _) => points.first().cloned().unwrap_or_default(),
    }
}

/// `updateViaRepresentativePoint`: the annotation with its representative
/// point at `p` (float32): the whole moved, or just the endpoint picked.
pub(crate) fn update_via_representative_point(g: &annotations::Geometry, p: &[f64], part: AnnotationPart) -> annotations::Geometry {
    use annotations::Geometry as G;
    let p: Vec<f64> = p.iter().map(|x| *x as f32 as f64).collect();
    // `old + (p - from)`, as float32 arrays hold it.
    let shift = |old: &[f64], from: &[f64]| -> Vec<f64> { old.iter().zip(from).zip(&p).map(|((o, f), x)| (o + (x - f)) as f32 as f64).collect() };
    match (g, part) {
        (G::Point(_), _) => G::Point(p),
        (G::Line(pa, _), AnnotationPart::Endpoint(0)) => G::Line(p, g_line_b(g).unwrap_or_else(|| pa.clone())),
        (G::Line(pa, _), AnnotationPart::Endpoint(1)) => G::Line(pa.clone(), p),
        (G::Line(pa, pb), _) => G::Line(p.clone(), shift(pb, pa)),
        (G::AxisAlignedBoundingBox(pa, pb), _) => G::AxisAlignedBoundingBox(p.clone(), shift(pb, pa)),
        (G::Ellipsoid { radii, .. }, _) => G::Ellipsoid { center: p, radii: radii.clone() },
        (G::Polyline(points), AnnotationPart::Endpoint(k)) if k < points.len() => {
            let mut points = points.clone();
            points[k] = p;
            G::Polyline(points)
        }
        (G::Polyline(points), _) => {
            let from = points.first().cloned().unwrap_or_default();
            G::Polyline(points.iter().map(|q| shift(q, &from)).collect())
        }
    }
}

fn g_line_b(g: &annotations::Geometry) -> Option<Vec<f64>> {
    match g {
        annotations::Geometry::Line(_, pb) => Some(pb.clone()),
        _ => None,
    }
}

/// An annotation's JSON with `g`'s points (`annotationToJson`).
pub(crate) fn write_geometry(v: &mut Value, g: &annotations::Geometry) {
    use annotations::Geometry as G;
    use ng_core::js::numbers as nums;
    let Some(o) = v.as_object_mut() else { return };
    match g {
        G::Point(p) => {
            o.insert("point".into(), nums(p));
        }
        G::Line(a, b) | G::AxisAlignedBoundingBox(a, b) => {
            o.insert("pointA".into(), nums(a));
            o.insert("pointB".into(), nums(b));
        }
        G::Ellipsoid { center, radii } => {
            o.insert("center".into(), nums(center));
            o.insert("radii".into(), nums(radii));
        }
        G::Polyline(points) => {
            o.insert("points".into(), Value::Array(points.iter().map(|p| nums(p)).collect()));
        }
    }
}
