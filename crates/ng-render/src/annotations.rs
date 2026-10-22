//! Annotations as neuroglancer draws them (`annotation/point.ts`,
//! `annotation/line.ts`, `webgl/lines.ts`, `webgl/circles.ts`): feathered
//! lines (`emitLine`) and bordered circles (`emitCircle`,
//! `getCircleColor`), positions in GL clip coordinates. In a cross-section
//! (`crossSectionFade`) alpha falls off with depth, `1 − 2·|0.5 − z|`, and
//! the depth range clips; blended `SRC_ALPHA, ONE_MINUS_SRC_ALPHA`.
//!
//! The quads are built on the CPU with the vertex shaders' arithmetic; the
//! target is drawn upside down (y mirrored), as neuroglancer's framebuffer.

use threers::renderer::PixelRect;
use wgpu::util::DeviceExt;

use crate::compose::{LAYER_BLEND, OFFSCREEN_FORMAT};

pub(crate) const SHADER: &str = r#"
struct LineIn {
    @location(0) position : vec3<f32>,
    @location(1) line_coord : f32,
    @location(2) feather_fraction : f32,
    @location(3) fade : f32,
    @location(4) color : vec4<f32>,
};
struct LineOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) line_coord : f32,
    @location(1) feather_fraction : f32,
    @location(2) fade : f32,
    @location(3) color : vec4<f32>,
};
@vertex
fn vs_line(in : LineIn) -> LineOut {
    var out : LineOut;
    out.clip = vec4<f32>(in.position, 1.0);
    out.line_coord = in.line_coord;
    out.feather_fraction = in.feather_fraction;
    out.fade = in.fade;
    out.color = in.color;
    return out;
}
fn cross_section_fade(z : f32, on : f32) -> f32 {
    return select(1.0, clamp(1.0 - 2.0 * abs(0.5 - z), 0.0, 1.0), on > 0.5);
}
@fragment
fn fs_line(in : LineOut) -> @location(0) vec4<f32> {
    let alpha = clamp((1.0 - abs(in.line_coord)) / in.feather_fraction, 0.0, 1.0);
    return vec4<f32>(in.color.rgb, in.color.a * alpha * cross_section_fade(in.clip.z, in.fade));
}

struct CircleIn {
    @location(0) clip : vec4<f32>,
    @location(1) coord : vec4<f32>,
    @location(2) interior : vec4<f32>,
    @location(3) border : vec4<f32>,
    @location(4) fade : f32,
    @location(5) pick : u32,
};
struct CircleOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) coord : vec4<f32>,
    @location(1) interior : vec4<f32>,
    @location(2) border : vec4<f32>,
    @location(3) fade : f32,
    @location(4) @interpolate(flat) pick : u32,
};
@vertex
fn vs_circle(in : CircleIn) -> CircleOut {
    var out : CircleOut;
    out.clip = in.clip;
    out.coord = in.coord;
    out.interior = in.interior;
    out.border = in.border;
    out.fade = in.fade;
    out.pick = in.pick;
    return out;
}
@fragment
fn fs_circle(in : CircleOut) -> @location(0) vec4<f32> {
    return circle_color(in);
}
// `getCircleColor`.
fn circle_color(in : CircleOut) -> vec4<f32> {
    let radius = length(in.coord.xy);
    if (radius > 1.0) {
        discard;
    }
    let border_fraction = clamp((radius - in.coord.z) / in.coord.w, 0.0, 1.0);
    let feather = clamp((1.0 - radius) / in.coord.w, 0.0, 1.0);
    let color = mix(in.interior, in.border, border_fraction);
    return vec4<f32>(color.rgb, color.a * feather * cross_section_fade(in.clip.z, in.fade));
}
@fragment
fn fs_circle_oit(in : CircleOut) -> OitOut {
    return emit_oit(circle_color(in), in.clip.z);
}
@fragment
fn fs_circle_emit(in : CircleOut) -> EmitOut {
    return emit(circle_color(in), in.clip.z, in.pick);
}
"#;

/// Marks drawn together: lines, then ellipses, then circles.
#[derive(Debug, Clone, Default)]
pub struct MarkBatch {
    pub lines: Vec<Line>,
    pub circles: Vec<Circle>,
    pub ellipses: Vec<Ellipse>,
}

/// A line (`emitLine`): endpoints in GL clip coordinates, width and feather
/// in pixels, colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    pub a: [f64; 4],
    pub b: [f64; 4],
    pub width: f64,
    pub feather: f64,
    pub color: [f32; 4],
    /// The colour at `b` (`setLineColor(start, end)`), mixed along it.
    pub color_b: [f32; 4],
    /// The pick ID (for hit-testing; 0 for none).
    pub pick: u32,
}

/// A circle (`emitCircle`): centre in GL clip coordinates, diameter, border
/// width and feather in pixels, interior and border colours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: [f64; 4],
    pub diameter: f64,
    pub border_width: f64,
    pub feather: f64,
    pub interior: [f32; 4],
    pub border: [f32; 4],
    /// The pick ID written in the 3D view (0 for none).
    pub pick: u32,
}

/// A filled ellipse (an ellipsoid's cross-section): its bounding quad's
/// corners in GL clip coordinates, for offsets (−1, −1), (1, −1), (1, 1),
/// (−1, 1); fragments outside the unit circle of those offsets are
/// discarded, without antialiasing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    pub corners: [[f64; 4]; 4],
    pub color: [f32; 4],
    /// The pick ID (for hit-testing; 0 for none).
    pub pick: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    position: [f32; 3],
    line_coord: f32,
    feather_fraction: f32,
    fade: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CircleVertex {
    clip: [f32; 4],
    coord: [f32; 4],
    interior: [f32; 4],
    border: [f32; 4],
    fade: f32,
    pick: u32,
    _pad: [f32; 2],
}

pub struct AnnotationRenderer {
    lines: wgpu::RenderPipeline,
    circles: wgpu::RenderPipeline,
}

impl AnnotationRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ng annotations"), source: wgpu::ShaderSource::Wgsl(format!("{}{}{SHADER}", crate::perspective::OIT_SHADER, crate::perspective::EMIT_SHADER).into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ng annotations"), bind_group_layouts: &[], immediate_size: 0 });
        let make = |label: &str, vs: &str, fs: &str, buffer: wgpu::VertexBufferLayout| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some(vs), buffers: &[Some(buffer)], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    targets: &[Some(wgpu::ColorTargetState { format: OFFSCREEN_FORMAT, blend: Some(LAYER_BLEND), write_mask: wgpu::ColorWrites::ALL })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let lines = make(
            "ng annotation lines",
            "vs_line",
            "fs_line",
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32, 3 => Float32, 4 => Float32x4],
            },
        );
        let circles = make(
            "ng annotation circles",
            "vs_circle",
            "fs_circle",
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CircleVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32, 5 => Uint32],
            },
        );
        Self { lines, circles }
    }

    /// Draws `lines` then `circles` over `target` (upside down), `size`
    /// pixels; `fade`: in a cross-section.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, size: PixelRect, lines: &[Line], circles: &[Circle], ellipses: &[Ellipse], fade: bool) {
        let batch = MarkBatch { lines: lines.to_vec(), circles: circles.to_vec(), ellipses: ellipses.to_vec() };
        self.draw_batches(device, encoder, target, size, std::slice::from_ref(&batch), fade);
    }

    /// [`draw`](Self::draw) for `batches` in turn, each its lines, then
    /// ellipses, then circles (a skeleton's edges then nodes; an
    /// annotation type's marks).
    pub fn draw_batches(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, size: PixelRect, batches: &[MarkBatch], fade: bool) {
        let (w, h) = (size.width as f64, size.height as f64);
        let fade = if fade { 1.0 } else { 0.0 };
        // Per batch, then the rest: line and circle vertex ranges.
        let mut line_vertices: Vec<LineVertex> = vec![];
        let mut circle_vertices: Vec<CircleVertex> = vec![];
        let mut ranges: Vec<(std::ops::Range<u32>, std::ops::Range<u32>)> = vec![];
        let mut add = |lines: &[Line], ellipses: &[Ellipse], circles: &[Circle], line_vertices: &mut Vec<LineVertex>, circle_vertices: &mut Vec<CircleVertex>| {
            let (l0, c0) = (line_vertices.len() as u32, circle_vertices.len() as u32);
            line_vertices.extend(lines.iter().filter_map(|l| line_quad(l, w, h, fade)).flatten());
            // Ellipses first (`SliceViewRenderHelper` order is by annotation
            // type: points, lines, boxes, ellipsoids — drawn as given here).
            circle_vertices.extend(ellipses.iter().flat_map(ellipse_quad));
            circle_vertices.extend(circles.iter().flat_map(|c| circle_quad(c, w, h, fade)));
            ranges.push((l0..line_vertices.len() as u32, c0..circle_vertices.len() as u32));
        };
        for b in batches {
            add(&b.lines, &b.ellipses, &b.circles, &mut line_vertices, &mut circle_vertices);
        }
        if line_vertices.is_empty() && circle_vertices.is_empty() {
            return;
        }
        let buffer = |label, contents: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::VERTEX });
        let line_buffer = (!line_vertices.is_empty()).then(|| buffer("ng annotation lines", bytemuck::cast_slice(&line_vertices)));
        let circle_buffer = (!circle_vertices.is_empty()).then(|| buffer("ng annotation circles", bytemuck::cast_slice(&circle_vertices)));
        let mut pass = threers::renderer::fullscreen::begin(encoder, target, threers::renderer::fullscreen::Load::Keep);
        pass.set_viewport(0.0, 0.0, size.width as f32, size.height as f32, 0.0, 1.0);
        for (lines, circles) in ranges {
            if let Some(b) = line_buffer.as_ref().filter(|_| !lines.is_empty()) {
                pass.set_pipeline(&self.lines);
                pass.set_vertex_buffer(0, b.slice(..));
                pass.draw(lines, 0..1);
            }
            if let Some(b) = circle_buffer.as_ref().filter(|_| !circles.is_empty()) {
                pass.set_pipeline(&self.circles);
                pass.set_vertex_buffer(0, b.slice(..));
                pass.draw(circles, 0..1);
            }
        }
    }
}

/// `clipLineToDepthRange` on GL clip coordinates.
pub fn clip_to_depth_range(a: [f64; 4], b: [f64; 4]) -> Option<([f64; 4], [f64; 4])> {
    let (mut tmin, mut tmax) = (0.0f64, 1.0f64);
    let k1 = b[3] - a[3] + a[2] - b[2];
    let k2 = a[3] - b[3] + a[2] - b[2];
    let q1 = (a[2] - a[3]) / k1;
    let q2 = (a[2] + a[3]) / k2;
    if k1 > 0.0 {
        tmin = tmin.max(q1);
    } else if k1 < 0.0 {
        tmax = tmax.min(q1);
    }
    if k2 > 0.0 {
        tmax = tmax.min(q2);
    } else if k2 < 0.0 {
        tmin = tmin.max(q2);
    }
    if tmin > tmax {
        return None;
    }
    let mix = |t: f64| std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t);
    Some((mix(tmin), mix(tmax)))
}

/// `emitLine`: the quad in device coordinates, y mirrored, depth
/// `(z + 1) / 2`.
fn line_quad(l: &Line, width: f64, height: f64, fade: f32) -> Option<[LineVertex; 6]> {
    let total = if l.width == 0.0 { 0.0 } else { l.width + 2.0 * l.feather };
    if total == 0.0 {
        return None;
    }
    let (a, b) = clip_to_depth_range(l.a, l.b)?;
    // `getLineEndpointCoefficient`: the end of the (clipped) quad, so the
    // colours span what is drawn.
    let color_at = |x: f64| -> [f32; 4] { std::array::from_fn(|i| l.color[i] + (l.color_b[i] - l.color[i]) * x as f32) };
    let mut da = [a[0] / a[3], a[1] / a[3], a[2] / a[3]];
    let mut db = [b[0] / b[3], b[1] / b[3], b[2] / b[3]];
    let params = [1.0 / width, 1.0 / height];
    let d = [db[0] - da[0], db[1] - da[1]];
    let pixel_length = ((d[0] / params[0] * 0.5).powi(2) + (d[1] / params[1] * 0.5).powi(2)).sqrt();
    let dir = if pixel_length < 1e-3 {
        da[2] = 0.0;
        db[2] = 0.0;
        [1.0, 0.0]
    } else {
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        [d[0] / len, d[1] / len]
    };
    let n = [dir[1] / params[1] * params[0], -dir[0] / params[0] * params[1]];
    let nl = (n[0] * n[0] + n[1] * n[1]).sqrt();
    let n = [n[0] / nl, n[1] / nl];
    let feather_fraction = (l.feather.max(1e-6) / total) as f32;
    let vertex = |x: f64, y: f64| {
        let p = [da[0] + (db[0] - da[0]) * x, da[1] + (db[1] - da[1]) * x, da[2] + (db[2] - da[2]) * x];
        let px = p[0] + y * n[0] * total * params[0];
        let py = p[1] + y * n[1] * total * params[1];
        LineVertex { position: [px as f32, -py as f32, ((p[2] + 1.0) / 2.0) as f32], line_coord: y as f32, feather_fraction, fade, color: color_at(x) }
    };
    let (v00, v10, v01, v11) = (vertex(0.0, -1.0), vertex(1.0, -1.0), vertex(0.0, 1.0), vertex(1.0, 1.0));
    Some([v00, v10, v11, v00, v11, v01])
}

/// An ellipse's quad through the circle shader: no border, a vanishing
/// feather, so only the discard shapes it.
fn ellipse_quad(e: &Ellipse) -> Vec<CircleVertex> {
    let offsets = [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let v = |k: usize| {
        let p = e.corners[k];
        CircleVertex {
            clip: [p[0] as f32, -p[1] as f32, ((p[2] + p[3]) / 2.0) as f32, p[3] as f32],
            coord: [offsets[k].0, offsets[k].1, 2.0, 1e-6],
            interior: e.color,
            border: e.color,
            fade: 0.0,
            pick: 0,
            _pad: [0.0; 2],
        }
    };
    vec![v(0), v(1), v(2), v(0), v(2), v(3)]
}

/// `emitCircle`: a quad `diameter + 2 · (border + feather)` pixels across.
pub(crate) fn circle_quad(c: &Circle, width: f64, height: f64, fade: f32) -> Vec<CircleVertex> {
    let mut total = c.diameter + 2.0 * (c.border_width + c.feather);
    if c.diameter == 0.0 {
        total = 0.0;
    }
    let (z, w) = if c.border_width == 0.0 { (total, 1e-6) } else { (c.diameter / total, c.feather / total) };
    let p = c.center;
    let vertex = |cx: f64, cy: f64| {
        let x = p[0] + cx * (1.0 / width) * p[3] * total;
        let y = p[1] + cy * (1.0 / height) * p[3] * total;
        CircleVertex {
            // Mirrored, and z to WebGPU's depth range.
            clip: [x as f32, -y as f32, ((p[2] + p[3]) / 2.0) as f32, p[3] as f32],
            coord: [cx as f32, cy as f32, z as f32, w as f32],
            interior: c.interior,
            border: c.border,
            fade,
            pick: c.pick,
            _pad: [0.0; 2],
        }
    };
    let (a, b, cc, d) = (vertex(-1.0, -1.0), vertex(1.0, -1.0), vertex(1.0, 1.0), vertex(-1.0, 1.0));
    vec![a, b, cc, a, cc, d]
}
