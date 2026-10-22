//! Anti-aliased lines as neuroglancer draws them (`src/webgl/lines.ts`): each
//! segment a quad `lineWidth + 2 · featherWidth` pixels wide, centred on the
//! segment and not extended past its ends; alpha
//! `clamp((1 − |lineCoord|) / featherFraction, 0, 1)` across it, where
//! `lineCoord` runs −1 … 1 over the width and `featherFraction =
//! max(1e-6, feather) / totalWidth`. Blended `SRC_ALPHA, ONE_MINUS_SRC_ALPHA`.
//!
//! The quads are built on the CPU with the vertex shader's arithmetic, in
//! normalized device coordinates of the panel.

use threers::renderer::PixelRect;
use wgpu::util::DeviceExt;

use crate::compose::{LAYER_BLEND, OFFSCREEN_FORMAT};

const SHADER: &str = r#"
struct VertexIn {
    @location(0) position : vec2<f32>,
    @location(1) line_coord : f32,
    @location(2) feather_fraction : f32,
    @location(3) color : vec4<f32>,
};
struct VertexOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) line_coord : f32,
    @location(1) feather_fraction : f32,
    @location(2) color : vec4<f32>,
};
@vertex
fn vs(in : VertexIn) -> VertexOut {
    var out : VertexOut;
    out.clip = vec4<f32>(in.position, 0.0, 1.0);
    out.line_coord = in.line_coord;
    out.feather_fraction = in.feather_fraction;
    out.color = in.color;
    return out;
}
@fragment
fn fs(in : VertexOut) -> @location(0) vec4<f32> {
    // `getLineAlpha`.
    let alpha = clamp((1.0 - abs(in.line_coord)) / in.feather_fraction, 0.0, 1.0);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
"#;

/// One line: endpoints in panel pixels (top-left origin), width and feather
/// in pixels, colour (display-referred RGB) and alpha.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    pub a: [f64; 2],
    pub b: [f64; 2],
    pub width: f64,
    pub feather: f64,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    line_coord: f32,
    feather_fraction: f32,
    color: [f32; 4],
}

pub struct LineRenderer {
    pipeline: wgpu::RenderPipeline,
}

impl LineRenderer {
    /// Draws into panel textures ([`OFFSCREEN_FORMAT`]).
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ng lines"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ng lines layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ng lines"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32, 2 => Float32, 3 => Float32x4],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: OFFSCREEN_FORMAT,
                    blend: Some(LAYER_BLEND),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline }
    }

    /// The quad of one line (two triangles) in NDC of a `width` × `height`
    /// panel: `emitLine` with `getQuadVertexPosition(vec2(0, -1), vec2(1, 1))`.
    fn quad(line: &Line, width: f64, height: f64) -> Option<[Vertex; 6]> {
        let total = if line.width == 0.0 { 0.0 } else { line.width + 2.0 * line.feather };
        if total == 0.0 {
            return None;
        }
        // Panel pixels (y down) → NDC (y up), in f32 as the shader works.
        let ndc = |p: [f64; 2]| [(p[0] / width * 2.0 - 1.0) as f32, (1.0 - p[1] / height * 2.0) as f32];
        let (a, b) = (ndc(line.a), ndc(line.b));
        let params = [(1.0 / width) as f32, (1.0 / height) as f32];
        let d = [b[0] - a[0], b[1] - a[1]];
        let pixel_length = ((d[0] / params[0] * 0.5).powi(2) + (d[1] / params[1] * 0.5).powi(2)).sqrt();
        let dir = if pixel_length < 1e-3 {
            [1.0f32, 0.0]
        } else {
            let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
            [d[0] / l, d[1] / l]
        };
        // `normalize(vec2(dir.y, -dir.x) / uLineParams.yx * uLineParams.xy)`.
        let n = [dir[1] / params[1] * params[0], -dir[0] / params[0] * params[1]];
        let nl = (n[0] * n[0] + n[1] * n[1]).sqrt();
        let n = [n[0] / nl, n[1] / nl];
        let feather_fraction = (line.feather.max(1e-6) / total) as f32;
        let total = total as f32;
        let vertex = |x: f32, y: f32| {
            let base = [a[0] + (b[0] - a[0]) * x, a[1] + (b[1] - a[1]) * x];
            Vertex {
                position: [base[0] + y * n[0] * total * params[0], base[1] + y * n[1] * total * params[1]],
                line_coord: y,
                feather_fraction,
                color: line.color,
            }
        };
        let (v00, v10, v01, v11) = (vertex(0.0, -1.0), vertex(1.0, -1.0), vertex(0.0, 1.0), vertex(1.0, 1.0));
        Some([v00, v10, v11, v00, v11, v01])
    }

    /// Draws `lines` over `panel` (a panel texture `size` pixels), blending.
    /// `flipped`: `panel` is upside down (row 0 at the bottom); the lines are
    /// given the right way up.
    pub fn draw(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, panel: &wgpu::TextureView, size: PixelRect, lines: &[Line], flipped: bool) {
        let h = size.height as f64;
        let flip = |p: [f64; 2]| if flipped { [p[0], h - p[1]] } else { p };
        let vertices: Vec<Vertex> = lines
            .iter()
            .map(|l| Line { a: flip(l.a), b: flip(l.b), ..*l })
            .filter_map(|l| Self::quad(&l, size.width as f64, h))
            .flatten()
            .collect();
        if vertices.is_empty() {
            return;
        }
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ng lines"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut pass = threers::renderer::fullscreen::begin(encoder, panel, threers::renderer::fullscreen::Load::Keep);
        pass.set_viewport(0.0, 0.0, size.width as f32, size.height as f32, 0.0, 1.0);
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
}

const AXES_SHADER: &str = r#"
struct VertexIn {
    @location(0) position : vec2<f32>,
    @location(1) color : vec4<f32>,
};
struct VertexOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
};
@vertex
fn vs(in : VertexIn) -> VertexOut {
    var out : VertexOut;
    out.clip = vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}
@fragment
fn fs(in : VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid : u32) -> @builtin(position) vec4<f32> {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}
@fragment
fn fs_zero_alpha() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
"#;

/// The axis lines of a cross-section (`AxesLineHelper`): 1 px `GL_LINES`
/// from the position along each display axis, red, green, blue at alpha 0.5.
/// Drawn as neuroglancer draws them: the panel's alpha cleared to 0 first,
/// then blended `ONE_MINUS_DST_ALPHA, DST_ALPHA` — so a line shows in full
/// colour where it is alone, and overlapping lines mix.
pub struct AxisLineRenderer {
    zero_alpha: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AxisVertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl AxisLineRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ng axis lines"),
            source: wgpu::ShaderSource::Wgsl(AXES_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ng axis lines layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let zero_alpha = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ng clear panel alpha"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_zero_alpha"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: OFFSCREEN_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALPHA,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
            dst_factor: wgpu::BlendFactor::DstAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let lines = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ng axis lines"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<AxisVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: OFFSCREEN_FORMAT,
                    blend: Some(wgpu::BlendState { color: component, alpha: component }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { zero_alpha, lines }
    }

    /// Draws the three axes over `panel` (`size` pixels): `origin` and the
    /// ends of the x, y, z axes, in the panel's NDC. `flipped`: `panel` is
    /// upside down (as neuroglancer's GL framebuffer is, which decides which
    /// pixels a line on a boundary, or a line's end, takes).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        panel: &wgpu::TextureView,
        size: PixelRect,
        origin: [f32; 2],
        ends: [[f32; 2]; 3],
        flipped: bool,
    ) {
        const ALPHA: f32 = 0.5;
        let flip = |p: [f32; 2]| if flipped { [p[0], -p[1]] } else { p };
        let origin = flip(origin);
        let ends = ends.map(flip);
        let colors = [[1.0, 0.0, 0.0, ALPHA], [0.0, 1.0, 0.0, ALPHA], [0.0, 0.0, 1.0, ALPHA]];
        let vertices: Vec<AxisVertex> = (0..3)
            .flat_map(|i| [AxisVertex { position: origin, color: colors[i] }, AxisVertex { position: ends[i], color: colors[i] }])
            .collect();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ng axis lines"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut pass = threers::renderer::fullscreen::begin(encoder, panel, threers::renderer::fullscreen::Load::Keep);
        pass.set_viewport(0.0, 0.0, size.width as f32, size.height as f32, 0.0, 1.0);
        pass.set_pipeline(&self.zero_alpha);
        pass.draw(0..3, 0..1);
        pass.set_pipeline(&self.lines);
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(0..6, 0..1);
    }
}
