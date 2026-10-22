//! The 3D panel as neuroglancer draws it by default (`PerspectivePanel`,
//! `src/perspective_view/panel.ts`): the projection background, the
//! cross-sections as planes (each slice view's framebuffer, `NEAREST`
//! sampled, the background where its alpha is 0, lit by
//! `ambient + |light · normal| · directional`), annotation lines (data-bounds
//! box edges), and axis lines — all with a depth buffer.
//!
//! Geometry arrives in GL clip coordinates (z from −w to w). It is drawn with
//! y mirrored into an upside-down target, in the orientation of
//! neuroglancer's GL framebuffer, so rasterization ties fall the same way;
//! z becomes `(z + w) / 2` for WebGPU's depth range, which keeps the same
//! window depths.

use wgpu::util::DeviceExt;

use crate::compose::{DEPTH_FORMAT, LAYER_BLEND, OFFSCREEN_FORMAT};

/// `glsl_perspectivePanelEmitOIT`: weighted blended order-independent
/// transparency; the colour is premultiplied by alpha.
pub(crate) const OIT_SHADER: &str = r#"
struct OitOut {
    @location(0) accum : vec4<f32>,
    @location(1) alpha : vec4<f32>,
};
fn compute_oit_weight(alpha : f32, depth : f32) -> f32 {
    let a = min(1.0, alpha) * 8.0 + 0.01;
    let b = -depth * 0.95 + 1.0;
    return a * a * a * b * b * b;
}
fn emit_oit(color : vec4<f32>, depth : f32) -> OitOut {
    let weight = compute_oit_weight(color.a, depth);
    let accum = color * weight;
    return OitOut(vec4<f32>(accum.rgb, color.a), vec4<f32>(accum.a, 0.0, 0.0, 0.0));
}
"#;

/// `glsl_perspectivePanelEmit` for the opaque pass's three targets.
pub(crate) const EMIT_SHADER: &str = r#"
// `glsl_perspectivePanelEmit`: the colour, `1 − depth` (0 where nothing
// is drawn) and the pick ID.
struct EmitOut {
    @location(0) color : vec4<f32>,
    @location(1) z : vec4<f32>,
    @location(2) pick : vec4<u32>,
};
fn emit(color : vec4<f32>, depth : f32, pick : u32) -> EmitOut {
    let z = 1.0 - depth;
    return EmitOut(color, vec4<f32>(z, z, z, 1.0), vec4<u32>(pick, pick, pick, 1u));
}
"#;

const SHADER: &str = r#"
struct QuadIn {
    @location(0) clip : vec4<f32>,
    @location(1) uv : vec2<f32>,
};
struct QuadOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv : vec2<f32>,
};
struct QuadUniforms {
    background : vec4<f32>,
    factor : vec4<f32>,
};
@group(0) @binding(0) var slice_texture : texture_2d<f32>;
@group(0) @binding(1) var slice_sampler : sampler;
@group(0) @binding(2) var<uniform> quad : QuadUniforms;

@vertex
fn vs_quad(in : QuadIn) -> QuadOut {
    var out : QuadOut;
    out.clip = in.clip;
    out.uv = in.uv;
    return out;
}
@fragment
fn fs_quad(in : QuadOut) -> EmitOut {
    var c = textureSample(slice_texture, slice_sampler, in.uv);
    if (c.a == 0.0) {
        // `hideCrossSectionBackground3D`: a negative alpha marks it.
        if (quad.background.a < 0.0) {
            discard;
        }
        c = quad.background;
    }
    return emit(c * quad.factor, in.clip.z, 0u);
}

struct LineIn {
    @location(0) position : vec3<f32>,
    @location(1) line_coord : f32,
    @location(2) feather_fraction : f32,
    @location(3) color : vec4<f32>,
    @location(4) pick : u32,
};
struct LineOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) line_coord : f32,
    @location(1) feather_fraction : f32,
    @location(2) color : vec4<f32>,
    @location(3) @interpolate(flat) pick : u32,
};
@vertex
fn vs_line(in : LineIn) -> LineOut {
    var out : LineOut;
    out.clip = vec4<f32>(in.position, 1.0);
    out.line_coord = in.line_coord;
    out.feather_fraction = in.feather_fraction;
    out.color = in.color;
    out.pick = in.pick;
    return out;
}
@fragment
fn fs_line(in : LineOut) -> EmitOut {
    return emit(line_color(in), in.clip.z, in.pick);
}
fn line_color(in : LineOut) -> vec4<f32> {
    let alpha = clamp((1.0 - abs(in.line_coord)) / in.feather_fraction, 0.0, 1.0);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}

struct AxisIn {
    @location(0) clip : vec4<f32>,
    @location(1) color : vec4<f32>,
};
struct AxisOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
};
@vertex
fn vs_axis(in : AxisIn) -> AxisOut {
    var out : AxisOut;
    out.clip = in.clip;
    out.color = in.color;
    return out;
}
@fragment
fn fs_axis(in : AxisOut) -> EmitOut {
    return emit(in.color, in.clip.z, 0u);
}

// `MeshShaderManager`: lit by `|normal · light| + ambient`.
struct MeshUniforms {
    mvp : mat4x4<f32>,
    normal0 : vec4<f32>,
    normal1 : vec4<f32>,
    normal2 : vec4<f32>,
    light : vec4<f32>,
    color : vec4<f32>,
    // `uFragmentOrigin`, `uFragmentShape`: positions are fractions of the
    // fragment's box (multi-resolution meshes; else 0 and 1). The shape's w:
    // `uSilhouettePower`, 0 for none.
    fragment_origin : vec4<f32>,
    fragment_shape : vec4<f32>,
};
@group(0) @binding(0) var<uniform> mesh : MeshUniforms;
struct MeshOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
    // `uPickID`: the origin's w, as bits.
    @location(1) @interpolate(flat) pick : u32,
};
fn decode_normal(e : vec2<f32>) -> vec3<f32> {
    var v = vec3<f32>(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        v = vec3<f32>((1.0 - abs(v.yx)) * vec2<f32>(select(-1.0, 1.0, v.x > 0.0), select(-1.0, 1.0, v.y > 0.0)), v.z);
    }
    return normalize(v);
}
@vertex
fn vs_mesh(@location(0) position : vec3<f32>, @location(1) encoded : vec2<f32>) -> MeshOut {
    var out : MeshOut;
    let vertex_position = mesh.fragment_origin.xyz + mesh.fragment_shape.xyz * position;
    let c = mesh.mvp * vec4<f32>(vertex_position, 1.0);
    out.clip = vec4<f32>(c.x, -c.y, (c.z + c.w) * 0.5, c.w);
    let normal_matrix = mat3x3<f32>(mesh.normal0.xyz, mesh.normal1.xyz, mesh.normal2.xyz);
    let normal_multiplier = 1.0 / mesh.fragment_shape.xyz;
    let normal = normalize(normal_matrix * (normal_multiplier * decode_normal(encoded)));
    let abs_cos_angle = abs(dot(normal, mesh.light.xyz));
    let lighting = abs_cos_angle + mesh.light.w;
    out.color = vec4<f32>(lighting * mesh.color.rgb, mesh.color.a);
    out.pick = bitcast<u32>(mesh.fragment_origin.w);
    if (mesh.fragment_shape.w > 0.0) {
        out.color = out.color * pow(1.0 - abs_cos_angle, mesh.fragment_shape.w);
    }
    return out;
}
@fragment
fn fs_mesh(in : MeshOut) -> EmitOut {
    return emit(in.color, in.clip.z, in.pick);
}
@fragment
fn fs_mesh_oit(in : MeshOut) -> OitOut {
    return emit_oit(in.color, in.clip.z);
}
@fragment
fn fs_line_oit(in : LineOut) -> OitOut {
    return emit_oit(line_color(in), in.clip.z);
}

// `transparencyCopyHelper`: the accumulated colour over what is drawn,
// blended `(ONE_MINUS_SRC_ALPHA, SRC_ALPHA)`.
@group(0) @binding(0) var oit_accum : texture_2d<f32>;
@group(0) @binding(1) var oit_alpha : texture_2d<f32>;
@vertex
fn vs_oit_copy(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs_oit_copy(@builtin(position) p : vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(p.xy);
    let v0 = textureLoad(oit_accum, xy, 0);
    let v1 = textureLoad(oit_alpha, xy, 0);
    // Nothing transparent here: 0 / 0 in GL, left as it was by the
    // revealage of 1.
    if (v1.r == 0.0) {
        discard;
    }
    return vec4<f32>(v0.rgb / v1.r, v0.a);
}
"#;

/// A segment's mesh on the GPU: positions (float32×3), normals
/// (snorm8×2), triangle indices.
pub struct GpuMesh {
    pub positions: wgpu::Buffer,
    pub normals: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub count: u32,
}

impl GpuMesh {
    pub fn new(device: &wgpu::Device, positions: &[f32], normals: &[[i8; 2]], indices: &[u32]) -> Self {
        let buffer = |label, contents: &[u8], usage| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage });
        // Snorm8x2 attributes are 2 bytes; vertex strides must be 4-aligned.
        let padded: Vec<[i8; 4]> = normals.iter().map(|n| [n[0], n[1], 0, 0]).collect();
        GpuMesh {
            positions: buffer("ng mesh positions", bytemuck::cast_slice(positions), wgpu::BufferUsages::VERTEX),
            normals: buffer("ng mesh normals", bytemuck::cast_slice(&padded), wgpu::BufferUsages::VERTEX),
            indices: buffer("ng mesh indices", bytemuck::cast_slice(indices), wgpu::BufferUsages::INDEX),
            count: indices.len() as u32,
        }
    }
}

/// A mesh to draw: its model-view-projection (GL clip, column-major),
/// normal matrix (column-major 3×3), the light (`uLightDirection`: the
/// direction times the directional part, the ambient part in w), and its
/// colour (alpha-premultiplied).
pub struct MeshDraw<'a> {
    pub mesh: &'a GpuMesh,
    pub mvp: [f32; 16],
    pub normal_matrix: [f32; 9],
    pub light: [f32; 4],
    pub color: [f32; 4],
    /// The indices to draw (a sub-chunk range, or all).
    pub indices: std::ops::Range<u32>,
    /// Drawn with the transparent layers (`isTransparent`: object alpha
    /// below 1), blended order-independently.
    pub transparent: bool,
    /// `uFragmentOrigin` and `uFragmentShape` (0 and 1 for meshes in their
    /// own coordinates).
    pub fragment_origin: [f32; 3],
    pub fragment_shape: [f32; 3],
    /// `uSilhouettePower` (`meshSilhouetteRendering`); 0 for none.
    pub silhouette: f32,
    /// The pick ID written (0 for none).
    pub pick: u32,
}

const MESH_UNIFORM_STRIDE: u64 = 256;
const MESH_UNIFORM_BYTES: u64 = 176;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    clip: [f32; 4],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    position: [f32; 3],
    line_coord: f32,
    feather_fraction: f32,
    color: [f32; 4],
    pick: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AxisVertex {
    clip: [f32; 4],
    color: [f32; 4],
}

/// GL clip coordinates → the mirrored WebGPU target's.
fn to_target(c: [f64; 4]) -> [f32; 4] {
    [c[0] as f32, -c[1] as f32, ((c[2] + c[3]) / 2.0) as f32, c[3] as f32]
}

/// A cross-section to show in 3D: its framebuffer, its corners in GL clip
/// coordinates (for the quad corners (−1, −1), (1, −1), (1, 1), (−1, 1)),
/// and its lighting factor.
pub struct SlicePlane<'a> {
    pub texture: &'a wgpu::TextureView,
    pub corners: [[f64; 4]; 4],
    pub factor: f32,
}

/// A volume rendering layer's pipeline (`VolumeRenderingRenderLayer`):
/// its compiled shader (`compile_volume_rendering`) over the transparent
/// pass's targets.
pub struct VolumePipeline {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// Max or min projection: drawn into the projection targets.
    projection: bool,
}

impl VolumePipeline {
    /// For `renderer`'s transparent targets.
    pub fn new(device: &wgpu::Device, renderer: &PerspectiveRenderer, compiled: &crate::user_shader::CompiledShader, mode: crate::user_shader::VolumeMode) -> Self {
        let projection = mode != crate::user_shader::VolumeMode::On;
        let sample_type = match compiled.sample {
            threers::volume::VolumeSample::Float => wgpu::TextureSampleType::Float { filterable: false },
            threers::volume::VolumeSample::Uint => wgpu::TextureSampleType::Uint,
        };
        let texture = |binding, sample_type, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type, view_dimension, multisampled: false },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng volume rendering"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                texture(1, sample_type, wgpu::TextureViewDimension::D3),
                texture(2, wgpu::TextureSampleType::Uint, wgpu::TextureViewDimension::D2),
                texture(3, wgpu::TextureSampleType::Float { filterable: false }, wgpu::TextureViewDimension::D2),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ng volume rendering"), source: wgpu::ShaderSource::Wgsl(compiled.wgsl.clone().into()) });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ng volume rendering"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let target = Some(wgpu::ColorTargetState { format: renderer.oit_format, blend: Some(OIT_BLEND), write_mask: wgpu::ColorWrites::ALL });
        // Projection: the colour and Z unblended, the intensity as depth,
        // kept where greater (`depthFunc(GREATER)`, cleared to 0).
        let targets = if projection {
            [Some(wgpu::ColorTargetState { format: renderer.oit_format, blend: None, write_mask: wgpu::ColorWrites::ALL }), Some(wgpu::ColorTargetState { format: Z_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })]
        } else {
            [target.clone(), target]
        };
        let depth_stencil = projection.then(|| wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: Default::default(),
            bias: Default::default(),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ng volume rendering"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_volume"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs_volume"), targets: &targets, compilation_options: Default::default() }),
            // `cullFace(FRONT)`: each ray once, from the box's far side. The
            // target is mirrored, so GL's front faces wind clockwise here.
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, layout, projection }
    }
}

/// One chunk to ray-march: its layer's pipeline, the chunk's texture, the
/// controls' lookup and its uniforms.
pub struct VolumeDraw<'a> {
    /// Its layer (a projection layer's chunks are drawn together, then
    /// copied into the transparent targets).
    pub layer: usize,
    pub pipeline: &'a VolumePipeline,
    pub volume: &'a wgpu::TextureView,
    pub lookup: &'a wgpu::TextureView,
    pub uniforms: VolumeUniforms,
}

/// `fs_volume`'s uniforms: the chunk layout's model-view-projection (GL
/// clip) and its inverse, the chunk's translation and size and the clip
/// bounds (chunk layout voxels), the near and far limit fractions, the
/// brightness factor (`optimalSamples / depthSamples`), the gain
/// (`exp(volumeRenderingGain)`) and the steps.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeUniforms {
    pub mvp: [f32; 16],
    pub inv_mvp: [f32; 16],
    pub translation: [f32; 4],
    pub chunk_size: [f32; 4],
    pub lower_clip: [f32; 4],
    pub upper_clip: [f32; 4],
    pub params: [f32; 4],
    pub steps: [u32; 4],
}

/// Lines and circles of transparent layers (skeletons with object alpha
/// below 1), drawn with the transparent meshes.
#[derive(Debug, Clone, Default)]
pub struct TransparentMarks {
    pub lines: Vec<Line3>,
    pub circles: Vec<crate::annotations::Circle>,
}

/// A feathered line in 3D (`emitLine`): endpoints in GL clip coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Line3 {
    pub a: [f64; 4],
    pub b: [f64; 4],
    pub width: f64,
    pub feather: f64,
    pub color: [f32; 4],
    /// The colour at `b`, mixed along it.
    pub color_b: [f32; 4],
    /// The pick ID written (0 for none).
    pub pick: u32,
}

pub struct PerspectiveRenderer {
    quad: wgpu::RenderPipeline,
    quad_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    lines: Targeted,
    axes: Targeted,
    mesh: wgpu::RenderPipeline,
    mesh_layout: wgpu::BindGroupLayout,
    /// Annotation circles (`emitCircle`): blended, `depthFunc(LEQUAL)`.
    circles: Targeted,
    /// Coloured triangles (ellipsoids' spheres), blended, `LEQUAL`.
    triangles: Targeted,
    /// The transparent layers' pass (`perspectivePanelEmitOIT`): meshes,
    /// lines and circles into the accumulation and revealage targets, and
    /// the copy over the panel.
    oit_mesh: wgpu::RenderPipeline,
    oit_lines: wgpu::RenderPipeline,
    oit_circles: wgpu::RenderPipeline,
    oit_copy: wgpu::RenderPipeline,
    oit_copy_layout: wgpu::BindGroupLayout,
    /// RGBA32F where it blends (`EXT_float_blend`), else RGBA16F.
    oit_format: wgpu::TextureFormat,
    oit_targets: std::sync::Mutex<Option<OitTargets>>,
    pick_targets: std::sync::Mutex<Option<PickTargets>>,
    /// `maxProjectionColorCopyHelper` and the projection targets (colour,
    /// `1 − depth`, the intensity as depth).
    projection_copy: wgpu::RenderPipeline,
    projection_copy_layout: wgpu::BindGroupLayout,
    projection_targets: std::sync::Mutex<Option<ProjectionTargets>>,
}

struct ProjectionTargets {
    size: (u32, u32),
    color: wgpu::TextureView,
    z: wgpu::TextureView,
    depth: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

/// The Z and pick buffers (`OffscreenTextures.Z`, `.PICK`).
struct PickTargets {
    size: (u32, u32),
    z: wgpu::Texture,
    pick: wgpu::Texture,
    z_view: wgpu::TextureView,
    pick_view: wgpu::TextureView,
}

/// The transparent pass's two targets (`transparentConfiguration`).
struct OitTargets {
    size: (u32, u32),
    accum: wgpu::TextureView,
    alpha: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

/// The Z buffer's format (`OffscreenTextures.Z`) and the pick buffer's.
pub const Z_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
pub const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// The 3D panel's targets: the colour (`blend`), then the Z and pick
/// buffers, unblended (their alpha is 1), written by what `emits`.
fn panel_targets(blend: Option<wgpu::BlendState>, emits: bool) -> [Option<wgpu::ColorTargetState>; 3] {
    let mask = if emits { wgpu::ColorWrites::ALL } else { wgpu::ColorWrites::empty() };
    [
        Some(wgpu::ColorTargetState { format: OFFSCREEN_FORMAT, blend, write_mask: wgpu::ColorWrites::ALL }),
        Some(wgpu::ColorTargetState { format: Z_FORMAT, blend: None, write_mask: mask }),
        Some(wgpu::ColorTargetState { format: PICK_FORMAT, blend: None, write_mask: mask }),
    ]
}

/// Which of the 3D panel's targets a pipeline writes.
#[derive(Clone, Copy)]
enum TargetKind {
    /// Colour, Z and pick at once ([`panel_targets`]).
    Panel,
    /// The colour alone.
    Color,
    /// Z and pick alone, unblended (the colour slot left empty, so the
    /// shader's outputs keep their locations).
    Ids,
}

fn targets(kind: TargetKind, blend: Option<wgpu::BlendState>, emits: bool) -> Vec<Option<wgpu::ColorTargetState>> {
    match kind {
        TargetKind::Panel => panel_targets(blend, emits).to_vec(),
        TargetKind::Color => vec![Some(wgpu::ColorTargetState { format: OFFSCREEN_FORMAT, blend, write_mask: wgpu::ColorWrites::ALL })],
        TargetKind::Ids => vec![
            None,
            Some(wgpu::ColorTargetState { format: Z_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
            Some(wgpu::ColorTargetState { format: PICK_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
        ],
    }
}

/// A pipeline over the 3D panel's colour, Z and pick targets: one pass
/// where its targets may blend differently, else (WebGL2) the colour,
/// blended, in one pass and Z and pick, unblended, in a second under the
/// same depth test: the fragments at the final depth pass it again, so the
/// targets end as one pass leaves them.
enum Targeted {
    Single(wgpu::RenderPipeline),
    Split { color: wgpu::RenderPipeline, ids: Option<wgpu::RenderPipeline> },
}

/// `build` once for one pass, or for the split where the GPU needs it.
fn targeted(emits: bool, build: impl Fn(TargetKind, &str) -> wgpu::RenderPipeline) -> Targeted {
    if crate::capabilities::independent_blend() {
        Targeted::Single(build(TargetKind::Panel, ""))
    } else {
        Targeted::Split { color: build(TargetKind::Color, " (colour)"), ids: emits.then(|| build(TargetKind::Ids, " (ids)")) }
    }
}

/// `maxProjectionColorCopyHelper`: a projection's colour into the
/// transparent targets, weighted at its depth (`1 − bufferDepth`), its
/// alpha the revealage factor.
const PROJECTION_COPY_SHADER: &str = r#"
@group(0) @binding(0) var projection_color : texture_2d<f32>;
@group(0) @binding(1) var projection_z : texture_2d<f32>;
@vertex
fn vs_copy(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs_copy(@builtin(position) p : vec4<f32>) -> OitOut {
    let xy = vec2<i32>(p.xy);
    let color = textureLoad(projection_color, xy, 0);
    let buffer_depth = textureLoad(projection_z, xy, 0).r;
    let weight = compute_oit_weight(color.a, 1.0 - buffer_depth);
    let accum = color * weight;
    return OitOut(vec4<f32>(accum.rgb, color.a), vec4<f32>(accum.a, 0.0, 0.0, 0.0));
}
"#;

/// The depth buffer as earlier passes left it.
fn keep_depth() -> wgpu::Operations<f32> {
    wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }
}

/// `blendFuncSeparate(ONE, ONE, ZERO, ONE_MINUS_SRC_ALPHA)`.
const OIT_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
};

/// A vertex of a coloured triangle (every three a triangle), in GL clip
/// coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColoredVertex {
    pub clip: [f64; 4],
    pub color: [f32; 4],
}

impl PerspectiveRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ng perspective"),
            source: wgpu::ShaderSource::Wgsl(format!("{OIT_SHADER}{EMIT_SHADER}{SHADER}").into()),
        });
        let quad_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng perspective slice"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ng perspective slice"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let make = |label: &str,
                    layouts: &[Option<&wgpu::BindGroupLayout>],
                    vs: &str,
                    fs: &str,
                    buffer: wgpu::VertexBufferLayout,
                    topology: wgpu::PrimitiveTopology,
                    blend: Option<wgpu::BlendState>,
                    compare: wgpu::CompareFunction,
                    emits: bool,
                    kind: TargetKind| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    buffers: &[Some(buffer)],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    targets: &targets(kind, blend, emits),
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState { topology, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(compare),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let quad = make(
            "ng perspective slice",
            &[Some(&quad_layout)],
            "vs_quad",
            "fs_quad",
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<QuadVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x2],
            },
            wgpu::PrimitiveTopology::TriangleList,
            None,
            wgpu::CompareFunction::Less,
            true,
            TargetKind::Panel,
        );
        // Annotations: blended, `depthFunc(LEQUAL)`.
        let lines = targeted(true, |kind, suffix| {
            make(
                &format!("ng perspective lines{suffix}"),
                &[],
                "vs_line",
                "fs_line",
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32, 3 => Float32x4, 4 => Uint32],
                },
                wgpu::PrimitiveTopology::TriangleList,
                Some(LAYER_BLEND),
                wgpu::CompareFunction::LessEqual,
                true,
                kind,
            )
        });
        // Axis lines: unblended (`blend = false`), `depthFunc(LESS)`; not
        // in the Z and pick buffers.
        let axes = targeted(false, |kind, suffix| {
            make(
                &format!("ng perspective axes{suffix}"),
                &[],
                "vs_axis",
                "fs_axis",
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<AxisVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4],
                },
                wgpu::PrimitiveTopology::LineList,
                None,
                wgpu::CompareFunction::Less,
                false,
                kind,
            )
        });
        let mesh_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng perspective mesh"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(MESH_UNIFORM_BYTES) },
                count: None,
            }],
        });
        // Opaque meshes (`!isTransparent`): no blending, `depthFunc(LESS)`.
        let mesh = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ng perspective mesh"), bind_group_layouts: &[Some(&mesh_layout)], immediate_size: 0 });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ng perspective mesh"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_mesh"),
                    buffers: &[
                        Some(wgpu::VertexBufferLayout { array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3] }),
                        Some(wgpu::VertexBufferLayout { array_stride: 4, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![1 => Snorm8x2] }),
                    ],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_mesh"),
                    targets: &panel_targets(None, true),
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let circles = {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ng perspective circles"), source: wgpu::ShaderSource::Wgsl(format!("{OIT_SHADER}{EMIT_SHADER}{}", crate::annotations::SHADER).into()) });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ng perspective circles"), bind_group_layouts: &[], immediate_size: 0 });
            targeted(true, |kind, suffix| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("ng perspective circles{suffix}")),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_circle"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<crate::annotations::CircleVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32, 5 => Uint32],
                    })],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_circle_emit"),
                    targets: &targets(kind, Some(LAYER_BLEND), true),
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            }))
        };
        let triangles = targeted(true, |kind, suffix| make(
            &format!("ng perspective triangles{suffix}"),
            &[],
            "vs_axis",
            "fs_axis",
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<AxisVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4],
            },
            wgpu::PrimitiveTopology::TriangleList,
            Some(LAYER_BLEND),
            wgpu::CompareFunction::LessEqual,
            true,
            kind,
        ));
        let oit_format = if device.features().contains(wgpu::Features::FLOAT32_BLENDABLE) { wgpu::TextureFormat::Rgba32Float } else { wgpu::TextureFormat::Rgba16Float };
        let oit_targets = [Some(wgpu::ColorTargetState { format: oit_format, blend: Some(OIT_BLEND), write_mask: wgpu::ColorWrites::ALL }), Some(wgpu::ColorTargetState { format: oit_format, blend: Some(OIT_BLEND), write_mask: wgpu::ColorWrites::ALL })];
        // `depthMask(false)`, `depthFunc(LESS)` against the opaque layers.
        let oit_depth = Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: Some(false), depth_compare: Some(wgpu::CompareFunction::Less), stencil: Default::default(), bias: Default::default() });
        let oit_pipeline = |label: &str, module: &wgpu::ShaderModule, layouts: &[Option<&wgpu::BindGroupLayout>], vs: &str, fs: &str, buffers: &[Option<wgpu::VertexBufferLayout>]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: layouts, immediate_size: 0 });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module, entry_point: Some(vs), buffers, compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module, entry_point: Some(fs), targets: &oit_targets, compilation_options: Default::default() }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: oit_depth.clone(),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let oit_mesh = oit_pipeline(
            "ng perspective mesh (transparent)",
            &shader,
            &[Some(&mesh_layout)],
            "vs_mesh",
            "fs_mesh_oit",
            &[
                Some(wgpu::VertexBufferLayout { array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3] }),
                Some(wgpu::VertexBufferLayout { array_stride: 4, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![1 => Snorm8x2] }),
            ],
        );
        let oit_lines = oit_pipeline(
            "ng perspective lines (transparent)",
            &shader,
            &[],
            "vs_line",
            "fs_line_oit",
            &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32, 3 => Float32x4, 4 => Uint32],
            })],
        );
        let circle_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ng perspective circles (transparent)"), source: wgpu::ShaderSource::Wgsl(format!("{OIT_SHADER}{EMIT_SHADER}{}", crate::annotations::SHADER).into()) });
        let oit_circles = oit_pipeline(
            "ng perspective circles (transparent)",
            &circle_module,
            &[],
            "vs_circle",
            "fs_circle_oit",
            &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<crate::annotations::CircleVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32, 5 => Uint32],
            })],
        );
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let oit_copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ng perspective transparency copy"), entries: &[texture_entry(0), texture_entry(1)] });
        let oit_copy = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ng perspective transparency copy"), bind_group_layouts: &[Some(&oit_copy_layout)], immediate_size: 0 });
            // `blendFunc(ONE_MINUS_SRC_ALPHA, SRC_ALPHA)`.
            let component = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::OneMinusSrcAlpha, dst_factor: wgpu::BlendFactor::SrcAlpha, operation: wgpu::BlendOperation::Add };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ng perspective transparency copy"),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_oit_copy"), buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_oit_copy"),
                    targets: &[Some(wgpu::ColorTargetState { format: OFFSCREEN_FORMAT, blend: Some(wgpu::BlendState { color: component, alpha: component }), write_mask: wgpu::ColorWrites::ALL })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let projection_copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ng projection copy"), entries: &[texture_entry(0), texture_entry(1)] });
        let projection_copy = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ng projection copy"), bind_group_layouts: &[Some(&projection_copy_layout)], immediate_size: 0 });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ng projection copy"), source: wgpu::ShaderSource::Wgsl(format!("{OIT_SHADER}{PROJECTION_COPY_SHADER}").into()) });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ng projection copy"),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_copy"), buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs_copy"), targets: &oit_targets, compilation_options: Default::default() }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            quad,
            quad_layout,
            sampler,
            lines,
            axes,
            mesh,
            mesh_layout,
            circles,
            triangles,
            oit_mesh,
            oit_lines,
            oit_circles,
            oit_copy,
            oit_copy_layout,
            oit_format,
            oit_targets: std::sync::Mutex::new(None),
            pick_targets: std::sync::Mutex::new(None),
            projection_copy,
            projection_copy_layout,
            projection_targets: std::sync::Mutex::new(None),
        }
    }

    /// The projection targets, `size` pixels (kept between frames).
    fn projection_targets(&self, device: &wgpu::Device, size: (u32, u32)) -> std::sync::MutexGuard<'_, Option<ProjectionTargets>> {
        let mut guard = self.projection_targets.lock().unwrap();
        if !matches!(&*guard, Some(t) if t.size == size) {
            let make = |label, format| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            };
            let (color, z, depth) = (make("ng projection colour", self.oit_format), make("ng projection z", Z_FORMAT), make("ng projection depth", DEPTH_FORMAT));
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ng projection copy"),
                layout: &self.projection_copy_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&color) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&z) }],
            });
            *guard = Some(ProjectionTargets { size, color, z, depth, bind_group });
        }
        guard
    }

    /// The Z and pick buffers, `size` pixels (kept between frames).
    fn pick_targets(&self, device: &wgpu::Device, size: (u32, u32)) -> std::sync::MutexGuard<'_, Option<PickTargets>> {
        let mut guard = self.pick_targets.lock().unwrap();
        if !matches!(&*guard, Some(t) if t.size == size) {
            let make = |label, format| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            };
            let (z, pick) = (make("ng perspective z", Z_FORMAT), make("ng perspective pick", PICK_FORMAT));
            *guard = Some(PickTargets { size, z_view: z.create_view(&Default::default()), pick_view: pick.create_view(&Default::default()), z, pick });
        }
        guard
    }

    /// `issuePickRequest`: copies the Z and pick buffers' window of
    /// `radius` around `(x, y)` (GL window pixels, from the bottom) of the
    /// frame last drawn; [`crate::pick::PickReadback::map`] once submitted.
    pub fn read_pick(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, x: i64, y: i64, radius: u32) -> Option<crate::pick::PickReadback> {
        let guard = self.pick_targets.lock().unwrap();
        let targets = guard.as_ref()?;
        Some(crate::pick::PickReadback::copy(device, encoder, &targets.z, &targets.pick, targets.size, x, y, radius))
    }

    /// The transparent pass's targets, `size` pixels (kept between frames).
    fn oit_targets(&self, device: &wgpu::Device, size: (u32, u32)) -> std::sync::MutexGuard<'_, Option<OitTargets>> {
        let mut guard = self.oit_targets.lock().unwrap();
        if !matches!(&*guard, Some(t) if t.size == size) {
            let make = |label| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: self.oit_format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            };
            let (accum, alpha) = (make("ng perspective accumulation"), make("ng perspective revealage"));
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ng perspective transparency copy"),
                layout: &self.oit_copy_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&accum) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&alpha) }],
            });
            *guard = Some(OitTargets { size, accum, alpha, bind_group });
        }
        guard
    }

    /// Draws the 3D panel into `color` (upside down) with `depth`, both
    /// `size` pixels: cleared to `background` at alpha 0, then the planes,
    /// the lines, the axes (`origin` and the x, y, z ends, GL clip).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        size: (u32, u32),
        background: [f32; 3],
        cross_section_background: Option<[f32; 3]>,
        meshes: &[MeshDraw],
        planes: &[SlicePlane],
        lines: &[Line3],
        circles: &[crate::annotations::Circle],
        triangles: &[ColoredVertex],
        axes: Option<([f64; 4], [[f64; 4]; 3])>,
        transparent: &TransparentMarks,
        volumes: &[VolumeDraw],
    ) {
        let (w, h) = (size.0 as f64, size.1 as f64);
        // The meshes' uniforms, one 256-byte slot each.
        let mesh_bind = (!meshes.is_empty()).then(|| {
            let mut bytes = vec![0u8; meshes.len() * MESH_UNIFORM_STRIDE as usize];
            for (k, m) in meshes.iter().enumerate() {
                let n = m.normal_matrix;
                let mut u = [0f32; 44];
                u[..16].copy_from_slice(&m.mvp);
                for c in 0..3 {
                    u[16 + 4 * c..16 + 4 * c + 3].copy_from_slice(&n[3 * c..3 * c + 3]);
                }
                u[28..32].copy_from_slice(&m.light);
                u[32..36].copy_from_slice(&m.color);
                u[36..39].copy_from_slice(&m.fragment_origin);
                u[39] = f32::from_bits(m.pick);
                u[40..43].copy_from_slice(&m.fragment_shape);
                u[43] = m.silhouette;
                let at = k * MESH_UNIFORM_STRIDE as usize;
                bytes[at..at + MESH_UNIFORM_BYTES as usize].copy_from_slice(bytemuck::cast_slice(&u));
            }
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ng perspective meshes"), contents: &bytes, usage: wgpu::BufferUsages::UNIFORM });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ng perspective meshes"),
                layout: &self.mesh_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &buffer, offset: 0, size: wgpu::BufferSize::new(MESH_UNIFORM_BYTES) }) }],
            })
        });
        // Per-plane vertex buffers and bind groups, made before the pass.
        let plane_draws: Vec<(wgpu::Buffer, wgpu::BindGroup)> = planes
            .iter()
            .map(|p| {
                // `TRIANGLE_FAN` over the square's corners; texture v runs up in
                // GL and down in ours.
                let uv = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
                let v = |i: usize| QuadVertex { clip: to_target(p.corners[i]), uv: uv[i] };
                let vertices = [v(0), v(1), v(2), v(0), v(2), v(3)];
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ng perspective slice"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let background_rgba = match cross_section_background {
                    Some(c) => [c[0], c[1], c[2], 1.0],
                    None => [0.0, 0.0, 0.0, -1.0],
                };
                let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ng perspective slice"),
                    contents: bytemuck::cast_slice(&[
                        background_rgba[0],
                        background_rgba[1],
                        background_rgba[2],
                        background_rgba[3],
                        p.factor,
                        p.factor,
                        p.factor,
                        1.0,
                    ]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ng perspective slice"),
                    layout: &self.quad_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(p.texture) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        wgpu::BindGroupEntry { binding: 2, resource: uniforms.as_entire_binding() },
                    ],
                });
                (buffer, bind_group)
            })
            .collect();
        let line_vertices: Vec<LineVertex> = lines.iter().filter_map(|l| line_quad(l, w, h)).flatten().collect();
        let line_buffer = (!line_vertices.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ng perspective lines"),
                contents: bytemuck::cast_slice(&line_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let triangle_vertices: Vec<AxisVertex> = triangles.iter().map(|v| AxisVertex { clip: to_target(v.clip), color: v.color }).collect();
        let triangle_buffer = (!triangle_vertices.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ng perspective triangles"), contents: bytemuck::cast_slice(&triangle_vertices), usage: wgpu::BufferUsages::VERTEX })
        });
        let circle_vertices: Vec<crate::annotations::CircleVertex> = circles.iter().flat_map(|c| crate::annotations::circle_quad(c, w, h, 0.0)).collect();
        let circle_buffer = (!circle_vertices.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ng perspective circles"), contents: bytemuck::cast_slice(&circle_vertices), usage: wgpu::BufferUsages::VERTEX })
        });
        let axis_buffer = axes.map(|(origin, ends)| {
            const ALPHA: f32 = 0.5;
            let colors = [[1.0, 0.0, 0.0, ALPHA], [0.0, 1.0, 0.0, ALPHA], [0.0, 0.0, 1.0, ALPHA]];
            let vertices: Vec<AxisVertex> = (0..3)
                .flat_map(|i| {
                    [
                        AxisVertex { clip: to_target(origin), color: colors[i] },
                        AxisVertex { clip: to_target(ends[i]), color: colors[i] },
                    ]
                })
                .collect();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ng perspective axes"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });

        let pick_targets = self.pick_targets(device, size);
        let pick_targets = pick_targets.as_ref().unwrap();
        // The Z and pick buffers cleared to 0 (`kZeroVec4`).
        let zero = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ng perspective"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: background[0] as f64,
                            g: background[1] as f64,
                            b: background[2] as f64,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment { view: &pick_targets.z_view, depth_slice: None, resolve_target: None, ops: zero }),
                Some(wgpu::RenderPassColorAttachment { view: &pick_targets.pick_view, depth_slice: None, resolve_target: None, ops: zero }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
        // Opaque layers first (`drawWithPicking`), then the slices.
        if let Some(bind_group) = &mesh_bind {
            pass.set_pipeline(&self.mesh);
            for (k, m) in meshes.iter().enumerate().filter(|(_, m)| !m.transparent) {
                pass.set_bind_group(0, bind_group, &[(k as u64 * MESH_UNIFORM_STRIDE) as u32]);
                pass.set_vertex_buffer(0, m.mesh.positions.slice(..));
                pass.set_vertex_buffer(1, m.mesh.normals.slice(..));
                pass.set_index_buffer(m.mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(m.indices.clone(), 0, 0..1);
            }
        }
        pass.set_pipeline(&self.quad);
        for (buffer, bind_group) in &plane_draws {
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
        // The annotations and axes, in order: in this pass where the targets
        // blend apart, else in passes of their own after it.
        let groups: [(&Targeted, Option<&wgpu::Buffer>, u32); 4] = [
            (&self.lines, line_buffer.as_ref(), line_vertices.len() as u32),
            (&self.triangles, triangle_buffer.as_ref(), triangle_vertices.len() as u32),
            (&self.circles, circle_buffer.as_ref(), circle_vertices.len() as u32),
            (&self.axes, axis_buffer.as_ref(), 6),
        ];
        for &(pipeline, buffer, count) in &groups {
            if let (Targeted::Single(pipeline), Some(buffer)) = (pipeline, buffer) {
                pass.set_pipeline(pipeline);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..count, 0..1);
            }
        }
        drop(pass);
        let keep = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
        let attach = |view| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: keep });
        let depth_attachment = || Some(wgpu::RenderPassDepthStencilAttachment { view: depth, depth_ops: Some(keep_depth()), stencil_ops: None });
        for (pipeline, buffer, count) in groups {
            let (Targeted::Split { color: color_pipeline, ids }, Some(buffer)) = (pipeline, buffer) else { continue };
            let mut passes = vec![("ng perspective (colour)", color_pipeline, vec![attach(color)])];
            if let Some(ids) = ids {
                passes.push(("ng perspective (ids)", ids, vec![None, attach(&pick_targets.z_view), attach(&pick_targets.pick_view)]));
            }
            for (label, pipeline, color_attachments) in passes {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(label),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: depth_attachment(),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
                pass.set_pipeline(pipeline);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..count, 0..1);
            }
        }

        // Transparent layers (`hasTransparent`): accumulated, then copied
        // over the panel.
        let transparent_meshes = meshes.iter().any(|m| m.transparent);
        if !transparent_meshes && transparent.lines.is_empty() && transparent.circles.is_empty() && volumes.is_empty() {
            return;
        }
        let line_vertices: Vec<LineVertex> = transparent.lines.iter().filter_map(|l| line_quad(l, w, h)).flatten().collect();
        let circle_vertices: Vec<crate::annotations::CircleVertex> = transparent.circles.iter().flat_map(|c| crate::annotations::circle_quad(c, w, h, 0.0)).collect();
        let buffer = |label, contents: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::VERTEX });
        let line_buffer = (!line_vertices.is_empty()).then(|| buffer("ng perspective lines (transparent)", bytemuck::cast_slice(&line_vertices)));
        let circle_buffer = (!circle_vertices.is_empty()).then(|| buffer("ng perspective circles (transparent)", bytemuck::cast_slice(&circle_vertices)));
        let targets = self.oit_targets(device, size);
        let targets = targets.as_ref().unwrap();
        // `clearColor(0, 0, 0, 1)`.
        let clear = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }), store: wgpu::StoreOp::Store };
        let attachment = |view| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: clear });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ng perspective transparent"),
            color_attachments: &[attachment(&targets.accum), attachment(&targets.alpha)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
        if let Some(bind_group) = mesh_bind.as_ref().filter(|_| transparent_meshes) {
            pass.set_pipeline(&self.oit_mesh);
            for (k, m) in meshes.iter().enumerate().filter(|(_, m)| m.transparent) {
                pass.set_bind_group(0, bind_group, &[(k as u64 * MESH_UNIFORM_STRIDE) as u32]);
                pass.set_vertex_buffer(0, m.mesh.positions.slice(..));
                pass.set_vertex_buffer(1, m.mesh.normals.slice(..));
                pass.set_index_buffer(m.mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(m.indices.clone(), 0, 0..1);
            }
        }
        if let Some(b) = &line_buffer {
            pass.set_pipeline(&self.oit_lines);
            pass.set_vertex_buffer(0, b.slice(..));
            pass.draw(0..line_vertices.len() as u32, 0..1);
        }
        if let Some(b) = &circle_buffer {
            pass.set_pipeline(&self.oit_circles);
            pass.set_vertex_buffer(0, b.slice(..));
            pass.draw(0..circle_vertices.len() as u32, 0..1);
        }
        drop(pass);
        // Volume rendering (`isVolumeRendering`): ray-marched into the same
        // targets, without the depth test (rays stop at the Z buffer).
        if !volumes.is_empty() {
            let bind_groups: Vec<wgpu::BindGroup> = volumes
                .iter()
                .map(|v| {
                    let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ng volume rendering"), contents: bytemuck::bytes_of(&v.uniforms), usage: wgpu::BufferUsages::UNIFORM });
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ng volume rendering"),
                        layout: &v.pipeline.layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(v.volume) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(v.lookup) },
                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&pick_targets.z_view) },
                        ],
                    })
                })
                .collect();
            let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
            let attachment = |view| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: load });
            if volumes.iter().any(|v| !v.pipeline.projection) {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ng volume rendering"),
                    color_attachments: &[attachment(&targets.accum), attachment(&targets.alpha)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
                for (v, group) in volumes.iter().zip(&bind_groups).filter(|(v, _)| !v.pipeline.projection) {
                    pass.set_pipeline(&v.pipeline.pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.draw(0..36, 0..1);
                }
            }
            // Each projection layer: its chunks into the cleared projection
            // targets, then copied over.
            let mut layers: Vec<usize> = volumes.iter().filter(|v| v.pipeline.projection).map(|v| v.layer).collect();
            layers.dedup();
            if !layers.is_empty() {
                let projection = self.projection_targets(device, size);
                let projection = projection.as_ref().unwrap();
                let zero = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store };
                for layer in layers {
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("ng projection"),
                            color_attachments: &[
                                Some(wgpu::RenderPassColorAttachment { view: &projection.color, depth_slice: None, resolve_target: None, ops: zero }),
                                Some(wgpu::RenderPassColorAttachment { view: &projection.z, depth_slice: None, resolve_target: None, ops: zero }),
                            ],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &projection.depth,
                                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
                        for (v, group) in volumes.iter().zip(&bind_groups).filter(|(v, _)| v.pipeline.projection && v.layer == layer) {
                            pass.set_pipeline(&v.pipeline.pipeline);
                            pass.set_bind_group(0, group, &[]);
                            pass.draw(0..36, 0..1);
                        }
                    }
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("ng projection copy"),
                        color_attachments: &[attachment(&targets.accum), attachment(&targets.alpha)],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
                    pass.set_pipeline(&self.projection_copy);
                    pass.set_bind_group(0, &projection.bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
        }
        let mut pass = threers::renderer::fullscreen::begin(encoder, color, threers::renderer::fullscreen::Load::Keep);
        pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
        pass.set_pipeline(&self.oit_copy);
        pass.set_bind_group(0, &targets.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// `clipLineToDepthRange` on GL clip coordinates.
fn clip_line_to_depth_range(mut a: [f64; 4], mut b: [f64; 4]) -> Option<([f64; 4], [f64; 4])> {
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
    let (ta, tb) = (a, b);
    let mix = |t: f64| std::array::from_fn(|i| ta[i] + (tb[i] - ta[i]) * t);
    a = mix(tmin);
    b = mix(tmax);
    Some((a, b))
}

/// `emitLine` in 3D: the quad, in device coordinates (w = 1), mirrored in y
/// and with depth `(z + 1) / 2`.
fn line_quad(l: &Line3, width: f64, height: f64) -> Option<[LineVertex; 6]> {
    let total = if l.width == 0.0 { 0.0 } else { l.width + 2.0 * l.feather };
    if total == 0.0 {
        return None;
    }
    let (a, b) = clip_line_to_depth_range(l.a, l.b)?;
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
        LineVertex {
            position: [px as f32, -py as f32, ((p[2] + 1.0) / 2.0) as f32],
            line_coord: y as f32,
            feather_fraction,
            color: color_at(x),
            pick: l.pick,
        }
    };
    let (v00, v10, v01, v11) = (vertex(0.0, -1.0), vertex(1.0, -1.0), vertex(0.0, 1.0), vertex(1.0, 1.0));
    Some([v00, v10, v11, v00, v11, v01])
}
