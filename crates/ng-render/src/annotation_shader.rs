//! Annotation shaders (`annotation/type_handler.ts`, the render helpers of
//! `annotation/{point,line,bounding_box,ellipsoid,polyline}.ts`): the
//! user's `main` with neuroglancer's setters (`setColor`,
//! `setPointMarkerSize`, `setLineColor(start, end)`, …), `defaultColor()`,
//! `prop_<id>()`, the `#uicontrol`s and `discard`.
//!
//! neuroglancer runs the shader in the vertex stage of each mark it draws,
//! every setter but those of the part being drawn a no-op, the two-ended
//! ones mixing their values by the vertex's end. Here it runs once per
//! annotation in a compute pass, every setter storing what it is given (both
//! ends of the two-ended ones), into an [`AnnotationStyle`] the marks are
//! then built from: the same values, as the shader sees nothing of the
//! vertex. `PROJECTION_VIEW` stays a constant: cross-sections and the 3D
//! view each have their own pipeline.

use std::sync::Arc;

use threers::renderer::readback::AsyncReadback;

use crate::user_shader::{CompileError, Control, ControlValue, ParsedShader, SelectValue};

/// A property's type in the shader (`annotationPropertyTypeRenderHandlers`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyShaderType {
    /// `rgb`: a `vec3` of its bytes normalized.
    Rgb,
    /// `rgba`: a `vec4`.
    Rgba,
    Float,
    Uint,
    Int,
}

impl PropertyShaderType {
    fn glsl(self) -> &'static str {
        match self {
            Self::Rgb => "vec3",
            Self::Rgba => "vec4",
            Self::Float => "float",
            Self::Uint => "uint",
            Self::Int => "int",
        }
    }

    /// The expression of `prop_<id>()` from its 32 bits.
    fn read_bits(self, bits: &str) -> String {
        match self {
            Self::Rgb => format!("vec3(float({bits} & 255u), float(({bits} >> 8u) & 255u), float(({bits} >> 16u) & 255u)) / 255.0"),
            Self::Rgba => format!("vec4(float({bits} & 255u), float(({bits} >> 8u) & 255u), float(({bits} >> 16u) & 255u), float({bits} >> 24u)) / 255.0"),
            Self::Float => format!("uintBitsToFloat({bits})"),
            Self::Uint => bits.to_owned(),
            Self::Int => format!("int({bits})"),
        }
    }

    /// A property value's 32 bits (colours packed little-endian; integers
    /// as they are; sign-extended for signed types).
    pub fn bits(self, value: f64) -> u32 {
        match self {
            Self::Float => (value as f32).to_bits(),
            Self::Int => value as i64 as i32 as u32,
            _ => value as u32,
        }
    }
}

/// How many `vec4`s one annotation's style takes.
pub const STYLE_VEC4S: usize = 21;

/// What the shader set for one annotation: each part's colours and sizes
/// (two-ended setters' both ends), and whether it discarded the
/// annotation. Unset ones keep the render helpers' starting values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnotationStyle {
    pub point_color: [f32; 4],
    pub point_border_color: [f32; 4],
    pub point_size: f32,
    pub point_border_width: f32,
    pub line_color: [[f32; 4]; 2],
    pub line_width: f32,
    pub marker_color: [[f32; 4]; 2],
    pub marker_border_color: [[f32; 4]; 2],
    pub marker_size: [f32; 2],
    pub marker_border_width: [f32; 2],
    pub poly_line_color: [[f32; 4]; 2],
    pub poly_line_width: f32,
    pub poly_marker_color: [[f32; 4]; 2],
    pub poly_marker_border_color: [[f32; 4]; 2],
    pub poly_marker_size: [f32; 2],
    pub poly_marker_border_width: [f32; 2],
    pub box_border_color: [f32; 4],
    pub box_border_width: f32,
    pub box_fill_color: [f32; 4],
    pub ellipsoid_fill_color: [f32; 4],
    pub discard: bool,
}

const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const NONE: [f32; 4] = [0.0; 4];

impl Default for AnnotationStyle {
    /// The render helpers' values before the user's `main` runs: no colour,
    /// black borders, 5 px markers with 1 px borders, 1 px lines.
    fn default() -> Self {
        Self {
            point_color: NONE,
            point_border_color: BLACK,
            point_size: 5.0,
            point_border_width: 1.0,
            line_color: [NONE; 2],
            line_width: 1.0,
            marker_color: [NONE; 2],
            marker_border_color: [BLACK; 2],
            marker_size: [5.0; 2],
            marker_border_width: [1.0; 2],
            poly_line_color: [NONE; 2],
            poly_line_width: 1.0,
            poly_marker_color: [NONE; 2],
            poly_marker_border_color: [BLACK; 2],
            poly_marker_size: [5.0; 2],
            poly_marker_border_width: [1.0; 2],
            box_border_color: NONE,
            box_border_width: 1.0,
            box_fill_color: NONE,
            ellipsoid_fill_color: NONE,
            discard: false,
        }
    }
}

impl AnnotationStyle {
    /// The default shader's (`setColor(defaultColor())`).
    pub fn default_shader(color: [f32; 3], projection: bool) -> Self {
        let c = [color[0], color[1], color[2], 1.0];
        Self {
            point_color: c,
            line_color: [c; 2],
            marker_color: [c; 2],
            poly_line_color: [c; 2],
            poly_marker_color: [c; 2],
            box_border_color: c,
            ellipsoid_fill_color: [color[0], color[1], color[2], if projection { 1.0 } else { 0.5 }],
            ..Self::default()
        }
    }

    /// From its `vec4`s as the compute pass writes them.
    pub fn from_vec4s(v: &[[f32; 4]]) -> Self {
        Self {
            point_color: v[0],
            point_border_color: v[1],
            point_size: v[2][0],
            point_border_width: v[2][1],
            line_width: v[2][2],
            poly_line_width: v[2][3],
            line_color: [v[3], v[4]],
            marker_color: [v[5], v[6]],
            marker_border_color: [v[7], v[8]],
            marker_size: [v[9][0], v[9][1]],
            marker_border_width: [v[9][2], v[9][3]],
            poly_line_color: [v[10], v[11]],
            poly_marker_color: [v[12], v[13]],
            poly_marker_border_color: [v[14], v[15]],
            poly_marker_size: [v[16][0], v[16][1]],
            poly_marker_border_width: [v[16][2], v[16][3]],
            box_border_color: v[17],
            box_fill_color: v[18],
            box_border_width: v[19][0],
            discard: v[19][1] != 0.0,
            ellipsoid_fill_color: v[20],
        }
    }
}

/// The shader's API (`AnnotationRenderHelper.getDependentShader`), every
/// setter storing into the style.
const SETTERS: &str = r#"
bool ng_discardValue;
#define discard ng_discard()
void ng_discard() { ng_discardValue = true; }
vec4 ng_s[21];
void setPointMarkerColor(vec4 color) { ng_s[0] = color; }
void setPointMarkerBorderColor(vec4 color) { ng_s[1] = color; }
void setPointMarkerSize(float size) { ng_s[2].x = size; }
void setPointMarkerBorderWidth(float size) { ng_s[2].y = size; }
void setSingleLineWidth(float width) { ng_s[2].z = width; }
void setPolyLineWidth(float width) { ng_s[2].w = width; }
void setSingleLineColor(vec4 startColor, vec4 endColor) { ng_s[3] = startColor; ng_s[4] = endColor; }
void setLineEndpointMarkerColor(vec4 startColor, vec4 endColor) { ng_s[5] = startColor; ng_s[6] = endColor; }
void setLineEndpointMarkerBorderColor(vec4 startColor, vec4 endColor) { ng_s[7] = startColor; ng_s[8] = endColor; }
void setLineEndpointMarkerSize(float startSize, float endSize) { ng_s[9].x = startSize; ng_s[9].y = endSize; }
void setLineEndpointMarkerBorderWidth(float startSize, float endSize) { ng_s[9].z = startSize; ng_s[9].w = endSize; }
void setPolyLineColor(vec4 startColor, vec4 endColor) { ng_s[10] = startColor; ng_s[11] = endColor; }
void setPolyEndpointMarkerColor(vec4 startColor, vec4 endColor) { ng_s[12] = startColor; ng_s[13] = endColor; }
void setPolyEndpointMarkerBorderColor(vec4 startColor, vec4 endColor) { ng_s[14] = startColor; ng_s[15] = endColor; }
void setPolyEndpointMarkerSize(float startSize, float endSize) { ng_s[16].x = startSize; ng_s[16].y = endSize; }
void setPolyEndpointMarkerBorderWidth(float startSize, float endSize) { ng_s[16].z = startSize; ng_s[16].w = endSize; }
void setBoundingBoxBorderColor(vec4 color) { ng_s[17] = color; }
void setBoundingBoxFillColor(vec4 color) { ng_s[18] = color; }
void setBoundingBoxBorderWidth(float size) { ng_s[19].x = size; }
void setEllipsoidFillColor(vec4 color) { ng_s[20] = color; }

void setLineWidth(float width) {
  setSingleLineWidth(width);
  setPolyLineWidth(width);
}
void setPointMarkerColor(vec3 color) { setPointMarkerColor(vec4(color, 1.0)); }
void setPointMarkerBorderColor(vec3 color) { setPointMarkerBorderColor(vec4(color, 1.0)); }
void setEndpointMarkerColor(vec4 startColor, vec4 endColor) {
  setLineEndpointMarkerColor(startColor, endColor);
  setPolyEndpointMarkerColor(startColor, endColor);
}
void setEndpointMarkerColor(vec3 startColor, vec3 endColor) {
  setEndpointMarkerColor(vec4(startColor, 1.0), vec4(endColor, 1.0));
}
void setEndpointMarkerColor(vec3 color) { setEndpointMarkerColor(color, color); }
void setEndpointMarkerColor(vec4 color) { setEndpointMarkerColor(color, color); }
void setEndpointMarkerBorderColor(vec4 startColor, vec4 endColor) {
  setLineEndpointMarkerBorderColor(startColor, endColor);
  setPolyEndpointMarkerBorderColor(startColor, endColor);
}
void setEndpointMarkerBorderColor(vec3 startColor, vec3 endColor) {
  setEndpointMarkerBorderColor(vec4(startColor, 1.0), vec4(endColor, 1.0));
}
void setEndpointMarkerBorderColor(vec3 color) { setEndpointMarkerBorderColor(color, color); }
void setEndpointMarkerBorderColor(vec4 color) { setEndpointMarkerBorderColor(color, color); }
void setEndpointMarkerSize(float startSize, float endSize) {
  setLineEndpointMarkerSize(startSize, endSize);
  setPolyEndpointMarkerSize(startSize, endSize);
}
void setEndpointMarkerSize(float size) { setEndpointMarkerSize(size, size); }
void setEndpointMarkerBorderWidth(float startSize, float endSize) {
  setLineEndpointMarkerBorderWidth(startSize, endSize);
  setPolyEndpointMarkerBorderWidth(startSize, endSize);
}
void setEndpointMarkerBorderWidth(float size) { setEndpointMarkerBorderWidth(size, size); }
void setLineColor(vec4 startColor, vec4 endColor) {
  setSingleLineColor(startColor, endColor);
  setPolyLineColor(startColor, endColor);
}
void setLineColor(vec4 color) { setLineColor(color, color); }
void setLineColor(vec3 color) { setLineColor(vec4(color, 1.0)); }
void setLineColor(vec3 startColor, vec3 endColor) { setLineColor(vec4(startColor, 1.0), vec4(endColor, 1.0)); }
void setPolyEndpointMarkerColor(vec3 startColor, vec3 endColor) {
  setPolyEndpointMarkerColor(vec4(startColor, 1.0), vec4(endColor, 1.0));
}
void setPolyEndpointMarkerColor(vec3 color) { setPolyEndpointMarkerColor(color, color); }
void setPolyEndpointMarkerColor(vec4 color) { setPolyEndpointMarkerColor(color, color); }
void setPolyEndpointMarkerBorderColor(vec3 startColor, vec3 endColor) {
  setPolyEndpointMarkerBorderColor(vec4(startColor, 1.0), vec4(endColor, 1.0));
}
void setPolyEndpointMarkerBorderColor(vec3 color) { setPolyEndpointMarkerBorderColor(color, color); }
void setPolyEndpointMarkerBorderColor(vec4 color) { setPolyEndpointMarkerBorderColor(color, color); }
void setPolyEndpointMarkerSize(float size) { setPolyEndpointMarkerSize(size, size); }
void setPolyEndpointMarkerBorderWidth(float size) { setPolyEndpointMarkerBorderWidth(size, size); }
void setPolyLineColor(vec4 color) { setPolyLineColor(color, color); }
void setPolyLineColor(vec3 color) { setPolyLineColor(vec4(color, 1.0)); }
void setPolyLineColor(vec3 startColor, vec3 endColor) { setPolyLineColor(vec4(startColor, 1.0), vec4(endColor, 1.0)); }

void setColor(vec4 color) {
  setPointMarkerColor(color);
  setLineColor(color);
  setEndpointMarkerColor(color);
  setBoundingBoxBorderColor(color);
  setEllipsoidFillColor(vec4(color.rgb, color.a * (PROJECTION_VIEW ? 1.0 : 0.5)));
}
void setEllipsoidFillColor(vec3 color) { setEllipsoidFillColor(vec4(color, 1.0)); }
void setBoundingBoxFillColor(vec3 color) { setBoundingBoxFillColor(vec4(color, 1.0)); }
void setBoundingBoxBorderColor(vec3 color) { setBoundingBoxBorderColor(vec4(color, 1.0)); }
void setColor(vec3 color) { setColor(vec4(color, 1.0)); }
"#;

/// A compiled annotation shader: its WGSL compute module and how many
/// control values and properties it reads.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledAnnotationShader {
    /// `ng_annotation_style`, a compute entry point over the annotations.
    pub wgsl: String,
    /// The same as a render pass, for GPUs without compute shaders
    /// (WebGL2): `ng_annotation_vs`/`ng_annotation_fs` over a grid with
    /// one pixel a style `vec4` (see [`AnnotationShaderPipeline::new_render`]).
    pub wgsl_render: String,
    pub controls: usize,
    pub properties: usize,
}

/// The GLSL: the setters, the controls (as image shaders have them),
/// `defaultColor()`, the properties, then the user's code as `main`.
pub fn glsl(parsed: &ParsedShader, values: &[ControlValue], properties: &[(String, PropertyShaderType)], projection: bool) -> String {
    let n = control_count(parsed);
    let mut s = format!(
        "#version 450\n#define highp\n#define mediump\n#define lowp\nconst bool PROJECTION_VIEW = {projection};\nfloat ng_controls[{n}];\nvec3 ng_default_color;\nuint ng_props[{p}];\nvec3 defaultColor() {{ return ng_default_color; }}\n{string_t}{SETTERS}{colormaps}",
        string_t = crate::user_shader::STRING_T,
        p = properties.len().max(1),
        colormaps = crate::user_shader::COLORMAPS,
    );
    for (k, (id, t)) in properties.iter().enumerate() {
        s += &format!("{} prop_{id}() {{ return {}; }}\n", t.glsl(), t.read_bits(&format!("ng_props[{k}]")));
    }
    let mut k = 0;
    for ((name, control), value) in parsed.controls.iter().zip(values) {
        // Property invlerps and transfer functions are not supported in
        // annotation shaders here yet: no define, no uniforms.
        s += &crate::user_shader::plain_control_define(name, control, value, k);
        k += control_width(control);
    }
    s += "#define main ng_user_main\n";
    s += &parsed.code;
    s += "\n#undef main\nvoid main() { ng_user_main(); }\n";
    s
}

/// The values the controls read, in [`glsl`]'s order.
fn control_count(parsed: &ParsedShader) -> usize {
    parsed.controls.iter().map(|(_, c)| control_width(c)).sum::<usize>().max(1)
}

/// How many floats a control takes in `ng_controls`.
fn control_width(control: &Control) -> usize {
    match control {
        Control::Slider { .. } | Control::Select { .. } => 1,
        Control::Color { .. } => 3,
        _ => 0,
    }
}

/// The control values for [`glsl`]: sliders, colours and selects.
pub fn control_values(parsed: &ParsedShader, values: &[ControlValue]) -> Vec<f32> {
    let mut out = vec![];
    for ((_, control), value) in parsed.controls.iter().zip(values) {
        match (control, value) {
            (Control::Slider { .. }, ControlValue::Number(v)) => out.push(*v as f32),
            (Control::Color { .. }, ControlValue::Color(c)) => out.extend(c),
            // A string by its literal's id (0 if the code never names it).
            (Control::Select { .. }, ControlValue::Select(SelectValue::String(s))) => out.push(parsed.string_literals.iter().position(|l| l == s).map_or(0.0, |k| (k + 1) as f32)),
            (Control::Select { .. }, ControlValue::Select(SelectValue::Number(n))) => out.push(*n as f32),
            (Control::Slider { default, .. }, _) => out.push(*default as f32),
            (Control::Color { default, .. }, _) => out.extend(default),
            (Control::Select { .. }, _) => out.push(0.0),
            _ => {}
        }
    }
    out.resize(control_count(parsed), 0.0);
    out
}

/// `parsed` compiled for annotations with `properties`, in cross-sections
/// or (`projection`) the 3D view: the WGSL of `ng_annotation_style`, a
/// compute entry point over the annotations.
pub fn compile(parsed: &ParsedShader, values: &[ControlValue], properties: &[(String, PropertyShaderType)], projection: bool) -> Result<CompiledAnnotationShader, CompileError> {
    parsed.check()?;
    let source = glsl(parsed, values, properties, projection);
    let wgsl = crate::user_shader::fragment_to_wgsl(parsed, &source)?;
    let wgsl = crate::user_shader::without_entry_point(&wgsl);
    let (n, p) = (control_count(parsed), properties.len().max(1));
    let defaults = AnnotationStyle::default();
    let v4 = |c: [f32; 4]| format!("vec4<f32>({:?}, {:?}, {:?}, {:?})", c[0], c[1], c[2], c[3]);
    let starting = [
        v4(defaults.point_color),
        v4(defaults.point_border_color),
        v4([defaults.point_size, defaults.point_border_width, defaults.line_width, defaults.poly_line_width]),
        v4(defaults.line_color[0]),
        v4(defaults.line_color[1]),
        v4(defaults.marker_color[0]),
        v4(defaults.marker_color[1]),
        v4(defaults.marker_border_color[0]),
        v4(defaults.marker_border_color[1]),
        v4([defaults.marker_size[0], defaults.marker_size[1], defaults.marker_border_width[0], defaults.marker_border_width[1]]),
        v4(defaults.poly_line_color[0]),
        v4(defaults.poly_line_color[1]),
        v4(defaults.poly_marker_color[0]),
        v4(defaults.poly_marker_color[1]),
        v4(defaults.poly_marker_border_color[0]),
        v4(defaults.poly_marker_border_color[1]),
        v4([defaults.poly_marker_size[0], defaults.poly_marker_size[1], defaults.poly_marker_border_width[0], defaults.poly_marker_border_width[1]]),
        v4(defaults.box_border_color),
        v4(defaults.box_fill_color),
        v4([defaults.box_border_width, 0.0, 0.0, 0.0]),
        v4(defaults.ellipsoid_fill_color),
    ];
    let starting: String = starting.iter().enumerate().map(|(k, v)| format!("    ng_s[{k}] = {v};\n")).collect();
    let wrapper = format!(
        r#"
struct NgAnnotationParams {{
    count : u32,
    default_color : vec3<f32>,
}}
@group(0) @binding(0) var<uniform> ng_params : NgAnnotationParams;
@group(0) @binding(1) var<storage, read> ng_in_controls : array<f32>;
@group(0) @binding(2) var<storage, read> ng_in_props : array<u32>;
@group(0) @binding(3) var<storage, read_write> ng_out : array<vec4<f32>>;

@compute @workgroup_size(64)
fn ng_annotation_style(@builtin(global_invocation_id) id : vec3<u32>) {{
    let i = id.x;
    if (i >= ng_params.count) {{
        return;
    }}
    for (var k = 0u; k < {n}u; k++) {{
        ng_controls[k] = ng_in_controls[k];
    }}
    for (var k = 0u; k < {p}u; k++) {{
        ng_props[k] = ng_in_props[i * {p}u + k];
    }}
    ng_default_color = ng_params.default_color;
{starting}    ng_discardValue = false;
    ng_user_main();
    if (ng_discardValue) {{
        ng_s[19].y = 1.0;
    }}
    for (var k = 0u; k < {STYLE_VEC4S}u; k++) {{
        ng_out[i * {STYLE_VEC4S}u + k] = ng_s[k];
    }}
}}
"#
    );
    let control_vec4s = n.div_ceil(4).max(1);
    let render_wrapper = format!(
        r#"
struct NgAnnotationRenderParams {{
    count : u32,
    per_row : u32,
    props_width : u32,
    rows : u32,
    default_color : vec4<f32>,
    controls : array<vec4<f32>, {control_vec4s}>,
}}
@group(0) @binding(0) var<uniform> ng_rparams : NgAnnotationRenderParams;
@group(0) @binding(1) var ng_in_props_texture : texture_2d<u32>;

struct NgAnnotationRow {{
    @builtin(position) position : vec4<f32>,
    @location(0) @interpolate(flat) row : u32,
}}

// One quad a row of the grid, the row passed on: a fragment's own y is
// flipped on GL (WebGL2), which renders the grid upside down.
@vertex
fn ng_annotation_vs(@builtin(vertex_index) v : u32, @builtin(instance_index) row : u32) -> NgAnnotationRow {{
    // Two triangles: (0, 0) (1, 0) (0, 1), (0, 1) (1, 0) (1, 1).
    let corner = vec2<f32>(f32(v == 1u || v == 4u || v == 5u), f32(v == 2u || v == 3u || v == 5u));
    let y = 1.0 - 2.0 * (f32(row) + corner.y) / f32(ng_rparams.rows);
    return NgAnnotationRow(vec4<f32>(corner.x * 2.0 - 1.0, y, 0.0, 1.0), row);
}}

@fragment
fn ng_annotation_fs(input : NgAnnotationRow) -> @location(0) vec4<f32> {{
    let column = u32(input.position.x);
    let k = column % {STYLE_VEC4S}u;
    let i = input.row * ng_rparams.per_row + column / {STYLE_VEC4S}u;
    if (i >= ng_rparams.count) {{
        return vec4<f32>(0.0);
    }}
    for (var c = 0u; c < {n}u; c++) {{
        ng_controls[c] = ng_rparams.controls[c / 4u][c % 4u];
    }}
    for (var c = 0u; c < {p}u; c++) {{
        let at = i * {p}u + c;
        ng_props[c] = textureLoad(ng_in_props_texture, vec2<i32>(i32(at % ng_rparams.props_width), i32(at / ng_rparams.props_width)), 0).r;
    }}
    ng_default_color = ng_rparams.default_color.xyz;
{starting}    ng_discardValue = false;
    ng_user_main();
    if (ng_discardValue) {{
        ng_s[19].y = 1.0;
    }}
    return ng_s[k];
}}
"#
    );
    Ok(CompiledAnnotationShader { wgsl: format!("{wgsl}{wrapper}"), wgsl_render: format!("{wgsl}{render_wrapper}"), controls: n, properties: p })
}

/// How a compiled annotation shader runs.
enum Runner {
    Compute(wgpu::ComputePipeline),
    /// Without compute shaders: a render pass into an `Rgba32Float` grid.
    Render(wgpu::RenderPipeline),
}

/// A compiled annotation shader's pipeline: a compute pass, or a render
/// pass where the GPU has no compute shaders.
pub struct AnnotationShaderPipeline {
    runner: Runner,
    layout: wgpu::BindGroupLayout,
    pub controls: usize,
    pub properties: usize,
}

impl AnnotationShaderPipeline {
    /// The compute pipeline, or the render one on a GPU without compute
    /// shaders ([`crate::capabilities::compute`]).
    pub fn new(device: &wgpu::Device, compiled: &CompiledAnnotationShader) -> Self {
        if crate::capabilities::compute() {
            Self::new_compute(device, compiled)
        } else {
            Self::new_render(device, compiled)
        }
    }

    /// The render pipeline, whatever the GPU: each pixel of a grid
    /// ([`STYLE_VEC4S`] columns an annotation) works out its annotation's
    /// style and keeps one `vec4` of it.
    pub fn new_render(device: &wgpu::Device, compiled: &CompiledAnnotationShader) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ng annotation shader (render)"), source: wgpu::ShaderSource::Wgsl(compiled.wgsl_render.as_str().into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng annotation shader (render)"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ng annotation shader (render)"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ng annotation shader (render)"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("ng_annotation_vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("ng_annotation_fs"),
                targets: &[Some(wgpu::ColorTargetState { format: STYLE_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { runner: Runner::Render(pipeline), layout, controls: compiled.controls, properties: compiled.properties }
    }

    /// The compute pipeline.
    pub fn new_compute(device: &wgpu::Device, compiled: &CompiledAnnotationShader) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ng annotation shader"), source: wgpu::ShaderSource::Wgsl(compiled.wgsl.as_str().into()) });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng annotation shader"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ng annotation shader"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ng annotation shader"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("ng_annotation_style"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { runner: Runner::Compute(pipeline), layout, controls: compiled.controls, properties: compiled.properties }
    }

    /// Runs the shader over `count` annotations (`props`: each one's
    /// property bits, [`Self::properties`] apiece) into a buffer read back
    /// later.
    pub fn evaluate(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, controls: &[f32], default_color: [f32; 3], props: &[u32], count: usize) -> StyleReadback {
        use wgpu::util::DeviceExt;
        let pipeline = match &self.runner {
            Runner::Compute(pipeline) => pipeline,
            Runner::Render(pipeline) => return self.evaluate_render(pipeline, device, encoder, controls, default_color, props, count),
        };
        let mut params = [0u32; 8];
        params[0] = count as u32;
        for k in 0..3 {
            params[4 + k] = default_color[k].to_bits();
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ng annotation params"), contents: bytemuck_u32(&params), usage: wgpu::BufferUsages::UNIFORM });
        let mut c: Vec<u32> = controls.iter().map(|v| v.to_bits()).collect();
        c.resize(self.controls.max(1), 0);
        let controls = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ng annotation controls"), contents: bytemuck_u32(&c), usage: wgpu::BufferUsages::STORAGE });
        let mut p = props.to_vec();
        p.resize((count * self.properties).max(1), 0);
        let props = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ng annotation properties"), contents: bytemuck_u32(&p), usage: wgpu::BufferUsages::STORAGE });
        let size = (count.max(1) * STYLE_VEC4S * 16) as u64;
        let out = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ng annotation styles"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let readback = AsyncReadback::new(device, size, "ng annotation styles readback");
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ng annotation shader"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: controls.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: props.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ng annotation shader"), timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((count as u32).div_ceil(64).max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, readback.buffer(), 0, size);
        StyleReadback { readback, count, layout: StyleLayout::Packed }
    }

    /// [`Self::evaluate`] as a render pass: the controls in a uniform, the
    /// property bits in an `R32Uint` texture, the styles a grid of
    /// `Rgba32Float` pixels read back row by row.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_render(&self, pipeline: &wgpu::RenderPipeline, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, controls: &[f32], default_color: [f32; 3], props: &[u32], count: usize) -> StyleReadback {
        use wgpu::util::DeviceExt;
        let max = device.limits().max_texture_dimension_2d as usize;
        // Annotations a row, and as many as the grid holds (the rest keep
        // the default style).
        let per_row = (max / STYLE_VEC4S).clamp(1, count.max(1));
        let evaluated = count.min(per_row * max);
        let rows = evaluated.div_ceil(per_row).max(1);
        let width = (per_row * STYLE_VEC4S) as u32;
        // The property bits, `props_width` a row.
        let p = self.properties.max(1);
        let mut bits = props.to_vec();
        bits.resize((evaluated * p).max(1), 0);
        let props_width = bits.len().min(max).max(1);
        let props_rows = bits.len().div_ceil(props_width).min(max);
        let padded_row = (props_width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        let mut padded = vec![0u8; padded_row * props_rows];
        for (r, chunk) in bits.chunks(props_width).take(props_rows).enumerate() {
            padded[r * padded_row..r * padded_row + chunk.len() * 4].copy_from_slice(bytemuck_u32(chunk));
        }
        let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ng annotation properties"), contents: &padded, usage: wgpu::BufferUsages::COPY_SRC });
        let props_size = wgpu::Extent3d { width: props_width as u32, height: props_rows as u32, depth_or_array_layers: 1 };
        let props_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ng annotation properties"),
            size: props_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_row as u32), rows_per_image: Some(props_rows as u32) } },
            props_texture.as_image_copy(),
            props_size,
        );
        // count, per_row, props_width, pad; the default colour; the controls.
        let control_vec4s = self.controls.div_ceil(4).max(1);
        let mut params = vec![0u32; 8 + 4 * control_vec4s];
        params[0] = evaluated as u32;
        params[1] = per_row as u32;
        params[2] = props_width as u32;
        params[3] = rows as u32;
        for k in 0..3 {
            params[4 + k] = default_color[k].to_bits();
        }
        for (k, v) in controls.iter().take(self.controls).enumerate() {
            params[8 + k] = v.to_bits();
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ng annotation params"), contents: bytemuck_u32(&params), usage: wgpu::BufferUsages::UNIFORM });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ng annotation shader (render)"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&props_texture.create_view(&Default::default())) },
            ],
        });
        let size = wgpu::Extent3d { width, height: rows as u32, depth_or_array_layers: 1 };
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ng annotation styles"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: STYLE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ng annotation shader (render)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..6, 0..rows as u32);
        }
        let row_bytes = (width as usize * 16).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        let readback = AsyncReadback::new(device, (row_bytes * rows) as u64, "ng annotation styles readback");
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: readback.buffer(), layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row_bytes as u32), rows_per_image: Some(rows as u32) } },
            size,
        );
        StyleReadback { readback, count, layout: StyleLayout::Grid { row_bytes, per_row, evaluated } }
    }
}

/// The styles' format in the render pass's grid.
const STYLE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// How the styles lie in the buffer read back.
enum StyleLayout {
    /// One after the other ([`STYLE_VEC4S`] `vec4`s each).
    Packed,
    /// In rows of `per_row` annotations, `row_bytes` apart; past
    /// `evaluated`, the default style.
    Grid { row_bytes: usize, per_row: usize, evaluated: usize },
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    // SAFETY: `u32` has no padding and any byte pattern is a valid `u8`.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

/// Styles being read back.
pub struct StyleReadback {
    readback: AsyncReadback,
    count: usize,
    layout: StyleLayout,
}

impl StyleReadback {
    /// Once the work is submitted: maps the buffer, calling `wake` when it
    /// can be read.
    pub fn map(&self, wake: Arc<dyn Fn() + Send + Sync>) {
        self.readback.map(wake);
    }

    /// The styles, once mapped.
    pub fn read(&self) -> Option<Vec<AnnotationStyle>> {
        let data = self.readback.read()?;
        let v4_at = |at: usize| -> [f32; 4] { std::array::from_fn(|c| f32::from_le_bytes(data[at + 4 * c..at + 4 * c + 4].try_into().unwrap())) };
        let style = |at: usize| -> AnnotationStyle {
            let v: Vec<[f32; 4]> = (0..STYLE_VEC4S).map(|k| v4_at(at + 16 * k)).collect();
            AnnotationStyle::from_vec4s(&v)
        };
        Some(
            (0..self.count)
                .map(|i| match self.layout {
                    StyleLayout::Packed => style(i * STYLE_VEC4S * 16),
                    StyleLayout::Grid { row_bytes, per_row, evaluated } if i < evaluated => style((i / per_row) * row_bytes + (i % per_row) * STYLE_VEC4S * 16),
                    StyleLayout::Grid { .. } => AnnotationStyle::default(),
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wgsl_validates(wgsl: &str) {
        let module = naga::front::wgsl::parse_str(wgsl).unwrap_or_else(|e| panic!("{}\n{wgsl}", e.emit_to_string(wgsl)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{e:?}\n{wgsl}"));
    }

    #[test]
    fn annotation_shaders_compile() {
        let props = [("color".to_owned(), PropertyShaderType::Rgb), ("size".to_owned(), PropertyShaderType::Float), ("kind".to_owned(), PropertyShaderType::Uint)];
        for code in [
            "\nvoid main() {\n  setColor(defaultColor());\n}\n",
            "#uicontrol float scale slider(min=1, max=10, default=2)\n#uicontrol vec3 tint color(default=\"red\")\nvoid main() {\n  setColor(prop_color() * tint);\n  setPointMarkerSize(prop_size() * scale);\n  setEndpointMarkerColor(vec3(1.0), vec3(0.0));\n  setLineWidth(2.0);\n  if (prop_kind() == 3u) discard;\n  setBoundingBoxFillColor(vec4(colormapJet(0.5), 0.3));\n}\n",
        ] {
            let parsed = crate::user_shader::parse(code, ng_core::DataType::Float32);
            let values = crate::user_shader::control_values(&parsed, ng_core::DataType::Float32, None);
            for projection in [false, true] {
                let compiled = compile(&parsed, &values, &props, projection).unwrap_or_else(|e| panic!("{e:?}\n{}", glsl(&parsed, &values, &props, projection)));
                wgsl_validates(&compiled.wgsl);
            }
        }
        let parsed = crate::user_shader::parse("void main() { setColor(prop_missing()); }", ng_core::DataType::Float32);
        assert!(compile(&parsed, &[], &props, false).is_err());
    }

    #[test]
    fn default_shader_style() {
        let s = AnnotationStyle::default_shader([1.0, 1.0, 0.0], false);
        assert_eq!((s.point_color, s.ellipsoid_fill_color[3], s.point_size), ([1.0, 1.0, 0.0, 1.0], 0.5, 5.0));
        assert_eq!(PropertyShaderType::Int.bits(-2.0), 0xffff_fffe);
    }
}
