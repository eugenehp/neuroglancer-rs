//! Annotation shaders without compute shaders (WebGL2): the render pass's
//! WGSL translates to WebGL2 GLSL, and it works out the same styles as the
//! compute pass, bit for bit.

use ng_render::annotation_shader::{compile, AnnotationShaderPipeline, AnnotationStyle, PropertyShaderType};
use threers::gpu::testing;

const CODE: &str = "#uicontrol float scale slider(min=1, max=10, default=3)\n\
#uicontrol vec3 tint color(default=\"#ff8000\")\n\
void main() {\n\
  setColor(prop_color() * tint);\n\
  setPointMarkerSize(prop_size() * scale);\n\
  setEndpointMarkerColor(vec3(1.0), vec3(0.0));\n\
  setLineWidth(2.0 + float(prop_kind()));\n\
  if (prop_kind() == 3u) discard;\n\
  setBoundingBoxFillColor(vec4(colormapJet(prop_size() / 40.0), 0.3));\n\
}\n";

fn properties() -> Vec<(String, PropertyShaderType)> {
    vec![("color".to_owned(), PropertyShaderType::Rgb), ("size".to_owned(), PropertyShaderType::Float), ("kind".to_owned(), PropertyShaderType::Uint)]
}

fn compiled(projection: bool) -> ng_render::annotation_shader::CompiledAnnotationShader {
    let parsed = ng_render::user_shader::parse(CODE, ng_core::DataType::Float32);
    let values = ng_render::user_shader::control_values(&parsed, ng_core::DataType::Float32, None);
    compile(&parsed, &values, &properties(), projection).unwrap()
}

#[test]
fn the_render_pass_translates_for_webgl2() {
    for projection in [false, true] {
        let source = compiled(projection).wgsl_render;
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{e:?}"));
        for (stage, entry) in [(naga::ShaderStage::Vertex, "ng_annotation_vs"), (naga::ShaderStage::Fragment, "ng_annotation_fs")] {
            let options = naga::back::glsl::Options { version: naga::back::glsl::Version::Embedded { version: 300, is_webgl: true }, ..Default::default() };
            let pipeline = naga::back::glsl::PipelineOptions { shader_stage: stage, entry_point: entry.to_string(), multiview: None };
            let mut out = String::new();
            naga::back::glsl::Writer::new(&mut out, &module, &info, &options, &pipeline, Default::default())
                .and_then(|mut w| w.write().map(|_| ()))
                .unwrap_or_else(|e| panic!("{entry}: {e}"));
        }
    }
}

/// Runs `pipeline` over the annotations and waits for the styles.
fn styles(gpu: &threers::gpu::GpuContext, pipeline: &AnnotationShaderPipeline, controls: &[f32], props: &[u32], count: usize) -> Vec<AnnotationStyle> {
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let readback = pipeline.evaluate(&gpu.device, &mut encoder, controls, [0.2, 0.4, 0.6], props, count);
    gpu.queue.submit([encoder.finish()]);
    readback.map(std::sync::Arc::new(|| {}));
    loop {
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        if let Some(styles) = readback.read() {
            return styles;
        }
    }
}

#[test]
fn the_render_pass_works_out_the_compute_passes_styles() {
    let Some(gpu) = testing::context("ng-render annotation shader") else { return };
    let compiled = compiled(false);
    let parsed = ng_render::user_shader::parse(CODE, ng_core::DataType::Float32);
    let values = ng_render::user_shader::control_values(&parsed, ng_core::DataType::Float32, None);
    let controls = ng_render::annotation_shader::control_values(&parsed, &values);
    // Enough annotations for several rows of the render pass's grid (780 a
    // row at 16384 px, 97 at WebGL2's 2048), each with its colour (packed
    // RGB), size and kind.
    let count = 2000;
    let props: Vec<u32> = (0..count as u32)
        .flat_map(|i| [PropertyShaderType::Rgb.bits(f64::from(0x10_20_30 + i * 0x01_03_07)), PropertyShaderType::Float.bits(f64::from(i % 40) + 0.5), i % 5])
        .collect();
    let compute = styles(&gpu, &AnnotationShaderPipeline::new_compute(&gpu.device, &compiled), &controls, &props, count);
    let render = styles(&gpu, &AnnotationShaderPipeline::new_render(&gpu.device, &compiled), &controls, &props, count);
    assert_eq!(compute.len(), count);
    for (i, (c, r)) in compute.iter().zip(&render).enumerate() {
        assert_eq!(format!("{c:?}"), format!("{r:?}"), "annotation {i}");
    }
    // The shader ran: sizes follow the property, and kind 3 is discarded.
    assert_ne!(format!("{:?}", compute[1]), format!("{:?}", compute[2]));
}
