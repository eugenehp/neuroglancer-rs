//! Fills a rectangle of the target with one color: a panel's background.

use threers::renderer::{fullscreen, PixelRect};
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
@group(0) @binding(0) var<uniform> color : vec4<f32>;
@fragment
fn fs(in : FullscreenVertex) -> @location(0) vec4<f32> {
    return color;
}
"#;

pub struct FillRenderer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    srgb_target: bool,
}

/// One rectangle's color. Each rectangle drawn in a frame needs its own, for
/// the same reason as `threers::volume::SliceBinding`.
pub struct FillBinding {
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl FillRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng fill layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ng fill pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ng fill shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{}\n{SHADER}", fullscreen::VS).into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ng fill pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(fullscreen::VS_ENTRY),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(target_format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            layout,
            srgb_target: target_format.is_srgb(),
        }
    }

    pub fn create_binding(&self, device: &wgpu::Device) -> FillBinding {
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ng fill color"),
            contents: bytemuck::bytes_of(&[0f32; 4]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ng fill bind group"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        FillBinding {
            uniforms,
            bind_group,
        }
    }

    /// `rgb` is display-referred (what neuroglancer writes to its canvas);
    /// it is linearized for an sRGB target so it shows the same.
    pub fn update(&self, queue: &wgpu::Queue, binding: &FillBinding, rgb: [f32; 3]) {
        let rgb = if self.srgb_target { rgb.map(srgb_to_linear) } else { rgb };
        queue.write_buffer(&binding.uniforms, 0, bytemuck::bytes_of(&[rgb[0], rgb[1], rgb[2], 1.0]));
    }

    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, rect: PixelRect, binding: &FillBinding) {
        if rect.width == 0 || rect.height == 0 {
            return;
        }
        let mut pass = fullscreen::begin(encoder, target, fullscreen::Load::Keep);
        pass.set_viewport(rect.x as f32, rect.y as f32, rect.width as f32, rect.height as f32, 0.0, 1.0);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &binding.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
