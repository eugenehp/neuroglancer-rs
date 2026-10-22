//! How a cross-section panel is put together, as `SliceView.updateRendering`
//! and `SliceViewRenderHelper` (`src/sliceview/frontend.ts`) do it:
//!
//! 1. Layers draw in order into an offscreen RGBA8 texture cleared to
//!    transparent. The first replaces what is there, alpha included; later
//!    ones blend with `blendFunc(SRC_ALPHA, ONE_MINUS_SRC_ALPHA)`, on alpha
//!    too.
//! 2. Within a layer, data draws finest scale first at depth 0 against a
//!    depth buffer cleared to 1 per layer, so coarser data only fills gaps.
//! 3. The panel (a second RGBA8 texture, the panel's own framebuffer) gets
//!    the offscreen colour where its alpha is nonzero and the background
//!    elsewhere; annotations then blend over it.
//! 4. The panel goes to the canvas. neuroglancer's canvas is premultiplied
//!    over a black page, so what shows is the panel's RGB, alpha aside.

use bytemuck::Zeroable;
use threers::renderer::{fullscreen, PixelRect};
use threers::volume::{SliceRenderer, SliceTarget, VolumeSample};
use wgpu::util::DeviceExt;

/// The offscreen colour format (`makeTextureBuffers`' default, `RGBA8`).
pub const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// `blendFunc(SRC_ALPHA, ONE_MINUS_SRC_ALPHA)`: the same factors for colour
/// and alpha (wgpu's `ALPHA_BLENDING` uses `ONE` for source alpha).
pub const LAYER_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

/// A layer's two pipelines: as the first layer (replacing) and over others.
pub struct LayerPipelines {
    pub first: SliceRenderer,
    pub blended: SliceRenderer,
}

impl LayerPipelines {
    /// `lookup`: the colour function reads a lookup table
    /// (`SliceRenderer::with_lookup`).
    pub fn new(device: &wgpu::Device, sample: VolumeSample, color_wgsl: &str, lookup: bool) -> Self {
        let target = |blend| SliceTarget {
            format: OFFSCREEN_FORMAT,
            blend,
            depth: Some((DEPTH_FORMAT, wgpu::CompareFunction::Less)),
        };
        let make = |blend| {
            if lookup {
                SliceRenderer::with_lookup(device, target(blend), sample, color_wgsl)
            } else {
                SliceRenderer::with_target(device, target(blend), sample, color_wgsl)
            }
        };
        Self {
            first: make(None),
            blended: make(Some(LAYER_BLEND)),
        }
    }

    /// The pipeline for the `index`th layer drawn (`setGLBlendMode`).
    pub fn for_layer(&self, index: usize) -> &SliceRenderer {
        if index == 0 {
            &self.first
        } else {
            &self.blended
        }
    }
}

/// A panel's offscreen colour and depth (the slice view's framebuffer), and
/// the panel's own colour (what annotations draw over).
pub struct PanelTarget {
    pub width: u32,
    pub height: u32,
    pub color: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub depth_view: wgpu::TextureView,
    pub panel_view: wgpu::TextureView,
    /// The panel upside down: overlays draw here, in the orientation of
    /// neuroglancer's GL framebuffer (row 0 at the bottom), so rasterization
    /// ties (a line on a pixel boundary, a line's excluded last pixel) fall
    /// the way they do there.
    pub flipped_view: wgpu::TextureView,
    /// Whether the depth can be bound as a texture (see
    /// [`with_sampled_depth`](Self::with_sampled_depth)).
    pub sampled_depth: bool,
}

impl PanelTarget {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        Self::create(device, width, height, false)
    }

    /// A target whose depth can also be bound as a texture: for a host that
    /// composites the panel into its own frame by depth (`Viewer::render_scene`).
    pub fn with_sampled_depth(device: &wgpu::Device, width: u32, height: u32) -> Self {
        Self::create(device, width, height, true)
    }

    fn create(device: &wgpu::Device, width: u32, height: u32, sampled_depth: bool) -> Self {
        let make = |format, usage, label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = make(
            OFFSCREEN_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            "ng panel color",
        );
        // bindable for the host's composite, copyable for reading it back
        let depth_usage = if sampled_depth {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
        let depth = make(DEPTH_FORMAT, depth_usage, "ng panel depth");
        let panel = make(
            OFFSCREEN_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            "ng panel",
        );
        let flipped = make(
            OFFSCREEN_FORMAT,
            if sampled_depth {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC
            } else {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            },
            "ng panel, flipped",
        );
        Self {
            width,
            height,
            color_view: color.create_view(&Default::default()),
            depth_view: depth.create_view(&Default::default()),
            panel_view: panel.create_view(&Default::default()),
            flipped_view: flipped.create_view(&Default::default()),
            color,
            sampled_depth,
        }
    }

    /// Begins the pass for the `index`th layer: colour cleared to
    /// transparent for the first, kept after; depth cleared for each.
    pub fn begin_layer<'e>(&self, encoder: &'e mut wgpu::CommandEncoder, index: usize) -> wgpu::RenderPass<'e> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ng panel layer"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if index == 0 { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Clears the colour to transparent (a panel with no layers).
    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        drop(fullscreen::begin(encoder, &self.color_view, fullscreen::Load::Clear));
    }
}

const COMPOSITE: &str = r#"
struct CompositeUniforms {
    background : vec4<f32>,
    origin : vec2<i32>,
    srgb_target : u32,
    keep_alpha : u32,
};
@group(0) @binding(0) var<uniform> u : CompositeUniforms;
@group(0) @binding(1) var panel : texture_2d<f32>;

fn srgb_to_linear(c : vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@fragment
fn fs(in : FullscreenVertex) -> @location(0) vec4<f32> {
    var c = textureLoad(panel, vec2<i32>(floor(in.clip.xy)) - u.origin, 0);
    if (c.a == 0.0 && u.keep_alpha == 1u) {
        c = u.background;
    }
    var rgb = c.rgb;
    if (u.srgb_target == 1u) {
        rgb = srgb_to_linear(rgb);
    }
    return vec4<f32>(rgb, select(1.0, c.a, u.keep_alpha == 1u));
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    background: [f32; 4],
    origin: [i32; 2],
    srgb_target: u32,
    keep_alpha: u32,
}

/// The compositing module (entry points `vs_main` and `fs`), for checking.
pub fn composite_shader_source() -> String {
    format!("{}\n{COMPOSITE}", fullscreen::VS)
}

/// Draws a texture into a rectangle of another: the slice view's framebuffer
/// into the panel ([`Compositor::panel`], `SliceViewRenderHelper`: the
/// background where alpha is 0, alpha kept), and the panel onto the frame
/// ([`Compositor::new`], `OffscreenCopyHelper`: RGB as is).
pub struct Compositor {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    srgb_target: bool,
    keep_alpha: bool,
}

/// One panel's uniforms and texture.
pub struct CompositeBinding {
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Compositor {
    /// Into a panel texture ([`OFFSCREEN_FORMAT`]), keeping alpha.
    pub fn panel(device: &wgpu::Device) -> Self {
        Self { keep_alpha: true, ..Self::new(device, OFFSCREEN_FORMAT) }
    }

    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng composite layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ng composite pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ng composite shader"),
            source: wgpu::ShaderSource::Wgsl(composite_shader_source().into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ng composite pipeline"),
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
            keep_alpha: false,
        }
    }

    /// A binding reading `source`.
    pub fn create_binding(&self, device: &wgpu::Device, source: &wgpu::TextureView) -> CompositeBinding {
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ng composite uniforms"),
            contents: bytemuck::bytes_of(&CompositeUniforms::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ng composite bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(source) },
            ],
        });
        CompositeBinding { uniforms, bind_group }
    }

    /// Draws the binding's texture into `rect` of `target`; `background` is
    /// display-referred RGB (`crossSectionBackgroundColor`).
    pub fn draw(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        rect: PixelRect,
        binding: &CompositeBinding,
        background: [f32; 3],
    ) {
        if rect.width == 0 || rect.height == 0 {
            return;
        }
        let u = CompositeUniforms {
            background: [background[0], background[1], background[2], 1.0],
            origin: [rect.x as i32, rect.y as i32],
            srgb_target: self.srgb_target as u32,
            keep_alpha: self.keep_alpha as u32,
        };
        queue.write_buffer(&binding.uniforms, 0, bytemuck::bytes_of(&u));
        let mut pass = fullscreen::begin(encoder, target, fullscreen::Load::Keep);
        pass.set_viewport(rect.x as f32, rect.y as f32, rect.width as f32, rect.height as f32, 0.0, 1.0);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &binding.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

const FLIP: &str = r#"
@group(0) @binding(0) var source : texture_2d<f32>;
@fragment
fn fs(in : FullscreenVertex) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(source));
    let p = vec2<i32>(floor(in.clip.xy));
    return textureLoad(source, vec2<i32>(p.x, size.y - 1 - p.y), 0);
}
"#;

/// Copies a panel texture upside down into another (both
/// [`OFFSCREEN_FORMAT`], the same size).
pub struct Flipper {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl Flipper {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng flip"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ng flip"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ng flip"),
            source: wgpu::ShaderSource::Wgsl(format!("{}\n{FLIP}", fullscreen::VS).into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ng flip"),
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
                targets: &[Some(OFFSCREEN_FORMAT.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// Copies `source` into `target`, upside down.
    pub fn flip(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, source: &wgpu::TextureView, target: &wgpu::TextureView) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ng flip"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) }],
        });
        let mut pass = fullscreen::begin(encoder, target, fullscreen::Load::Keep);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

const IMAGE_OVERLAY: &str = r#"
@group(0) @binding(0) var image: texture_2d<f32>;

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(image, vec2<i32>(p.xy), 0);
}
"#;

/// Blends an RGBA8 image (straight alpha, sRGB-encoded, as a 2D canvas or
/// threers' `Overlay` holds it) over a frame of the same size: a UI drawn on
/// the CPU, over the panels.
pub struct ImageOverlay {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    texture: Option<(wgpu::Texture, wgpu::BindGroup, u32, u32)>,
}

impl ImageOverlay {
    /// For targets of `target_format`. On an sRGB target the image is
    /// decoded and blended in linear light; otherwise as stored.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let format = if target_format.is_srgb() { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng image overlay"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ng image overlay"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ng image overlay"),
            source: wgpu::ShaderSource::Wgsl(format!("{}\n{IMAGE_OVERLAY}", fullscreen::VS).into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ng image overlay"),
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
        Self { pipeline, layout, format, texture: None }
    }

    /// Uploads `rgba` (`width` × `height`, rows from the top) and blends it
    /// over `target`.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        if width == 0 || height == 0 || rgba.len() < (width * height * 4) as usize {
            return;
        }
        if !matches!(&self.texture, Some((_, _, w, h)) if *w == width && *h == height) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ng image overlay"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ng image overlay"),
                layout: &self.layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) }],
            });
            self.texture = Some((texture, bind_group, width, height));
        }
        let (texture, bind_group, _, _) = self.texture.as_ref().unwrap();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(height) },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        let mut pass = fullscreen::begin(encoder, target, fullscreen::Load::Keep);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
