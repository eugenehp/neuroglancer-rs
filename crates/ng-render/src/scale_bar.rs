//! The cross-section scale bar (`src/widget/scale_bar.ts`): a bar of a round
//! physical length near a quarter of the panel (at most 100 px), labelled
//! with that length, on a translucent backing, at the panel's bottom left;
//! one bar per distinct unit and relative scale among the displayed
//! dimensions, stacked upward.
//!
//! The bar's length, label and layout are computed as neuroglancer does. The
//! image is painted by a [`ScaleBarPainter`]: in the browser with the same
//! 2D-canvas calls neuroglancer makes (so the text is the browser's own),
//! natively with threers' font rasterizer.

use ng_core::js;
use ng_core::navigation::DisplayDimensionRenderInfo;
use ng_core::si_units::exp10;

/// `ScaleBarOptions` (`defaultScaleBarOptions`).
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleBarOptions {
    pub scale_factor: f64,
    pub text_height: f64,
    pub bar_height: f64,
    pub bar_top_margin: f64,
    pub font_name: String,
    pub padding: f64,
    pub max_width: f64,
    pub max_width_fraction: f64,
    pub left_offset: f64,
    pub bottom_offset: f64,
    pub margin_between: f64,
}

impl Default for ScaleBarOptions {
    fn default() -> Self {
        Self {
            scale_factor: 1.0,
            text_height: 15.0,
            bar_height: 8.0,
            bar_top_margin: 5.0,
            font_name: "sans-serif".into(),
            padding: 2.0,
            max_width: 100.0,
            max_width_fraction: 0.25,
            left_offset: 10.0,
            bottom_offset: 10.0,
            margin_between: 5.0,
        }
    }
}

/// `pickSiPrefix`: the first preferred prefix whose exponent is at most
/// `log10(x)`, else the smallest.
pub fn pick_si_prefix(x: f64) -> (&'static str, i32) {
    let p = ng_core::si_units::pick_si_prefix(x);
    (p.prefix, p.exponent)
}

/// One bar: its label (`"x y: "` when several bars show), its length in
/// pixels, and the physical length it stands for.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleBar {
    pub label: String,
    pub length_in_pixels: f64,
    pub physical_length: f64,
    pub physical_unit: String,
}

impl ScaleBar {
    /// The text drawn: `${label}${physicalLength} ${physicalUnit}`.
    pub fn text(&self) -> String {
        format!("{}{} {}", self.label, js::number_to_string(self.physical_length), self.physical_unit)
    }
}

/// `ScaleBarDimensions.update`.
fn dimensions(target_length: f64, size_per_pixel: f64, base_unit: &str) -> (f64, f64, String) {
    const ALLOWED: [f64; 6] = [1.5, 2.0, 3.0, 5.0, 7.5, 10.0];
    let target_size = target_length * size_per_pixel;
    let exponent = js::log10(target_size).floor() as i32;
    let ten = exp10(exponent);
    let target_significand = target_size / ten;
    let mut best = 1.0;
    for a in ALLOWED {
        if (a - target_significand).abs() < (best - target_significand).abs() {
            best = a;
        } else {
            break;
        }
    }
    let physical_size = best * ten;
    let (prefix, prefix_exponent) = pick_si_prefix(physical_size);
    let length = js::round(physical_size / size_per_pixel);
    (length, best * exp10(exponent - prefix_exponent), format!("{prefix}{base_unit}"))
}

/// `MultipleScaleBarTextures.draw`'s bars, bottom first: one per distinct
/// (unit, relative scale) among the display dimensions.
pub fn scale_bars(
    info: &DisplayDimensionRenderInfo,
    relative_factors: &[f64],
    zoom: f64,
    logical_width: f64,
    options: &ScaleBarOptions,
) -> Vec<ScaleBar> {
    let target = (options.max_width_fraction * logical_width).min(options.max_width * options.scale_factor);
    // (unit, factor, size per pixel, label)
    let mut bars: Vec<(String, f64, f64, String)> = vec![];
    for i in 0..info.display_rank {
        let dim = info.display_dimension_indices[i] as usize;
        let unit = &info.display_dimension_units[i];
        let factor = relative_factors.get(dim).copied().unwrap_or(1.0);
        let k = match bars.iter().position(|b| b.0 == *unit && b.1 == factor) {
            Some(k) => k,
            None => {
                let size = info.display_dimension_scales[i] * zoom / info.canonical_voxel_factors[i];
                bars.push((unit.clone(), factor, size, String::new()));
                bars.len() - 1
            }
        };
        bars[k].3.push_str(&format!("{} ", info.global_dimension_names[dim]));
    }
    let single = bars.len() == 1;
    bars.into_iter()
        .rev()
        .map(|(unit, _, size, label)| {
            let (length, physical_length, physical_unit) = dimensions(target, size, &unit);
            ScaleBar {
                label: if single { String::new() } else { format!("{label}: ") },
                length_in_pixels: length,
                physical_length,
                physical_unit,
            }
        })
        .collect()
}

/// A painted bar: RGBA, not premultiplied, row 0 at the top.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleBarImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Paints a bar as `makeScaleBarTexture` does: a canvas
/// `max(length, textWidth) + 2·padding` wide and
/// `textHeight + barTopMargin + barHeight + 2·padding` tall, filled with
/// `rgba(0, 0, 0, 0.3)`; the text (`bold <textHeight>px <font>`, white,
/// centred, baseline at `height − padding − barHeight − barTopMargin`); and
/// the bar (white, at `(padding, height − padding − barHeight)`).
pub trait ScaleBarPainter {
    fn paint(&self, bar: &ScaleBar, options: &ScaleBarOptions) -> ScaleBarImage;
}

/// Natively: threers' caption font (SF Compact on macOS when installed, its
/// bitmap face elsewhere), bold by overdraw. Close to the browser's text, not
/// the same pixels.
#[cfg(not(target_arch = "wasm32"))]
pub struct NativeScaleBarPainter {
    font: std::sync::Mutex<threers::captions::CaptionFont>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for NativeScaleBarPainter {
    fn default() -> Self {
        Self { font: std::sync::Mutex::new(threers::captions::CaptionFont::ui()) }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ScaleBarPainter for NativeScaleBarPainter {
    fn paint(&self, bar: &ScaleBar, o: &ScaleBarOptions) -> ScaleBarImage {
        let mut font = self.font.lock().unwrap();
        let text = bar.text();
        let size = (o.text_height * o.scale_factor) as f32;
        let text_width = font.measure(&text, size, 0.0) as f64 + 1.0;
        let inner_width = bar.length_in_pixels.max(text_width);
        let bar_height = o.bar_height * o.scale_factor;
        let margin = o.bar_top_margin * o.scale_factor;
        let padding = o.padding * o.scale_factor;
        let height = (size as f64 + margin + bar_height + 2.0 * padding) as u32;
        let width = (inner_width + 2.0 * padding) as u32;
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        // Backing: black at 0.3 (77 of 255).
        for p in rgba.chunks_exact_mut(4) {
            p[3] = 77;
        }
        let mut put = |x: i32, y: i32, coverage: f32| {
            if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                return;
            }
            let i = ((y as u32 * width + x as u32) * 4) as usize;
            // White over the backing, unpremultiplied.
            let a0 = rgba[i + 3] as f32 / 255.0;
            let a = coverage + a0 * (1.0 - coverage);
            let c = if a > 0.0 { coverage / a } else { 0.0 };
            for k in 0..3 {
                rgba[i + k] = (c * 255.0).round() as u8;
            }
            rgba[i + 3] = (a * 255.0).round() as u8;
        };
        let baseline = height as f64 - padding - bar_height - margin;
        let mut pen = width as f64 / 2.0 - text_width / 2.0;
        let mut previous = None;
        for c in text.chars() {
            if let Some(p) = previous {
                pen += font.kerning(p, c, size) as f64;
            }
            let g = font.glyph(c, size).clone();
            for gy in 0..g.height {
                for gx in 0..g.width {
                    let cov = g.coverage[(gy * g.width + gx) as usize] as f32 / 255.0;
                    if cov > 0.0 {
                        let x = pen as i32 + g.offset_x + gx as i32;
                        let y = baseline as i32 + g.offset_y + gy as i32;
                        put(x, y, cov);
                        put(x + 1, y, cov); // bold
                    }
                }
            }
            pen += g.advance as f64;
            previous = Some(c);
        }
        let top = (height as f64 - padding - bar_height) as i32;
        for y in top..top + bar_height as i32 {
            for x in padding as i32..(padding + bar.length_in_pixels) as i32 {
                put(x, y, 1.0);
            }
        }
        ScaleBarImage { width, height, rgba }
    }
}

const SHADER: &str = r#"
@group(0) @binding(0) var image : texture_2d<f32>;
// x, y: the image's corner; z: 1 to read its rows bottom-up.
@group(0) @binding(1) var<uniform> origin : vec4<f32>;
@fragment
fn fs(in : FullscreenVertex) -> @location(0) vec4<f32> {
    // 1:1 with the image (neuroglancer samples LINEAR at texel centres).
    var p = vec2<i32>(floor(in.clip.xy - origin.xy));
    if (origin.z == 1.0) {
        p.y = i32(textureDimensions(image).y) - 1 - p.y;
    }
    return textureLoad(image, p, 0);
}
"#;

/// Blends painted bars into a panel (`OffscreenCopyHelper` under
/// `blendFunc(SRC_ALPHA, ONE_MINUS_SRC_ALPHA)`).
pub struct ScaleBarRenderer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl ScaleBarRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        use threers::renderer::fullscreen;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng scale bar"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ng scale bar"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ng scale bar"),
            source: wgpu::ShaderSource::Wgsl(format!("{}\n{SHADER}", fullscreen::VS).into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ng scale bar"),
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
                    format: crate::compose::OFFSCREEN_FORMAT,
                    blend: Some(crate::compose::LAYER_BLEND),
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
        Self { pipeline, layout }
    }

    /// Draws `image` with its top-left at `(x, y)` of `panel`; `flipped`:
    /// `panel` is upside down, and `(x, y)` its bottom-left corner there.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        panel: &wgpu::TextureView,
        image: &ScaleBarImage,
        x: i32,
        y: i32,
        flipped: bool,
    ) {
        use wgpu::util::DeviceExt;
        if image.width == 0 || image.height == 0 {
            return;
        }
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("ng scale bar"),
                size: wgpu::Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &image.rgba,
        );
        let origin = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ng scale bar origin"),
            contents: bytemuck::cast_slice(&[x as f32, y as f32, flipped as u32 as f32, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let view = texture.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ng scale bar"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: origin.as_entire_binding() },
            ],
        });
        let mut pass = threers::renderer::fullscreen::begin(encoder, panel, threers::renderer::fullscreen::Load::Keep);
        pass.set_viewport(x as f32, y as f32, image.width as f32, image.height as f32, 0.0, 1.0);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
