//! A user shader's invlerp legend (`ColorLegendPanel`): the layer's shader
//! run once per legend pixel, its data value the window's at that pixel
//! (`ng_colorLegendLerp`), the opacity 1, read back.
//!
//! The strip renders through the layer's own pipeline into a texture a row
//! high and is copied out; the copy maps asynchronously (as screenshots
//! do), so a legend shows a frame or two after it is asked for.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ng_core::DataType;
use threers::math::Matrix4;
use threers::volume::{Data3DTexture, GpuVolume, SliceParams};

/// What a legend was drawn for: the values, the shader and its uniforms.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LegendKey {
    pub values: Vec<u64>,
    pub shader: String,
    pub uniforms: Vec<u32>,
}

/// A legend's copy under way.
pub(crate) struct LegendJob {
    pub key: LegendKey,
    pub buffer: wgpu::Buffer,
    pub mapped: Arc<AtomicBool>,
    pub width: u32,
}

/// The values as the data type's voxels, uploaded as its volumes are.
fn voxels(t: DataType, values: &[f64]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(values.len() * t.bytes());
    for &v in values {
        match t {
            DataType::Uint8 => raw.push(v.clamp(0.0, 255.0) as u8),
            DataType::Int8 => raw.push(v.clamp(-128.0, 127.0) as i8 as u8),
            DataType::Uint16 => raw.extend_from_slice(&(v.clamp(0.0, 65535.0) as u16).to_le_bytes()),
            DataType::Int16 => raw.extend_from_slice(&(v.clamp(-32768.0, 32767.0) as i16).to_le_bytes()),
            DataType::Uint32 => raw.extend_from_slice(&(v.clamp(0.0, u32::MAX as f64) as u32).to_le_bytes()),
            DataType::Int32 => raw.extend_from_slice(&(v.clamp(i32::MIN as f64, i32::MAX as f64) as i32).to_le_bytes()),
            DataType::Uint64 => raw.extend_from_slice(&(v.max(0.0) as u64).to_le_bytes()),
            DataType::Float32 => raw.extend_from_slice(&(v as f32).to_le_bytes()),
        }
    }
    ng_render::image::upload_bytes(t, &raw)
}

/// Renders `values` through `pipeline` (the layer's, its uniforms in
/// `lookup`) and starts copying the row out.
pub(crate) fn start(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &threers::volume::SliceRenderer,
    lookup: &wgpu::TextureView,
    t: DataType,
    values: &[f64],
    key: LegendKey,
) -> LegendJob {
    let width = values.len().max(1) as u32;
    let data = Data3DTexture::new(width, 1, 1, ng_render::image::volume_format(t), voxels(t, values));
    let volume = GpuVolume::from_texture(device, queue, &data);
    let make = |format, usage, label| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color = make(ng_render::compose::OFFSCREEN_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, "ng legend");
    let depth = make(ng_render::compose::DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT, "ng legend depth");
    let (color_view, depth_view) = (color.create_view(&Default::default()), depth.create_view(&Default::default()));
    let binding = pipeline.create_binding_with_lookup(device, &volume, lookup);
    // NDC x across the row's voxels, y and z at the voxel's middle.
    let half = width as f32 / 2.0;
    let mut e = [0.0f32; 16];
    e[0] = half;
    e[10] = 1.0;
    e[12] = half;
    e[13] = 0.5;
    e[14] = 0.5;
    e[15] = 1.0;
    pipeline.update(queue, &binding, &SliceParams { ndc_to_voxel: Matrix4 { elements: e }, params: [[0.0; 4]; 4] });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ng legend"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pipeline.draw_in_pass(&mut pass, threers::renderer::PixelRect { x: 0, y: 0, width, height: 1 }, &binding);
    }
    let padded_row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ng legend copy"),
        size: padded_row as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &color, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_row), rows_per_image: Some(1) } },
        wgpu::Extent3d { width, height: 1, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    let mapped = Arc::new(AtomicBool::new(false));
    let flag = mapped.clone();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        if r.is_ok() {
            flag.store(true, Ordering::SeqCst);
        }
    });
    LegendJob { key, buffer, mapped, width }
}

impl LegendJob {
    /// The colours, once the copy has mapped.
    pub(crate) fn take(&self) -> Option<Vec<[u8; 4]>> {
        if !self.mapped.load(Ordering::SeqCst) {
            return None;
        }
        let colors = {
            let data = self.buffer.slice(..).get_mapped_range().ok()?;
            data[..self.width as usize * 4].chunks_exact(4).map(|p| [p[0], p[1], p[2], p[3]]).collect()
        };
        self.buffer.unmap();
        Some(colors)
    }
}
