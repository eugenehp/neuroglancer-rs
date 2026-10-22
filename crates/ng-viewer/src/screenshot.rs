//! Screenshots (`util/screenshot_manager.ts`): the data panels rendered at a
//! scale, once their chunks have loaded, cropped to the panels' viewports
//! and encoded as PNG.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// `MAX_RENDER_AREA_PIXELS`.
pub const MAX_RENDER_AREA_PIXELS: f64 = 5100.0 * 5100.0;

/// What a screenshot is waiting for.
pub(crate) enum Stage {
    /// Rendering, until no chunk is loading.
    Render,
    /// The pixels being copied back.
    Mapping { buffer: wgpu::Buffer, mapped: Arc<AtomicBool>, padded_row: u32 },
}

pub(crate) struct Job {
    /// Scale of the render over the view.
    pub scale: f64,
    /// Whether the slice views keep their field of view (the render has
    /// more pixels per voxel) or show more around it.
    pub keep_fov: bool,
    pub name: String,
    pub texture: wgpu::Texture,
    pub size: (u32, u32),
    pub stage: Stage,
    /// Frames rendered so far.
    pub frames: u32,
}

/// A finished screenshot.
#[derive(Debug, Clone)]
pub struct Screenshot {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub png: Vec<u8>,
}

/// `setExtension`: the name with `.png` (replacing any other extension).
pub fn png_name(name: &str) -> String {
    if name.ends_with(".png") {
        return name.to_owned();
    }
    match name.rfind('.') {
        Some(dot) => format!("{}.png", &name[..dot]),
        None => format!("{name}.png"),
    }
}

/// The default name: `neuroglancer-screenshot-w{w}px-h{h}px-at-{time}`.
pub fn default_name(width: u32, height: u32, time: &str) -> String {
    format!("neuroglancer-screenshot-w{width}px-h{height}px-at-{time}")
}

/// The scale a screenshot may use: at most [`MAX_RENDER_AREA_PIXELS`] (`handleScreenshotZoomAndResize`).
pub fn clamp_scale(scale: f64, width: u32, height: u32) -> f64 {
    let pixels = width as f64 * height as f64;
    if scale > 1.0 && pixels * scale * scale > MAX_RENDER_AREA_PIXELS {
        (MAX_RENDER_AREA_PIXELS / pixels).sqrt()
    } else {
        scale
    }
}

/// RGBA rows (`width` × `height`) of a mapped copy whose rows are
/// `padded_row` bytes, from `region` (x, y, w, h); BGRA swapped.
pub(crate) fn crop(data: &[u8], padded_row: u32, region: (u32, u32, u32, u32), bgra: bool) -> Vec<u8> {
    let (x, y, w, h) = region;
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in y..y + h {
        let start = (row * padded_row + x * 4) as usize;
        for px in data[start..start + (w * 4) as usize].chunks_exact(4) {
            if bgra {
                out.extend_from_slice(&[px[2], px[1], px[0], 255]);
            } else {
                out.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
    }
    out
}

/// PNG bytes of RGBA pixels.
pub fn encode_png(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![];
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        if let Ok(mut writer) = encoder.write_header() {
            let _ = writer.write_image_data(rgba);
        }
    }
    out
}

/// Starts mapping `texture`'s copy into a new buffer.
pub(crate) fn start_copy(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, size: (u32, u32)) -> Stage {
    let padded_row = (size.0 * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot"),
        size: (padded_row * size.1) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_row), rows_per_image: Some(size.1) } },
        wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    let mapped = Arc::new(AtomicBool::new(false));
    let flag = mapped.clone();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        if r.is_ok() {
            flag.store(true, Ordering::SeqCst);
        }
    });
    Stage::Mapping { buffer, mapped, padded_row }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_scales_as_neuroglancer() {
        assert_eq!(png_name("a.jpg"), "a.png");
        assert_eq!(png_name("shot"), "shot.png");
        assert_eq!(png_name("x.png"), "x.png");
        assert_eq!(default_name(1196, 750, "t"), "neuroglancer-screenshot-w1196px-h750px-at-t");
        assert_eq!(clamp_scale(4.0, 1196, 750), 4.0);
        assert!((clamp_scale(8.0, 1196, 750) - (MAX_RENDER_AREA_PIXELS / (1196.0 * 750.0)).sqrt()).abs() < 1e-9);
        let data = [1u8, 2, 3, 9, 4, 5, 6, 9, 0, 0, 0, 0, 7, 8, 9, 9];
        assert_eq!(crop(&data, 8, (1, 0, 1, 2), true), vec![6, 5, 4, 255, 9, 8, 7, 255]);
        let png = encode_png(&[255, 0, 0, 255], 1, 1);
        assert_eq!(&png[1..4], b"PNG");
    }
}
