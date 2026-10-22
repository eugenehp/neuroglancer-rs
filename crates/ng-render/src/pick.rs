//! Picking in the 3D panel (`rendered_data_panel.ts`,
//! `rendered_data_panel_picking.ts`): a window of the Z and pick buffers
//! around the pointer copied out after a frame, read once the GPU is done
//! with it, and its nearest drawn sample.

use std::sync::Arc;

use threers::renderer::readback::AsyncReadback;

/// `getPickDiameter`.
pub fn pick_diameter(radius: u32) -> u32 {
    1 + 2 * radius
}

/// `getPickOffsetSequence`: offsets into the C-order window within
/// `radius` of its centre, nearest first (ties in column-major order, as
/// the stable sort leaves them).
pub fn pick_offset_sequence(radius: u32) -> Vec<u32> {
    let d = pick_diameter(radius);
    let r = radius as i64;
    let dist2 = |x: u32, y: u32| (x as i64 - r).pow(2) + (y as i64 - r).pow(2);
    let mut offsets = vec![];
    for x in 0..d {
        for y in 0..d {
            if dist2(x, y) <= r * r {
                offsets.push(y * d + x);
            }
        }
    }
    offsets.sort_by_key(|&o| dist2(o % d, o / d));
    offsets
}

/// `ResolvedPanelPickSample`: the window position, Z (`1 − depth`) and
/// pick ID of the nearest sample something was drawn at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PickSample {
    pub relative_x: u32,
    pub relative_y: u32,
    pub depth: f32,
    pub pick: u32,
}

/// `resolveNearestPanelPickSample` with depth: the first sample in offset
/// order with a non-zero Z.
pub fn resolve_nearest(z: &[f32], pick: &[u32], radius: u32) -> Option<PickSample> {
    let d = pick_diameter(radius);
    pick_offset_sequence(radius).into_iter().find_map(|o| {
        let depth = *z.get(o as usize)?;
        (depth != 0.0).then(|| PickSample { relative_x: o % d, relative_y: o / d, depth, pick: pick.get(o as usize).copied().unwrap_or(0) })
    })
}

/// A pick window being read back: the request's centre and radius, and
/// its buffer (Z rows, then pick rows; outside the panel zero, as
/// `clearOutOfBoundsPickData` leaves it).
pub struct PickReadback {
    pub x: i64,
    pub y: i64,
    pub radius: u32,
    /// The panel's size when drawn.
    pub size: (u32, u32),
    readback: AsyncReadback,
    bytes_per_row: u32,
}

impl PickReadback {
    /// Copies the window around `(x, y)` of `z` and `pick` (`size`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, z: &wgpu::Texture, pick: &wgpu::Texture, size: (u32, u32), x: i64, y: i64, radius: u32) -> Self {
        let d = pick_diameter(radius);
        let bytes_per_row = (d * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let image = (bytes_per_row * d) as u64;
        let readback = AsyncReadback::new(device, 2 * image, "ng pick window");
        // The part of the window inside the panel.
        let (x0, y0) = (x - radius as i64, y - radius as i64);
        let (cx0, cy0) = (x0.max(0), y0.max(0));
        let (cx1, cy1) = ((x0 + d as i64).min(size.0 as i64), (y0 + d as i64).min(size.1 as i64));
        if cx1 > cx0 && cy1 > cy0 {
            for (k, texture) in [z, pick].into_iter().enumerate() {
                let offset = k as u64 * image + (cy0 - y0) as u64 * bytes_per_row as u64 + (cx0 - x0) as u64 * 4;
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d { x: cx0 as u32, y: cy0 as u32, z: 0 }, aspect: wgpu::TextureAspect::All },
                    wgpu::TexelCopyBufferInfo { buffer: readback.buffer(), layout: wgpu::TexelCopyBufferLayout { offset, bytes_per_row: Some(bytes_per_row), rows_per_image: Some(d) } },
                    wgpu::Extent3d { width: (cx1 - cx0) as u32, height: (cy1 - cy0) as u32, depth_or_array_layers: 1 },
                );
            }
        }
        Self { x, y, radius, size, readback, bytes_per_row }
    }

    /// Once the copy is submitted: maps the buffer, calling `wake` when it
    /// can be read.
    pub fn map(&self, wake: Arc<dyn Fn() + Send + Sync>) {
        self.readback.map(wake);
    }

    /// The window's Z and pick values, once mapped.
    pub fn read(&self) -> Option<(Vec<f32>, Vec<u32>)> {
        let data = self.readback.read()?;
        let d = pick_diameter(self.radius) as usize;
        let bpr = self.bytes_per_row as usize;
        let row = |image: usize, y: usize| &data[image * bpr * d + y * bpr..][..d * 4];
        let mut z = Vec::with_capacity(d * d);
        let mut pick = Vec::with_capacity(d * d);
        for y in 0..d {
            z.extend(row(0, y).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
            pick.extend(row(1, y).chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
        }
        Some((z, pick))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_samples_first() {
        let seq = pick_offset_sequence(1);
        // The centre, then the four edge neighbours in column-major order;
        // the corners are beyond the radius.
        assert_eq!(seq, vec![4, 3, 1, 7, 5]);
        let mut z = vec![0.0; 9];
        let mut pick = vec![0; 9];
        z[5] = 0.25;
        pick[5] = 7;
        z[2] = 0.5;
        assert_eq!(resolve_nearest(&z, &pick, 1), Some(PickSample { relative_x: 2, relative_y: 1, depth: 0.25, pick: 7 }));
        assert_eq!(resolve_nearest(&[0.0; 9], &[0; 9], 1), None);
    }
}
