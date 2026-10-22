//! Segmentation layers in cross-sections: `SegmentationRenderLayer`
//! (`src/sliceview/volume/segmentation_renderlayer.ts`) with neuroglancer's
//! segment colours (`SegmentColorShaderManager`, `glsl_hashCombine`,
//! `glsl_hsvToRgb`).
//!
//! Each voxel's uint64 ID is hashed with the layer's seed to a hue and a
//! saturation; the colour is mixed toward white by `saturation`, at
//! `selectedAlpha` for visible segments and `notSelectedAlpha` otherwise. The
//! segment under the pointer (hover highlight) is lightened.
//!
//! The visible-segment set (`visibleSegments`) is a sorted table of IDs in a
//! lookup texture, searched per pixel ([`VisibleSet`]); neuroglancer uses a GPU
//! hash table, with the same answers. An empty set shows every segment
//! (`ignoreNullVisibleSet`).
//!
//! `segmentDefaultColor` replaces the hashed colour of every segment.
//!
//! Not yet: equivalences, stated colours, a highlight colour.

use ng_core::DataType;
use threers::volume::{SliceParams, VolumeFormat, VolumeSample};

/// `HAS_SELECTED_SEGMENT_FLAG`.
pub const HAS_SELECTED_SEGMENT: u32 = 1;
/// `SHOW_ALL_SEGMENTS_FLAG`.
pub const SHOW_ALL_SEGMENTS: u32 = 2;
/// A default colour is set (`segmentDefaultColor`).
pub const HAS_DEFAULT_COLOR: u32 = 4;

/// The `slice_color` WGSL (for `SliceRenderer::with_lookup`, the lookup
/// being the [`VisibleSet`]). `params[0]` = (seed low 16 bits, seed high 16
/// bits, selectedAlpha, notSelectedAlpha), `params[1]` = (saturation, flags,
/// hideSegmentZero, default red), `params[2]` = the selected segment's four
/// 16-bit parts, lowest first, `params[3]` = the visible set's size as two
/// 16-bit halves, default green and blue. 32-bit values travel as 16-bit halves: floats hold
/// those exactly, where a bit cast could lose NaN payloads on WebGL2.
pub const SEGMENTATION_WGSL: &str = r#"
fn ng_hash_combine(state_in : u32, value_in : u32) -> u32 {
    var value = value_in * 0xcc9e2d51u;
    value = (value << 15u) | (value >> 17u);
    value = value * 0x1b873593u;
    var state = state_in ^ value;
    state = (state << 13u) | (state >> 19u);
    return state * 5u + 0xe6546b64u;
}
fn ng_hue_to_rgb(hue : f32) -> vec3<f32> {
    let hue6 = hue * 6.0;
    let r = abs(hue6 - 3.0) - 1.0;
    let g = 2.0 - abs(hue6 - 2.0);
    let b = 2.0 - abs(hue6 - 4.0);
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}
fn ng_hsv_to_rgb(c : vec3<f32>) -> vec3<f32> {
    let hue_rgb = ng_hue_to_rgb(c.x);
    return c.z * ((hue_rgb - 1.0) * c.y + 1.0);
}
fn ng_segment_color_hash(seed : u32, x : vec2<u32>) -> vec3<f32> {
    var h = ng_hash_combine(ng_hash_combine(seed, x.x), x.y);
    var v : vec2<f32>;
    v.x = f32(h & 0xFFu) / 255.0;
    h = h >> 8u;
    v.y = f32(h & 0xFFu) / 255.0;
    let hsv = vec3<f32>(v.x, 0.5 + v.y * 0.5, 1.0);
    return ng_hsv_to_rgb(hsv);
}
fn ng_u32(lo : f32, hi : f32) -> u32 {
    return u32(lo) | (u32(hi) << 16u);
}
fn ng_visible_at(i : u32) -> vec2<u32> {
    let w = textureDimensions(slice_lookup).x;
    return textureLoad(slice_lookup, vec2<i32>(i32(i % w), i32(i / w)), 0).rg;
}
fn ng_visible(value : vec2<u32>, count : u32) -> bool {
    // Binary search of the sorted IDs (high word, then low).
    var lo = 0u;
    var hi = count;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        let m = ng_visible_at(mid);
        if (m.y < value.y || (m.y == value.y && m.x < value.x)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo < count && all(ng_visible_at(lo) == value);
}
fn slice_color(data : vec4<u32>, params : array<vec4<f32>, 4>) -> vec4<f32> {
    let value = data.rg;
    let seed = ng_u32(params[0].x, params[0].y);
    var alpha = params[0].z;
    var saturation = params[1].x;
    let flags = u32(params[1].y);
    let selected = vec2<u32>(ng_u32(params[2].x, params[2].y), ng_u32(params[2].z, params[2].w));
    if (params[1].z != 0.0 && value.x == 0u && value.y == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    var has = true;
    if ((flags & 2u) == 0u) {
        has = ng_visible(value, ng_u32(params[3].x, params[3].y));
    }
    if ((flags & 1u) != 0u && all(selected == value)) {
        let adjustment = select(0.75, 0.5, has);
        if (saturation > adjustment) {
            saturation -= adjustment;
        } else {
            saturation += adjustment;
        }
    } else if (!has) {
        alpha = params[0].w;
    }
    var rgba = vec4<f32>(ng_segment_color_hash(seed, value), 0.0);
    if ((flags & 4u) != 0u) {
        rgba = vec4<f32>(params[1].w, params[3].z, params[3].w, 0.0);
    }
    if (rgba.a > 0.0) {
        alpha = rgba.a;
    }
    return vec4<f32>(mix(vec3<f32>(1.0, 1.0, 1.0), rgba.rgb, saturation), alpha);
}
"#;

/// A segmentation layer's display settings (`SegmentationUserLayer`'s
/// defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationDisplay {
    pub color_seed: u32,
    pub selected_alpha: f32,
    pub not_selected_alpha: f32,
    pub saturation: f32,
    pub hide_segment_zero: bool,
    pub hover_highlight: bool,
    pub ignore_null_visible_set: bool,
    /// `segmentDefaultColor`: one colour for every segment.
    pub default_color: Option<[f32; 3]>,
}

impl Default for SegmentationDisplay {
    fn default() -> Self {
        Self {
            color_seed: 0,
            selected_alpha: 0.5,
            not_selected_alpha: 0.0,
            saturation: 1.0,
            hide_segment_zero: true,
            hover_highlight: true,
            ignore_null_visible_set: true,
            default_color: None,
        }
    }
}

impl SegmentationDisplay {
    /// The slice parameters, with the segment under the pointer and the
    /// number of visible segments (the [`VisibleSet`]'s length).
    pub fn slice_params(&self, ndc_to_voxel: threers::math::Matrix4, selected: Option<u64>, visible: usize) -> SliceParams {
        let mut flags = 0;
        let (mut lo, mut hi) = (0u32, 0u32);
        if let (Some(id), true) = (selected, self.hover_highlight) {
            lo = id as u32;
            hi = (id >> 32) as u32;
            flags |= HAS_SELECTED_SEGMENT;
        }
        if visible == 0 && self.ignore_null_visible_set {
            flags |= SHOW_ALL_SEGMENTS;
        }
        let [dr, dg, db] = self.default_color.unwrap_or([0.0; 3]);
        if self.default_color.is_some() {
            flags |= HAS_DEFAULT_COLOR;
        }
        let half = |v: u32| [(v & 0xffff) as f32, (v >> 16) as f32];
        let [s0, s1] = half(self.color_seed);
        let ([l0, l1], [h0, h1]) = (half(lo), half(hi));
        SliceParams {
            ndc_to_voxel,
            params: [
                [s0, s1, self.selected_alpha, self.not_selected_alpha],
                [self.saturation, flags as f32, self.hide_segment_zero as u32 as f32, dr],
                [l0, l1, h0, h1],
                {
                    let [c0, c1] = half(visible as u32);
                    [c0, c1, dg, db]
                },
            ],
        }
    }
}

/// Segmentation values go up as unsigned integers: 64-bit as `Rg32Uint`, the
/// rest as `R32Uint` (`toUint64`).
pub fn volume_format(t: DataType) -> VolumeFormat {
    match t {
        DataType::Uint64 => VolumeFormat::Rg32Uint,
        _ => VolumeFormat::R32Uint,
    }
}

/// Little-endian voxels of `t` as the bytes of [`volume_format`]. Signed
/// 8- and 16-bit values are sign-extended to 32 bits.
pub fn upload_bytes(t: DataType, data: &[u8]) -> Vec<u8> {
    let u32s = |f: &dyn Fn(&[u8]) -> u32, n: usize| data.chunks_exact(n).flat_map(|c| f(c).to_le_bytes()).collect();
    match t {
        DataType::Uint8 => u32s(&|c| c[0] as u32, 1),
        DataType::Int8 => u32s(&|c| c[0] as i8 as i32 as u32, 1),
        DataType::Uint16 => u32s(&|c| u16::from_le_bytes([c[0], c[1]]) as u32, 2),
        DataType::Int16 => u32s(&|c| i16::from_le_bytes([c[0], c[1]]) as i32 as u32, 2),
        DataType::Float32 => u32s(&|c| f32::from_le_bytes(c.try_into().unwrap()) as u32, 4),
        DataType::Uint32 | DataType::Int32 | DataType::Uint64 => data.to_vec(),
    }
}

pub fn sample() -> VolumeSample {
    VolumeSample::Uint
}

/// Width of the [`VisibleSet`] texture; it grows in rows.
const SET_WIDTH: u32 = 1024;

/// The visible segments on the GPU: sorted IDs in an `Rg32Uint` texture (low
/// word, high word), read by [`SEGMENTATION_WGSL`].
pub struct VisibleSet {
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    capacity: usize,
    len: usize,
    /// Changes when the texture is replaced (bindings must be remade).
    pub generation: u64,
}

impl VisibleSet {
    pub fn new(device: &wgpu::Device) -> Self {
        let (texture, view) = Self::allocate(device, 1);
        Self { texture, view, capacity: SET_WIDTH as usize, len: 0, generation: 0 }
    }

    fn allocate(device: &wgpu::Device, rows: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ng visible segments"),
            size: wgpu::Extent3d { width: SET_WIDTH, height: rows, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Replaces the set's contents.
    pub fn set(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, ids: &[u64]) {
        let mut ids = ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() > self.capacity {
            let rows = (ids.len() as u32).div_ceil(SET_WIDTH).next_power_of_two();
            (self.texture, self.view) = Self::allocate(device, rows);
            self.capacity = (rows * SET_WIDTH) as usize;
            self.generation += 1;
        }
        self.len = ids.len();
        if ids.is_empty() {
            return;
        }
        let rows = (ids.len() as u32).div_ceil(SET_WIDTH);
        let mut data = vec![0u32; (rows * SET_WIDTH * 2) as usize];
        for (i, id) in ids.iter().enumerate() {
            data[2 * i] = *id as u32;
            data[2 * i + 1] = (id >> 32) as u32;
        }
        queue.write_texture(
            self.texture.as_image_copy(),
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(SET_WIDTH * 8), rows_per_image: None },
            wgpu::Extent3d { width: SET_WIDTH, height: rows, depth_or_array_layers: 1 },
        );
    }
}
