//! Image layers with neuroglancer's default shader:
//!
//! ```glsl
//! #uicontrol invlerp normalized
//! void main() { emitGrayscale(normalized()); }
//! ```
//!
//! `normalized()` maps the control's range (default: `defaultDataTypeRange`,
//! `src/util/lerp.ts`) to [0, 1]. With one layer, the output shows as that
//! grey: the first layer is drawn unblended, and the background only shows
//! where nothing was drawn (`SliceViewRenderHelper`).
//!
//! How each data type reaches the GPU, read back exactly:
//! - 8- and 16-bit integers and float32 as `R32Float` (every value exact).
//! - 32-bit integers as `R32Uint` (signed ones reinterpreted in the shader).
//! - uint64 as `Rg32Uint`, low word first.

use ng_core::DataType;
use threers::volume::{SliceParams, VolumeFormat, VolumeSample};

/// An invlerp range, as neuroglancer holds it (`DataTypeInterval`): numbers,
/// or exact 64-bit integers for uint64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Interval {
    Number([f64; 2]),
    Uint64([u64; 2]),
}

impl Interval {
    /// The bounds as numbers (for display and checks).
    pub fn as_f64(&self) -> [f64; 2] {
        match *self {
            Interval::Number(r) => r,
            Interval::Uint64([lo, hi]) => [lo as f64, hi as f64],
        }
    }
}

/// `defaultDataTypeRange`.
pub fn default_data_type_range(t: DataType) -> Interval {
    Interval::Number(match t {
        DataType::Uint8 => [0.0, 255.0],
        DataType::Int8 => [-128.0, 127.0],
        DataType::Uint16 => [0.0, 65535.0],
        DataType::Int16 => [-32768.0, 32767.0],
        DataType::Uint32 => [0.0, 4294967295.0],
        DataType::Int32 => [-2147483648.0, 2147483647.0],
        DataType::Uint64 => return Interval::Uint64([0, u64::MAX]),
        DataType::Float32 => [0.0, 1.0],
    })
}

/// The volume format a data type is uploaded as.
pub fn volume_format(t: DataType) -> VolumeFormat {
    match t {
        DataType::Uint8 | DataType::Int8 | DataType::Uint16 | DataType::Int16 | DataType::Float32 => {
            VolumeFormat::R32Float
        }
        DataType::Uint32 | DataType::Int32 => VolumeFormat::R32Uint,
        DataType::Uint64 => VolumeFormat::Rg32Uint,
    }
}

/// Converts little-endian voxels of `t` to the bytes of [`volume_format`].
pub fn upload_bytes(t: DataType, data: &[u8]) -> Vec<u8> {
    let to_f32 = |v: f32| v.to_le_bytes();
    match t {
        DataType::Uint8 => data.iter().flat_map(|&b| to_f32(b as f32)).collect(),
        DataType::Int8 => data.iter().flat_map(|&b| to_f32(b as i8 as f32)).collect(),
        DataType::Uint16 => data
            .chunks_exact(2)
            .flat_map(|c| to_f32(u16::from_le_bytes([c[0], c[1]]) as f32))
            .collect(),
        DataType::Int16 => data
            .chunks_exact(2)
            .flat_map(|c| to_f32(i16::from_le_bytes([c[0], c[1]]) as f32))
            .collect(),
        // Already in the uploaded layout.
        DataType::Float32 | DataType::Uint32 | DataType::Int32 | DataType::Uint64 => data.to_vec(),
    }
}

/// `computeInvlerp` per data type (`src/webgl/lerp.ts`), then
/// `emitGrayscale` at the layer's opacity. Parameters, from [`slice_params`]:
///
/// - 8/16-bit and float32: `(v - params[0].x) * params[0].y`.
/// - 32-bit: `(|v - offset| >> shift) * ±params[0].y`, exact in integers.
/// - uint64: the same on 64 bits, keeping the low word after the shift.
///
/// `params[0].z` is the opacity, `params[0].w` the shift, and `params[1]` the
/// integer offset as four 16-bit parts, lowest first.
///
/// A NaN (a float voxel that is NaN, or `0 · ∞` when the range is empty)
/// becomes 0: GLSL defines `clamp(x, lo, hi)` as `min(max(x, lo), hi)`, which
/// gives `lo` for NaN, where WGSL leaves it unspecified. It is caught by its
/// bits, which survive fast-math compilation.
pub fn color_wgsl(t: DataType) -> (VolumeSample, String) {
    let (sample, scalar, invlerp) = match t {
        DataType::Uint8 | DataType::Int8 | DataType::Uint16 | DataType::Int16 | DataType::Float32 => {
            (VolumeSample::Float, "f32", "return (value.r - params[0].x) * params[0].y;")
        }
        DataType::Uint32 => (
            VolumeSample::Uint,
            "u32",
            r#"let v = value.r;
    let offset = ng_u32(params[1].x, params[1].y);
    var multiplier = params[0].y;
    var x : u32;
    if (v >= offset) {
        x = v - offset;
    } else {
        x = offset - v;
        multiplier = -multiplier;
    }
    x = x >> u32(params[0].w);
    return f32(x) * multiplier;"#,
        ),
        DataType::Int32 => (
            VolumeSample::Uint,
            "u32",
            r#"let v = bitcast<i32>(value.r);
    let offset_bits = ng_u32(params[1].x, params[1].y);
    let offset = bitcast<i32>(offset_bits);
    var multiplier = params[0].y;
    var x : u32;
    // `uint(v - offset)`: the difference, wrapped to 32 bits.
    if (v >= offset) {
        x = value.r - offset_bits;
    } else {
        x = offset_bits - value.r;
        multiplier = -multiplier;
    }
    x = x >> u32(params[0].w);
    return f32(x) * multiplier;"#,
        ),
        DataType::Uint64 => (
            VolumeSample::Uint,
            "u32",
            r#"var v = value.rg;
    let offset = vec2<u32>(ng_u32(params[1].x, params[1].y), ng_u32(params[1].z, params[1].w));
    var multiplier = params[0].y;
    var a = v;
    var b = offset;
    if (v.y < offset.y || (v.y == offset.y && v.x < offset.x)) {
        a = offset;
        b = v;
        multiplier = -multiplier;
    }
    // `subtract`.
    var d = a;
    if (a.x < b.x) {
        d.y = d.y - 1u;
    }
    d = d - b;
    // `shiftRight(...).value[0]`.
    let shift = u32(params[0].w);
    var low : u32;
    if (shift >= 32u) {
        low = d.y >> (shift - 32u);
    } else if (shift == 0u) {
        low = d.x;
    } else {
        low = (d.x >> shift) | (d.y << (32u - shift));
    }
    return f32(low) * multiplier;"#,
        ),
    };
    let wgsl = format!(
        r#"
fn ng_u32(lo : f32, hi : f32) -> u32 {{
    return u32(lo) | (u32(hi) << 16u);
}}
fn ng_invlerp(value : vec4<{scalar}>, params : array<vec4<f32>, 4>) -> f32 {{
    {invlerp}
}}
fn slice_color(value : vec4<{scalar}>, params : array<vec4<f32>, 4>) -> vec4<f32> {{
    var v = ng_invlerp(value, params);
    if ((bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u) {{
        v = 0.0;
    }}
    let g = clamp(v, 0.0, 1.0);
    return vec4<f32>(g, g, g, params[0].z);
}}
"#
    );
    (sample, wgsl)
}

/// The image layer's default `opacity`.
pub const DEFAULT_OPACITY: f32 = 0.5;

/// `enableLerpShaderFunction`: the uniforms for `range` over `t`, as
/// `[[lower or 0, scalar, 0, shift], offset as four 16-bit parts]`.
pub fn lerp_params(t: DataType, range: Interval) -> [[f32; 4]; 2] {
    let halves = |v: u64| [(v & 0xffff) as f32, ((v >> 16) & 0xffff) as f32, ((v >> 32) & 0xffff) as f32, (v >> 48) as f32];
    match (t, range) {
        (DataType::Uint32 | DataType::Int32, Interval::Number([lower, upper])) => {
            let diff = upper - lower;
            let shift = (diff.abs().log2().ceil() - 24.0).max(0.0);
            let scalar = 2f64.powf(shift) / diff;
            // `uniform2ui` / `uniform2i` of the lower bound: the same bits.
            let offset = ng_core::js::to_uint32(lower);
            [[0.0, scalar as f32, 0.0, shift as f32], halves(offset as u64)]
        }
        (DataType::Uint64, Interval::Uint64([lower, upper])) => {
            let diff = lower.abs_diff(upper);
            let num_bits = (diff as f64).log2().ceil();
            let shift = (num_bits - 24.0).max(0.0);
            let shifted = diff.checked_shr(shift as u32).unwrap_or(0);
            let mut scalar = 1.0 / shifted as f64;
            if lower > upper {
                scalar *= -1.0;
            }
            [[0.0, scalar as f32, 0.0, shift as f32], halves(lower)]
        }
        (_, range) => {
            let [lower, upper] = range.as_f64();
            [[lower as f32, (1.0 / (upper - lower)) as f32, 0.0, 0.0], [0.0; 4]]
        }
    }
}

/// Slice parameters for invlerp over `range` at `opacity`.
pub fn slice_params(ndc_to_voxel: threers::math::Matrix4, t: DataType, range: Interval, opacity: f32) -> SliceParams {
    let [mut p0, p1] = lerp_params(t, range);
    p0[2] = opacity;
    SliceParams {
        ndc_to_voxel,
        params: [p0, p1, [0.0; 4], [0.0; 4]],
    }
}

/// The grey the GPU computes for `value`, in [0, 1], by the same
/// arithmetic (for checking renders).
pub fn expected_gray(t: DataType, range: Interval, value: f64) -> f64 {
    let [p0, p1] = lerp_params(t, range);
    let v = match t {
        DataType::Uint32 | DataType::Int32 => {
            let offset = (p1[0] as u64 | (p1[1] as u64) << 16) as u32;
            let offset = if t == DataType::Int32 { offset as i32 as i64 } else { offset as i64 };
            let (x, m) = if value as i64 >= offset {
                ((value as i64 - offset) as u64, p0[1])
            } else {
                ((offset - value as i64) as u64, -p0[1])
            };
            ((x as u32) >> p0[3] as u32) as f32 * m
        }
        DataType::Uint64 => {
            let offset = p1[0] as u64 | (p1[1] as u64) << 16 | (p1[2] as u64) << 32 | (p1[3] as u64) << 48;
            let v = value as u64;
            let (d, m) = if v >= offset { (v - offset, p0[1]) } else { (offset - v, -p0[1]) };
            (d >> p0[3] as u32) as u32 as f32 * m
        }
        _ => (value as f32 - p0[0]) * p0[1],
    };
    if v.is_nan() {
        0.0
    } else {
        v.clamp(0.0, 1.0) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_keeps_values_exact() {
        let bytes = upload_bytes(DataType::Int16, &(-32768i16).to_le_bytes());
        assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()), -32768.0);
        let bytes = upload_bytes(DataType::Int8, &[0xff]);
        assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()), -1.0);
    }
}
