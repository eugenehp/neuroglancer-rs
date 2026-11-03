//! Data histograms (`webgl/empirical_cdf.ts`): what an invlerp control's
//! CDF panel plots. Neuroglancer writes each slice pixel's value, through
//! the control's window, to a second render target and adds up 2¹⁴ hashed
//! sample points of it; here the same points look the value up in the
//! loaded chunks.

use std::sync::OnceLock;

use ng_core::data_type::DataType;

/// `histogramSamples`.
pub const SAMPLES: usize = 1 << 14;
/// `histogramSamplesPerInstance`.
const PER_INSTANCE: usize = 4096;
/// `HISTOGRAM_SAMPLES_PER_INSTANCE` of volume rendering.
const PER_INSTANCE_3D: usize = 256;

/// `simpleFloatHash`, in single precision.
pub fn simple_float_hash(x: f32, y: f32) -> f32 {
    let v = (x * 12.9898 + y * 78.233).sin() * 43758.547;
    v - v.floor()
}

/// `simpleFloatHash` as the GPU computes it: neuroglancer hashes in its
/// shaders, and `fract(sin(…) · 43758.5453)` turns the last bits of the
/// GPU's `sin` into different points than the CPU's (WebGPU's `sin` gives
/// WebGL's bits; the CPU's matches under half of them). The slice
/// histogram's points, then the 3D one's, once computed.
struct GpuHashes {
    points: Vec<(f32, f32)>,
    points_3d: Vec<[f32; 3]>,
}

static GPU_HASHES: OnceLock<GpuHashes> = OnceLock::new();

/// Volume rendering's instances (at most `SAMPLES_3D / PER_INSTANCE_3D`).
const INSTANCES_3D: usize = 64;

const HASH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> out : array<f32>;
fn h(x : f32, y : f32) -> f32 {
    return fract(sin(dot(vec2<f32>(x, y), vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let k = id.x;
    if (k < 16384u) {
        let v = f32(k % 4096u);
        let i = f32(k / 4096u);
        out[2u * k] = h(v, i);
        out[2u * k + 1u] = h(v + 10.0, 5.0 + i);
    } else if (k < 16384u + 256u * 64u) {
        let j = k - 16384u;
        let v = f32(j % 256u);
        let i = f32(j / 256u);
        let o = 2u * 16384u + 3u * j;
        out[o] = h(v, i);
        out[o + 1u] = h(v + 10.0, 5.0 + i);
        out[o + 2u] = h(v + 20.0, 15.0 + i);
    }
}
"#;

/// The GPU's hashes being read back.
pub struct HashReadback(threers::renderer::readback::AsyncReadback);

impl HashReadback {
    /// Starts the hashes on `device`, `wake` called once they can be read.
    pub fn start(device: &wgpu::Device, queue: &wgpu::Queue, wake: std::sync::Arc<dyn Fn() + Send + Sync>) -> Self {
        let floats = 2 * SAMPLES + 3 * PER_INSTANCE_3D * INSTANCES_3D;
        let size = (floats * 4) as u64;
        let out = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ng hashes"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let readback = threers::renderer::readback::AsyncReadback::new(device, size, "ng hashes readback");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ng hashes"), source: wgpu::ShaderSource::Wgsl(HASH_SHADER.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("ng hashes"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("ng hashes"), layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: out.as_entire_binding() }] });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(((SAMPLES + PER_INSTANCE_3D * INSTANCES_3D) as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, readback.buffer(), 0, size);
        queue.submit([encoder.finish()]);
        readback.map(wake);
        Self(readback)
    }

    /// Takes the hashes in once read; whether they are now in use.
    pub fn poll(&self) -> bool {
        let Some(data) = self.0.read() else { return false };
        let f: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let points = f[..2 * SAMPLES].chunks_exact(2).map(|c| (c[0], c[1])).collect();
        let points_3d = f[2 * SAMPLES..].chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let _ = GPU_HASHES.set(GpuHashes { points, points_3d });
        true
    }
}

/// Where the histogram samples the panel: fractions of its width and
/// height, as `TextureHistogramGenerator`'s vertex shader picks them (input
/// 0, seed 0, vertex and instance IDs): the GPU's hashes once they are in,
/// the CPU's meanwhile.
pub fn sample_points() -> &'static [(f32, f32)] {
    if let Some(g) = GPU_HASHES.get() {
        return &g.points;
    }
    static POINTS: OnceLock<Vec<(f32, f32)>> = OnceLock::new();
    POINTS.get_or_init(|| {
        (0..SAMPLES / PER_INSTANCE)
            .flat_map(|instance| {
                (0..PER_INSTANCE).map(move |vertex| {
                    let (v, i) = (vertex as f32, instance as f32);
                    (simple_float_hash(v, i), simple_float_hash(v + 10.0, 5.0 + i))
                })
            })
            .collect()
    })
}

/// The bin of a value whose position in the window is `x`: 0 below, 255
/// above, 1–254 across it (`(1 + x·253) / 255`, stored as 8 bits).
pub fn bin(x: f64) -> usize {
    if x.is_nan() || x < 0.0 {
        0
    } else if x > 1.0 {
        255
    } else {
        (1.0 + x * 253.0).round() as usize
    }
}

/// The empirical CDF's vertices (`createCDFLineShader`): vertex `i` at
/// `x` (its fraction of the panel) and the fraction of samples in bins
/// `0..=i` (the last vertex leaving out bin 255, as the shader does).
pub fn cdf_vertices(counts: &[u32; 256], bounds_fraction: f64) -> Vec<(f64, f64)> {
    let total = (counts.iter().map(|&c| c as f64).sum::<f64>()).max(1.0);
    let mut sum = 0.0;
    let mut out = Vec::with_capacity(256);
    for (i, &count) in counts.iter().enumerate().take(255) {
        sum += count as f64;
        let x = if i == 0 { 0.0 } else { i as f64 / 254.0 * bounds_fraction };
        out.push((x, sum / total));
    }
    out.push((1.0, sum / total));
    out
}

/// `BIN_SIZE_MULTIPLIER_FOR_WINDOW`.
const WINDOW_BINS: f64 = 64.0;
/// `MAX_AUTO_RANGE_ITERATIONS`.
const MAX_ITERATIONS: usize = 16;
/// `FLOAT_EQUAL_TOLERANCE`.
const FLOAT_TOLERANCE: f64 = 1e-3;

/// Counts of `values` in the 256 bins of `window`.
pub fn bin_values(values: &[f64], window: [f64; 2]) -> [u32; 256] {
    let mut counts = [0u32; 256];
    for &v in values {
        counts[bin((v - window[0]) / (window[1] - window[0]))] += 1;
    }
    counts
}

/// `adjustBound`: `bound` moved by `change` (rounded, and kept in the type's
/// range, for integer types).
fn adjust_bound(bound: f64, t: DataType, change: f64, increase: bool, type_range: [f64; 2]) -> f64 {
    if t != DataType::Float32 && bound == if increase { type_range[1] } else { type_range[0] } {
        return bound;
    }
    let delta = if t == DataType::Float32 { change } else { ng_core::js::round(change) };
    let v = bound + if increase { delta } else { -delta };
    if t == DataType::Float32 {
        v
    } else {
        v.clamp(type_range[0], type_range[1])
    }
}

/// `computePercentilesFromEmpiricalHistogram`: the range between the
/// percentiles as the histogram over `range` resolves them, and a window a
/// little wider.
pub fn percentiles(counts: &[u32; 256], lower: f64, upper: f64, range: [f64; 2], t: DataType, type_range: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    let total: f64 = counts.iter().map(|&c| c as f64).sum();
    if total == 0.0 {
        return (range, range);
    }
    let mut sum = 0.0;
    let cdf: Vec<f64> = counts.iter().map(|&c| {
        sum += c as f64;
        sum / total
    }).collect();
    let bin_size = (range[1] - range[0]) / 254.0;
    let mut lower_index = 0;
    for (i, &c) in cdf.iter().enumerate() {
        lower_index = i;
        if c > lower {
            break;
        }
    }
    let upper_index = cdf.iter().position(|&c| c >= upper).unwrap_or(255);
    let (mut lo, mut hi) = (range[0], range[1]);
    if lower_index == 0 {
        let mut shift = bin_size / 2.0;
        if t == DataType::Float32 {
            shift = shift.max(1f64.max((lo / 2.0).abs()));
        }
        lo = adjust_bound(lo, t, shift, false, type_range);
    } else {
        lo = adjust_bound(lo, t, bin_size * (lower_index - 1) as f64, true, type_range);
    }
    if upper_index == 255 {
        let mut shift = bin_size / 2.0;
        if t == DataType::Float32 {
            shift = shift.max(1f64.max((hi / 2.0).abs()));
        }
        hi = adjust_bound(hi, t, shift, true, type_range);
    } else {
        hi = adjust_bound(hi, t, bin_size * (254 - upper_index) as f64, false, type_range);
    }
    let mut expand = bin_size * WINDOW_BINS;
    if t != DataType::Float32 {
        expand = expand.max(1.0);
    }
    let window = [adjust_bound(lo, t, expand, false, type_range), adjust_bound(hi, t, expand, true, type_range)];
    ([lo, hi], window)
}

/// `AutoRangeFinder.autoComputeRange`, its frames run at once over the same
/// samples: from the widest range, each histogram's percentiles narrow it,
/// until a range repeats (within 10⁻³ for floats), its ends meet, or 16
/// steps pass. Returns the range (reversed if `inverted`) and the window.
pub fn auto_range(values: &[f64], lower: f64, upper: f64, t: DataType, type_range: [f64; 2], inverted: bool) -> ([f64; 2], [f64; 2]) {
    let mut range = if t == DataType::Float32 { [-65536.0, 65536.0] } else { type_range };
    let mut previous: Vec<[f64; 2]> = vec![];
    loop {
        let (new_range, new_window) = percentiles(&bin_values(values, range), lower, upper, range, t, type_range);
        let found = previous.iter().any(|p| {
            if t == DataType::Float32 {
                (p[0] - new_range[0]).abs() < FLOAT_TOLERANCE && (p[1] - new_range[1]).abs() < FLOAT_TOLERANCE
            } else {
                *p == new_range
            }
        });
        let exceeded = previous.len() > MAX_ITERATIONS;
        previous.push(new_range);
        if found || exceeded || new_range[0] == new_range[1] {
            let mut r = new_range;
            if inverted {
                r.reverse();
            }
            // `ensureWindowBoundsNotEqual` applies only without a range.
            return (r, new_window);
        }
        range = new_range;
    }
}

/// The volume rendering histogram's samples (`histogramShaderGetter`): per
/// chunk of display size `size` (of `total` voxels drawn), instances of 256
/// points at hashed positions within it.
pub fn volume_sample_points(size: [u32; 3], total: f64) -> Vec<[u32; 3]> {
    // `NUM_HISTOGRAM_SAMPLES`, `HISTOGRAM_SAMPLES_PER_INSTANCE`.
    const SAMPLES_3D: f64 = 16384.0;
    const PER_INSTANCE_3D: u32 = self::PER_INSTANCE_3D as u32;
    let volume = size.iter().map(|&s| s as f64).product::<f64>();
    let wanted = SAMPLES_3D * (volume / total);
    let instances = ((wanted.min(volume / 2.0) / PER_INSTANCE_3D as f64).round() as u32).max(1);
    let mut out = Vec::with_capacity((instances * PER_INSTANCE_3D) as usize);
    for instance in 0..instances {
        for vertex in 0..PER_INSTANCE_3D {
            let (v, i) = (vertex as f32, instance as f32);
            let gpu = GPU_HASHES.get().and_then(|g| g.points_3d.get(instance as usize * PER_INSTANCE_3D as usize + vertex as usize));
            let r = gpu.copied().unwrap_or_else(|| [simple_float_hash(v, i), simple_float_hash(v + 10.0, 5.0 + i), simple_float_hash(v + 20.0, 15.0 + i)]);
            // `floor(rand · (size − 1))`, clamped to the chunk.
            out.push(std::array::from_fn(|k| ((r[k] * (size[k] as f32 - 1.0)).floor().max(0.0) as u32).min(size[k].saturating_sub(1))));
        }
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_and_samples_as_neuroglancer() {
        assert_eq!((bin(-0.1), bin(0.0), bin(1.0), bin(1.01), bin(f64::NAN)), (0, 1, 254, 255, 0));
        let p = sample_points();
        assert_eq!(p.len(), SAMPLES);
        assert!(p.iter().all(|&(x, y)| (0.0..1.0).contains(&x) && (0.0..1.0).contains(&y)));
        // Spread over the panel.
        let left = p.iter().filter(|(x, _)| *x < 0.5).count();
        assert!((7000..9400).contains(&left), "{left}");
        let mut counts = [0u32; 256];
        counts[1] = 1;
        counts[254] = 3;
        let v = cdf_vertices(&counts, 1.0);
        assert_eq!((v[0], v[1], v[253], v[254], v[255]), ((0.0, 0.0), (1.0 / 254.0, 0.25), (253.0 / 254.0, 0.25), (1.0, 1.0), (1.0, 1.0)));
    }

    #[test]
    fn auto_ranges_converge_on_the_percentiles() {
        let values: Vec<f64> = (0..10000).map(|i| (i % 200) as f64 + 20.0).collect();
        let (range, window) = auto_range(&values, 0.0, 1.0, DataType::Uint8, [0.0, 255.0], false);
        assert_eq!(range, [20.0, 219.0]);
        assert_eq!((window[0], window[1]), (0.0, 255.0));
        let (range, _) = auto_range(&values, 0.05, 0.95, DataType::Uint8, [0.0, 255.0], true);
        assert!(range[0] > range[1] && (range[1] - 30.0).abs() <= 2.0 && (range[0] - 209.0).abs() <= 2.0, "{range:?}");
        let floats: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let (range, _) = auto_range(&floats, 0.0, 1.0, DataType::Float32, [0.0, 1.0], false);
        assert!(range[0] <= 0.0 && range[0] > -0.01 && range[1] >= 0.999 && range[1] < 1.01, "{range:?}");
    }
}
