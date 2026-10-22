//! Prefetching for slice views: a port of neuroglancer's
//! `VelocityEstimator` (`util/velocity_estimation.ts`) and
//! `getPrefetchChunkOffsets` (`sliceview/backend.ts`). While the position
//! moves, the chunks it is likely to reach within two seconds load ahead of
//! it, behind every visible chunk.

const VELOCITY_HALF_LIFE_MS: f64 = 50.0;
const MODEL_HALF_LIFE_MS: f64 = 1000.0;

/// An exponentially weighted estimate of each global dimension's velocity:
/// its mean and variance (`VelocityEstimator`).
#[derive(Debug, Clone, Default)]
pub struct VelocityEstimator {
    last_time: f64,
    num_samples: u64,
    prev_position: Vec<f32>,
    velocity: Vec<f32>,
    pub mean: Vec<f32>,
    pub variance: Vec<f32>,
}

impl VelocityEstimator {
    pub fn new() -> Self {
        Self { last_time: f64::NEG_INFINITY, ..Default::default() }
    }

    fn reset(&mut self, rank: usize) {
        self.last_time = f64::NEG_INFINITY;
        self.num_samples = 0;
        self.velocity = vec![0.0; rank];
        self.prev_position = vec![0.0; rank];
        self.mean = vec![0.0; rank];
        self.variance = vec![0.0; rank];
    }

    /// `addSample(position, time)`, `time` in milliseconds. (As
    /// neuroglancer's, the first sample's interval is infinite: its
    /// velocity is 0.)
    pub fn add_sample(&mut self, position: &[f32], time: f64) {
        let rank = position.len();
        if rank != self.mean.len() {
            self.reset(rank);
        }
        let num_samples = self.num_samples;
        self.num_samples += 1;
        let delta_t = time - self.last_time;
        self.last_time = time;
        let velocity_alpha = 1.0 - 2f64.powf(-(delta_t / VELOCITY_HALF_LIFE_MS));
        let model_alpha = 1.0 - 2f64.powf(-(delta_t / MODEL_HALF_LIFE_MS));
        for (i, &p) in position.iter().enumerate() {
            // `Float32Array` arithmetic: each stored value rounds to f32.
            let sample = (p as f64 - self.prev_position[i] as f64) / delta_t.max(1.0);
            self.prev_position[i] = p;
            let prev = self.velocity[i] as f64;
            let new = (prev + velocity_alpha * (sample - prev)) as f32;
            self.velocity[i] = new;
            if num_samples == 1 {
                self.mean[i] = new;
            } else {
                let mean_prev = self.mean[i] as f64;
                let var_prev = self.variance[i] as f64;
                let delta = new as f64 - mean_prev;
                self.mean[i] = (mean_prev + model_alpha * delta) as f32;
                self.variance[i] = ((1.0 - model_alpha) * (var_prev + model_alpha * delta * delta)) as f32;
            }
        }
    }
}

/// `erf`: Abramowitz and Stegun 7.1.26, as neuroglancer computes it.
pub fn erf(x: f64) -> f64 {
    let (a1, a2, a3, a4, a5, p) = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911);
    let t = 1.0 / (1.0 + p * x.abs());
    let y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    // `Math.sign`: 0 at 0.
    if x == 0.0 {
        0.0
    } else {
        x.signum() * y
    }
}

const PREFETCH_MS: f64 = 2000.0;
const MAX_PREFETCH_VELOCITY: f64 = 0.1;
const MAX_SINGLE_DIRECTION_PREFETCH_CHUNKS: i64 = 32;
const PREFETCH_PROBABILITY_CUTOFF: f64 = 0.05;

/// One chunk to prefetch next to each visible one: `delta` chunks along
/// chunk dimension `dim`, needed within the horizon with `probability`.
/// Entries come in groups, one a direction; a group stops at the first
/// chunk out of `[min, max]` or already visible (`group_end`: the index of
/// the next group).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefetchOffset {
    pub dim: usize,
    pub delta: i64,
    pub min: i64,
    pub max: i64,
    pub probability: f64,
    pub group_end: usize,
}

/// What [`prefetch_offsets`] needs of one chunk source in one view.
pub struct SourceGeometry<'a> {
    /// Per chunk dimension, its coefficient on each global dimension.
    pub global_to_chunk: &'a [Vec<f64>],
    /// Chunk dimensions shown in the view.
    pub display_dims: &'a [usize],
    pub chunk_data_size: &'a [u32],
    /// The position's chunk and offset within it, per chunk dimension.
    pub position_chunk: &'a [i64],
    pub position_in_chunk: &'a [f64],
    /// Clip bounds in voxels.
    pub lower_clip: &'a [f64],
    pub upper_clip: &'a [f64],
}

/// `getPrefetchChunkOffsets`.
pub fn prefetch_offsets(v: &VelocityEstimator, s: &SourceGeometry) -> Vec<PrefetchOffset> {
    let mut offsets: Vec<PrefetchOffset> = vec![];
    for chunk_dim in 0..s.chunk_data_size.len() {
        let is_display = s.display_dims.contains(&chunk_dim);
        let coeffs = &s.global_to_chunk[chunk_dim];
        let (mut mean, mut variance) = (0.0f64, 0.0f64);
        for (g, &c) in coeffs.iter().enumerate().take(v.mean.len()) {
            mean += c * v.mean[g] as f64;
            variance += c * c * v.variance[g] as f64;
        }
        if mean > MAX_PREFETCH_VELOCITY {
            continue;
        }
        let chunk_size = s.chunk_data_size[chunk_dim] as f64;
        let initial_fraction = if is_display { 0.0 } else { s.position_in_chunk[chunk_dim] / chunk_size };
        let adjusted_mean = mean / chunk_size * PREFETCH_MS;
        let stddev_sqrt2 = (2.0 * variance).sqrt() / chunk_size * PREFETCH_MS;
        if adjusted_mean.abs() < 1e-3 && stddev_sqrt2 < 1e-3 {
            continue;
        }
        let stddev_sqrt2 = stddev_sqrt2.max(1e-6);
        let cdf = |x: f64| 0.5 * (1.0 + erf((x - adjusted_mean) / stddev_sqrt2));
        let cur = s.position_chunk[chunk_dim];
        let min = (s.lower_clip[chunk_dim] / chunk_size).floor() as i64;
        let max = (s.upper_clip[chunk_dim] / chunk_size).ceil() as i64 - 1;
        let start = offsets.len();
        for i in 1..=MAX_SINGLE_DIRECTION_PREFETCH_CHUNKS {
            if !is_display && cur + i > max {
                break;
            }
            let probability = 1.0 - cdf(i as f64 - initial_fraction);
            if probability < PREFETCH_PROBABILITY_CUTOFF {
                break;
            }
            offsets.push(PrefetchOffset { dim: chunk_dim, delta: i, min, max, probability, group_end: 0 });
        }
        let end = offsets.len();
        offsets[start..end].iter_mut().for_each(|o| o.group_end = end);
        let start = end;
        for i in 1..=MAX_SINGLE_DIRECTION_PREFETCH_CHUNKS {
            if !is_display && cur - i < min {
                break;
            }
            let probability = cdf(-i as f64 + 1.0 - initial_fraction);
            if probability < PREFETCH_PROBABILITY_CUTOFF {
                break;
            }
            offsets.push(PrefetchOffset { dim: chunk_dim, delta: -i, min, max, probability, group_end: 0 });
        }
        let end = offsets.len();
        offsets[start..end].iter_mut().for_each(|o| o.group_end = end);
    }
    offsets
}

/// The chunks to prefetch around `visible` (grid positions): for each, each
/// group's offsets in turn until one is out of bounds or visible. With their
/// probabilities, highest first per chunk.
pub fn prefetch_chunks(visible: &[Vec<u64>], offsets: &[PrefetchOffset]) -> Vec<(Vec<u64>, f64)> {
    let seen: std::collections::HashSet<&Vec<u64>> = visible.iter().collect();
    let mut out = vec![];
    for chunk in visible {
        let mut j = 0;
        while j < offsets.len() {
            let o = offsets[j];
            let new = chunk[o.dim] as i64 + o.delta;
            if new < o.min || new > o.max || new < 0 {
                j = o.group_end;
                continue;
            }
            let mut grid = chunk.clone();
            grid[o.dim] = new as u64;
            if seen.contains(&grid) {
                j = o.group_end;
                continue;
            }
            out.push((grid, o.probability));
            j += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_is_abramowitz_and_stegun() {
        assert_eq!(erf(0.0), 0.0);
        assert!((erf(1.0) - 0.8427).abs() < 1e-4);
        assert!((erf(-1.0) + 0.8427).abs() < 1e-4);
    }

    fn geometry<'a>(g2c: &'a [Vec<f64>], display: &'a [usize], pos: &'a [i64], within: &'a [f64], lo: &'a [f64], hi: &'a [f64], size: &'a [u32]) -> SourceGeometry<'a> {
        SourceGeometry { global_to_chunk: g2c, display_dims: display, chunk_data_size: size, position_chunk: pos, position_in_chunk: within, lower_clip: lo, upper_clip: hi }
    }

    #[test]
    fn a_still_position_prefetches_nothing_and_a_moving_one_looks_ahead() {
        let g2c = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let (display, pos, within, lo, hi, size) = ([0usize, 1], [1i64, 1, 2], [0.0, 0.0, 12.0], [0.0; 3], [256.0; 3], [64u32, 64, 64]);
        let s = geometry(&g2c, &display, &pos, &within, &lo, &hi, &size);
        let mut v = VelocityEstimator::new();
        for t in 0..20 {
            v.add_sample(&[100.0, 100.0, 140.0], t as f64 * 16.0);
        }
        assert!(prefetch_offsets(&v, &s).is_empty());
        // Stepping through z, a slice a frame (0.06 voxels/ms): chunks ahead
        // in z, none behind it or in the plane.
        let mut v = VelocityEstimator::new();
        for t in 0..60 {
            v.add_sample(&[100.0, 100.0, 140.0 + t as f32], t as f64 * 16.0);
        }
        let offsets = prefetch_offsets(&v, &s);
        assert!(!offsets.is_empty());
        assert!(offsets.iter().all(|o| o.dim == 2 && o.delta > 0), "{offsets:?}");
        // Around the visible chunk (1, 1, 2) of a 4×4×4 grid: z 3, then the
        // grid ends.
        let chunks = prefetch_chunks(&[vec![1, 1, 2]], &offsets);
        assert_eq!(chunks.first().map(|c| c.0.clone()), Some(vec![1, 1, 3]));
        assert!(chunks.iter().all(|c| c.0[2] <= 3));
    }
}
