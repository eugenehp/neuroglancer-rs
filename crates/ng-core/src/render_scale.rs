//! `render_scale_statistics.ts`: the histogram of how finely each layer's
//! data was drawn last frame, by spatial scale, and the render scale target
//! it is drawn at.

/// `numRenderScaleHistogramBins`.
pub const NUM_BINS: usize = 40;
/// `renderScaleHistogramBinSize`: bins per power of two, inverted.
pub const BIN_SIZE: f64 = 0.5;
/// `renderScaleHistogramOrigin`: log₂ of the first bin's scale.
pub const ORIGIN: f64 = -4.0;

/// `VOLUME_RENDERING_DEPTH_SAMPLES_LOG_SCALE_ORIGIN`: the volume rendering
/// histogram's first bin (depth samples).
pub const VOLUME_RENDERING_ORIGIN: f64 = 1.0;

/// `VOLUME_RENDERING_RESOLUTION_INDICATOR_BAR_HEIGHT`: the count an other
/// scale's indicator bar stands for.
pub const VOLUME_RENDERING_INDICATOR_BAR_HEIGHT: u32 = 10;

/// `getRenderScaleHistogramOffset`: the (fractional) bin of `render_scale`.
pub fn histogram_offset(render_scale: f64) -> f64 {
    histogram_offset_at(render_scale, ORIGIN)
}

/// [`histogram_offset`] over a histogram starting at `origin`.
pub fn histogram_offset_at(render_scale: f64, origin: f64) -> f64 {
    (render_scale.log2() - origin) / BIN_SIZE
}

/// [`scale_from_histogram_offset`] over a histogram starting at `origin`.
pub fn scale_from_histogram_offset_at(offset: f64, origin: f64) -> f64 {
    (offset * BIN_SIZE + origin).exp2()
}

/// `getRenderScaleFromHistogramOffset`.
pub fn scale_from_histogram_offset(offset: f64) -> f64 {
    (offset * BIN_SIZE + ORIGIN).exp2()
}

/// The interval `trackableRenderScaleTarget` accepts: 2⁻⁴ to 2¹⁶ − 1.
pub fn target_interval() -> (f64, f64) {
    (ORIGIN.exp2(), (BIN_SIZE * NUM_BINS as f64 + ORIGIN).round().exp2() - 1.0)
}

/// A render scale target read from JSON (`verifyNumberInInterval`):
/// `default` unless a number in [`target_interval`].
pub fn parse_target(v: Option<&serde_json::Value>, default: f64) -> f64 {
    let (lo, hi) = target_interval();
    v.and_then(serde_json::Value::as_f64).filter(|x| (lo..=hi).contains(x)).unwrap_or(default)
}

/// `RenderScaleHistogram`: chunk counts per (spatial scale, render scale
/// bin), present and not.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderScaleHistogram {
    /// Spatial scales (metres) in the order first added; row `i` of `rows`.
    pub spatial_scales: Vec<f64>,
    /// Per spatial scale: present counts per bin, then not-present counts.
    pub rows: Vec<[u32; 2 * NUM_BINS]>,
    /// `frameNumber` the counts are for.
    pub frame: Option<u64>,
    /// `logScaleOrigin`: log₂ of the first bin's scale.
    pub origin: f64,
    /// `fakeChunkCount`: not-present counts that only mark a bar.
    pub fake_chunk_count: u32,
}

impl Default for RenderScaleHistogram {
    fn default() -> Self {
        Self { spatial_scales: vec![], rows: vec![], frame: None, origin: ORIGIN, fake_chunk_count: 0 }
    }
}

impl RenderScaleHistogram {
    /// `begin`: starts counting `frame`, unless it already is.
    pub fn begin(&mut self, frame: u64) {
        if self.frame != Some(frame) {
            self.frame = Some(frame);
            self.spatial_scales.clear();
            self.rows.clear();
            self.fake_chunk_count = 0;
        }
    }

    /// The volume rendering histogram (`volumeRenderingChunkResolutionHistogram`):
    /// depth samples from 2¹.
    pub fn volume_rendering() -> Self {
        Self { origin: VOLUME_RENDERING_ORIGIN, ..Self::default() }
    }

    /// `add(…, renderOnly = true)`: a bar that marks a scale, not chunks.
    pub fn add_indicator(&mut self, spatial_scale: f64, render_scale: f64, height: u32) {
        self.add(spatial_scale, render_scale, 0, height);
        self.fake_chunk_count += height;
    }

    /// `add`: chunks of data `spatial_scale` metres across drawn at
    /// `render_scale` screen pixels each.
    pub fn add(&mut self, spatial_scale: f64, render_scale: f64, present: u32, not_present: u32) {
        let row = match self.spatial_scales.iter().position(|&s| s == spatial_scale) {
            Some(r) => r,
            None => {
                self.spatial_scales.push(spatial_scale);
                self.rows.push([0; 2 * NUM_BINS]);
                self.rows.len() - 1
            }
        };
        let bin = histogram_offset_at(render_scale, self.origin).round().clamp(0.0, (NUM_BINS - 1) as f64) as usize;
        self.rows[row][bin] += present;
        self.rows[row][bin + NUM_BINS] += not_present;
    }

    /// Total (present, not present) chunks.
    pub fn totals(&self) -> (u32, u32) {
        let (p, n) = self.rows.iter().fold((0, 0), |(p, n), r| (p + r[..NUM_BINS].iter().sum::<u32>(), n + r[NUM_BINS..].iter().sum::<u32>()));
        (p, n.saturating_sub(self.fake_chunk_count))
    }
}

/// `medianOf3`.
pub fn median_of_3(a: f64, b: f64, c: f64) -> f64 {
    a.max(b).min(a.min(b).max(c))
}

#[cfg(test)]
mod tests {
    #[test]
    fn volume_rendering_histograms_count_without_indicators() {
        let mut h = RenderScaleHistogram::volume_rendering();
        h.begin(1);
        h.add_indicator(4e-9, 16.0, VOLUME_RENDERING_INDICATOR_BAR_HEIGHT);
        h.add(8e-9, 64.0, 3, 2);
        assert_eq!(h.totals(), (3, 2));
        // 64 samples: (log₂ 64 − 1) / 0.5 = bin 10.
        assert_eq!(h.rows[1][10], 3);
        h.begin(2);
        assert_eq!((h.totals(), h.fake_chunk_count), ((0, 0), 0));
    }

    use super::*;

    #[test]
    fn bins_and_targets_as_neuroglancer() {
        assert_eq!(histogram_offset(1.0), 8.0);
        assert_eq!(scale_from_histogram_offset(8.0), 1.0);
        assert_eq!(target_interval(), (0.0625, 65535.0));
        assert_eq!(parse_target(Some(&serde_json::json!(2)), 1.0), 2.0);
        assert_eq!(parse_target(Some(&serde_json::json!(0.01)), 1.0), 1.0);
        assert_eq!(parse_target(Some(&serde_json::json!("2")), 1.0), 1.0);
        let mut h = RenderScaleHistogram::default();
        h.begin(1);
        h.add(4e-9, 1.0, 3, 1);
        h.add(4e-9, 1e9, 1, 0);
        h.add(8e-9, 0.5, 2, 2);
        assert_eq!(h.spatial_scales, vec![4e-9, 8e-9]);
        assert_eq!((h.rows[0][8], h.rows[0][8 + NUM_BINS], h.rows[0][NUM_BINS - 1], h.rows[1][6]), (3, 1, 1, 2));
        assert_eq!(h.totals(), (6, 3));
        h.begin(1);
        assert_eq!(h.totals(), (6, 3));
        h.begin(2);
        assert_eq!(h.totals(), (0, 0));
        assert_eq!(median_of_3(3.0, 1.0, 2.0), 2.0);
        assert_eq!(median_of_3(4.0, 4.0, 40.0), 4.0);
    }
}
