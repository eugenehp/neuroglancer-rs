//! Segment colors. Port of `SegmentColorHash` (`src/segment_color.ts`) and
//! `hashCombine` (`src/gpu_hash/hash_function.ts`).
//!
//! A segment's color depends only on its uint64 ID and the layer's seed, so
//! these must match neuroglancer exactly for the same state to look the same.

const K1: u32 = 0xcc9e_2d51;
const K2: u32 = 0x1b87_3593;

/// MurmurHash3's per-block mix, without the final mixing steps.
pub fn hash_combine(state: u32, value: u32) -> u32 {
    let value = value.wrapping_mul(K1).rotate_left(15).wrapping_mul(K2);
    (state ^ value)
        .rotate_left(13)
        .wrapping_mul(5)
        .wrapping_add(0xe654_6b64)
}

/// HSV → RGB, each component in `[0, 1]`. Port of `hsvToRgb`
/// (`src/util/colorspace.ts`), in `f32` because the TS writes into a
/// `Float32Array`.
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f32; 3] {
    let h = h * 6.0;
    let hue_index = h.floor();
    let remainder = h - hue_index;
    let val1 = v * (1.0 - s);
    let val2 = v * (1.0 - s * remainder);
    let val3 = v * (1.0 - s * (1.0 - remainder));
    let rgb = match (hue_index as i64).rem_euclid(6) {
        0 => [v, val3, val1],
        1 => [val2, v, val1],
        2 => [val1, v, val3],
        3 => [val1, val2, v],
        4 => [val3, val1, v],
        _ => [v, val1, val2],
    };
    rgb.map(|c| c as f32)
}

/// The per-layer seed that decides segment colors. `0` is the default and is
/// omitted from the layer's JSON.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SegmentColorHash {
    pub seed: u32,
}

impl SegmentColorHash {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    pub fn compute(&self, id: u64) -> [f32; 3] {
        let h = hash_combine(self.seed, id as u32);
        let h = hash_combine(h, (id >> 32) as u32);
        let c0 = (h & 0xff) as f64 / 255.0;
        let c1 = ((h >> 8) & 0xff) as f64 / 255.0;
        hsv_to_rgb(c0, 0.5 + 0.5 * c1, 1.0)
    }

    /// The JSON form: absent for the default seed, otherwise the number.
    pub fn to_json(&self) -> Option<serde_json::Value> {
        (self.seed != 0).then(|| self.seed.into())
    }

    /// Accepts what `restoreState` does: any number, truncated to uint32 the
    /// way JavaScript's `x >>> 0` does. Anything else becomes the default seed.
    pub fn from_json(value: &serde_json::Value) -> Self {
        Self::new(value.as_f64().map(crate::js::to_uint32).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_seed_is_omitted_from_json() {
        assert_eq!(SegmentColorHash::default().to_json(), None);
        assert_eq!(SegmentColorHash::new(7).to_json(), Some(7.into()));
    }
}
