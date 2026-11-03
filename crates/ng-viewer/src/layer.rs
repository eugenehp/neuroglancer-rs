//! A volume layer: a data source's multiscale volume, placed in the global
//! coordinate space through its transform (`getRenderLayerTransform`,
//! `src/render_coordinate_transform.ts`).
//!
//! The transform is the source's own with the state's `transform` spec over
//! it, after the global combiner has rescaled it to the global units. Its
//! output dimensions that are global (not channel `^` or local `'`) map to
//! the global dimensions of the same name.
//!
//! A spec can add dimensions beyond the source's (`sourceRank`): those have
//! the singleton extent [0, 1), as their bounding boxes give them.

use std::sync::Arc;

use ng_core::coordinate_space::is_global_dimension;
use ng_core::transform::CoordinateSpaceTransform;
use ng_core::{CoordinateSpace, DataType};
use ng_datasource::model::{DataSource, VolumeSource, VolumeType};

pub struct VolumeLayer {
    /// One chunk layout per scale, finest first.
    pub scales: Vec<Arc<VolumeSource>>,
    pub data_type: DataType,
    pub volume_type: VolumeType,
    /// The source's (chunk) rank.
    pub source_rank: usize,
    /// The transform's rank: `source_rank` plus dimensions a spec added.
    pub rank: usize,
    pub transform: CoordinateSpaceTransform,
    /// `(rank + 1)²`, column-major: model coordinates → layer coordinates
    /// (`scaleTransformSubmatrix`, rounded to `f32` as it stores it).
    pub model_to_layer: Vec<f64>,
    /// For each layer (output) dimension, its global dimension, or `None` for
    /// channel and local dimensions.
    pub layer_to_global: Vec<Option<usize>>,
    /// The invlerp range (`defaultDataTypeRange` unless the state sets it).
    pub range: ng_render::image::Interval,
    /// Channel dimensions (`^`, `getChannelSpace`), in the layer's order.
    pub channels: Vec<ChannelDim>,
}

/// A channel dimension: its layer dimension, the chunk dimension holding
/// it, and its extent (`[0, extent)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelDim {
    pub layer_dim: usize,
    pub chunk_dim: usize,
    pub extent: u32,
}

/// `source` with its clip bounds moved in from each face of its first three
/// (spatial) dimensions by `trim[c]` voxels of the finest scale — fewer of
/// its own on a coarser one (its `chunk_to_multiscale_transform` scale), so
/// the same width of the data goes at every level — and no further than its
/// middle.
pub fn trim_edges(source: &VolumeSource, trim: [f64; 3]) -> VolumeSource {
    let mut s = source.clone();
    let spec = &source.spec;
    let rank = spec.rank.min(3);
    let n = spec.rank + 1;
    let mut lower: Vec<f32> = s.lower_clip_bound.clone().unwrap_or_else(|| spec.lower_voxel_bound.clone());
    let mut upper: Vec<f32> = s.upper_clip_bound.clone().unwrap_or_else(|| spec.upper_voxel_bound.clone());
    for c in 0..rank {
        let relative = source.chunk_to_multiscale_transform.get(c * n + c).map_or(1.0, |&v| v as f64).abs().max(1e-9);
        let (lo, hi) = (lower[c] as f64, upper[c] as f64);
        let k = (trim[c] / relative).min((hi - lo) / 2.0 - 0.5).max(0.0);
        lower[c] = (lo + k) as f32;
        upper[c] = (hi - k) as f32;
    }
    s.lower_clip_bound = Some(lower);
    s.upper_clip_bound = Some(upper);
    s
}

impl VolumeLayer {
    pub fn new(source: &DataSource, transform: CoordinateSpaceTransform, global: &CoordinateSpace) -> anyhow::Result<Self> {
        let volume = source.volume().ok_or_else(|| anyhow::anyhow!("the data source has no volume"))?;
        let scales: Vec<Arc<VolumeSource>> = volume.sources.first().map(|alts| alts.iter().cloned().map(Arc::new).collect()).unwrap_or_default();
        let source_rank = scales.first().map_or(transform.source_rank, |s| s.spec.rank);
        anyhow::ensure!(
            source_rank == transform.source_rank,
            "the volume's rank ({source_rank}) differs from its transform's source rank ({})",
            transform.source_rank
        );
        let mut layer = Self {
            scales,
            data_type: volume.data_type,
            volume_type: volume.volume_type,
            source_rank,
            rank: transform.rank,
            model_to_layer: vec![],
            layer_to_global: vec![],
            range: ng_render::image::default_data_type_range(volume.data_type),
            transform: transform.clone(),
            channels: vec![],
        };
        layer.set_transform(transform, global);
        Ok(layer)
    }

    /// A layer with a transform but no volume (an annotation layer's).
    pub fn without_volume(transform: CoordinateSpaceTransform, global: &CoordinateSpace) -> Self {
        let mut layer = Self {
            scales: vec![],
            data_type: DataType::Uint8,
            volume_type: VolumeType::Unknown,
            source_rank: transform.source_rank,
            rank: transform.rank,
            model_to_layer: vec![],
            layer_to_global: vec![],
            range: ng_render::image::default_data_type_range(DataType::Uint8),
            transform: transform.clone(),
            channels: vec![],
        };
        layer.set_transform(transform, global);
        layer
    }

    /// The source's default transform: its model transform as given.
    pub fn default_transform(source: &DataSource) -> CoordinateSpaceTransform {
        source.model_transform.clone().normalized()
    }

    /// Takes a new (bound) transform or global space.
    pub fn set_transform(&mut self, transform: CoordinateSpaceTransform, global: &CoordinateSpace) {
        let rank = transform.rank;
        let n = rank + 1;
        let input = &transform.input_space;
        let output = &transform.output_space;
        // `scaleTransformSubmatrix`: each linear entry scaled by input scale
        // / output scale (the output's are the global, local or channel
        // space's, which the combiners keep equal to the layer's).
        let mut m = transform.transform.clone();
        for input_dim in 0..rank {
            for out in 0..rank {
                m[input_dim * n + out] *= input.scales[input_dim] / output.scales[out];
            }
        }
        self.model_to_layer = m.iter().map(|&x| x as f32 as f64).collect();
        self.layer_to_global = output
            .names
            .iter()
            .map(|name| if is_global_dimension(name) { global.names.iter().position(|g| g == name) } else { None })
            .collect();
        self.rank = rank;
        // Channel dimensions: those the source's dimensions map onto, over
        // the finest scale's voxel bounds.
        self.channels = match self.scales.first() {
            Some(s0) => output
                .names
                .iter()
                .enumerate()
                .filter(|(_, name)| name.ends_with('^'))
                .filter_map(|(layer_dim, _)| {
                    let chunk_dim = (0..self.source_rank.min(rank)).find(|&c| self.model_to_layer[c * n + layer_dim] != 0.0)?;
                    let extent = (s0.spec.upper_voxel_bound[chunk_dim] - s0.spec.lower_voxel_bound[chunk_dim]).max(1.0) as u32;
                    Some(ChannelDim { layer_dim, chunk_dim, extent })
                })
                .collect(),
            None => vec![],
        };
        self.transform = transform;
    }

    /// The channel extents, first fastest (`getChannelSpace`'s shape).
    pub fn channel_extents(&self) -> Vec<u32> {
        self.channels.iter().map(|c| c.extent).collect()
    }

    /// A global position in layer coordinates, homogeneous (`rank + 1`,
    /// ending in 1): layer dimensions no global one maps to at `unmapped`.
    /// `None` if `global` is too short.
    pub fn layer_position(&self, global: &[f32], unmapped: f64) -> Option<Vec<f64>> {
        let mut p = vec![1.0; self.rank + 1];
        for (d, g) in self.layer_to_global.iter().enumerate() {
            p[d] = match g {
                Some(g) => *global.get(*g)? as f64,
                None => unmapped,
            };
        }
        Some(p)
    }

    /// `(rank + 1)²`, column-major: chunk voxels of scale `s` → layer
    /// coordinates. Dimensions past the source's map through unchanged.
    pub fn chunk_to_layer(&self, s: usize) -> Vec<f64> {
        let n = self.rank + 1;
        let sn = self.source_rank + 1;
        let c = &self.scales[s].chunk_to_multiscale_transform;
        // `extendHomogeneousTransform`: the source's rows and columns, the
        // identity for the added dimensions.
        let map = |i: usize| if i == self.rank { Some(self.source_rank) } else if i < self.source_rank { Some(i) } else { None };
        let mut ext = vec![0.0; n * n];
        for col in 0..n {
            for row in 0..n {
                ext[col * n + row] = match (map(col), map(row)) {
                    (Some(sc), Some(sr)) => c[sc * sn + sr] as f64,
                    _ => (col == row) as u8 as f64,
                };
            }
        }
        ng_core::matrix::multiply(&self.model_to_layer, &ext, n)
    }
}

#[cfg(test)]
mod trim_tests {
    use super::*;
    use ng_datasource::model::{ChunkSourceParams, VolumeChunkSpec};

    fn source(upper: [f32; 3], clip: Option<(Vec<f32>, Vec<f32>)>) -> VolumeSource {
        VolumeSource {
            spec: VolumeChunkSpec {
                rank: 3,
                data_type: DataType::Uint8,
                chunk_data_size: vec![64, 64, 64],
                lower_voxel_bound: vec![0.0; 3],
                upper_voxel_bound: upper.to_vec(),
                base_voxel_offset: vec![0.0; 3],
                compressed_segmentation_block_size: None,
            },
            params: ChunkSourceParams::Nifti { url: String::new(), data: None },
            chunk_to_multiscale_transform: vec![],
            lower_clip_bound: clip.as_ref().map(|c| c.0.clone()),
            upper_clip_bound: clip.map(|c| c.1),
        }
    }

    #[test]
    fn trims_each_face_from_the_clip_bounds_or_the_voxel_bounds() {
        let t = trim_edges(&source([252.0, 175.0, 11.0], None), [2.0, 2.0, 0.0]);
        assert_eq!(t.lower_clip_bound, Some(vec![2.0, 2.0, 0.0]));
        assert_eq!(t.upper_clip_bound, Some(vec![250.0, 173.0, 11.0]));
        let t = trim_edges(&source([100.0; 3], Some((vec![10.0, 0.0, 0.0], vec![90.0, 100.0, 100.0]))), [1.0; 3]);
        assert_eq!((t.lower_clip_bound.unwrap()[0], t.upper_clip_bound.unwrap()[0]), (11.0, 89.0));
    }

    #[test]
    fn a_coarser_scale_loses_fewer_of_its_voxels() {
        // a level at 4× the finest voxel: 8 finest voxels are 2 of its own
        let mut s = source([100.0; 3], None);
        s.chunk_to_multiscale_transform = vec![4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let t = trim_edges(&s, [8.0, 8.0, 0.0]);
        assert_eq!(t.lower_clip_bound, Some(vec![2.0, 2.0, 0.0]));
        assert_eq!(t.upper_clip_bound, Some(vec![98.0, 98.0, 100.0]));
    }

    #[test]
    fn a_thin_volume_keeps_its_middle() {
        // 3 voxels deep: at most one comes off each face, one stays
        let t = trim_edges(&source([100.0, 100.0, 3.0], None), [0.0, 0.0, 2.0]);
        assert_eq!((t.lower_clip_bound.unwrap()[2], t.upper_clip_bound.unwrap()[2]), (1.0, 2.0));
    }
}
