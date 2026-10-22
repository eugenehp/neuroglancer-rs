//! What a data source provides, independent of format: a model coordinate
//! space and transform, and subsources (a multiscale volume, the data-bounds
//! annotation). Port of the parts of `src/datasource/index.ts` and
//! `src/sliceview/volume/*` that describe volumes.
//!
//! [`DataSource::snapshot`] renders it the way neuroglancer's
//! `tests/datasource/test_util.ts` does, so it can be compared with
//! neuroglancer's own metadata snapshots.

use std::sync::Arc;

use ng_core::coordinate_space::CoordinateSpaceBounds;
use ng_core::{js, matrix, CoordinateSpace, DataType};
use serde_json::{json, Map, Value};

/// `VolumeType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeType {
    Unknown,
    Image,
    Segmentation,
}

impl VolumeType {
    pub fn name(self) -> &'static str {
        match self {
            VolumeType::Unknown => "UNKNOWN",
            VolumeType::Image => "IMAGE",
            VolumeType::Segmentation => "SEGMENTATION",
        }
    }
}

pub use ng_core::transform::CoordinateSpaceTransform;

/// `VolumeChunkSpecification`: one resolution's chunk grid.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeChunkSpec {
    pub rank: usize,
    pub data_type: DataType,
    pub chunk_data_size: Vec<u32>,
    pub lower_voxel_bound: Vec<f32>,
    pub upper_voxel_bound: Vec<f32>,
    /// Added to a chunk's grid position × size to get its voxel position.
    pub base_voxel_offset: Vec<f32>,
    pub compressed_segmentation_block_size: Option<[u32; 3]>,
}

impl VolumeChunkSpec {
    /// `computeChunkBounds`: the voxel origin and (clipped) size of the chunk
    /// at `grid`.
    pub fn chunk_bounds(&self, grid: &[u64]) -> (Vec<f64>, Vec<u32>) {
        let mut origin = Vec::with_capacity(self.rank);
        let mut size = Vec::with_capacity(self.rank);
        for (i, &g) in grid.iter().enumerate().take(self.rank) {
            let start = g as f64 * self.chunk_data_size[i] as f64;
            let upper = (self.upper_voxel_bound[i] as f64).min(start + self.chunk_data_size[i] as f64);
            size.push((upper - start) as u32);
            origin.push(start + self.base_voxel_offset[i] as f64);
        }
        (origin, size)
    }

    /// Chunks along each dimension (`upperChunkBound` with a zero lower bound).
    pub fn grid_shape(&self) -> Vec<u64> {
        (0..self.rank)
            .map(|i| ((self.upper_voxel_bound[i] as f64 - 1.0) / self.chunk_data_size[i] as f64 + 1.0).floor() as u64)
            .collect()
    }
}

/// How to fetch and decode one resolution's chunks.
#[derive(Debug, Clone)]
pub enum ChunkSourceParams {
    Precomputed(crate::precomputed::ChunkParams),
    Zarr(crate::zarr::ChunkParams),
    N5(crate::n5::ChunkParams),
    DeepZoom(crate::deepzoom::ChunkParams),
    /// A NIfTI file is one chunk, held in memory once read (`None` for
    /// metadata only).
    Nifti { url: String, data: Option<Arc<Vec<u8>>> },
}

/// `SliceViewSingleResolutionSource`.
#[derive(Debug, Clone)]
pub struct VolumeSource {
    pub spec: VolumeChunkSpec,
    pub params: ChunkSourceParams,
    /// `(rank + 1)²`, column-major: chunk voxels → model (multiscale) voxels.
    pub chunk_to_multiscale_transform: Vec<f32>,
    pub lower_clip_bound: Option<Vec<f32>>,
    pub upper_clip_bound: Option<Vec<f32>>,
}

/// `MultiscaleVolumeChunkSource.getSources(...)`: `sources[alternative][scale]`,
/// finest scale first. Alternatives are chunk layouts of the same data.
#[derive(Debug, Clone)]
pub struct MultiscaleVolume {
    pub data_type: DataType,
    pub volume_type: VolumeType,
    pub rank: usize,
    pub sources: Vec<Vec<VolumeSource>>,
}

#[derive(Debug, Clone)]
pub enum Subsource {
    Volume(MultiscaleVolume),
    /// `makeDataBoundsBoundingBoxAnnotationSet`.
    DataBounds { lower: Vec<f64>, upper: Vec<f64> },
    /// Object meshes.
    Mesh(Box<crate::mesh::MeshSource>),
    /// Object skeletons.
    Skeleton(Box<crate::skeleton::SkeletonSource>),
    /// A precomputed annotation collection.
    Annotations(Box<crate::annotation::AnnotationInfo>),
    /// `local://annotations`: the layer's own annotations, in its state.
    LocalAnnotations,
}

#[derive(Debug, Clone)]
pub struct SubsourceEntry {
    pub id: String,
    pub default: bool,
    pub subsource: Subsource,
}

#[derive(Debug, Clone)]
pub struct DataSource {
    pub canonical_url: String,
    pub model_transform: CoordinateSpaceTransform,
    pub subsources: Vec<SubsourceEntry>,
}

impl DataSource {
    /// The standard pair every volume source offers: the volume and its bounds.
    pub fn volume_with_bounds(canonical_url: String, model_transform: CoordinateSpaceTransform, volume: MultiscaleVolume, bounds: &CoordinateSpaceBounds) -> Self {
        DataSource {
            canonical_url,
            model_transform,
            subsources: vec![
                SubsourceEntry {
                    id: "default".into(),
                    default: true,
                    subsource: Subsource::Volume(volume),
                },
                SubsourceEntry {
                    id: "bounds".into(),
                    default: true,
                    subsource: Subsource::DataBounds {
                        lower: bounds.lower_bounds.clone(),
                        upper: bounds.upper_bounds.clone(),
                    },
                },
            ],
        }
    }

    pub fn volume(&self) -> Option<&MultiscaleVolume> {
        self.subsources.iter().find_map(|s| match &s.subsource {
            Subsource::Volume(v) => Some(v),
            _ => None,
        })
    }

    /// `getDatasourceSnapshot` (neuroglancer's `tests/datasource/test_util.ts`).
    pub fn snapshot(&self) -> Value {
        json!({
            "subsources": self.subsources.iter().map(subsource_snapshot).collect::<Vec<_>>(),
            "modelTransform": transform_snapshot(&self.model_transform),
            "canonicalUrl": self.canonical_url,
        })
    }
}

/// A float as the snapshot prints it: infinities and NaN as strings.
fn num(x: f64) -> Value {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        (if x > 0.0 { "Infinity" } else { "-Infinity" }).into()
    } else {
        js::number(x)
    }
}

fn nums<T: Copy + Into<f64>>(v: &[T]) -> Value {
    Value::Array(v.iter().map(|&x| num(x.into())).collect())
}

fn space_snapshot(s: &CoordinateSpace) -> Value {
    json!({
        "bounds": {
            "lowerBounds": nums(&s.bounds.lower_bounds),
            "upperBounds": nums(&s.bounds.upper_bounds),
            "voxelCenterAtIntegerCoordinates": s.bounds.voxel_center_at_integer_coordinates,
        },
        "coordinateArrays": s.coordinate_arrays.iter().map(|a| match a {
            None => Value::Null,
            Some(a) => json!({"explicit": a.explicit, "coordinates": a.coordinates, "labels": a.labels}),
        }).collect::<Vec<_>>(),
        "names": s.names,
        "scales": nums(&s.scales),
        "units": s.units,
        "valid": s.valid,
    })
}

/// `coordinateSpacesEqual`, for whether the snapshot shows `outputSpace`.
fn spaces_equal(a: &CoordinateSpace, b: &CoordinateSpace) -> bool {
    a == b
}

fn matrix_rows<T: Copy + Into<f64>>(m: &[T], rows: usize) -> Value {
    let cols = m.len() / rows;
    Value::Array(
        (0..rows)
            .map(|r| Value::Array((0..cols).map(|c| num(m[c * rows + r].into())).collect()))
            .collect(),
    )
}

fn transform_snapshot(t: &CoordinateSpaceTransform) -> Value {
    let mut out = Map::new();
    out.insert("inputSpace".into(), space_snapshot(&t.input_space));
    if !spaces_equal(&t.input_space, &t.output_space) {
        out.insert("outputSpace".into(), space_snapshot(&t.output_space));
    }
    if !matrix::is_identity(&t.transform, t.rank + 1, t.rank + 1) {
        out.insert("transform".into(), matrix_rows(&t.transform, t.rank + 1));
    }
    Value::Object(out)
}

fn subsource_snapshot(e: &SubsourceEntry) -> Value {
    let subsource = match &e.subsource {
        Subsource::Volume(v) => json!({"volume": {
            "dataType": v.data_type.name(),
            "volumeType": v.volume_type.name(),
            "rank": v.rank,
            "sources": v.sources.iter().map(|alts| alts.iter().map(source_snapshot).collect::<Vec<_>>()).collect::<Vec<_>>(),
        }}),
        Subsource::DataBounds { lower, upper } => json!({"staticAnnotations": [{
            "description": "Data Bounds",
            "id": "data-bounds",
            "pointA": nums(lower),
            "pointB": nums(upper),
            "type": "axis_aligned_bounding_box",
        }]}),
        Subsource::Mesh(m) => json!({"mesh": {"url": m.location.url(), "transform": nums(&m.to_model)}}),
        Subsource::Skeleton(m) => json!({"skeleton": {"url": m.location.url(), "transform": nums(&m.to_model)}}),
        Subsource::LocalAnnotations => json!({"local": "annotations"}),
        Subsource::Annotations(a) => json!({"annotation": {"type": format!("{:?}", a.annotation_type), "rank": a.rank, "levels": a.spatial.len()}}),
    };
    json!({"subsource": subsource, "default": e.default, "id": e.id})
}

fn source_snapshot(s: &VolumeSource) -> Value {
    let spec = &s.spec;
    let mut out = Map::new();
    if let Some(b) = &s.lower_clip_bound {
        out.insert("lowerClipBound".into(), nums(b));
    }
    if let Some(b) = &s.upper_clip_bound {
        out.insert("upperClipBound".into(), nums(b));
    }
    let parameters = match &s.params {
        ChunkSourceParams::Precomputed(p) => p.snapshot(),
        ChunkSourceParams::Zarr(p) => json!({"url": p.url()}),
        // `VolumeChunkEncoding` by its index.
        ChunkSourceParams::N5(p) => json!({"url": p.location.url(), "encoding": p.encoding as u8}),
        ChunkSourceParams::DeepZoom(p) => json!({"url": p.location.url(), "encoding": p.encoding as u8, "format": p.format, "overlap": p.overlap, "tilesize": p.tilesize}),
        ChunkSourceParams::Nifti { url, .. } => json!({"url": url}),
    };
    out.insert(
        "chunkSource".into(),
        json!({
            "parameters": parameters,
            "spec": {
                "baseVoxelOffset": nums(&spec.base_voxel_offset),
                "chunkDataSize": spec.chunk_data_size,
                "compressedSegmentationBlockSize": spec.compressed_segmentation_block_size,
                "dataType": spec.data_type.name(),
                "lowerVoxelBound": nums(&spec.lower_voxel_bound),
                "upperVoxelBound": nums(&spec.upper_voxel_bound),
            },
        }),
    );
    let n = spec.rank + 1;
    let t: Vec<f64> = s.chunk_to_multiscale_transform.iter().map(|&x| x as f64).collect();
    if !matrix::is_identity(&t, n, n) {
        out.insert("chunkToMultiscaleTransform".into(), matrix_rows(&s.chunk_to_multiscale_transform, n));
    }
    Value::Object(out)
}
