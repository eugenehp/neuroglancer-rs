//! NIfTI volumes. Port of `src/datasource/nifti/backend.ts` (volume info and
//! chunk decode); the file itself is parsed by `exg_nifti::parse_raw`.
//!
//! As in neuroglancer: values are shown as stored (no `scl_slope`/`scl_inter`),
//! int64 is read as uint64, float64 is narrowed to float32, and the geometry is
//! the qform alone, computed in `f32` the way gl-matrix computes it.

use anyhow::{bail, Result};
use exg_nifti::{DataType as NiftiType, RawNifti};
use ng_core::DataType;

/// Port of `NiftiVolumeInfo` (`src/datasource/nifti/base.ts`).
#[derive(Debug, Clone, PartialEq)]
pub struct NiftiVolumeInfo {
    pub rank: usize,
    /// `i, j, k, m, c^, c1^, c2^`, truncated to `rank`.
    pub source_names: Vec<String>,
    /// `x, y, z, t, c^, c1^, c2^`, truncated to `rank`.
    pub view_names: Vec<String>,
    pub view_scales: Vec<f64>,
    pub source_scales: Vec<f64>,
    pub units: Vec<String>,
    pub data_type: DataType,
    /// Homogeneous `(rank + 1) × (rank + 1)` source → view transform,
    /// column-major.
    pub transform: Vec<f64>,
    pub description: String,
    pub volume_size: Vec<u32>,
}

/// A whole NIfTI volume: its info and its voxels, little-endian, in
/// `info.data_type`, x fastest.
#[derive(Debug, Clone)]
pub struct NiftiVolume {
    pub info: NiftiVolumeInfo,
    pub data: Vec<u8>,
}

/// `NIFTI1.*` unit codes from nifti-reader-js.
mod units {
    pub const SPATIAL_MASK: u8 = 0x07;
    pub const TEMPORAL_MASK: u8 = 0x38;
    pub const METER: u8 = 1;
    pub const MM: u8 = 2;
    pub const MICRON: u8 = 3;
    pub const SEC: u8 = 8;
    pub const MSEC: u8 = 16;
    pub const USEC: u8 = 24;
    pub const HZ: u8 = 32;
    pub const RADS: u8 = 48;
}

/// `DATA_TYPE_CONVERSIONS`.
fn convert_data_type(t: NiftiType) -> DataType {
    match t {
        NiftiType::I8 => DataType::Int8,
        NiftiType::U8 => DataType::Uint8,
        NiftiType::I16 => DataType::Int16,
        NiftiType::U16 => DataType::Uint16,
        NiftiType::I32 => DataType::Int32,
        NiftiType::U32 => DataType::Uint32,
        NiftiType::I64 | NiftiType::U64 => DataType::Uint64,
        // Downcast unsupported float64 to float32 for visualization.
        NiftiType::F32 | NiftiType::F64 => DataType::Float32,
    }
}

/// Port of the `GET_NIFTI_VOLUME_INFO_RPC_ID` handler.
pub fn volume_info(raw: &RawNifti) -> Result<NiftiVolumeInfo> {
    let h = &raw.header;
    let (spatial_inv_scale, spatial_unit) = match h.xyzt_units & units::SPATIAL_MASK {
        units::METER => (1.0, "m"),
        units::MM => (1e3, "m"),
        units::MICRON => (1e6, "m"),
        _ => (1.0, ""),
    };
    let (time_inv_scale, time_unit) = match h.xyzt_units & units::TEMPORAL_MASK {
        units::SEC => (1.0, "s"),
        units::MSEC => (1e3, "s"),
        units::USEC => (1e6, "s"),
        units::HZ => (1.0, "Hz"),
        units::RADS => (1.0, "rad/s"),
        _ => (1.0, ""),
    };
    let rank = h.n_dims;
    if !(1..=7).contains(&rank) {
        bail!("NIfTI dim[0] = {rank} is out of range");
    }
    // nifti-reader-js reads pixdim as float32; exg widens the same values.
    let p = &h.pixdim;
    let units = [spatial_unit, spatial_unit, spatial_unit, time_unit, "", "", ""];
    let source_scales = [
        p[0] / spatial_inv_scale,
        p[1] / spatial_inv_scale,
        p[2] / spatial_inv_scale,
        p[3] / time_inv_scale,
        p[4],
        p[5],
        p[6],
    ];
    let view_scales = [
        1.0 / spatial_inv_scale,
        1.0 / spatial_inv_scale,
        1.0 / spatial_inv_scale,
        1.0 / time_inv_scale,
        1.0,
        1.0,
        1.0,
    ];
    let source_names = ["i", "j", "k", "m", "c^", "c1^", "c2^"];
    let view_names = ["x", "y", "z", "t", "c^", "c1^", "c2^"];

    let method2 = qform_transform(raw);
    let n = rank + 1;
    let mut transform = vec![0.0; n * n];
    for i in 0..n {
        transform[i * n + i] = 1.0;
    }
    let copy_rank = rank.min(3);
    for row in 0..copy_rank {
        for col in 0..copy_rank {
            transform[col * n + row] = method2[col * 4 + row] as f64;
        }
        transform[rank * n + row] = method2[12 + row] as f64;
    }

    Ok(NiftiVolumeInfo {
        rank,
        source_names: source_names[..rank].iter().map(|s| s.to_string()).collect(),
        view_names: view_names[..rank].iter().map(|s| s.to_string()).collect(),
        units: units[..rank].iter().map(|s| s.to_string()).collect(),
        source_scales: source_scales[..rank].to_vec(),
        view_scales: view_scales[..rank].to_vec(),
        data_type: convert_data_type(h.datatype),
        transform,
        description: description(raw),
        volume_size: h.dim[..rank].iter().map(|&d| d as u32).collect(),
    })
}

/// `descrip` as nifti-reader-js's `getStringAt` reads it: every byte of the
/// 80-byte field except NULs, each as a Latin-1 character, untrimmed. A C string
/// would stop at the first NUL; files do put text after one, and neuroglancer
/// shows it.
fn description(raw: &RawNifti) -> String {
    let offset = match raw.header.version {
        exg_nifti::NiftiVersion::Nifti1 => 148,
        exg_nifti::NiftiVersion::Nifti2 => 240,
    };
    raw.header_bytes
        .get(offset..offset + 80)
        .unwrap_or_default()
        .iter()
        .filter(|&&b| b != 0)
        .map(|&b| b as char)
        .collect()
}

/// NIfTI "method 2" (qform) as neuroglancer computes it:
/// `translationRotationScaleZReflectionToMat4` over gl-matrix's
/// `fromRotationTranslationScale`. Inputs are stored in `Float32Array`s and so
/// rounded to `f32`; the arithmetic is JavaScript's (`f64`); each output is
/// rounded to `f32` when stored. Column-major 4×4.
fn qform_transform(raw: &RawNifti) -> [f32; 16] {
    let h = &raw.header;
    let [b, c, d] = h.quatern;
    // nifti-reader-js hands over float32 values; `quatern_a` is computed in
    // f64 and rounded as `quat.fromValues` stores it.
    let a = (1.0 - b * b - c * c - d * d).sqrt();
    let [x, y, z, w] = [b, c, d, a].map(|v| v as f32 as f64);
    // `qfac` is -1 only for exactly -1: neuroglancer compares `pixDims[0]`.
    let qfac = if raw.pixdim0 == -1.0 { -1.0 } else { 1.0 };
    let (sx, sy, sz) = (1.0f32 as f64, 1.0f32 as f64, (1.0 * qfac) as f32 as f64);
    let v = h.qoffset.map(|o| o as f32);

    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    let f = |v: f64| v as f32;
    [
        f((1.0 - (yy + zz)) * sx),
        f((xy + wz) * sx),
        f((xz - wy) * sx),
        0.0,
        f((xy - wz) * sy),
        f((1.0 - (xx + zz)) * sy),
        f((yz + wx) * sy),
        0.0,
        f((xz + wy) * sz),
        f((yz - wx) * sz),
        f((1.0 - (xx + yy)) * sz),
        0.0,
        v[0],
        v[1],
        v[2],
        1.0,
    ]
}

/// Parses a `.nii` / `.nii.gz` and decodes its voxels the way
/// `NiftiVolumeChunkSource.download` does.
pub fn load(bytes: &[u8]) -> Result<NiftiVolume> {
    let raw = exg_nifti::parse_raw(bytes)?;
    let info = volume_info(&raw)?;
    let narrow_f64 = raw.header.datatype == NiftiType::F64;
    let mut data = raw.into_little_endian();
    if narrow_f64 {
        data = data
            .chunks_exact(8)
            .flat_map(|c| (f64::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes())
            .collect();
    }
    Ok(NiftiVolume { info, data })
}

impl NiftiVolumeInfo {
    /// The data source neuroglancer builds (`getDataSource` in
    /// `datasource/nifti/frontend.ts`) for the file at `url`.
    pub fn data_source(&self, url: &str) -> crate::model::DataSource {
        self.data_source_with(url, None)
    }

    fn data_source_with(&self, url: &str, data: Option<std::sync::Arc<Vec<u8>>>) -> crate::model::DataSource {
        use crate::model::*;
        use ng_core::coordinate_space::{BoundingBox, TransformedBoundingBox};
        use std::sync::Arc;
        let rank = self.rank;
        let size: Vec<f64> = self.volume_size.iter().map(|&s| s as f64).collect();
        let input = ng_core::CoordinateSpace::new(true, self.source_names.clone(), self.units.clone(), self.source_scales.clone())
            .with_bounding_boxes(vec![TransformedBoundingBox::identity(BoundingBox {
                lower_bounds: vec![0.0; rank],
                upper_bounds: size.clone(),
            })]);
        let output = ng_core::CoordinateSpace::new(true, self.view_names.clone(), self.units.clone(), self.view_scales.clone());
        let volume = MultiscaleVolume {
            data_type: self.data_type,
            volume_type: VolumeType::Unknown,
            rank,
            sources: vec![vec![VolumeSource {
                spec: VolumeChunkSpec {
                    rank,
                    data_type: self.data_type,
                    chunk_data_size: self.volume_size.clone(),
                    lower_voxel_bound: vec![0.0; rank],
                    upper_voxel_bound: size.iter().map(|&s| s as f32).collect(),
                    base_voxel_offset: vec![0.0; rank],
                    compressed_segmentation_block_size: None,
                },
                params: ChunkSourceParams::Nifti { url: url.to_owned(), data },
                chunk_to_multiscale_transform: {
                    let id = ng_core::matrix::identity(rank + 1, rank + 1);
                    id.iter().map(|&x| x as f32).collect()
                },
                lower_clip_bound: None,
                upper_clip_bound: None,
            }]],
        };
        let bounds = input.bounds.clone();
        DataSource::volume_with_bounds(
            format!("{url}|nifti:"),
            CoordinateSpaceTransform::new(Arc::new(input), Arc::new(output), self.transform.clone()),
            volume,
            &bounds,
        )
    }
}

impl NiftiVolume {
    /// The data source, with the voxels in memory as its one chunk.
    pub fn data_source(self, url: &str) -> crate::model::DataSource {
        self.info.data_source_with(url, Some(std::sync::Arc::new(self.data)))
    }
}

/// Reads the file at `location` (gzipped or not) and builds the data source.
pub async fn get(location: &ng_kvstore::KvStoreLocation, url: &str) -> anyhow::Result<crate::model::DataSource> {
    use anyhow::Context;
    let bytes = location
        .read("", ng_kvstore::ByteRange::Full)
        .await?
        .with_context(|| format!("no NIfTI file at {}", location.url()))?;
    Ok(load(&bytes)?.data_source(url))
}
