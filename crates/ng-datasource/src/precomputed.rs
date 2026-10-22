//! Neuroglancer precomputed volumes: the `info` file and chunk addressing.
//! Port of the volume parts of `src/datasource/precomputed/frontend.ts` and
//! `backend.ts`. Chunk decoding is in `crate::decode`.

use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use ng_core::coordinate_space::{BoundingBox, TransformedBoundingBox};
use ng_core::{CoordinateSpace, DataType};
use serde_json::{json, Value};

use crate::model::{
    ChunkSourceParams, CoordinateSpaceTransform, DataSource, MultiscaleVolume, VolumeChunkSpec, VolumeSource, VolumeType,
};

/// `VolumeChunkEncoding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Raw = 0,
    Jpeg = 1,
    CompressedSegmentation = 2,
    Compresso = 3,
    Png = 4,
    Jxl = 5,
    Crackle = 6,
}

/// `DataEncoding` (sharding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataEncoding {
    Raw = 0,
    Gzip = 1,
}

/// `ShardingHashFunction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardingHash {
    Identity = 0,
    MurmurHash3X86_128 = 1,
}

/// `ShardingParameters` (`neuroglancer_uint64_sharded_v1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardingParameters {
    pub hash: ShardingHash,
    pub preshift_bits: u32,
    pub minishard_bits: u32,
    pub shard_bits: u32,
    pub minishard_index_encoding: DataEncoding,
    pub data_encoding: DataEncoding,
}

/// `VolumeChunkSourceParameters`.
#[derive(Debug, Clone)]
pub struct ChunkParams {
    /// The scale's directory.
    pub location: ng_kvstore::KvStoreLocation,
    pub encoding: Encoding,
    pub sharding: Option<ShardingParameters>,
}

impl ChunkParams {
    /// The scale's directory URL, ending in `/`.
    pub fn url(&self) -> String {
        self.location.url()
    }

    pub fn snapshot(&self) -> Value {
        json!({
            "encoding": self.encoding as u8,
            "sharding": self.sharding.map(|s| json!({
                "hash": s.hash as u8,
                "preshiftBits": s.preshift_bits,
                "minishardBits": s.minishard_bits,
                "shardBits": s.shard_bits,
                "minishardIndexEncoding": s.minishard_index_encoding as u8,
                "dataEncoding": s.data_encoding as u8,
            })),
            "url": self.url(),
        })
    }

    /// The key of the unsharded chunk at `origin` (voxels) with `size`:
    /// `x0-x1_y0-y1_z0-z1`.
    pub fn chunk_key(origin: &[f64], size: &[u32]) -> String {
        (0..3)
            .map(|i| format!("{}-{}", origin[i], origin[i] + size[i] as f64))
            .collect::<Vec<_>>()
            .join("_")
    }
}

/// `verifyEnumString`: case-insensitive, must start with a letter.
fn enum_str<'a>(v: &'a Value, what: &str) -> Result<&'a str> {
    v.as_str()
        .filter(|s| s.starts_with(|c: char| c.is_ascii_alphabetic()))
        .ok_or_else(|| anyhow!("Invalid enum value for {what}: {v}."))
}

fn parse_encoding(v: &Value) -> Result<Encoding> {
    Ok(match enum_str(v, "encoding")?.to_ascii_uppercase().as_str() {
        "RAW" => Encoding::Raw,
        "JPEG" => Encoding::Jpeg,
        "COMPRESSED_SEGMENTATION" => Encoding::CompressedSegmentation,
        "COMPRESSO" => Encoding::Compresso,
        "PNG" => Encoding::Png,
        "JXL" => Encoding::Jxl,
        "CRACKLE" => Encoding::Crackle,
        _ => bail!("Invalid enum value: {v}."),
    })
}

fn parse_data_encoding(v: Option<&Value>) -> Result<DataEncoding> {
    match v {
        None | Some(Value::Null) => Ok(DataEncoding::Raw),
        Some(v) => Ok(match enum_str(v, "encoding")?.to_ascii_uppercase().as_str() {
            "RAW" => DataEncoding::Raw,
            "GZIP" => DataEncoding::Gzip,
            _ => bail!("Invalid enum value: {v}."),
        }),
    }
}

fn int(v: &Value, what: &str) -> Result<i64> {
    v.as_f64()
        .filter(|f| f.fract() == 0.0)
        .map(|f| f as i64)
        .ok_or_else(|| anyhow!("Expected integer for {what}, but received: {v}"))
}

/// `parseShardingParameters`.
pub fn parse_sharding(v: Option<&Value>) -> Result<Option<ShardingParameters>> {
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let o = v.as_object().context("sharding is not an object")?;
    let t = o.get("@type").and_then(Value::as_str).unwrap_or_default();
    ensure!(t == "neuroglancer_uint64_sharded_v1", "Unsupported sharding format: {t:?}");
    let hash = match enum_str(o.get("hash").unwrap_or(&Value::Null), "hash")?.to_ascii_uppercase().as_str() {
        "IDENTITY" => ShardingHash::Identity,
        "MURMURHASH3_X86_128" => ShardingHash::MurmurHash3X86_128,
        other => bail!("Invalid enum value: {other:?}."),
    };
    let bits = |k: &str| -> Result<u32> {
        Ok(int(o.get(k).with_context(|| format!("sharding missing {k}"))?, k)? as u32)
    };
    Ok(Some(ShardingParameters {
        hash,
        preshift_bits: bits("preshift_bits")?,
        shard_bits: bits("shard_bits")?,
        minishard_bits: bits("minishard_bits")?,
        minishard_index_encoding: parse_data_encoding(o.get("minishard_index_encoding"))?,
        data_encoding: parse_data_encoding(o.get("data_encoding"))?,
    }))
}

/// `ScaleInfo`.
#[derive(Debug, Clone)]
pub struct ScaleInfo {
    pub key: String,
    pub encoding: Encoding,
    pub resolution: Vec<f64>,
    /// `Float32Array`s in the TS.
    pub voxel_offset: Vec<f32>,
    pub size: Vec<f32>,
    pub chunk_sizes: Vec<Vec<u32>>,
    pub compressed_segmentation_block_size: Option<[u32; 3]>,
    pub sharding: Option<ShardingParameters>,
    pub hidden: bool,
}

impl ScaleInfo {
    fn parse(v: &Value, num_channels: u32) -> Result<Self> {
        let o = v.as_object().context("scale is not an object")?;
        let rank = if num_channels == 1 { 3 } else { 4 };
        let vec3 = |k: &str, f: &dyn Fn(&Value) -> Result<f64>| -> Result<Vec<f64>> {
            let a = o.get(k).and_then(Value::as_array).with_context(|| format!("scale `{k}` is not an array"))?;
            ensure!(a.len() == 3, "scale `{k}` must have 3 elements");
            a.iter().map(f).collect()
        };
        let mut resolution = vec3("resolution", &|x| ng_core::js::verify_finite_positive_float(x))?;
        let mut voxel_offset: Vec<f32> = match o.get("voxel_offset") {
            None | Some(Value::Null) => vec![0.0; 3],
            Some(_) => vec3("voxel_offset", &|x| int(x, "voxel_offset").map(|i| i as f64))?
                .into_iter()
                .map(|x| x as f32)
                .collect(),
        };
        let mut size: Vec<f32> = vec3("size", &|x| {
            let i = int(x, "size")?;
            ensure!(i > 0, "size must be positive");
            Ok(i as f64)
        })?
        .into_iter()
        .map(|x| x as f32)
        .collect();
        if rank == 4 {
            resolution.push(1.0);
            voxel_offset.push(0.0);
            size.push(num_channels as f32);
        }
        let chunk_sizes = o
            .get("chunk_sizes")
            .and_then(Value::as_array)
            .context("`chunk_sizes` is not an array")?
            .iter()
            .map(|c| {
                let a = c.as_array().filter(|a| a.len() == 3).context("chunk size must have 3 elements")?;
                let mut v = a
                    .iter()
                    .map(|x| {
                        let i = int(x, "chunk_sizes")?;
                        ensure!(i > 0, "chunk size must be positive");
                        Ok(i as u32)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if rank == 4 {
                    v.push(num_channels);
                }
                Ok(v)
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(!chunk_sizes.is_empty(), "No chunk sizes specified.");
        let sharding = parse_sharding(o.get("sharding"))?;
        ensure!(sharding.is_none() || chunk_sizes.len() == 1, "Sharding requires a single chunk size per scale");
        let encoding = parse_encoding(o.get("encoding").unwrap_or(&Value::Null))?;
        let compressed_segmentation_block_size = if encoding == Encoding::CompressedSegmentation {
            let a = o
                .get("compressed_segmentation_block_size")
                .and_then(Value::as_array)
                .filter(|a| a.len() == 3)
                .context("compressed_segmentation_block_size must have 3 elements")?;
            let mut b = [0u32; 3];
            for (i, x) in a.iter().enumerate() {
                let n = int(x, "compressed_segmentation_block_size")?;
                ensure!(n > 0, "block size must be positive");
                b[i] = n as u32;
            }
            Some(b)
        } else {
            None
        };
        Ok(ScaleInfo {
            key: o.get("key").and_then(Value::as_str).context("scale missing `key`")?.to_owned(),
            encoding,
            resolution,
            voxel_offset,
            size,
            chunk_sizes,
            compressed_segmentation_block_size,
            sharding,
            hidden: o.get("hidden").and_then(Value::as_bool).unwrap_or(false),
        })
    }
}

/// `MultiscaleVolumeInfo`.
#[derive(Debug, Clone)]
pub struct MultiscaleVolumeInfo {
    pub data_type: DataType,
    pub volume_type: VolumeType,
    pub mesh: Option<String>,
    pub skeletons: Option<String>,
    pub segment_property_map: Option<String>,
    pub scales: Vec<ScaleInfo>,
    pub model_space: Arc<CoordinateSpace>,
}

/// `parseMultiscaleVolumeInfo`.
pub fn parse_info(v: &Value) -> Result<MultiscaleVolumeInfo> {
    let o = v.as_object().context("info is not an object")?;
    let data_type = DataType::from_name(enum_str(o.get("data_type").unwrap_or(&Value::Null), "data_type")?)
        .with_context(|| format!("Invalid enum value: {}.", o["data_type"]))?;
    let num_channels = int(o.get("num_channels").context("info missing num_channels")?, "num_channels")?;
    ensure!(num_channels > 0, "num_channels must be positive");
    let num_channels = num_channels as u32;
    let volume_type = match enum_str(o.get("type").unwrap_or(&Value::Null), "type")?.to_ascii_uppercase().as_str() {
        "UNKNOWN" => VolumeType::Unknown,
        "IMAGE" => VolumeType::Image,
        "SEGMENTATION" => VolumeType::Segmentation,
        other => bail!("Invalid enum value: {other:?}."),
    };
    let opt_str = |k: &str| o.get(k).and_then(Value::as_str).map(str::to_owned);
    let scales = o
        .get("scales")
        .and_then(Value::as_array)
        .context("`scales` is not an array")?
        .iter()
        .map(|s| ScaleInfo::parse(s, num_channels))
        .collect::<Result<Vec<_>>>()?;
    ensure!(!scales.is_empty(), "Expected at least one scale");
    let base = &scales[0];
    let rank = if num_channels == 1 { 3 } else { 4 };
    let mut names: Vec<String> = ["x", "y", "z"].map(String::from).to_vec();
    let mut units: Vec<String> = vec!["m".into(); 3];
    let mut scale_values = vec![0.0; rank];
    let mut lower = vec![0.0; rank];
    let mut upper = vec![0.0; rank];
    for i in 0..3 {
        scale_values[i] = base.resolution[i] / 1e9;
        lower[i] = base.voxel_offset[i] as f64;
        upper[i] = lower[i] + base.size[i] as f64;
    }
    if rank == 4 {
        scale_values[3] = 1.0;
        upper[3] = num_channels as f64;
        names.push("c^".into());
        units.push(String::new());
    }
    let model_space = CoordinateSpace::new(true, names, units, scale_values).with_bounding_boxes(vec![
        TransformedBoundingBox::identity(BoundingBox {
            lower_bounds: lower,
            upper_bounds: upper,
        }),
    ]);
    Ok(MultiscaleVolumeInfo {
        data_type,
        volume_type,
        mesh: opt_str("mesh"),
        skeletons: opt_str("skeletons"),
        segment_property_map: opt_str("segment_properties"),
        scales,
        model_space: Arc::new(model_space),
    })
}

impl MultiscaleVolumeInfo {
    /// `PrecomputedMultiscaleVolumeChunkSource.getSources`, for the volume in
    /// directory `dir`.
    pub fn volume(&self, dir: &ng_kvstore::KvStoreLocation) -> MultiscaleVolume {
        let rank = self.model_space.rank();
        let model_resolution = &self.scales[0].resolution;
        let base_box = &self.model_space.bounding_boxes[0].bounding_box;
        let n = rank + 1;
        // One list per scale, then transposed to one per chunk-size alternative.
        let per_scale: Vec<Vec<VolumeSource>> = self
            .scales
            .iter()
            .filter(|s| !s.hidden)
            .map(|scale| {
                let mut transform = vec![0f32; n * n];
                transform[n * n - 1] = 1.0;
                let mut lower_clip = vec![0f32; rank];
                let mut upper_clip = vec![0f32; rank];
                for i in 0..3 {
                    let relative = scale.resolution[i] / model_resolution[i];
                    transform[n * i + i] = relative as f32;
                    let offset = scale.voxel_offset[i] as f64;
                    transform[n * rank + i] = (offset * relative) as f32;
                    lower_clip[i] = (base_box.lower_bounds[i] / relative - offset) as f32;
                    upper_clip[i] = (base_box.upper_bounds[i] / relative - offset) as f32;
                }
                if rank == 4 {
                    transform[n * 3 + 3] = 1.0;
                    lower_clip[3] = base_box.lower_bounds[3] as f32;
                    upper_clip[3] = base_box.upper_bounds[3] as f32;
                }
                scale
                    .chunk_sizes
                    .iter()
                    .map(|chunk_size| VolumeSource {
                        spec: VolumeChunkSpec {
                            rank,
                            data_type: self.data_type,
                            chunk_data_size: chunk_size.clone(),
                            lower_voxel_bound: vec![0.0; rank],
                            upper_voxel_bound: scale.size.clone(),
                            base_voxel_offset: scale.voxel_offset.clone(),
                            compressed_segmentation_block_size: scale.compressed_segmentation_block_size,
                        },
                        params: ChunkSourceParams::Precomputed(ChunkParams {
                            location: dir.child(scale.key.trim_end_matches('/')).as_directory(),
                            encoding: scale.encoding,
                            sharding: scale.sharding,
                        }),
                        chunk_to_multiscale_transform: transform.clone(),
                        lower_clip_bound: Some(lower_clip.clone()),
                        upper_clip_bound: Some(upper_clip.clone()),
                    })
                    .collect()
            })
            .collect();
        let alternatives = per_scale.iter().map(Vec::len).max().unwrap_or(0);
        let sources = (0..alternatives)
            .map(|a| per_scale.iter().filter_map(|s| s.get(a).cloned()).collect())
            .collect();
        MultiscaleVolume {
            data_type: self.data_type,
            volume_type: self.volume_type,
            rank,
            sources,
        }
    }

    /// The data source for the volume in directory `dir`.
    pub fn data_source(&self, dir: &ng_kvstore::KvStoreLocation) -> DataSource {
        let dir = dir.as_directory();
        let url = dir.url();
        let rank = self.model_space.rank();
        let transform = CoordinateSpaceTransform {
            rank,
            source_rank: rank,
            input_space: self.model_space.clone(),
            output_space: self.model_space.clone(),
            transform: ng_core::matrix::identity(rank + 1, rank + 1),
        };
        DataSource::volume_with_bounds(
            format!("{url}|neuroglancer-precomputed:"),
            transform,
            self.volume(&dir),
            &self.model_space.bounds,
        )
    }
}

/// Reads `info` from `location` and builds the data source.
pub async fn get(location: &ng_kvstore::KvStoreLocation) -> Result<DataSource> {
    let dir = location.as_directory();
    let bytes = dir
        .read("info", ng_kvstore::ByteRange::Full)
        .await?
        .with_context(|| format!("no precomputed info at {}", dir.child("info").url()))?;
    let info: Value = serde_json::from_slice(&bytes).context("parsing precomputed info")?;
    match info.get("@type").and_then(Value::as_str) {
        Some("neuroglancer_skeletons") => return skeletons_data_source(&dir, &info),
        Some("neuroglancer_legacy_mesh" | "neuroglancer_multilod_draco") => return mesh_data_source(&dir, &info),
        Some("neuroglancer_annotations_v1") => return annotation_data_source(&dir, &info),
        _ => {}
    }
    let info = parse_info(&info)?;
    let mut source = info.data_source(&dir);
    if let Some(mesh) = &info.mesh {
        // `getMeshSource`: a directory whose `info`, if any, names its
        // format (none: the legacy single-resolution one).
        let location = dir.child(mesh).as_directory();
        let meta = match location.read("info", ng_kvstore::ByteRange::Full).await {
            Ok(Some(bytes)) => serde_json::from_slice::<Value>(&bytes).ok(),
            _ => None,
        };
        let kind = meta.as_ref().and_then(|v| v.get("@type")).and_then(Value::as_str).unwrap_or("neuroglancer_legacy_mesh");
        let multiscale = match kind {
            "neuroglancer_legacy_mesh" => Ok(None),
            "neuroglancer_multilod_draco" => crate::multiscale_mesh::parse_info(meta.as_ref().unwrap()).map(Some),
            other => Err(anyhow::anyhow!("Unsupported mesh type: {other:?}")),
        };
        if let Ok(multiscale) = multiscale {
            // `getSubsourceToModelSubspaceTransform` (nanometres to voxels of
            // the first scale) times the mesh's own transform.
            let to_model = match &multiscale {
                Some(m) => info.subsource_to_model(&m.transform),
                None => info.subsource_to_model(&IDENTITY4),
            };
            source.subsources.push(crate::model::SubsourceEntry {
                id: "mesh".into(),
                default: true,
                subsource: crate::model::Subsource::Mesh(Box::new(crate::mesh::MeshSource { location, to_model, multiscale })),
            });
        }
    }
    if let Some(skeletons) = &info.skeletons {
        // `getSkeletonSource`: the info is required; failures leave the
        // subsource out, as neuroglancer reports and skips them.
        let location = dir.child(skeletons).as_directory();
        let meta = location.read("info", ng_kvstore::ByteRange::Full).await;
        let parsed = match meta {
            Ok(Some(bytes)) => serde_json::from_slice::<Value>(&bytes).map_err(anyhow::Error::from).and_then(|v| crate::skeleton::parse_info(&v)),
            Ok(None) => Err(anyhow::anyhow!("no skeleton info at {}", location.child("info").url())),
            Err(e) => Err(e),
        };
        if let Ok(skeleton_info) = parsed {
            let to_model = info.subsource_to_model(&skeleton_info.transform);
            source.subsources.push(crate::model::SubsourceEntry {
                id: "skeletons".into(),
                default: true,
                subsource: crate::model::Subsource::Skeleton(Box::new(crate::skeleton::SkeletonSource { location, info: skeleton_info, to_model })),
            });
        }
    }
    Ok(source)
}

/// `getDefaultCoordinateSpace`: x, y, z in nanometres, unbounded.
fn object_data_source(dir: &ng_kvstore::KvStoreLocation, subsource: crate::model::Subsource) -> DataSource {
    let space = Arc::new(CoordinateSpace::new(true, ["x", "y", "z"].map(String::from).to_vec(), vec!["m".into(); 3], vec![1e-9; 3]));
    DataSource {
        canonical_url: format!("{}|neuroglancer-precomputed:", dir.url()),
        model_transform: CoordinateSpaceTransform { rank: 3, source_rank: 3, input_space: space.clone(), output_space: space, transform: ng_core::matrix::identity(4, 4) },
        subsources: vec![crate::model::SubsourceEntry { id: "default".into(), default: true, subsource }],
    }
}

/// `getSkeletonsDataSource`: a skeleton directory on its own, its
/// transform to the default space.
fn skeletons_data_source(dir: &ng_kvstore::KvStoreLocation, info: &Value) -> Result<DataSource> {
    let info = crate::skeleton::parse_info(info)?;
    let to_model = info.transform;
    Ok(object_data_source(dir, crate::model::Subsource::Skeleton(Box::new(crate::skeleton::SkeletonSource { location: dir.clone(), info, to_model }))))
}

/// `getAnnotationDataSource`: the collection's own coordinate space (its
/// bounds), the identity onto it.
fn annotation_data_source(dir: &ng_kvstore::KvStoreLocation, info: &Value) -> Result<DataSource> {
    let info = crate::annotation::parse_info(dir, info)?;
    let rank = info.rank;
    let space = Arc::new(CoordinateSpace::new(true, info.names.clone(), info.units.clone(), info.scales.clone()).with_bounding_boxes(vec![TransformedBoundingBox::identity(BoundingBox { lower_bounds: info.lower_bound.clone(), upper_bounds: info.upper_bound.clone() })]));
    Ok(DataSource {
        canonical_url: format!("{}|neuroglancer-precomputed:", dir.url()),
        model_transform: CoordinateSpaceTransform { rank, source_rank: rank, input_space: space.clone(), output_space: space, transform: ng_core::matrix::identity(rank + 1, rank + 1) },
        subsources: vec![crate::model::SubsourceEntry { id: "default".into(), default: true, subsource: crate::model::Subsource::Annotations(Box::new(info)) }],
    })
}

/// `getMeshDataSource`: a mesh directory on its own.
fn mesh_data_source(dir: &ng_kvstore::KvStoreLocation, info: &Value) -> Result<DataSource> {
    let multiscale = match info.get("@type").and_then(Value::as_str) {
        Some("neuroglancer_multilod_draco") => Some(crate::multiscale_mesh::parse_info(info)?),
        _ => None,
    };
    let to_model = multiscale.as_ref().map_or(IDENTITY4, |m| m.transform);
    Ok(object_data_source(dir, crate::model::Subsource::Mesh(Box::new(crate::mesh::MeshSource { location: dir.clone(), to_model, multiscale }))))
}

const IDENTITY4: [f64; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

impl MultiscaleVolumeInfo {
    /// `getSubsourceToModelSubspaceTransform` (nanometres to voxels of the
    /// first scale) times `transform` (column-major 4×4).
    fn subsource_to_model(&self, transform: &[f64; 16]) -> [f64; 16] {
        let mut scale = [0.0; 16];
        scale[15] = 1.0;
        for i in 0..3 {
            scale[5 * i] = 1.0 / self.scales[0].resolution[i];
        }
        let mut out = [0.0; 16];
        for c in 0..4 {
            for r in 0..4 {
                out[c * 4 + r] = (0..4).map(|k| scale[k * 4 + r] * transform[c * 4 + k]).sum();
            }
        }
        out
    }
}
