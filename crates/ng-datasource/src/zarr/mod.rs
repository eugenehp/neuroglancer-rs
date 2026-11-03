//! Zarr v2/v3 arrays and OME-NGFF multiscales. Port of
//! `src/datasource/zarr/frontend.ts` and `metadata/parse.ts`, with the codec
//! work done by `exg_zarr::codec`.
//!
//! neuroglancer is stricter than the Zarr spec in places, and this follows it:
//! only its eight data types (no bool, int64, float16 or float64), only the
//! blosc, gzip, zlib and zstd v2 compressors, and integer fill values must be
//! JSON integers.

pub mod ome;

use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use exg_zarr::codec::{ArrayMetadata, ArrayToBytes, CodecChain, DataType as ZarrType};
use ng_core::coordinate_space::{BoundingBox, TransformedBoundingBox};
use ng_core::si_units::{all_si_prefixes, exp10};
use ng_core::{matrix, CoordinateSpace, DataType};
use ng_kvstore::{ByteRange, KvStoreLocation};
use serde_json::{Map, Value};

use crate::model::{
    ChunkSourceParams, CoordinateSpaceTransform, DataSource, MultiscaleVolume, VolumeChunkSpec, VolumeSource, VolumeType,
};

/// A Zarr array as neuroglancer sees it (`ArrayMetadata` in
/// `metadata/index.ts`), over exg's codec model.
#[derive(Debug, Clone)]
pub struct ZarrArray {
    pub codec: ArrayMetadata,
    pub data_type: DataType,
    pub dimension_names: Vec<Option<String>>,
    pub dimension_units: Vec<Option<String>>,
    /// Physical (stored) dimension order → logical dimension, outermost first
    /// (`codecs.layoutInfo[0].physicalToLogicalDimension`).
    pub physical_to_logical: Vec<usize>,
    /// The unit neuroglancer reads: the chunk, or the sub-chunk of a shard.
    pub read_chunk_shape: Vec<usize>,
}

/// A Zarr node: an array, or a group with its attributes.
#[derive(Debug, Clone)]
pub enum Node {
    Array(Box<ZarrArray>),
    Group(Map<String, Value>),
}

fn data_type(t: ZarrType) -> Result<DataType> {
    Ok(match t {
        ZarrType::Uint8 => DataType::Uint8,
        ZarrType::Int8 => DataType::Int8,
        ZarrType::Uint16 => DataType::Uint16,
        ZarrType::Int16 => DataType::Int16,
        ZarrType::Uint32 => DataType::Uint32,
        ZarrType::Int32 => DataType::Int32,
        ZarrType::Uint64 => DataType::Uint64,
        ZarrType::Float32 => DataType::Float32,
        other => bail!("Unsupported data type: {}", other.name()),
    })
}

/// `parseFillValue`: integers must be JSON integers; float32 also takes
/// "Infinity", "-Infinity", "NaN" and `0x…` bit patterns.
fn check_fill_value(t: DataType, v: &Value) -> Result<()> {
    match t {
        DataType::Float32 => match v {
            Value::Number(_) => Ok(()),
            Value::String(s) if matches!(s.as_str(), "Infinity" | "-Infinity" | "NaN") => Ok(()),
            Value::String(s) if s.len() > 2 && s.starts_with("0x") && s[2..].bytes().all(|b| b.is_ascii_hexdigit()) => Ok(()),
            _ => bail!("Expected number, \"Infinity\", \"-Infinity\", \"NaN\", or hex string but received: {v}"),
        },
        _ => match v.as_f64() {
            Some(f) if f.fract() == 0.0 && v.is_number() => Ok(()),
            _ => bail!("Expected integer but received: {v}"),
        },
    }
}

/// The physical dimension order and read shape of a codec chain, as
/// neuroglancer's codec `layoutInfo` computes them: the serializer sees the
/// array after the transposes (and a shard's layout is its sub-chunks').
fn layout(chain: &CodecChain, chunk_shape: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let (mut p2l, mut read) = match &chain.serializer {
        ArrayToBytes::Bytes { .. } => {
            let mut shape = chunk_shape.to_vec();
            for order in &chain.transposes {
                shape = order.iter().map(|&i| shape[i]).collect();
            }
            ((0..shape.len()).collect::<Vec<_>>(), shape)
        }
        ArrayToBytes::Sharding(spec) => layout(&spec.codecs, &spec.sub_chunk_shape),
    };
    // Undo the transposes, last first: encoded dimension e is decoded
    // dimension order[e].
    for order in chain.transposes.iter().rev() {
        p2l = p2l.iter().map(|&e| order[e]).collect();
        let mut decoded = vec![0; read.len()];
        for (e, &d) in order.iter().enumerate() {
            decoded[d] = read[e];
        }
        read = decoded;
    }
    (p2l, read)
}

fn optional_strings(v: Option<&Value>, rank: usize, what: &str) -> Result<Vec<Option<String>>> {
    match v {
        None | Some(Value::Null) => Ok(vec![None; rank]),
        Some(Value::Array(a)) => {
            ensure!(a.len() == rank, "Expected array of length {rank} for {what}");
            a.iter()
                .map(|x| match x {
                    Value::Null => Ok(None),
                    Value::String(s) => Ok(Some(s.clone())),
                    _ => bail!("Expected string or null in {what}, but received: {x}"),
                })
                .collect()
        }
        Some(other) => bail!("Expected array for {what}, but received: {other}"),
    }
}

impl ZarrArray {
    fn from_codec(codec: ArrayMetadata, raw: &Value, attrs: &Map<String, Value>) -> Result<Self> {
        let data_type = data_type(codec.data_type)?;
        let rank = codec.rank();
        match raw.get("fill_value") {
            None => bail!("Missing \"fill_value\" property"),
            Some(Value::Null) if codec.zarr_format == 2 => {}
            Some(v) => check_fill_value(data_type, v).context("Error parsing \"fill_value\" property")?,
        }
        if codec.zarr_format == 2 {
            if let Some(id) = raw.get("compressor").and_then(|c| c.get("id")).and_then(Value::as_str) {
                ensure!(matches!(id, "blosc" | "gzip" | "zlib" | "zstd"), "Unsupported compressor: {id:?}");
            }
        }
        let dimension_names = match codec.zarr_format {
            2 => optional_strings(attrs.get("_ARRAY_DIMENSIONS"), rank, "_ARRAY_DIMENSIONS")?,
            _ => optional_strings(raw.get("dimension_names"), rank, "dimension_names")?,
        };
        let dimension_units = optional_strings(attrs.get("dimension_units"), rank, "dimension_units")?;
        let (physical_to_logical, read_chunk_shape) = layout(&codec.codecs, &codec.chunk_shape);
        Ok(ZarrArray {
            codec,
            data_type,
            dimension_names,
            dimension_units,
            physical_to_logical,
            read_chunk_shape,
        })
    }

    pub fn rank(&self) -> usize {
        self.codec.rank()
    }

    pub fn zarr_version(&self) -> u8 {
        self.codec.zarr_format
    }
}

/// `parseV2Metadata` (`.zarray` and `.zattrs`).
pub fn parse_v2(zarray: &[u8], zattrs: Option<&[u8]>) -> Result<ZarrArray> {
    let raw: Value = serde_json::from_slice(zarray).context("parsing .zarray")?;
    let codec = ArrayMetadata::from_v2(zarray, zattrs).context("Error parsing zarr v2 metadata")?;
    let attrs = codec.attributes.clone();
    ZarrArray::from_codec(codec, &raw, &attrs).context("Error parsing zarr v2 metadata")
}

/// `parseV3Metadata` (`zarr.json`): an array or a group.
pub fn parse_v3(zarr_json: &[u8]) -> Result<Node> {
    let raw: Value = serde_json::from_slice(zarr_json).context("parsing zarr.json")?;
    let attrs = raw.get("attributes").and_then(Value::as_object).cloned().unwrap_or_default();
    if raw.get("node_type").and_then(Value::as_str) == Some("group") {
        return Ok(Node::Group(attrs));
    }
    let codec = ArrayMetadata::from_v3(zarr_json).context("Error parsing zarr v3 metadata")?;
    Ok(Node::Array(Box::new(ZarrArray::from_codec(codec, &raw, &attrs).context("Error parsing zarr v3 metadata")?)))
}

/// `UNITS` in `metadata/parse.ts`: `(unit, scale)`.
fn zarr_unit(name: &str) -> Option<(&'static str, f64)> {
    if name.is_empty() {
        return Some(("", 1.0));
    }
    named_unit(name, true)
}

/// The units zarr's `UNITS` and OME's `OME_UNITS` share, `(unit, scale)`:
/// imperial lengths, parsecs, minutes to days, and metres and seconds with
/// each SI prefix (`"micrometer"`; with `short`, also `"um"`).
pub(crate) fn named_unit(name: &str, short: bool) -> Option<(&'static str, f64)> {
    Some(match name {
        "angstrom" => ("m", 1e-10),
        "foot" => ("m", 0.3048),
        "inch" => ("m", 0.0254),
        "mile" => ("m", 1609.34),
        // neuroglancer's constant, digit for digit.
        #[allow(clippy::excessive_precision)]
        "parsec" => ("m", 3.0856775814913673e16),
        "yard" => ("m", 0.9144),
        "minute" => ("s", 60.0),
        "hour" => ("s", 3600.0),
        "day" => ("s", 86400.0),
        _ => {
            for (long, abbreviation, unit) in [("meter", "m", "m"), ("second", "s", "s")] {
                for p in all_si_prefixes() {
                    if name.strip_suffix(long) == Some(p.long_prefix) || (short && name.strip_suffix(abbreviation) == Some(p.prefix)) {
                        return Some((unit, exp10(p.exponent)));
                    }
                }
            }
            return None;
        }
    })
}

/// `parseDimensionUnit`: `"4 nm"` → `("m", 4e-9)`; null is unitless.
pub fn parse_dimension_unit(v: Option<&str>) -> Result<(&'static str, f64)> {
    let Some(s) = v else { return Ok(("", 1.0)) };
    let s = s.trim();
    // /^([-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?\d+)?)\s*(.*)/
    let b = s.as_bytes();
    let mut i = 0;
    if matches!(b.first(), Some(b'+' | b'-')) {
        i += 1;
    }
    let digits = |mut j: usize| {
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        j
    };
    let mut end = if b.get(i) == Some(&b'.') && b.get(i + 1).is_some_and(u8::is_ascii_digit) {
        Some(digits(i + 1))
    } else if b.get(i).is_some_and(u8::is_ascii_digit) {
        let e = digits(i);
        Some(if b.get(e) == Some(&b'.') { digits(e + 1) } else { e })
    } else {
        None
    };
    if let Some(e) = end {
        if matches!(b.get(e), Some(b'e' | b'E')) {
            let mut k = e + 1;
            if matches!(b.get(k), Some(b'+' | b'-')) {
                k += 1;
            }
            if b.get(k).is_some_and(u8::is_ascii_digit) {
                end = Some(digits(k));
            }
        }
    }
    let (scale, unit) = match end {
        None => (1.0, s),
        Some(e) => (
            s[..e].trim_end_matches('.').parse::<f64>().map_err(|_| anyhow!("bad number in unit {s:?}"))?,
            s[e..].trim_start(),
        ),
    };
    let (unit_name, unit_scale) = zarr_unit(unit).ok_or_else(|| anyhow!("Unsupported unit: {unit:?}"))?;
    Ok((unit_name, scale * unit_scale))
}

/// `getNormalizedDimensionNames`: unnamed dimensions become `d{i}` (v2) or
/// `dim_{i}` (v3); repeats get a numeric suffix.
pub fn normalized_dimension_names(names: &[Option<String>], zarr_version: u8) -> Vec<String> {
    let prefix = if zarr_version == 2 { "d" } else { "dim_" };
    let mut seen = std::collections::HashSet::new();
    names
        .iter()
        .enumerate()
        .map(|(i, name)| match name {
            None => {
                let mut j = i;
                loop {
                    let n = format!("{prefix}{j}");
                    if seen.insert(n.clone()) {
                        return n;
                    }
                    j += 1;
                }
            }
            Some(name) if seen.insert(name.clone()) => name.clone(),
            Some(name) => {
                let mut j = 1;
                loop {
                    let n = format!("{name}{j}");
                    if seen.insert(n.clone()) {
                        return n;
                    }
                    j += 1;
                }
            }
        })
        .collect()
}

/// One scale: where its array is, its metadata, and its transform into the
/// model space.
#[derive(Debug, Clone)]
pub struct ZarrScale {
    /// The array's directory.
    pub location: KvStoreLocation,
    pub transform: Vec<f64>,
    pub array: Arc<ZarrArray>,
}

/// `ZarrMultiscaleInfo`.
#[derive(Debug, Clone)]
pub struct ZarrMultiscale {
    pub coordinate_space: Arc<CoordinateSpace>,
    pub data_type: DataType,
    pub scales: Vec<ZarrScale>,
    pub base_transform: Vec<f64>,
}

/// `VolumeChunkSourceParameters` for Zarr: where the array is and how it is
/// laid out.
#[derive(Debug, Clone)]
pub struct ChunkParams {
    /// The array's directory.
    pub location: KvStoreLocation,
    pub array: Arc<ZarrArray>,
}

impl ChunkParams {
    /// The array's directory URL, ending in `/`.
    pub fn url(&self) -> String {
        self.location.url()
    }
}

/// `getMultiscaleInfoForSingleArray`.
pub fn single_array_multiscale(location: &KvStoreLocation, array: ZarrArray) -> Result<ZarrMultiscale> {
    let rank = array.rank();
    let names = normalized_dimension_names(&array.dimension_names, array.zarr_version());
    let units_scales = array
        .dimension_units
        .iter()
        .map(|u| parse_dimension_unit(u.as_deref()))
        .collect::<Result<Vec<_>>>()?;
    let space = CoordinateSpace::new(
        true,
        names,
        units_scales.iter().map(|(u, _)| u.to_string()).collect(),
        units_scales.iter().map(|(_, s)| *s).collect(),
    )
    .with_bounding_boxes(vec![TransformedBoundingBox::identity(BoundingBox {
        lower_bounds: vec![0.0; rank],
        upper_bounds: array.codec.shape.iter().map(|&s| s as f64).collect(),
    })]);
    let transform = matrix::identity(rank + 1, rank + 1);
    Ok(ZarrMultiscale {
        coordinate_space: Arc::new(space),
        data_type: array.data_type,
        scales: vec![ZarrScale {
            location: location.as_directory(),
            transform: transform.clone(),
            array: Arc::new(array),
        }],
        base_transform: transform,
    })
}

impl ZarrMultiscale {
    /// `MultiscaleVolumeChunkSource.getSources`: chunks in their physical
    /// dimension order, innermost first (neuroglancer's x-fastest layout),
    /// mapped onto the model space.
    pub fn volume(&self) -> MultiscaleVolume {
        let rank = self.coordinate_space.rank();
        let n = rank + 1;
        let scales = self
            .scales
            .iter()
            .map(|scale| {
                let a = &scale.array;
                let mut chunk = vec![0u32; rank];
                let mut data = vec![0f32; rank];
                let mut order = vec![0f32; n * n];
                order[n * n - 1] = 1.0;
                for i in 0..rank {
                    let d = a.physical_to_logical[rank - 1 - i];
                    chunk[i] = a.read_chunk_shape[d] as u32;
                    data[i] = a.codec.shape[d] as f32;
                    order[i + d * n] = 1.0;
                }
                // `matrix.multiply` into a Float32Array: f64 sums, f32 stores.
                let order64: Vec<f64> = order.iter().map(|&x| x as f64).collect();
                let transform: Vec<f32> = matrix::multiply(&scale.transform, &order64, n).iter().map(|&x| x as f32).collect();
                VolumeSource {
                    spec: VolumeChunkSpec {
                        rank,
                        data_type: self.data_type,
                        chunk_data_size: chunk,
                        lower_voxel_bound: vec![0.0; rank],
                        upper_voxel_bound: data,
                        base_voxel_offset: vec![0.0; rank],
                        compressed_segmentation_block_size: None,
                    },
                    params: ChunkSourceParams::Zarr(ChunkParams {
                        location: scale.location.clone(),
                        array: a.clone(),
                    }),
                    chunk_to_multiscale_transform: transform,
                    lower_clip_bound: None,
                    upper_clip_bound: None,
                }
            })
            .collect();
        MultiscaleVolume {
            data_type: self.data_type,
            volume_type: VolumeType::Image,
            rank,
            sources: vec![scales],
        }
    }

    pub fn data_source(&self, url: &str, zarr_version: u8) -> DataSource {
        DataSource::volume_with_bounds(
            format!("{url}|zarr{zarr_version}:"),
            CoordinateSpaceTransform::over(self.coordinate_space.clone(), self.base_transform.clone()),
            self.volume(),
            &self.coordinate_space.bounds,
        )
    }
}

async fn read_json(location: &KvStoreLocation, key: &str) -> Result<Option<Vec<u8>>> {
    location.read(key, ByteRange::Full).await
}

/// `getMetadata` with no version given: tries v2 and v3; both is an error.
pub async fn get_node(location: &KvStoreLocation) -> Result<Option<(Node, u8)>> {
    get_node_version(location, None).await
}

/// `getMetadata`: with a version, only that version's metadata is read (a
/// `zarr3://` source asks for no `.zarray`/`.zattrs`, which 404 on every scale).
pub async fn get_node_version(location: &KvStoreLocation, version: Option<u8>) -> Result<Option<(Node, u8)>> {
    let dir = location.as_directory();
    let v2 = version != Some(3);
    let v3 = version != Some(2);
    let none = || async { Ok(None) };
    let (zarray, zattrs, zarr_json) = futures::join!(
        async { if v2 { read_json(&dir, ".zarray").await } else { none().await } },
        async { if v2 { read_json(&dir, ".zattrs").await } else { none().await } },
        async { if v3 { read_json(&dir, "zarr.json").await } else { none().await } }
    );
    let (zarray, zattrs, zarr_json) = (zarray?, zattrs?, zarr_json?);
    let v2 = match (&zarray, &zattrs) {
        (Some(a), attrs) => Some(Node::Array(Box::new(parse_v2(a, attrs.as_deref())?))),
        (None, Some(attrs)) => {
            let v: Value = serde_json::from_slice(attrs).context("parsing .zattrs")?;
            Some(Node::Group(v.as_object().cloned().context(".zattrs is not an object")?))
        }
        (None, None) => None,
    };
    let v3 = zarr_json.as_deref().map(parse_v3).transpose()?;
    match (v2, v3) {
        (Some(_), Some(_)) => bail!("Both zarr v2 and v3 metadata found"),
        (Some(n), None) => Ok(Some((n, 2))),
        (None, Some(n)) => Ok(Some((n, 3))),
        (None, None) => Ok(None),
    }
}

/// `resolveOmeMultiscale`.
async fn resolve_ome(location: &KvStoreLocation, ome: ome::OmeMultiscale, zarr_version: u8) -> Result<ZarrMultiscale> {
    let base = location.as_directory();
    let mut arrays = vec![];
    let mut locations = vec![];
    for scale in &ome.scales {
        // `scale.url` is the base URL plus the dataset path.
        let relative = scale.url.strip_prefix(&base.url()).with_context(|| format!("{} is not under {}", scale.url, base.url()))?;
        let location = base.child(relative.trim_end_matches('/')).as_directory();
        // the group's own version: its arrays are the same
        let node = get_node_version(&location, Some(zarr_version)).await?;
        locations.push(location);
        match node {
            Some((Node::Array(a), v)) if v == zarr_version => arrays.push(*a),
            _ => bail!("zarr v{zarr_version} array metadata not found at {}", scale.url),
        }
    }
    let rank = ome.coordinate_space.rank();
    let data_type = arrays[0].data_type;
    for (scale, a) in ome.scales.iter().zip(&arrays) {
        ensure!(a.rank() == rank, "Expected zarr array at {:?} to have rank {rank}, but received: {}", scale.url, a.rank());
        ensure!(
            a.data_type == data_type,
            "Expected zarr array at {:?} to have data type {}, but received: {}",
            scale.url,
            data_type.name(),
            a.data_type.name()
        );
    }
    let n = rank + 1;
    let lower: Vec<f64> = (0..rank).map(|i| ome.scales[0].transform[n * rank + i]).collect();
    let upper: Vec<f64> = (0..rank).map(|i| lower[i] + arrays[0].codec.shape[i] as f64).collect();
    let mut space = CoordinateSpace::new(
        true,
        ome.coordinate_space.names.clone(),
        ome.coordinate_space.units.clone(),
        ome.coordinate_space.scales.clone(),
    );
    space = space.with_bounding_boxes(vec![TransformedBoundingBox::identity(BoundingBox {
        lower_bounds: lower,
        upper_bounds: upper,
    })]);
    Ok(ZarrMultiscale {
        coordinate_space: Arc::new(space),
        data_type,
        scales: ome
            .scales
            .into_iter()
            .zip(arrays)
            .zip(locations)
            .map(|((s, a), location)| ZarrScale {
                location,
                transform: s.transform,
                array: Arc::new(a),
            })
            .collect(),
        base_transform: ome.base_transform,
    })
}

/// `ZarrDataSource.get` for a directory: a single array, or an OME
/// multiscale group.
pub async fn get(location: &KvStoreLocation) -> Result<DataSource> {
    get_version(location, None).await
}

/// [`get`] for a known Zarr version (`zarr2://`, `zarr3://`).
pub async fn get_version(location: &KvStoreLocation, version: Option<u8>) -> Result<DataSource> {
    let dir = location.as_directory();
    let url = dir.url();
    let (node, version) = get_node_version(&dir, version).await?.context("No zarr metadata found")?;
    let multiscale = match node {
        Node::Array(a) => single_array_multiscale(&dir, *a)?,
        Node::Group(attrs) => {
            let ome = ome::parse(&url, &Value::Object(attrs), version)?
                .context("Neither array nor OME multiscale metadata found")?;
            resolve_ome(&dir, ome, version).await?
        }
    };
    Ok(multiscale.data_source(&url, version))
}
