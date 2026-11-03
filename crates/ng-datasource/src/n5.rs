//! N5 (`datasource/n5`): a dataset's `attributes.json`, merged with those of
//! the directories above it (`getAttributes`); a single-resolution array
//! (`dimensions`, `blockSize`, `dataType`, `compression`) or a multiscale
//! group of `s0`, `s1`, … (n5-viewer's `scales` or per-scale
//! `downsamplingFactors`), with `resolution`/`pixelResolution`, `units` and
//! `axes`; and its chunks (`decodeChunk`): a big-endian header (mode,
//! rank, shape) before the compressed big-endian elements, first dimension
//! fastest.

use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use ng_core::coordinate_space::{BoundingBox, TransformedBoundingBox};
use ng_core::js::{verify_array, verify_finite_positive_float, verify_positive_int};
use ng_core::{CoordinateSpace, DataType};
use ng_kvstore::{ByteRange, KvStoreLocation};
use serde_json::{Map, Value};

use crate::model::{ChunkSourceParams, CoordinateSpaceTransform, DataSource, MultiscaleVolume, VolumeChunkSpec, VolumeSource, VolumeType};

/// `VolumeChunkEncoding`: the `compression` type (`gzip` with `useZlib`:
/// zlib).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Raw,
    Zlib,
    Gzip,
    Blosc,
    Zstd,
}

impl Encoding {
    fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "raw" => Self::Raw,
            "zlib" => Self::Zlib,
            "gzip" => Self::Gzip,
            "blosc" => Self::Blosc,
            "zstd" => Self::Zstd,
            _ => return None,
        })
    }
}

/// One scale's chunks: its directory and encoding.
#[derive(Debug, Clone)]
pub struct ChunkParams {
    pub location: KvStoreLocation,
    pub encoding: Encoding,
}

/// `ScaleMetadata`.
#[derive(Debug, Clone)]
struct ScaleMetadata {
    location: KvStoreLocation,
    data_type: DataType,
    encoding: Encoding,
    size: Vec<f64>,
    chunk_size: Vec<u32>,
    downsampling_factors: Vec<f64>,
}

fn rank_check(rank: &mut Option<usize>, n: usize) -> Result<()> {
    match *rank {
        Some(r) if r != n => bail!("Rank mismatch, received {n} but expected {r}"),
        _ => *rank = Some(n),
    }
    Ok(())
}

/// `parseScaleMetadata`.
fn parse_scale_metadata(location: KvStoreLocation, o: &Map<String, Value>, scale_index: usize, factors: Option<Vec<f64>>) -> Result<ScaleMetadata> {
    let t = o.get("dataType").and_then(Value::as_str).context("dataType")?;
    let data_type = DataType::from_name(t).with_context(|| format!("Invalid enum value: {t:?}."))?;
    let size: Vec<f64> = verify_array(o.get("dimensions").context("dimensions")?)?.iter().map(|v| verify_positive_int(v).map(|n| n as f32 as f64)).collect::<Result<_>>()?;
    let block = verify_array(o.get("blockSize").context("blockSize")?)?;
    ensure!(block.len() == size.len(), "Expected array of length {}, but received: {:?}", size.len(), block);
    let chunk_size: Vec<u32> = block.iter().map(|v| verify_positive_int(v).map(|n| n as u32)).collect::<Result<_>>()?;
    let encoding = match o.get("compression").filter(|v| !v.is_null()) {
        Some(c) => {
            let t = c.get("type").and_then(Value::as_str).context("type")?;
            let e = Encoding::parse(t).with_context(|| format!("Invalid enum value: {t:?}."))?;
            if e == Encoding::Gzip && c.get("useZlib").and_then(Value::as_bool) == Some(true) {
                Encoding::Zlib
            } else {
                e
            }
        }
        None => {
            let t = o.get("compressionType").and_then(Value::as_str).context("compressionType")?;
            Encoding::parse(t).with_context(|| format!("Invalid enum value: {t:?}."))?
        }
    };
    let downsampling_factors = match factors {
        Some(f) => f,
        None => match o.get("downsamplingFactors").filter(|v| !v.is_null()) {
            Some(v) => {
                let a = verify_array(v)?;
                ensure!(a.len() == size.len(), "Expected array of length {}, but received: {v}", size.len());
                a.iter().map(verify_finite_positive_float).collect::<Result<_>>()?
            }
            None if scale_index == 0 => vec![1.0; size.len()],
            None => bail!("Expected downsamplingFactors attribute"),
        },
    };
    Ok(ScaleMetadata { location, data_type, encoding, size, chunk_size, downsampling_factors })
}

/// `getAttributes`: `attributes.json` of `dir` and each directory above it,
/// merged (deeper ones winning); the outermost directory holding one and
/// the path from it to `dir`. `None` if there is none.
async fn get_attributes(dir: &KvStoreLocation) -> Result<Option<(Map<String, Value>, String, String)>> {
    let full = dir.as_directory().path;
    let mut path = full.trim_end_matches('/').to_owned();
    let mut levels = vec![];
    loop {
        let directory = if path.is_empty() { String::new() } else { format!("{path}/") };
        let location = KvStoreLocation { store: dir.store.clone(), path: directory.clone() };
        levels.push((location, full[directory.len()..].to_owned()));
        if path.is_empty() {
            break;
        }
        path = path.rfind('/').map_or(String::new(), |i| path[..i].to_owned());
    }
    let reads = futures::future::join_all(levels.iter().map(|(l, _)| l.read("attributes.json", ByteRange::Full))).await;
    let mut metadata = vec![];
    for ((l, _), r) in levels.iter().zip(reads) {
        metadata.push(match r? {
            Some(bytes) => {
                let v: Value = serde_json::from_slice(&bytes).with_context(|| format!("Error reading attributes from {}", l.child("attributes.json").url()))?;
                Some(v.as_object().cloned().with_context(|| format!("Error reading attributes from {}: Expected object, but received: {v}", l.child("attributes.json").url()))?)
            }
            None => None,
        });
    }
    let Some(root) = metadata.iter().rposition(Option::is_some) else { return Ok(None) };
    let mut merged = Map::new();
    for m in metadata.into_iter().rev().flatten() {
        merged.extend(m);
    }
    let (root_location, relative) = &levels[root];
    Ok(Some((merged, root_location.url(), relative.clone())))
}

/// `getDefaultAxes`.
fn default_axes(rank: usize) -> Vec<String> {
    (0..rank).map(|i| ["x", "y", "z", "t", "c"].get(i).map_or_else(|| format!("d{}", i + 1), |s| (*s).to_owned())).collect()
}

/// An `n5`-style unit (`unitFromJson`): its base unit and exponent.
fn unit(v: &Value) -> Result<ng_core::si_units::ParsedUnit> {
    v.as_str().and_then(ng_core::si_units::parse_unit).with_context(|| format!("Invalid unit: {v}"))
}

/// `getMultiscaleMetadata`: the model space, and each scale's directory and
/// (if the group gives them) downsampling factors.
async fn multiscale_metadata(dir: &KvStoreLocation, a: &Map<String, Value>) -> Result<(CoordinateSpace, Vec<(KvStoreLocation, Option<Vec<f64>>)>)> {
    let mut rank = None;
    let floats = |v: &Value| -> Result<Vec<f64>> { verify_array(v)?.iter().map(verify_finite_positive_float).collect() };
    let mut scales = match a.get("resolution").filter(|v| !v.is_null()) {
        Some(v) => {
            let s = floats(v)?;
            rank_check(&mut rank, s.len())?;
            Some(s)
        }
        None => None,
    };
    let axes = match a.get("axes").filter(|v| !v.is_null()) {
        Some(v) => {
            let names: Vec<String> = verify_array(v)?.iter().map(|x| x.as_str().map(str::to_owned).with_context(|| format!("Expected string, but received: {x}"))).collect::<Result<_>>()?;
            rank_check(&mut rank, names.len())?;
            Some(names)
        }
        None => None,
    };
    let mut units = match a.get("units").filter(|v| !v.is_null()) {
        Some(v) => {
            let u: Vec<_> = verify_array(v)?.iter().map(unit).collect::<Result<_>>()?;
            rank_check(&mut rank, u.len())?;
            Some(u)
        }
        None => None,
    };
    let mut default_unit = ng_core::si_units::ParsedUnit { unit: "m", exponent: -9 };
    let mut single_factors = None;
    let mut all_factors: Option<Vec<Vec<f64>>> = None;
    let multi = |v: &Value, rank: &mut Option<usize>| -> Result<Vec<Vec<f64>>> {
        let a = verify_array(v)?;
        ensure!(!a.is_empty(), "Expected non-empty array");
        a.iter()
            .map(|f| {
                let f = floats(f)?;
                rank_check(rank, f.len())?;
                Ok(f)
            })
            .collect()
    };
    if let Some(v) = a.get("downsamplingFactors").filter(|v| !v.is_null()) {
        let arr = verify_array(v)?;
        ensure!(!arr.is_empty(), "Expected non-empty array");
        if arr[0].is_array() {
            all_factors = Some(multi(v, &mut rank)?);
        } else {
            let f = floats(v)?;
            rank_check(&mut rank, f.len())?;
            single_factors = Some(f);
        }
    }
    // n5-viewer's `pixelResolution` and `scales`.
    if let Some(r) = a.get("pixelResolution").filter(|v| !v.is_null()) {
        default_unit = unit(r.get("unit").context("unit")?)?;
        if let Some(d) = r.get("dimensions").filter(|v| !v.is_null()) {
            let s = floats(d)?;
            rank_check(&mut rank, s.len())?;
            scales = Some(s);
        }
    }
    if let Some(v) = a.get("scales").filter(|v| !v.is_null()) {
        all_factors = Some(multi(v, &mut rank)?);
    }
    let dimensions = match a.get("dimensions").filter(|v| !v.is_null()) {
        Some(v) => {
            let d: Vec<u64> = verify_array(v)?.iter().map(verify_positive_int).collect::<Result<_>>()?;
            rank_check(&mut rank, d.len())?;
            Some(d)
        }
        None => None,
    };
    let rank = rank.context("Unable to determine rank of dataset")?;
    let units = units.take().unwrap_or_else(|| vec![default_unit; rank]);
    let scales: Vec<f64> = scales.unwrap_or_else(|| vec![1.0; rank]).iter().zip(&units).map(|(&s, u)| ng_core::si_units::scale_by_exp10(s, u.exponent)).collect();
    let names = axes.unwrap_or_else(|| default_axes(rank));
    let space = CoordinateSpace::new(true, names, units.iter().map(|u| u.unit.to_owned()).collect(), scales);
    let dir = dir.as_directory();
    let levels = if dimensions.is_none() {
        match all_factors {
            Some(all) => all.into_iter().enumerate().map(|(i, f)| (dir.child(&format!("s{i}/")), Some(f))).collect(),
            None => {
                // `findScaleDirectories`, by asking for each `s<i>` in turn
                // (listing is not available over every kvstore here).
                let mut found = vec![];
                for i in 0.. {
                    let scale = dir.child(&format!("s{i}/"));
                    if scale.read("attributes.json", ByteRange::Full).await?.is_none() {
                        break;
                    }
                    found.push((scale, None));
                }
                ensure!(!found.is_empty(), "Not valid single-resolution or multi-resolution dataset");
                found
            }
        }
    } else {
        vec![(dir.clone(), Some(single_factors.unwrap_or_else(|| vec![1.0; rank])))]
    };
    Ok((space, levels))
}

/// `N5DataSource.get`.
pub async fn get(location: &KvStoreLocation) -> Result<DataSource> {
    let dir = location.as_directory();
    let (attributes, root_url, path_from_root) = get_attributes(&dir).await?.context("N5 metadata not found")?;
    let (space, levels) = multiscale_metadata(&dir, &attributes).await?;
    let rank = space.rank();
    // `getAllScales`.
    let mut scales = vec![];
    for (i, (scale_dir, factors)) in levels.into_iter().enumerate() {
        let Some((a, _, _)) = get_attributes(&scale_dir).await? else { continue };
        let url = scale_dir.url();
        scales.push(parse_scale_metadata(scale_dir, &a, i, factors).with_context(|| format!("Error parsing array metadata at {url}"))?);
    }
    let base = scales.first().context("At least one scale must be specified.")?;
    for (i, s) in scales.iter().enumerate() {
        ensure!(s.data_type == base.data_type, "Scale s{i} has data type {} but expected {}.", s.data_type.name().to_uppercase(), base.data_type.name().to_uppercase());
        ensure!(s.size.len() == rank, "Rank mismatch, received {} but expected {rank}", s.size.len());
    }
    // The bounds: the base scale's voxels through its downsampling factors.
    let mut transform = vec![0.0; rank * (rank + 1)];
    for i in 0..rank {
        transform[i * rank + i] = base.downsampling_factors[i];
    }
    let space = Arc::new(space.with_bounding_boxes(vec![TransformedBoundingBox { bounding_box: BoundingBox { lower_bounds: vec![0.0; rank], upper_bounds: base.size.clone() }, transform }]));
    let n = rank + 1;
    let sources = scales
        .iter()
        .map(|s| {
            let mut m = vec![0f32; n * n];
            m[n * n - 1] = 1.0;
            for i in 0..rank {
                m[i * n + i] = s.downsampling_factors[i] as f32;
            }
            VolumeSource {
                spec: VolumeChunkSpec {
                    rank,
                    data_type: s.data_type,
                    chunk_data_size: s.chunk_size.clone(),
                    lower_voxel_bound: vec![0.0; rank],
                    upper_voxel_bound: s.size.iter().map(|&x| x as f32).collect(),
                    base_voxel_offset: vec![0.0; rank],
                    compressed_segmentation_block_size: None,
                },
                params: ChunkSourceParams::N5(ChunkParams { location: s.location.clone(), encoding: s.encoding }),
                chunk_to_multiscale_transform: m,
                lower_clip_bound: None,
                upper_clip_bound: None,
            }
        })
        .collect();
    let volume = MultiscaleVolume { data_type: base.data_type, volume_type: VolumeType::Image, rank, sources: vec![sources] };
    Ok(DataSource::volume_with_bounds(
        format!("{root_url}|n5:{path_from_root}"),
        CoordinateSpaceTransform::identity(space.clone()),
        volume,
        &space.bounds,
    ))
}

/// `decodeChunk`: the chunk's shape (from its header) and its elements,
/// decompressed and made little-endian.
pub fn decode_chunk(bytes: &[u8], encoding: Encoding, data_type: DataType, rank: usize) -> Result<(Vec<u32>, Vec<u8>)> {
    let u16_at = |o: usize| -> Result<u16> { Ok(u16::from_be_bytes(bytes.get(o..o + 2).context("N5 chunk header truncated")?.try_into().unwrap())) };
    let mode = u16_at(0)?;
    ensure!(mode == 0, "Unsupported mode: {mode}.");
    let dims = u16_at(2)? as usize;
    ensure!(dims == rank, "Number of dimensions must be 3.");
    let mut offset = 4;
    let mut shape = Vec::with_capacity(dims);
    for _ in 0..dims {
        shape.push(u32::from_be_bytes(bytes.get(offset..offset + 4).context("N5 chunk header truncated")?.try_into().unwrap()));
        offset += 4;
    }
    let data = &bytes[offset..];
    let mut out = match encoding {
        Encoding::Raw => data.to_vec(),
        Encoding::Gzip | Encoding::Zlib => {
            use std::io::Read;
            let mut out = vec![];
            if encoding == Encoding::Gzip {
                flate2::read::GzDecoder::new(data).read_to_end(&mut out).context("gzip")?;
            } else {
                flate2::read::ZlibDecoder::new(data).read_to_end(&mut out).context("zlib")?;
            }
            out
        }
        Encoding::Blosc => exg_zarr::codec::blosc_decompress(data)?,
        Encoding::Zstd => exg_zarr::codec::zstd_decode(data)?,
    };
    let element = data_type.bytes();
    let numel: usize = shape.iter().map(|&s| s as usize).product();
    ensure!(
        out.len() == numel * element,
        "Raw-format chunk is {} bytes, but {numel} * {element} = {} bytes are expected.",
        out.len(),
        numel * element
    );
    // Big-endian elements (`decodeRawChunk` with `Endianness.BIG`).
    if element > 1 {
        for w in out.chunks_exact_mut(element) {
            w.reverse();
        }
    }
    Ok((shape, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_decode_as_neuroglancer() {
        let mut b = vec![0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1];
        b.extend([0x01, 0x02, 0x03, 0x04]);
        let (shape, data) = decode_chunk(&b, Encoding::Raw, DataType::Uint16, 2).unwrap();
        assert_eq!((shape, data), (vec![2, 1], vec![0x02, 0x01, 0x04, 0x03]));
        b[1] = 1;
        assert!(decode_chunk(&b, Encoding::Raw, DataType::Uint16, 2).is_err());
        assert_eq!(default_axes(7), ["x", "y", "z", "t", "c", "d6", "d7"]);
    }
}
