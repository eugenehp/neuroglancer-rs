//! Deep Zoom images (`datasource/deepzoom`): a `.dzi` file (`TileSize`,
//! `Overlap`, `Format`, `Size`) over a pyramid of JPEG or PNG tiles in
//! `<name>_files/<level>/<x>_<y>.<format>`, level 0 the 1 × 1 one. The model
//! space is x, y in nanometres and a channel dimension `c^` of 3; each
//! level a scale of `TileSize × TileSize × 3` chunks.

use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use ng_core::coordinate_space::{BoundingBox, TransformedBoundingBox};
use ng_core::{CoordinateSpace, DataType};
use ng_kvstore::{ByteRange, KvStoreLocation};

use crate::chunk::ChunkData;
use crate::model::{ChunkSourceParams, CoordinateSpaceTransform, DataSource, MultiscaleVolume, VolumeChunkSpec, VolumeSource, VolumeType};

/// `ImageTileEncoding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Jpg = 0,
    Jpeg = 1,
    Png = 2,
}

/// One level's tiles.
#[derive(Debug, Clone)]
pub struct ChunkParams {
    pub location: KvStoreLocation,
    pub encoding: Encoding,
    pub format: String,
    pub overlap: u32,
    pub tilesize: u32,
}

/// `DZIMetaData`.
#[derive(Debug, Clone, PartialEq)]
pub struct Metadata {
    pub width: u32,
    pub height: u32,
    pub tilesize: u32,
    pub overlap: u32,
    pub format: String,
}

/// An attribute's value in the first `<tag …>` of `xml`.
fn attribute(xml: &str, tag: &str, name: &str) -> Option<String> {
    let start = xml.find(&format!("<{tag}"))?;
    let element = &xml[start..start + xml[start..].find('>')?];
    let at = element.find(&format!(" {name}="))? + name.len() + 2;
    let quote = element[at..].chars().next()?;
    let rest = &element[at + 1..];
    Some(rest[..rest.find(quote)?].to_owned())
}

/// `getDZIMetadata`.
pub fn parse_metadata(text: &str) -> Result<Metadata> {
    let int = |tag: &str, name: &str, positive: bool| -> Result<u32> {
        let v = attribute(text, tag, name).with_context(|| format!("Expected string, but received: undefined ({tag} {name})"))?;
        let n: i64 = v.trim().parse().with_context(|| format!("Expected integer, but received: {v:?}"))?;
        ensure!(n >= 0 && (!positive || n > 0), "Expected positive integer, but received: {n}");
        Ok(n as u32)
    };
    Ok(Metadata {
        width: int("Size", "Width", true)?,
        height: int("Size", "Height", true)?,
        tilesize: int("Image", "TileSize", true)?,
        overlap: int("Image", "Overlap", false)?,
        format: attribute(text, "Image", "Format").context("Expected string, but received: undefined (Format)")?,
    })
}

/// Whether `prefix` is a Deep Zoom image (`detectFormat`).
pub fn detect(prefix: &[u8]) -> bool {
    let text = String::from_utf8_lossy(prefix);
    text.contains("<Image") && text.contains("http://schemas.microsoft.com/deepzoom/2009")
}

/// `DeepzoomDataSource.get` for the `.dzi` at `location`.
pub async fn get(location: &KvStoreLocation) -> Result<DataSource> {
    let url = location.url();
    ensure!(!url.ends_with(".json") && !url.contains(".json?"), "DZI-JSON: OpenSeadragon hack not supported yet.");
    let bytes = location.read("", ByteRange::Full).await?.with_context(|| format!("{url} not found"))?;
    let metadata = parse_metadata(&String::from_utf8_lossy(&bytes))?;
    data_source(location, &metadata)
}

/// `getImageDataSource`.
pub fn data_source(location: &KvStoreLocation, m: &Metadata) -> Result<DataSource> {
    let encoding = match m.format.to_ascii_uppercase().as_str() {
        "JPG" => Encoding::Jpg,
        "JPEG" => Encoding::Jpeg,
        "PNG" => Encoding::Png,
        _ => anyhow::bail!("Invalid enum value: {:?}.", m.format),
    };
    // `buildPyramidalImageInfo`: halving (rounding up) down to 1 × 1.
    let mut levels = vec![];
    let (mut w, mut h) = (m.width, m.height);
    while w > 1 || h > 1 {
        levels.push((w, h));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    levels.push((w, h));
    let rank = 3;
    let upper = [m.width as f64, m.height as f64, 3.0];
    let space = Arc::new(
        CoordinateSpace::new(true, vec!["x".into(), "y".into(), "c^".into()], vec!["m".into(), "m".into(), String::new()], vec![1e-9, 1e-9, 1.0])
            .with_bounding_boxes(vec![TransformedBoundingBox::identity(BoundingBox { lower_bounds: vec![0.0; rank], upper_bounds: upper.to_vec() })]),
    );
    // `<name>_files`, beside the `.dzi`.
    let base = match location.path.rfind('.') {
        Some(i) => location.path[..i].to_owned(),
        None => String::new(),
    };
    let files = KvStoreLocation { store: location.store.clone(), path: format!("{base}_files") };
    let n = rank + 1;
    let count = levels.len();
    let sources = levels
        .iter()
        .enumerate()
        .map(|(index, &(lw, lh))| {
            let relative = (1u64 << index) as f32;
            let mut transform = vec![0f32; n * n];
            transform[n * n - 1] = 1.0;
            transform[0] = relative;
            transform[n + 1] = relative;
            transform[2 * n + 2] = 1.0;
            VolumeSource {
                spec: VolumeChunkSpec {
                    rank,
                    data_type: DataType::Uint8,
                    chunk_data_size: vec![m.tilesize, m.tilesize, 3],
                    lower_voxel_bound: vec![0.0; rank],
                    upper_voxel_bound: vec![lw as f32, lh as f32, 3.0],
                    base_voxel_offset: vec![0.0; rank],
                    compressed_segmentation_block_size: None,
                },
                params: ChunkSourceParams::DeepZoom(ChunkParams { location: files.child(&format!("{}/", count - 1 - index)), encoding, format: m.format.clone(), overlap: m.overlap, tilesize: m.tilesize }),
                chunk_to_multiscale_transform: transform,
                lower_clip_bound: None,
                upper_clip_bound: Some(vec![(upper[0] / relative as f64) as f32, (upper[1] / relative as f64) as f32, 3.0]),
            }
        })
        .collect();
    let volume = MultiscaleVolume { data_type: DataType::Uint8, volume_type: VolumeType::Image, rank, sources: vec![sources] };
    Ok(DataSource::volume_with_bounds(
        format!("{}|deepzoom:", location.url()),
        CoordinateSpaceTransform::identity(space.clone()),
        volume,
        &space.bounds,
    ))
}

/// `DeepzoomImageTileSource.download`: tile `<x>_<y>` without its overlap,
/// planar, in a `TileSize² × 3` buffer, copied as neuroglancer copies it
/// (rows of the tile's width at the tile size's stride, so a wider tile's
/// row ends spill into the next row before it is written), of which the
/// chunk's clipped part is kept.
/// A missing tile is zeros.
pub(crate) async fn read_tile(source: &VolumeSource, p: &ChunkParams, grid: &[u64]) -> Result<ChunkData> {
    let (origin, size) = source.spec.chunk_bounds(grid);
    let numel: usize = size.iter().map(|&s| s as usize).product();
    let key = format!("{}_{}.{}", grid[0], grid[1], p.format);
    let Some(bytes) = p.location.read(&key, ByteRange::Full).await? else {
        return Ok(ChunkData { origin, size, data: vec![0; numel], memory_bytes: 0 });
    };
    let (tw, th, planar) = match p.encoding {
        Encoding::Png => {
            let mut decoder = png::Decoder::new(&bytes[..]);
            decoder.set_transformations(png::Transformations::IDENTITY);
            let mut reader = decoder.read_info().context("png: invalid header")?;
            let (w, h) = (reader.info().width as usize, reader.info().height as usize);
            ensure!(
                reader.info().color_type == png::ColorType::Rgb && reader.info().bit_depth == png::BitDepth::Eight,
                "png: Image decode parameters did not match expected chunk parameters.  Expected: width: undefined height: undefined area: undefined channels: 3 bytes per pixel: 1."
            );
            let mut image = vec![0; reader.output_buffer_size()];
            let frame = reader.next_frame(&mut image).context("png: Failed to decode png image.")?;
            image.truncate(frame.buffer_size());
            // `decompressPng` makes it planar, then the tile source
            // transposes it again (as though interleaved): reproduced.
            let planar = crate::chunk::transpose_2d(&image, w * h, 3);
            (w, h, crate::chunk::transpose_2d(&planar, w * h, 3))
        }
        Encoding::Jpg | Encoding::Jpeg => {
            // `decodeJpeg` with 3 components: planar.
            let image = crate::jpeg::decode(&bytes).context("JPEG decoding failed")?;
            let (w, h, c) = (image.width, image.height, image.num_components);
            ensure!(c == 3, "JPEG data does not have the expected number of components: components={c}, expected=3");
            (w, h, crate::chunk::transpose_2d(&image.data, w * h, 3))
        }
    };
    let ts = p.tilesize as usize;
    let t2 = ts * ts;
    let (ox, oy) = (if grid[0] == 0 { 0 } else { p.overlap as usize }, if grid[1] == 0 { 0 } else { p.overlap as usize });
    let twh = tw * th;
    let mut d = vec![0u8; t2 * 3];
    for k in 0..3 {
        for j in 0..th {
            for i in 0..tw {
                let to = i + j * ts + k * t2;
                if to < d.len() {
                    // Past the tile's end, `undefined`: stored as 0.
                    d[to] = planar.get(i + ox + (j + oy) * tw + k * twh).copied().unwrap_or(0);
                }
            }
        }
    }
    // The chunk's clipped part of the tile-size buffer.
    let (sx, sy) = (size[0] as usize, size[1] as usize);
    let mut data = Vec::with_capacity(numel);
    for k in 0..size[2] as usize {
        for j in 0..sy {
            data.extend_from_slice(&d[j * ts + k * t2..j * ts + k * t2 + sx]);
        }
    }
    // Its buffer is the whole tile's.
    Ok(ChunkData { origin, size, data, memory_bytes: t2 * 3 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dzi_parses_as_neuroglancer() {
        let text = r#"<?xml version="1.0" encoding="utf-8"?>
<Image TileSize="254" Overlap="1" Format="jpg" xmlns="http://schemas.microsoft.com/deepzoom/2009">
<Size Width="48023" Height="34907" />
</Image>"#;
        assert!(detect(text.as_bytes()));
        assert_eq!(parse_metadata(text).unwrap(), Metadata { width: 48023, height: 34907, tilesize: 254, overlap: 1, format: "jpg".into() });
    }
}
