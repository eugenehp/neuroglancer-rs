//! Reading one chunk of a volume source: fetch it, decode it, and lay it out
//! as neuroglancer holds chunk data — the spec's dimensions, first fastest
//! (x-fastest), little-endian, in the volume's data type.
//!
//! Ports `download` of `src/datasource/precomputed/backend.ts` and
//! `src/datasource/zarr/backend.ts`, and the decoders in
//! `src/sliceview/backend_chunk_decoders/`. Where neuroglancer keeps
//! compressed segmentation compressed on the GPU, this decodes it to plain
//! values.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::decode::DecodeJob;
use anyhow::{bail, ensure, Context, Result};
use exg_zarr::codec::{IndexRange, ShardIndex};
use ng_kvstore::ByteRange;

use crate::model::{ChunkSourceParams, VolumeSource};
use crate::precomputed::Encoding;

/// One decoded chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkData {
    /// Voxel origin of the chunk in its source.
    pub origin: Vec<f64>,
    /// Size along each spec dimension; edge chunks may be smaller than the
    /// spec's chunk size.
    pub size: Vec<u32>,
    /// `size` elements, first dimension fastest, little-endian.
    pub data: Vec<u8>,
    /// What neuroglancer holds for it (`gpuMemoryBytes`): the data, but a
    /// compressed segmentation chunk as it came (neuroglancer decodes it in
    /// its shader), and nothing for a chunk never written.
    pub memory_bytes: usize,
}

/// Reads the chunk at grid position `grid` of `source`. A chunk that was
/// never written comes back filled with the source's fill value (zero for
/// precomputed), as neuroglancer renders it.
pub async fn read_chunk(source: &VolumeSource, grid: &[u64]) -> Result<ChunkData> {
    match &source.params {
        ChunkSourceParams::Precomputed(p) => read_precomputed(source, p, grid).await,
        ChunkSourceParams::Zarr(p) => read_zarr(source, p, grid).await,
        ChunkSourceParams::N5(p) => read_n5(source, p, grid).await,
        ChunkSourceParams::DeepZoom(p) => crate::deepzoom::read_tile(source, p, grid).await,
        ChunkSourceParams::Nifti { data, .. } => {
            let data = data.as_ref().context("NIfTI source without its data")?;
            ensure!(grid.iter().all(|&g| g == 0), "a NIfTI volume is a single chunk");
            Ok(ChunkData {
                origin: vec![0.0; grid.len()],
                size: source.spec.chunk_data_size.clone(),
                data: data.as_ref().clone(),
                memory_bytes: data.len(),
            })
        }
    }
}

async fn read_precomputed(source: &VolumeSource, p: &crate::precomputed::ChunkParams, grid: &[u64]) -> Result<ChunkData> {
    let spec = &source.spec;
    let (origin, size) = spec.chunk_bounds(grid);
    let bytes_per = spec.data_type.bytes();
    let numel: usize = size.iter().map(|&s| s as usize).product();
    let bytes = match &p.sharding {
        None => {
            let key = crate::precomputed::ChunkParams::chunk_key(&origin, &size);
            p.location.read(&key, ByteRange::Full).await?
        }
        Some(sharding) => {
            // `gridShape`: chunks along x, y, z.
            let shape = spec.grid_shape();
            let key = crate::sharded::volume_chunk_key([shape[0], shape[1], shape[2]], [grid[0], grid[1], grid[2]]);
            crate::sharded::read(&p.location, sharding, key).await?
        }
    };
    let Some(bytes) = bytes else {
        return Ok(ChunkData {
            origin,
            size,
            data: vec![0; numel * bytes_per],
            memory_bytes: 0,
        });
    };
    let encoded = bytes.len();
    let data = match p.encoding {
        Encoding::Raw => {
            ensure!(
                bytes.len() == numel * bytes_per,
                "Raw-format chunk is {} bytes, but {numel} * {bytes_per} = {} bytes are expected.",
                bytes.len(),
                numel * bytes_per
            );
            bytes
        }
        Encoding::CompressedSegmentation => {
            let block = spec.compressed_segmentation_block_size.context("compressed segmentation without a block size")?;
            let channels = size.get(3).copied().unwrap_or(1);
            let job = DecodeJob::CompressedSegmentation {
                size: [size[0], size[1], size[2], channels],
                block,
                uint64: spec.data_type == ng_core::DataType::Uint64,
            };
            crate::decode::decode(job, bytes).await?
        }
        Encoding::Jpeg => {
            let area = size[..3].iter().map(|&s| s as usize).product();
            crate::decode::decode(DecodeJob::Jpeg { area, channels: size.get(3).copied().unwrap_or(1) as usize }, bytes).await?
        }
        Encoding::Png => {
            let area = size[..3].iter().map(|&s| s as usize).product();
            let job = DecodeJob::Png { area, channels: size.get(3).copied().unwrap_or(1) as usize, bytes_per_sample: bytes_per };
            let png = crate::decode::decode(job, bytes).await?;
            ensure!(
                png.len() == numel * bytes_per,
                "Raw-format chunk is {} bytes, but {numel} * {bytes_per} = {} bytes are expected.",
                png.len(),
                numel * bytes_per
            );
            png
        }
        other => bail!("precomputed {other:?} chunks are not supported yet"),
    };
    let memory_bytes = if p.encoding == Encoding::CompressedSegmentation { encoded } else { data.len() };
    Ok(ChunkData { origin, size, data, memory_bytes })
}

/// `n5` `download`: the chunk at `<g0>/<g1>/…`; one never written is zeros.
/// Its header gives its shape; any part beyond the volume is cut off.
async fn read_n5(source: &VolumeSource, p: &crate::n5::ChunkParams, grid: &[u64]) -> Result<ChunkData> {
    let spec = &source.spec;
    let (origin, size) = spec.chunk_bounds(grid);
    let element = spec.data_type.bytes();
    let numel: usize = size.iter().map(|&s| s as usize).product();
    let key = grid.iter().map(u64::to_string).collect::<Vec<_>>().join("/");
    let Some(bytes) = p.location.read(&key, ByteRange::Full).await? else {
        return Ok(ChunkData { origin, size, data: vec![0; numel * element], memory_bytes: 0 });
    };
    let (shape, data) = crate::n5::decode_chunk(&bytes, p.encoding, spec.data_type, spec.rank).with_context(|| format!("decoding {}", p.location.child(&key).url()))?;
    if shape == size {
        let memory_bytes = data.len();
        return Ok(ChunkData { origin, size, data, memory_bytes });
    }
    // Copy the overlap, row by row along the first dimension.
    let mut out = vec![0u8; numel * element];
    let rank = size.len();
    let rows: usize = size[1..].iter().map(|&s| s as usize).product();
    let width = size[0].min(shape[0]) as usize * element;
    for row in 0..rows {
        let (mut src, mut dst, mut rest) = (0usize, 0usize, row);
        let (mut src_stride, mut dst_stride) = (shape[0] as usize, size[0] as usize);
        let mut inside = true;
        for d in 1..rank {
            let i = rest % size[d] as usize;
            rest /= size[d] as usize;
            inside &= i < shape[d] as usize;
            src += i * src_stride;
            dst += i * dst_stride;
            src_stride *= shape[d] as usize;
            dst_stride *= size[d] as usize;
        }
        if inside {
            out[dst * element..dst * element + width].copy_from_slice(&data[src * element..src * element + width]);
        }
    }
    let memory_bytes = out.len();
    Ok(ChunkData { origin, size, data: out, memory_bytes })
}

/// Shard indexes already read, per shard key; `None` for an absent shard.
type ShardIndexCache = Mutex<HashMap<String, Option<Arc<ShardIndex>>>>;

fn shard_cache() -> &'static ShardIndexCache {
    static CACHE: std::sync::OnceLock<ShardIndexCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}

async fn read_zarr(source: &VolumeSource, p: &crate::zarr::ChunkParams, grid: &[u64]) -> Result<ChunkData> {
    let spec = &source.spec;
    let a = &p.array;
    let m = &a.codec;
    let rank = a.rank();
    let p2l = &a.physical_to_logical;
    // Spec dimension i is physical dimension rank-1-i.
    let mut logical_grid = vec![0u64; rank];
    for (i, &g) in grid.iter().enumerate() {
        logical_grid[p2l[rank - 1 - i]] = g;
    }
    // The stored chunk (a shard, when sharded) and, within it, the sub-chunk.
    let key_coords: Vec<u64> = (0..rank)
        .map(|d| logical_grid[d] * a.read_chunk_shape[d] as u64 / m.chunk_shape[d] as u64)
        .collect();
    let key = m.chunk_key(&key_coords);
    let origin: Vec<f64> = (0..rank).map(|i| grid[i] as f64 * spec.chunk_data_size[i] as f64).collect();
    let size = spec.chunk_data_size.clone();
    let logical = match m.sharding() {
        None => match p.location.read(&key, ByteRange::Full).await? {
            Some(bytes) => Some(m.decode_chunk(&bytes).with_context(|| format!("decoding {}", p.location.child(&key).url()))?),
            None => None,
        },
        Some(spec) => {
            let sub: Vec<usize> = (0..rank)
                .map(|d| (logical_grid[d] % spec.sub_chunk_grid_shape[d] as u64) as usize)
                .collect();
            let cache_key = p.location.child(&key).url();
            let cached = shard_cache().lock().unwrap().get(&cache_key).cloned();
            let index = match cached {
                Some(index) => index,
                None => {
                    let range = match spec.index_range()? {
                        IndexRange::Prefix(n) => ByteRange::Range { offset: 0, length: n },
                        IndexRange::Suffix(n) => ByteRange::Suffix(n),
                    };
                    let index = match p.location.read(&key, range).await? {
                        Some(bytes) => Some(Arc::new(spec.decode_index(&bytes, m.verify_checksums)?)),
                        None => None,
                    };
                    shard_cache().lock().unwrap().insert(cache_key, index.clone());
                    index
                }
            };
            match index.as_ref().and_then(|i| i.locate(&sub)) {
                None => None,
                Some(r) => match p.location.read(&key, ByteRange::Range { offset: r.start, length: r.end - r.start }).await? {
                    Some(bytes) => Some(spec.decode_sub_chunk(&bytes, m.data_type, m.verify_checksums)?),
                    None => None,
                },
            }
        }
    };
    let elem = m.fill_value.len();
    let data = match logical {
        // Logical C order → physical C order, which is the spec's x-fastest
        // layout.
        Some(logical) => to_physical(&logical, &a.read_chunk_shape, p2l, elem),
        None => m.fill_value.repeat(a.read_chunk_shape.iter().product()),
    };
    let memory_bytes = data.len();
    Ok(ChunkData { origin, size, data, memory_bytes })
}

/// Reorders a logical C-order array of `shape` into physical C order, where
/// physical dimension `k` is logical dimension `p2l[k]`.
fn to_physical(logical: &[u8], shape: &[usize], p2l: &[usize], elem: usize) -> Vec<u8> {
    let rank = shape.len();
    if p2l.iter().enumerate().all(|(k, &l)| k == l) {
        return logical.to_vec();
    }
    let mut lstride = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        lstride[d] = lstride[d + 1] * shape[d + 1];
    }
    let pshape: Vec<usize> = p2l.iter().map(|&l| shape[l]).collect();
    let numel: usize = shape.iter().product();
    let mut out = vec![0u8; logical.len()];
    let mut idx = vec![0usize; rank];
    for k in 0..numel {
        let src: usize = (0..rank).map(|pk| idx[pk] * lstride[p2l[pk]]).sum();
        out[k * elem..(k + 1) * elem].copy_from_slice(&logical[src * elem..(src + 1) * elem]);
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < pshape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Decodes a JPEG chunk as `decodeJpegChunk` does: the image is the chunk's
/// x·y·z voxels in any width × height of that area, with one component per
/// channel; a colour image comes back planar (channel slowest).
pub fn decode_jpeg(data: &[u8], area: usize, channels: usize) -> Result<Vec<u8>> {
    let image = crate::jpeg::decode(data).context("JPEG decoding failed")?;
    let (width, height, components) = (image.width, image.height, image.num_components);
    ensure!(
        width * height == area,
        "JPEG data does not have the expected dimensions: width={width}, height={height}, \
         expected width=undefined, expected height=undefined, expected area={area}"
    );
    ensure!(
        components == channels,
        "JPEG data does not have the expected number of components: components={components}, expected={channels}"
    );
    let pixels = image.data;
    match components {
        1 => Ok(pixels),
        3 => Ok(transpose_2d(&pixels, area, 3)),
        _ => bail!("JPEG data has an unsupported number of components: components={components}"),
    }
}

/// Decodes a PNG chunk as `decodePngChunk` does (spng's `SPNG_FMT_PNG`:
/// samples as stored, host-endian, so little-endian here): 8- or 16-bit
/// grey, grey + alpha, RGB or RGBA, with one sample per channel and as many
/// bytes per sample as the data type has.
///
/// A multi-channel image is made planar with `transposeArray2d` over its
/// bytes, as neuroglancer does; for 16-bit samples that moves bytes rather
/// than samples and only covers the first half of the image. Reproduced, not
/// fixed.
pub fn decode_png(data: &[u8], area: usize, channels: usize, bytes_per_sample: usize) -> Result<Vec<u8>> {
    let mut decoder = png::Decoder::new(data);
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().context("png: invalid header")?;
    let info = reader.info();
    let (sx, sy) = (info.width as usize, info.height as usize);
    let depth = info.bit_depth as usize;
    let (sample_bytes, components) = match info.color_type {
        png::ColorType::Grayscale => (if depth <= 8 { 1 } else { 2 }, 1),
        png::ColorType::Rgb => (depth / 8, 3),
        png::ColorType::Indexed => (1, 3),
        png::ColorType::GrayscaleAlpha => (depth / 8, 2),
        png::ColorType::Rgba => (depth / 8, 4),
    };
    ensure!(
        sx * sy == area && components == channels && sample_bytes == bytes_per_sample,
        "png: Image decode parameters did not match expected chunk parameters.  \
         Expected: width: undefined height: undefined area: {area} channels: {channels} \
         bytes per pixel: {bytes_per_sample}.  Decoded:  width: {sx} height: {sy} \
         channels: {components} bytes per pixel: {sample_bytes}.  Convert to Grayscale? false"
    );
    // spng's decoded size for indexed and sub-byte images is not
    // width·height·channels·bytes, so neuroglancer rejects them.
    ensure!(
        info.color_type != png::ColorType::Indexed && depth >= 8,
        "png: Failed to decode png image. decoder code: 8"
    );
    let mut image = vec![0; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut image).context("png: Failed to decode png image. decoder code: 9")?;
    image.truncate(frame.buffer_size());
    ensure!(image.len() == sx * sy * components * sample_bytes, "png: unexpected decoded size");
    if sample_bytes == 2 {
        for pair in image.chunks_exact_mut(2) {
            pair.swap(0, 1);
        }
    }
    Ok(if components == 1 { image } else { transpose_2d(&image, sx * sy, components) })
}

/// `transposeArray2d` on bytes: `major × minor` row-major to `minor × major`.
/// Like neuroglancer's, it reads only the first `major · minor` elements and
/// leaves the rest of the result zero.
pub(crate) fn transpose_2d(array: &[u8], major: usize, minor: usize) -> Vec<u8> {
    let mut out = vec![0; array.len()];
    for index in 0..major {
        for j in 0..minor {
            out[j * major + index] = array[index * minor + j];
        }
    }
    out
}

/// Decodes multi-channel compressed segmentation (the precomputed
/// `compressed_segmentation` encoding; format in neuroglancer's
/// `src/sliceview/compressed_segmentation/README.md`) to plain values: x
/// fastest, then y, z, channel.
pub fn decode_compressed_segmentation(data: &[u8], size: [u32; 4], block: [u32; 3], uint64: bool) -> Result<Vec<u8>> {
    ensure!(data.len().is_multiple_of(4), "compressed segmentation data is not a whole number of uint32s");
    let words: Vec<u32> = data.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let [sx, sy, sz, channels] = size.map(|v| v as usize);
    let [bx, by, bz] = block.map(|v| v as usize);
    ensure!(bx > 0 && by > 0 && bz > 0, "zero compressed segmentation block size");
    let per = if uint64 { 2 } else { 1 };
    let elem = 4 * per;
    let n = sx * sy * sz;
    let mut out = vec![0u8; n * channels * elem];
    let (gx, gy) = (sx.div_ceil(bx), sy.div_ceil(by));
    let word = |i: usize| words.get(i).copied().context("compressed segmentation offset past the end of the data");
    for c in 0..channels {
        let base = word(c)? as usize;
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let block_index = x / bx + gx * (y / by + gy * (z / bz));
                    let h0 = word(base + 2 * block_index)?;
                    let h1 = word(base + 2 * block_index + 1)?;
                    let table = (h0 & 0xff_ffff) as usize;
                    let bits = h0 >> 24;
                    ensure!(matches!(bits, 0 | 1 | 2 | 4 | 8 | 16 | 32), "invalid encoded bits {bits}");
                    let mut index = 0usize;
                    if bits > 0 {
                        let within = x % bx + bx * (y % by + by * (z % bz));
                        let bit = within * bits as usize;
                        // As neuroglancer: the encoded-values offset is masked
                        // to 24 bits.
                        let values = ((base + h1 as usize) & 0xff_ffff) + bit / 32;
                        let w = word(values)?;
                        let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
                        index = ((w >> (bit % 32)) & mask) as usize;
                    }
                    let v = base + table + per * index;
                    let o = ((c * n) + x + sx * (y + sy * z)) * elem;
                    out[o..o + 4].copy_from_slice(&word(v)?.to_le_bytes());
                    if uint64 {
                        out[o + 4..o + 8].copy_from_slice(&word(v + 1)?.to_le_bytes());
                    }
                }
            }
        }
    }
    Ok(out)
}

/// How many bytes neuroglancer's compressed segmentation encoding of `data`
/// takes (`encodeChannels`: each channel's offset, then per channel a
/// two-word header per block, each block's indices at the fewest of 0, 1,
/// 2, 4, 8, 16 or 32 bits its distinct values need over the whole block
/// size, and its sorted value table unless the channel already has one
/// alike). `size` is x, y, z and channels; `data` x fastest, little-endian.
/// What neuroglancer holds for a raw segmentation chunk it transcodes.
pub fn compressed_segmentation_encoded_bytes(data: &[u8], size: [usize; 4], block: [usize; 3], uint64: bool) -> usize {
    let element = if uint64 { 8 } else { 4 };
    let value = |i: usize| -> u64 {
        let b = &data[i * element..(i + 1) * element];
        if uint64 {
            u64::from_le_bytes(b.try_into().unwrap())
        } else {
            u32::from_le_bytes(b.try_into().unwrap()) as u64
        }
    };
    let grid: [usize; 3] = std::array::from_fn(|i| size[i].div_ceil(block[i].max(1)));
    let block_elements = block[0] * block[1] * block[2];
    let mut words = size[3];
    for c in 0..size[3] {
        let mut tables: std::collections::HashSet<Vec<u64>> = Default::default();
        words += 2 * grid[0] * grid[1] * grid[2];
        for bz in 0..grid[2] {
            for by in 0..grid[1] {
                for bx in 0..grid[0] {
                    let lo = [bx * block[0], by * block[1], bz * block[2]];
                    let hi: [usize; 3] = std::array::from_fn(|i| (lo[i] + block[i]).min(size[i]));
                    let mut values = vec![];
                    for z in lo[2]..hi[2] {
                        for y in lo[1]..hi[1] {
                            for x in lo[0]..hi[0] {
                                values.push(value(x + size[0] * (y + size[1] * (z + size[2] * c))));
                            }
                        }
                    }
                    if values.is_empty() {
                        continue;
                    }
                    values.sort_unstable();
                    values.dedup();
                    let n = values.len();
                    let bits = if n == 1 {
                        0
                    } else {
                        let mut b = 1;
                        while (1usize << b) < n {
                            b *= 2;
                        }
                        b
                    };
                    words += (bits * block_elements).div_ceil(32);
                    if tables.insert(values) {
                        words += n * (element / 4);
                    }
                }
            }
        }
    }
    words * 4
}
