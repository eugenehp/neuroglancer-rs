//! Adapters over another store's value (`kvstore/zip`, `kvstore/gzip`): a
//! zip archive's entries by name (its central directory read once, from the
//! end, ZIP64 included; entries stored or deflated), and a gzip-compressed
//! value decompressed.

use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use futures::lock::Mutex;

use crate::{ByteRange, KvStore, KvStoreLocation, ReadFuture};

/// An entry: where its local header is, how it is stored and its sizes.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Entry {
    header_offset: u64,
    method: u16,
    compressed_size: u64,
    size: u64,
}

/// A zip archive at `base`, read as a store of its entries.
pub struct ZipStore(Arc<ZipArchive>);

struct ZipArchive {
    base: KvStoreLocation,
    entries: Mutex<Option<Arc<HashMap<String, Entry>>>>,
}

fn u16_at(b: &[u8], o: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(b.get(o..o + 2).context("zip: truncated")?.try_into().unwrap()))
}

fn u32_at(b: &[u8], o: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(b.get(o..o + 4).context("zip: truncated")?.try_into().unwrap()))
}

fn u64_at(b: &[u8], o: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(b.get(o..o + 8).context("zip: truncated")?.try_into().unwrap()))
}

/// The central directory's entries, from `directory` (its bytes) holding
/// `count` of them.
fn parse_directory(directory: &[u8], count: u64) -> Result<HashMap<String, Entry>> {
    let mut entries = HashMap::new();
    let mut o = 0;
    for _ in 0..count {
        ensure!(u32_at(directory, o)? == 0x0201_4b50, "zip: bad central directory entry");
        let method = u16_at(directory, o + 10)?;
        let mut compressed_size = u32_at(directory, o + 20)? as u64;
        let mut size = u32_at(directory, o + 24)? as u64;
        let (name_len, extra_len, comment_len) = (u16_at(directory, o + 28)? as usize, u16_at(directory, o + 30)? as usize, u16_at(directory, o + 32)? as usize);
        let mut header_offset = u32_at(directory, o + 42)? as u64;
        let name = String::from_utf8_lossy(directory.get(o + 46..o + 46 + name_len).context("zip: truncated")?).into_owned();
        // ZIP64: the sizes and offset that did not fit, in that order.
        let extra = directory.get(o + 46 + name_len..o + 46 + name_len + extra_len).context("zip: truncated")?;
        let mut e = 0;
        while e + 4 <= extra.len() {
            let (id, len) = (u16_at(extra, e)?, u16_at(extra, e + 2)? as usize);
            if id == 1 {
                let mut f = e + 4;
                for field in [&mut size, &mut compressed_size, &mut header_offset] {
                    if *field == 0xffff_ffff {
                        *field = u64_at(extra, f)?;
                        f += 8;
                    }
                }
            }
            e += 4 + len;
        }
        entries.insert(name, Entry { header_offset, method, compressed_size, size });
        o += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

impl ZipStore {
    // (On wasm the base store is not `Send`: nothing there is.)
    #[cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]
    pub fn new(base: KvStoreLocation) -> Self {
        Self(Arc::new(ZipArchive { base, entries: Mutex::new(None) }))
    }
}

impl ZipArchive {
    /// The central directory, read once: the end record (in the last
    /// 64 KiB + 22 bytes), its ZIP64 counterpart if needed, the directory.
    async fn entries(&self) -> Result<Arc<HashMap<String, Entry>>> {
        let mut guard = self.entries.lock().await;
        if let Some(e) = guard.as_ref() {
            return Ok(e.clone());
        }
        let url = self.base.url();
        let tail = self.base.read("", ByteRange::Suffix(65536 + 22)).await?.with_context(|| format!("{url} not found"))?;
        let end = (0..=tail.len().saturating_sub(22)).rev().find(|&i| tail[i..i + 4] == [0x50, 0x4b, 0x05, 0x06]).with_context(|| format!("{url}: not a zip archive"))?;
        let mut count = u16_at(&tail, end + 10)? as u64;
        let mut size = u32_at(&tail, end + 12)? as u64;
        let mut offset = u32_at(&tail, end + 16)? as u64;
        if (count == 0xffff || size == 0xffff_ffff || offset == 0xffff_ffff) && end >= 20 && tail[end - 20..end - 16] == [0x50, 0x4b, 0x06, 0x07] {
            let record = u64_at(&tail, end - 12)?;
            let z = self.base.read("", ByteRange::Range { offset: record, length: 56 }).await?.context("zip: ZIP64 end record missing")?;
            ensure!(u32_at(&z, 0)? == 0x0606_4b50, "zip: bad ZIP64 end record");
            count = u64_at(&z, 32)?;
            size = u64_at(&z, 40)?;
            offset = u64_at(&z, 48)?;
        }
        let directory = self.base.read("", ByteRange::Range { offset, length: size }).await?.context("zip: central directory missing")?;
        let entries = Arc::new(parse_directory(&directory, count)?);
        *guard = Some(entries.clone());
        Ok(entries)
    }

    async fn read_entry(&self, key: &str, range: ByteRange) -> Result<Option<Vec<u8>>> {
        let entries = self.entries().await?;
        let Some(&e) = entries.get(key) else { return Ok(None) };
        let header = self.base.read("", ByteRange::Range { offset: e.header_offset, length: 30 }).await?.context("zip: local header missing")?;
        ensure!(u32_at(&header, 0)? == 0x0403_4b50, "zip: bad local header for {key:?}");
        let data_offset = e.header_offset + 30 + u16_at(&header, 26)? as u64 + u16_at(&header, 28)? as u64;
        let data = match e.method {
            // Stored: only the bytes asked for.
            0 => {
                let (offset, length) = range.resolve(e.size);
                return self.base.read("", ByteRange::Range { offset: data_offset + offset, length }).await;
            }
            8 => {
                let raw = self.base.read("", ByteRange::Range { offset: data_offset, length: e.compressed_size }).await?.context("zip: entry data missing")?;
                let mut out = Vec::with_capacity(e.size as usize);
                flate2::read::DeflateDecoder::new(&raw[..]).read_to_end(&mut out).with_context(|| format!("zip: inflating {key:?}"))?;
                out
            }
            m => bail!("zip: unsupported compression method {m} for {key:?}"),
        };
        let (offset, length) = range.resolve(data.len() as u64);
        Ok(Some(data[offset as usize..(offset + length) as usize].to_vec()))
    }
}

impl KvStore for ZipStore {
    fn read(&self, key: &str, range: ByteRange) -> ReadFuture {
        let archive = self.0.clone();
        let key = key.to_owned();
        Box::pin(async move { archive.read_entry(&key, range).await })
    }

    fn url(&self, key: &str) -> String {
        format!("{}|zip:{key}", self.0.base.url())
    }
}

/// A gzip-compressed value at `base`, decompressed (`|gzip:`).
pub struct GzipStore {
    base: KvStoreLocation,
}

impl GzipStore {
    pub fn new(base: KvStoreLocation) -> Self {
        Self { base }
    }
}

impl KvStore for GzipStore {
    fn read(&self, key: &str, range: ByteRange) -> ReadFuture {
        let base = self.base.clone();
        let key = key.to_owned();
        Box::pin(async move {
            if !key.is_empty() {
                return Ok(None);
            }
            let Some(raw) = base.read("", ByteRange::Full).await? else { return Ok(None) };
            let mut out = vec![];
            flate2::read::MultiGzDecoder::new(&raw[..]).read_to_end(&mut out).context("gzip")?;
            let (offset, length) = range.resolve(out.len() as u64);
            Ok(Some(out[offset as usize..(offset + length) as usize].to_vec()))
        })
    }

    fn url(&self, key: &str) -> String {
        format!("{}|gzip:{key}", self.base.url())
    }
}
