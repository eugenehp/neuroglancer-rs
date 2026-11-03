//! Key-value stores: where data sources read their files from. Port of the
//! reading side of `src/kvstore/` (http, gcs, s3, file).
//!
//! A store reads `key` (a path relative to the store's base) whole or by byte
//! range. A read that finds nothing returns `Ok(None)`: HTTP 404, and also 403
//! and network-level failures, which neuroglancer treats as "not found" too
//! (`isNotFoundError`: S3 answers 403 for absent keys).
//!
//! Reads are futures. Natively, HTTP reads wait on I/O threads of their own
//! through one keep-alive client, and file reads block (run them on worker
//! threads); in the browser they wrap `fetch`.

use std::sync::Arc;

use anyhow::{bail, Result};

#[cfg(not(target_arch = "wasm32"))]
mod disk_cache;
#[cfg(target_arch = "wasm32")]
mod browser_cache;
#[cfg(target_arch = "wasm32")]
mod global;
mod file;
mod http;
mod memory;
mod zip;

#[cfg(target_arch = "wasm32")]
pub use browser_cache::set_browser_cache;
#[cfg(not(target_arch = "wasm32"))]
pub use disk_cache::set_disk_cache;

/// A cache entry's name for a read: a 128-bit hash of the object's stable
/// name and the range, in hex.
pub(crate) fn cache_key(name: &str, range: ByteRange) -> String {
    use std::hash::{Hash, Hasher};
    let hash = |seed: u64| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut h);
        name.hash(&mut h);
        match range {
            ByteRange::Full => 0u8.hash(&mut h),
            ByteRange::Range { offset, length } => (1u8, offset, length).hash(&mut h),
            ByteRange::Suffix(n) => (2u8, n).hash(&mut h),
        }
        h.finish()
    };
    format!("{:016x}{:016x}", hash(0x6e67), hash(0x7273))
}
pub use file::FileStore;
pub use http::{encode_path_for_url, HttpStore};
pub use memory::{global as memory, MemoryStore};
pub use zip::{GzipStore, ZipStore};

/// Which bytes of a value to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteRange {
    Full,
    /// `length` bytes from `offset`.
    Range { offset: u64, length: u64 },
    /// The last `n` bytes.
    Suffix(u64),
}

impl ByteRange {
    /// The concrete `[offset, offset + length)` within a value of `total`
    /// bytes, clamped as neuroglancer's `composeByteRangeRequest` does.
    pub fn resolve(self, total: u64) -> (u64, u64) {
        match self {
            ByteRange::Full => (0, total),
            ByteRange::Range { offset, length } => {
                let offset = offset.min(total);
                (offset, length.min(total - offset))
            }
            ByteRange::Suffix(n) => {
                let n = n.min(total);
                (total - n, n)
            }
        }
    }
}

pub type ReadResult = Result<Option<Vec<u8>>>;

#[cfg(not(target_arch = "wasm32"))]
pub type ReadFuture = futures::future::BoxFuture<'static, ReadResult>;
#[cfg(target_arch = "wasm32")]
pub type ReadFuture = futures::future::LocalBoxFuture<'static, ReadResult>;

/// `Send + Sync` natively, where reads run on worker threads; nothing in the
/// browser, where `fetch` futures are not `Send`.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSendSync: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync> MaybeSendSync for T {}
#[cfg(target_arch = "wasm32")]
pub trait MaybeSendSync {}
#[cfg(target_arch = "wasm32")]
impl<T> MaybeSendSync for T {}

pub trait KvStore: MaybeSendSync {
    /// Reads `key`; `Ok(None)` if it does not exist.
    fn read(&self, key: &str, range: ByteRange) -> ReadFuture;

    /// A URL for `key`, for messages.
    fn url(&self, key: &str) -> String;
}

/// A store and a path within it: what a kvstore URL denotes.
#[derive(Clone)]
pub struct KvStoreLocation {
    pub store: Arc<dyn KvStore>,
    /// Relative to the store; no leading `/`. A directory ends in `/`.
    pub path: String,
}

impl std::fmt::Debug for KvStoreLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KvStoreLocation({})", self.url())
    }
}

impl KvStoreLocation {
    /// Reads `path` joined with `key`.
    pub fn read(&self, key: &str, range: ByteRange) -> ReadFuture {
        self.store.read(&join(&self.path, key), range)
    }

    /// The location of `key` under this one.
    pub fn child(&self, key: &str) -> KvStoreLocation {
        KvStoreLocation {
            store: self.store.clone(),
            path: join(&self.path, key),
        }
    }

    /// This location as a directory (ending in `/`).
    pub fn as_directory(&self) -> KvStoreLocation {
        let mut path = self.path.clone();
        if !path.is_empty() && !path.ends_with('/') {
            path.push('/');
        }
        KvStoreLocation {
            store: self.store.clone(),
            path,
        }
    }

    pub fn url(&self) -> String {
        self.store.url(&self.path)
    }
}

/// `a` + `b`, with exactly one `/` between them unless `a` is empty.
pub fn join(a: &str, b: &str) -> String {
    if a.is_empty() {
        return b.trim_start_matches('/').to_owned();
    }
    if b.is_empty() {
        return a.to_owned();
    }
    format!("{}/{}", a.trim_end_matches('/'), b.trim_start_matches('/'))
}

/// Parses a kvstore URL (the part of a data source URL before any `|`):
///
/// - `http://…`, `https://…`
/// - `gs://bucket/path` — the GCS JSON API, as neuroglancer's `gs://` uses
/// - `gs+xml://bucket/path` — `https://storage.googleapis.com/bucket/path`
/// - `s3://bucket/path` — `https://bucket.s3.amazonaws.com/path`
/// - `file:///abs/path` (native only)
pub fn parse_url(url: &str) -> Result<KvStoreLocation> {
    let Some((scheme, rest)) = url.split_once("://") else {
        bail!("not a kvstore URL: {url:?}");
    };
    let (bucket, path) = rest.split_once('/').unwrap_or((rest, ""));
    let location = |store: Arc<dyn KvStore>, path: &str| KvStoreLocation {
        store,
        path: path.to_owned(),
    };
    Ok(match scheme {
        "http" | "https" => {
            // The origin is the store; the path is the location within it.
            let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
            location(Arc::new(HttpStore::new(format!("{scheme}://{host}"))), path)
        }
        "gs" => location(Arc::new(HttpStore::gcs(bucket)), path),
        "gs+xml" => location(
            Arc::new(HttpStore::new(format!("https://storage.googleapis.com/{bucket}"))),
            path,
        ),
        "s3" => location(Arc::new(HttpStore::new(format!("https://{bucket}.s3.amazonaws.com"))), path),
        // bytes handed over directly (`ng_kvstore::memory().insert(name, bytes)`)
        "memory" => location(Arc::new(memory::global().clone()), rest),
        #[cfg(not(target_arch = "wasm32"))]
        "file" => {
            // `file:///abs` → root `/`, path `abs`.
            let abs = rest.strip_prefix('/').unwrap_or(rest);
            location(Arc::new(FileStore::new("/")), abs)
        }
        other => bail!("unsupported kvstore scheme `{other}://`"),
    })
}

/// Parses a kvstore pipeline URL: a base kvstore URL then adapters, each
/// `|<scheme>:<path>` — `zip:` (an archive's entries, `<path>` within it)
/// and `gzip:` (the value decompressed).
#[cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]
pub fn parse_pipeline(url: &str) -> Result<KvStoreLocation> {
    let mut parts = url.split('|');
    let mut location = parse_url(parts.next().unwrap_or_default())?;
    for part in parts {
        let (scheme, path) = part.split_once(':').unwrap_or((part, ""));
        let store: Arc<dyn KvStore> = match scheme {
            "zip" => Arc::new(ZipStore::new(location)),
            "gzip" => {
                if !path.is_empty() {
                    bail!("gzip: expected an empty path, but received: {path:?}");
                }
                Arc::new(GzipStore::new(location))
            }
            other => bail!("unsupported kvstore adapter `{other}:`"),
        };
        location = KvStoreLocation { store, path: path.to_owned() };
    }
    Ok(location)
}

/// HTTP requests asked again since the page started (a server pushing back,
/// or failing for now): a host tuning its downloads at once reads it.
pub static RETRIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many requests have been retried so far (`RETRIED`).
pub fn retried() -> u64 {
    RETRIED.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls() {
        let l = parse_url("https://example.com/a/b/").unwrap();
        assert_eq!(l.url(), "https://example.com/a/b/");
        assert_eq!(l.child("info").url(), "https://example.com/a/b/info");
        let l = parse_url("s3://bucket/data").unwrap();
        assert_eq!(l.child("x").url(), "https://bucket.s3.amazonaws.com/data/x");
        let l = parse_url("gs+xml://bucket/data").unwrap();
        assert_eq!(l.child("x").url(), "https://storage.googleapis.com/bucket/data/x");
        // Named as `GcsKvStore.getUrl` names it (fetched through the JSON API).
        let l = parse_url("gs://bucket/dir/data").unwrap();
        assert_eq!(l.child("x y").url(), "gs://bucket/dir/data/x y");
        assert!(parse_url("ftp://x/y").is_err());
    }

    #[test]
    fn byte_ranges_clamp() {
        assert_eq!(ByteRange::Suffix(10).resolve(4), (0, 4));
        assert_eq!(ByteRange::Range { offset: 3, length: 10 }.resolve(5), (3, 2));
        assert_eq!(ByteRange::Full.resolve(7), (0, 7));
    }

    #[test]
    fn join_has_one_slash() {
        assert_eq!(join("a/", "/b"), "a/b");
        assert_eq!(join("", "b"), "b");
        assert_eq!(join("a", ""), "a");
    }
}
