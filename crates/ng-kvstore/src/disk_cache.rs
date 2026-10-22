//! A cache of remote reads on disk, natively, off unless asked for
//! ([`set_disk_cache`], or the `NG_CACHE_DIR` environment variable): a
//! dataset seen before loads from disk. Neuroglancer keeps no such cache, so
//! data changed on the server since it was cached is served as cached;
//! "not found" answers are never kept, so data written since shows.

use std::path::PathBuf;
use std::sync::Mutex;

use crate::ByteRange;

struct DiskCache {
    dir: PathBuf,
    max_bytes: u64,
    /// Bytes in the directory, as counted when it was opened and since.
    used: u64,
}

/// Default size: 10 GB.
const DEFAULT_MAX_BYTES: u64 = 10_000_000_000;

fn state() -> &'static Mutex<Option<DiskCache>> {
    static STATE: std::sync::OnceLock<Mutex<Option<DiskCache>>> = std::sync::OnceLock::new();
    STATE.get_or_init(|| {
        let from_env = std::env::var_os("NG_CACHE_DIR").map(PathBuf::from);
        Mutex::new(from_env.and_then(|dir| open(dir, DEFAULT_MAX_BYTES)))
    })
}

fn open(dir: PathBuf, max_bytes: u64) -> Option<DiskCache> {
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::warn!("no disk cache at {}: {e}", dir.display());
        return None;
    }
    let used = std::fs::read_dir(&dir).map_or(0, |entries| entries.flatten().filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum());
    Some(DiskCache { dir, max_bytes, used })
}

/// Keeps remote reads in `dir` (at most `max_bytes`, the least recently
/// written going first), and serves them from there; `None`: no cache.
pub fn set_disk_cache(dir: Option<PathBuf>, max_bytes: Option<u64>) {
    *state().lock().unwrap() = dir.and_then(|d| open(d, max_bytes.unwrap_or(DEFAULT_MAX_BYTES)));
}

/// The cached bytes of a read, if cached.
pub(crate) fn get(name: &str, range: ByteRange) -> Option<Vec<u8>> {
    let path = state().lock().unwrap().as_ref()?.dir.join(crate::cache_key(name, range));
    std::fs::read(path).ok()
}

/// Keeps a read's bytes.
pub(crate) fn put(name: &str, range: ByteRange, bytes: &[u8]) {
    let mut guard = state().lock().unwrap();
    let Some(cache) = guard.as_mut() else { return };
    let path = cache.dir.join(crate::cache_key(name, range));
    // Written whole, then renamed into place: a reader never sees half.
    let temp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&temp, bytes).and_then(|()| std::fs::rename(&temp, &path)).is_err() {
        let _ = std::fs::remove_file(&temp);
        return;
    }
    cache.used += bytes.len() as u64;
    if cache.used > cache.max_bytes {
        trim(cache);
    }
}

/// Deletes the oldest files until the cache is at 90 % of its size.
fn trim(cache: &mut DiskCache) {
    let Ok(entries) = std::fs::read_dir(&cache.dir) else { return };
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((m.modified().ok()?, m.len(), e.path()))
        })
        .collect();
    files.sort();
    let mut used: u64 = files.iter().map(|f| f.1).sum();
    let target = cache.max_bytes / 10 * 9;
    for (_, len, path) in files {
        if used <= target {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            used -= len;
        }
    }
    cache.used = used;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_come_back_from_disk_and_the_oldest_go_when_full() {
        let dir = std::env::temp_dir().join(format!("ng-disk-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        set_disk_cache(Some(dir.clone()), Some(100));
        let range = ByteRange::Range { offset: 8, length: 40 };
        assert_eq!(get("gs://b/x", range), None);
        put("gs://b/x", range, &[7; 40]);
        assert_eq!(get("gs://b/x", range), Some(vec![7; 40]));
        // Another range of the same object is another entry.
        assert_eq!(get("gs://b/x", ByteRange::Full), None);
        std::thread::sleep(std::time::Duration::from_millis(20));
        put("gs://b/y", range, &[1; 40]);
        std::thread::sleep(std::time::Duration::from_millis(20));
        // Over 100 bytes: the oldest (x) goes.
        put("gs://b/z", range, &[2; 40]);
        assert_eq!(get("gs://b/x", range), None);
        assert_eq!(get("gs://b/z", range), Some(vec![2; 40]));
        set_disk_cache(None, None);
        assert_eq!(get("gs://b/z", range), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
