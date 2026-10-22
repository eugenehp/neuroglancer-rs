//! An in-memory store, for tests and for data handed over directly (a file
//! dropped on the page).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{ByteRange, KvStore, ReadFuture};

#[derive(Default, Clone)]
pub struct MemoryStore {
    values: Arc<Mutex<HashMap<String, Arc<Vec<u8>>>>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, key: impl Into<String>, value: Vec<u8>) {
        self.values.lock().unwrap().insert(key.into(), Arc::new(value));
    }
}

/// The process-wide store behind `memory://` URLs: an embedder (or a page
/// a file is dropped on) puts bytes here under a name, then opens
/// `memory://<name>` like any other location — `nifti://memory://scan.nii`.
pub fn global() -> &'static MemoryStore {
    static GLOBAL: std::sync::OnceLock<MemoryStore> = std::sync::OnceLock::new();
    GLOBAL.get_or_init(MemoryStore::new)
}

impl KvStore for MemoryStore {
    fn read(&self, key: &str, range: ByteRange) -> ReadFuture {
        let value = self.values.lock().unwrap().get(key).cloned();
        Box::pin(async move {
            Ok(value.map(|v| {
                let (offset, length) = range.resolve(v.len() as u64);
                v[offset as usize..(offset + length) as usize].to_vec()
            }))
        })
    }

    fn url(&self, key: &str) -> String {
        format!("memory://{key}")
    }
}

#[cfg(test)]
mod tests {
    use crate::ByteRange;

    #[test]
    fn memory_urls_read_what_was_put() {
        super::global().insert("dir/a.bin", vec![1, 2, 3, 4]);
        let loc = crate::parse_url("memory://dir/a.bin").unwrap();
        let got = futures::executor::block_on(loc.read("", ByteRange::Range { offset: 1, length: 2 })).unwrap();
        assert_eq!(got, Some(vec![2, 3]));
    }
}
