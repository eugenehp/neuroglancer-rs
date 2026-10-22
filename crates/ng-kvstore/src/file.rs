//! Local files (native only), for `file://` URLs and plain paths.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use crate::{ByteRange, KvStore, ReadFuture};

pub struct FileStore {
    root: PathBuf,
}

impl FileStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl KvStore for FileStore {
    fn read(&self, key: &str, range: ByteRange) -> ReadFuture {
        let path = self.root.join(key);
        Box::pin(async move {
            let mut file = match std::fs::File::open(&path) {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(e) => return Err(anyhow::anyhow!("{}: {e}", path.display())),
            };
            let meta = file.metadata()?;
            if meta.is_dir() {
                return Ok(None);
            }
            let (offset, length) = range.resolve(meta.len());
            file.seek(SeekFrom::Start(offset))?;
            let mut buf = vec![0u8; length as usize];
            file.read_exact(&mut buf)?;
            Ok(Some(buf))
        })
    }

    fn url(&self, key: &str) -> String {
        format!("file://{}", self.root.join(key).display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_ranges_and_reports_missing() {
        let dir = std::env::temp_dir().join(format!("ng-kvstore-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("v"), b"0123456789").unwrap();
        let s = FileStore::new(&dir);
        let read = |k: &str, r| futures::executor::block_on(s.read(k, r)).unwrap();
        assert_eq!(read("v", ByteRange::Full).unwrap(), b"0123456789");
        assert_eq!(read("v", ByteRange::Range { offset: 2, length: 3 }).unwrap(), b"234");
        assert_eq!(read("v", ByteRange::Suffix(4)).unwrap(), b"6789");
        assert_eq!(read("missing", ByteRange::Full), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
