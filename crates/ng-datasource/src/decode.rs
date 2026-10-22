//! Chunk decoding as a job that can run elsewhere. Natively a chunk loads on
//! a worker thread already, so its job runs in place; in a browser the page
//! has one thread, and the host can hand jobs to Web Workers
//! ([`set_offload`]), as neuroglancer decodes in its workers.

use anyhow::{bail, Result};

/// How to decode one chunk's bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeJob {
    /// [`crate::chunk::decode_jpeg`].
    Jpeg { area: usize, channels: usize },
    /// [`crate::chunk::decode_png`].
    Png { area: usize, channels: usize, bytes_per_sample: usize },
    /// [`crate::chunk::decode_compressed_segmentation`].
    CompressedSegmentation { size: [u32; 4], block: [u32; 3], uint64: bool },
}

impl DecodeJob {
    /// Decodes `data`, here.
    pub fn run(&self, data: &[u8]) -> Result<Vec<u8>> {
        match *self {
            DecodeJob::Jpeg { area, channels } => crate::chunk::decode_jpeg(data, area, channels),
            DecodeJob::Png { area, channels, bytes_per_sample } => crate::chunk::decode_png(data, area, channels, bytes_per_sample),
            DecodeJob::CompressedSegmentation { size, block, uint64 } => crate::chunk::decode_compressed_segmentation(data, size, block, uint64),
        }
    }

    /// The job as words, to send to a worker ([`from_words`](Self::from_words)).
    pub fn to_words(&self) -> Vec<u32> {
        match *self {
            DecodeJob::Jpeg { area, channels } => vec![0, area as u32, channels as u32],
            DecodeJob::Png { area, channels, bytes_per_sample } => vec![1, area as u32, channels as u32, bytes_per_sample as u32],
            DecodeJob::CompressedSegmentation { size, block, uint64 } => {
                vec![2, size[0], size[1], size[2], size[3], block[0], block[1], block[2], uint64 as u32]
            }
        }
    }

    pub fn from_words(w: &[u32]) -> Result<Self> {
        Ok(match w {
            [0, area, channels] => DecodeJob::Jpeg { area: *area as usize, channels: *channels as usize },
            [1, area, channels, bytes] => DecodeJob::Png { area: *area as usize, channels: *channels as usize, bytes_per_sample: *bytes as usize },
            [2, s0, s1, s2, s3, b0, b1, b2, u] => DecodeJob::CompressedSegmentation { size: [*s0, *s1, *s2, *s3], block: [*b0, *b1, *b2], uint64: *u != 0 },
            _ => bail!("not a decode job: {w:?}"),
        })
    }
}

/// Runs jobs somewhere else: the decoded bytes, or why not.
#[cfg(target_arch = "wasm32")]
pub type Offload = std::rc::Rc<dyn Fn(DecodeJob, Vec<u8>) -> futures::future::LocalBoxFuture<'static, Result<Vec<u8>>>>;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static OFFLOAD: std::cell::RefCell<Option<Offload>> = const { std::cell::RefCell::new(None) };
}

/// Where this thread's decode jobs run from now on (a browser's workers).
#[cfg(target_arch = "wasm32")]
pub fn set_offload(offload: Offload) {
    OFFLOAD.with(|o| *o.borrow_mut() = Some(offload));
}

/// Jobs run here again.
#[cfg(target_arch = "wasm32")]
pub fn clear_offload() {
    OFFLOAD.with(|o| *o.borrow_mut() = None);
}

/// Decodes `data` by `job`: through the off-loader if one is set, else here.
pub async fn decode(job: DecodeJob, data: Vec<u8>) -> Result<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    if let Some(offload) = OFFLOAD.with(|o| o.borrow().clone()) {
        return offload(job, data).await;
    }
    job.run(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_survive_the_trip_as_words() {
        for job in [
            DecodeJob::Jpeg { area: 4096, channels: 3 },
            DecodeJob::Png { area: 64, channels: 2, bytes_per_sample: 2 },
            DecodeJob::CompressedSegmentation { size: [64, 64, 64, 1], block: [8, 8, 8], uint64: true },
        ] {
            assert_eq!(DecodeJob::from_words(&job.to_words()).unwrap(), job);
        }
        assert!(DecodeJob::from_words(&[9]).is_err());
    }
}
