//! The precomputed sharded format (`neuroglancer_uint64_sharded_v1`): a port
//! of `src/datasource/precomputed/sharded.ts`, with the chunk key of
//! `PrecomputedVolumeChunkSource.download` (`encodeZIndexCompressed3d`) and the
//! hash of `src/util/hash.ts`.
//!
//! A chunk's key is hashed (after `preshift_bits`) to a shard and a minishard.
//! The shard file starts with a fixed index of `2^minishard_bits` byte
//! ranges, one minishard index each; a minishard index lists (key, byte range)
//! for its chunks, delta-encoded.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{bail, ensure, Context, Result};
use ng_kvstore::{ByteRange, KvStoreLocation};

use crate::precomputed::{DataEncoding, ShardingHash, ShardingParameters};

/// `murmurHash3_x86_128Hash64Bits_Bigint`: the low 64 bits of MurmurHash3
/// x86_128 of the 8 little-endian bytes of `input`.
pub fn murmurhash3_x86_128_hash64(seed: u32, input: u64) -> u64 {
    const C1: u32 = 0x239b961b;
    const C2: u32 = 0xab0e9789;
    const C3: u32 = 0x38b34ae5;
    fn mix(mut h: u32) -> u32 {
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;
        h
    }
    let (mut h1, mut h2, mut h3, mut h4) = (seed, seed, seed, seed);
    let k2 = ((input >> 32) as u32).wrapping_mul(C2).rotate_left(16).wrapping_mul(C3);
    h2 ^= k2;
    let k1 = (input as u32).wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
    h1 ^= k1;
    let len = 8;
    h1 ^= len;
    h2 ^= len;
    h3 ^= len;
    h4 ^= len;
    h1 = h1.wrapping_add(h2).wrapping_add(h3).wrapping_add(h4);
    h2 = h2.wrapping_add(h1);
    h3 = h3.wrapping_add(h1);
    h4 = h4.wrapping_add(h1);
    h1 = mix(h1);
    h2 = mix(h2);
    h3 = mix(h3);
    h4 = mix(h4);
    h1 = h1.wrapping_add(h2).wrapping_add(h3).wrapping_add(h4);
    h2 = h2.wrapping_add(h1);
    h1 as u64 | (h2 as u64) << 32
}

/// `encodeZIndexCompressed3d`: the bits of x, y and z interleaved, each
/// dimension dropping out once its `bits` are used.
pub fn encode_z_index_compressed_3d(bits: [u32; 3], position: [u64; 3]) -> u128 {
    let max_bits = bits.into_iter().max().unwrap_or(0);
    let (mut index, mut output_bit) = (0u128, 0);
    for bit in 0..max_bits {
        for d in 0..3 {
            if bit < bits[d] {
                // `(x >> bit) & 1` on a 32-bit grid coordinate.
                index |= (((position[d] as u32).checked_shr(bit).unwrap_or(0) & 1) as u128) << output_bit;
                output_bit += 1;
            }
        }
    }
    index
}

/// `Math.ceil(Math.log2(n))` for a grid size `n ≥ 1`.
pub fn ceil_log2(n: u64) -> u32 {
    if n <= 1 {
        0
    } else {
        64 - (n - 1).leading_zeros()
    }
}

/// The key of the chunk at `grid` in a volume `grid_shape` chunks across.
pub fn volume_chunk_key(grid_shape: [u64; 3], grid: [u64; 3]) -> u128 {
    encode_z_index_compressed_3d(grid_shape.map(ceil_log2), grid)
}

/// A decoded minishard index: keys, then absolute start and end offsets.
struct MinishardIndex {
    keys: Vec<u64>,
    starts: Vec<u64>,
    ends: Vec<u64>,
    shard_path: String,
}

/// A minishard index read, or one being read: those who asked meanwhile
/// wait for it (neuroglancer's minishard index is a chunk of its own, loaded
/// once however many chunks are in the minishard).
enum Cached {
    /// `None` where the minishard is empty or the shard absent; with when
    /// it was last used.
    Ready(Option<Arc<MinishardIndex>>, u64),
    Reading(Vec<futures::channel::oneshot::Sender<Result<Option<Arc<MinishardIndex>>, String>>>),
}

/// Minishard indexes by directory URL and shard + minishard, at most
/// [`MAX_CACHED`] read ones (the least recently used go first).
#[derive(Default)]
struct Cache {
    entries: HashMap<(String, u64), Cached>,
    /// Counts uses, to order them.
    tick: u64,
}

type MinishardCache = Mutex<Cache>;

/// Indexes kept: a minishard index lists up to a few thousand chunks, so
/// this covers far more than any view needs at once.
const MAX_CACHED: usize = 8192;

impl Cache {
    fn get(&self, key: &(String, u64)) -> Option<&Cached> {
        self.entries.get(key)
    }

    fn remove(&mut self, key: &(String, u64)) -> Option<Cached> {
        self.entries.remove(key)
    }

    fn insert_ready(&mut self, key: (String, u64), index: Option<Arc<MinishardIndex>>) {
        self.tick += 1;
        self.entries.insert(key, Cached::Ready(index, self.tick));
        if self.entries.len() > MAX_CACHED {
            // The oldest quarter of the read ones goes.
            let mut ready: Vec<(u64, (String, u64))> = self
                .entries
                .iter()
                .filter_map(|(k, c)| match c {
                    Cached::Ready(_, used) => Some((*used, k.clone())),
                    Cached::Reading(_) => None,
                })
                .collect();
            ready.sort_unstable_by_key(|(used, _)| *used);
            for (_, k) in ready.into_iter().take(MAX_CACHED / 4) {
                self.entries.remove(&k);
            }
        }
    }
}

/// Ends a read of a minishard index: if it ends without a result (its
/// future dropped), those waiting are let go to read it themselves.
struct Reading((String, u64));

impl Drop for Reading {
    fn drop(&mut self) {
        let mut cache = cache().lock().unwrap();
        if matches!(cache.get(&self.0), Some(Cached::Reading(_))) {
            cache.remove(&self.0);
        }
    }
}

fn cache() -> &'static MinishardCache {
    static CACHE: OnceLock<MinishardCache> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// A sharded file's gzip through the decode jobs ([`crate::decode::decode`]):
/// off the render thread where the host has given it workers.
async fn decode(data: Vec<u8>, encoding: DataEncoding) -> Result<Vec<u8>> {
    match encoding {
        DataEncoding::Raw => Ok(data),
        DataEncoding::Gzip => crate::decode::decode(crate::decode::DecodeJob::Gzip, data).await.context("gzip decoding failed"),
    }
}

/// `readFileHandle` with `strictByteRange`: a short read is an error.
async fn read_strict(location: &KvStoreLocation, path: &str, offset: u64, length: u64) -> Result<Option<Vec<u8>>> {
    let data = location.read(path, ByteRange::Range { offset, length }).await?;
    if let Some(d) = &data {
        ensure!(
            d.len() as u64 == length,
            "Received truncated response for {}{path}, expected {{\"offset\":{offset},\"length\":{length}}} but received length={}",
            location.url(),
            d.len()
        );
    }
    Ok(data)
}

async fn minishard_index(
    location: &KvStoreLocation,
    sharding: &ShardingParameters,
    shard_and_minishard: u64,
) -> Result<Option<Arc<MinishardIndex>>> {
    let cache_key = (location.url(), shard_and_minishard);
    loop {
        let waiting = {
            let mut cache = cache().lock().unwrap();
            cache.tick += 1;
            let tick = cache.tick;
            match cache.entries.get_mut(&cache_key) {
                Some(Cached::Ready(hit, used)) => {
                    *used = tick;
                    return Ok(hit.clone());
                }
                Some(Cached::Reading(waiters)) => {
                    let (send, receive) = futures::channel::oneshot::channel();
                    waiters.push(send);
                    receive
                }
                None => {
                    cache.entries.insert(cache_key.clone(), Cached::Reading(vec![]));
                    break;
                }
            }
        };
        match waiting.await {
            Ok(result) => return result.map_err(anyhow::Error::msg),
            // The read was dropped: read it here.
            Err(_) => continue,
        }
    }
    let reading = Reading(cache_key);
    let result = read_minishard_index(location, sharding, shard_and_minishard).await;
    let mut cache = cache().lock().unwrap();
    let waiters = match cache.remove(&reading.0) {
        Some(Cached::Reading(waiters)) => waiters,
        _ => vec![],
    };
    if let Ok(index) = &result {
        cache.insert_ready(reading.0.clone(), index.clone());
    }
    drop(cache);
    for w in waiters {
        let _ = w.send(result.as_ref().map(Clone::clone).map_err(|e| format!("{e:#}")));
    }
    result
}

async fn read_minishard_index(
    location: &KvStoreLocation,
    sharding: &ShardingParameters,
    shard_and_minishard: u64,
) -> Result<Option<Arc<MinishardIndex>>> {
    let mask = |bits: u32| ((1u128 << bits) - 1) as u64;
    let minishard = shard_and_minishard & mask(sharding.minishard_bits);
    let shard = mask(sharding.shard_bits) & shard_and_minishard.checked_shr(sharding.minishard_bits).unwrap_or(0);
    let digits = sharding.shard_bits.div_ceil(4) as usize;
    let shard_path = format!("{shard:0digits$x}.shard");
    let shard_index_size = 16u64 << sharding.minishard_bits;

    let index = async {
        let Some(entry) = read_strict(location, &shard_path, minishard << 4, 16).await? else {
            return Ok(None);
        };
        let start = u64::from_le_bytes(entry[..8].try_into().unwrap());
        let end = u64::from_le_bytes(entry[8..].try_into().unwrap());
        if start == end {
            return Ok(None);
        }
        // Offsets are relative to the end of the shard index.
        let (start, end) = (start + shard_index_size, end + shard_index_size);
        ensure!(end >= start, "invalid minishard index range {start}..{end}");
        let raw = location
            .read(&shard_path, ByteRange::Range { offset: start, length: end - start })
            .await?
            .with_context(|| format!("{}{shard_path} not found", location.url()))?;
        let raw = decode(raw, sharding.minishard_index_encoding).await?;
        if raw.len() % 24 != 0 {
            bail!("Invalid minishard index length: {}", raw.len());
        }
        let words: Vec<u64> = raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let n = words.len() / 3;
        let (mut keys, mut starts, mut ends) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut prev_key, mut prev_end) = (0u64, shard_index_size);
        // BigUint64Array arithmetic wraps at 64 bits.
        for i in 0..n {
            let key = prev_key.wrapping_add(words[i]);
            let start = prev_end.wrapping_add(words[n + i]);
            let end = start.wrapping_add(words[2 * n + i]);
            keys.push(key);
            starts.push(start);
            ends.push(end);
            prev_key = key;
            prev_end = end;
        }
        Ok::<_, anyhow::Error>(Some(Arc::new(MinishardIndex { keys, starts, ends, shard_path })))
    }
    .await?;
    Ok(index)
}

/// `ShardedKvStore.read`: the (decoded) value stored under `key`, or `None`
/// if it was never written.
pub async fn read(location: &KvStoreLocation, sharding: &ShardingParameters, key: u128) -> Result<Option<Vec<u8>>> {
    Ok(read_with_shard_info(location, sharding, key).await?.map(|(data, _)| data))
}

/// Where a value lies (`ShardInfo`): its shard file and byte offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardInfo {
    pub shard_path: String,
    pub offset: u64,
}

/// `ShardedKvStore.readWithShardInfo`: [`read`], and where the value lies.
pub async fn read_with_shard_info(location: &KvStoreLocation, sharding: &ShardingParameters, key: u128) -> Result<Option<(Vec<u8>, ShardInfo)>> {
    let shifted = key.checked_shr(sharding.preshift_bits).unwrap_or(0);
    let hash = match sharding.hash {
        ShardingHash::Identity => shifted,
        // `Number(input >> 32n)` through `Math.imul`: bits past 64 drop out.
        ShardingHash::MurmurHash3X86_128 => murmurhash3_x86_128_hash64(0, shifted as u64) as u128,
    };
    let bits = sharding.minishard_bits + sharding.shard_bits;
    let shard_and_minishard = (hash & ((1u128 << bits.min(127)) - 1)) as u64;
    let Some(index) = minishard_index(location, sharding, shard_and_minishard).await? else {
        return Ok(None);
    };
    let Some(i) = index.keys.iter().position(|&k| k as u128 == key) else {
        return Ok(None);
    };
    let (start, end) = (index.starts[i], index.ends[i]);
    ensure!(end >= start, "invalid chunk byte range {start}..{end}");
    let data = location
        .read(&index.shard_path, ByteRange::Range { offset: start, length: end - start })
        .await?
        .with_context(|| format!("{}{} not found", location.url(), index.shard_path))?;
    let info = ShardInfo { shard_path: index.shard_path.clone(), offset: start };
    Ok(Some((decode(data, sharding.data_encoding).await?, info)))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ng_kvstore::{KvStore, MemoryStore, ReadFuture};

    /// A memory store counting its reads, each pending once (as a fetch
    /// is), so reads begun together overlap.
    struct Counting(MemoryStore, Arc<AtomicUsize>);

    impl KvStore for Counting {
        fn read(&self, key: &str, range: ByteRange) -> ReadFuture {
            self.1.fetch_add(1, Ordering::Relaxed);
            let read = self.0.read(key, range);
            let mut yielded = false;
            let yield_once = futures::future::poll_fn(move |cx| {
                if yielded {
                    return std::task::Poll::Ready(());
                }
                yielded = true;
                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            });
            Box::pin(async move {
                yield_once.await;
                read.await
            })
        }
        fn url(&self, key: &str) -> String {
            format!("counting://{key}")
        }
    }

    #[test]
    fn the_index_cache_drops_the_least_recently_used_when_full() {
        let mut cache = Cache::default();
        let key = |i: u64| ("memory://v/".to_owned(), i);
        for i in 0..MAX_CACHED as u64 {
            cache.insert_ready(key(i), None);
        }
        // Index 0 used again: the oldest is now 1.
        cache.tick += 1;
        if let Some(Cached::Ready(_, used)) = cache.entries.get_mut(&key(0)) {
            *used = cache.tick;
        }
        cache.insert_ready(key(MAX_CACHED as u64), None);
        assert_eq!(cache.entries.len(), MAX_CACHED + 1 - MAX_CACHED / 4);
        assert!(cache.get(&key(0)).is_some() && cache.get(&key(MAX_CACHED as u64)).is_some());
        assert!(cache.get(&key(1)).is_none());
    }

    #[test]
    fn chunks_read_at_once_share_their_minishard_index() {
        // One shard, one minishard holding keys 1 and 2: the shard index
        // (16 bytes), the chunks, then the minishard index (raw, deltas).
        let (a, b) = (b"first".to_vec(), b"second".to_vec());
        let index: Vec<u64> = vec![1, 1, 0, 0, a.len() as u64, b.len() as u64];
        let data_end = (a.len() + b.len()) as u64;
        let mut shard = vec![];
        shard.extend(data_end.to_le_bytes());
        shard.extend((data_end + 48).to_le_bytes());
        shard.extend(&a);
        shard.extend(&b);
        shard.extend(index.iter().flat_map(|w| w.to_le_bytes()));
        let memory = MemoryStore::new();
        memory.insert("v/0.shard", shard);
        let reads = Arc::new(AtomicUsize::new(0));
        let location = KvStoreLocation { store: Arc::new(Counting(memory, reads.clone())), path: "v/".into() };
        let sharding = ShardingParameters {
            hash: ShardingHash::Identity,
            preshift_bits: 0,
            minishard_bits: 0,
            shard_bits: 0,
            minishard_index_encoding: DataEncoding::Raw,
            data_encoding: DataEncoding::Raw,
        };
        let (x, y) = futures::executor::block_on(futures::future::join(read(&location, &sharding, 1), read(&location, &sharding, 2)));
        assert_eq!((x.unwrap().unwrap(), y.unwrap().unwrap()), (a, b));
        // The index entry and the minishard index once, then each chunk.
        assert_eq!(reads.load(Ordering::Relaxed), 4);
    }
}
