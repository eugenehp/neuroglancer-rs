//! The chunk manager: which chunks to fetch, in what order, and which to keep.
//! A simplified port of `src/chunk_manager/` — the same model (each frame,
//! views request the chunks they need with a priority; downloads are bounded
//! and go highest-priority first; memory is a budget, and what was not asked
//! for recently is evicted first), without neuroglancer's separate
//! system-memory and GPU-memory tiers or its prefetching.
//!
//! The manager is generic over what a chunk is: a key, a loader future, and
//! the loaded value with its size. Loading runs on an [`Executor`]: worker
//! threads natively, the page's event loop in the browser.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use futures::future::{AbortHandle, Abortable};

use ng_kvstore::MaybeSendSync;

mod adaptive;
pub use adaptive::Adaptive;

/// A value loading in the background, as the executor runs it.
#[cfg(not(target_arch = "wasm32"))]
pub type Task = futures::future::BoxFuture<'static, ()>;
#[cfg(target_arch = "wasm32")]
pub type Task = futures::future::LocalBoxFuture<'static, ()>;

/// A loader's future for one chunk.
#[cfg(not(target_arch = "wasm32"))]
pub type LoadFuture<V> = futures::future::BoxFuture<'static, anyhow::Result<V>>;
#[cfg(target_arch = "wasm32")]
pub type LoadFuture<V> = futures::future::LocalBoxFuture<'static, anyhow::Result<V>>;

/// Runs loading tasks.
pub trait Executor: MaybeSendSync {
    fn spawn(&self, task: Task);
}

/// Native: a pool of threads polling the tasks. A task waiting (on a
/// download, on another's read of an index) yields its thread to the rest,
/// so the downloads in flight are bounded by the chunk manager's limit, not
/// by the threads, and the threads go to decoding.
#[cfg(not(target_arch = "wasm32"))]
pub struct ThreadPool {
    pool: futures::executor::ThreadPool,
}

#[cfg(not(target_arch = "wasm32"))]
impl ThreadPool {
    pub fn new(threads: usize) -> Self {
        let pool = futures::executor::ThreadPool::builder()
            .pool_size(threads.max(1))
            .name_prefix("ng-chunk-")
            .create()
            .expect("spawning the chunk threads");
        Self { pool }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Executor for ThreadPool {
    fn spawn(&self, task: Task) {
        self.pool.spawn_ok(task);
    }
}

/// Browser: `spawn_local` on the page's event loop.
#[cfg(target_arch = "wasm32")]
pub struct LocalExecutor;

#[cfg(target_arch = "wasm32")]
impl Executor for LocalExecutor {
    fn spawn(&self, task: Task) {
        wasm_bindgen_futures::spawn_local(task);
    }
}

/// Where a chunk is (`ChunkState`, reduced).
#[derive(Debug, Clone)]
pub enum State<V> {
    /// Requested, waiting for a download slot.
    Queued,
    Downloading,
    Ready(Arc<V>),
    Failed(Arc<str>),
}

struct Entry<V> {
    state: State<V>,
    /// Highest priority requested this frame; `None` if not requested.
    priority: Option<f64>,
    /// Requested this frame as visible, not only to prefetch.
    visible: bool,
    /// Last frame it was requested in.
    last_used: u64,
    bytes: usize,
    /// How long its download took (ms), once done, if a clock is set.
    latency: Option<f64>,
    /// While downloading: stops it (and its HTTP request) when unwanted.
    abort: Option<AbortHandle>,
    /// Which download this entry waits on: a result from one aborted or
    /// superseded since is not this entry's.
    download: u64,
}

/// Results handed back by finished tasks, with the download they came from.
type Inbox<K, V> = Arc<Mutex<Vec<(K, u64, anyhow::Result<(V, usize)>, Option<f64>)>>>;

pub struct ChunkManager<K, V> {
    entries: HashMap<K, Entry<V>>,
    inbox: Inbox<K, V>,
    executor: Arc<dyn Executor>,
    /// Called (from any thread) when a result arrives: schedule a redraw.
    wake: Arc<dyn Fn() + Send + Sync>,
    frame: u64,
    in_flight: usize,
    /// Numbers each download started.
    downloads: u64,
    pub max_concurrent_downloads: usize,
    /// Bytes of ready chunks to keep (`systemMemoryLimit`).
    pub memory_budget: usize,
    memory_used: usize,
    /// Milliseconds from some origin (`performance.now()`), to time
    /// downloads by.
    pub clock: Option<fn() -> f64>,
    /// Running totals of the downloads (`DownloadStats`).
    pub stats: DownloadStats,
    /// Counts changes to what is held: a chunk ready or failed, a ready one
    /// evicted or dropped. A host that builds from the held values (a batch
    /// of skeletons) rebuilds only as it moves.
    revision: u64,
}

/// Downloads finished since the manager was made, for a host that tunes
/// `max_concurrent_downloads` by what it measures: counts, bytes, summed
/// latency (when a clock is set), and how many were left waiting for a slot
/// at the end of the last frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct DownloadStats {
    pub done: u64,
    pub failed: u64,
    pub bytes: u64,
    pub latency_ms: f64,
    pub timed: u64,
    pub waiting: usize,
}

/// What a load returns: the value and its size in bytes.
pub type Loaded<V> = (V, usize);

impl<K, V> ChunkManager<K, V>
where
    K: Clone + Eq + Hash + MaybeSendSync + 'static,
    V: MaybeSendSync + 'static,
{
    pub fn new(executor: Arc<dyn Executor>, wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            entries: HashMap::new(),
            inbox: Arc::default(),
            executor,
            wake,
            frame: 0,
            in_flight: 0,
            downloads: 0,
            // neuroglancer's defaults: `concurrentDownloads` 100, 2 GB system memory.
            max_concurrent_downloads: 100,
            memory_budget: 2_000_000_000,
            memory_used: 0,
            clock: None,
            stats: DownloadStats::default(),
            revision: 0,
        }
    }

    /// Starts a frame: priorities from the previous one are forgotten.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        for e in self.entries.values_mut() {
            e.priority = None;
            e.visible = false;
        }
    }

    /// Asks for `key` at `priority` (higher first) this frame; returns its
    /// state. Not yet known chunks are queued.
    pub fn request(&mut self, key: &K, priority: f64) -> &State<V> {
        let e = self.entry(key, priority);
        e.visible = true;
        &e.state
    }

    /// Asks for `key` ahead of need (`ChunkPriorityTier.PREFETCH`): queued
    /// at `priority` like [`request`](Self::request), but not counted as
    /// visible.
    pub fn request_prefetch(&mut self, key: &K, priority: f64) -> &State<V> {
        &self.entry(key, priority).state
    }

    fn entry(&mut self, key: &K, priority: f64) -> &mut Entry<V> {
        let frame = self.frame;
        let e = self.entries.entry(key.clone()).or_insert(Entry {
            state: State::Queued,
            priority: None,
            visible: false,
            last_used: frame,
            bytes: 0,
            latency: None,
            abort: None,
            download: 0,
        });
        e.priority = Some(e.priority.map_or(priority, |p| p.max(priority)));
        e.last_used = frame;
        e
    }

    /// Every chunk known: its key, state, whether it was requested as
    /// visible this frame, and its size in memory.
    pub fn for_each(&self, mut f: impl FnMut(&K, &State<V>, bool, usize, Option<f64>)) {
        for (k, e) in &self.entries {
            f(k, &e.state, e.visible, e.bytes, e.latency);
        }
    }

    /// Chunks requested this frame ahead of need only (prefetched).
    pub fn prefetching(&self) -> usize {
        self.entries.values().filter(|e| e.priority.is_some() && !e.visible).count()
    }

    /// The ready value of `key`, if any, without requesting it.
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        match self.entries.get(key).map(|e| &e.state) {
            Some(State::Ready(v)) => Some(v.clone()),
            _ => None,
        }
    }

    /// Takes in finished loads. Returns whether any arrived.
    pub fn poll(&mut self) -> bool {
        let done: Vec<_> = std::mem::take(&mut *self.inbox.lock().unwrap());
        let mut any = false;
        for (key, download, result, latency) in done {
            // a cancelled download was counted off when it was cancelled
            let Some(e) = self.entries.get_mut(&key) else { continue };
            if !matches!(e.state, State::Downloading) || e.download != download {
                continue;
            }
            self.in_flight -= 1;
            any = true;
            self.revision += 1;
            e.abort = None;
            e.latency = latency;
            if let Some(ms) = latency {
                self.stats.latency_ms += ms;
                self.stats.timed += 1;
            }
            match result {
                Ok((value, bytes)) => {
                    e.state = State::Ready(Arc::new(value));
                    e.bytes = bytes;
                    self.memory_used += bytes;
                    self.stats.done += 1;
                    self.stats.bytes += bytes as u64;
                }
                Err(err) => {
                    e.state = State::Failed(format!("{err:#}").into());
                    self.stats.failed += 1;
                }
            }
        }
        any
    }

    /// Ends a frame: starts the highest-priority queued loads that fit, and
    /// evicts least-recently-used ready chunks over budget. `load` makes the
    /// future for a key.
    pub fn end_frame(&mut self, mut load: impl FnMut(&K) -> LoadFuture<Loaded<V>>) {
        // Queued chunks no longer wanted are dropped, so the queue holds this
        // frame's requests only.
        self.entries
            .retain(|_, e| !matches!(e.state, State::Queued) || e.priority.is_some());
        let mut queued: Vec<(K, f64)> = self
            .entries
            .iter()
            .filter(|(_, e)| matches!(e.state, State::Queued))
            .map(|(k, e)| (k.clone(), e.priority.unwrap_or(f64::NEG_INFINITY)))
            .collect();
        queued.sort_by(|a, b| b.1.total_cmp(&a.1));
        let wanted = queued.len();
        let mut started = 0;
        for (key, _) in queued {
            if self.in_flight >= self.max_concurrent_downloads {
                break;
            }
            started += 1;
            self.downloads += 1;
            let download = self.downloads;
            let (abort, registration) = AbortHandle::new_pair();
            let e = self.entries.get_mut(&key).unwrap();
            e.state = State::Downloading;
            e.abort = Some(abort);
            e.download = download;
            self.in_flight += 1;
            let future = Abortable::new(load(&key), registration);
            let inbox = self.inbox.clone();
            let wake = self.wake.clone();
            let clock = self.clock;
            let start = clock.map(|c| c());
            self.executor.spawn(Box::pin(async move {
                // aborted: the future (and the request under it) is dropped
                let Ok(result) = future.await else { return };
                let latency = clock.zip(start).map(|(c, t)| c() - t);
                inbox.lock().unwrap().push((key, download, result, latency));
                wake();
            }));
        }
        self.stats.waiting = wanted - started;
        self.evict();
    }

    fn evict(&mut self) {
        if self.memory_used <= self.memory_budget {
            return;
        }
        let frame = self.frame;
        let mut ready: Vec<(K, u64, usize)> = self
            .entries
            .iter()
            .filter(|(_, e)| matches!(e.state, State::Ready(_)) && e.last_used < frame)
            .map(|(k, e)| (k.clone(), e.last_used, e.bytes))
            .collect();
        ready.sort_by_key(|&(_, used, _)| used);
        for (key, _, bytes) in ready {
            if self.memory_used <= self.memory_budget {
                break;
            }
            self.entries.remove(&key);
            self.memory_used -= bytes;
            self.revision += 1;
        }
    }

    /// Forgets every chunk whose key fails `keep` — those of a layer that is
    /// gone (a source switched away): downloads still running are cancelled,
    /// their HTTP requests with them, so the slots go to what is wanted now
    /// instead of the new source queueing behind the old one's backlog; ready
    /// data is freed. Returns how many downloads were cancelled.
    pub fn retain_keys(&mut self, mut keep: impl FnMut(&K) -> bool) -> usize {
        let (mut cancelled, mut freed, mut dropped) = (0, 0, 0);
        self.entries.retain(|k, e| {
            if keep(k) {
                return true;
            }
            if let Some(a) = e.abort.take() {
                a.abort();
            }
            match e.state {
                State::Downloading => cancelled += 1,
                State::Ready(_) => freed += e.bytes,
                _ => {}
            }
            dropped += 1;
            false
        });
        if dropped > 0 {
            self.revision += 1;
        }
        self.in_flight -= cancelled;
        self.memory_used -= freed;
        cancelled
    }

    /// Loads still running.
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    pub fn memory_used(&self) -> usize {
        self.memory_used
    }

    /// Swaps `key`'s ready value for `value` (`bytes` in memory): a smaller
    /// form of it, once what it was loaded for holds the rest elsewhere.
    /// False (and nothing changed) unless `key` is ready.
    pub fn replace(&mut self, key: &K, value: Arc<V>, bytes: usize) -> bool {
        let Some(e) = self.entries.get_mut(key) else { return false };
        if !matches!(e.state, State::Ready(_)) {
            return false;
        }
        e.state = State::Ready(value);
        self.memory_used = self.memory_used - e.bytes + bytes;
        e.bytes = bytes;
        true
    }

    /// Changes with what is held (see the field): equal values, the same
    /// ready set as before.
    pub fn revision(&self) -> u64 {
        self.revision
    }
}

impl<K, V> Drop for ChunkManager<K, V> {
    /// A manager going away (its source closed) stops its downloads.
    fn drop(&mut self) {
        for e in self.entries.values_mut() {
            if let Some(a) = e.abort.take() {
                a.abort();
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Runs tasks immediately, in order.
    struct Inline;
    impl Executor for Inline {
        fn spawn(&self, task: Task) {
            futures::executor::block_on(task);
        }
    }

    fn manager(budget: usize, concurrent: usize) -> (ChunkManager<u32, Vec<u8>>, Arc<AtomicUsize>) {
        let wakes = Arc::new(AtomicUsize::new(0));
        let w = wakes.clone();
        let mut m = ChunkManager::new(Arc::new(Inline), Arc::new(move || {
            w.fetch_add(1, Ordering::SeqCst);
        }));
        m.memory_budget = budget;
        m.max_concurrent_downloads = concurrent;
        (m, wakes)
    }

    fn load(k: &u32) -> LoadFuture<Loaded<Vec<u8>>> {
        let k = *k;
        Box::pin(async move { Ok((vec![k as u8; 10], 10)) })
    }

    #[test]
    fn highest_priority_loads_first_within_the_download_limit() {
        let (mut m, wakes) = manager(1000, 2);
        m.begin_frame();
        for (k, p) in [(1, 1.0), (2, 5.0), (3, 3.0)] {
            m.request(&k, p);
        }
        m.end_frame(load);
        assert!(m.poll());
        assert!(m.get(&2).is_some() && m.get(&3).is_some());
        assert!(m.get(&1).is_none(), "lowest priority waits for a slot");
        assert_eq!(wakes.load(Ordering::SeqCst), 2);
        // Next frame it gets its turn.
        m.begin_frame();
        m.request(&1, 1.0);
        m.end_frame(load);
        m.poll();
        assert!(m.get(&1).is_some());
    }

    #[test]
    fn prefetched_chunks_load_behind_visible_ones_and_are_not_visible() {
        let (mut m, _) = manager(1000, 1);
        m.begin_frame();
        m.request_prefetch(&7, 100.0);
        m.request(&1, -5.0);
        m.request_prefetch(&1, 200.0);
        m.end_frame(load);
        m.poll();
        // One download slot: the visible chunk took it. (A chunk asked for
        // both ways is visible, at the higher priority.)
        assert!(m.get(&1).is_some() && m.get(&7).is_none());
        let mut visible = vec![];
        m.for_each(|k, _, v, _, _| visible.push((*k, v)));
        visible.sort();
        assert_eq!(visible, [(1, true), (7, false)]);
        m.begin_frame();
        m.request_prefetch(&7, 100.0);
        m.end_frame(load);
        m.poll();
        assert!(m.get(&7).is_some(), "prefetched once a slot is free");
    }

    #[test]
    fn a_ready_value_is_replaced_with_its_size() {
        let (mut m, _) = manager(1000, 10);
        assert!(!m.replace(&1, Arc::new(vec![]), 0), "not loaded yet");
        m.begin_frame();
        m.request(&1, 0.0);
        m.end_frame(load);
        m.poll();
        assert_eq!(m.memory_used(), 10);
        assert!(m.replace(&1, Arc::new(vec![9]), 1));
        assert_eq!((m.memory_used(), m.get(&1).unwrap().as_slice()), (1, &[9u8][..]));
    }

    #[test]
    fn revision_moves_with_what_is_held() {
        let (mut m, _) = manager(15, 10);
        let r0 = m.revision();
        m.begin_frame();
        m.request(&1, 0.0);
        m.end_frame(load);
        assert_eq!(m.revision(), r0, "downloading changes nothing held");
        m.poll();
        let r1 = m.revision();
        assert!(r1 > r0, "a chunk arrived");
        m.begin_frame();
        m.request(&1, 0.0);
        m.end_frame(load);
        m.poll();
        assert_eq!(m.revision(), r1, "asked again, nothing new");
        // a second chunk over the budget evicts the first
        m.begin_frame();
        m.request(&2, 0.0);
        m.end_frame(load);
        m.poll();
        let r2 = m.revision();
        m.begin_frame();
        m.end_frame(load);
        assert!(m.get(&1).is_none() && m.revision() > r2, "evicted");
        let r3 = m.revision();
        m.retain_keys(|_| false);
        assert!(m.revision() > r3, "dropped");
    }

    #[test]
    fn unrequested_queue_entries_are_dropped_and_lru_evicted_over_budget() {
        let (mut m, _) = manager(25, 10);
        m.begin_frame();
        for k in 0..3 {
            m.request(&k, 0.0);
        }
        m.end_frame(load);
        m.poll();
        assert_eq!(m.memory_used(), 30);
        // Only chunk 2 is still wanted: the others go, oldest first, until under budget.
        m.begin_frame();
        m.request(&2, 0.0);
        m.end_frame(load);
        assert!(m.memory_used() <= 25);
        assert!(m.get(&2).is_some(), "a chunk requested this frame is never evicted");
    }

    #[test]
    fn failures_are_kept_and_reported() {
        let (mut m, _) = manager(100, 10);
        m.begin_frame();
        m.request(&7, 0.0);
        m.end_frame(|_| Box::pin(async { Err(anyhow::anyhow!("boom")) }));
        m.poll();
        m.begin_frame();
        assert!(matches!(m.request(&7, 0.0), State::Failed(e) if e.contains("boom")));
    }

    /// Holds tasks until told to run them: downloads that stay in flight.
    #[derive(Default)]
    struct Held(Mutex<Vec<Task>>);
    impl Executor for Held {
        fn spawn(&self, task: Task) {
            self.0.lock().unwrap().push(task);
        }
    }

    #[test]
    fn a_dropped_layers_downloads_are_cancelled_and_free_their_slots() {
        let held = Arc::new(Held::default());
        let mut m: ChunkManager<u32, Vec<u8>> = ChunkManager::new(held.clone(), Arc::new(|| {}));
        m.max_concurrent_downloads = 2;
        let dropped = Arc::new(AtomicUsize::new(0));
        // a load that never finishes and counts being dropped (its request aborted)
        struct OnDrop(Arc<AtomicUsize>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let slow = |d: Arc<AtomicUsize>| {
            move |_: &u32| -> LoadFuture<Loaded<Vec<u8>>> {
                let g = OnDrop(d.clone());
                Box::pin(async move {
                    let _g = g;
                    futures::future::pending::<anyhow::Result<Loaded<Vec<u8>>>>().await
                })
            }
        };
        // polls each held task once; keeps those still running
        let step = |held: &Held| {
            let mut tasks = held.0.lock().unwrap();
            let waker = futures::task::noop_waker();
            let mut cx = std::task::Context::from_waker(&waker);
            tasks.retain_mut(|t| t.as_mut().poll(&mut cx).is_pending());
        };
        // source A wants 1 and 2: both start
        m.begin_frame();
        m.request(&1, 1.0);
        m.request(&2, 1.0);
        m.end_frame(slow(dropped.clone()));
        assert_eq!(m.in_flight(), 2);
        // switched to source B: A's keys (1, 2) are forgotten, its downloads
        // cancelled; 3 and 4 take the slots in the same frame
        assert_eq!(m.retain_keys(|k| *k > 2), 2);
        m.begin_frame();
        m.request(&3, 1.0);
        m.request(&4, 1.0);
        m.end_frame(slow(dropped.clone()));
        // A's downloads were cancelled (an aborted task ends at its next
        // poll, dropping its load); B's took the slots and run on
        step(&held);
        assert_eq!(dropped.load(Ordering::SeqCst), 2, "A's loads dropped unfinished");
        assert_eq!(held.0.lock().unwrap().len(), 2);
        assert_eq!(m.in_flight(), 2);
        assert!(!m.poll(), "no result from a cancelled download");
        m.begin_frame();
        assert!(matches!(m.request(&3, 1.0), State::Downloading));
        // dropping the manager (its source closed) cancels the rest
        drop(m);
        step(&held);
        assert_eq!(dropped.load(Ordering::SeqCst), 4, "B's loads cancelled with the manager");
    }
}
