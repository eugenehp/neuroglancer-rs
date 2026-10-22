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

use ng_kvstore::MaybeSendSync;

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
}

/// Results handed back by finished tasks.
type Inbox<K, V> = Arc<Mutex<Vec<(K, anyhow::Result<(V, usize)>, Option<f64>)>>>;

pub struct ChunkManager<K, V> {
    entries: HashMap<K, Entry<V>>,
    inbox: Inbox<K, V>,
    executor: Arc<dyn Executor>,
    /// Called (from any thread) when a result arrives: schedule a redraw.
    wake: Arc<dyn Fn() + Send + Sync>,
    frame: u64,
    in_flight: usize,
    pub max_concurrent_downloads: usize,
    /// Bytes of ready chunks to keep (`systemMemoryLimit`).
    pub memory_budget: usize,
    memory_used: usize,
    /// Milliseconds from some origin (`performance.now()`), to time
    /// downloads by.
    pub clock: Option<fn() -> f64>,
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
            // neuroglancer's defaults: `concurrentDownloads` 100, 2 GB system memory.
            max_concurrent_downloads: 100,
            memory_budget: 2_000_000_000,
            memory_used: 0,
            clock: None,
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
        let any = !done.is_empty();
        for (key, result, latency) in done {
            self.in_flight -= 1;
            let Some(e) = self.entries.get_mut(&key) else { continue };
            e.latency = latency;
            match result {
                Ok((value, bytes)) => {
                    e.state = State::Ready(Arc::new(value));
                    e.bytes = bytes;
                    self.memory_used += bytes;
                }
                Err(err) => e.state = State::Failed(format!("{err:#}").into()),
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
        for (key, _) in queued {
            if self.in_flight >= self.max_concurrent_downloads {
                break;
            }
            self.entries.get_mut(&key).unwrap().state = State::Downloading;
            self.in_flight += 1;
            let future = load(&key);
            let inbox = self.inbox.clone();
            let wake = self.wake.clone();
            let clock = self.clock;
            let start = clock.map(|c| c());
            self.executor.spawn(Box::pin(async move {
                let result = future.await;
                let latency = clock.zip(start).map(|(c, t)| c() - t);
                inbox.lock().unwrap().push((key, result, latency));
                wake();
            }));
        }
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
        }
    }

    /// Loads still running.
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    pub fn memory_used(&self) -> usize {
        self.memory_used
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
}
