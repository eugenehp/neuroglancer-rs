//! How many volume chunks to download at once, kept as high as the server
//! allows.
//!
//! Measured on FAFB through a faithful slow line (atlas's tools/shaper.mjs:
//! one shared bandwidth budget and a round trip, unlike Chrome's own
//! throttling, which lets parallel requests exceed its cap and so made
//! fewer look better): the whole load takes the same time at any number on a
//! line that is full (5 and 20 Mbit/s: within 5%), but the EM — the layer on
//! show — completes far sooner with more in flight (5 Mbit/s: 10.6 s at 128,
//! 18.4 s at 16), because a few large segmentation chunks otherwise hold the
//! slots; and on a fast line more is simply faster (100 Mbit/s: 5.4 s at 128,
//! 6.0 at 64, 11.9 at 16). Latency per chunk grows with the number in flight,
//! but chunks are requested highest-priority first, so that only delays the
//! ones the view needs least. A latency target (keep each chunk near 450 ms)
//! cut the number to 12 on the slow line and took the EM twice as long.
//!
//! So: as many as allowed (128; past ~100 HTTP/2 queues them in the browser
//! anyway), and backing off only when the server pushes back — requests
//! refused or failing (429, 5xx: the host counts its retries in `failed` as
//! well as the chunks that gave up) in a window halve the number; each clean
//! window after adds an eighth back. Windows of ~¾ s, counted only while
//! chunks wait for a slot or fail (else the number limits nothing).
//!
//! `Viewer::settings.adaptive_downloads` runs one over the volume chunks'
//! downloads, its pushback the chunks that failed and the HTTP requests
//! retried (`ng_kvstore::retried`).

use crate::DownloadStats;

/// A window's length (ms).
const WINDOW_MS: f64 = 750.0;
/// Clean windows to wait after a back-off before adding again.
const COOL_DOWN: u32 = 2;

#[derive(Debug, Clone, Copy)]
struct Window {
    t0: f64,
    base: DownloadStats,
}

#[derive(Debug, Clone)]
pub struct Adaptive {
    /// Chunks to download at once now.
    pub concurrency: usize,
    min: usize,
    max: usize,
    window: Option<Window>,
    /// clean windows still to wait before adding again
    cool: u32,
    /// The last window's throughput (bytes per ms of chunk data, decoded),
    /// mean latency per chunk (ms; NaN when not timed), and failures.
    pub rate: f64,
    pub latency: f64,
    pub failed: u64,
    /// Windows measured so far, and back-offs.
    pub windows: u32,
    pub backoffs: u32,
}

impl Adaptive {
    pub fn new(start: usize, min: usize, max: usize) -> Self {
        Self {
            concurrency: start.clamp(min, max),
            min,
            max,
            window: None,
            cool: 0,
            rate: f64::NAN,
            latency: f64::NAN,
            failed: 0,
            windows: 0,
            backoffs: 0,
        }
    }

    /// Takes the downloads' totals at `now` (ms). Returns whether
    /// `concurrency` changed.
    pub fn observe(&mut self, now: f64, s: DownloadStats) -> bool {
        let Some(w) = self.window else {
            self.window = Some(Window { t0: now, base: s });
            return false;
        };
        let failed = s.failed - w.base.failed;
        // nothing waiting and nothing failing: the number limits nothing
        if s.waiting == 0 && failed == 0 {
            self.window = Some(Window { t0: now, base: s });
            return false;
        }
        let ms = now - w.t0;
        if ms < WINDOW_MS {
            return false;
        }
        self.window = Some(Window { t0: now, base: s });
        let timed = s.timed - w.base.timed;
        self.rate = (s.bytes - w.base.bytes) as f64 / ms;
        self.latency = if timed > 0 {
            (s.latency_ms - w.base.latency_ms) / timed as f64
        } else {
            f64::NAN
        };
        self.failed = failed;
        self.windows += 1;
        let before = self.concurrency;
        if failed > 0 {
            self.concurrency = (before / 2).max(self.min);
            self.cool = COOL_DOWN;
            self.backoffs += 1;
        } else if self.cool > 0 {
            self.cool -= 1;
        } else {
            self.concurrency = (before + (before / 8).max(1)).min(self.max);
        }
        self.concurrency != before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `secs` seconds (two windows each) of downloads that fail `fails(i)`
    /// times in second `i`; the number after each second.
    fn run(a: &mut Adaptive, fails: impl Fn(usize) -> u64, secs: usize) -> Vec<usize> {
        let mut s = DownloadStats {
            waiting: 50,
            ..Default::default()
        };
        let mut now = 0.0;
        a.observe(now, s);
        let mut seen = vec![];
        for i in 0..secs {
            for _ in 0..2 {
                now += WINDOW_MS;
                s.done += 40;
                s.failed += fails(i);
                a.observe(now, s);
            }
            seen.push(a.concurrency);
        }
        seen
    }

    #[test]
    fn stays_at_the_most_while_nothing_fails() {
        let mut a = Adaptive::new(128, 4, 128);
        assert!(run(&mut a, |_| 0, 10).iter().all(|&c| c == 128));
        // and climbs back to it from lower
        let mut a = Adaptive::new(16, 4, 128);
        let seen = run(&mut a, |_| 0, 20);
        assert_eq!(*seen.last().unwrap(), 128, "{seen:?}");
    }

    #[test]
    fn backs_off_when_the_server_pushes_back() {
        // rate-limited for three seconds: halved each window, then recovers
        let mut a = Adaptive::new(128, 4, 128);
        let seen = run(&mut a, |i| if (2..5).contains(&i) { 3 } else { 0 }, 40);
        assert!(seen[4] <= 4, "{seen:?}");
        assert_eq!(a.backoffs, 6);
        assert_eq!(*seen.last().unwrap(), 128, "{seen:?}");
    }

    #[test]
    fn idle_stretches_are_not_measured() {
        let mut a = Adaptive::new(64, 4, 128);
        let idle = DownloadStats::default();
        for t in 0..10 {
            assert!(!a.observe(t as f64 * 1000.0, idle));
        }
        assert_eq!((a.concurrency, a.windows), (64, 0));
    }
}
