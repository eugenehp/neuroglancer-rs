//! HTTP(S), GCS and S3. Port of `src/kvstore/http/read.ts`:
//!
//! - 404 and 403 are "not found" (`isNotFoundError`; S3 answers 403 for an
//!   absent key). In the browser, a failed request (status 0, usually CORS) is
//!   too. Natively a transport error is reported, since there it means the
//!   server could not be reached rather than a policy refusal.
//! - A suffix range is a `HEAD` for the length and then a `Range` request, not
//!   `bytes=-N`, which forces a CORS preflight and which some servers mishandle.
//! - A redirect to `<url>/` (a directory listing) is "not found".
//! - A server that ignores `Range` and answers 200 is sliced locally.

use crate::{ByteRange, KvStore, ReadFuture};

pub struct HttpStore {
    /// No trailing `/`.
    base: String,
    /// `gs://` buckets go through the GCS JSON API.
    gcs_bucket: Option<String>,
}

impl HttpStore {
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            base: base.into().trim_end_matches('/').to_owned(),
            gcs_bucket: None,
        }
    }

    /// A GCS bucket through the JSON API, as neuroglancer's `gs://` reads it.
    pub fn gcs(bucket: &str) -> Self {
        Self {
            base: "https://storage.googleapis.com".to_owned(),
            gcs_bucket: Some(bucket.to_owned()),
        }
    }

    fn object_url(&self, key: &str) -> String {
        match &self.gcs_bucket {
            // The random parameter keeps the browser from serving a cached
            // response whose CORS headers were for another origin
            // (`GcsKvStore.getObjectUrl`).
            Some(bucket) => format!(
                "https://storage.googleapis.com/storage/v1/b/{bucket}/o/{}?alt=media&neuroglancer={}",
                encode_uri_component(key),
                random_hex()
            ),
            None => format!("{}/{}", self.base, key.trim_start_matches('/')),
        }
    }
}

/// `encodeURIComponent`.
pub fn encode_uri_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.!~*'()".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `encodePathForUrl` (`src/kvstore/url.ts`): `encodeURI`, then `?`, `#` and
/// `@` escaped too.
pub fn encode_path_for_url(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b";,/:&=+$-_.!~*'()".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn random_hex() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);
    // A splitmix64 step: distinct per call, which is all the parameter needs.
    let mut z = COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    format!("{:016x}", z ^ (z >> 31))
}

/// `Range` header value for `[offset, offset + length)`.
fn range_header(offset: u64, length: u64) -> String {
    format!("bytes={}-{}", offset, offset + length - 1)
}

/// A ranged GET's body: a 200 means the server ignored the range and sent
/// the whole object, so the range is sliced out here.
fn ranged_body(status: u16, body: Vec<u8>, offset: u64, length: Option<u64>) -> Vec<u8> {
    match (status, length) {
        (200, Some(length)) => {
            let start = (offset as usize).min(body.len());
            let end = (start + length as usize).min(body.len());
            body[start..end].to_vec()
        }
        _ => body,
    }
}

impl KvStore for HttpStore {
    fn read(&self, key: &str, range: ByteRange) -> ReadFuture {
        let url = self.object_url(key);
        Box::pin(read(url, self.url(key), range))
    }

    /// `getUrl`: what the key is called, stable across calls (caches key on
    /// it) — a bucket's `gs://bucket/key`, as `GcsKvStore` names it, not the
    /// fetched URL with its one-off parameter.
    fn url(&self, key: &str) -> String {
        match &self.gcs_bucket {
            Some(bucket) => format!("gs://{bucket}/{}", key.trim_start_matches('/')),
            None => self.object_url(key),
        }
    }
}

// ── native ──────────────────────────────────────────────────────────────────

/// One HTTP client for every request: connections (and TLS sessions) kept
/// alive and reused, as a browser does.
#[cfg(not(target_arch = "wasm32"))]
fn agent() -> &'static ureq::Agent {
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    AGENT.get_or_init(|| ureq::AgentBuilder::new().max_idle_connections(256).max_idle_connections_per_host(128).build())
}

/// Threads that wait on the network (`ureq` blocks), so the chunk threads
/// do not: as many as neuroglancer's downloads in flight.
#[cfg(not(target_arch = "wasm32"))]
const IO_THREADS: usize = 100;

#[cfg(not(target_arch = "wasm32"))]
type IoJob = Box<dyn FnOnce() + Send>;

#[cfg(not(target_arch = "wasm32"))]
fn io_queue() -> &'static std::sync::Mutex<std::sync::mpsc::Sender<IoJob>> {
    static QUEUE: std::sync::OnceLock<std::sync::Mutex<std::sync::mpsc::Sender<IoJob>>> = std::sync::OnceLock::new();
    QUEUE.get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<IoJob>();
        let receiver = std::sync::Arc::new(std::sync::Mutex::new(receiver));
        for i in 0..IO_THREADS {
            let receiver = receiver.clone();
            let _ = std::thread::Builder::new().name(format!("ng-http-{i}")).spawn(move || loop {
                let job = match receiver.lock().unwrap().recv() {
                    Ok(job) => job,
                    Err(_) => return,
                };
                job();
            });
        }
        std::sync::Mutex::new(sender)
    })
}

/// A read on an I/O thread (through the disk cache, if one is set: `name`
/// is the object's stable name); the future waits without holding its own.
#[cfg(not(target_arch = "wasm32"))]
async fn read(url: String, name: String, range: ByteRange) -> crate::ReadResult {
    let (send, receive) = futures::channel::oneshot::channel();
    let job: IoJob = Box::new(move || {
        let result = match crate::disk_cache::get(&name, range) {
            Some(bytes) => Ok(Some(bytes)),
            None => {
                let result = read_blocking(url, range);
                if let Ok(Some(bytes)) = &result {
                    crate::disk_cache::put(&name, range, bytes);
                }
                result
            }
        };
        let _ = send.send(result);
    });
    let _ = io_queue().lock().unwrap().send(job);
    receive.await.unwrap_or_else(|_| Err(anyhow::anyhow!("the download thread went away")))
}

#[cfg(not(target_arch = "wasm32"))]
fn read_blocking(url: String, range: ByteRange) -> crate::ReadResult {
    use anyhow::{anyhow, Context};

    fn not_found(code: u16) -> bool {
        code == 404 || code == 403
    }

    let (offset, length) = match range {
        ByteRange::Full => (0, None),
        ByteRange::Range { offset, length } => (offset, Some(length)),
        ByteRange::Suffix(_) => {
            let head = match agent().head(&url).call() {
                Ok(r) => r,
                Err(ureq::Error::Status(code, _)) if not_found(code) => return Ok(None),
                Err(e) => return Err(anyhow!("HEAD {url}: {e}")),
            };
            let total: u64 = head
                .header("content-length")
                .and_then(|v| v.parse().ok())
                .with_context(|| format!("no content-length for {url}, needed to read a suffix"))?;
            let (offset, length) = range.resolve(total);
            (offset, Some(length))
        }
    };
    if length == Some(0) {
        return Ok(Some(vec![]));
    }
    let mut request = agent().get(&url);
    if let Some(length) = length {
        request = request.set("Range", &range_header(offset, length));
    }
    let response = match request.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(code, _)) if not_found(code) => return Ok(None),
        Err(e) => return Err(anyhow!("GET {url}: {e}")),
    };
    if redirected_to_directory(&url, response.get_url()) {
        return Ok(None);
    }
    let status = response.status();
    let mut body = Vec::new();
    std::io::Read::read_to_end(&mut response.into_reader(), &mut body).with_context(|| format!("reading {url}"))?;
    Ok(Some(ranged_body(status, body, offset, length)))
}

/// `wasRedirectedToDirectoryListing`.
fn redirected_to_directory(requested: &str, final_url: &str) -> bool {
    let path = |u: &str| {
        let after_scheme = u.split_once("://").map_or(u, |(_, r)| r);
        let path = after_scheme.find('/').map_or("/", |i| &after_scheme[i..]);
        path.split(['?', '#']).next().unwrap_or("").to_owned()
    };
    format!("{}/", path(requested)) == path(final_url)
}

// ── browser ─────────────────────────────────────────────────────────────────

/// A read through the browser cache, if one is set (`name`: the object's
/// stable name).
#[cfg(target_arch = "wasm32")]
async fn read(url: String, name: String, range: ByteRange) -> crate::ReadResult {
    if let Some(bytes) = crate::browser_cache::get(&name, range).await {
        return Ok(Some(bytes));
    }
    let result = fetch_range(url, range).await;
    if let Ok(Some(bytes)) = &result {
        // Kept in the background: the chunk does not wait for the write.
        let bytes = bytes.clone();
        wasm_bindgen_futures::spawn_local(async move { crate::browser_cache::put(&name, range, &bytes).await });
    }
    result
}

#[cfg(target_arch = "wasm32")]
async fn fetch_range(url: String, range: ByteRange) -> crate::ReadResult {
    use anyhow::{anyhow, Context};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    /// Aborts the request when the read is dropped before it finishes: a chunk
    /// nobody wants any more (a source switched away, a view moved on) stops
    /// holding one of the browser's connections to the host, which the
    /// chunks that are wanted now would otherwise queue behind.
    struct AbortOnDrop(Option<web_sys::AbortController>);
    impl Drop for AbortOnDrop {
        fn drop(&mut self) {
            if let Some(c) = self.0.take() {
                c.abort();
            }
        }
    }

    /// `fetch_once`, retried as neuroglancer's `fetchOk` does: a server
    /// pushing back (429) or failing for now (500, 502, 503, 504) is asked
    /// again after a pause — its `Retry-After` if it names one, else 0.5 s
    /// doubling to 8 s, jittered so many chunks do not come back at once —
    /// up to `RETRIES` times. Without it a rate limit's refusals were holes
    /// for good (FAFB at a 24-at-once limit: 9 tiles in 10 drawn missing).
    /// A request that never got a response (a dropped connection,
    /// `net::ERR_FAILED`) is asked again `NETWORK_RETRIES` times before it
    /// counts as absent: one blip on a `zarr.json` left a whole volume
    /// unopened. A CORS refusal fails the same way every time, so it
    /// costs only those few extra requests.
    async fn fetch(url: &str, method: &str, range: Option<String>, signal: Option<&web_sys::AbortSignal>) -> anyhow::Result<Option<web_sys::Response>> {
        const RETRIES: u32 = 6;
        const NETWORK_RETRIES: u32 = 2;
        let mut attempt = 0;
        let mut network = 0;
        loop {
            match fetch_once(url, method, range.clone(), signal).await? {
                Fetched::Retry(after) if attempt < RETRIES => {
                    crate::RETRIED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let backoff = (500.0 * 2f64.powi(attempt as i32)).min(8000.0) * (0.5 + 0.5 * js_sys::Math::random());
                    crate::global::sleep(after.unwrap_or(backoff) as i32).await;
                    attempt += 1;
                }
                Fetched::Retry(_) => return Err(anyhow!("{method} {url}: still refused after {RETRIES} retries")),
                Fetched::Unreached if network < NETWORK_RETRIES && !signal.is_some_and(|s| s.aborted()) => {
                    crate::RETRIED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    crate::global::sleep((300.0 * (1.0 + js_sys::Math::random()) * f64::from(network + 1)) as i32).await;
                    network += 1;
                }
                Fetched::Unreached => return Ok(None),
                Fetched::Done(r) => return Ok(r),
            }
        }
    }

    enum Fetched {
        Done(Option<web_sys::Response>),
        /// worth asking again, after this long (ms) if the server said
        Retry(Option<f64>),
        /// no response at all: a network or CORS failure, or an abort
        Unreached,
    }

    async fn fetch_once(url: &str, method: &str, range: Option<String>, signal: Option<&web_sys::AbortSignal>) -> anyhow::Result<Fetched> {
        let init = web_sys::RequestInit::new();
        init.set_method(method);
        if let Some(s) = signal {
            init.set_signal(Some(s));
        }
        if let Some(r) = range {
            let headers = web_sys::Headers::new().map_err(|e| anyhow!("{e:?}"))?;
            headers.set("range", &r).map_err(|e| anyhow!("{e:?}"))?;
            init.set_headers(&headers);
        }
        // the page's or a worker's `fetch`: an embedder may run the viewer in a worker
        let promise = crate::global::fetch(url, &init).context("no fetch in this global scope")?;
        let response = match JsFuture::from(promise).await {
            Ok(r) => r.dyn_into::<web_sys::Response>().map_err(|e| anyhow!("{e:?}"))?,
            // A network/CORS failure: status 0, "not found" to neuroglancer
            // once the retries are spent.
            Err(_) => return Ok(Fetched::Unreached),
        };
        match response.status() {
            404 | 403 => Ok(Fetched::Done(None)),
            s if (200..300).contains(&s) => Ok(Fetched::Done(Some(response))),
            429 | 500 | 502 | 503 | 504 => {
                // Retry-After in seconds (the HTTP-date form is left to the back-off)
                let after = response.headers().get("retry-after").ok().flatten().and_then(|v| v.trim().parse::<f64>().ok()).map(|s| (s * 1000.0).min(30_000.0));
                Ok(Fetched::Retry(after))
            }
            s => Err(anyhow!("{method} {url}: HTTP {s}")),
        }
    }

    let mut guard = AbortOnDrop(web_sys::AbortController::new().ok());
    let signal = guard.0.as_ref().map(|c| c.signal());
    let signal = signal.as_ref();
    let (offset, length) = match range {
        ByteRange::Full => (0, None),
        ByteRange::Range { offset, length } => (offset, Some(length)),
        ByteRange::Suffix(n) => {
            let Some(head) = fetch(&url, "HEAD", None, signal).await? else { return Ok(None) };
            let total: u64 = head
                .headers()
                .get("content-length")
                .ok()
                .flatten()
                .and_then(|v| v.parse().ok())
                .with_context(|| format!("no content-length for {url}, needed to read a suffix"))?;
            let (offset, length) = ByteRange::Suffix(n).resolve(total);
            (offset, Some(length))
        }
    };
    if length == Some(0) {
        return Ok(Some(vec![]));
    }
    let Some(response) = fetch(&url, "GET", length.map(|l| range_header(offset, l)), signal).await? else {
        return Ok(None);
    };
    if redirected_to_directory(&url, &response.url()) {
        return Ok(None);
    }
    let status = response.status();
    let buffer = JsFuture::from(response.array_buffer().map_err(|e| anyhow!("{e:?}"))?)
        .await
        .map_err(|e| anyhow!("{e:?}"))?;
    let body = js_sys::Uint8Array::new(&buffer).to_vec();
    // read to the end: nothing left to abort
    guard.0 = None;
    Ok(Some(ranged_body(status, body, offset, length)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_uri_component_matches_js() {
        assert_eq!(encode_uri_component("a/b c?d=é"), "a%2Fb%20c%3Fd%3D%C3%A9");
        assert_eq!(encode_uri_component("A-z_0.9!~*'()"), "A-z_0.9!~*'()");
    }

    #[test]
    fn encode_path_for_url_matches_js() {
        // encodeURI("a b/c?d#e@f[g]é").replace(/[?#@]/g, …)
        assert_eq!(encode_path_for_url("a b/c?d#e@f[g]é"), "a%20b/c%3Fd%23e%40f%5Bg%5D%C3%A9");
        assert_eq!(encode_path_for_url("scale0/image"), "scale0/image");
    }

    #[test]
    fn a_buckets_urls_are_stable_and_its_fetches_are_not() {
        let gcs = HttpStore::gcs("bucket");
        // Caches key on `url`: the same key, the same name (`GcsKvStore.getUrl`).
        assert_eq!(gcs.url("a/0.shard"), "gs://bucket/a/0.shard");
        assert_eq!(gcs.url("a/0.shard"), gcs.url("a/0.shard"));
        // Each fetch its own parameter (cached CORS headers).
        let fetched = gcs.object_url("a/0.shard");
        assert!(fetched.starts_with("https://storage.googleapis.com/storage/v1/b/bucket/o/a%2F0.shard?alt=media&neuroglancer="), "{fetched}");
        assert_ne!(fetched, gcs.object_url("a/0.shard"));
    }

    #[test]
    fn directory_redirects() {
        assert!(redirected_to_directory("http://h/a/b", "http://h/a/b/"));
        assert!(!redirected_to_directory("http://h/a/b", "http://h/a/b"));
    }
}
