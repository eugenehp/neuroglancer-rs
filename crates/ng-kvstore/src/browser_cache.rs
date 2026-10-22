//! A cache of remote reads in the browser's Cache Storage, off unless asked
//! for ([`set_browser_cache`]): a dataset seen before loads without the
//! network. Neuroglancer keeps no such cache (it even keeps the browser's
//! HTTP cache off for buckets), so data changed on the server since it was
//! cached is served as cached; "not found" answers are never kept. Cache
//! Storage exists on secure pages only (HTTPS, localhost); elsewhere this
//! does nothing. The browser bounds its size.

use std::cell::RefCell;

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::ByteRange;

thread_local! {
    static CACHE: RefCell<Option<String>> = const { RefCell::new(None) };
    /// The cache, once opened.
    static OPENED: RefCell<Option<web_sys::Cache>> = const { RefCell::new(None) };
}

/// Keeps remote reads in the Cache Storage cache called `name`, and serves
/// them from there; `None`: no cache.
pub fn set_browser_cache(name: Option<&str>) {
    CACHE.with(|c| *c.borrow_mut() = name.map(str::to_owned));
    OPENED.with(|o| *o.borrow_mut() = None);
}

/// Where an entry is filed: a URL of its own (Cache Storage keys requests).
fn entry_url(name: &str, range: ByteRange) -> String {
    format!("https://neuroglancer-rs.cache.invalid/{}", crate::cache_key(name, range))
}

async fn open() -> Option<web_sys::Cache> {
    if let Some(cache) = OPENED.with(|o| o.borrow().clone()) {
        return Some(cache);
    }
    let cache_name = CACHE.with(|c| c.borrow().clone())?;
    let storage = crate::global::caches()?;
    let cache: web_sys::Cache = JsFuture::from(storage.open(&cache_name)).await.ok()?.dyn_into().ok()?;
    OPENED.with(|o| *o.borrow_mut() = Some(cache.clone()));
    Some(cache)
}

/// The cached bytes of a read, if cached.
pub(crate) async fn get(name: &str, range: ByteRange) -> Option<Vec<u8>> {
    let cache = open().await?;
    let hit = JsFuture::from(cache.match_with_str(&entry_url(name, range))).await.ok()?;
    let response: web_sys::Response = hit.dyn_into().ok()?;
    let buffer = JsFuture::from(response.array_buffer().ok()?).await.ok()?;
    Some(js_sys::Uint8Array::new(&buffer).to_vec())
}

/// Keeps a read's bytes.
pub(crate) async fn put(name: &str, range: ByteRange, bytes: &[u8]) {
    let Some(cache) = open().await else { return };
    let mut body = bytes.to_vec();
    let Ok(response) = web_sys::Response::new_with_opt_u8_array(Some(&mut body)) else { return };
    let _ = JsFuture::from(cache.put_with_str(&entry_url(name, range), &response)).await;
}
