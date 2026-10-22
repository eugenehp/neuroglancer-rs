//! The browser's global scope, page or worker: `fetch` and `caches` exist on
//! both (`WindowOrWorkerGlobalScope`), but `web_sys` puts them on `Window`
//! and `WorkerGlobalScope` separately. Reading through `web_sys::window()`
//! alone fails in a Web Worker, where an embedder may run the viewer (an
//! `OffscreenCanvas` render worker).
use wasm_bindgen::JsCast;

/// `fetch(url, init)` on whichever global this runs under.
pub(crate) fn fetch(url: &str, init: &web_sys::RequestInit) -> Option<js_sys::Promise> {
    let global = js_sys::global();
    if let Some(w) = global.dyn_ref::<web_sys::Window>() {
        return Some(w.fetch_with_str_and_init(url, init));
    }
    global.dyn_ref::<web_sys::WorkerGlobalScope>().map(|w| w.fetch_with_str_and_init(url, init))
}

/// `caches` on whichever global this runs under (absent off HTTPS/localhost).
pub(crate) fn caches() -> Option<web_sys::CacheStorage> {
    let global = js_sys::global();
    if let Some(w) = global.dyn_ref::<web_sys::Window>() {
        return w.caches().ok();
    }
    global.dyn_ref::<web_sys::WorkerGlobalScope>().and_then(|w| w.caches().ok())
}
