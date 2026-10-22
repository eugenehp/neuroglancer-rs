//! Browser shell: `neuroglancer::web`'s full-window canvas around the engine.
//!
//! Build with `scripts/build-web.sh`, serve `crates/ng-web/www/`, and open
//! `index.html?url=<data source>` — anything neuroglancer accepts
//! (`precomputed://https://…`, `zarr://…`, `https://…/x.nii.gz`) — or
//! `index.html#!{…}` with neuroglancer's viewer state, as in its share links
//! (also `#!gs://…/state.json`). The data must allow cross-origin requests,
//! as for neuroglancer.
//!
//! `?ui=0`: the panels alone (neuroglancer's `showUIControls` off).
//! `?domtext=0` / `?domtext=1`: all UI text on the canvas / all as DOM.
//! `?workers=0`: chunks decoded on the page rather than in workers.
//! `?credits=0`: without the port's credits (footer, Settings' About).
//! `?cache=1`: downloads kept in the browser's Cache Storage (HTTPS or
//! localhost only), so a dataset seen before loads from there.

#[cfg(target_arch = "wasm32")]
mod web {
    use neuroglancer::web::{with_engine, DomText, WebOptions};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    pub async fn start() -> Result<(), JsValue> {
        console_error_panic_hook::set_once();
        let _ = console_log::init_with_level(log::Level::Info);
        // In a decode worker (`decode_worker.js`): nothing to set up.
        let Some(window) = web_sys::window() else { return Ok(()) };
        let params = web_sys::UrlSearchParams::new_with_str(&window.location().search()?)?;
        let dom_text = match params.get("domtext").as_deref() {
            Some("0") => DomText::None,
            Some("1") => DomText::All,
            _ => DomText::Translucent,
        };
        // `?workers=0`: decode on the page.
        let decode_workers = (params.get("workers").as_deref() != Some("0")).then(|| "./decode_worker.js".to_owned());
        // `?cache=1`: what is downloaded kept in the browser, for next time.
        let cache = (params.get("cache").as_deref() == Some("1")).then(|| "neuroglancer-rs".to_owned());
        // `?credits=0`: without the port's credits (for comparing with neuroglancer).
        let no_attribution = params.get("credits").as_deref() == Some("0");
        neuroglancer::web::start(WebOptions { canvas: None, no_ui: params.get("ui").as_deref() == Some("0"), dom_text, url_hash: true, decode_workers, cache, no_attribution }).await?;

        let hash = window.location().hash()?;
        if hash.starts_with("#!") && hash.len() > 2 {
            // Restored by `start`.
        } else if let Some(url) = params.get("url").or_else(|| params.get("nifti")) {
            // A path relative to the page becomes an absolute URL; one with a
            // scheme of its own (`precomputed://gs://…`, a pipeline) is kept as
            // it is (the URL parser would take `precomputed:` for the scheme
            // and mangle the rest).
            let absolute = if url.contains("://") {
                url.clone()
            } else {
                web_sys::Url::new_with_base(&url, &window.location().href()?).map(|u| u.href()).unwrap_or(url.clone())
            };
            match neuroglancer::datasource::url::get(&absolute).await {
                Ok(source) => {
                    with_engine(|engine| {
                        if let Err(e) = engine.viewer_mut().add_layer(source) {
                            log::error!("could not show {url}: {e:#}");
                        }
                    });
                }
                Err(e) => log::error!("could not open {url}: {e:#}"),
            }
        } else {
            log::info!("open with ?url=<data source> or #!<viewer state> to load data");
        }
        Ok(())
    }

    /// `viewer.isReady()`: every layer's sources loaded (or failed) and
    /// nothing downloading, for screenshotting when the data is in.
    #[wasm_bindgen(js_name = isReady)]
    pub fn is_ready() -> bool {
        with_engine(|e| e.is_ready()).unwrap_or(false)
    }

    /// Layer `layer`'s data histogram (the invlerp's CDF counts), as JSON:
    /// for comparing with neuroglancer's.
    #[wasm_bindgen(js_name = dataHistogram)]
    pub fn data_histogram(layer: usize) -> String {
        with_engine(|e| match e.viewer().layers().get(layer).and_then(|l| l.data_histogram()) {
            Some(h) => serde_json::to_string(&h.to_vec()).unwrap_or_default(),
            None => "null".into(),
        })
        .unwrap_or_else(|| "null".into())
    }

    /// The UI's text runs and their line boxes, as JSON
    /// (`[[text, x, y, width, height], …]`), for comparing with
    /// neuroglancer's DOM.
    #[wasm_bindgen(js_name = uiTextRects)]
    pub fn ui_text_rects() -> String {
        with_engine(|e| {
            let Some(ui) = e.ui() else { return "[]".into() };
            let rows: Vec<String> = ui
                .text_rects()
                .into_iter()
                .map(|(t, r)| format!("[{},{},{},{},{}]", serde_json::Value::from(t), r.x, r.y, r.width, r.height))
                .collect();
            format!("[{}]", rows.join(","))
        })
        .unwrap_or_else(|| "[]".into())
    }
}
