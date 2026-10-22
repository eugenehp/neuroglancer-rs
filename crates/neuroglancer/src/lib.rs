//! neuroglancer as a library: neuroglancer-rs, © 2026 Eugene Hauptmann, MIT
//! OR Apache-2.0 (<https://github.com/eugenehp/neuroglancer-rs>).
//!
//! [`Engine`] is the viewer (data sources, chunk loading, rendering,
//! navigation, neuroglancer's state) and its UI (layer bar, side panels,
//! dialogs), driven through one platform-neutral interface: input in DOM
//! terms, frames rendered into any `wgpu` texture view on the caller's
//! device, and [`Request`]s for what only the host can do (clipboard, files,
//! links). Embed it in your own window, canvas or offscreen renderer:
//!
//! ```no_run
//! # async fn demo(device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>, view: wgpu::TextureView) {
//! use std::sync::Arc;
//! use neuroglancer::prelude::*;
//!
//! let mut engine = Engine::new(EngineOptions {
//!     device,
//!     queue,
//!     format: wgpu::TextureFormat::Rgba8Unorm,
//!     width: 800,
//!     height: 600,
//!     scale_factor: 1.0,
//!     executor: Arc::new(ThreadPool::new(4)),
//!     wake: Arc::new(|| {}),
//!     ui: false,
//!     now: || 0.0,
//! });
//! engine.restore_url("https://neuroglancer-demo.appspot.com/#!gs://neuroglancer-public-data/flyem_fib-25/state.json");
//! while !engine.is_ready() {
//!     engine.draw_frame(&view, None, String::new);
//! }
//! # }
//! ```
//!
//! `use neuroglancer::prelude::*;` brings in what embedding takes.
//!
//! Ready-made hosts: [`winit`] (feature `winit`, a desktop window) and
//! [`web`] (feature `web`, wasm32: a page's canvas). The layers underneath
//! are re-exported for finer control: [`viewer`] (the viewer alone, no UI),
//! [`render`] (the GPU passes), [`datasource`], [`kvstore`], [`chunk`],
//! [`core`] (coordinate spaces, navigation, state), [`ui`].

mod engine;
pub mod prelude;

pub use engine::{Engine, EngineOptions, Frame, Request, Response, UiCanvas};

pub use ng_chunk as chunk;
pub use ng_core as core;
pub use ng_datasource as datasource;
pub use ng_kvstore as kvstore;
pub use ng_render as render;
pub use ng_ui as ui;
pub use ng_viewer as viewer;

pub use ng_ui::{ColorSetting, Cursor};
pub use ng_viewer::input::Modifiers;
pub use ng_viewer::{Button, Viewer};

/// The GPU and UI engine underneath.
pub use threers;

/// Fetches a viewer state: a share link (`…#!{…}`, `#!gs://…/state.json`),
/// its JSON, or (natively) a file holding it.
pub use ng_viewer::state::fetch_state;

#[cfg(all(feature = "winit", not(target_arch = "wasm32")))]
pub mod winit;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub mod web;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub mod decode_pool;
