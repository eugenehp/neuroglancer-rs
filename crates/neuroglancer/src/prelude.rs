//! What embedding the engine takes, in one import:
//!
//! ```
//! use neuroglancer::prelude::*;
//! ```
//!
//! The engine and what it exchanges with its host ([`Engine`],
//! [`EngineOptions`], [`Frame`], [`Request`], [`Response`], [`UiCanvas`] and
//! the [`Painter`] one paints with), input ([`Button`], [`Modifiers`],
//! [`Cursor`], [`ColorSetting`]), where background work runs
//! ([`ThreadPool`] natively, `LocalExecutor` in a browser), the viewer
//! underneath ([`Viewer`]), and with their features the ready-made hosts'
//! entry points.

pub use crate::engine::{Engine, EngineOptions, Frame, Request, Response, UiCanvas};
pub use crate::fetch_state;
pub use ng_chunk::Executor;
#[cfg(not(target_arch = "wasm32"))]
pub use ng_chunk::ThreadPool;
#[cfg(target_arch = "wasm32")]
pub use ng_chunk::LocalExecutor;
pub use ng_ui::{ColorSetting, Cursor};
pub use ng_viewer::input::Modifiers;
pub use ng_viewer::{Button, Viewer};
pub use threers::ui::{Painter, TextMeasure};

#[cfg(all(feature = "winit", not(target_arch = "wasm32")))]
pub use crate::winit::{run as run_window, Open, WindowOptions};

#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub use crate::web::{start as start_web, with_engine, DomText, WebOptions};
