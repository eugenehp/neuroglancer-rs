//! [`Engine`]: the viewer and its UI behind one platform-neutral interface.
//! A host (a window, a canvas, a test) owns the GPU surface and the event
//! source; it hands the engine input in DOM terms, asks it for frames, and
//! carries out the few [`Request`]s only it can (the clipboard, files, links).

use std::sync::Arc;

use ng_chunk::Executor;
use ng_ui::{Action, ColorSetting, Cursor, Effect, NgUi};
use ng_viewer::input::Modifiers;
use ng_viewer::screenshot::Screenshot;
use ng_viewer::host::Loader;
use ng_viewer::{Button, PendingSource, Viewer};
use serde_json::Value;
use threers::ui::Painter;

/// What the engine needs the host to do: things outside the viewer.
#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    /// Put this text on the clipboard.
    Clipboard(String),
    /// Put a link to the current state on the clipboard (the page's URL in a
    /// browser; [`Engine::state_fragment`] after a viewer URL elsewhere).
    CopyUrl,
    /// Open this link (`<a target="_blank">`).
    OpenUrl(String),
    /// Let the user pick a colour for `setting`, starting at `current`; hand
    /// each choice to [`Engine::set_color`] (or call
    /// [`Engine::open_color_picker`] for the UI's own picker).
    PickColor { setting: ColorSetting, current: [f32; 3] },
    /// Save `bytes` (of type `mime`) as a file called `name`.
    Download { name: String, bytes: Vec<u8>, mime: String },
}

/// What an input did.
#[derive(Debug, Default)]
pub struct Response {
    /// Something changed: draw a frame.
    pub redraw: bool,
    /// The input was used (a browser host prevents its default action).
    pub handled: bool,
    /// For the host to carry out.
    pub requests: Vec<Request>,
}

impl Response {
    fn changed(changed: bool) -> Self {
        Self { redraw: changed, handled: changed, requests: vec![] }
    }
}

/// What a frame left to do.
#[derive(Debug, Default)]
pub struct Frame {
    /// A screenshot finished in this frame: for the host to save.
    pub screenshot: Option<Screenshot>,
    /// Draw another frame straight away (playback, a screenshot under way,
    /// a legend being drawn).
    pub animate: bool,
    /// Draw again in this many milliseconds (a caret's blink).
    pub redraw_in: Option<f64>,
}

/// Where the UI paints: a 2D canvas over the GPU's, a CPU overlay blended
/// over the frame, … The engine lays the UI out with the same painter (it
/// measures text) before and after rendering, then paints it between
/// [`begin`](Self::begin) and [`finish`](Self::finish).
pub trait UiCanvas {
    /// A new frame's painting starts (clear, resize).
    fn begin(&mut self) {}
    /// Calls `f` with the painter.
    fn with_painter(&mut self, f: &mut dyn FnMut(&mut dyn Painter));
    /// The frame's painting is done.
    fn finish(&mut self) {}
}

/// neuroglancer: the [`Viewer`] (data, rendering, navigation) and, unless
/// it is left out, its UI ([`NgUi`]: the layer bar, side panels, dialogs).
pub struct Engine {
    viewer: Viewer,
    ui: Option<NgUi>,
    /// States and data sources loading in the background.
    loader: Loader,
    /// A state to restore and sources to load, at the next frame.
    restore: Option<Value>,
    load: Vec<PendingSource>,
    /// The drawing area, in device pixels and in CSS pixels.
    size: (u32, u32),
    css: (f32, f32),
    now: fn() -> f64,
}

/// What [`Engine::new`] needs: the GPU, the target's format and size, and
/// where background work runs.
pub struct EngineOptions {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    /// The format of the textures frames are rendered to.
    pub format: wgpu::TextureFormat,
    /// The drawing area in device pixels, and device pixels per CSS pixel.
    pub width: u32,
    pub height: u32,
    pub scale_factor: f64,
    /// Runs downloads and decoding: a thread pool natively
    /// ([`ng_chunk::ThreadPool`]), the page's event loop in a browser
    /// ([`ng_chunk::LocalExecutor`]).
    pub executor: Arc<dyn Executor>,
    /// Called (from any thread) when background work finished: the host
    /// calls [`Engine::poll`] (or draws a frame, which polls).
    pub wake: Arc<dyn Fn() + Send + Sync>,
    /// With neuroglancer's UI, or the panels alone.
    pub ui: bool,
    /// Milliseconds on a monotonic clock (download latencies, playback, the
    /// caret's blink).
    pub now: fn() -> f64,
}

impl Engine {
    pub fn new(options: EngineOptions) -> Self {
        let EngineOptions { device, queue, format, width, height, scale_factor, executor, wake, ui, now } = options;
        let mut viewer = Viewer::new(device, queue, format, width, height, scale_factor, executor.clone(), wake.clone());
        viewer.set_clock(now);
        let css = (width as f32 / scale_factor as f32, height as f32 / scale_factor as f32);
        Self {
            viewer,
            ui: ui.then(NgUi::new),
            loader: Loader::new(executor, wake),
            restore: None,
            load: vec![],
            size: (width, height),
            css,
            now,
        }
    }

    pub fn viewer(&self) -> &Viewer {
        &self.viewer
    }

    pub fn viewer_mut(&mut self) -> &mut Viewer {
        &mut self.viewer
    }

    /// The UI, unless the engine was made without it.
    pub fn ui(&self) -> Option<&NgUi> {
        self.ui.as_ref()
    }

    pub fn ui_mut(&mut self) -> Option<&mut NgUi> {
        self.ui.as_mut()
    }

    /// The viewer and the UI together (a UI call that reads the viewer).
    pub fn parts(&mut self) -> (&mut Viewer, Option<&mut NgUi>) {
        (&mut self.viewer, self.ui.as_mut())
    }

    /// The drawing area changed: `width` × `height` device pixels,
    /// `scale_factor` of them a CSS pixel.
    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f64) {
        self.size = (width, height);
        self.css = ((width as f64 / scale_factor) as f32, (height as f64 / scale_factor) as f32);
        self.viewer.resize(width, height, scale_factor);
    }

    /// The drawing area's size in CSS pixels, where it is not exactly the
    /// device pixels over the scale factor (a canvas's `clientWidth`, whose
    /// device size was rounded): the UI's layout size, and how pointer
    /// positions map to device pixels.
    pub fn set_css_size(&mut self, width: f32, height: f32) {
        self.css = (width, height);
    }

    /// The panels' input units (device pixels) for a point in CSS pixels.
    fn device(&self, x: f32, y: f32) -> (f64, f64) {
        let sx = self.size.0 as f64 / (self.css.0 as f64).max(1.0);
        let sy = self.size.1 as f64 / (self.css.1 as f64).max(1.0);
        (x as f64 * sx, y as f64 * sy)
    }

    // ---- State and data ----

    /// Restores a viewer state now (`viewer.state.reset(); restoreState`);
    /// its data sources load in the background, each showing as it arrives.
    pub fn restore_state(&mut self, state: &Value) -> anyhow::Result<()> {
        self.loader.restore_state(&mut self.viewer, state)
    }

    /// Fetches a state (a share link `…#!{…}`, `#!gs://…/state.json`, JSON)
    /// in the background, then restores it.
    pub fn restore_url(&mut self, arg: &str) {
        self.loader.restore_url(arg);
    }

    /// Loads data sources in the background; [`poll`](Self::poll) hands
    /// each to its layer.
    pub fn load_sources(&mut self, pending: Vec<PendingSource>) {
        self.loader.load_sources(pending);
    }

    /// Takes in what background work finished (sources, states, chunks);
    /// whether a frame would show something new.
    pub fn poll(&mut self) -> bool {
        self.loader.poll(&mut self.viewer)
    }

    /// Every layer's sources loaded (or failed) and nothing downloading
    /// (`viewer.isReady()`).
    pub fn is_ready(&self) -> bool {
        Loader::is_ready(&self.viewer)
    }

    /// The state as a URL fragment (`#!{…}`).
    pub fn state_fragment(&mut self) -> String {
        self.viewer.state_fragment()
    }

    /// One line on the position and the values under the pointer (a window
    /// title).
    pub fn status(&mut self) -> String {
        self.viewer.status()
    }

    // ---- Frames ----

    /// Draws a frame into `target`: what arrived since the last one, the
    /// UI laid out (it decides where the panels go), the panels rendered,
    /// a screenshot's next step, then the UI laid out again with what the
    /// frame drew and painted on `ui`. `screenshot_time` names a finished
    /// screenshot's file.
    pub fn draw_frame(&mut self, target: &wgpu::TextureView, mut ui: Option<&mut dyn UiCanvas>, screenshot_time: impl FnOnce() -> String) -> Frame {
        if let Some(state) = self.restore.take() {
            if let Err(e) = self.restore_state(&state) {
                log::error!("could not restore the state: {e:#}");
            }
        }
        let pending = std::mem::take(&mut self.load);
        if !pending.is_empty() {
            self.load_sources(pending);
        }
        let now = (self.now)();
        // Playing dimensions advance, and keep drawing.
        let playing = self.viewer.step_playback(now);
        self.poll();
        let (w, h) = self.css;
        if let (Some(u), Some(canvas)) = (&mut self.ui, ui.as_deref_mut()) {
            u.set_time(now);
            canvas.with_painter(&mut |p| {
                u.update(&mut self.viewer, w, h, p);
            });
        }
        self.viewer.render(target);
        let mut frame = Frame::default();
        // A screenshot under way: its scaled render, then its file.
        if self.viewer.screenshot_active() {
            if let Some(shot) = self.viewer.step_screenshot(&screenshot_time()) {
                if let Some(u) = &mut self.ui {
                    u.screenshot_finished();
                }
                frame.screenshot = Some(shot);
            }
        }
        if let (Some(u), Some(canvas)) = (&mut self.ui, ui) {
            // Again, with what this frame drew (chunk progress, hover values).
            canvas.with_painter(&mut |p| {
                u.update(&mut self.viewer, w, h, p);
            });
            canvas.begin();
            canvas.with_painter(&mut |p| u.paint(p));
            canvas.finish();
            frame.redraw_in = u.next_blink();
        }
        frame.animate = self.viewer.screenshot_active() || playing || self.viewer.legend_pending();
        frame
    }

    // ---- Input (CSS pixels, DOM semantics) ----

    /// Carries out a UI action; what it asks of the host.
    fn run(&mut self, action: &Action, requests: &mut Vec<Request>) {
        let Some(ui) = &mut self.ui else { return };
        if let Action::Unported(title) = action {
            log::info!("{title}: not ported yet");
        }
        match ui.apply(&mut self.viewer, action) {
            Effect::None => {}
            // At the next frame, as neuroglancer applies them.
            Effect::RestoreState(state) => self.restore = Some(state),
            Effect::LoadSources(pending) => self.load.extend(pending),
            Effect::Clipboard(text) => requests.push(Request::Clipboard(text)),
            Effect::CopyUrl => requests.push(Request::CopyUrl),
            Effect::OpenUrl(url) => requests.push(Request::OpenUrl(url)),
            Effect::PickColor { setting, current } => requests.push(Request::PickColor { setting, current }),
            Effect::Download { name, bytes, mime } => requests.push(Request::Download { name, bytes, mime }),
        }
    }

    /// Runs `action` if there is one: a [`Response`] that redraws.
    fn act(&mut self, action: Option<Action>) -> Option<Response> {
        let action = action?;
        let mut requests = vec![];
        self.run(&action, &mut requests);
        Some(Response { redraw: true, handled: true, requests })
    }

    /// A colour the host's picker chose ([`Request::PickColor`]).
    pub fn set_color(&mut self, setting: ColorSetting, color: [f32; 3]) {
        let mut requests = vec![];
        self.run(&Action::SetColor(setting, color), &mut requests);
    }

    /// The UI's own colour picker, for a host without one.
    pub fn open_color_picker(&mut self, setting: ColorSetting, current: [f32; 3]) {
        if let Some(u) = &mut self.ui {
            u.open_color_picker(setting, current);
        }
    }

    /// The pointer's shape at `(x, y)`.
    pub fn cursor(&self, x: f32, y: f32) -> Cursor {
        self.ui.as_ref().map_or(Cursor::Default, |u| u.cursor(x, y))
    }

    pub fn pointer_down(&mut self, x: f32, y: f32, button: Button, modifiers: Modifiers) -> Response {
        let mut response = Response::default();
        if let Some(u) = &mut self.ui {
            u.set_time((self.now)());
            // The UI first. A press on the panels still reaches them (and
            // may also end an edit in the UI).
            let captured = u.captures(x, y);
            let action = u.pointer_down(x, y, button, modifiers, &self.viewer);
            if let Some(r) = self.act(action) {
                response = r;
            }
            if captured {
                return response;
            }
        }
        let (dx, dy) = self.device(x, y);
        self.viewer.pointer_move(dx, dy);
        // Presses on a panel the UI acts on (selecting a position).
        let action = self.ui.as_mut().and_then(|u| u.data_panel_press(button, modifiers, &self.viewer));
        if let Some(r) = self.act(action) {
            response.requests.extend(r.requests);
            response.redraw = true;
            response.handled = true;
            return response;
        }
        let changed = self.viewer.pointer_down(button, modifiers);
        response.redraw |= changed;
        response.handled |= changed;
        response
    }

    pub fn pointer_up(&mut self, button: Button) -> Response {
        // A layer item's click (when its press did not drag), or the drag's end.
        let action = self.ui.as_mut().and_then(|u| u.pointer_up_action());
        let mut response = self.act(action).unwrap_or_default();
        let ui = self.ui.as_mut().is_some_and(|u| u.pointer_up());
        response.redraw |= ui | self.viewer.pointer_up(button);
        response
    }

    pub fn pointer_move(&mut self, x: f32, y: f32) -> Response {
        let mut response = Response::default();
        if let Some(u) = &mut self.ui {
            u.set_time((self.now)());
            let action = u.pointer_move(x, y);
            if let Some(r) = self.act(action) {
                response = r;
            }
            // Over the UI (a panel's controls, say) the panels see the
            // pointer leave, unless dragging.
            if self.ui.as_ref().is_some_and(|u| u.covers(x, y)) && !self.viewer.is_dragging() {
                response.redraw |= self.viewer.pointer_leave();
                return response;
            }
        }
        let (dx, dy) = self.device(x, y);
        response.redraw |= self.viewer.pointer_move(dx, dy);
        response
    }

    pub fn pointer_leave(&mut self) -> Response {
        let hover = self.ui.as_mut().is_some_and(|u| u.pointer_leave());
        Response::changed(self.viewer.pointer_leave() | hover)
    }

    pub fn double_click(&mut self, x: f32, y: f32, modifiers: Modifiers) -> Response {
        let Some(u) = &mut self.ui else { return Response::default() };
        let action = u.double_click(x, y);
        if let Some(r) = self.act(action) {
            return r;
        }
        // On a data panel: `select`/`star` the segments under it.
        let u = self.ui.as_mut().unwrap();
        if !u.captures(x, y) {
            let action = u.data_panel_double_click(modifiers, &self.viewer);
            if let Some(r) = self.act(action) {
                return r;
            }
        }
        Response::default()
    }

    /// A wheel turned at `(x, y)` (DOM `WheelEvent`: `delta_y` positive
    /// toward the user, `delta_mode` 0 pixels, 1 lines, 2 pages).
    pub fn wheel(&mut self, x: f32, y: f32, delta_x: f64, delta_y: f64, delta_mode: u32, modifiers: Modifiers) -> Response {
        let action = self.ui.as_mut().and_then(|u| u.wheel(x, y, delta_y, modifiers));
        if let Some(r) = self.act(action) {
            return r;
        }
        let changed = self.viewer.wheel(delta_x, delta_y, delta_mode, modifiers);
        Response { redraw: changed, handled: true, requests: vec![] }
    }

    /// A key pressed: `key` and `code` as DOM `KeyboardEvent`s name them.
    pub fn key_down(&mut self, key: &str, code: &str, modifiers: Modifiers) -> Response {
        // A focused field in the UI takes the keyboard.
        if let Some(u) = &mut self.ui {
            u.set_time((self.now)());
            if u.has_keyboard_focus() {
                let action = u.key_down(key, modifiers, &self.viewer);
                if let Some(r) = self.act(action) {
                    return r;
                }
            }
        }
        if self.viewer.key_down(code, modifiers) {
            return Response::changed(true);
        }
        // Then the viewer's global bindings (`getDefaultGlobalBindings`).
        let action = self.ui.as_mut().and_then(|u| u.shortcut(code, modifiers, &self.viewer));
        self.act(action).unwrap_or_default()
    }

    /// A key released (a momentary tool ends with its key).
    pub fn key_up(&mut self, code: &str) -> Response {
        Response::changed(self.ui.as_mut().is_some_and(|u| u.key_up(code)))
    }

    /// The window lost the focus (momentary tools end).
    pub fn blur(&mut self) -> Response {
        Response::changed(self.ui.as_mut().is_some_and(|u| u.blur_window()))
    }
}
