//! A page's canvas around the [`Engine`] (wasm32): WebGPU, or WebGL2 where
//! there is none; DOM input; the UI on a 2D canvas over the GPU's (text as
//! Chrome draws a page's); the state kept in the URL hash as neuroglancer's
//! `UrlHashBinding` keeps it.
//!
//! [`start`] sets it up; [`with_engine`] reaches the engine afterwards
//! (to open data, read state, add bindings of your own).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use threers::ui::canvas::CanvasPainter;
use threers::ui::{Painter, TextStyle};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent, PointerEvent, WheelEvent};

use crate::{Button, Cursor, Engine, EngineOptions, Modifiers, Request, Response, UiCanvas};

/// Which UI text goes on the page as DOM, over the canvas, rather than on
/// the canvas: Chrome rasterizes some text differently on a canvas than in
/// a page (translucent colours, for one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DomText {
    /// Translucent text as DOM, the rest on the canvas (which matches the
    /// page there): text as neuroglancer's.
    #[default]
    Translucent,
    /// All on the canvas.
    None,
    /// All as DOM.
    All,
}

/// How [`start`] sets the page up.
#[derive(Default)]
pub struct WebOptions {
    /// The canvas to draw on; by default a new one filling the window.
    pub canvas: Option<HtmlCanvasElement>,
    /// Without neuroglancer's UI (as its `showUIControls` off: the panels
    /// fill the canvas).
    pub no_ui: bool,
    pub dom_text: DomText,
    /// Keep the state in the URL hash (`#!{…}`) and restore it from there,
    /// as neuroglancer's page does.
    pub url_hash: bool,
    /// Decode chunks in Web Workers running this module worker script (one
    /// a core, up to 8): see [`crate::decode_pool`]. Without it, chunks
    /// decode on the page.
    pub decode_workers: Option<String>,
    /// Keep what is downloaded in this Cache Storage cache, and load from
    /// there what was seen before (see [`ng_kvstore::set_browser_cache`]).
    pub cache: Option<String>,
    /// Without the port's credits (the footer and the Settings panel's
    /// About), as neuroglancer's page shows none.
    pub no_attribution: bool,
}

/// neuroglancer's UI, painted on a 2D canvas over the GPU canvas.
struct UiLayer {
    painter: CanvasPainter,
    canvas: HtmlCanvasElement,
    /// The GPU canvas, whose size the UI's takes.
    under: HtmlCanvasElement,
}

impl UiCanvas for UiLayer {
    fn begin(&mut self) {
        let (w, h) = (self.under.width(), self.under.height());
        if self.canvas.width() != w || self.canvas.height() != h {
            self.canvas.set_width(w);
            self.canvas.set_height(h);
        }
        self.painter.begin(w, h, dpr());
    }

    fn with_painter(&mut self, f: &mut dyn FnMut(&mut dyn Painter)) {
        f(&mut self.painter);
    }

    fn finish(&mut self) {
        self.painter.finish();
    }
}

struct App {
    engine: Engine,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// Kept alive: on WebGL2 the context belongs to the instance.
    _gpu: threers::gpu::GpuContext,
    canvas: HtmlCanvasElement,
    /// A frame is already requested; more changes before it runs share it,
    /// as with neuroglancer's `animationFrameDebounce`.
    frame_pending: bool,
    /// The URL hash as last written or read (`UrlHashBinding`'s
    /// `prevStateString`), when the state is kept there.
    hash: Option<String>,
    ui: Option<UiLayer>,
    /// A repaint for the caret's blink is scheduled.
    blink_pending: bool,
}

type Shared = Rc<RefCell<App>>;

thread_local! {
    /// Schedules a frame; set once the app exists. Background work calls it.
    static WAKE: RefCell<Option<Box<dyn Fn()>>> = const { RefCell::new(None) };
    static APP: RefCell<Option<Shared>> = const { RefCell::new(None) };
}

/// Runs `f` on the engine, once [`start`] has made it (`None` before);
/// a frame follows.
pub fn with_engine<R>(f: impl FnOnce(&mut Engine) -> R) -> Option<R> {
    let app = APP.with(|a| a.borrow().clone())?;
    let result = f(&mut app.borrow_mut().engine);
    schedule_redraw(&app);
    Some(result)
}

/// Sets up the page: the GPU canvas (and the UI's over it), the engine, its
/// input, its frames. With [`WebOptions::url_hash`], a state in the URL hash
/// is restored.
pub async fn start(options: WebOptions) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    ng_kvstore::set_browser_cache(options.cache.as_deref());
    if let Some(script) = &options.decode_workers {
        let cores = window.navigator().hardware_concurrency() as usize;
        if let Err(e) = crate::decode_pool::start(script, cores.clamp(1, 8)) {
            log::error!("no decode workers ({e:?}): decoding on the page");
        }
    }
    let full_window = options.canvas.is_none();
    let canvas = match options.canvas {
        Some(c) => c,
        None => {
            let canvas: HtmlCanvasElement = document.create_element("canvas")?.dyn_into()?;
            canvas.style().set_css_text("display:block;width:100vw;height:100vh;touch-action:none;outline:0");
            document.body().ok_or("no body")?.append_child(&canvas)?;
            canvas
        }
    };
    canvas.set_tab_index(0);
    let (width, height) = canvas_pixel_size(&canvas);
    canvas.set_width(width);
    canvas.set_height(height);

    // WebGPU, or WebGL2 where there is none; then the device with the
    // limits threers' renderer needs.
    let (instance, surface, adapter) = threers::wasm::open_browser_gpu(&canvas).await?;
    // Float32 blending for the 3D view's transparent pass, where the
    // browser has it (`float32-blendable`).
    let gpu = threers::gpu::GpuContext::open_with_optional_features(instance, adapter, wgpu::Features::FLOAT32_BLENDABLE).await?;
    // WebGL2: no compute shaders, colour targets blended alike.
    ng_render::capabilities::set_from_adapter(&gpu.adapter);
    log::info!("adapter: {}", gpu.describe());
    // Not sRGB: browser canvases are plain 8-bit; threers and the slice
    // pass both adjust to the target.
    let config = gpu.surface_config(&surface, width, height, false);
    surface.configure(&gpu.device, &config);
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());

    let mut engine = Engine::new(EngineOptions {
        device: device.clone(),
        queue: queue.clone(),
        format: config.format,
        width,
        height,
        scale_factor: dpr(),
        executor: Arc::new(ng_chunk::LocalExecutor),
        wake: Arc::new(|| {
            WAKE.with(|w| {
                if let Some(wake) = &*w.borrow() {
                    wake();
                }
            })
        }),
        ui: !options.no_ui,
        now,
    });
    engine.set_css_size(canvas.client_width() as f32, canvas.client_height() as f32);
    engine.viewer_mut().mac = window.navigator().platform().map(|p| p.starts_with("Mac")).unwrap_or(false);
    if let Some(ui) = engine.ui_mut() {
        ui.attribution = !options.no_attribution;
    }
    engine.viewer_mut().set_scale_bar_painter(Box::new(CanvasScaleBarPainter));

    let hash = window.location().hash()?;
    if options.url_hash && hash.starts_with("#!") && hash.len() > 2 {
        match crate::fetch_state(&hash).await.and_then(|state| engine.restore_state(&state)) {
            Ok(()) => {}
            Err(e) => log::error!("could not restore the state: {e:#}"),
        }
    }
    document.set_title(&engine.status());

    let ui = if options.no_ui {
        None
    } else {
        let overlay: HtmlCanvasElement = document.create_element("canvas")?.dyn_into()?;
        if full_window {
            overlay.style().set_css_text("position:fixed;left:0;top:0;width:100vw;height:100vh;pointer-events:none");
            document.body().ok_or("no body")?.append_child(&overlay)?;
        } else {
            // Over the canvas, in its parent.
            overlay.style().set_css_text(&format!(
                "position:absolute;left:{}px;top:{}px;width:{}px;height:{}px;pointer-events:none",
                canvas.offset_left(),
                canvas.offset_top(),
                canvas.client_width(),
                canvas.client_height()
            ));
            canvas.after_with_node_1(&overlay)?;
        }
        let ctx: web_sys::CanvasRenderingContext2d = overlay.get_context("2d")?.ok_or("no 2d context")?.dyn_into()?;
        let painter = match options.dom_text {
            DomText::None => CanvasPainter::new(ctx, dpr()),
            DomText::All => CanvasPainter::with_dom_text(ctx, dpr()),
            DomText::Translucent => CanvasPainter::with_dom_text_for(ctx, dpr(), |s: &TextStyle| s.color[3] < 1.0),
        };
        Some(UiLayer { painter, canvas: overlay, under: canvas.clone() })
    };

    let app: Shared = Rc::new(RefCell::new(App {
        engine,
        surface,
        config,
        device,
        queue,
        _gpu: gpu,
        canvas: canvas.clone(),
        frame_pending: false,
        hash: options.url_hash.then(|| hash.clone()),
        ui,
        blink_pending: false,
    }));

    listen(&canvas, "pointerdown", &app, |app, e: PointerEvent| {
        let _ = app.canvas.set_pointer_capture(e.pointer_id());
        let _ = app.canvas.focus();
        let (x, y) = (e.offset_x() as f32, e.offset_y() as f32);
        match button(e.button()) {
            Some(b) => app.engine.pointer_down(x, y, b, modifiers(&e)),
            None => app.engine.pointer_move(x, y),
        }
    });
    listen(&canvas, "pointerup", &app, |app, e: PointerEvent| app.engine.pointer_up(button(e.button()).unwrap_or(Button::Primary)));
    listen(&canvas, "pointermove", &app, |app, e: PointerEvent| {
        let (x, y) = (e.offset_x() as f32, e.offset_y() as f32);
        let response = app.engine.pointer_move(x, y);
        if app.ui.is_some() {
            let cursor = match app.engine.cursor(x, y) {
                Cursor::Default => "default",
                Cursor::Pointer => "pointer",
                Cursor::Crosshair => "crosshair",
                Cursor::Text => "text",
            };
            let _ = app.canvas.style().set_property("cursor", cursor);
        }
        response
    });
    listen(&canvas, "pointerleave", &app, |app, _e: PointerEvent| app.engine.pointer_leave());
    listen(&canvas, "dblclick", &app, |app, e: MouseEvent| app.engine.double_click(e.offset_x() as f32, e.offset_y() as f32, modifiers(&e)));
    listen(&canvas, "wheel", &app, |app, e: WheelEvent| {
        e.prevent_default();
        app.engine.wheel(e.offset_x() as f32, e.offset_y() as f32, e.delta_x(), e.delta_y(), e.delta_mode(), modifiers(&e))
    });
    // On the window: keys reach the viewer's bindings wherever the focus
    // is (neuroglancer binds them on the document).
    listen(&window, "keydown", &app, |app, e: KeyboardEvent| {
        let m = Modifiers { control: e.ctrl_key(), alt: e.alt_key(), meta: e.meta_key(), shift: e.shift_key() };
        let response = app.engine.key_down(&e.key(), &e.code(), m);
        if response.handled {
            e.prevent_default();
        }
        response
    });
    // A momentary tool ends with its key (`keyup`), or when the window
    // loses focus (`blur`).
    listen(&window, "keyup", &app, |app, e: KeyboardEvent| app.engine.key_up(&e.code()));
    listen(&window, "blur", &app, |app, _e: web_sys::Event| app.engine.blur());
    // Right-click moves to the pointer; keep the browser's menu out of it.
    listen(&canvas, "contextmenu", &app, |_app, e: web_sys::Event| {
        e.prevent_default();
        Response::default()
    });
    listen(&window, "resize", &app, |app, _e: web_sys::Event| {
        let (w, h) = canvas_pixel_size(&app.canvas);
        app.canvas.set_width(w);
        app.canvas.set_height(h);
        app.config.width = w;
        app.config.height = h;
        app.surface.configure(&app.device, &app.config);
        app.engine.resize(w, h, dpr());
        let (cw, ch) = (app.canvas.client_width() as f32, app.canvas.client_height() as f32);
        app.engine.set_css_size(cw, ch);
        Response { redraw: true, ..Default::default() }
    });

    APP.with(|a| *a.borrow_mut() = Some(app.clone()));
    {
        let app = app.clone();
        WAKE.with(|w| *w.borrow_mut() = Some(Box::new(move || schedule_redraw(&app))));
    }
    // Editing the URL hash restores that state (`updateFromUrlHash`).
    if options.url_hash {
        listen(&window, "hashchange", &app, |app, _e: web_sys::HashChangeEvent| {
            let hash = web_sys::window().and_then(|w| w.location().hash().ok()).unwrap_or_default();
            if app.hash.as_ref() != Some(&hash) {
                app.hash = Some(hash.clone());
                app.engine.restore_url(&hash);
            }
            Response { redraw: true, ..Default::default() }
        });
    }
    schedule_redraw(&app);
    Ok(())
}

/// Does what the engine asks of the page.
fn carry_out(request: Request) {
    match request {
        Request::Clipboard(text) => set_clipboard(&text),
        Request::OpenUrl(url) => {
            if let Some(w) = web_sys::window() {
                let _ = w.open_with_url_and_target(&url, "_blank");
            }
        }
        Request::CopyUrl => {
            if let Some(href) = web_sys::window().and_then(|w| w.location().href().ok()) {
                set_clipboard(&href);
            }
        }
        Request::PickColor { setting, current } => pick_color(setting, current),
        Request::Download { name, bytes, mime } => download(&name, &bytes, &mime),
    }
}

/// Writes `text` to the clipboard (`navigator.clipboard.writeText`).
pub fn set_clipboard(text: &str) {
    if let Some(w) = web_sys::window() {
        let promise = w.navigator().clipboard().write_text(text);
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) = wasm_bindgen_futures::JsFuture::from(promise).await {
                log::warn!("could not copy to the clipboard: {e:?}");
            }
        });
    }
}

/// Saves `bytes` as a file (`saveBlobToFile`: a Blob link clicked).
pub fn download(name: &str, bytes: &[u8], mime: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else { return };
    let parts = js_sys::Array::of1(&js_sys::Uint8Array::from(bytes));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type(mime);
    let Ok(blob) = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options) else { return };
    let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) else { return };
    let Ok(link) = document.create_element("a") else { return };
    let link: web_sys::HtmlAnchorElement = link.unchecked_into();
    link.set_href(&url);
    link.set_download(name);
    link.click();
    let _ = web_sys::Url::revoke_object_url(&url);
}

/// The browser's colour picker (an `<input type=color>` out of sight):
/// each colour chosen goes to the UI.
fn pick_color(setting: crate::ColorSetting, current: [f32; 3]) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else { return };
    let Ok(input) = document.create_element("input") else { return };
    let input: web_sys::HtmlInputElement = input.unchecked_into();
    input.set_type("color");
    input.set_value(&ng_viewer::state::serialize_color(current));
    input.style().set_css_text("position:fixed;left:-100px;top:0;opacity:0");
    if let Some(body) = document.body() {
        let _ = body.append_child(&input);
    }
    let target = input.clone();
    let on_input = Closure::<dyn FnMut()>::new(move || {
        let Some(c) = ng_viewer::state::parse_color(&target.value()) else { return };
        with_engine(|engine| engine.set_color(setting, c));
    });
    let _ = input.add_event_listener_with_callback("input", on_input.as_ref().unchecked_ref());
    on_input.forget();
    input.click();
}

fn modifiers(e: &MouseEvent) -> Modifiers {
    Modifiers { control: e.ctrl_key(), alt: e.alt_key(), meta: e.meta_key(), shift: e.shift_key() }
}

fn button(b: i16) -> Option<Button> {
    match b {
        0 => Some(Button::Primary),
        1 => Some(Button::Middle),
        2 => Some(Button::Secondary),
        _ => None,
    }
}

/// Adds a listener: what the engine asks of the page is carried out, and a
/// frame drawn on change.
fn listen<E: JsCast + 'static>(target: &web_sys::EventTarget, name: &str, app: &Shared, mut handler: impl FnMut(&mut App, E) -> Response + 'static) {
    let app = app.clone();
    let closure = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
        let response = handler(&mut app.borrow_mut(), e.unchecked_into());
        for request in response.requests {
            carry_out(request);
        }
        if response.redraw {
            schedule_redraw(&app);
        }
    });
    // `passive: false` so the wheel handler may `preventDefault` page zoom.
    let options = web_sys::AddEventListenerOptions::new();
    options.set_passive(false);
    target
        .add_event_listener_with_callback_and_add_event_listener_options(name, closure.as_ref().unchecked_ref(), &options)
        .expect("addEventListener");
    closure.forget();
}

fn schedule_redraw(app: &Shared) {
    // Asked while the app is in use (on WebGL2 a readback can finish, and
    // wake, within a frame): again once that is over.
    let Ok(mut a) = app.try_borrow_mut() else {
        let app = app.clone();
        wasm_bindgen_futures::spawn_local(async move { schedule_redraw(&app) });
        return;
    };
    if std::mem::replace(&mut a.frame_pending, true) {
        return;
    }
    drop(a);
    let app = app.clone();
    let closure = Closure::once_into_js(move || {
        let animate = {
            let mut a = app.borrow_mut();
            a.frame_pending = false;
            draw(&mut a)
        };
        // A screenshot renders until it is done; playback until paused.
        if animate {
            schedule_redraw(&app);
        }
    });
    web_sys::window().unwrap().request_animation_frame(closure.unchecked_ref()).expect("requestAnimationFrame");
}

/// Draws a frame; whether another should follow straight away.
fn draw(app: &mut App) -> bool {
    let mut animate = false;
    match app.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture) | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
            let view = texture.texture.create_view(&Default::default());
            let ui = app.ui.as_mut().map(|u| u as &mut dyn UiCanvas);
            // `new Date().toLocaleString().replace(", ", "-")`.
            let frame = app.engine.draw_frame(&view, ui, || String::from(js_sys::Date::new_0().to_locale_string("en-US", &JsValue::UNDEFINED)).replacen(", ", "-", 1));
            app.queue.present(texture);
            if let Some(shot) = frame.screenshot {
                download(&shot.name, &shot.png, "image/png");
            }
            animate = frame.animate;
            // A caret blinks: paint again when it does.
            if let Some(ms) = frame.redraw_in {
                if !std::mem::replace(&mut app.blink_pending, true) {
                    let closure = Closure::once_into_js(|| {
                        APP.with(|a| {
                            if let Some(app) = &*a.borrow() {
                                app.borrow_mut().blink_pending = false;
                                schedule_redraw(app);
                            }
                        })
                    });
                    let _ = web_sys::window().unwrap().set_timeout_with_callback_and_timeout_and_arguments_0(closure.unchecked_ref(), ms.ceil() as i32);
                }
            }
        }
        wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
            app.surface.configure(&app.device, &app.config);
        }
        e => log::error!("surface error: {e:?}"),
    }
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        doc.set_title(&app.engine.status());
    }
    // Keep the URL hash on the state (`UrlHashBinding.setUrlHash`), once
    // there is a state to show.
    if app.hash.is_some() && !app.engine.viewer().layers().is_empty() {
        let fragment = app.engine.state_fragment();
        if app.hash.as_ref() != Some(&fragment) {
            let window = web_sys::window().unwrap();
            let url = if fragment == "#!%7B%7D" { "#".to_owned() } else { fragment.clone() };
            if let Ok(history) = window.history() {
                let _ = history.replace_state_with_url(&JsValue::NULL, "", Some(&url));
            }
            app.hash = Some(fragment);
        }
    }
    animate
}

/// Scale bars painted as `makeScaleBarTexture` paints them, with the
/// page's own 2D canvas: the same text, the same pixels as neuroglancer.
struct CanvasScaleBarPainter;

impl ng_render::scale_bar::ScaleBarPainter for CanvasScaleBarPainter {
    fn paint(&self, bar: &ng_render::scale_bar::ScaleBar, o: &ng_render::scale_bar::ScaleBarOptions) -> ng_render::scale_bar::ScaleBarImage {
        let empty = ng_render::scale_bar::ScaleBarImage { width: 0, height: 0, rgba: vec![] };
        let Some(document) = web_sys::window().and_then(|w| w.document()) else { return empty };
        let Ok(canvas) = document.create_element("canvas") else { return empty };
        let canvas: HtmlCanvasElement = canvas.unchecked_into();
        let Ok(Some(ctx)) = canvas.get_context("2d") else { return empty };
        let ctx: web_sys::CanvasRenderingContext2d = ctx.unchecked_into();
        let text_height = o.text_height * o.scale_factor;
        let font = format!("bold {text_height}px {}", o.font_name);
        ctx.set_font(&font);
        ctx.set_fill_style_str("white");
        let text = bar.text();
        let text_width = ctx.measure_text(&text).map(|m| m.width()).unwrap_or(0.0);
        let inner_width = bar.length_in_pixels.max(text_width);
        let bar_height = o.bar_height * o.scale_factor;
        let bar_top_margin = o.bar_top_margin * o.scale_factor;
        let inner_height = bar_height + bar_top_margin + text_height;
        let padding = o.padding * o.scale_factor;
        let total_height = inner_height + 2.0 * padding;
        let total_width = inner_width + 2.0 * padding;
        // `canvas.width = totalWidth`: the setter truncates.
        canvas.set_width(total_width as u32);
        canvas.set_height(total_height as u32);
        ctx.set_font(&font);
        ctx.set_text_align("center");
        ctx.set_fill_style_str("rgba(0, 0, 0, 0.3)");
        ctx.fill_rect(0.0, 0.0, total_width, total_height);
        ctx.set_fill_style_str("white");
        let _ = ctx.fill_text(&text, total_width / 2.0, total_height - padding - bar_height - bar_top_margin);
        ctx.fill_rect(padding, total_height - padding - bar_height, bar.length_in_pixels, bar_height);
        let (w, h) = (canvas.width(), canvas.height());
        match ctx.get_image_data(0.0, 0.0, w as f64, h as f64) {
            // Not premultiplied, as WebGL's upload of a canvas gives it.
            Ok(data) => ng_render::scale_bar::ScaleBarImage { width: w, height: h, rgba: data.data().0 },
            Err(_) => empty,
        }
    }
}

/// Milliseconds since the page loaded (`performance.now()`).
pub fn now() -> f64 {
    web_sys::window().and_then(|w| w.performance()).map_or(0.0, |p| p.now())
}

fn dpr() -> f64 {
    web_sys::window().unwrap().device_pixel_ratio()
}

/// The canvas's CSS size in device pixels.
fn canvas_pixel_size(canvas: &HtmlCanvasElement) -> (u32, u32) {
    let w = (canvas.client_width() as f64 * dpr()).round().max(1.0) as u32;
    let h = (canvas.client_height() as f64 * dpr()).round().max(1.0) as u32;
    (w, h)
}
