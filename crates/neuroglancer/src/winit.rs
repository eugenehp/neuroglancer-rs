//! A desktop window around the [`Engine`]: winit and a wgpu surface (Metal
//! on Apple, Vulkan or DX12 elsewhere), the UI drawn into a CPU overlay and
//! blended over the panels.
//!
//! Input follows neuroglancer's default bindings: drag to pan, shift-drag to
//! rotate, wheel to step through slices, ctrl/⌘-wheel to zoom, right-click to
//! move to the pointer, arrows / `,` `.` to step, `z` to snap. The window
//! title shows the position and value under the pointer.

use std::sync::{Arc, Mutex};

use ng_chunk::ThreadPool;
use threers::gpu::GpuContext;
use threers::ui::Painter;
use winit::window::CursorIcon;
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoopBuilder,
    keyboard::{ModifiersState, PhysicalKey},
    window::WindowBuilder,
};

use crate::{Button, Cursor, Engine, EngineOptions, Modifiers, Request, UiCanvas};

/// What to show.
pub enum Open {
    /// A data source: anything neuroglancer accepts (`precomputed://gs://…`,
    /// `zarr://https://…`, `nifti://…`), or a local path or kvstore URL whose
    /// format is detected.
    Source(String),
    /// A viewer state: a share link (`…#!{…}`, `#!gs://…`), its JSON, or a
    /// file holding it.
    State(String),
}

/// The window's settings.
pub struct WindowOptions {
    pub title: String,
    /// The initial size, in logical pixels.
    pub width: u32,
    pub height: u32,
    /// neuroglancer's UI, or the panels alone.
    pub ui: bool,
}

impl Default for WindowOptions {
    fn default() -> Self {
        Self { title: "neuroglancer-rs".into(), width: 1100, height: 800, ui: true }
    }
}

/// The UI painted into an overlay the size of the window.
struct OverlayUi {
    overlay: threers::overlay::Overlay,
    scale: f32,
    size: (u32, u32),
}

impl UiCanvas for OverlayUi {
    fn begin(&mut self) {
        self.overlay.resize(self.size.0, self.size.1);
        self.overlay.clear();
    }

    fn with_painter(&mut self, f: &mut dyn FnMut(&mut dyn Painter)) {
        f(&mut threers::ui::OverlayPainter::with_scale(&mut self.overlay, self.scale));
    }
}

/// Milliseconds since the first call.
fn now() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// Opens a window showing `open` and runs until it is closed.
pub fn run(open: Option<Open>, options: WindowOptions) {
    pollster::block_on(run_async(open, options));
}

async fn run_async(open: Option<Open>, options: WindowOptions) {
    // A user event wakes the loop when a chunk finishes loading.
    let event_loop = EventLoopBuilder::<()>::with_user_event().build().expect("event loop");
    let proxy = Mutex::new(event_loop.create_proxy());
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(&options.title)
            .with_inner_size(winit::dpi::LogicalSize::new(options.width, options.height))
            .build(&event_loop)
            .expect("window"),
    );

    // Float32 blending for the 3D view's transparent pass, where the
    // adapter has it (`EXT_float_blend`).
    let (gpu, surface) = GpuContext::with_surface_and_optional_features(wgpu::SurfaceTarget::from(window.clone()), wgpu::Features::FLOAT32_BLENDABLE)
        .await
        .expect("GPU");
    log::info!("adapter: {}", gpu.describe());
    ng_render::capabilities::set_from_adapter(&gpu.adapter);
    let size = window.inner_size();
    // sRGB: threers encodes on the way out, and the slice pass adjusts to it.
    let mut config = gpu.surface_config(&surface, size.width, size.height, true);
    surface.configure(&gpu.device, &config);
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut engine = Engine::new(EngineOptions {
        device: device.clone(),
        queue: queue.clone(),
        format: config.format,
        width: config.width,
        height: config.height,
        scale_factor: window.scale_factor(),
        executor: Arc::new(ThreadPool::new(threads)),
        wake: Arc::new(move || {
            let _ = proxy.lock().unwrap().send_event(());
        }),
        ui: options.ui,
        now,
    });
    match &open {
        Some(Open::Source(url)) => {
            if let Err(e) = engine.viewer_mut().open(url) {
                eprintln!("could not open {url}: {e:#}");
                std::process::exit(1);
            }
        }
        Some(Open::State(arg)) => {
            let viewer = engine.viewer_mut();
            let state = crate::fetch_state(arg).await.and_then(|s| viewer.open_state(&s));
            if let Err(e) = state {
                eprintln!("could not restore the state: {e:#}");
                std::process::exit(1);
            }
            for m in &viewer.messages {
                eprintln!("{m}");
            }
            for l in viewer.layers() {
                for m in &l.messages {
                    eprintln!("{}: {m}", l.name());
                }
            }
        }
        None => {}
    }

    let window_for_loop = window.clone();
    let mut modifiers = Modifiers::default();
    let sf = window.scale_factor() as f32;
    let mut ui = OverlayUi { overlay: threers::overlay::Overlay::new(config.width, config.height), scale: sf, size: (config.width, config.height) };
    let mut image_overlay = ng_render::compose::ImageOverlay::new(&device, config.format);
    // The pointer, in logical pixels.
    let mut cursor: Option<(f32, f32)> = None;
    // The last primary press, for double clicks (winit has none).
    let mut last_press: Option<(std::time::Instant, f32, f32)> = None;

    event_loop
        .run(move |event, target| {
            // A caret's blink is due.
            if let Event::NewEvents(winit::event::StartCause::ResumeTimeReached { .. }) = event {
                target.set_control_flow(winit::event_loop::ControlFlow::Wait);
                window_for_loop.request_redraw();
                return;
            }
            if let Event::UserEvent(()) = event {
                if engine.poll() {
                    window_for_loop.request_redraw();
                }
                return;
            }
            let Event::WindowEvent { event, window_id } = event else {
                return;
            };
            if window_id != window_for_loop.id() {
                return;
            }
            let response = match event {
                WindowEvent::CloseRequested => {
                    target.exit();
                    None
                }
                WindowEvent::Resized(new_size) => {
                    config.width = new_size.width.max(1);
                    config.height = new_size.height.max(1);
                    surface.configure(&device, &config);
                    let sf = window_for_loop.scale_factor();
                    ui.scale = sf as f32;
                    ui.size = (config.width, config.height);
                    engine.resize(config.width, config.height, sf);
                    Some(crate::Response { redraw: true, ..Default::default() })
                }
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    ui.scale = scale_factor as f32;
                    engine.resize(config.width, config.height, scale_factor);
                    Some(crate::Response { redraw: true, ..Default::default() })
                }
                WindowEvent::ModifiersChanged(m) => {
                    let s = m.state();
                    modifiers = Modifiers {
                        control: s.contains(ModifiersState::CONTROL),
                        alt: s.contains(ModifiersState::ALT),
                        meta: s.contains(ModifiersState::SUPER),
                        shift: s.contains(ModifiersState::SHIFT),
                    };
                    None
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let button = match button {
                        MouseButton::Left => Button::Primary,
                        MouseButton::Middle => Button::Middle,
                        MouseButton::Right => Button::Secondary,
                        _ => return,
                    };
                    match state {
                        ElementState::Pressed => {
                            let Some((x, y)) = cursor else { return };
                            let mut response = crate::Response::default();
                            if button == Button::Primary {
                                let now = std::time::Instant::now();
                                let double = last_press.is_some_and(|(t, px, py)| now - t < std::time::Duration::from_millis(500) && (px - x).abs() < 4.0 && (py - y).abs() < 4.0);
                                last_press = (!double).then_some((now, x, y));
                                if double {
                                    response = engine.double_click(x, y, modifiers);
                                }
                            }
                            let down = engine.pointer_down(x, y, button, modifiers);
                            response.redraw |= down.redraw;
                            response.requests.extend(down.requests);
                            Some(response)
                        }
                        ElementState::Released => Some(engine.pointer_up(button)),
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let sf = window_for_loop.scale_factor();
                    let (x, y) = ((position.x / sf) as f32, (position.y / sf) as f32);
                    cursor = Some((x, y));
                    let response = engine.pointer_move(x, y);
                    window_for_loop.set_cursor_icon(match engine.cursor(x, y) {
                        Cursor::Default => CursorIcon::Default,
                        Cursor::Pointer => CursorIcon::Pointer,
                        Cursor::Crosshair => CursorIcon::Crosshair,
                        Cursor::Text => CursorIcon::Text,
                    });
                    Some(response)
                }
                WindowEvent::CursorLeft { .. } => {
                    cursor = None;
                    Some(engine.pointer_leave())
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    // DOM semantics: deltaY positive toward the user. winit's
                    // is positive away from the user.
                    let (dx, dy, mode) = match delta {
                        MouseScrollDelta::LineDelta(x, y) => (-x as f64, -y as f64, 1),
                        MouseScrollDelta::PixelDelta(p) => (-p.x, -p.y, 0),
                    };
                    let (x, y) = cursor.unwrap_or((-1.0, -1.0));
                    Some(engine.wheel(x, y, dx, dy, mode, modifiers))
                }
                // A momentary tool ends with its key.
                WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Released => match event.physical_key {
                    PhysicalKey::Code(code) => Some(engine.key_up(&format!("{code:?}"))),
                    _ => None,
                },
                WindowEvent::Focused(false) => Some(engine.blur()),
                WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                    // DOM `key` (winit's named keys are named as the DOM's)
                    // and `code` (winit's `KeyCode` variants are named as
                    // `KeyboardEvent.code` values are).
                    let key = match &event.logical_key {
                        winit::keyboard::Key::Character(c) => c.to_string(),
                        winit::keyboard::Key::Named(n) => format!("{n:?}"),
                        _ => String::new(),
                    };
                    let code = match event.physical_key {
                        PhysicalKey::Code(code) => format!("{code:?}"),
                        _ => String::new(),
                    };
                    Some(engine.key_down(&key, &code, modifiers))
                }
                WindowEvent::RedrawRequested => {
                    match surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(frame_texture)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(frame_texture) => {
                            let view = frame_texture.texture.create_view(&Default::default());
                            ui.size = (config.width, config.height);
                            let ui_canvas = engine.ui().is_some().then_some(&mut ui as &mut dyn UiCanvas);
                            let frame = engine.draw_frame(&view, ui_canvas, || {
                                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()).to_string()
                            });
                            if engine.ui().is_some() {
                                let mut encoder = device.create_command_encoder(&Default::default());
                                image_overlay.draw(&device, &queue, &mut encoder, &view, ui.overlay.pixels(), config.width, config.height);
                                queue.submit([encoder.finish()]);
                            }
                            queue.present(frame_texture);
                            // A finished screenshot: saved in the working directory.
                            if let Some(shot) = frame.screenshot {
                                match std::fs::write(&shot.name, &shot.png) {
                                    Ok(()) => log::info!("saved {}", shot.name),
                                    Err(e) => log::error!("could not save {}: {e}", shot.name),
                                }
                            }
                            if frame.animate {
                                window_for_loop.request_redraw();
                            }
                            if let Some(ms) = frame.redraw_in {
                                let at = std::time::Instant::now() + std::time::Duration::from_secs_f64(ms / 1000.0);
                                target.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(at));
                            }
                        }
                        wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                            surface.configure(&device, &config);
                            window_for_loop.request_redraw();
                        }
                        e => log::error!("surface error: {e:?}"),
                    }
                    None
                }
                _ => None,
            };
            let Some(response) = response else { return };
            for request in response.requests {
                carry_out(&mut engine, request);
            }
            // Redraw on change only, as neuroglancer's `scheduleRedraw` does.
            if response.redraw {
                window_for_loop.set_title(&engine.status());
                window_for_loop.request_redraw();
            }
        })
        .expect("event loop");
}

/// Does what the engine asks of the desktop.
fn carry_out(engine: &mut Engine, request: Request) {
    match request {
        Request::Clipboard(text) => set_clipboard(&text),
        Request::CopyUrl => set_clipboard(&format!("https://neuroglancer-demo.appspot.com/{}", engine.state_fragment())),
        // The system's browser.
        Request::OpenUrl(url) => {
            let opener = if cfg!(target_os = "macos") { "open" } else if cfg!(windows) { "explorer" } else { "xdg-open" };
            if let Err(e) = std::process::Command::new(opener).arg(&url).spawn() {
                log::error!("could not open {url}: {e}");
            }
        }
        // No system colour picker: the UI's own.
        Request::PickColor { setting, current } => engine.open_color_picker(setting, current),
        // Saved in the working directory.
        Request::Download { name, bytes, .. } => match std::fs::write(&name, bytes) {
            Ok(()) => log::info!("saved {name}"),
            Err(e) => log::error!("could not save {name}: {e}"),
        },
    }
}

/// Puts `text` on the system clipboard, through the platform's copy
/// command (`pbcopy`, `clip`, `wl-copy` or `xclip`); printed if none runs.
pub fn set_clipboard(text: &str) {
    use std::io::Write;
    let commands: &[(&str, &[&str])] = if cfg!(target_os = "macos") {
        &[("pbcopy", &[])]
    } else if cfg!(windows) {
        &[("clip", &[])]
    } else {
        &[("wl-copy", &[]), ("xclip", &["-selection", "clipboard"])]
    };
    for (command, args) in commands {
        let child = std::process::Command::new(command).args(*args).stdin(std::process::Stdio::piped()).spawn();
        if let Ok(mut child) = child {
            // Written, then closed: the command reads to its end.
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes());
            }
            if child.wait().is_ok_and(|s| s.success()) {
                return;
            }
        }
    }
    println!("{text}");
}
