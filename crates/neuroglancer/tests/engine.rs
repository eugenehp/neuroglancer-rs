//! The engine as a downstream crate uses it: headless, on its own device,
//! rendering into a texture. Its frames are the viewer's own, pixel for
//! pixel; with the UI, the UI paints over them. Skips without a GPU.

use std::sync::Arc;

use neuroglancer::prelude::*;
use neuroglancer::threers::renderer::readback;

const W: u32 = 640;
const H: u32 = 480;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const NIFTI: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../neuroglancer/testdata/datasource/nifti/standard.nii");

fn target(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// The UI painted into a CPU overlay.
struct Overlay(neuroglancer::threers::overlay::Overlay);

impl UiCanvas for Overlay {
    fn begin(&mut self) {
        self.0.clear();
    }
    fn with_painter(&mut self, f: &mut dyn FnMut(&mut dyn Painter)) {
        f(&mut neuroglancer::threers::ui::OverlayPainter::with_scale(&mut self.0, 1.0));
    }
}

fn engine(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>, ui: bool) -> Engine {
    let mut engine = Engine::new(EngineOptions {
        device: device.clone(),
        queue: queue.clone(),
        format: FORMAT,
        width: W,
        height: H,
        scale_factor: 1.0,
        executor: Arc::new(ThreadPool::new(2)),
        wake: Arc::new(|| {}),
        ui,
        now: || 0.0,
    });
    engine.viewer_mut().open(NIFTI).unwrap();
    engine
}

/// Draws until the data is in (a minute at most).
fn draw_loaded(engine: &mut Engine, view: &wgpu::TextureView, mut ui: Option<&mut Overlay>) {
    let start = std::time::Instant::now();
    loop {
        engine.draw_frame(view, ui.as_deref_mut().map(|u| u as &mut dyn UiCanvas), String::new);
        if engine.is_ready() {
            engine.draw_frame(view, ui.as_deref_mut().map(|u| u as &mut dyn UiCanvas), String::new);
            return;
        }
        assert!(start.elapsed().as_secs() < 60, "still loading");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn engine_frames_are_the_viewers_and_the_ui_paints_over_them() {
    let Some(gpu) = neuroglancer::threers::gpu::testing::context("neuroglancer engine") else {
        return;
    };
    eprintln!("adapter: {}", gpu.describe());
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());

    // The viewer alone.
    let texture = target(&device);
    let view = texture.create_view(&Default::default());
    let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
    viewer.open(NIFTI).unwrap();
    let start = std::time::Instant::now();
    loop {
        viewer.render(&view);
        if !viewer.is_loading() {
            viewer.poll();
            viewer.render(&view);
            if !viewer.is_loading() {
                break;
            }
        }
        assert!(start.elapsed().as_secs() < 60, "still loading");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let expected = readback::texture_bytes(&device, &queue, &texture);

    // The engine without its UI: the same frame.
    let texture = target(&device);
    let view = texture.create_view(&Default::default());
    let mut panels = engine(&device, &queue, false);
    draw_loaded(&mut panels, &view, None);
    let got = readback::texture_bytes(&device, &queue, &texture);
    assert!(got.chunks(4).any(|p| p[..3] != got[..3]), "a blank frame");
    assert!(got == expected, "the engine's frame differs from the viewer's");

    // With the UI: it lays out over the panels and paints (the layer bar).
    let mut overlay = Overlay(neuroglancer::threers::overlay::Overlay::new(W, H));
    let mut full = engine(&device, &queue, true);
    draw_loaded(&mut full, &view, Some(&mut overlay));
    let painted = overlay.0.pixels().chunks(4).filter(|p| p[3] > 0).count();
    assert!(painted > 1000, "the UI painted {painted} pixels");
    assert!(full.ui().is_some_and(|u| !u.text_rects().is_empty()));
    eprintln!("engine: {painted} UI pixels over a frame identical to the viewer's");
}
