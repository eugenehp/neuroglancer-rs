//! A GPU without independent blending or compute shaders (WebGL2): the 3D
//! panel's annotations drawn in split passes, the annotation styles and
//! histogram points worked out without compute, give the same frame as the
//! full path. In a test binary of its own: the capabilities are
//! process-wide.

use std::sync::Arc;

use ng_chunk::ThreadPool;
use ng_viewer::Viewer;
use threers::renderer::readback;

const W: u32 = 800;
const H: u32 = 600;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Every annotation type in 3D (lines, points, ellipsoids, boxes,
/// polylines), the image and segmentation slices, axis lines; 4 panels.
const STATE: &str = r##"{"dimensions": {"x": [4e-09, "m"], "y": [4e-09, "m"], "z": [4e-08, "m"]}, "position": [19.123, 14.777, 10.61], "crossSectionScale": 0.3171, "layers": [{"type": "image", "name": "em", "source": "precomputed://file://{FIXTURES}/image", "opacity": 0.8, "shaderControls": {"normalized": {"range": [50, 200]}}}, {"type": "segmentation", "name": "seg", "source": {"url": "precomputed://file://{FIXTURES}/seg", "transform": {"outputDimensions": {"x": [8e-09, "m"], "y": [8e-09, "m"], "z": [8e-08, "m"]}, "matrix": [[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0]]}}, "colorSeed": 123, "selectedAlpha": 0.6, "segments": ["75", "!82", "106", "44"]}, {"type": "annotation", "name": "ann", "source": "local://annotations", "annotations": [{"type": "point", "point": [19, 14, 10.6], "id": "p1"}, {"type": "point", "point": [25, 10, 10.6], "id": "p2"}, {"type": "point", "point": [10, 20, 11.4], "id": "p3"}, {"type": "line", "pointA": [5, 5, 10.6], "pointB": [30, 25, 10.6], "id": "l1"}, {"type": "line", "pointA": [8, 25, 8], "pointB": [30, 5, 13], "id": "l2"}, {"type": "axis_aligned_bounding_box", "pointA": [12, 8, 9], "pointB": [22, 18, 12], "id": "b1"}, {"type": "ellipsoid", "center": [28, 22, 10], "radii": [6, 4, 2], "id": "e1"}, {"type": "polyline", "points": [[6, 24, 10.6], [12, 27, 10.6], [18, 24, 10.6], [24, 27, 10.6]], "id": "pl1"}]}], "showAxisLines": true, "showScaleBar": false, "crossSectionBackgroundColor": "#203040", "layout": "4panel", "projectionOrientation": [0.25, 0.2, 0.05, 0.9460443964212251], "projectionScale": 250}"##;

fn frame(gpu: &threers::gpu::GpuContext) -> Vec<u8> {
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let state: serde_json::Value = serde_json::from_str(&STATE.replace("{FIXTURES}", fixtures)).unwrap();
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let mut viewer = Viewer::new(gpu.device.clone(), gpu.queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
    viewer.open_state(&state).unwrap();
    let start = std::time::Instant::now();
    loop {
        viewer.render(&view);
        viewer.poll();
        if !viewer.is_loading() {
            viewer.render(&view);
            viewer.poll();
            if !viewer.is_loading() {
                break;
            }
        }
        assert!(start.elapsed().as_secs() < 60, "still loading");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    viewer.render(&view);
    readback::texture_bytes(&gpu.device, &gpu.queue, &target)
}

#[test]
fn a_downlevel_gpu_draws_the_same_frame() {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer downlevel") else { return };
    ng_render::capabilities::set(true, true);
    let full = frame(&gpu);
    ng_render::capabilities::set(false, false);
    let downlevel = frame(&gpu);
    ng_render::capabilities::set(true, true);
    assert!(full.chunks(4).filter(|p| p[..3] != full[..3]).count() > 1000, "a blank frame");
    let differing = full.chunks(4).zip(downlevel.chunks(4)).filter(|(a, b)| a != b).count();
    if differing > 0 {
        let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
        std::fs::write(out.join("ng-downlevel-full.png"), threers::encode_png(W, H, &full)).unwrap();
        std::fs::write(out.join("ng-downlevel-split.png"), threers::encode_png(W, H, &downlevel)).unwrap();
    }
    assert_eq!(differing, 0, "pixels differ between the full and the downlevel paths");
}
