//! Scene mode (`Viewer::render_scene`): the 3D content drawn from a host's
//! camera, as an embedder composites it into its own frame.
//!
//! The host's world here is millimetres with the data at its physical
//! position; the camera looks down −z at the navigation position. The
//! cross-section through that position must land in the middle of the
//! frame, at the depth the host's own projection gives that point, with
//! nothing drawn far outside the data. Skips without a GPU.

use std::sync::Arc;

use ng_chunk::ThreadPool;
use ng_core::matrix::IDENTITY4;
use ng_viewer::scene::{HostCamera, SceneView};
use ng_viewer::Viewer;
use threers::renderer::readback;

const W: u32 = 320;
const H: u32 = 240;
const NIFTI: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../neuroglancer/testdata/datasource/nifti/standard.nii");

/// `gl-matrix`'s `perspective` (column-major, GL clip).
fn perspective(fovy: f64, aspect: f64, near: f64, far: f64) -> [f64; 16] {
    let f = 1.0 / (fovy / 2.0).tan();
    let mut m = [0.0; 16];
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) / (near - far);
    m[11] = -1.0;
    m[14] = 2.0 * far * near / (near - far);
    m
}

fn transform(m: &[f64; 16], p: [f64; 3]) -> [f64; 4] {
    std::array::from_fn(|r| (0..3).map(|c| m[c * 4 + r] * p[c]).sum::<f64>() + m[12 + r])
}

#[test]
fn scene_draws_the_cross_section_at_the_host_cameras_depth() {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer scene") else {
        return;
    };
    if !std::path::Path::new(NIFTI).exists() {
        eprintln!("skipped: {NIFTI} (neuroglancer's test data) is absent");
        return;
    }
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut viewer = Viewer::new(device.clone(), queue.clone(), format, W, H, 1.0, Arc::new(ThreadPool::new(4)), Arc::new(|| {}));
    viewer.open(NIFTI).unwrap();
    // Wait for the source, so the navigation has its position and scales.
    let start = std::time::Instant::now();
    while viewer.navigation().render_info().display_rank == 0 && start.elapsed().as_secs() < 30 {
        viewer.poll();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let info = viewer.navigation().render_info().clone();
    let position = viewer.navigation().position().to_vec();
    // The navigation position in host-world millimetres.
    let centre: [f64; 3] = std::array::from_fn(|k| position[info.display_dimension_indices[k] as usize] as f64 * info.voxel_physical_scales[k] * 1e3);
    let mut physical_from_world = IDENTITY4;
    for k in 0..3 {
        physical_from_world[k * 5] = 1e-3;
    }
    // Looking down −z from twice the data's largest extent above the centre
    // (`standard.nii`: 4 × 5 × 7 voxels of 1 × 3 × 2, anisotropic on purpose).
    let extent_mm = info.voxel_physical_scales.iter().fold(0f64, |a, &s| a.max(s)) * 1e3 * 8.0;
    let distance = 2.0 * extent_mm;
    let mut view = IDENTITY4;
    for k in 0..3 {
        view[12 + k] = -centre[k];
    }
    view[14] -= distance;
    let projection = perspective(std::f64::consts::FRAC_PI_4, W as f64 / H as f64, distance / 10.0, distance * 10.0);
    let scene = SceneView { width: W, height: H, slice_size: 256, camera: HostCamera { view, projection, physical_from_world }, bounds: None };

    let start = std::time::Instant::now();
    loop {
        viewer.render_scene(scene);
        viewer.poll();
        if !viewer.is_loading() || start.elapsed().as_secs() > 60 {
            viewer.render_scene(scene);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let out = viewer.scene_output().expect("scene output after render_scene");
    assert_eq!((out.width, out.height), (W, H));
    let color = readback::texture_bytes(&device, &queue, out.color.texture());
    let depth = readback::texture_bytes(&device, &queue, out.depth.texture());
    let png = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/ng-viewer-scene.png");
    std::fs::write(&png, threers::encode_png(W, H, &color)).unwrap();

    // Rows are bottom-up; the centre is the centre either way.
    let px = |x: u32, y: u32| ((y * W + x) * 4) as usize;
    let (cx, cy) = (W / 2, H / 2);
    assert!(color[px(cx, cy) + 3] > 0, "nothing drawn at the centre");
    for (x, y) in [(1, 1), (W - 2, 1), (1, H - 2), (W - 2, H - 2)] {
        assert_eq!(color[px(x, y) + 3], 0, "drawn far outside the data at ({x}, {y})");
    }
    // The xy cross-section faces the camera, so its depth is the host's
    // projection of the navigation position at every pixel it covers. Read
    // it off the centre lines, where the xz and yz planes are seen edge-on.
    let clip = transform(&projection, std::array::from_fn(|k| transform(&view, centre)[k]));
    let expected = (clip[2] / clip[3] + 1.0) / 2.0;
    let mut read = 0;
    for (dx, dy) in [(6i32, 6i32), (-6, 6), (6, -6), (-6, -6)] {
        let (x, y) = ((cx as i32 + dx) as u32, (cy as i32 + dy) as u32);
        if color[px(x, y) + 3] == 0 {
            continue;
        }
        let d = f32::from_le_bytes(depth[px(x, y)..px(x, y) + 4].try_into().unwrap()) as f64;
        assert!((d - expected).abs() < 1e-4, "depth {d} at ({x}, {y}), the host's projection gives {expected}");
        read += 1;
    }
    assert!(read > 0, "no data drawn beside the centre");

    // A ray straight down through the centre meets the xy cross-section at the
    // navigation position, where the volume has a value.
    let origin = [centre[0] + 0.3, centre[1] + 0.2, centre[2] + distance];
    let pick = viewer.scene_probe(&physical_from_world, origin, [0.0, 0.0, -1.0], scene.slice_size).expect("a cross-section under the ray");
    assert_eq!(pick.plane, 0, "the xy cross-section faces the ray");
    for k in 0..2 {
        assert!((pick.world[k] - origin[k]).abs() < 1e-6, "{:?} vs {origin:?}", pick.world);
    }
    assert!((pick.world[2] - centre[2]).abs() < 1e-4, "{:?} vs centre {centre:?}", pick.world);
    assert!(!pick.values.is_empty() && !pick.values[0].1.is_empty(), "{:?}", pick.values);
    // A ray along x meets the yz cross-section (all three are drawn under
    // the default layout); with a layout of one slice (`xy-3d`) only that
    // slice is drawn, and only it is picked.
    let side = [centre[0] - distance, centre[1] + 0.2, centre[2] + 0.1];
    let along_x = viewer.scene_probe(&physical_from_world, side, [1.0, 0.0, 0.0], scene.slice_size);
    assert_eq!(along_x.map(|p| p.plane), Some(2), "the yz cross-section under the default layout");
    viewer.layout = ng_viewer::Layout::SliceAnd3d(ng_core::panel::NamedAxes::Xy);
    assert!(viewer.scene_probe(&physical_from_world, side, [1.0, 0.0, 0.0], scene.slice_size).is_none(), "an undrawn slice was picked");
    assert_eq!(viewer.scene_probe(&physical_from_world, origin, [0.0, 0.0, -1.0], scene.slice_size).map(|p| p.plane), Some(0));
    viewer.layout = ng_viewer::Layout::FourPanel;
    // A plane the host hides (`scene_planes`) is neither drawn nor picked;
    // the others still are.
    viewer.scene_planes = [true, true, false];
    assert!(viewer.scene_probe(&physical_from_world, side, [1.0, 0.0, 0.0], scene.slice_size).is_none(), "a hidden plane was picked");
    assert_eq!(viewer.scene_probe(&physical_from_world, origin, [0.0, 0.0, -1.0], scene.slice_size).map(|p| p.plane), Some(0));
    viewer.scene_planes = [true; 3];
    // Moving the crossing point moves the cross-section with it.
    let p0 = viewer.scene_position().unwrap();
    assert!(viewer.scene_set_position([p0[0], p0[1], p0[2] + 1e-3]));
    let moved = viewer.scene_probe(&physical_from_world, origin, [0.0, 0.0, -1.0], scene.slice_size);
    if let Some(m) = moved {
        assert!((m.world[2] - (centre[2] + 1.0)).abs() < 1e-4, "{:?}", m.world);
    }
}

/// A host camera close to the cross-section, relative to the slice's field of
/// view (an embedder zoomed in on tissue): the plane facing the camera must
/// still be drawn across the frame. `far_offset_mm` moves the whole world
/// far from the origin, as a brain's coordinates are.
fn close_camera_draws(distance_factor: f64, far_offset_mm: f64) -> u32 {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer scene close") else {
        return u32::MAX;
    };
    if !std::path::Path::new(NIFTI).exists() {
        return u32::MAX;
    }
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let mut viewer = Viewer::new(device.clone(), queue.clone(), wgpu::TextureFormat::Rgba8Unorm, W, H, 1.0, Arc::new(ThreadPool::new(4)), Arc::new(|| {}));
    viewer.open(NIFTI).unwrap();
    let start = std::time::Instant::now();
    while viewer.navigation().render_info().display_rank == 0 && start.elapsed().as_secs() < 30 {
        viewer.poll();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let info = viewer.navigation().render_info().clone();
    let position = viewer.navigation().position().to_vec();
    let centre_phys: [f64; 3] = std::array::from_fn(|k| position[info.display_dimension_indices[k] as usize] as f64 * info.voxel_physical_scales[k] * 1e3);
    // world = physical (mm) + offset
    let mut physical_from_world = IDENTITY4;
    for k in 0..3 {
        physical_from_world[k * 5] = 1e-3;
        physical_from_world[12 + k] = -far_offset_mm * 1e-3;
    }
    let centre: [f64; 3] = std::array::from_fn(|k| centre_phys[k] + far_offset_mm);
    let extent_mm = info.voxel_physical_scales.iter().fold(0f64, |a, &s| a.max(s)) * 1e3 * 8.0;
    let distance = extent_mm * distance_factor;
    let mut view = IDENTITY4;
    for k in 0..3 {
        view[12 + k] = -centre[k];
    }
    view[14] -= distance;
    let projection = perspective(std::f64::consts::FRAC_PI_4, W as f64 / H as f64, distance / 50.0, distance * 1000.0);
    let scene = SceneView { width: W, height: H, slice_size: 256, camera: HostCamera { view, projection, physical_from_world }, bounds: None };
    let start = std::time::Instant::now();
    loop {
        viewer.render_scene(scene);
        viewer.poll();
        if !viewer.is_loading() || start.elapsed().as_secs() > 60 {
            viewer.render_scene(scene);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let out = viewer.scene_output().unwrap();
    let color = readback::texture_bytes(&device, &queue, out.color.texture());
    let png = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../target/ng-viewer-scene-close-{distance_factor}-{far_offset_mm}.png"));
    std::fs::write(&png, threers::encode_png(W, H, &color)).unwrap();
    (0..W * H).filter(|i| color[(i * 4 + 3) as usize] > 0).count() as u32
}

#[test]
fn scene_draws_the_cross_section_with_the_camera_close() {
    for (factor, offset) in [(2.0, 0.0), (0.2, 0.0), (0.02, 0.0), (2.0, 60.0), (0.2, 60.0), (0.02, 60.0)] {
        let n = close_camera_draws(factor, offset);
        if n == u32::MAX {
            return;
        }
        eprintln!("camera at {factor} × extent, world offset {offset} mm: {n} of {} pixels drawn", W * H);
        assert!(n > W * H / 4, "camera at {factor} × the data's extent (offset {offset} mm): only {n} pixels drawn");
    }
}

/// Volume rendering in scene mode: an image layer with `volumeRendering: on`
/// is ray-marched in the host's 3D view — the host composites the 3D panel,
/// so the volume has to be in it — and with it off (and the slices hidden)
/// nothing is drawn.
#[test]
fn scene_volume_renders_images() {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer scene volume rendering") else {
        return;
    };
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let format = wgpu::TextureFormat::Rgba8Unorm;
    // the host world in micrometres; the volume (40 × 30 × 20 voxels of
    // 4 × 4 × 40 nm) centred at (0.082, 0.062, 0.42) µm, looked at down −z
    let mut physical_from_world = IDENTITY4;
    for k in 0..3 {
        physical_from_world[k * 5] = 1e-6;
    }
    let centre = [0.082, 0.062, 0.42];
    // just above the volume's top (z 0.8 µm)
    let distance = 0.6;
    let mut view = IDENTITY4;
    for k in 0..3 {
        view[12 + k] = -centre[k];
    }
    view[14] -= distance;
    let projection = perspective(std::f64::consts::FRAC_PI_4, W as f64 / H as f64, 0.05, 50.0);
    let scene = SceneView { width: W, height: H, slice_size: 128, camera: HostCamera { view, projection, physical_from_world }, bounds: None };
    let draw = |mode: &str, slices: bool| {
        let state = serde_json::json!({
            "dimensions": { "x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"] },
            "position": [20.5, 15.5, 10.5],
            "crossSectionScale": 1,
            "layers": [{ "type": "image", "name": "em", "source": format!("precomputed://file://{fixtures}/image"), "volumeRendering": mode }],
            "showAxisLines": false,
            "showDefaultAnnotations": false,
            "showSlices": slices,
            "layout": "4panel"
        });
        let mut viewer = Viewer::new(device.clone(), queue.clone(), format, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
        viewer.open_state(&state).unwrap();
        let start = std::time::Instant::now();
        let mut frames = 0;
        loop {
            viewer.render_scene(scene);
            viewer.poll();
            frames += 1;
            if (!viewer.is_loading() && start.elapsed().as_millis() > 2000) || start.elapsed().as_secs() > 60 {
                viewer.render_scene(scene);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        eprintln!("{mode} slices={slices}: {frames} frames, loading={}", viewer.is_loading());
        let out = viewer.scene_output().expect("scene output");
        let color = readback::texture_bytes(&device, &queue, out.color.texture());
        let png = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../target/ng-viewer-scene-volume-{mode}-{slices}.png"));
        std::fs::write(&png, threers::encode_png(W, H, &color)).unwrap();
        color.chunks_exact(4).filter(|p| p[3] > 0).count()
    };
    // the camera sees the data: its slices are drawn
    let slices = draw("off", true);
    assert!(slices > 2000, "the slices, as a check on the camera: {slices} pixels");
    let off = draw("off", false);
    let on = draw("on", false);
    assert_eq!(off, 0, "slices hidden, no volume rendering: nothing drawn");
    assert!(on > slices / 2, "the volume is ray-marched in the host's view: {on} pixels drawn");
}

/// The slices are drawn again only when what they show changes: a host
/// camera moving around them (orbiting) redraws none, and the frame it comes
/// back to is the same, pixel for pixel; moving the crossing point redraws.
#[test]
fn scene_slices_are_not_redrawn_for_a_moving_camera() {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer scene slices kept") else {
        return;
    };
    if !std::path::Path::new(NIFTI).exists() {
        eprintln!("skipped: {NIFTI} (neuroglancer's test data) is absent");
        return;
    }
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let mut viewer = Viewer::new_scene(device.clone(), queue.clone(), Arc::new(ThreadPool::new(4)), Arc::new(|| {}), Default::default());
    viewer.open(NIFTI).unwrap();
    let start = std::time::Instant::now();
    while viewer.navigation().render_info().display_rank == 0 && start.elapsed().as_secs() < 30 {
        viewer.poll();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let info = viewer.navigation().render_info().clone();
    let position = viewer.navigation().position().to_vec();
    let centre: [f64; 3] = std::array::from_fn(|k| position[info.display_dimension_indices[k] as usize] as f64 * info.voxel_physical_scales[k] * 1e3);
    let mut physical_from_world = IDENTITY4;
    for k in 0..3 {
        physical_from_world[k * 5] = 1e-3;
    }
    let extent_mm = info.voxel_physical_scales.iter().fold(0f64, |a, &s| a.max(s)) * 1e3 * 8.0;
    let distance = 2.0 * extent_mm;
    // looking down −z at the centre, from straight above or a little aside
    let camera = |dx: f64| {
        let mut view = IDENTITY4;
        view[12] = -centre[0] - dx;
        view[13] = -centre[1];
        view[14] = -centre[2] - distance;
        let projection = perspective(std::f64::consts::FRAC_PI_4, W as f64 / H as f64, distance / 10.0, distance * 10.0);
        SceneView { width: W, height: H, slice_size: 128, camera: HostCamera { view, projection, physical_from_world }, bounds: None }
    };
    let (a, b) = (camera(0.0), camera(extent_mm * 0.3));
    let start = std::time::Instant::now();
    loop {
        viewer.render_scene(a);
        viewer.poll();
        if !viewer.is_loading() || start.elapsed().as_secs() > 60 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    // everything loaded: one more frame settles what the slices show
    viewer.render_scene(a);
    viewer.render_scene(a);
    let read = |viewer: &Viewer| readback::texture_bytes(&device, &queue, viewer.scene_output().unwrap().color.texture());
    let first = read(&viewer);
    let drawn = viewer.slice_redraws();
    viewer.render_scene(b);
    let aside = read(&viewer);
    viewer.render_scene(a);
    let back = read(&viewer);
    assert_eq!(viewer.slice_redraws(), drawn, "the camera moved, and the slices were drawn again");
    assert_ne!(first, aside, "the camera moved and nothing changed");
    assert!(first == back, "back where it was, the frame differs");
    // the crossing point moved: the slices show something else
    let p = viewer.scene_position().unwrap();
    assert!(viewer.scene_set_position([p[0] + info.voxel_physical_scales[0], p[1], p[2]]));
    viewer.render_scene(a);
    assert!(viewer.slice_redraws() > drawn, "the slices moved, and were not drawn again");

    // Composited over a host frame: shown where the host's depth is clear,
    // hidden behind a host surface nearer than everything (depth 0).
    let target = |label| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let host_depth = |d: f32| {
        let t = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("host depth"),
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = t.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(d), store: wgpu::StoreOp::Store }), stencil_ops: None }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
        queue.submit([encoder.finish()]);
        view
    };
    let mut drawn_pixels = |depth: f32| {
        let t = target("host frame");
        let view = t.create_view(&Default::default());
        viewer.composite_scene(&view, wgpu::TextureFormat::Rgba8Unorm, &host_depth(depth), 1, false);
        readback::texture_bytes(&device, &queue, &t).chunks_exact(4).filter(|p| p[3] > 0).count()
    };
    assert!(drawn_pixels(1.0) > 0, "nothing composited over a clear depth");
    assert_eq!(drawn_pixels(0.0), 0, "composited through a nearer host surface");
}
