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
use ng_viewer::scene::{HostCamera, SceneView};
use ng_viewer::Viewer;
use threers::renderer::readback;

const W: u32 = 320;
const H: u32 = 240;
const NIFTI: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../neuroglancer/testdata/datasource/nifti/standard.nii");

fn identity() -> [f64; 16] {
    let mut m = [0.0; 16];
    for k in 0..4 {
        m[k * 5] = 1.0;
    }
    m
}

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
    let mut physical_from_world = identity();
    for k in 0..3 {
        physical_from_world[k * 5] = 1e-3;
    }
    // Looking down −z from twice the data's largest extent above the centre
    // (`standard.nii`: 4 × 5 × 7 voxels of 1 × 3 × 2, anisotropic on purpose).
    let extent_mm = info.voxel_physical_scales.iter().fold(0f64, |a, &s| a.max(s)) * 1e3 * 8.0;
    let distance = 2.0 * extent_mm;
    let mut view = identity();
    for k in 0..3 {
        view[12 + k] = -centre[k];
    }
    view[14] -= distance;
    let projection = perspective(std::f64::consts::FRAC_PI_4, W as f64 / H as f64, distance / 10.0, distance * 10.0);
    let scene = SceneView { width: W, height: H, slice_size: 256, camera: HostCamera { view, projection, physical_from_world } };

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
    // Moving the crossing point moves the cross-section with it.
    let p0 = viewer.scene_position().unwrap();
    assert!(viewer.scene_set_position([p0[0], p0[1], p0[2] + 1e-3]));
    let moved = viewer.scene_probe(&physical_from_world, origin, [0.0, 0.0, -1.0], scene.slice_size);
    if let Some(m) = moved {
        assert!((m.world[2] - (centre[2] + 1.0)).abs() < 1e-4, "{:?}", m.world);
    }
}
