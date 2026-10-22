//! Renders neuroglancer's NIfTI test files offscreen and checks each sampled
//! pixel against the value under the pointer there.
//!
//! The expectation takes a separate path from the GPU: the pointer's global
//! position comes from `SlicePanel::mouse_position` (the bit-exact port of
//! neuroglancer's), the voxel from `ImageLayer::value_at` (the inverse
//! transform), the grey from neuroglancer's invlerp over the data type's
//! default range, and the background is `crossSectionBackgroundColor` where
//! there is no data. The GPU draws through the composed NDC → source matrix.
//!
//! Writes each frame to `target/ng-viewer-<file>.png`. Skips without a GPU.

use std::sync::Arc;

use ng_chunk::ThreadPool;
use ng_core::navigation::View;
use ng_core::panel::NamedAxes;
use ng_core::segment_color::SegmentColorHash;
use ng_viewer::{LayerKind, Layout, Panel, Viewer, CROSS_SECTION_BACKGROUND};
use threers::renderer::readback;

const W: u32 = 800;
const H: u32 = 600;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../neuroglancer/testdata/datasource/nifti/");

/// Whether pixel `(px, py)` of a panel at `r` is covered by an annotation line
/// the viewer drew there (data bounds): those pixels show the line.
fn under_annotation(viewer: &Viewer, axes: NamedAxes, r: threers::renderer::PixelRect, px: u32, py: u32) -> bool {
    let (x, y) = ((px - r.x) as f64 + 0.5, (py - r.y) as f64 + 0.5);
    viewer.annotation_lines(axes).iter().any(|l| {
        let half = l.width / 2.0 + l.feather + 0.5;
        let (dx, dy) = (l.b[0] - l.a[0], l.b[1] - l.a[1]);
        let len2 = dx * dx + dy * dy;
        let t = if len2 == 0.0 { 0.0 } else { (((x - l.a[0]) * dx + (y - l.a[1]) * dy) / len2).clamp(0.0, 1.0) };
        let (cx, cy) = (l.a[0] + t * dx, l.a[1] + t * dy);
        ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() <= half
    })
}

/// Renders until every requested chunk has loaded (a minute at most, for
/// sources over the network).
fn render_loaded(viewer: &mut Viewer, view: &wgpu::TextureView) {
    let start = std::time::Instant::now();
    while start.elapsed() < std::time::Duration::from_secs(60) {
        viewer.render(view);
        if !viewer.is_loading() {
            viewer.poll();
            viewer.render(view);
            if !viewer.is_loading() {
                return;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    panic!("chunks still loading");
}

fn check_file(name: &str, zoom: f64) {
    check_url(&format!("{TESTDATA}{name}"), name.split('.').next().unwrap(), zoom);
}

fn check_url(url: &str, stem: &str, zoom: f64) {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer headless render") else {
        return;
    };
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let target = device.create_texture(&wgpu::TextureDescriptor {
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

    let mut viewer = Viewer::new(
        device.clone(),
        queue.clone(),
        FORMAT,
        W,
        H,
        1.0,
        Arc::new(ThreadPool::new(4)),
        Arc::new(|| {}),
    );
    viewer.open(url).unwrap_or_else(|e| panic!("{url}: {e:#}"));
    let name = stem;
    viewer.navigation_mut().set_zoom(View::CrossSection, zoom);
    // Over the 3D panel while drawing: no segment is hovered.
    viewer.pointer_move(2.0, H as f64 - 2.0);
    render_loaded(&mut viewer, &view);
    let rgba = readback::texture_bytes(&device, &queue, &target);
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../target/ng-viewer-{stem}.png"));
    std::fs::write(&out, threers::encode_png(W, H, &rgba)).unwrap();

    let background = (CROSS_SECTION_BACKGROUND[0] * 255.0).round() as i32;
    let (mut checked, mut inside, mut mismatched) = (0, 0, 0);
    for (panel, r) in viewer.layout.panels(W, H) {
        let Panel::Slice(axes) = panel else { continue };
        for gy in 1..24 {
            for gx in 1..24 {
                let px = r.x + r.width * gx / 24;
                let py = r.y + r.height * gy / 24;
                if under_annotation(&viewer, axes, r, px, py) {
                    continue;
                }
                viewer.pointer_move(px as f64 + 0.5, py as f64 + 0.5);
                let pos = viewer.mouse_position().unwrap();
                // Skip pixels whose center is on a voxel face: either side is
                // right, and the GPU and CPU may round differently.
                // Only the displayed dimensions vary across the panel.
                let voxel = viewer.layer_voxel(0, &pos).unwrap();
                if voxel.iter().take(3).any(|c| (c - c.round()).abs() < 1e-3) {
                    continue;
                }
                // A display dimension the data lacks has extent [0, 1)
                // (neuroglancer's chunk display bounds for it): outside that,
                // the panel shows background.
                let display_rank = viewer.navigation().render_info().display_rank;
                let point = viewer.mouse_display_point().unwrap();
                let off_plane = (display_rank..3).any(|i| !(0.0..1.0).contains(&point[i]));
                // On or near the edge of that extent, either side is right —
                // except in the xy panel, where the missing dimension does not
                // vary across the plane, so an exact 0 is unambiguous.
                let unambiguous = |v: f32| axes == NamedAxes::Xy && v == 0.0;
                if (display_rank..3).any(|i| !unambiguous(point[i]) && (point[i] - point[i].round()).abs() < 1e-3) {
                    continue;
                }
                let value = if off_plane { None } else { viewer.layer_value_at(0, &pos) };
                let expected: [i32; 3] = match (value, viewer.layers()[0].kind().unwrap()) {
                    (Some(v), LayerKind::Image { .. }) => {
                        inside += 1;
                        let t = viewer.layers()[0].data_type().unwrap();
                        let g = (ng_render::image::expected_gray(t, range(&viewer), v.parse().unwrap()) * 255.0).round() as i32;
                        [g; 3]
                    }
                    (Some(v), LayerKind::Segmentation(display)) => {
                        inside += 1;
                        let id: u64 = v.parse().unwrap();
                        if id == 0 {
                            [background; 3]
                        } else {
                            // Not hovered, all segments shown: the hash colour
                            // at full saturation.
                            assert_eq!(display.saturation, 1.0);
                            SegmentColorHash::new(display.color_seed).compute(id).map(|c| (c * 255.0).round() as i32)
                        }
                    }
                    (None, _) => [background; 3],
                };
                checked += 1;
                let i = ((py * W + px) * 4) as usize;
                let got = &rgba[i..i + 4];
                if (0..3).any(|c| (got[c] as i32 - expected[c]).abs() > 1) {
                    mismatched += 1;
                    eprintln!("{name} {axes:?} ({px},{py}) pos {pos:?}: got {got:?}, expected {expected:?}");
                }
            }
        }
    }
    eprintln!("{name}: {checked} pixels, {inside} inside the volume, {mismatched} off; {}", out.display());
    assert!(inside >= 50, "{name}: too few samples inside the volume ({inside}/{checked})");
    assert_eq!(mismatched, 0, "{name}");
}

fn range(viewer: &Viewer) -> ng_render::image::Interval {
    viewer.layers()[0].range().unwrap()
}

#[test]
fn standard_nii() {
    // 4×5×7 uint8, pixdim (1, 3, 2): 20 pixels per global voxel.
    check_file("standard.nii", 0.05);
}

#[test]
fn standard_nii_gz() {
    check_file("standard.nii.gz", 0.05);
}

#[test]
fn example_nifti2_rotated_4d() {
    // 32×20×12×2 int16 with a qform rotation, in mm.
    check_file("example_nifti2.nii.gz", 0.25);
}

#[test]
fn ome_zarr_multiscale_2d() {
    // 10×10 uint16 at three scales; 2-D, so only the xy panel shows data.
    check_url(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../../neuroglancer/testdata/datasource/zarr/ome_zarr/simple_0.4"),
        "ome_simple_0.4",
        0.1,
    );
}

/// A uint64 segmentation (precomputed, compressed segmentation in a shard,
/// written by tensorstore): neuroglancer's segment colours, segment 0 hidden.
#[test]
fn segmentation_uint64_sharded() {
    check_url(
        concat!(
            "precomputed://",
            env!("CARGO_MANIFEST_DIR"),
            "/../ng-datasource/tests/fixtures/sharded/one_shard_cseg_u64"
        ),
        "segmentation",
        0.05,
    );
}

/// The README's example: precomputed JPEG chunks from `gs://`, at the finest
/// scale (one 8 nm voxel per pixel). Needs the network:
/// `cargo test -p ng-viewer --test headless_render -- --ignored`.
#[test]
#[ignore]
fn fib25_precomputed_jpeg_over_gcs() {
    // Not a whole zoom: at 1 voxel a pixel, centred on a voxel corner, every
    // sampled pixel centre is on a voxel face (and skipped).
    check_url("precomputed://gs://neuroglancer-public-data/flyem_fib-25/image", "fib25", 0.9);
}

/// Sharded precomputed JPEG (hemibrain EM, identity hash, gzip minishard
/// indexes), at a fine scale. Needs the network.
#[test]
#[ignore]
fn hemibrain_sharded_precomputed_over_gcs() {
    check_url("precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg", "hemibrain", 0.9);
}

#[test]
fn sharded_zarr_3d() {
    // 5×7×3 uint16 in 4×6×2 shards of 2×3×1 sub-chunks, some never written.
    check_url(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../../exg/crates/exg-zarr/tests/fixtures/codecs/v3_shard_end_u16.zarr"),
        "shard_end_u16",
        0.02,
    );
}

/// Viewer state with two layers (`tests/fixtures`, written by tensorstore):
/// an image with a `shaderControls` range and `opacity`, and a segmentation at
/// half the resolution, shifted by a `transform` spec, with its own colour
/// seed and alpha and a list of visible segments (the rest at
/// `notSelectedAlpha`, 0), over explicit `dimensions` and a background
/// colour. Each
/// sampled pixel must be neuroglancer's composite of the two layers' values
/// under it: the image replaces, the segmentation blends over it
/// (`SRC_ALPHA, ONE_MINUS_SRC_ALPHA` on colour and alpha, in 8 bits), and
/// alpha 0 shows the background.
#[test]
fn state_with_image_and_segmentation_layers() {
    check_two_layer_state("4panel", "state");
}

/// The same state in other layouts: one big cross-section, and a
/// cross-section beside the 3D view (`xz`, `yz-3d`).
#[test]
fn state_layouts() {
    check_two_layer_state("xz", "state-xz");
    check_two_layer_state("yz-3d", "state-yz-3d");
}

fn check_two_layer_state(layout: &str, stem: &str) {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer state render") else {
        return;
    };
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let state = serde_json::json!({
        "dimensions": { "x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"] },
        "position": [20.5, 15.5, 10.5],
        "crossSectionScale": 0.2,
        "layers": [
            { "type": "image", "name": "em", "source": format!("precomputed://file://{fixtures}/image"),
              "opacity": 0.8, "shaderControls": { "normalized": { "range": [50, 200] } } },
            { "type": "segmentation", "name": "seg",
              "source": { "url": format!("precomputed://file://{fixtures}/seg"),
                          "transform": { "outputDimensions": { "x": [8e-9, "m"], "y": [8e-9, "m"], "z": [8e-8, "m"] },
                                         "matrix": [[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0]] } },
              "colorSeed": 123, "selectedAlpha": 0.6, "segments": ["75", "!82", 106, "44"] }
        ],
        "crossSectionBackgroundColor": "#203040",
        "layout": layout
    });
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let target = device.create_texture(&wgpu::TextureDescriptor {
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
    let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(4)), Arc::new(|| {}));
    viewer.open_state(&state).unwrap();
    assert!(viewer.messages.is_empty(), "{:?}", viewer.messages);
    for l in viewer.layers() {
        assert!(l.messages.is_empty(), "{}: {:?}", l.name(), l.messages);
    }
    // The state's dimensions hold; the segmentation's transform is rescaled
    // into them (its translation of 2 × 8 nm becomes 4 × 4 nm).
    let dims = viewer.dimensions();
    assert_eq!(dims.names, ["x", "y", "z"]);
    assert_eq!(dims.scales, [4e-9, 4e-9, 4e-8]);
    assert_eq!(viewer.navigation().position(), &[20.5, 15.5, 10.5]);
    assert_eq!(viewer.layout, Layout::from_name(layout).unwrap());

    // Written back as neuroglancer would: the transform spec relative to the
    // source's own (as given), the other settings as they were, keys in
    // neuroglancer's order; restoring that gives the same state again, and
    // its share link reads back.
    let written = viewer.state_json();
    let keys: Vec<&str> = written.as_object().unwrap().keys().map(String::as_str).collect();
    // `projectionScale` too: once the bounds are known, neuroglancer derives
    // and writes the 3D zoom; `uiControlVisibility` is `{}` at its defaults
    // (as neuroglancer's JSON editor shows).
    assert_eq!(keys, ["dimensions", "position", "crossSectionScale", "projectionScale", "layers", "crossSectionBackgroundColor", "layout", "uiControlVisibility"]);
    let num = |v: &serde_json::Value| serde_json::from_str::<serde_json::Value>(&v.to_string().replace(".0,", ",").replace(".0]", "]")).unwrap();
    // The transform in the merged units (4 nm, so the translation of 2 × 8 nm
    // reads 4), as neuroglancer writes it (`combiner.mts`).
    assert_eq!(
        num(&written["layers"][1]["source"]["transform"]),
        num(&serde_json::json!({ "matrix": [[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0]],
            "outputDimensions": { "x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"] } }))
    );
    assert_eq!(written["layers"][0]["source"], state["layers"][0]["source"]);
    assert_eq!(written["layers"][1]["type"], "segmentation");
    assert_eq!(written["crossSectionBackgroundColor"], "#203040");
    assert_eq!(written["layout"], layout);
    assert_eq!(written["dimensions"], serde_json::json!({ "x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"] }));
    let fragment = viewer.state_fragment();
    match ng_core::json::parse_fragment(&fragment).unwrap() {
        ng_core::json::Fragment::State(v) => assert_eq!(v, written),
        f => panic!("{f:?}"),
    }
    let mut again = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
    again.open_state(&written).unwrap();
    assert_eq!(again.state_json(), written);

    // Off every slice panel while drawing: no segment is hovered.
    let park = viewer.layout.panels(W, H).into_iter().find(|(p, _)| *p == Panel::Perspective).map_or((W as f64 + 10.0, 0.0), |(_, r)| (r.x as f64 + 2.0, r.y as f64 + 2.0));
    viewer.pointer_move(park.0, park.1);
    render_loaded(&mut viewer, &view);
    let rgba = readback::texture_bytes(&device, &queue, &target);
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../target/ng-viewer-{stem}.png"));
    std::fs::write(&out, threers::encode_png(W, H, &rgba)).unwrap();

    let (checked, both, mismatched) = check_image_and_segmentation(&mut viewer, &rgba, [0x20, 0x30, 0x40], 0.8, 123, |id| {
        if [75, 106, 44].contains(&id) { 0.6 } else { 0.0 }
    });
    eprintln!("{stem}: {checked} pixels, {both} with both layers, {mismatched} off; {}", out.display());
    assert!(both >= 100, "too few pixels with both layers ({both}/{checked})");
    assert_eq!(mismatched, 0);
}

/// Checks every sampled pixel of a frame showing an image layer (0) and a
/// segmentation layer (1) against neuroglancer's composite of their values
/// under it: the image replaces, the segmentation blends over it
/// (`SRC_ALPHA, ONE_MINUS_SRC_ALPHA` on colour and alpha, in 8 bits) — also
/// where the image has nothing, since the blend goes by the layer's place in
/// the order — and alpha 0 shows the background. `seg_alpha` gives a
/// segment's alpha. Returns (pixels checked, with both layers, mismatched).
fn check_image_and_segmentation(
    viewer: &mut Viewer,
    rgba: &[u8],
    background: [u8; 3],
    image_opacity: f64,
    seed: u32,
    seg_alpha: impl Fn(u64) -> f64,
) -> (usize, usize, usize) {
    let bg = background.map(|c| c as f64);
    let q = |x: f64| (x * 255.0).round().clamp(0.0, 255.0);
    let (mut checked, mut both, mut mismatched) = (0, 0, 0);
    for (panel, r) in viewer.layout.panels(W, H) {
        let Panel::Slice(axes) = panel else { continue };
        for gy in 1..24 {
            for gx in 1..24 {
                let px = r.x + r.width * gx / 24;
                let py = r.y + r.height * gy / 24;
                if under_annotation(viewer, axes, r, px, py) {
                    continue;
                }
                viewer.pointer_move(px as f64 + 0.5, py as f64 + 0.5);
                let pos = viewer.mouse_position().unwrap();
                // Skip voxel faces of either layer.
                let near_face = (0..2).any(|l| match viewer.layer_voxel(l, &pos) {
                    Some(v) => v.iter().any(|c| (c - c.round()).abs() < 1e-3),
                    None => true,
                });
                if near_face {
                    continue;
                }
                let image = viewer.layer_value_at(0, &pos);
                let seg = viewer.layer_value_at(1, &pos);
                let mut dst = [0.0f64; 4];
                if let Some(v) = &image {
                    let t = viewer.layers()[0].data_type().unwrap();
                    let range = viewer.layers()[0].range().unwrap();
                    let g = ng_render::image::expected_gray(t, range, v.parse().unwrap());
                    dst = [q(g), q(g), q(g), q(image_opacity)];
                }
                if let Some(v) = &seg {
                    let id: u64 = v.parse().unwrap();
                    let (c, a) = if id == 0 {
                        ([0.0; 3], 0.0)
                    } else {
                        (SegmentColorHash::new(seed).compute(id).map(|c| c as f64), seg_alpha(id))
                    };
                    for ch in 0..3 {
                        dst[ch] = q(c[ch] * a + dst[ch] / 255.0 * (1.0 - a));
                    }
                    dst[3] = q(a * a + dst[3] / 255.0 * (1.0 - a));
                }
                let expected = if dst[3] == 0.0 { bg } else { [dst[0], dst[1], dst[2]] };
                if image.is_some() && seg.is_some() {
                    both += 1;
                }
                checked += 1;
                let i = ((py * W + px) * 4) as usize;
                let got = &rgba[i..i + 3];
                if (0..3).any(|c| (got[c] as f64 - expected[c]).abs() > 1.0) {
                    mismatched += 1;
                    eprintln!("{axes:?} ({px},{py}) pos {pos:?} image {image:?} seg {seg:?}: got {got:?}, expected {expected:?}");
                }
            }
        }
    }
    (checked, both, mismatched)
}

/// A FlyEM-style state over the network: hemibrain EM (sharded JPEG) under
/// the v1.2 segmentation (sharded compressed segmentation, uint64), at 8 nm.
/// Needs the network: `cargo test -p ng-viewer --test headless_render -- --ignored`.
#[test]
#[ignore]
fn hemibrain_state_over_gcs() {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer hemibrain state") else {
        return;
    };
    let state = serde_json::json!({
        "dimensions": { "x": [8e-9, "m"], "y": [8e-9, "m"], "z": [8e-9, "m"] },
        "position": [17216.5, 19776.5, 20704.5],
        // Not 1: pixel centres would all be on voxel faces (and skipped).
        "crossSectionScale": 0.9,
        "layers": [
            { "type": "image", "name": "emdata", "source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg" },
            { "type": "segmentation", "name": "segmentation", "source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation" }
        ],
        "layout": "4panel"
    });
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let target = device.create_texture(&wgpu::TextureDescriptor {
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
    let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(8)), Arc::new(|| {}));
    viewer.open_state(&state).unwrap();
    for l in viewer.layers() {
        assert!(l.messages.is_empty(), "{}: {:?}", l.name(), l.messages);
    }
    viewer.pointer_move(2.0, H as f64 - 2.0);
    render_loaded(&mut viewer, &view);
    let rgba = readback::texture_bytes(&device, &queue, &target);
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/ng-viewer-hemibrain-state.png");
    std::fs::write(&out, threers::encode_png(W, H, &rgba)).unwrap();
    let (checked, both, mismatched) = check_image_and_segmentation(&mut viewer, &rgba, [128, 128, 128], 0.5, 0, |_| 0.5);
    eprintln!("hemibrain state: {checked} pixels, {both} with both layers, {mismatched} off; {}", out.display());
    assert!(both >= 1000);
    assert_eq!(mismatched, 0);
}

/// A screenshot at twice the scale: the panels' area, twice as many pixels
/// each way, once its chunks load; a PNG of the same pixels as the view.
#[test]
fn screenshots_render_the_panels_scaled() {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer screenshot") else {
        return;
    };
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let state = serde_json::json!({
        "dimensions": { "x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"] },
        "position": [20.5, 15.5, 10.5],
        "crossSectionScale": 0.2,
        "layers": [{ "type": "image", "name": "em", "source": format!("precomputed://file://{fixtures}/image") }],
        "layout": "xy"
    });
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let target = device.create_texture(&wgpu::TextureDescriptor {
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
    let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
    viewer.open_state(&state).unwrap();
    render_loaded(&mut viewer, &view);
    let (_, _, w, h) = viewer.panel_viewport_union();
    viewer.start_screenshot(2.0, true, String::new());
    let start = std::time::Instant::now();
    let shot = loop {
        viewer.render(&view);
        if let Some(s) = viewer.step_screenshot("now") {
            break s;
        }
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        assert!(start.elapsed() < std::time::Duration::from_secs(30), "screenshot never finished");
    };
    assert!(!viewer.screenshot_active());
    assert_eq!((shot.width, shot.height), (2 * w, 2 * h));
    assert_eq!(shot.name, format!("neuroglancer-screenshot-w{}px-h{}px-at-now.png", 2 * w, 2 * h));
    let decoder = png::Decoder::new(std::io::Cursor::new(&shot.png));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!((info.width, info.height), (2 * w, 2 * h));
    // The image is there: not all one colour.
    let first = &buf[..4];
    assert!(buf.chunks_exact(4).any(|p| p != first));
}

/// A user shader: one written as the default is (but not its text) draws
/// what the built-in path draws, bit for bit; another draws its own
/// colours; one that does not compile reports why and draws the default.
#[test]
fn user_shaders_compile_and_draw() {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer user shaders") else {
        return;
    };
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let state = |shader: Option<&str>| {
        let mut layer = serde_json::json!({ "type": "image", "name": "em", "source": format!("precomputed://file://{fixtures}/image"),
            "opacity": 0.8, "shaderControls": { "normalized": { "range": [50, 200] } } });
        if let Some(s) = shader {
            layer["shader"] = s.into();
        }
        serde_json::json!({
            "dimensions": { "x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"] },
            "position": [20.5, 15.5, 10.5],
            "crossSectionScale": 0.2,
            "layers": [layer],
            "crossSectionBackgroundColor": "#203040",
            "layout": "xy"
        })
    };
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let target = device.create_texture(&wgpu::TextureDescriptor {
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
    let render = |shader: Option<&str>| {
        let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
        viewer.open_state(&state(shader)).unwrap();
        render_loaded(&mut viewer, &view);
        let error = viewer.layers()[0].shader_error().map(|e| e.message.clone());
        (readback::texture_bytes(&device, &queue, &target), error)
    };
    let (default, _) = render(None);
    let same = "#uicontrol invlerp normalized\n// the default, said again\nvoid main() {\n  emitGrayscale(normalized());\n}\n";
    let (custom, error) = render(Some(same));
    assert_eq!(error, None);
    assert!(default == custom, "the default as a user shader differs");
    let jet = "#uicontrol invlerp normalized\n#uicontrol float gain slider(min=0, max=2, default=1)\nvoid main() {\n  emitRGB(colormapJet(normalized() * gain));\n}\n";
    let (colored, error) = render(Some(jet));
    assert_eq!(error, None);
    assert!(colored != default);
    // Some pixel is coloured (not grey) where there is data.
    assert!(colored.chunks_exact(4).any(|p| p[0] != p[1] || p[1] != p[2]));
    // A `select` of strings: gray by default, as the default draws; "jet"
    // as the colormap shader draws.
    let select = "#uicontrol invlerp normalized\n#uicontrol string_t mode select(options=[\"gray\", \"jet\"])\nvoid main() {\n  if (mode == \"jet\") { emitRGB(colormapJet(normalized())); } else { emitGrayscale(normalized()); }\n}\n";
    let (gray, error) = render(Some(select));
    assert_eq!(error, None);
    assert!(gray == default, "the select's default");
    let jet_selected = {
        let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
        let mut s = state(Some(select));
        s["layers"][0]["shaderControls"]["mode"] = "jet".into();
        viewer.open_state(&s).unwrap();
        render_loaded(&mut viewer, &view);
        readback::texture_bytes(&device, &queue, &target)
    };
    assert!(jet_selected == colored, "the select's other option");
    // A transfer function from black to white, opaque: the grey of the raw
    // value, as the lookup table is a byte per step.
    let ramp = "#uicontrol transferFunction tf(controlPoints=[[0, \"#000000\", 1], [255, \"#ffffff\", 1]])\nvoid main() {\n  emitRGBA(tf());\n}\n";
    let (tf, error) = render(Some(ramp));
    assert_eq!(error, None);
    let grey = "void main() {\n  emitRGB(vec3(toNormalized(getDataValue())));\n}\n";
    let (grey, error) = render(Some(grey));
    assert_eq!(error, None);
    let worst = tf.iter().zip(&grey).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
    assert!(worst <= 1, "the transfer function's ramp off by {worst}");
    assert!(tf != default);
    let (broken, error) = render(Some("void main() { emitRGB(nothing); }\n"));
    assert!(error.is_some_and(|e| e.contains("nothing")), "a compile error");
    // The fallback: the default shader at its default range.
    let default = {
        let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
        let mut s = state(None);
        s["layers"][0].as_object_mut().unwrap().remove("shaderControls");
        viewer.open_state(&s).unwrap();
        render_loaded(&mut viewer, &view);
        readback::texture_bytes(&device, &queue, &target)
    };
    let diff: Vec<usize> = broken.chunks_exact(4).zip(default.chunks_exact(4)).enumerate().filter(|(_, (a, b))| a != b).map(|(i, _)| i).collect();
    assert!(diff.is_empty(), "the default drawn instead: {} differ, first at {:?}: {:?} vs {:?}", diff.len(), diff.first().map(|i| (i % W as usize, i / W as usize)), diff.first().map(|i| &broken[i * 4..i * 4 + 4]), diff.first().map(|i| &default[i * 4..i * 4 + 4]));
}

#[test]
fn segmentation_meshes_draw_in_3d() {
    mesh_render_check("seg_mesh", "mesh", 1.0);
    mesh_render_check("seg_multires", "mesh", 1.0);
    mesh_render_check("seg_multires_sharded", "mesh", 1.0);
    // Transparent: the order-independent pass.
    mesh_render_check("seg_mesh", "mesh", 0.5);
    mesh_render_check("seg_skeleton", "skeletons", 1.0);
    mesh_render_check("seg_skeleton", "skeletons", 0.5);
}

/// `fixture`'s `subsource` (meshes or skeletons) of segment 75, at
/// `alpha`, changes pixels of the 3D view to shades of the segment's
/// colour.
fn mesh_render_check(fixture: &str, subsource: &str, alpha: f64) {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer meshes") else {
        return;
    };
    let fixture = fixture.to_owned();
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let state = |meshes: bool| {
        serde_json::json!({
            "dimensions": { "x": [8e-9, "m"], "y": [8e-9, "m"], "z": [8e-8, "m"] },
            "position": [10.5, 7.5, 5.5],
            "projectionScale": 250,
            "projectionOrientation": [0.25, 0.2, 0.05, 0.9447],
            "layers": [{ "type": "segmentation", "name": "seg",
                "source": { "url": format!("precomputed://file://{fixtures}/{fixture}"), "subsources": { subsource: meshes } },
                "segments": ["75"], "colorSeed": 123, "objectAlpha": alpha }],
            "showAxisLines": false,
            "showDefaultAnnotations": false,
            "layout": "3d"
        })
    };
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let target = device.create_texture(&wgpu::TextureDescriptor {
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
    let render = |meshes: bool| {
        let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
        viewer.open_state(&state(meshes)).unwrap();
        render_loaded(&mut viewer, &view);
        readback::texture_bytes(&device, &queue, &target)
    };
    let with = render(true);
    if let Some(unsharded) = fixture.strip_suffix("_sharded") {
        // The same meshes as the unsharded fixture's.
        let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
        let mut s = state(true);
        s["layers"][0]["source"]["url"] = format!("precomputed://file://{fixtures}/{unsharded}").into();
        viewer.open_state(&s).unwrap();
        render_loaded(&mut viewer, &view);
        assert!(readback::texture_bytes(&device, &queue, &target) == with, "{fixture}: as {unsharded}");
    }
    let without = render(false);
    if alpha == 1.0 && !fixture.ends_with("_sharded") {
        // Picked in 3D (`completePickRequest`): over a pixel the object
        // covers, the layer's value is segment 75 and the position is on it.
        let k = with.chunks_exact(4).zip(without.chunks_exact(4)).position(|(a, b)| a != b).unwrap();
        let (x, y) = ((k as u32 % W) as f64 + 0.5, (k as u32 / W) as f64 + 0.5);
        let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
        viewer.open_state(&state(true)).unwrap();
        render_loaded(&mut viewer, &view);
        viewer.pointer_move(x, y);
        let mut position = None;
        for _ in 0..200 {
            viewer.render(&view);
            viewer.poll();
            position = viewer.mouse_position();
            if position.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let position = position.unwrap_or_else(|| panic!("{fixture}: a position under the pointer"));
        assert_eq!(viewer.layer_value_at(0, &position).as_deref(), Some("75"), "{fixture}: segment 75 picked");
        // Within the volume (20 × 15 × 10 voxels).
        assert!(position.iter().zip([20.0, 15.0, 10.0]).all(|(p, n)| (-0.5..n + 0.5).contains(p)), "{fixture}: {position:?}");
    }
    let changed: Vec<&[u8]> = with.chunks_exact(4).zip(without.chunks_exact(4)).filter(|(a, b)| a != b).map(|(a, _)| a).collect();
    let least = if subsource == "mesh" { 100 } else { 10 };
    assert!(changed.len() > least, "{fixture}: the {subsource} cover some pixels ({})", changed.len());
    // Each a lit shade of the segment's colour: `(|n·l| + 0.2) · colour`.
    let c = ng_core::SegmentColorHash::new(123).compute(75);
    let dominant = (0..3).max_by(|&a, &b| c[a].total_cmp(&c[b])).unwrap();
    assert!(changed.iter().all(|p| (0..3).all(|k| p[k] <= p[dominant])), "shades of segment 75's colour");
}


/// Volume rendering ("on", the default shader) draws the image in 3D:
/// grey, over what is drawn without it.
#[test]
fn images_are_volume_rendered() {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer volume rendering") else {
        return;
    };
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let state = |mode: &str| {
        serde_json::json!({
            "dimensions": { "x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"] },
            "position": [20.5, 15.5, 10.5],
            "projectionScale": 300,
            "projectionOrientation": [0.25, 0.2, 0.05, 0.9460443964212251],
            "layers": [{ "type": "image", "name": "em", "source": format!("precomputed://file://{fixtures}/image"), "volumeRendering": mode }],
            "showAxisLines": false,
            "showDefaultAnnotations": false,
            "showSlices": false,
            "layout": "3d"
        })
    };
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let target = device.create_texture(&wgpu::TextureDescriptor {
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
    let render = |mode: &str| {
        let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
        viewer.open_state(&state(mode)).unwrap();
        render_loaded(&mut viewer, &view);
        readback::texture_bytes(&device, &queue, &target)
    };
    let on = render("on");
    let off = render("off");
    let changed: Vec<&[u8]> = on.chunks_exact(4).zip(off.chunks_exact(4)).filter(|(a, b)| a != b).map(|(a, _)| a).collect();
    assert!(changed.len() > 1000, "the volume covers some pixels ({})", changed.len());
    // Grey: the default shader's `emitGrayscale`.
    assert!(changed.iter().all(|p| p[0].abs_diff(p[1]) <= 1 && p[1].abs_diff(p[2]) <= 1), "grey");
}

/// Milliseconds on a clock the test moves (`Viewer::set_clock` takes a `fn`).
static TEST_CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn test_clock() -> f64 {
    TEST_CLOCK.load(std::sync::atomic::Ordering::Relaxed) as f64
}

/// Stepping through z, a slice a frame: the chunks ahead are prefetched
/// (behind the visible ones); standing still, or with `prefetch` off, none.
#[test]
fn moving_through_slices_prefetches_the_chunks_ahead() {
    let Some(gpu) = threers::gpu::testing::context("ng-viewer prefetch") else {
        return;
    };
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let target = device.create_texture(&wgpu::TextureDescriptor {
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
    let run = |prefetch: bool, step: f32| {
        let mut viewer = Viewer::new(device.clone(), queue.clone(), FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
        viewer.set_clock(test_clock);
        // 40×30×20 voxels in 16×16×8 chunks: three chunks along z.
        let state = serde_json::json!({
            "dimensions": { "x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"] },
            "position": [20.5, 15.5, 1.5],
            "crossSectionScale": 0.2,
            "layers": [{ "type": "image", "name": "em", "source": format!("precomputed://file://{fixtures}/image") }],
            "layout": "xy",
            "prefetch": prefetch
        });
        viewer.open_state(&state).unwrap();
        let mut most = 0;
        for frame in 0..12 {
            TEST_CLOCK.fetch_add(16, std::sync::atomic::Ordering::Relaxed);
            let mut position = viewer.navigation().position().to_vec();
            position[2] = 1.5 + step * frame as f32;
            viewer.navigation_mut().set_position(&position);
            viewer.render(&view);
            viewer.poll();
            most = most.max(viewer.prefetching_chunks());
        }
        most
    };
    assert_eq!(run(true, 0.0), 0, "standing still");
    assert_eq!(run(false, 0.5), 0, "prefetch off");
    assert!(run(true, 0.5) > 0, "moving: nothing prefetched");
}
