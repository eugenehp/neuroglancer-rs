//! The image and segmentation shaders on the GPU, one voxel per pixel, against
//! the same arithmetic on the CPU (`expected_gray`) and neuroglancer's segment
//! colours (`ng_core::segment_color`, checked against neuroglancer itself).
//! Skips (passes) without a GPU.

use ng_core::segment_color::SegmentColorHash;
use ng_core::DataType;
use ng_render::image::{self, Interval};
use ng_render::segmentation::{self, SegmentationDisplay, VisibleSet};
use threers::gpu::{testing, GpuContext};
use threers::math::Matrix4;
use threers::renderer::{fullscreen, readback, PixelRect};
use threers::volume::{Data3DTexture, GpuVolume, SliceParams, SliceRenderer, SliceTarget, VolumeFormat, VolumeSample};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// NDC across a `width` × 1 target → voxel `(x, 0.5, 0.5)`.
fn row(width: usize) -> Matrix4 {
    let w = width as f32 / 2.0;
    Matrix4 {
        elements: [w, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, w, 0.5, 0.5, 1.0],
    }
}

/// Draws a `width` × 1 × 1 volume into a `width` × 1 target cleared to
/// transparent, replacing; returns the RGBA bytes. `set`: the lookup table
/// (segmentation's visible set).
#[allow(clippy::too_many_arguments)]
fn draw(
    gpu: &GpuContext,
    sample: VolumeSample,
    wgsl: &str,
    format: VolumeFormat,
    bytes: Vec<u8>,
    width: usize,
    params: SliceParams,
    set: Option<&VisibleSet>,
) -> Vec<u8> {
    let target = SliceTarget { format: FORMAT, blend: None, depth: None };
    let volume = GpuVolume::from_texture(&gpu.device, &gpu.queue, &Data3DTexture::new(width as u32, 1, 1, format, bytes));
    let (slices, binding) = match set {
        Some(set) => {
            let slices = SliceRenderer::with_lookup(&gpu.device, target, sample, wgsl);
            let binding = slices.create_binding_with_lookup(&gpu.device, &volume, &set.view);
            (slices, binding)
        }
        None => {
            let slices = SliceRenderer::with_target(&gpu.device, target, sample, wgsl);
            let binding = slices.create_binding(&gpu.device, &volume);
            (slices, binding)
        }
    };
    slices.update(&gpu.queue, &binding, &params);
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: width as u32, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    drop(fullscreen::begin(&mut encoder, &view, fullscreen::Load::Clear));
    slices.draw(&mut encoder, &view, PixelRect { x: 0, y: 0, width: width as u32, height: 1 }, &binding);
    gpu.queue.submit([encoder.finish()]);
    readback::texture_bytes(&gpu.device, &gpu.queue, &texture)
}

/// Renders `values` (little-endian bytes of `t`) with `range` and checks each
/// pixel's grey against `expected_gray`, and the alpha against the opacity.
fn check_image(gpu: &GpuContext, t: DataType, bytes: &[u8], values: &[f64], range: Interval) {
    let (sample, wgsl) = image::color_wgsl(t);
    let n = values.len();
    let params = image::slice_params(row(n), t, range, 0.5);
    let rgba = draw(gpu, sample, &wgsl, image::volume_format(t), image::upload_bytes(t, bytes), n, params, None);
    for (i, &v) in values.iter().enumerate() {
        let want = (image::expected_gray(t, range, v) * 255.0).round() as i32;
        let got = &rgba[i * 4..i * 4 + 4];
        assert!(
            (got[0] as i32 - want).abs() <= 1 && got[0] == got[1] && got[1] == got[2] && got[3].abs_diff(128) <= 1,
            "{t:?} {range:?} value {v}: got {got:?}, want {want}"
        );
    }
}

#[test]
fn image_invlerp_matches_neuroglancers_arithmetic() {
    let Some(gpu) = testing::context("ng-render image shading") else { return };

    // float32, with NaN and infinities: NaN shows as 0 (GLSL's clamp).
    let floats = [0.0f32, 0.25, 0.5, 1.0, -3.0, 7.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -f32::NAN];
    let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    let values: Vec<f64> = floats.iter().map(|&f| f as f64).collect();
    check_image(&gpu, DataType::Float32, &bytes, &values, image::default_data_type_range(DataType::Float32));
    // An empty range: `0 · ∞` is NaN at the bound itself, ±∞ elsewhere.
    check_image(&gpu, DataType::Float32, &bytes, &values, Interval::Number([0.5, 0.5]));
    // A reversed range.
    check_image(&gpu, DataType::Float32, &bytes, &values, Interval::Number([1.0, 0.0]));

    let u8s: Vec<u8> = vec![0, 1, 127, 128, 254, 255];
    check_image(&gpu, DataType::Uint8, &u8s, &u8s.iter().map(|&v| v as f64).collect::<Vec<_>>(), image::default_data_type_range(DataType::Uint8));
    check_image(&gpu, DataType::Uint8, &u8s, &u8s.iter().map(|&v| v as f64).collect::<Vec<_>>(), Interval::Number([128.0, 128.0]));
    let i16s = [i16::MIN, -1, 0, 1, 1000, i16::MAX];
    check_image(
        &gpu,
        DataType::Int16,
        &i16s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        &i16s.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        image::default_data_type_range(DataType::Int16),
    );

    // 32-bit: exact integer offsets, shifted.
    let u32s = [0u32, 1, 255, 1 << 24, (1 << 24) + 1, 3_000_000_000, u32::MAX - 1, u32::MAX];
    let u32_bytes: Vec<u8> = u32s.iter().flat_map(|v| v.to_le_bytes()).collect();
    let u32_values: Vec<f64> = u32s.iter().map(|&v| v as f64).collect();
    for range in [
        image::default_data_type_range(DataType::Uint32),
        Interval::Number([16777216.0, 16777217.0]),
        Interval::Number([4294967295.0, 0.0]),
        Interval::Number([7.0, 7.0]),
    ] {
        check_image(&gpu, DataType::Uint32, &u32_bytes, &u32_values, range);
    }
    let i32s = [i32::MIN, i32::MIN + 1, -1, 0, 1, 1 << 30, i32::MAX];
    let i32_bytes: Vec<u8> = i32s.iter().flat_map(|v| v.to_le_bytes()).collect();
    let i32_values: Vec<f64> = i32s.iter().map(|&v| v as f64).collect();
    for range in [image::default_data_type_range(DataType::Int32), Interval::Number([-5.0, 5.0])] {
        check_image(&gpu, DataType::Int32, &i32_bytes, &i32_values, range);
    }

    // uint64: both words matter.
    let u64s = [0u64, 1, 1 << 40, (1 << 40) + (1 << 39), u64::MAX / 2, u64::MAX - (1 << 41), u64::MAX];
    let u64_bytes: Vec<u8> = u64s.iter().flat_map(|v| v.to_le_bytes()).collect();
    let u64_values: Vec<f64> = u64s.iter().map(|&v| v as f64).collect();
    for range in [
        image::default_data_type_range(DataType::Uint64),
        Interval::Uint64([1 << 40, 1 << 41]),
        Interval::Uint64([u64::MAX, 0]),
        Interval::Uint64([5, 5]),
    ] {
        check_image(&gpu, DataType::Uint64, &u64_bytes, &u64_values, range);
    }
}

/// GLSL's `mix(x, y, a)`.
fn mix(x: f32, y: f32, a: f32) -> f32 {
    x * (1.0 - a) + y * a
}

#[test]
fn segment_colors_match_neuroglancer() {
    let Some(gpu) = testing::context("ng-render segmentation shading") else { return };
    let ids: Vec<u64> = vec![0, 1, 2, 3, 255, 1 << 32, (1 << 32) + 7, 0xdead_beef_cafe_babe, u64::MAX, 864691135, 42];
    let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let n = ids.len();
    for seed in [0u32, 1, 0x7fc0_0001, u32::MAX] {
        for hovered in [None, Some(42u64)] {
            let display = SegmentationDisplay {
                color_seed: seed,
                ..Default::default()
            };
            let params = display.slice_params(row(n), hovered, 0);
            let empty = VisibleSet::new(&gpu.device);
            let rgba = draw(
                &gpu,
                segmentation::sample(),
                segmentation::SEGMENTATION_WGSL,
                segmentation::volume_format(DataType::Uint64),
                segmentation::upload_bytes(DataType::Uint64, &bytes),
                n,
                params,
                Some(&empty),
            );
            let hash = SegmentColorHash::new(seed);
            for (i, &id) in ids.iter().enumerate() {
                let got = &rgba[i * 4..i * 4 + 4];
                if id == 0 {
                    assert_eq!(got, [0, 0, 0, 0], "segment 0 is hidden");
                    continue;
                }
                // Hovering a visible segment lightens it: saturation 1 → 0.5.
                let saturation = if hovered == Some(id) { 0.5 } else { 1.0 };
                let rgb = hash.compute(id).map(|c| mix(1.0, c, saturation));
                for c in 0..3 {
                    let want = (rgb[c] * 255.0).round() as i32;
                    assert!((got[c] as i32 - want).abs() <= 1, "seed {seed} id {id} hovered {hovered:?}: got {got:?}, want {rgb:?}");
                }
                assert!(got[3].abs_diff(128) <= 1, "alpha: {got:?}");
            }
        }
    }
}

#[test]
fn small_segmentation_types_widen_to_u32() {
    let Some(gpu) = testing::context("ng-render segmentation u16") else { return };
    let ids: [u16; 4] = [0, 1, 300, u16::MAX];
    let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let display = SegmentationDisplay::default();
    let rgba = draw(
        &gpu,
        segmentation::sample(),
        segmentation::SEGMENTATION_WGSL,
        segmentation::volume_format(DataType::Uint16),
        segmentation::upload_bytes(DataType::Uint16, &bytes),
        4,
        display.slice_params(row(4), None, 0),
        Some(&VisibleSet::new(&gpu.device)),
    );
    let hash = SegmentColorHash::default();
    for (i, &id) in ids.iter().enumerate().skip(1) {
        let rgb = hash.compute(id as u64);
        for c in 0..3 {
            assert!((rgba[i * 4 + c] as i32 - (rgb[c] * 255.0).round() as i32).abs() <= 1, "id {id}");
        }
    }
}

/// The invlerp uniforms equal what neuroglancer's `enableLerpShaderFunction`
/// sets (`scripts/golden/lerp_params.mts`, from a mock GL).
#[test]
fn lerp_uniforms_match_neuroglancer() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/lerp_params.json");
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let num = |v: &serde_json::Value| -> f32 {
        match v {
            serde_json::Value::String(s) => s.parse::<f32>().unwrap_or_else(|_| if s == "-Infinity" { f32::NEG_INFINITY } else { f32::INFINITY }),
            v => v.as_f64().unwrap() as f32,
        }
    };
    for case in cases.as_array().unwrap() {
        let t = DataType::from_name(&case["type"].as_str().unwrap().to_lowercase()).unwrap();
        let iv: Vec<&str> = case["interval"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let range = if t == DataType::Uint64 {
            Interval::Uint64([iv[0].parse().unwrap(), iv[1].parse().unwrap()])
        } else {
            Interval::Number([iv[0].parse().unwrap(), iv[1].parse().unwrap()])
        };
        let [p0, p1] = image::lerp_params(t, range);
        let offset = p1[0] as u64 | (p1[1] as u64) << 16 | (p1[2] as u64) << 32 | (p1[3] as u64) << 48;
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits();
        if let Some(params) = case.get("params") {
            let want = [num(&params[0]), num(&params[1])];
            assert!(same(p0[0], want[0]) && same(p0[1], want[1]), "{case}: got {p0:?}");
        } else {
            let bounds: Vec<u64> = case["bounds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            let scalar = num(&case["scalar"]);
            let (want_offset, want_shift) = match bounds.len() {
                2 => (bounds[0], bounds[1]),
                _ => (bounds[0] | bounds[1] << 32, bounds[2]),
            };
            assert_eq!(offset, want_offset, "{case}: offset");
            assert_eq!(p0[3] as u64, want_shift, "{case}: shift");
            assert!(same(p0[1], scalar), "{case}: scalar {} vs {scalar}", p0[1]);
        }
    }
}

/// Every shader this crate makes validates and translates to WebGL2 GLSL, as
/// the browser fallback needs.
#[test]
fn shaders_translate_for_webgl2() {
    let mut modules: Vec<(String, String)> = vec![];
    for t in [DataType::Uint8, DataType::Float32, DataType::Uint32, DataType::Int32, DataType::Uint64] {
        let (sample, wgsl) = image::color_wgsl(t);
        modules.push((format!("image {t:?}"), SliceRenderer::shader_source(sample, &wgsl)));
    }
    modules.push(("segmentation".into(), SliceRenderer::shader_source_with_lookup(segmentation::sample(), segmentation::SEGMENTATION_WGSL)));
    modules.push(("composite".into(), ng_render::compose::composite_shader_source()));
    for (name, source) in modules {
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&source)));
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for (stage, entry) in [(naga::ShaderStage::Vertex, "vs_main"), (naga::ShaderStage::Fragment, "fs")] {
            let options = naga::back::glsl::Options {
                version: naga::back::glsl::Version::Embedded { version: 300, is_webgl: true },
                ..Default::default()
            };
            let pipeline = naga::back::glsl::PipelineOptions { shader_stage: stage, entry_point: entry.to_string(), multiview: None };
            let mut out = String::new();
            naga::back::glsl::Writer::new(&mut out, &module, &info, &options, &pipeline, Default::default())
                .and_then(|mut w| w.write().map(|_| ()))
                .unwrap_or_else(|e| panic!("{name} {entry}: {e}"));
        }
    }
}

/// With visible segments listed, those show at `selectedAlpha` and the rest
/// at `notSelectedAlpha`; hovering a hidden one shows it at `selectedAlpha`,
/// lightened by 0.75 rather than 0.5 (saturation 1 → 0.25, not 0.5). The set spans more than one texture
/// row.
#[test]
fn visible_segment_sets_pick_the_alpha() {
    let Some(gpu) = testing::context("ng-render visible set") else { return };
    let ids: Vec<u64> = vec![5, 6, 7, 1 << 33, (1 << 33) + 1, 2000, 2001, u64::MAX];
    let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    // Visible: 6, 2^33, 2001, u64::MAX, and 1500 others (two rows of 1024).
    let mut visible: Vec<u64> = vec![6, 1 << 33, 2001, u64::MAX];
    visible.extend((10_000..11_500).map(|i| i * 3));
    let mut set = VisibleSet::new(&gpu.device);
    set.set(&gpu.device, &gpu.queue, &visible);
    assert_eq!(set.len(), visible.len());
    let display = SegmentationDisplay { not_selected_alpha: 0.25, ..Default::default() };
    for hovered in [None, Some(7u64), Some(6u64)] {
        let rgba = draw(
            &gpu,
            segmentation::sample(),
            segmentation::SEGMENTATION_WGSL,
            segmentation::volume_format(DataType::Uint64),
            segmentation::upload_bytes(DataType::Uint64, &bytes),
            ids.len(),
            display.slice_params(row(ids.len()), hovered, set.len()),
            Some(&set),
        );
        let hash = SegmentColorHash::default();
        for (i, &id) in ids.iter().enumerate() {
            let has = visible.contains(&id);
            let got = &rgba[i * 4..i * 4 + 4];
            let (alpha, saturation) = match (hovered == Some(id), has) {
                (true, true) => (0.5, 0.5),
                (true, false) => (0.5, 0.25),
                (false, true) => (0.5, 1.0),
                (false, false) => (0.25, 1.0),
            };
            let rgb = hash.compute(id).map(|c| mix(1.0, c, saturation));
            for c in 0..3 {
                let want = (rgb[c] * 255.0).round() as i32;
                assert!((got[c] as i32 - want).abs() <= 1, "id {id} hovered {hovered:?}: got {got:?}, want {rgb:?}");
            }
            assert!((got[3] as f32 - alpha * 255.0).abs() <= 1.0, "id {id} hovered {hovered:?}: alpha {got:?}, want {alpha}");
        }
    }
}

/// An annotation shader run over two annotations on the GPU: colours from
/// a property, a size from another, a discard, the controls, and what the
/// shader leaves unset at the render helpers' starting values.
#[test]
fn annotation_shader_sets_styles() {
    use ng_render::annotation_shader::{compile, control_values, AnnotationShaderPipeline, PropertyShaderType as P};
    let Some(gpu) = testing::context("ng-render annotation shader") else { return };
    let code = "#uicontrol float scale slider(min=1, max=10, default=2)\nvoid main() {\n  setColor(prop_color());\n  setPointMarkerSize(prop_size() * scale);\n  setEndpointMarkerColor(vec3(1.0, 0.0, 0.0), defaultColor());\n  if (prop_size() > 5.0) discard;\n}\n";
    let parsed = ng_render::user_shader::parse(code, DataType::Float32);
    let values = ng_render::user_shader::control_values(&parsed, DataType::Float32, None);
    let props = [("color".to_owned(), P::Rgb), ("size".to_owned(), P::Float)];
    let compiled = compile(&parsed, &values, &props, true).unwrap();
    let pipeline = AnnotationShaderPipeline::new(&gpu.device, &compiled);
    let bits = [P::Rgb.bits((255 | 128 << 8) as f64), P::Float.bits(1.5), P::Rgb.bits(0.0), P::Float.bits(6.0)];
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let readback = pipeline.evaluate(&gpu.device, &mut encoder, &control_values(&parsed, &values), [0.0, 0.0, 1.0], &bits, 2);
    gpu.queue.submit([encoder.finish()]);
    readback.map(std::sync::Arc::new(|| {}));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let styles = readback.read().expect("mapped");
    let s = &styles[0];
    assert_eq!(s.point_color, [1.0, 128.0 / 255.0, 0.0, 1.0]);
    assert_eq!(s.point_size, 3.0);
    assert_eq!(s.marker_color, [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]);
    assert_eq!((s.point_border_color, s.point_border_width, s.line_width), ([0.0, 0.0, 0.0, 1.0], 1.0, 1.0));
    assert_eq!(s.ellipsoid_fill_color[3], 1.0);
    assert!(!s.discard && styles[1].discard);
}
