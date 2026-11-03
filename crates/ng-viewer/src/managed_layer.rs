//! A layer as the viewer manages it: its data sources as they load, what
//! draws it, and its annotations.

use super::*;

/// How a layer is shaded: the user layer type (`type` in the state, or as
/// `detectLayerType` picks it: segmentation volumes get a segmentation layer,
/// the rest an image layer).
#[derive(Debug, Clone, PartialEq)]
pub enum LayerKind {
    Image { opacity: f32 },
    Segmentation(SegmentationDisplay),
}

/// What draws a layer: its volume, shading and textures.
pub(crate) struct LayerState {
    /// The data source it draws (the first with a volume).
    pub(crate) source_index: usize,
    pub(crate) layer: VolumeLayer,
    pub(crate) kind: LayerKind,
    pub(crate) pipelines: LayerPipelines,
    pub(crate) textures: HashMap<TextureKey, ChunkTexture>,
    /// Segmentation: the visible segments (`visibleSegments`), and the set's
    /// generation the textures' bindings were made with.
    pub(crate) visible: Option<(VisibleSet, u64)>,
    /// Changes whenever the pipelines or the textures' bindings are made
    /// again (a shader compiled, a lookup replaced): what was drawn with the
    /// old ones is out of date. Unique across states (`next_epoch`).
    pub(crate) epoch: u64,
    /// An image's own shader (not the default), compiled.
    pub(crate) shader: Option<UserShader>,
    /// Its shader compiled for volume rendering, while that is on.
    pub(crate) volume: Option<VolumeShader>,
}

/// An image's shader compiled for volume rendering: what it was compiled
/// from, its controls, the pipeline and the controls' texture.
pub(crate) struct VolumeShader {
    key: (String, Vec<bool>, Vec<Vec<u32>>, ng_render::user_shader::VolumeMode),
    pub(crate) parsed: ng_render::user_shader::ParsedShader,
    pub(crate) values: Vec<ng_render::user_shader::ControlValue>,
    pub(crate) pipeline: ng_render::perspective::VolumePipeline,
    pub(crate) uniforms: ng_render::user_shader::UniformTexture,
}

/// A compiled user shader: what it was compiled from (the code and its
/// checkboxes, which are compiled in), its controls and their values, and
/// the texture the values go to.
pub(crate) struct UserShader {
    pub(crate) key: (String, Vec<bool>, Vec<Vec<u32>>),
    pub(crate) parsed: ng_render::user_shader::ParsedShader,
    pub(crate) values: Vec<ng_render::user_shader::ControlValue>,
    pub(crate) uniforms: ng_render::user_shader::UniformTexture,
}

impl LayerState {
    /// An image layer's shader brought up to date with its JSON: compiled
    /// again when its code or a checkbox changes; the default built in; on
    /// an error the default too (`fallbackShaderParameters`), and the error.
    pub(crate) fn update_shader(&mut self, device: &wgpu::Device, json: &serde_json::Map<String, Value>) -> Option<ng_render::user_shader::CompileError> {
        use ng_render::user_shader as us;
        if !matches!(self.kind, LayerKind::Image { .. }) {
            return None;
        }
        let t = self.layer.data_type;
        let text = json.get("shader").and_then(Value::as_str).unwrap_or(state::DEFAULT_IMAGE_SHADER);
        let extents = self.layer.channel_extents();
        // With channel dimensions the chunks hold every channel (stacked),
        // which only the compiled shaders read: the default one too.
        let default_pipelines = |device: &wgpu::Device| {
            if !extents.is_empty() {
                let parsed = us::parse_with_channels(state::DEFAULT_IMAGE_SHADER, t, extents.len());
                let values = us::control_values(&parsed, t, None);
                if let Ok(compiled) = us::compile_with_channels(t, &parsed, &values, &extents) {
                    return LayerPipelines::new(device, compiled.sample, &compiled.wgsl, true);
                }
            }
            let (sample, color) = ng_render::image::color_wgsl(t);
            LayerPipelines::new(device, sample, &color, false)
        };
        if text == state::DEFAULT_IMAGE_SHADER && extents.is_empty() {
            if self.shader.take().is_some() {
                self.pipelines = default_pipelines(device);
                self.clear_bindings();
            }
            return None;
        }
        let parsed = us::parse_with_channels(text, t, extents.len());
        let values = us::control_values(&parsed, t, json.get("shaderControls"));
        let (checks, channels) = us::compiled_values(&values);
        let key = (text.to_owned(), checks, channels);
        if let Some(s) = &mut self.shader {
            if s.key == key {
                s.values = values;
                return None;
            }
        }
        match us::compile_with_channels(t, &parsed, &values, &extents) {
            Ok(compiled) => {
                self.pipelines = LayerPipelines::new(device, compiled.sample, &compiled.wgsl, true);
                let uniforms = us::UniformTexture::new(device, compiled.texels);
                self.shader = Some(UserShader { key, parsed, values, uniforms });
                self.clear_bindings();
                None
            }
            Err(e) => {
                if self.shader.take().is_some() {
                    self.pipelines = default_pipelines(device);
                    self.clear_bindings();
                }
                Some(e)
            }
        }
    }

    /// The shader compiled for volume rendering (the layer's, or the
    /// default), recompiled when the code, its checkboxes or channels
    /// change; `None` if it does not compile.
    pub(crate) fn volume_shader(&mut self, device: &wgpu::Device, renderer: &ng_render::perspective::PerspectiveRenderer, json: &serde_json::Map<String, Value>, mode: ng_render::user_shader::VolumeMode) -> Option<&VolumeShader> {
        use ng_render::user_shader as us;
        let t = self.layer.data_type;
        let extents = self.layer.channel_extents();
        let text = json.get("shader").and_then(Value::as_str).unwrap_or(state::DEFAULT_IMAGE_SHADER);
        let parsed = us::parse_with_channels(text, t, extents.len());
        let values = us::control_values(&parsed, t, json.get("shaderControls"));
        let (checks, channels) = us::compiled_values(&values);
        let key = (text.to_owned(), checks, channels, mode);
        if !matches!(&self.volume, Some(v) if v.key == key) {
            self.volume = us::compile_volume_rendering_mode(t, &parsed, &values, &extents, mode).ok().map(|compiled| VolumeShader {
                key,
                pipeline: ng_render::perspective::VolumePipeline::new(device, renderer, &compiled, mode),
                uniforms: us::UniformTexture::new(device, compiled.texels),
                parsed: parsed.clone(),
                values: values.clone(),
            });
        }
        let v = self.volume.as_mut()?;
        v.parsed = parsed;
        v.values = values;
        Some(v)
    }

    fn clear_bindings(&mut self) {
        for t in self.textures.values_mut() {
            t.bindings = Default::default();
        }
        self.epoch = next_epoch();
    }
}

/// A value no layer state's `epoch` had before.
pub(crate) fn next_epoch() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl LayerState {
    pub(crate) fn upload_format(&self) -> (VolumeFormat, UploadFn) {
        match self.kind {
            LayerKind::Image { .. } => (ng_render::image::volume_format(self.layer.data_type), ng_render::image::upload_bytes),
            LayerKind::Segmentation(_) => (
                ng_render::segmentation::volume_format(self.layer.data_type),
                ng_render::segmentation::upload_bytes,
            ),
        }
    }

    pub(crate) fn slice_params(&self, ndc_to_voxel: Matrix4, hovered: Option<u64>) -> SliceParams {
        match &self.kind {
            LayerKind::Image { opacity } => {
                ng_render::image::slice_params(ndc_to_voxel, self.layer.data_type, self.layer.range, *opacity)
            }
            LayerKind::Segmentation(display) => {
                display.slice_params(ndc_to_voxel, hovered, self.visible.as_ref().map_or(0, |(set, _)| set.len()))
            }
        }
    }
}

/// A data source of a layer, as it loads.
pub(crate) enum SourceLoad {
    Pending,
    Failed,
    Loaded(Box<LoadedSource>),
}

pub(crate) struct LoadedSource {
    pub(crate) source: DataSource,
    /// The source's transform with the state's spec over it; the global
    /// combiner rescales it.
    pub(crate) transform: CoordinateSpaceTransform,
    pub(crate) binding: Binding,
}

pub(crate) struct SourceSlot {
    pub(crate) spec: SourceSpec,
    pub(crate) load: SourceLoad,
    /// Its URL was typed in (`setManually`).
    pub(crate) set_manually: bool,
}

/// A data source of a layer, for display.
#[derive(Debug, Clone)]
pub struct SourceView {
    pub url: String,
    pub state: SourceState,
    pub enable_default_subsources: bool,
    pub subsources: Vec<SubsourceView>,
    /// The source's transform, as the global space rescaled it.
    pub transform: Option<CoordinateSpaceTransform>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceState {
    Loading,
    Failed,
    Loaded,
}

/// A subsource: its ID (empty for `default`), what it is, whether on.
#[derive(Debug, Clone)]
pub struct SubsourceView {
    pub id: String,
    pub description: String,
    pub enabled: bool,
}

/// A segmentation layer's listed segments: ID, and whether it is shown.
pub(crate) type SegmentList = Arc<[(u64, bool)]>;

/// A layer (`ManagedUserLayer`): its state, its data sources, and what draws
/// it once they load.
pub struct ManagedLayer {
    pub id: LayerId,
    pub spec: LayerSpec,
    pub(crate) sources: Vec<SourceSlot>,
    pub(crate) render: Option<LayerState>,
    /// Problems to show: failed sources, unsupported settings.
    pub messages: Vec<String>,
    /// (needed, available) visible chunks, last frame.
    pub(crate) progress: (u32, u32),
    /// `sliceViewRenderScaleHistogram`: the slice views' chunks last frame.
    pub(crate) render_scale_histogram: ng_core::render_scale::RenderScaleHistogram,
    /// The meshes' (`displayState.renderScaleHistogram`).
    pub(crate) mesh_render_scale_histogram: ng_core::render_scale::RenderScaleHistogram,
    /// An annotation layer's spatial index spacing, in cross-sections and
    /// in 3D (`annotationCrossSectionRenderScaleHistogram`, …).
    pub(crate) annotation_histograms: [ng_core::render_scale::RenderScaleHistogram; 2],
    /// Volume rendering's (`volumeRenderingChunkResolutionHistogram`).
    pub(crate) volume_rendering_histogram: ng_core::render_scale::RenderScaleHistogram,
    /// The chunks volume rendering drew last frame (for its data histogram
    /// when there is no slice view): data, chunk display dimensions, fixed
    /// offsets.
    pub(crate) volume_chunks: Vec<VolumeChunk>,
    /// The `normalized` control's data histogram, while its CDF panel shows
    /// (`dataHistogramSpecifications`): wanted, and last frame's counts.
    pub(crate) histogram_wanted: bool,
    pub(crate) data_histogram: Option<Box<[u32; 256]>>,
    /// The values the histogram counted (for `AutoRangeFinder`).
    pub(crate) histogram_samples: Vec<f64>,
    /// Other invlerp controls' windows to count over (a user shader's), and
    /// last frame's counts over each.
    pub(crate) histogram_windows: Vec<[f64; 2]>,
    pub(crate) window_histograms: Vec<([f64; 2], Box<[u32; 256]>)>,
    /// A user shader's legends: asked for (by their values), being drawn,
    /// and drawn (each with what it was drawn for).
    pub(crate) legend_wants: Vec<Vec<f64>>,
    pub(crate) legend_jobs: Vec<legend::LegendJob>,
    pub(crate) legends: Vec<(legend::LegendKey, Vec<[u8; 4]>)>,
    /// Why the layer's shader did not compile (it shows the default).
    pub(crate) shader_error: Option<ng_render::user_shader::CompileError>,
    /// An annotation layer's local annotations, placed by their source's
    /// transform.
    pub(crate) annotation: Option<AnnotationLayerState>,
    /// An annotation layer's shader, compiled for cross-sections and 3D.
    pub(crate) annotation_shader: Option<AnnotationShaderCache>,
    /// Drawn only at this cross-section scale or finer (global units per
    /// pixel; not neuroglancer's): a full-resolution source whose coarsest
    /// level is still far finer than a whole-brain view would otherwise
    /// download hundreds of megabytes for it, where a coarse copy of the same
    /// data shows the view in a few. `spec.visible` stays the reader's.
    pub max_cross_section_scale: Option<f64>,
    /// Hidden by `max_cross_section_scale` at the scale now.
    pub(crate) zoom_hidden: bool,
    /// The segments listed in `segments` (id, shown) as last parsed, and a
    /// hash of the JSON they were parsed from.
    pub(crate) segments_memo: std::sync::Mutex<Option<(u64, SegmentList)>>,
}

/// An annotation layer's shader as last compiled: from what code and
/// properties, its controls' values, and a pipeline per view (`None` for
/// the default shader, whose styles need no GPU).
pub(crate) struct AnnotationShaderCache {
    pub(crate) code: String,
    pub(crate) properties: Vec<(String, ng_render::annotation_shader::PropertyShaderType)>,
    pub(crate) controls: Vec<f32>,
    pub(crate) pipelines: Option<[ng_render::annotation_shader::AnnotationShaderPipeline; 2]>,
}

/// An annotation as the Annotations tab lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnotationListEntry {
    pub id: String,
    pub icon: &'static str,
    pub rows: Vec<Vec<Option<f64>>>,
    pub description: Option<String>,
}

/// An annotation as the Selection panel shows it: as listed, whether its
/// source is read-only, and its properties (id, type, value).
#[derive(Debug, Clone, PartialEq)]
pub struct AnnotationDetails {
    pub entry: AnnotationListEntry,
    pub readonly: bool,
    pub properties: Vec<(String, ng_datasource::annotation::PropertyType, f64)>,
}

/// An annotation layer's local annotations and where they are.
pub struct AnnotationLayerState {
    pub source_index: usize,
    pub layer: VolumeLayer,
    pub annotations: Vec<annotations::Annotation>,
    /// A precomputed source's metadata: its annotations come from the
    /// cells of its spatial index.
    pub spatial: Option<Arc<ng_datasource::annotation::AnnotationInfo>>,
}

impl ManagedLayer {
    pub fn name(&self) -> &str {
        &self.spec.name
    }

    /// Drawn: the reader shows it (`spec.visible`) and its zoom gate
    /// (`max_cross_section_scale`) lets it.
    pub fn shown(&self) -> bool {
        self.spec.visible && !self.zoom_hidden
    }

    /// The `segments` list (id, shown), parsed once per change: hosts and
    /// each frame's meshes and skeletons ask for it many times.
    pub(crate) fn segments(&self) -> SegmentList {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        match self.spec.json.get("segments") {
            Some(Value::Array(a)) => {
                for v in a {
                    match v {
                        Value::String(s) => s.hash(&mut h),
                        other => other.to_string().hash(&mut h),
                    }
                }
            }
            Some(other) => other.to_string().hash(&mut h),
            None => 0u8.hash(&mut h),
        }
        let key = h.finish();
        let mut memo = self.segments_memo.lock().unwrap();
        if let Some((k, list)) = memo.as_ref() {
            if *k == key {
                return list.clone();
            }
        }
        let list: SegmentList = state::segmentation_settings(&self.spec.json).map(|(_, s)| s.into()).unwrap_or_else(|_| Arc::from(vec![]));
        *memo = Some((key, list.clone()));
        list
    }

    /// The settings neuroglancer applies in place (`set_layer_option`
    /// without a reload) as the layer's JSON holds them: `opacity`,
    /// `shader`, `shaderControls`, `crossSectionRenderScale`,
    /// `volumeRendering`, `volumeRenderingDepthSamples` (`null` where unset).
    pub fn live_settings(&self) -> serde_json::Map<String, Value> {
        crate::LIVE_LAYER_OPTIONS.iter().map(|&k| (k.to_owned(), self.spec.json.get(k).cloned().unwrap_or(Value::Null))).collect()
    }

    /// Why the layer's shader did not compile, if it did not.
    pub fn shader_error(&self) -> Option<&ng_render::user_shader::CompileError> {
        self.shader_error.as_ref()
    }

    /// Whether every data source has loaded or failed.
    pub fn is_ready(&self) -> bool {
        self.sources.iter().all(|s| !matches!(s.load, SourceLoad::Pending))
    }

    pub fn kind(&self) -> Option<&LayerKind> {
        self.render.as_ref().map(|r| &r.kind)
    }

    pub fn kind_mut(&mut self) -> Option<&mut LayerKind> {
        self.render.as_mut().map(|r| &mut r.kind)
    }

    pub fn data_type(&self) -> Option<DataType> {
        self.render.as_ref().map(|r| r.layer.data_type)
    }

    pub fn range(&self) -> Option<ng_render::image::Interval> {
        self.render.as_ref().map(|r| r.layer.range)
    }

    /// `annotationColor` (`UserLayerWithAnnotationsMixin`), yellow by default.
    pub fn annotation_color(&self) -> [f32; 3] {
        self.spec.json.get("annotationColor").and_then(Value::as_str).and_then(state::parse_color).unwrap_or([1.0, 1.0, 0.0])
    }

    /// `layerBarColors`: the colours a layer's chip in the layer bar shows,
    /// sRGB. Empty for none; `None` where neuroglancer shows none either
    /// (every segment in its own colour, or too many colours). Only
    /// segmentation layers have them (`supportsLayerBarColorSyncOption`), and
    /// only while visible.
    pub fn layer_bar_colors(&self) -> Option<Vec<[f32; 3]>> {
        const MAX_LAYER_BAR_UI_INDICATOR_COLORS: usize = 6;
        // Annotation layers (`automaticLayerBarColors`): the annotation
        // colour while the shader uses `defaultColor` (the default does).
        if self.spec.layer_type == "annotation" {
            let shader = self.spec.json.get("shader").and_then(Value::as_str).unwrap_or("defaultColor");
            return Some(if self.spec.visible && shader.contains("defaultColor") { vec![self.annotation_color()] } else { vec![] });
        }
        let Some(LayerKind::Segmentation(display)) = self.kind() else { return Some(vec![]) };
        if !self.spec.visible {
            return Some(vec![]);
        }
        let json = &self.spec.json;
        let (_, segments) = state::segmentation_settings(json).ok()?;
        let visible: std::collections::BTreeSet<u64> = segments.iter().filter(|s| s.1).map(|s| s.0).collect();
        let fixed = json.get("segmentDefaultColor").and_then(Value::as_str).and_then(state::parse_color);
        let stated: HashMap<u64, [f32; 3]> = json
            .get("segmentColors")
            .and_then(Value::as_object)
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| Some((k.parse::<u64>().ok()?, state::parse_color(v.as_str()?)?)))
                    .collect()
            })
            .unwrap_or_default();
        let fixed_only = fixed.is_some() && stated.is_empty();
        if visible.is_empty() {
            // `hasVolume`: this layer draws a volume.
            if !display.ignore_null_visible_set {
                return Some(vec![]);
            }
            return if fixed_only { Some(vec![fixed.unwrap()]) } else { None };
        }
        if fixed_only {
            return Some(vec![fixed.unwrap()]);
        }
        if stated.is_empty() && visible.len() > MAX_LAYER_BAR_UI_INDICATOR_COLORS {
            return None;
        }
        let hash = ng_core::SegmentColorHash::new(display.color_seed);
        // By ID (a `BTreeSet`), then the distinct CSS colours in order.
        let mut colors: Vec<[f32; 3]> = vec![];
        let css = |c: [f32; 3]| c.map(|v| (v * 255.0).round() as i32);
        for id in visible {
            let c = stated.get(&id).copied().or(fixed).unwrap_or_else(|| hash.compute(id));
            if !colors.iter().any(|&o| css(o) == css(c)) {
                colors.push(c);
            }
        }
        (colors.len() <= MAX_LAYER_BAR_UI_INDICATOR_COLORS).then_some(colors)
    }

    /// Visible chunks this layer needed last frame, and how many of them
    /// were loaded (`numVisibleChunksNeeded`/`Available`).
    pub fn chunk_progress(&self) -> (u32, u32) {
        self.progress
    }

    /// The `normalized` control's window, for an image layer.
    /// The channel the shader's first invlerp reads (`channel`), one index
    /// per channel dimension.
    pub fn invlerp_channel(&self) -> Vec<u32> {
        use ng_render::user_shader as us;
        let extents = self.channel_extents();
        let (Some(t), false) = (self.data_type(), extents.is_empty()) else { return vec![] };
        let text = self.spec.json.get("shader").and_then(Value::as_str).unwrap_or(state::DEFAULT_IMAGE_SHADER);
        let parsed = us::parse_with_channels(text, t, extents.len());
        us::control_values(&parsed, t, self.spec.json.get("shaderControls"))
            .into_iter()
            .find_map(|v| match v {
                us::ControlValue::Invlerp { channel, .. } => Some(channel),
                _ => None,
            })
            .unwrap_or_else(|| vec![0; extents.len()])
    }

    pub fn invlerp_window(&self) -> Option<ng_render::image::Interval> {
        self.data_type().map(|t| state::image_window(&self.spec.json, t))
    }

    /// Last frame's data histogram (see [`Viewer::want_data_histogram`]).
    pub fn data_histogram(&self) -> Option<&[u32; 256]> {
        self.data_histogram.as_deref()
    }

    /// Last frame's histogram over `window` (see
    /// [`Viewer::want_window_histogram`]).
    pub fn window_histogram(&self, window: [f64; 2]) -> Option<&[u32; 256]> {
        self.window_histograms.iter().find(|(w, _)| *w == window).map(|(_, h)| &**h)
    }

    /// The data values last frame's histogram sampled.
    pub fn histogram_samples(&self) -> &[f64] {
        &self.histogram_samples
    }

    /// `sliceViewRenderScaleHistogram`.
    pub fn render_scale_histogram(&self) -> &ng_core::render_scale::RenderScaleHistogram {
        &self.render_scale_histogram
    }

    /// `crossSectionRenderScale` (`sliceViewRenderScaleTarget`), default 1.
    pub fn render_scale_target(&self) -> f64 {
        ng_core::render_scale::parse_target(self.spec.json.get("crossSectionRenderScale"), 1.0)
    }

    /// Turns a subsource on or off (recorded only where it differs from its
    /// default).
    pub fn set_subsource_enabled(&mut self, source: usize, subsource: usize, enabled: bool) {
        let Some(slot) = self.sources.get_mut(source) else { return };
        let SourceLoad::Loaded(l) = &slot.load else { return };
        let Some(e) = l.source.subsources.get(subsource) else { return };
        let default = e.default && slot.spec.enable_default_subsources;
        let id = e.id.clone();
        if enabled == default {
            slot.spec.subsources.remove(&id);
        } else {
            slot.spec.subsources.insert(id, enabled);
        }
        if let Some(spec) = self.spec.sources.get_mut(source) {
            *spec = slot.spec.clone();
        }
    }

    /// `enableDefaultSubsources` for a source (explicit choices reset).
    pub fn set_default_subsources(&mut self, source: usize, enabled: bool) {
        let Some(slot) = self.sources.get_mut(source) else { return };
        slot.spec.enable_default_subsources = enabled;
        slot.spec.subsources.clear();
        if let Some(spec) = self.spec.sources.get_mut(source) {
            *spec = slot.spec.clone();
        }
    }

    /// Each data source as the layer panel's Source tab shows it.
    pub fn source_views(&self) -> Vec<SourceView> {
        self.sources
            .iter()
            .map(|slot| {
                let (state, subsources, transform) = match &slot.load {
                    SourceLoad::Pending => (SourceState::Loading, vec![], None),
                    SourceLoad::Failed => (SourceState::Failed, vec![], None),
                    SourceLoad::Loaded(l) => {
                        let subs = l
                            .source
                            .subsources
                            .iter()
                            .map(|e| SubsourceView {
                                // `default` is shown without its ID.
                                id: if e.id == "default" { String::new() } else { e.id.clone() },
                                description: match &e.subsource {
                                    ng_datasource::model::Subsource::Volume(v) => format!("{} volume", v.data_type.name().to_lowercase()),
                                    ng_datasource::model::Subsource::DataBounds { .. } => "default annotations".into(),
                                    ng_datasource::model::Subsource::Mesh(m) if m.multiscale.is_some() => "meshes (multi-res.)".into(),
                                    ng_datasource::model::Subsource::Mesh(_) => "meshes (single-res.)".into(),
                                    ng_datasource::model::Subsource::Skeleton(_) => "skeletons".into(),
                                    ng_datasource::model::Subsource::Annotations(_) => "annotations".into(),
                                    ng_datasource::model::Subsource::LocalAnnotations => "Local annotations".into(),
                                },
                                enabled: slot.spec.subsources.get(&e.id).copied().unwrap_or(e.default && slot.spec.enable_default_subsources),
                            })
                            .collect();
                        (SourceState::Loaded, subs, Some(l.transform.clone()))
                    }
                };
                SourceView { url: slot.spec.url.clone(), state, enable_default_subsources: slot.spec.enable_default_subsources, subsources, transform }
            })
            .collect()
    }

    /// The data-bounds boxes (`makeDataBoundsBoundingBoxAnnotationSet`) of
    /// loaded sources, where that subsource is on: by default, unless the
    /// source spec turns default subsources or `bounds` off.
    pub fn data_bounds(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        let mut out = vec![];
        for slot in &self.sources {
            let SourceLoad::Loaded(l) = &slot.load else { continue };
            for entry in &l.source.subsources {
                if let ng_datasource::model::Subsource::DataBounds { lower, upper } = &entry.subsource {
                    let enabled = slot.spec.subsources.get(&entry.id).copied().unwrap_or(entry.default && slot.spec.enable_default_subsources);
                    if enabled {
                        out.push((lower.clone(), upper.clone()));
                    }
                }
            }
        }
        out
    }

    /// An annotation layer's annotations as its Annotations tab lists them
    /// (`makeAnnotationListElement`): each one's ID, type icon, positions
    /// (`visitTransformedAnnotationGeometry`, in the global dimensions the
    /// layer maps, `None` for others; vectors without translation) and
    /// description. `None` for other layers.
    pub fn annotation_list(&self, global_rank: usize) -> Option<Vec<AnnotationListEntry>> {
        let ann = self.annotation.as_ref()?;
        ann.annotations.iter().map(|a| self.annotation_entry(a, global_rank)).collect()
    }

    /// One annotation as the list (and the Selection panel) shows it.
    pub(crate) fn annotation_entry(&self, a: &annotations::Annotation, global_rank: usize) -> Option<AnnotationListEntry> {
        let ann = self.annotation.as_ref()?;
        let l = &ann.layer;
        let n = l.rank + 1;
        let m2l = &l.model_to_layer;
        let to_global = |model: &[f64], vector: bool| -> Vec<Option<f64>> {
            let mut out = vec![None; global_rank];
            for (ld, g) in l.layer_to_global.iter().enumerate() {
                let Some(g) = g else { continue };
                if *g >= global_rank {
                    continue;
                }
                let mut v = if vector { 0.0 } else { m2l[l.rank * n + ld] };
                for (c, x) in model.iter().enumerate().take(l.source_rank) {
                    v += m2l[c * n + ld] * x;
                }
                out[*g] = Some(v);
            }
            out
        };
        use annotations::Geometry as G;
        let (icon, rows): (&'static str, Vec<Vec<Option<f64>>>) = match &a.geometry {
            G::Point(p) => ("⚬", vec![to_global(p, false)]),
            G::Line(pa, pb) => ("ꕹ", vec![to_global(pa, false), to_global(pb, false)]),
            G::AxisAlignedBoundingBox(pa, pb) => ("❑", vec![to_global(pa, false), to_global(pb, false)]),
            G::Ellipsoid { center, radii } => ("◎", vec![to_global(center, false), to_global(radii, true)]),
            G::Polyline(points) => ("⤤", points.iter().map(|p| to_global(p, false)).collect()),
        };
        Some(AnnotationListEntry { id: a.id.clone(), icon, rows, description: a.description.clone() })
    }

    /// Global dimensions an annotation layer's transform maps
    /// (`globalDimensionIndices`).
    pub fn annotation_dimensions(&self) -> Option<Vec<usize>> {
        let ann = self.annotation.as_ref()?;
        let mut dims: Vec<usize> = ann.layer.layer_to_global.iter().flatten().copied().collect();
        dims.sort_unstable();
        dims.dedup();
        Some(dims)
    }

    /// The mesh sources of loaded sources, where that subsource is on.
    ///
    /// Meshes are drawn through the layer's volume source's model-to-layer
    /// transform, but `to_model` takes a mesh into its own source's model
    /// space: a mesh-only source beside the voxels (BANC's meshes at 8 nm,
    /// its segmentation at 16) would be scaled by their ratio. Such a mesh is
    /// carried into the volume source's model units — exact when the two
    /// sources' transforms differ only in their model scales.
    pub fn mesh_sources(&self) -> Vec<ng_datasource::mesh::MeshSource> {
        let volume_scales = self.render.as_ref().and_then(|r| match &self.sources.get(r.source_index)?.load {
            SourceLoad::Loaded(l) => Some((r.source_index, l.transform.input_space.scales.clone())),
            _ => None,
        });
        let mut out = vec![];
        for (si, slot) in self.sources.iter().enumerate() {
            let SourceLoad::Loaded(l) = &slot.load else { continue };
            for entry in &l.source.subsources {
                if let ng_datasource::model::Subsource::Mesh(m) = &entry.subsource {
                    if slot.spec.subsources.get(&entry.id).copied().unwrap_or(entry.default && slot.spec.enable_default_subsources) {
                        let mut m = (**m).clone();
                        if let Some((vi, vs)) = &volume_scales {
                            if *vi != si {
                                let ms = &l.transform.input_space.scales;
                                for r in 0..3 {
                                    let k = match (ms.get(r), vs.get(r)) {
                                        (Some(a), Some(b)) if *b > 0.0 => a / b,
                                        _ => 1.0,
                                    };
                                    // row r of the column-major 4 × 4
                                    for c in 0..4 {
                                        m.to_model[c * 4 + r] *= k;
                                    }
                                }
                            }
                        }
                        out.push(m);
                    }
                }
            }
        }
        out
    }

    /// The layer's channel dimensions: name and extent.
    pub fn channel_dimensions(&self) -> Vec<(String, u32)> {
        self.render.as_ref().map_or(vec![], |r| r.layer.channels.iter().map(|c| (r.layer.transform.output_space.names[c.layer_dim].clone(), c.extent)).collect())
    }

    /// The extents of the layer's channel dimensions (`getChannelSpace`),
    /// first fastest; none for a single channel.
    pub fn channel_extents(&self) -> Vec<u32> {
        self.render.as_ref().map_or(vec![], |r| r.layer.channel_extents())
    }

    /// The meshes' render scale histogram (`Resolution (mesh)`).
    pub fn mesh_render_scale_histogram(&self) -> &ng_core::render_scale::RenderScaleHistogram {
        &self.mesh_render_scale_histogram
    }

    /// The spatial index spacing histogram of cross-sections (`false`) or
    /// the 3D view (`true`).
    pub fn annotation_histogram(&self, projection: bool) -> &ng_core::render_scale::RenderScaleHistogram {
        &self.annotation_histograms[usize::from(projection)]
    }

    /// `crossSectionAnnotationSpacing` or (`projection`)
    /// `projectionAnnotationSpacing`: 8 pixels by default.
    pub fn annotation_spacing_target(&self, projection: bool) -> f64 {
        let key = if projection { "projectionAnnotationSpacing" } else { "crossSectionAnnotationSpacing" };
        ng_core::render_scale::parse_target(self.spec.json.get(key), 8.0)
    }

    /// The relationships of the layer's annotations (`relationships`): a
    /// precomputed source's, else local annotations' `segments`.
    pub fn annotation_relationships(&self) -> Vec<String> {
        match self.annotation.as_ref() {
            Some(AnnotationLayerState { spatial: Some(info), .. }) => info.relationships.iter().map(|(id, _)| id.clone()).collect(),
            _ => vec!["segments".into()],
        }
    }

    /// The annotation shader's properties: a precomputed source's, else the
    /// layer's `annotationProperties`.
    pub fn annotation_property_types(&self) -> Vec<(String, ng_render::annotation_shader::PropertyShaderType)> {
        use annotations::PropertyType as T;
        use ng_render::annotation_shader::PropertyShaderType as S;
        let shader_type = |k: T| match k {
            T::Rgb => S::Rgb,
            T::Rgba => S::Rgba,
            T::Float32 => S::Float,
            T::Uint8 | T::Uint16 | T::Uint32 => S::Uint,
            T::Int8 | T::Int16 | T::Int32 => S::Int,
        };
        match self.annotation.as_ref().and_then(|a| a.spatial.as_ref()) {
            Some(info) => info.properties.iter().map(|p| (p.id.clone(), shader_type(p.kind))).collect(),
            None => annotations::parse_property_specs(self.spec.json.get("annotationProperties")).into_iter().map(|p| (p.id, shader_type(p.kind))).collect(),
        }
    }

    /// The annotation properties as the Rendering tab lists them: id,
    /// type name and description.
    pub fn annotation_properties(&self) -> Vec<(String, &'static str, Option<String>)> {
        match self.annotation.as_ref().and_then(|a| a.spatial.as_ref()) {
            Some(info) => info.properties.iter().map(|p| (p.id.clone(), p.kind.name(), p.description.clone())).collect(),
            None => annotations::parse_property_specs(self.spec.json.get("annotationProperties")).into_iter().map(|p| (p.id, p.kind.name(), p.description)).collect(),
        }
    }

    /// Recompiles the annotation shader if its code or properties changed
    /// (its error kept for the editor); refreshes its controls' values.
    pub(crate) fn update_annotation_shader(&mut self, device: &wgpu::Device) {
        if self.annotation.is_none() {
            return;
        }
        let code = self.spec.json.get("shader").and_then(Value::as_str).unwrap_or(state::DEFAULT_ANNOTATION_SHADER).to_owned();
        let properties = self.annotation_property_types();
        let parsed = ng_render::user_shader::parse(&code, DataType::Float32);
        let values = ng_render::user_shader::control_values(&parsed, DataType::Float32, self.spec.json.get("shaderControls"));
        let controls = ng_render::annotation_shader::control_values(&parsed, &values);
        if let Some(cache) = self.annotation_shader.as_mut().filter(|c| c.code == code && c.properties == properties) {
            cache.controls = controls;
            return;
        }
        let pipelines = if code == state::DEFAULT_ANNOTATION_SHADER {
            self.shader_error = None;
            None
        } else {
            let compiled = [false, true].map(|projection| ng_render::annotation_shader::compile(&parsed, &values, &properties, projection));
            match compiled {
                [Ok(slice), Ok(projection)] => {
                    self.shader_error = None;
                    Some([ng_render::annotation_shader::AnnotationShaderPipeline::new(device, &slice), ng_render::annotation_shader::AnnotationShaderPipeline::new(device, &projection)])
                }
                [Err(e), _] | [_, Err(e)] => {
                    self.shader_error = Some(e);
                    None
                }
            }
        };
        self.annotation_shader = Some(AnnotationShaderCache { code, properties, controls, pipelines });
    }

    /// Whether the annotations come from a precomputed spatial index.
    pub fn has_spatial_annotations(&self) -> bool {
        self.annotation.as_ref().is_some_and(|a| a.spatial.is_some())
    }

    /// Volume rendering's chunk resolution histogram (`Resolution (3D)`).
    pub fn volume_rendering_histogram(&self) -> &ng_core::render_scale::RenderScaleHistogram {
        &self.volume_rendering_histogram
    }

    /// `meshRenderScale` (`renderScaleTarget`): the meshes' level-of-detail
    /// cutoff, in pixels.
    pub fn mesh_render_scale_target(&self) -> f64 {
        ng_core::render_scale::parse_target(self.spec.json.get("meshRenderScale"), 1.0)
    }

    /// `meshSilhouetteRendering`: a finite non-negative power, else 0.
    pub fn silhouette_rendering(&self) -> f64 {
        self.spec.json.get("meshSilhouetteRendering").and_then(Value::as_f64).filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(0.0)
    }

    /// `has3dLayer`: meshes or skeletons to draw in 3D.
    pub fn has_3d(&self) -> bool {
        !self.mesh_sources().is_empty() || self.has_skeletons()
    }

    /// `hasSkeletonsLayer`.
    pub fn has_skeletons(&self) -> bool {
        !self.skeleton_sources().is_empty()
    }

    /// The first skeleton source's vertex attributes (beyond positions),
    /// as the Render tab lists them: GLSL type and name.
    pub fn skeleton_vertex_attributes(&self) -> Vec<(String, String)> {
        let sources = self.skeleton_sources();
        let Some(s) = sources.first() else { return vec![] };
        s.info.vertex_attributes.iter().map(|a| (if a.num_components > 1 { format!("vec{}", a.num_components) } else { "float".into() }, a.id.clone())).collect()
    }

    /// The skeleton sources of loaded sources, where that subsource is on.
    pub fn skeleton_sources(&self) -> Vec<ng_datasource::skeleton::SkeletonSource> {
        let mut out = vec![];
        for slot in &self.sources {
            let SourceLoad::Loaded(l) = &slot.load else { continue };
            for entry in &l.source.subsources {
                if let ng_datasource::model::Subsource::Skeleton(m) = &entry.subsource {
                    if slot.spec.subsources.get(&entry.id).copied().unwrap_or(entry.default && slot.spec.enable_default_subsources) {
                        out.push((**m).clone());
                    }
                }
            }
        }
        out
    }

    /// The finest scale's voxel size along each global dimension (metres,
    /// say), where the layer has one: its chunk-to-global transform's
    /// column lengths, in the global dimensions' units.
    pub fn voxel_size_global(&self, global_rank: usize) -> Vec<Option<f64>> {
        let mut out = vec![None; global_rank];
        let Some(render) = &self.render else { return out };
        let layer = &render.layer;
        let n = layer.rank + 1;
        let m = layer.chunk_to_layer(0);
        for (ld, g) in layer.layer_to_global.iter().enumerate() {
            let Some(g) = *g else { continue };
            if g >= global_rank {
                continue;
            }
            // The chunk dimension that maps onto this one.
            let v = (0..layer.rank).map(|c| m[c * n + ld].abs()).fold(0.0, f64::max);
            if v > 0.0 {
                out[g] = Some(v);
            }
        }
        out
    }

    /// [`data_bounds`](Self::data_bounds) in global coordinates (one entry
    /// per global dimension, `NaN` where the layer has none): the corners
    /// through the source's transform.
    pub fn data_bounds_global(&self, global_rank: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
        self.data_bounds().iter().map(|(lo, hi)| (self.model_to_global(lo, global_rank), self.model_to_global(hi, global_rank))).collect()
    }

    /// Each data box's eight corners (its first three model dimensions) in
    /// global coordinates: the box as it stands, turned with the layer's
    /// transform — `data_bounds_global` maps only two of them.
    pub fn data_box_corners_global(&self, global_rank: usize) -> Vec<Vec<Vec<f64>>> {
        self.data_bounds()
            .iter()
            .map(|(lo, hi)| {
                (0..8)
                    .map(|c| {
                        let corner: Vec<f64> = (0..lo.len()).map(|k| if k < 3 && (c >> k) & 1 == 1 { hi[k] } else { lo[k] }).collect();
                        self.model_to_global(&corner, global_rank)
                    })
                    .collect()
            })
            .collect()
    }

    /// A model point in global coordinates (NaN where a global dimension is
    /// not the layer's).
    fn model_to_global(&self, model: &[f64], global_rank: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; global_rank];
        let Some(render) = &self.render else { return out };
        let layer = &render.layer;
        let (rank, n) = (layer.rank, layer.rank + 1);
        let m2l = &layer.model_to_layer;
        for (ld, g) in layer.layer_to_global.iter().enumerate() {
            let Some(g) = *g else { continue };
            if g < global_rank {
                out[g] = (0..model.len().min(rank)).map(|k| m2l[k * n + ld] * model[k]).sum::<f64>() + m2l[rank * n + ld];
            }
        }
        out
    }

    /// `ManagedUserLayer.toJSON`: `type` (as resolved), `source` (each data
    /// source's spec, with its transform relative to the source's own), the
    /// layer's other keys as restored, then `name` and `visible`/`archived`.
    pub fn to_json(&self) -> Value {
        let mut o = serde_json::Map::new();
        let layer_type = match self.kind() {
            Some(LayerKind::Image { .. }) => "image".to_owned(),
            Some(LayerKind::Segmentation(_)) => "segmentation".to_owned(),
            None => self.spec.layer_type.clone(),
        };
        o.insert("type".into(), layer_type.into());
        let sources: Vec<Value> = self.sources.iter().map(|s| s.to_json()).collect();
        match sources.len() {
            0 => {}
            1 => {
                o.insert("source".into(), sources.into_iter().next().unwrap());
            }
            _ => {
                o.insert("source".into(), Value::Array(sources));
            }
        }
        for (k, v) in &self.spec.json {
            // `transform` here is the legacy form, now inside `source`.
            if !matches!(k.as_str(), "type" | "source" | "name" | "visible" | "archived" | "transform") {
                o.insert(k.clone(), v.clone());
            }
        }
        o.insert("name".into(), self.spec.name.clone().into());
        if !self.spec.visible {
            if self.spec.archived {
                o.insert("archived".into(), true.into());
            } else {
                o.insert("visible".into(), false.into());
            }
        }
        Value::Object(o)
    }
}

impl SourceSlot {
    /// `LayerDataSource.toJSON`: the URL alone, or with the transform spec.
    fn to_json(&self) -> Value {
        let transform = match &self.load {
            SourceLoad::Loaded(l) => l.transform.spec_over(&VolumeLayer::default_transform(&l.source)),
            // Not loaded: the spec as given.
            _ => self.spec.transform.clone(),
        };
        // `layerDataSourceSpecificationToJson`: the URL alone unless there
        // is more to say.
        let defaults = self.spec.enable_default_subsources;
        if transform.is_none() && self.spec.subsources.is_empty() && defaults {
            return Value::String(self.spec.url.clone());
        }
        let mut o = serde_json::Map::new();
        o.insert("url".into(), self.spec.url.clone().into());
        if let Some(t) = transform {
            o.insert("transform".into(), t.to_json());
        }
        if !self.spec.subsources.is_empty() {
            o.insert("subsources".into(), Value::Object(self.spec.subsources.iter().map(|(k, v)| (k.clone(), Value::Bool(*v))).collect()));
        }
        if !defaults {
            o.insert("enableDefaultSubsources".into(), false.into());
        }
        Value::Object(o)
    }
}

/// Whether a source's volume subsource is on: its explicit setting, else
/// its default unless the spec turns default subsources off.
pub(crate) fn volume_enabled(slot: &SourceSlot, source: &ng_datasource::model::DataSource) -> bool {
    source.subsources.iter().any(|e| {
        matches!(e.subsource, ng_datasource::model::Subsource::Volume(_)) && slot.spec.subsources.get(&e.id).copied().unwrap_or(e.default && slot.spec.enable_default_subsources)
    })
}

/// A data source to load for [`Viewer::source_loaded`].
#[derive(Debug, Clone, PartialEq)]
pub struct PendingSource {
    pub layer: LayerId,
    pub index: usize,
    pub url: String,
}
