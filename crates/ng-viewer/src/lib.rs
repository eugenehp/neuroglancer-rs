//! The viewer itself, independent of window system: neuroglancer's 4panel
//! layout, its navigation and input bindings, chunked volume loading and
//! frame drawing. `ng-native` and `ng-web` create a surface, an executor for
//! loading, and forward events.
//!
//! Units: the shells pass physical pixels and a scale factor. Navigation works
//! in logical pixels, which is what neuroglancer's "viewport pixels" are (its
//! canvas is sized in CSS pixels); rendering uses the full resolution.
//!
//! Cross-section panels render as neuroglancer's do (`ng_render::compose`):
//! layers into an offscreen texture, then onto the background.
//!
//! The 3D panel is neuroglancer's default perspective view: the
//! cross-sections as planes, data bounds and axis lines (no meshes or volume
//! rendering yet).

pub mod annotations;
pub mod chunks;
pub mod histogram;
pub mod host;
mod legend;
pub mod playback;
pub mod prefetch;
pub mod screenshot;
pub mod input;
pub mod layer;
pub mod scene;
pub mod state;
pub mod tool_query;
pub mod tools;
/// User image shaders (`#uicontrol`s, GLSL to WGSL).
pub use ng_render::user_shader;

pub use ng_render::image::{default_data_type_range, expected_gray, Interval};

use std::collections::HashMap;
use std::sync::Arc;

use chunks::{panel_sources, visible_chunks, visible_scales, PanelCamera};
use input::{lookup, Modifiers, PanelKind};
use layer::VolumeLayer;
use ng_chunk::{ChunkManager, Executor, State};
use ng_core::navigation::{Navigation, View};
use ng_core::panel::{wheel_zoom_amount, NamedAxes, PanelPointer, SlicePanel};
use ng_core::combiner::{Binding, BoundSpace, CoordinateSpaceCombiner};
use ng_core::coordinate_space::is_global_dimension;
use ng_core::transform::CoordinateSpaceTransform;
use ng_core::{CoordinateSpace, DataType};
use serde_json::Value;
use state::{LayerSpec, SourceSpec};
use ng_datasource::chunk::{read_chunk, ChunkData};
use ng_datasource::model::DataSource;
use ng_render::compose::{CompositeBinding, Compositor, LayerPipelines, PanelTarget};
use ng_render::segmentation::{SegmentationDisplay, VisibleSet};
use threers::math::Matrix4;
use threers::renderer::PixelRect;
use threers::volume::{Data3DTexture, GpuVolume, SliceBinding, SliceParams, VolumeFormat};
use ng_datasource::model::VolumeType;


/// `crossSectionBackgroundColor`'s default.
pub const CROSS_SECTION_BACKGROUND: [f32; 3] = [0.5, 0.5, 0.5];

/// A data panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Slice(NamedAxes),
    Perspective,
}

/// neuroglancer's data panel layouts (`LAYOUTS`, `src/data_panel_layout.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// xy and xz above, 3d and yz below (`FourPanelLayout`).
    FourPanel,
    /// xy and yz above, xz and 3d below (`FourPanelAltLayout`).
    FourPanelAlt,
    /// The 3D view alone (`SinglePerspectiveLayout`).
    Perspective,
    /// One cross-section (`SinglePanelLayout`).
    Single(NamedAxes),
    /// A cross-section left of the 3D view (`SliceViewPerspectiveTwoPanelLayout`, row).
    SliceAnd3d(NamedAxes),
}

impl Layout {
    pub const ALL: [Layout; 9] = [
        Layout::FourPanel,
        Layout::FourPanelAlt,
        Layout::Perspective,
        Layout::Single(NamedAxes::Xy),
        Layout::Single(NamedAxes::Xz),
        Layout::Single(NamedAxes::Yz),
        Layout::SliceAnd3d(NamedAxes::Xy),
        Layout::SliceAnd3d(NamedAxes::Xz),
        Layout::SliceAnd3d(NamedAxes::Yz),
    ];

    /// The state's name for it.
    pub fn name(self) -> String {
        let axes = |a: NamedAxes| match a {
            NamedAxes::Xy => "xy",
            NamedAxes::Xz => "xz",
            NamedAxes::Yz => "yz",
        };
        match self {
            Layout::FourPanel => "4panel".into(),
            Layout::FourPanelAlt => "4panel-alt".into(),
            Layout::Perspective => "3d".into(),
            Layout::Single(a) => axes(a).into(),
            Layout::SliceAnd3d(a) => format!("{}-3d", axes(a)),
        }
    }

    /// `getLayoutByName`.
    pub fn from_name(name: &str) -> Option<Layout> {
        Layout::ALL.into_iter().find(|l| l.name() == name)
    }

    /// The panels and their rectangles in a `width` × `height` area (pixels,
    /// top-left origin). Panels split the area evenly (`flex: 1`); with
    /// `borders` (neuroglancer's `showPanelBorders`, on by default) each has a
    /// 2 px border and draws inside it. Rounded as `RenderedDataPanel` does:
    /// the left/top edge `Math.round`ed, the size the element's integer
    /// `clientWidth`/`clientHeight`, clipped to the area.
    pub fn panels_with(self, width: u32, height: u32, borders: bool) -> Vec<(Panel, PixelRect)> {
        self.panels_scaled(width, height, if borders { 1.0 } else { 0.0 })
    }

    /// [`panels_with`](Self::panels_with) borders, in an area of physical
    /// pixels `scale` to a CSS pixel: the 2 px borders are `2 × scale`
    /// physical pixels (`scale` 0 for none).
    pub fn panels_scaled(self, width: u32, height: u32, scale: f64) -> Vec<(Panel, PixelRect)> {
        let inset = 2.0 * scale;
        let cell = |col: u32, row: u32, cols: u32, rows: u32| {
            let (w, h) = (width as f64 / cols as f64, height as f64 / rows as f64);
            let x = ng_core::js::round(col as f64 * w + inset);
            let y = ng_core::js::round(row as f64 * h + inset);
            let cw = ng_core::js::round(w - 2.0 * inset).max(1.0);
            let ch = ng_core::js::round(h - 2.0 * inset).max(1.0);
            PixelRect {
                x: x as u32,
                y: y as u32,
                width: (cw.min(width as f64 - x)).max(1.0) as u32,
                height: (ch.min(height as f64 - y)).max(1.0) as u32,
            }
        };
        self.cells().into_iter().map(|(p, [c, r, cs, rs])| (p, cell(c, r, cs, rs))).collect()
    }

    /// Each panel's cell: column, row, and the grid's columns and rows
    /// (panels share the area evenly, `flex: 1`).
    pub fn cells(self) -> Vec<(Panel, [u32; 4])> {
        let s = Panel::Slice;
        match self {
            Layout::FourPanel => vec![
                (s(NamedAxes::Xy), [0, 0, 2, 2]),
                (s(NamedAxes::Xz), [1, 0, 2, 2]),
                (Panel::Perspective, [0, 1, 2, 2]),
                (s(NamedAxes::Yz), [1, 1, 2, 2]),
            ],
            Layout::FourPanelAlt => vec![
                (s(NamedAxes::Xy), [0, 0, 2, 2]),
                (s(NamedAxes::Yz), [1, 0, 2, 2]),
                (s(NamedAxes::Xz), [0, 1, 2, 2]),
                (Panel::Perspective, [1, 1, 2, 2]),
            ],
            Layout::Perspective => vec![(Panel::Perspective, [0, 0, 1, 1])],
            Layout::Single(a) => vec![(s(a), [0, 0, 1, 1])],
            Layout::SliceAnd3d(a) => vec![(s(a), [0, 0, 2, 1]), (Panel::Perspective, [1, 0, 2, 1])],
        }
    }

    /// [`panels_with`](Self::panels_with) panel borders.
    pub fn panels(self, width: u32, height: u32) -> Vec<(Panel, PixelRect)> {
        self.panels_with(width, height, true)
    }

    /// The rectangle of `panel`, if the layout has it.
    pub fn rect(self, panel: Panel, width: u32, height: u32) -> Option<PixelRect> {
        self.panels(width, height).into_iter().find(|(p, _)| *p == panel).map(|(_, r)| r)
    }

    /// The panel containing `(x, y)`, if any (not a gap).
    pub fn panel_at(self, x: f64, y: f64, width: u32, height: u32) -> Option<Panel> {
        self.panels(width, height).into_iter().find_map(|(p, r)| {
            (x >= r.x as f64 && x < (r.x + r.width) as f64 && y >= r.y as f64 && y < (r.y + r.height) as f64).then_some(p)
        })
    }
}

/// Mouse buttons, numbered as DOM `MouseEvent.button` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Button {
    Primary = 0,
    Middle = 1,
    Secondary = 2,
}

#[derive(Debug, Clone)]
enum DragAction {
    Translate,
    /// Through the global point under the pointer when the drag began.
    Rotate { initial_position: Vec<f32> },
    /// The 3D panel: rotate the projection orientation.
    PerspectiveRotate,
    /// The 3D panel: move the position in the view plane.
    PerspectiveTranslate,
    /// `move-annotation`: an annotation of layer `layer` (local) dragged by
    /// its representative point (`getRepresentativePoint`), from its
    /// geometry when the drag began, `total` logical pixels so far.
    MoveAnnotation { layer: LayerId, id: String, part: AnnotationPart, point: Vec<f64>, geometry: annotations::Geometry, total: (f64, f64) },
}

#[derive(Debug, Clone)]
struct Drag {
    panel: Panel,
    action: DragAction,
}

/// Identifies a layer for as long as it exists.
pub type LayerId = u64;

/// A chunk: its layer, scale and grid position.
pub type ChunkKey = (LayerId, usize, Vec<u64>);

/// A chunk's texture: the chunk, and the offsets along non-displayed
/// dimensions of the slab it holds.
type TextureKey = (usize, Vec<u64>, Vec<u32>);

struct ChunkTexture {
    volume: GpuVolume,
    /// Per slice panel (uniforms differ per panel), and per pipeline (first
    /// layer drawn, or blended over others).
    bindings: [[Option<SliceBinding>; 2]; 3],
    last_used: u64,
}

/// A slice panel as the 3D panel shows it: its index, `invViewMatrix`, and
/// physical size.
type SliceViewFrame = (usize, ng_core::glm::Mat4, (u32, u32));

/// A chunk source's statistics (`ChunkStatistics`, its visible tier).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SourceStatistics {
    /// `getChunkSourceIdentifier`, flattened (`.type`, `.parameters.url`,
    /// …), in order.
    pub properties: Vec<(String, String)>,
    pub visible_total: u32,
    pub downloading: u32,
    pub system: u32,
    pub gpu: u32,
    pub failed: u32,
    pub gpu_bytes: usize,
    /// The mean download time (ms), if timed.
    pub download_latency: Option<f64>,
}

/// A source's identifying properties, as neuroglancer's key flattens them.
fn source_properties(params: &ng_datasource::model::ChunkSourceParams) -> Vec<(String, String)> {
    use ng_datasource::model::ChunkSourceParams as P;
    match params {
        P::Precomputed(p) => vec![(".type".into(), "precomputed/VolumeChunkSource".into()), (".parameters.url".into(), p.url())],
        P::Zarr(p) => vec![(".type".into(), "zarr/VolumeChunkSource".into()), (".parameters.url".into(), p.url())],
        P::Nifti { url, .. } => vec![(".type".into(), "nifti/VolumeChunkSource".into()), (".parameters.url".into(), url.clone())],
        P::N5(p) => vec![(".type".into(), "n5/VolumeChunkSource".into()), (".parameters.url".into(), p.location.url())],
        P::DeepZoom(p) => vec![(".type".into(), "deepzoom/ImageTileSource".into()), (".parameters.url".into(), p.location.url())],
    }
}

/// Converts a slab's little-endian voxels to texture bytes.
type UploadFn = fn(DataType, &[u8]) -> Vec<u8>;

/// How a layer is shaded: the user layer type (`type` in the state, or as
/// `detectLayerType` picks it: segmentation volumes get a segmentation layer,
/// the rest an image layer).
#[derive(Debug, Clone, PartialEq)]
pub enum LayerKind {
    Image { opacity: f32 },
    Segmentation(SegmentationDisplay),
}

/// What draws a layer: its volume, shading and textures.
struct LayerState {
    /// The data source it draws (the first with a volume).
    source_index: usize,
    layer: VolumeLayer,
    kind: LayerKind,
    pipelines: LayerPipelines,
    textures: HashMap<TextureKey, ChunkTexture>,
    /// Segmentation: the visible segments (`visibleSegments`), and the set's
    /// generation the textures' bindings were made with.
    visible: Option<(VisibleSet, u64)>,
    /// An image's own shader (not the default), compiled.
    shader: Option<UserShader>,
    /// Its shader compiled for volume rendering, while that is on.
    volume: Option<VolumeShader>,
}

/// An image's shader compiled for volume rendering: what it was compiled
/// from, its controls, the pipeline and the controls' texture.
struct VolumeShader {
    key: (String, Vec<bool>, Vec<Vec<u32>>, ng_render::user_shader::VolumeMode),
    parsed: ng_render::user_shader::ParsedShader,
    values: Vec<ng_render::user_shader::ControlValue>,
    pipeline: ng_render::perspective::VolumePipeline,
    uniforms: ng_render::user_shader::UniformTexture,
}

/// A compiled user shader: what it was compiled from (the code and its
/// checkboxes, which are compiled in), its controls and their values, and
/// the texture the values go to.
struct UserShader {
    key: (String, Vec<bool>, Vec<Vec<u32>>),
    parsed: ng_render::user_shader::ParsedShader,
    values: Vec<ng_render::user_shader::ControlValue>,
    uniforms: ng_render::user_shader::UniformTexture,
}

impl LayerState {
    /// An image layer's shader brought up to date with its JSON: compiled
    /// again when its code or a checkbox changes; the default built in; on
    /// an error the default too (`fallbackShaderParameters`), and the error.
    fn update_shader(&mut self, device: &wgpu::Device, json: &serde_json::Map<String, Value>) -> Option<ng_render::user_shader::CompileError> {
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
        // What is compiled in: checkboxes, and the channels read.
        let checks: Vec<bool> = values.iter().filter_map(|v| if let us::ControlValue::Bool(b) = v { Some(*b) } else { None }).collect();
        let channels: Vec<Vec<u32>> = values
            .iter()
            .filter_map(|v| match v {
                us::ControlValue::Invlerp { channel, .. } => Some(channel.clone()),
                us::ControlValue::TransferFunction(tf) => Some(tf.channel.clone()),
                _ => None,
            })
            .collect();
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
    fn volume_shader(&mut self, device: &wgpu::Device, renderer: &ng_render::perspective::PerspectiveRenderer, json: &serde_json::Map<String, Value>, mode: ng_render::user_shader::VolumeMode) -> Option<&VolumeShader> {
        use ng_render::user_shader as us;
        let t = self.layer.data_type;
        let extents = self.layer.channel_extents();
        let text = json.get("shader").and_then(Value::as_str).unwrap_or(state::DEFAULT_IMAGE_SHADER);
        let parsed = us::parse_with_channels(text, t, extents.len());
        let values = us::control_values(&parsed, t, json.get("shaderControls"));
        let checks: Vec<bool> = values.iter().filter_map(|v| if let us::ControlValue::Bool(b) = v { Some(*b) } else { None }).collect();
        let channels: Vec<Vec<u32>> = values
            .iter()
            .filter_map(|v| match v {
                us::ControlValue::Invlerp { channel, .. } => Some(channel.clone()),
                us::ControlValue::TransferFunction(tf) => Some(tf.channel.clone()),
                _ => None,
            })
            .collect();
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
    }
}

impl LayerState {
    fn upload_format(&self) -> (VolumeFormat, UploadFn) {
        match self.kind {
            LayerKind::Image { .. } => (ng_render::image::volume_format(self.layer.data_type), ng_render::image::upload_bytes),
            LayerKind::Segmentation(_) => (
                ng_render::segmentation::volume_format(self.layer.data_type),
                ng_render::segmentation::upload_bytes,
            ),
        }
    }

    fn slice_params(&self, ndc_to_voxel: Matrix4, hovered: Option<u64>) -> SliceParams {
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
enum SourceLoad {
    Pending,
    Failed,
    Loaded(Box<LoadedSource>),
}

struct LoadedSource {
    source: DataSource,
    /// The source's transform with the state's spec over it; the global
    /// combiner rescales it.
    transform: CoordinateSpaceTransform,
    binding: Binding,
}

struct SourceSlot {
    spec: SourceSpec,
    load: SourceLoad,
    /// Its URL was typed in (`setManually`).
    set_manually: bool,
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

/// A layer (`ManagedUserLayer`): its state, its data sources, and what draws
/// it once they load.
pub struct ManagedLayer {
    pub id: LayerId,
    pub spec: LayerSpec,
    sources: Vec<SourceSlot>,
    render: Option<LayerState>,
    /// Problems to show: failed sources, unsupported settings.
    pub messages: Vec<String>,
    /// (needed, available) visible chunks, last frame.
    progress: (u32, u32),
    /// `sliceViewRenderScaleHistogram`: the slice views' chunks last frame.
    render_scale_histogram: ng_core::render_scale::RenderScaleHistogram,
    /// The meshes' (`displayState.renderScaleHistogram`).
    mesh_render_scale_histogram: ng_core::render_scale::RenderScaleHistogram,
    /// An annotation layer's spatial index spacing, in cross-sections and
    /// in 3D (`annotationCrossSectionRenderScaleHistogram`, …).
    annotation_histograms: [ng_core::render_scale::RenderScaleHistogram; 2],
    /// Volume rendering's (`volumeRenderingChunkResolutionHistogram`).
    volume_rendering_histogram: ng_core::render_scale::RenderScaleHistogram,
    /// The chunks volume rendering drew last frame (for its data histogram
    /// when there is no slice view): data, chunk display dimensions, fixed
    /// offsets.
    volume_chunks: Vec<VolumeChunk>,
    /// The `normalized` control's data histogram, while its CDF panel shows
    /// (`dataHistogramSpecifications`): wanted, and last frame's counts.
    histogram_wanted: bool,
    data_histogram: Option<Box<[u32; 256]>>,
    /// The values the histogram counted (for `AutoRangeFinder`).
    histogram_samples: Vec<f64>,
    /// Other invlerp controls' windows to count over (a user shader's), and
    /// last frame's counts over each.
    histogram_windows: Vec<[f64; 2]>,
    window_histograms: Vec<([f64; 2], Box<[u32; 256]>)>,
    /// A user shader's legends: asked for (by their values), being drawn,
    /// and drawn (each with what it was drawn for).
    legend_wants: Vec<Vec<f64>>,
    legend_jobs: Vec<legend::LegendJob>,
    legends: Vec<(legend::LegendKey, Vec<[u8; 4]>)>,
    /// Why the layer's shader did not compile (it shows the default).
    shader_error: Option<ng_render::user_shader::CompileError>,
    /// An annotation layer's local annotations, placed by their source's
    /// transform.
    annotation: Option<AnnotationLayerState>,
    /// An annotation layer's shader, compiled for cross-sections and 3D.
    annotation_shader: Option<AnnotationShaderCache>,
}

/// An annotation layer's shader as last compiled: from what code and
/// properties, its controls' values, and a pipeline per view (`None` for
/// the default shader, whose styles need no GPU).
struct AnnotationShaderCache {
    code: String,
    properties: Vec<(String, ng_render::annotation_shader::PropertyShaderType)>,
    controls: Vec<f32>,
    pipelines: Option<[ng_render::annotation_shader::AnnotationShaderPipeline; 2]>,
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
    fn annotation_entry(&self, a: &annotations::Annotation, global_rank: usize) -> Option<AnnotationListEntry> {
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
    pub fn mesh_sources(&self) -> Vec<ng_datasource::mesh::MeshSource> {
        let mut out = vec![];
        for slot in &self.sources {
            let SourceLoad::Loaded(l) = &slot.load else { continue };
            for entry in &l.source.subsources {
                if let ng_datasource::model::Subsource::Mesh(m) = &entry.subsource {
                    if slot.spec.subsources.get(&entry.id).copied().unwrap_or(entry.default && slot.spec.enable_default_subsources) {
                        out.push((**m).clone());
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
    fn update_annotation_shader(&mut self, device: &wgpu::Device) {
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
        let Some(render) = &self.render else { return vec![] };
        let layer = &render.layer;
        let (rank, n) = (layer.rank, layer.rank + 1);
        let m2l = &layer.model_to_layer;
        let map = |model: &[f64]| -> Vec<f64> {
            let mut out = vec![f64::NAN; global_rank];
            for (ld, g) in layer.layer_to_global.iter().enumerate() {
                let Some(g) = *g else { continue };
                if g < global_rank {
                    out[g] = (0..model.len().min(rank)).map(|k| m2l[k * n + ld] * model[k]).sum::<f64>() + m2l[rank * n + ld];
                }
            }
            out
        };
        self.data_bounds().iter().map(|(lo, hi)| (map(lo), map(hi))).collect()
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

/// A data source to load for [`Viewer::source_loaded`].
#[derive(Debug, Clone, PartialEq)]
pub struct PendingSource {
    pub layer: LayerId,
    pub index: usize,
    pub url: String,
}

/// A slice panel's offscreen targets and their compositing bindings: the
/// slice view into the panel, the panel onto the frame.
struct PanelFrame {
    target: PanelTarget,
    to_panel: CompositeBinding,
    to_frame: CompositeBinding,
}

/// Viewer settings from the state, with neuroglancer's defaults
/// (`TrackableBoolean`/`TrackableValue`: written back only when changed).
#[derive(Debug, Clone, PartialEq)]
pub struct ViewerSettings {
    /// `title` (the page title).
    pub title: Option<String>,
    /// `showDefaultAnnotations`: data bounds.
    pub show_default_annotations: bool,
    /// `hideCrossSectionBackground3D`.
    pub hide_cross_section_background_3d: bool,
    /// `wireFrame`: chunk outlines in cross-sections.
    pub wire_frame: bool,
    /// `enableAdaptiveDownsampling`.
    pub enable_adaptive_downsampling: bool,
    /// `prefetch`.
    pub prefetch: bool,
    /// `gpuMemoryLimit`, `systemMemoryLimit` (bytes), `concurrentDownloads`.
    pub gpu_memory_limit: f64,
    pub system_memory_limit: f64,
    pub concurrent_downloads: f64,
}

impl Default for ViewerSettings {
    fn default() -> Self {
        Self {
            title: None,
            show_default_annotations: true,
            hide_cross_section_background_3d: false,
            wire_frame: false,
            enable_adaptive_downsampling: true,
            prefetch: true,
            gpu_memory_limit: 1e9,
            system_memory_limit: 2e9,
            concurrent_downloads: 100.0,
        }
    }
}

impl ViewerSettings {
    fn from_state(o: &serde_json::Map<String, Value>) -> Self {
        let d = Self::default();
        let b = |k: &str, v: bool| o.get(k).and_then(Value::as_bool).unwrap_or(v);
        // Limits: positive integers (`verifyPositiveInt`), else the default.
        let n = |k: &str, v: f64| o.get(k).and_then(Value::as_f64).filter(|x| *x > 0.0 && x.fract() == 0.0).unwrap_or(v);
        Self {
            title: o.get("title").and_then(Value::as_str).map(str::to_owned),
            show_default_annotations: b("showDefaultAnnotations", d.show_default_annotations),
            hide_cross_section_background_3d: b("hideCrossSectionBackground3D", d.hide_cross_section_background_3d),
            wire_frame: b("wireFrame", d.wire_frame),
            enable_adaptive_downsampling: b("enableAdaptiveDownsampling", d.enable_adaptive_downsampling),
            prefetch: b("prefetch", d.prefetch),
            gpu_memory_limit: n("gpuMemoryLimit", d.gpu_memory_limit),
            system_memory_limit: n("systemMemoryLimit", d.system_memory_limit),
            concurrent_downloads: n("concurrentDownloads", d.concurrent_downloads),
        }
    }

    fn to_json(&self, key: &str) -> Option<Value> {
        let d = Self::default();
        let b = |v: bool, dv: bool| (v != dv).then_some(Value::Bool(v));
        let n = |v: f64, dv: f64| (v != dv).then(|| ng_core::js::number(v));
        match key {
            "title" => self.title.clone().map(Value::String),
            "showDefaultAnnotations" => b(self.show_default_annotations, d.show_default_annotations),
            "hideCrossSectionBackground3D" => b(self.hide_cross_section_background_3d, d.hide_cross_section_background_3d),
            "wireFrame" => b(self.wire_frame, d.wire_frame),
            "enableAdaptiveDownsampling" => b(self.enable_adaptive_downsampling, d.enable_adaptive_downsampling),
            "prefetch" => b(self.prefetch, d.prefetch),
            "gpuMemoryLimit" => n(self.gpu_memory_limit, d.gpu_memory_limit),
            "systemMemoryLimit" => n(self.system_memory_limit, d.system_memory_limit),
            "concurrentDownloads" => n(self.concurrent_downloads, d.concurrent_downloads),
            _ => None,
        }
    }
}

/// A segment's mesh: its layer, the layer's mesh source, the segment.
type MeshKey = (LayerId, usize, u64);

/// A precomputed annotation source's spatial index cell: the layer, the
/// level (finest first) and its grid position.
type AnnotationCellKey = (LayerId, usize, Vec<u64>);
/// A multi-resolution mesh's fragment: the segment's key and its octree row.
type FragmentKey = (LayerId, usize, u64, u32);

/// Where a mesh draw's vertices are.
#[derive(Clone, Copy)]
enum MeshRef {
    Whole(MeshKey),
    Fragment(FragmentKey),
}

/// A mesh draw: what, its model-view-projection, normal matrix, colour,
/// index range, and fragment box.
struct MeshJob {
    mesh: MeshRef,
    /// Its pick ID.
    pick: u32,
    mvp: [f32; 16],
    normal_matrix: [f32; 9],
    color: [f32; 4],
    silhouette: f32,
    indices: std::ops::Range<u32>,
    origin: [f32; 3],
    shape: [f32; 3],
}

pub struct Viewer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// The target's format (screenshots render to the same).
    target_format: wgpu::TextureFormat,
    /// A segment hovered outside the panels (a list entry:
    /// `segmentSelectionState.set`), by layer.
    segment_hover: Option<(usize, u64)>,
    /// `velocity`: each dimension's playback (by name), and the dimensions
    /// playing now.
    velocities: std::collections::BTreeMap<String, playback::DimensionVelocity>,
    /// Edits to the global space, in order (`Date.now()` timestamps: an
    /// edit outranks what bound spaces say).
    edit_counter: f64,
    playing: HashMap<usize, playback::Running>,
    /// A screenshot under way, and the region it crops to.
    screenshot: Option<screenshot::Job>,
    screenshot_region: (u32, u32, u32, u32),
    /// Physical pixels.
    width: u32,
    height: u32,
    scale_factor: f64,
    /// Where the data panels go, in logical pixels of the target (the rest
    /// belongs to a UI drawn over it); `None` for all of it.
    viewport: Option<[f64; 4]>,
    /// Whether Command stands in for Control (macOS).
    pub mac: bool,

    nav: Navigation,
    panels: Vec<(NamedAxes, SlicePanel)>,
    /// Each slice panel's estimate of how the position moves (prefetching).
    velocity: [prefetch::VelocityEstimator; 3],
    /// The global coordinate space (`dimensions`), merged from the layers.
    global: CoordinateSpace,
    combiner: CoordinateSpaceCombiner,
    /// Bound data sources, in the order they bound (the combiner's set).
    bind_order: Vec<(LayerId, usize)>,
    layers: Vec<ManagedLayer>,
    next_layer_id: LayerId,
    /// `crossSectionBackgroundColor`.
    pub background: [f32; 3],
    /// `layout`.
    pub layout: Layout,
    /// `showAxisLines`.
    pub show_axis_lines: bool,
    /// `showScaleBar`.
    pub show_scale_bar: bool,
    /// `showSlices`: cross-sections in the 3D panel.
    pub show_slices: bool,
    /// The rest of the viewer's settings (the Settings panel's).
    pub settings: ViewerSettings,
    /// `layout.orthographicProjection`.
    pub orthographic_projection: bool,
    /// `projectionBackgroundColor`.
    pub projection_background: [f32; 3],
    perspective: ng_render::perspective::PerspectiveRenderer,
    perspective_frame: Option<PanelFrame>,
    scale_bar_painter: Box<dyn ng_render::scale_bar::ScaleBarPainter>,
    scale_bar_renderer: ng_render::scale_bar::ScaleBarRenderer,
    /// Painted bars by (text, length in pixels).
    scale_bar_cache: HashMap<(String, u64), ng_render::scale_bar::ScaleBarImage>,
    /// The state last restored: keys this viewer does not act on are written
    /// back as they were.
    restored: serde_json::Map<String, Value>,
    /// Problems restoring the state.
    pub messages: Vec<String>,
    chunks: ChunkManager<ChunkKey, ChunkData>,
    /// Segments' meshes (layer, mesh source, segment), and those on the GPU
    /// (with the data they were made from).
    meshes: ChunkManager<MeshKey, ng_datasource::mesh::MeshData>,
    gpu_meshes: HashMap<MeshKey, (Arc<ng_datasource::mesh::MeshData>, ng_render::perspective::GpuMesh)>,
    /// Segments' skeletons (layer, skeleton source, segment); empty where
    /// the segment has none.
    skeletons: ChunkManager<MeshKey, ng_datasource::skeleton::SkeletonData>,
    /// Precomputed annotation sources' spatial index cells.
    annotation_cells: ChunkManager<AnnotationCellKey, Vec<annotations::Annotation>>,
    /// Precomputed annotations by ID (`by_id`), `None` where there is none.
    annotations_by_id: ChunkManager<(LayerId, u64), Option<annotations::Annotation>>,
    /// Those asked for since the last frame.
    annotation_wants: Vec<(LayerId, u64)>,
    /// Runs loads; here also wakes deferred past the caller.
    executor: Arc<dyn Executor>,
    /// Annotation shaders' styles by layer, view (3D or not) and what they
    /// were computed from.
    annotation_styles: HashMap<(LayerId, bool, u64), StyleSlot>,
    /// The histograms' hashed sample points, from the GPU (until read).
    hashes: Option<histogram::HashReadback>,
    /// The annotation hovered (`hoverState`): its layer and ID.
    annotation_hover: Option<(LayerId, String)>,
    /// This frame's pick IDs in the 3D panel (`pickIDs`): ID `k + 1` is
    /// entry `k`'s layer and segment.
    pick_ids: Vec<PickTarget>,
    /// Each slice panel's pickable marks as last drawn.
    slice_picks: [SlicePicks; 3],
    /// A pick window being read back, and the last one resolved.
    pending_pick: Option<PendingPick>,
    perspective_pick: Option<PerspectivePick>,
    wake: Arc<dyn Fn() + Send + Sync>,
    /// The frame whose skeletons were last counted in the layers' progress
    /// (each panel asks for them).
    skeletons_counted: u64,
    /// Multi-resolution meshes' manifests and fragments, the fragments on
    /// the GPU, and this frame's fragment reads (location, segment, byte
    /// range, quantization bits, whether partitioned).
    manifests: ChunkManager<MeshKey, ng_datasource::multiscale_mesh::Manifest>,
    fragments: ChunkManager<FragmentKey, ng_datasource::multiscale_mesh::Fragment>,
    gpu_fragments: HashMap<FragmentKey, (Arc<ng_datasource::multiscale_mesh::Fragment>, ng_render::perspective::GpuMesh)>,
    fragment_reads: HashMap<FragmentKey, (ng_kvstore::KvStoreLocation, String, u64, u64, u32, bool)>,
    compositor: Compositor,
    panel_compositor: Compositor,
    lines: ng_render::lines::LineRenderer,
    annotation_renderer: ng_render::annotations::AnnotationRenderer,
    axis_lines: ng_render::lines::AxisLineRenderer,
    flipper: ng_render::compose::Flipper,
    /// The annotation lines each slice panel drew last frame (physical
    /// pixels, panel-relative).
    last_lines: [Vec<ng_render::lines::Line>; 3],
    /// Per slice panel, sized to its physical rectangle.
    panel_frames: Vec<Option<PanelFrame>>,
    frame: u64,


    /// Logical pixels.
    /// `None` until the pointer first moves over the viewer (neuroglancer's
    /// `mouseState.active`): nothing is hovered before.
    pointer: Option<(f64, f64)>,
    drag: Option<Drag>,
    /// Scene mode ([`render_scene`](Self::render_scene)): the host's camera
    /// and frame for the frame being drawn, `None` otherwise.
    scene: Option<scene::SceneView>,
    /// The layout target scene mode draws into and discards.
    scene_scratch: Option<scene::Scratch>,
}

impl Viewer {
    /// `executor` runs chunk loads; `wake` is called (from any thread) when
    /// one finishes, to schedule a redraw.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        scale_factor: f64,
        executor: Arc<dyn Executor>,
        wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        let compositor = Compositor::new(&device, format);
        let panel_compositor = Compositor::panel(&device);
        let lines = ng_render::lines::LineRenderer::new(&device);
        let annotation_renderer = ng_render::annotations::AnnotationRenderer::new(&device);
        let axis_lines = ng_render::lines::AxisLineRenderer::new(&device);
        let scale_bar_renderer = ng_render::scale_bar::ScaleBarRenderer::new(&device);
        let flipper = ng_render::compose::Flipper::new(&device);
        let perspective = ng_render::perspective::PerspectiveRenderer::new(&device);
        // The GPU's hashes where it has compute shaders; the CPU's otherwise.
        let hashes = ng_render::capabilities::compute().then(|| histogram::HashReadback::start(&device, &queue, wake.clone()));
        Self {
            target_format: format,
            segment_hover: None,
            velocities: Default::default(),
            edit_counter: 0.0,
            playing: HashMap::new(),
            screenshot: None,
            screenshot_region: (0, 0, 0, 0),
            device,
            queue,
            width: width.max(1),
            height: height.max(1),
            scale_factor,
            viewport: None,
            mac: cfg!(target_os = "macos"),
            nav: Navigation::new(Arc::new(CoordinateSpace::empty_invalid())),
            panels: NamedAxes::ALL.iter().map(|&a| (a, SlicePanel::new(a))).collect(),
            velocity: std::array::from_fn(|_| prefetch::VelocityEstimator::new()),
            global: CoordinateSpace::empty_invalid(),
            combiner: CoordinateSpaceCombiner::new(is_global_dimension),
            bind_order: vec![],
            layers: vec![],
            next_layer_id: 1,
            background: CROSS_SECTION_BACKGROUND,
            layout: Layout::FourPanel,
            show_axis_lines: true,
            show_scale_bar: true,
            show_slices: true,
            settings: ViewerSettings::default(),
            orthographic_projection: false,
            projection_background: [0.0; 3],
            perspective,
            perspective_frame: None,
            scale_bar_painter: default_scale_bar_painter(),
            scale_bar_renderer,
            scale_bar_cache: HashMap::new(),
            restored: serde_json::Map::new(),
            messages: vec![],
            chunks: ChunkManager::new(executor.clone(), wake.clone()),
            meshes: ChunkManager::new(executor.clone(), wake.clone()),
            gpu_meshes: HashMap::new(),
            skeletons: ChunkManager::new(executor.clone(), wake.clone()),
            annotation_cells: ChunkManager::new(executor.clone(), wake.clone()),
            annotations_by_id: ChunkManager::new(executor.clone(), wake.clone()),
            annotation_wants: vec![],
            annotation_styles: HashMap::new(),
            hashes,
            skeletons_counted: 0,
            annotation_hover: None,
            pick_ids: vec![],
            slice_picks: Default::default(),
            pending_pick: None,
            perspective_pick: None,
            wake: wake.clone(),
            manifests: ChunkManager::new(executor.clone(), wake.clone()),
            fragments: ChunkManager::new(executor.clone(), wake.clone()),
            executor,
            gpu_fragments: HashMap::new(),
            fragment_reads: HashMap::new(),
            compositor,
            panel_compositor,
            lines,
            annotation_renderer,
            axis_lines,
            flipper,
            last_lines: Default::default(),
            panel_frames: (0..3).map(|_| None).collect(),
            frame: 0,
            pointer: None,
            drag: None,
            scene: None,
            scene_scratch: None,
        }
    }

    /// Back to an empty viewer (`root.reset()`).
    pub fn reset(&mut self) {
        self.layers.clear();
        self.bind_order.clear();
        self.global = CoordinateSpace::empty_invalid();
        self.combiner = CoordinateSpaceCombiner::new(is_global_dimension);
        self.nav = Navigation::new(Arc::new(CoordinateSpace::empty_invalid()));
        self.panels = NamedAxes::ALL.iter().map(|&a| (a, SlicePanel::new(a))).collect();
        self.background = CROSS_SECTION_BACKGROUND;
        self.layout = Layout::FourPanel;
        self.show_axis_lines = true;
        self.show_scale_bar = true;
        self.show_slices = true;
        self.settings = ViewerSettings::default();
        self.orthographic_projection = false;
        self.projection_background = [0.0; 3];
        self.restored.clear();
        self.messages.clear();
    }

    /// Restores viewer state JSON (`#!{…}`), as neuroglancer does from a
    /// reset viewer: `dimensions`, then the navigation keys, then `layers`,
    /// then `crossSectionBackgroundColor`. Returns the data sources to load;
    /// hand each to [`source_loaded`](Self::source_loaded). Problems that do
    /// not stop the rest are in [`messages`](Self::messages).
    pub fn restore_state(&mut self, state: &Value) -> anyhow::Result<Vec<PendingSource>> {
        let obj = state.as_object().ok_or_else(|| anyhow::anyhow!("Expected object, but received: {state}"))?;
        self.reset();
        self.restored = obj.clone();
        if let Some(dims) = obj.get("dimensions") {
            self.global = CoordinateSpace::from_json(Some(dims), false).map_err(|e| anyhow::anyhow!("Error restoring property \"dimensions\": {e:#}"))?;
            self.nav.set_coordinate_space(Arc::new(self.global.clone()));
        }
        self.nav.restore_state(obj)?;
        self.velocities = obj
            .get("velocity")
            .and_then(Value::as_object)
            .map(|o| o.iter().filter_map(|(k, v)| Some((k.clone(), playback::DimensionVelocity::from_json(v)?))).collect())
            .unwrap_or_default();
        self.playing.clear();
        let mut pending = vec![];
        if let Some(layers) = obj.get("layers") {
            let (specs, errors) = state::parse_layers(layers).map_err(|e| anyhow::anyhow!("Error restoring property \"layers\": {e:#}"))?;
            self.messages.extend(errors);
            for spec in specs {
                let id = self.new_layer(spec);
                let layer = self.layers.last().unwrap();
                for (index, slot) in layer.sources.iter().enumerate() {
                    pending.push(PendingSource { layer: id, index, url: slot.spec.url.clone() });
                }
            }
            // `markLoading`: dimensions stay while sources load.
            for _ in &pending {
                self.combiner.retain();
            }
        }
        if let Some(c) = obj.get("crossSectionBackgroundColor") {
            match c.as_str().and_then(state::parse_color) {
                Some(rgb) => self.background = rgb,
                None => self.messages.push(format!("unsupported crossSectionBackgroundColor {c}")),
            }
        }
        // `TrackableBoolean`: anything but a boolean gives the default.
        self.show_axis_lines = obj.get("showAxisLines").and_then(Value::as_bool).unwrap_or(true);
        self.show_scale_bar = obj.get("showScaleBar").and_then(Value::as_bool).unwrap_or(true);
        self.show_slices = obj.get("showSlices").and_then(Value::as_bool).unwrap_or(true);
        self.settings = ViewerSettings::from_state(obj);
        self.orthographic_projection =
            obj.get("layout").and_then(|l| l.get("orthographicProjection")).and_then(Value::as_bool).unwrap_or(false);
        if let Some(c) = obj.get("projectionBackgroundColor") {
            match c.as_str().and_then(state::parse_color) {
                Some(rgb) => self.projection_background = rgb,
                None => self.messages.push(format!("unsupported projectionBackgroundColor {c}")),
            }
        }
        if let Some(v) = obj.get("layout") {
            // A name, or `{"type": name, …}` (`DataPanelLayoutSpecification`);
            // an invalid name leaves the default (`TrackableValue`).
            let name = v.as_str().or_else(|| v.get("type").and_then(Value::as_str));
            match name.and_then(Layout::from_name) {
                Some(layout) => self.layout = layout,
                None => self.messages.push(format!("invalid layout {v}")),
            }
            if v.get("crossSections").is_some_and(|c| c.as_object().is_some_and(|o| !o.is_empty())) {
                self.messages.push("layout crossSections are not shown yet".into());
            }
        }
        Ok(pending)
    }

    fn new_layer(&mut self, spec: LayerSpec) -> LayerId {
        let id = self.next_layer_id;
        self.next_layer_id += 1;
        let sources = spec.sources.iter().map(|s| SourceSlot { spec: s.clone(), load: SourceLoad::Pending, set_manually: false }).collect();
        let mut messages = vec![];
        if !matches!(spec.layer_type.as_str(), "image" | "segmentation" | "annotation" | "auto" | "new") {
            messages.push(format!("{} layers are not supported yet", spec.layer_type));
        }
        self.layers.push(ManagedLayer { id, spec, sources, render: None, messages, progress: (0, 0), render_scale_histogram: Default::default(), mesh_render_scale_histogram: Default::default(), annotation_histograms: Default::default(), volume_rendering_histogram: ng_core::render_scale::RenderScaleHistogram::volume_rendering(), volume_chunks: vec![], histogram_wanted: false, data_histogram: None, histogram_samples: vec![], histogram_windows: vec![], window_histograms: vec![], legend_wants: vec![], legend_jobs: vec![], legends: vec![], shader_error: None, annotation: None, annotation_shader: None });
        id
    }

    /// A data source finished loading (or failed): bind its transform, merge
    /// dimensions, and set the layer up to draw.
    pub fn source_loaded(&mut self, layer: LayerId, index: usize, result: anyhow::Result<DataSource>) {
        let Some(li) = self.layers.iter().position(|l| l.id == layer) else { return };
        let Some(slot) = self.layers[li].sources.get_mut(index) else { return };
        if !matches!(slot.load, SourceLoad::Pending) {
            return;
        }
        match result {
            Ok(source) => {
                // The URL typed (or of a layer whose type is still to be
                // found) becomes the source's canonical one
                // (`originalCanonicalUrl`).
                let layer_type = self.layers[li].spec.layer_type.clone();
                let slot = &mut self.layers[li].sources[index];
                if (matches!(layer_type.as_str(), "auto" | "new") || slot.set_manually) && source.canonical_url != slot.spec.url {
                    slot.spec.url = source.canonical_url.clone();
                    let url = slot.spec.url.clone();
                    if let Some(spec) = self.layers[li].spec.sources.get_mut(index) {
                        spec.url = url;
                    }
                }
                let slot = &mut self.layers[li].sources[index];
                // `local://annotations` without a state transform: the
                // identity onto the global space as it is now
                // (`LocalDataSourceProvider.get`).
                let local = source.subsources.iter().any(|e| matches!(e.subsource, ng_datasource::model::Subsource::LocalAnnotations));
                let transform = if local && slot.spec.transform.is_none() {
                    identity_over(&self.global)
                } else {
                    VolumeLayer::default_transform(&source).with_spec(slot.spec.transform.as_ref())
                };
                slot.load = SourceLoad::Loaded(Box::new(LoadedSource { source, transform, binding: Binding::default() }));
                self.bind_order.push((layer, index));
            }
            Err(e) => {
                let msg = format!("{}: {e:#}", slot.spec.url);
                slot.load = SourceLoad::Failed;
                self.layers[li].messages.push(msg);
            }
        }
        self.combiner.release();
        self.update_dimensions();
        if let Err(e) = self.setup_render(li) {
            let layer = &mut self.layers[li];
            layer.messages.push(format!("{e:#}"));
        }
    }

    /// Runs the global combiner over the bound sources and passes the result
    /// on: to navigation, and to each layer's transform.
    fn update_dimensions(&mut self) {
        let order = self.bind_order.clone();
        {
            let mut slots: Vec<(LayerId, usize, &mut SourceSlot)> = vec![];
            for layer in &mut self.layers {
                let id = layer.id;
                for (i, slot) in layer.sources.iter_mut().enumerate() {
                    slots.push((id, i, slot));
                }
            }
            let mut by_order: Vec<(usize, &mut Binding, &mut dyn BoundSpace)> = vec![];
            for (id, i, slot) in slots {
                if let SourceLoad::Loaded(loaded) = &mut slot.load {
                    let LoadedSource { transform, binding, .. } = &mut **loaded;
                    if let Some(rank) = order.iter().position(|&o| o == (id, i)) {
                        by_order.push((rank, binding, transform as &mut dyn BoundSpace));
                    }
                }
            }
            by_order.sort_by_key(|(r, _, _)| *r);
            let mut bound: Vec<(&mut Binding, &mut dyn BoundSpace)> = by_order.into_iter().map(|(_, b, t)| (b, t)).collect();
            self.combiner.update(&mut self.global, &mut bound);
        }
        if **self.nav.coordinate_space() != self.global {
            self.nav.set_coordinate_space(Arc::new(self.global.clone()));
        }
        let global = self.global.clone();
        for li in 0..self.layers.len() {
            if self.layers[li].annotation.is_some() {
                self.setup_annotations(li);
            }
        }
        for layer in &mut self.layers {
            let Some(render) = &mut layer.render else { continue };
            if let SourceLoad::Loaded(loaded) = &layer.sources[render.source_index].load {
                render.layer.set_transform(loaded.transform.clone(), &global);
            }
        }
    }

    /// An annotation layer's local annotations, once their source has
    /// loaded: its transform (the global space's identity if the state names
    /// none, as `LocalDataSourceProvider` makes it) and the parsed list.
    // (Shared with the loading tasks: across threads natively; on wasm,
    // where the kvstore is not `Send`, on the one thread.)
    #[allow(clippy::arc_with_non_send_sync)]
    fn setup_annotations(&mut self, li: usize) {
        let global = self.global.clone();
        let layer = &mut self.layers[li];
        if !matches!(layer.spec.layer_type.as_str(), "annotation" | "auto") {
            return;
        }
        // The local annotations, or a precomputed collection's.
        let Some((index, transform, spatial)) = layer.sources.iter().enumerate().find_map(|(i, s)| match &s.load {
            SourceLoad::Loaded(l) => l.source.subsources.iter().find_map(|e| match &e.subsource {
                ng_datasource::model::Subsource::LocalAnnotations => Some((i, l.transform.clone(), None)),
                ng_datasource::model::Subsource::Annotations(info) => Some((i, l.transform.clone(), Some(Arc::new((**info).clone())))),
                _ => None,
            }),
            _ => None,
        }) else {
            return;
        };
        if layer.spec.layer_type == "auto" {
            layer.spec.layer_type = "annotation".into();
        }
        let rank = transform.source_rank;
        let specs = annotations::parse_property_specs(layer.spec.json.get("annotationProperties"));
        let annotations = if spatial.is_some() { vec![] } else { annotations::parse_annotations_with(layer.spec.json.get("annotations"), rank, &specs) };
        layer.annotation = Some(AnnotationLayerState { source_index: index, layer: VolumeLayer::without_volume(transform, &global), annotations, spatial });
    }

    /// Sets up drawing for layer `li` once a volume source has loaded.
    fn setup_render(&mut self, li: usize) -> anyhow::Result<()> {
        self.setup_annotations(li);
        let global = self.global.clone();
        let layer = &mut self.layers[li];
        if layer.render.is_some() {
            return Ok(());
        }
        let found = layer.sources.iter().enumerate().find_map(|(i, s)| match &s.load {
            SourceLoad::Loaded(l) if l.source.volume().is_some() => Some((i, &l.source, &l.transform)),
            _ => None,
        });
        let (source_index, mut volume) = match found {
            Some((i, source, transform)) => (i, VolumeLayer::new(source, transform.clone(), &global)?),
            None => {
                // Objects only (`precomputed://…` of meshes or skeletons): a
                // segmentation layer without a volume.
                let objects = layer.sources.iter().enumerate().find_map(|(i, s)| match &s.load {
                    SourceLoad::Loaded(l) if l.source.subsources.iter().any(|e| matches!(e.subsource, ng_datasource::model::Subsource::Mesh(_) | ng_datasource::model::Subsource::Skeleton(_))) => Some((i, &l.transform)),
                    _ => None,
                });
                match objects {
                    Some((i, transform)) if matches!(layer.spec.layer_type.as_str(), "segmentation" | "auto") => {
                        let mut v = VolumeLayer::without_volume(transform.clone(), &global);
                        v.volume_type = VolumeType::Segmentation;
                        (i, v)
                    }
                    _ => return Ok(()),
                }
            }
        };
        let mut visible = None;
        let kind = match layer.spec.layer_type.as_str() {
            "image" => LayerKind::Image { opacity: 0.0 },
            "segmentation" => LayerKind::Segmentation(SegmentationDisplay::default()),
            "auto" => match volume.volume_type {
                VolumeType::Segmentation => LayerKind::Segmentation(SegmentationDisplay::default()),
                _ => LayerKind::Image { opacity: 0.0 },
            },
            _ => return Ok(()),
        };
        let kind = match kind {
            LayerKind::Image { .. } => {
                let (opacity, range, warning) = state::image_settings(&layer.spec.json, volume.data_type);
                volume.range = range;
                layer.messages.extend(warning);
                LayerKind::Image { opacity }
            }
            LayerKind::Segmentation(_) => {
                let (display, segments) = state::segmentation_settings(&layer.spec.json)?;
                let ids: Vec<u64> = segments.iter().filter(|&&(_, visible)| visible).map(|&(id, _)| id).collect();
                let mut set = VisibleSet::new(&self.device);
                set.set(&self.device, &self.queue, &ids);
                visible = Some((set, 0));
                LayerKind::Segmentation(display)
            }
        };
        let pipelines = match kind {
            LayerKind::Image { .. } => {
                let (sample, color) = ng_render::image::color_wgsl(volume.data_type);
                LayerPipelines::new(&self.device, sample, &color, false)
            }
            LayerKind::Segmentation(_) => LayerPipelines::new(
                &self.device,
                ng_render::segmentation::sample(),
                ng_render::segmentation::SEGMENTATION_WGSL,
                true,
            ),
        };
        let mut state = LayerState { source_index, layer: volume, kind, pipelines, textures: HashMap::new(), visible, shader: None, volume: None };
        layer.shader_error = state.update_shader(&self.device, &layer.spec.json);
        layer.render = Some(state);
        Ok(())
    }

    /// Sets (or, with `None`, removes) a member of layer `layer`'s JSON, and
    /// applies the display settings it holds: opacity, the `normalized`
    /// range, segment colours and alphas, render scale.
    pub fn set_layer_option(&mut self, layer: usize, key: &str, value: Option<Value>) {
        let Some(l) = self.layers.get_mut(layer) else { return };
        match key.split_once('.') {
            // A member of an object (`skeletonRendering.mode2d`): the object
            // left out once empty, as its `toJSON` does.
            Some((outer, inner)) => {
                let mut o = l.spec.json.get(outer).and_then(Value::as_object).cloned().unwrap_or_default();
                match value {
                    Some(v) => {
                        o.insert(inner.to_owned(), v);
                    }
                    None => {
                        o.remove(inner);
                    }
                }
                if o.is_empty() {
                    l.spec.json.remove(outer);
                } else {
                    l.spec.json.insert(outer.to_owned(), Value::Object(o));
                }
            }
            None => match value {
                Some(v) => {
                    l.spec.json.insert(key.to_owned(), v);
                }
                None => {
                    l.spec.json.remove(key);
                }
            },
        }
        let json = l.spec.json.clone();
        if let (Some(render), "shader" | "shaderControls") = (&mut l.render, key) {
            l.shader_error = render.update_shader(&self.device, &json);
        }
        if let Some(render) = &mut l.render {
            match &mut render.kind {
                LayerKind::Image { opacity } => {
                    let (o, range, _) = state::image_settings(&json, render.layer.data_type);
                    *opacity = o;
                    render.layer.range = range;
                }
                LayerKind::Segmentation(display) => {
                    if let Ok((d, _)) = state::segmentation_settings(&json) {
                        *display = d;
                    }
                }
            }
        }
    }

    /// `ManagedUserLayer.setArchived`: an archived layer is hidden and out
    /// of the layer bar, but stays in the state (and its sources bound).
    pub fn set_archived(&mut self, layer: usize, archived: bool) {
        let Some(l) = self.layers.get_mut(layer) else { return };
        if l.spec.archived == archived {
            return;
        }
        if archived {
            l.spec.visible = false;
        }
        l.spec.archived = archived;
    }

    /// Layer `from` moved to index `to` of the list without it (the layer
    /// bar's drag; `managedLayers.splice`).
    pub fn move_layer(&mut self, from: usize, to: usize) {
        if from >= self.layers.len() {
            return;
        }
        let layer = self.layers.remove(from);
        let to = to.min(self.layers.len());
        self.layers.insert(to, layer);
    }

    /// A copy of layer `i` (Shift+drag): its state, named uniquely
    /// (`getUniqueLayerName`), inserted at `at`. Returns its sources to load.
    pub fn copy_layer(&mut self, i: usize, at: usize) -> Vec<PendingSource> {
        let Some(json) = self.layers.get(i).map(|l| l.to_json()) else { return vec![] };
        let Ok((mut specs, _)) = state::parse_layers(&Value::Array(vec![json])) else { return vec![] };
        let Some(mut spec) = specs.pop() else { return vec![] };
        let base = spec.name.clone();
        let mut suffix = 0;
        while self.layers.iter().any(|l| l.name() == spec.name) {
            suffix += 1;
            spec.name = format!("{base}{suffix}");
        }
        let id = self.new_layer(spec);
        let li = self.layers.len() - 1;
        self.move_layer(li, at);
        let layer = self.layers.iter().find(|l| l.id == id).unwrap();
        let pending: Vec<PendingSource> = layer.sources.iter().enumerate().map(|(index, slot)| PendingSource { layer: id, index, url: slot.spec.url.clone() }).collect();
        for _ in &pending {
            self.combiner.retain();
        }
        pending
    }

    /// `addNewLayer`: a layer of type `new` named `new layer` (made
    /// unique), without sources. Returns its index.
    pub fn add_new_layer(&mut self) -> usize {
        let mut name = "new layer".to_owned();
        let mut suffix = 0;
        while self.layers.iter().any(|l| l.name() == name) {
            suffix += 1;
            name = format!("new layer{suffix}");
        }
        let spec = state::LayerSpec { name, layer_type: "new".into(), visible: true, archived: false, sources: vec![], json: Default::default() };
        self.new_layer(spec);
        self.layers.len() - 1
    }

    /// A layer's data source set to `url` (none if empty), replacing any;
    /// returns it to load (as [`restore_state`](Self::restore_state) does).
    pub fn set_layer_source(&mut self, layer: usize, url: &str) -> Vec<PendingSource> {
        let Some(l) = self.layers.get_mut(layer) else { return vec![] };
        l.spec.sources = if url.is_empty() {
            vec![]
        } else {
            vec![state::SourceSpec { url: url.to_owned(), transform: None, enable_default_subsources: true, subsources: Default::default() }]
        };
        l.sources = l.spec.sources.iter().map(|s| SourceSlot { spec: s.clone(), load: SourceLoad::Pending, set_manually: true }).collect();
        l.render = None;
        l.messages.clear();
        let id = l.id;
        let pending: Vec<PendingSource> = l.sources.iter().enumerate().map(|(index, slot)| PendingSource { layer: id, index, url: slot.spec.url.clone() }).collect();
        for _ in &pending {
            self.combiner.retain();
        }
        pending
    }

    /// `NewUserLayer.detectedLayerConstructor`: the type its loaded data
    /// suggests (`image` or `segmentation`).
    pub fn detected_layer_type(&self, layer: usize) -> Option<&'static str> {
        self.layers.get(layer)?.sources.iter().find_map(|s| match &s.load {
            SourceLoad::Loaded(l) => l.source.volume().map(|v| if matches!(v.volume_type, VolumeType::Segmentation) { "segmentation" } else { "image" }),
            _ => None,
        })
    }

    /// `changeLayerType`: the layer redone as `layer_type` from its JSON.
    pub fn change_layer_type(&mut self, layer: usize, layer_type: &str) {
        let Some(l) = self.layers.get_mut(layer) else { return };
        l.spec.layer_type = layer_type.to_owned();
        l.render = None;
        l.messages.retain(|m| !m.ends_with("layers are not supported yet"));
        if let Err(e) = self.setup_render(layer) {
            self.layers[layer].messages.push(format!("{e:#}"));
        }
    }

    /// `getRenameValidity`: each name valid (`/^[a-zA-Z][a-zA-Z_0-9]*['^]?$/`),
    /// unique, and (if new) not one a bound space already uses.
    pub fn dimension_rename_validity(&self, names: &[String]) -> Vec<bool> {
        let bound: Vec<String> = self
            .layers
            .iter()
            .flat_map(|l| l.sources.iter())
            .filter_map(|s| match &s.load {
                SourceLoad::Loaded(l) => Some(l.transform.space().names.clone()),
                _ => None,
            })
            .flatten()
            .collect();
        names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                ng_core::coordinate_space::is_valid_dimension_name(n, false)
                    && !names.iter().enumerate().any(|(j, m)| j != i && m == n)
                    && (self.global.names.contains(n) || !bound.contains(n))
            })
            .collect()
    }

    /// `PositionWidget.updateNames`: the global dimensions renamed (and the
    /// layers' transforms after them). `false` if a name is not valid or
    /// nothing changed.
    pub fn rename_dimensions(&mut self, names: &[String]) -> bool {
        if names.len() != self.global.rank() || self.dimension_rename_validity(names).contains(&false) || names == self.global.names.as_slice() {
            return false;
        }
        self.edit_counter += 1.0;
        let old = self.global.names.clone();
        let mut space = self.global.clone();
        for (i, n) in names.iter().enumerate() {
            if old[i] != *n {
                space.timestamps[i] = self.edit_counter;
            }
        }
        space.names = names.to_vec();
        self.global = space;
        self.update_dimensions();
        // What is keyed by name follows: playback, the dimension tools.
        for (o, n) in old.iter().zip(names) {
            if o != n {
                if let Some(v) = self.velocities.remove(o) {
                    self.velocities.insert(n.clone(), v);
                }
            }
        }
        self.rename_dimension_tools(&old, names);
        true
    }

    /// `PositionWidget.updateScales`: each dimension's scale and unit
    /// (`parseScale`) set. `false` if one does not parse or nothing changed.
    pub fn set_dimension_scales(&mut self, texts: &[String]) -> bool {
        if texts.len() != self.global.rank() {
            return false;
        }
        let Some(parsed) = texts.iter().map(|t| ng_core::si_units::parse_scale(t)).collect::<Option<Vec<_>>>() else { return false };
        let mut space = self.global.clone();
        let mut changed = false;
        self.edit_counter += 1.0;
        for (i, (scale, unit)) in parsed.into_iter().enumerate() {
            if space.scales[i] != scale || space.units[i] != unit {
                space.scales[i] = scale;
                space.units[i] = unit.to_owned();
                space.timestamps[i] = self.edit_counter;
                changed = true;
            }
        }
        if !changed {
            return false;
        }
        self.global = space;
        self.update_dimensions();
        true
    }

    /// `changeLayerName`: renames a layer, made unique
    /// (`getUniqueLayerName`: `name`, `name1`, `name2` …). Returns the name
    /// it got, or `None` if unchanged.
    pub fn rename_layer(&mut self, layer: usize, name: &str) -> Option<String> {
        if self.layers.get(layer)?.name() == name {
            return None;
        }
        let mut unique = name.to_owned();
        let mut suffix = 0;
        while self.layers.iter().any(|l| l.name() == unique) {
            suffix += 1;
            unique = format!("{name}{suffix}");
        }
        self.layers[layer].spec.name = unique.clone();
        Some(unique)
    }

    /// A segmentation layer's visible segments changed in its spec
    /// (`segments`): rebuilds the set it draws with.
    pub fn refresh_segments(&mut self, layer: usize) {
        let (device, queue) = (self.device.clone(), self.queue.clone());
        let Some(l) = self.layers.get_mut(layer) else { return };
        let Ok((_, segments)) = state::segmentation_settings(&l.spec.json) else { return };
        let Some(render) = l.render.as_mut() else { return };
        if let Some((set, _)) = &mut render.visible {
            let ids: Vec<u64> = segments.iter().filter(|s| s.1).map(|s| s.0).collect();
            set.set(&device, &queue, &ids);
        }
    }

    /// `recolor` on every segmentation layer
    /// (`segmentColorHash.randomize()`): a new colour seed.
    pub fn recolor(&mut self, seed: impl Fn() -> u32) {
        for l in &mut self.layers {
            if let Some(LayerKind::Segmentation(display)) = l.kind_mut() {
                let s = seed();
                display.color_seed = s;
                l.spec.json.insert("colorSeed".into(), s.into());
            }
        }
    }

    /// `clear-segments` on every segmentation layer that picks: no
    /// segments visible.
    pub fn clear_segments(&mut self) {
        for i in 0..self.layers.len() {
            let l = &mut self.layers[i];
            let picks = l.spec.json.get("pick").and_then(Value::as_bool).unwrap_or(true);
            if picks && matches!(l.kind(), Some(LayerKind::Segmentation(_))) {
                l.spec.json.remove("segments");
                self.refresh_segments(i);
            }
        }
    }

    /// `deleteLayer`: removes the layer, and its data sources from the
    /// global coordinate space.
    pub fn remove_layer(&mut self, layer: usize) {
        if layer >= self.layers.len() {
            return;
        }
        let removed = self.layers.remove(layer);
        self.bind_order.retain(|&(id, _)| id != removed.id);
        // Sources still loading held the dimensions (`markLoading`).
        for slot in &removed.sources {
            if matches!(slot.load, SourceLoad::Pending) {
                self.combiner.release();
            }
        }
        self.update_dimensions();
    }

    /// Shows `source` as the only layer (an `auto` layer named after it).
    pub fn add_layer(&mut self, source: DataSource) -> anyhow::Result<()> {
        self.reset();
        let url = source.canonical_url.clone();
        let name = url.trim_end_matches('/').rsplit(['/', ':']).next().filter(|s| !s.is_empty()).unwrap_or("layer").to_owned();
        let spec = LayerSpec {
            name,
            layer_type: "auto".into(),
            visible: true,
            archived: false,
            sources: vec![SourceSpec { url, transform: None, enable_default_subsources: true, subsources: Default::default() }],
            json: serde_json::Map::new(),
        };
        let id = self.new_layer(spec);
        self.combiner.retain();
        self.source_loaded(id, 0, Ok(source));
        let layer = self.layers.last().unwrap();
        match layer.messages.first() {
            Some(m) if layer.render.is_none() => anyhow::bail!("{m}"),
            _ if layer.render.is_none() => anyhow::bail!("the data source has no volume"),
            _ => Ok(()),
        }
    }

    /// Opens the data source at `url` and shows it. Natively this blocks while
    /// the metadata loads; in the browser use [`ng_datasource::url::get`] and
    /// [`add_layer`](Self::add_layer).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(&mut self, url: &str) -> anyhow::Result<()> {
        let source = futures::executor::block_on(ng_datasource::url::get(url))?;
        self.add_layer(source)
    }

    /// Restores `state` and loads its data sources, blocking (native).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_state(&mut self, state: &Value) -> anyhow::Result<()> {
        let pending = self.restore_state(state)?;
        for p in pending {
            let result = futures::executor::block_on(ng_datasource::url::get(&p.url));
            self.source_loaded(p.layer, p.index, result);
        }
        Ok(())
    }

    /// Takes in finished chunk loads; returns whether any arrived (redraw).
    pub fn poll(&mut self) -> bool {
        self.chunks.poll() | self.meshes.poll() | self.skeletons.poll() | self.annotation_cells.poll() | self.annotations_by_id.poll() | self.manifests.poll() | self.fragments.poll() | self.poll_pick() | self.poll_styles()
    }

    /// Takes in annotation styles read back; whether any arrived.
    fn poll_styles(&mut self) -> bool {
        let mut pending = false;
        let mut any = false;
        if let Some(h) = &self.hashes {
            if h.poll() {
                self.hashes = None;
                any = true;
            } else {
                pending = true;
            }
        }
        for slot in self.annotation_styles.values_mut() {
            if let StyleState::Pending(r) = &slot.state {
                match r.read() {
                    Some(styles) => {
                        slot.state = StyleState::Ready(Arc::new(styles));
                        any = true;
                    }
                    None => pending = true,
                }
            }
        }
        // The mapping completes as the device is polled: natively and on
        // WebGL2 only then (a browser's WebGPU completes it by itself).
        // Asked again until it does.
        if pending {
            let _ = self.device.poll(wgpu::PollType::Poll);
            (self.wake)();
        }
        any
    }

    /// `completePickRequest`: once the pick window is read, the position
    /// under the pointer (from the nearest drawn sample's depth, through
    /// the inverse view-projection) and the object there. Whether that
    /// changed.
    fn poll_pick(&mut self) -> bool {
        let Some(pending) = self.pending_pick.as_ref().filter(|p| p.mapped) else { return false };
        // The mapping completes as the device is polled (natively, WebGL2).
        let _ = self.device.poll(wgpu::PollType::Poll);
        let Some((z, pick)) = pending.readback.read() else {
            // Not yet: asked again.
            (self.wake)();
            return false;
        };
        let pending = self.pending_pick.take().unwrap();
        let r = &pending.readback;
        let result = ng_render::pick::resolve_nearest(&z, &pick, r.radius).map(|sample| {
            let (w, h) = (r.size.0 as f64, r.size.1 as f64);
            let ndc = [
                2.0 * (r.x + sample.relative_x as i64 - r.radius as i64) as f64 / w - 1.0,
                2.0 * (r.y + sample.relative_y as i64 - r.radius as i64) as f64 / h - 1.0,
                2.0 * (1.0 - sample.depth as f64) - 1.0,
            ];
            let p = ng_core::glm::vec4_transform_mat4([ndc[0], ndc[1], ndc[2], 1.0], &pending.inv_vp);
            let mut position = pending.position.clone();
            for (k, &g) in pending.display.iter().enumerate() {
                if g >= 0 {
                    position[g as usize] = (p[k] / p[3]) as f32;
                }
            }
            let picked = (sample.pick as usize).checked_sub(1).and_then(|k| pending.ids.get(k).cloned());
            PerspectivePick { position, picked }
        });
        let changed = result != self.perspective_pick;
        self.perspective_pick = result;
        changed
    }

    /// Whether chunks are still loading.
    pub fn is_loading(&self) -> bool {
        self.chunks.in_flight() + self.meshes.in_flight() + self.skeletons.in_flight() + self.annotation_cells.in_flight() + self.annotations_by_id.in_flight() + self.manifests.in_flight() + self.fragments.in_flight() > 0
    }

    /// `chunkQueueManager.getStatistics()`: each chunk source with chunks,
    /// in layer and scale order, its identifying properties and its visible
    /// chunks' counts.
    /// The volume chunks wanted this frame: layer index, scale and grid
    /// position (what the statistics count as visible).
    pub fn visible_chunks(&self) -> Vec<(usize, usize, Vec<u64>)> {
        let order: HashMap<LayerId, usize> = self.layers.iter().enumerate().map(|(i, l)| (l.id, i)).collect();
        let mut out = vec![];
        self.chunks.for_each(|(id, scale, grid), _, visible, _, _| {
            if let (true, Some(&li)) = (visible, order.get(id)) {
                out.push((li, *scale, grid.clone()));
            }
        });
        out.sort();
        out
    }

    pub fn chunk_statistics(&self) -> Vec<SourceStatistics> {
        let mut by_source: std::collections::BTreeMap<(usize, usize), SourceStatistics> = Default::default();
        let order: HashMap<LayerId, usize> = self.layers.iter().enumerate().map(|(i, l)| (l.id, i)).collect();
        let mut latencies: std::collections::BTreeMap<(usize, usize), (f64, u32)> = Default::default();
        self.chunks.for_each(|(id, scale, _), state, visible, bytes, latency| {
            let Some(&li) = order.get(id) else { return };
            let entry = by_source.entry((li, *scale)).or_insert_with(|| {
                let properties = self.layers[li]
                    .render
                    .as_ref()
                    .and_then(|r| r.layer.scales.get(*scale))
                    .map(|s| source_properties(&s.params))
                    .unwrap_or_default();
                SourceStatistics { properties, ..Default::default() }
            });
            if let Some(ms) = latency {
                let l = latencies.entry((li, *scale)).or_default();
                l.0 += ms;
                l.1 += 1;
            }
            if !visible {
                return;
            }
            entry.visible_total += 1;
            match state {
                State::Queued => {}
                State::Downloading => entry.downloading += 1,
                // Drawn chunks are on the GPU.
                State::Ready(_) => {
                    entry.gpu += 1;
                    entry.gpu_bytes += bytes;
                }
                State::Failed(_) => entry.failed += 1,
            }
        });
        for (k, (sum, n)) in latencies {
            if let Some(s) = by_source.get_mut(&k) {
                s.download_latency = Some(sum / n as f64);
            }
        }
        by_source.into_values().collect()
    }

    /// Times chunk downloads by `clock` (milliseconds), for their latency.
    pub fn set_clock(&mut self, clock: fn() -> f64) {
        self.chunks.clock = Some(clock);
    }

    /// Dimension `dim`'s playback, if enabled.
    pub fn velocity(&self, dim: usize) -> Option<playback::DimensionVelocity> {
        self.velocities.get(self.global.names.get(dim)?).copied()
    }

    /// Enables (with `Some`), changes or disables dimension `dim`'s
    /// playback.
    pub fn set_velocity(&mut self, dim: usize, velocity: Option<playback::DimensionVelocity>) {
        let Some(name) = self.global.names.get(dim).cloned() else { return };
        match velocity {
            Some(v) => {
                self.velocities.insert(name, v);
            }
            None => {
                self.velocities.remove(&name);
            }
        }
    }

    /// `PlaybackManager.updateStarted`: advances the playing dimensions to
    /// time `now` (ms); a dimension moved otherwise since stops. Returns
    /// whether any plays (draw again).
    pub fn step_playback(&mut self, now: f64) -> bool {
        let rank = self.global.rank();
        let mut position = self.nav.position().to_vec();
        let mut moved = false;
        let mut any = false;
        for d in 0..rank.min(position.len()) {
            let Some(name) = self.global.names.get(d).cloned() else { continue };
            let Some(v) = self.velocities.get(&name).copied() else {
                self.playing.remove(&d);
                continue;
            };
            if v.paused || v.velocity == 0.0 {
                self.playing.remove(&d);
                continue;
            }
            any = true;
            let run = *self.playing.entry(d).or_insert(playback::Running { prev_time: now, prev_coordinate: position[d] });
            if position[d].floor() != run.prev_coordinate.floor() {
                self.velocities.insert(name, playback::DimensionVelocity { paused: true, ..v });
                self.playing.remove(&d);
                continue;
            }
            let (lower, upper) = (self.global.bounds.lower_bounds[d], self.global.bounds.upper_bounds[d]);
            let (c, v2) = playback::step(position[d], v, now - run.prev_time, lower, upper);
            if v2 != v {
                self.velocities.insert(name, v2);
            }
            if c != position[d] {
                position[d] = c;
                moved = true;
            }
            self.playing.insert(d, playback::Running { prev_time: now, prev_coordinate: c });
        }
        if moved {
            self.nav.set_position(&position);
        }
        any
    }

    /// Volume chunks requested in the last frame ahead of need (prefetched
    /// for where the position is heading).
    pub fn prefetching_chunks(&self) -> usize {
        self.chunks.prefetching()
    }

    /// Chunks downloading now.
    pub fn loading_chunks(&self) -> usize {
        self.chunks.in_flight()
    }

    pub fn has_layer(&self) -> bool {
        self.layers.iter().any(|l| l.render.is_some())
    }

    /// Paints scale bars (the browser shell passes one using the 2D canvas,
    /// as neuroglancer does).
    pub fn set_scale_bar_painter(&mut self, painter: Box<dyn ng_render::scale_bar::ScaleBarPainter>) {
        self.scale_bar_painter = painter;
        self.scale_bar_cache.clear();
    }

    /// The annotation lines (data bounds) panel `axes` drew last frame, in
    /// physical pixels relative to the panel.
    pub fn annotation_lines(&self, axes: NamedAxes) -> &[ng_render::lines::Line] {
        &self.last_lines[NamedAxes::ALL.iter().position(|&a| a == axes).unwrap()]
    }

    pub fn layers(&self) -> &[ManagedLayer] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [ManagedLayer] {
        &mut self.layers
    }

    /// The global coordinate space.
    pub fn dimensions(&self) -> &CoordinateSpace {
        &self.global
    }

    /// The viewer state as neuroglancer writes it (`toJSON`): its keys in
    /// neuroglancer's order, with what this viewer holds (dimensions,
    /// navigation, layers, background, layout) from its current state, and
    /// the other keys of the restored state as they were. Legacy keys
    /// (`navigation`, …) are not written, as in neuroglancer.
    pub fn state_json(&mut self) -> Value {
        let mut nav = self.nav.to_json();
        let mut out = serde_json::Map::new();
        for &key in state::VIEWER_STATE_KEYS {
            let value = match key {
                "dimensions" => self.global.to_json(),
                "relativeDisplayScales" | "displayDimensions" | "position" | "crossSectionOrientation"
                | "crossSectionScale" | "crossSectionDepth" | "projectionOrientation" | "projectionScale"
                | "projectionDepth" => nav.remove(key),
                "layers" => Some(Value::Array(self.layers.iter().map(|l| l.to_json()).collect())),
                // `CoordinateSpacePlaybackVelocity.toJSON`: by dimension, in order.
                "velocity" => {
                    let o: serde_json::Map<String, Value> =
                        self.global.names.iter().filter_map(|n| Some((n.clone(), self.velocities.get(n)?.to_json()))).collect();
                    (!o.is_empty()).then_some(Value::Object(o))
                }
                "showAxisLines" => (!self.show_axis_lines).then_some(Value::Bool(false)),
                "showScaleBar" => (!self.show_scale_bar).then_some(Value::Bool(false)),
                "showSlices" => (!self.show_slices).then_some(Value::Bool(false)),
                "projectionBackgroundColor" => {
                    (self.projection_background != [0.0; 3]).then(|| Value::String(state::serialize_color(self.projection_background)))
                }
                "title" | "showDefaultAnnotations" | "hideCrossSectionBackground3D" | "wireFrame" | "enableAdaptiveDownsampling"
                | "prefetch" | "gpuMemoryLimit" | "systemMemoryLimit" | "concurrentDownloads" => self.settings.to_json(key),
                "crossSectionBackgroundColor" => {
                    (self.background != CROSS_SECTION_BACKGROUND).then(|| Value::String(state::serialize_color(self.background)))
                }
                "layout" => Some(match self.restored.get("layout") {
                    // An object form (cross sections, projection) keeps its
                    // other members.
                    Some(Value::Object(o)) => {
                        let mut o = o.clone();
                        o.insert("type".into(), self.layout.name().into());
                        Value::Object(o)
                    }
                    _ => Value::String(self.layout.name()),
                }),
                // A `CompoundTrackable`: `{}` when all its members are defaults.
                "uiControlVisibility" => Some(self.restored.get(key).cloned().unwrap_or_else(|| Value::Object(Default::default()))),
                _ => self.restored.get(key).cloned(),
            };
            if let Some(v) = value {
                out.insert(key.to_owned(), v);
            }
        }
        Value::Object(out)
    }

    /// A state key the viewer keeps but does not act on (the UI's: panel
    /// locations, `selectedLayer`, `toolPalettes` …), as restored or set.
    pub fn ui_state(&self, key: &str) -> Option<&Value> {
        self.restored.get(key)
    }

    /// Sets (or, with `None`, removes) such a key; it is written back with
    /// the state.
    pub fn set_ui_state(&mut self, key: &str, value: Option<Value>) {
        match value {
            Some(v) => {
                self.restored.insert(key.to_owned(), v);
            }
            None => {
                self.restored.remove(key);
            }
        }
    }

    /// The state as a share-link fragment, `#!…` (`encodeStateAsFragment`).
    pub fn state_fragment(&mut self) -> String {
        format!("#!{}", ng_core::json::encode_state_as_fragment(&self.state_json()))
    }

    pub fn navigation(&self) -> &Navigation {
        &self.nav
    }

    pub fn navigation_mut(&mut self) -> &mut Navigation {
        &mut self.nav
    }

    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f64) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.scale_factor = scale_factor;
    }

    /// Confines the data panels to `rect` (x, y, width, height in logical
    /// pixels of the target), as neuroglancer's panels sit below its top bar
    /// and layer bar; `None` gives them the whole target. Pointer positions
    /// stay relative to the target. Returns whether it changed.
    pub fn set_viewport(&mut self, rect: Option<[f64; 4]>) -> bool {
        let changed = self.viewport != rect;
        self.viewport = rect;
        changed
    }

    pub fn viewport(&self) -> Option<[f64; 4]> {
        self.viewport
    }

    /// The data panels' region in logical pixels.
    fn logical_region(&self) -> [f64; 4] {
        let s = self.scale_factor;
        self.viewport.unwrap_or([0.0, 0.0, self.width as f64 / s, self.height as f64 / s])
    }

    /// The data panels' region's size in logical pixels.
    fn logical_size(&self) -> (u32, u32) {
        let [_, _, w, h] = self.logical_region();
        (w.round().max(1.0) as u32, h.round().max(1.0) as u32)
    }

    /// The data panels' region in physical pixels: its offset and size.
    fn physical_region(&self) -> (u32, u32, u32, u32) {
        let s = self.scale_factor;
        let [x, y, w, h] = self.logical_region();
        let (x, y) = ((x * s).round().max(0.0) as u32, (y * s).round().max(0.0) as u32);
        let w = ((w * s).round().max(1.0) as u32).min(self.width.saturating_sub(x).max(1));
        let h = ((h * s).round().max(1.0) as u32).min(self.height.saturating_sub(y).max(1));
        (x, y, w, h)
    }

    /// Each panel's element box in logical pixels of the target, borders
    /// included (as `getBoundingClientRect` gives it): the area the UI
    /// positions a panel's controls in.
    pub fn logical_panel_rects(&self) -> Vec<(Panel, [f64; 4])> {
        // The elements' boxes, unrounded (as `getBoundingClientRect`).
        let [ox, oy, w, h] = self.logical_region();
        self.layout
            .cells()
            .into_iter()
            .map(|(p, [c, r, cs, rs])| {
                let (cw, ch) = (w / cs as f64, h / rs as f64);
                (p, [ox + c as f64 * cw, oy + r as f64 * ch, cw, ch])
            })
            .collect()
    }

    /// The names of the displayed dimensions, in display order.
    pub fn display_dimension_names(&self) -> Vec<String> {
        let info = self.nav.render_info();
        info.display_dimension_indices[..info.display_rank]
            .iter()
            .filter(|&&i| i >= 0)
            .map(|&i| info.global_dimension_names[i as usize].clone())
            .collect()
    }

    /// Each panel's rectangle in physical pixels of the target.
    /// The panels' viewports together (`calculatePanelViewportBounds`): x,
    /// y, width, height in physical pixels.
    pub fn panel_viewport_union(&self) -> (u32, u32, u32, u32) {
        let rects = self.panel_rects();
        let x0 = rects.iter().map(|(_, r)| r.x).min().unwrap_or(0);
        let y0 = rects.iter().map(|(_, r)| r.y).min().unwrap_or(0);
        let x1 = rects.iter().map(|(_, r)| r.x + r.width).max().unwrap_or(0);
        let y1 = rects.iter().map(|(_, r)| r.y + r.height).max().unwrap_or(0);
        (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
    }

    /// Starts a screenshot (`takeScreenshot`) at `scale` (clamped to
    /// [`MAX_RENDER_AREA_PIXELS`](screenshot::MAX_RENDER_AREA_PIXELS)),
    /// keeping the slice views' field of view or not; `name` empty for the
    /// default. Drive it with [`step_screenshot`](Self::step_screenshot).
    pub fn start_screenshot(&mut self, scale: f64, keep_fov: bool, name: String) {
        let (_, _, w, h) = self.panel_viewport_union();
        let scale = screenshot::clamp_scale(scale, w, h);
        let size = (((self.width as f64) * scale).round().max(1.0) as u32, ((self.height as f64) * scale).round().max(1.0) as u32);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.screenshot = Some(screenshot::Job { scale, keep_fov, name, texture, size, stage: screenshot::Stage::Render, frames: 0 });
    }

    /// Whether a screenshot is under way (keep drawing frames).
    pub fn screenshot_active(&self) -> bool {
        self.screenshot.is_some()
    }

    /// `cancelScreenshot`.
    pub fn cancel_screenshot(&mut self) {
        self.screenshot = None;
    }

    /// Advances a screenshot: renders it scaled until no chunk is loading,
    /// then reads it back. Returns it once done (named with `time` if it
    /// has no name).
    pub fn step_screenshot(&mut self, time: &str) -> Option<screenshot::Screenshot> {
        let job = self.screenshot.take()?;
        match job.stage {
            screenshot::Stage::Render => {
                let (w, h, sf, viewport) = (self.width, self.height, self.scale_factor, self.viewport);
                // Kept: as many logical pixels, each more physical ones;
                // else more logical pixels (and a wider view) at the same
                // density.
                if job.keep_fov {
                    self.resize(job.size.0, job.size.1, sf * job.scale);
                } else {
                    self.resize(job.size.0, job.size.1, sf);
                    self.viewport = viewport.map(|v| v.map(|c| c * job.scale));
                }
                let view = job.texture.create_view(&Default::default());
                self.render(&view);
                let region = self.panel_viewport_union();
                (self.width, self.height, self.scale_factor, self.viewport) = (w, h, sf, viewport);
                let frames = job.frames + 1;
                if frames >= 2 && !self.is_loading() {
                    let stage = screenshot::start_copy(&self.device, &self.queue, &job.texture, job.size);
                    self.screenshot_region = region;
                    self.screenshot = Some(screenshot::Job { stage, frames, ..job });
                } else {
                    self.screenshot = Some(screenshot::Job { frames, ..job });
                }
                None
            }
            screenshot::Stage::Mapping { ref buffer, ref mapped, padded_row } => {
                let _ = self.device.poll(wgpu::PollType::Poll);
                if !mapped.load(std::sync::atomic::Ordering::SeqCst) {
                    self.screenshot = Some(job);
                    return None;
                }
                let bgra = matches!(self.target_format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
                let region = self.screenshot_region;
                let rgba = match buffer.slice(..).get_mapped_range() {
                    Ok(data) => screenshot::crop(&data, padded_row, region, bgra),
                    Err(_) => return None,
                };
                buffer.unmap();
                let (_, _, width, height) = region;
                let name = if job.name.is_empty() { screenshot::default_name(width, height, time) } else { job.name.clone() };
                Some(screenshot::Screenshot { name: screenshot::png_name(&name), width, height, png: screenshot::encode_png(&rgba, width, height) })
            }
        }
    }

    /// A panel's rectangle in logical pixels: the layout's, or scene mode's.
    fn panel_logical_rect(&self, panel: Panel, lw: u32, lh: u32) -> Option<PixelRect> {
        match &self.scene {
            Some(scene) => Some(scene.logical_rect(panel)),
            None => self.layout.rect(panel, lw, lh),
        }
    }

    /// Draws a frame in scene mode (see [`scene`]): the viewer's 3D content
    /// from the host's camera, `view.width × view.height` physical pixels,
    /// with the cross-sections at `view.slice_size`. Requests the chunks it
    /// needs, as [`render`](Self::render) does; [`poll`](Self::poll) and draw
    /// again as they arrive. Read the result with
    /// [`scene_output`](Self::scene_output).
    ///
    /// The state's layout, panels and pointer are left alone; the host routes
    /// its own input and composites the output into its frame by depth.
    pub fn render_scene(&mut self, view: scene::SceneView) {
        let (w, h) = (view.width.max(view.slice_size).max(1), view.height.max(view.slice_size).max(1));
        if !matches!(&self.scene_scratch, Some(s) if s.width >= w && s.height >= h) {
            self.scene_scratch = Some(scene::Scratch::new(&self.device, self.target_format, w, h));
        }
        // `render` composites into its target: in scene mode, a scratch one
        let scratch = self.scene_scratch.take().expect("allocated above");
        let saved = (self.width, self.height, self.scale_factor, self.viewport);
        (self.width, self.height, self.scale_factor, self.viewport) = (w, h, 1.0, None);
        self.scene = Some(view);
        self.render(&scratch.view);
        self.scene = None;
        (self.width, self.height, self.scale_factor, self.viewport) = saved;
        self.scene_scratch = Some(scratch);
    }

    /// The 3D output of the last [`render_scene`](Self::render_scene).
    pub fn scene_output(&self) -> Option<scene::SceneOutput<'_>> {
        let pf = self.perspective_frame.as_ref().filter(|pf| pf.target.sampled_depth)?;
        Some(scene::SceneOutput { color: &pf.target.flipped_view, depth: &pf.target.depth_view, width: pf.target.width, height: pf.target.height })
    }

    pub fn panel_rects(&self) -> Vec<(Panel, PixelRect)> {
        if let Some(scene) = &self.scene {
            return scene.panel_rects(self.show_slices);
        }
        let (x, y, w, h) = self.physical_region();
        self.layout
            .panels_scaled(w, h, self.scale_factor)
            .into_iter()
            .map(|(p, r)| (p, PixelRect { x: r.x + x, y: r.y + y, ..r }))
            .collect()
    }

    /// The panel under the pointer, and the pointer in that panel's logical
    /// pixels.
    fn pointer_panel(&self) -> Option<(Panel, PanelPointer)> {
        // the host routes its own input in scene mode
        if self.scene.is_some() {
            return None;
        }
        let (w, h) = self.logical_size();
        let [ox, oy, _, _] = self.logical_region();
        let (x, y) = self.pointer?;
        let (x, y) = (x - ox, y - oy);
        let panel = self.layout.panel_at(x, y, w, h)?;
        let r = self.layout.rect(panel, w, h)?;
        let p = PanelPointer {
            width: r.width as f64,
            height: r.height as f64,
            x: x - r.x as f64,
            y: y - r.y as f64,
        };
        Some((panel, p))
    }

    fn panel_mut(&mut self, axes: NamedAxes) -> SlicePanel {
        // Panels are small; take a copy to sidestep borrowing `self.nav` and
        // `self.panels` together, then put it back.
        self.panels.iter().find(|(a, _)| *a == axes).unwrap().1.clone()
    }

    fn put_panel(&mut self, axes: NamedAxes, panel: SlicePanel) {
        self.panels.iter_mut().find(|(a, _)| *a == axes).unwrap().1 = panel;
    }

    /// The global position under the pointer, if it is over a slice panel.
    pub fn mouse_position(&mut self) -> Option<Vec<f32>> {
        let (panel, p) = self.pointer_panel()?;
        let pos = if panel == Panel::Perspective {
            // The last pick's (`mouseState` from the pick buffers).
            self.perspective_pick.as_ref()?.position.clone()
        } else {
            let Panel::Slice(axes) = panel else { return None };
            if !self.nav.valid() {
                return None;
            }
            let mut sp = self.panel_mut(axes);
            let pos = sp.mouse_position(&mut self.nav, p);
            self.put_panel(axes, sp);
            pos
        };
        Some(self.snap_to_picked(pos))
    }

    /// Annotation `id` of layer `li`: a local one, or one of a loaded
    /// spatial index cell (the finest level holding it).
    fn find_annotation(&self, li: usize, id: &str) -> Option<annotations::Annotation> {
        let layer = self.layers.get(li)?;
        let ann = layer.annotation.as_ref()?;
        if let Some(a) = ann.annotations.iter().find(|a| a.id == id) {
            return Some(a.clone());
        }
        ann.spatial.as_ref()?;
        if let Some(Some(a)) = id.parse::<u64>().ok().and_then(|n| self.annotations_by_id.get(&(layer.id, n))).as_deref() {
            return Some(a.clone());
        }
        let mut found: Option<(usize, annotations::Annotation)> = None;
        self.annotation_cells.for_each(|(l, level, _), state, _, _, _| {
            if *l != layer.id || found.as_ref().is_some_and(|(f, _)| f <= level) {
                return;
            }
            if let State::Ready(list) = state {
                if let Some(a) = list.iter().find(|a| a.id == id) {
                    found = Some((*level, a.clone()));
                }
            }
        });
        found.map(|(_, a)| a)
    }

    /// Annotation `id` of layer `li` for the Selection panel; `None` until
    /// it is loaded.
    pub fn annotation_details(&self, li: usize, id: &str) -> Option<AnnotationDetails> {
        let layer = self.layers.get(li)?;
        let a = self.find_annotation(li, id)?;
        let entry = layer.annotation_entry(&a, self.global.rank())?;
        let spatial = layer.annotation.as_ref()?.spatial.as_ref();
        let properties = spatial.map_or(vec![], |info| info.properties.iter().zip(&a.properties).map(|(p, v)| (p.id.clone(), p.kind, *v)).collect());
        Some(AnnotationDetails { entry, readonly: spatial.is_some(), properties })
    }

    /// Asks for annotation `id` of a precomputed layer `li` from its `by_id`
    /// index (`getReference`); whether it may still come (`false`: there
    /// is no such annotation).
    pub fn request_annotation(&mut self, li: usize, id: &str) -> bool {
        let Some(layer) = self.layers.get(li) else { return false };
        if !layer.annotation.as_ref().is_some_and(|a| a.spatial.as_ref().is_some_and(|i| i.by_id.is_some())) {
            return false;
        }
        let Ok(n) = id.parse::<u64>() else { return false };
        let key = (layer.id, n);
        let mut known = None;
        self.annotations_by_id.for_each(|k, state, _, _, _| {
            if *k == key {
                known = Some(matches!(state, State::Queued | State::Downloading));
            }
        });
        match known {
            Some(pending) => pending,
            None => {
                self.annotation_wants.push(key);
                // A frame to ask for it in, once the caller (building the
                // UI, inside the app) is done.
                let wake = self.wake.clone();
                self.executor.spawn(Box::pin(async move { wake() }));
                true
            }
        }
    }

    /// Annotation `id`'s first property value of layer `li`, formatted.
    fn picked_annotation_value(&self, li: usize, id: &str) -> Option<String> {
        let spec = self.layers.get(li)?.annotation.as_ref()?.spatial.as_ref()?.properties.first()?;
        let a = self.find_annotation(li, id)?;
        Some(annotations::format_property_value(spec.kind, *a.properties.first()?))
    }

    /// `updateMouseState` for a picked annotation (`snapPosition`): the
    /// position moved onto the point, the line (or polyline's first
    /// segment), the endpoint or the box corner picked; boxes' and
    /// ellipsoids' bodies leave it.
    fn snap_to_picked(&self, mut position: Vec<f32>) -> Vec<f32> {
        let Some(PickTarget::Annotation(layer_id, id, part)) = self.picked() else { return position };
        let Some(li) = self.layers.iter().position(|l| l.id == layer_id) else { return position };
        let Some(a) = self.find_annotation(li, &id) else { return position };
        let Some(p) = self.global_to_annotation_model(li, &position) else { return position };
        use annotations::Geometry as G;
        // `projectPointToLineSegment`.
        let project = |a: &[f64], b: &[f64]| -> Vec<f64> {
            let den: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
            let num: f64 = a.iter().zip(b).zip(&p).map(|((x, y), q)| -(x - q) * (y - x)).sum();
            let t = (num / den.max(1e-6)).clamp(0.0, 1.0);
            a.iter().zip(b).map(|(x, y)| x + t * (y - x)).collect()
        };
        let snapped: Option<Vec<f64>> = match (&a.geometry, part) {
            (G::Point(q), _) => Some(q.clone()),
            (G::Line(pa, pb), AnnotationPart::Body) => Some(project(pa, pb)),
            (G::Line(pa, pb), AnnotationPart::Endpoint(k)) => Some(if k == 0 { pa.clone() } else { pb.clone() }),
            (G::Polyline(points), AnnotationPart::Body) if points.len() >= 2 => Some(project(&points[0], &points[1])),
            (G::Polyline(points), AnnotationPart::Endpoint(k)) => points.get(k).cloned(),
            // `snapPositionToCorner`: per dimension the nearer bound.
            (G::AxisAlignedBoundingBox(pa, pb), AnnotationPart::Corner(_)) => Some(pa.iter().zip(pb).zip(&p).map(|((a, b), x)| if (a - x).abs() < (b - x).abs() { *a } else { *b }).collect()),
            _ => None,
        };
        if let Some(model) = snapped {
            let l = &self.layers[li].annotation.as_ref().unwrap().layer;
            let n = l.rank + 1;
            let m2l = &l.model_to_layer;
            for (ld, g) in l.layer_to_global.iter().enumerate() {
                let Some(g) = *g else { continue };
                let mut v = m2l[l.rank * n + ld];
                for (c, x) in model.iter().enumerate().take(l.source_rank) {
                    v += m2l[c * n + ld] * x;
                }
                if v.is_finite() {
                    if let Some(slot) = position.get_mut(g) {
                        *slot = v as f32;
                    }
                }
            }
        }
        position
    }

    /// The pointer moved to `(x, y)`, physical pixels from the top-left.
    /// Returns whether the view changed.
    /// Whether a drag (pan, rotate) is under way.
    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// The pointer left the viewer: nothing is hovered (`mouseState.setActive(false)`).
    pub fn pointer_leave(&mut self) -> bool {
        self.pointer.take().is_some()
    }

    pub fn pointer_move(&mut self, x: f64, y: f64) -> bool {
        let (x, y) = (x / self.scale_factor, y / self.scale_factor);
        let (dx, dy) = self.pointer.map_or((0.0, 0.0), |(px, py)| (x - px, y - py));
        self.pointer = Some((x, y));
        let Some(drag) = self.drag.clone() else {
            // The status line follows the pointer.
            return true;
        };
        match (drag.panel, drag.action) {
            (Panel::Slice(axes), DragAction::Translate) => {
                let mut sp = self.panel_mut(axes);
                sp.translate_by_viewport_pixels(&mut self.nav, dx, dy);
                self.put_panel(axes, sp);
            }
            (Panel::Slice(axes), DragAction::Rotate { initial_position }) => {
                let mut sp = self.panel_mut(axes);
                sp.rotate_via_mouse_drag(&mut self.nav, dx, dy, &initial_position);
                self.put_panel(axes, sp);
            }
            // `PerspectivePanel`'s "rotate-via-mouse-drag".
            (Panel::Perspective, DragAction::PerspectiveRotate) => {
                self.nav.rotate_relative(View::Projection, ng_core::glm::AXES[1], dx / 4.0 * std::f64::consts::PI / 180.0);
                self.nav.rotate_relative(View::Projection, ng_core::glm::AXES[0], -dy / 4.0 * std::f64::consts::PI / 180.0);
            }
            (Panel::Perspective, DragAction::PerspectiveTranslate) => self.perspective_translate(dx, dy),
            (panel, DragAction::MoveAnnotation { layer, id, part, point, geometry, total }) => {
                let total = (total.0 + dx, total.1 + dy);
                if let Some(d) = self.drag.as_mut() {
                    if let DragAction::MoveAnnotation { total: t, .. } = &mut d.action {
                        *t = total;
                    }
                }
                self.move_annotation(panel, layer, &id, part, &point, &geometry, total);
            }
            _ => {}
        }
        true
    }

    pub fn pointer_down(&mut self, button: Button, modifiers: Modifiers) -> bool {
        let Some((panel, _)) = self.pointer_panel() else { return false };
        let kind = match panel {
            Panel::Slice(_) => PanelKind::Slice,
            Panel::Perspective => PanelKind::Perspective,
        };
        let key = format!("mousedown{}", button as u8);
        let Some(action) = lookup(kind, &key, modifiers, self.mac) else {
            return false;
        };
        match (panel, action) {
            (Panel::Slice(_), "translate-via-mouse-drag") => {
                self.drag = Some(Drag { panel, action: DragAction::Translate });
                false
            }
            (Panel::Slice(_), "rotate-via-mouse-drag") => {
                let Some(initial_position) = self.mouse_position() else { return false };
                self.drag = Some(Drag { panel, action: DragAction::Rotate { initial_position } });
                false
            }
            (Panel::Slice(_), "move-to-mouse-position") => match self.mouse_position() {
                Some(p) => {
                    self.nav.set_position(&p);
                    true
                }
                None => false,
            },
            (Panel::Perspective, "rotate-via-mouse-drag") => {
                self.drag = Some(Drag { panel, action: DragAction::PerspectiveRotate });
                false
            }
            (Panel::Perspective, "translate-via-mouse-drag") => {
                self.drag = Some(Drag { panel, action: DragAction::PerspectiveTranslate });
                false
            }
            (_, "move-annotation") => {
                let Some(PickTarget::Annotation(layer, id, part)) = self.picked() else { return false };
                let Some(a) = self.layers.iter().find(|l| l.id == layer).and_then(|l| l.annotation.as_ref()?.annotations.iter().find(|a| a.id == id)).cloned() else { return false };
                let point = representative_point(&a.geometry, part);
                self.drag = Some(Drag { panel, action: DragAction::MoveAnnotation { layer, id, part, point, geometry: a.geometry, total: (0.0, 0.0) } });
                false
            }
            (_, "delete-annotation") => match self.picked_annotation() {
                Some((layer, id)) => {
                    self.delete_annotation(layer, &id);
                    true
                }
                None => false,
            },
            _ => false,
        }
    }

    /// `move-annotation`'s drag: the representative point moved `total`
    /// logical pixels in the panel (`translateDataPointByViewportPixels`),
    /// the annotation updated from it (`updateViaRepresentativePoint`).
    #[allow(clippy::too_many_arguments)]
    fn move_annotation(&mut self, panel: Panel, layer: LayerId, id: &str, part: AnnotationPart, point: &[f64], geometry: &annotations::Geometry, total: (f64, f64)) {
        let Some(li) = self.layers.iter().position(|l| l.id == layer) else { return };
        let info = self.nav.render_info().clone();
        let slice_inv_view = match panel {
            Panel::Slice(axes) => {
                let mut sp = self.panel_mut(axes);
                let inv_view = sp.inv_view_matrix(&mut self.nav);
                self.put_panel(axes, sp);
                Some(inv_view)
            }
            Panel::Perspective => None,
        };
        let Some(ann) = self.layers[li].annotation.as_ref() else { return };
        let al = &ann.layer;
        let display = model_to_display(al, &info, point);
        let moved = match panel {
            Panel::Slice(_) => {
                let Some(inv_view) = slice_inv_view else { return };
                // View pixels (x right, y down) through the view's linear part.
                std::array::from_fn::<f64, 3, _>(|r| display[r] + inv_view[r] as f64 * total.0 + inv_view[4 + r] as f64 * total.1)
            }
            Panel::Perspective => {
                let Some(camera) = self.perspective_camera() else { return };
                let vp = camera.view_projection;
                let Some(inv) = ng_core::glm::mat4_invert(&vp) else { return };
                let (lw, lh) = self.logical_size();
                let Some(r) = self.layout.rect(Panel::Perspective, lw, lh) else { return };
                let mut c = ng_core::glm::vec4_transform_mat4([display[0], display[1], display[2], 1.0], &vp);
                c[0] += 2.0 * total.0 / r.width as f64 * c[3];
                c[1] -= 2.0 * total.1 / r.height as f64 * c[3];
                let d = ng_core::glm::vec4_transform_mat4(c, &inv);
                [d[0] / d[3], d[1] / d[3], d[2] / d[3]]
            }
        };
        // The point's global position with the display dimensions moved,
        // back into the source's coordinates.
        let mut global: Vec<f32> = self.nav.position().to_vec();
        {
            let l = al;
            let n = l.rank + 1;
            for (ld, g) in l.layer_to_global.iter().enumerate() {
                let Some(g) = *g else { continue };
                let mut v = l.model_to_layer[l.rank * n + ld];
                for (c, x) in point.iter().enumerate().take(l.source_rank) {
                    v += l.model_to_layer[c * n + ld] * x;
                }
                if let Some(slot) = global.get_mut(g) {
                    *slot = v as f32;
                }
            }
        }
        for (k, &g) in info.display_dimension_indices.iter().enumerate() {
            if g >= 0 {
                if let Some(slot) = global.get_mut(g as usize) {
                    *slot = moved[k] as f32;
                }
            }
        }
        let Some(new_point) = self.global_to_annotation_model(li, &global) else { return };
        let updated = update_via_representative_point(geometry, &new_point, part);
        self.update_annotation(li, id, |v| write_geometry(v, &updated));
    }

    pub fn pointer_up(&mut self, _button: Button) -> bool {
        self.drag = None;
        false
    }

    /// A wheel event with DOM semantics: `delta_y` positive toward the user,
    /// `delta_mode` 0 pixels / 1 lines / 2 pages.
    pub fn wheel(&mut self, delta_x: f64, delta_y: f64, delta_mode: u32, modifiers: Modifiers) -> bool {
        let Some((panel, p)) = self.pointer_panel() else { return false };
        match panel {
            Panel::Slice(axes) => {
                let Some(action) = lookup(PanelKind::Slice, "wheel", modifiers, self.mac) else {
                    return false;
                };
                let mut sp = self.panel_mut(axes);
                match action {
                    "z+1-via-wheel" => sp.z_via_wheel(&mut self.nav, delta_y, delta_x, 1.0),
                    "z+10-via-wheel" => sp.z_via_wheel(&mut self.nav, delta_y, delta_x, 10.0),
                    "zoom-via-wheel" => sp.zoom_by_mouse(&mut self.nav, wheel_zoom_amount(delta_y, delta_mode), p),
                    "adjust-depth-range-via-wheel" => {
                        let d = self.nav.depth_range(View::CrossSection) * wheel_zoom_amount(delta_y, delta_mode);
                        self.nav.set_depth_range(View::CrossSection, d);
                    }
                    _ => {}
                }
                self.put_panel(axes, sp);
                true
            }
            Panel::Perspective => match lookup(PanelKind::Perspective, "wheel", modifiers, self.mac) {
                // `zoomByMouse`: the projection zoom.
                Some("zoom-via-wheel") => {
                    self.nav.zoom_by(View::Projection, wheel_zoom_amount(delta_y, delta_mode));
                    true
                }
                Some("adjust-depth-range-via-wheel") => {
                    let d = self.nav.depth_range(View::Projection) * wheel_zoom_amount(delta_y, delta_mode);
                    self.nav.set_depth_range(View::Projection, d);
                    true
                }
                _ => false,
            },
        }
    }

    /// A key press: `code` is DOM `KeyboardEvent.code` (any case). Acts on the
    /// panel under the pointer, as neuroglancer's focused panel does.
    pub fn key_down(&mut self, code: &str, modifiers: Modifiers) -> bool {
        let Some((panel, _)) = self.pointer_panel() else { return false };
        let Panel::Slice(axes) = panel else { return false };
        let key = code.to_ascii_lowercase();
        let Some(action) = lookup(PanelKind::Slice, &key, modifiers, self.mac) else {
            return false;
        };
        self.slice_action(axes, action)
    }

    /// A data panel action by name (`snap`, `zoom-in`, `x+`, …), as the
    /// command palette runs it: in the slice panel under the pointer, else
    /// the layout's first. Returns whether it was one.
    pub fn panel_action(&mut self, action: &str) -> bool {
        let axes = match self.pointer_panel() {
            Some((Panel::Slice(axes), _)) => axes,
            _ => match self.layout.cells().into_iter().find_map(|(p, _)| if let Panel::Slice(a) = p { Some(a) } else { None }) {
                Some(a) => a,
                None => return false,
            },
        };
        self.slice_action(axes, action)
    }

    fn slice_action(&mut self, axes: NamedAxes, action: &str) -> bool {
        let mut sp = self.panel_mut(axes);
        let nav = &mut self.nav;
        let handled = match action {
            "snap" => {
                sp.snap(nav);
                true
            }
            "zoom-in" => {
                sp.with(nav, |n| n.zoom_by(View::CrossSection, 0.5));
                true
            }
            "zoom-out" => {
                sp.with(nav, |n| n.zoom_by(View::CrossSection, 2.0));
                true
            }
            "depth-range-decrease" | "depth-range-increase" => {
                let f = if action == "depth-range-decrease" { 0.5 } else { 2.0 };
                let d = nav.depth_range(View::CrossSection) * f;
                nav.set_depth_range(View::CrossSection, d);
                true
            }
            a => {
                let axis_sign = |s: &str| -> Option<(usize, f32)> {
                    let axis = match s.as_bytes().first()? {
                        b'x' => 0,
                        b'y' => 1,
                        b'z' => 2,
                        _ => return None,
                    };
                    let sign = match &s[1..] {
                        "+" => 1.0,
                        "-" => -1.0,
                        _ => return None,
                    };
                    Some((axis, sign))
                };
                if let Some(rest) = a.strip_prefix("rotate-relative-") {
                    axis_sign(rest).map(|(axis, sign)| sp.rotate_relative_axis(nav, axis, sign as f64)).is_some()
                } else if let Some((axis, sign)) = axis_sign(a) {
                    sp.translate_axis(nav, axis, sign);
                    true
                } else {
                    false
                }
            }
        };
        self.put_panel(axes, sp);
        handled
    }

    /// The 3D panel's camera, for its logical size.
    fn perspective_camera(&self) -> Option<ng_core::panel::PerspectiveCamera> {
        if let Some(scene) = &self.scene {
            return scene.camera(self.nav.render_info());
        }
        let (lw, lh) = self.logical_size();
        let r = self.layout.rect(Panel::Perspective, lw, lh)?;
        Some(ng_core::panel::perspective_camera(&self.nav, r.width as f64, r.height as f64, self.orthographic_projection))
    }

    /// `PerspectivePanel.translateByViewportPixels`.
    fn perspective_translate(&mut self, dx: f64, dy: f64) {
        let (lw, lh) = self.logical_size();
        let Some(r) = self.layout.rect(Panel::Perspective, lw, lh) else { return };
        let Some(camera) = self.perspective_camera() else { return };
        let Some(inv) = ng_core::glm::mat4_invert(&camera.view_projection) else { return };
        let (w, h) = (r.width as f64, r.height as f64);
        self.nav.update_display_position(|pos| {
            let t = ng_core::glm::vec3_transform_mat4(*pos, camera.view_projection);
            let t = [t[0] + (-2.0 * dx / w) as f32, t[1] + (2.0 * dy / h) as f32, t[2]];
            *pos = ng_core::glm::vec3_transform_mat4(t, inv);
            true
        });
    }

    fn layer_render(&self, layer: usize) -> Option<&LayerState> {
        self.layers.get(layer)?.render.as_ref()
    }

    /// Asks for layer `layer`'s data histogram from the next frame on, or
    /// (`false`) no longer: the histogram's widget shows. The last counts
    /// stay until replaced.
    pub fn want_data_histogram(&mut self, layer: usize, wanted: bool) {
        if let Some(l) = self.layers.get_mut(layer) {
            l.histogram_wanted = wanted;
            if !wanted {
                l.histogram_windows.clear();
            }
        }
    }

    /// The legend of a user shader for data `values` (one per pixel): its
    /// colours (RGBA, as drawn over transparent) once drawn — the last
    /// drawn as wide meanwhile; drawn from the next frame.
    pub fn shader_legend(&mut self, layer: usize, values: Vec<f64>) -> Option<Vec<[u8; 4]>> {
        let key = self.legend_key(layer, &values)?;
        let l = self.layers.get_mut(layer)?;
        if let Some((_, colors)) = l.legends.iter().find(|(k, _)| *k == key) {
            return Some(colors.clone());
        }
        if !l.legend_wants.contains(&values) {
            l.legend_wants.push(values.clone());
        }
        l.legends.iter().rev().find(|(k, _)| k.values.len() == values.len()).map(|(_, c)| c.clone())
    }

    /// What a legend over `values` is drawn for: the shader and its
    /// uniforms at opacity 1.
    fn legend_key(&self, layer: usize, values: &[f64]) -> Option<legend::LegendKey> {
        let l = self.layers.get(layer)?;
        let state = l.render.as_ref()?;
        let shader = state.shader.as_ref()?;
        let uniforms = ng_render::user_shader::uniform_bits(&shader.parsed, state.layer.data_type, &shader.values, 1.0);
        Some(legend::LegendKey { values: values.iter().map(|v| v.to_bits()).collect(), shader: shader.key.0.clone(), uniforms })
    }

    /// Whether a legend is being drawn (draw again for it to show).
    pub fn legend_pending(&self) -> bool {
        self.layers.iter().any(|l| !l.legend_jobs.is_empty() || !l.legend_wants.is_empty())
    }

    /// Starts the legends asked for, collects those drawn.
    fn process_legends(&mut self) {
        for li in 0..self.layers.len() {
            let wants = std::mem::take(&mut self.layers[li].legend_wants);
            for values in wants {
                let Some(key) = self.legend_key(li, &values) else { continue };
                let l = &self.layers[li];
                if l.legend_jobs.iter().any(|j| j.key == key) || l.legends.iter().any(|(k, _)| *k == key) {
                    continue;
                }
                let Some(state) = l.render.as_ref() else { continue };
                let Some(shader) = state.shader.as_ref() else { continue };
                let t = state.layer.data_type;
                // Opacity 1, as the legend's.
                shader.uniforms.set(&self.queue, &ng_render::user_shader::uniform_bits(&shader.parsed, t, &shader.values, 1.0));
                let job = legend::start(&self.device, &self.queue, &state.pipelines.first, &shader.uniforms.view, t, &values, key);
                self.layers[li].legend_jobs.push(job);
            }
            // Natively, a copy's mapping completes as the device is polled.
            if !self.layers[li].legend_jobs.is_empty() {
                let _ = self.device.poll(wgpu::PollType::Poll);
            }
            let l = &mut self.layers[li];
            let mut done = vec![];
            l.legend_jobs.retain(|j| match j.take() {
                Some(colors) => {
                    done.push((j.key.clone(), colors));
                    false
                }
                None => true,
            });
            for d in done {
                l.legends.push(d);
                // A few kept (several controls, a width changing).
                if l.legends.len() > 8 {
                    l.legends.remove(0);
                }
            }
        }
    }

    /// A histogram of layer `layer`'s data over `window` too (an invlerp
    /// control of a user shader), for the next frames: its values are
    /// sampled once for all.
    pub fn want_window_histogram(&mut self, layer: usize, window: [f64; 2]) {
        if let Some(l) = self.layers.get_mut(layer) {
            if !l.histogram_windows.contains(&window) {
                l.histogram_windows.push(window);
            }
        }
    }

    /// Each scale's layer-to-chunk transform (inverted `chunk_to_layer`).
    fn layer_to_chunk(layer: &VolumeLayer) -> Vec<Vec<f64>> {
        let n = layer.rank + 1;
        (0..layer.scales.len()).map(|s| ng_core::matrix::inverse(&layer.chunk_to_layer(s), n).0).collect()
    }

    /// Layer `layer`'s value at a global position, from the finest loaded
    /// chunk containing it, formatted as neuroglancer's layer panel shows it.
    pub fn layer_value_at(&self, layer: usize, position: &[f32]) -> Option<String> {
        // The object picked in the 3D panel (`pickedRenderLayer`'s layer:
        // `pickedValue`).
        if let Some(id) = self.picked_segment(layer) {
            return Some(id.to_string());
        }
        // A picked annotation's first property (`transformPickedValue`).
        if let Some(Some(v)) = self.layers.get(layer).and_then(|l| match self.picked() {
            Some(PickTarget::Annotation(id, annotation, _)) if id == l.id => Some(self.picked_annotation_value(layer, &annotation)),
            _ => None,
        }) {
            return Some(v);
        }
        let state = self.layer_render(layer)?;
        let (chunk, index) = self.locate(layer, &Self::layer_to_chunk(&state.layer), position)?;
        let l = &state.layer;
        if l.channels.is_empty() || !matches!(state.kind, LayerKind::Image { .. }) {
            return Some(format_value(l.data_type, &chunk.data, index));
        }
        // Every channel's value (`getChannelSpace` order), joined.
        let count: u32 = l.channels.iter().map(|c| c.extent).product();
        let values: Vec<String> = (0..count)
            .map(|flat| {
                let mut rest = flat;
                let coords: Vec<u32> = l
                    .channels
                    .iter()
                    .map(|c| {
                        let v = rest % c.extent;
                        rest /= c.extent;
                        v
                    })
                    .collect();
                channel_offset(l, &chunk, &coords).map_or(String::new(), |o| format_value(l.data_type, &chunk.data, index + o))
            })
            .collect();
        Some(values.join(","))
    }

    /// The finest loaded chunk of layer `layer` holding a global position,
    /// and the voxel's index in it.
    fn locate(&self, layer: usize, l2c_by_scale: &[Vec<f64>], position: &[f32]) -> Option<(std::sync::Arc<ng_datasource::chunk::ChunkData>, usize)> {
        let id = self.layers.get(layer)?.id;
        let state = self.layer_render(layer)?;
        let layer = &state.layer;
        let rank = layer.rank;
        let n = rank + 1;
        let mut layer_pos = vec![0.0; n];
        layer_pos[rank] = 1.0;
        for (d, g) in layer.layer_to_global.iter().enumerate() {
            layer_pos[d] = match g {
                Some(g) => *position.get(*g)? as f64,
                None => 0.5,
            };
        }
        for (s, l2c) in l2c_by_scale.iter().enumerate() {
            let spec = &layer.scales[s].spec;
            let voxel: Vec<f64> = (0..rank).map(|c| (0..n).map(|k| l2c[k * n + c] * layer_pos[k]).sum::<f64>().floor()).collect();
            let sr = layer.source_rank;
            if (sr..rank).any(|c| voxel[c] != 0.0) || (0..sr).any(|c| voxel[c] < 0.0 || voxel[c] >= spec.upper_voxel_bound[c] as f64) {
                continue;
            }
            let grid: Vec<u64> = (0..sr).map(|c| (voxel[c] / spec.chunk_data_size[c] as f64).floor() as u64).collect();
            let Some(chunk) = self.chunks.get(&(id, s, grid.clone())) else { continue };
            let mut index = 0usize;
            let mut stride = 1usize;
            for c in 0..sr {
                let off = voxel[c] as usize - grid[c] as usize * spec.chunk_data_size[c] as usize;
                if off >= chunk.size[c] as usize {
                    return None;
                }
                index += off * stride;
                stride *= chunk.size[c] as usize;
            }
            return Some((chunk, index));
        }
        None
    }

    /// Fills the wanted data histograms from this frame's slice panels.
    fn compute_data_histograms(&mut self, slice_views: &[SliceViewFrame]) {
        let info = self.nav.render_info().clone();
        let position = self.nav.position().to_vec();
        for li in 0..self.layers.len() {
            let l = &self.layers[li];
            if !(l.histogram_wanted || !l.histogram_windows.is_empty()) || !l.spec.visible {
                continue;
            }
            let (Some(state), Some(window)) = (l.render.as_ref(), l.invlerp_window()) else { continue };
            let t = state.layer.data_type;
            let w = window.as_f64();
            let windows = l.histogram_windows.clone();
            // The invlerp's channel (`dataHistogramChannelSpecifications`).
            let channel = l.invlerp_channel();
            let l2c = Self::layer_to_chunk(&state.layer);
            let mut counts = [0u32; 256];
            let mut samples = Vec::with_capacity(histogram::SAMPLES * slice_views.len());
            // No slice view: volume rendering samples its chunks
            // (`needToDrawHistogram && !sliceViewsPresent`).
            if slice_views.is_empty() && !l.volume_chunks.is_empty() {
                let size_of = |c: &ChunkData, dims: &[Option<usize>; 3]| -> [u32; 3] { dims.map(|d| d.map_or(1, |d| c.size[d])) };
                let total: f64 = l.volume_chunks.iter().map(|(c, dims, _)| size_of(c, dims).iter().map(|&s| s as f64).product::<f64>()).sum();
                for (chunk, dims, fixed) in &l.volume_chunks {
                    let rank = chunk.size.len();
                    let mut strides = vec![1usize; rank];
                    for k in 1..rank {
                        strides[k] = strides[k - 1] * chunk.size[k - 1] as usize;
                    }
                    let mut base = 0usize;
                    for &(c, _, off) in fixed {
                        if !state.layer.channels.iter().any(|ch| ch.chunk_dim == c) {
                            base += off as usize * strides[c];
                        }
                    }
                    let Some(offset) = channel_offset(&state.layer, chunk, &channel) else { continue };
                    for p in histogram::volume_sample_points(size_of(chunk, dims), total) {
                        let mut index = base + offset;
                        for (k, d) in dims.iter().enumerate() {
                            if let Some(d) = d {
                                index += p[k] as usize * strides[*d];
                            }
                        }
                        let value = histogram::value_f64(t, &chunk.data, index);
                        counts[histogram::bin((value - w[0]) / (w[1] - w[0]))] += 1;
                        samples.push(value);
                    }
                }
            }
            for (_, inv_view, (pw, ph)) in slice_views {
                let (lw, lh) = (*pw as f64 / self.scale_factor, *ph as f64 / self.scale_factor);
                let mut global = position.clone();
                for &(u, v) in histogram::sample_points() {
                    let view = [((u as f64 - 0.5) * lw) as f32, ((v as f64 - 0.5) * lh) as f32, 0.0];
                    let plane = ng_core::glm::vec3_transform_mat4(view, *inv_view);
                    for i in 0..info.display_rank {
                        global[info.display_dimension_indices[i] as usize] = plane[i];
                    }
                    // Pixels no chunk was drawn on (depth 1) are skipped.
                    let Some((chunk, index)) = self.locate(li, &l2c, &global) else { continue };
                    let Some(offset) = channel_offset(&state.layer, &chunk, &channel) else { continue };
                    let value = histogram::value_f64(t, &chunk.data, index + offset);
                    counts[histogram::bin((value - w[0]) / (w[1] - w[0]))] += 1;
                    samples.push(value);
                }
            }
            let window_histograms = windows
                .into_iter()
                .map(|w| {
                    let mut c = [0u32; 256];
                    for v in &samples {
                        c[histogram::bin((v - w[0]) / (w[1] - w[0]))] += 1;
                    }
                    (w, Box::new(c))
                })
                .collect();
            let l = &mut self.layers[li];
            l.data_histogram = Some(Box::new(counts));
            l.histogram_samples = samples;
            l.window_histograms = window_histograms;
        }
    }

    /// The point under the pointer in display coordinates, all three (a
    /// display dimension the space lacks still has a coordinate on the
    /// panel's plane), if it is over a slice panel.
    pub fn mouse_display_point(&mut self) -> Option<[f32; 3]> {
        let (panel, p) = self.pointer_panel()?;
        let Panel::Slice(axes) = panel else { return None };
        let mut sp = self.panel_mut(axes);
        let inv = sp.inv_view_matrix(&mut self.nav);
        self.put_panel(axes, sp);
        let view = [(p.x - p.width / 2.0) as f32, (p.y - p.height / 2.0) as f32, 0.0];
        Some(ng_core::glm::vec3_transform_mat4(view, inv))
    }

    /// Continuous voxel coordinates of a global position in layer `layer`'s
    /// finest scale (chunk dimensions; voxel `i` spans `[i, i + 1)`).
    pub fn layer_voxel(&self, layer: usize, position: &[f32]) -> Option<Vec<f64>> {
        let layer = &self.layer_render(layer)?.layer;
        let rank = layer.rank;
        let n = rank + 1;
        let mut layer_pos = vec![0.0; n];
        layer_pos[rank] = 1.0;
        for (d, g) in layer.layer_to_global.iter().enumerate() {
            layer_pos[d] = match g {
                Some(g) => *position.get(*g)? as f64,
                None => 0.5,
            };
        }
        let (l2c, _) = ng_core::matrix::inverse(&layer.chunk_to_layer(0), n);
        Some((0..layer.source_rank).map(|c| (0..n).map(|k| l2c[k * n + c] * layer_pos[k]).sum()).collect())
    }

    /// The segment under the pointer in each layer (`segmentSelectionState`,
    /// for hover highlighting); `None` for other layers.
    /// Segmentation layers' skeletons to draw in a cross-section
    /// (`view_2d`) or the 3D view (`SkeletonLayer.draw`), requesting those
    /// not loaded: each visible segment's, in its colour.
    fn skeleton_draws(&mut self, hovered: &[Option<u64>], info: &ng_core::navigation::DisplayDimensionRenderInfo, view_2d: bool) -> Vec<SkeletonDraw> {
        let mut out = vec![];
        let mut wanted = vec![];
        for (li, managed) in self.layers.iter().enumerate() {
            if !managed.spec.visible {
                continue;
            }
            let (Some(state), Some(LayerKind::Segmentation(display))) = (managed.render.as_ref(), managed.kind()) else { continue };
            let sources = managed.skeleton_sources();
            if sources.is_empty() {
                continue;
            }
            let alpha = managed.spec.json.get("objectAlpha").and_then(Value::as_f64).unwrap_or(1.0).min(1.0) as f32;
            if alpha <= 0.0 {
                continue;
            }
            let (points, width) = skeleton_options(&managed.spec.json, view_2d);
            let Ok((_, segments)) = state::segmentation_settings(&managed.spec.json) else { continue };
            for (k, source) in sources.iter().enumerate() {
                let Some(model) = mesh_model_matrix(&state.layer, &source.to_model, info) else { continue };
                for &(id, visible) in &segments {
                    if visible {
                        let color = object_color(display, hovered.get(li).copied().flatten(), id, alpha);
                        wanted.push(((managed.id, k, id), model, color, points, width));
                    }
                }
            }
        }
        let count = std::mem::replace(&mut self.skeletons_counted, self.frame) != self.frame;
        for (key, model, color, points, width) in wanted {
            self.skeletons.request(&key, 0.0);
            let ready = self.skeletons.get(&key);
            if let Some(l) = self.layers.iter_mut().find(|l| l.id == key.0).filter(|_| count) {
                l.progress.0 += 1;
                l.progress.1 += u32::from(ready.is_some());
            }
            if let Some(data) = ready.filter(|d| d.num_vertices() > 0) {
                out.push(SkeletonDraw { segment: (key.0, key.2), model, color, data, points, width });
            }
        }
        out
    }

    /// What is picked under the pointer: in the 3D panel from its pick
    /// buffers, in a slice panel the nearest mark drawn within the pick
    /// radius (last drawn on top).
    fn picked(&self) -> Option<PickTarget> {
        let (panel, p) = self.pointer_panel()?;
        match panel {
            Panel::Perspective => self.perspective_pick.as_ref()?.picked.clone(),
            Panel::Slice(axes) => {
                let i = NamedAxes::ALL.iter().position(|&a| a == axes)?;
                self.slice_picks[i].pick(p.x * self.scale_factor, p.y * self.scale_factor)
            }
        }
    }

    /// The segment of layer `layer` picked under the pointer (a mesh or
    /// skeleton).
    fn picked_segment(&self, layer: usize) -> Option<u64> {
        let Some(PickTarget::Segment(id, segment)) = self.picked() else { return None };
        (self.layers.get(layer)?.id == id).then_some(segment)
    }

    /// The annotation picked under the pointer: its layer's index and ID.
    pub fn picked_annotation(&self) -> Option<(usize, String)> {
        self.picked_annotation_part().map(|(l, a, _)| (l, a))
    }

    /// The annotation picked under the pointer, and its part
    /// (`annotationPart`).
    pub fn picked_annotation_part(&self) -> Option<(usize, String, u32)> {
        let Some(PickTarget::Annotation(id, a, part)) = self.picked() else { return None };
        Some((self.layers.iter().position(|l| l.id == id)?, a, part.index()))
    }

    pub fn hovered_segments(&mut self) -> Vec<Option<u64>> {
        if let Some((layer, id)) = self.segment_hover {
            return (0..self.layers.len()).map(|i| (i == layer).then_some(id)).collect();
        }
        let pos = self.mouse_position();
        (0..self.layers.len())
            .map(|i| {
                if let Some(id) = self.picked_segment(i) {
                    return Some(id);
                }
                let pos = pos.as_ref()?;
                matches!(self.layers[i].kind(), Some(LayerKind::Segmentation(_))).then_some(())?;
                self.layer_value_at(i, pos)?.parse().ok()
            })
            .collect()
    }

    /// `getMousePositionInAnnotationCoordinates`: the pointer's (unsnapped)
    /// position in annotation layer `layer`'s source coordinates, as
    /// float32.
    pub fn mouse_annotation_position(&mut self, layer: usize) -> Option<Vec<f64>> {
        let position = self.mouse_position()?;
        self.global_to_annotation_model(layer, &position)
    }

    /// A global position in annotation layer `layer`'s source coordinates
    /// (float32).
    fn global_to_annotation_model(&self, layer: usize, position: &[f32]) -> Option<Vec<f64>> {
        global_to_model(&self.layers.get(layer)?.annotation.as_ref()?.layer, position)
    }

    /// Deletes annotation `id` of annotation layer `layer`
    /// (`LocalAnnotationSource.delete`).
    pub fn delete_annotation(&mut self, layer: usize, id: &str) {
        let Some(l) = self.layers.get_mut(layer) else { return };
        let Some(Value::Array(list)) = l.spec.json.get_mut("annotations") else { return };
        list.retain(|a| a.get("id").and_then(Value::as_str) != Some(id));
        self.setup_annotations(layer);
    }

    /// Adds an annotation (`LocalAnnotationSource.add`, its JSON) to
    /// annotation layer `layer`.
    pub fn add_annotation(&mut self, layer: usize, annotation: Value) {
        let Some(l) = self.layers.get_mut(layer) else { return };
        let mut list = l.spec.json.get("annotations").and_then(Value::as_array).cloned().unwrap_or_default();
        list.push(annotation);
        l.spec.json.insert("annotations".into(), Value::Array(list));
        self.setup_annotations(layer);
    }

    /// Changes annotation `id` of annotation layer `layer` in place
    /// (`LocalAnnotationSource.update`): `None` if it is not there, else
    /// whether it changed.
    pub fn update_annotation(&mut self, layer: usize, id: &str, f: impl FnOnce(&mut Value)) -> Option<bool> {
        let l = self.layers.get_mut(layer)?;
        let Some(Value::Array(list)) = l.spec.json.get_mut("annotations") else { return None };
        let a = list.iter_mut().find(|a| a.get("id").and_then(Value::as_str) == Some(id))?;
        let before = a.clone();
        f(a);
        let changed = *a != before;
        if changed {
            self.setup_annotations(layer);
        }
        Some(changed)
    }

    /// `moveToSegment`: the position moves to the loaded mesh vertex of
    /// segment `id` nearest it (`MeshLayer.getObjectPosition`, looking at
    /// about 100 000 vertices). Whether one was loaded.
    pub fn move_to_segment(&mut self, layer: usize, id: u64) -> bool {
        let Some(managed) = self.layers.get(layer) else { return false };
        let Some(state) = managed.render.as_ref() else { return false };
        let l = &state.layer;
        let (rank, n) = (l.rank, l.rank + 1);
        let model_rank = l.source_rank.min(3);
        for (k, source) in managed.mesh_sources().iter().enumerate() {
            // Multi-resolution: the centre of the manifest's bounds.
            let multiscale_center = if source.multiscale.is_some() {
                let Some(manifest) = self.manifests.get(&(managed.id, k, id)) else { continue };
                Some(std::array::from_fn::<f32, 3, _>(|i| ((manifest.clip_lower[i] as f64 + manifest.clip_upper[i] as f64) / 2.0) as f32))
            } else {
                None
            };
            let mesh = match multiscale_center {
                Some(_) => Arc::new(ng_datasource::mesh::MeshData::default()),
                None => {
                    let Some(mesh) = self.meshes.get(&(managed.id, k, id)) else { continue };
                    if mesh.positions.len() < 3 {
                        continue;
                    }
                    mesh
                }
            };
            // Mesh coordinates to the layer's: `to_model` in the model's
            // first dimensions, then model to layer.
            let mut to_model = ng_core::matrix::identity(n, n);
            for c in 0..model_rank {
                for r in 0..model_rank {
                    to_model[c * n + r] = source.to_model[c * 4 + r];
                }
                to_model[(n - 1) * n + c] = source.to_model[12 + c];
            }
            let a = ng_core::matrix::multiply(&l.model_to_layer, &to_model, n);
            let (inv, _) = ng_core::matrix::inverse(&a, n);
            let position = self.nav.position().to_vec();
            let mut layer_pos = vec![0.0; n];
            layer_pos[rank] = 1.0;
            for (d, g) in l.layer_to_global.iter().enumerate() {
                if let Some(g) = g {
                    layer_pos[d] = position.get(*g).copied().unwrap_or(0.0) as f64;
                }
            }
            let apply = |m: &[f64], p: &[f64]| -> Vec<f64> { (0..rank).map(|r| (0..n).map(|c| m[c * n + r] * p[c]).sum()).collect() };
            let near = apply(&inv, &layer_pos);
            let vertex_count = mesh.positions.len() / 3;
            let interval = ((vertex_count as f64 / 100_000.0).floor() as usize).max(1);
            let mut best: Option<(f64, [f32; 3])> = None;
            for v in mesh.positions.chunks_exact(3).step_by(interval) {
                let d: f64 = (0..3).map(|j| (v[j] as f64 - near.get(j).copied().unwrap_or(0.0)).powi(2)).sum();
                if best.is_none_or(|(b, _)| d < b) {
                    best = Some((d, [v[0], v[1], v[2]]));
                }
            }
            let Some((_, v)) = multiscale_center.map(|c| (0.0, c)).or(best) else { continue };
            let mut p = vec![0.0; n];
            p[..3.min(rank)].copy_from_slice(&[v[0] as f64, v[1] as f64, v[2] as f64][..3.min(rank)]);
            p[rank] = 1.0;
            let target = apply(&a, &p);
            let mut position = position;
            for (d, g) in l.layer_to_global.iter().enumerate() {
                if let Some(g) = g {
                    if let Some(x) = position.get_mut(*g) {
                        *x = target[d] as f32;
                    }
                }
            }
            self.nav.set_position(&position);
            return true;
        }
        false
    }

    /// The annotation hovered (`hoverState`): a list entry under the
    /// pointer, or one picked in a panel; whether that changed.
    pub fn set_annotation_hover(&mut self, hover: Option<(usize, String)>) -> bool {
        let hover = hover.and_then(|(l, id)| Some((self.layers.get(l)?.id, id)));
        let changed = self.annotation_hover != hover;
        self.annotation_hover = hover;
        changed
    }

    /// The annotation hovered: its layer's index and ID.
    pub fn annotation_hover(&self) -> Option<(usize, String)> {
        let (id, a) = self.annotation_hover.as_ref()?;
        Some((self.layers.iter().position(|l| l.id == *id)?, a.clone()))
    }

    /// The ID of annotation layer `layer`'s `n`th annotation.
    pub fn annotation_id(&self, layer: usize, n: usize) -> Option<String> {
        Some(self.layers.get(layer)?.annotation.as_ref()?.annotations.get(n)?.id.clone())
    }

    /// A segment hovered in a list (`None` once the pointer leaves it);
    /// whether that changed.
    pub fn set_segment_hover(&mut self, hover: Option<(usize, u64)>) -> bool {
        let changed = self.segment_hover != hover;
        self.segment_hover = hover;
        changed
    }

    /// One line describing the pointer's position and the values under it.
    pub fn status(&mut self) -> String {
        if !self.has_layer() {
            return "neuroglancer-rs — no layer".to_owned();
        }
        let space = self.nav.coordinate_space().clone();
        let fmt = |pos: &[f32]| {
            space
                .names
                .iter()
                .zip(pos)
                .map(|(n, c)| format!("{n} {}", c.floor()))
                .collect::<Vec<_>>()
                .join("  ")
        };
        match self.mouse_position() {
            Some(pos) => {
                let values: Vec<String> = (0..self.layers.len())
                    .filter(|&i| self.layers[i].render.is_some())
                    .map(|i| format!("{}: {}", self.layers[i].name(), self.layer_value_at(i, &pos).unwrap_or_default()))
                    .collect();
                format!("{}  |  {}", fmt(&pos), values.join("  "))
            }
            None => fmt(self.nav.position()),
        }
    }

    /// The 3D panel (`PerspectivePanel.drawWithPicking`, default content):
    /// the slice views as planes, the layers' data bounds, the axis lines.
    #[allow(clippy::too_many_arguments)]
    fn render_perspective(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        phys: PixelRect,
        slice_views: &[SliceViewFrame],
        info: &ng_core::navigation::DisplayDimensionRenderInfo,
        position: &[f32],
        cross_section_background: [f32; 3],
    ) {
        use ng_core::glm::{mat4_invert, mat4_multiply, vec3_transform_quat, vec4_transform_mat4};
        let Some(camera) = self.perspective_camera() else { return };
        let hovered = self.hovered_segments();
        let skeletons = self.skeleton_draws(&hovered, info, false);
        self.pick_ids.clear();
        let (lw, lh) = self.logical_size();
        let logical = self.panel_logical_rect(Panel::Perspective, lw, lh).unwrap_or(phys);
        let vp = camera.view_projection;
        // Lighting: `lightDirection` is the view's −z — in scene mode the host
        // camera's, whose z axis (in display coordinates) is `inv_view`'s third
        // column; the planes' shading uses |light · normal|, so only the axis counts.
        let l = match &self.scene {
            Some(_) => {
                let z = [camera.inv_view[8], camera.inv_view[9], camera.inv_view[10]];
                let n = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2]).sqrt().max(1e-30);
                [z[0] / n, z[1] / n, z[2] / n]
            }
            None => vec3_transform_quat([0.0, 0.0, 1.0], self.nav.orientation(View::Projection)),
        };
        let light = [-(l[0] as f64), -(l[1] as f64), -(l[2] as f64)];
        // in scene mode the slices are drawn whatever the state's layout
        let show_planes = self.show_slices && (self.scene.is_some() || !matches!(self.layout, Layout::Perspective));
        let planes: Vec<ng_render::perspective::SlicePlane> = slice_views
            .iter()
            .filter(|_| show_planes)
            .filter_map(|&(i, inv_view, (sw, sh))| {
                let frame = self.panel_frames[i].as_ref()?;
                // `viewportNormalInCanonicalCoordinates`.
                let view = mat4_invert(&inv_view)?;
                let mut normal = [0f64; 3];
                for k in 0..3 {
                    normal[k] = (view[k * 4 + 2] as f64 / info.canonical_voxel_factors[k]) as f32 as f64;
                }
                let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                if len > 0.0 {
                    normal = normal.map(|x| (x * (1.0 / len)) as f32 as f64);
                }
                let scalar = (light[0] * normal[0] + light[1] * normal[1] + light[2] * normal[2]).abs();
                let factor = (0.2 + scalar * 0.8) as f32;
                // Maps (±1, ±1, 0) to the slice view's corners, then to clip.
                let mut m = [0f32; 16];
                m[0] = (sw as f64 / 2.0) as f32;
                m[5] = (-(sh as f64) / 2.0) as f32;
                m[10] = 1.0;
                m[15] = 1.0;
                let m = mat4_multiply(&vp, &mat4_multiply(&inv_view, &m));
                let corner = |x: f64, y: f64| vec4_transform_mat4([x, y, 0.0, 1.0], &m);
                Some(ng_render::perspective::SlicePlane {
                    texture: &frame.target.color_view,
                    corners: [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)],
                    factor,
                })
            })
            .collect();
        // Annotations: the data-bounds box edges.
        let inv_vp = mat4_invert(&vp).unwrap_or(vp);
        let mut lines = vec![];
        for managed in self.layers.iter().filter(|l| l.spec.visible) {
            let Some(state) = managed.render.as_ref() else { continue };
            let color = managed.annotation_color();
            for (lower, upper) in if self.settings.show_default_annotations { managed.data_bounds() } else { vec![] } {
                let Some((edges, clip)) = chunks::box_edges_3d(&state.layer, &lower, &upper, info, position, &inv_vp) else {
                    continue;
                };
                for (a, b) in edges {
                    lines.push(ng_render::perspective::Line3 {
                        a: vec4_transform_mat4([a[0], a[1], a[2], 1.0], &vp),
                        b: vec4_transform_mat4([b[0], b[1], b[2], 1.0], &vp),
                        width: 1.0,
                        feather: 1.0,
                        color: [color[0], color[1], color[2], clip as f32],
                        color_b: [color[0], color[1], color[2], clip as f32],
                        pick: 0,
                    });
                }
            }
        }
        // Annotation layers' annotations in 3D (`PerspectiveViewAnnotationLayer`).
        let mut circles = vec![];
        let mut triangles = vec![];
        let light_vec = [light[0] * 0.8, light[1] * 0.8, light[2] * 0.8, 0.2];
        let projection = camera.projection.map(|x| x as f64);
        let view = mat4_inverse_f64(&camera.inv_view.map(|x| x as f64));
        let mut spatial: HashMap<LayerId, Vec<annotations::Annotation>> = HashMap::new();
        for li in 0..self.layers.len() {
            if !self.layers[li].spec.visible {
                continue;
            }
            if let Some(list) = spatial_annotations(&mut self.layers, &mut self.annotation_cells, li, AnnotationView::Projection, self.frame, info, position, &projection, &view, (phys.width as f64, phys.height as f64)) {
                spatial.insert(self.layers[li].id, list);
            }
        }
        // Each layer's styles (its shader's), once computed.
        let mut styles: HashMap<LayerId, Arc<Vec<ng_render::annotation_shader::AnnotationStyle>>> = HashMap::new();
        for managed in self.layers.iter().filter(|l| l.spec.visible) {
            let Some(ann) = managed.annotation.as_ref() else { continue };
            let list: &[annotations::Annotation] = spatial.get(&managed.id).map_or(&ann.annotations, |v| v);
            if let Some(s) = annotation_styles(&mut self.annotation_styles, &self.device, &self.queue, &self.wake, managed, list, true, self.frame) {
                styles.insert(managed.id, s);
            }
        }
        for managed in self.layers.iter().filter(|l| l.spec.visible) {
            let Some(ann) = managed.annotation.as_ref() else { continue };
            let list: &[annotations::Annotation] = spatial.get(&managed.id).map_or(&ann.annotations, |v| v);
            let Some(layer_styles) = styles.get(&managed.id) else { continue };
            let clip = |model: &[f64]| {
                let d = model_to_display(&ann.layer, info, model);
                vec4_transform_mat4([d[0], d[1], d[2], 1.0], &vp)
            };
            let hovered = self.annotation_hover.as_ref().filter(|(l, _)| *l == managed.id).map(|(_, id)| id.clone());
            for (k, a) in list.iter().enumerate() {
                let Some(s) = layer_styles.get(k).filter(|s| !s.discard) else { continue };
                let lit = |c: [f32; 4]| highlight(c, hovered.as_deref() == Some(a.id.as_str()));
                // The parts' pick IDs: the object, its endpoints, a corner.
                let [body, end0, end1] = [AnnotationPart::Body, AnnotationPart::Endpoint(0), AnnotationPart::Endpoint(1)].map(|part| {
                    self.pick_ids.push(PickTarget::Annotation(managed.id, a.id.clone(), part));
                    self.pick_ids.len() as u32
                });
                let corner_picks: Vec<u32> = if matches!(a.geometry, annotations::Geometry::AxisAlignedBoundingBox(..)) {
                    (0..8)
                        .map(|k| {
                            self.pick_ids.push(PickTarget::Annotation(managed.id, a.id.clone(), AnnotationPart::Corner(k)));
                            self.pick_ids.len() as u32
                        })
                        .collect()
                } else {
                    vec![]
                };
                let circle = |center: [f64; 4], diameter: f32, border_width: f32, feather: f64, interior: [f32; 4], border: [f32; 4], pick: u32| ng_render::annotations::Circle { center, diameter: diameter as f64, border_width: border_width as f64, feather, interior: lit(interior), border, pick };
                let line = |a: [f64; 4], b: [f64; 4], width: f32, colors: [[f32; 4]; 2]| ng_render::perspective::Line3 { a, b, width: width as f64, feather: 1.0, color: lit(colors[0]), color_b: lit(colors[1]), pick: body };
                match &a.geometry {
                    annotations::Geometry::Point(p) => circles.push(circle(clip(p), s.point_size, s.point_border_width, 1.0, s.point_color, s.point_border_color, body)),
                    annotations::Geometry::Line(pa, pb) => {
                        let (ca, cb) = (clip(pa), clip(pb));
                        lines.push(line(ca, cb, s.line_width, s.line_color));
                        for (e, (c, pick)) in [(ca, end0), (cb, end1)].into_iter().enumerate() {
                            circles.push(circle(c, s.marker_size[e], s.marker_border_width[e], 0.5, s.marker_color[e], s.marker_border_color[e], pick));
                        }
                    }
                    annotations::Geometry::Polyline(points) => {
                        let clips: Vec<[f64; 4]> = points.iter().map(|p| clip(p)).collect();
                        for w in clips.windows(2) {
                            lines.push(line(w[0], w[1], s.poly_line_width, s.poly_line_color));
                        }
                        if clips.len() >= 2 {
                            for (k, &c) in clips.iter().enumerate() {
                                let e = usize::from(k + 1 == clips.len());
                                circles.push(circle(c, s.poly_marker_size[e], s.poly_marker_border_width[e], 0.5, s.poly_marker_color[e], s.poly_marker_border_color[e], if e == 1 { end1 } else { end0 }));
                            }
                        }
                    }
                    annotations::Geometry::Ellipsoid { center, radii } => {
                        // `SphereRenderHelper(10, 10)`, lit per vertex.
                        let fill = lit(s.ellipsoid_fill_color);
                        let c = model_to_display(&ann.layer, info, center);
                        let r = linear_to_display(&ann.layer, info, radii);
                        for v in sphere_triangles(10, 10) {
                            let p = [v[0] * r[0] + c[0], v[1] * r[1] + c[1], v[2] * r[2] + c[2]];
                            let n = [v[0] / r[0].max(1e-6), v[1] / r[1].max(1e-6), v[2] / r[2].max(1e-6)];
                            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                            let n = n.map(|x| x / len);
                            let lighting = ((n[0] * light_vec[0] + n[1] * light_vec[1] + n[2] * light_vec[2]).abs() + light_vec[3]) as f32;
                            triangles.push(ng_render::perspective::ColoredVertex {
                                clip: vec4_transform_mat4([p[0], p[1], p[2], 1.0], &vp),
                                color: [fill[0] * lighting, fill[1] * lighting, fill[2] * lighting, fill[3]],
                            });
                        }
                    }
                    annotations::Geometry::AxisAlignedBoundingBox(pa, pb) => {
                        // Twelve edges (`vec4(vColor.rgb, getLineAlpha())`:
                        // opaque), and the corners as circles one line-width
                        // across without border or feather.
                        let border = s.box_border_color;
                        let opaque = [border[0], border[1], border[2], 1.0];
                        let corner = |bits: usize| -> Vec<f64> { (0..pa.len()).map(|d| if d < 3 && bits >> d & 1 == 1 { pa[d].max(pb[d]) } else { pa[d].min(pb[d]) }).collect() };
                        let corners: Vec<[f64; 4]> = (0..8).map(|b| clip(&corner(b))).collect();
                        for i in 0..8 {
                            for k in 0..3 {
                                let j = i | (1 << k);
                                if j != i {
                                    lines.push(line(corners[i], corners[j], s.box_border_width, [opaque; 2]));
                                }
                            }
                        }
                        for (k, c) in corners.into_iter().enumerate() {
                            circles.push(circle(c, s.box_border_width, 0.0, 0.0, opaque, opaque, corner_picks[k]));
                        }
                    }
                }
            }
        }
        // Segmentation layers' skeletons (`PerspectiveViewSkeletonLayer`),
        // drawn with the opaque layers, before the annotations.
        // Transparent ones (object alpha below 1) with the transparent
        // meshes.
        let (mut skeleton_lines, mut skeleton_circles) = (vec![], vec![]);
        let mut transparent = ng_render::perspective::TransparentMarks::default();
        let pick_ids = &mut self.pick_ids;
        for d in &skeletons {
            // `pickIDs.registerUint64`: each segment's ID, from 1.
            pick_ids.push(PickTarget::Segment(d.segment.0, d.segment.1));
            let pick = pick_ids.len() as u32;
            let line3 = |l: ng_render::annotations::Line| ng_render::perspective::Line3 { a: l.a, b: l.b, width: l.width, feather: l.feather, color: l.color, color_b: l.color_b, pick };
            let (mut l, mut c) = (vec![], vec![]);
            d.marks(&|p| vec4_transform_mat4([p[0], p[1], p[2], 1.0], &vp), 0.0, &mut l, &mut c);
            for circle in &mut c {
                circle.pick = pick;
            }
            if d.color[3] < 1.0 {
                transparent.lines.extend(l.into_iter().map(line3));
                transparent.circles.extend(c);
            } else {
                skeleton_lines.extend(l.into_iter().map(line3));
                skeleton_circles.extend(c);
            }
        }
        lines.splice(0..0, skeleton_lines);
        circles.splice(0..0, skeleton_circles);
        // Axis lines (`drawAxisLines`): `zoom · min(w, h) / h / 4` long.
        let axes = self.show_axis_lines.then(|| {
            let r = logical;
            let axis_length = self.nav.zoom(View::Projection) * ((r.width.min(r.height) as f64) / r.height as f64 / 4.0);
            let mut origin = [0.0; 3];
            for (k, o) in origin.iter_mut().enumerate() {
                let g = info.display_dimension_indices[k];
                *o = if g < 0 { 0.0 } else { position[g as usize] as f64 };
            }
            let clip = |p: [f64; 3]| vec4_transform_mat4([p[0], p[1], p[2], 1.0], &vp);
            let ends = std::array::from_fn(|k| {
                let mut p = origin;
                p[k] += axis_length / info.canonical_voxel_factors[k];
                clip(p)
            });
            (clip(origin), ends)
        });
        // Segmentation layers' meshes (`MeshLayer.draw`,
        // `MultiscaleMeshLayer.draw`): each visible segment's, in its colour
        // (`getObjectColor`), lit by `uLightDirection`.
        let light_vec = [(light[0] * 0.8) as f32, (light[1] * 0.8) as f32, (light[2] * 0.8) as f32, 0.2];
        let mut mesh_jobs: Vec<MeshJob> = vec![];
        // Segments' meshes or manifests wanted, by layer and source.
        // (layer, source, segment, mvp, normal matrix, colour, detail cutoff,
        // the source if multi-resolution).
        // Then the silhouette power and the scale multiplier (the model
        // matrix's physical scale, for the render scale histogram).
        type Wanted = (LayerId, usize, u64, [f32; 16], [f32; 9], [f32; 4], f64, Option<ng_datasource::mesh::MeshSource>, f32, f64);
        // Render scale histogram counts: (layer, spatial scale, render
        // scale, present, not present).
        let mut histogram_adds: Vec<(LayerId, f64, f64, u32, u32)> = vec![];
        let mut histogram_layers: Vec<LayerId> = vec![];
        let mut segments_wanted: Vec<Wanted> = vec![];
        for (li, managed) in self.layers.iter().enumerate() {
            if !managed.spec.visible {
                continue;
            }
            let (Some(state), Some(LayerKind::Segmentation(display))) = (managed.render.as_ref(), managed.kind()) else { continue };
            let sources = managed.mesh_sources();
            if sources.is_empty() {
                continue;
            }
            let alpha = managed.spec.json.get("objectAlpha").and_then(Value::as_f64).unwrap_or(1.0).min(1.0) as f32;
            if alpha <= 0.0 {
                continue;
            }
            // `meshRenderScale` (`renderScaleTarget`): the level-of-detail
            // cutoff, in pixels.
            let detail_cutoff = managed.mesh_render_scale_target();
            let silhouette = managed.silhouette_rendering() as f32;
            histogram_layers.push(managed.id);
            let Ok((_, segments)) = state::segmentation_settings(&managed.spec.json) else { continue };
            for (k, source) in sources.iter().enumerate() {
                let Some(model) = mesh_model_matrix(&state.layer, &source.to_model, info) else { continue };
                let mvp = mat4_multiply(&vp, &model);
                let normal_matrix = mesh_normal_matrix(&model, info.canonical_voxel_factors);
                let scale_multiplier = mesh_scale_multiplier(&model, info.voxel_physical_scales);
                for &(id, visible) in &segments {
                    if !visible {
                        continue;
                    }
                    let color = object_color(display, hovered.get(li).copied().flatten(), id, alpha);
                    let multiscale = source.multiscale.is_some().then(|| source.clone());
                    segments_wanted.push((managed.id, k, id, mvp, normal_matrix, color, detail_cutoff, multiscale, silhouette, scale_multiplier));
                }
            }
        }
        let (pw, ph) = (phys.width as f64, phys.height as f64);
        self.fragment_reads.clear();
        for (layer_id, k, id, mvp, normal_matrix, color, detail_cutoff, multiscale, silhouette, scale_multiplier) in segments_wanted {
            let key = (layer_id, k, id);
            let mut counted = (1u32, 0u32);
            let is_multiscale = multiscale.is_some();
            match multiscale {
                None => {
                    self.meshes.request(&key, 0.0);
                    let ready = self.meshes.get(&key);
                    counted.1 = u32::from(ready.is_some());
                    if ready.is_none() {
                        histogram_adds.push((layer_id, f64::INFINITY, f64::INFINITY, 0, 1));
                        if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                            l.progress.0 += counted.0;
                        }
                        continue;
                    }
                    if let Some(data) = ready {
                        // The manifest and its fragments.
                        histogram_adds.push((layer_id, f64::INFINITY, f64::INFINITY, 1 + data.fragments_loaded, data.fragments - data.fragments_loaded));
                        let fresh = !matches!(self.gpu_meshes.get(&key), Some((d, _)) if Arc::ptr_eq(d, &data));
                        if fresh && !data.indices.is_empty() {
                            let gpu = ng_render::perspective::GpuMesh::new(&self.device, &data.positions, &data.normals, &data.indices);
                            self.gpu_meshes.insert(key, (data, gpu));
                        }
                        if let Some((_, gpu)) = self.gpu_meshes.get(&key) {
                            self.pick_ids.push(PickTarget::Segment(layer_id, id));
                            let pick = self.pick_ids.len() as u32;
                            mesh_jobs.push(MeshJob { mesh: MeshRef::Whole(key), pick, mvp, normal_matrix, color, silhouette, indices: 0..gpu.count, origin: [0.0; 3], shape: [1.0; 3] });
                        }
                    }
                }
                Some(source) => {
                    let info_ms = source.multiscale.clone().unwrap();
                    self.manifests.request(&key, 1.0);
                    let Some(manifest) = self.manifests.get(&key) else {
                        if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                            l.progress.0 += 1;
                        }
                        histogram_adds.push((layer_id, f64::INFINITY, f64::INFINITY, 0, 1));
                        continue;
                    };
                    counted.1 = 1;
                    self.pick_ids.push(PickTarget::Segment(layer_id, id));
                    let segment_pick = self.pick_ids.len() as u32;
                    let planes = ng_datasource::multiscale_mesh::frustum_planes(&mvp);
                    // The backend's requests: every fragment wanted, coarser first.
                    let mut wanted = vec![];
                    ng_datasource::multiscale_mesh::desired_chunks(&manifest, &mvp, &planes, detail_cutoff, pw, ph, &mut |lod, row, _, empty| {
                        if !empty {
                            wanted.push((lod, row));
                        }
                    });
                    for &(lod, row) in &wanted {
                        let fkey = (layer_id, k, id, row as u32);
                        self.fragments.request(&fkey, lod as f64);
                        counted.0 += 1;
                        counted.1 += u32::from(self.fragments.get(&fkey).is_some());
                        let (path, begin, length) = manifest.fragment_range(id, row);
                        self.fragment_reads.insert(fkey, (source.location.clone(), path, begin, length, info_ms.vertex_quantization_bits, lod != 0));
                    }
                    let fragments = &self.fragments;
                    let mut draws = vec![];
                    ng_datasource::multiscale_mesh::chunks_to_draw(
                        &manifest,
                        &mvp,
                        &planes,
                        detail_cutoff,
                        pw,
                        ph,
                        &mut |lod, row, render_scale| {
                            let has = fragments.get(&(layer_id, k, id, row as u32)).is_some();
                            histogram_adds.push((layer_id, manifest.lod_scales[lod] as f64 * scale_multiplier, render_scale, u32::from(has), u32::from(!has)));
                            has
                        },
                        &mut |lod, row, b, e, _| draws.push((lod, row, b, e)),
                    );
                    for (lod, row, b, e) in draws {
                        let fkey = (layer_id, k, id, row as u32);
                        let Some(data) = self.fragments.get(&fkey) else { continue };
                        let fresh = !matches!(self.gpu_fragments.get(&fkey), Some((d, _)) if Arc::ptr_eq(d, &data));
                        if fresh && !data.indices.is_empty() {
                            let gpu = ng_render::perspective::GpuMesh::new(&self.device, &data.positions, &data.normals, &data.indices);
                            self.gpu_fragments.insert(fkey, (data.clone(), gpu));
                        }
                        let (Some(&ib), Some(&ie)) = (data.sub_chunk_offsets.get(b), data.sub_chunk_offsets.get(e)) else { continue };
                        if ib == ie || !self.gpu_fragments.contains_key(&fkey) {
                            continue;
                        }
                        // `uFragmentOrigin` and `uFragmentShape` (float32).
                        let scale = (1u64 << lod) as f64;
                        let o = &manifest.octree[row * 5..row * 5 + 3];
                        let origin = std::array::from_fn(|i| (manifest.grid_origin[i] as f64 + o[i] as f64 * manifest.chunk_shape[i] as f64 * scale + manifest.vertex_offsets[lod * 3 + i] as f64) as f32);
                        let shape = std::array::from_fn(|i| (manifest.chunk_shape[i] as f64 * scale) as f32);
                        mesh_jobs.push(MeshJob { mesh: MeshRef::Fragment(fkey), pick: segment_pick, mvp, normal_matrix, color, silhouette, indices: ib..ie, origin, shape });
                    }
                }
            }
            // The manifest at an infinite scale (single-resolution meshes'
            // with their fragments, above).
            if is_multiscale {
                histogram_adds.push((layer_id, f64::INFINITY, f64::INFINITY, 1, 0));
            }
            // Counted with the layer's visible chunks (the layer bar's
            // progress), as the manifests and fragments are.
            if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                l.progress.0 += counted.0;
                l.progress.1 += counted.1;
            }
        }
        // `renderScaleHistogram.begin` and `add` for each mesh layer drawn.
        let frame = self.frame;
        for l in self.layers.iter_mut().filter(|l| histogram_layers.contains(&l.id)) {
            l.mesh_render_scale_histogram.begin(frame);
        }
        for (layer_id, spatial, render, present, missing) in histogram_adds {
            if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                l.mesh_render_scale_histogram.add(spatial, render, present, missing);
            }
        }
        let mesh_draws: Vec<ng_render::perspective::MeshDraw> = mesh_jobs
            .iter()
            .filter_map(|j| {
                let mesh = match j.mesh {
                    MeshRef::Whole(key) => &self.gpu_meshes.get(&key)?.1,
                    MeshRef::Fragment(key) => &self.gpu_fragments.get(&key)?.1,
                };
                Some(ng_render::perspective::MeshDraw {
                    mesh,
                    mvp: j.mvp,
                    normal_matrix: j.normal_matrix,
                    light: light_vec,
                    color: j.color,
                    indices: j.indices.clone(),
                    fragment_origin: j.origin,
                    fragment_shape: j.shape,
                    // `isTransparent`: object alpha below 1, or silhouettes.
                    transparent: j.color[3] < 1.0 || j.silhouette > 0.0,
                    silhouette: j.silhouette,
                    pick: j.pick,
                })
            })
            .collect();
        // Image layers' volume rendering (`VolumeRenderingRenderLayer`, "on"):
        // each layer's chunks within the frustum at the scale its depth
        // samples call for.
        let mut volume_jobs: Vec<(usize, TextureKey, ng_render::perspective::VolumeUniforms)> = vec![];
        for l in &mut self.layers {
            l.volume_chunks.clear();
        }
        for li in 0..self.layers.len() {
            let managed = &self.layers[li];
            if !managed.spec.visible {
                continue;
            }
            let (mode, samples, gain) = state::volume_rendering_settings(&managed.spec.json);
            let mode = match mode {
                "on" => ng_render::user_shader::VolumeMode::On,
                "max" => ng_render::user_shader::VolumeMode::Max,
                "min" => ng_render::user_shader::VolumeMode::Min,
                _ => continue,
            };
            if !matches!(managed.kind(), Some(LayerKind::Image { .. })) {
                continue;
            }
            let json = managed.spec.json.clone();
            let id = managed.id;
            let frame = self.frame;
            let Some(state) = self.layers[li].render.as_mut() else { continue };
            let t = state.layer.data_type;
            let Some(shader) = state.volume_shader(&self.device, &self.perspective, &json, mode) else { continue };
            shader.uniforms.set(&self.queue, &ng_render::user_shader::uniform_bits(&shader.parsed, t, &shader.values, 1.0));
            let sources = chunks::panel_sources(&state.layer, info, position);
            let Some(vrc) = chunks::volume_rendering_chunks(&state.layer, &sources, info, &camera.projection, &camera.inv_view, &vp, samples) else { continue };
            let ps = &vrc.source;
            let model = ps.chunk_to_display;
            let mvp64 = ng_core::matrix::multiply(&vp.map(|x| x as f64), &model, 4);
            let mvp: [f32; 16] = std::array::from_fn(|i| mvp64[i] as f32);
            let inv_mvp = ng_core::glm::mat4_invert(&mvp).unwrap_or(mvp);
            // `getVolumeRenderingNearFarBounds`.
            let planes = ng_datasource::multiscale_mesh::frustum_planes(&mvp);
            let (mut min_z, mut max_z) = (0.0f64, 0.0f64);
            for k in 0..3 {
                let c = planes[16 + k] as f64;
                let (a, b) = (c * vrc.lower_clip[k], c * vrc.upper_clip[k]);
                min_z += a.min(b);
                max_z += a.max(b);
            }
            let near = -(planes[19] as f64);
            let far = planes[23] as f64;
            let (adjusted_near, adjusted_far) = (near.max(min_z), far.min(max_z));
            let params = [((adjusted_near - near) / (far - near)) as f32, ((adjusted_far - near) / (far - near)) as f32, (vrc.optimal_samples / samples) as f32, gain.exp() as f32];
            let offsets: Vec<u32> = ps.fixed.iter().map(|f| f.2).collect();
            let (mut present, mut missing) = (0u32, 0u32);
            let mut drawn = vec![];
            for grid in &vrc.chunks {
                let key: ChunkKey = (id, ps.scale, grid.clone());
                let requested = self.chunks.request(&key, 0.0).clone();
                let State::Ready(chunk) = requested else {
                    missing += 1;
                    continue;
                };
                present += 1;
                drawn.push((chunk.clone(), ps.chunk_display_dims, ps.fixed.clone()));
                let tkey: TextureKey = (ps.scale, grid.clone(), offsets.clone());
                let Some(state) = self.layers[li].render.as_mut() else { break };
                if !state.textures.contains_key(&tkey) {
                    let draw = chunks::ChunkDraw { scale: ps.scale, grid: grid.clone(), chunk_display_dims: ps.chunk_display_dims, fixed: ps.fixed.clone(), rect: [0.0; 4], ndc_to_texture: [0.0; 16] };
                    let (format, upload) = state.upload_format();
                    let volume = upload_slab(&self.device, &self.queue, &state.layer, &chunk, &draw, format, upload, true);
                    state.textures.insert(tkey.clone(), ChunkTexture { volume, bindings: Default::default(), last_used: frame });
                }
                if let Some(t) = state.textures.get_mut(&tkey) {
                    t.last_used = frame;
                }
                let mut translation = [0.0f32; 4];
                let mut size = [1.0f32; 4];
                for (k, cd) in ps.chunk_display_dims.iter().enumerate() {
                    if let Some(c) = cd {
                        translation[k] = (grid[*c] as f64 * vrc.chunk_size[k]) as f32;
                        size[k] = chunk.size[*c] as f32;
                    }
                }
                let lower = vrc.lower_clip.map(|x| x as f32);
                let upper = vrc.upper_clip.map(|x| x as f32);
                volume_jobs.push((
                    li,
                    tkey,
                    ng_render::perspective::VolumeUniforms {
                        mvp,
                        inv_mvp,
                        translation,
                        chunk_size: size,
                        lower_clip: [lower[0], lower[1], lower[2], 0.0],
                        upper_clip: [upper[0], upper[1], upper[2], 0.0],
                        params,
                        steps: [samples as u32, 0, 0, 0],
                    },
                ));
            }
            // The histogram: the other scales as indicator bars (each
            // samples count once), then the chunks drawn.
            self.layers[li].volume_chunks = drawn;
            // Counted with the layer's visible chunks (the layer bar).
            self.layers[li].progress.0 += present + missing;
            self.layers[li].progress.1 += present;
            let histogram = &mut self.layers[li].volume_rendering_histogram;
            histogram.begin(frame);
            if present + missing > 0 {
                use ng_core::render_scale::{scale_from_histogram_offset_at, BIN_SIZE, NUM_BINS, VOLUME_RENDERING_INDICATOR_BAR_HEIGHT, VOLUME_RENDERING_ORIGIN};
                // `clampAndRoundResolutionTargetValue`.
                let max = scale_from_histogram_offset_at(NUM_BINS as f64, VOLUME_RENDERING_ORIGIN).log2().round().exp2() - 1.0;
                let _ = BIN_SIZE;
                let round = |v: f64| v.round().clamp(VOLUME_RENDERING_ORIGIN.exp2(), max) as i64;
                let mut stored = vec![round(vrc.optimal_samples)];
                for (index, &(spacing, optimal)) in (0..vrc.scales.len()).rev().zip(&vrc.scales) {
                    let r = round(optimal);
                    if index != vrc.active && !stored.contains(&r) {
                        histogram.add_indicator(spacing, optimal, VOLUME_RENDERING_INDICATOR_BAR_HEIGHT);
                        stored.push(r);
                    }
                }
                histogram.add(vrc.physical_spacing, vrc.optimal_samples, present, missing);
            }
        }
        // scene mode hands the depth to the host, which binds it to composite
        let sampled = self.scene.is_some();
        if !matches!(&self.perspective_frame, Some(pf) if pf.target.width == phys.width && pf.target.height == phys.height && pf.target.sampled_depth == sampled) {
            let target = if sampled { PanelTarget::with_sampled_depth(&self.device, phys.width, phys.height) } else { PanelTarget::new(&self.device, phys.width, phys.height) };
            let to_panel = self.panel_compositor.create_binding(&self.device, &target.color_view);
            let to_frame = self.compositor.create_binding(&self.device, &target.panel_view);
            self.perspective_frame = Some(PanelFrame { target, to_panel, to_frame });
        }
        let pf = self.perspective_frame.as_ref().unwrap();
        self.perspective.draw(
            &self.device,
            encoder,
            &pf.target.flipped_view,
            &pf.target.depth_view,
            (phys.width, phys.height),
            // scene mode: transparent black outside the data (premultiplied,
            // so the host's frame shows through), and no slice background
            if self.scene.is_some() { [0.0; 3] } else { self.projection_background },
            (self.scene.is_none() && !self.settings.hide_cross_section_background_3d).then_some(cross_section_background),
            &mesh_draws,
            &planes,
            &lines,
            &circles,
            &triangles,
            axes,
            &transparent,
            &volume_jobs
                .iter()
                .filter_map(|(li, tkey, uniforms)| {
                    let state = self.layers[*li].render.as_ref()?;
                    let shader = state.volume.as_ref()?;
                    Some(ng_render::perspective::VolumeDraw { layer: *li, pipeline: &shader.pipeline, volume: &state.textures.get(tkey)?.volume.view, lookup: &shader.uniforms.view, uniforms: *uniforms })
                })
                .collect::<Vec<_>>(),
        );
        // `attemptToIssuePickRequest`: the window around the pointer, if
        // it is over this panel and no request is outstanding.
        if self.pending_pick.is_none() {
            if let Some((Panel::Perspective, p)) = self.pointer_panel() {
                let s = self.scale_factor;
                let x = (p.x * s).floor() as i64;
                let y = (phys.height as f64 - p.y * s).floor() as i64;
                if let Some(readback) = self.perspective.read_pick(&self.device, encoder, x, y, PICK_RADIUS) {
                    let display = info.display_dimension_indices;
                    self.pending_pick = Some(PendingPick { readback, inv_vp, position: position.to_vec(), display, ids: self.pick_ids.clone(), mapped: false });
                }
            }
        }
        self.flipper.flip(&self.device, encoder, &pf.target.flipped_view, &pf.target.panel_view);
        self.compositor.draw(&self.queue, encoder, target, phys, &pf.to_frame, cross_section_background);
    }

    /// Draws a frame into `target`: `width` × `height` physical pixels in the
    /// format the viewer was created with. Requests the chunks it needs;
    /// [`poll`](Self::poll) and redraw as they arrive.
    pub fn render(&mut self, target: &wgpu::TextureView) {
        self.frame += 1;
        self.process_legends();
        self.chunks.max_concurrent_downloads = self.settings.concurrent_downloads.max(1.0) as usize;
        self.chunks.memory_budget = self.settings.system_memory_limit.max(0.0) as usize;
        self.chunks.poll();
        self.chunks.begin_frame();
        self.meshes.poll();
        self.meshes.begin_frame();
        self.skeletons.poll();
        self.skeletons.begin_frame();
        self.annotation_cells.poll();
        self.annotation_cells.begin_frame();
        for layer in &mut self.layers {
            layer.update_annotation_shader(&self.device);
        }
        self.annotations_by_id.poll();
        self.annotations_by_id.begin_frame();
        for key in std::mem::take(&mut self.annotation_wants) {
            self.annotations_by_id.request(&key, 0.0);
        }
        self.manifests.poll();
        self.manifests.begin_frame();
        self.fragments.poll();
        self.fragments.begin_frame();
        let panel_rects = self.panel_rects();
        for l in &mut self.layers {
            l.progress = (0, 0);
        }

        // Black between panels (the page behind neuroglancer's canvas).
        let layout = self.layout;
        {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            drop(threers::renderer::fullscreen::begin(&mut encoder, target, threers::renderer::fullscreen::Load::Clear));
            self.queue.submit([encoder.finish()]);
        }
        // Each slice panel's view, for the 3D panel: (axes, invViewMatrix,
        // physical size).
        let mut slice_views: Vec<SliceViewFrame> = vec![];

        let hovered = self.hovered_segments();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let (lw, lh) = self.logical_size();
        let position = self.nav.position().to_vec();
        let info = self.nav.render_info().clone();
        let valid = self.nav.valid();
        let frame = self.frame;
        let background = self.background;
        let has_slices = panel_rects.iter().any(|(p, _)| matches!(p, Panel::Slice(_)));
        let slice_skeletons = if has_slices && valid { self.skeleton_draws(&hovered, &info, true) } else { vec![] };
        for &(panel, phys) in &panel_rects {
            let Panel::Slice(axes) = panel else { continue };
            let i = NamedAxes::ALL.iter().position(|&a| a == axes).unwrap();
            let logical = self.panel_logical_rect(panel, lw, lh).unwrap();
            let pf = match &mut self.panel_frames[i] {
                Some(pf) if pf.target.width == phys.width && pf.target.height == phys.height => pf,
                slot => {
                    let target = PanelTarget::new(&self.device, phys.width, phys.height);
                    let to_panel = self.panel_compositor.create_binding(&self.device, &target.color_view);
                    let to_frame = self.compositor.create_binding(&self.device, &target.panel_view);
                    slot.insert(PanelFrame { target, to_panel, to_frame })
                }
            };
            let panel_rect_local = PixelRect { x: 0, y: 0, width: phys.width, height: phys.height };
            pf.target.clear(&mut encoder);
            if !valid {
                self.panel_compositor.draw(&self.queue, &mut encoder, &pf.target.panel_view, panel_rect_local, &pf.to_panel, background);
                self.compositor.draw(&self.queue, &mut encoder, target, phys, &pf.to_frame, background);
                continue;
            }
            // `velocityEstimator.addSample` (with a clock to time it by).
            let prefetching = match (self.settings.prefetch, self.chunks.clock) {
                (true, Some(clock)) => {
                    self.velocity[i].add_sample(&position, clock());
                    true
                }
                _ => false,
            };
            let mut sp = self.panels[i].1.clone();
            let inv_view = sp.inv_view_matrix(&mut self.nav);
            self.panels[i].1 = sp;
            slice_views.push((i, inv_view, (phys.width, phys.height)));
            let camera = PanelCamera::new(inv_view, logical.width as f64, logical.height as f64);
            let sx = phys.width as f64 / logical.width as f64;
            let sy = phys.height as f64 / logical.height as f64;
            // Visible layers in order; the first drawn replaces, later ones
            // blend (`setGLBlendMode`).
            let mut drawn = 0;
            for (li, managed) in self.layers.iter_mut().enumerate() {
                if !managed.spec.visible {
                    continue;
                }
                let id = managed.id;
                let Some(state) = managed.render.as_mut() else { continue };
                // A user shader's control values (the opacity first).
                if let (Some(shader), LayerKind::Image { opacity }) = (&state.shader, &state.kind) {
                    let values = ng_render::user_shader::uniform_bits(&shader.parsed, state.layer.data_type, &shader.values, *opacity);
                    shader.uniforms.set(&self.queue, &values);
                }
                // A replaced visible-set texture invalidates the bindings.
                if let Some((set, seen)) = &mut state.visible {
                    if *seen != set.generation {
                        *seen = set.generation;
                        for t in state.textures.values_mut() {
                            t.bindings = Default::default();
                        }
                    }
                }
                let sources = panel_sources(&state.layer, &info, &position);
                let pixel_size = camera.pixel_size(&info);
                let target = ng_core::render_scale::parse_target(managed.spec.json.get("crossSectionRenderScale"), 1.0);
                let scales = visible_scales(&sources, pixel_size, target);
                if !scales.is_empty() {
                    managed.render_scale_histogram.begin(frame);
                }
                let order = drawn;
                drawn += 1;
                let pipeline = state.pipelines.for_layer(order);
                let mut pass = pf.target.begin_layer(&mut encoder, order);
                // `scales` runs coarse to fine. Draw finest first: under the
                // depth test, coarser data only fills gaps.
                for (priority_index, &s) in scales.iter().enumerate().rev() {
                    let Some(ps) = sources[s].as_ref() else { continue };
                    let draws = visible_chunks(&state.layer, ps, &camera);
                    // `renderScaleHistogram.add`: the median voxel size
                    // (`effectiveVoxelSize` is a `Float32Array`).
                    let v = ps.effective_voxel_size.map(|x| x as f32 as f64);
                    let median = ng_core::render_scale::median_of_3(v[0], v[1], v[2]);
                    let present = draws.iter().filter(|d| self.chunks.get(&(id, s, d.grid.clone())).is_some()).count() as u32;
                    if !draws.is_empty() {
                        managed.render_scale_histogram.add(median, median / pixel_size, present, draws.len() as u32 - present);
                    }
                    // Chunks the moving position will reach, behind every
                    // visible one (the `PREFETCH` tier).
                    if prefetching && !draws.is_empty() {
                        let src = &state.layer.scales[s];
                        let display_dims: Vec<usize> = ps.chunk_display_dims.iter().flatten().copied().collect();
                        let sizes: Vec<u32> = src.spec.chunk_data_size.clone();
                        let rank = sizes.len().min(ps.position_voxel.len());
                        let position_chunk: Vec<i64> = (0..rank).map(|c| (ps.position_voxel[c] / sizes[c] as f64).floor() as i64).collect();
                        let within: Vec<f64> = (0..rank).map(|c| ps.position_voxel[c] - position_chunk[c] as f64 * sizes[c] as f64).collect();
                        let lower: Vec<f64> = (0..rank).map(|c| src.lower_clip_bound.as_ref().map_or(src.spec.lower_voxel_bound[c] as f64, |b| b[c] as f64)).collect();
                        let upper: Vec<f64> = (0..rank).map(|c| src.upper_clip_bound.as_ref().map_or(src.spec.upper_voxel_bound[c] as f64, |b| b[c] as f64)).collect();
                        let geometry = prefetch::SourceGeometry {
                            global_to_chunk: &ps.global_to_chunk[..rank],
                            display_dims: &display_dims,
                            chunk_data_size: &sizes[..rank],
                            position_chunk: &position_chunk,
                            position_in_chunk: &within,
                            lower_clip: &lower,
                            upper_clip: &upper,
                        };
                        let offsets = prefetch::prefetch_offsets(&self.velocity[i], &geometry);
                        if !offsets.is_empty() {
                            let visible: Vec<Vec<u64>> = draws.iter().map(|d| d.grid.clone()).collect();
                            for (grid, probability) in prefetch::prefetch_chunks(&visible, &offsets) {
                                self.chunks.request_prefetch(&(id, s, grid), PREFETCH_TIER + 1e9 * priority_index as f64 + probability);
                            }
                        }
                    }
                    for draw in draws {
                        // Finer scales first; within one, nearer the center first.
                        let cx = draw.rect[0] + draw.rect[2] / 2.0 - camera.width / 2.0;
                        let cy = draw.rect[1] + draw.rect[3] / 2.0 - camera.height / 2.0;
                        let priority = 1e9 * priority_index as f64 - (cx * cx + cy * cy).sqrt();
                        let key: ChunkKey = (id, s, draw.grid.clone());
                        let requested = self.chunks.request(&key, priority).clone();
                        managed.progress.0 += 1;
                        let State::Ready(chunk) = requested else { continue };
                        managed.progress.1 += 1;
                        // Panel-local pixels of the offscreen target.
                        let x0 = (draw.rect[0] * sx).floor().max(0.0);
                        let y0 = (draw.rect[1] * sy).floor().max(0.0);
                        let x1 = ((draw.rect[0] + draw.rect[2]) * sx).ceil().min(phys.width as f64);
                        let y1 = ((draw.rect[1] + draw.rect[3]) * sy).ceil().min(phys.height as f64);
                        if x1 <= x0 || y1 <= y0 {
                            continue;
                        }
                        let rect = PixelRect {
                            x: x0 as u32,
                            y: y0 as u32,
                            width: (x1 - x0) as u32,
                            height: (y1 - y0) as u32,
                        };
                        // The texture matrix was for the logical rectangle; the
                        // physical one is the same region of the panel, rounded
                        // outward, so re-derive its NDC mapping.
                        let ndc_to_voxel = Matrix4 {
                            elements: subrect_matrix(draw.ndc_to_texture, draw.rect, [x0 / sx, y0 / sy, (x1 - x0) / sx, (y1 - y0) / sy]),
                        };
                        let params = state.slice_params(ndc_to_voxel, hovered.get(li).copied().flatten());
                        let offsets: Vec<u32> = draw.fixed.iter().map(|f| f.2).collect();
                        let tkey: TextureKey = (s, draw.grid.clone(), offsets);
                        let (format, upload) = state.upload_format();
                        let texture = state.textures.entry(tkey).or_insert_with(|| ChunkTexture {
                            volume: upload_slab(&self.device, &self.queue, &state.layer, &chunk, &draw, format, upload, matches!(state.kind, LayerKind::Image { .. })),
                            bindings: Default::default(),
                            last_used: frame,
                        });
                        texture.last_used = frame;
                        // Bindings are made per pipeline: a layer's place in
                        // the drawing order picks it.
                        let slot = &mut texture.bindings[i][(order > 0) as usize];
                        if slot.is_none() {
                            // The lookup (the fields, apart from the textures).
                            let lookup = state.visible.as_ref().map(|(set, _)| &set.view).or(state.shader.as_ref().map(|s| &s.uniforms.view));
                            *slot = Some(match lookup {
                                Some(view) => pipeline.create_binding_with_lookup(&self.device, &texture.volume, view),
                                None => pipeline.create_binding(&self.device, &texture.volume),
                            });
                        }
                        let binding = slot.as_ref().unwrap();
                        pipeline.update(&self.queue, binding, &params);
                        pipeline.draw_in_pass(&mut pass, rect, binding);
                    }
                }
            }
            self.panel_compositor.draw(&self.queue, &mut encoder, &pf.target.panel_view, panel_rect_local, &pf.to_panel, background);
            // Annotations over the panel: each visible layer's data bounds.
            let mut lines = vec![];
            for managed in self.layers.iter().filter(|l| l.spec.visible) {
                let Some(state) = managed.render.as_ref() else { continue };
                let color = managed.annotation_color();
                for (lower, upper) in if self.settings.show_default_annotations { managed.data_bounds() } else { vec![] } {
                    let Some((polygon, clip)) = chunks::box_cross_section(&state.layer, &lower, &upper, &info, &position, &camera) else {
                        continue;
                    };
                    for k in 0..polygon.len() {
                        let (a, b) = (polygon[k], polygon[(k + 1) % polygon.len()]);
                        lines.push(ng_render::lines::Line {
                            a: [a[0] * sx, a[1] * sy],
                            b: [b[0] * sx, b[1] * sy],
                            width: 1.0,
                            feather: 1.0,
                            color: [color[0], color[1], color[2], clip as f32],
                        });
                    }
                }
            }
            // Annotation layers' annotations (`SliceViewAnnotationLayer`):
            // boxes' cross-sections with the lines above; points and lines
            // in clip coordinates, fading with depth.
            let mut marks = AnnotationMarks::default();
            let relative_depth = self.nav.relative_depth_range(View::CrossSection);
            let vp = camera.view_projection();
            // Segmentation layers' skeletons (`SliceViewPanelSkeletonLayer`):
            // each segment's edges, then its nodes.
            // What can be picked here (`pickIDs`), ID `k + 1` entry `k`.
            let mut targets: Vec<PickTarget> = vec![];
            let mut skeleton_batches = vec![];
            for d in &slice_skeletons {
                let to_clip = |p: [f64; 3]| {
                    let c = chunks::apply(&vp, p);
                    [c[0], c[1], -c[2] / relative_depth, 1.0]
                };
                let (mut l, mut c) = (vec![], vec![]);
                d.marks(&to_clip, 1.0, &mut l, &mut c);
                targets.push(PickTarget::Segment(d.segment.0, d.segment.1));
                let pick = targets.len() as u32;
                l.iter_mut().for_each(|x| x.pick = pick);
                c.iter_mut().for_each(|x| x.pick = pick);
                skeleton_batches.push(ng_render::annotations::MarkBatch { lines: l, circles: c, ellipses: vec![] });
            }
            // Box cross-sections' lines: from which line, which pick.
            let first_box_line = lines.len();
            let mut box_picks: Vec<(usize, u32)> = vec![];
            // Precomputed sources' spatial index cells in view.
            let mut slice_projection = [0.0; 16];
            slice_projection[0] = 2.0 / camera.width;
            slice_projection[5] = -2.0 / camera.height;
            slice_projection[10] = -1.0 / relative_depth;
            slice_projection[15] = 1.0;
            let slice_view = mat4_inverse_f64(&camera.inv_view);
            let mut spatial: HashMap<LayerId, Vec<annotations::Annotation>> = HashMap::new();
            for li in 0..self.layers.len() {
                if !self.layers[li].spec.visible {
                    continue;
                }
                if let Some(list) = spatial_annotations(&mut self.layers, &mut self.annotation_cells, li, AnnotationView::CrossSection, frame, &info, &position, &slice_projection, &slice_view, (phys.width as f64, phys.height as f64)) {
                    spatial.insert(self.layers[li].id, list);
                }
            }
            // Each layer's styles (its shader's), once computed.
            let mut styles: HashMap<LayerId, Arc<Vec<ng_render::annotation_shader::AnnotationStyle>>> = HashMap::new();
            for managed in self.layers.iter().filter(|l| l.spec.visible) {
                let Some(ann) = managed.annotation.as_ref() else { continue };
                let list: &[annotations::Annotation] = spatial.get(&managed.id).map_or(&ann.annotations, |v| v);
                if let Some(s) = annotation_styles(&mut self.annotation_styles, &self.device, &self.queue, &self.wake, managed, list, false, frame) {
                    styles.insert(managed.id, s);
                }
            }
            for managed in self.layers.iter().filter(|l| l.spec.visible) {
                let Some(ann) = managed.annotation.as_ref() else { continue };
                let list: &[annotations::Annotation] = spatial.get(&managed.id).map_or(&ann.annotations, |v| v);
                let Some(layer_styles) = styles.get(&managed.id) else { continue };
                let to_clip = |model: &[f64]| -> [f64; 4] {
                    let d = model_to_display(&ann.layer, &info, model);
                    let c = chunks::apply(&vp, d);
                    // `mat4.ortho(…, -relativeDepthRange, relativeDepthRange)`.
                    [c[0], c[1], -c[2] / relative_depth, 1.0]
                };
                let hovered = self.annotation_hover.as_ref().filter(|(l, _)| *l == managed.id).map(|(_, id)| id.as_str());
                let mut register = |id: &str, part: AnnotationPart| {
                    targets.push(PickTarget::Annotation(managed.id, id.to_owned(), part));
                    targets.len() as u32
                };
                let mut ellipses = vec![None; list.len()];
                for (k, a) in list.iter().enumerate() {
                    if let annotations::Geometry::Ellipsoid { center, radii } = &a.geometry {
                        let fill = highlight(layer_styles[k].ellipsoid_fill_color, hovered == Some(a.id.as_str()));
                        let to_display = |m: &[f64]| model_to_display(&ann.layer, &info, m);
                        let display_to_clip = |d: [f64; 3]| {
                            let c = chunks::apply(&vp, d);
                            [c[0], c[1], -c[2] / relative_depth, 1.0]
                        };
                        if let Some(corners) = ellipsoid_cross_section(&camera, to_display(center), linear_to_display(&ann.layer, &info, radii)) {
                            // `setEllipsoidFillColor` (`setColor`'s: half alpha here).
                            ellipses[k] = Some(ng_render::annotations::Ellipse { corners: corners.map(display_to_clip), color: fill, pick: 0 });
                        }
                    }
                }
                marks.add(list, &ellipses, layer_styles, hovered, &mut register, &to_clip, &mut |lower, upper, color, width, pick| {
                    let Some((polygon, clip)) = chunks::box_cross_section(&ann.layer, lower, upper, &info, &position, &camera) else { return };
                    box_picks.push((lines.len(), pick));
                    for k in 0..polygon.len() {
                        let (a, b) = (polygon[k], polygon[(k + 1) % polygon.len()]);
                        // `vColor.a * getLineAlpha() * vClipCoefficient`.
                        lines.push(ng_render::lines::Line { a: [a[0] * sx, a[1] * sy], b: [b[0] * sx, b[1] * sy], width, feather: 1.0, color: [color[0], color[1], color[2], color[3] * clip as f32] });
                    }
                });
            }
            // Overlays draw on the panel turned upside down, as in
            // neuroglancer's GL framebuffer, then it turns back.
            self.flipper.flip(&self.device, &mut encoder, &pf.target.panel_view, &pf.target.flipped_view);
            self.lines.draw(&self.device, &mut encoder, &pf.target.flipped_view, panel_rect_local, &lines, true);
            skeleton_batches.append(&mut marks.batches);
            self.annotation_renderer.draw_batches(&self.device, &mut encoder, &pf.target.flipped_view, panel_rect_local, &skeleton_batches, true);
            // The pickable marks in drawing order, in panel pixels.
            let (w, h) = (phys.width as f64, phys.height as f64);
            let mut shapes = vec![];
            for (k, l) in lines.iter().enumerate().skip(first_box_line) {
                let Some(&(_, pick)) = box_picks.iter().rev().find(|(start, _)| *start <= k) else { continue };
                shapes.push(PickShape::segment(l.a, l.b, l.width, l.feather, pick));
            }
            for b in &skeleton_batches {
                shapes.extend(b.lines.iter().filter_map(|l| PickShape::from_line(l, w, h)));
                shapes.extend(b.ellipses.iter().filter_map(|e| PickShape::from_ellipse(e, w, h)));
                shapes.extend(b.circles.iter().filter_map(|c| PickShape::from_circle(c, w, h)));
            }
            self.slice_picks[i] = SlicePicks { size: (phys.width, phys.height), targets, shapes };
            self.last_lines[i] = lines;
            if self.show_axis_lines {
                // `computeAxisLineMatrix`: from the position, `axisLength`
                // pixels (¼ · 1.5 of the panel's smaller side) along each
                // display axis; `disableZProjection` keeps them unclipped.
                let axis_length = (logical.width.min(logical.height) as f64 / 4.0) * 1.5;
                let zoom = self.nav.zoom(View::CrossSection);
                let vp = camera.view_projection();
                let mut origin = [0.0; 3];
                for (k, o) in origin.iter_mut().enumerate() {
                    let g = info.display_dimension_indices[k];
                    *o = if g < 0 { 0.0 } else { position[g as usize] as f64 };
                }
                let ndc = |p: [f64; 3]| {
                    let q = chunks::apply(&vp, p);
                    [q[0] as f32, q[1] as f32]
                };
                let ends = std::array::from_fn(|k| {
                    let mut p = origin;
                    p[k] += axis_length * zoom / info.canonical_voxel_factors[k];
                    ndc(p)
                });
                let o = ndc(origin);
                self.axis_lines.draw(&self.device, &mut encoder, &pf.target.flipped_view, panel_rect_local, o, ends, true);
                // Recorded with the annotation lines (panel pixels).
                let px = |p: [f32; 2]| [(p[0] as f64 + 1.0) / 2.0 * phys.width as f64, (1.0 - p[1] as f64) / 2.0 * phys.height as f64];
                for e in ends {
                    self.last_lines[i].push(ng_render::lines::Line { a: px(o), b: px(e), width: 1.0, feather: 0.0, color: [1.0; 4] });
                }
            }
            // The scale bar: in `4panel` and `4panel-alt` on the xy panel only
            // (the others get `sliceViewerStateWithoutScaleBar`).
            let has_scale_bar = match layout {
                Layout::FourPanel | Layout::FourPanelAlt => axes == NamedAxes::Xy,
                _ => true,
            };
            if self.show_scale_bar && has_scale_bar {
                let options = ng_render::scale_bar::ScaleBarOptions::default();
                let factors = self.nav.relative_display_factors();
                let zoom = self.nav.zoom(View::CrossSection);
                let bars = ng_render::scale_bar::scale_bars(&info, &factors, zoom, logical.width as f64, &options);
                let mut bottom = options.bottom_offset * options.scale_factor;
                for bar in &bars {
                    let key = (bar.text(), bar.length_in_pixels.to_bits());
                    let painter = &self.scale_bar_painter;
                    let image = self.scale_bar_cache.entry(key).or_insert_with(|| painter.paint(bar, &options));
                    // `gl.viewport(left, bottom, …)`, from the panel's bottom:
                    // the flipped panel's top-down y.
                    let x = (options.left_offset * options.scale_factor) as i32;
                    self.scale_bar_renderer.draw(&self.device, &self.queue, &mut encoder, &pf.target.flipped_view, image, x, bottom as i32, true);
                    let y = phys.height as i32 - bottom as i32 - image.height as i32;
                    // Recorded for tests, as one line as thick as the bar.
                    let (x0, y0, x1, y1) = (x as f64, y as f64, (x + image.width as i32) as f64, (y + image.height as i32) as f64);
                    self.last_lines[i].push(ng_render::lines::Line {
                        a: [x0, (y0 + y1) / 2.0],
                        b: [x1, (y0 + y1) / 2.0],
                        width: y1 - y0,
                        feather: 0.0,
                        color: [1.0; 4],
                    });
                    bottom += image.height as f64 + options.margin_between * options.scale_factor;
                }
            }
            self.flipper.flip(&self.device, &mut encoder, &pf.target.flipped_view, &pf.target.panel_view);
            self.compositor.draw(&self.queue, &mut encoder, target, phys, &pf.to_frame, background);
        }
        if valid {
            if let Some(&(_, phys)) = panel_rects.iter().find(|(p, _)| *p == Panel::Perspective) {
                self.render_perspective(&mut encoder, target, phys, &slice_views, &info, &position, background);
            }
        }
        self.queue.submit([encoder.finish()]);
        if let Some(pick) = self.pending_pick.as_mut().filter(|p| !p.mapped) {
            pick.readback.map(self.wake.clone());
            pick.mapped = true;
            // Another frame polls for it: natively and on WebGL2 the mapping
            // only completes as the device is polled, and nothing else may
            // ask for one.
            (self.wake)();
        }
        if valid {
            self.compute_data_histograms(&slice_views);
        }

        // Start loads; drop textures of chunks no longer drawn or held.
        let scales: HashMap<LayerId, Vec<Arc<ng_datasource::model::VolumeSource>>> = self
            .layers
            .iter()
            .filter_map(|l| Some((l.id, l.render.as_ref()?.layer.scales.clone())))
            .collect();
        let mesh_sources: HashMap<LayerId, Vec<ng_datasource::mesh::MeshSource>> = self.layers.iter().map(|l| (l.id, l.mesh_sources())).collect();
        self.meshes.end_frame(|(id, k, segment)| {
            let source = mesh_sources.get(id).and_then(|v| v.get(*k)).cloned();
            let segment = *segment;
            Box::pin(async move {
                let source = source.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                let mesh = ng_datasource::mesh::load(&source, segment).await?;
                let bytes = mesh.bytes();
                Ok((mesh, bytes))
            })
        });
        let skeleton_sources: HashMap<LayerId, Vec<ng_datasource::skeleton::SkeletonSource>> = self.layers.iter().map(|l| (l.id, l.skeleton_sources())).collect();
        self.skeletons.end_frame(|(id, k, segment)| {
            let source = skeleton_sources.get(id).and_then(|v| v.get(*k)).cloned();
            let segment = *segment;
            Box::pin(async move {
                let source = source.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                let skeleton = ng_datasource::skeleton::load(&source, segment).await?.unwrap_or_default();
                let bytes = skeleton.bytes();
                Ok((skeleton, bytes))
            })
        });
        let annotation_infos: HashMap<LayerId, Arc<ng_datasource::annotation::AnnotationInfo>> = self.layers.iter().filter_map(|l| Some((l.id, l.annotation.as_ref()?.spatial.clone()?))).collect();
        self.annotation_cells.end_frame(|(id, level, cell)| {
            let info = annotation_infos.get(id).cloned();
            let (level, cell) = (*level, cell.clone());
            Box::pin(async move {
                let info = info.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                // Finest first here; the info lists the coarsest first.
                let decoded = ng_datasource::annotation::load_cell(&info, info.spatial.len() - 1 - level, &cell).await?;
                let list: Vec<annotations::Annotation> = decoded.iter().map(|a| annotations::from_precomputed(info.annotation_type, a)).collect();
                let bytes = list.len() * (64 + 8 * info.rank * 2);
                Ok((list, bytes))
            })
        });
        // Styles not drawn this frame (nor being computed) are dropped.
        let frame = self.frame;
        self.annotation_styles.retain(|_, slot| slot.frame == frame || matches!(slot.state, StyleState::Pending(_)));
        let by_id_infos = annotation_infos.clone();
        self.annotations_by_id.end_frame(|(id, annotation)| {
            let info = by_id_infos.get(id).cloned();
            let annotation = *annotation;
            Box::pin(async move {
                let info = info.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                let a = ng_datasource::annotation::load_by_id(&info, annotation).await?.map(|a| annotations::from_precomputed(info.annotation_type, &a));
                Ok((a, 64))
            })
        });
        let meshes = &self.meshes;
        self.gpu_meshes.retain(|k, (data, _)| meshes.get(k).is_some_and(|d| Arc::ptr_eq(&d, data)));
        let manifest_sources = mesh_sources;
        self.manifests.end_frame(|(id, k, segment)| {
            let source = manifest_sources.get(id).and_then(|v| v.get(*k)).cloned();
            let segment = *segment;
            Box::pin(async move {
                let source = source.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                let multiplier = source.multiscale.as_ref().map_or(1.0, |m| m.lod_scale_multiplier);
                // `<id>.index`, or the segment's entry in the shards.
                let (bytes, shard) = match source.multiscale.as_ref().and_then(|m| m.sharding) {
                    Some(sharding) => {
                        let (bytes, info) = ng_datasource::sharded::read_with_shard_info(&source.location, &sharding, segment as u128).await?.ok_or_else(|| anyhow::anyhow!("no manifest for segment {segment}"))?;
                        (bytes, Some(info))
                    }
                    None => (source.location.read(&format!("{segment}.index"), ng_kvstore::ByteRange::Full).await?.ok_or_else(|| anyhow::anyhow!("no manifest for segment {segment}"))?, None),
                };
                let mut manifest = ng_datasource::multiscale_mesh::parse_manifest(&bytes, multiplier)?;
                manifest.shard = shard;
                let size = manifest.octree.len() * 4 + manifest.offsets.len() * 8;
                Ok((manifest, size))
            })
        });
        let reads = std::mem::take(&mut self.fragment_reads);
        self.fragments.end_frame(|key| {
            let read = reads.get(key).cloned();
            Box::pin(async move {
                let (location, path, offset, length, bits, partition) = read.ok_or_else(|| anyhow::anyhow!("the fragment is not wanted"))?;
                let bytes = location.read(&path, ng_kvstore::ByteRange::Range { offset, length }).await?.ok_or_else(|| anyhow::anyhow!("no fragment data in {path}"))?;
                // `strictByteRange`.
                anyhow::ensure!(bytes.len() as u64 == length, "Received truncated response for {path}");
                let fragment = ng_datasource::multiscale_mesh::decode_fragment(&bytes, bits, partition)?;
                let size = fragment.bytes();
                Ok((fragment, size))
            })
        });
        let fragments = &self.fragments;
        self.gpu_fragments.retain(|k, (data, _)| fragments.get(k).is_some_and(|d| Arc::ptr_eq(&d, data)));
        // Segmentation scales neuroglancer transcodes, and into what blocks.
        let info = self.nav.render_info().clone();
        let mut blocks: HashMap<(LayerId, usize), [usize; 3]> = HashMap::new();
        for l in &self.layers {
            let Some(render) = l.render.as_ref() else { continue };
            for s in 0..render.layer.scales.len() {
                if let Some(b) = chunks::transcode_block(&render.layer, s, &info) {
                    blocks.insert((l.id, s), b);
                }
            }
        }
        self.chunks.end_frame(|(id, s, grid)| {
            let source = scales.get(id).and_then(|v| v.get(*s)).cloned();
            let block = blocks.get(&(*id, *s)).copied();
            let grid = grid.clone();
            Box::pin(async move {
                let source = source.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                let data = read_chunk(&source, &grid).await?;
                // As neuroglancer counts it (the statistics' memory): a
                // chunk it transcodes at its encoded size.
                let bytes = match block {
                    Some(block) if data.memory_bytes > 0 => {
                        let z = &data.size;
                        let shape = [z[0] as usize, z[1] as usize, z[2] as usize, z.get(3).copied().unwrap_or(1) as usize];
                        ng_datasource::chunk::compressed_segmentation_encoded_bytes(&data.data, shape, block, source.spec.data_type == DataType::Uint64)
                    }
                    _ => data.memory_bytes,
                };
                Ok((data, bytes))
            })
        });
        let chunks = &self.chunks;
        for layer in &mut self.layers {
            let id = layer.id;
            if let Some(state) = &mut layer.render {
                state
                    .textures
                    .retain(|(s, grid, _), t| t.last_used + 60 > frame && chunks.get(&(id, *s, grid.clone())).is_some());
            }
        }
    }
}

/// A global position in a layer's model coordinates (float32, as
/// neuroglancer keeps them); its unmapped layer dimensions at 0.
fn global_to_model(l: &VolumeLayer, position: &[f32]) -> Option<Vec<f64>> {
    let n = l.rank + 1;
    let mut layer_pos = vec![0.0; n];
    layer_pos[l.rank] = 1.0;
    for (d, g) in l.layer_to_global.iter().enumerate() {
        layer_pos[d] = match g {
            Some(g) => *position.get(*g)? as f64,
            None => 0.0,
        };
    }
    let (inv, _) = ng_core::matrix::inverse(&l.model_to_layer, n);
    Some((0..l.source_rank).map(|c| ((0..n).map(|k| inv[k * n + c] * layer_pos[k]).sum::<f64>()) as f32 as f64).collect())
}

/// Which view a spatial index is drawn in, for its histogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnnotationView {
    CrossSection,
    Projection,
}

/// `SpatiallyIndexedAnnotationLayer.draw` for layer `li`: the annotations
/// of its precomputed source's spatial index cells within the view of
/// `projection · view` (`size` pixels), each cell's first
/// `round(count · drawFraction)` (the writer shuffles them), requesting the
/// cells. `None` for local annotations.
#[allow(clippy::too_many_arguments)]
fn spatial_annotations(
    layers: &mut [ManagedLayer],
    cells: &mut ChunkManager<AnnotationCellKey, Vec<annotations::Annotation>>,
    li: usize,
    view_kind: AnnotationView,
    frame: u64,
    info: &ng_core::navigation::DisplayDimensionRenderInfo,
    position: &[f32],
    projection: &[f64; 16],
    view: &[f64; 16],
    size: (f64, f64),
) -> Option<Vec<annotations::Annotation>> {
    let managed = layers.get_mut(li)?;
    let id = managed.id;
    let ann = managed.annotation.as_ref()?;
    let spatial = ann.spatial.as_ref()?;
    let key = match view_kind {
        AnnotationView::CrossSection => "crossSectionAnnotationSpacing",
        AnnotationView::Projection => "projectionAnnotationSpacing",
    };
    let target = ng_core::render_scale::parse_target(managed.spec.json.get(key), 8.0);
    let layouts = annotation_level_layouts(&ann.layer, spatial, info, position);
    let canonical_to_physical: f64 = info.voxel_physical_scales.iter().product();
    let visible = chunks::visible_annotation_cells(&layouts, projection, view, size, canonical_to_physical, target);
    let histogram = &mut managed.annotation_histograms[view_kind as usize];
    histogram.begin(frame);
    let mut out = vec![];
    for c in visible {
        let key = (id, c.level, c.cell.clone());
        cells.request(&key, -(c.level as f64));
        managed.progress.0 += 1;
        let present = match cells.get(&key) {
            Some(list) => {
                let count = (list.len() as f64 * c.draw_fraction).round() as usize;
                out.extend(list.iter().take(count).cloned());
                managed.progress.1 += 1;
                1
            }
            None => 0,
        };
        let histogram = &mut managed.annotation_histograms[view_kind as usize];
        histogram.add(c.physical_spacing, c.pixel_spacing, present, 1 - present);
    }
    Some(out)
}

/// A precomputed annotation source's spatial index levels, finest first,
/// as chunk layouts in the display (`getVolumetricTransformedSources` with
/// each level's `chunkToMultiscaleTransform`, a translation to the lower
/// bound).
fn annotation_level_layouts(layer: &VolumeLayer, spatial: &ng_datasource::annotation::AnnotationInfo, info: &ng_core::navigation::DisplayDimensionRenderInfo, position: &[f32]) -> Vec<chunks::AnnotationLevelLayout> {
    let rank = spatial.rank;
    let origin = model_to_display(layer, info, &spatial.lower_bound);
    let columns: Vec<[f64; 3]> = (0..rank)
        .map(|c| {
            let mut e = vec![0.0; rank];
            e[c] = 1.0;
            linear_to_display(layer, info, &e)
        })
        .collect();
    // Each display dimension's model dimension.
    let mut dims = [None; 3];
    for i in 0..3 {
        dims[i] = (0..rank).find(|&c| columns[c][i] != 0.0 && !dims.contains(&Some(c)));
    }
    let model_position = global_to_model(layer, position);
    spatial
        .spatial
        .iter()
        .rev()
        .map(|level| {
            let mut to_display = [0.0; 16];
            to_display[15] = 1.0;
            to_display[12..15].copy_from_slice(&origin);
            let mut chunk_size = [1.0; 3];
            let mut grid = [1.0; 3];
            for (i, d) in dims.iter().enumerate() {
                match *d {
                    Some(c) => {
                        to_display[4 * i..4 * i + 3].copy_from_slice(&columns[c]);
                        chunk_size[i] = level.chunk_size[c];
                        grid[i] = level.grid_shape[c] as f64;
                    }
                    None => to_display[5 * i] = 1.0,
                }
            }
            // The other dimensions: the cell at the position, and the
            // fraction of the level one slice through them holds.
            let mut base = Some(vec![0u64; rank]);
            let mut slice_fraction = 1.0;
            for c in (0..rank).filter(|c| !dims.contains(&Some(*c))) {
                slice_fraction /= level.grid_shape[c] as f64 * level.chunk_size[c];
                let cell = model_position.as_ref().map_or(-1.0, |p| ((p[c] - spatial.lower_bound[c]) / level.chunk_size[c]).floor());
                match base.as_mut() {
                    Some(b) if cell >= 0.0 && cell < level.grid_shape[c] as f64 => b[c] = cell as u64,
                    _ => base = None,
                }
            }
            chunks::AnnotationLevelLayout { to_display, dims, chunk_size, grid, base, limit: level.limit as f64, slice_fraction }
        })
        .collect()
}

/// Model coordinates of `layer` to the display dimensions' (global)
/// coordinates.
fn model_to_display(layer: &VolumeLayer, info: &ng_core::navigation::DisplayDimensionRenderInfo, model: &[f64]) -> [f64; 3] {
    let n = layer.rank + 1;
    let m2l = &layer.model_to_layer;
    let mut out = [0.0; 3];
    for (i, slot) in out.iter_mut().enumerate().take(info.display_rank) {
        let g = info.display_dimension_indices[i];
        if g < 0 {
            continue;
        }
        if let Some(ld) = layer.layer_to_global.iter().position(|&x| x == Some(g as usize)) {
            *slot = (0..layer.source_rank.min(model.len())).map(|c| m2l[c * n + ld] * model[c]).sum::<f64>() + m2l[layer.rank * n + ld];
        }
    }
    out
}

/// A model-space vector (no translation) in the display dimensions.
fn linear_to_display(layer: &VolumeLayer, info: &ng_core::navigation::DisplayDimensionRenderInfo, v: &[f64]) -> [f64; 3] {
    let a = model_to_display(layer, info, v);
    let o = model_to_display(layer, info, &vec![0.0; v.len()]);
    [a[0] - o[0], a[1] - o[1], a[2] - o[2]]
}

/// An axis-aligned ellipsoid's cross-section with the panel's plane
/// (`computeCrossSectionEllipse`, `computeCenterOrientEllipse`): the
/// corners of its bounding quad, in display coordinates, for offsets
/// (−1, −1), (1, −1), (1, 1), (−1, 1) along its axes; `None` if the plane
/// misses it.
fn ellipsoid_cross_section(camera: &chunks::PanelCamera, center: [f64; 3], radii: [f64; 3]) -> Option<[[f64; 3]; 4]> {
    let inv = &camera.inv_view;
    let e1 = [inv[0], inv[1], inv[2]];
    let e2 = [inv[4], inv[5], inv[6]];
    let c0 = [inv[12], inv[13], inv[14]];
    let r: [f64; 3] = std::array::from_fn(|i| radii[i].abs().max(1e-3));
    let d: [f64; 3] = std::array::from_fn(|i| c0[i] - center[i]);
    let sum = |f: &dyn Fn(usize) -> f64| (0..3).map(|i| f(i) / (r[i] * r[i])).sum::<f64>();
    let (a11, a12, a22) = (sum(&|i| e1[i] * e1[i]), sum(&|i| e1[i] * e2[i]), sum(&|i| e2[i] * e2[i]));
    let (b1, b2, c) = (sum(&|i| d[i] * e1[i]), sum(&|i| d[i] * e2[i]), sum(&|i| d[i] * d[i]));
    let det = a11 * a22 - a12 * a12;
    if det <= 0.0 {
        return None;
    }
    let k = [-(a22 * b1 - a12 * b2) / det, -(-a12 * b1 + a11 * b2) / det];
    let rhs = 1.0 - (c + b1 * k[0] + b2 * k[1]);
    if rhs <= 0.0 {
        return None;
    }
    let mean = (a11 + a22) / 2.0;
    let root = (((a11 - a22) / 2.0).powi(2) + a12 * a12).sqrt();
    let (l1, l2) = (mean - root, mean + root);
    let u1 = if a12.abs() > 1e-12 { let v = [l1 - a22, a12]; let n = (v[0] * v[0] + v[1] * v[1]).sqrt(); [v[0] / n, v[1] / n] } else if a11 <= a22 { [1.0, 0.0] } else { [0.0, 1.0] };
    let u2 = [-u1[1], u1[0]];
    let (a, b) = ((rhs / l1).sqrt(), (rhs / l2).sqrt());
    let corner = |cx: f64, cy: f64| -> [f64; 3] {
        let u = k[0] + u1[0] * cx * a + u2[0] * cy * b;
        let v = k[1] + u1[1] * cx * a + u2[1] * cy * b;
        std::array::from_fn(|i| c0[i] + u * e1[i] + v * e2[i])
    };
    Some([corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)])
}

/// `getSphereVertexArray` and `getSphereIndexArray`: the unit sphere's
/// triangles, vertex by vertex.
fn sphere_triangles(lat: usize, lon: usize) -> Vec<[f64; 3]> {
    let vertex = |li: usize, oi: usize| {
        let theta = li as f64 * std::f64::consts::PI / lat as f64;
        let phi = oi as f64 * 2.0 * std::f64::consts::PI / lon as f64;
        // Stored as float32.
        [(phi.cos() * theta.sin()) as f32 as f64, theta.cos() as f32 as f64, (phi.sin() * theta.sin()) as f32 as f64]
    };
    let mut out = vec![];
    for li in 0..lat {
        for oi in 0..lon {
            let (first, second) = ((li, oi), (li + 1, oi));
            for (a, b) in [first, second, (li, oi + 1), second, (li + 1, oi + 1), (li, oi + 1)] {
                out.push(vertex(a, b));
            }
        }
    }
    out
}

/// A chunk volume rendering drew: data, chunk display dimensions, fixed
/// offsets.
type VolumeChunk = (Arc<ChunkData>, [Option<usize>; 3], Vec<(usize, u64, u32)>);

/// `getObjectColor`: segment `id`'s colour in `display`, highlighted if
/// `hovered`, premultiplied by `alpha`.
fn object_color(display: &ng_render::segmentation::SegmentationDisplay, hovered: Option<u64>, id: u64, alpha: f32) -> [f32; 4] {
    let base = display.default_color.unwrap_or_else(|| ng_core::SegmentColorHash::new(display.color_seed).compute(id));
    let mut saturation = display.saturation;
    if display.hover_highlight && hovered == Some(id) {
        saturation += if saturation > 0.5 { -0.5 } else { 0.5 };
    }
    let c = |x: f32| (x * saturation + (1.0 - saturation)) * alpha;
    [c(base[0]), c(base[1]), c(base[2]), alpha]
}

/// `SkeletonRenderingOptions`' view-specific options (`mode2d` and
/// `lineWidth2d`, or the 3D ones): whether nodes are drawn as points, and
/// the line width in pixels. Invalid values are the defaults, as
/// `TrackableValue.restoreState` resets them.
fn skeleton_options(json: &serde_json::Map<String, Value>, view_2d: bool) -> (bool, f64) {
    let o = json.get("skeletonRendering").and_then(Value::as_object);
    let (mode_key, width_key, default_points, default_width) = if view_2d { ("mode2d", "lineWidth2d", true, 2.0) } else { ("mode3d", "lineWidth3d", false, 1.0) };
    // `verifyEnumString`: case-insensitive.
    let points = match o.and_then(|o| o.get(mode_key)).and_then(Value::as_str).map(str::to_ascii_uppercase).as_deref() {
        Some("LINES") => false,
        Some("LINES_AND_POINTS") => true,
        _ => default_points,
    };
    let width = o.and_then(|o| o.get(width_key)).and_then(Value::as_f64).filter(|w| w.is_finite() && *w > 0.0).unwrap_or(default_width);
    (points, width)
}

/// A segment's skeleton to draw: its model matrix to the display
/// dimensions, colour, data, and the view's options.
struct SkeletonDraw {
    /// The layer and segment (for picking).
    segment: (LayerId, u64),
    model: [f32; 16],
    color: [f32; 4],
    data: Arc<ng_datasource::skeleton::SkeletonData>,
    points: bool,
    width: f64,
}

impl SkeletonDraw {
    /// `drawSkeleton`: each edge a line, each vertex a circle (`max(5, 2 ·
    /// width)` across with points, else `width`), without border; feathered
    /// 1 px in cross-sections, not in 3D. `to_clip` maps display coordinates
    /// to GL clip coordinates.
    fn marks(&self, to_clip: &dyn Fn([f64; 3]) -> [f64; 4], feather: f64, lines: &mut Vec<ng_render::annotations::Line>, circles: &mut Vec<ng_render::annotations::Circle>) {
        let m = &self.model;
        let clips: Vec<[f64; 4]> = self
            .data
            .positions
            .chunks_exact(3)
            .map(|p| {
                let d = ng_core::glm::vec4_transform_mat4([p[0] as f64, p[1] as f64, p[2] as f64, 1.0], m);
                to_clip([d[0], d[1], d[2]])
            })
            .collect();
        for e in self.data.indices.chunks_exact(2) {
            let (Some(&a), Some(&b)) = (clips.get(e[0] as usize), clips.get(e[1] as usize)) else { continue };
            lines.push(ng_render::annotations::Line { a, b, width: self.width, feather, color: self.color, color_b: self.color, pick: 0 });
        }
        let diameter = if self.points { (self.width * 2.0).max(5.0) } else { self.width };
        for &center in &clips {
            circles.push(ng_render::annotations::Circle { center, diameter, border_width: 0.0, feather, interior: self.color, border: self.color, pick: 0 });
        }
    }
}

/// `pickRadius`'s default.
const PICK_RADIUS: u32 = 5;

/// A pick window being read: the frame's inverse view-projection, position,
/// display dimensions and pick IDs, and whether its buffer is mapping.
struct PendingPick {
    readback: ng_render::pick::PickReadback,
    inv_vp: [f32; 16],
    position: Vec<f32>,
    display: [i32; 3],
    ids: Vec<PickTarget>,
    mapped: bool,
}

/// What a pick ID stands for (`PickIDManager`'s registrations): a
/// layer's segment (a mesh, a skeleton) or annotation.
#[derive(Debug, Clone, PartialEq)]
enum PickTarget {
    Segment(LayerId, u64),
    Annotation(LayerId, String, AnnotationPart),
}

/// The part of an annotation picked (its `partIndex`), for snapping.
#[derive(Debug, Clone, Copy, PartialEq)]
enum AnnotationPart {
    Body,
    /// A line's endpoint marker (`ENDPOINTS_PICK_OFFSET + k`).
    Endpoint(usize),
    /// A box's corner (3D): which (`CORNERS_PICK_OFFSET + k`).
    Corner(usize),
}

impl AnnotationPart {
    /// `annotationPart` (`pickedOffset`) as neuroglancer numbers it: 0 for
    /// the object, then the endpoints or corners.
    fn index(self) -> u32 {
        match self {
            AnnotationPart::Body => 0,
            AnnotationPart::Endpoint(k) | AnnotationPart::Corner(k) => 1 + k as u32,
        }
    }
}

/// A resolved 3D pick: the global position under the pointer, and what
/// was drawn there.
#[derive(Debug, Clone, PartialEq)]
struct PerspectivePick {
    position: Vec<f32>,
    picked: Option<PickTarget>,
}

/// A pickable mark in a slice panel, in its pixels (from the top left),
/// as its quad covers them.
#[derive(Debug, Clone, PartialEq)]
enum PickShapeKind {
    /// A line's quad: from `a` to `b`, `half` pixels to each side.
    Segment { a: [f64; 2], b: [f64; 2], half: f64 },
    Disc { center: [f64; 2], radius: f64 },
    /// `center + s·u + t·v` for `s² + t² ≤ 1`.
    Ellipse { center: [f64; 2], u: [f64; 2], v: [f64; 2] },
}

#[derive(Debug, Clone, PartialEq)]
struct PickShape {
    kind: PickShapeKind,
    pick: u32,
}

/// GL clip coordinates to pixels from the top left of a `w` × `h` panel;
/// `None` beyond the depth range.
fn clip_to_pixels(c: [f64; 4], w: f64, h: f64) -> Option<[f64; 2]> {
    (c[3] > 0.0 && (c[2] / c[3]).abs() <= 1.0).then(|| [(c[0] / c[3] + 1.0) / 2.0 * w, (1.0 - c[1] / c[3]) / 2.0 * h])
}

impl PickShape {
    fn segment(a: [f64; 2], b: [f64; 2], width: f64, feather: f64, pick: u32) -> Self {
        let half = if width == 0.0 { 0.0 } else { width / 2.0 + feather };
        PickShape { kind: PickShapeKind::Segment { a, b, half }, pick }
    }

    fn from_line(l: &ng_render::annotations::Line, w: f64, h: f64) -> Option<Self> {
        let (a, b) = ng_render::annotations::clip_to_depth_range(l.a, l.b)?;
        let px = |c: [f64; 4]| [(c[0] / c[3] + 1.0) / 2.0 * w, (1.0 - c[1] / c[3]) / 2.0 * h];
        (l.pick != 0).then(|| Self::segment(px(a), px(b), l.width, l.feather, l.pick))
    }

    fn from_circle(c: &ng_render::annotations::Circle, w: f64, h: f64) -> Option<Self> {
        let center = clip_to_pixels(c.center, w, h)?;
        let total = if c.diameter == 0.0 { 0.0 } else { c.diameter + 2.0 * (c.border_width + c.feather) };
        (c.pick != 0).then_some(PickShape { kind: PickShapeKind::Disc { center, radius: total / 2.0 }, pick: c.pick })
    }

    fn from_ellipse(e: &ng_render::annotations::Ellipse, w: f64, h: f64) -> Option<Self> {
        let p: Vec<[f64; 2]> = e.corners.iter().map(|&c| clip_to_pixels(c, w, h)).collect::<Option<_>>()?;
        let center = [(p[0][0] + p[2][0]) / 2.0, (p[0][1] + p[2][1]) / 2.0];
        let u = [(p[1][0] - p[0][0]) / 2.0, (p[1][1] - p[0][1]) / 2.0];
        let v = [(p[3][0] - p[0][0]) / 2.0, (p[3][1] - p[0][1]) / 2.0];
        (e.pick != 0).then_some(PickShape { kind: PickShapeKind::Ellipse { center, u, v }, pick: e.pick })
    }

    /// Whether the quad covers the pixel centred at `p`.
    fn covers(&self, p: [f64; 2]) -> bool {
        match self.kind {
            PickShapeKind::Segment { a, b, half } => {
                let d = [b[0] - a[0], b[1] - a[1]];
                let len2 = d[0] * d[0] + d[1] * d[1];
                if len2 < 1e-12 {
                    return (p[0] - a[0]).abs() <= half && (p[1] - a[1]).abs() <= half;
                }
                let r = [p[0] - a[0], p[1] - a[1]];
                let t = (r[0] * d[0] + r[1] * d[1]) / len2;
                let perp = (r[0] * d[1] - r[1] * d[0]).abs() / len2.sqrt();
                (0.0..=1.0).contains(&t) && perp <= half
            }
            PickShapeKind::Disc { center, radius } => (p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2) <= radius * radius,
            PickShapeKind::Ellipse { center, u, v } => {
                let det = u[0] * v[1] - u[1] * v[0];
                if det.abs() < 1e-12 {
                    return false;
                }
                let r = [p[0] - center[0], p[1] - center[1]];
                let s = (r[0] * v[1] - r[1] * v[0]) / det;
                let t = (u[0] * r[1] - u[1] * r[0]) / det;
                s * s + t * t <= 1.0
            }
        }
    }
}

/// A slice panel's pickable marks (`pickIDs` and what the pick buffer
/// would hold): its size, the targets and the shapes in drawing order.
#[derive(Debug, Clone, Default)]
struct SlicePicks {
    size: (u32, u32),
    targets: Vec<PickTarget>,
    shapes: Vec<PickShape>,
}

impl SlicePicks {
    /// `completePickRequest` in a slice panel: from the pixel under `(x,
    /// y)` (physical, from the panel's top left) outwards within the pick
    /// radius, the first covered by a mark — the last one drawn there.
    fn pick(&self, x: f64, y: f64) -> Option<PickTarget> {
        if self.shapes.is_empty() {
            return None;
        }
        let (w, h) = (self.size.0 as i64, self.size.1 as i64);
        let gx = x.floor() as i64;
        let gy = (h as f64 - y).floor() as i64;
        let r = PICK_RADIUS as i64;
        let d = ng_render::pick::pick_diameter(PICK_RADIUS) as i64;
        for o in ng_render::pick::pick_offset_sequence(PICK_RADIUS) {
            let (px, py) = (gx + o as i64 % d - r, gy + o as i64 / d - r);
            if px < 0 || py < 0 || px >= w || py >= h {
                continue;
            }
            let centre = [px as f64 + 0.5, h as f64 - (py as f64 + 0.5)];
            if let Some(s) = self.shapes.iter().rev().find(|s| s.covers(centre)) {
                return self.targets.get(s.pick as usize - 1).cloned();
            }
        }
        None
    }
}

/// Annotations as marks in clip coordinates, a batch per layer and
/// annotation type in drawing order (`annotationTypes`: points, lines,
/// boxes, ellipsoids, polylines).
#[derive(Default)]
struct AnnotationMarks {
    batches: Vec<ng_render::annotations::MarkBatch>,
}

impl AnnotationMarks {
    /// The marks of `annotations` by type (`annotationTypes`), each in its
    /// style (the shader's): points as circles with their borders, lines
    /// with their endpoint markers, polylines' segments and markers,
    /// ellipsoids' cross-sections (`ellipses`, by annotation); boxes go to
    /// `boxes` (model-space corners, border colour and width). The hovered
    /// annotation's colours three quarters of the way to white
    /// (`setPartIndex`); discarded ones are left out.
    #[allow(clippy::too_many_arguments)]
    fn add(&mut self, annotations: &[annotations::Annotation], ellipses: &[Option<ng_render::annotations::Ellipse>], styles: &[ng_render::annotation_shader::AnnotationStyle], hovered: Option<&str>, register: &mut dyn FnMut(&str, AnnotationPart) -> u32, to_clip: &dyn Fn(&[f64]) -> [f64; 4], boxes: &mut BoxSink) {
        use annotations::Geometry as G;
        type Kind = fn(&G) -> bool;
        let kinds: [Kind; 5] = [|g| matches!(g, G::Point(_)), |g| matches!(g, G::Line(..)), |g| matches!(g, G::AxisAlignedBoundingBox(..)), |g| matches!(g, G::Ellipsoid { .. }), |g| matches!(g, G::Polyline(_))];
        for kind in kinds {
            let mut batch = ng_render::annotations::MarkBatch::default();
            for (k, a) in annotations.iter().enumerate().filter(|(_, a)| kind(&a.geometry)) {
                let Some(s) = styles.get(k).filter(|s| !s.discard) else { continue };
                let lit = |c: [f32; 4]| highlight(c, hovered == Some(a.id.as_str()));
                let pick = register(&a.id, AnnotationPart::Body);
                let marker = |center: [f64; 4], end: usize, pick: u32| ng_render::annotations::Circle {
                    center,
                    diameter: s.marker_size[end] as f64,
                    border_width: s.marker_border_width[end] as f64,
                    feather: 0.5,
                    interior: lit(s.marker_color[end]),
                    border: s.marker_border_color[end],
                    pick,
                };
                match &a.geometry {
                    G::Point(p) => batch.circles.push(ng_render::annotations::Circle {
                        center: to_clip(p),
                        diameter: s.point_size as f64,
                        border_width: s.point_border_width as f64,
                        feather: 1.0,
                        interior: lit(s.point_color),
                        border: s.point_border_color,
                        pick,
                    }),
                    G::Line(pa, pb) => {
                        let (ca, cb) = (to_clip(pa), to_clip(pb));
                        batch.lines.push(ng_render::annotations::Line { a: ca, b: cb, width: s.line_width as f64, feather: 1.0, color: lit(s.line_color[0]), color_b: lit(s.line_color[1]), pick });
                        batch.circles.push(marker(ca, 0, register(&a.id, AnnotationPart::Endpoint(0))));
                        batch.circles.push(marker(cb, 1, register(&a.id, AnnotationPart::Endpoint(1))));
                    }
                    G::Polyline(points) => {
                        // Each segment a line from the start colour to the
                        // end one; a marker at each segment's start (the
                        // start's style) and at the last one's end.
                        let clips: Vec<[f64; 4]> = points.iter().map(|p| to_clip(p)).collect();
                        for w in clips.windows(2) {
                            batch.lines.push(ng_render::annotations::Line { a: w[0], b: w[1], width: s.poly_line_width as f64, feather: 1.0, color: lit(s.poly_line_color[0]), color_b: lit(s.poly_line_color[1]), pick });
                        }
                        if clips.len() >= 2 {
                            // `snapPosition`'s endpoints: each marker the start of
                            // its segment (the first point), the last the end.
                            let (start, end) = (register(&a.id, AnnotationPart::Endpoint(0)), register(&a.id, AnnotationPart::Endpoint(1)));
                            for (k, &c) in clips.iter().enumerate() {
                                let e = usize::from(k + 1 == clips.len());
                                batch.circles.push(ng_render::annotations::Circle {
                                    center: c,
                                    diameter: s.poly_marker_size[e] as f64,
                                    border_width: s.poly_marker_border_width[e] as f64,
                                    feather: 0.5,
                                    interior: lit(s.poly_marker_color[e]),
                                    border: s.poly_marker_border_color[e],
                                    pick: if e == 1 { end } else { start },
                                });
                            }
                        }
                    }
                    G::AxisAlignedBoundingBox(pa, pb) => {
                        let lower: Vec<f64> = pa.iter().zip(pb).map(|(a, b)| a.min(*b)).collect();
                        let upper: Vec<f64> = pa.iter().zip(pb).map(|(a, b)| a.max(*b)).collect();
                        boxes(&lower, &upper, lit(s.box_border_color), s.box_border_width as f64, pick);
                    }
                    G::Ellipsoid { .. } => {
                        if let Some(Some(e)) = ellipses.get(k) {
                            batch.ellipses.push(ng_render::annotations::Ellipse { pick, ..*e });
                        }
                    }
                }
            }
            self.batches.push(batch);
        }
    }
}

/// `setPartIndex`'s highlight: the hovered annotation's colour three
/// quarters of the way to white, its alpha kept.
fn highlight(c: [f32; 4], hovered: bool) -> [f32; 4] {
    if hovered {
        [c[0] + (1.0 - c[0]) * 0.75, c[1] + (1.0 - c[1]) * 0.75, c[2] + (1.0 - c[2]) * 0.75, c[3]]
    } else {
        c
    }
}

/// Where boxes go: their lower and upper corners (model space), border
/// colour and width, and pick ID.
type BoxSink<'a> = dyn FnMut(&[f64], &[f64], [f32; 4], f64, u32) + 'a;

/// `getRepresentativePoint`: what a drag moves (a line's or box's first
/// point unless an endpoint is picked; a polyline's first point or the
/// endpoint; an ellipsoid's centre).
fn representative_point(g: &annotations::Geometry, part: AnnotationPart) -> Vec<f64> {
    use annotations::Geometry as G;
    match (g, part) {
        (G::Point(p), _) => p.clone(),
        (G::Line(_, pb), AnnotationPart::Endpoint(1)) => pb.clone(),
        (G::Line(pa, _), _) | (G::AxisAlignedBoundingBox(pa, _), _) => pa.clone(),
        (G::Ellipsoid { center, .. }, _) => center.clone(),
        (G::Polyline(points), AnnotationPart::Endpoint(k)) => points.get(k).or(points.first()).cloned().unwrap_or_default(),
        (G::Polyline(points), _) => points.first().cloned().unwrap_or_default(),
    }
}

/// `updateViaRepresentativePoint`: the annotation with its representative
/// point at `p` (float32): the whole moved, or just the endpoint picked.
fn update_via_representative_point(g: &annotations::Geometry, p: &[f64], part: AnnotationPart) -> annotations::Geometry {
    use annotations::Geometry as G;
    let p: Vec<f64> = p.iter().map(|x| *x as f32 as f64).collect();
    // `old + (p - from)`, as float32 arrays hold it.
    let shift = |old: &[f64], from: &[f64]| -> Vec<f64> { old.iter().zip(from).zip(&p).map(|((o, f), x)| (o + (x - f)) as f32 as f64).collect() };
    match (g, part) {
        (G::Point(_), _) => G::Point(p),
        (G::Line(pa, _), AnnotationPart::Endpoint(0)) => G::Line(p, g_line_b(g).unwrap_or_else(|| pa.clone())),
        (G::Line(pa, _), AnnotationPart::Endpoint(1)) => G::Line(pa.clone(), p),
        (G::Line(pa, pb), _) => G::Line(p.clone(), shift(pb, pa)),
        (G::AxisAlignedBoundingBox(pa, pb), _) => G::AxisAlignedBoundingBox(p.clone(), shift(pb, pa)),
        (G::Ellipsoid { radii, .. }, _) => G::Ellipsoid { center: p, radii: radii.clone() },
        (G::Polyline(points), AnnotationPart::Endpoint(k)) if k < points.len() => {
            let mut points = points.clone();
            points[k] = p;
            G::Polyline(points)
        }
        (G::Polyline(points), _) => {
            let from = points.first().cloned().unwrap_or_default();
            G::Polyline(points.iter().map(|q| shift(q, &from)).collect())
        }
    }
}

fn g_line_b(g: &annotations::Geometry) -> Option<Vec<f64>> {
    match g {
        annotations::Geometry::Line(_, pb) => Some(pb.clone()),
        _ => None,
    }
}

/// An annotation's JSON with `g`'s points (`annotationToJson`).
fn write_geometry(v: &mut Value, g: &annotations::Geometry) {
    use annotations::Geometry as G;
    let nums = |p: &[f64]| Value::Array(p.iter().map(|x| ng_core::js::number(*x)).collect());
    let Some(o) = v.as_object_mut() else { return };
    match g {
        G::Point(p) => {
            o.insert("point".into(), nums(p));
        }
        G::Line(a, b) | G::AxisAlignedBoundingBox(a, b) => {
            o.insert("pointA".into(), nums(a));
            o.insert("pointB".into(), nums(b));
        }
        G::Ellipsoid { center, radii } => {
            o.insert("center".into(), nums(center));
            o.insert("radii".into(), nums(radii));
        }
        G::Polyline(points) => {
            o.insert("points".into(), Value::Array(points.iter().map(|p| nums(p)).collect()));
        }
    }
}

/// `LocalDataSourceProvider`'s transform without a state transform: the
/// identity from dimensions named `0`, `1`, … to the global space's.
fn identity_over(global: &ng_core::CoordinateSpace) -> ng_core::transform::CoordinateSpaceTransform {
    let rank = global.rank();
    let input = ng_core::CoordinateSpace::new(true, (0..rank).map(|i| i.to_string()).collect(), global.units.clone(), global.scales.clone());
    ng_core::transform::CoordinateSpaceTransform {
        rank,
        source_rank: rank,
        input_space: Arc::new(input),
        // `makeCoordinateSpace` without bounds: unbounded.
        output_space: Arc::new(ng_core::CoordinateSpace::new(true, global.names.clone(), global.units.clone(), global.scales.clone())),
        transform: ng_core::matrix::identity(rank + 1, rank + 1),
    }
}

/// `update3dRenderLayerAttachment` for a mesh source: its coordinates
/// (`to_model`, column-major 4×4) through the layer's model-to-layer
/// transform to the display dimensions (as float32, gl-matrix's storage).
fn mesh_model_matrix(layer: &layer::VolumeLayer, to_model: &[f64; 16], info: &ng_core::navigation::DisplayDimensionRenderInfo) -> Option<[f32; 16]> {
    let n = layer.rank + 1;
    let m2l = &layer.model_to_layer;
    let model_rank = layer.source_rank.min(3);
    let mut out = [0f64; 16];
    out[15] = 1.0;
    for i in 0..3 {
        let g = info.display_dimension_indices[i];
        if g < 0 {
            continue;
        }
        let Some(ld) = layer.layer_to_global.iter().position(|&x| x == Some(g as usize)) else { continue };
        for c in 0..4 {
            let mut v = if c == 3 { m2l[(n - 1) * n + ld] } else { 0.0 };
            for d in 0..model_rank {
                v += m2l[d * n + ld] * to_model[c * 4 + d];
            }
            out[c * 4 + i] = v;
        }
    }
    out.iter().all(|v| v.is_finite()).then(|| out.map(|v| v as f32))
}

/// `MultiscaleMeshLayer.draw`'s `scaleMultiplier`: the cube root of the
/// model matrix's 3×3 determinant, its rows scaled to physical units
/// (`scaleMat3Output`, float32 as gl-matrix stores it).
fn mesh_scale_multiplier(model: &[f32; 16], scales: [f64; 3]) -> f64 {
    let m = |c: usize, r: usize| ((model[c * 4 + r] as f64) * scales[r]) as f32 as f64;
    let det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) - m(1, 0) * (m(0, 1) * m(2, 2) - m(2, 1) * m(0, 2)) + m(2, 0) * (m(0, 1) * m(1, 2) - m(1, 1) * m(0, 2));
    det.abs().powf(1.0 / 3.0)
}

/// `beginModel`'s normal matrix: the model matrix's 3×3, its rows scaled by
/// the canonical voxel factors (`scaleMat3Output`), inverted, transposed.
fn mesh_normal_matrix(model: &[f32; 16], factors: [f64; 3]) -> [f32; 9] {
    let mut m = [0f64; 9];
    for c in 0..3 {
        for r in 0..3 {
            // `mat3FromMat4`, stored as float32, then scaled (float32 again).
            m[c * 3 + r] = ((model[c * 4 + r] as f64) * factors[r]) as f32 as f64;
        }
    }
    // `mat3.invert`.
    let [a00, a01, a02, a10, a11, a12, a20, a21, a22] = m;
    let b01 = a22 * a11 - a12 * a21;
    let b11 = -a22 * a10 + a12 * a20;
    let b21 = a21 * a10 - a11 * a20;
    let det = a00 * b01 + a01 * b11 + a02 * b21;
    if det == 0.0 {
        return [0.0; 9];
    }
    let d = 1.0 / det;
    let inv = [
        b01 * d,
        (-a22 * a01 + a02 * a21) * d,
        (a12 * a01 - a02 * a11) * d,
        b11 * d,
        (a22 * a00 - a02 * a20) * d,
        (-a12 * a00 + a02 * a10) * d,
        b21 * d,
        (-a21 * a00 + a01 * a20) * d,
        (a11 * a00 - a01 * a10) * d,
    ]
    .map(|v| v as f32);
    // `mat3.transpose`.
    [inv[0], inv[3], inv[6], inv[1], inv[4], inv[7], inv[2], inv[5], inv[8]]
}

#[cfg(not(target_arch = "wasm32"))]
fn default_scale_bar_painter() -> Box<dyn ng_render::scale_bar::ScaleBarPainter> {
    Box::new(ng_render::scale_bar::NativeScaleBarPainter::default())
}

/// In the browser the shell sets a canvas painter; until then, none.
#[cfg(target_arch = "wasm32")]
fn default_scale_bar_painter() -> Box<dyn ng_render::scale_bar::ScaleBarPainter> {
    struct Blank;
    impl ng_render::scale_bar::ScaleBarPainter for Blank {
        fn paint(&self, _: &ng_render::scale_bar::ScaleBar, _: &ng_render::scale_bar::ScaleBarOptions) -> ng_render::scale_bar::ScaleBarImage {
            ng_render::scale_bar::ScaleBarImage { width: 0, height: 0, rgba: vec![] }
        }
    }
    Box::new(Blank)
}

/// Re-maps an NDC → texture matrix for rectangle `from` (panel pixels) to the
/// same mapping expressed for rectangle `to`.
fn subrect_matrix(m: [f32; 16], from: [f64; 4], to: [f64; 4]) -> [f32; 16] {
    // ndc_to: NDC in `to` → NDC in `from`: x_from = a * x_to + b.
    let ax = to[2] / from[2];
    let bx = ((to[0] + to[2] / 2.0) - (from[0] + from[2] / 2.0)) / from[2] * 2.0;
    let ay = to[3] / from[3];
    let by = -((to[1] + to[3] / 2.0) - (from[1] + from[3] / 2.0)) / from[3] * 2.0;
    let mut t = [0.0f64; 16];
    t[0] = ax;
    t[5] = ay;
    t[10] = 1.0;
    t[15] = 1.0;
    t[12] = bx;
    t[13] = by;
    let m64 = m.map(|x| x as f64);
    let r = ng_core::matrix::multiply(&m64, &t, 4);
    std::array::from_fn(|i| r[i] as f32)
}

/// A chunk's displayed slab as a 3D texture: the chunk dimensions shown
/// along display x, y, z, at the fixed offsets of the others, cut at the
/// source's upper clip bound (neuroglancer discards beyond it: Zarr stores
/// edge chunks full-size, padded past the array's end).
#[allow(clippy::too_many_arguments)]
fn upload_slab(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layer: &VolumeLayer,
    chunk: &ChunkData,
    draw: &chunks::ChunkDraw,
    format: VolumeFormat,
    upload: UploadFn,
    stack_channels: bool,
) -> GpuVolume {
    let data_type = layer.data_type;
    let source = &layer.scales[draw.scale];
    let rank = chunk.size.len();
    let clip = |c: usize| -> usize {
        let upper = source.upper_clip_bound.as_ref().map_or(source.spec.upper_voxel_bound[c], |b| b[c]) as f64;
        let origin = draw.grid[c] as f64 * source.spec.chunk_data_size[c] as f64;
        ((upper - origin).ceil().max(0.0) as usize).min(chunk.size[c] as usize)
    };
    let mut strides = vec![1usize; rank];
    for c in 1..rank {
        strides[c] = strides[c - 1] * chunk.size[c - 1] as usize;
    }
    // Channel dimensions: every channel, each its own slab stacked along
    // z (first channel dimension fastest), when the layer reads them.
    let channels: &[layer::ChannelDim] = if stack_channels { &layer.channels } else { &[] };
    let mut base = 0usize;
    for &(c, _, off) in &draw.fixed {
        if channels.iter().any(|ch| ch.chunk_dim == c) {
            continue;
        }
        base += off as usize * strides[c];
    }
    let dims: [(usize, usize); 3] = draw.chunk_display_dims.map(|cd| match cd {
        Some(c) => (clip(c).max(1), strides[c]),
        None => (1, 0),
    });
    let elem = data_type.bytes();
    let count: usize = channels.iter().map(|c| c.extent as usize).product();
    let mut slab = Vec::with_capacity(dims[0].0 * dims[1].0 * dims[2].0 * elem * count);
    for flat in 0..count {
        // The channel's offset in the chunk, or none if it is not there.
        let mut offset = Some(base);
        let mut rest = flat;
        for ch in channels {
            let coordinate = rest % ch.extent as usize;
            rest /= ch.extent as usize;
            let origin = draw.grid[ch.chunk_dim] as usize * source.spec.chunk_data_size[ch.chunk_dim] as usize;
            offset = offset.and_then(|o| coordinate.checked_sub(origin).filter(|&k| k < chunk.size[ch.chunk_dim] as usize).map(|k| o + k * strides[ch.chunk_dim]));
        }
        for z in 0..dims[2].0 {
            for y in 0..dims[1].0 {
                // A row along the chunk's contiguous axis: copied whole.
                if let (Some(o), 1) = (offset, dims[0].1) {
                    let i = o + y * dims[1].1 + z * dims[2].1;
                    slab.extend_from_slice(&chunk.data[i * elem..(i + dims[0].0) * elem]);
                    continue;
                }
                for x in 0..dims[0].0 {
                    match offset {
                        Some(o) => {
                            let i = o + x * dims[0].1 + y * dims[1].1 + z * dims[2].1;
                            slab.extend_from_slice(&chunk.data[i * elem..(i + 1) * elem]);
                        }
                        None => slab.extend(std::iter::repeat_n(0u8, elem)),
                    }
                }
            }
        }
    }
    let bytes = upload(data_type, &slab);
    let texture = Data3DTexture::new(dims[0].0 as u32, dims[1].0 as u32, (dims[2].0 * count) as u32, format, bytes);
    GpuVolume::from_texture(device, queue, &texture)
}

/// Prefetched chunks' priorities sit this far below every visible chunk's
/// (`ChunkPriorityTier.PREFETCH` after `VISIBLE`).
const PREFETCH_TIER: f64 = -1e15;

/// The offset, in elements, of channel `coords` (one per channel
/// dimension) from channel 0's in a chunk; `None` if the chunk lacks it.
fn channel_offset(layer: &VolumeLayer, chunk: &ChunkData, coords: &[u32]) -> Option<usize> {
    let mut offset = 0;
    for (ch, &c) in layer.channels.iter().zip(coords) {
        if c as usize >= chunk.size.get(ch.chunk_dim).copied()? as usize {
            return None;
        }
        let stride: usize = chunk.size[..ch.chunk_dim].iter().map(|&n| n as usize).product();
        offset += c as usize * stride;
    }
    Some(offset)
}

fn format_value(t: DataType, data: &[u8], i: usize) -> String {
    let b = &data[i * t.bytes()..(i + 1) * t.bytes()];
    match t {
        DataType::Uint8 => b[0].to_string(),
        DataType::Int8 => (b[0] as i8).to_string(),
        DataType::Uint16 => u16::from_le_bytes([b[0], b[1]]).to_string(),
        DataType::Int16 => i16::from_le_bytes([b[0], b[1]]).to_string(),
        DataType::Uint32 => u32::from_le_bytes(b.try_into().unwrap()).to_string(),
        DataType::Int32 => i32::from_le_bytes(b.try_into().unwrap()).to_string(),
        DataType::Uint64 => u64::from_le_bytes(b.try_into().unwrap()).to_string(),
        DataType::Float32 => format!("{}", f32::from_le_bytes(b.try_into().unwrap())),
    }
}

/// The inverse of a column-major 4×4.
fn mat4_inverse_f64(m: &[f64; 16]) -> [f64; 16] {
    let (inv, _) = ng_core::matrix::inverse(m, 4);
    inv.try_into().unwrap_or(*m)
}

/// An annotation list's styles: being computed, or ready.
enum StyleState {
    Pending(ng_render::annotation_shader::StyleReadback),
    Ready(Arc<Vec<ng_render::annotation_shader::AnnotationStyle>>),
}

struct StyleSlot {
    state: StyleState,
    /// The last frame they were asked for in.
    frame: u64,
}

/// The styles of layer `managed`'s annotations `list` in a view (3D:
/// `projection`): the default shader's at once; a user shader's once its
/// compute pass is read back (`None` meanwhile, the pass started).
#[allow(clippy::too_many_arguments)]
fn annotation_styles(
    cache: &mut HashMap<(LayerId, bool, u64), StyleSlot>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wake: &Arc<dyn Fn() + Send + Sync>,
    managed: &ManagedLayer,
    list: &[annotations::Annotation],
    projection: bool,
    frame: u64,
) -> Option<Arc<Vec<ng_render::annotation_shader::AnnotationStyle>>> {
    use std::hash::{Hash, Hasher};
    let color = managed.annotation_color();
    let shader = managed.annotation_shader.as_ref();
    let Some((shader, pipelines)) = shader.and_then(|s| Some((s, s.pipelines.as_ref()?))) else {
        return Some(Arc::new(vec![ng_render::annotation_shader::AnnotationStyle::default_shader(color, projection); list.len()]));
    };
    // Each annotation's property bits, in the shader's property order.
    let mut bits = Vec::with_capacity(list.len() * shader.properties.len());
    for a in list {
        for (k, (_, t)) in shader.properties.iter().enumerate() {
            bits.push(t.bits(a.properties.get(k).copied().unwrap_or(0.0)));
        }
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    shader.code.hash(&mut h);
    shader.controls.iter().for_each(|c| c.to_bits().hash(&mut h));
    color.iter().for_each(|c| c.to_bits().hash(&mut h));
    list.iter().for_each(|a| a.id.hash(&mut h));
    bits.hash(&mut h);
    let key = (managed.id, projection, h.finish());
    if let Some(slot) = cache.get_mut(&key) {
        slot.frame = frame;
        if let StyleState::Pending(r) = &slot.state {
            if let Some(styles) = r.read() {
                slot.state = StyleState::Ready(Arc::new(styles));
            }
        }
        return match &slot.state {
            StyleState::Ready(styles) => Some(styles.clone()),
            StyleState::Pending(_) => None,
        };
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    let readback = pipelines[usize::from(projection)].evaluate(device, &mut encoder, &shader.controls, color, &bits, list.len());
    queue.submit([encoder.finish()]);
    readback.map(wake.clone());
    // A frame to poll for it (as for picks).
    wake();
    cache.insert(key, StyleSlot { state: StyleState::Pending(readback), frame });
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_tile_the_target_without_overlap() {
        let (w, h) = (800, 600);
        for layout in Layout::ALL {
            assert_eq!(Layout::from_name(&layout.name()), Some(layout));
            let panels = layout.panels(w, h);
            for (i, &(p, r)) in panels.iter().enumerate() {
                assert!(r.x + r.width <= w && r.y + r.height <= h, "{layout:?} {p:?} {r:?}");
                let (cx, cy) = (r.x + r.width / 2, r.y + r.height / 2);
                assert_eq!(layout.panel_at(cx as f64, cy as f64, w, h), Some(p));
                for &(_, o) in &panels[i + 1..] {
                    let apart = r.x + r.width <= o.x || o.x + o.width <= r.x || r.y + r.height <= o.y || o.y + o.height <= r.y;
                    assert!(apart, "{layout:?}: {r:?} overlaps {o:?}");
                }
            }
        }
        assert_eq!(Layout::from_name("xy-3d"), Some(Layout::SliceAnd3d(NamedAxes::Xy)));
        assert_eq!(Layout::from_name("4panel-alt").unwrap().rect(Panel::Perspective, w, h).unwrap().x, 402);
        assert_eq!(Layout::from_name("bogus"), None);
    }

    /// Panel rectangles as neuroglancer computes them from its layout's
    /// elements (measured in Chrome: element boxes, `clientLeft`,
    /// `clientWidth`).
    #[test]
    fn panel_geometry_matches_neuroglancer() {
        let r = |x, y, width, height| PixelRect { x, y, width, height };
        let four = Layout::FourPanel;
        assert_eq!(four.rect(Panel::Slice(NamedAxes::Xz), 800, 600), Some(r(402, 2, 396, 296)));
        assert_eq!(four.rect(Panel::Slice(NamedAxes::Yz), 1000, 700), Some(r(502, 352, 496, 346)));
        // 801 × 599: elements 400.5 × 299.5, so x = round(402.5) = 403, and
        // the sizes round half up (396.5 → 397, 295.5 → 296).
        assert_eq!(four.rect(Panel::Slice(NamedAxes::Xz), 801, 599), Some(r(403, 2, 397, 296)));
        assert_eq!(four.rect(Panel::Perspective, 801, 599), Some(r(2, 302, 397, 296)));
        assert_eq!(Layout::SliceAnd3d(NamedAxes::Xy).rect(Panel::Perspective, 801, 599), Some(r(403, 2, 397, 595)));
        assert_eq!(Layout::Single(NamedAxes::Xy).rect(Panel::Slice(NamedAxes::Xy), 801, 599), Some(r(2, 2, 797, 595)));
        // Without borders: the element boxes themselves.
        assert_eq!(four.panels_with(800, 600, false)[1].1, r(400, 0, 400, 300));
    }

    #[test]
    fn subrect_matrix_is_identity_for_the_same_rect() {
        let m: [f32; 16] = std::array::from_fn(|i| i as f32 + 1.0);
        let r = [3.0, 4.0, 50.0, 60.0];
        assert_eq!(subrect_matrix(m, r, r), m);
    }
}
