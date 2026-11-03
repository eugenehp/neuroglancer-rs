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
//! The 3D panel is neuroglancer's perspective view: the cross-sections as
//! planes, data bounds and axis lines, segment meshes and skeletons,
//! annotations, and volume rendering.

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
mod layers;
mod layout;
mod managed_layer;
mod pick;
mod pointer;
mod render;
mod selection;
mod skeleton_draw;
mod values;
/// User image shaders (`#uicontrol`s, GLSL to WGSL).
pub use ng_render::user_shader;
// the atlas records WebGL2-vs-WebGPU from threers' adapter through this
pub use ng_render::capabilities;

pub use ng_render::image::{default_data_type_range, expected_gray, Interval};

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
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

pub use layout::{Layout, Panel};
pub use managed_layer::{AnnotationDetails, AnnotationLayerState, AnnotationListEntry, LayerKind, ManagedLayer, PendingSource, SourceState, SourceView, SubsourceView};
use managed_layer::*;
use pick::*;
use render::*;


/// `crossSectionBackgroundColor`'s default.
pub const CROSS_SECTION_BACKGROUND: [f32; 3] = [0.5, 0.5, 0.5];

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
    /// The layer (index in `layers()`) and scale the source is.
    pub layer: usize,
    pub scale: usize,
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

/// Layer settings [`Viewer::set_layer_option`] applies in place, without
/// reloading the layer ([`ManagedLayer::live_settings`]).
pub const LIVE_LAYER_OPTIONS: [&str; 6] = ["opacity", "shader", "shaderControls", "crossSectionRenderScale", "volumeRendering", "volumeRenderingDepthSamples"];

/// A layer's volume chunks for the last frame, over every scale
/// ([`Viewer::layer_progress`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerProgress {
    /// Wanted by the view, on the GPU (drawn), downloading, failed.
    pub wanted: u32,
    pub ready: u32,
    pub downloading: u32,
    pub failed: u32,
    /// Bytes of the ready ones on the GPU.
    pub bytes: usize,
}

/// Converts a slab's little-endian voxels to texture bytes.
type UploadFn = fn(DataType, &[u8]) -> Vec<u8>;

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

/// How segments' meshes and skeletons are drawn. The default is
/// neuroglancer's (every mesh and skeleton, at full detail); a host showing
/// hundreds of neurons caps and thins them so a frame costs about the same
/// with a thousand as with ten.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneObjects {
    /// Draw a segment's skeleton where its layer has meshes too (neuroglancer
    /// does). Off: the skeleton still loads (asked for, kept) but is not
    /// drawn under the mesh — its lines and points are rebuilt on the CPU
    /// every frame, millions for a few detailed neurons (FlyWire's).
    pub draw_skeletons_with_meshes: bool,
    /// Skeletons with more edges than this are drawn thinned (their branch
    /// points, tips and every few vertices between), cached per skeleton:
    /// each drawn vertex is placed on the CPU every frame. 0: never thinned.
    pub skeleton_draw_edges: usize,
    /// A budget of skeleton edges drawn in all (0: none): shared among the
    /// skeletons drawn, each thinned to its share (at least 256; in powers of
    /// two, so it is thinned again only as the count crosses one).
    pub skeleton_total_edges: usize,
    /// Meshes' level of detail coarsened as more segments are shown (0: not):
    /// past this many, the detail cutoff grows with the square root of the
    /// count over it — the triangles drawn stay about level.
    pub mesh_detail_segments: usize,
    /// Draw skeletons at all (neuroglancer does). Off: they still load for
    /// the segments shown — the host draws them itself, batched
    /// (`scene_segment_skeleton_into`), a frame then not paying for each
    /// vertex.
    pub skeletons_drawn: bool,
    /// At most this many segments per layer drawn as meshes (0: all) — the
    /// last shown, where the layer has skeletons too: the rest stand for
    /// themselves by their skeletons, a mesh each being many draws.
    pub meshes_max_segments: usize,
}

impl Default for SceneObjects {
    fn default() -> Self {
        Self {
            draw_skeletons_with_meshes: true,
            skeleton_draw_edges: 0,
            skeleton_total_edges: 0,
            mesh_detail_segments: 0,
            skeletons_drawn: true,
            meshes_max_segments: 0,
        }
    }
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
    /// Scene mode (not neuroglancer's): which of the layout's cross-sections
    /// are drawn, by `NamedAxes::ALL` (xy, xz, yz) — a host showing or
    /// hiding one plane of the three without changing the layout.
    pub scene_planes: [bool; 3],
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
    /// (the cache's value it stands for — its slim form once uploaded —,
    /// the GPU's copy, the bytes uploaded)
    gpu_meshes: HashMap<MeshKey, (Arc<ng_datasource::mesh::MeshData>, ng_render::perspective::GpuMesh, usize)>,
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
    /// How segments' meshes and skeletons are drawn: neuroglancer's way by
    /// default; a host drawing many neurons tunes it ([`SceneObjects`]).
    pub objects: SceneObjects,
    /// Tunes `settings.concurrent_downloads` as the volume chunks download
    /// (`None`: the number stays as set), and the HTTP retries counted
    /// before this viewer (`ng_kvstore::retried` is the page's).
    pub adaptive_downloads: Option<ng_chunk::Adaptive>,
    retried_base: u64,
    /// Each slice panel's layers as last drawn (a hash of every chunk, rect,
    /// uniform and shader): equal, its colour is still right and is not
    /// drawn again.
    slice_drawn: [Option<u64>; 3],
    /// Slice panels drawn (not skipped as unchanged) since the viewer was made.
    slice_redraws: u64,
    thinned_skeletons: HashMap<MeshKey, (Arc<ng_datasource::skeleton::SkeletonData>, usize, Arc<ng_datasource::skeleton::SkeletonData>)>,
    /// Multi-resolution meshes' manifests and fragments, the fragments on
    /// the GPU, and this frame's fragment reads (location, segment, byte
    /// range, quantization bits, whether partitioned).
    manifests: ChunkManager<MeshKey, ng_datasource::multiscale_mesh::Manifest>,
    fragments: ChunkManager<FragmentKey, ng_datasource::multiscale_mesh::Fragment>,
    /// The layers' ids at the last frame: when they change, the caches drop
    /// (and stop downloading) what belonged to layers that are gone.
    live_layers: Vec<LayerId>,
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
    /// The near and far planes the last scene frame drew with (eye
    /// distances; fitted to `SceneView::bounds`), for the composite.
    scene_depth: Option<scene::DepthRanges>,
    /// The composite into the host's frame, made on first use.
    scene_composite: Option<scene::Composite>,
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
            scene_planes: [true; 3],
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
            objects: SceneObjects::default(),
            adaptive_downloads: None,
            retried_base: ng_kvstore::retried(),
            slice_drawn: [None; 3],
            slice_redraws: 0,
            thinned_skeletons: HashMap::new(),
            annotation_hover: None,
            pick_ids: vec![],
            slice_picks: Default::default(),
            pending_pick: None,
            perspective_pick: None,
            wake: wake.clone(),
            manifests: ChunkManager::new(executor.clone(), wake.clone()),
            fragments: ChunkManager::new(executor.clone(), wake.clone()),
            live_layers: Vec::new(),
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
            scene_depth: None,
            scene_composite: None,
        }
    }

    /// A viewer for scene mode only ([`render_scene`](Self::render_scene)):
    /// no layout target of its own (its panels are offscreen), segments'
    /// objects drawn as `objects` says.
    pub fn new_scene(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, executor: Arc<dyn Executor>, wake: Arc<dyn Fn() + Send + Sync>, objects: SceneObjects) -> Self {
        // the format is only the (unused) layout compositor's
        let mut v = Self::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, 1, 1, 1.0, executor, wake);
        v.objects = objects;
        v
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

    /// Takes in finished chunk loads; returns whether any arrived (redraw).
    pub fn poll(&mut self) -> bool {
        self.chunks.poll() | self.meshes.poll() | self.skeletons.poll() | self.annotation_cells.poll() | self.annotations_by_id.poll() | self.manifests.poll() | self.fragments.poll() | self.poll_pick() | self.poll_styles()
    }

    /// Whether chunks are still loading.
    pub fn is_loading(&self) -> bool {
        self.chunks.in_flight() + self.meshes.in_flight() + self.skeletons.in_flight() + self.annotation_cells.in_flight() + self.annotations_by_id.in_flight() + self.manifests.in_flight() + self.fragments.in_flight() > 0
    }

    /// The volume chunks' download totals (`ng_chunk::DownloadStats`), and
    /// how many are in flight now.
    pub fn chunk_downloads(&self) -> (ng_chunk::DownloadStats, usize) {
        (self.chunks.stats, self.chunks.in_flight())
    }

    /// Downloads in flight: volume chunks, and everything else (meshes,
    /// skeletons, annotations, manifests).
    pub fn downloads_in_flight(&self) -> (usize, usize) {
        let other = self.meshes.in_flight() + self.skeletons.in_flight() + self.annotation_cells.in_flight() + self.annotations_by_id.in_flight() + self.manifests.in_flight() + self.fragments.in_flight();
        (self.chunks.in_flight(), other)
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

    /// What the viewer holds, by cache: bytes and entries — for a host that
    /// has to fit it in a bounded memory (a 32-bit wasm module) and wants to
    /// see where it goes.
    pub fn memory_statistics(&self) -> serde_json::Value {
        let mut chunk_entries = 0usize;
        self.chunks.for_each(|_, _, _, _, _| chunk_entries += 1);
        let textures: usize = self.layers.iter().filter_map(|l| l.render.as_ref()).map(|r| r.textures.len()).sum();
        serde_json::json!({
            "chunks": { "bytes": self.chunks.memory_used(), "entries": chunk_entries, "budget": self.chunks.memory_budget },
            "meshes": self.meshes.memory_used(),
            "skeletons": self.skeletons.memory_used(),
            "manifests": self.manifests.memory_used(),
            "fragments": self.fragments.memory_used(),
            "annotationCells": self.annotation_cells.memory_used(),
            "chunkTextures": textures,
            "gpuMeshes": self.gpu_meshes.len(),
            "gpuFragments": self.gpu_fragments.len(),
            "meshedSegments": self.scene_meshed_segments().len(),
            "skeletonSources": self.layers.iter().map(|l| l.skeleton_sources().len()).sum::<usize>(),
            // what the meshes on the GPU were made from (the GPU's copy, about as big)
            "gpuMeshBytes": self.gpu_meshes.values().map(|(_, _, b)| b).sum::<usize>() + self.gpu_fragments.values().map(|(d, _)| d.bytes()).sum::<usize>(),
            "sliceRedraws": self.slice_redraws,
        })
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
                SourceStatistics { properties, layer: li, scale: *scale, ..Default::default() }
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
    /// its own input and composites the output into its frame by depth
    /// ([`composite_scene`](Self::composite_scene)).
    pub fn render_scene(&mut self, mut view: scene::SceneView) {
        let (w, h) = (view.width.max(view.slice_size).max(1), view.height.max(view.slice_size).max(1));
        let (camera, depth) = view.fitted();
        view.camera = camera;
        self.scene_depth = depth;
        let saved = (self.width, self.height, self.scale_factor, self.viewport);
        (self.width, self.height, self.scale_factor, self.viewport) = (w, h, 1.0, None);
        self.scene = Some(view);
        self.render_into(None);
        self.scene = None;
        (self.width, self.height, self.scale_factor, self.viewport) = saved;
    }

    /// The near and far planes the last scene frame drew with, and the
    /// host's (`None` for an orthographic camera: the host's depth as is).
    pub fn scene_depth_ranges(&self) -> Option<scene::DepthRanges> {
        self.scene_depth
    }

    /// Draws the last scene frame over the host's: into `view` (`format`),
    /// hidden where `scene_depth` (the host's depth for the frame, `samples`
    /// per pixel) is nearer, unless `on_top`. The viewer's depth is taken to
    /// the host's range first, so the two compare as one.
    pub fn composite_scene(&mut self, view: &wgpu::TextureView, format: wgpu::TextureFormat, scene_depth: &wgpu::TextureView, samples: u32, on_top: bool) {
        if !matches!(&self.scene_composite, Some(c) if c.matches(format, samples)) {
            self.scene_composite = Some(scene::Composite::new(&self.device, format, samples, !capabilities::compute()));
        }
        let (Some(out), Some(composite)) = (self.scene_output(), self.scene_composite.as_ref()) else { return };
        composite.draw(&self.device, &self.queue, view, &out, scene_depth, on_top, self.scene_depth);
    }

    /// The index of the layer named `name`.
    pub fn layer_index(&self, name: &str) -> Option<usize> {
        self.layers.iter().position(|l| l.name() == name)
    }

    /// Shows layer `layer` only at cross-section scale `max` or finer
    /// (`ManagedLayer::max_cross_section_scale`; `None`: always).
    pub fn set_zoom_gate(&mut self, layer: usize, max: Option<f64>) {
        if let Some(l) = self.layers.get_mut(layer) {
            l.max_cross_section_scale = max;
        }
        self.update_zoom_gates();
    }

    /// Each layer's zoom gate for the cross-section scale now; whether any
    /// changed.
    pub(crate) fn update_zoom_gates(&mut self) -> bool {
        let zoom = self.nav.zoom(View::CrossSection);
        let mut changed = false;
        for l in &mut self.layers {
            // Hidden past the limit, and when the zoom is NaN.
            let hidden = l.max_cross_section_scale.is_some_and(|m| !matches!(zoom.partial_cmp(&m), Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)));
            changed |= l.zoom_hidden != hidden;
            l.zoom_hidden = hidden;
        }
        changed
    }

    /// Each layer's volume chunks for the last frame, over every scale: as
    /// [`chunk_statistics`](Self::chunk_statistics), summed by layer
    /// (indexed as [`layers`](Self::layers)).
    pub fn layer_progress(&self) -> Vec<LayerProgress> {
        let mut out = vec![LayerProgress::default(); self.layers.len()];
        for st in self.chunk_statistics() {
            let Some(o) = out.get_mut(st.layer) else { continue };
            o.wanted += st.visible_total;
            o.ready += st.gpu;
            o.downloading += st.downloading;
            o.failed += st.failed;
            o.bytes += st.gpu_bytes;
        }
        out
    }

    /// Slice panels drawn since the viewer was made: those whose layers
    /// changed (a slice whose draws are as last frame's is not drawn again).
    pub fn slice_redraws(&self) -> u64 {
        self.slice_redraws
    }

    /// HTTP requests retried (the server pushing back, or failing for now)
    /// since this viewer was made.
    pub fn retried(&self) -> u64 {
        ng_kvstore::retried().saturating_sub(self.retried_base)
    }

    /// Changes whenever the segments a host draws for might: a segment shown
    /// or hidden, a layer shown, hidden or gone, a skeleton or mesh arriving
    /// or freed. Equal values: `scene_shown_segments`, `scene_meshed_segments`
    /// and the segments' skeletons are as they were.
    pub fn scene_segments_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for l in &self.layers {
            (l.id, l.shown(), matches!(l.kind(), Some(LayerKind::Segmentation(_)))).hash(&mut h);
            if l.shown() {
                let segments = l.segments();
                segments.len().hash(&mut h);
                for &(id, visible) in segments.iter() {
                    (id, visible).hash(&mut h);
                }
            }
        }
        (self.skeletons.revision(), self.meshes.revision(), self.fragments.revision(), self.gpu_fragments.len(), self.objects.meshes_max_segments).hash(&mut h);
        h.finish()
    }

    /// The 3D output of the last [`render_scene`](Self::render_scene).
    pub fn scene_output(&self) -> Option<scene::SceneOutput<'_>> {
        let pf = self.perspective_frame.as_ref().filter(|pf| pf.target.sampled_depth)?;
        Some(scene::SceneOutput { color: &pf.target.flipped_view, depth: &pf.target.depth_view, width: pf.target.width, height: pf.target.height })
    }

    /// The cross-sections scene mode shows for the layout: that slice for one
    /// of a cross-section and the 3D view (`xy-3d`) or a slice alone, else
    /// all three (before `scene_planes` picks among them).
    pub fn scene_axes(&self) -> Vec<NamedAxes> {
        match &self.layout {
            Layout::SliceAnd3d(a) | Layout::Single(a) => vec![*a],
            _ => NamedAxes::ALL.to_vec(),
        }
    }

    pub fn panel_rects(&self) -> Vec<(Panel, PixelRect)> {
        if let Some(scene) = &self.scene {
            // a layout of one cross-section and the 3D view (`xy-3d`, or one
            // slice alone) shows that slice in the scene; any other, all three
            let axes: Vec<NamedAxes> = self
                .scene_axes()
                .into_iter()
                .filter(|a| self.scene_planes[NamedAxes::ALL.iter().position(|b| b == a).unwrap_or(0)])
                .collect();
            return scene.panel_rects(self.show_slices, &axes);
        }
        let (x, y, w, h) = self.physical_region();
        self.layout
            .panels_scaled(w, h, self.scale_factor)
            .into_iter()
            .map(|(p, r)| (p, PixelRect { x: r.x + x, y: r.y + y, ..r }))
            .collect()
    }

}

/// A global position in a layer's model coordinates (float32, as
/// neuroglancer keeps them); its unmapped layer dimensions at 0.
fn global_to_model(l: &VolumeLayer, position: &[f32]) -> Option<Vec<f64>> {
    let n = l.rank + 1;
    let layer_pos = l.layer_position(position, 0.0)?;
    let (inv, _) = ng_core::matrix::inverse(&l.model_to_layer, n);
    Some((0..l.source_rank).map(|c| ((0..n).map(|k| inv[k * n + c] * layer_pos[k]).sum::<f64>()) as f32 as f64).collect())
}

/// A chunk volume rendering drew: data, chunk display dimensions, fixed
/// offsets.
type VolumeChunk = (Arc<ChunkData>, [Option<usize>; 3], Vec<(usize, u64, u32)>);

/// Vertices a single-resolution mesh keeps in memory once on the GPU
/// (`MeshData::slim`).
const MESH_SLIM_POINTS: usize = 4096;

/// `pickRadius`'s default.
const PICK_RADIUS: u32 = 5;

/// Where boxes go: their lower and upper corners (model space), border
/// colour and width, and pick ID.
type BoxSink<'a> = dyn FnMut(&[f64], &[f64], [f32; 4], f64, u32) + 'a;

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

/// Prefetched chunks' priorities sit this far below every visible chunk's
/// (`ChunkPriorityTier.PREFETCH` after `VISIBLE`).
const PREFETCH_TIER: f64 = -1e15;
