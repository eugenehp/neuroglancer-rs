//! neuroglancer's user interface over the engine: the top bar (position
//! widget with its dimension dropdowns, mouse position, panel buttons), the
//! layer bar, and each data panel's controls (display dimensions, layout
//! buttons, the 3D panel's "Sections" checkbox).
//!
//! The engine crates (`ng-core` … `ng-viewer`) know nothing of this crate;
//! it reads a [`Viewer`] and acts on it. It is built with [`threers::ui`] and
//! its [`style!`](threers::style) macro: boxes styled with neuroglancer's own
//! CSS (read from its stylesheets, checked against Chrome's layout), laid out
//! with flexbox, painted through a [`Painter`] — the page's 2D canvas in a
//! browser, so text renders as the DOM's does, or threers' overlay natively.
//!
//! Each frame a shell calls [`NgUi::update`] (which also confines the
//! viewer's panels to the area below the bars), renders the viewer, then
//! paints the UI over it with [`NgUi::paint`]. Input goes to the UI first —
//! [`NgUi::pointer_move`], [`pointer_down`](NgUi::pointer_down),
//! [`pointer_up`](NgUi::pointer_up), [`wheel`](NgUi::wheel),
//! [`key_down`](NgUi::key_down) — and an [`Action`] it returns is carried
//! out with [`NgUi::apply`]; what the UI does not take goes on to the viewer.

pub mod about;
pub mod icons;
mod annotation_tools;
mod annotations;
mod dialogs;
mod layer_panel;
mod rendering;
mod segments;
mod sources;
mod statistics;
pub use segments::SegmentChange;
pub use rendering::InvlerpCtl;
mod palette;
mod panels;
mod plot;
pub mod side;
mod palettes;
mod shader_ui;
mod tools;
mod view;

use std::collections::HashMap;

use ng_core::js::number_to_string;
use ng_viewer::{LayerId, Layout, Viewer};
use threers::ui::{NodeId, Painter, Rect, TextMeasure, Ui};

pub use icons::Icon;
pub use side::{Location, PanelId, Side};
use threers::ui::{FieldKey, KeyModifiers, TextField};
pub use ng_viewer::input::Modifiers;
pub use ng_viewer::Button;

/// What an input asks of the viewer, for [`NgUi::apply`].
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Only the UI changed (hover, typing): paint it again.
    Repaint,
    /// A layer bar item: show or hide the layer.
    ToggleLayerVisible(usize),
    /// Alt+click on a layer: `pickEnabled` on or off.
    TogglePick(usize),
    /// Right-click on a layer: select it and show its side panel
    /// (`selectedLayer.layer = layer; visible = true`).
    SelectLayer(usize),
    /// A layer's close button: archive it (`setArchived(true)`).
    ArchiveLayer(usize),
    /// A layer's delete button (`deleteLayer`).
    DeleteLayer(usize),
    /// A panel's layout button.
    SetLayout(Layout),
    /// The 3D panel's "Sections" checkbox (`showSlices`).
    ToggleShowSlices,
    /// The position widget's copy button: `formatPositionText` to the
    /// clipboard.
    CopyPosition,
    /// The top bar's clipboard button: the page's URL to the clipboard.
    CopyUrl,
    /// Coordinates typed in, or one picked on a dimension's plot. A whole
    /// number typed, or any picked, goes to the voxel centre on dimensions
    /// whose centres are at half-integers (`updatePosition`,
    /// `setPositionFromMouse`).
    SetCoordinates(Vec<Coordinate>),
    /// The wheel or ↑/↓ over a dimension: step it, to a voxel centre
    /// (`adjustDimensionPosition`).
    AdjustCoordinate(usize, f64),
    /// A top-bar button or shortcut: show or hide a side panel.
    TogglePanel(PanelId),
    /// A side panel's close button.
    ClosePanel(PanelId),
    /// Panels moved or resized (a gutter dragged).
    SetLocations(Vec<(PanelId, Location)>),
    /// A Settings checkbox, or its shortcut.
    Toggle(Setting),
    /// A Settings number committed.
    SetNumber(NumberSetting, f64),
    /// The title committed.
    SetTitle(String),
    /// A colour chosen for a Settings colour input.
    SetColor(ColorSetting, [f32; 3]),
    /// A Settings colour input clicked: the shell shows a colour picker.
    PickColor(ColorSetting),
    /// The Layers panel's eye: the layer bar shown or hidden
    /// (`uiControlVisibility.showLayerPanel`).
    ToggleLayerBar,
    /// A Layers panel checkbox: archived or not.
    SetArchived(usize, bool),
    /// A layer's panel toggle (`makeSelectedLayerSidePanelCheckboxIcon`).
    ToggleLayerSidePanel(usize),
    /// A layer renamed.
    RenameLayer(usize, String),
    /// `select-position` (Ctrl/⌘+right-click on a panel): the selection
    /// captured and pinned, its panel shown; `unpin-selected-position`
    /// (with Shift): unpinned, following the pointer.
    SelectPosition { pin: bool },
    /// The selection panel's back/forward buttons.
    SelectionHistory(i32),
    /// The selection panel's pin.
    ToggleSelectionPin,
    /// Move-to buttons: the position set.
    MoveTo(Vec<f32>),
    /// Copy buttons: text to the clipboard.
    CopyText(String),
    /// A segment shown or hidden (`visibleSegments`).
    ToggleSegment(usize, u64),
    /// A layer panel tab chosen (the layer's `tab`).
    SelectLayerTab(usize, &'static str),
    /// A source's "Enable default subsource set" (layer, source).
    ToggleDefaultSubsources(usize, usize),
    /// A subsource checkbox (layer, source, subsource).
    ToggleSubsource(usize, usize, usize),
    /// A layer's JSON member set (or removed), and the settings it holds
    /// applied (`Viewer::set_layer_option`).
    SetLayerOption(usize, String, Option<serde_json::Value>),
    /// Segments starred, unstarred, shown or hidden (the Seg. tab).
    SegmentChange(usize, segments::SegmentChange),
    /// `select` (show or hide) or `star` the segment under the pointer in
    /// each visible segmentation layer that picks
    /// (`SegmentationUserLayer.handleAction`).
    ToggleHoveredSegments { star: bool },
    /// `annotate` (Ctrl+click on a panel): the selected layer's annotation
    /// tool acts at the pointer.
    Annotate,
    /// An annotation layer's list entry `n`: pinned (`pin-annotation`,
    /// click), or the position moved to its centre (`move-to-annotation`,
    /// right-click).
    PinAnnotation(usize, usize),
    /// `finish-annotation` (Enter) or `undo-annotation-step` (Backspace) on
    /// a panel: the selected layer's tool's `complete` or `undo`.
    AnnotationStep { finish: bool },
    MoveToAnnotation(usize, usize),
    /// Deletes an annotation (layer, ID).
    DeleteAnnotation(usize, String),
    /// `moveToSegment` (right-click on a segment's entry): to the segment's
    /// loaded mesh, else a message.
    MoveToSegment(usize, u64),
    /// An invlerp range finder button: the range between these percentiles
    /// of the sampled data (`AutoRangeFinder`).
    AutoRange { layer: usize, ctl: rendering::InvlerpCtl, range: [f64; 2], lower: f64, upper: f64 },
    /// A segmentation layer's randomize button: a new colour seed
    /// (`segmentColorHash.randomize()`).
    RandomizeSeed(usize),
    /// The top bar's `{}`: the JSON state editor (`edit-json-state`).
    EditJsonState,
    /// A dialog's Close (or Escape).
    CloseDialog,
    /// The layer bar's `+` (`add-layer`): a new layer, its Source tab open
    /// with the URL focused (`addNewLayer`).
    AddLayer,
    /// A layer's source URL committed: loads it.
    SetLayerSource(usize, String),
    /// `Create as … layer` (`changeLayerType`).
    ChangeLayerType(usize, &'static str),
    /// A completion chosen for the focused URL.
    Complete(String),
    /// `open-command-palette`; a row run (by place); back out of a group;
    /// a command run by id.
    OpenPalette,
    PaletteRun(usize),
    PaletteBack,
    PaletteCommand(String),
    /// Playback: enabled (`Some`) or not, and its settings, for a dimension;
    /// paused or not (`toggle-playback`); its speed scaled (Alt+wheel); its
    /// boundary behaviour chosen.
    SetVelocity(usize, Option<ng_viewer::playback::DimensionVelocity>),
    TogglePlayback(usize),
    MultiplyVelocity(usize, f64),
    SetBoundary(usize, String),
    /// A tool bound to a key (`LocalToolBinder.set`), unbound
    /// (`removeJsonString`); the wheel over the data panels with a tool
    /// active.
    BindTool(char, ng_viewer::tools::ToolOwner, serde_json::Value),
    UnbindTool(ng_viewer::tools::ToolOwner, serde_json::Value),
    ToolWheel { delta_y: f64, modifiers: Modifiers },
    /// Tool palettes: the wrench's list shown or not; one added (named,
    /// with a query); one deleted; its query's results made its list; a
    /// tool removed from its list.
    TogglePaletteList,
    AddPalette { name: String, query: String },
    DeletePalette(u16),
    ConvertPalette(u16),
    RemovePaletteTool(u16, usize),
    RenamePalette(u16, String),
    /// The global dimensions renamed, rescaled (`updateNames`, `updateScales`).
    RenameDimensions(Vec<String>),
    /// A link followed (a help button).
    OpenUrl(String),
    /// Layer `i`'s shader in a larger editor (`ShaderCodeOverlay`).
    OpenShaderOverlay(usize),
    /// A pinned layer panel's tab chosen (layer, panel, tab); the selected
    /// layer's panel pinned to its layer; a pinned one made the selected
    /// layer's again.
    SelectPinnedTab(usize, usize, &'static str),
    PinLayerPanel,
    UnpinLayerPanel(usize, usize),
    /// A user shader's control set (`None`: its default).
    SetShaderControl(usize, &'static str, Option<serde_json::Value>),
    /// An invlerp's channel coordinate typed: layer, control, dimension,
    /// the text.
    SetShaderChannel(usize, &'static str, usize, String),
    /// A dragged layer over layer `Some(i)` of the bar, or its empty end.
    LayerDragOver(Option<usize>),
    SetDimensionScales(Vec<String>),
    SetPaletteQuery(u16, String),
    /// The select segments tool pressed on a panel (`drag-select-segments`,
    /// Alt to deselect) at the pointer.
    SelectSegments { layer: usize, deselect: bool, x: f32, y: f32 },
    /// The top bar's camera: the screenshot dialog; its scale, its FOV
    /// checkbox, Take, Cancel.
    OpenScreenshot,
    ScreenshotScale(f64),
    ScreenshotKeepFov,
    TakeScreenshot,
    CancelScreenshot,
    /// The state editor: a fold marker clicked (by line); Apply; Download.
    ToggleFold(usize),
    ApplyState,
    DownloadState,
    /// A global shortcut (`getDefaultGlobalBindings`), by action name.
    Shortcut(String),
    /// A control whose panel or dialog is not ported yet, by its title.
    Unported(&'static str),
    /// Several, in order.
    Many(Vec<Action>),
}

/// A coordinate to set (see [`Action::SetCoordinates`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate {
    pub dim: usize,
    pub value: f64,
    pub to_voxel_center: bool,
}

/// A boolean viewer setting (the Settings panel's checkboxes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Setting {
    ShowAxisLines,
    ShowScaleBar,
    ShowSlices,
    HideCrossSectionBackground3D,
    ShowDefaultAnnotations,
    ShowStatistics,
    WireFrame,
    Prefetch,
    AdaptiveDownsampling,
    OrthographicProjection,
}

/// A numeric viewer setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumberSetting {
    GpuMemoryLimit,
    SystemMemoryLimit,
    ConcurrentDownloads,
}

/// A colour viewer setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSetting {
    CrossSectionBackground,
    ProjectionBackground,
    /// A segmentation layer's `segmentDefaultColor`.
    SegmentDefault(usize),
    /// A layer's `annotationColor`.
    Annotation(usize),
    /// A user shader's colour control (layer, control).
    ShaderColor(usize, &'static str),
}

/// A text field outside the position widget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldId {
    Title,
    Number(NumberSetting),
    /// A layer's name (Layers panel, layer panel).
    LayerName(usize),
    /// An invlerp bound: layer, window (else range), endpoint.
    Bound(usize, bool, usize),
    /// A layer slider's number (`.range-slider input[type=number]`), by the
    /// JSON member it sets.
    Slider(usize, &'static str),
    /// A segmentation layer's colour seed (`TextInputWidget`).
    ColorSeed(usize),
    /// A segmentation layer's segment query (`segmentQuery`).
    SegmentQuery(usize),
    /// The screenshot dialog's name.
    ScreenshotName,
    /// A layer's data source URL.
    SourceUrl(usize),
    /// A dimension's playback speed.
    Velocity(usize),
    /// A user shader's slider control (layer, control).
    ShaderSlider(usize, &'static str),
    /// An invlerp's channel coordinate along a channel dimension: layer,
    /// control, dimension.
    ShaderChannel(usize, &'static str, usize),
    /// A global dimension's name, its scale (double-clicked to edit).
    DimensionName(usize),
    DimensionScale(usize),
    /// A tool palette's name, its query.
    PaletteName(u16),
    PaletteQuery(u16),
}

/// What the shell has to do after [`NgUi::apply`].
#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    /// Show a colour picker for `setting`, starting at `current`; report the
    /// choice with [`Action::SetColor`].
    PickColor { setting: ColorSetting, current: [f32; 3] },
    /// Nothing more (the next frame shows the change).
    None,
    /// Load these data sources (a layer's source set), then hand each to
    /// `Viewer::source_loaded`.
    LoadSources(Vec<ng_viewer::PendingSource>),
    /// Restore this state (the JSON editor's Apply), loading its sources.
    RestoreState(serde_json::Value),
    /// Save `bytes` (of type `mime`) as a file called `name` (the JSON
    /// editor's Download, a screenshot).
    Download { name: String, bytes: Vec<u8>, mime: String },
    /// Put this text on the system clipboard.
    Clipboard(String),
    /// Put the page's URL (the viewer state in its fragment) on the
    /// clipboard.
    CopyUrl,
    /// Open this link in a new window (`<a target="_blank">`).
    OpenUrl(String),
}

/// The pointer's shape over a spot (CSS `cursor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cursor {
    Default,
    Pointer,
    Crosshair,
    Text,
}

/// What a node is, for hover styling and input. Node IDs change as the tree
/// is rebuilt; keys stay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Icon(u32),
    LayerPanel,
    LayerItem(usize),
    LayoutButton(usize, usize),
    Sections,
    /// A panel's display dimensions or layout controls: over the panel, but
    /// not the panel (`event.target !== element`).
    PanelWidget(usize),
    /// `.neuroglancer-position-dimension`, its name, scale and coordinate.
    Dimension(usize),
    DimensionName(usize),
    DimensionScale(usize),
    Coordinate(usize),
    /// The open dimension dropdown, its plot canvas, its tool key binding.
    Dropdown,
    Plot,
    /// A tool's key binding widget, a bound tool in the top bar (by place
    /// this frame), the status messages.
    ToolBinding(u32),
    ToolStatus(u32),
    Status,
    /// A tool palette's layer group (palette, layer), a tool in it (by
    /// place); the wrench's list, a palette in it, its shown checkbox, a
    /// canned palette.
    PaletteGroup(u16, usize),
    PaletteTool(u16, u16),
    PaletteDropdown,
    PaletteListItem(u16),
    PaletteVisible(u16),
    PaletteCanned(usize),
    /// The colour picker (where the system has none): its box, its
    /// saturation/value field, its hue strip.
    ColorPicker,
    ColorPickerField,
    ColorPickerHue,
    /// A layer's shader editor, a line of it (layer, line), an error marker.
    ShaderEditor(usize),
    ShaderLine(usize, usize),
    ShaderLint(usize, usize),
    /// A side panel, its close button, the gutter beside a column of
    /// panels (by side and column), between two panels (and the first's
    /// index).
    Panel(PanelId),
    PanelClose(PanelId),
    Gutter(Side, usize),
    FlexGutter(Side, usize, usize),
    /// Settings controls.
    Setting(Setting),
    Field(FieldId),
    Color(ColorSetting),
    /// The help panel's scrolling list.
    Scroll(PanelId),
    /// The layer panel's tabs, the Source tab's checkboxes
    /// (layer, source, subsource).
    LayerTab(PanelId, usize),
    SourceDefaults(usize, usize),
    Subsource(usize, usize, usize),
    /// A selection panel layer title; a segment row (layer, ID).
    SelectionLayer(usize),
    SegmentEntry(usize, u64),
    /// A Layers panel row, its archive checkbox, its colour chip.
    ListItem(usize),
    ListArchive(usize),
    ListColor(usize),
    /// A layer control, by layer and name (`blend`, `opacity`, …).
    Control(usize, &'static str),
    /// An open `<select>`'s option.
    MenuItem(usize),
    /// A statistics table row (by place).
    StatisticsRow(usize),
    /// A source URL's completion (by place).
    Completion(usize),
    /// A command palette row (by place).
    PaletteRow(usize),
    /// An Annotations tab entry (layer, index).
    AnnotationEntry(usize, usize),
    /// The open dialog's box, its buttons, its editor, the editor's
    /// scrolling body, a shown line (by place), a fold marker (by line).
    Dialog,
    DialogButton(u8),
    DialogEditor,
    DialogScroll,
    DialogLine(usize),
    DialogFold(usize),
}

/// A gutter being dragged: what it resizes and from where.
#[derive(Debug, Clone, PartialEq)]
enum GutterDrag {
    /// A column's width (a row's height): the pointer along that axis at
    /// the start, the size then, and the group's panels.
    Column { side: Side, x0: f32, size0: f64, panels: Vec<(PanelId, Location)> },
    /// Two neighbouring panels' shares: their locations, where the first
    /// starts and the second ends along the group (`across` for a row).
    Split { first: (PanelId, Location), second: (PanelId, Location), top: f32, bottom: f32, across: bool },
}

/// A coordinate being typed (the focused `<input>`): its text and selection
/// (`start..end`, in characters).
#[derive(Debug, Clone, PartialEq)]
struct Edit {
    dim: usize,
    text: String,
    start: usize,
    end: usize,
    /// Typed into since focused or last committed (`widget.modified`).
    modified: bool,
}

/// The open dropdown's plot as laid out: its dimension, canvas box and
/// bounds.
#[derive(Debug, Clone)]
struct PlotGeometry {
    dim: usize,
    canvas: Rect,
    bounds: plot::DimensionBounds,
}

/// neuroglancer's UI around a [`Viewer`].
pub struct NgUi {
    /// The port's credits: the footer and the Settings panel's About
    /// (neuroglancer has neither; off to compare with it).
    pub attribution: bool,
    ui: Ui,
    keys: HashMap<NodeId, Key>,
    actions: HashMap<Key, Action>,
    /// Keys under the pointer, innermost first.
    hovered: Vec<Key>,
    pointer: Option<(f32, f32)>,
    /// `maxPositionWidthSeen` per dimension name.
    max_position_seen: HashMap<String, usize>,
    /// The widest hover value seen per layer, in characters (`maxLength`).
    max_value_len: HashMap<LayerId, usize>,
    /// `showLayerHoverValues`.
    pub show_layer_hover_values: bool,
    /// `showAllDimensionPlotBounds`.
    pub show_all_plot_bounds: bool,
    /// The data panels' boxes as last laid out, and the layout they are of.
    panels: Vec<[f32; 4]>,
    layout: Option<Layout>,
    /// The panel the pointer is in, and the one with focus: a panel takes it
    /// when the pointer enters (`AutomaticallyFocusedElement`) and shows a
    /// white border while it has it (`:focus-within`).
    entered: Option<usize>,
    focused: Option<usize>,
    /// An `<input>` (a range slider, a number) has the focus: panels the
    /// pointer enters leave it (`isInputTextTarget`) until a press
    /// elsewhere.
    input_focused: bool,
    /// Each layer's hovered segment as of this update
    /// (`segmentSelectionState`): its list entries are `data-selected`.
    hovered_segments: Vec<Option<u64>>,
    /// `StatusMessage.showTemporaryMessage`s: text, and when it goes (ms).
    temporary_messages: Vec<(String, f64)>,
    /// An annotation being placed (`MultiStepAnnotationTool.inProgressAnnotation`).
    annotating: Option<annotation_tools::InProgress>,
    /// An annotation to scroll into view in its list (`scrollItemIntoView`
    /// on selection), and, while building, its list and entry.
    scroll_to_annotation: Option<(usize, String)>,
    annotation_scroll: Option<(threers::ui::NodeId, threers::ui::NodeId)>,
    /// The focused element is going away with the panels (a layout button
    /// had focus): the first new panel takes it (`maybeUpdateFocus`).
    refocus: bool,
    /// The coordinate being edited, the dimension whose dropdown is open,
    /// and that dropdown's plot.
    edit: Option<Edit>,
    dropdown: Option<usize>,
    plot: Option<PlotGeometry>,
    /// The coordinate under the pointer on the plot, and whether the
    /// position is being dragged along it.
    plot_hover: Option<f64>,
    plot_drag: bool,
    /// The pointer has moved since the press (`dragging`): the label shows
    /// the position, and no hover line.
    plot_dragged: bool,
    /// Coordinates typed into but not committed, other than the focused
    /// one's (`widget.modified`): shown, and committed with it.
    pending: std::collections::BTreeMap<usize, String>,
    /// A gutter being dragged.
    gutter_drag: Option<GutterDrag>,
    /// The open dialog, its shown lines (by place), the monospace advance
    /// it was laid out with.
    dialog: Option<dialogs::Dialog>,
    /// Tooltips of keyed controls, as last built; when the hovered set
    /// last changed.
    titles: HashMap<Key, String>,
    hover_since: f64,
    /// Dimensions' playback, as last built (for input without the viewer).
    velocities: HashMap<usize, ng_viewer::playback::DimensionVelocity>,
    /// Starred segments per layer, for the selection panel's stars.
    starred: HashMap<usize, Vec<u64>>,
    /// The highlighted source URL completion (↑/↓).
    completion_active: Option<usize>,
    dialog_lines: Vec<usize>,
    mono_advance: f32,
    /// The palette's results' visible height (to scroll the active row in).
    palette_results_height: f32,
    /// Layer controls as last built, the one being dragged (and the
    /// endpoint it moves), the `<select>` whose options are open.
    controls: HashMap<Key, rendering::Control>,
    control_drag: Option<(Key, usize)>,
    menu: Option<Key>,
    /// Tools as last built (by place), the one whose binding widget is
    /// held (the next letter binds it), the active tool's key.
    tool_refs: Vec<tools::ToolRef>,
    binding_tool: Option<tools::ToolRef>,
    active_tool: Option<char>,
    active_ref: Option<tools::ToolRef>,
    /// The select segments tool's drag.
    select_paint: Option<tools::SelectPaint>,
    /// A layer item pressed (and maybe dragged).
    layer_drag: Option<LayerDrag>,
    /// The layer panel being built (its tabs scroll under its key).
    layer_panel: PanelId,
    /// Whether the wrench's list of palettes shows.
    palette_list: bool,
    /// The wrench, as last built.
    palette_button: Option<NodeId>,
    /// The colour picker open (which setting, the colour in HSV), and the
    /// part being dragged (`true` the field, `false` the hue).
    color_picker: Option<(ColorSetting, [f32; 3])>,
    picker_drag: Option<bool>,
    /// The shader being edited, the `ch` of its font, its shown lines as
    /// last built, and whether a press elsewhere has taken its keyboard.
    shader_edit: Option<shader_ui::ShaderEdit>,
    shader_advance: f32,
    shader_lines: Vec<(NodeId, usize)>,
    shader_blur_pending: bool,
    /// Invlerp legends of user shaders: their widths as last laid out, and
    /// the colours the viewer drew for them (by slot and control).
    legend_widths: HashMap<(usize, &'static str), usize>,
    legend_colors: HashMap<(usize, &'static str), Vec<[u8; 4]>>,
    /// The page's size, as last built.
    viewport: (f32, f32),
    /// Each global dimension's name and scale text, as last built.
    dimension_texts: Vec<(String, String)>,
    /// The invlerp channel cells built (`channelInvlerpLayerControl`'s
    /// position widgets): field, cell, the dimension's extent.
    channel_cells: Vec<(FieldId, NodeId, u32)>,
    /// The focused text field, and the scroll offsets of scrolling panels.
    field: Option<(FieldId, TextField)>,
    scroll: HashMap<PanelId, f32>,
    /// The selection's pin, and its history (`TrackableDataSelectionState`).
    selection_pinned: bool,
    selection_history: Vec<panels::Selection>,
    selection_index: usize,
    /// Milliseconds, from [`set_time`](Self::set_time), and when the caret
    /// last moved: it blinks from then, 500 ms on, 500 ms off.
    now: f64,
    caret_since: f64,
}

/// A layer item pressed: which layer, where, whether it has become a
/// drag (and a copy, Shift held), its click otherwise, what it is over, the
/// copy made.
#[derive(Debug, Clone, PartialEq)]
struct LayerDrag {
    id: ng_viewer::LayerId,
    start: (f32, f32),
    dragging: bool,
    copy: bool,
    click: Option<Action>,
    target: Option<Key>,
    copy_id: Option<ng_viewer::LayerId>,
}

impl NgUi {
    /// `registerLayerBarDropHandlers`' `update` on entering `target` (a
    /// layer, or the bar's end): the dragged layer (or its copy, made on
    /// the first) moves before it — after it, coming from before.
    fn layer_drag_over(&mut self, viewer: &mut Viewer, target: Option<usize>) -> Effect {
        let Some(drag) = self.layer_drag.clone() else { return Effect::None };
        let index_of = |viewer: &Viewer, id: ng_viewer::LayerId| viewer.layers().iter().position(|l| l.id == id);
        let moving = if drag.copy { drag.copy_id } else { Some(drag.id) };
        let target_id = target.and_then(|t| viewer.layers().get(t).map(|l| l.id));
        if target_id.is_some() && target_id == moving {
            return Effect::None;
        }
        match moving {
            None => {
                // The copy: inserted at the target.
                let Some(source) = index_of(viewer, drag.id) else { return Effect::None };
                let at = target.unwrap_or(viewer.layers().len());
                let pending = viewer.copy_layer(source, at);
                let copy = viewer.layers().get(at.min(viewer.layers().len() - 1)).map(|l| l.id);
                if let Some(d) = &mut self.layer_drag {
                    d.copy_id = copy;
                }
                if !pending.is_empty() {
                    return Effect::LoadSources(pending);
                }
            }
            Some(id) => {
                let Some(from) = index_of(viewer, id) else { return Effect::None };
                let to = match target_id.and_then(|t| index_of(viewer, t)) {
                    Some(t) => {
                        let without = if from < t { t - 1 } else { t };
                        if from <= without { without + 1 } else { without }
                    }
                    None => viewer.layers().len() - 1,
                };
                viewer.move_layer(from, to);
            }
        }
        Effect::None
    }

    /// The drag's status (`.neuroglancer-drag-status`): yellow, at the
    /// top, on the side away from the pointer.
    pub(crate) fn layer_drag_status(&self, viewer: &Viewer) -> Option<(String, bool)> {
        let drag = self.layer_drag.as_ref().filter(|d| d.dragging)?;
        let (x, _) = self.pointer?;
        let left = x >= self.ui.rect(self.ui.root()).width / 2.0;
        let _ = viewer;
        // Over the drop zone, what a drop does; else (over the layers too,
        // as Chrome reports it) what dragging does.
        let text = if drag.target == Some(Key::LayerPanel) {
            if drag.copy {
                "Drop to copy 1 layer (release SHIFT to move)".to_owned()
            } else {
                "Drop to move 1 layer (hold SHIFT to copy)".to_owned()
            }
        } else {
            "Drag layer to another layer bar/panel (including in another Neuroglancer window), or to the left/top/right/bottom edge of a layer group".to_owned()
        };
        Some((text, left))
    }
}

impl Default for NgUi {
    fn default() -> Self {
        Self::new()
    }
}

/// The text of the mouse position widget (`MousePositionWidget`).
fn mouse_position_text(names: &[String], p: &[f32]) -> String {
    names.iter().zip(p).map(|(n, v)| format!("{n} {}", number_to_string((*v as f64).floor()))).collect::<Vec<_>>().join("  ")
}

/// `formatPositionText`.
pub fn format_position_text(p: &[f32]) -> String {
    p.iter().map(|v| number_to_string((*v as f64).floor())).collect::<Vec<_>>().join(", ")
}

/// A coordinate as the position widget shows it.
fn coordinate_text(v: f32) -> String {
    number_to_string((v as f64).floor())
}

/// A typed coordinate (`updatePosition`); `None` if it is not a number.
fn typed_coordinate(dim: usize, text: &str) -> Option<Coordinate> {
    let value = js_number(text);
    value.is_finite().then(|| Coordinate {
        dim,
        value,
        // "If `valueString` contains a decimal point, don't adjust to voxel
        // center."
        to_voxel_center: value.fract() == 0.0 && !text.contains('.'),
    })
}

/// JavaScript's `Number(s)` for what a coordinate field can hold (`-`,
/// digits, `.`): the empty string is 0, anything else unparsable NaN.
fn js_number(s: &str) -> f64 {
    let t = s.trim();
    if t.is_empty() {
        return 0.0;
    }
    if t.ends_with('.') && !t.starts_with('.') && t.len() > 1 {
        return t[..t.len() - 1].parse().unwrap_or(f64::NAN);
    }
    t.parse().unwrap_or(f64::NAN)
}

impl NgUi {
    pub fn new() -> Self {
        Self {
            attribution: true,
            ui: Ui::new(),
            keys: HashMap::new(),
            actions: HashMap::new(),
            hovered: vec![],
            pointer: None,
            max_position_seen: HashMap::new(),
            max_value_len: HashMap::new(),
            show_layer_hover_values: true,
            show_all_plot_bounds: true,
            panels: vec![],
            layout: None,
            entered: None,
            focused: None,
            input_focused: false,
            hovered_segments: vec![],
            temporary_messages: vec![],
            annotating: None,
            scroll_to_annotation: None,
            annotation_scroll: None,
            refocus: false,
            edit: None,
            dropdown: None,
            plot: None,
            plot_hover: None,
            plot_drag: false,
            plot_dragged: false,
            pending: Default::default(),
            now: 0.0,
            caret_since: 0.0,
            gutter_drag: None,
            dialog: None,
            completion_active: None,
            starred: HashMap::new(),
            velocities: HashMap::new(),
            titles: HashMap::new(),
            hover_since: 0.0,
            dialog_lines: vec![],
            mono_advance: 8.0,
            palette_results_height: 448.0,
            controls: HashMap::new(),
            control_drag: None,
            menu: None,
            tool_refs: vec![],
            binding_tool: None,
            active_tool: None,
            active_ref: None,
            select_paint: None,
            layer_drag: None,
            layer_panel: PanelId::Layer,
            palette_list: false,
            palette_button: None,
            color_picker: None,
            picker_drag: None,
            shader_edit: None,
            shader_advance: 8.0,
            shader_lines: vec![],
            shader_blur_pending: false,
            legend_widths: HashMap::new(),
            legend_colors: HashMap::new(),
            viewport: (0.0, 0.0),
            dimension_texts: vec![],
            channel_cells: vec![],
            field: None,
            scroll: HashMap::new(),
            selection_pinned: true,
            selection_history: vec![],
            selection_index: 0,
        }
    }

    /// The laid-out tree, for inspection.
    pub fn tree(&self) -> &Ui {
        &self.ui
    }

    /// Every text run and its line box, in painting order: for comparing
    /// the layout with neuroglancer's DOM.
    pub fn text_rects(&self) -> Vec<(String, Rect)> {
        let mut out = vec![];
        let mut stack = vec![self.ui.root()];
        while let Some(n) = stack.pop() {
            if let threers::ui::NodeKind::Text(t) = self.ui.kind(n) {
                out.push((t.clone(), self.ui.rect(n)));
            }
            stack.extend(self.ui.paint_order(n).into_iter().rev());
        }
        out
    }

    /// Rebuilds and lays out the UI for `viewer` in a `width` × `height`
    /// (logical pixels) window, and confines the viewer's panels to the area
    /// below the bars. Returns whether that area changed.
    pub fn update(&mut self, viewer: &mut Viewer, width: f32, height: f32, text: &mut dyn TextMeasure) -> bool {
        // A segment list entry under the pointer is the hovered segment
        // (`segmentSelectionState.set`), highlighted in the panels.
        let hover = self.hovered.iter().find_map(|k| if let Key::SegmentEntry(layer, id) = k { Some((*layer, *id)) } else { None });
        viewer.set_segment_hover(hover);
        // An annotation list entry under the pointer is the hovered
        // annotation (`mouseenter`), highlighted in the panels; else the
        // one picked in a panel.
        let entry = self.hovered.iter().find_map(|k| if let Key::AnnotationEntry(layer, n) = k { Some((*layer, *n)) } else { None });
        let annotation = match entry {
            Some((layer, n)) => viewer.annotation_id(layer, n).map(|id| (layer, id)),
            None => viewer.picked_annotation(),
        };
        viewer.set_annotation_hover(annotation);
        self.annotation_preview(viewer);
        self.hovered_segments = viewer.hovered_segments();
        self.build(viewer, width, height, text)
    }

    /// Paints the UI as last updated.
    pub fn paint(&self, painter: &mut dyn Painter) {
        self.ui.paint(painter);
    }

    fn keys_at(&self, x: f32, y: f32) -> Vec<Key> {
        let Some(hit) = self.ui.hit_test(x, y) else { return vec![] };
        self.ui.ancestors(hit).filter_map(|n| self.keys.get(&n).copied()).collect()
    }

    fn panel_at(&self, x: f32, y: f32) -> Option<usize> {
        self.panels.iter().position(|&[px, py, w, h]| x >= px && y >= py && x < px + w && y < py + h)
    }

    /// The coordinate at a point on (or dragged beyond) the open plot.
    fn plot_coordinate(&self, y: f32) -> Option<(usize, f64)> {
        let p = self.plot.as_ref()?;
        let t = ((y - p.canvas.y) / p.canvas.height) as f64;
        Some((p.dim, plot::coordinate_at(&p.bounds, t)))
    }

    /// The pointer moved to `(x, y)`, logical pixels. Returns an action
    /// when the UI changed: hover styles, the plot, focus — or the position,
    /// while dragging on a plot.
    pub fn pointer_move(&mut self, x: f32, y: f32) -> Option<Action> {
        self.pointer = Some((x, y));
        if let Some(field) = self.picker_drag {
            return self.picker_set(field, x, y);
        }
        // A layer dragged (past Chrome's 4px): over another item it moves
        // there, over the bar's empty end to the end (`dragenter`).
        if let Some(drag) = &mut self.layer_drag {
            let (dx, dy) = (x - drag.start.0, y - drag.start.1);
            if !drag.dragging && dx.abs().max(dy.abs()) > 4.0 {
                drag.dragging = true;
            }
            if drag.dragging {
                let keys = self.keys_at(x, y);
                // The drop zone: the bar past its items and `+`.
                let over_add = keys.iter().any(|k| matches!(k, Key::Icon(view::ADD_LAYER_ICON)));
                let target = keys.iter().copied().find(|k| matches!(k, Key::LayerItem(_))).or_else(|| keys.iter().copied().find(|k| matches!(k, Key::LayerPanel)).filter(|_| !over_add));
                if let Some(drag) = &mut self.layer_drag {
                    if target.is_some() && target != drag.target {
                        drag.target = target;
                        let to = match target {
                            Some(Key::LayerItem(i)) => Some(i),
                            _ => None,
                        };
                        return Some(Action::LayerDragOver(to));
                    }
                    if target.is_none() {
                        drag.target = None;
                    }
                }
                return Some(Action::Repaint);
            }
        }
        // The select tool paints once the pointer has moved 5px.
        if let Some(p) = &mut self.select_paint {
            let (dx, dy) = (x - p.start.0, y - p.start.1);
            if !p.painting && dx * dx + dy * dy > 25.0 {
                p.painting = true;
            }
        }
        if let Some(drag) = &self.gutter_drag {
            return Some(match drag {
                GutterDrag::Column { side, x0, size0, panels } => {
                    // `size -= outwards · delta`, at least the minimum.
                    let min = panels.iter().map(|(_, l)| l.min_size).fold(0.0, f64::max);
                    let at = if side.horizontal() { y } else { x };
                    let size = (size0 - (side.outwards() * (at - x0)) as f64).round().max(min);
                    Action::SetLocations(panels.iter().map(|(id, l)| (*id, Location { size, ..l.clone() })).collect())
                }
                GutterDrag::Split { first, second, top, bottom, across } => {
                    let at = if *across { x } else { y };
                    let fraction = ((at - top) / (bottom - top)).clamp(0.1, 0.9) as f64;
                    let sum = first.1.flex + second.1.flex;
                    let round = |v: f64| ng_core::js::round(v * 100.0) / 100.0;
                    Action::SetLocations(vec![
                        (first.0, Location { flex: round(fraction * sum), ..first.1.clone() }),
                        (second.0, Location { flex: round((1.0 - fraction) * sum), ..second.1.clone() }),
                    ])
                }
            });
        }
        if let Some((key, endpoint)) = self.control_drag {
            let (Some(node), Some(control)) = (self.node_of(key), self.controls.get(&key)) else { return None };
            return control.drag(self.ui.rect(node), x, endpoint);
        }
        let keys = self.keys_at(x, y);
        let mut changed = keys != self.hovered;
        if changed {
            // A tooltip shown goes; the next waits for the pointer to rest.
            changed |= self.tooltip().is_some();
            self.hover_since = self.now;
        }
        self.hovered = keys;
        // The dropdown is over the panels, not in them.
        let inside = if self.hovered.contains(&Key::Dropdown) { None } else { self.panel_at(x, y) };
        if inside != self.entered {
            self.entered = inside;
            if inside.is_some() && inside != self.focused && !self.input_focused {
                self.focused = inside;
                changed = true;
            }
        }
        if self.plot_drag {
            // `startRelativeMouseDrag`: the plot follows the pointer anywhere.
            self.plot_hover = None;
            self.plot_dragged = true;
            return self.plot_coordinate(y).map(|(dim, c)| Action::SetCoordinates(vec![Coordinate { dim, value: c, to_voxel_center: true }]));
        }
        let hover = if self.hovered.contains(&Key::Plot) { self.plot_coordinate(y).map(|(_, c)| c) } else { None };
        if hover != self.plot_hover {
            self.plot_hover = hover;
            changed = true;
        }
        changed.then_some(Action::Repaint)
    }

    pub fn pointer_leave(&mut self) -> bool {
        self.pointer = None;
        self.entered = None;
        self.plot_hover = None;
        !std::mem::take(&mut self.hovered).is_empty()
    }

    /// Whether `(x, y)` is over part of the UI rather than a bare panel: the
    /// viewer should then see the pointer leave (neuroglancer's panels track
    /// the mouse only when they are the event's target).
    pub fn covers(&self, x: f32, y: f32) -> bool {
        self.dialog.is_some() || !self.keys_at(x, y).is_empty()
    }

    /// Whether `(x, y)` is over a control, which takes presses and the wheel.
    pub fn captures(&self, x: f32, y: f32) -> bool {
        // A dialog's shade covers the viewer.
        if self.dialog.is_some() {
            return true;
        }
        self.keys_at(x, y).iter().any(|k| {
            self.actions.contains_key(k)
                || matches!(k, Key::Dimension(_) | Key::Dropdown | Key::LayerPanel | Key::Panel(_) | Key::Gutter(..) | Key::FlexGutter(..) | Key::Field(_))
        })
    }

    /// The node a key marks, as last laid out.
    fn node_of(&self, key: Key) -> Option<NodeId> {
        self.keys.iter().find(|(_, k)| **k == key).map(|(n, _)| *n)
    }

    /// A click on an invlerp row (`<label>`) off its inputs and buttons:
    /// the lower range bound focused, its text selected. Whether it was.
    fn label_click(&mut self, keys: &[Key]) -> bool {
        let Some(i) = keys.iter().find_map(|k| if let Key::Control(i, "invlerpRow") = k { Some(*i) } else { None }) else { return false };
        let interactive = keys.iter().any(|k| matches!(k, Key::Field(_) | Key::Control(_, "minMax" | "p1" | "p5")));
        if interactive {
            return false;
        }
        let Some(c) = self.controls.get(&Key::Control(i, "cdf")) else { return false };
        self.field = Some((FieldId::Bound(i, false, 0), TextField::selected(c.bound_text(false, 0))));
        self.caret_since = self.now;
        true
    }

    /// `change`: a focused field's text committed (on Enter or blur).
    fn commit_field(&mut self) -> Option<Action> {
        let (id, field) = self.field.as_ref()?;
        Some(match id {
            FieldId::Bound(i, window, endpoint) => {
                let control = self.controls.get(&Key::Control(*i, "cdf"))?;
                control.set_bound(*window, *endpoint, field.text.trim()).unwrap_or(Action::Repaint)
            }
            FieldId::Slider(i, key) => slider_action(self.controls.get(&Key::Control(*i, key)), &field.text).unwrap_or(Action::Repaint),
            // `parseUint32` of the seed; anything else leaves it.
            FieldId::SegmentQuery(i) => Action::SetLayerOption(*i, "segmentQuery".into(), (!field.text.is_empty()).then(|| field.text.clone().into())),
            FieldId::SourceUrl(i) => Action::SetLayerSource(*i, field.text.trim().to_owned()),
            // `NumberInputWidget`: a number, else left as it was.
            FieldId::Velocity(dim) => match self.velocities.get(dim) {
                Some(v) => {
                    let n = ng_core::js::parse_float(field.text.trim());
                    if n.is_finite() {
                        Action::SetVelocity(*dim, Some(ng_viewer::playback::DimensionVelocity { velocity: n, ..*v }))
                    } else {
                        Action::Repaint
                    }
                }
                None => Action::Repaint,
            },
            FieldId::ScreenshotName => {
                let text = field.text.clone();
                if let Some(dialogs::Dialog::Screenshot(d)) = &mut self.dialog {
                    d.name = text;
                }
                Action::Repaint
            }
            FieldId::ColorSeed(i) => match field.text.trim().parse::<u32>() {
                Ok(seed) => Action::SetLayerOption(*i & rendering::LAYER_MASK, "colorSeed".into(), Some(seed.into())),
                Err(_) => Action::Repaint,
            },
            FieldId::Title => Action::SetTitle(field.text.clone()),
            FieldId::PaletteName(k) => Action::RenamePalette(*k, field.text.clone()),
            FieldId::ShaderChannel(i, name, dim) => Action::SetShaderChannel(*i, name, *dim, field.text.clone()),
            // `NumberInputWidget` over the slider: a number in range.
            FieldId::ShaderSlider(i, name) => match self.controls.values().find_map(|c| match c {
                rendering::Control::ShaderSlider { layer, name: n, min, max, default, .. } if layer == i && n == name => Some((*min, *max, *default)),
                _ => None,
            }) {
                Some((min, max, default)) => {
                    let v = ng_core::js::parse_float(field.text.trim());
                    if v.is_finite() && v >= min && v <= max {
                        Action::SetShaderControl(*i, name, shader_ui::slider_json(v, default))
                    } else {
                        Action::Repaint
                    }
                }
                None => Action::Repaint,
            },
            // `updateNames`, `updateScales`: all the dimensions at once.
            FieldId::DimensionName(d) => {
                let mut names: Vec<String> = self.dimension_texts.iter().map(|t| t.0.clone()).collect();
                if let Some(n) = names.get_mut(*d) {
                    *n = field.text.clone();
                }
                Action::RenameDimensions(names)
            }
            FieldId::DimensionScale(d) => {
                let mut scales: Vec<String> = self.dimension_texts.iter().map(|t| t.1.clone()).collect();
                if let Some(n) = scales.get_mut(*d) {
                    *n = field.text.clone();
                }
                Action::SetDimensionScales(scales)
            }
            FieldId::PaletteQuery(k) => Action::SetPaletteQuery(*k, field.text.clone()),
            FieldId::LayerName(i) => Action::RenameLayer(*i, field.text.clone()),
            // `parseFloat`; a number that is not valid leaves the value.
            FieldId::Number(n) => {
                let v = ng_core::js::parse_float(field.text.trim());
                if v.is_nan() {
                    Action::Repaint
                } else {
                    Action::SetNumber(*n, v)
                }
            }
        })
    }

    /// A press on a data panel that the UI acts on: `select-position` and
    /// `unpin-selected-position` (`getDefaultSelectBindings`). Call before
    /// handing the press to the viewer; `Some` means the UI took it.
    pub fn data_panel_press(&mut self, button: Button, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        // The active tool's bindings first (they are the panels' parents,
        // at the highest priority).
        if let Some(a) = self.tool_press(button, modifiers, viewer) {
            return Some(a);
        }
        let key = format!("mousedown{}", button as u8);
        match ng_viewer::input::lookup(ng_viewer::input::PanelKind::Slice, &key, modifiers, viewer.mac) {
            Some("select-position") => Some(Action::SelectPosition { pin: true }),
            Some("annotate") => Some(Action::Annotate),
            Some("unpin-selected-position") => Some(Action::SelectPosition { pin: false }),
            _ => None,
        }
    }

    /// A double-click on a data panel (`dblclick0`): `select` or `star`
    /// the segments under the pointer. Call when the UI's own
    /// [`NgUi::double_click`] did not take it and the pointer is over a
    /// panel.
    pub fn data_panel_double_click(&mut self, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        match ng_viewer::input::lookup(ng_viewer::input::PanelKind::Slice, "dblclick0", modifiers, viewer.mac) {
            Some("select") => Some(Action::ToggleHoveredSegments { star: false }),
            Some("star") => Some(Action::ToggleHoveredSegments { star: true }),
            _ => None,
        }
    }

    /// A global shortcut (`getDefaultGlobalBindings`) for a key not taken
    /// by a field or a panel: `code` is the DOM `KeyboardEvent.code`.
    pub fn shortcut(&mut self, code: &str, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        // The active dimension tool's `toggle-playback` (`shift?+alt?+space`).
        if code == "Space" && !modifiers.control && !modifiers.meta {
            if let Some(dim) = self.active_dimension_tool(viewer) {
                return Some(Action::TogglePlayback(dim));
            }
        }
        let key = code.to_ascii_lowercase();
        // A focused data panel's annotation steps (Enter, Backspace).
        if self.focused.is_some() {
            match ng_viewer::input::lookup(ng_viewer::input::PanelKind::Slice, &key, modifiers, viewer.mac) {
                Some("finish-annotation") => return Some(Action::AnnotationStep { finish: true }),
                Some("undo-annotation-step") => return Some(Action::AnnotationStep { finish: false }),
                _ => {}
            }
        }
        ng_viewer::input::lookup_global(&key, modifiers, viewer.mac).map(|a| Action::Shortcut(a.to_owned()))
    }

    /// A screenshot finished (saved by the shell): its dialog closes
    /// (`screenshotFinished`).
    pub fn screenshot_finished(&mut self) {
        if matches!(self.dialog, Some(dialogs::Dialog::Screenshot(_))) {
            self.dialog = None;
            self.field = None;
        }
    }

    /// The time now, in milliseconds (`performance.now()`, say): the caret
    /// blinks by it. Call before handling input and before updating.
    pub fn set_time(&mut self, ms: f64) {
        self.now = ms;
    }

    /// Whether the caret shows now.
    fn caret_on(&self) -> bool {
        ((self.now - self.caret_since) / 500.0).floor() as i64 % 2 == 0
    }

    /// Milliseconds until the caret blinks, while one shows: when to paint
    /// again.
    pub fn next_blink(&self) -> Option<f64> {
        // A caret (in a coordinate, a field, the state editor, the palette).
        let caret = self.edit.as_ref().is_some_and(|e| e.start == e.end)
            || self.field.as_ref().is_some_and(|(_, f)| f.anchor == f.caret)
            || matches!(&self.dialog, Some(dialogs::Dialog::StateEditor(e)) if e.focused)
            || matches!(&self.dialog, Some(dialogs::Dialog::Palette(_)));
        let blink = caret.then(|| 500.0 - (self.now - self.caret_since).rem_euclid(500.0));
        // A tooltip about to show.
        let tip = (self.pointer.is_some() && self.tooltip().is_none() && self.hovered.iter().any(|k| self.titles.contains_key(k)))
            .then(|| (TOOLTIP_DELAY - (self.now - self.hover_since)).max(1.0));
        // A temporary message going.
        let message = self.temporary_messages.iter().map(|(_, until)| (until - self.now).max(1.0)).reduce(f64::min);
        [blink, tip, message].into_iter().flatten().reduce(f64::min)
    }

    /// The tooltip showing: the innermost hovered control's title, once
    /// the pointer has rested on it.
    fn tooltip(&self) -> Option<&str> {
        if self.pointer.is_none() || self.now - self.hover_since < TOOLTIP_DELAY || self.dialog.is_some() {
            return None;
        }
        self.hovered.iter().find_map(|k| self.titles.get(k)).map(String::as_str)
    }

    /// Whether the UI holds the keyboard (a field is being edited).
    pub fn has_keyboard_focus(&self) -> bool {
        self.dialog.is_some() || self.edit.is_some() || self.field.is_some() || self.binding_tool.is_some() || self.color_picker.is_some() || self.shader_focused()
    }

    /// For shells without a system colour picker: the UI's own, for
    /// `setting`, starting at `current` ([`Effect::PickColor`]).
    pub fn open_color_picker(&mut self, setting: ColorSetting, current: [f32; 3]) {
        self.color_picker = Some((setting, threers::ui::controls::rgb_to_hsv(current)));
        self.picker_drag = None;
    }

    /// The colour picker's part under `(x, y)` set from there: its colour.
    fn picker_set(&mut self, field: bool, x: f32, y: f32) -> Option<Action> {
        let (setting, mut hsv) = self.color_picker?;
        let key = if field { Key::ColorPickerField } else { Key::ColorPickerHue };
        let r = self.ui.rect(self.node_of(key)?);
        if field {
            let (s, v) = threers::ui::controls::color_picker_field(r, x, y);
            (hsv[1], hsv[2]) = (s, v);
        } else {
            hsv[0] = threers::ui::controls::color_picker_hue(r, x);
        }
        self.color_picker = Some((setting, hsv));
        Some(Action::SetColor(setting, threers::ui::controls::hsv_to_rgb(hsv)))
    }

    /// The pointer's shape at `(x, y)`.
    pub fn cursor(&self, x: f32, y: f32) -> Cursor {
        let keys = self.keys_at(x, y);
        match keys.first() {
            Some(Key::Icon(_) | Key::LayerItem(_)) => Cursor::Pointer,
            Some(Key::Coordinate(_)) if self.edit.is_some() => Cursor::Text,
            Some(Key::Plot) => Cursor::Default,
            Some(_) => {
                if keys.iter().any(|k| matches!(k, Key::Icon(_) | Key::LayerItem(_))) {
                    Cursor::Pointer
                } else {
                    Cursor::Default
                }
            }
            None if self.panel_at(x, y).is_some() => Cursor::Crosshair,
            None => Cursor::Default,
        }
    }

    /// Ends editing: commits a modified coordinate (as the field's `blur`
    /// does) and closes the dropdown.
    fn blur(&mut self) -> Option<Action> {
        self.dropdown = None;
        self.plot = None;
        self.plot_hover = None;
        let commit = self.commit();
        self.edit = None;
        Some(commit)
    }

    /// `updatePosition`: every coordinate typed into, as typed (those that
    /// are not numbers are dropped).
    fn commit(&mut self) -> Action {
        let mut typed: Vec<(usize, String)> = std::mem::take(&mut self.pending).into_iter().collect();
        if let Some(e) = self.edit.as_mut().filter(|e| e.modified) {
            e.modified = false;
            typed.push((e.dim, e.text.clone()));
        }
        let coordinates: Vec<Coordinate> = typed.iter().filter_map(|(dim, text)| typed_coordinate(*dim, text)).collect();
        if coordinates.is_empty() {
            Action::Repaint
        } else {
            Action::SetCoordinates(coordinates)
        }
    }

    /// Focuses dimension `dim`'s coordinate with all of it selected
    /// (`coordinate.focus(); coordinate.select()`), opening its dropdown.
    fn focus_coordinate(&mut self, dim: usize, text: String) {
        // A field typed into keeps its text (and stays to be committed).
        let (text, modified) = match self.pending.remove(&dim) {
            Some(t) => (t, true),
            None => (text, false),
        };
        let n = text.chars().count();
        self.caret_since = self.now;
        self.edit = Some(Edit { dim, text, start: 0, end: n, modified });
        self.dropdown = Some(dim);
    }

    /// A button press at `(x, y)`; `None` leaves it to the viewer.
    pub fn pointer_down(&mut self, x: f32, y: f32, button: Button, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        // A press outside the data panels takes the focus from the one
        // that had it (`AutomaticallyFocusedElement` focuses again only on
        // the next `mouseenter`).
        let keys = self.keys_at(x, y);
        let blurred = self.focused.is_some() && (self.panel_at(x, y).is_none() || keys.contains(&Key::Dropdown));
        if blurred {
            self.focused = None;
        }
        self.input_focused = keys.iter().any(|k| matches!(k, Key::Field(_)) || matches!(self.controls.get(k), Some(rendering::Control::Slider { .. } | rendering::Control::ShaderSlider { .. })));
        // A press in a panel focuses it (`pointerdown`).
        if let Some(panel) = self.panel_at(x, y).filter(|_| !keys.contains(&Key::Dropdown) && self.focused.is_none()) {
            self.focused = Some(panel);
        }
        self.press(x, y, button, modifiers, viewer).or(blurred.then_some(Action::Repaint))
    }

    fn press(&mut self, x: f32, y: f32, button: Button, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        let keys = self.keys_at(x, y);
        // A segment entry's plain right-click (`onMousedown`): to the segment.
        let plain = !modifiers.control && !modifiers.alt && !modifiers.meta && !modifiers.shift;
        if button == Button::Secondary && plain && self.dialog.is_none() {
            if let Some((layer, id)) = keys.iter().find_map(|k| if let Key::SegmentEntry(l, id) = k { Some((*l, *id)) } else { None }) {
                return Some(Action::MoveToSegment(layer, id));
            }
            // An annotation layer's entry (`mousedown2`: move-to-annotation).
            if let Some((layer, n)) = keys.iter().find_map(|k| if let Key::AnnotationEntry(l, n) = k { Some((*l, *n)) } else { None }) {
                if viewer.layers().get(layer).is_some_and(|l| l.spec.layer_type == "annotation") {
                    return Some(Action::MoveToAnnotation(layer, n));
                }
            }
        }
        // The colour picker: its field and hue strip drag; a press outside
        // closes it.
        if self.color_picker.is_some() {
            for (key, field) in [(Key::ColorPickerField, true), (Key::ColorPickerHue, false)] {
                if keys.contains(&key) && button == Button::Primary {
                    self.picker_drag = Some(field);
                    return self.picker_set(field, x, y);
                }
            }
            if !keys.contains(&Key::ColorPicker) {
                self.color_picker = None;
            }
            return Some(Action::Repaint);
        }
        if self.dialog.is_some() {
            return if button == Button::Primary { self.dialog_press(viewer, &keys, x, y, modifiers.shift) } else { Some(Action::Repaint) };
        }
        // The shader editor: a press on it places the caret; elsewhere, it
        // loses the keyboard (what was typed is set).
        if button == Button::Primary && keys.iter().any(|k| matches!(k, Key::ShaderEditor(_) | Key::ShaderLine(..))) {
            if let Some(a) = self.shader_press(viewer, &keys, x, y, modifiers.shift) {
                return Some(a);
            }
        } else if self.shader_focused() {
            self.shader_blur_pending = true;
        }
        // The wrench's list closes on a press outside it and the wrench
        // (`focusout`).
        if self.palette_list && !keys.iter().any(|k| matches!(k, Key::PaletteDropdown | Key::Icon(0))) {
            self.palette_list = false;
        }
        // A tool's binding widget held: the next letter binds the tool.
        if button == Button::Primary {
            if let Some(tool) = keys.iter().find_map(|k| match k {
                Key::ToolBinding(n) | Key::ToolStatus(n) => self.tool_refs.get(*n as usize),
                _ => None,
            }) {
                self.binding_tool = Some(tool.clone());
                return Some(Action::Repaint);
            }
        }

        // A focused field loses focus to a press elsewhere (`change`).
        let mut committed = None;
        if let Some((id, _)) = &self.field {
            if !keys.contains(&Key::Field(*id)) {
                committed = self.commit_field();
                self.field = None;
            }
        }
        // A completion chosen (inside its field, which keeps the focus).
        if let Some(a) = keys.iter().find_map(|k| if matches!(k, Key::Completion(_)) { self.actions.get(k).cloned() } else { None }) {
            return Some(a);
        }
        // An open `<select>`: an option chosen, or the menu closed.
        if let Some(menu) = self.menu.take() {
            if let Some(k) = keys.iter().find_map(|k| if let Key::MenuItem(k) = k { Some(*k) } else { None }) {
                if self.controls.get(&menu).and_then(rendering::Control::options).is_some_and(|(options, _)| k < options.len()) {
                    return self.controls[&menu].choose(k).or(Some(Action::Repaint));
                }
            }
            if keys.contains(&menu) {
                return Some(Action::Repaint);
            }
        }
        // Layer controls: a `<select>` opens; the others take the press
        // (and maybe a drag).
        if button == Button::Primary {
            if let Some(key) = keys.iter().copied().find(|k| matches!(k, Key::Control(..)) && self.controls.contains_key(k)) {
                if self.controls.get(&key).and_then(rendering::Control::options).is_some() {
                    self.menu = Some(key);
                    return Some(Action::Repaint);
                }
                let r = self.ui.rect(self.node_of(key)?);
                if let Some((action, drag)) = self.controls[&key].press(r, x) {
                    self.control_drag = drag.map(|e| (key, e));
                    self.label_click(&keys);
                    return Some(action);
                }
            }
            // A click elsewhere on the invlerp control's `<label>` focuses its
            // first input, the range's lower bound, text selected.
            if self.label_click(&keys) {
                return Some(Action::Repaint);
            }
        }
        let early = self.panel_press(&keys, x, y, button, modifiers, viewer);
        match (committed, early) {
            (c, Some(a)) => Some(match c {
                Some(c) => Action::Many(vec![c, a]),
                None => a,
            }),
            (Some(c), None) if keys.iter().any(|k| matches!(k, Key::Panel(_))) => Some(c),
            (c, None) => {
                let rest = self.pointer_down_bars(&keys, x, y, button, modifiers, viewer);
                match (c, rest) {
                    (Some(c), Some(r)) => Some(Action::Many(vec![c, r])),
                    (c, r) => c.or(r),
                }
            }
        }
    }

    /// A press on the side panels: gutters, fields, colour inputs; `None`
    /// for the rest.
    fn panel_press(&mut self, keys: &[Key], x: f32, y: f32, button: Button, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        // A Layers panel row: right-click or Ctrl/⌘+click toggles the
        // layer's panel, Alt+click its picking (`LayerListItem`).
        if let Some(i) = keys.iter().find_map(|k| if let Key::ListItem(i) = k { Some(*i) } else { None }) {
            if button == Button::Secondary || (button == Button::Primary && (modifiers.control || modifiers.meta)) {
                return Some(Action::ToggleLayerSidePanel(i));
            }
            if button == Button::Primary && modifiers.alt {
                return Some(Action::TogglePick(i));
            }
        }
        if button != Button::Primary {
            return keys.iter().any(|k| matches!(k, Key::Panel(_))).then_some(Action::Repaint);
        }
        for k in keys {
            match *k {
                Key::Gutter(side, gi) => {
                    let group = side::groups(viewer, side).into_iter().nth(gi)?;
                    let at = if side.horizontal() { y } else { x };
                    self.gutter_drag = Some(GutterDrag::Column { side, x0: at, size0: group.size, panels: group.panels });
                    return Some(Action::Repaint);
                }
                Key::FlexGutter(side, gi, index) => {
                    let group = side::groups(viewer, side).into_iter().nth(gi)?;
                    let (first, second) = (group.panels.get(index)?.clone(), group.panels.get(index + 1)?.clone());
                    let a = self.ui.rect(self.node_of(Key::Panel(first.0))?);
                    let r = self.ui.rect(self.node_of(Key::Panel(second.0))?);
                    let across = side.horizontal();
                    let (top, bottom) = if across { (a.x, r.x + r.width) } else { (a.y, r.y + r.height) };
                    self.gutter_drag = Some(GutterDrag::Split { first, second, top, bottom, across });
                    let _ = y;
                    return Some(Action::Repaint);
                }
                Key::Field(id) => {
                    // Focus: the caret at the end (a click past the text).
                    if self.field.as_ref().is_none_or(|(f, _)| *f != id) {
                        let text = match id {
                            FieldId::Title => viewer.settings.title.clone().unwrap_or_default(),
                            FieldId::LayerName(i) => viewer.layers().get(i).map_or(String::new(), |l| l.name().to_owned()),
                            FieldId::Number(n) => number_to_string(number_setting(viewer, n)),
                            FieldId::Bound(i, window, endpoint) => match self.controls.get(&Key::Control(i, "cdf")) {
                                Some(c) => c.bound_text(window, endpoint),
                                None => String::new(),
                            },
                            FieldId::Slider(i, key) => match self.controls.get(&Key::Control(i, key)) {
                                Some(rendering::Control::Slider { value, .. }) => number_to_string(*value),
                                _ => String::new(),
                            },
                            FieldId::DimensionName(d) => self.dimension_texts.get(d).map_or(String::new(), |t| t.0.clone()),
                            FieldId::DimensionScale(d) => self.dimension_texts.get(d).map_or(String::new(), |t| t.1.clone()),
                            FieldId::ShaderChannel(i, name, dim) => {
                                let l = &viewer.layers()[i];
                                let t = l.data_type().unwrap_or(ng_core::DataType::Uint8);
                                let shader = l.spec.json.get("shader").and_then(serde_json::Value::as_str).unwrap_or(ng_viewer::state::DEFAULT_IMAGE_SHADER);
                                let parsed = ng_viewer::user_shader::parse_with_channels(shader, t, l.channel_extents().len());
                                let values = ng_viewer::user_shader::control_values(&parsed, t, l.spec.json.get("shaderControls"));
                                parsed
                                    .controls
                                    .iter()
                                    .position(|(n, _)| n == name)
                                    .and_then(|k| match &values[k] {
                                        ng_viewer::user_shader::ControlValue::Invlerp { channel, .. } => channel.get(dim).map(u32::to_string),
                                        _ => None,
                                    })
                                    .unwrap_or_default()
                            }
                            FieldId::ShaderSlider(i, name) => viewer
                                .layers()
                                .get(i)
                                .and_then(|l| l.spec.json.get("shaderControls")?.get(name)?.as_f64())
                                .or_else(|| self.controls.values().find_map(|c| match c {
                                    rendering::Control::ShaderSlider { layer, name: n, value, .. } if *layer == i && *n == name => Some(*value),
                                    _ => None,
                                }))
                                .map_or(String::new(), number_to_string),
                            FieldId::PaletteName(k) => palettes::entry(viewer, k as usize).map_or(String::new(), |e| e.0),
                            FieldId::PaletteQuery(k) => palettes::entry(viewer, k as usize).and_then(|e| e.1.get("query").and_then(serde_json::Value::as_str).map(str::to_owned)).unwrap_or_default(),
                            FieldId::Velocity(d) => self.velocities.get(&d).map_or(String::new(), |v| number_to_string(v.velocity)),
                            FieldId::SourceUrl(i) => viewer.layers().get(i).and_then(|l| l.spec.sources.first()).map_or(String::new(), |s| s.url.clone()),
                            FieldId::ScreenshotName => match &self.dialog {
                                Some(dialogs::Dialog::Screenshot(d)) => d.name.clone(),
                                _ => String::new(),
                            },
                            FieldId::SegmentQuery(i) => viewer.layers().get(i).and_then(|l| l.spec.json.get("segmentQuery")?.as_str().map(str::to_owned)).unwrap_or_default(),
                            FieldId::ColorSeed(i) => viewer
                                .layers()
                                .get(i & rendering::LAYER_MASK)
                                .and_then(|l| match l.kind() {
                                    Some(ng_viewer::LayerKind::Segmentation(d)) => Some(d.color_seed.to_string()),
                                    _ => None,
                                })
                                .unwrap_or_default(),
                        };
                        // The invlerp bounds select their text on focus.
                        let field = if matches!(id, FieldId::Bound(..)) { TextField::selected(text) } else { TextField::new(text) };
                        self.field = Some((id, field));
                    }
                    self.caret_since = self.now;
                    // A coordinate loses focus (`blur` commits it); its
                    // dropdown, holding this field, stays.
                    if self.edit.is_some() {
                        let commit = self.commit();
                        self.edit = None;
                        return Some(commit);
                    }
                    return Some(Action::Repaint);
                }
                _ => {}
            }
        }
        None
    }

    fn pointer_down_bars(&mut self, keys: &[Key], x: f32, y: f32, button: Button, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        let _ = x;
        let keys = keys.to_vec();
        // A press outside the open dimension closes its dropdown and blurs
        // its field (the document's capturing `pointerdown`).
        // (The dropdown is the open dimension's.)
        let in_dimension = |dim: usize| keys.contains(&Key::Dimension(dim)) || keys.contains(&Key::Dropdown);
        let mut result = None;
        if let Some(dim) = self.dropdown.or(self.edit.as_ref().map(|e| e.dim)) {
            if !in_dimension(dim) {
                result = self.blur();
            }
        }
        if keys.contains(&Key::Plot) && button == Button::Primary {
            if modifiers.control || modifiers.alt || modifiers.meta {
                return result.or(Some(Action::Repaint));
            }
            // The hover line stays until the pointer moves (a drag's move
            // clears it).
            self.plot_drag = true;
            return self.plot_coordinate(y).map(|(dim, c)| Action::SetCoordinates(vec![Coordinate { dim, value: c, to_voxel_center: true }]));
        }
        if keys.contains(&Key::Dropdown) {
            let action = if button == Button::Primary { keys.iter().find_map(|k| self.actions.get(k).cloned()) } else { None };
            return Some(action.or(result).unwrap_or(Action::Repaint));
        }
        // A press in a dimension (outside its inputs and dropdown) focuses
        // its coordinate.
        if let Some(dim) = keys.iter().find_map(|k| if let Key::Dimension(d) = k { Some(*d) } else { None }) {
            if button == Button::Primary && !keys.contains(&Key::Dropdown) {
                if self.edit.as_ref().is_some_and(|e| e.dim == dim) {
                    // Inside the focused field: a click moves the caret to
                    // the end (as a click past the text does).
                    if let Some(e) = &mut self.edit {
                        let n = e.text.chars().count();
                        (e.start, e.end) = (n, n);
                    }
                } else {
                    let space = viewer.dimensions();
                    let text = if viewer.navigation().valid() && dim < space.rank() {
                        coordinate_text(viewer.navigation().position()[dim])
                    } else {
                        String::new()
                    };
                    self.focus_coordinate(dim, text);
                }
                return Some(result.unwrap_or(Action::Repaint));
            }
            return result.or(Some(Action::Repaint));
        }
        let item = keys.iter().find_map(|k| if let Key::LayerItem(i) = k { Some(*i) } else { None });
        if let (Some(i), Button::Secondary) = (item, button) {
            // `contextmenu`: select the layer and show its panel.
            return Some(Action::SelectLayer(i));
        }
        if button != Button::Primary {
            return result;
        }
        let action = keys.iter().find_map(|k| self.actions.get(k).cloned());
        let action = match (action, item) {
            (Some(Action::ToggleLayerVisible(i)), _) if modifiers.control || modifiers.meta => Some(Action::ToggleLayerSidePanel(i)),
            (Some(Action::ToggleLayerVisible(i)), _) if modifiers.alt => Some(Action::TogglePick(i)),
            // ⌘/Ctrl+click on the wrench makes a palette.
            (Some(Action::TogglePaletteList), _) if modifiers.control || modifiers.meta => Some(Action::AddPalette { name: "Palette".into(), query: String::new() }),
            (a, _) => a,
        };
        // A layer item's click waits for the release: a press that moves
        // drags the layer instead (`draggable`; Shift copies it).
        if let (Some(i), Some(a @ (Action::ToggleLayerVisible(_) | Action::ToggleLayerSidePanel(_) | Action::TogglePick(_)))) = (item, &action) {
            if let Some(layer) = viewer.layers().get(i) {
                self.layer_drag = Some(LayerDrag { id: layer.id, start: (x, y), dragging: false, copy: modifiers.shift, click: Some(a.clone()), target: None, copy_id: None });
                return result.or(Some(Action::Repaint));
            }
        }
        action.or(result)
    }

    /// The release's action, before [`pointer_up`](Self::pointer_up): a
    /// layer item's click when the press did not drag.
    pub fn pointer_up_action(&mut self) -> Option<Action> {
        let drag = self.layer_drag.take()?;
        if drag.dragging {
            Some(Action::Repaint)
        } else {
            drag.click
        }
    }

    /// A button released: ends a drag along a plot or a gutter.
    pub fn pointer_up(&mut self) -> bool {
        self.plot_dragged = false;
        if self.picker_drag.take().is_some() {
            return true;
        }
        // The select tool's drag ends (back to idle).
        if self.select_paint.take().is_some() {
            return true;
        }
        // `addToolKeyBindHandlers`: binding lasts while the button is held.
        if self.binding_tool.take().is_some() {
            return true;
        }
        let gutter = self.gutter_drag.take().is_some() | self.control_drag.take().is_some();
        std::mem::take(&mut self.plot_drag) | gutter
    }

    /// Double-click at `(x, y)`: a dimension's name or scale becomes
    /// editable (not ported yet).
    pub fn double_click(&mut self, x: f32, y: f32) -> Option<Action> {
        let keys = self.keys_at(x, y);
        // A tool's binding widget, or the tool in the top bar: unbound; a
        // tool in an explicit palette (the event reaching its container
        // too): removed.
        let unbind = keys.iter().find_map(|k| match k {
            Key::ToolBinding(n) | Key::ToolStatus(n) => self.tool_refs.get(*n as usize),
            _ => None,
        });
        let unbind = unbind.map(|tool| Action::UnbindTool(tool.owner, tool.json.clone()));
        let remove = keys.iter().find_map(|k| if let Key::PaletteTool(p, i) = k { Some((*p, *i as usize)) } else { None }).map(|(p, i)| Action::RemovePaletteTool(p, i));
        // The selected layer's legacy tool in the top bar: unset.
        if let Some(i) = keys.iter().find_map(|k| if let Key::Control(i, "legacyTool") = k { Some(*i) } else { None }) {
            return Some(Action::SetLayerOption(i, "tool".into(), None));
        }
        match (unbind, remove) {
            (Some(a), Some(b)) => return Some(Action::Many(vec![a, b])),
            (Some(a), None) | (None, Some(a)) => return Some(a),
            (None, None) => {}
        }
        if let Some(a) = keys.iter().find_map(|k| self.controls.get(k)?.reset()) {
            return Some(a);
        }
        // A dimension's name or scale becomes editable, its text selected;
        // the coordinate loses the focus (and commits), the dropdown stays.
        let name = keys.iter().find_map(|k| if let Key::DimensionName(d) = k { Some(*d) } else { None });
        let scale = keys.iter().find_map(|k| if let Key::DimensionScale(d) = k { Some(*d) } else { None });
        if let Some((id, dim)) = name.map(|d| (FieldId::DimensionName(d), d)).or(scale.map(|d| (FieldId::DimensionScale(d), d))) {
            let commit = self.edit.is_some().then(|| self.commit());
            self.edit = None;
            self.dropdown = Some(dim);
            let text = match id {
                FieldId::DimensionName(d) => self.dimension_texts.get(d).map(|t| t.0.clone()),
                _ => self.dimension_texts.get(dim).map(|t| t.1.clone()),
            };
            self.field = Some((id, TextField::selected(text.unwrap_or_default())));
            self.caret_since = self.now;
            return Some(commit.unwrap_or(Action::Repaint));
        }
        None
    }

    /// The wheel at `(x, y)` (`delta_y` positive toward the user). Over a
    /// dimension it steps that coordinate (`adjust-via-wheel`); over another
    /// control it is swallowed.
    pub fn wheel(&mut self, x: f32, y: f32, delta_y: f64, modifiers: Modifiers) -> Option<Action> {
        let keys = self.keys_at(x, y);
        // The active tool's wheel bindings on the data panels.
        if keys.is_empty() && self.dialog.is_none() && self.panel_at(x, y).is_some() {
            if let Some(a) = self.tool_wheel_action(delta_y, modifiers) {
                return Some(a);
            }
        }
        if let Some(dialogs::Dialog::StateEditor(ed)) = &mut self.dialog {
            if keys.contains(&Key::DialogScroll) {
                ed.scroll = (ed.scroll + delta_y as f32).max(0.0);
            }
            return Some(Action::Repaint);
        }
        // The render scale's `adjust-via-wheel`, before its panel scrolls.
        if let Some(c) = keys.iter().find_map(|k| match self.controls.get(k)? {
            c @ rendering::Control::RenderScale { .. } => Some(c),
            _ => None,
        }) {
            return c.wheel(delta_y).or(Some(Action::Repaint));
        }
        // A range input's wheel: a step up (down the page) or down.
        // `EnumSelectWidget`'s own wheel listener: the previous or next
        // option.
        if let Some(c @ (rendering::Control::Select { .. } | rendering::Control::ShaderSelect { .. })) = keys.iter().find_map(|k| self.controls.get(k)) {
            if delta_y != 0.0 {
                return c.wheel(delta_y);
            }
        }
        if let Some(c @ rendering::Control::Slider { .. }) = keys.iter().find_map(|k| self.controls.get(k)) {
            if delta_y != 0.0 {
                return c.wheel(delta_y);
            }
        }
        // A scrolling panel scrolls (clamped when next laid out).
        if let Some(id) = keys.iter().find_map(|k| if let Key::Scroll(id) = k { Some(*id) } else { None }) {
            let s = self.scroll.entry(id).or_insert(0.0);
            *s = (*s + delta_y as f32).max(0.0);
            return Some(Action::Repaint);
        }
        if let Some(dim) = keys.iter().find_map(|k| if let Key::Dimension(d) = k { Some(*d) } else { None }) {
            if modifiers.alt {
                // `adjust-velocity-via-wheel`.
                return Some(Action::MultiplyVelocity(dim, ng_core::panel::wheel_zoom_amount(delta_y, 0)));
            }
            if delta_y == 0.0 {
                return Some(Action::Repaint);
            }
            return Some(Action::AdjustCoordinate(dim, delta_y.signum()));
        }
        self.captures(x, y).then_some(Action::Repaint)
    }

    /// A key pressed (`key` is the DOM `KeyboardEvent.key`: `"5"`,
    /// `"Enter"`, `"ArrowLeft"` …). `None` when the UI does not hold the
    /// keyboard, so the viewer gets it.
    pub fn key_down(&mut self, key: &str, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        // The shader editor's keys.
        if self.shader_focused() {
            return self.shader_key(key, KeyModifiers { shift: modifiers.shift, control: modifiers.control, alt: modifiers.alt, meta: modifiers.meta });
        }
        // The colour picker closes on Escape or Enter (its colour kept).
        if self.color_picker.is_some() {
            if matches!(key, "Escape" | "Enter") {
                self.color_picker = None;
            }
            return Some(Action::Repaint);
        }
        // A tool's binding widget held: A–Z binds (by key, any case).
        if let Some(tool) = &self.binding_tool {
            let mut chars = key.chars();
            return match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_ascii_alphabetic() => Some(tools::bind_action(c.to_ascii_uppercase(), tool)),
                _ => Some(Action::Repaint),
            };
        }
        self.caret_since = self.now;
        if matches!(self.dialog, Some(dialogs::Dialog::Palette(_))) {
            let m = KeyModifiers { shift: modifiers.shift, control: modifiers.control, alt: modifiers.alt, meta: modifiers.meta };
            return self.palette_key(viewer, key, m).or(Some(Action::Repaint));
        }
        if matches!(self.dialog, Some(dialogs::Dialog::StateEditor(_))) {
            let m = KeyModifiers { shift: modifiers.shift, control: modifiers.control, alt: modifiers.alt, meta: modifiers.meta };
            return self.dialog_key(key, m).or(Some(Action::Repaint));
        }
        // Other dialogs: their field takes keys; Escape closes them.
        if self.dialog.is_some() && self.field.is_none() {
            return Some(if key == "Escape" { Action::CloseDialog } else { Action::Repaint });
        }
        // A source URL's completions: ↑/↓ highlight one, Tab takes it.
        if let Some((FieldId::SourceUrl(_), field)) = &mut self.field {
            let completions = sources::completions(&field.text);
            if !completions.is_empty() {
                let n = completions.len();
                match key {
                    "ArrowDown" => {
                        self.completion_active = Some(self.completion_active.map_or(0, |k| (k + 1) % n));
                        return Some(Action::Repaint);
                    }
                    "ArrowUp" => {
                        self.completion_active = Some(self.completion_active.map_or(n - 1, |k| (k + n - 1) % n));
                        return Some(Action::Repaint);
                    }
                    "Tab" => {
                        let k = self.completion_active.unwrap_or(0).min(n - 1);
                        return Some(Action::Complete(completions[k].0.to_owned()));
                    }
                    _ => self.completion_active = None,
                }
            }
        }
        if let Some((id, field)) = &mut self.field {
            let numeric = matches!(id, FieldId::Number(_));
            let m = KeyModifiers { shift: modifiers.shift, control: modifiers.control, alt: modifiers.alt, meta: modifiers.meta };
            return match field.key(key, m, |c| !numeric || c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+') || !c.is_control()) {
                FieldKey::Commit => {
                    let a = self.commit_field();
                    // A source URL committed goes back to showing the source's.
                    if matches!(self.field, Some((FieldId::SourceUrl(_), _))) {
                        self.field = None;
                        self.completion_active = None;
                    }
                    a
                }
                FieldKey::Tab { .. } => {
                    let a = self.commit_field();
                    self.field = None;
                    a
                }
                FieldKey::Cancel => {
                    // A name field's `cancel`: back to the name, unfocused.
                    if matches!(id, FieldId::LayerName(_)) {
                        self.field = None;
                    }
                    Some(Action::Repaint)
                }
                // The number's `input`: each valid value applies at once.
                FieldKey::Edited => Some(match id {
                    FieldId::Slider(i, key) => slider_action(self.controls.get(&Key::Control(*i, key)), &field.text).unwrap_or(Action::Repaint),
                    // The list follows the query as typed.
                    FieldId::SegmentQuery(i) => Action::SetLayerOption(*i, "segmentQuery".into(), (!field.text.is_empty()).then(|| field.text.clone().into())),
                    _ => Action::Repaint,
                }),
                FieldKey::Moved => Some(Action::Repaint),
                FieldKey::Ignored => None,
            };
        }
        let edit = self.edit.as_mut()?;
        let n = edit.text.chars().count();
        let rank = viewer.dimensions().rank();
        let adjacent = |dim: usize, dir: i32| -> Option<usize> {
            let d = dim as i32 + dir;
            (d >= 0 && (d as usize) < rank).then_some(d as usize)
        };
        let move_to = |this: &mut Self, dim: usize| -> Action {
            // Focus moving to another coordinate commits nothing (its `blur`
            // skips that): what was typed waits, for the next commit.
            if let Some(p) = this.edit.take().filter(|p| p.modified) {
                this.pending.insert(p.dim, p.text);
            }
            let text = if viewer.navigation().valid() { coordinate_text(viewer.navigation().position()[dim]) } else { String::new() };
            this.focus_coordinate(dim, text);
            Action::Repaint
        };
        match key {
            "Enter" => Some(self.commit()),
            "Escape" => {
                // `cancel`: the fields show the position again; focus goes.
                self.edit = None;
                self.pending.clear();
                self.dropdown = None;
                self.plot = None;
                Some(Action::Repaint)
            }
            "ArrowUp" | "ArrowDown" => {
                // `adjustDimensionPosition` commits what was typed first.
                let dim = edit.dim;
                let commit = self.commit();
                Some(Action::Many(vec![commit, Action::AdjustCoordinate(dim, if key == "ArrowUp" { -1.0 } else { 1.0 })]))
            }
            "Tab" => {
                let dir = if modifiers.shift { -1 } else { 1 };
                // Past the last (or first) field focus stays: the binding
                // prevents the browser's tabbing.
                match adjacent(edit.dim, dir) {
                    Some(d) => Some(move_to(self, d)),
                    None => Some(Action::Repaint),
                }
            }
            "ArrowLeft" | "ArrowRight" => {
                let dir = if key == "ArrowRight" { 1 } else { -1 };
                let at_edge = edit.start == edit.end && edit.start == if dir == 1 { n } else { 0 };
                if at_edge {
                    if let Some(d) = adjacent(edit.dim, dir) {
                        return Some(move_to(self, d));
                    }
                }
                let caret = if edit.start != edit.end {
                    if dir == 1 {
                        edit.end
                    } else {
                        edit.start
                    }
                } else {
                    (edit.start as i32 + dir).clamp(0, n as i32) as usize
                };
                (edit.start, edit.end) = (caret, caret);
                Some(Action::Repaint)
            }
            "Home" | "End" => {
                let c = if key == "Home" { 0 } else { n };
                (edit.start, edit.end) = (c, c);
                Some(Action::Repaint)
            }
            "Backspace" | "Delete" => {
                if key == "Backspace" && edit.start == edit.end && edit.start == 0 {
                    // `delete-backward` at the start: the previous coordinate.
                    return Some(match adjacent(edit.dim, -1) {
                        Some(d) => move_to(self, d),
                        None => Action::Repaint,
                    });
                }
                let (mut a, mut b) = (edit.start, edit.end);
                if a == b {
                    if key == "Backspace" {
                        a -= 1;
                    } else if b < n {
                        b += 1;
                    }
                }
                let chars: Vec<char> = edit.text.chars().collect();
                edit.text = chars[..a].iter().chain(&chars[b..]).collect();
                (edit.start, edit.end) = (a, a);
                edit.modified = true;
                Some(Action::Repaint)
            }
            "a" if modifiers.control || modifiers.meta => {
                (edit.start, edit.end) = (0, n);
                Some(Action::Repaint)
            }
            k if k.chars().count() == 1 && !(modifiers.control || modifiers.meta) => {
                let c = k.chars().next().unwrap();
                // The `input` handler drops all but `-`, digits and `.`.
                if !(c == '-' || c == '.' || c.is_ascii_digit() || c == ' ' || c == ',') {
                    return Some(Action::Repaint);
                }
                let chars: Vec<char> = edit.text.chars().collect();
                let typed: String = chars[..edit.start].iter().chain(std::iter::once(&c)).chain(&chars[edit.end..]).collect();
                // `^(-?\d+(?:\.(?:\d+)?)?)((?:\s+(?![\s,]))|(?:\s*,\s*))$`,
                // typed at the end: on to the next coordinate.
                let at_end = edit.end == n;
                let separator = c == ' ' || c == ',';
                let number = typed.trim_end_matches([' ', ',']);
                let is_number = {
                    let s = number.strip_prefix('-').unwrap_or(number);
                    let (int, frac) = s.split_once('.').unwrap_or((s, ""));
                    !int.is_empty() && int.chars().all(|c| c.is_ascii_digit()) && frac.chars().all(|c| c.is_ascii_digit())
                };
                if separator {
                    edit.text = typed.chars().filter(|c| *c == '-' || *c == '.' || c.is_ascii_digit()).collect();
                    edit.start = edit.text.chars().count().min(edit.start);
                    edit.end = edit.start;
                    edit.modified = true;
                    if at_end && is_number {
                        if let Some(d) = adjacent(edit.dim, 1) {
                            return Some(move_to(self, d));
                        }
                    }
                    return Some(Action::Repaint);
                }
                edit.text = typed;
                edit.start += 1;
                edit.end = edit.start;
                edit.modified = true;
                Some(Action::Repaint)
            }
            _ => Some(Action::Repaint),
        }
    }

    /// Carries out `action` on `viewer`.
    pub fn apply(&mut self, viewer: &mut Viewer, action: &Action) -> Effect {
        match action {
            Action::Repaint | Action::Unported(_) => Effect::None,
            Action::TogglePanel(id) => {
                self.toggle_panel(viewer, *id);
                Effect::None
            }
            Action::ClosePanel(id) => {
                let mut l = id.location(viewer);
                l.visible = false;
                id.set_location(viewer, &l);
                Effect::None
            }
            Action::SetLocations(locations) => {
                for (id, l) in locations {
                    id.set_location(viewer, l);
                }
                Effect::None
            }
            Action::Toggle(setting) => {
                toggle_setting(viewer, *setting);
                Effect::None
            }
            Action::SetNumber(n, v) => {
                // `verifyPositiveInt`: anything else leaves the value.
                if *v > 0.0 && v.fract() == 0.0 {
                    match n {
                        NumberSetting::GpuMemoryLimit => viewer.settings.gpu_memory_limit = *v,
                        NumberSetting::SystemMemoryLimit => viewer.settings.system_memory_limit = *v,
                        NumberSetting::ConcurrentDownloads => viewer.settings.concurrent_downloads = *v,
                    }
                }
                if let Some((FieldId::Number(f), t)) = &mut self.field {
                    if f == n {
                        *t = TextField::new(number_to_string(number_setting(viewer, *n)));
                    }
                }
                Effect::None
            }
            Action::ToggleLayerBar => {
                let shown = panels::layer_bar_shown(viewer);
                let mut o = viewer.ui_state("uiControlVisibility").and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
                o.insert("showLayerPanel".into(), (!shown).into());
                viewer.set_ui_state("uiControlVisibility", Some(serde_json::Value::Object(o)));
                Effect::None
            }
            Action::SetArchived(i, archived) => {
                viewer.set_archived(*i, *archived);
                Effect::None
            }
            Action::ToggleLayerSidePanel(i) => {
                // Shown for this layer, or hidden if it already is.
                let shown = PanelId::Layer.location(viewer).visible && PanelId::Layer.layer(viewer) == Some(*i);
                if shown {
                    return self.apply(viewer, &Action::ClosePanel(PanelId::Layer));
                }
                if let Some(name) = viewer.layers().get(*i).map(|l| l.name().to_owned()) {
                    self.select_layer_named(viewer, &name, true);
                }
                Effect::None
            }
            Action::RenameLayer(i, name) => {
                let old = viewer.layers().get(*i).map(|l| l.name().to_owned());
                if let (Some(old), Some(new)) = (old, viewer.rename_layer(*i, name)) {
                    // `selectedLayer` names it.
                    if viewer.ui_state("selectedLayer").and_then(|v| v.get("layer")).and_then(serde_json::Value::as_str) == Some(old.as_str()) {
                        self.select_layer_named(viewer, &new, false);
                    }
                }
                if let Some((FieldId::LayerName(f), t)) = &mut self.field {
                    if f == i {
                        *t = TextField::new(viewer.layers().get(*i).map_or(String::new(), |l| l.name().to_owned()));
                    }
                }
                Effect::None
            }
            Action::SetTitle(t) => {
                viewer.settings.title = (!t.is_empty()).then(|| t.clone());
                Effect::None
            }
            Action::SetColor(setting, c) => {
                match setting {
                    ColorSetting::CrossSectionBackground => viewer.background = *c,
                    ColorSetting::ProjectionBackground => viewer.projection_background = *c,
                    ColorSetting::SegmentDefault(i) => viewer.set_layer_option(*i, "segmentDefaultColor", Some(ng_viewer::state::serialize_color(*c).into())),
                    ColorSetting::Annotation(i) => viewer.set_layer_option(*i, "annotationColor", Some(ng_viewer::state::serialize_color(*c).into())),
                    ColorSetting::ShaderColor(i, name) => {
                        // `TrackableRGB`: the default left out.
                        let Some(l) = viewer.layers().get(*i) else { return Effect::None };
                        let t = l.data_type().unwrap_or(ng_core::DataType::Uint8);
                        let text = l.spec.json.get("shader").and_then(serde_json::Value::as_str).unwrap_or(ng_viewer::state::DEFAULT_IMAGE_SHADER);
                        let parsed = ng_viewer::user_shader::parse_with_channels(text, t, l.channel_extents().len());
                        let default = parsed.controls.iter().find(|(n, _)| n == name).and_then(|(_, c)| match c {
                            ng_viewer::user_shader::Control::Color { default, .. } => Some(*default),
                            _ => None,
                        });
                        let hex = ng_viewer::state::serialize_color(*c);
                        let value = (default.map(ng_viewer::state::serialize_color).as_deref() != Some(hex.as_str())).then(|| hex.into());
                        let controls = shader_ui::set_control(l.spec.json.get("shaderControls"), name, value);
                        viewer.set_layer_option(*i, "shaderControls", controls);
                    }
                }
                Effect::None
            }
            Action::PickColor(setting) => Effect::PickColor {
                setting: *setting,
                current: match setting {
                    ColorSetting::CrossSectionBackground => viewer.background,
                    ColorSetting::ProjectionBackground => viewer.projection_background,
                    ColorSetting::SegmentDefault(i) => rendering::layer_color(viewer, *i, "segmentDefaultColor").unwrap_or([1.0, 0.0, 0.0]),
                    ColorSetting::Annotation(i) => viewer.layers().get(*i).map_or([1.0, 1.0, 0.0], |l| l.annotation_color()),
                    ColorSetting::ShaderColor(i, name) => viewer
                        .layers()
                        .get(*i)
                        .and_then(|l| {
                            let t = l.data_type()?;
                            let text = l.spec.json.get("shader").and_then(serde_json::Value::as_str).unwrap_or(ng_viewer::state::DEFAULT_IMAGE_SHADER);
                            let parsed = ng_viewer::user_shader::parse_with_channels(text, t, l.channel_extents().len());
                            let values = ng_viewer::user_shader::control_values(&parsed, t, l.spec.json.get("shaderControls"));
                            parsed.controls.iter().zip(values).find(|((n, _), _)| n == name).and_then(|(_, v)| match v {
                                ng_viewer::user_shader::ControlValue::Color(c) => Some(c),
                                _ => None,
                            })
                        })
                        .unwrap_or([1.0, 1.0, 1.0]),
                },
            },
            Action::Shortcut(name) => self.run_shortcut(viewer, name),
            Action::BindTool(key, owner, json) => {
                viewer.bind_tool(*key, *owner, json.clone());
                Effect::None
            }
            Action::UnbindTool(owner, json) => {
                viewer.unbind_tool(*owner, json);
                if self.active_tool.is_some_and(|k| viewer.tool(k).is_none()) {
                    self.active_tool = None;
                }
                Effect::None
            }
            Action::SelectSegments { layer, deselect, x, y } => {
                self.select_paint = Some(tools::SelectPaint { layer: *layer, deselect: *deselect, start: (*x, *y), painting: false, last: None });
                self.try_select_segment(viewer);
                Effect::None
            }
            Action::TogglePaletteList => {
                self.palette_list = !self.palette_list;
                Effect::None
            }
            Action::AddPalette { name, query } => {
                palettes::add_new(viewer, name, query);
                self.palette_list = false;
                Effect::None
            }
            Action::DeletePalette(k) => {
                palettes::remove(viewer, *k as usize);
                self.scroll.retain(|id, _| !matches!(id, PanelId::Palette(_)));
                Effect::None
            }
            Action::ConvertPalette(k) => {
                palettes::convert_to_explicit(viewer, *k as usize);
                Effect::None
            }
            Action::LayerDragOver(target) => self.layer_drag_over(viewer, *target),
            // `PositionWidget` over the channel space: the coordinate
            // floored, kept within the dimension; `channel` left out at
            // the directive's default.
            Action::SetShaderChannel(i, name, dim, text) => {
                let Some(l) = viewer.layers().get(*i) else { return Effect::None };
                let Some(t) = l.data_type() else { return Effect::None };
                let extents = l.channel_extents();
                let shader = l.spec.json.get("shader").and_then(serde_json::Value::as_str).unwrap_or(ng_viewer::state::DEFAULT_IMAGE_SHADER);
                let parsed = ng_viewer::user_shader::parse_with_channels(shader, t, extents.len());
                let values = ng_viewer::user_shader::control_values(&parsed, t, l.spec.json.get("shaderControls"));
                let Some(k) = parsed.controls.iter().position(|(n, _)| n == name) else { return Effect::None };
                let (ng_viewer::user_shader::Control::Invlerp { channel: default, .. }, ng_viewer::user_shader::ControlValue::Invlerp { channel, .. }) = (&parsed.controls[k].1, &values[k]) else { return Effect::None };
                let v = ng_core::js::parse_float(text.trim());
                let Some(&extent) = extents.get(*dim) else { return Effect::None };
                if !v.is_finite() || v < 0.0 || v >= extent as f64 {
                    return Effect::None;
                }
                let mut channel = channel.clone();
                channel[*dim] = v.floor() as u32;
                let controls = l.spec.json.get("shaderControls").cloned();
                let mut o = controls.as_ref().and_then(|c| c.get(*name)).and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
                if channel == *default {
                    o.remove("channel");
                } else {
                    o.insert("channel".into(), channel.iter().map(|&c| serde_json::Value::from(c)).collect());
                }
                let value = (!o.is_empty()).then_some(serde_json::Value::Object(o));
                let controls = shader_ui::set_control(controls.as_ref(), name, value);
                viewer.set_layer_option(*i, "shaderControls", controls);
                Effect::None
            }
            Action::SetShaderControl(i, name, value) => {
                let controls = viewer.layers().get(*i).and_then(|l| shader_ui::set_control(l.spec.json.get("shaderControls"), name, value.clone()));
                viewer.set_layer_option(*i, "shaderControls", controls);
                Effect::None
            }
            Action::RenameDimensions(names) => {
                viewer.rename_dimensions(names);
                Effect::None
            }
            Action::SetDimensionScales(scales) => {
                viewer.set_dimension_scales(scales);
                Effect::None
            }
            Action::RenamePalette(k, name) => {
                palettes::rename(viewer, *k as usize, name);
                Effect::None
            }
            Action::SetPaletteQuery(k, query) => {
                palettes::set_query(viewer, *k as usize, query);
                Effect::None
            }
            Action::RemovePaletteTool(k, index) => {
                palettes::remove_tool(viewer, *k as usize, *index);
                Effect::None
            }
            Action::ToolWheel { delta_y, modifiers } => {
                self.tool_wheel(viewer, *delta_y, *modifiers);
                Effect::None
            }
            Action::SelectLayerTab(i, tab) => {
                if let Some(l) = viewer.layers_mut().get_mut(*i) {
                    l.spec.json.insert("tab".into(), (*tab).into());
                }
                self.scroll.remove(&PanelId::Layer);
                Effect::None
            }
            Action::SelectPinnedTab(i, k, tab) => {
                layer_panel::set_pinned_tab(viewer, *i, *k, tab);
                self.scroll.remove(&PanelId::Pinned(*i as u16, *k as u16));
                Effect::None
            }
            Action::OpenUrl(url) => Effect::OpenUrl(url.clone()),
            Action::OpenShaderOverlay(i) => {
                self.shader_blur_pending = self.shader_focused();
                self.dialog = Some(dialogs::Dialog::ShaderCode(*i));
                Effect::None
            }
            Action::PinLayerPanel => {
                layer_panel::pin(viewer);
                Effect::None
            }
            Action::UnpinLayerPanel(i, k) => {
                layer_panel::unpin(viewer, *i, *k);
                Effect::None
            }
            Action::ToggleDefaultSubsources(i, s) => {
                if let Some(l) = viewer.layers_mut().get_mut(*i) {
                    let on = l.source_views().get(*s).is_some_and(|v| v.enable_default_subsources);
                    l.set_default_subsources(*s, !on);
                }
                Effect::None
            }
            Action::ToggleSubsource(i, s, k) => {
                if let Some(l) = viewer.layers_mut().get_mut(*i) {
                    let on = l.source_views().get(*s).and_then(|v| v.subsources.get(*k).map(|x| x.enabled)).unwrap_or(false);
                    l.set_subsource_enabled(*s, *k, !on);
                }
                Effect::None
            }
            Action::SelectPosition { pin } => {
                let mut l = PanelId::Selection.location(viewer);
                l.visible = true;
                PanelId::Selection.set_location(viewer, &l);
                self.selection_pinned = *pin;
                if *pin {
                    if let Some(sel) = panels::Selection::capture(viewer) {
                        sel.store(viewer);
                        // `history`: after the current entry.
                        self.selection_history.truncate(self.selection_index + 1);
                        self.selection_history.push(sel);
                        self.selection_index = self.selection_history.len() - 1;
                    }
                }
                Effect::None
            }
            Action::SelectionHistory(dir) => {
                let i = self.selection_index as i32 + dir;
                if i >= 0 && (i as usize) < self.selection_history.len() {
                    self.selection_index = i as usize;
                    self.selection_pinned = true;
                    self.selection_history[i as usize].clone().store(viewer);
                }
                Effect::None
            }
            Action::ToggleSelectionPin => {
                self.selection_pinned = !self.selection_pinned;
                if self.selection_pinned {
                    if let Some(sel) = panels::Selection::capture(viewer) {
                        sel.store(viewer);
                    }
                }
                Effect::None
            }
            Action::MoveTo(p) => {
                viewer.navigation_mut().set_position(p);
                Effect::None
            }
            Action::CopyText(t) => Effect::Clipboard(t.clone()),
            Action::ToggleSegment(layer, id) => {
                if let Some(l) = viewer.layers_mut().get_mut(*layer) {
                    let visible = panels::segment_visible(l, *id);
                    let mut segs: Vec<serde_json::Value> = l.spec.json.get("segments").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                    segs.retain(|v| {
                        let s = v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string());
                        s.trim_start_matches('!') != id.to_string()
                    });
                    // Hidden ones stay listed with `!` (the ordered set keeps them).
                    segs.push(if visible { format!("!{id}") } else { id.to_string() }.into());
                    l.spec.json.insert("segments".into(), serde_json::Value::Array(segs));
                }
                viewer.refresh_segments(*layer);
                Effect::None
            }
            Action::ToggleLayerVisible(i) => {
                if let Some(l) = viewer.layers_mut().get_mut(*i) {
                    l.spec.visible = !l.spec.visible;
                }
                Effect::None
            }
            Action::TogglePick(i) => {
                if let Some(l) = viewer.layers_mut().get_mut(*i) {
                    // `pickEnabled`, saved as `"pick": false` when off.
                    let enabled = l.spec.json.get("pick").and_then(serde_json::Value::as_bool).unwrap_or(true);
                    if enabled {
                        l.spec.json.insert("pick".into(), false.into());
                    } else {
                        l.spec.json.remove("pick");
                    }
                }
                Effect::None
            }
            Action::SetVelocity(dim, v) => {
                viewer.set_velocity(*dim, *v);
                Effect::None
            }
            Action::TogglePlayback(dim) => {
                // `togglePlayback`: enabled if it was not, then (un)paused.
                let v = viewer.velocity(*dim).unwrap_or_default();
                viewer.set_velocity(*dim, Some(ng_viewer::playback::DimensionVelocity { paused: !v.paused, ..v }));
                Effect::None
            }
            Action::MultiplyVelocity(dim, factor) => {
                let v = viewer.velocity(*dim).unwrap_or_default();
                viewer.set_velocity(*dim, Some(ng_viewer::playback::DimensionVelocity { velocity: ng_viewer::playback::multiply(v.velocity, *factor), ..v }));
                Effect::None
            }
            Action::SetBoundary(dim, name) => {
                if let Some(v) = viewer.velocity(*dim) {
                    let at_boundary = ng_viewer::playback::Boundary::ALL.into_iter().find(|b| b.name() == name).unwrap_or(v.at_boundary);
                    viewer.set_velocity(*dim, Some(ng_viewer::playback::DimensionVelocity { at_boundary, ..v }));
                }
                Effect::None
            }
            Action::AddLayer => {
                let i = viewer.add_new_layer();
                let name = viewer.layers()[i].name().to_owned();
                self.select_layer_named(viewer, &name, true);
                self.field = Some((FieldId::SourceUrl(i), TextField::new("")));
                self.caret_since = self.now;
                Effect::None
            }
            Action::SetLayerSource(i, url) => {
                // A new layer takes its name from the URL (`suggestLayerName`).
                if viewer.layers().get(*i).is_some_and(|l| l.spec.layer_type == "new") {
                    if let Some(name) = sources::suggest_layer_name(url) {
                        if let (Some(old), Some(new)) = (viewer.layers().get(*i).map(|l| l.name().to_owned()), viewer.rename_layer(*i, &name)) {
                            if viewer.ui_state("selectedLayer").and_then(|v| v.get("layer")).and_then(serde_json::Value::as_str) == Some(old.as_str()) {
                                self.select_layer_named(viewer, &new, false);
                            }
                        }
                    }
                }
                Effect::LoadSources(viewer.set_layer_source(*i, url))
            }
            Action::Complete(value) => {
                if let Some((FieldId::SourceUrl(_), field)) = &mut self.field {
                    *field = TextField::new(value.clone());
                }
                self.completion_active = None;
                Effect::None
            }
            Action::ChangeLayerType(i, t) => {
                viewer.change_layer_type(*i, t);
                Effect::None
            }
            Action::Annotate => self.annotate(viewer),
            Action::AnnotationStep { finish } => self.annotation_step(viewer, *finish),
            Action::PinAnnotation(i, n) => {
                let rank = viewer.dimensions().rank();
                if let Some(e) = viewer.layers().get(*i).and_then(|l| l.annotation_list(rank)).and_then(|l| l.into_iter().nth(*n)) {
                    self.select_annotation(viewer, *i, &e.id);
                }
                Effect::None
            }
            Action::MoveToAnnotation(i, n) => {
                // `getCenterPosition`: a point's or ellipsoid's first row; a
                // line's, box's or polyline's first two rows' middle.
                let rank = viewer.dimensions().rank();
                if let Some(e) = viewer.layers().get(*i).and_then(|l| l.annotation_list(rank)).and_then(|l| l.into_iter().nth(*n)) {
                    let two = matches!(e.icon, "ꕹ" | "❑" | "⤤") && e.rows.len() >= 2;
                    let mut position = viewer.navigation().position().to_vec();
                    for (d, p) in position.iter_mut().enumerate() {
                        let a = e.rows[0].get(d).copied().flatten();
                        let v = if two { a.zip(e.rows[1].get(d).copied().flatten()).map(|(a, b)| (a + b) / 2.0) } else { a };
                        if let Some(v) = v {
                            *p = v as f32;
                        }
                    }
                    viewer.navigation_mut().set_position(&position);
                }
                Effect::None
            }
            Action::DeleteAnnotation(i, id) => {
                viewer.delete_annotation(*i, id);
                Effect::None
            }
            Action::MoveToSegment(layer, id) => {
                if !viewer.move_to_segment(*layer, *id) {
                    // `showTemporaryMessage`'s default: 2 s.
                    self.temporary_messages.push((format!("No position information loaded for segment {id}"), self.now + 2000.0));
                }
                Effect::None
            }
            Action::ToggleHoveredSegments { star } => {
                // `LayerActionContext.segmentationToggleSegmentState`: once
                // any layer turns its segment on, only those turning one on
                // act.
                let hovered = viewer.hovered_segments();
                let mut toggles = vec![];
                let mut state = None;
                for (i, l) in viewer.layers().iter().enumerate() {
                    let picks = l.spec.json.get("pick").and_then(serde_json::Value::as_bool).unwrap_or(true);
                    let Some(Some(id)) = hovered.get(i).copied() else { continue };
                    let Ok((display, segs)) = ng_viewer::state::segmentation_settings(&l.spec.json) else { continue };
                    if !l.spec.visible || !picks || (id == 0 && display.hide_segment_zero) {
                        continue;
                    }
                    let on = if *star { !segs.iter().any(|s| s.0 == id) } else { !segs.iter().any(|s| s.0 == id && s.1) };
                    if on || state.is_none() {
                        state = Some(on);
                    }
                    toggles.push((i, id, on));
                }
                for (i, id, on) in toggles {
                    if Some(on) == state {
                        let change = if *star { segments::SegmentChange::Star(vec![id], on) } else { segments::SegmentChange::Show(vec![id], on) };
                        self.apply(viewer, &Action::SegmentChange(i, change));
                    }
                }
                Effect::None
            }
            Action::SegmentChange(i, change) => {
                if let Some(l) = viewer.layers().get(*i) {
                    let after = segments::apply_change(segments::starred(l), change);
                    viewer.set_layer_option(*i, "segments", (!after.is_empty()).then(|| segments::segments_json(&after)));
                    viewer.refresh_segments(*i);
                }
                Effect::None
            }
            Action::AutoRange { layer, ctl, range, lower, upper } => {
                let Some(l) = viewer.layers().get(*layer) else { return Effect::None };
                let Some(t) = l.data_type() else { return Effect::None };
                let type_range = ng_viewer::default_data_type_range(t).as_f64();
                let (range, window) = ng_viewer::histogram::auto_range(l.histogram_samples(), *lower, *upper, t, type_range, range[0] > range[1]);
                let controls = rendering::invlerp_controls_json(l.spec.json.get("shaderControls"), t, *ctl, range, window);
                viewer.set_layer_option(*layer, "shaderControls", controls);
                Effect::None
            }
            Action::EditJsonState
            | Action::CloseDialog
            | Action::ToggleFold(_)
            | Action::ApplyState
            | Action::DownloadState
            | Action::OpenScreenshot
            | Action::ScreenshotScale(_)
            | Action::ScreenshotKeepFov
            | Action::TakeScreenshot
            | Action::CancelScreenshot
            | Action::OpenPalette
            | Action::PaletteRun(_)
            | Action::PaletteBack
            | Action::PaletteCommand(_) => dialogs::apply(self, viewer, action),
            Action::RandomizeSeed(i) => {
                // A splitmix64 step of the time: a new 32-bit seed.
                let mut z = (self.now * 1e3) as u64 ^ 0x9e37_79b9_7f4a_7c15;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                viewer.set_layer_option(*i, "colorSeed", Some(((z ^ (z >> 31)) as u32).into()));
                Effect::None
            }
            // The layer type select: its abbreviation chosen.
            Action::SetLayerOption(i, key, Some(serde_json::Value::String(abbreviation))) if key == "type" => {
                if let Some((t, _)) = layer_panel::LAYER_TYPES.iter().find(|(_, a)| a == abbreviation) {
                    viewer.change_layer_type(*i, t);
                }
                Effect::None
            }
            Action::SetLayerOption(i, key, value) => {
                viewer.set_layer_option(*i, key, value.clone());
                Effect::None
            }
            Action::SelectLayer(i) => {
                if let Some(name) = viewer.layers().get(*i).map(|l| l.name().to_owned()) {
                    self.select_layer_named(viewer, &name, true);
                }
                Effect::None
            }
            Action::ArchiveLayer(i) => {
                viewer.set_archived(*i, true);
                Effect::None
            }
            Action::DeleteLayer(i) => {
                viewer.remove_layer(*i);
                Effect::None
            }
            Action::SetLayout(layout) => {
                // The button had focus; it goes with the old panels.
                self.refocus = viewer.layout != *layout;
                viewer.layout = *layout;
                Effect::None
            }
            Action::ToggleShowSlices => {
                viewer.show_slices = !viewer.show_slices;
                Effect::None
            }
            Action::CopyPosition => {
                if viewer.navigation().valid() {
                    Effect::Clipboard(format_position_text(viewer.navigation().position()))
                } else {
                    Effect::Clipboard(String::new())
                }
            }
            Action::CopyUrl => Effect::CopyUrl,
            Action::SetCoordinates(coordinates) => {
                for c in coordinates {
                    self.set_coordinate(viewer, c.dim, |space, _| {
                        if c.to_voxel_center && !space.bounds.voxel_center_at_integer_coordinates[c.dim] {
                            c.value + 0.5
                        } else {
                            c.value
                        }
                    });
                }
                Effect::None
            }
            Action::Many(actions) => {
                let mut effect = Effect::None;
                for a in actions {
                    let e = self.apply(viewer, a);
                    if e != Effect::None {
                        effect = e;
                    }
                }
                effect
            }
            Action::AdjustCoordinate(dim, delta) => {
                self.set_coordinate(viewer, *dim, |space, current| space.bounds.clamp_and_round_to_voxel_center(*dim, current + delta));
                Effect::None
            }
        }
    }

    /// A top-bar button: a panel shown or hidden. The layer panel shows the
    /// selected layer, or the first if none is (`selectedLayer.visible`).
    fn toggle_panel(&mut self, viewer: &mut Viewer, id: PanelId) {
        let mut l = id.location(viewer);
        l.visible = !l.visible;
        if id == PanelId::Layer && l.visible && id.layer(viewer).is_none() {
            let Some(first) = viewer.layers().iter().find(|l| !l.spec.archived).map(|l| l.name().to_owned()) else { return };
            self.select_layer_named(viewer, &first, true);
            return;
        }
        id.set_location(viewer, &l);
    }

    /// `selectedLayer.layer = …` (and shown, if `show`).
    fn select_layer_named(&mut self, viewer: &mut Viewer, name: &str, show: bool) {
        let mut o = viewer.ui_state("selectedLayer").and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
        o.insert("layer".into(), name.into());
        viewer.set_ui_state("selectedLayer", Some(serde_json::Value::Object(o)));
        if show {
            let mut l = PanelId::Layer.location(viewer);
            l.visible = true;
            PanelId::Layer.set_location(viewer, &l);
        }
    }

    /// The `n`th layer not archived (`getLayerByNonArchivedIndex`).
    fn nth_layer(viewer: &Viewer, n: usize) -> Option<usize> {
        viewer.layers().iter().enumerate().filter(|(_, l)| !l.spec.archived).nth(n).map(|(i, _)| i)
    }

    /// A global action (`Viewer.registerActionListeners`, the data panel
    /// layout's `toggle-layout`).
    fn run_shortcut(&mut self, viewer: &mut Viewer, name: &str) -> Effect {
        if let Some(key) = name.strip_prefix("tool-").and_then(|k| k.chars().next()) {
            self.activate_tool(viewer, key);
            return Effect::None;
        }
        let digit = |prefix: &str| name.strip_prefix(prefix).and_then(|d| d.parse::<usize>().ok());
        if let Some(n) = digit("toggle-layer-") {
            if let Some(i) = Self::nth_layer(viewer, n - 1) {
                return self.apply(viewer, &Action::ToggleLayerVisible(i));
            }
        } else if let Some(n) = digit("toggle-pick-layer-") {
            if let Some(i) = Self::nth_layer(viewer, n - 1) {
                return self.apply(viewer, &Action::TogglePick(i));
            }
        } else if let Some(n) = digit("select-layer-") {
            if let Some(i) = Self::nth_layer(viewer, n - 1) {
                let name = viewer.layers()[i].name().to_owned();
                self.select_layer_named(viewer, &name, true);
            }
        } else {
            match name {
                "toggle-axis-lines" => toggle_setting(viewer, Setting::ShowAxisLines),
                "toggle-scale-bar" => toggle_setting(viewer, Setting::ShowScaleBar),
                "toggle-default-annotations" => toggle_setting(viewer, Setting::ShowDefaultAnnotations),
                "toggle-show-slices" => toggle_setting(viewer, Setting::ShowSlices),
                "toggle-orthographic-projection" => toggle_setting(viewer, Setting::OrthographicProjection),
                "toggle-show-statistics" => toggle_setting(viewer, Setting::ShowStatistics),
                "help" => self.toggle_panel(viewer, PanelId::Help),
                "recolor" => {
                    let seed = || {
                        // `getRandomUint32`.
                        let mut b = [0u8; 4];
                        getrandom_fill(&mut b);
                        u32::from_le_bytes(b)
                    };
                    viewer.recolor(seed);
                }
                "clear-segments" => viewer.clear_segments(),
                // The focused panel's related layouts (`registerRelatedLayouts`).
                "toggle-layout" | "toggle-layout-alternative" => {
                    if let Some(p) = self.focused {
                        let panel = viewer.logical_panel_rects().get(p).map(|(panel, _)| *panel);
                        if let Some(panel) = panel {
                            let related = view::related_layouts(viewer.layout, panel);
                            let i = usize::from(name == "toggle-layout-alternative").min(related.len().saturating_sub(1));
                            if let Some(&target) = related.get(i) {
                                return self.apply(viewer, &Action::SetLayout(target));
                            }
                        }
                    }
                }
                "add-layer" => return self.apply(viewer, &Action::AddLayer),
                "open-command-palette" => return self.apply(viewer, &Action::OpenPalette),
                _ => {}
            }
        }
        Effect::None
    }

    fn set_coordinate(&mut self, viewer: &mut Viewer, dim: usize, f: impl FnOnce(&ng_core::CoordinateSpace, f64) -> f64) {
        if !viewer.navigation().valid() || dim >= viewer.dimensions().rank() {
            return;
        }
        let space = viewer.dimensions().clone();
        let mut p = viewer.navigation().position().to_vec();
        p[dim] = f(&space, p[dim] as f64) as f32;
        viewer.navigation_mut().set_position(&p);
        // The fields show the new position (`updateView` sets their values,
        // which leaves the caret at the end).
        self.pending.clear();
        if let Some(e) = &mut self.edit {
            e.text = coordinate_text(p[e.dim]);
            let n = e.text.chars().count();
            (e.start, e.end, e.modified) = (n, n, false);
        }
    }
}

/// A numeric setting's value.
fn number_setting(viewer: &Viewer, n: NumberSetting) -> f64 {
    match n {
        NumberSetting::GpuMemoryLimit => viewer.settings.gpu_memory_limit,
        NumberSetting::SystemMemoryLimit => viewer.settings.system_memory_limit,
        NumberSetting::ConcurrentDownloads => viewer.settings.concurrent_downloads,
    }
}

/// Flips a boolean setting.
fn toggle_setting(viewer: &mut Viewer, s: Setting) {
    match s {
        Setting::ShowAxisLines => viewer.show_axis_lines ^= true,
        Setting::ShowScaleBar => viewer.show_scale_bar ^= true,
        Setting::ShowSlices => viewer.show_slices ^= true,
        Setting::HideCrossSectionBackground3D => viewer.settings.hide_cross_section_background_3d ^= true,
        Setting::ShowDefaultAnnotations => viewer.settings.show_default_annotations ^= true,
        Setting::WireFrame => viewer.settings.wire_frame ^= true,
        Setting::Prefetch => viewer.settings.prefetch ^= true,
        Setting::AdaptiveDownsampling => viewer.settings.enable_adaptive_downsampling ^= true,
        Setting::OrthographicProjection => viewer.orthographic_projection ^= true,
        // The statistics panel's location (`statistics`).
        Setting::ShowStatistics => {
            let visible = viewer.ui_state("statistics").and_then(|v| v.get("visible")).and_then(serde_json::Value::as_bool).unwrap_or(false);
            let mut o = viewer.ui_state("statistics").and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
            if visible {
                o.remove("visible");
            } else {
                o.insert("visible".into(), true.into());
            }
            viewer.set_ui_state("statistics", (!o.is_empty()).then_some(serde_json::Value::Object(o)));
        }
    }
}

/// Random bytes (colour seeds, annotation IDs): `Math.random` in the
/// browser; natively the clock's nanoseconds, mixed with a fresh random
/// hasher per 8 bytes.
fn getrandom_fill(b: &mut [u8]) {
    #[cfg(target_arch = "wasm32")]
    for x in b.iter_mut() {
        *x = (js_sys::Math::random() * 256.0) as u8;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::hash::{BuildHasher, Hasher};
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        for (k, chunk) in b.chunks_mut(8).enumerate() {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u128(nanos);
            h.write_usize(k);
            for (x, y) in chunk.iter_mut().zip(h.finish().to_le_bytes()) {
                *x = y;
            }
        }
    }
}

/// How long the pointer rests before a tooltip shows (ms).
const TOOLTIP_DELAY: f64 = 1000.0;

/// A slider's number typed: applied if a number in [0, 1]
/// (`trackableAlphaValue`'s check).
fn slider_action(control: Option<&rendering::Control>, text: &str) -> Option<Action> {
    let Some(rendering::Control::Slider { layer, key, spec, .. }) = control else { return None };
    let v = ng_core::js::parse_float(text.trim());
    // A status-bar copy's control: its layer without the slot bit.
    let layer = layer & rendering::LAYER_MASK;
    (spec.min..=spec.max).contains(&v).then(|| Action::SetLayerOption(layer, (*key).into(), spec.json(v)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_positions_as_neuroglancer() {
        assert_eq!(format_position_text(&[19.5, 14.5, 10.5]), "19, 14, 10");
        assert_eq!(format_position_text(&[-0.5]), "-1");
        assert_eq!(mouse_position_text(&["x".into(), "y".into()], &[3.9, -2.1]), "x 3  y -3");
    }

    #[test]
    fn reads_typed_numbers_as_javascript() {
        assert_eq!(js_number(""), 0.0);
        assert_eq!(js_number("12"), 12.0);
        assert_eq!(js_number("-3.5"), -3.5);
        assert_eq!(js_number("4."), 4.0);
        assert!(js_number("-").is_nan());
        assert!(js_number("1-2").is_nan());
    }
}
