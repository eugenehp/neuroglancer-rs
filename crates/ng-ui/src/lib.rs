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
mod apply;
mod dialogs;
mod input;
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
use apply::{getrandom_fill, number_setting};
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
