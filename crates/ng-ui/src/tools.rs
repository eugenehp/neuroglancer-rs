//! Tools (`ui/tool.ts`): the key binding widgets (`ToolBindingWidget`),
//! the bound tools in the top bar (`AnnotationToolStatusWidget`), the status
//! messages at the bottom (`StatusMessage`), and what an active tool shows.

use ng_viewer::tools::{ToolBinding, ToolOwner};
use ng_viewer::Viewer;
use serde_json::Value;
use threers::style;
use threers::ui::{Edges, NodeId};

use crate::view::Builder;
use crate::{Action, Key, NgUi};
use ng_core::js::number_to_string;

const SMALL: f32 = 40.0 / 3.0;
const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
/// `font: 9pt monospace`.
const KEY_FONT: f32 = 12.0;

/// A tool, by the binder it belongs to and its JSON.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolRef {
    pub owner: ToolOwner,
    pub json: Value,
}

impl ToolRef {
    pub(crate) fn layer(i: usize, json: impl Into<Value>) -> Self {
        ToolRef { owner: ToolOwner::Layer(i), json: json.into() }
    }

    /// `{type: "dimension", dimension}` of the viewer.
    pub(crate) fn dimension(name: &str) -> Self {
        ToolRef { owner: ToolOwner::Global, json: serde_json::json!({ "type": "dimension", "dimension": name }) }
    }

    /// The tool's type (`toolJson` or its `type`).
    pub(crate) fn id(&self) -> &str {
        match &self.json {
            Value::String(s) => s,
            Value::Object(o) => o.get("type").and_then(Value::as_str).unwrap_or(""),
            _ => "",
        }
    }
}

impl From<&ToolBinding> for ToolRef {
    fn from(b: &ToolBinding) -> Self {
        ToolRef { owner: b.owner, json: b.json.clone() }
    }
}

/// A layer control's label (`LayerControlDefinition.label`) by its tool ID.
pub(crate) fn control_label(id: &str) -> Option<&'static str> {
    Some(match id {
        "crossSectionRenderScale" => "Resolution (slice)",
        "blend" => "Blending (slice)",
        "opacity" => "Opacity (slice)",
        "volumeRendering" => "Volume rendering (experimental)",
        "colorSeed" => "Color seed",
        "segmentDefaultColor" => "Fixed color",
        "saturation" => "Saturation",
        "selectedAlpha" => "Opacity (on)",
        "notSelectedAlpha" => "Opacity (off)",
        "hideSegmentZero" => "Hide segment ID 0",
        "baseSegmentColoring" => "Base segment coloring",
        "ignoreNullVisibleSet" => "Show all by default",
        "hoverHighlight" => "Highlight on hover",
        _ => return None,
    })
}

/// `Tool.description`.
pub(crate) fn description(tool: &ToolRef) -> String {
    let field = |k: &str| tool.json.get(k).and_then(Value::as_str).unwrap_or("").to_owned();
    match tool.id() {
        "dimension" => format!("dim {}", field("dimension")),
        "selectSegments" => "select".into(),
        "selectPreviousAnnotation" => "select previous annotation".into(),
        "selectNextAnnotation" => "select next annotation".into(),
        "shaderControl" => field("control"),
        id => control_label(id).map_or_else(|| id.to_owned(), str::to_owned),
    }
}

/// `nonArchivedLayerIndex + 1` of layer `i`.
pub(crate) fn layer_number(viewer: &Viewer, i: usize) -> Option<usize> {
    let layers = viewer.layers();
    (!layers.get(i)?.spec.archived).then(|| layers[..i].iter().filter(|l| !l.spec.archived).count() + 1)
}

impl Builder<'_> {
    /// The key a tool is bound to.
    pub(crate) fn tool_key(&self, tool: &ToolRef) -> Option<char> {
        self.bindings.iter().find(|b| b.owner == tool.owner && b.json == tool.json).map(|b| b.key)
    }

    /// Registers a tool for input on `node` (by place this frame).
    fn tool_slot(&mut self, tool: ToolRef) -> u32 {
        self.tools.push(tool);
        self.tools.len() as u32 - 1
    }

    /// `ToolBindingWidget`: `[K]`, or `[ ]` while unbound, the brackets
    /// grey; outlined under the pointer. A press binds the next letter
    /// typed, a double click unbinds.
    pub(crate) fn tool_binding(&mut self, parent: NodeId, tool: ToolRef, title: Option<&str>) -> NodeId {
        let key_text = self.tool_key(&tool).map_or(" ".to_owned(), |k| k.to_string());
        let slot = self.tool_slot(tool);
        let key = Key::ToolBinding(slot);
        let node = self.ui.add_box(parent, style! { display: flex; flex-shrink: 0; font-family: monospace; font-size: {KEY_FONT}; color: #0ff; });
        self.key(node, key, None);
        if self.hover(key) {
            // `:hover { outline: 1px solid #fff }`, outside the box.
            let s = self.ui.style_mut(node);
            s.border = Edges::all(1.0);
            s.border_color = WHITE;
            s.margin = Edges::all(-1.0);
        }
        let hint = "click → bind key, dbclick → unbind";
        self.title(key, title.map_or(hint.to_owned(), |t| format!("{t}\n({hint})")));
        for (t, color) in [("[", "#999"), (key_text.as_str(), "#0ff"), ("]", "#999")] {
            self.text_box(node, t, style! { font-family: monospace; font-size: {KEY_FONT}; color: {color}; white-space: pre; });
        }
        node
    }
}

impl NgUi {
    /// `.neuroglancer-annotation-tool-status`: each bound tool, by key —
    /// `[K]`, the layer's number, the description in green.
    pub(crate) fn tool_status_widget(&mut self, b: &mut Builder, row: NodeId, viewer: &Viewer) {
        let mut bindings = b.bindings.clone();
        // The selected layer's legacy tool (an annotation layer's `tool`).
        let legacy = viewer
            .ui_state("selectedLayer")
            .and_then(|v| v.get("layer"))
            .and_then(serde_json::Value::as_str)
            .and_then(|name| viewer.layers().iter().position(|l| l.name() == name))
            .and_then(|i| {
                let tool = viewer.layers()[i].spec.json.get("tool")?.as_str()?;
                let description = match tool {
                    "annotatePoint" => "annotate point",
                    "annotateLine" => "annotate line",
                    "annotateBoundingBox" => "annotate bounding box",
                    "annotateSphere" => "annotate ellipsoid",
                    "annotatePolyline" => "annotate polyline",
                    _ => return None,
                };
                Some((i, description))
            });
        if bindings.is_empty() && legacy.is_none() {
            return;
        }
        bindings.sort_by_key(|t| t.key);
        let status = b.ui.add_box(row, style! { display: flex; align-items: center; flex-shrink: 1; min-width: 0; overflow: hidden; });
        if let Some((i, description)) = legacy {
            let widget = b.ui.add_box(status, style! { display: flex; align-items: center; flex-shrink: 0; });
            let key = Key::Control(i, "legacyTool");
            // dblclick: `tool.layer.tool.value = undefined`.
            b.key(widget, key, None);
            b.title(key, "dblclick → unbind");
            if let Some(number) = layer_number(viewer, i) {
                b.text_box(widget, &number.to_string(), style! { font-family: sans-serif; font-size: {SMALL}; font-weight: bold; color: white; background: #9a7518; });
            }
            b.text_box(widget, description, style! { font-family: sans-serif; font-size: {SMALL}; color: #3c3; white-space: nowrap; });
        }
        for (n, binding) in bindings.iter().enumerate() {
            let n = n + usize::from(legacy.is_some());
            let tool = ToolRef::from(binding);
            let slot = b.tool_slot(tool.clone());
            let key = Key::ToolStatus(slot);
            let widget = b.ui.add_box(status, style! { display: flex; align-items: center; flex-shrink: 0; });
            if n > 0 {
                b.ui.style_mut(widget).margin.left = 3.0;
            }
            b.key(widget, key, None);
            b.title(key, "dblclick → unbind, click → bind key");
            if b.hover(key) {
                let s = b.ui.style_mut(widget);
                s.border = Edges::all(1.0);
                s.border_color = WHITE;
                s.margin = Edges::new(-1.0, -1.0, -1.0, if n > 0 { 2.0 } else { -1.0 });
            }
            let k = b.ui.add_box(widget, style! { display: flex; font-family: monospace; font-size: {KEY_FONT}; color: #0ff; });
            for (t, color) in [("[", "#999"), (&binding.key.to_string()[..], "#0ff"), ("]", "#999")] {
                b.text_box(k, t, style! { font-family: monospace; font-size: {KEY_FONT}; color: {color}; });
            }
            if let ToolOwner::Layer(i) = binding.owner {
                if let Some(number) = layer_number(viewer, i) {
                    b.text_box(widget, &number.to_string(), style! { font-family: sans-serif; font-size: {SMALL}; font-weight: bold; color: white; background: #9a7518; });
                }
            }
            b.text_box(widget, &description(&tool), style! { font-family: sans-serif; font-size: {SMALL}; color: #3c3; white-space: nowrap; });
        }
    }

    /// `#neuroglancer-status-container`: at the bottom, over everything,
    /// one `<li>` per message — the key binding prompt, the active tool's.
    pub(crate) fn status_messages(&mut self, b: &mut Builder, root: NodeId, viewer: &mut Viewer, width: f32) {
        let binding = self.binding_tool.is_some();
        let active = self.active_tool.and_then(|k| viewer.tool(k));
        let now = self.now;
        self.temporary_messages.retain(|(_, until)| *until > now);
        if !binding && active.is_none() && self.temporary_messages.is_empty() {
            return;
        }
        let container = b.ui.add_box(
            root,
            style! { position: absolute; left: 0; bottom: 0; display: flex; flex-direction: column; background: #808080; color: white; font-family: sans-serif; font-size: {SMALL}; },
        );
        b.key(container, Key::Status, None);
        let li = |b: &mut Builder| b.ui.add_box(container, style! { display: flex; flex-direction: column; width: {width}; box-sizing: border-box; padding: 0 0 5px 10px; });
        if binding {
            let item = li(b);
            b.text_box(item, "Press A-Z to bind key", style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
        }
        if let Some(active) = active {
            let item = li(b);
            self.tool_status(b, item, viewer, &ToolRef::from(&active));
        }
        for (text, _) in self.temporary_messages.clone() {
            let item = li(b);
            b.text_box(item, &text, style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
        }
    }

    /// What an active tool shows (`makeToolActivationStatusMessage…`): a
    /// header and a body side by side, then the tool's input bindings.
    fn tool_status(&mut self, b: &mut Builder, item: NodeId, viewer: &mut Viewer, tool: &ToolRef) {
        let content = b.ui.add_box(item, style! { display: flex; });
        let header = b.ui.add_box(content, style! { display: flex; align-items: center; padding: 3px; background: #555; flex-shrink: 0; });
        let body = b.ui.add_box(content, style! { display: flex; flex: 1; padding: 2px; min-width: 0; });
        let font = style! { font-family: sans-serif; font-size: {SMALL}; color: white; };
        let bindings = match (tool.owner, tool.id()) {
            (ToolOwner::Global, "dimension") => {
                let name = tool.json.get("dimension").and_then(Value::as_str).unwrap_or("").to_owned();
                if let Some(dim) = viewer.dimensions().names.iter().position(|n| *n == name) {
                    b.ui.clear_children(content);
                    self.dimension_tool_status(b, content, viewer, dim);
                }
                "wheel→adjust-position-via-wheel, alt+wheel→adjust-velocity-via-wheel, {space,mousedown0}→toggle-playback"
            }
            (ToolOwner::Layer(i), "selectSegments") => {
                // `updateStatus`.
                // The header is `width: min-content`: its text wraps at
                // every space. Blink takes each word's width between its
                // ends' offsets in the whole run, each rounded up to 1/64 px
                // (so "segments" after "Select " is 57.78125, alone 57.796875).
                let min_content = |b: &mut Builder, text: &str| {
                    let mut widest = 0.0f32;
                    let mut start = 0;
                    for word in text.split(' ') {
                        let end = start + word.len();
                        widest = widest.max(b.width(&text[..end], &font) - b.width(&text[..start], &font));
                        start = end + 1;
                    }
                    // A block of that width (flex text does not grow when it wraps).
                    let mut f = font.clone();
                    f.display = threers::ui::Display::Block;
                    f.width = threers::ui::Length::Px(widest);
                    f.flex_shrink = 0.0;
                    f.wrap = true;
                    f
                };
                match self.select_paint.as_ref().filter(|p| p.layer == i) {
                    None => {
                        let f = min_content(b, "Select/Deselect segments");
                        b.text_box(header, "Select/Deselect segments", f);
                        b.text_box(body, "mousedown0 to select segments; alt+mousedown0 to deselect segments.", font.clone());
                    }
                    Some(p) => {
                        let verb = if p.deselect { "deselect" } else { "select" };
                        let shown = viewer.layers().get(i).map_or(0, |l| crate::segments::starred(l).iter().filter(|s| s.1).count());
                        let text = format!("{} segments", if p.deselect { "Deselect" } else { "Select" });
                        let f = min_content(b, &text);
                        b.text_box(header, &text, f);
                        b.text_box(body, &format!("Drag to {verb} segments ({shown} selected)."), font.clone());
                    }
                }
                "mousedown0→drag-select-segments"
            }
            (ToolOwner::Layer(i), _) => {
                // `LayerControlTool`: the control's label in the header, the
                // control in the body.
                let Some((label, kind)) = self.layer_control(b, body, viewer, i, tool, crate::rendering::STATUS_SLOT) else {
                    b.ui.remove(content);
                    return;
                };
                b.ui.reparent(label, header);
                match kind {
                    ControlKind::Wheel => "shift+wheel→adjust-via-wheel",
                    ControlKind::Invlerp => "shift+wheel→adjust-contrast-via-wheel, shift+mousedown0→adjust-via-drag, shift+mousedown2→invert-range",
                    ControlKind::None => "",
                }
            }
            _ => "",
        };
        if !bindings.is_empty() {
            // `→` comes from a fallback font whose line is a pixel taller.
            b.text_box(item, bindings, style! { font-family: sans-serif; font-size: {SMALL}; color: white; background: #333; height: 16px; padding-top: 1px; box-sizing: border-box; });
        }
    }
}

/// The select segments tool's drag (`ToolState.SELECT`/`DESELECT`): from
/// where, whether past 5px (painting), the segment last acted on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SelectPaint {
    pub layer: usize,
    pub deselect: bool,
    pub start: (f32, f32),
    pub painting: bool,
    pub last: Option<u64>,
}

/// The input an active layer control tool takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ControlKind {
    /// `adjust-via-wheel`.
    Wheel,
    /// The invlerp's contrast, drag and inversion.
    Invlerp,
    /// None (a checkbox toggles on activation).
    None,
}

/// `Action`s of the tool input the shells route here.
pub(crate) fn bind_action(key: char, tool: &ToolRef) -> Action {
    Action::BindTool(key, tool.owner, tool.json.clone())
}

impl NgUi {
    /// `GlobalToolBinder.activate`: the tool bound to `key`, active until
    /// the key is released; its activation's immediate effect (a checkbox
    /// toggles).
    pub(crate) fn activate_tool(&mut self, viewer: &mut Viewer, key: char) {
        let Some(binding) = viewer.tool(key) else {
            self.active_tool = None;
            return;
        };
        if self.active_tool == Some(key) {
            return;
        }
        // `SelectPrevious/NextAnnotationTool`: a step through the list,
        // then `activation.cancel()`.
        if let (ToolOwner::Layer(i), Value::String(id)) = (binding.owner, &binding.json) {
            let offset = match id.as_str() {
                "selectPreviousAnnotation" => Some(-1),
                "selectNextAnnotation" => Some(1),
                _ => None,
            };
            if let Some(offset) = offset {
                self.shift_selected_annotation(viewer, i, offset);
                self.active_tool = None;
                return;
            }
        }
        self.active_tool = Some(key);
        self.active_ref = Some(ToolRef::from(&binding));
        // `checkboxLayerControl`'s `activateTool`.
        if let (ToolOwner::Layer(i), Value::String(id)) = (binding.owner, &binding.json) {
            if let Some(&(_, _, default)) = crate::rendering::SEGMENTATION_CHECKBOXES.iter().find(|(_, k, _)| k == id) {
                let on = viewer.layers().get(i).and_then(|l| l.spec.json.get(id.as_str())?.as_bool()).unwrap_or(default);
                let value = (!on != default).then_some(Value::Bool(!on));
                viewer.set_layer_option(i, id, value);
            }
        }
    }

    /// A key released: a momentary tool ends with its key
    /// (`event.code === "Key" + key`).
    pub fn key_up(&mut self, code: &str) -> bool {
        match self.active_tool {
            Some(k) if code.strip_prefix("Key").is_some_and(|c| c.len() == 1 && c.starts_with(k)) => {
                self.active_tool = None;
                self.active_ref = None;
                true
            }
            _ => false,
        }
    }

    /// The window lost the focus: a momentary tool ends (`blur`), as does
    /// binding a key.
    pub fn blur_window(&mut self) -> bool {
        let was = self.active_tool.is_some() || self.binding_tool.is_some();
        self.active_tool = None;
        self.active_ref = None;
        self.binding_tool = None;
        was
    }

    /// The active tool, if a dimension tool: its dimension.
    pub(crate) fn active_dimension_tool(&self, viewer: &Viewer) -> Option<usize> {
        let tool = self.active_ref.as_ref().filter(|t| t.owner == ToolOwner::Global && t.id() == "dimension")?;
        let name = tool.json.get("dimension")?.as_str()?;
        viewer.dimensions().names.iter().position(|n| n == name)
    }

    /// The wheel over the data panels while a tool is active: the tool's,
    /// if it binds it with these modifiers.
    pub(crate) fn tool_wheel_action(&self, delta_y: f64, modifiers: crate::Modifiers) -> Option<Action> {
        let tool = self.active_ref.as_ref()?;
        let takes = match (tool.owner, tool.id()) {
            // `at:shift?+wheel`, `at:shift?+alt+wheel`.
            (ToolOwner::Global, "dimension") => true,
            // `at:shift+wheel`.
            (ToolOwner::Layer(_), id) => modifiers.shift && (control_label(id).is_some() || id == "shaderControl"),
            _ => false,
        };
        takes.then_some(Action::ToolWheel { delta_y, modifiers })
    }

    /// The active tool's wheel action (`adjust-…-via-wheel`).
    pub(crate) fn tool_wheel(&mut self, viewer: &mut Viewer, delta_y: f64, modifiers: crate::Modifiers) {
        let Some(tool) = self.active_ref.clone() else { return };
        match (tool.owner, tool.id()) {
            (ToolOwner::Global, "dimension") => {
                let Some(dim) = self.active_dimension_tool(viewer) else { return };
                let action = if modifiers.alt {
                    Action::MultiplyVelocity(dim, ng_core::panel::wheel_zoom_amount(delta_y, 0))
                } else if delta_y != 0.0 {
                    // `adjustDimensionPosition`: a voxel toward the wheel.
                    Action::AdjustCoordinate(dim, delta_y.signum())
                } else {
                    return;
                };
                self.apply(viewer, &action);
            }
            (ToolOwner::Layer(i), "shaderControl") => {
                // `adjust-contrast-via-wheel`.
                if let Some(control) = self.controls.get(&crate::Key::Control(i | crate::rendering::STATUS_SLOT, "cdf")).cloned() {
                    if let Some(a) = control.contrast(ng_core::panel::wheel_zoom_amount(delta_y, 0)) {
                        self.apply(viewer, &a);
                    }
                }
            }
            (ToolOwner::Layer(i), id) => {
                let Some(key) = self.controls.keys().copied().find(|k| matches!(k, crate::Key::Control(s, n) if *s == i | crate::rendering::STATUS_SLOT && *n == id)) else { return };
                let control = self.controls[&key].clone();
                if let Some(a) = control.wheel(delta_y) {
                    self.apply(viewer, &a);
                }
            }
            (ToolOwner::Global, _) => {}
        }
    }

    /// A press on the data panels while a tool is active.
    pub(crate) fn tool_press(&mut self, button: crate::Button, modifiers: crate::Modifiers, viewer: &Viewer) -> Option<Action> {
        if button != crate::Button::Primary || modifiers.control || modifiers.meta {
            return None;
        }
        if let Some(ToolRef { owner: ToolOwner::Layer(layer), .. }) = self.active_ref.as_ref().filter(|t| t.id() == "selectSegments") {
            let (x, y) = self.pointer.unwrap_or_default();
            return Some(Action::SelectSegments { layer: *layer, deselect: modifiers.alt, x, y });
        }
        // The dimension tool's `at:shift?+alt?+mousedown0`.
        self.active_dimension_tool(viewer).map(Action::TogglePlayback)
    }

    /// `trySelectSegment`: the segment under the pointer selected (starred
    /// and shown) or deselected, once per segment the drag passes over.
    pub(crate) fn try_select_segment(&mut self, viewer: &mut Viewer) {
        let Some(p) = &self.select_paint else { return };
        let (layer, deselect) = (p.layer, p.deselect);
        let Some(id) = viewer.hovered_segments().get(layer).copied().flatten().filter(|&id| id != 0) else { return };
        if p.last == Some(id) {
            return;
        }
        if let Some(p) = &mut self.select_paint {
            p.last = Some(id);
        }
        let change = if deselect { crate::SegmentChange::Star(vec![id], false) } else { crate::SegmentChange::Show(vec![id], true) };
        self.apply(viewer, &Action::SegmentChange(layer, change));
    }
}

impl NgUi {
    /// The dimension tool's status (`DimensionTool.makeTool`,
    /// `.neuroglancer-position-tool` on #474747): the dimension as the
    /// position widget shows it, its plot along the row, and the playback
    /// checkbox icon.
    pub(crate) fn dimension_tool_status(&mut self, b: &mut Builder, content: NodeId, viewer: &Viewer, dim: usize) {
        b.ui.style_mut(content).background = Some([71.0 / 255.0, 71.0 / 255.0, 71.0 / 255.0, 1.0]);
        self.dimension_tool_widgets(b, content, content, viewer, dim, PLAYBACK_STATUS_ICON);
    }

    /// The dimension tool's widgets (`makeTool`): in `row` the dimension
    /// (`PositionWidget`, one dimension, centred), then the playback
    /// `CheckboxIcon`; its `PositionPlot` (row orientation) between them,
    /// or (in a palette, `plot_parent` apart from `row`) under them.
    pub(crate) fn dimension_tool_widgets(&mut self, b: &mut Builder, row: NodeId, plot_parent: NodeId, viewer: &Viewer, dim: usize, icon_id: u32) -> NodeId {
        use crate::view::{input, lu};
        let space = viewer.dimensions().clone();
        let ch = b.ch("monospace", 13.0);
        let widget = b.ui.add_box(row, style! { display: flex; align-items: center; flex-shrink: 0; });
        // A line of the font (19px with its border), the inputs at its top.
        let d = b.ui.add_box(widget, style! { display: flex; align-items: flex-start; height: 19px; box-sizing: border-box; border: 1px solid; border-color: transparent; flex-shrink: 0; });
        let name = &space.names[dim];
        input(b, d, name, name.chars().count().max(1), "#ff6", ch, false, None);
        let (scale, prefix, unit) = ng_core::si_units::format_scale_with_unit(space.scales[dim], &space.units[dim]);
        let scale = format!("{scale}{prefix}{unit}");
        let container = b.ui.add_box(d, style! { display: flex; margin: {Edges::new(0.0, if scale.is_empty() { 2.0 } else { 5.0 }, 0.0, 4.0)}; align-items: flex-start; flex-shrink: 0; });
        // `::before`/`::after` in the surrounding font.
        let paren = style! { color: #aaa; font-family: sans-serif; font-size: {SMALL}; line-height: 15px; margin-top: 1px; flex-shrink: 0; };
        if !scale.is_empty() {
            b.text_box(container, "(", paren.clone());
        }
        input(b, container, &scale, scale.chars().count().max(1), "#bbb", ch, false, None);
        if !scale.is_empty() {
            b.text_box(container, ")", paren);
        }
        let position = viewer.navigation().position().get(dim).copied().unwrap_or(0.0) as f64;
        let coordinate = if viewer.navigation().valid() { number_to_string(position.floor()) } else { String::new() };
        input(b, d, &coordinate, coordinate.chars().count().max(1), "white", ch, true, None);

        let make_plot = |b: &mut Builder, parent: NodeId| {
            // The bounds over the canvas, which is as wide as what is left
            // (less half the wider label each side).
            let plot = b.ui.add_box(parent, style! { display: flex; flex-direction: column; flex: 1; min-width: 0; position: relative; overflow: hidden; });
            if parent != row {
                // Under the row: its own height.
                let st = b.ui.style_mut(plot);
                st.flex_grow = 0.0;
                st.flex_shrink = 0.0;
                st.flex_basis = threers::ui::Length::Auto;
            }
            let labels = b.ui.add_box(plot, style! { position: relative; height: 15px; flex-shrink: 0; });
            b.ui.add_box(plot, style! { height: 3px; flex-shrink: 0; });
            let canvas = b.ui.add_box(plot, style! { position: relative; height: {crate::plot::ROW_CANVAS_HEIGHT as f32}; flex-shrink: 0; });
            // Drawn once laid out: the canvas's width sets the columns.
            b.deferred.push(crate::rendering::Deferred::RowPlot { labels, canvas, space: Box::new(space.clone()), dim, position, ch: lu(ch) });
            plot
        };
        let plot = if plot_parent == row {
            Some(make_plot(b, row))
        } else {
            b.ui.add_box(row, style! { flex: 1; });
            None
        };
        // The playback `CheckboxIcon` (dark scheme).
        let enabled = viewer.velocity(dim).is_some();
        let toggle = if enabled { None } else { Some(ng_viewer::playback::DimensionVelocity::default()) };
        let icon = b.icon(row, icon_id, Some(crate::icons::VIDEO), "", Edges::symmetric(0.0, 1.0), Action::SetVelocity(dim, toggle));
        if enabled {
            b.ui.style_mut(icon).background = Some([1.0, 1.0, 1.0, 0.2]);
        }
        b.title(Key::Icon(icon_id), if enabled { "Disable playback/velocity" } else { "Enable playback/velocity" });
        plot.unwrap_or_else(|| make_plot(b, plot_parent))
    }
}

/// The dimension tool status's playback icon.
const PLAYBACK_STATUS_ICON: u32 = 3100;
