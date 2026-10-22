//! Tool palettes (`ui/tool_palette.ts`): side panels of tools, each listed
//! explicitly or by a query (`+`: every tool; `type:shaderControl`…), kept
//! in the `toolPalettes` state key (name → location, `tools`, `query`); the
//! wrench's list of them (`MultiToolPaletteDropdownButton`).

use ng_viewer::tools::ToolOwner;
use ng_viewer::Viewer;
use serde_json::{Map, Value};
use threers::style;
use threers::ui::{Edges, NodeId};

use crate::tools::ToolRef;
use crate::view::Builder;
use crate::{icons, Action, FieldId, Key, NgUi, PanelId};

const SMALL: f32 = 40.0 / 3.0;

/// `CANNED_PALETTES`: name, description, query.
pub(crate) const CANNED: [(&str, Option<&str>, &str); 3] = [("Palette", Some("New empty palette"), ""), ("All controls", None, "+"), ("Shader controls", None, "type:shaderControl")];

/// The palettes in the state, in order.
fn all(viewer: &Viewer) -> Vec<(String, Map<String, Value>)> {
    match viewer.ui_state("toolPalettes") {
        Some(Value::Object(o)) => o.iter().map(|(k, v)| (k.clone(), v.as_object().cloned().unwrap_or_default())).collect(),
        _ => vec![],
    }
}

fn set_all(viewer: &mut Viewer, palettes: Vec<(String, Map<String, Value>)>) {
    let o: Map<String, Value> = palettes.into_iter().map(|(k, v)| (k, Value::Object(v))).collect();
    viewer.set_ui_state("toolPalettes", (!o.is_empty()).then_some(Value::Object(o)));
}

pub(crate) fn count(viewer: &Viewer) -> usize {
    all(viewer).len()
}

/// Palette `k`: its name and JSON.
pub(crate) fn entry(viewer: &Viewer, k: usize) -> Option<(String, Map<String, Value>)> {
    all(viewer).into_iter().nth(k)
}

/// Replaces palette `k` (its name too).
pub(crate) fn set_entry(viewer: &mut Viewer, k: usize, name: &str, json: Map<String, Value>) {
    let mut palettes = all(viewer);
    if let Some(p) = palettes.get_mut(k) {
        *p = (name.to_owned(), json);
    }
    set_all(viewer, palettes);
}

/// `checkTitles`: `name`, or with the first numeric suffix free.
fn unique_name(palettes: &[(String, Map<String, Value>)], name: &str, except: Option<usize>) -> String {
    let taken = |n: &str| palettes.iter().enumerate().any(|(i, p)| Some(i) != except && p.0 == n);
    if !taken(name) {
        return name.to_owned();
    }
    (1..).map(|s| format!("{name}{s}")).find(|n| !taken(n)).unwrap()
}

/// `MultiToolPaletteState.addNew`: a palette shown at the default
/// location; its index.
pub(crate) fn add_new(viewer: &mut Viewer, name: &str, query: &str) -> usize {
    let mut palettes = all(viewer);
    let name = unique_name(&palettes, name, None);
    let mut json = Map::new();
    if !query.is_empty() {
        json.insert("query".into(), query.into());
    }
    palettes.push((name, json));
    let k = palettes.len() - 1;
    set_all(viewer, palettes);
    k
}

pub(crate) fn remove(viewer: &mut Viewer, k: usize) {
    let mut palettes = all(viewer);
    if k < palettes.len() {
        palettes.remove(k);
    }
    set_all(viewer, palettes);
}

/// `PaletteNameWidget`: renamed (made unique).
pub(crate) fn rename(viewer: &mut Viewer, k: usize, name: &str) {
    let palettes = all(viewer);
    let Some((_, json)) = palettes.get(k).cloned() else { return };
    let name = unique_name(&palettes, name, Some(k));
    set_entry(viewer, k, &name, json);
}

fn query_of(json: &Map<String, Value>) -> &str {
    json.get("query").and_then(Value::as_str).unwrap_or("")
}

fn explicit_tools(json: &Map<String, Value>) -> Vec<Value> {
    json.get("tools").and_then(Value::as_array).cloned().unwrap_or_default()
}

/// `queryDefined`: no explicit tools.
fn query_defined(json: &Map<String, Value>) -> bool {
    explicit_tools(json).is_empty()
}

/// The tools palette `k` shows: its query's matches (`QueryResults`), or
/// its explicit list.
pub(crate) fn tools(viewer: &Viewer, json: &Map<String, Value>) -> Vec<Value> {
    if !query_defined(json) {
        return explicit_tools(json);
    }
    match ng_viewer::tool_query::parse(query_of(json)) {
        Ok(q) => ng_viewer::tool_query::matching_tools(&q, &viewer.tool_lists()),
        Err(_) => vec![],
    }
}

/// `convertToExplicitPalette`: the query's results become the list.
pub(crate) fn convert_to_explicit(viewer: &mut Viewer, k: usize) {
    let Some((name, mut json)) = entry(viewer, k) else { return };
    let results: Vec<Value> = tools(viewer, &json)
        .into_iter()
        .map(|mut t| {
            if let Some(o) = t.as_object_mut() {
                o.remove("layerType");
            }
            t
        })
        .collect();
    json.remove("query");
    json.insert("tools".into(), Value::Array(results));
    set_entry(viewer, k, &name, json);
}

/// A tool removed from an explicit palette (double click, `unbind`).
pub(crate) fn remove_tool(viewer: &mut Viewer, k: usize, index: usize) {
    let Some((name, mut json)) = entry(viewer, k) else { return };
    let mut list = explicit_tools(&json);
    if index < list.len() {
        list.remove(index);
    }
    if list.is_empty() {
        json.remove("tools");
    } else {
        json.insert("tools".into(), Value::Array(list));
    }
    set_entry(viewer, k, &name, json);
}

/// A palette's query set (`onCommit`).
pub(crate) fn set_query(viewer: &mut Viewer, k: usize, query: &str) {
    let Some((name, mut json)) = entry(viewer, k) else { return };
    if query.is_empty() {
        json.remove("query");
    } else {
        json.insert("query".into(), query.into());
    }
    set_entry(viewer, k, &name, json);
}

/// `updateTooltip`: the tool's properties, and what dragging does.
fn tool_title(json: &Value, query_defined: bool) -> String {
    let props: Vec<String> = match json {
        Value::Object(o) => o.iter().filter(|(k, _)| *k != "layer" && *k != "layerType").map(|(k, v)| format!("{k}:{}", v.as_str().map_or_else(|| v.to_string(), str::to_owned))).collect(),
        Value::String(s) => vec![format!("type:{s}")],
        _ => vec![],
    };
    let mut text = props.join(" ");
    text.push_str(if query_defined { "\nDrag to copy to another palette" } else { "\nDrag to move/copy, dblclick to remove" });
    text
}

impl NgUi {
    /// `ToolPalettePanel`: the title bar (name, convert-the-query, delete,
    /// close), the query while the palette is query-defined, the tools —
    /// a layer's grouped under its eye and name.
    pub(crate) fn palette_panel(&mut self, b: &mut Builder, panel: NodeId, viewer: &mut Viewer, k: u16) {
        let id = PanelId::Palette(k);
        let Some((name, json)) = entry(viewer, k as usize) else { return };
        let bar = b.ui.add_box(panel, style! { display: flex; align-items: center; font-family: sans-serif; font-size: {SMALL}; background: #03c; padding: 2px; color: white; flex-shrink: 0; });
        self.palette_name(b, bar, k, &name, false);
        let base = 2100 + 8 * k as u32;
        let defined = query_defined(&json);
        let query = query_of(&json).to_owned();
        if !query.is_empty() {
            // `CheckboxIcon` over `hasQuery`, checked.
            let search = b.icon(bar, base, Some(icons::SEARCH), "", Edges::symmetric(0.0, 1.0), Action::ConvertPalette(k));
            // `.light-background[data-checked=true]`.
            b.ui.style_mut(search).background = Some([0.0, 0.0, 0.0, 0.5]);
            b.title(Key::Icon(base), "Convert query results to an explicit tool palette");
        }
        b.icon(bar, base + 1, Some(icons::BIN), "", Edges::symmetric(0.0, 1.0), Action::DeletePalette(k));
        b.title(Key::Icon(base + 1), "Delete tool palette");
        let close = b.icon(bar, 2000 + id.ordinal(), Some(icons::CLOSE), "", Edges::symmetric(0.0, 1.0), Action::ClosePanel(id));
        b.key(close, Key::PanelClose(id), Some(Action::ClosePanel(id)));
        b.title(Key::PanelClose(id), "Close panel");

        let body = b.ui.add_box(panel, style! { display: flex; flex-direction: column; flex: 1; min-height: 0; });
        if defined {
            // `ToolQueryWidget`: an `AutocompleteTextInput`, then the errors.
            let wrap = b.ui.add_box(body, style! { display: flex; flex-direction: column; flex-shrink: 0; });
            let input = b.ui.add_box(wrap, style! { display: flex; position: relative; margin-top: 2px; padding: 2px; min-height: 18px; background: #222; font-family: monospace; font-size: 13px; color: white; });
            let fid = FieldId::PaletteQuery(k);
            b.key(input, Key::Field(fid), None);
            let editing = self.field.as_ref().filter(|(f, _)| *f == fid).map(|(_, t)| t);
            let font = style! { font-family: monospace; font-size: 13px; color: white; line-height: 18px; };
            let view = threers::ui::controls::TextView {
                value: &query,
                placeholder: "Enter tool query or drag in tools",
                placeholder_color: [0.67, 0.67, 0.67, 1.0],
                editing,
                caret_on: self.caret_on(),
                font: &font,
                x_offset: 0.0,
                selection: threers::ui::controls::SELECTION,
            };
            threers::ui::controls::text_contents(&mut b.ui, input, view, b.text);
            let errors = b.ui.add_box(wrap, style! { display: flex; flex-direction: column; margin-bottom: 2px; flex-shrink: 0; });
            if let Err(errs) = ng_viewer::tool_query::parse(&query) {
                for e in errs {
                    b.text_box(errors, &e.message, style! { padding: 2px; font-family: sans-serif; font-size: {SMALL}; color: red; });
                }
            }
        }
        // `.neuroglancer-tool-palette-items`.
        let items = b.ui.add_box(body, scroll_style(self.scroll.get(&id).copied().unwrap_or(0.0)));
        b.key(items, Key::Scroll(id), None);
        let list = tools(viewer, &json);
        let mut index = 0;
        while index < list.len() {
            let Some((owner, local)) = viewer.tool_from_palette_json(&list[index]) else {
                index += 1;
                continue;
            };
            match owner {
                ToolOwner::Layer(i) => {
                    // `RenderedLayerGroup`: this layer's run of tools.
                    let group = b.ui.add_box(items, style! { display: flex; flex-direction: column; border-left: 2px solid; flex-shrink: 0; });
                    let gkey = Key::PaletteGroup(k, i);
                    b.key(group, gkey, None);
                    let hovered = b.hover(gkey);
                    b.ui.style_mut(group).border_color = if hovered { [0.8, 0.8, 0.8, 1.0] } else { [24.0 / 255.0, 24.0 / 255.0, 24.0 / 255.0, 1.0] };
                    let header = b.ui.add_box(group, style! { display: flex; align-items: flex-start; border-bottom: 2px solid black; flex-shrink: 0; });
                    b.ui.style_mut(header).background = Some(if hovered { [40.0 / 255.0, 40.0 / 255.0, 40.0 / 255.0, 1.0] } else { [24.0 / 255.0, 24.0 / 255.0, 24.0 / 255.0, 1.0] });
                    let visible = viewer.layers().get(i).is_some_and(|l| l.spec.visible);
                    let eye = b.icon(header, 2200 + 64 * k as u32 + i as u32 % 64, Some(if visible { icons::EYE } else { icons::EYE_CROSSED }), "", Edges::ZERO, Action::ToggleLayerVisible(i));
                    let s = b.ui.style_mut(eye);
                    (s.width, s.height, s.min_width, s.min_height) = (threers::ui::Length::Px(20.0), threers::ui::Length::Px(20.0), threers::ui::Length::Px(20.0), threers::ui::Length::Px(20.0));
                    s.padding = Edges::ZERO;
                    s.align_self = Some(threers::ui::AlignItems::FlexStart);
                    let layer_name = viewer.layers().get(i).map_or(String::new(), |l| l.name().to_owned());
                    // `LayerNameWidget` (`.neuroglancer-layer-side-panel-name`).
                    let name = self.name_field(b, header, FieldId::LayerName(i), &layer_name, "white");
                    // The `<input>`'s 2px of padding before the text.
                    b.ui.style_mut(name).padding.left = 2.0;
                    let content = b.ui.add_box(group, style! { display: flex; flex-direction: column; flex-shrink: 0; });
                    while index < list.len() {
                        match viewer.tool_from_palette_json(&list[index]) {
                            Some((ToolOwner::Layer(j), local)) if j == i => {
                                self.rendered_tool(b, content, viewer, k, index, &list[index], ToolRef { owner: ToolOwner::Layer(i), json: local }, defined);
                                index += 1;
                            }
                            _ => break,
                        }
                    }
                }
                ToolOwner::Global => {
                    self.rendered_tool(b, items, viewer, k, index, &list[index], ToolRef { owner, json: local }, defined);
                    index += 1;
                }
            }
        }
        // `.neuroglancer-tool-palette-drop-zone`.
        b.ui.add_box(items, style! { flex: 1; });
    }

    /// `RenderedTool`: the key binding, then the tool as palettes show it
    /// (`renderInPalette`).
    #[allow(clippy::too_many_arguments)]
    fn rendered_tool(&mut self, b: &mut Builder, parent: NodeId, viewer: &mut Viewer, k: u16, index: usize, json: &Value, tool: ToolRef, defined: bool) {
        let key = Key::PaletteTool(k, index as u16);
        let row = b.ui.add_box(parent, style! { display: flex; align-items: center; padding: 2px 0; flex-shrink: 0; });
        b.key(row, key, None);
        if b.hover(key) {
            b.ui.style_mut(row).background = threers::ui::css::color("#282828");
        }
        b.title(key, tool_title(json, defined));
        b.tool_binding(row, tool.clone(), None);
        let content = b.ui.add_box(row, style! { display: flex; flex: 1; min-width: 0; });
        let slot = crate::rendering::palette_slot(k);
        match (tool.owner, tool.id()) {
            (ToolOwner::Global, "dimension") => {
                let name = tool.json.get("dimension").and_then(Value::as_str).unwrap_or("");
                match viewer.dimensions().names.iter().position(|n| n == name) {
                    Some(dim) => self.dimension_palette_tool(b, content, viewer, dim, k),
                    None => {
                        b.text_box(content, "Loading...", style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
                    }
                }
            }
            (ToolOwner::Layer(i), _) if self.layer_control(b, content, viewer, i, &tool, slot).is_none() => {
                b.text_box(content, "Loading...", style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
            }
            _ => {}
        }
    }

    /// The dimension tool in a palette (`.neuroglancer-position-tool-in-palette`):
    /// the dimension and the playback icon over its plot.
    fn dimension_palette_tool(&mut self, b: &mut Builder, content: NodeId, viewer: &Viewer, dim: usize, k: u16) {
        b.ui.style_mut(content).flex_direction = threers::ui::FlexDirection::Column;
        let top = b.ui.add_box(content, style! { display: flex; align-items: flex-start; flex-shrink: 0; });
        let plot_box = self.dimension_tool_widgets(b, top, content, viewer, dim, 3200 + 16 * k as u32 + dim as u32 % 16);
        let _ = plot_box;
    }
}

impl NgUi {
    /// `PaletteNameWidget` (`.neuroglancer-tool-palette-name`): an
    /// `<input>` filling the line, transparent, borderless.
    fn palette_name(&self, b: &mut Builder, parent: NodeId, k: u16, name: &str, stretch: bool) {
        let id = FieldId::PaletteName(k);
        let node = b.ui.add_box(parent, style! { display: flex; position: relative; align-items: center; flex: 1; width: 0; min-width: 0; height: 17px; padding: 1px 2px; box-sizing: border-box; overflow: hidden; font-family: Arial; font-size: {SMALL}; color: white; });
        // In the list's rows (`align-items: stretch`), the row's height.
        if stretch {
            let s = b.ui.style_mut(node);
            s.height = threers::ui::Length::Auto;
            s.align_self = Some(threers::ui::AlignItems::Stretch);
        }
        b.key(node, Key::Field(id), None);
        b.title(Key::Field(id), "Rename tool palette");
        let editing = self.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t);
        let font = style! { font-family: Arial; font-size: {SMALL}; color: white; };
        self.field_contents(b, node, editing, name, "", &font, 0.0);
    }
}

/// `.neuroglancer-tool-palette-items`: a column scrolling its overflow.
fn scroll_style(top: f32) -> threers::ui::Style {
    let mut s = style! { display: flex; flex-direction: column; flex: 1; flex-basis: 0px; min-height: 0; overflow: hidden; font-family: sans-serif; font-size: {SMALL}; color: white; };
    s.scroll = true;
    s.scroll_top = top;
    s
}

impl NgUi {
    /// The wrench (`MultiToolPaletteDropdownButton`)'s list
    /// (`MultiToolPaletteDropdown`): each palette (shown, name, delete),
    /// then the canned ones not already there.
    pub(crate) fn palette_dropdown(&mut self, b: &mut Builder, root: NodeId, viewer: &Viewer, anchor: threers::ui::Rect, width: f32) {
        let palettes = all(viewer);
        let w = (width / 2.0).min(300.0);
        let dropdown = b.ui.add_box(
            root,
            style! { position: absolute; top: {anchor.y + anchor.height}; display: flex; flex-direction: column; min-width: {w}; border: 1px solid #aaa; background: black; padding: 2px; font-family: sans-serif; font-size: {SMALL}; color: white; },
        );
        // `positionRelativeDropdown` against the button (the icon and its
        // 1px margins): left-aligned if there is more room to its right
        // than its left — measured, as it is written, against the
        // viewport's height — else right-aligned; `min-width` outranks
        // its `max-width`.
        let (left_edge, right_edge) = (anchor.x - 1.0, anchor.x + anchor.width + 1.0);
        let outer = w + 6.0;
        let left = if left_edge < b.viewport_height - right_edge { left_edge } else { right_edge - outer };
        b.ui.style_mut(dropdown).left = threers::ui::Length::Px(left);
        b.key(dropdown, Key::PaletteDropdown, None);
        let item = |b: &mut Builder, key: Key| {
            let li = b.ui.add_box(dropdown, style! { display: flex; align-items: center; padding: 2px; font-family: sans-serif; font-size: {SMALL}; flex-shrink: 0; });
            b.key(li, key, None);
            if b.hover(key) {
                b.ui.style_mut(li).background = threers::ui::css::color("#333");
            }
            li
        };
        let mut seen = vec![];
        for (k, (name, json)) in palettes.iter().enumerate() {
            let k = k as u16;
            seen.push(query_of(json).to_owned());
            let li = item(b, Key::PaletteListItem(k));
            let visible = PanelId::Palette(k).location(viewer).visible;
            // `TrackableBooleanCheckbox`: a native checkbox and its margins.
            let cell = b.ui.add_box(li, style! { display: flex; flex-shrink: 0; margin: 3px 3px 3px 4px; });
            crate::view::checkbox_sized(b, cell, visible, Key::PaletteVisible(k), Action::TogglePanel(PanelId::Palette(k)), 13.0);
            b.title(Key::PaletteVisible(k), if visible { "Hide tool palette" } else { "Show tool palette" });
            self.palette_name(b, li, k, name, true);
            let base = 2600 + 2 * k as u32;
            b.icon(li, base, Some(icons::BIN), "", Edges::symmetric(0.0, 1.0), Action::DeletePalette(k));
            b.title(Key::Icon(base), "Delete tool palette");
        }
        let mut first = true;
        for (c, (name, description, query)) in CANNED.iter().enumerate() {
            if !query.is_empty() && seen.iter().any(|q| q == query) {
                continue;
            }
            if first {
                first = false;
                if !palettes.is_empty() {
                    b.ui.add_box(dropdown, style! { height: 1px; margin: 3px 0; background: #ccc; flex-shrink: 0; });
                }
            }
            let li = item(b, Key::PaletteCanned(c));
            b.actions.insert(Key::PaletteCanned(c), Action::AddPalette { name: (*name).into(), query: (*query).into() });
            b.text_box(li, description.unwrap_or(name), style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
        }
    }
}
