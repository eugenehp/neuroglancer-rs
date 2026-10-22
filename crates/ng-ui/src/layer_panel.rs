//! The layer side panel (`ui/layer_side_panel.ts`): title bar (type, name,
//! pick, pin, delete), tabs (`widget/tab_view.ts`), and the Source tab
//! (`ui/layer_data_sources_tab.ts`, `widget/coordinate_transform.ts`).

use ng_core::js::number_to_string;
use ng_core::si_units::format_scale_with_unit_as_string;
use ng_viewer::{LayerKind, SourceState, Viewer};
use threers::style;
use threers::ui::{Edges, Length, NodeId, Style};

use crate::panels::{layer_type_abbreviation, scroll_style};
use crate::view::{checkbox, Builder};
use crate::{icons, Action, Key, NgUi, PanelId};

/// `10pt`.
const SMALL: f32 = 40.0 / 3.0;

/// A layer type's tabs: ID and label (`UserLayer.tabs`).
pub(crate) fn tabs(layer: &ng_viewer::ManagedLayer) -> &'static [(&'static str, &'static str)] {
    match layer.kind() {
        Some(LayerKind::Segmentation(_)) => &[("source", "Source"), ("rendering", "Render"), ("segments", "Seg."), ("annotations", "Annotations")],
        // `NewUserLayer`: its data sources only.
        None if layer.spec.layer_type == "new" => &[("source", "Source")],
        None if layer.spec.layer_type == "annotation" => &[("source", "Source"), ("rendering", "Rendering"), ("annotations", "Annotations"), ("schema", "Schema")],
        _ => &[("source", "Source"), ("rendering", "Rendering"), ("annotations", "Annotations")],
    }
}

/// `layerTypes` in registration order: type and abbreviation.
pub(crate) const LAYER_TYPES: [(&str, &str); 6] = [("new", "new"), ("auto", "auto"), ("image", "img"), ("segmentation", "seg"), ("annotation", "ann"), ("mesh", "mesh")];
const LAYER_TYPE_ABBREVIATIONS: [&str; 6] = ["new", "auto", "img", "seg", "ann", "mesh"];

/// The selected tab (`tab`, else the first).
pub(crate) fn selected_tab(layer: &ng_viewer::ManagedLayer) -> &'static str {
    let tabs = tabs(layer);
    let want = layer.spec.json.get("tab").and_then(serde_json::Value::as_str);
    tabs.iter().find(|(id, _)| Some(*id) == want).map_or(tabs[0].0, |(id, _)| id)
}

/// Pinned panel `k`'s tab (its `tab`, else the first).
pub(crate) fn pinned_tab(layer: &ng_viewer::ManagedLayer, k: usize) -> &'static str {
    let tabs = tabs(layer);
    let panels = crate::side::pinned_panels(layer);
    let want = panels.get(k).and_then(|p| p.get("tab")).and_then(serde_json::Value::as_str);
    tabs.iter().find(|(id, _)| Some(*id) == want).map_or(tabs[0].0, |(id, _)| id)
}

/// A pinned panel's tab set.
pub(crate) fn set_pinned_tab(viewer: &mut Viewer, i: usize, k: usize, tab: &str) {
    let Some(layer) = viewer.layers().get(i) else { return };
    let mut panels = crate::side::pinned_panels(layer);
    if let Some(serde_json::Value::Object(o)) = panels.get_mut(k) {
        o.insert("tab".into(), tab.into());
    }
    viewer.set_layer_option(i, "panels", Some(serde_json::Value::Array(panels)));
}

/// `UserLayerSidePanelState.pin`: the selected layer's panel stays where
/// it is as one of its layer's (`panels`, its tab with it); the selected
/// layer's panel, left without tabs, hides (and sorts after it).
pub(crate) fn pin(viewer: &mut Viewer) {
    let Some(i) = PanelId::Layer.layer(viewer) else { return };
    let mut location = PanelId::Layer.location(viewer);
    let tab = selected_tab(&viewer.layers()[i]);
    let mut panels = crate::side::pinned_panels(&viewer.layers()[i]);
    let pinned = PanelId::Pinned(i as u16, panels.len() as u16);
    let mut entry = location.to_json(&pinned.default_location());
    entry.insert("tab".into(), tab.into());
    panels.push(serde_json::Value::Object(entry));
    viewer.set_layer_option(i, "panels", Some(serde_json::Value::Array(panels)));
    viewer.set_layer_option(i, "tab", None);
    location.visible = false;
    PanelId::Layer.set_location(viewer, &location);
}

/// `unpin`: pinned panel `k` of layer `i` becomes the selected layer's
/// (where it is, its tab with it); the panel of the layer selected before,
/// if shown, is pinned first.
pub(crate) fn unpin(viewer: &mut Viewer, i: usize, k: usize) {
    if let Some(other) = PanelId::Layer.layer(viewer) {
        if other != i && PanelId::Layer.location(viewer).visible {
            pin(viewer);
        }
    }
    let Some(layer) = viewer.layers().get(i) else { return };
    let location = PanelId::Pinned(i as u16, k as u16).location(viewer);
    let mut panels = crate::side::pinned_panels(layer);
    if k >= panels.len() {
        return;
    }
    let entry = panels.remove(k);
    let tab = entry.get("tab").cloned();
    viewer.set_layer_option(i, "panels", (!panels.is_empty()).then_some(serde_json::Value::Array(panels)));
    viewer.set_layer_option(i, "tab", tab);
    let name = viewer.layers()[i].name().to_owned();
    let mut selected = viewer.ui_state("selectedLayer").and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
    selected.insert("layer".into(), name.into());
    viewer.set_ui_state("selectedLayer", Some(serde_json::Value::Object(selected)));
    PanelId::Layer.set_location(viewer, &location);
}

impl NgUi {
    /// `LayerSidePanel` for the selected layer.
    pub(crate) fn layer_side_panel_body(&mut self, b: &mut Builder, panel: NodeId, viewer: &mut Viewer, id: PanelId) {
        let (i, pinned) = match id {
            PanelId::Pinned(i, k) => (i as usize, Some(k as usize)),
            _ => match PanelId::Layer.layer(viewer) {
                Some(i) => (i, None),
                None => return,
            },
        };
        let Some(layer) = viewer.layers().get(i) else { return };
        self.layer_panel = id;
        // The title bar's icons, apart per panel.
        let icon_base = 50000 + 8 * id.ordinal();
        let visible = layer.spec.visible;
        let segmentation = matches!(layer.kind(), Some(LayerKind::Segmentation(_)));
        let picks = layer.spec.json.get("pick").and_then(serde_json::Value::as_bool).unwrap_or(true);
        // The title bar, green while the panel follows the selection
        // (`[data-neuroglancer-layer-panel-pinned=false]`).
        let bar = b.ui.add_box(
            panel,
            style! { display: flex; align-items: center; font-family: sans-serif; font-size: {SMALL}; background: #393; padding: 2px; color: white; flex-shrink: 0; },
        );
        // Pinned: the side panels' blue.
        if pinned.is_some() {
            b.ui.style_mut(bar).background = threers::ui::css::color("#03c");
        }
        // `LayerTypeWidget`: a select as wide as its abbreviation.
        let abbreviation = layer_type_abbreviation(layer);
        let type_font = style! { font-family: sans-serif; font-size: {SMALL}; font-weight: bold; color: white; };
        let w = b.width(abbreviation, &type_font) + 6.0;
        let select = b.text_box(
            bar,
            abbreviation,
            style! { ..type_font.clone(); width: {w.ceil()}; box-sizing: border-box; padding: 0 3px; margin-right: 5px; background: rgba(0, 0, 0, 0.5); flex-shrink: 0; },
        );
        // Its options: the registered types by abbreviation; choosing one
        // runs `changeLayerType`.
        b.key(select, Key::Control(i, "type"), Some(Action::Repaint));
        b.title(Key::Control(i, "type"), "Change layer type");
        let current = LAYER_TYPES.iter().find(|(_, a)| *a == abbreviation).map_or("new", |(_, a)| *a);
        b.controls.insert(
            Key::Control(i, "type"),
            crate::rendering::Control::Select { layer: i, key: "type", options: &LAYER_TYPE_ABBREVIATIONS, value: current, default: "" },
        );
        // `LayerNameWidget`, struck through while the layer is hidden.
        let name = self.name_field(b, bar, crate::FieldId::LayerName(i), layer.name(), "white");
        {
            let s = b.ui.style_mut(name);
            s.height = Length::Px(15.0);
            s.line_height = None;
            s.padding = Edges::symmetric(1.0, 2.0);
            s.width = Length::Px(0.0);
            s.line_through = !visible;
        }
        // The input's line box is the font's (`line-height: normal`).
        for c in b.ui.children(name).to_vec() {
            let s = b.ui.style_mut(c);
            s.line_through = !visible;
            s.line_height = None;
            for t in b.ui.children(c).to_vec() {
                let s = b.ui.style_mut(t);
                s.line_through = !visible;
                s.line_height = None;
            }
        }
        if segmentation {
            let pick = b.icon(bar, icon_base, Some(icons::CURSOR), "", Edges::symmetric(0.0, 1.0), Action::TogglePick(i));
            // `.light-background[data-checked=true]`: dark while picking.
            if picks && !b.hover(Key::Icon(icon_base)) {
                b.ui.style_mut(pick).background = Some([0.0, 0.0, 0.0, 0.5]);
            }
        }
        let pin_action = match pinned {
            Some(k) => Action::UnpinLayerPanel(i, k),
            None => Action::PinLayerPanel,
        };
        let pin = b.icon(bar, icon_base + 1, None, "📌\u{fe0e}", Edges::symmetric(0.0, 1.0), pin_action);
        b.title(Key::Icon(icon_base + 1), if pinned.is_some() { "Unpin panel to this layer" } else { "Pin panel to this layer" });
        b.ui.style_mut(pin).min_height = Length::Px(20.0);
        // `.light-background[data-checked=true]` while pinned.
        if pinned.is_some() {
            b.ui.style_mut(pin).background = Some([0.0, 0.0, 0.0, 0.5]);
        }
        // Chrome's line box here is the emoji fallback font's (20px), with
        // the run 4px down rather than centred.
        let glyph = b.ui.children(pin)[0];
        let s = b.ui.style_mut(glyph);
        s.position = threers::ui::Position::Relative;
        s.top = Length::Px(1.0);
        b.icon(bar, icon_base + 2, Some(icons::BIN), "", Edges::symmetric(0.0, 1.0), Action::DeleteLayer(i));
        b.title(Key::Icon(icon_base + 2), "Delete layer");
        b.icon(bar, icon_base + 3, Some(icons::CLOSE), "", Edges::symmetric(0.0, 1.0), Action::ClosePanel(id));
        b.title(Key::Icon(icon_base + 3), "Close panel");

        // `TabView`.
        let view = b.ui.add_box(panel, style! { display: flex; flex-direction: column; flex: 1; min-height: 0; position: relative; font-family: sans-serif; font-size: {SMALL}; color: white; });
        let tab_bar = b.ui.add_box(view, style! { display: flex; align-items: flex-start; background: #333; border-width: 0 0 1px 0; border-color: white; flex-shrink: 0; });
        let selected = match pinned {
            Some(k) => pinned_tab(layer, k),
            None => selected_tab(layer),
        };
        let panel_id = id;
        for (k, (id, label)) in tabs(layer).iter().enumerate() {
            let key = Key::LayerTab(panel_id, k);
            let is = *id == selected;
            let color = if b.hover(key) { "#daa520" } else { "white" };
            let node = b.text_box(
                tab_bar,
                label,
                style! {
                    font-family: sans-serif;
                    font-size: {SMALL};
                    font-weight: bold;
                    color: {color};
                    border-width: {if is { Edges::new(1.0, 1.0, 1.0, 1.0) } else { Edges::new(1.0, 1.0, 0.0, 1.0) }};
                    border-color: {if is { "white" } else { "transparent" }};
                    padding: {Edges::new(1.0, 2.0, if is { 1.0 } else { 0.0 }, 2.0)};
                    margin: {Edges::new(0.0, 4.0, if is { -1.0 } else { 0.0 }, 0.0)};
                    background: {is.then_some([0.0, 0.0, 0.0, 1.0])};
                    flex-shrink: 0;
                },
            );
            if is {
                // `border-bottom: 1px solid black` over the bar's white line.
                b.ui.style_mut(node).border_color = [1.0; 4];
                b.ui.add_box(node, style! { position: absolute; left: 0; right: 0; bottom: -1px; height: 1px; background: black; });
                b.ui.style_mut(node).position = threers::ui::Position::Relative;
            }
            let action = match pinned {
                Some(p) => Action::SelectPinnedTab(i, p, id),
                None => Action::SelectLayerTab(i, id),
            };
            b.key(node, key, Some(action));
        }
        let content = b.ui.add_box(view, style! { display: flex; flex-direction: column; flex: 1; flex-basis: 0px; height: 0; min-height: 0; padding: 2px; });
        match selected {
            "source" => self.source_tab(b, content, viewer, i),
            "rendering" if viewer.layers()[i].spec.layer_type == "annotation" => self.annotation_rendering_tab(b, content, viewer, i),
            "rendering" if !segmentation => self.image_rendering_tab(b, content, viewer, i),
            "rendering" => self.segmentation_rendering_tab(b, content, viewer, i),
            "segments" => self.segments_tab(b, content, viewer, i),
            "annotations" => self.annotations_tab(b, content, viewer, i),
            "schema" => self.schema_tab(b, content, viewer, i),
            _ => {}
        }
    }

    /// `LayerDataSourcesTab`.
    fn source_tab(&mut self, b: &mut Builder, content: NodeId, viewer: &mut Viewer, i: usize) {
        let scroll = b.ui.add_box(content, scroll_style(self.scroll.get(&self.layer_panel).copied().unwrap_or(0.0)));
        b.key(scroll, Key::Scroll(self.layer_panel), None);
        let views = viewer.layers()[i].source_views();
        // A layer without sources still has its (empty) URL to type in.
        if views.is_empty() {
            let block = b.ui.add_box(scroll, style! { display: flex; flex-direction: column; flex-shrink: 0; });
            self.source_url(b, block, i, "");
        }
        for (si, source) in views.into_iter().enumerate() {
            let block = b.ui.add_box(scroll, style! { display: flex; flex-direction: column; flex-shrink: 0; });
            self.source_url(b, block, i, &source.url);
            match source.state {
                SourceState::Loading => {
                    b.text_box(block, "Loading", style! { color: #ccc; font-family: sans-serif; font-size: {SMALL}; });
                }
                SourceState::Failed => {
                    for m in &viewer.layers()[i].messages {
                        b.text_box(block, m, style! { color: red; font-family: sans-serif; font-size: {SMALL}; white-space: normal; word-wrap: break-word; });
                    }
                }
                SourceState::Loaded => {
                    // "Enable default subsource set", then each subsource.
                    let line = b.ui.add_box(block, style! { display: flex; align-items: flex-end; margin-bottom: 4px; flex-shrink: 0; });
                    checkbox(b, line, source.enable_default_subsources, Key::SourceDefaults(i, si), Action::ToggleDefaultSubsources(i, si), Edges::new(3.0, 3.0, 3.0, 4.0));
                    b.text_box(line, "Enable default subsource set", style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
                    for (k, sub) in source.subsources.iter().enumerate() {
                        let line = b.ui.add_box(block, style! { display: flex; align-items: flex-end; flex-shrink: 0; });
                        checkbox(b, line, sub.enabled, Key::Subsource(i, si, k), Action::ToggleSubsource(i, si, k), Edges::new(3.0, 3.0, 3.0, 4.0));
                        let struck = !sub.enabled || viewer.layers()[i].spec.layer_type == "new";
                        if !sub.id.is_empty() {
                            // The brackets are `::before` and `::after`:
                            // runs of their own, shaped apart.
                            let id = b.ui.add_box(line, style! { display: flex; align-items: flex-end; margin-right: 4px; flex-shrink: 0; });
                            for t in ["[", sub.id.as_str(), "]"] {
                                let run = b.text_box(id, t, style! { font-family: sans-serif; font-size: {SMALL}; font-weight: bold; color: #0ff; });
                                b.ui.style_mut(run).line_through = struck;
                                for c in b.ui.children(run).to_vec() {
                                    b.ui.style_mut(c).line_through = struck;
                                }
                            }
                        }
                        let t = b.text_box(line, &sub.description, style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
                        // Struck through when off, or not used (a `new` layer
                        // uses none yet).
                        if struck {
                            b.ui.style_mut(t).line_through = true;
                            for c in b.ui.children(t).to_vec() {
                                b.ui.style_mut(c).line_through = true;
                            }
                        }
                    }
                    if let Some(t) = &source.transform {
                        // `canChangeModelSpaceRank`: local annotations.
                        transform_grid(b, block, t, source.url == "local://annotations");
                    }

                }
            }
        }
        // Add another data source.
        if !viewer.layers()[i].spec.sources.is_empty() {
            let add = b.ui.add_box(scroll, style! { display: flex; flex-shrink: 0; });
            b.icon(add, 2600, Some(icons::PLUS), "", Edges::symmetric(0.0, 1.0), Action::Unported("Add data source"));
        }
        // `NewUserLayer`: the detected type to become, on #ff9 bars at the
        // bottom (`.neuroglancer-layer-data-sources-tab-type-detection`).
        let layer = &viewer.layers()[i];
        let editing = matches!(&self.field, Some((crate::FieldId::SourceUrl(f), _)) if *f == i);
        if layer.spec.layer_type == "new" && !editing {
            if let Some(t) = viewer.detected_layer_type(i) {
                let options: &[&str] = if t == "image" { &[t, "multi-channel image"] } else { &[t] };
                for (k, label) in options.iter().enumerate() {
                    let key = Key::Control(i, if k == 0 { "createAs" } else { "createAsMulti" });
                    let bg = if b.hover(key) { "#ffc" } else { "#ff9" };
                    let bar = b.ui.add_box(content, style! { display: flex; justify-content: center; background: {bg}; flex-shrink: 0; });
                    if k > 0 {
                        b.ui.style_mut(bar).margin.top = 0.5 * SMALL;
                    }
                    let font = style! { font-family: sans-serif; font-size: {SMALL}; color: black; };
                    b.text_box(bar, "Create as ", font.clone());
                    b.text_box(bar, label, style! { ..font.clone(); font-weight: bold; });
                    b.text_box(bar, " layer", font);
                    b.key(bar, key, Some(Action::ChangeLayerType(i, t)));
                }
            }
        }
    }

    /// A source's URL (`MultilineAutocomplete`): wrapping after each `|`,
    /// `:`, `/`, `_` run (the syntax highlighter's `<wbr>`s); editable, with
    /// its completions below while focused.
    fn source_url(&mut self, b: &mut Builder, block: NodeId, i: usize, url: &str) {
        let id = crate::FieldId::SourceUrl(i);
        let editing = self.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t.clone());
        // Positioned: its dropdown paints over what follows.
        // A block, so the wrapped URL sets its height.
        let node = b.ui.add_box(block, style! { position: relative; margin-top: 2px; padding: 2px; min-height: 15px; background: #222; flex-shrink: 0; });
        b.key(node, Key::Field(id), None);
        let font = style! { font-family: monospace; font-size: 13px; color: white; };
        match &editing {
            Some(field) => {
                let view = threers::ui::controls::TextView {
                    value: url,
                    placeholder: "",
                    placeholder_color: [0.5; 4],
                    editing: Some(field),
                    caret_on: self.caret_on(),
                    font: &font,
                    x_offset: 2.0,
                    selection: threers::ui::controls::SELECTION,
                };
                threers::ui::controls::text_contents(&mut b.ui, node, view, b.text);
                let completions = crate::sources::completions(&field.text);
                if !completions.is_empty() {
                    self.source_completions(b, node, &completions);
                }
            }
            None => {
                b.text_box(node, &url_with_breaks(url), style! { ..font.clone(); white-space: normal; word-wrap: break-word; });
            }
        }
    }

    /// `.neuroglancer-multiline-autocomplete-dropdown`: each completion and
    /// its description (italic, #f9f), every other one on #2b2b2b, the
    /// active one on #666.
    fn source_completions(&self, b: &mut Builder, input: NodeId, completions: &[(&'static str, &'static str)]) {
        let list = b.ui.add_box(
            input,
            style! { position: absolute; left: 0; right: -2px; top: 100%; display: flex; flex-direction: column; border: 1px solid #aaa; background: #181818; },
        );
        let font = style! { font-family: monospace; font-size: 13px; color: white; };
        for (k, (value, description)) in completions.iter().enumerate() {
            let row = b.ui.add_box(list, style! { display: flex; flex-direction: column; flex-shrink: 0; });
            let key = Key::Completion(k);
            b.key(row, key, Some(Action::Complete(value.to_string())));
            let bg = if self.completion_active == Some(k) {
                Some("#666")
            } else if k % 2 == 1 {
                Some("#2b2b2b")
            } else {
                None
            };
            b.ui.style_mut(row).background = bg.and_then(threers::ui::css::color);
            b.text_box(row, value, font.clone());
            b.text_box(row, description, style! { ..font.clone(); font-style: italic; color: #f9f; });
        }
    }
}

/// The syntax highlighter's pieces (`/\|?[^|:/_]*(?:[:/_]+)?/g`), joined by
/// zero-width spaces where a line may break.
pub(crate) fn url_with_breaks(url: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = url.chars().collect();
    let mut i = 0;
    let mut first = true;
    while i < chars.len() {
        let start = i;
        if chars[i] == '|' {
            i += 1;
        }
        while i < chars.len() && !matches!(chars[i], '|' | ':' | '/' | '_') {
            i += 1;
        }
        while i < chars.len() && matches!(chars[i], ':' | '/' | '_') {
            i += 1;
        }
        if i == start {
            i += 1;
        }
        if !first {
            out.push('\u{200b}');
        }
        first = false;
        out.extend(&chars[start..i]);
    }
    out
}

/// `CoordinateSpaceTransformWidget`: a grid of the source's dimensions
/// (columns) against the output's (rows), placed on its tracks.
fn transform_grid(b: &mut Builder, parent: NodeId, t: &ng_core::transform::CoordinateSpaceTransform, mutable_source_rank: bool) {
    let rank = t.rank;
    let (input, output) = (&t.input_space, &t.output_space);
    // Columns: the label (`min-content`: a 15 px vertical line + 2), then
    // `1fr` for output names, output scales and each source dimension plus
    // the translation; the scale column holds its inputs' minimum.
    const WIDTH: f32 = 296.0;
    const LABEL: f32 = 17.0;
    const SCALE_MIN: f32 = 48.98;
    let n = (rank.max(1) + 3) as f32;
    let share = (WIDTH - LABEL) / n;
    let (scale_w, fr) = if share < SCALE_MIN { (SCALE_MIN, (WIDTH - LABEL - SCALE_MIN) / (n - 1.0)) } else { (share, share) };
    let col_names = LABEL;
    let col_scales = col_names + fr;
    let col_dim = |k: usize| col_scales + scale_w + k as f32 * fr;
    // Rows: label, names, lower, upper, scales (+3), each output dimension,
    // the last (add dimension, reset).
    let (r_label, r_names, r_lower, r_upper, r_scales) = (0.0, 15.0, 30.0, 45.0, 60.0);
    // An output row grows to hold its scale's suggestion (19 + 2 + 36 + 2
    // and its border, against the cells' 49).
    let row_h: Vec<f32> = (0..rank)
        .map(|o| if inferred_output_scale(t, o).is_some_and(|(s, u)| s != output.scales[o] || u != output.units[o]) { 61.0 } else { 49.0 })
        .collect();
    let r_out = |k: usize| 84.0 + row_h[..k].iter().sum::<f32>();
    // The extend cell: `+S` (a mutable source rank) above `+V`.
    let extend_h = if mutable_source_rank { 36.0 } else { 18.0 };
    let height = r_out(rank) + extend_h;
    let grid = b.ui.add_box(parent, style! { position: relative; height: {height}; flex-shrink: 0; });
    let label = style! { font-family: sans-serif; font-size: {SMALL}; font-style: italic; color: #fcc; };
    let place = |b: &mut Builder, x: f32, y: f32, w: f32, h: f32, style: Style| -> NodeId {
        b.ui.add_box(grid, style! { ..style; position: absolute; left: {x}; top: {y}; width: {w}; height: {h}; })
    };
    // "Source dimensions" over the source columns.
    let l = label.clone();
    let n_src = rank.max(1);
    let node = place(b, col_dim(0), r_label, n_src as f32 * fr, 15.0, style! { ..l.clone(); display: flex; justify-content: center; });
    b.ui.add_text(node, "Source dimensions");
    for (text, y, h) in [("Lower", r_lower, 15.0), ("Upper", r_upper, 15.0), ("Scale", r_scales, 21.0)] {
        let node = place(b, 0.0, y, col_dim(0) - 4.0, h, style! { ..l.clone(); display: flex; justify-content: flex-end; align-items: center; });
        b.ui.add_text(node, text);
    }
    // The vertical labels: "Translation" (bottom of the header, centred in
    // its column) and "Output dimensions" (centred on the output rows).
    let node = place(b, col_dim(rank) + (fr - 15.0) / 2.0, r_label, 15.0, r_scales + 24.0 - 2.0, style! { ..l.clone(); display: flex; flex-direction: column; justify-content: flex-end; writing-mode: vertical-lr; });
    b.ui.add_text(node, "Translation");
    let node = place(b, 0.0, r_out(0), LABEL - 2.0, r_out(rank) - r_out(0), style! { ..l.clone(); display: flex; flex-direction: column; justify-content: center; writing-mode: vertical-lr; });
    b.ui.add_text(node, "Output dimensions");
    let mono = style! { font-family: monospace; font-size: {SMALL}; color: white; };
    let bounds_font = style! { font-family: monospace; font-size: 12px; color: #aff; };
    let field = |b: &mut Builder, parent: NodeId, text: &str, color: &str, right: bool| {
        let node = b.ui.add_box(
            parent,
            style! { ..mono.clone(); display: flex; align-items: center; justify-content: {if right { threers::ui::JustifyContent::FlexEnd } else { threers::ui::JustifyContent::FlexStart }}; background: black; padding: 2px; height: 15px; flex-shrink: 0; color: {color}; overflow: hidden; },
        );
        b.ui.add_text(node, text);
        node
    };
    for k in 0..rank {
        let x = col_dim(k);
        let name = input.names.get(k).cloned().filter(|s| !s.is_empty()).unwrap_or_else(|| k.to_string());
        let node = place(b, x, r_names, fr, 15.0, style! { ..mono.clone(); display: flex; justify-content: center; color: #ff6; });
        b.ui.add_text(node, name);
        let (lo, hi) = format_bounds(input.bounds.lower_bounds[k], input.bounds.upper_bounds[k], input.bounds.voxel_center_at_integer_coordinates[k]);
        for (text, y) in [(lo, r_lower), (hi, r_upper)] {
            let node = place(b, x, y, fr, 15.0, style! { ..bounds_font.clone(); display: flex; justify-content: flex-end; align-items: flex-start; padding: 0 2px; box-sizing: border-box; });
            b.ui.add_text(node, text);
        }
        let cell = place(b, x, r_scales, fr, 21.0, style! { display: flex; flex-direction: column; border: 1px solid #333; box-sizing: border-box; });
        field(b, cell, &format_scale_with_unit_as_string(input.scales[k], &input.units[k], false), "white", true);
    }
    for (o, &h) in row_h.iter().enumerate() {
        let y = r_out(o);
        // Name and bounds.
        let cell = place(b, col_names, y, fr - 3.0, h, style! { display: flex; flex-direction: column; border: 1px solid #333; box-sizing: border-box; });
        field(b, cell, &output.names[o], "#ff6", false);
        let (lo, hi) = format_bounds(output.bounds.lower_bounds[o], output.bounds.upper_bounds[o], output.bounds.voxel_center_at_integer_coordinates[o]);
        for text in [lo, hi] {
            b.text_box(cell, &text, style! { ..bounds_font.clone(); display: flex; justify-content: flex-end; padding: 0 2px; height: 14px; line-height: 14px; flex-shrink: 0; });
        }
        let cell = place(b, col_scales, y, scale_w - 3.0, h, style! { display: flex; flex-direction: column; border: 1px solid #333; box-sizing: border-box; });
        field(b, cell, &format_scale_with_unit_as_string(output.scales[o], &output.units[o], false), "white", true);
        // `getInferredOutputScale`, offered when it differs: an arrow over
        // the scale (the two wrap in the cell's width).
        if let Some((scale, unit)) = inferred_output_scale(t, o).filter(|(s, u)| *s != output.scales[o] || *u != output.units[o]) {
            let text = format_scale_with_unit_as_string(scale, &unit, false);
            let key = Key::Icon(2620 + o as u32);
            let hover = b.hover(key);
            let color = if hover { "white" } else { "#aaa" };
            let suggestion = b.ui.add_box(cell, style! { display: flex; flex-direction: column; align-self: flex-end; margin: 2px; border: 1px solid #333; border-radius: 20%; flex-shrink: 0; width: {scale_w - 3.0 - 2.0 - 4.0}; box-sizing: border-box; color: {color}; });
            if hover {
                let st = b.ui.style_mut(suggestion);
                st.background = threers::ui::css::color("#db4437");
                st.border_color = [0.0; 4];
            }
            b.ui.add_path(suggestion, style! { width: 16px; height: 16px; margin-bottom: 3px; flex-shrink: 0; color: {color}; }, icons::ARROW_UP.paths.iter().map(|p| p.to_string()).collect(), [0.0, 0.0, 24.0, 24.0], Some(2.0), false);
            b.text_box(suggestion, &text, style! { font-family: monospace; font-size: {SMALL}; color: {color}; });
            let title = format!("Change coordinates of {} dimension {} to inferred scale of {text}", if output.names[o].ends_with('\'') { "local" } else { "global" }, output.names[o]);
            b.key(suggestion, key, Some(Action::Unported("Change coordinates to the inferred scale")));
            b.title(key, title);
        }
        for k in 0..=rank {
            let coefficient = t.transform[k * (rank + 1) + o];
            let (x, w) = if k == rank { (col_dim(k) + 3.0, fr - 3.0) } else { (col_dim(k), fr) };
            let cell = place(b, x, y, w, h, style! { display: flex; flex-direction: column; justify-content: center; border: 1px solid #333; box-sizing: border-box; background: black; });
            let f = field(b, cell, &number_to_string(coefficient), "white", false);
            b.ui.style_mut(f).flex_grow = 1.0;
        }
    }
    // Add an output dimension.
    if mutable_source_rank {
        let add = place(b, col_names, r_out(rank), fr, 18.0, style! { display: flex; });
        b.icon(add, 2611, Some(icons::PLUS), "S", Edges::symmetric(0.0, 1.0), Action::Unported("Extend to additional source dimension"));
        b.title(Key::Icon(2611), "Extend to additional source dimension");
    }
    let add = place(b, col_names, r_out(rank) + extend_h - 18.0, fr, 18.0, style! { display: flex; });
    b.icon(add, 2610, Some(icons::PLUS), "V", Edges::symmetric(0.0, 1.0), Action::Unported("Add output dimension"));
    b.title(Key::Icon(2610), "Embed in additional output dimension");
}

/// `getInferredOutputScale`: output dimension `o`'s scale from the one
/// source dimension it depends on (its coefficient's magnitude times that
/// dimension's scale).
fn inferred_output_scale(t: &ng_core::transform::CoordinateSpaceTransform, o: usize) -> Option<(f64, String)> {
    let rank = t.rank;
    let inputs: Vec<usize> = (0..rank).filter(|&k| t.transform[k * (rank + 1) + o] != 0.0).collect();
    let [k] = inputs[..] else { return None };
    let coefficient = t.transform[k * (rank + 1) + o].abs();
    Some((t.input_space.scales[k] * coefficient, t.input_space.units[k].clone()))
}

/// `formatBounds`.
fn format_bounds(lower: f64, upper: f64, centers_at_integers: bool) -> (String, String) {
    let shift = if centers_at_integers { 0.5 } else { 0.0 };
    let (lower, upper) = (lower + shift, upper + shift);
    let lo = if lower == f64::NEG_INFINITY { "(-∞,".to_owned() } else { format!("[{},", number_to_string(lower.floor())) };
    let hi = if upper == f64::INFINITY { "+∞)".to_owned() } else { format!("{})", number_to_string(upper.floor())) };
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_break_where_neuroglancer_puts_wbr() {
        assert_eq!(
            url_with_breaks("precomputed://http://127.0.0.1:8792/data/image"),
            "precomputed://\u{200b}http://\u{200b}127.0.0.1:\u{200b}8792/\u{200b}data/\u{200b}image"
        );
        assert_eq!(format_bounds(0.0, 40.0, false), ("[0,".into(), "40)".into()));
    }
}
