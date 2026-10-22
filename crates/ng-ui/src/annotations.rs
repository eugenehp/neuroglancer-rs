//! A layer's Annotations tab (`ui/annotations.ts` `AnnotationLayerView`):
//! the annotation colour, the list's header (each global dimension's name
//! and scale) and its entries. The annotations listed are the data-bounds
//! boxes of the layer's sources (`makeDataBoundsBoundingBoxAnnotationSet`).

use ng_core::si_units::format_scale_with_unit_as_string;
use ng_viewer::Viewer;
use threers::style;
use threers::ui::{Edges, NodeId};

use crate::panels::scroll_style;
use crate::view::Builder;
use crate::{Action, ColorSetting, Key, NgUi};

/// `10pt`.
const SMALL: f32 = 40.0 / 3.0;

/// The list's columns in `ch` of its font: the symbol, each dimension
/// (`setColumnWidth`: the widest of its header, `name (scale)`, and its
/// coordinates, plus 2), the delete button.
pub(crate) fn column_widths(headers: &[(String, String)], rows: &[Vec<f64>]) -> Vec<usize> {
    headers
        .iter()
        .enumerate()
        .map(|(i, (name, scale))| {
            let header = name.chars().count() + scale.chars().count() + 3;
            let coords = rows.iter().map(|r| coordinate_text(r[i]).len()).max().unwrap_or(0);
            header.max(coords) + 2
        })
        .collect()
}

/// A coordinate as listed: `Math.floor`, as an integer.
fn coordinate_text(v: f64) -> String {
    if v.is_finite() {
        ng_core::js::number_to_string(v.floor())
    } else {
        String::new()
    }
}

impl NgUi {
    /// `AnnotationLayerView` of annotation layer `i`: the related-segment
    /// filters (`initializeAnnotationLayerViewTab`), the toolbox (the
    /// annotation tools, help, previous and next), the header and the list
    /// of its annotations.
    fn annotation_layer_tab(&mut self, b: &mut Builder, content: NodeId, viewer: &mut Viewer, i: usize) {
        let dims = viewer.dimensions().clone();
        let layer = &viewer.layers()[i];
        let json = layer.spec.json.clone();
        let entries = layer.annotation_list(dims.rank()).unwrap_or_default();
        let selected = self.selected_annotation(viewer, i);
        let layer = &viewer.layers()[i];
        let shown = layer.annotation_dimensions().unwrap_or_default();
        let font = style! { font-family: sans-serif; font-size: {SMALL}; color: white; };
        let readonly = layer.has_spatial_annotations();
        let relationships = layer.annotation_relationships();

        // A chunked source's spacing (`renderScaleControls`): in
        // cross-sections, then in 3D.
        if readonly {
            for (projection, prompt, key, json_key) in [(false, "Spacing (cross section)", "annotationSpacingCrossSection", "crossSectionAnnotationSpacing"), (true, "Spacing (projection)", "annotationSpacingProjection", "projectionAnnotationSpacing")] {
                let layer = &viewer.layers()[i];
                let target = layer.annotation_spacing_target(projection);
                crate::rendering::render_scale_widget_with_prompt(b, content, i, key, layer.annotation_histogram(projection), target, "px", prompt, font.clone());
                b.controls.insert(Key::Control(i, key), crate::rendering::Control::RenderScale { layer: i, key: json_key, target, origin: ng_core::render_scale::ORIGIN, invert_wheel: false });
            }
        }

        // "Ignore null related segment filter" (`ignoreNullSegmentFilter`,
        // true by default), its checkbox after the text.
        // (The checkbox, inline and baseline-aligned after the text, makes
        // the label 19 px, the text 4 px down.)
        let row = b.ui.add_box(content, style! { display: flex; align-items: flex-start; flex-shrink: 0; });
        let text = b.text_box(row, "Ignore null related segment filter", font.clone());
        b.ui.style_mut(text).margin.top = 4.0;
        let ignore = json.get("ignoreNullSegmentFilter").and_then(serde_json::Value::as_bool).unwrap_or(true);
        let toggle = Action::SetLayerOption(i, "ignoreNullSegmentFilter".into(), ignore.then_some(false.into()));
        let checkbox = threers::ui::controls::checkbox(&mut b.ui, row, ignore, 13.0, threers::ui::controls::BORDER, Edges::new(3.0, 3.0, 3.0, 4.0));
        b.key(checkbox, Key::Control(i, "ignoreNullSegmentFilter"), Some(toggle.clone()));
        b.key(row, Key::Control(i, "ignoreNullSegmentFilterRow"), Some(toggle));
        b.title(Key::Control(i, "ignoreNullSegmentFilterRow"), "Display all annotations if filtering by related segments is enabled but no segments are selected");

        // Each relationship (`LinkedSegmentationLayerWidget`): show matches
        // only, the name, the linked layer.
        if relationships.iter().any(|r| r == "segments") {
            // (A 21 px label, its text 4 px down.)
            let row = b.ui.add_box(content, style! { display: flex; align-items: center; flex-shrink: 0; margin-top: 2px; });
            let filter = json.get("filterBySegmentation").and_then(serde_json::Value::as_array).is_some_and(|a| a.iter().any(|v| v == "segments"));
            let checkbox = threers::ui::controls::checkbox(&mut b.ui, row, filter, 13.0, threers::ui::controls::BORDER, Edges::new(3.0, 3.0, 3.0, 4.0));
            b.key(checkbox, Key::Control(i, "filterBySegmentation"), Some(Action::Unported("Filter by related segments")));
            b.text_box(row, "segments", font.clone());
            let linked = json.get("linkedSegmentationLayer").and_then(|v| v.get("segments")).and_then(serde_json::Value::as_str).unwrap_or("").to_owned();
            let names: Vec<&'static str> = std::iter::once("").chain(viewer.layers().iter().filter(|l| matches!(l.kind(), Some(ng_viewer::LayerKind::Segmentation(_)))).map(|l| crate::shader_ui::intern(l.name()))).collect();
            let current = crate::shader_ui::intern(&linked);
            let sel = crate::rendering::select(b, row, current, &names, Key::Control(i, "linkedSegmentationLayer"), Action::Repaint);
            b.ui.style_mut(sel).margin.left = 0.0;
            if let Some(spacer) = b.ui.children(row).iter().rev().nth(1).copied() {
                b.ui.style_mut(spacer).flex_grow = 0.0;
            }
        }

        // The toolbox: the tools (`PlacePointTool` …), help, then the
        // previous and next tool buttons.
        // Hidden for a read-only source (`mutableControls`, `navRow`).
        let toolbox = b.ui.add_box(content, style! { display: flex; align-items: stretch; flex-shrink: 0; });
        if readonly {
            b.ui.style_mut(toolbox).display = threers::ui::Display::None;
        }
        // The tools fill the row; help sits at its end (`margin-left: auto`).
        let tools = b.ui.add_box(toolbox, style! { display: flex; align-items: center; flex: 1; });
        let tool_ids = ["annotatePoint", "annotateBoundingBox", "annotateLine", "annotateSphere", "annotatePolyline"];
        for (k, (icon, title)) in [("⚬", "Annotate point"), ("❑", "Annotate bounding box"), ("ꕹ", "Annotate line"), ("◎", "Annotate ellipsoid"), ("⤤", "Annotate polyline")].into_iter().enumerate() {
            // `layer.tool.value = new Place…Tool(…)`: the layer's `tool`.
            b.icon(tools, 2900 + k as u32, None, icon, Edges::symmetric(0.0, 1.0), Action::SetLayerOption(i, "tool".into(), Some(tool_ids[k].into())));
            b.title(Key::Icon(2900 + k as u32), title);
        }
        b.ui.add_box(tools, style! { flex: 1; });
        b.icon(tools, 2905, Some(crate::icons::HELP), "", Edges::symmetric(0.0, 1.0), Action::OpenUrl("https://neuroglancer-docs.web.app/user-guide/annotations.html".into()));
        b.title(Key::Icon(2905), "The left icons allow you to select the type of the annotation. Color and other display settings are available in the 'Rendering' tab. Click for documentation.");
        let nav = b.ui.add_box(toolbox, style! { display: flex; align-items: center; });
        for (k, (tool, label, title)) in [("selectPreviousAnnotation", "\u{25c0} Prev", "Select previous annotation (click to bind key)"), ("selectNextAnnotation", "Next \u{25b6}", "Select next annotation (click to bind key)")].into_iter().enumerate() {
            let button = b.ui.add_box(nav, style! { display: flex; align-items: center; });
            b.tool_binding(button, crate::tools::ToolRef::layer(i, tool), Some(title));
            b.text_box(button, label, font.clone());
            let _ = k;
        }

        // The header and the list (`makeAnnotationListElement`).
        let headers: Vec<(String, String)> = shown.iter().map(|&d| (dims.names[d].clone(), format_scale_with_unit_as_string(dims.scales[d], &dims.units[d], true))).collect();
        let rows: Vec<Vec<f64>> = entries.iter().flat_map(|e| e.rows.iter().map(|r| shown.iter().map(|&d| r[d].unwrap_or(f64::NAN)).collect::<Vec<f64>>())).collect();
        let widths = column_widths(&headers, &rows);
        let ch = b.ch("sans-serif", SMALL);
        let mono = b.ch("monospace", SMALL);
        let col = |w: usize| crate::view::lu(w as f32 * ch);
        let header = b.ui.add_box(content, style! { display: flex; padding-bottom: 2px; flex-shrink: 0; });
        b.ui.add_box(header, style! { width: {col(2)}; flex-shrink: 0; });
        for (k, (name, scale)) in headers.iter().enumerate() {
            let cell = b.ui.add_box(header, style! { display: flex; justify-content: flex-end; width: {col(widths[k])}; flex-shrink: 0; });
            b.text_box(cell, name, style! { font-family: monospace; font-size: {SMALL}; color: #ff6; });
            if !scale.is_empty() {
                b.text_box(cell, &format!("({scale})"), style! { font-family: monospace; font-size: {SMALL}; color: #bbb; margin-left: {mono}; });
            }
        }
        let list = b.ui.add_box(content, scroll_style(self.scroll.get(&self.layer_panel).copied().unwrap_or(0.0)));
        b.ui.style_mut(list).margin = Edges::new(2.0, 0.0, 0.0, 0.0);
        b.key(list, Key::Scroll(self.layer_panel), None);
        let mono_font = style! { font-family: monospace; font-size: {SMALL}; color: white; };
        for (n, e) in entries.iter().enumerate() {
            // `grid-auto-rows: min-content`: the symbol (its fallback font's
            // 18 px line) spans the rows, so a one-row entry is 18 px.
            let entry = b.ui.add_box(list, style! { display: flex; flex-direction: column; flex-shrink: 0; });
            if e.rows.len() + usize::from(e.description.is_some()) == 1 {
                b.ui.style_mut(entry).min_height = threers::ui::Length::Px(18.0);
            }
            // `.neuroglancer-annotation-selected`, `-hover`, both.
            let is_selected = selected.as_deref() == Some(e.id.as_str());
            // `.neuroglancer-annotation-hover`: the hovered annotation's.
            let hovered = viewer.annotation_hover().is_some_and(|(l, id)| l == i && id == e.id);
            let background = match (is_selected, hovered) {
                (true, true) => Some("#969"),
                (true, false) => Some("#939"),
                (false, true) => Some("#333"),
                (false, false) => None,
            };
            if let Some(c) = background {
                b.ui.style_mut(entry).background = Some(threers::ui::css::color(c).unwrap());
            }
            b.key(entry, Key::AnnotationEntry(i, n), Some(Action::PinAnnotation(i, n)));
            if self.scroll_to_annotation.as_ref().is_some_and(|(l, id)| *l == i && *id == e.id) {
                self.annotation_scroll = Some((list, entry));
            }
            for (r, row) in e.rows.iter().enumerate() {
                let line = b.ui.add_box(entry, style! { display: flex; height: 15px; flex-shrink: 0; });
                let symbol = b.ui.add_box(line, style! { width: {col(2)}; flex-shrink: 0; });
                if r == 0 {
                    b.text_box(symbol, e.icon, style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
                }
                for (k, &d) in shown.iter().enumerate() {
                    let cell = b.ui.add_box(line, style! { display: flex; justify-content: flex-end; width: {col(widths[k])}; flex-shrink: 0; });
                    b.text_box(cell, &row[d].map(coordinate_text).unwrap_or_default(), mono_font.clone());
                }
            }
            if let Some(d) = &e.description {
                let line = b.ui.add_box(entry, style! { display: flex; height: 15px; flex-shrink: 0; });
                b.ui.add_box(line, style! { width: {col(2)}; flex-shrink: 0; });
                b.text_box(line, d, style! { font-family: sans-serif; font-size: {SMALL}; color: white; white-space: pre; });
            }
        }
    }

    /// An annotation's details in the Selection panel (`displaySelectionState`
    /// of `UserLayerWithAnnotations`): its icon, each position (copy, the
    /// layer dimensions' coordinates, move to it), delete; its ID; the
    /// `segments` relationship; the description.
    pub(crate) fn annotation_details(&mut self, b: &mut Builder, parent: NodeId, viewer: &mut Viewer, i: usize, id: &str) {
        let dims = viewer.dimensions().clone();
        let Some(details) = viewer.annotation_details(i, id) else {
            // `.neuroglancer-selection-annotation-status`: a precomputed
            // one may still come from its `by_id` index.
            let status = if viewer.request_annotation(i, id) { "Loading..." } else { "Annotation not found" };
            b.text_box(parent, status, style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
            return;
        };
        let entry = details.entry;
        let shown = viewer.layers()[i].annotation_dimensions().unwrap_or_default();
        let mono = style! { font-family: monospace; font-size: 13px; color: white; };
        let ch = b.ch("monospace", 13.0);
        // `.neuroglancer-selected-annotation-details-position-grid`.
        let grid = b.ui.add_box(parent, style! { display: flex; flex-shrink: 0; });
        b.text_box(grid, entry.icon, mono.clone());
        let rows = b.ui.add_box(grid, style! { display: flex; flex-direction: column; });
        for (r, row) in entry.rows.iter().enumerate() {
            // Grid cells keep their text at the top of the 18 px row.
            let line = b.ui.add_box(rows, style! { display: flex; align-items: flex-start; });
            let coords: Vec<f64> = shown.iter().map(|&d| row[d].unwrap_or(0.0).floor()).collect();
            let text = coords.iter().map(|v| ng_core::js::number_to_string(*v)).collect::<Vec<_>>().join(", ");
            b.icon(line, 3000 + r as u32 * 2, Some(crate::icons::COPY), "", Edges::symmetric(0.0, 1.0), Action::CopyText(text));
            b.title(Key::Icon(3000 + r as u32 * 2), "Copy position");
            for (k, &d) in shown.iter().enumerate() {
                // The names head the grid's columns: the first row only.
                let name_color = if r == 0 { [1.0, 1.0, 0x66 as f32 / 255.0, 1.0] } else { [0.0; 4] };
                b.text_box(line, &dims.names[d], style! { font-family: monospace; font-size: 13px; color: {name_color}; margin-left: {ch}; });
                b.text_box(line, &ng_core::js::number_to_string(coords[k]), style! { font-family: monospace; font-size: 13px; color: white; margin-left: {ch / 2.0}; });
            }
            // A vector (an ellipsoid's radii) has nowhere to move to.
            let vector = entry.icon == "◎" && r == 1;
            if !vector {
                let target: Vec<f32> = (0..dims.rank()).map(|d| row[d].unwrap_or(0.0) as f32).collect();
                b.icon(line, 3001 + r as u32 * 2, None, "↗", Edges::symmetric(0.0, 1.0), Action::MoveTo(target));
                b.title(Key::Icon(3001 + r as u32 * 2), "Move to position");
            }
        }
        if !details.readonly {
            let delete = b.icon(grid, 3099, Some(crate::icons::BIN), "", Edges::symmetric(0.0, 1.0), Action::DeleteAnnotation(i, entry.id.clone()));
            b.ui.style_mut(delete).align_self = Some(threers::ui::AlignItems::FlexStart);
            b.title(Key::Icon(3099), "Delete annotation");
        }

        // The ID (`.neuroglancer-annotation-property-id`), 14 px.
        let label = b.ui.add_box(parent, style! { display: flex; align-items: center; margin-bottom: 8px; flex-shrink: 0; });
        // Its label's 5 px margin, then the row's 12 px gap.
        b.text_box(label, "ID", style! { font-family: sans-serif; font-size: 12px; color: {[1.0, 1.0, 1.0, 0.8]}; margin-right: 17px; });
        // `text-overflow: ellipsis`.
        b.text_box(label, &entry.id, style! { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; font-family: monospace; font-size: 12px; color: white; });

        // Each property (`neuroglancer-annotation-property`): its name, and
        // its value at the right, read-only here.
        for (k, (name, kind, value)) in details.properties.iter().enumerate() {
            let row = b.ui.add_box(parent, style! { display: flex; align-items: center; padding: 2px; border-width: 0 0 1px 0; border-color: #222; flex-shrink: 0; });
            let key = Key::Control(i, crate::shader_ui::intern(&format!("annotationProperty{k}")));
            b.key(row, key, None);
            if b.hover(key) {
                b.ui.style_mut(row).background = Some([0x11 as f32 / 255.0, 0x11 as f32 / 255.0, 0x11 as f32 / 255.0, 1.0]);
            }
            let name_box = b.ui.add_box(row, style! { flex: 1; min-width: 0; display: flex; });
            b.text_box(name_box, name, style! { font-family: sans-serif; font-size: 13px; color: white; margin-right: 5px; });
            b.ui.add_box(row, style! { width: 12px; flex-shrink: 0; });
            let value_box = b.ui.add_box(row, style! { flex: 1; min-width: 0; display: flex; justify-content: flex-end; });
            let text = ng_viewer::annotations::format_property_value(*kind, *value);
            if matches!(kind, ng_viewer::annotations::PropertyType::Rgb | ng_viewer::annotations::PropertyType::Rgba) {
                // `makeReadonlyColorProperty`: a swatch, then the colour in
                // capitals.
                let text = text.to_uppercase();
                let container = b.ui.add_box(value_box, style! { display: flex; align-items: center; });
                let preview = b.ui.add_box(container, style! { display: flex; justify-content: center; align-items: center; width: 16px; height: 16px; border-radius: 2px; background: #333; flex-shrink: 0; margin-right: 2px; });
                let c = |k: u32| ((*value as u32) >> (8 * k) & 0xff) as f32 / 255.0;
                b.ui.add_box(preview, style! { width: 8px; height: 8px; background: {[c(0), c(1), c(2), 1.0]}; });
                b.text_box(container, &text, style! { font-family: sans-serif; font-size: 13px; color: white; });
            } else {
                b.text_box(value_box, &text, style! { font-family: monospace; font-size: 12px; color: white; overflow: hidden; text-overflow: ellipsis; });
            }
        }
        if details.readonly {
            // No relationships here, no description.
            return;
        }

        // The `segments` relationship (`makeRelatedSegmentList`), empty.
        let list = b.ui.add_box(parent, style! { display: flex; align-items: center; margin: 4px 0; flex-shrink: 0; });
        b.icon(list, 3100, Some(crate::icons::COPY), "", Edges::symmetric(0.0, 1.0), Action::CopyText(String::new()));
        b.title(Key::Icon(3100), "Copy segment IDs");
        b.icon(list, 3101, Some(crate::icons::BIN), "", Edges::symmetric(0.0, 1.0), Action::Unported("Remove all IDs"));
        b.title(Key::Icon(3101), "Remove all IDs");
        b.text_box(list, "segments", style! { font-family: sans-serif; font-size: 13px; color: white; background: #333; });
        b.icon(list, 3102, Some(crate::icons::PLUS), "", Edges::symmetric(0.0, 1.0), Action::Unported("Add related segment ID"));

        // The description: a 3-row textarea.
        let description = b.ui.add_box(parent, style! { display: flex; background: #222; height: 45px; padding: 2px; flex-shrink: 0; });
        match &entry.description {
            Some(d) => b.text_box(description, d, style! { font-family: sans-serif; font-size: {SMALL}; color: white; }),
            None => b.text_box(description, "Description", style! { font-family: sans-serif; font-size: {SMALL}; color: #757575; }),
        };
    }

    /// `AnnotationSchemaTab` (`ui/annotation_schema_tab.ts`) of a layer
    /// without properties: the explanation, the schema's actions
    /// (download, upload, copy, paste), the table's header and the add
    /// button.
    pub(crate) fn schema_tab(&mut self, b: &mut Builder, content: NodeId, _viewer: &mut Viewer, i: usize) {
        let _ = i;
        let dim = [1.0, 1.0, 1.0, 0.6];
        let scroll = b.ui.add_box(content, scroll_style(self.scroll.get(&self.layer_panel).copied().unwrap_or(0.0)));
        b.key(scroll, Key::Scroll(self.layer_panel), None);
        // `.neuroglancer-annotation-schema-text-container`: 0.25rem.
        let text = b.ui.add_box(scroll, style! { display: flex; flex-direction: column; padding: 4px; flex-shrink: 0; });
        let main = b.ui.add_box(text, style! { padding: 4px; margin-bottom: 4px; flex-shrink: 0; });
        b.text_box(
            main,
            "Set annotation property (metadata) schema for this layer which applies to all annotations in this layer. Changing a default value in the schema is not retroactive and only applies to new annotations.",
            style! { font-family: sans-serif; font-size: {SMALL}; color: {dim}; white-space: normal; },
        );
        let actions = b.ui.add_box(text, style! { display: flex; align-items: center; flex-shrink: 0; });
        for (k, (icon, title, action)) in [
            (crate::icons::DOWNLOAD, "Download schema", Action::Unported("Download schema")),
            (crate::icons::UPLOAD, "Upload schema", Action::Unported("Upload schema")),
            (crate::icons::COPY, "Copy schema to clipboard", Action::Unported("Copy schema")),
            (crate::icons::CLIPBOARD, "Paste schema from clipboard", Action::Unported("Paste schema")),
        ]
        .into_iter()
        .enumerate()
        {
            let id = 2950 + k as u32;
            let node = b.icon(actions, id, Some(icon), "", Edges::symmetric(0.0, 1.0), action);
            b.title(Key::Icon(id), title);
            // `stroke: rgba(255, 255, 255, 0.6)`, white on hover.
            if !b.hover(Key::Icon(id)) {
                if let Some(&path) = b.ui.children(node).first() {
                    b.ui.style_mut(path).color = dim;
                }
            }
        }
        // `.neuroglancer-annotation-schema-grid`: the header, the (empty)
        // body, the add button.
        let grid = b.ui.add_box(scroll, style! { display: flex; flex-direction: column; padding: 4px; flex-shrink: 0; });
        let header = b.ui.add_box(grid, style! { display: flex; align-items: flex-start; padding: 4px 2px; border-width: 0 0 1px 0; border-color: #222; flex-shrink: 0; });
        let cell_font = style! { font-family: sans-serif; font-size: {SMALL}; color: {dim}; white-space: normal; };
        let titles = ["Hover a name to see the property description", "The type of the property", "For enums, these are the available options, first entry is the default value."];
        for (k, name) in ["Name", "Type", "Default value"].into_iter().enumerate() {
            if k > 0 {
                b.ui.add_box(header, style! { width: 8px; flex-shrink: 0; });
            }
            // A block, so the text wraps at its width ("Default value").
            let cell = b.ui.add_box(header, style! { flex: 1; min-width: 0; });
            b.text_box(cell, name, cell_font.clone());
            let key = Key::Control(i, ["schemaName", "schemaType", "schemaDefault"][k]);
            b.key(cell, key, None);
            b.title(key, titles[k]);
        }
        // The description and delete headers, at most 1.25rem.
        for _ in 0..2 {
            b.ui.add_box(header, style! { width: 8px; flex-shrink: 0; });
            b.ui.add_box(header, style! { flex: 1; max-width: 20px; });
        }
        let add = b.ui.add_box(grid, style! { display: flex; padding: 4px 0; flex-shrink: 0; });
        b.icon(add, 2960, Some(crate::icons::PLUS), "", Edges::symmetric(0.0, 1.0), Action::Unported("Add property"));
        b.title(Key::Icon(2960), "Add property");
    }

    /// `AnnotationLayerView` of layer `i`.
    pub(crate) fn annotations_tab(&mut self, b: &mut Builder, content: NodeId, viewer: &mut Viewer, i: usize) {
        if viewer.layers()[i].spec.layer_type == "annotation" {
            return self.annotation_layer_tab(b, content, viewer, i);
        }
        let dims = viewer.dimensions().clone();
        let rank = dims.rank();
        let layer = &viewer.layers()[i];
        let color = layer.annotation_color();
        let boxes = layer.data_bounds_global(rank);

        // The toolbox: the annotation colour.
        let toolbox = b.ui.add_box(content, style! { display: flex; flex-shrink: 0; });
        let well = threers::ui::controls::color_well(&mut b.ui, toolbox, color, 50.0);
        b.key(well, Key::Color(ColorSetting::Annotation(i)), Some(Action::PickColor(ColorSetting::Annotation(i))));

        // Columns, in `ch` of the list's font.
        let headers: Vec<(String, String)> = (0..rank).map(|d| (dims.names[d].clone(), format_scale_with_unit_as_string(dims.scales[d], &dims.units[d], true))).collect();
        let rows: Vec<Vec<f64>> = boxes.iter().flat_map(|(lo, hi)| [lo.clone(), hi.clone()]).collect();
        let widths = column_widths(&headers, &rows);
        let ch = b.ch("sans-serif", SMALL);
        let mono = b.ch("monospace", SMALL);
        let col = |w: usize| crate::view::lu(w as f32 * ch);

        let header = b.ui.add_box(content, style! { display: flex; padding-bottom: 2px; flex-shrink: 0; });
        b.ui.add_box(header, style! { width: {col(2)}; flex-shrink: 0; });
        for (k, (name, scale)) in headers.iter().enumerate() {
            let cell = b.ui.add_box(header, style! { display: flex; justify-content: flex-end; width: {col(widths[k])}; flex-shrink: 0; });
            b.text_box(cell, name, style! { font-family: monospace; font-size: {SMALL}; color: #ff6; });
            if !scale.is_empty() {
                b.text_box(cell, &format!("({scale})"), style! { font-family: monospace; font-size: {SMALL}; color: #bbb; margin-left: {mono}; });
            }
        }

        let list = b.ui.add_box(content, scroll_style(self.scroll.get(&self.layer_panel).copied().unwrap_or(0.0)));
        b.ui.style_mut(list).margin = Edges::new(2.0, 0.0, 0.0, 0.0);
        b.key(list, Key::Scroll(self.layer_panel), None);
        let mono_font = style! { font-family: monospace; font-size: {SMALL}; color: white; };
        for (n, (lo, hi)) in boxes.iter().enumerate() {
            let entry = b.ui.add_box(list, style! { display: flex; flex-direction: column; flex-shrink: 0; });
            let center: Vec<f32> = lo.iter().zip(hi).map(|(a, z)| ((a + z) / 2.0) as f32).collect();
            b.key(entry, Key::AnnotationEntry(i, n), Some(Action::MoveTo(center)));
            for (r, corner) in [lo, hi].into_iter().enumerate() {
                let line = b.ui.add_box(entry, style! { display: flex; height: 15px; flex-shrink: 0; });
                let symbol = b.ui.add_box(line, style! { width: {col(2)}; flex-shrink: 0; });
                if r == 0 {
                    // `❑`, the bounding box's symbol, overflowing its cell.
                    b.text_box(symbol, "❑", style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
                }
                for (k, v) in corner.iter().enumerate() {
                    let cell = b.ui.add_box(line, style! { display: flex; justify-content: flex-end; width: {col(widths[k])}; flex-shrink: 0; });
                    b.text_box(cell, &coordinate_text(*v), mono_font.clone());
                }
            }
            let line = b.ui.add_box(entry, style! { display: flex; height: 15px; flex-shrink: 0; });
            b.ui.add_box(line, style! { width: {col(2)}; flex-shrink: 0; });
            b.text_box(line, "Data Bounds", style! { font-family: sans-serif; font-size: {SMALL}; color: white; white-space: pre; });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_as_neuroglancer() {
        let headers = vec![("x".to_owned(), "4nm".to_owned()), ("z".to_owned(), "40nm".to_owned())];
        assert_eq!(column_widths(&headers, &[vec![4.0, 0.0], vec![44.0, 20.0]]), vec![9, 10]);
        assert_eq!(column_widths(&headers, &[vec![1234567890.5, 0.0]]), vec![12, 10]);
        assert_eq!(coordinate_text(-0.5), "-1");
    }
}
