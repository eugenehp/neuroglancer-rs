//! Side panels' frames (`SidePanel`: title bar, close button; the manager's
//! columns and resize gutters) and the Settings and Help panels' bodies.

use ng_viewer::Viewer;
use threers::style;
use threers::ui::{Edges, NodeId, Rgba};

use crate::side::{self, Group, PanelId, Side};
use crate::view::{checkbox, checkbox_sized, lu, Builder};
use crate::{icons, Action, ColorSetting, FieldId, Key, NgUi, NumberSetting, Setting};

/// `10pt`.
const SMALL: f32 = 40.0 / 3.0;
/// `4ex` in Chrome's sans-serif at 10pt (the help panel's indent).
const FOUR_EX_SANS_10PT: f32 = 27.8837;
/// Chrome's placeholder text colour on `#333`.
const PLACEHOLDER: Rgba = [117.0 / 255.0, 117.0 / 255.0, 117.0 / 255.0, 1.0];

impl NgUi {
    /// The panels docked on `side`, as columns with gutters, in `parent` (the
    /// row around the centre). On the left the gutter follows each column;
    /// on the right it precedes it.
    pub(crate) fn side_columns(&mut self, b: &mut Builder, parent: NodeId, viewer: &mut Viewer, side: Side) {
        for (gi, group) in side::groups(viewer, side).iter().enumerate() {
            if side == Side::Right {
                gutter(b, parent, Key::Gutter(side, gi), true);
            }
            self.column(b, parent, viewer, group, gi);
            if side == Side::Left {
                gutter(b, parent, Key::Gutter(side, gi), true);
            }
        }
    }

    /// The panels docked on `side` (top or bottom), as rows with gutters,
    /// in `parent` (the centre column). On top the gutter follows each row;
    /// at the bottom it precedes it.
    pub(crate) fn side_rows(&mut self, b: &mut Builder, parent: NodeId, viewer: &mut Viewer, side: Side) {
        for (gi, group) in side::groups(viewer, side).iter().enumerate() {
            if side == Side::Bottom {
                gutter(b, parent, Key::Gutter(side, gi), false);
            }
            let row = b.ui.add_box(parent, style! { display: flex; height: {group.size as f32}; flex-shrink: 0; min-height: 0; });
            let n = group.panels.len();
            for (i, (id, location)) in group.panels.iter().enumerate() {
                let flex = if n > 1 { location.flex as f32 } else { 1.0 };
                let panel = b.ui.add_box(
                    row,
                    style! { display: flex; flex-direction: column; flex-grow: {flex}; flex-basis: 0%; min-width: 0; min-height: 0; overflow: hidden; },
                );
                b.key(panel, Key::Panel(*id), None);
                self.panel_contents(b, panel, viewer, *id);
                if i + 1 < n {
                    gutter(b, row, Key::FlexGutter(side, gi, i), true);
                }
            }
            if side == Side::Top {
                gutter(b, parent, Key::Gutter(side, gi), false);
            }
        }
    }

    /// `StatisticsPanel`: a table of each chunk source's visible chunks.
    fn statistics_body(&mut self, b: &mut Builder, panel: NodeId, viewer: &Viewer) {
        let scroll = self.scroll_body(b, panel, PanelId::Statistics, SMALL);
        let stats = viewer.chunk_statistics();
        let names = crate::statistics::formatted_names(&stats.iter().map(|s| s.properties.clone()).collect::<Vec<_>>());
        let font = style! { font-family: sans-serif; font-size: {SMALL}; color: white; };
        let bold = style! { ..font.clone(); font-weight: bold; };
        let rows: Vec<Vec<String>> = stats
            .iter()
            .zip(&names)
            .map(|(s, name)| {
                let latency = s.download_latency.map(ng_core::js::number_to_string).unwrap_or_default();
                vec![name.clone(), s.visible_total.to_string(), s.downloading.to_string(), s.system.to_string(), s.gpu.to_string(), s.failed.to_string(), s.gpu_bytes.to_string(), latency]
            })
            .collect();
        let header2 = ["", "T", "D", "M", "G", "F", "", ""];
        // `table-layout: auto` at `width: 100%`: each column its widest
        // content, the rest shared in proportion; "Visible chunks" spans five.
        let mut widths: Vec<f32> = (0..8).map(|c| {
            let mut w = b.width(header2[c], &bold);
            for r in &rows {
                w = w.max(b.width(&r[c], &font));
            }
            w + 2.0
        }).collect();
        for (c, h) in [(0, "Name"), (6, "Visible memory"), (7, "Download latency")] {
            widths[c] = widths[c].max(b.width(h, &bold) + 2.0);
        }
        // A spanning header wider than its columns widens them in
        // proportion to their content.
        let span = b.width("Visible chunks", &bold) + 2.0;
        let content: f32 = widths[1..6].iter().sum();
        if span > content + 8.0 {
            let scale = (span - 8.0) / content;
            for w in &mut widths[1..6] {
                *w *= scale;
            }
        }
        let table = b.ui.add_box(scroll, style! { display: flex; flex-direction: column; flex-shrink: 0; padding: 2px; });
        let mut cells: Vec<(NodeId, usize, usize)> = vec![];
        let row_box = |b: &mut Builder| b.ui.add_box(table, style! { display: flex; margin-bottom: 2px; flex-shrink: 0; });
        let mut cell = |b: &mut Builder, row: NodeId, c: usize, span: usize, text: &str, style: &threers::ui::Style, bg: Option<&str>| {
            let n = b.ui.add_box(row, style! { display: flex; box-sizing: border-box; padding: 1px; margin-right: 2px; flex-shrink: 0; overflow: hidden; });
            b.ui.style_mut(n).background = bg.and_then(threers::ui::css::color);
            cells.push((n, c, span));
            b.text_box(n, text, style! { ..style.clone(); white-space: pre; });
        };
        let r1 = row_box(b);
        cell(b, r1, 0, 1, "Name", &bold, Some("#333"));
        cell(b, r1, 1, 5, "Visible chunks", &bold, Some("#333"));
        cell(b, r1, 6, 1, "Visible memory", &bold, Some("#333"));
        cell(b, r1, 7, 1, "Download latency", &bold, Some("#333"));
        let r2 = row_box(b);
        for (c, h) in header2.iter().enumerate() {
            cell(b, r2, c, 1, h, &bold, Some("#333"));
        }
        for (k, r) in rows.iter().enumerate() {
            let row = row_box(b);
            let key = Key::StatisticsRow(k);
            b.key(row, key, None);
            if b.hover(key) {
                b.ui.style_mut(row).background = threers::ui::css::color("#336");
            }
            for (c, t) in r.iter().enumerate() {
                cell(b, row, c, 1, t, &font, None);
            }
        }
        b.deferred.push(crate::rendering::Deferred::Table { table, widths, cells });
    }

    /// `.neuroglancer-side-panel-column`: the group's panels, stacked by
    /// their `flex` shares, gutters between.
    fn column(&mut self, b: &mut Builder, parent: NodeId, viewer: &mut Viewer, group: &Group, gi: usize) {
        let col = b.ui.add_box(
            parent,
            style! { display: flex; flex-direction: column; width: {group.size as f32}; flex-shrink: 0; min-width: 0; },
        );
        let n = group.panels.len();
        for (i, (id, location)) in group.panels.iter().enumerate() {
            let flex = if n > 1 { location.flex as f32 } else { 1.0 };
            let panel = b.ui.add_box(
                col,
                style! {
                    display: flex;
                    flex-direction: column;
                    flex-grow: {flex};
                    flex-basis: 0%;
                    min-width: 0;
                    min-height: 0;
                    overflow: hidden;
                },
            );
            b.key(panel, Key::Panel(*id), None);
            self.panel_contents(b, panel, viewer, *id);
            if i + 1 < n {
                gutter(b, col, Key::FlexGutter(group.side, gi, i), false);
            }
        }
    }

    fn panel_contents(&mut self, b: &mut Builder, panel: NodeId, viewer: &mut Viewer, id: PanelId) {
        match id {
            PanelId::Settings => {
                title_bar(b, panel, "Settings", id);
                self.settings_body(b, panel, viewer);
            }
            PanelId::Help => {
                title_bar(b, panel, "Help", id);
                self.help_body(b, panel, viewer);
            }
            PanelId::LayerList => {
                self.layer_list_panel(b, panel, viewer);
            }
            PanelId::Selection => {
                self.selection_panel(b, panel, viewer);
            }
            PanelId::Layer | PanelId::Pinned(..) => {
                self.layer_side_panel_body(b, panel, viewer, id);
            }
            PanelId::Statistics => {
                title_bar(b, panel, "Chunk statistics", id);
                self.statistics_body(b, panel, viewer);
            }
            PanelId::Palette(k) => self.palette_panel(b, panel, viewer, k),
        }
    }

    /// A scrolling column filling the rest of a panel (`flex: 1; height: 0;
    /// overflow: auto`), scrolled as last left.
    pub(crate) fn scroll_body(&self, b: &mut Builder, panel: NodeId, id: PanelId, font_size: f32) -> NodeId {
        let body = b.ui.add_box(
            panel,
            style! {
                display: flex;
                flex-direction: column;
                flex: 1;
                height: 0;
                min-height: 0;
                font-family: sans-serif;
                font-size: {font_size};
                color: white;
            },
        );
        let scroll = b.ui.add_box(
            body,
            style! {
                display: flex;
                flex-direction: column;
                flex: 1;
                position: relative;
                overflow: auto;
                scroll-top: {self.scroll.get(&id).copied().unwrap_or(0.0)};
            },
        );
        b.key(scroll, Key::Scroll(id), None);
        scroll
    }

    /// `ViewerSettingsPanel`.
    fn settings_body(&mut self, b: &mut Builder, panel: NodeId, viewer: &Viewer) {
        let scroll = self.scroll_body(b, panel, PanelId::Settings, SMALL);
        let arial_ch = b.ch("Arial", SMALL);
        let input_width = lu(11.0 * arial_ch);
        // The title: a text input across the panel.
        let title = viewer.settings.title.clone().unwrap_or_default();
        let field = self.text_field(b, scroll, FieldId::Title, &title, "Title", None);
        b.ui.style_mut(field).margin.top = 3.0;
        // `NumberInputWidget`s: a label, and the input pushed right.
        for (label, setting, value) in [
            ("GPU memory limit", NumberSetting::GpuMemoryLimit, viewer.settings.gpu_memory_limit),
            ("System memory limit", NumberSetting::SystemMemoryLimit, viewer.settings.system_memory_limit),
            ("Concurrent chunk requests", NumberSetting::ConcurrentDownloads, viewer.settings.concurrent_downloads),
        ] {
            let row = settings_row(b, scroll, label);
            self.text_field(b, row, FieldId::Number(setting), &ng_core::js::number_to_string(value), "", Some(input_width));
        }
        let checks = [
            ("Show axis lines", Setting::ShowAxisLines, viewer.show_axis_lines),
            ("Show scale bar", Setting::ShowScaleBar, viewer.show_scale_bar),
            ("Show cross sections in 3-d", Setting::ShowSlices, viewer.show_slices),
            ("Hide sections background 3-d", Setting::HideCrossSectionBackground3D, viewer.settings.hide_cross_section_background_3d),
            ("Show default annotations", Setting::ShowDefaultAnnotations, viewer.settings.show_default_annotations),
            ("Show chunk statistics", Setting::ShowStatistics, PanelId::Statistics.location(viewer).visible),
            ("Wire frame rendering", Setting::WireFrame, viewer.settings.wire_frame),
            ("Enable prefetching", Setting::Prefetch, viewer.settings.prefetch),
            ("Enable adaptive downsampling", Setting::AdaptiveDownsampling, viewer.settings.enable_adaptive_downsampling),
        ];
        for (label, setting, on) in checks {
            let row = settings_row(b, scroll, label);
            // `TrackableBooleanCheckbox`, `width: 11ch`, the box centred in it.
            let cell = b.ui.add_box(
                row,
                style! { display: flex; justify-content: center; width: {input_width}; height: 13px; margin: 3px 3px 3px 5px; flex-shrink: 0; },
            );
            let check = checkbox(b, cell, on, Key::Setting(setting), Action::Toggle(setting), Edges::ZERO);
            b.centred.push((cell, check));
        }
        for (label, setting, color) in [
            ("Cross-section background", ColorSetting::CrossSectionBackground, viewer.background),
            ("Projection background", ColorSetting::ProjectionBackground, viewer.projection_background),
        ] {
            let row = settings_row(b, scroll, label);
            color_input(b, row, setting, color, input_width);
        }
        // The port's credits (not in neuroglancer).
        if self.attribution {
            let heading = style! { font-family: sans-serif; font-size: {SMALL}; font-weight: bold; color: white; background: #333; padding: 4px; margin-top: 12px; flex-shrink: 0; };
            b.text_box(scroll, "About", heading);
            let line = style! { font-family: sans-serif; font-size: {SMALL}; color: white; margin-top: 3px; flex-shrink: 0; };
            b.text_box(scroll, &format!("{} {}: a Rust port of neuroglancer", crate::about::NAME, crate::about::VERSION), line.clone());
            b.text_box(scroll, &crate::about::copyright(), line.clone());
            b.text_box(scroll, &format!("License: {}", crate::about::LICENSE), line.clone());
            let hovered = b.hovered.contains(&Key::Icon(crate::view::ABOUT_LINK));
            let link = b.text_box(
                scroll,
                crate::about::repository_label(),
                style! { ..line; color: {if hovered { [0.75, 0.88, 1.0, 1.0] } else { [0.53, 0.75, 1.0, 1.0] }}; },
            );
            b.key(link, Key::Icon(crate::view::ABOUT_LINK), Some(Action::OpenUrl(crate::about::REPOSITORY.into())));
        }
    }

    /// `InputEventBindingHelpDialog`: each binding map's strokes and actions.
    fn help_body(&mut self, b: &mut Builder, panel: NodeId, viewer: &Viewer) {
        let scroll = self.scroll_body(b, panel, PanelId::Help, 16.0);
        let groups = ng_viewer::input::help_bindings(&ng_viewer::input::viewer_bindings(), viewer.mac);
        for (gi, (label, entries)) in groups.iter().enumerate() {
            // `h2`: bold 10pt on #333; `div + h2` 1em above.
            b.text_box(
                scroll,
                label,
                style! {
                    font-family: sans-serif;
                    font-size: {SMALL};
                    font-weight: bold;
                    color: white;
                    background: #333;
                    padding: 4px;
                    margin: {Edges::new(if gi > 0 { SMALL } else { 0.0 }, 0.0, 0.83 * SMALL, 0.0)};
                    flex-shrink: 0;
                },
            );
            for (stroke, action) in entries {
                b.text_box(scroll, stroke, style! { font-family: monospace; font-size: 13px; font-weight: bold; color: yellow; flex-shrink: 0; });
                b.text_box(scroll, action, style! { font-family: sans-serif; font-size: {SMALL}; color: white; margin-left: {FOUR_EX_SANS_10PT}; flex-shrink: 0; });
            }
        }
    }

    /// A text `<input>` (Arial 10pt, white on `#333`, 1 × 2 px padding) for
    /// `id`, showing `value` or, focused, what is being typed.
    pub(crate) fn text_field(&self, b: &mut Builder, parent: NodeId, id: FieldId, value: &str, placeholder: &str, width: Option<f32>) -> NodeId {
        let editing = self.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t);
        let node = b.ui.add_box(
            parent,
            style! {
                display: flex;
                position: relative;
                align-items: center;
                height: 15px;
                padding: 1px 2px;
                background: #333;
                font-family: Arial;
                font-size: {SMALL};
                color: white;
                flex-shrink: 0;
                overflow: hidden;
                margin-left: {if width.is_some() { 5.0 } else { 0.0 }};
            },
        );
        if let Some(w) = width {
            b.ui.style_mut(node).width = threers::ui::Length::Px(w);
        }
        b.key(node, Key::Field(id), None);
        let font = style! { font-family: Arial; font-size: {SMALL}; color: white; };
        self.field_contents(b, node, editing, value, placeholder, &font, 2.0);
        node
    }
}

/// A label row of the settings list: text, and the control at the right
/// (`justify-content: space-between`), 3 px below the one before.
fn settings_row(b: &mut Builder, parent: NodeId, label: &str) -> NodeId {
    let row = b.ui.add_box(
        parent,
        style! {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-top: 3px;
            flex-shrink: 0;
            font-family: sans-serif;
            font-size: {SMALL};
            color: white;
        },
    );
    b.text_box(row, label, style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
    row
}

/// Chrome's `<input type=color>`: a light grey button (1 px black border,
/// 1 × 2 px padding), the colour in a swatch with a grey edge.
fn color_input(b: &mut Builder, parent: NodeId, setting: ColorSetting, color: [f32; 3], width: f32) {
    let node = threers::ui::controls::color_well(&mut b.ui, parent, color, width);
    b.ui.style_mut(node).margin.left = 5.0;
    b.key(node, Key::Color(setting), Some(Action::PickColor(setting)));
}

/// `addTitleBar`: the title on blue, the close button at the end.
pub(crate) fn title_bar(b: &mut Builder, panel: NodeId, title: &str, id: PanelId) -> NodeId {
    let bar = b.ui.add_box(
        panel,
        style! {
            display: flex;
            align-items: center;
            font-family: sans-serif;
            font-size: {SMALL};
            background: #03c;
            padding: 2px;
            color: white;
            flex-shrink: 0;
        },
    );
    b.text_box(bar, title, style! { flex: 1; overflow: hidden; font-family: sans-serif; font-size: {SMALL}; color: white; });
    let close = b.icon(bar, 2000 + id.ordinal(), Some(icons::CLOSE), "", Edges::symmetric(0.0, 1.0), Action::ClosePanel(id));
    b.key(close, Key::PanelClose(id), Some(Action::ClosePanel(id)));
    b.title(Key::PanelClose(id), "Close panel");
    bar
}

/// A resize gutter: a 1 px `#333` line with 2 px of padding either side
/// (`background-clip: content-box`), across columns (`vertical`) or
/// between stacked panels.
fn gutter(b: &mut Builder, parent: NodeId, key: Key, between_columns: bool) {
    let node = if between_columns {
        b.ui.add_box(parent, style! { width: 1px; padding: 0 2px; background: #333; background-clip: content-box; flex-shrink: 0; })
    } else {
        b.ui.add_box(parent, style! { height: 1px; padding: 2px 0; background: #333; background-clip: content-box; flex-shrink: 0; })
    };
    b.key(node, key, None);
}

impl NgUi {
    /// `LayerListPanel`: every layer, archived ones too.
    fn layer_list_panel(&mut self, b: &mut Builder, panel: NodeId, viewer: &mut Viewer) {
        let layers = viewer.layers();
        let visible = layers.iter().filter(|l| l.spec.visible).count();
        let hidden = layers.iter().filter(|l| !l.spec.visible && !l.spec.archived).count();
        let archived = layers.iter().filter(|l| l.spec.archived).count();
        let mut title = String::from("Layers");
        if visible + hidden + archived > 0 {
            title.push_str(" (");
            let mut sep = "";
            if visible + hidden > 0 {
                title.push_str(&format!("{visible}/{} visible", visible + hidden));
                sep = ", ";
            }
            if archived > 0 {
                title.push_str(&format!("{sep}{archived} archived"));
            }
            title.push(')');
        }
        let bar = title_bar(b, panel, &title, PanelId::LayerList);
        // The layer bar's visibility (`uiControlVisibility.showLayerPanel`),
        // before the close button.
        let shown = layer_bar_shown(viewer);
        let n = b.ui.children(bar).len();
        let eye = b.icon(bar, 2100, Some(icons::EYE), "", Edges::symmetric(0.0, 1.0), Action::ToggleLayerBar);
        if shown {
            b.ui.style_mut(eye).background = Some([1.0, 1.0, 1.0, 0.2]);
        }
        // `order: 50`: before the close button.
        b.ui.move_child(bar, eye, n - 1);
        let items = self.scroll_body(b, panel, PanelId::LayerList, 16.0);
        let selected = PanelId::Layer.layer(viewer);
        let non_archived = layers.iter().filter(|l| !l.spec.archived).count();
        let sans16_ch = b.ch("sans-serif", 16.0);
        let number_width = lu(number_to_string_len(non_archived) as f32 * sans16_ch);
        let mut number = 0;
        for (i, layer) in viewer.layers().iter().enumerate() {
            let key = Key::ListItem(i);
            let hover = b.hover(key);
            let (archived, visible) = (layer.spec.archived, layer.spec.visible);
            let border = if archived {
                "#666"
            } else if selected == Some(i) {
                "#3c3"
            } else {
                "#aaa"
            };
            let item = b.ui.add_box(
                items,
                style! {
                    display: flex;
                    align-items: center;
                    gap: 2px;
                    padding: 2px 4px;
                    border: 1px solid;
                    border-color: {border};
                    margin: 2px;
                    flex-shrink: 0;
                    background: {hover.then_some([0.2, 0.2, 0.2, 1.0])};
                    color: {if visible { "white" } else { "#bbb" }};
                },
            );
            b.key(item, key, None);
            // Not archived (`TrackableBooleanCheckbox`, 1rem).
            let cell = b.ui.add_box(item, style! { display: flex; flex-shrink: 0; margin: 3px 3px 3px 4px; });
            checkbox_sized(b, cell, !archived, Key::ListArchive(i), Action::SetArchived(i, !archived), 16.0);
            if !archived {
                number += 1;
            }
            // Archived: `visibility: hidden`, its width kept.
            let label = if archived { String::new() } else { number.to_string() };
            b.text_box(
                item,
                &label,
                style! {
                    display: flex;
                    justify-content: flex-end;
                    width: {number_width};
                    line-height: 18px;
                    font-family: sans-serif;
                    font-size: 16px;
                    font-weight: bold;
                    color: {if visible { "white" } else { "#bbb" }};
                    flex-shrink: 0;
                },
            );
            // The colour, or the "unsupported" checker; slashed while hidden.
            let wrapper = b.ui.add_box(item, style! { position: relative; padding: 5px; flex-shrink: 0; });
            b.key(wrapper, Key::ListColor(i), Some(Action::ToggleLayerVisible(i)));
            let chip = b.ui.add_box(
                wrapper,
                style! {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    overflow: hidden;
                    opacity: {if visible { 1.0 } else { 0.3 }};
                    position: relative;
                },
            );
            if !archived {
                color_chip(b, chip, layer.layer_bar_colors());
            }
            if !visible {
                // `::before`: an 18 px line at 135°.
                b.ui.add_path(
                    wrapper,
                    style! { position: absolute; left: 0; top: 0; width: 20px; height: 20px; color: #bbb; },
                    vec![format!("M{} {}L{} {}", 10.0 - 6.364, 10.0 - 6.364, 10.0 + 6.364, 10.0 + 6.364)],
                    [0.0, 0.0, 20.0, 20.0],
                    Some(1.0),
                    false,
                );
            }
            // `LayerTypeIndicatorWidget`.
            let abbreviation = layer_type_abbreviation(layer);
            let type_color = if archived {
                "#999"
            } else if visible {
                "white"
            } else {
                "#bbb"
            };
            b.text_box(
                item,
                abbreviation,
                style! {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 2px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 4px;
                    font-family: sans-serif;
                    font-size: {SMALL};
                    font-weight: bold;
                    color: {type_color};
                    height: 16px;
                    width: 26px;
                    flex-shrink: 0;
                },
            );
            // `LayerNameWidget`.
            let name_color = if archived {
                "#999"
            } else if visible {
                "white"
            } else {
                "#bbb"
            };
            let name = self.name_field(b, item, FieldId::LayerName(i), layer.name(), name_color);
            b.ui.style_mut(name).margin.left = 4.0;
            if hover {
                // On hover: the layer panel's toggle, the eye, delete.
                let side = b.icon(item, 2200 + i as u32 * 4, Some(icons::CONTROLS_ALT), "", Edges::symmetric(0.0, 1.0), Action::ToggleLayerSidePanel(i));
                if selected == Some(i) && PanelId::Layer.location(viewer).visible {
                    b.ui.style_mut(side).background = Some([1.0, 1.0, 1.0, 0.2]);
                }
                let eye = b.icon(
                    item,
                    2201 + i as u32 * 4,
                    Some(if visible { icons::EYE } else { icons::EYE_CROSSED }),
                    "",
                    Edges::ZERO,
                    Action::ToggleLayerVisible(i),
                );
                let s = b.ui.style_mut(eye);
                (s.width, s.height, s.padding) = (threers::ui::Length::Px(20.0), threers::ui::Length::Px(20.0), Edges::ZERO);
                b.icon(item, 2202 + i as u32 * 4, Some(icons::BIN), "", Edges::symmetric(0.0, 1.0), Action::DeleteLayer(i));
            }
        }
        // `layerDropZone`.
        b.ui.add_box(items, style! { flex: 1; });
    }

    /// An `<input>` holding a name: Arial 10pt on a 20 px line, no box.
    pub(crate) fn name_field(&self, b: &mut Builder, parent: NodeId, id: FieldId, value: &str, color: &str) -> NodeId {
        let editing = self.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t);
        let node = b.ui.add_box(
            parent,
            style! {
                display: flex;
                position: relative;
                align-items: center;
                flex: 1;
                min-width: 0;
                height: 20px;
                line-height: 20px;
                overflow: hidden;
                font-family: Arial;
                font-size: {SMALL};
                color: {color};
            },
        );
        b.key(node, Key::Field(id), None);
        let font = style! { font-family: Arial; font-size: {SMALL}; color: {color}; line-height: 20px; };
        self.field_contents(b, node, editing, value, "", &font, 0.0);
        node
    }

    /// A field's text, and its selection or caret while it is edited.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn field_contents(&self, b: &mut Builder, node: NodeId, editing: Option<&threers::ui::TextField>, value: &str, placeholder: &str, font: &threers::ui::Style, x_offset: f32) {
        let view = threers::ui::controls::TextView {
            value,
            placeholder,
            placeholder_color: PLACEHOLDER,
            editing,
            caret_on: self.caret_on(),
            font,
            x_offset,
            selection: threers::ui::controls::SELECTION,
        };
        threers::ui::controls::text_contents(&mut b.ui, node, view, b.text);
    }

    /// A field's contents without a placeholder.
    pub(crate) fn text_contents(&self, b: &mut Builder, node: NodeId, editing: Option<&threers::ui::TextField>, value: &str, font: &threers::ui::Style, x_offset: f32) {
        self.field_contents(b, node, editing, value, "", font, x_offset);
    }

    fn selection_panel(&mut self, b: &mut Builder, panel: NodeId, viewer: &mut Viewer) {
        self.selection_panel_body(b, panel, viewer);
    }

}

/// Whether the layer bar shows (`uiControlVisibility.showLayerPanel`, unset
/// meaning shown).
pub(crate) fn layer_bar_shown(viewer: &Viewer) -> bool {
    viewer.ui_state("uiControlVisibility").and_then(|v| v.get("showLayerPanel")).and_then(serde_json::Value::as_bool).unwrap_or(true)
}

fn number_to_string_len(n: usize) -> usize {
    n.to_string().len()
}

/// `typeAbbreviation` of a layer's type.
pub(crate) fn layer_type_abbreviation(layer: &ng_viewer::ManagedLayer) -> &'static str {
    match layer.kind() {
        Some(ng_viewer::LayerKind::Image { .. }) => "img",
        Some(ng_viewer::LayerKind::Segmentation(_)) => "seg",
        None => match layer.spec.layer_type.as_str() {
            "image" => "img",
            "segmentation" => "seg",
            "annotation" => "ann",
            "mesh" => "mesh",
            "skeleton" => "skel",
            _ => "new",
        },
    }
}

/// `LayerColorWidget`'s indicator: one colour, several (a conic stepped
/// gradient, drawn as wedges), or the grey checker of "no colour".
fn color_chip(b: &mut Builder, chip: NodeId, colors: Option<Vec<[f32; 3]>>) {
    match colors {
        Some(c) if c.is_empty() => {
            // The four 45° gradients over a 4 px tile: a 2 px checker.
            let mut d = String::new();
            for y in (0..10).step_by(2) {
                for x in (0..10).step_by(2) {
                    if (x / 2 + y / 2) % 2 == 1 {
                        d.push_str(&format!("M{x} {y}h2v2h-2Z"));
                    }
                }
            }
            b.ui.add_path(chip, style! { width: 10px; height: 10px; color: gray; }, vec![d], [0.0, 0.0, 10.0, 10.0], None, true);
        }
        Some(c) if c.len() == 1 => {
            b.ui.style_mut(chip).background = Some([c[0][0], c[0][1], c[0][2], 1.0]);
        }
        Some(mut c) => {
            // `createSteppedCssGradient(colors.reverse(), conic)`: wedges
            // clockwise from the top.
            c.reverse();
            let n = c.len() as f64;
            for (i, col) in c.iter().enumerate() {
                let a0 = (ng_core::js::round(i as f64 * 360.0 / n)).to_radians();
                let a1 = (ng_core::js::round((i + 1) as f64 * 360.0 / n)).to_radians();
                let p = |a: f64| (5.0 + 8.0 * a.sin(), 5.0 - 8.0 * a.cos());
                let ((x0, y0), (x1, y1)) = (p(a0), p(a1));
                let large = if a1 - a0 > std::f64::consts::PI { 1 } else { 0 };
                let d = format!("M5 5L{x0} {y0}A8 8 0 {large} 1 {x1} {y1}Z");
                b.ui.add_path(
                    chip,
                    style! { position: absolute; left: 0; top: 0; width: 10px; height: 10px; color: {[col[0], col[1], col[2], 1.0]}; },
                    vec![d],
                    [0.0, 0.0, 10.0, 10.0],
                    None,
                    true,
                );
            }
        }
        // Rainbow: neuroglancer's conic of five colours.
        None => {
            let stops = [[0xab, 0x47, 0xbc], [0xff, 0xca, 0x28], [0x66, 0xbb, 0x6a], [0x26, 0xa6, 0x9a], [0x5c, 0x6b, 0xc0]];
            let colors: Vec<[f32; 3]> = stops.iter().map(|c| c.map(|v| v as f32 / 255.0)).rev().collect();
            color_chip(b, chip, Some(colors));
        }
    }
}

/// A captured selection (`PersistentViewerSelectionState`): where, and each
/// layer's value there.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Selection {
    pub position: Option<Vec<f32>>,
    pub layers: Vec<(String, Option<serde_json::Value>)>,
}

impl Selection {
    /// What the pointer is over now (`capturePersistentViewerSelectionState`).
    pub(crate) fn capture(viewer: &mut Viewer) -> Option<Selection> {
        let position = viewer.mouse_position()?;
        let picked = viewer.picked_annotation_part();
        let mut layers = vec![];
        for (i, l) in viewer.layers().iter().enumerate() {
            if l.spec.archived {
                continue;
            }
            // An annotation layer's picked annotation (`captureSelectionState`).
            if let Some((_, id, part)) = picked.as_ref().filter(|(l, _, _)| *l == i) {
                layers.push((l.name().to_owned(), Some(serde_json::json!({ "annotationId": id, "annotationPart": part, "annotationSource": 0, "annotationSubsource": "default" }))));
                continue;
            }
            let value = viewer.layer_value_at(i, &position).map(|v| match l.kind() {
                // Segment IDs as strings (`selectionStateToJson`), values as numbers.
                Some(ng_viewer::LayerKind::Segmentation(_)) => serde_json::Value::String(v),
                _ => v.parse::<f64>().map(ng_core::js::number).unwrap_or(serde_json::Value::String(v)),
            });
            layers.push((l.name().to_owned(), value));
        }
        Some(Selection { position: Some(position), layers })
    }

    /// From the `selection` state key.
    pub(crate) fn from_state(viewer: &Viewer) -> Option<Selection> {
        let o = viewer.ui_state("selection")?.as_object()?;
        let position = o.get("position").and_then(|p| p.as_array()).map(|a| a.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect());
        // An annotation's selection is its source's state itself
        // (`annotationId`, …); others hold a `value`.
        let layers = o.get("layers")?.as_object()?.iter().map(|(k, v)| (k.clone(), if v.get("annotationId").is_some() { Some(v.clone()) } else { v.get("value").cloned() })).collect();
        Some(Selection { position, layers })
    }

    /// Into the `selection` state key (`toJSON`, pinned and shown).
    pub(crate) fn store(&self, viewer: &mut Viewer) {
        let mut o = viewer.ui_state("selection").and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
        if let Some(p) = &self.position {
            o.insert("position".into(), ng_core::js::numbers(p));
        }
        let mut layers = serde_json::Map::new();
        for (name, value) in &self.layers {
            let data = match value {
                Some(v) if v.get("annotationId").is_some() => v.clone(),
                Some(v) => serde_json::json!({ "value": v }),
                None => serde_json::Value::Null,
            };
            if !data.is_null() {
                layers.insert(name.clone(), data);
            }
        }
        o.insert("layers".into(), serde_json::Value::Object(layers));
        viewer.set_ui_state("selection", Some(serde_json::Value::Object(o)));
    }
}

impl NgUi {
    /// `SelectionDetailsPanel`.
    fn selection_panel_body(&mut self, b: &mut Builder, panel: NodeId, viewer: &mut Viewer) {
        let bar = title_bar(b, panel, "Selection", PanelId::Selection);
        // Back, forward and pin before the close button.
        let close_at = b.ui.children(bar).len() - 1;
        let back = b.icon(bar, 2300, Some(icons::ARROW_LEFT), "", Edges::symmetric(0.0, 1.0), Action::SelectionHistory(-1));
        let forward = b.icon(bar, 2301, Some(icons::ARROW_RIGHT), "", Edges::symmetric(0.0, 1.0), Action::SelectionHistory(1));
        b.ui.style_mut(back).opacity = if self.selection_index > 0 { 1.0 } else { 0.0 };
        b.ui.style_mut(forward).opacity = if self.selection_index + 1 < self.selection_history.len() { 1.0 } else { 0.0 };
        let pin = b.icon(bar, 2302, None, "📌\u{fe0e}", Edges::symmetric(0.0, 1.0), Action::ToggleSelectionPin);
        // The emoji's fallback font makes its line 20 px.
        b.ui.style_mut(pin).min_height = threers::ui::Length::Px(20.0);
        if self.selection_pinned {
            // `.light-background[data-checked=true]`.
            b.ui.style_mut(pin).background = Some([0.0, 0.0, 0.0, 0.5]);
        }
        for (k, node) in [back, forward, pin].into_iter().enumerate() {
            b.ui.move_child(bar, node, close_at + k);
        }
        let body = self.scroll_body(b, panel, PanelId::Selection, 16.0);
        let selection = if self.selection_pinned { Selection::from_state(viewer) } else { Selection::capture(viewer) };
        let Some(selection) = selection else { return };
        if let Some(position) = &selection.position {
            let row = b.ui.add_box(body, style! { display: flex; align-items: center; font-family: monospace; font-size: 13px; flex-shrink: 0; });
            b.icon(row, 2310, Some(icons::COPY), "", Edges::symmetric(0.0, 1.0), Action::CopyText(crate::format_position_text(position)));
            let names = viewer.dimensions().names.clone();
            for (i, (name, v)) in names.iter().zip(position).enumerate() {
                let dim = b.ui.add_box(row, style! { display: flex; margin-left: {if i > 0 { 8.0 } else { 0.0 }}; flex-shrink: 0; });
                b.text_box(dim, name, style! { font-family: monospace; font-size: 13px; color: #ff6; });
                b.text_box(dim, &crate::coordinate_text(*v), style! { font-family: monospace; font-size: 13px; color: white; margin-left: 5px; });
            }
            b.icon(row, 2311, None, "↗", Edges::symmetric(0.0, 1.0), Action::MoveTo(position.clone()));
        }
        let shown: Vec<usize> = selection.layers.iter().enumerate().filter(|(_, (n, v))| v.is_some() && viewer.layers().iter().any(|l| l.name() == n)).map(|(i, _)| i).collect();
        for (li, (name, value)) in selection.layers.iter().enumerate() {
            let Some(value) = value else { continue };
            let Some(layer_index) = viewer.layers().iter().position(|l| l.name() == name) else { continue };
            // `margin: 4px 0` in a block: adjacent margins collapse.
            let last = shown.last() == Some(&li);
            let section = b.ui.add_box(
                body,
                style! { display: flex; flex-direction: column; margin: {Edges::new(4.0, 0.0, if last { 4.0 } else { 0.0 }, 0.0)}; border: 1px solid #1f1f1f; flex-shrink: 0; },
            );
            let key = Key::SelectionLayer(li);
            let title_bg = if b.hover(key) { "#333" } else { "#1f1f1f" };
            let title = b.text_box(section, name, style! { background: {title_bg}; font-family: sans-serif; font-size: {SMALL}; color: white; padding: 2px 4px; });
            b.key(title, key, Some(Action::ToggleLayerSidePanel(layer_index)));
            let layer_body = b.ui.add_box(section, style! { display: flex; flex-direction: column; padding: 4px; });
            if let Some(id) = value.get("annotationId").and_then(serde_json::Value::as_str) {
                self.annotation_details(b, layer_body, viewer, layer_index, id);
                continue;
            }
            match viewer.layers()[layer_index].kind() {
                Some(ng_viewer::LayerKind::Segmentation(display)) => {
                    let id: u64 = value.as_str().and_then(|s| s.parse().ok()).or_else(|| value.as_u64()).unwrap_or(0);
                    let visible = segment_visible(&viewer.layers()[layer_index], id);
                    let ids: Vec<u64> = crate::segments::starred(&viewer.layers()[layer_index]).into_iter().map(|(i, _)| i).collect();
                    self.starred.insert(layer_index, ids);
                    self.segment_entry(b, layer_body, layer_index, id, display.color_seed, visible, 2400 + li as u32 * 8);
                }
                _ => {
                    let grid = b.ui.add_box(layer_body, style! { display: flex; align-items: center; font-family: monospace; font-size: 13px; });
                    let text = match value {
                        serde_json::Value::String(s) => s.clone(),
                        v => ng_core::js::to_string(v),
                    };
                    b.icon(grid, 2400 + li as u32 * 8, Some(icons::COPY), "", Edges::symmetric(0.0, 1.0), Action::CopyText(text.clone()));
                    b.text_box(grid, &text, style! { font-family: monospace; font-size: 13px; color: white; });
                }
            }
        }
    }

    /// A segment's row (`SegmentListEntry`): copy (on hover), star, eye, and
    /// the ID on the segment's colour.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn segment_entry(&self, b: &mut Builder, parent: NodeId, layer: usize, id: u64, seed: u32, visible: bool, icon_base: u32) -> NodeId {
        let key = Key::SegmentEntry(layer, id);
        let entry = b.ui.add_box(parent, style! { display: flex; align-items: center; background: black; flex-shrink: 0; });
        if self.hovered_segments.get(layer).copied().flatten() == Some(id) {
            b.ui.style_mut(entry).background = Some(threers::ui::css::color("#222").unwrap());
        }
        b.key(entry, key, None);
        let copy = b.icon(entry, icon_base, Some(icons::COPY), "", Edges::symmetric(0.0, 1.0), Action::CopyText(id.to_string()));
        b.ui.style_mut(copy).opacity = if b.hover(key) { 1.0 } else { 0.0 };
        b.ui.add_box(entry, style! { width: 0; });
        // Starred: in the layer's `segments`.
        let starred = self.starred.get(&layer).is_some_and(|s| s.contains(&id));
        let tri = if starred { crate::segments::Tri::On } else { crate::segments::Tri::Off };
        crate::segments::star_icon(b, entry, icon_base + 1, tri, Action::SegmentChange(layer, crate::segments::SegmentChange::Star(vec![id], !starred)));
        b.icon(
            entry,
            icon_base + 2,
            Some(if visible { icons::EYE } else { icons::EYE_CROSSED }),
            "",
            Edges::symmetric(0.0, 1.0),
            Action::ToggleSegment(layer, id),
        );
        let color = ng_core::SegmentColorHash::new(seed).compute(id);
        let text = if use_white(color) { "white" } else { "black" };
        b.text_box(entry, &id.to_string(), style! { font-family: monospace; font-size: 13px; color: {text}; background: {[color[0], color[1], color[2], 1.0]}; });
        entry
    }
}

/// Whether a segment is in a segmentation layer's visible set.
pub(crate) fn segment_visible(layer: &ng_viewer::ManagedLayer, id: u64) -> bool {
    ng_viewer::state::segmentation_settings(&layer.spec.json).is_ok_and(|(_, s)| s.iter().any(|&(i, v)| i == id && v))
}

pub(crate) fn use_white(c: [f32; 3]) -> bool {
    let e = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    0.2126 * e(c[0]) + 0.7152 * e(c[1]) + 0.0722 * e(c[2]) <= 0.179
}

/// A scrolling column (`overflow-y: auto; flex: 1; height: 0`), scrolled
/// `top` pixels.
pub(crate) fn scroll_style(top: f32) -> threers::ui::Style {
    style! { display: flex; flex-direction: column; flex: 1; height: 0; min-height: 0; overflow: auto; scroll-top: {top}; }
}
