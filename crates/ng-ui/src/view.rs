//! The tree: neuroglancer's DOM and stylesheets, as `style!` declarations.

use std::collections::HashMap;

use ng_core::js::number_to_string;
use ng_core::panel::NamedAxes;
use ng_viewer::{Layout, Panel, Viewer};
use threers::style;
use threers::ui::{Edges, JustifyContent, Length, NodeId, Rect, Rgba, Style, TextMeasure, Ui};

use crate::{coordinate_text, icons, mouse_position_text, plot, Action, Icon, Key, NgUi, PlotGeometry};

const WHITE: Rgba = [1.0; 4];
/// The page's default font (`body` has no `font-family`: Chrome's `Times`).
const SERIF: &str = "Times";
/// `font-size: medium` for `monospace` in Chrome.
const MONOSPACE_MEDIUM: f32 = 13.0;
/// `10pt`.
const SMALL: f32 = 40.0 / 3.0;
/// `dimensionColors` of the display dimensions widget.
const DIMENSION_COLORS: [&str; 3] = ["#f00", "#0f0", "#99f"];
/// A selection in a focused field, as Chrome (headless) paints it.
const SELECTION: Rgba = [77.0 / 255.0, 113.0 / 255.0, 152.0 / 255.0, 1.0];

/// The top bar's icon buttons, left to right after the position widget
/// (`Viewer.makeUI`), with their titles.
const TOP_ICONS: [(Option<Icon>, &str, &str); 9] = [
    (Some(icons::TOOL), "", "Tool palettes"),
    (Some(icons::LAYERS), "", "Show layer list panel"),
    (Some(icons::LIST), "", "Show selection details panel"),
    (Some(icons::CONTROLS_ALT), "", "Show layer side panel"),
    (None, "{}", "Edit JSON state"),
    (Some(icons::CAMERA), "", "Screenshot"),
    (Some(icons::CLIPBOARD), "", "Copy URL to clipboard"),
    (None, "?", "Show help panel"),
    (Some(icons::SETTINGS), "", "Show settings panel"),
];
const COPY_ICON: u32 = 100;
pub(crate) const ADD_LAYER_ICON: u32 = 101;
/// The footer's and the About section's links to the repository.
pub(crate) const FOOTER_LINK: u32 = 3901;
pub(crate) const ABOUT_LINK: u32 = 3902;
const PLAYBACK_ICON: u32 = 102;
/// Dimension `i`'s play/pause button is `PLAY_ICON + i`.
const PLAY_ICON: u32 = 3000;
/// Layer `i`'s close and delete buttons.
fn layer_icon(i: usize, delete: bool) -> u32 {
    1000 + 2 * i as u32 + delete as u32
}

/// A length as Blink's `LayoutUnit` holds it: truncated to 1/64 px. Widths
/// in `ch` (`3ch` = 3 × the `0` advance) land where Chrome puts them only
/// with this; text is placed at quarter pixels, so the drift shows.
pub(crate) fn lu(v: f32) -> f32 {
    (v * 64.0).trunc() / 64.0
}

/// `getRelativeLuminance(c) <= 0.179`: white text reads better on `c`.
fn use_white_background(c: [f32; 3]) -> bool {
    let e = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    0.2126 * e(c[0]) + 0.7152 * e(c[1]) + 0.0722 * e(c[2]) <= 0.179
}

/// `LAYOUT_SYMBOLS`.
fn layout_symbol(layout: Layout) -> &'static str {
    match layout {
        Layout::FourPanel => "◱",
        Layout::FourPanelAlt => "◲",
        Layout::Perspective | Layout::Single(_) => "◻",
        Layout::SliceAnd3d(_) => "◫",
    }
}

/// A panel's related layouts (`registerRelatedLayouts`): what its
/// buttons, `toggle-layout` and `toggle-layout-alternative` switch to.
pub(crate) fn related_layouts(layout: Layout, panel: Panel) -> Vec<Layout> {
    panel_furniture(layout, panel).1
}

/// For a panel of `layout`: whether it has a display dimensions widget, its
/// related layouts (`registerRelatedLayouts`), and whether it shows the
/// "Sections" checkbox (`showSliceViewsCheckbox`).
fn panel_furniture(layout: Layout, panel: Panel) -> (bool, Vec<Layout>, bool) {
    match (layout, panel) {
        (Layout::FourPanel | Layout::FourPanelAlt, Panel::Slice(axes)) => (axes == NamedAxes::Xy, vec![Layout::Single(axes), Layout::SliceAnd3d(axes)], false),
        (Layout::FourPanel, Panel::Perspective) => (true, vec![Layout::Perspective, Layout::FourPanelAlt], true),
        (Layout::FourPanelAlt, Panel::Perspective) => (true, vec![Layout::Perspective, Layout::FourPanel], true),
        (Layout::SliceAnd3d(axes), Panel::Slice(_)) => (true, vec![Layout::Single(axes), Layout::FourPanelAlt], false),
        (Layout::SliceAnd3d(_), Panel::Perspective) => (true, vec![Layout::Perspective, Layout::FourPanelAlt], true),
        (Layout::Single(axes), _) => (true, vec![Layout::FourPanelAlt, Layout::SliceAnd3d(axes)], false),
        (Layout::Perspective, _) => (true, vec![Layout::FourPanelAlt], false),
    }
}

/// Builds nodes with a running text measurer.
pub(crate) struct Builder<'a> {
    pub(crate) ui: Ui,
    pub(crate) keys: HashMap<NodeId, Key>,
    pub(crate) actions: HashMap<Key, Action>,
    pub(crate) hovered: &'a [Key],
    pub(crate) text: &'a mut dyn TextMeasure,
    /// Native controls drawn inside a wider box (a checkbox in an
    /// `<input>` 11ch wide): Chrome centres them in the box snapped to
    /// pixels.
    pub(crate) centred: Vec<(NodeId, NodeId)>,
    /// Native controls Chrome paints at whole pixels (a range input), where
    /// layout leaves them at fractions.
    pub(crate) snapped: Vec<NodeId>,
    /// Canvas contents to draw once laid out.
    pub(crate) deferred: Vec<crate::rendering::Deferred>,
    /// Layer controls' models, for input.
    pub(crate) controls: HashMap<Key, crate::rendering::Control>,
    /// The page's height (`100vh`).
    pub(crate) viewport_height: f32,
    /// Tooltips (`title`) of keyed controls.
    pub(crate) titles: HashMap<Key, String>,
    /// The tools bound to keys; tools keyed this frame (by place).
    pub(crate) bindings: Vec<ng_viewer::tools::ToolBinding>,
    pub(crate) tools: Vec<crate::tools::ToolRef>,
}

impl Builder<'_> {
    pub(crate) fn hover(&self, key: Key) -> bool {
        self.hovered.contains(&key)
    }

    /// A control's tooltip (`title`).
    pub(crate) fn title(&mut self, key: Key, text: impl Into<String>) {
        self.titles.insert(key, text.into());
    }

    pub(crate) fn key(&mut self, node: NodeId, key: Key, action: Option<Action>) {
        self.keys.insert(node, key);
        if let Some(a) = action {
            self.actions.insert(key, a);
        }
    }

    /// The width of `0` (the `ch` unit) in a font.
    pub(crate) fn ch(&mut self, family: &str, size: f32) -> f32 {
        self.text.measure("0", &style! { font-family: {family}; font-size: {size}; }).0
    }

    /// A run's width as layout gives it (rounded up to 1/64 px).
    pub(crate) fn width(&mut self, text: &str, style: &Style) -> f32 {
        (self.text.measure(text, style).0 * 64.0).ceil() / 64.0
    }

    /// A box holding one line of text.
    pub(crate) fn text_box(&mut self, parent: NodeId, text: &str, style: Style) -> NodeId {
        let b = self.ui.add_box(parent, style);
        self.ui.add_text(b, text);
        b
    }

    /// `.neuroglancer-icon`: an 18 px (minimum) box, 2 px side padding, 1 px
    /// side margins, the icon or text centred, `#db4437` under the pointer.
    pub(crate) fn icon(&mut self, parent: NodeId, id: u32, icon: Option<Icon>, text: &str, margin: Edges, action: Action) -> NodeId {
        let key = Key::Icon(id);
        let b = self.ui.add_box(
            parent,
            style! {
                display: flex;
                min-width: 18px;
                min-height: 18px;
                padding: 0 2px;
                margin: {margin};
                align-self: center;
                align-items: center;
                justify-content: center;
                border-radius: 20%;
                background: {self.hover(key).then_some([0xdb as f32 / 255.0, 0x44 as f32 / 255.0, 0x37 as f32 / 255.0, 1.0])};
                color: white;
                font-family: sans-serif;
                font-size: 12px;
                font-weight: 900;
            },
        );
        // `makeIcon({svg, text})`: the icon, then the text.
        if let Some(icon) = icon {
            self.ui.add_path(
                b,
                style! { width: 16px; height: 16px; flex-shrink: 0; color: white; },
                icon.paths.iter().map(|s| s.to_string()).collect(),
                [0.0, 0.0, 24.0, 24.0],
                Some(2.0),
                icon.fill,
            );
        }
        if !text.is_empty() {
            self.ui.add_text(b, text);
        }
        self.key(b, key, Some(action));
        b
    }

    /// Filled rectangles in a box's coordinates, as one path.
    pub(crate) fn rects(&mut self, parent: NodeId, rects: &[[f64; 4]], color: &str, size: (f32, f32)) {
        if rects.is_empty() {
            return;
        }
        self.ui.add_path(
            parent,
            style! { position: absolute; left: 0; top: 0; width: {size.0}; height: {size.1}; color: {color}; },
            vec![rects.iter().map(|r| plot::rect_path(*r)).collect::<String>()],
            [0.0, 0.0, size.0, size.1],
            None,
            true,
        );
    }
}

impl NgUi {
    pub(crate) fn build(&mut self, viewer: &mut Viewer, width: f32, height: f32, text: &mut dyn TextMeasure) -> bool {
        let hovered = std::mem::take(&mut self.hovered);
        self.viewport = (width, height);
        self.channel_cells.clear();
        // Boxes on Blink's 1/64 px grid where float error is all that keeps
        // them off it: where a run of text lands decides its quarter-pixel
        // glyph position.
        let mut ui = Ui::new();
        ui.set_layout_units(true);
        let mut b = Builder { ui, keys: HashMap::new(), actions: HashMap::new(), hovered: &hovered, text, centred: vec![], snapped: vec![], deferred: vec![], controls: HashMap::new(), viewport_height: height, titles: HashMap::new(), bindings: viewer.tool_bindings(), tools: vec![] };
        let root = b.ui.root();
        // `body`: white text in the default font (black, behind the panels'
        // canvas, is not painted here); `.neuroglancer-viewer` a column.
        b.ui.set_style(
            root,
            style! {
                display: flex;
                flex-direction: column;
                position: relative;
                color: white;
                font-family: {SERIF};
                font-size: 16px;
            },
        );
        // The shader typed sets after a pause, or as the editor blurs.
        if std::mem::take(&mut self.shader_blur_pending) {
            self.shader_blur(viewer);
        }
        self.shader_settle(viewer, false);
        // The side panels' locations renumbered (`renderSide`).
        crate::side::normalize_locations(viewer);
        // Data histograms are computed while a widget shows one.
        for li in 0..viewer.layers().len() {
            viewer.want_data_histogram(li, false);
        }
        let dimensions = self.top_row(&mut b, root, viewer);
        // `SidePanelManager`: docked panels left and right of the centre.
        let body = b.ui.add_box(root, style! { display: flex; flex: 1; min-height: 0; });
        self.side_columns(&mut b, body, viewer, crate::Side::Left);
        let center = b.ui.add_box(body, style! { display: flex; flex-direction: column; flex: 1; flex-basis: 0px; min-width: 0; });
        // Top and bottom panels around the centre, in its column.
        self.side_rows(&mut b, center, viewer, crate::Side::Top);
        // `.neuroglancer-layer-group-viewer`: the layer bar over the panels.
        let group = b.ui.add_box(center, style! { display: flex; flex-direction: column; flex: 1; min-width: 0; min-height: 0; });
        if crate::panels::layer_bar_shown(viewer) {
            self.layer_bar(&mut b, group, viewer);
        }
        let data = b.ui.add_box(group, style! { flex: 1; });
        self.side_rows(&mut b, center, viewer, crate::Side::Bottom);
        self.side_columns(&mut b, body, viewer, crate::Side::Right);

        b.ui.layout(width, height, b.text);
        let r = b.ui.rect(data);
        let changed = viewer.set_viewport(Some([r.x as f64, r.y as f64, r.width as f64, r.height as f64]));
        self.panel_controls(&mut b, root, viewer);
        // The status messages, over the panels at the bottom.
        if self.active_tool.is_some_and(|k| viewer.tool(k).is_none()) {
            self.active_tool = None;
            self.active_ref = None;
        }
        // The select tool's painting follows the segment under the pointer.
        if self.select_paint.as_ref().is_some_and(|p| p.painting) {
            self.try_select_segment(viewer);
        }
        self.status_messages(&mut b, root, viewer, width);
        // A layer being dragged: `.neuroglancer-drag-status`.
        if let Some((text, left)) = self.layer_drag_status(viewer) {
            let node = b.text_box(root, &text, style! { position: absolute; top: 0; padding: 2px; background: yellow; font-family: sans-serif; font-size: {SMALL}; color: black; });
            let s = b.ui.style_mut(node);
            if left {
                s.left = threers::ui::Length::Px(0.0);
            } else {
                s.right = threers::ui::Length::Px(0.0);
            }
        }
        // An open `<select>`'s options, under it.
        if let Some(key) = self.menu {
            match (b.keys.iter().find(|(_, k)| **k == key).map(|(n, _)| *n), b.controls.get(&key).cloned()) {
                (Some(node), Some(control)) if control.options().is_some() => {
                    let (options, value) = control.options().unwrap();
                    let r = b.ui.rect(node);
                    crate::rendering::select_menu(&mut b, root, r, &options, value);
                }
                _ => self.menu = None,
            }
        }
        // The colour picker, under the colour it sets.
        if let Some((setting, hsv)) = self.color_picker {
            let anchor = b.keys.iter().find(|(_, k)| **k == Key::Color(setting)).map(|(n, _)| b.ui.rect(*n));
            let (w, h) = (218.0, 196.0);
            let (x, y) = match anchor {
                Some(r) => (r.x.min(width - w).max(0.0), if r.y + r.height + h <= height { r.y + r.height + 2.0 } else { (r.y - h - 2.0).max(0.0) }),
                None => ((width - w) / 2.0, (height - h) / 2.0),
            };
            let font = style! { font-family: monospace; font-size: 13px; color: black; };
            let nodes = threers::ui::controls::color_picker(&mut b.ui, root, x, y, hsv, &font);
            b.key(nodes.popup, Key::ColorPicker, None);
            b.key(nodes.field, Key::ColorPickerField, None);
            b.key(nodes.hue, Key::ColorPickerHue, None);
        }
        // The wrench's list of palettes, under it.
        if self.palette_list {
            if let Some(node) = self.palette_button {
                // The button's box: the icon across, the bar down.
                let (icon, bar) = (b.ui.rect(node), b.ui.parent(node).map(|p| b.ui.rect(p)).unwrap_or_default());
                let r = threers::ui::Rect { x: icon.x, y: bar.y, width: icon.width, height: bar.height };
                self.palette_dropdown(&mut b, root, viewer, r, width);
            }
        }
        // A focused channel coordinate's dropdown: its plot.
        if let Some((_, cell, extent)) = self.channel_cells.iter().find(|(id, _, _)| self.field.as_ref().is_some_and(|(f, _)| f == id)).copied() {
            let current = self.field.as_ref().map_or(0.0, |(_, t)| ng_core::js::parse_float(t.text.trim()));
            let rect = b.ui.rect(cell);
            channel_dropdown(&mut b, root, rect, extent, current);
        }
        // The open dimension's dropdown hangs below it, over everything.
        let mut plot = None;
        if let Some(dim) = self.dropdown {
            if let Some(&(_, container)) = dimensions.iter().find(|(d, _)| *d == dim) {
                let rect = b.ui.rect(container);
                plot = self.dimension_dropdown(&mut b, root, viewer, dim, rect);
            }
        }
        // The port's credits, at the bottom over the panels; a click opens
        // the repository.
        let mut footer_node = None;
        if self.attribution {
            let font = style! { font-family: sans-serif; font-size: 11px; };
            let line = crate::about::footer();
            let w = b.width(&line, &font) + 12.0;
            let footer = b.ui.add_box(
                root,
                style! {
                    position: absolute;
                    left: {((width - w) / 2.0).max(0.0).round()};
                    top: {(height - 17.0).round()};
                    padding: 1px 6px;
                    border-radius: 3px;
                    background: {[0.0, 0.0, 0.0, 0.55]};
                    font-family: sans-serif;
                    font-size: 11px;
                    color: {[1.0, 1.0, 1.0, if b.hovered.contains(&Key::Icon(FOOTER_LINK)) { 1.0 } else { 0.7 }]};
                    white-space: nowrap;
                },
            );
            b.ui.add_text(footer, line);
            footer_node = Some(footer);
            b.key(footer, Key::Icon(FOOTER_LINK), Some(Action::OpenUrl(crate::about::REPOSITORY.into())));
            b.title(Key::Icon(FOOTER_LINK), crate::about::REPOSITORY);
        }
        // The tooltip of what the pointer has rested on (`title`).
        if let (Some((x, y)), Some(text)) = (self.pointer, self.tooltip()) {
            let tip = b.ui.add_box(
                root,
                style! { position: absolute; left: {x}; top: {y + 20.0}; max-width: 300px; padding: 2px 4px; border: 1px solid #767676; background: white; font-family: sans-serif; font-size: 12px; color: black; white-space: normal; },
            );
            b.ui.add_text(tip, text);
        }
        // A dialog over everything.
        if self.dialog.is_some() {
            self.mono_advance = b.ch("monospace", 13.0);
            self.dialog(&mut b, root, viewer);
        }
        b.ui.layout(width, height, b.text);
        // The footer clear of the 3D panel's "Sections" (its label to the
        // checkbox's left): moved left of them where they meet.
        if let Some(footer) = footer_node {
            let f = b.ui.rect(footer);
            if let Some(check) = b.keys.iter().find(|(_, k)| **k == Key::Sections).map(|(n, _)| *n) {
                let c = b.ui.rect(check);
                let (left, right) = (c.x - 64.0, c.x + c.width + 4.0);
                if f.y < c.y + c.height + 4.0 && f.x < right && f.x + f.width > left {
                    // Left of them where it fits, else right of them, else
                    // a line above them.
                    if f.width + 6.0 <= left {
                        b.ui.offset(footer, left - 6.0 - f.width - f.x, 0.0);
                    } else if right + 6.0 + f.width <= width {
                        b.ui.offset(footer, right + 6.0 - f.x, 0.0);
                    } else {
                        b.ui.offset(footer, 0.0, c.y - 4.0 - (f.y + f.height));
                    }
                }
            }
        }
        for &(cell, inner) in &b.centred {
            let (c, i) = (b.ui.rect(cell), b.ui.rect(inner));
            let snap = |a: f32, len: f32| {
                let s = (a + 0.5).floor();
                (s, (a + len + 0.5).floor() - s)
            };
            let ((sx, sw), (sy, sh)) = (snap(c.x, c.width), snap(c.y, c.height));
            let x = sx + (sw - i.width) / 2.0;
            let y = sy + (sh - i.height) / 2.0;
            b.ui.offset(inner, x - i.x, y - i.y);
        }
        for &node in &b.snapped {
            let r = b.ui.rect(node);
            b.ui.offset(node, r.x.round() - r.x, r.y.round() - r.y);
        }
        if !b.deferred.is_empty() {
            // Legends' widths, for the colours asked for next.
            for d in &b.deferred {
                if let crate::rendering::Deferred::Legend(l) = d {
                    let w = (b.ui.rect(l.node).width - 2.0).round().max(1.0) as usize;
                    self.legend_widths.insert(l.key, w);
                }
            }
            for d in std::mem::take(&mut b.deferred) {
                d.draw(&mut b.ui);
            }
            b.ui.layout(width, height, b.text);
        }
        // The selected annotation's entry scrolled into view: as little as
        // shows it whole.
        self.scroll_to_annotation = None;
        if let Some((list, entry)) = self.annotation_scroll.take() {
            let (l, e) = (b.ui.rect(list), b.ui.rect(entry));
            let top = b.ui.style(list).scroll_top;
            let target = if e.y < l.y { top - (l.y - e.y) } else if e.y + e.height > l.y + l.height { top + (e.y + e.height - l.y - l.height) } else { top };
            if target != top {
                self.scroll.insert(self.layer_panel, target.max(0.0));
                b.ui.style_mut(list).scroll_top = target.max(0.0);
                b.ui.layout(width, height, b.text);
            }
        }
        self.plot = plot.map(|(dim, canvas, bounds)| PlotGeometry { dim, canvas: b.ui.rect(canvas), bounds });

        let Builder { ui, keys, actions, controls, titles, tools, .. } = b;
        self.tool_refs = tools;
        self.controls = controls;
        self.titles = titles;
        self.ui = ui;
        self.keys = keys;
        self.actions = actions;
        self.hovered = hovered;
        // What is under the pointer may have moved.
        if let Some((x, y)) = self.pointer {
            self.hovered = self.keys_at(x, y);
        }
        changed
    }

    /// `.neuroglancer-viewer-top-row`. Returns each dimension's container.
    fn top_row(&mut self, b: &mut Builder, root: NodeId, viewer: &mut Viewer) -> Vec<(usize, NodeId)> {
        let row = b.ui.add_box(
            root,
            style! {
                display: flex;
                height: 20px;
                flex-shrink: 0;
                align-items: center;
                background: #222;
                margin-bottom: 3px;
                padding-right: 2px;
            },
        );
        // `.neuroglancer-position-widget`.
        let widget = b.ui.add_box(row, style! { display: flex; align-items: center; align-self: stretch; });
        let space = viewer.dimensions().clone();
        let valid = viewer.navigation().valid();
        let position: Vec<f32> = viewer.navigation().position().to_vec();
        let mono_ch = b.ch("monospace", MONOSPACE_MEDIUM);
        let serif_ch = b.ch(SERIF, 16.0);
        let mut dimensions = vec![];
        self.dimension_texts = (0..space.rank())
            .map(|i| {
                let (scale, prefix, unit) = ng_core::si_units::format_scale_with_unit(space.scales[i], &space.units[i]);
                (space.names[i].clone(), format!("{scale}{prefix}{unit}"))
            })
            .collect();
        if space.valid {
            for (i, name) in space.names.iter().enumerate() {
                let editing = self.edit.as_ref().filter(|e| e.dim == i);
                // `:focus-within` and `[data-dropdown-visible]`: a grey border
                // on black.
                // A name or scale being edited (`nameElement.disabled = false`).
                let field = |id: crate::FieldId| self.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t.clone());
                let name_field = field(crate::FieldId::DimensionName(i));
                let scale_field = field(crate::FieldId::DimensionScale(i));
                let focused = editing.is_some() || self.dropdown == Some(i) || name_field.is_some() || scale_field.is_some();
                // `.neuroglancer-position-dimension`: 1 px transparent border,
                // `1ch` apart.
                let dim = b.ui.add_box(
                    widget,
                    style! {
                        display: flex;
                        border: 1px solid;
                        border-color: {if focused { "#aaa" } else { "transparent" }};
                        background: {focused.then_some([0.0, 0.0, 0.0, 1.0])};
                        // The row's height; the inputs sit a pixel down
                        // (their baselines on the line's).
                        padding-top: 1px;
                        margin-left: {if i > 0 { lu(serif_ch) } else { 0.0 }};
                        align-self: stretch;
                        align-items: flex-start;
                        flex-shrink: 0;
                        position: relative;
                    },
                );
                b.key(dim, Key::Dimension(i), None);
                dimensions.push((i, dim));
                // A field's text, and its selection (or the caret, while it
                // blinks on).
                let caret_on = self.caret_on();
                let shown = |f: &Option<threers::ui::TextField>, value: &str| match f {
                    Some(t) => {
                        let (a, z) = (t.anchor.min(t.caret), t.anchor.max(t.caret));
                        (t.text.clone(), (a != z || caret_on).then_some((a, z)))
                    }
                    None => (value.to_owned(), None),
                };
                let (name_text, name_selection) = shown(&name_field, name);
                let name_box = input(b, dim, &name_text, name_text.chars().count().max(1), "#ff6", mono_ch, false, name_selection);
                b.key(name_box, Key::DimensionName(i), None);
                let (scale, prefix, unit) = ng_core::si_units::format_scale_with_unit(space.scales[i], &space.units[i]);
                let scale = format!("{scale}{prefix}{unit}");
                // `.neuroglancer-position-dimension-scale-container`, with
                // `(` and `)` (`::before`, `::after`) when not empty.
                let container = b.ui.add_box(
                    dim,
                    style! {
                        display: flex;
                        margin: {Edges::new(0.0, if scale.is_empty() { 2.0 } else { 5.0 }, 0.0, 4.0)};
                        align-items: flex-start;
                        flex-shrink: 0;
                    },
                );
                // The serif parentheses share the inputs' baseline.
                let paren = style! {
                    color: #aaa;
                    font-family: {SERIF};
                    font-size: 16px;
                    line-height: 18px;
                    margin-top: -1.5px;
                    flex-shrink: 0;
                };
                if !scale.is_empty() {
                    b.text_box(container, "(", paren.clone());
                }
                let (scale_text, scale_selection) = shown(&scale_field, &scale);
                let scale_box = input(b, container, &scale_text, scale_text.chars().count().max(1), "#bbb", mono_ch, false, scale_selection);
                b.key(scale_box, Key::DimensionScale(i), None);
                if !scale.is_empty() {
                    b.text_box(container, ")", paren);
                }
                // Play or pause while it has a velocity (`showPlayback`):
                // `.neuroglancer-position-dimension > .neuroglancer-icon`, 14px
                // with a 12px icon, `top: 1px` (a margin, as it is centred).
                if let Some(v) = viewer.velocity(i) {
                    let key = Key::Icon(PLAY_ICON + i as u32);
                    let button = b.ui.add_box(
                        dim,
                        style! {
                            display: flex;
                            min-width: 14px;
                            min-height: 14px;
                            padding: 0 2px;
                            margin: -2px 1px 0 1px;
                            align-self: center;
                            align-items: center;
                            justify-content: center;
                            border-radius: 20%;
                            background: {b.hover(key).then_some([0xdb as f32 / 255.0, 0x44 as f32 / 255.0, 0x37 as f32 / 255.0, 1.0])};
                            flex-shrink: 0;
                        },
                    );
                    let icon = if v.paused { icons::PLAY } else { icons::PAUSE };
                    b.ui.add_path(button, style! { width: 12px; height: 12px; flex-shrink: 0; color: white; }, icon.paths.iter().map(|s| s.to_string()).collect(), [0.0, 0.0, 24.0, 24.0], Some(2.0), icon.fill);
                    b.key(button, key, Some(Action::SetVelocity(i, Some(ng_viewer::playback::DimensionVelocity { paused: !v.paused, ..v }))));
                }
                // `.neuroglancer-position-dimension-coordinate`, right-aligned,
                // as wide as the widest coordinate the bounds allow.
                let coordinate = match (editing, self.pending.get(&i)) {
                    (Some(e), _) => e.text.clone(),
                    (None, Some(typed)) => typed.clone(),
                    (None, None) if valid => coordinate_text(position[i]),
                    (None, None) => String::new(),
                };
                let len = coordinate.chars().count();
                let digits = |v: f64| if v.is_finite() { number_to_string(v).len() } else { 0 };
                let bounds = &space.bounds;
                let max_width = digits(bounds.lower_bounds[i].floor()).max(digits(bounds.upper_bounds[i].ceil()));
                let seen = self.max_position_seen.entry(name.clone()).or_insert(max_width);
                *seen = (*seen).max(len);
                let width = max_width.min(*seen).max(len).max(1);
                // A caret only while it blinks on.
                let caret_on = self.caret_on();
                let selection = editing.map(|e| (e.start, e.end)).filter(|(a, z)| a != z || caret_on);
                let c = input(b, dim, &coordinate, width, "white", mono_ch, true, selection);
                b.key(c, Key::Coordinate(i), None);
            }
        }
        b.icon(widget, crate::view::COPY_ICON, Some(icons::COPY), "", Edges::symmetric(0.0, 1.0), Action::CopyPosition);
        b.title(Key::Icon(crate::view::COPY_ICON), "Copy position to clipboard");

        // `.neuroglancer-mouse-position-widget`: orange, `flex: 1`.
        let names = space.names.clone();
        let mouse = viewer.mouse_position().map(|p| mouse_position_text(&names, &p)).unwrap_or_default();
        let m = b.ui.add_box(
            row,
            style! {
                flex: 1;
                margin-left: {lu(mono_ch)};
                align-self: center;
                overflow: hidden;
                font-family: monospace;
                font-size: {MONOSPACE_MEDIUM};
                color: orange;
            },
        );
        if !mouse.is_empty() {
            b.ui.add_text(m, mouse);
        }
        // `AnnotationToolStatusWidget`: the bound tools.
        self.tool_status_widget(b, row, viewer);
        // `LayerArchiveCountWidget`: shown layers / all, once any is archived.
        let total = viewer.layers().len();
        let archived = viewer.layers().iter().filter(|l| l.spec.archived).count();
        let count = if archived > 0 { format!("{}/{total}", total - archived) } else { String::new() };
        // `MultiToolPaletteDropdownButton`: shown palettes / all, while
        // some are hidden.
        let palettes: Vec<bool> = (0..crate::palettes::count(viewer) as u16).map(|k| crate::PanelId::Palette(k).location(viewer).visible).collect();
        let shown = palettes.iter().filter(|v| **v).count();
        let palette_count = if shown < palettes.len() { format!("{shown}/{}", palettes.len()) } else { String::new() };
        let mod_label = if viewer.mac { "⌘" } else { "Ctrl" };
        for (i, (icon, text, title)) in TOP_ICONS.iter().enumerate() {
            let text = match *title {
                "Show layer list panel" => count.as_str(),
                "Tool palettes" => palette_count.as_str(),
                _ => text,
            };
            // Panel buttons are `CheckboxIcon`s: lit while their panel shows.
            let panel = match *title {
                "Show layer list panel" => Some(crate::PanelId::LayerList),
                "Show selection details panel" => Some(crate::PanelId::Selection),
                "Show layer side panel" => Some(crate::PanelId::Layer),
                "Show help panel" => Some(crate::PanelId::Help),
                "Show settings panel" => Some(crate::PanelId::Settings),
                _ => None,
            };
            let action = match (panel, *title) {
                (Some(p), _) => Action::TogglePanel(p),
                (None, "Copy URL to clipboard") => Action::CopyUrl,
                (None, "Edit JSON state") => Action::EditJsonState,
                (None, "Screenshot") => Action::OpenScreenshot,
                (None, "Tool palettes") => Action::TogglePaletteList,
                (None, t) => Action::Unported(t),
            };
            let node = b.icon(row, i as u32, *icon, text, Edges::symmetric(0.0, 1.0), action);
            if *title == "Tool palettes" {
                b.title(Key::Icon(i as u32), if self.palette_list { "Hide tool palette list".to_owned() } else { format!("Show tool palette list ({mod_label}+click to create new)") });
                if self.palette_list {
                    b.ui.style_mut(node).background = Some([1.0, 1.0, 1.0, 0.2]);
                }
                self.palette_button = Some(node);
            } else {
                b.title(Key::Icon(i as u32), *title);
            }
            if icon.is_some() && !text.is_empty() {
                // The archive count goes before the icon
                // (`insertAdjacentElement("afterbegin", …)`).
                let t = *b.ui.children(node).last().unwrap();
                b.ui.move_child(node, t, 0);
            }
            if panel.is_some_and(|p| p.location(viewer).visible) {
                // `.dark-background[data-checked=true]`, which outranks
                // `:hover`.
                b.ui.style_mut(node).background = Some([1.0, 1.0, 1.0, 0.2]);
            }
        }
        dimensions
    }

    /// `.neuroglancer-position-dimension-dropdown` under dimension `dim`
    /// (whose container is at `container`): the tool key, the plot, the
    /// playback header. Returns the plot's canvas node and bounds.
    fn dimension_dropdown(&mut self, b: &mut Builder, root: NodeId, viewer: &Viewer, dim: usize, container: Rect) -> Option<(usize, NodeId, plot::DimensionBounds)> {
        let space = viewer.dimensions();
        if dim >= space.rank() {
            return None;
        }
        // `left: -1px` from the padding box, below the content; at least as
        // wide as the dimension (`calc(100% + 2px)`).
        let dropdown = b.ui.add_box(
            root,
            style! {
                position: absolute;
                left: {container.x};
                top: {container.y + container.height - 1.0};
                min-width: {container.width};
                box-sizing: border-box;
                border: 1px solid #aaa;
                padding: 2px;
                background: black;
                display: flex;
                flex-direction: column;
                align-items: stretch;
            },
        );
        b.key(dropdown, Key::Dropdown, None);
        // The tool button (`makeToolButton`): an unbound key, `[ ]`, on a
        // line of the dropdown's font.
        let line = b.ui.add_box(dropdown, style! { display: flex; height: 18px; align-items: flex-start; });
        let name = space.names.get(dim).cloned().unwrap_or_default();
        let key = b.tool_binding(line, crate::tools::ToolRef::dimension(&name), None);
        let m = &mut b.ui.style_mut(key).margin;
        m.top += 3.0;
        // The plot (`PositionPlot`, column orientation): bound labels, 3 px,
        // the canvas with 0.5em above and below.
        let mut bounds = plot::dimension_bounds(space, dim, plot::CANVAS_HEIGHT);
        let mut canvas = None;
        if let Some(bounds) = &mut bounds {
            if !self.show_all_plot_bounds && !bounds.boxes.is_empty() {
                let lo = bounds.boxes.iter().map(|b| b.0).fold(f64::INFINITY, f64::min);
                let hi = bounds.boxes.iter().map(|b| b.1).fold(f64::NEG_INFINITY, f64::max);
                bounds.boxes = vec![(lo, hi)];
            }
            let label_style = style! { font-family: monospace; font-size: {MONOSPACE_MEDIUM}; color: white; };
            let current = viewer.navigation().position()[dim] as f64;
            let position_row = plot::indicator(bounds, current, plot::CANVAS_HEIGHT);
            // While dragging, the label shows the position.
            let hover = if self.plot_dragged { Some(current.floor()) } else { self.plot_hover };
            let hover_row = if self.plot_dragged { position_row } else { hover.and_then(|h| plot::indicator(bounds, h, plot::CANVAS_HEIGHT)) };
            let (lower, upper) = (number_to_string(bounds.lower), number_to_string(bounds.upper));
            let hover_text = hover.map(number_to_string);
            let labels_width = [Some(&lower), Some(&upper), hover_text.as_ref().filter(|_| hover_row.is_some())]
                .into_iter()
                .flatten()
                .map(|t| b.width(t, &label_style))
                .fold(0.0f32, f32::max);
            let grid = b.ui.add_box(dropdown, style! { display: flex; position: relative; overflow: hidden; });
            let labels = b.ui.add_box(grid, style! { position: relative; width: {labels_width}; height: 116px; flex-shrink: 0; });
            // Bounds labels hide where the hover label would cover them.
            let show_lower = hover_row.is_none_or(|o| o as f32 > 15.0);
            let show_upper = hover_row.is_none_or(|o| (o as f32) < plot::CANVAS_HEIGHT as f32 - 17.0);
            let label = |b: &mut Builder, text: &str, top: f32, shown: bool| {
                b.text_box(labels, text, style! { ..label_style.clone(); position: absolute; right: 0; top: {top}; opacity: {if shown { 1.0 } else { 0.0 }}; });
            };
            label(b, &lower, 0.0, show_lower);
            label(b, &upper, plot::CANVAS_HEIGHT as f32 - 1.0, show_upper);
            if let (Some(t), Some(row)) = (&hover_text, hover_row) {
                label(b, t, row as f32, true);
            }
            let c = b.ui.add_box(
                grid,
                style! {
                    position: relative;
                    width: {plot::CANVAS_WIDTH as f32};
                    height: {plot::CANVAS_HEIGHT as f32};
                    margin: 8px 0 8px 3px;
                    flex-shrink: 0;
                },
            );
            b.key(c, Key::Plot, None);
            let size = (plot::CANVAS_WIDTH as f32, plot::CANVAS_HEIGHT as f32);
            let (ticks, bars) = plot::bars(bounds);
            b.rects(c, &ticks, "#fff", size);
            b.rects(c, &bars, "#ccc", size);
            let line = |row: f64| [0.0, row, plot::CANVAS_WIDTH, 1.0];
            if let Some(row) = position_row {
                b.rects(c, &[line(row)], "#f66", size);
            }
            if let (Some(row), false) = (hover_row, self.plot_dragged) {
                b.rects(c, &[line(row)], "#66f", size);
            }
            canvas = Some(c);
        }
        // `.neuroglancer-position-dimension-playback`: the header (a
        // `CheckboxIcon`, dark scheme), then, while enabled, the speed and
        // what happens at the bounds.
        let velocity = viewer.velocity(dim);
        if let Some(v) = velocity {
            self.velocities.insert(dim, v);
        } else {
            self.velocities.remove(&dim);
        }
        let playback = b.ui.add_box(dropdown, style! { display: flex; flex-direction: column; font-family: sans-serif; font-size: {SMALL}; color: white; });
        let header = b.ui.add_box(playback, style! { display: flex; align-items: center; });
        let toggle = if velocity.is_some() { None } else { Some(ng_viewer::playback::DimensionVelocity::default()) };
        let icon = b.icon(header, PLAYBACK_ICON, Some(icons::VIDEO), "", Edges::symmetric(0.0, 1.0), Action::SetVelocity(dim, toggle));
        // `.dark-background[data-checked="true"]`, over the hover colour too.
        if velocity.is_some() {
            b.ui.style_mut(icon).background = Some([1.0, 1.0, 1.0, 0.2]);
        }
        b.title(Key::Icon(PLAYBACK_ICON), if velocity.is_some() { "Disable playback/velocity" } else { "Enable playback/velocity" });
        b.text_box(header, "Playback", style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
        if let Some(v) = velocity {
            let row = b.ui.add_box(playback, style! { display: flex; align-items: center; margin-bottom: 2px; });
            b.icon(row, PLAYBACK_ICON + 1, None, "±", Edges::symmetric(0.0, 1.0), Action::SetVelocity(dim, Some(ng_viewer::playback::DimensionVelocity { velocity: -v.velocity, ..v })));
            b.title(Key::Icon(PLAYBACK_ICON + 1), "Negate velocity");
            let id = crate::FieldId::Velocity(dim);
            let editing = self.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t.clone());
            // An `<input>`: 13.333px (the UA's), not monospace's 13px.
            let ch = b.ch("monospace", SMALL);
            let input = b.ui.add_box(row, style! { display: flex; position: relative; justify-content: flex-end; align-items: center; width: {lu(8.0 * ch)}; padding: 1px 2px; background: {[1.0, 1.0, 1.0, 0.3]}; flex-shrink: 0; });
            b.key(input, Key::Field(id), None);
            b.title(Key::Field(id), "Velocity in coordinates per second");
            let font = style! { font-family: monospace; font-size: {SMALL}; color: white; };
            let text = number_to_string(v.velocity);
            let view = threers::ui::controls::TextView {
                value: &text,
                placeholder: "",
                placeholder_color: [0.5; 4],
                editing: editing.as_ref(),
                caret_on: self.caret_on(),
                font: &font,
                x_offset: 2.0,
                selection: threers::ui::controls::SELECTION,
            };
            threers::ui::controls::text_contents(&mut b.ui, input, view, b.text);
            b.text_box(row, "/s", style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
            let select_font = style! { font-family: Arial; font-size: {SMALL}; color: black; };
            let select = threers::ui::controls::select(&mut b.ui, playback, v.at_boundary.name(), &crate::rendering::BOUNDARIES, &select_font, b.text);
            b.ui.style_mut(select).align_self = Some(threers::ui::AlignItems::Center);
            let key = Key::Control(usize::MAX - dim, "atBoundary");
            b.key(select, key, Some(Action::Repaint));
            b.title(key, "Behavior when lower/upper bound is reached");
            b.controls.insert(key, crate::rendering::Control::Boundary { dim, value: v.at_boundary.name() });
        }
        Some((dim, canvas?, bounds?))
    }

    /// `.neuroglancer-layer-panel`.
    fn layer_bar(&mut self, b: &mut Builder, parent: NodeId, viewer: &mut Viewer) {
        let panel = b.ui.add_box(
            parent,
            style! {
                display: flex;
                width: 100%;
                margin: -1px 0 0 -5px;
                flex-shrink: 0;
                align-items: stretch;
            },
        );
        b.key(panel, Key::LayerPanel, None);
        let panel_hover = b.hover(Key::LayerPanel);
        let mono_ch = b.ch("monospace", MONOSPACE_MEDIUM);
        let position = viewer.mouse_position();
        // Archived layers stay in the state, out of the bar; the rest are
        // numbered in order.
        let shown = viewer.layers().iter().enumerate().filter(|(_, l)| !l.spec.archived);
        let selected = crate::PanelId::Layer.layer(viewer);
        for (number, (i, layer)) in shown.enumerate() {
            let key = Key::LayerItem(i);
            let visible = layer.spec.visible;
            let color = if visible { "white" } else { "#bbb" };
            // `[data-selected]`: green; under the pointer, gold (or brighter
            // green).
            let border = match (selected == Some(i), b.hover(key)) {
                (true, true) => "#3c3",
                (true, false) => "#8c8",
                (false, true) => "#daa520",
                (false, false) => "#ccc",
            };
            let item = b.ui.add_box(
                panel,
                style! {
                    display: flex;
                    position: relative;
                    align-items: center;
                    color: {color};
                    border: 1px solid;
                    border-color: {border};
                    font-family: sans-serif;
                    font-size: {SMALL};
                    background: black;
                    padding: 1px;
                    margin: 1px 1px 1px 5px;
                    flex-shrink: 0;
                },
            );
            b.key(item, key, Some(Action::ToggleLayerVisible(i)));
            // `LayerWidget.update`.
            let mut title = format!("Click to {}, {}+click to show side panel", if visible { "hide" } else { "show" }, if viewer.mac { "⌘" } else { "control" });
            if matches!(layer.kind(), Some(ng_viewer::LayerKind::Segmentation(_))) {
                let picks = layer.spec.json.get("pick").and_then(serde_json::Value::as_bool).unwrap_or(true);
                title += &format!(", {}+click to {} spatial object selection", if viewer.mac { "⌥" } else { "alt" }, if picks { "disable" } else { "enable" });
            }
            title += ", drag to move, shift+drag to copy";
            b.title(key, title);
            // Loaded fractions of the visible and prefetched chunks.
            let (needed, available) = layer.chunk_progress();
            let fraction = available as f32 / needed.max(1) as f32;
            b.ui.add_box(item, style! { position: absolute; left: 0; top: 0; height: 4px; width: {Length::Percent(fraction * 100.0)}; background: #666; });
            b.ui.add_box(item, style! { position: absolute; left: 0; bottom: 0; height: 4px; width: 0%; background: #666; });
            b.text_box(
                item,
                &(number + 1).to_string(),
                style! {
                    padding: 0 1px;
                    background: #9a7518;
                    font-family: sans-serif;
                    font-size: {SMALL};
                    font-weight: bold;
                    color: {color};
                },
            );
            // The label over its colour (`setColor`).
            let wrapper = b.ui.add_box(item, style! { display: flex; position: relative; overflow: hidden; margin: 0 2px; });
            let colors = layer.layer_bar_colors().unwrap_or_default();
            let label_color = match colors.len() {
                0 => color,
                1 if use_white_background(colors[0]) => "white",
                1 => "black",
                _ => "white",
            };
            if !colors.is_empty() {
                // `.neuroglancer-layer-item-label-color`: one colour fills the
                // label; several are `createSteppedCssGradient`, scaled by 1.2
                // and blurred 8 px (`[data-color="multi"]`).
                let multi = colors.len() > 1;
                let grow = if multi { 10.0 } else { 0.0 };
                let rgba = |c: [f32; 3]| [c[0], c[1], c[2], 1.0];
                let step = 100.0 / colors.len() as f64;
                let gradient: Vec<(f32, Rgba)> = if multi {
                    colors
                        .iter()
                        .enumerate()
                        .flat_map(|(i, &c)| {
                            let start = ng_core::js::round(i as f64 * step) as f32 / 100.0;
                            let end = ng_core::js::round((i + 1) as f64 * step) as f32 / 100.0;
                            [(start, rgba(c)), (end, rgba(c))]
                        })
                        .collect()
                } else {
                    vec![]
                };
                b.ui.add_box(
                    wrapper,
                    style! {
                        position: absolute;
                        left: {Length::Percent(-grow)};
                        top: {Length::Percent(-grow)};
                        width: {Length::Percent(100.0 + 2.0 * grow)};
                        height: {Length::Percent(100.0 + 2.0 * grow)};
                        background: {(!multi).then(|| rgba(colors[0]))};
                        background-image: {gradient};
                        // `blur(8px)` before `scale(1.2)`.
                        filter: {if multi { 8.0 * 1.2 } else { 0.0 }};
                    },
                );
            }
            // `position: relative; z-index: 1`: over the colour.
            b.text_box(
                wrapper,
                layer.name(),
                style! {
                    position: relative;
                    padding: 0 1px;
                    text-decoration: {!visible};
                    font-family: sans-serif;
                    font-size: {SMALL};
                    color: {label_color};
                },
            );
            let value = if self.show_layer_hover_values {
                position.as_ref().and_then(|p| viewer.layer_value_at(i, p)).unwrap_or_default()
            } else {
                String::new()
            };
            let max_len = self.max_value_len.entry(layer.id).or_insert(0);
            *max_len = (*max_len).max(value.chars().count());
            // `.neuroglancer-layer-item-value-container`: a grid cell holding
            // the value and the buttons over each other, as wide as the wider.
            // Neither is positioned (the progress strips paint over both), so
            // they overlap here by a negative margin.
            let value_width = if *max_len > 0 { lu(*max_len as f32 * mono_ch) } else { 0.0 };
            let value_outer = lu(mono_ch) + value_width;
            let cell = value_outer.max(48.0);
            let values = b.ui.add_box(item, style! { display: flex; width: {cell}; height: 18px; align-items: center; flex-shrink: 0; });
            let v = b.ui.add_box(
                values,
                style! {
                    display: flex;
                    font-family: monospace;
                    font-size: {MONOSPACE_MEDIUM};
                    color: white;
                    width: {value_width};
                    margin-left: {lu(mono_ch)};
                    justify-content: center;
                    opacity: {if panel_hover { 0.0 } else { 1.0 }};
                    flex-shrink: 0;
                },
            );
            if !value.is_empty() {
                b.ui.add_text(v, value);
            }
            // `justify-self: right`; hidden (`visibility: hidden`) until the
            // bar is under the pointer, which a press on them always is.
            let buttons = b.ui.add_box(
                values,
                style! {
                    display: flex;
                    margin-left: {cell - 48.0 - value_outer};
                    height: 18px;
                    flex-shrink: 0;
                    align-items: center;
                    opacity: {if panel_hover { 1.0 } else { 0.0 }};
                },
            );
            b.icon(buttons, layer_icon(i, false), Some(icons::CLOSE), "", Edges::symmetric(0.0, 1.0), Action::ArchiveLayer(i));
            b.icon(buttons, layer_icon(i, true), Some(icons::BIN), "", Edges::symmetric(0.0, 1.0), Action::DeleteLayer(i));
            b.title(Key::Icon(layer_icon(i, false)), "Remove layer from this layer group");
            b.title(Key::Icon(layer_icon(i, true)), "Delete this layer");
        }
        b.icon(panel, ADD_LAYER_ICON, Some(icons::PLUS), "", Edges::new(1.0, 1.0, 1.0, 5.0), Action::AddLayer);
        // `.neuroglancer-layer-panel-drop-zone`.
        b.ui.add_box(panel, style! { flex: 1; });
    }

    /// Each data panel's controls, over its padding box (inside the 2 px
    /// border): the display dimensions, the layout buttons
    /// (`registerRelatedLayouts`), "Sections" on the 3D panel.
    fn panel_controls(&mut self, b: &mut Builder, root: NodeId, viewer: &Viewer) {
        let layout = viewer.layout;
        let rects = viewer.logical_panel_rects();
        self.panels = rects.iter().map(|(_, r)| r.map(|v| v as f32)).collect();
        if self.layout != Some(layout) {
            // New panels. Chrome follows a layout change with a synthetic
            // mouse move, so the panel under the pointer is entered and takes
            // focus; failing that, the first takes it if the focused element
            // went with the old panels.
            self.layout = Some(layout);
            self.entered = self.pointer.and_then(|(x, y)| self.panel_at(x, y));
            self.focused = self.entered.or(std::mem::take(&mut self.refocus).then_some(0));
        }
        let names = viewer.display_dimension_names();
        let mono12_ch = b.ch("monospace", 12.0);
        for (pi, (panel, [x, y, w, h])) in rects.into_iter().enumerate() {
            let (x, y, w, h) = (x as f32, y as f32, w as f32, h as f32);
            // The panel's border (`showPanelBorders`), painted over the
            // canvas: black, or white with focus (`:focus-within`).
            // (A dialog's input with the keyboard leaves no panel focused.)
            let keyboard_elsewhere = matches!(&self.dialog, Some(crate::dialogs::Dialog::Palette(_))) || matches!(&self.dialog, Some(crate::dialogs::Dialog::StateEditor(e)) if e.focused);
            let border = if self.focused == Some(pi) && !keyboard_elsewhere { "white" } else { "black" };
            b.ui.add_box(root, style! { position: absolute; left: {x}; top: {y}; width: {w - 4.0}; height: {h - 4.0}; border: 2px solid; border-color: {border}; });
            let overlay = b.ui.add_box(root, style! { position: absolute; left: {x + 2.0}; top: {y + 2.0}; width: {w - 4.0}; height: {h - 4.0}; });
            let (dimensions_widget, related, sections) = panel_furniture(layout, panel);
            if dimensions_widget {
                let widget = b.ui.add_box(
                    overlay,
                    style! {
                        display: flex;
                        flex-direction: column;
                        position: absolute;
                        top: 2px;
                        left: 2px;
                        padding: 2px 4px;
                        background: rgba(0, 0, 0, 0.3);
                    },
                );
                b.key(widget, Key::PanelWidget(pi), None);
                // Always three rows; a dimension not displayed has an empty
                // name (still `1ch` wide; `min-width: 10px` holds the border
                // box).
                for (i, color) in DIMENSION_COLORS.iter().enumerate() {
                    let name = names.get(i).map_or("", String::as_str);
                    b.text_box(
                        widget,
                        name,
                        style! {
                            display: flex;
                            width: {lu(name.chars().count().max(1) as f32 * mono12_ch).max(10.0 - 6.0)};
                            height: 14px;
                            padding: 2px;
                            border: 1px solid transparent;
                            justify-content: flex-end;
                            align-items: center;
                            font-family: monospace;
                            font-size: 12px;
                            color: {*color};
                        },
                    );
                }
            }
            let controls = b.ui.add_box(overlay, style! { display: flex; position: absolute; top: 2px; right: 2px; align-items: flex-start; });
            b.key(controls, Key::PanelWidget(pi), None);
            for (k, target) in related.iter().enumerate() {
                let key = Key::LayoutButton(pi, k);
                let button = b.ui.add_box(
                    controls,
                    style! {
                        display: flex;
                        box-sizing: border-box;
                        width: 18px;
                        height: 18px;
                        margin: 2px;
                        padding: 1px 6px;
                        border: 1px solid white;
                        background: rgba(0, 0, 0, 0.5);
                        align-items: center;
                        justify-content: center;
                        flex-shrink: 0;
                    },
                );
                b.key(button, key, Some(Action::SetLayout(*target)));
                let color = if b.hover(key) { "#6ff" } else { "white" };
                b.text_box(
                    button,
                    layout_symbol(*target),
                    style! { width: 15px; height: 15px; line-height: 12px; font-family: Arial; font-size: 15px; color: {color}; },
                );
            }
            if sections {
                let label = b.ui.add_box(
                    overlay,
                    style! {
                        display: flex;
                        position: absolute;
                        right: 2px;
                        bottom: 2px;
                        align-items: flex-end;
                    },
                );
                b.text_box(label, "Sections", style! { font-family: sans-serif; font-size: 13px; color: white; });
                checkbox(b, label, viewer.show_slices, Key::Sections, Action::ToggleShowSlices, Edges::new(3.0, 3.0, 3.0, 4.0));
            }
        }
    }
}

/// A channel position's dropdown (`PositionWidget` over the channel space,
/// no tool or playback): the plot of `[0, extent)` with the coordinate.
fn channel_dropdown(b: &mut Builder, root: NodeId, container: Rect, extent: u32, current: f64) {
    let dropdown = b.ui.add_box(
        root,
        style! {
            position: absolute;
            left: {container.x};
            top: {container.y + container.height - 1.0};
            min-width: {container.width};
            box-sizing: border-box;
            border: 1px solid #aaa;
            padding: 2px;
            background: black;
            display: flex;
            flex-direction: column;
            align-items: stretch;
        },
    );
    b.key(dropdown, Key::Dropdown, None);
    let space = ng_core::CoordinateSpace::new(true, vec!["c".into()], vec![String::new()], vec![1.0]).with_bounding_boxes(vec![ng_core::coordinate_space::TransformedBoundingBox::identity(ng_core::coordinate_space::BoundingBox {
        lower_bounds: vec![0.0],
        upper_bounds: vec![extent as f64],
    })]);
    let Some(bounds) = plot::dimension_bounds(&space, 0, plot::CANVAS_HEIGHT) else { return };
    let label_style = style! { font-family: monospace; font-size: {MONOSPACE_MEDIUM}; color: white; };
    let (lower, upper) = (number_to_string(bounds.lower), number_to_string(bounds.upper));
    let labels_width = [&lower, &upper].iter().map(|t| b.width(t, &label_style)).fold(0.0f32, f32::max);
    let grid = b.ui.add_box(dropdown, style! { display: flex; position: relative; overflow: hidden; });
    // The canvas's `0.5em` margins, in the layer panel's `small` font
    // (the bitmap lands on the whole pixel below the fractional top).
    let em = SMALL / 2.0;
    let labels = b.ui.add_box(grid, style! { position: relative; width: {labels_width}; height: {plot::CANVAS_HEIGHT as f32 + 2.0 * em}; flex-shrink: 0; });
    b.text_box(labels, &lower, style! { ..label_style.clone(); position: absolute; right: 0; top: 0; });
    b.text_box(labels, &upper, style! { ..label_style.clone(); position: absolute; right: 0; top: {plot::CANVAS_HEIGHT as f32 - 1.0}; });
    let c = b.ui.add_box(grid, style! { position: relative; width: {plot::CANVAS_WIDTH as f32}; height: {plot::CANVAS_HEIGHT as f32}; margin: {Edges::new(em.ceil(), 0.0, 2.0 * em - em.ceil(), 3.0)}; flex-shrink: 0; });
    let size = (plot::CANVAS_WIDTH as f32, plot::CANVAS_HEIGHT as f32);
    let (ticks, bars) = plot::bars(&bounds);
    b.rects(c, &ticks, "#fff", size);
    b.rects(c, &bars, "#ccc", size);
    if let Some(row) = plot::indicator(&bounds, current, plot::CANVAS_HEIGHT) {
        b.rects(c, &[[0.0, row, plot::CANVAS_WIDTH, 1.0]], "#f66", size);
    }
}

/// An `<input>` of `length` ch (Chrome's 1 px × 2 px padding, `padding-left:
/// 2px` of neuroglancer's): text in monospace, right-aligned for
/// coordinates; with `selection`, focused: the selected characters on
/// [`SELECTION`], or a caret.
#[allow(clippy::too_many_arguments)]
pub(crate) fn input(b: &mut Builder, parent: NodeId, text: &str, length: usize, color: &str, ch: f32, right: bool, selection: Option<(usize, usize)>) -> NodeId {
    // `text-align: right` aligns a line that fits; one wider than the box
    // (by its last 1/64 px, as 5 digits of 7.803 px in 39 px) starts at
    // the left, as Blink starts any overflowing line.
    let fits = b.width(text, &style! { font-family: monospace; font-size: {MONOSPACE_MEDIUM}; }) <= lu(length as f32 * ch);
    let right = right && fits;
    let node = b.ui.add_box(
        parent,
        style! {
            display: flex;
            position: relative;
            width: {lu(length as f32 * ch)};
            height: 15px;
            padding: 1px 2px;
            flex-shrink: 0;
            justify-content: {if right { JustifyContent::FlexEnd } else { JustifyContent::FlexStart }};
            align-items: center;
            font-family: monospace;
            font-size: {MONOSPACE_MEDIUM};
            color: {color};
        },
    );
    if let Some((start, end)) = selection {
        // Where the text starts in the content box (monospace: `ch` a
        // character).
        let width = lu(length as f32 * ch);
        let text_width = text.chars().count() as f32 * ch;
        let x0 = 2.0 + if right { width - text_width } else { 0.0 };
        let (a, z) = (x0 + start as f32 * ch, x0 + end as f32 * ch);
        let span = if start == end { (a, 1.0, threers::ui::css::color(color).unwrap_or(WHITE)) } else { (a, z - a, SELECTION) };
        b.ui.add_box(node, style! { position: absolute; left: {span.0}; top: 1px; width: {span.1}; height: 15px; background: {span.2}; });
        // The text over its selection: positioned too, and after it.
        let run = b.ui.add_box(node, style! { position: relative; font-family: monospace; font-size: {MONOSPACE_MEDIUM}; color: {color}; });
        b.ui.add_text(run, text);
        return node;
    }
    if fits {
        b.ui.add_text(node, text);
    } else {
        // The editor inside an `<input>` clips its text to the content box.
        let clip = b.ui.add_box(node, style! { display: flex; width: {lu(length as f32 * ch)}; flex-shrink: 0; overflow: hidden; font-family: monospace; font-size: {MONOSPACE_MEDIUM}; color: {color}; });
        b.ui.add_text(clip, text);
    }
    node
}

/// A checked or unchecked `<input type=checkbox>` as Chrome draws it: 13 px,
/// `margin: 3px 3px 3px 4px`, a blue (`AccentColor`) rounded box with a white
/// tick, or a dark grey rounded ring around white.
/// [`checkbox`] at `size` px (`width: 1rem`), no margin.
pub(crate) fn checkbox_sized(b: &mut Builder, parent: NodeId, checked: bool, key: Key, action: Action, size: f32) {
    let node = threers::ui::controls::checkbox(&mut b.ui, parent, checked, size, [0x4f as f32 / 255.0, 0x4f as f32 / 255.0, 0x4f as f32 / 255.0, 1.0], Edges::ZERO);
    b.key(node, key, Some(action));
}

pub(crate) fn checkbox(b: &mut Builder, parent: NodeId, checked: bool, key: Key, action: Action, margin: Edges) -> NodeId {
    // Over neuroglancer's dark panels Chrome borders an unchecked box #4f4f4f.
    let node = threers::ui::controls::checkbox(&mut b.ui, parent, checked, 13.0, [0x4f as f32 / 255.0, 0x4f as f32 / 255.0, 0x4f as f32 / 255.0, 1.0], margin);
    b.key(node, key, Some(action));
    node
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_label_text_contrasts_with_its_colour() {
        assert!(use_white_background([0.0, 0.0, 0.5]));
        assert!(!use_white_background([1.0, 1.0, 0.0]));
    }

    #[test]
    fn panels_get_neuroglancers_controls() {
        let (dims, related, sections) = panel_furniture(Layout::FourPanel, Panel::Slice(NamedAxes::Xz));
        assert!(!dims && !sections);
        assert_eq!(related.iter().map(|l| layout_symbol(*l)).collect::<String>(), "◻◫");
        let (dims, related, sections) = panel_furniture(Layout::FourPanel, Panel::Perspective);
        assert!(dims && sections);
        assert_eq!(related.iter().map(|l| layout_symbol(*l)).collect::<String>(), "◻◲");
    }

    #[test]
    fn styles_read_as_neuroglancers_css() {
        use threers::ui::AlignItems;
        let s = style! { display: flex; margin: 1px 1px 1px 5px; border: 1px solid #ccc; font-size: {SMALL}; };
        assert_eq!(s.margin, Edges::new(1.0, 1.0, 1.0, 5.0));
        assert_eq!(s.border_color, [0.8, 0.8, 0.8, 1.0]);
        assert_eq!(s.align_items, AlignItems::Stretch);
    }
}
