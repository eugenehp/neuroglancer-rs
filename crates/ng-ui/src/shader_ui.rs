//! An image layer's shader in its Rendering tab: the code, editable
//! (`ShaderCodeWidget`: CodeMirror, the shader set 500 ms after the last
//! change, errors marked in the lint gutter and the border red), and the
//! controls its `#uicontrol`s declare (`ShaderControls`): sliders, colours,
//! checkboxes and invlerps.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use ng_core::js::number_to_string;
use ng_core::DataType;
use ng_viewer::user_shader::{Control as ShaderControl, SliderType};
use ng_viewer::Viewer;
use serde_json::Value;
use threers::style;
use threers::ui::{controls, Edges, FieldKey, KeyModifiers, NodeId, TextArea};

use crate::rendering::{self, Control, InvlerpCtl};
use crate::tools::ToolRef;
use crate::view::Builder;
use crate::{Action, ColorSetting, FieldId, Key, NgUi};

const SMALL: f32 = 40.0 / 3.0;
/// `SHADER_UPDATE_DELAY`.
const UPDATE_DELAY: f64 = 500.0;
/// The editor's line height and height (`height: 8em`).
const LINE_HEIGHT: f32 = 15.0;

/// A control name as a key (names live as long as the program; each is
/// kept once).
pub(crate) fn intern(s: &str) -> &'static str {
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut names = NAMES.get_or_init(Default::default).lock().unwrap();
    if let Some(&n) = names.get(s) {
        return n;
    }
    let n: &'static str = Box::leak(s.to_owned().into_boxed_str());
    names.insert(n);
    n
}

/// The shader being edited: which layer's, its text, whether it has the
/// keyboard, its scroll, and when it last changed (not yet set).
#[derive(Debug, Clone)]
pub(crate) struct ShaderEdit {
    pub layer: usize,
    pub area: TextArea,
    pub focused: bool,
    pub scroll: f32,
    pub edited_at: Option<f64>,
}

/// `shaderControls` with control `name` set to `value` (`None`: its
/// default, left out).
pub(crate) fn set_control(controls: Option<&Value>, name: &str, value: Option<Value>) -> Option<Value> {
    let mut o = controls.and_then(Value::as_object).cloned().unwrap_or_default();
    match value {
        Some(v) => {
            o.insert(name.to_owned(), v);
        }
        None => {
            o.remove(name);
        }
    }
    (!o.is_empty()).then_some(Value::Object(o))
}

/// The layer type's default shader.
pub(crate) fn default_shader(viewer: &Viewer, i: usize) -> &'static str {
    match viewer.layers().get(i).map(|l| l.spec.layer_type.as_str()) {
        Some("annotation") => ng_viewer::state::DEFAULT_ANNOTATION_SHADER,
        Some("segmentation") => ng_viewer::state::DEFAULT_SKELETON_SHADER,
        _ => ng_viewer::state::DEFAULT_IMAGE_SHADER,
    }
}

/// The layer's shader member: a segmentation layer's is its skeletons'.
pub(crate) fn shader_key(viewer: &Viewer, i: usize) -> &'static str {
    match viewer.layers().get(i).map(|l| l.spec.layer_type.as_str()) {
        Some("segmentation") => "skeletonRendering.shader",
        _ => "shader",
    }
}

/// The layer's shader text (its default when not set).
pub(crate) fn shader_text(viewer: &Viewer, i: usize) -> String {
    let json = viewer.layers().get(i).map(|l| &l.spec.json);
    let text = match shader_key(viewer, i).split_once('.') {
        Some((outer, inner)) => json.and_then(|j| j.get(outer)?.get(inner)),
        None => json.and_then(|j| j.get("shader")),
    };
    text.and_then(Value::as_str).unwrap_or(default_shader(viewer, i)).to_owned()
}

/// What the state holds for an edited shader: nothing for the default.
fn shader_json(text: &str, default: &str) -> Option<Value> {
    (text != default).then(|| text.into())
}

/// The editor's height in a tab: the image tab's `8em`, else CodeMirror's
/// own 300 px (annotation layers).
fn tab_editor_height(viewer: &Viewer, i: usize) -> f32 {
    match viewer.layers().get(i).map(|l| l.spec.layer_type.as_str()) {
        Some("annotation") => 300.0,
        // `.neuroglancer-segmentation-rendering-tab`'s `6em`.
        Some("segmentation") => 6.0 * SMALL,
        _ => 8.0 * SMALL,
    }
}

impl NgUi {
    /// Sets the shader typed if it has rested for `UPDATE_DELAY` (or `now`
    /// regardless): called each frame.
    pub(crate) fn shader_settle(&mut self, viewer: &mut Viewer, force: bool) {
        let Some(edit) = &mut self.shader_edit else { return };
        let Some(at) = edit.edited_at else { return };
        if force || self.now - at >= UPDATE_DELAY {
            edit.edited_at = None;
            let (layer, text) = (edit.layer, edit.area.text.clone());
            let default = default_shader(viewer, layer);
            let key = shader_key(viewer, layer);
            viewer.set_layer_option(layer, key, shader_json(&text, default));
        }
    }

    /// The editor (`ShaderCodeWidget`): CodeMirror with the GLSL mode's
    /// colours, `8em` tall in a 1px border, red while the shader has
    /// errors (marked in the lint gutter, their messages as titles).
    pub(crate) fn shader_editor(&mut self, b: &mut Builder, parent: NodeId, viewer: &Viewer, i: usize) {
        self.shader_editor_sized(b, parent, viewer, i, None);
    }

    /// [`shader_editor`](Self::shader_editor) at `size` (the overlay's),
    /// else the tab's.
    pub(crate) fn shader_editor_sized(&mut self, b: &mut Builder, parent: NodeId, viewer: &Viewer, i: usize, size: Option<(f32, f32)>) {
        // The overlay's CodeMirror is outside the side panel's `small`
        // font: `medium` monospace, 13px.
        let font_size = if size.is_some() { 13.0 } else { SMALL };
        self.shader_advance = b.ch("monospace", font_size);
        let area = match &self.shader_edit {
            Some(e) if e.layer == i => e.area.clone(),
            _ => TextArea::new(shader_text(viewer, i)),
        };
        let (focused, scroll) = match &self.shader_edit {
            Some(e) if e.layer == i => (e.focused, e.scroll),
            _ => (false, 0.0),
        };
        // Errors: the directives', then the compiler's (once the shader
        // in the state is the one shown).
        let t = viewer.layers()[i].data_type().unwrap_or(DataType::Uint8);
        let parsed = ng_viewer::user_shader::parse_with_channels(&area.text, t, viewer.layers()[i].channel_extents().len());
        let mut errors: Vec<(usize, String)> = parsed.errors.iter().map(|e| (e.line, e.message.clone())).collect();
        if let Some(e) = viewer.layers()[i].shader_error() {
            if area.text == shader_text(viewer, i) {
                errors.push((e.line.unwrap_or(0), e.message.clone()));
            }
        }
        // `.valid-input`: shown in the overlay, and for annotation layers
        // (whose shader compiles as the tab shows it).
        let valid_shown = matches!(viewer.layers()[i].spec.layer_type.as_str(), "annotation" | "segmentation");
        let tokens = rendering::glsl_tokens(&area.text);
        let token_runs = |line: usize| -> Vec<(String, threers::ui::Rgba)> {
            tokens.get(line).map_or(vec![], |runs| runs.iter().map(|(s, k)| (s.clone(), threers::ui::css::color(rendering::token_color(*k)).unwrap_or([0.0, 0.0, 0.0, 1.0]))).collect())
        };
        let font = style! { font-family: monospace; font-size: {font_size}; color: black; };
        let marker = |b: &mut Builder, parent: NodeId, line: usize, left: f32, top: f32, messages: &[&str]| {
            // `.CodeMirror-lint-marker-error`: a red disc with a white cross
            // in the lint gutter, the line's messages its title.
            let m = b.ui.add_box(parent, style! { position: absolute; left: {left}; top: {top}; width: 12px; height: 12px; border-radius: 6px; background: #f44; });
            b.ui.add_path(m, style! { position: absolute; left: 3px; top: 3px; width: 6px; height: 6px; color: white; }, vec!["M0 0L6 6M6 0L0 6".into()], [0.0, 0.0, 6.0, 6.0], Some(1.5), false);
            let key = Key::ShaderLint(i, line);
            b.key(m, key, None);
            b.title(key, messages.join("\n"));
        };
        let at = |line: usize| errors.iter().filter(|(l, _)| *l == line).map(|(_, m)| m.as_str()).collect::<Vec<&str>>();
        if !focused {
            // As CodeMirror shows it at rest.
            let lines: Vec<Vec<(String, threers::ui::Rgba)>> = (0..tokens.len()).map(token_runs).collect();
            // In the overlay the content is 80vw by 80vh, inside the border.
            let editor = controls::code_view(&mut b.ui, parent, &lines, &font, LINE_HEIGHT, size.map_or(tab_editor_height(viewer, i) + 2.0, |s| s.1 + 2.0));
            if let Some((w, _)) = size {
                b.ui.style_mut(editor).width = threers::ui::Length::Px(w + 2.0);
            }
            b.key(editor, Key::ShaderEditor(i), None);
            // `.invalid-input` red; in the overlay (made once the shader has
            // compiled) `.valid-input` green. Drawn over the view's white,
            // which reaches under its border.
            let border = if !errors.is_empty() {
                Some([1.0, 0.0, 0.0, 1.0])
            } else if size.is_some() || valid_shown {
                Some([0.0, 128.0 / 255.0, 0.0, 1.0])
            } else {
                None
            };
            if let Some(color) = border {
                let edge = b.ui.add_box(editor, style! { position: absolute; left: -1px; top: -1px; right: -1px; bottom: -1px; border: 1px solid; });
                b.ui.style_mut(edge).border_color = color;
            }
            let mut marked: Vec<usize> = errors.iter().map(|(l, _)| *l).collect();
            marked.dedup();
            for line in marked {
                let messages = at(line);
                marker(b, editor, line, 1.0, 4.0 + line as f32 * LINE_HEIGHT + 1.0, &messages);
            }
            self.shader_lines.clear();
            return;
        }
        let height = size.map_or(tab_editor_height(viewer, i), |s| s.1);
        // The border (`.invalid-input`), white beneath (the editor's
        // background reaches under it).
        let border = if !errors.is_empty() {
            [1.0, 0.0, 0.0, 1.0]
        } else if size.is_some() || valid_shown {
            [0.0, 128.0 / 255.0, 0.0, 1.0]
        } else {
            [0.0; 4]
        };
        let frame = b.ui.add_box(parent, style! { display: flex; flex-direction: column; border: 1px solid; background: white; flex-shrink: 0; });
        b.ui.style_mut(frame).border_color = border;
        if let Some((w, _)) = size {
            b.ui.style_mut(frame).width = threers::ui::Length::Px(w + 2.0);
        }
        let folds = threers::ui::code::Folds::default();
        let view = controls::CodeEditorView {
            area: &area,
            folds: &folds,
            tokens: &token_runs,
            font: &font,
            line_height: LINE_HEIGHT,
            height,
            focused,
            caret_on: self.caret_on(),
            scroll_top: scroll,
            fold_gutter: false,
        };
        let nodes = controls::code_editor(&mut b.ui, frame, &view, b.text);
        b.key(nodes.editor, Key::ShaderEditor(i), None);
        for (node, shown) in &nodes.lines {
            b.key(*node, Key::ShaderLine(i, shown.line), None);
            let messages = at(shown.line);
            if !messages.is_empty() {
                marker(b, *node, shown.line, -15.0, 1.0, &messages);
            }
        }
        self.shader_lines = nodes.lines.iter().map(|(node, s)| (*node, s.line)).collect();
    }

    /// A press on the editor: the caret to the line and column under it
    /// (extending the selection with Shift), and the keyboard.
    pub(crate) fn shader_press(&mut self, viewer: &Viewer, keys: &[Key], x: f32, y: f32, shift: bool) -> Option<Action> {
        let (layer, line, rect) = match keys.iter().find_map(|k| if let Key::ShaderLine(l, line) = k { Some((*l, *line)) } else { None }) {
            Some((layer, line)) => (layer, line, self.ui.rect(self.node_of(Key::ShaderLine(layer, line))?)),
            None => {
                // The editor at rest: its lines from 1 + 4px down, the text
                // past the 16px gutter and 4px.
                let layer = keys.iter().find_map(|k| if let Key::ShaderEditor(l) = k { Some(*l) } else { None })?;
                let r = self.ui.rect(self.node_of(Key::ShaderEditor(layer))?);
                let n = shader_text(viewer, layer).split('\n').count();
                let line = (((y - r.y - 5.0) / LINE_HEIGHT).floor().max(0.0) as usize).min(n.saturating_sub(1));
                (layer, line, threers::ui::Rect { x: r.x + 18.0, y: r.y + 5.0 + line as f32 * LINE_HEIGHT, width: r.width - 18.0, height: LINE_HEIGHT })
            }
        };
        if self.shader_edit.as_ref().is_none_or(|e| e.layer != layer) {
            self.shader_edit = Some(ShaderEdit { layer, area: TextArea::new(shader_text(viewer, layer)), focused: false, scroll: 0.0, edited_at: None });
        }
        let advance = self.shader_advance;
        let edit = self.shader_edit.as_mut()?;
        let text = edit.area.lines().get(line).map(|s| s.to_string()).unwrap_or_default();
        let font = style! { font-family: monospace; font-size: {SMALL}; };
        let col = controls::code_column(rect, x, &text, &font, &mut threers::ui::FixedWidthText(advance));
        let at = edit.area.index_of(line, col);
        edit.area.move_to(at, shift);
        edit.focused = true;
        self.caret_since = self.now;
        Some(Action::Repaint)
    }

    /// Whether the shader editor has the keyboard.
    pub(crate) fn shader_focused(&self) -> bool {
        self.shader_edit.as_ref().is_some_and(|e| e.focused)
    }

    /// A key in the editor: edits wait to be set; Escape leaves (setting
    /// what was typed).
    pub(crate) fn shader_key(&mut self, key: &str, m: KeyModifiers) -> Option<Action> {
        let now = self.now;
        let edit = self.shader_edit.as_mut().filter(|e| e.focused)?;
        self.caret_since = now;
        match edit.area.key(key, m) {
            FieldKey::Edited => {
                edit.edited_at = Some(now);
                Some(Action::Repaint)
            }
            FieldKey::Cancel => {
                // Set as the next frame builds.
                self.shader_blur_pending = true;
                Some(Action::Repaint)
            }
            FieldKey::Ignored => None,
            _ => Some(Action::Repaint),
        }
    }

    /// The editor loses the keyboard: what was typed is set now.
    pub(crate) fn shader_blur(&mut self, viewer: &mut Viewer) {
        self.shader_settle(viewer, true);
        self.shader_edit = None;
    }

    /// The controls of a user shader (`ShaderControls`): each as a layer
    /// control, its tool binding first.
    pub(crate) fn shader_controls(&mut self, b: &mut Builder, parent: NodeId, viewer: &mut Viewer, i: usize) {
        // Annotation shaders' controls do not depend on data.
        let annotation = viewer.layers()[i].spec.layer_type == "annotation";
        let Some(t) = viewer.layers()[i].data_type().or(annotation.then_some(ng_core::DataType::Float32)) else { return };
        let json = viewer.layers()[i].spec.json.clone();
        let text = shader_text(viewer, i);
        let parsed = ng_viewer::user_shader::parse_with_channels(&text, t, viewer.layers()[i].channel_extents().len());
        let values = ng_viewer::user_shader::control_values(&parsed, t, json.get("shaderControls"));
        let controls_json = json.get("shaderControls").cloned();
        for (c, ((name, control), value)) in parsed.controls.iter().zip(&values).enumerate() {
            let name = intern(name);
            // Each control's keys apart: its place above the layer's bits.
            let slot = i | ((c + 1) << 28);
            let tool = ToolRef::layer(i, serde_json::json!({ "type": "shaderControl", "control": name }));
            use ng_viewer::user_shader::ControlValue as V;
            // `channelInvlerpLayerControl`: the channel as a position under
            // the label.
            let row = match value {
                V::Invlerp { channel, .. } if !channel.is_empty() => {
                    let (row, column) = rendering::control_row_with_label_column(b, parent, name, tool);
                    self.channel_position(b, column, viewer, i, name, channel);
                    row
                }
                _ => rendering::control_row_pub(b, parent, name, tool),
            };
            match (control, value) {
                (ShaderControl::Slider { value_type, min, max, step, default }, V::Number(v)) => {
                    self.shader_slider(b, row, i, slot, name, *v, (*min, *max, *step, *default), *value_type);
                }
                (ShaderControl::Color { default, .. }, V::Color(color)) => {
                    b.ui.add_box(row, style! { flex: 1; });
                    let well = controls::color_well(&mut b.ui, row, *color, 50.0);
                    b.ui.style_mut(well).margin.left = 5.0;
                    let setting = ColorSetting::ShaderColor(i, name);
                    b.key(well, Key::Color(setting), Some(Action::PickColor(setting)));
                    let _ = default;
                }
                (ShaderControl::Checkbox { default }, V::Bool(on)) => {
                    b.ui.add_box(row, style! { flex: 1; });
                    let node = controls::checkbox(&mut b.ui, row, *on, 13.0, controls::BORDER, Edges::new(3.0, 3.0, 3.0, 5.0));
                    // `TrackableBoolean`: the default left out.
                    let next = (!*on != *default).then_some(Value::Bool(!*on));
                    let action = Action::SetLayerOption(i, "shaderControls".into(), set_control(controls_json.as_ref(), name, next));
                    b.key(node, Key::Control(slot, "checkbox"), Some(action.clone()));
                    b.key(row, Key::Control(slot, "checkboxRow"), Some(action));
                }
                (ShaderControl::Invlerp { range: default_range, window: default_window, .. }, V::Invlerp { range, window, channel }) => {
                    let ctl = InvlerpCtl { name, default_range: default_range.as_f64(), default_window: *default_window };
                    viewer.want_window_histogram(i, *window);
                    let histogram = viewer.layers()[i].window_histogram(*window).copied();
                    // Its legend through the shader, at the width it had.
                    let width = self.legend_widths.get(&(slot, name)).copied().unwrap_or(200);
                    match viewer.shader_legend(i, rendering::legend_values(t, *window, width)) {
                        Some(colors) => {
                            self.legend_colors.insert((slot, name), colors);
                        }
                        None => {
                            self.legend_colors.remove(&(slot, name));
                        }
                    }
                    let w = ng_viewer::Interval::Number(*window);
                    rendering::invlerp_widget(self, b, row, slot, &json, t, *range, w, histogram.as_ref(), ctl, channel.is_empty());
                }
                // `selectLayerControl`: an `EnumSelectWidget` of the options'
                // labels.
                (ShaderControl::Select { options, default, .. }, V::Select(v)) => {
                    let labels: Vec<&'static str> = options.iter().map(|o| intern(&o.display())).collect();
                    let current = options.iter().position(|o| o.value == *v).unwrap_or(0);
                    let default = options.iter().position(|o| o.value == *default).unwrap_or(0);
                    let key = Key::Control(slot, "select");
                    rendering::select(b, row, labels[current], &labels, key, Action::Repaint);
                    let values = options.iter().map(|o| o.value.json()).collect();
                    b.controls.insert(key, rendering::Control::ShaderSelect { layer: i, name, labels, values, current, default });
                }
                _ => {}
            }
        }
    }

    /// An invlerp's channel (`PositionWidget` over the channel space, no
    /// copy button): each channel dimension's name and coordinate.
    fn channel_position(&mut self, b: &mut Builder, parent: NodeId, viewer: &Viewer, i: usize, name: &'static str, channel: &[u32]) {
        let dims = viewer.layers()[i].channel_dimensions();
        let ch = b.ch("monospace", 13.0);
        let gap = crate::view::lu(b.ch("serif", 16.0));
        let line = b.ui.add_box(parent, style! { display: flex; align-items: flex-start; height: 20px; flex-shrink: 0; });
        for (k, ((dim, extent), c)) in dims.iter().zip(channel).enumerate() {
            let id = FieldId::ShaderChannel(i, name, k);
            // `:focus-within`: a grey border on black.
            let focused = self.field.as_ref().is_some_and(|(f, _)| *f == id);
            let cell = b.ui.add_box(line, style! { display: flex; border: 1px solid; border-color: {if focused { "#aaa" } else { "transparent" }}; background: {focused.then_some([0.0, 0.0, 0.0, 1.0])}; margin-left: {if k > 0 { gap } else { 0.0 }}; align-items: flex-start; flex-shrink: 0; });
            self.channel_cells.push((id, cell, *extent));
            crate::view::input(b, cell, dim, dim.chars().count().max(1), "#ff6", ch, false, None);
            // The scale container, its input empty for a unitless
            // dimension (still a character wide).
            let container = b.ui.add_box(cell, style! { display: flex; margin: 0 2px 0 4px; flex-shrink: 0; });
            crate::view::input(b, container, "", 1, "#bbb", ch, false, None);
            let editing = self.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t);
            let text = editing.map_or(c.to_string(), |t| t.text.clone());
            let width = extent.to_string().len().max(text.chars().count()).max(1);
            let node = crate::view::input(b, cell, &text, width, "white", ch, true, None);
            b.key(node, Key::Field(id), None);
        }
    }

    /// A slider control (`rangeLayerControl`): Chrome's range input over
    /// `[min, max]` by `step`, then the number (`maxNumberWidth + 2` ch).
    #[allow(clippy::too_many_arguments)]
    fn shader_slider(&self, b: &mut Builder, row: NodeId, i: usize, slot: usize, name: &'static str, value: f64, (min, max, step, default): (f64, f64, f64, f64), value_type: SliderType) {
        let slider = b.ui.add_box(row, style! { display: flex; align-items: center; justify-content: flex-end; flex: 1; max-width: 500px; margin-left: 5px; });
        let key = Key::Control(slot, "slider");
        let fraction = if max > min { ((value - min) / (max - min)) as f32 } else { 0.0 };
        let hovered = b.hover(key);
        let range = controls::range(&mut b.ui, slider, fraction, hovered);
        b.ui.style_mut(range).margin = Edges::all(2.0);
        b.snapped.push(range);
        b.key(range, key, None);
        b.controls.insert(key, Control::ShaderSlider { layer: i, name, value, min, max, step, default, int: value_type != SliderType::Float });
        let digits = [number_to_string(min), number_to_string(max), number_to_string(max.min(min + step)), number_to_string(min.max(max - step))].iter().map(|s| s.len()).max().unwrap_or(1);
        let ch = b.ch("Arial", SMALL);
        let width = crate::view::lu((digits + 2) as f32 * ch) + 4.0;
        let number = b.ui.add_box(slider, style! { display: flex; position: relative; align-items: center; width: {width}; height: 20px; padding: 1px 2px; box-sizing: border-box; background: {[1.0, 1.0, 1.0, 0.3]}; flex-shrink: 0; });
        let id = FieldId::ShaderSlider(i, name);
        b.key(number, Key::Field(id), None);
        let editing = self.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t);
        let font = style! { font-family: Arial; font-size: {SMALL}; color: white; };
        self.text_contents(b, number, editing, &number_to_string(value), &font, 0.0);
    }
}

/// A slider's value at fraction `f` of its run: `min + k · step`, nearest,
/// within `[min, max]` (an integer slider's an integer).
pub(crate) fn slider_value(f: f64, min: f64, max: f64, step: f64, int: bool) -> f64 {
    let raw = min + f.clamp(0.0, 1.0) * (max - min);
    let mut v = if step > 0.0 { min + ((raw - min) / step).round() * step } else { raw };
    // As browsers step: in decimal, so `0.1 · 3` is `0.3`.
    let decimals = number_to_string(step).split('.').nth(1).map_or(0, str::len).max(number_to_string(min).split('.').nth(1).map_or(0, str::len));
    let scale = 10f64.powi(decimals as i32);
    v = (v * scale).round() / scale;
    if int {
        v = v.round();
    }
    v.clamp(min, max)
}

/// A slider's JSON for value `v`: none at its default.
pub(crate) fn slider_json(v: f64, default: f64) -> Option<Value> {
    (v != default).then(|| ng_core::js::number(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliders_step_as_browsers_do() {
        assert_eq!(slider_value(0.3, 0.0, 1.0, 0.1, false), 0.3);
        assert_eq!(slider_value(0.5, -1.0, 1.0, 0.02, false), 0.0);
        assert_eq!(slider_value(0.26, 0.0, 10.0, 1.0, true), 3.0);
        assert_eq!(slider_value(2.0, 0.0, 10.0, 1.0, true), 10.0);
        assert_eq!(set_control(None, "g", Some(1.5.into())), Some(serde_json::json!({ "g": 1.5 })));
        assert_eq!(set_control(Some(&serde_json::json!({ "g": 1.5 })), "g", None), None);
        assert!(std::ptr::eq(intern("brightness"), intern("brightness")));
    }
}
