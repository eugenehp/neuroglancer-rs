//! Dialogs over the viewer (`overlay.ts`): the JSON state editor
//! (`ui/state_editor.ts`).

use serde_json::Value;
use threers::style;
use threers::ui::code::Folds;
use threers::ui::{controls, FieldKey, KeyModifiers, Length, NodeId, Rgba, TextArea};

use crate::view::Builder;
use crate::{Action, Effect, Key, NgUi};

/// An open dialog.
#[derive(Debug, Clone)]
pub(crate) enum Dialog {
    StateEditor(StateEditor),
    Screenshot(ScreenshotDialog),
    Palette(crate::palette::Palette),
    /// `ShaderCodeOverlay`: layer `i`'s shader in a larger editor.
    ShaderCode(usize),
}

/// `StateEditorDialog`: the state as JSON, folded but for its top level,
/// editable; Apply restores it once it parses.
#[derive(Debug, Clone)]
pub(crate) struct StateEditor {
    pub area: TextArea,
    pub folds: Folds,
    /// The text parsed since the last edit (`parsedValue`); Apply is enabled
    /// while it is.
    pub parsed: Option<Value>,
    pub scroll: f32,
    /// Whether the editor has the keyboard (clicked into); until then the
    /// dialog's box does, and only Escape does anything.
    pub focused: bool,
}

impl StateEditor {
    /// `updateView`: the state, 2-space indented, then `foldAll` and
    /// `unfold` at the caret (the first line).
    pub(crate) fn new(state: &Value) -> Self {
        let text = serde_json::to_string_pretty(state).unwrap_or_default();
        let mut folds = Folds::fold_all(&text);
        folds.unfold(0);
        StateEditor { area: TextArea::new(text), folds, parsed: None, scroll: 0.0, focused: false }
    }

    /// The text changed (`debouncedValueUpdater`).
    fn edited(&mut self) {
        self.parsed = serde_json::from_str(&self.area.text).ok();
        // Folds keep to the ranges still in the text.
        let now = threers::ui::code::brace_folds(&self.area.text);
        self.folds.folded.retain(|f| now.contains(f));
    }
}

/// CodeMirror's JavaScript mode on JSON (default theme): strings and
/// property names `#a11`, numbers `#164`, `true`/`false`/`null` `#219`, the
/// rest black.
pub(crate) fn json_tokens(line: &str) -> Vec<(String, Rgba)> {
    let color = |c: &str| threers::ui::css::color(c).unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let (string, number, atom, plain) = (color("#a11"), color("#164"), color("#219"), color("black"));
    let c: Vec<char> = line.chars().collect();
    let mut out: Vec<(String, Rgba)> = vec![];
    let mut push = |s: String, col: Rgba| match out.last_mut() {
        Some((p, pc)) if *pc == col => p.push_str(&s),
        _ => out.push((s, col)),
    };
    let mut i = 0;
    while i < c.len() {
        let start = i;
        let ch = c[i];
        if ch == '"' {
            i += 1;
            let mut escaped = false;
            while i < c.len() {
                let n = c[i];
                i += 1;
                if n == '"' && !escaped {
                    break;
                }
                escaped = !escaped && n == '\\';
            }
            push(c[start..i].iter().collect(), string);
        } else if ch.is_ascii_digit() || (ch == '-' && c.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            i += 1;
            while i < c.len() && (c[i].is_ascii_alphanumeric() || matches!(c[i], '.' | '+' | '-')) {
                i += 1;
            }
            push(c[start..i].iter().collect(), number);
        } else if ch.is_ascii_alphabetic() {
            while i < c.len() && c[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let word: String = c[start..i].iter().collect();
            let col = if matches!(word.as_str(), "true" | "false" | "null") { atom } else { plain };
            push(word, col);
        } else {
            i += 1;
            push(ch.to_string(), plain);
        }
    }
    out
}

/// Chrome's `<button>` in 13.33px Arial, greyed while `disabled`.
fn dialog_button(b: &mut Builder, parent: NodeId, label: &str, key: Key, action: Option<Action>, disabled: bool) -> NodeId {
    // `:disabled`: text, background and border at 30%.
    let color = if disabled { [16.0 / 255.0, 16.0 / 255.0, 16.0 / 255.0, 0.3] } else { [0.0, 0.0, 0.0, 1.0] };
    let font = style! { font-family: Arial; font-size: {40.0 / 3.0}; color: {color}; };
    let node = controls::button(&mut b.ui, parent, label, &font, b.text);
    b.ui.style_mut(node).margin.right = 0.0;
    if disabled {
        let s = b.ui.style_mut(node);
        // Chrome's button look paints the face opaque (#eee), only the
        // border faded.
        s.background = Some([238.0 / 255.0, 238.0 / 255.0, 238.0 / 255.0, 1.0]);
        s.border_color = [118.0 / 255.0, 118.0 / 255.0, 118.0 / 255.0, 0.3];
    }
    b.key(node, key, if disabled { None } else { action });
    node
}

impl NgUi {
    /// Builds the open dialog over everything.
    pub(crate) fn dialog(&mut self, b: &mut Builder, root: NodeId, viewer: &ng_viewer::Viewer) {
        if let Some(Dialog::Screenshot(_)) = &self.dialog {
            return self.screenshot_dialog(b, root, viewer);
        }
        if let Some(Dialog::Palette(_)) = &self.dialog {
            self.palette_results_height = 0.6 * b.viewport_height - 32.0;
            return self.command_palette(b, root, viewer);
        }
        if let Some(Dialog::ShaderCode(i)) = self.dialog {
            // `Overlay` with the code widget, 80vw by 80vh.
            let content = controls::overlay(&mut b.ui, root, Length::Auto);
            // The content has the focus (`tabindex`): Chrome's ring.
            b.ui.style_mut(content).box_sizing = threers::ui::BoxSizing::ContentBox;
            controls::focus_ring(&mut b.ui, content);
            b.key(content, Key::Dialog, None);
            let size = (0.8 * self.viewport.0, 0.8 * self.viewport.1);
            return self.shader_editor_sized(b, content, viewer, i, Some(size));
        }
        let caret_on = self.caret_on();
        let Some(Dialog::StateEditor(ed)) = &self.dialog else { return };
        let content = controls::overlay(&mut b.ui, root, Length::Percent(80.0));
        // The content has the focus (`tabindex`): Chrome's ring around it.
        b.ui.style_mut(content).box_sizing = threers::ui::BoxSizing::ContentBox;
        controls::focus_ring(&mut b.ui, content);
        b.key(content, Key::Dialog, None);
        let buttons = b.ui.add_box(content, style! { display: flex; position: relative; flex-shrink: 0; });
        dialog_button(b, buttons, "Apply changes", Key::DialogButton(0), Some(Action::ApplyState), ed.parsed.is_none());
        dialog_button(b, buttons, "Download", Key::DialogButton(1), Some(Action::DownloadState), false);
        b.ui.add_box(buttons, style! { flex: 1; });
        // `.close-button`: 15px from the right.
        let close = dialog_button(b, buttons, "Close", Key::DialogButton(2), Some(Action::CloseDialog), false);
        b.ui.style_mut(close).margin.right = -1.0;
        // CodeMirror's `font-family: monospace` at `medium`: 13px.
        let font = style! { font-family: monospace; font-size: 13px; color: black; };
        let lines: Vec<String> = ed.area.lines().iter().map(|s| s.to_string()).collect();
        let tokens = move |k: usize| json_tokens(lines.get(k).map_or("", String::as_str));
        let view = controls::CodeEditorView {
            area: &ed.area,
            folds: &ed.folds,
            tokens: &tokens,
            font: &font,
            line_height: 15.0,
            height: 300.0,
            focused: ed.focused,
            caret_on,
            scroll_top: ed.scroll,
            fold_gutter: true,
        };
        let nodes = controls::code_editor(&mut b.ui, content, &view, b.text);
        b.key(nodes.editor, Key::DialogEditor, None);
        b.key(nodes.body, Key::DialogScroll, None);
        for (k, (node, _)) in nodes.lines.iter().enumerate() {
            b.key(*node, Key::DialogLine(k), None);
        }
        for (node, line) in &nodes.markers {
            b.key(*node, Key::DialogFold(*line), Some(Action::ToggleFold(*line)));
        }
        self.dialog_lines = nodes.lines.iter().map(|(_, s)| s.line).collect();
    }

    /// A key while the dialog is open: the editor takes it; Escape closes.
    pub(crate) fn dialog_key(&mut self, key: &str, m: KeyModifiers) -> Option<Action> {
        // The shader overlay: the editor's keys; Escape closes it (what was
        // typed is set).
        if let Some(Dialog::ShaderCode(_)) = self.dialog {
            if key == "Escape" {
                self.shader_blur_pending = true;
                return Some(Action::CloseDialog);
            }
            return self.shader_key(key, m).or(Some(Action::Repaint));
        }
        let Some(Dialog::StateEditor(ed)) = &mut self.dialog else { return None };
        self.caret_since = self.now;
        if !ed.focused {
            return (key == "Escape").then_some(Action::CloseDialog);
        }
        match ed.area.key(key, m) {
            FieldKey::Cancel => Some(Action::CloseDialog),
            FieldKey::Edited => {
                ed.edited();
                Some(Action::Repaint)
            }
            FieldKey::Ignored => None,
            _ => Some(Action::Repaint),
        }
    }

    /// A press while the dialog is open (it takes every press): a line
    /// places the caret; buttons and fold markers act.
    pub(crate) fn dialog_press(&mut self, viewer: &ng_viewer::Viewer, keys: &[Key], x: f32, y: f32, shift: bool) -> Option<Action> {
        if let Some(Dialog::ShaderCode(_)) = self.dialog {
            return self.shader_press(viewer, keys, x, y, shift).or(Some(Action::Repaint));
        }
        if let Some(a) = keys.iter().find_map(|k| self.actions.get(k).cloned()) {
            self.field = None;
            return Some(a);
        }
        // A field in the dialog takes the focus; a press elsewhere commits it.
        if let Some(id) = keys.iter().find_map(|k| if let Key::Field(id) = k { Some(*id) } else { None }) {
            let text = match (&self.dialog, id) {
                (Some(Dialog::Screenshot(d)), crate::FieldId::ScreenshotName) => d.name.clone(),
                _ => String::new(),
            };
            if self.field.as_ref().is_none_or(|(f, _)| *f != id) {
                self.field = Some((id, threers::ui::TextField::new(text)));
            }
            self.caret_since = self.now;
            return Some(Action::Repaint);
        }
        if self.field.is_some() {
            let a = self.commit_field();
            self.field = None;
            return a;
        }
        let line_index = keys.iter().find_map(|k| if let Key::DialogLine(i) = k { Some(*i) } else { None });
        let node = line_index.and_then(|i| self.node_of(Key::DialogLine(i)));
        let Some(Dialog::StateEditor(ed)) = &mut self.dialog else { return None };
        if let (Some(i), Some(node)) = (line_index, node) {
            let line = self.dialog_lines.get(i).copied()?;
            let text = ed.area.lines()[line].to_string();
            let font = style! { font-family: monospace; font-size: 13px; };
            let col = controls::code_column(self.ui.rect(node), x, &text, &font, &mut threers::ui::FixedWidthText(self.mono_advance));
            let at = ed.area.index_of(line, col);
            ed.area.move_to(at, shift);
            ed.focused = true;
            self.caret_since = self.now;
        }
        Some(Action::Repaint)
    }
}

/// Carries out a dialog action.
pub(crate) fn apply(ui: &mut NgUi, viewer: &mut ng_viewer::Viewer, action: &Action) -> Effect {
    match action {
        Action::EditJsonState => {
            ui.dialog = Some(Dialog::StateEditor(StateEditor::new(&viewer.state_json())));
            Effect::None
        }
        Action::CloseDialog => {
            ui.dialog = None;
            ui.field = None;
            Effect::None
        }
        Action::OpenPalette => {
            ui.dialog = Some(Dialog::Palette(Default::default()));
            Effect::None
        }
        Action::PaletteBack => {
            match &mut ui.dialog {
                Some(Dialog::Palette(p)) if p.group.is_some() => {
                    p.group = None;
                    p.query = Default::default();
                    p.active = 0;
                    p.scroll = 0.0;
                }
                _ => ui.dialog = None,
            }
            Effect::None
        }
        Action::PaletteRun(k) => {
            let (entries, groups) = crate::palette::catalog(viewer);
            let Some(Dialog::Palette(p)) = &mut ui.dialog else { return Effect::None };
            match p.rows(&entries, &groups).get(*k).copied() {
                Some(crate::palette::Row::Group(g)) => {
                    p.group = Some(g);
                    p.query = Default::default();
                    p.active = 0;
                    p.scroll = 0.0;
                    Effect::None
                }
                Some(crate::palette::Row::Command(c)) => {
                    ui.dialog = None;
                    let action = crate::palette::command_action(&entries[c].id);
                    ui.apply(viewer, &action)
                }
                None => Effect::None,
            }
        }
        Action::PaletteCommand(id) => {
            if viewer.panel_action(id) {
                Effect::None
            } else {
                ui.apply(viewer, &Action::Shortcut(id.clone()))
            }
        }
        Action::OpenScreenshot => {
            ui.dialog = Some(Dialog::Screenshot(ScreenshotDialog { scale: 1.0, keep_fov: true, name: String::new(), scroll: 0.0 }));
            Effect::None
        }
        Action::ScreenshotScale(s) => {
            if let Some(Dialog::Screenshot(d)) = &mut ui.dialog {
                d.scale = *s;
            }
            Effect::None
        }
        Action::ScreenshotKeepFov => {
            if let Some(Dialog::Screenshot(d)) = &mut ui.dialog {
                d.keep_fov = !d.keep_fov;
            }
            Effect::None
        }
        Action::TakeScreenshot => {
            if let Some(Dialog::Screenshot(d)) = &ui.dialog {
                viewer.start_screenshot(d.scale, d.keep_fov, d.name.clone());
            }
            Effect::None
        }
        Action::CancelScreenshot => {
            viewer.cancel_screenshot();
            Effect::None
        }
        Action::ToggleFold(line) => {
            if let Some(Dialog::StateEditor(ed)) = &mut ui.dialog {
                ed.folds.toggle(&ed.area.text, *line);
            }
            Effect::None
        }
        Action::ApplyState => {
            let Some(Dialog::StateEditor(ed)) = &mut ui.dialog else { return Effect::None };
            // `applyChanges`: the state reset to the parsed value.
            match ed.parsed.take() {
                Some(state) => Effect::RestoreState(state),
                None => Effect::None,
            }
        }
        Action::DownloadState => {
            let Some(Dialog::StateEditor(ed)) = &ui.dialog else { return Effect::None };
            Effect::Download { name: "state.json".into(), bytes: ed.area.text.clone().into_bytes(), mime: "text/json".into() }
        }
        _ => Effect::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_colours_as_codemirror() {
        let t = json_tokens(r#"  "a": [-1.5e-9, true, "x\"y"],"#);
        let names: Vec<&str> = t.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(names, vec!["  ", "\"a\"", ": [", "-1.5e-9", ", ", "true", ", ", "\"x\\\"y\"", "],"]);
        let ed = StateEditor::new(&serde_json::json!({"a": {"b": 1}, "c": [1, 2], "d": 3}));
        let shown = threers::ui::code::shown_lines(&ed.area.text, &ed.folds);
        assert_eq!(shown.len(), 5);
        assert!(shown[1].fold.is_some() && shown[2].fold.is_some() && shown[3].fold.is_none());
    }
}

/// `ScreenshotDialog`'s settings and state.
#[derive(Debug, Clone)]
pub(crate) struct ScreenshotDialog {
    /// The scale factor (1, 2 or 4).
    pub scale: f64,
    /// `Keep slice FOV fixed with scale change`.
    pub keep_fov: bool,
    /// The name typed (empty for the default).
    pub name: String,
    /// The body's scroll.
    pub scroll: f32,
}

/// The screenshot dialog's palette (`.neuroglancer-screenshot-overlay`).
mod palette {
    pub const GRAY300: &str = "#d0d5dd";
    pub const GRAY600: &str = "#141415";
    pub const GRAY500: &str = "#f7f7f7";
    pub const GRAY50: &str = "#e6e6e6";
    pub const GRAY800: &str = "#344054";
    pub const GRAY700: [f32; 4] = [20.0 / 255.0, 20.0 / 255.0, 21.0 / 255.0, 0.6];
    pub const GRAY400: [f32; 4] = [20.0 / 255.0, 20.0 / 255.0, 21.0 / 255.0, 0.4];
    pub const GRAY200: [f32; 4] = [20.0 / 255.0, 20.0 / 255.0, 21.0 / 255.0, 0.8];
    pub const PRIMARY700: &str = "#0474ff";
}

/// `rem` in px.
const REM: f32 = 16.0;

/// The ikonate `help` icon (`?` in a circle) and `close`.
const HELP_PATHS: &[&str] = &["M12 21A9 9 0 1 1 12 3A9 9 0 1 1 12 21Z", "M12 14C12 12 13.576002 11.6652983 14.1186858 11.1239516C14.663127 10.5808518 15 9.83229128 15 9.00477884C15 7.34519421 13.6568542 6 12 6C11.1696235 6 10.4179473 6.33790488 9.87476487 6.88384034C9.32976664 7.43158493 9 8.18531559 9 9", "M12 17L12 17.01"];

impl NgUi {
    /// The screenshot dialog (`ScreenshotDialog`), framed.
    pub(crate) fn screenshot_dialog(&mut self, b: &mut Builder, root: NodeId, viewer: &ng_viewer::Viewer) {
        let Some(Dialog::Screenshot(d)) = &self.dialog else { return };
        let d = d.clone();
        let in_progress = viewer.screenshot_active();
        let content = controls::overlay_top(&mut b.ui, root, Length::Px(48.75 * REM), 1.25 * REM);
        b.ui.style_mut(content).font_family = "sans-serif".into();
        b.key(content, Key::Dialog, None);
        let title_font = style! { font-family: sans-serif; font-size: {0.938 * REM}; font-weight: 590; color: {palette::GRAY600}; };
        let label_font = style! { font-family: sans-serif; font-size: {0.813 * REM}; font-weight: 590; color: {palette::GRAY200}; };
        let value_font = style! { font-family: sans-serif; font-size: {0.813 * REM}; color: {palette::GRAY700}; };

        // The header: the title, the close icon.
        let header = b.ui.add_box(content, style! { display: flex; align-items: center; justify-content: space-between; height: 48px; padding: 0 16px; border-width: 0 0 1px 0; border-color: #ccc; margin-bottom: 1px; flex-shrink: 0; });
        b.text_box(header, "Screenshot", title_font.clone());
        let close = b.ui.add_box(header, style! { display: flex; width: 18px; height: 18px; });
        b.ui.add_path(close, style! { width: 18px; height: 18px; color: {[20.0 / 255.0, 20.0 / 255.0, 21.0 / 255.0, 0.6]}; }, icons_close(), [0.0, 0.0, 24.0, 24.0], Some(2.0), false);
        b.key(close, Key::DialogButton(10), (!in_progress).then_some(Action::CloseDialog));

        // The body, scrolling past `100vh − 200px`.
        let body = b.ui.add_box(content, style! { display: flex; flex-direction: column; max-height: {b.viewport_height - 200.0}; overflow: auto; scroll-top: {d.scroll}; flex-shrink: 1; min-height: 0; });
        b.key(body, Key::DialogScroll, None);

        // Settings: the name, the scale, keeping the FOV.
        let settings = b.ui.add_box(body, style! { display: flex; flex-direction: column; padding: 16px 16px 12px; flex-shrink: 0; });
        let sub = b.ui.add_box(settings, style! { display: flex; align-items: center; justify-content: space-between; height: 17px; flex-shrink: 0; });
        b.text_box(sub, "Settings", title_font.clone());
        help_icon(b, sub);
        b.text_box(settings, "Screenshot name", style! { ..label_font.clone(); color: {palette::GRAY800}; margin: 12px 0 6px; flex-shrink: 0; });
        let input = b.ui.add_box(
            settings,
            style! { display: flex; position: relative; align-items: center; box-sizing: border-box; height: 38px; padding: 8px 12px; border: 1px solid; border-color: {palette::GRAY300}; border-radius: 8px; background: white; flex-shrink: 0; },
        );
        let id = crate::FieldId::ScreenshotName;
        b.key(input, Key::Field(id), None);
        let editing = self.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t);
        let view = controls::TextView {
            value: &d.name,
            placeholder: "Enter optional screenshot name",
            placeholder_color: palette::GRAY700,
            editing,
            caret_on: self.caret_on(),
            font: &style! { font-family: Arial; font-size: {0.813 * REM}; color: {palette::GRAY700}; },
            x_offset: 12.0,
            selection: [0.7, 0.84, 1.0, 1.0],
        };
        controls::text_contents(&mut b.ui, input, view, b.text);

        let menu = b.ui.add_box(body, style! { display: flex; flex-direction: column; padding: 0 16px; flex-shrink: 0; });
        let scale_label = b.ui.add_box(menu, style! { display: flex; align-items: center; margin-bottom: 8px; flex-shrink: 0; });
        b.text_box(scale_label, "Screenshot scale factor", label_font.clone());
        let tip = help_icon(b, scale_label);
        b.ui.style_mut(tip).margin.left = 0.15 * REM;
        let radios = b.ui.add_box(menu, style! { display: flex; flex-shrink: 0; });
        for (k, s) in [1.0, 2.0, 4.0].into_iter().enumerate() {
            // A block: its line is 18px (16px text).
            let item = b.ui.add_box(radios, style! { display: flex; align-items: center; height: 18px; margin-right: {2.125 * REM}; });
            controls::radio(&mut b.ui, item, d.scale == s, threers::ui::Edges::new(0.0, 0.188 * REM, 0.0, 0.0));
            b.text_box(item, &format!("{}x", s as u32), style! { font-family: sans-serif; font-size: {0.75 * REM}; color: {palette::GRAY700}; });
            b.key(item, Key::DialogButton(20 + k as u8), (!in_progress).then_some(Action::ScreenshotScale(s)));
        }
        let keep = b.ui.add_box(menu, style! { display: flex; align-items: center; margin: 16px 0; flex-shrink: 0; });
        let check = controls::checkbox(&mut b.ui, keep, d.keep_fov, 13.0, controls::BORDER, threers::ui::Edges::ZERO);
        b.ui.style_mut(check).margin.right = 0.5 * REM;
        b.text_box(keep, "Keep slice FOV fixed with scale change", label_font.clone());
        b.key(keep, Key::DialogButton(30), (!in_progress).then_some(Action::ScreenshotKeepFov));

        // Preview: the size, the panels, the layers.
        let preview = b.ui.add_box(body, style! { display: flex; flex-direction: column; padding: 16px 16px 8px; border-width: 1px 0; border-color: #ccc; background: {palette::GRAY500}; flex-shrink: 0; });
        let top = b.ui.add_box(preview, style! { display: flex; align-items: flex-start; height: 18px; flex-shrink: 0; });
        b.text_box(top, "Preview", title_font.clone());
        b.ui.add_box(top, style! { flex: 1; });
        let copy = b.ui.add_box(top, style! { display: flex; padding: 0 2px; margin: 0 1px; });
        b.ui.add_path(copy, style! { width: 16px; height: 16px; color: {palette::GRAY400}; }, crate::icons::COPY.paths.iter().map(|p| p.to_string()).collect(), [0.0, 0.0, 24.0, 24.0], Some(2.0), false);
        let info = screenshot_info(viewer, d.scale, d.keep_fov);
        b.key(copy, Key::DialogButton(40), Some(Action::CopyText(info.json.clone())));
        let size = b.ui.add_box(preview, style! { display: flex; align-items: center; margin: 12px 0; flex-shrink: 0; });
        let l = b.text_box(size, "Screenshot size", label_font.clone());
        b.ui.style_mut(l).width = Length::Percent(33.33);
        let v = b.text_box(size, &format!("{} x {} px", info.width, info.height), style! { ..value_font.clone(); padding: 4px; });
        b.ui.style_mut(v).width = Length::Percent(33.33);
        table(b, preview, ["Panel type", "Pixel resolution", "Physical scale"], [true, false, true], &info.panels.iter().map(|r| vec![vec![(r.0.clone(), None)], vec![(r.1.clone(), None)], vec![(r.2.clone(), None)]]).collect::<Vec<_>>());
        table(b, preview, ["Layer name", "Data type", "Physical voxel resolution"], [true, false, false], &info.layers);

        // Progress.
        let stats = b.ui.add_box(body, style! { display: flex; flex-direction: column; padding: 16px; flex-shrink: 0; });
        let st = b.text_box(stats, "Screenshot progress", title_font.clone());
        b.ui.style_mut(st).padding.bottom = 8.0;
        b.ui.style_mut(st).flex_shrink = 0.0;
        let desc = b.text_box(
            stats,
            "The screenshot will take when all the chunks are loaded. If GPU memory is full, the screenshot will only capture the successfully loaded chunks. A screenshot scale larger than 1 may cause new chunks to be downloaded once the screenshot is in progress.",
            style! { ..value_font.clone(); line-height: 20px; white-space: normal; padding-bottom: 8px; flex-shrink: 0; },
        );
        let _ = desc;
        for (key, value) in &info.statistics {
            let row = b.ui.add_box(stats, style! { display: flex; padding: 8px 0; flex-shrink: 0; });
            let k = b.text_box(row, key, label_font.clone());
            b.ui.style_mut(k).width = Length::Percent(50.0);
            b.text_box(row, value, value_font.clone());
        }

        // The footer: progress, the buttons.
        let footer = b.ui.add_box(content, style! { display: flex; align-items: center; padding: 12px 16px; border-width: 1px 0 0 0; border-color: #ccc; flex-shrink: 0; });
        b.text_box(footer, if in_progress { "Screenshot in progress..." } else { "" }, style! { font-family: sans-serif; font-size: {0.813 * REM}; font-weight: 590; color: {palette::PRIMARY700}; });
        b.ui.add_box(footer, style! { flex: 1; });
        let (label, action) = if in_progress { ("Cancel screenshot", Action::CancelScreenshot) } else { ("Take screenshot", Action::TakeScreenshot) };
        let button = b.ui.add_box(
            footer,
            style! { display: flex; align-items: center; padding: 8px 12px; border: 1px solid; border-color: {palette::GRAY300}; border-radius: 8px; background: white; margin-left: 4px; },
        );
        b.text_box(button, label, style! { font-family: sans-serif; font-size: {0.813 * REM}; font-weight: 590; color: {palette::GRAY800}; });
        b.key(button, Key::DialogButton(50), Some(action));
    }
}

fn icons_close() -> Vec<String> {
    crate::icons::CLOSE.paths.iter().map(|p| p.to_string()).collect()
}

/// `.neuroglancer-screenshot-tooltip`: a 16px `?` circle.
fn help_icon(b: &mut Builder, parent: NodeId) -> NodeId {
    let n = b.ui.add_box(parent, style! { display: flex; width: 16px; height: 16px; margin: 0 1px; flex-shrink: 0; });
    b.ui.add_path(n, style! { width: 16px; height: 16px; color: {palette::GRAY400}; }, HELP_PATHS.iter().map(|p| p.to_string()).collect(), [0.0, 0.0, 24.0, 24.0], Some(2.0), false);
    n
}

/// A table column: a third of the dialog's 780px less paddings (744px).
const COLUMN: f32 = 744.0 * 0.3333;

/// A screenshot table: header cells (with a help icon where marked), rows
/// of cells made of runs (text, and a colour other than the cell's).
fn table(b: &mut Builder, parent: NodeId, headers: [&str; 3], help: [bool; 3], rows: &[TableRow]) {
    let t = b.ui.add_box(parent, style! { display: flex; flex-direction: column; padding: 4px 2px 2px; background: {palette::GRAY50}; margin-bottom: 12px; border-radius: 5.6px; flex-shrink: 0; });
    let head = b.ui.add_box(t, style! { display: flex; height: 24px; flex-shrink: 0; });
    for (k, h) in headers.iter().enumerate() {
        // `width: 33.33%` of the table's 744px.
        let cell = b.ui.add_box(head, style! { display: flex; align-items: flex-start; width: {COLUMN}; box-sizing: border-box; flex-shrink: 0; padding: 4px 6px; });
        b.text_box(cell, h, style! { font-family: sans-serif; font-size: {0.813 * REM}; font-weight: 590; color: {palette::GRAY200}; });
        if help[k] {
            let i = help_icon(b, cell);
            b.ui.style_mut(i).margin.left = 4.0;
        }
    }
    for row in rows {
        let r = b.ui.add_box(t, style! { display: flex; height: 32px; flex-shrink: 0; });
        for cell_runs in row {
            let cell = b.ui.add_box(r, style! { display: flex; align-items: center; width: {COLUMN}; box-sizing: border-box; flex-shrink: 0; padding: 6px; background: white; overflow: hidden; });
            for (text, color) in cell_runs {
                let c = color.map_or(palette::GRAY700, |c| threers::ui::css::color(c).unwrap_or(palette::GRAY700));
                b.text_box(cell, text, style! { font-family: sans-serif; font-size: {0.813 * REM}; color: {c}; white-space: pre; line-height: 20px; });
            }
        }
    }
}

/// A table cell: runs of text, with a colour other than the cell's.
pub(crate) type Cell = Vec<(String, Option<&'static str>)>;
/// A table row's cells.
pub(crate) type TableRow = Vec<Cell>;

/// What the screenshot dialog shows.
pub(crate) struct ScreenshotInfo {
    pub width: u32,
    pub height: u32,
    /// (type, pixel resolution, physical scale).
    pub panels: Vec<(String, String, String)>,
    /// Name, type, voxel resolution (runs: dimension names darker).
    pub layers: Vec<TableRow>,
    pub statistics: Vec<(String, String)>,
    pub json: String,
}

/// The preview's values at `scale` (`getViewerResolutionMetadata`).
pub(crate) fn screenshot_info(viewer: &ng_viewer::Viewer, scale: f64, keep_fov: bool) -> ScreenshotInfo {
    use ng_core::si_units::format_scale_with_unit_as_string;
    // `{precision: 2, elide1: false}`.
    let fmt2 = |v: f64, unit: &str| ng_core::si_units::format_scale_with_unit_as_string_precision(v, unit, false, 2);
    let (_, _, w, h) = viewer.panel_viewport_union();
    let s = ng_viewer::screenshot::clamp_scale(scale, w, h);
    let (width, height) = (((w as f64) * s).round() as u32, ((h as f64) * s).round() as u32);
    let mut panels = vec![];
    let nav = viewer.navigation();
    let info = nav.render_info();
    let unit = info.display_dimension_units[0].clone();
    let rects = viewer.panel_rects();
    // The slice zoom: per screenshot pixel when the FOV is kept.
    let slice_zoom = nav.zoom(ng_core::navigation::View::CrossSection) * info.canonical_voxel_physical_size / if keep_fov { s } else { 1.0 };
    let projection = nav.zoom(ng_core::navigation::View::Projection) * info.canonical_voxel_physical_size;
    for (kind, slice) in [("Slice view (2D)", true), ("Perspective projection (3D)", false)] {
        let Some((_, r)) = rects.iter().find(|(p, _)| matches!(p, ng_viewer::Panel::Slice(_)) == slice) else { continue };
        let px = format!("{} x {} px", ((r.width as f64) * s).round(), ((r.height as f64) * s).round());
        let phys = if slice { format!("{}/px", fmt2(slice_zoom, &unit)) } else { format!("{}/vh", fmt2(projection, &unit)) };
        panels.push((kind.to_owned(), px, phys));
    }
    let mut layers = vec![];
    let dims = viewer.dimensions();
    for l in viewer.layers().iter().filter(|l| l.spec.visible && !l.spec.archived) {
        let kind = match l.kind() {
            Some(ng_viewer::LayerKind::Image { .. }) => "Image slice (2D)",
            Some(ng_viewer::LayerKind::Segmentation(_)) => "Segmentation slice (2D)",
            None => continue,
        };
        let mut res: Vec<(String, Option<&'static str>)> = vec![];
        for (d, size) in l.voxel_size_global(dims.rank()).iter().enumerate() {
            if let Some(size) = size {
                res.push((dims.names[d].clone(), Some(palette::GRAY600)));
                res.push((format!(" {} ", format_scale_with_unit_as_string(*size * dims.scales[d], &dims.units[d], false)), None));
            }
        }
        layers.push(vec![vec![(l.name().to_owned(), None)], vec![(kind.to_owned(), None)], res]);
    }
    let (needed, available) = viewer.layers().iter().map(|l| l.chunk_progress()).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    let percent = if needed > 0 { 100.0 * available as f64 / needed as f64 } else { 0.0 };
    let statistics = vec![
        ("Number of loaded chunks".to_owned(), format!("{available} out of {needed} ({}%)", ng_core::js::to_fixed(percent, 2))),
        ("Visible chunk GPU memory usage".to_owned(), "0MB / 0MB (0.00% of total)".to_owned()),
        ("Number of downloading chunks".to_owned(), viewer.loading_chunks().to_string()),
    ];
    let json = serde_json::json!({ "width": width, "height": height, "panels": panels.iter().map(|p| serde_json::json!({"type": p.0, "pixelResolution": p.1, "physicalScale": p.2})).collect::<Vec<_>>() }).to_string();
    ScreenshotInfo { width, height, panels, layers, statistics, json }
}
