//! The command palette (`ui/command_palette.ts`, `ui/command_catalog.ts`,
//! `ui/default_commands.ts`): every command and tool, searchable, the
//! per-layer ones grouped.

use threers::style;
use threers::ui::{FieldKey, KeyModifiers, NodeId, TextField};

use crate::view::Builder;
use crate::{Action, Key, NgUi, PanelId};

/// `STATIC_COMMANDS` and `axisCommands()`, then the viewer's own: id and
/// label, in registration order.
const COMMANDS: &[(&str, &str)] = &[
    ("toggle-show-slices", "Toggle Slices in 3D"),
    ("toggle-scale-bar", "Toggle Scale Bar"),
    ("toggle-axis-lines", "Toggle Axis Lines"),
    ("toggle-orthographic-projection", "Toggle Orthographic Projection"),
    ("toggle-default-annotations", "Toggle Bounding Box"),
    ("toggle-show-statistics", "Toggle Statistics"),
    ("toggle-layout", "Toggle Layout"),
    ("toggle-layout-alternative", "Toggle Alternative Layout"),
    ("help", "Show Help"),
    ("open-command-palette", "Open Command Palette"),
    ("toggle-layer-list-panel", "Toggle Layer List"),
    ("toggle-selection-details-panel", "Toggle Selection Details"),
    ("toggle-layer-side-panel", "Toggle Layer Side Panel"),
    ("toggle-settings-panel", "Toggle Settings"),
    ("new-tool-palette", "New Tool Palette"),
    ("snap", "Snap to Axis"),
    ("zoom-in", "Zoom In"),
    ("zoom-out", "Zoom Out"),
    ("depth-range-decrease", "Decrease Depth Range"),
    ("depth-range-increase", "Increase Depth Range"),
    ("t-", "Previous Timestep"),
    ("t+", "Next Timestep"),
    ("add-layer", "Add Layer"),
    ("recolor", "Randomize Colors"),
    ("clear-segments", "Clear Selected Segments"),
    ("finish-annotation", "Finish Annotation"),
    ("undo-annotation-step", "Undo Annotation Step"),
    ("copy-url", "Copy URL"),
    ("copy-position", "Copy Position"),
    ("copy-selection-position", "Copy Selection Position"),
    ("deactivate-active-tool", "Deactivate Active Tool"),
    ("x-", "Move −X"),
    ("x+", "Move +X"),
    ("rotate-relative-x-", "Rotate −X"),
    ("rotate-relative-x+", "Rotate +X"),
    ("y-", "Move −Y"),
    ("y+", "Move +Y"),
    ("rotate-relative-y-", "Rotate −Y"),
    ("rotate-relative-y+", "Rotate +Y"),
    ("z-", "Move −Z"),
    ("z+", "Move +Z"),
    ("rotate-relative-z-", "Rotate −Z"),
    ("rotate-relative-z+", "Rotate +Z"),
    ("edit-json-state", "Edit JSON State"),
    ("screenshot", "Screenshot"),
];

/// A command (`CommandEntry`): its label, shortcut, what it does, its
/// group (by index), whether enabled.
#[derive(Debug, Clone)]
pub(crate) struct Entry {
    pub label: String,
    pub shortcut: String,
    pub id: String,
    pub group: Option<usize>,
    pub enabled: bool,
}

/// `CommandCatalog.rebuild`: the commands and groups.
pub(crate) fn catalog(viewer: &ng_viewer::Viewer) -> (Vec<Entry>, Vec<(String, String)>) {
    let shortcuts = ng_viewer::input::action_shortcuts(viewer.mac);
    let shortcut = |id: &str| shortcuts.iter().find(|(a, _)| a == id).map_or(String::new(), |(_, k)| k.clone());
    let range = |prefix: &str| {
        let s = shortcut(&format!("{prefix}-1"));
        if s.is_empty() {
            s
        } else {
            format!("{s}–9")
        }
    };
    let groups = vec![
        ("Toggle Layer Visibility".to_owned(), range("toggle-layer")),
        ("Select Layer".to_owned(), range("select-layer")),
        ("Toggle Layer Picking".to_owned(), range("toggle-pick-layer")),
    ];
    let mut out = vec![];
    let mut index = 0;
    for l in viewer.layers() {
        let enabled = !l.spec.archived;
        if enabled {
            index += 1;
        }
        for (g, prefix) in ["toggle-layer", "select-layer", "toggle-pick-layer"].into_iter().enumerate() {
            let id = format!("{prefix}-{index}");
            out.push(Entry { label: l.name().to_owned(), shortcut: shortcut(&id), id, group: Some(g), enabled });
        }
    }
    for (id, label) in COMMANDS {
        out.push(Entry { label: (*label).to_owned(), shortcut: shortcut(id), id: (*id).to_owned(), group: None, enabled: true });
    }
    // Bound actions nothing registered, labelled from their ids.
    for (id, key) in &shortcuts {
        let layer_numbered = ["toggle-layer-", "select-layer-", "toggle-pick-layer-"].iter().any(|p| id.strip_prefix(p).is_some_and(|n| n.bytes().all(|b| b.is_ascii_digit())));
        if COMMANDS.iter().any(|(c, _)| c == id) || layer_numbered || id.starts_with("tool-") {
            continue;
        }
        let label = id.split('-').map(|w| w[..1].to_uppercase() + &w[1..]).collect::<Vec<_>>().join(" ");
        out.push(Entry { label, shortcut: key.clone(), id: id.clone(), group: None, enabled: true });
    }
    // Tools: each dimension's, then each shown layer's controls.
    for name in &viewer.dimensions().names {
        out.push(Entry { label: format!("dim {name}"), shortcut: String::new(), id: format!("tool:dim:{name}"), group: None, enabled: true });
    }
    for l in viewer.layers().iter().filter(|l| l.spec.visible && !l.spec.archived) {
        let controls: &[&str] = match l.kind() {
            Some(ng_viewer::LayerKind::Image { .. }) => &["Resolution (slice)", "Blending (slice)", "Opacity (slice)", "Volume rendering (experimental)", "normalized"],
            Some(ng_viewer::LayerKind::Segmentation(_)) => &[
                "Color seed",
                "Fixed color",
                "Saturation",
                "Opacity (on)",
                "Opacity (off)",
                "Resolution (slice)",
                "Hide segment ID 0",
                "Base segment coloring",
                "Show all by default",
                "Highlight on hover",
            ],
            None => &[],
        };
        for c in controls {
            out.push(Entry { label: format!("{c} — {}", l.name()), shortcut: String::new(), id: format!("tool:{c}"), group: None, enabled: true });
        }
    }
    (out, groups)
}

/// `rankedMatches`: those starting with `query` (case-insensitively), then
/// those containing it.
pub(crate) fn ranked<'a, T>(items: impl IntoIterator<Item = &'a T>, label: impl Fn(&T) -> String, query: &str) -> Vec<&'a T>
where
    T: 'a,
{
    let items: Vec<&T> = items.into_iter().collect();
    if query.is_empty() {
        return items;
    }
    let q = query.to_lowercase();
    let (mut prefix, mut substring) = (vec![], vec![]);
    for item in items {
        let text = label(item).to_lowercase();
        if text.starts_with(&q) {
            prefix.push(item);
        } else if text.contains(&q) {
            substring.push(item);
        }
    }
    prefix.extend(substring);
    prefix
}

/// A row shown: a group (its index) or a command (its index).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Row {
    Group(usize),
    Command(usize),
}

/// The palette's state.
#[derive(Debug, Clone, Default)]
pub(crate) struct Palette {
    pub query: TextField,
    pub active: usize,
    /// The group opened (its index).
    pub group: Option<usize>,
    pub scroll: f32,
}

impl Palette {
    /// `computeDisplayRows`.
    pub(crate) fn rows(&self, entries: &[Entry], groups: &[(String, String)]) -> Vec<Row> {
        let q = self.query.text.as_str();
        let commands = |group: Option<usize>| -> Vec<Row> {
            let pool: Vec<usize> = (0..entries.len()).filter(|&k| entries[k].enabled && entries[k].group == group).collect();
            ranked(&pool, |&k| entries[k].label.clone(), q).into_iter().map(|&k| Row::Command(k)).collect()
        };
        let rows: Vec<Row> = if let Some(g) = self.group {
            commands(Some(g))
        } else if !q.is_empty() {
            let gs: Vec<usize> = (0..groups.len()).collect();
            let mut rows: Vec<Row> = ranked(&gs, |&g| groups[g].0.clone(), q).into_iter().map(|&g| Row::Group(g)).collect();
            rows.extend(commands(None));
            rows
        } else {
            let mut rows = vec![];
            let mut seen = vec![];
            for (k, e) in entries.iter().enumerate() {
                if !e.enabled {
                    continue;
                }
                match e.group {
                    None => rows.push(Row::Command(k)),
                    Some(g) if !seen.contains(&g) => {
                        seen.push(g);
                        rows.push(Row::Group(g));
                    }
                    _ => {}
                }
            }
            rows
        };
        rows.into_iter().filter(|r| !matches!(r, Row::Command(k) if entries[*k].id == "open-command-palette")).collect()
    }
}

/// Row height: 4px padding each way around 16px text.
const ROW: f32 = 26.0;

impl NgUi {
    /// `CommandPalette`: over a dimmed page, 500px wide, 20% down.
    pub(crate) fn command_palette(&mut self, b: &mut Builder, root: NodeId, viewer: &ng_viewer::Viewer) {
        let Some(crate::dialogs::Dialog::Palette(p)) = &self.dialog else { return };
        let p = p.clone();
        let (entries, groups) = catalog(viewer);
        let rows = p.rows(&entries, &groups);
        let shade = b.ui.add_box(root, style! { position: absolute; left: 0; top: 0; right: 0; bottom: 0; background: {[0.0, 0.0, 0.0, 0.8]}; });
        let height = b.viewport_height;
        let box_ = b.ui.add_box(
            shade,
            style! { position: absolute; left: 50%; margin-left: -250px; top: {0.2 * height}; width: 500px; max-height: {0.6 * height}; display: flex; flex-direction: column; background: #222; color: white; font-family: sans-serif; },
        );
        b.key(box_, Key::Dialog, None);
        // The group header, the query.
        let input_row = b.ui.add_box(box_, style! { display: flex; flex-direction: column; border-width: 0 0 1px 0; border-color: #444; flex-shrink: 0; });
        if let Some(g) = p.group {
            let header = b.text_box(input_row, &format!("← {}", groups[g].0), style! { font-family: sans-serif; font-size: 12px; color: #aaa; padding: 4px 8px; border-width: 0 0 1px 0; border-color: #333; });
            b.key(header, Key::DialogButton(60), Some(Action::PaletteBack));
        }
        let input = b.ui.add_box(input_row, style! { display: flex; position: relative; align-items: center; height: 31px; padding: 0 8px; flex-shrink: 0; });
        let font = style! { font-family: Arial; font-size: {40.0 / 3.0}; color: white; };
        let view = threers::ui::controls::TextView {
            value: "",
            placeholder: if p.group.is_some() { "" } else { "Type to find a command..." },
            placeholder_color: [117.0 / 255.0, 117.0 / 255.0, 117.0 / 255.0, 1.0],
            editing: Some(&p.query),
            caret_on: self.caret_on(),
            font: &font,
            x_offset: 8.0,
            selection: threers::ui::controls::SELECTION,
        };
        threers::ui::controls::text_contents(&mut b.ui, input, view, b.text);
        let results = b.ui.add_box(box_, style! { display: flex; flex-direction: column; overflow: auto; scroll-top: {p.scroll}; flex-shrink: 1; min-height: 0; });
        b.key(results, Key::DialogScroll, None);
        if rows.is_empty() {
            b.text_box(results, "No commands found.", style! { font-family: sans-serif; font-size: 16px; color: #888; padding: 8px; });
        }
        for (k, row) in rows.iter().enumerate() {
            let (label, shortcut) = match *row {
                Row::Group(g) => (groups[g].0.clone(), groups[g].1.clone()),
                Row::Command(c) => (entries[c].label.clone(), entries[c].shortcut.clone()),
            };
            let key = Key::PaletteRow(k);
            // The modifier symbols come from a fallback font whose line is a
            // pixel taller.
            let tall = shortcut.contains(['⌘', '⌥', '⇧', '⌃']);
            let r = b.ui.add_box(results, style! { display: flex; justify-content: space-between; align-items: center; height: {ROW + tall as u8 as f32}; padding: 0 12px; flex-shrink: 0; });
            if k == p.active || b.hover(key) {
                b.ui.style_mut(r).background = threers::ui::css::color("#444");
            }
            b.key(r, key, Some(Action::PaletteRun(k)));
            b.text_box(r, &label, style! { font-family: sans-serif; font-size: 16px; color: white; white-space: pre; });
            if !shortcut.is_empty() {
                b.text_box(r, &shortcut, style! { font-family: sans-serif; font-size: 16px; color: #aaa; margin-left: 12px; flex-shrink: 0; white-space: pre; });
            }
        }
    }

    /// A key in the palette: ↑/↓ move, Enter runs, Escape (or Backspace on
    /// nothing in a group) goes back; the rest edits the query.
    pub(crate) fn palette_key(&mut self, viewer: &ng_viewer::Viewer, key: &str, m: KeyModifiers) -> Option<Action> {
        let (entries, groups) = catalog(viewer);
        let height = self.palette_results_height;
        let Some(crate::dialogs::Dialog::Palette(p)) = &mut self.dialog else { return None };
        let n = p.rows(&entries, &groups).len();
        let wrap = |i: isize| if n == 0 { 0 } else { i.rem_euclid(n as isize) as usize };
        match key {
            "ArrowDown" | "ArrowUp" => {
                p.active = wrap(p.active as isize + if key == "ArrowDown" { 1 } else { -1 });
                // `scrollIntoView({block: "nearest"})`.
                let top = p.active as f32 * ROW;
                if top < p.scroll {
                    p.scroll = top;
                } else if top + ROW > p.scroll + height {
                    p.scroll = top + ROW - height;
                }
                return Some(Action::Repaint);
            }
            "Enter" => return (n > 0).then_some(Action::PaletteRun(p.active)),
            "Escape" => return Some(if p.group.is_some() { Action::PaletteBack } else { Action::CloseDialog }),
            "Backspace" if p.group.is_some() && p.query.text.is_empty() => return Some(Action::PaletteBack),
            _ => {}
        }
        match p.query.key(key, m, |_| true) {
            FieldKey::Edited => {
                p.active = 0;
                p.scroll = 0.0;
                Some(Action::Repaint)
            }
            FieldKey::Ignored => None,
            _ => Some(Action::Repaint),
        }
    }
}

/// What a command id does here.
pub(crate) fn command_action(id: &str) -> Action {
    match id {
        "toggle-layer-list-panel" => Action::TogglePanel(PanelId::LayerList),
        "toggle-selection-details-panel" => Action::TogglePanel(PanelId::Selection),
        "toggle-layer-side-panel" => Action::TogglePanel(PanelId::Layer),
        "toggle-settings-panel" => Action::TogglePanel(PanelId::Settings),
        "help" => Action::TogglePanel(PanelId::Help),
        "edit-json-state" => Action::EditJsonState,
        "screenshot" => Action::OpenScreenshot,
        "add-layer" => Action::AddLayer,
        "copy-url" => Action::CopyUrl,
        "copy-position" => Action::CopyPosition,
        id if id.starts_with("tool:") => Action::Unported("Tools"),
        "new-tool-palette" | "deactivate-active-tool" | "finish-annotation" | "undo-annotation-step" | "copy-selection-position" => Action::Unported("Tools"),
        id => Action::PaletteCommand(id.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_prefixes_first() {
        let items = ["Toggle Scale Bar", "Show Help", "Zoom In", "Toggle Axis Lines"];
        let r: Vec<&&str> = ranked(&items, |s| s.to_string(), "to");
        assert_eq!(r, [&"Toggle Scale Bar", &"Toggle Axis Lines"]);
        let r: Vec<&&str> = ranked(&items, |s| s.to_string(), "l");
        assert_eq!(r, [&"Toggle Scale Bar", &"Show Help", &"Toggle Axis Lines"]);
    }
}
