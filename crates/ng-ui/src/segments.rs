//! A segmentation layer's Seg. tab (`ui/segment_list.ts`
//! `SegmentDisplayTab`): the Select tool, the query, and the lists — the
//! query's matches (IDs typed) and the starred segments — each under a
//! header (copy all, star all, show all, copy visible, status).
//!
//! The starred segments are the layer's `segments`; a `!` before an ID
//! keeps it starred but hidden.

use ng_viewer::Viewer;
use threers::style;
use threers::ui::{Edges, NodeId, Rgba};

use crate::panels::scroll_style;
use crate::view::Builder;
use crate::{icons, Action, FieldId, Key, NgUi};

/// `10pt`.
const SMALL: f32 = 40.0 / 3.0;

/// A star or eye's state over several segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tri {
    Off,
    Some,
    On,
}

impl Tri {
    fn of(n: usize, total: usize) -> Tri {
        if total > 0 && n == total {
            Tri::On
        } else if n > 0 {
            Tri::Some
        } else {
            Tri::Off
        }
    }
}

/// A layer's starred segments in order, and whether each is visible.
pub(crate) fn starred(layer: &ng_viewer::ManagedLayer) -> Vec<(u64, bool)> {
    ng_viewer::state::segmentation_settings(&layer.spec.json).map(|(_, s)| s).unwrap_or_default()
}

/// The layer's `segments` as neuroglancer writes them.
pub(crate) fn segments_json(segments: &[(u64, bool)]) -> serde_json::Value {
    serde_json::Value::Array(segments.iter().map(|&(id, visible)| serde_json::Value::String(if visible { id.to_string() } else { format!("!{id}") })).collect())
}

/// `parseSegmentQuery` for a list of IDs (`/^[,\s]*[0-9]+(?:[,\s]+[0-9]+)*[,\s]*$/`):
/// the distinct IDs, sorted; `None` for any other query.
pub(crate) fn id_query(query: &str) -> Option<Vec<u64>> {
    let parts: Vec<&str> = query.split(|c: char| c == ',' || c.is_whitespace()).filter(|p| !p.is_empty()).collect();
    if parts.is_empty() || !parts.iter().all(|p| p.bytes().all(|b| b.is_ascii_digit())) {
        return None;
    }
    let mut ids: Vec<u64> = parts.iter().filter_map(|p| ng_viewer::state::parse_uint64(&serde_json::Value::String((*p).into())).ok()).collect();
    ids.sort_unstable();
    ids.dedup();
    Some(ids)
}

/// What a change to a set of segments does to the starred list.
#[derive(Debug, Clone, PartialEq)]
pub enum SegmentChange {
    /// Star (or unstar, which hides too) these.
    Star(Vec<u64>, bool),
    /// Show (starring them) or hide these.
    Show(Vec<u64>, bool),
}

/// `selectedSegments` / `visibleSegments` after `change`.
pub(crate) fn apply_change(mut segments: Vec<(u64, bool)>, change: &SegmentChange) -> Vec<(u64, bool)> {
    match change {
        SegmentChange::Star(ids, true) => {
            for &id in ids {
                if !segments.iter().any(|s| s.0 == id) {
                    segments.push((id, false));
                }
            }
        }
        SegmentChange::Star(ids, false) => segments.retain(|s| !ids.contains(&s.0)),
        SegmentChange::Show(ids, show) => {
            for &id in ids {
                match segments.iter_mut().find(|s| s.0 == id) {
                    Some(s) => s.1 = *show,
                    None if *show => segments.push((id, true)),
                    None => {}
                }
            }
        }
    }
    segments
}

impl NgUi {
    /// `SegmentDisplayTab` of segmentation layer `i`.
    pub(crate) fn segments_tab(&mut self, b: &mut Builder, content: NodeId, viewer: &mut Viewer, i: usize) {
        let layer = &viewer.layers()[i];
        let Some(ng_viewer::LayerKind::Segmentation(display)) = layer.kind() else { return };
        let seed = display.color_seed;
        let segments = starred(layer);
        let query = layer.spec.json.get("segmentQuery").and_then(serde_json::Value::as_str).unwrap_or("").to_owned();

        // The Select tool.
        let toolbox = b.ui.add_box(content, style! { display: flex; flex-shrink: 0; });
        let binding = b.tool_binding(toolbox, crate::tools::ToolRef::layer(i, "selectSegments"), Some("Select/Deselect segments"));
        b.ui.style_mut(binding).align_self = Some(threers::ui::AlignItems::Center);
        let select = b.text_box(toolbox, "Select", style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
        b.title(Key::Control(i, "selectTool"), "Select/Deselect segments\n(click → bind key, dbclick → unbind)");
        b.key(select, Key::Control(i, "selectTool"), None);

        // The query.
        let id = FieldId::SegmentQuery(i);
        let editing = self.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t);
        let input = b.ui.add_box(
            content,
            style! { display: flex; position: relative; align-items: center; box-sizing: border-box; height: 23px; padding: 2px; border: 2px solid #333; background: #151515; flex-shrink: 0; overflow: hidden; },
        );
        b.key(input, Key::Field(id), None);
        let font = style! { font-family: monospace; font-size: 13px; color: white; };
        let view = threers::ui::controls::TextView {
            value: &query,
            placeholder: "Enter ID, name prefix or /regexp",
            placeholder_color: [0xaa as f32 / 255.0, 0xaa as f32 / 255.0, 0xaa as f32 / 255.0, 1.0],
            editing,
            caret_on: self.caret_on(),
            font: &font,
            x_offset: 0.0,
            selection: threers::ui::controls::SELECTION,
        };
        threers::ui::controls::text_contents(&mut b.ui, input, view, b.text);

        // Each list shown takes an equal share of what is left; the starred
        // list scrolls with the panel's offset, the query's is clipped.
        let scroll = content;
        let matches = id_query(&query);
        // One width for every ID chip in the tab (the longest ID's).
        let digits = matches.iter().flatten().chain(segments.iter().map(|s| &s.0)).map(|id| id.to_string().len()).max().unwrap_or(1);
        let show_query = !query.is_empty();
        if show_query {
            let ids = matches.clone().unwrap_or_default();
            let visible = ids.iter().filter(|id| segments.iter().any(|s| s.0 == **id && s.1)).count();
            let starred_n = ids.iter().filter(|id| segments.iter().any(|s| s.0 == **id)).count();
            // `statusText`: the matches, with how many are visible and starred.
            let mut status = format!("{} ids", ids.len());
            if starred_n > 0 {
                status = if starred_n == visible {
                    format!("{starred_n} vis/{status}")
                } else if visible > 0 {
                    format!("{visible} vis/{starred_n} star/{status}")
                } else {
                    format!("{starred_n} star/{status}")
                };
            }
            if matches.is_none() {
                status = String::new();
            }
            self.segment_group(b, scroll, i, seed, &segments, &ids, &status, 3000, Tri::of(starred_n, ids.len()), Tri::of(visible, ids.len()), None, digits);
        }
        if !segments.is_empty() || !show_query {
            let ids: Vec<u64> = segments.iter().map(|s| s.0).collect();
            let visible = segments.iter().filter(|s| s.1).count();
            let status = format!("{visible}/{} visible", segments.len());
            let offset = self.scroll.get(&self.layer_panel).copied().unwrap_or(0.0);
            self.segment_group(b, scroll, i, seed, &segments, &ids, &status, 3100, Tri::of(ids.len(), ids.len()), Tri::of(visible, ids.len()), Some(offset), digits);
        }
    }

    /// A list's header and entries.
    #[allow(clippy::too_many_arguments)]
    fn segment_group(&self, b: &mut Builder, parent: NodeId, i: usize, seed: u32, segments: &[(u64, bool)], ids: &[u64], status: &str, icon_base: u32, star: Tri, eye: Tri, scroll: Option<f32>, digits: usize) {
        let header = b.ui.add_box(parent, style! { display: flex; align-items: center; height: 18px; background: #333; flex-shrink: 0; });
        let shown = !ids.is_empty();
        let sorted = |only_visible: bool| {
            let mut v: Vec<u64> = ids.iter().copied().filter(|id| !only_visible || segments.iter().any(|s| s.0 == *id && s.1)).collect();
            v.sort_unstable();
            v.iter().map(u64::to_string).collect::<Vec<_>>().join(", ")
        };
        let hide = |b: &mut Builder, n: NodeId, visible: bool| {
            if !visible {
                b.ui.style_mut(n).opacity = 0.0;
                b.keys.remove(&n);
            }
        };
        let n = b.icon(header, icon_base, Some(icons::COPY), "", Edges::symmetric(0.0, 1.0), Action::CopyText(sorted(false)));
        hide(b, n, shown);
        let n = star_icon(b, header, icon_base + 1, star, Action::SegmentChange(i, SegmentChange::Star(ids.to_vec(), star != Tri::On)));
        hide(b, n, shown);
        let n = eye_icon(b, header, icon_base + 2, eye, Action::SegmentChange(i, SegmentChange::Show(ids.to_vec(), eye != Tri::On)));
        hide(b, n, shown);
        let any_visible = eye != Tri::Off;
        let n = b.icon(header, icon_base + 3, Some(icons::COPY), "", Edges::symmetric(0.0, 1.0), Action::CopyText(sorted(true)));
        hide(b, n, any_visible);
        b.text_box(header, status, style! { font-family: sans-serif; font-size: {SMALL}; color: white; white-space: pre; });

        // The entries (`.neuroglancer-segment-list`, `flex: 1`): IDs as wide
        // as the longest, right-aligned.
        let parent = match scroll {
            Some(top) => {
                let list = b.ui.add_box(parent, scroll_style(top));
                b.key(list, Key::Scroll(self.layer_panel), None);
                list
            }
            None => b.ui.add_box(parent, style! { display: flex; flex-direction: column; flex: 1; height: 0; min-height: 0; overflow: hidden; }),
        };
        let digits = digits.max(1);
        let ch = b.ch("monospace", 13.0);
        for (k, &id) in ids.iter().enumerate() {
            let state = segments.iter().find(|s| s.0 == id);
            self.segment_row(b, parent, i, id, seed, state.is_some(), state.is_some_and(|s| s.1), digits as f32 * ch, icon_base + 10 + 4 * k as u32);
        }
    }

    /// One entry: copy (under the pointer), star, eye, the ID on its colour.
    #[allow(clippy::too_many_arguments)]
    fn segment_row(&self, b: &mut Builder, parent: NodeId, i: usize, id: u64, seed: u32, starred: bool, visible: bool, id_width: f32, icon_base: u32) {
        let key = Key::SegmentEntry(i, id);
        let entry = b.ui.add_box(parent, style! { display: flex; align-items: center; height: 18px; background: black; flex-shrink: 0; });
        // `[data-selected="true"]`: the hovered segment's entry.
        if self.hovered_segments.get(i).copied().flatten() == Some(id) {
            b.ui.style_mut(entry).background = Some(threers::ui::css::color("#222").unwrap());
        }
        b.key(entry, key, None);
        let copy = b.icon(entry, icon_base, Some(icons::COPY), "", Edges::symmetric(0.0, 1.0), Action::CopyText(id.to_string()));
        if !b.hover(key) {
            b.ui.style_mut(copy).opacity = 0.0;
        }
        // Off the pointer, an unstarred star and a hidden eye are not shown.
        let hovered = b.hover(key);
        let star = star_icon(b, entry, icon_base + 1, if starred { Tri::On } else { Tri::Off }, Action::SegmentChange(i, SegmentChange::Star(vec![id], !starred)));
        let eye = eye_icon(b, entry, icon_base + 2, if visible { Tri::On } else { Tri::Off }, Action::SegmentChange(i, SegmentChange::Show(vec![id], !visible)));
        for (node, on) in [(star, starred), (eye, visible)] {
            if !on && !hovered {
                b.ui.style_mut(node).opacity = 0.0;
            }
        }
        let color = ng_core::SegmentColorHash::new(seed).compute(id);
        let text = if crate::panels::use_white(color) { "white" } else { "black" };
        let chip = b.ui.add_box(entry, style! { display: flex; justify-content: flex-end; width: {crate::view::lu(id_width)}; flex-shrink: 0; background: {[color[0], color[1], color[2], 1.0]}; });
        b.text_box(chip, &id.to_string(), style! { font-family: monospace; font-size: 13px; color: {text}; });
    }
}

/// `makeStarButton`: an outline star, filled white when on, 40% when some.
pub(crate) fn star_icon(b: &mut Builder, parent: NodeId, id: u32, state: Tri, action: Action) -> NodeId {
    if state == Tri::On {
        // One path, filled and stroked, as the SVG is.
        return b.icon(parent, id, Some(icons::Icon { paths: icons::STAR.paths, fill: true }), "", Edges::symmetric(0.0, 1.0), action);
    }
    let node = b.icon(parent, id, Some(icons::STAR), "", Edges::symmetric(0.0, 1.0), action);
    if state == Tri::Some {
        under_icon(b, node, icons::STAR.paths, None, [1.0, 1.0, 1.0, 0.4]);
    }
    node
}

/// `makeEyeButton`: the eye, struck through unless visible (the stroke at
/// 40% when some are).
pub(crate) fn eye_icon(b: &mut Builder, parent: NodeId, id: u32, state: Tri, action: Action) -> NodeId {
    match state {
        Tri::On => b.icon(parent, id, Some(icons::EYE), "", Edges::symmetric(0.0, 1.0), action),
        Tri::Off => b.icon(parent, id, Some(icons::EYE_CROSSED), "", Edges::symmetric(0.0, 1.0), action),
        Tri::Some => {
            let node = b.icon(parent, id, Some(icons::EYE), "", Edges::symmetric(0.0, 1.0), action);
            let slash = &icons::EYE_CROSSED.paths[icons::EYE_CROSSED.paths.len() - 1..];
            under_icon(b, node, slash, Some(2.0), [1.0, 1.0, 1.0, 0.4]);
            node
        }
    }
}

/// Paths drawn with an icon's SVG (at its place in the 16px box): filled
/// (`stroke: None`) under it, or stroked over it.
fn under_icon(b: &mut Builder, icon: NodeId, paths: &[&str], stroke: Option<f32>, color: Rgba) {
    b.ui.style_mut(icon).position = threers::ui::Position::Relative;
    let svg = b.ui.children(icon)[0];
    let extra = b.ui.add_path(
        icon,
        style! { position: absolute; left: 2px; top: 1px; width: 16px; height: 16px; color: {color}; },
        paths.iter().map(|p| p.to_string()).collect(),
        [0.0, 0.0, 24.0, 24.0],
        stroke,
        stroke.is_none(),
    );
    if stroke.is_none() {
        // A fill goes under the outline: the outline positioned after it.
        b.ui.style_mut(svg).position = threers::ui::Position::Relative;
        let n = b.ui.children(icon).len();
        b.ui.move_child(icon, extra, n - 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_queries_and_changes_as_neuroglancer() {
        assert_eq!(id_query(" 106, 75 75\t3"), Some(vec![3, 75, 106]));
        assert_eq!(id_query("abc"), None);
        assert_eq!(id_query(""), None);
        let s = vec![(75, true), (82, false)];
        assert_eq!(apply_change(s.clone(), &SegmentChange::Star(vec![82, 5], true)), vec![(75, true), (82, false), (5, false)]);
        assert_eq!(apply_change(s.clone(), &SegmentChange::Star(vec![75], false)), vec![(82, false)]);
        assert_eq!(apply_change(s.clone(), &SegmentChange::Show(vec![82, 9], true)), vec![(75, true), (82, true), (9, true)]);
        assert_eq!(apply_change(s, &SegmentChange::Show(vec![75, 9], false)), vec![(75, false), (82, false)]);
        assert_eq!(segments_json(&[(75, true), (82, false)]), serde_json::json!(["75", "!82"]));
    }
}
