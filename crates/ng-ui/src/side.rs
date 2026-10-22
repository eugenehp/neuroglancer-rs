//! Side panels (`src/ui/side_panel.ts`, `side_panel_location.ts`): where each
//! panel docks, as the state records it, and how docked panels group.

use serde_json::{Map, Value};

use ng_viewer::Viewer;

/// The edge a panel docks to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
}

impl Side {
    fn name(self) -> &'static str {
        match self {
            Side::Left => "left",
            Side::Right => "right",
            Side::Top => "top",
            Side::Bottom => "bottom",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "left" => Side::Left,
            "right" => Side::Right,
            "top" => Side::Top,
            "bottom" => Side::Bottom,
            _ => return None,
        })
    }

    /// Whether panels on this side stack in rows (top, bottom) rather
    /// than columns.
    pub fn horizontal(self) -> bool {
        matches!(self, Side::Top | Side::Bottom)
    }

    /// `OUTWARDS_SIGN_FOR_SIDE`.
    pub fn outwards(self) -> f32 {
        match self {
            Side::Left | Side::Top => -1.0,
            Side::Right | Side::Bottom => 1.0,
        }
    }
}

/// `SidePanelLocation`.
#[derive(Debug, Clone, PartialEq)]
pub struct Location {
    pub side: Side,
    pub col: f64,
    pub row: f64,
    pub flex: f64,
    /// Width (left/right) or height (top/bottom), pixels.
    pub size: f64,
    pub min_size: f64,
    pub visible: bool,
}

/// `DEFAULT_SIDE_PANEL_LOCATION`.
pub const DEFAULT_LOCATION: Location =
    Location { side: Side::Right, col: 0.0, row: f64::INFINITY, flex: 1.0, size: 300.0, min_size: 100.0, visible: false };

impl Location {
    /// `TrackableSidePanelLocation.restoreState`: each member if valid, else
    /// the default's.
    pub fn restore(json: Option<&Value>, default: &Location) -> Location {
        let Some(o) = json.and_then(Value::as_object) else { return default.clone() };
        let f = |k: &str, d: f64| o.get(k).and_then(Value::as_f64).filter(|v| v.is_finite()).unwrap_or(d);
        Location {
            side: o.get("side").and_then(Value::as_str).and_then(Side::parse).unwrap_or(default.side),
            col: f("col", default.col),
            row: f("row", default.row),
            flex: f("flex", default.flex),
            size: o.get("size").and_then(Value::as_f64).filter(|v| *v > 0.0 && v.fract() == 0.0).unwrap_or(default.size),
            min_size: default.min_size,
            visible: o.get("visible").and_then(Value::as_bool).unwrap_or(default.visible),
        }
    }

    /// `toJSON`: the members that differ from the default.
    pub fn to_json(&self, default: &Location) -> Map<String, Value> {
        let mut o = Map::new();
        let num = |v: f64| ng_core::js::number(v);
        if self.side != default.side {
            o.insert("side".into(), self.side.name().into());
        }
        for (k, v, d) in [("col", self.col, default.col), ("row", self.row, default.row), ("flex", self.flex, default.flex), ("size", self.size, default.size)] {
            if v != d && v.is_finite() {
                o.insert(k.into(), num(v));
            }
        }
        if self.visible != default.visible {
            o.insert("visible".into(), self.visible.into());
        }
        o
    }
}

/// The viewer's side panels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PanelId {
    /// `LayerListPanel` (`layerListPanel`).
    LayerList,
    /// `InputEventBindingHelpDialog` (`helpPanel`).
    Help,
    /// `ViewerSettingsPanel` (`settingsPanel`).
    Settings,
    /// `SelectionDetailsPanel` (`selection`).
    Selection,
    /// `LayerSidePanel` for the selected layer (`selectedLayer`).
    Layer,
    /// `StatisticsPanel` (`statistics`).
    Statistics,
    /// A `ToolPalettePanel`: the `k`th of `toolPalettes`.
    Palette(u16),
    /// A pinned `LayerSidePanel`: layer `i`'s `k`th of its `panels`.
    Pinned(u16, u16),
}

impl PanelId {
    /// In the order the viewer registers them (`sidePanelManager.registerPanel`).
    pub const ALL: [PanelId; 6] = [PanelId::LayerList, PanelId::Selection, PanelId::Statistics, PanelId::Help, PanelId::Settings, PanelId::Layer];

    pub fn state_key(self) -> &'static str {
        match self {
            PanelId::LayerList => "layerListPanel",
            PanelId::Help => "helpPanel",
            PanelId::Settings => "settingsPanel",
            PanelId::Selection => "selection",
            PanelId::Layer => "selectedLayer",
            PanelId::Statistics => "statistics",
            PanelId::Palette(_) => "toolPalettes",
            PanelId::Pinned(..) => "panels",
        }
    }

    /// The fixed panels, then each tool palette.
    pub fn all(viewer: &Viewer) -> Vec<PanelId> {
        let palettes = crate::palettes::count(viewer) as u16;
        let pinned: Vec<PanelId> = viewer.layers().iter().enumerate().flat_map(|(i, l)| (0..pinned_panels(l).len() as u16).map(move |k| PanelId::Pinned(i as u16, k))).collect();
        // Pinning registers the selected layer's panel anew, after the
        // pinned ones (ties in `renderSide` go by registration).
        let any_pinned = !pinned.is_empty();
        let fixed = PanelId::ALL.iter().copied().filter(move |&p| !any_pinned || p != PanelId::Layer);
        let after = any_pinned.then_some(PanelId::Layer);
        fixed.chain((0..palettes).map(PanelId::Palette)).chain(pinned).chain(after).collect()
    }

    /// A small number per panel (for its widgets' keys).
    pub fn ordinal(self) -> u32 {
        match self {
            PanelId::LayerList => 0,
            PanelId::Help => 1,
            PanelId::Settings => 2,
            PanelId::Selection => 3,
            PanelId::Layer => 4,
            PanelId::Statistics => 5,
            PanelId::Palette(k) => 6 + k as u32,
            PanelId::Pinned(i, k) => 4096 + 64 * i as u32 + k as u32,
        }
    }

    pub fn default_location(self) -> Location {
        let left = |row: f64| Location { side: Side::Left, row, ..DEFAULT_LOCATION };
        match self {
            PanelId::LayerList => left(0.0),
            PanelId::Help => left(1.0),
            PanelId::Settings => left(2.0),
            // `DATA_SELECTION_STATE_DEFAULT_PANEL_LOCATION`.
            PanelId::Selection => Location { row: 1.0, min_size: 150.0, ..DEFAULT_LOCATION },
            // `SELECTED_LAYER_SIDE_PANEL_DEFAULT_LOCATION`.
            PanelId::Layer => Location { row: 0.0, ..DEFAULT_LOCATION },
            // `DEFAULT_STATISTICS_PANEL_LOCATION`.
            PanelId::Statistics => Location { side: Side::Bottom, col: 0.0, row: 0.0, flex: 1.0, size: 100.0, min_size: 50.0, visible: false },
            // `DEFAULT_TOOL_PALETTE_PANEL_LOCATION`.
            PanelId::Palette(_) => Location { side: Side::Right, row: 0.0, visible: true, ..DEFAULT_LOCATION },
            // `LAYER_SIDE_PANEL_DEFAULT_LOCATION`.
            PanelId::Pinned(..) => Location { row: 0.0, visible: true, ..DEFAULT_LOCATION },
        }
    }

    /// The panel's location in `viewer`'s state.
    pub fn location(self, viewer: &Viewer) -> Location {
        if let PanelId::Palette(k) = self {
            let entry = crate::palettes::entry(viewer, k as usize).map(|(_, v)| Value::Object(v));
            return Location::restore(entry.as_ref(), &self.default_location());
        }
        if let PanelId::Pinned(i, k) = self {
            let entry = viewer.layers().get(i as usize).and_then(|l| pinned_panels(l).get(k as usize).cloned());
            return Location::restore(entry.as_ref(), &self.default_location());
        }
        // A `selection` present is shown unless it says not.
        let default = match (self, viewer.ui_state(self.state_key())) {
            (PanelId::Selection, Some(_)) => Location { visible: true, ..self.default_location() },
            _ => self.default_location(),
        };
        let mut l = Location::restore(viewer.ui_state(self.state_key()), &default);
        // `selectedLayer` without a layer that exists is not shown.
        if self == PanelId::Layer && self.layer(viewer).is_none() {
            l.visible = false;
        }
        l
    }

    /// The selected layer's index (`selectedLayer.layer`), for
    /// [`PanelId::Layer`].
    pub fn layer(self, viewer: &Viewer) -> Option<usize> {
        let name = viewer.ui_state("selectedLayer")?.get("layer")?.as_str()?;
        viewer.layers().iter().position(|l| l.name() == name && !l.spec.archived)
    }

    /// Writes `location` into `viewer`'s state, keeping the key's other
    /// members (`selectedLayer.layer`, …).
    pub fn set_location(self, viewer: &mut Viewer, location: &Location) {
        if let PanelId::Pinned(i, k) = self {
            let Some(layer) = viewer.layers().get(i as usize) else { return };
            let mut panels = pinned_panels(layer);
            let Some(Value::Object(o)) = panels.get(k as usize).cloned() else { return };
            let mut out = location.to_json(&self.default_location());
            for (key, v) in o {
                if !["side", "col", "row", "flex", "size", "visible"].contains(&key.as_str()) {
                    out.insert(key, v);
                }
            }
            panels[k as usize] = Value::Object(out);
            viewer.set_layer_option(i as usize, "panels", Some(Value::Array(panels)));
            return;
        }
        if let PanelId::Palette(k) = self {
            let Some((name, mut o)) = crate::palettes::entry(viewer, k as usize) else { return };
            for k in ["side", "col", "row", "flex", "size", "visible"] {
                o.remove(k);
            }
            let mut out = location.to_json(&self.default_location());
            out.extend(o);
            crate::palettes::set_entry(viewer, k as usize, &name, out);
            return;
        }
        let mut o = viewer.ui_state(self.state_key()).and_then(Value::as_object).cloned().unwrap_or_default();
        for k in ["side", "col", "row", "flex", "size", "visible"] {
            o.remove(k);
        }
        if self == PanelId::Selection {
            // `TrackableDataSelectionState.toJSON`: shown, what differs from
            // a shown panel; hidden, what differs, then `visible: false`.
            if location.visible {
                o.extend(location.to_json(&Location { visible: true, ..self.default_location() }));
                if o.is_empty() {
                    viewer.set_ui_state(self.state_key(), Some(Value::Object(o)));
                    return;
                }
            } else {
                let diff = location.to_json(&self.default_location());
                if !diff.is_empty() {
                    o.extend(diff);
                    o.insert("visible".into(), false.into());
                }
            }
        } else {
            o.extend(location.to_json(&self.default_location()));
        }
        // `emptyToUndefined`.
        viewer.set_ui_state(self.state_key(), (!o.is_empty()).then_some(Value::Object(o)));
    }
}

/// A layer's pinned panels (`panels`), each its location, `tab` and
/// maybe `tabs`.
pub fn pinned_panels(layer: &ng_viewer::ManagedLayer) -> Vec<Value> {
    layer.spec.json.get("panels").and_then(Value::as_array).cloned().unwrap_or_default()
}

/// `SidePanelManager.renderSide`'s write-back: on each side, the panels
/// sorted by column then row (ties in registration order) and renumbered
/// from 0 — columns by group, rows within one — and the visible ones of a
/// group all its width (its first visible panel's, at least the largest
/// minimum).
pub fn normalize_locations(viewer: &mut Viewer) {
    let all = PanelId::all(viewer);
    for side in [Side::Left, Side::Right, Side::Top, Side::Bottom] {
        let mut panels: Vec<(PanelId, Location)> = all.iter().map(|&id| (id, id.location(viewer))).filter(|(_, l)| l.side == side).collect();
        let cross = |l: &Location| if side.horizontal() { l.row } else { l.col };
        let along = |l: &Location| if side.horizontal() { l.col } else { l.row };
        panels.sort_by(|a, b| cross(&a.1).total_cmp(&cross(&b.1)).then(along(&a.1).total_cmp(&along(&b.1))));
        let mut i = 0;
        let mut cross_index = 0.0;
        while i < panels.len() {
            let c = cross(&panels[i].1);
            let n = panels[i..].iter().take_while(|(_, l)| cross(l) == c).count();
            let group = &panels[i..i + n];
            let min = group.iter().filter(|(_, l)| l.visible).map(|(_, l)| l.min_size).fold(0.0, f64::max);
            let size = group.iter().find(|(_, l)| l.visible).map(|(_, l)| l.size.max(min));
            for (flex_index, (id, l)) in group.iter().enumerate() {
                let mut new = l.clone();
                if side.horizontal() {
                    (new.row, new.col) = (cross_index, flex_index as f64);
                } else {
                    (new.col, new.row) = (cross_index, flex_index as f64);
                }
                if let (true, Some(size)) = (l.visible, size) {
                    new.size = size;
                }
                if new != *l {
                    id.set_location(viewer, &new);
                }
            }
            i += n;
            cross_index += 1.0;
        }
    }
}

/// One column (left/right) of docked panels: its width and visible panels
/// in order (`SidePanelFlex`).
#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    pub side: Side,
    pub size: f64,
    pub panels: Vec<(PanelId, Location)>,
}

/// `renderSide`: the visible panels on `side`, grouped by column (`col`) and
/// ordered by `row` (top and bottom: by `row`, ordered by `col`), each group
/// as wide (tall) as its first panel says (at least the largest minimum).
pub fn groups(viewer: &Viewer, side: Side) -> Vec<Group> {
    let mut panels: Vec<(PanelId, Location)> = PanelId::all(viewer).into_iter().map(|id| (id, id.location(viewer))).filter(|(_, l)| l.side == side).collect();
    // `crossKey` and `flexKey`.
    let cross = |l: &Location| if side.horizontal() { l.row } else { l.col };
    let along = |l: &Location| if side.horizontal() { l.col } else { l.row };
    // A stable sort, as `Array.prototype.sort`.
    panels.sort_by(|a, b| cross(&a.1).total_cmp(&cross(&b.1)).then(along(&a.1).total_cmp(&along(&b.1))));
    let mut out: Vec<Group> = vec![];
    let mut i = 0;
    while i < panels.len() {
        let col = cross(&panels[i].1);
        let members: Vec<(PanelId, Location)> = panels[i..].iter().take_while(|(_, l)| cross(l) == col).cloned().collect();
        i += members.len();
        let visible: Vec<(PanelId, Location)> = members.into_iter().filter(|(_, l)| l.visible).collect();
        if visible.is_empty() {
            continue;
        }
        let min = visible.iter().map(|(_, l)| l.min_size).fold(0.0, f64::max);
        out.push(Group { side, size: visible[0].1.size.max(min), panels: visible });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locations_round_trip_as_neuroglancer_writes_them() {
        let d = PanelId::Settings.default_location();
        assert_eq!(Location::restore(None, &d), d);
        let l = Location::restore(Some(&serde_json::json!({"visible": true, "size": 250, "side": "right", "row": "x"})), &d);
        assert_eq!((l.visible, l.size, l.side, l.row), (true, 250.0, Side::Right, 2.0));
        assert_eq!(Value::Object(l.to_json(&d)), serde_json::json!({"side": "right", "size": 250, "visible": true}));
        assert!(DEFAULT_LOCATION.to_json(&DEFAULT_LOCATION).is_empty());
    }
}
