//! Events to action names: neuroglancer's `EventActionMap`
//! (`src/util/event_action_map.ts`, `hierarchical_map.ts`) and default
//! bindings (`src/ui/default_input_event_bindings.ts`), and the help
//! listing made from them (`src/help/input_event_bindings.ts`).
//!
//! A stroke is the key name with its modifiers, always in the order
//! `control+alt+meta+shift+` (neuroglancer's normalized identifier). Key
//! names are DOM `KeyboardEvent.code` lowercased (`keyz`, `arrowleft`,
//! `comma`), or `mousedown0`/`1`/`2`, `dblclick0`, `wheel`.

use std::sync::{Arc, OnceLock};

/// Modifier keys held during an event.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Modifiers {
    pub control: bool,
    pub alt: bool,
    pub meta: bool,
    pub shift: bool,
}

const CONTROL: u8 = 1;
const ALT: u8 = 2;
const META: u8 = 4;
const SHIFT: u8 = 8;

impl Modifiers {
    fn mask(self) -> u8 {
        ((self.control as u8) * CONTROL) | ((self.alt as u8) * ALT) | ((self.meta as u8) * META) | ((self.shift as u8) * SHIFT)
    }
}

/// `getStrokeIdentifier`.
fn stroke_of(key: &str, m: u8) -> String {
    let mut s = String::new();
    for (bit, name) in [(CONTROL, "control+"), (ALT, "alt+"), (META, "meta+"), (SHIFT, "shift+")] {
        if m & bit != 0 {
            s.push_str(name);
        }
    }
    s.push_str(key);
    s
}

/// `getStrokeIdentifier` for held modifiers.
pub fn stroke(key: &str, m: Modifiers) -> String {
    stroke_of(key, m.mask())
}

/// `parseEventIdentifier`: phase, key name, modifiers, optional modifiers.
fn parse(identifier: &str) -> (Option<&str>, &str, u8, u8) {
    let (phase, rest) = match identifier.split_once(':') {
        Some((p, r)) => (Some(p), r),
        None => (None, identifier),
    };
    let (mut key, mut m, mut opt) = (None, 0, 0);
    for part in rest.split('+') {
        match part {
            "control" => m |= CONTROL,
            "control?" => opt |= CONTROL,
            "alt" => m |= ALT,
            "alt?" => opt |= ALT,
            "meta" => m |= META,
            "meta?" => opt |= META,
            "shift" => m |= SHIFT,
            "shift?" => opt |= SHIFT,
            k => {
                assert!(key.is_none(), "invalid event identifier {identifier:?}");
                key = Some(k);
            }
        }
    }
    (phase, key.unwrap_or_else(|| panic!("invalid event identifier {identifier:?}")), m, opt)
}

/// `getStrokeIdentifierWithOptionalModifiers`.
fn original_identifier(key: &str, m: u8, opt: u8) -> String {
    let mut s = String::new();
    for (bit, name) in [(CONTROL, "control"), (ALT, "alt"), (META, "meta"), (SHIFT, "shift")] {
        if m & bit != 0 {
            s.push_str(name);
            s.push('+');
        }
        if opt & bit != 0 {
            s.push_str(name);
            s.push_str("?+");
        }
    }
    s.push_str(key);
    s
}

/// `friendlyEventIdentifier`: without phase and optional modifiers.
pub fn friendly_identifier(identifier: &str) -> String {
    let rest = identifier.split_once(':').filter(|(p, _)| matches!(*p, "at" | "bubble" | "capture")).map_or(identifier, |(_, r)| r);
    rest.split('+').filter(|p| !p.ends_with('?')).collect::<Vec<_>>().join("+")
}

/// What an event is bound to (`EventAction`).
#[derive(Debug, Clone, PartialEq)]
pub struct EventAction {
    pub action: String,
    /// The binding as written, optional modifiers included.
    pub original: String,
}

/// A map from normalized event identifiers (`at:control+keyz`) to actions,
/// with parent maps consulted by priority (`HierarchicalMap`).
#[derive(Debug, Default)]
pub struct EventActionMap {
    pub label: Option<String>,
    /// Highest priority first.
    parents: Vec<(Arc<EventActionMap>, f64)>,
    /// In insertion order (a JS `Map`).
    bindings: Vec<(String, EventAction)>,
}

impl EventActionMap {
    pub fn new(label: Option<&str>) -> Self {
        Self { label: label.map(str::to_owned), ..Default::default() }
    }

    /// `addParent`: after parents of higher or equal priority.
    pub fn add_parent(&mut self, parent: Arc<EventActionMap>, priority: f64) {
        let i = self.parents.iter().take_while(|(_, p)| priority < *p).count();
        self.parents.insert(i, (parent, priority));
    }

    /// `set`: every normalized identifier the (possibly optional-modifier)
    /// `identifier` stands for.
    pub fn set(&mut self, identifier: &str, action: &str) {
        let (phase, key, m, opt) = parse(identifier);
        let value = EventAction { action: action.to_owned(), original: original_identifier(key, m, opt) };
        // `getNormalizedStrokeIdentifiers`.
        let mut strokes = vec![];
        if opt == 0 {
            strokes.push(stroke_of(key, m));
        }
        for x in 0..16u8 {
            if x & (m | opt) == x && x & m == m {
                strokes.push(stroke_of(key, x));
            }
        }
        let phases: &[&str] = match phase {
            None => &["at", "bubble"],
            Some(p) => &[if p == "at" { "at" } else { "bubble" }],
        };
        for s in strokes {
            for p in phases {
                let k = format!("{p}:{s}");
                match self.bindings.iter_mut().find(|(b, _)| *b == k) {
                    Some(entry) => entry.1 = value.clone(),
                    None => self.bindings.push((k, value.clone())),
                }
            }
        }
    }

    fn with(label: Option<&str>, parents: &[(Arc<EventActionMap>, f64)], bindings: &[(&str, &str)]) -> Arc<Self> {
        let mut map = Self::new(label);
        for (p, priority) in parents {
            map.add_parent(p.clone(), *priority);
        }
        for (k, a) in bindings {
            map.set(k, a);
        }
        Arc::new(map)
    }

    /// `get`: parents of positive priority, then this map, then the rest.
    pub fn get(&self, key: &str) -> Option<&EventAction> {
        let (high, low): (Vec<_>, Vec<_>) = self.parents.iter().partition(|(_, p)| *p > 0.0);
        high.iter()
            .find_map(|(p, _)| p.get(key))
            .or_else(|| self.bindings.iter().find(|(k, _)| k == key).map(|(_, v)| v))
            .or_else(|| low.iter().find_map(|(p, _)| p.get(key)))
    }

    /// `dispatchEventWithModifiers` (the map lookup): on Mac, Command
    /// without Control reaches a `control+` binding when there is no
    /// `meta+` one.
    pub fn lookup(&self, phase: &str, key: &str, m: Modifiers, mac: bool) -> Option<&EventAction> {
        self.get(&format!("{phase}:{}", stroke(key, m))).or_else(|| {
            (mac && m.meta && !m.control).then(|| self.get(&format!("{phase}:{}", stroke(key, Modifiers { meta: false, control: true, ..m })))).flatten()
        })
    }
}

/// `getDefaultGlobalBindings`.
pub fn default_global_bindings() -> Arc<EventActionMap> {
    static MAP: OnceLock<Arc<EventActionMap>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = EventActionMap::new(None);
        for (k, a) in [
            ("keyl", "recolor"),
            ("keyx", "clear-segments"),
            ("keys", "toggle-show-slices"),
            ("keyb", "toggle-scale-bar"),
            ("keyv", "toggle-default-annotations"),
            ("keya", "toggle-axis-lines"),
            ("keyo", "toggle-orthographic-projection"),
        ] {
            map.set(k, a);
        }
        for i in 1..=9 {
            map.set(&format!("digit{i}"), &format!("toggle-layer-{i}"));
            map.set(&format!("control+digit{i}"), &format!("select-layer-{i}"));
            map.set(&format!("alt+digit{i}"), &format!("toggle-pick-layer-{i}"));
        }
        for c in b'a'..=b'z' {
            map.set(&format!("alt?+control?+shift+key{}", c as char), &format!("tool-{}", c.to_ascii_uppercase() as char));
        }
        for (k, a) in [
            ("keyn", "add-layer"),
            ("keyh", "help"),
            ("space", "toggle-layout"),
            ("shift+space", "toggle-layout-alternative"),
            ("backslash", "toggle-show-statistics"),
            ("control+keyp", "open-command-palette"),
            ("escape", "deactivate-active-tool"),
        ] {
            map.set(k, a);
        }
        Arc::new(map)
    })
    .clone()
}

/// `getDefaultSelectBindings`.
pub fn default_select_bindings() -> Arc<EventActionMap> {
    static MAP: OnceLock<Arc<EventActionMap>> = OnceLock::new();
    MAP.get_or_init(|| {
        EventActionMap::with(None, &[], &[("control+mousedown2", "select-position"), ("shift+control+mousedown2", "unpin-selected-position")])
    })
    .clone()
}

/// `getDefaultRenderedDataPanelBindings`.
pub fn default_rendered_data_panel_bindings() -> Arc<EventActionMap> {
    static MAP: OnceLock<Arc<EventActionMap>> = OnceLock::new();
    MAP.get_or_init(|| {
        EventActionMap::with(
            Some("All Data Panels"),
            &[(default_select_bindings(), 0.0)],
            &[
                ("arrowleft", "x-"),
                ("arrowright", "x+"),
                ("arrowup", "y-"),
                ("arrowdown", "y+"),
                ("comma", "z-"),
                ("period", "z+"),
                ("bracketleft", "t-"),
                ("bracketright", "t+"),
                ("keyz", "snap"),
                ("control+equal", "zoom-in"),
                ("alt+equal", "depth-range-decrease"),
                ("control+shift+equal", "zoom-in"),
                ("alt+shift+equal", "depth-range-decrease"),
                ("control+minus", "zoom-out"),
                ("alt+minus", "depth-range-increase"),
                ("keyr", "rotate-relative-z-"),
                ("keye", "rotate-relative-z+"),
                ("shift+arrowdown", "rotate-relative-x-"),
                ("shift+arrowup", "rotate-relative-x+"),
                ("shift+arrowleft", "rotate-relative-y-"),
                ("shift+arrowright", "rotate-relative-y+"),
                ("control+wheel", "zoom-via-wheel"),
                ("alt+wheel", "adjust-depth-range-via-wheel"),
                ("at:wheel", "z+1-via-wheel"),
                ("at:shift+wheel", "z+10-via-wheel"),
                ("at:dblclick0", "select"),
                ("at:shift+dblclick0", "star"),
                ("at:control+mousedown0", "annotate"),
                ("at:mousedown2", "move-to-mouse-position"),
                ("at:alt+mousedown0", "move-annotation"),
                ("at:control+alt+mousedown2", "delete-annotation"),
                ("enter", "finish-annotation"),
                ("backspace", "undo-annotation-step"),
                ("at:touchpinch", "zoom-via-touchpinch"),
                ("at:touchrotate", "rotate-in-plane-via-touchrotate"),
                ("at:touchtranslate2", "translate-in-plane-via-touchtranslate"),
                ("at:touchhold1", "move-to-mouse-position"),
                ("at:touchtap1x2", "select"),
                ("at:touchtap2x3", "snap"),
            ],
        )
    })
    .clone()
}

/// `getDefaultPerspectivePanelBindings`.
pub fn default_perspective_panel_bindings() -> Arc<EventActionMap> {
    static MAP: OnceLock<Arc<EventActionMap>> = OnceLock::new();
    MAP.get_or_init(|| {
        EventActionMap::with(
            None,
            &[(default_rendered_data_panel_bindings(), f64::NEG_INFINITY)],
            &[
                ("at:mousedown0", "rotate-via-mouse-drag"),
                ("at:shift+mousedown0", "translate-via-mouse-drag"),
                ("at:touchtranslate1", "rotate-out-of-plane-via-touchtranslate"),
            ],
        )
    })
    .clone()
}

/// `getDefaultSliceViewPanelBindings`.
pub fn default_slice_view_panel_bindings() -> Arc<EventActionMap> {
    static MAP: OnceLock<Arc<EventActionMap>> = OnceLock::new();
    MAP.get_or_init(|| {
        EventActionMap::with(
            None,
            &[(default_rendered_data_panel_bindings(), f64::NEG_INFINITY)],
            &[
                ("at:mousedown0", "translate-via-mouse-drag"),
                ("at:shift+mousedown0", "rotate-via-mouse-drag"),
                ("at:touchtranslate1", "translate-z-via-touchtranslate"),
            ],
        )
    })
    .clone()
}

/// The viewer's maps (`Viewer.inputEventBindings`): each an empty map over
/// the defaults (`setDefaultInputEventBindings`), with the labels the help
/// panel gives them.
pub fn viewer_bindings() -> [(&'static str, Arc<EventActionMap>); 3] {
    let over = |parent: Arc<EventActionMap>| {
        let mut m = EventActionMap::new(None);
        m.add_parent(parent, f64::NEG_INFINITY);
        Arc::new(m)
    };
    [
        ("Global", over(default_global_bindings())),
        ("Cross section view", over(default_slice_view_panel_bindings())),
        ("3-D projection view", over(default_perspective_panel_bindings())),
    ]
}

/// Which binding map a panel uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelKind {
    Slice,
    Perspective,
}

/// The action bound to `key` with `modifiers` in a panel (events at the
/// panel), or `None`.
pub fn lookup(panel: PanelKind, key: &str, modifiers: Modifiers, mac: bool) -> Option<&'static str> {
    let map: &'static EventActionMap = match panel {
        PanelKind::Slice => {
            static M: OnceLock<Arc<EventActionMap>> = OnceLock::new();
            M.get_or_init(default_slice_view_panel_bindings)
        }
        PanelKind::Perspective => {
            static M: OnceLock<Arc<EventActionMap>> = OnceLock::new();
            M.get_or_init(default_perspective_panel_bindings)
        }
    };
    map.lookup("at", key, modifiers, mac).map(|a| a.action.as_str())
}

/// A global key binding (events bubbling to the viewer).
pub fn lookup_global(key: &str, modifiers: Modifiers, mac: bool) -> Option<&'static str> {
    static M: OnceLock<Arc<EventActionMap>> = OnceLock::new();
    M.get_or_init(default_global_bindings).lookup("bubble", key, modifiers, mac).map(|a| a.action.as_str())
}

/// `pressedEventIdentifier`: the binding as shown, with `control+` as
/// `meta+` on Mac when no explicit `meta+` binding exists.
fn pressed_identifier(map: &EventActionMap, normalized: &str, action: &EventAction, mac: bool) -> String {
    let identifier = friendly_identifier(&action.original);
    if !mac {
        return identifier;
    }
    let (phase, key, m, _) = parse(normalized);
    if m & CONTROL == 0 || m & META != 0 {
        return identifier;
    }
    let meta = stroke_of(key, (m & !CONTROL) | META);
    if map.get(&format!("{}:{meta}", phase.unwrap_or("at"))).is_some() {
        return identifier;
    }
    identifier.replacen("control+", "meta+", 1)
}

/// `collectActionBindings`: each action's first keyboard binding in the
/// global, cross-section and 3-D maps (in that order, `entries()` order
/// within), as the command palette shows it (`formatKeyStroke`).
pub fn action_shortcuts(mac: bool) -> Vec<(String, String)> {
    fn entries<'a>(map: &'a EventActionMap, out: &mut Vec<(&'a str, &'a EventAction)>) {
        let (high, low): (Vec<_>, Vec<_>) = map.parents.iter().partition(|(_, p)| *p > 0.0);
        for (p, _) in high {
            entries(p, out);
        }
        out.extend(map.bindings.iter().map(|(k, v)| (k.as_str(), v)));
        for (p, _) in low {
            entries(p, out);
        }
    }
    let mut out: Vec<(String, String)> = vec![];
    for (_, map) in viewer_bindings() {
        let mut all = vec![];
        entries(&map, &mut all);
        for (event, action) in all {
            if ["mouse", "wheel", "touch", "click"].iter().any(|x| event.contains(x)) || out.iter().any(|(a, _)| *a == action.action) {
                continue;
            }
            out.push((action.action.clone(), format_key_stroke(&pressed_identifier(&map, event, action, mac), mac)));
        }
    }
    out
}

/// `formatKeyName`.
fn format_key_name(name: &str) -> &str {
    name.strip_prefix("key").or_else(|| name.strip_prefix("digit")).or_else(|| name.strip_prefix("arrow")).unwrap_or(name)
}

/// `formatKeyStroke`: modifier symbols on Mac (joined without `+` before a
/// one-character key).
pub fn format_key_stroke(stroke: &str, mac: bool) -> String {
    let parts: Vec<&str> = stroke.split('+').collect();
    if !mac {
        return parts.iter().map(|p| format_key_name(p)).collect::<Vec<_>>().join("+");
    }
    let formatted: Vec<&str> = parts
        .iter()
        .map(|p| match *p {
            "control" => "⌃",
            "meta" => "⌘",
            "alt" => "⌥",
            "shift" => "⇧",
            p => format_key_name(p),
        })
        .collect();
    let key = formatted.last().copied().unwrap_or("");
    formatted.join(if key.chars().count() == 1 { "" } else { "+" })
}

type Groups = Vec<(String, Vec<(String, String)>)>;

/// `collectBindings`: each labelled map's bindings (unlabelled parents
/// folded into their child), as `(binding, action)` with runs like
/// `digit1…digit9 → toggle-layer-1…9` written as ranges (`simplifyEntries`),
/// bindings formatted by [`format_key_stroke`].
pub fn help_bindings(maps: &[(&str, Arc<EventActionMap>)], mac: bool) -> Groups {
    fn add_entries(map: &EventActionMap, entries: &mut Vec<(String, String)>, out: &mut Groups, seen: &mut Vec<*const EventActionMap>, mac: bool) {
        for (parent, _) in &map.parents {
            match &parent.label {
                Some(label) => add_map(label, parent, out, seen, mac),
                None => add_entries(parent, entries, out, seen, mac),
            }
        }
        for (event, action) in &map.bindings {
            let id = pressed_identifier(map, event, action, mac);
            match entries.iter_mut().find(|(k, _)| *k == id) {
                Some(e) => e.1 = action.action.clone(),
                None => entries.push((id, action.action.clone())),
            }
        }
    }
    fn add_map(label: &str, map: &Arc<EventActionMap>, out: &mut Groups, seen: &mut Vec<*const EventActionMap>, mac: bool) {
        let ptr = Arc::as_ptr(map);
        if seen.contains(&ptr) {
            return;
        }
        let mut entries = vec![];
        add_entries(map, &mut entries, out, seen, mac);
        seen.push(ptr);
        out.push((label.to_owned(), simplify(entries)));
    }
    fn increment(x: &str, i: u32) -> String {
        let mut chars: Vec<char> = x.chars().collect();
        if let Some(last) = chars.last_mut() {
            *last = char::from_u32(*last as u32 + i).unwrap_or(*last);
        }
        chars.into_iter().collect()
    }
    fn make_range(x: &str, count: u32) -> String {
        let start = x.chars().last().unwrap();
        let end = increment(&start.to_string(), count - 1);
        format!("{}[{start}-{end}]", &x[..x.len() - start.len_utf8()])
    }
    fn simplify(mut entries: Vec<(String, String)>) -> Vec<(String, String)> {
        let mut remap: Vec<(String, (String, String))> = vec![];
        let mut i = 0;
        while i < entries.len() {
            let (id, action) = entries[i].clone();
            if (id.ends_with('a') && action.to_lowercase().ends_with('a')) || (id.ends_with('1') && action.ends_with('1')) {
                let mut k = 1;
                loop {
                    let (oid, oaction) = (increment(&id, k), increment(&action, k));
                    if let Some(j) = entries.iter().position(|(e, a)| *e == oid && *a == oaction) {
                        entries.remove(j);
                        k += 1;
                        continue;
                    }
                    if k != 1 {
                        remap.push((id.clone(), (make_range(&id, k), make_range(&action, k))));
                    }
                    break;
                }
            }
            i += 1;
        }
        entries.into_iter().map(|(id, a)| remap.iter().find(|(r, _)| *r == id).map_or((id, a), |(_, v)| v.clone())).collect()
    }
    let mut out = vec![];
    let mut seen = vec![];
    for (label, map) in maps {
        add_map(label, map, &mut out, &mut seen, mac);
    }
    for (_, entries) in &mut out {
        for e in entries.iter_mut() {
            e.0 = format_key_stroke(&e.0, mac);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const NONE: Modifiers = Modifiers { control: false, alt: false, meta: false, shift: false };

    #[test]
    fn strokes_use_neuroglancers_modifier_order() {
        let m = Modifiers { control: true, alt: true, meta: true, shift: true };
        assert_eq!(stroke("wheel", m), "control+alt+meta+shift+wheel");
        assert_eq!(friendly_identifier("at:alt?+control?+shift+keya"), "shift+keya");
    }

    #[test]
    fn panel_bindings_override_shared_ones() {
        assert_eq!(lookup(PanelKind::Slice, "mousedown0", NONE, false), Some("translate-via-mouse-drag"));
        assert_eq!(lookup(PanelKind::Perspective, "mousedown0", NONE, false), Some("rotate-via-mouse-drag"));
        assert_eq!(lookup(PanelKind::Slice, "wheel", NONE, false), Some("z+1-via-wheel"));
        let shift = Modifiers { shift: true, ..NONE };
        assert_eq!(lookup(PanelKind::Slice, "wheel", shift, false), Some("z+10-via-wheel"));
        assert_eq!(lookup(PanelKind::Slice, "mousedown2", Modifiers { control: true, ..NONE }, false), Some("select-position"));
    }

    #[test]
    fn command_reaches_control_bindings_on_mac_only() {
        let cmd = Modifiers { meta: true, ..NONE };
        assert_eq!(lookup(PanelKind::Slice, "wheel", cmd, true), Some("zoom-via-wheel"));
        assert_eq!(lookup(PanelKind::Slice, "wheel", cmd, false), None);
        assert_eq!(lookup(PanelKind::Slice, "equal", cmd, true), Some("zoom-in"));
    }

    #[test]
    fn global_bindings_with_optional_modifiers() {
        let shift = Modifiers { shift: true, ..NONE };
        assert_eq!(lookup_global("keya", shift, false), Some("tool-A"));
        assert_eq!(lookup_global("keya", Modifiers { control: true, ..shift }, false), Some("tool-A"));
        assert_eq!(lookup_global("keya", NONE, false), Some("toggle-axis-lines"));
        assert_eq!(lookup_global("digit3", Modifiers { alt: true, ..NONE }, false), Some("toggle-pick-layer-3"));
    }

    /// The help panel's contents, as neuroglancer shows them on a Mac
    /// (captured from its DOM: `tests/fixtures/help_bindings_mac.json`).
    #[test]
    fn help_lists_what_neuroglancer_lists() {
        let fixture: Vec<(String, String)> = serde_json::from_str(include_str!("../tests/fixtures/help_bindings_mac.json")).unwrap();
        let mut ours = vec![];
        for (label, entries) in help_bindings(&viewer_bindings(), true) {
            ours.push(("h2".to_owned(), label));
            for (k, a) in entries {
                ours.push(("dt".to_owned(), k));
                ours.push(("dd".to_owned(), a));
            }
        }
        assert_eq!(ours, fixture);
    }

    #[test]
    fn palette_shortcuts_as_neuroglancer() {
        let s = action_shortcuts(true);
        let get = |a: &str| s.iter().find(|(x, _)| x == a).map(|(_, k)| k.as_str());
        assert_eq!(get("toggle-layer-1"), Some("1"));
        assert_eq!(get("select-layer-1"), Some("⌘1"));
        assert_eq!(get("toggle-pick-layer-1"), Some("⌥1"));
        assert_eq!(get("toggle-show-statistics"), Some("backslash"));
        assert_eq!(get("toggle-layout-alternative"), Some("⇧+space"));
        assert_eq!(get("zoom-in"), Some("⌘+equal"));
        assert_eq!(get("rotate-relative-x-"), Some("⇧+down"));
        assert_eq!(get("t-"), Some("bracketleft"));
        assert_eq!(get("finish-annotation"), Some("enter"));
        assert_eq!(get("toggle-layer-list-panel"), None);
    }
}
