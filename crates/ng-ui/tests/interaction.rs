//! The UI over a loaded viewer: its layout against neuroglancer's measured
//! DOM, and its controls doing what neuroglancer's do. Skips without a GPU.

use std::sync::Arc;

use ng_chunk::ThreadPool;
use ng_ui::{Action, Button, Effect, Modifiers, NgUi};
use ng_viewer::{Layout, Viewer};
use threers::ui::{Style, TextMeasure};

const W: u32 = 1200;
const H: u32 = 800;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Chrome's metrics for the fonts neuroglancer's UI uses, near enough:
/// monospace `0` is 0.6 em, line boxes as Chrome's (15 px at 13 px).
struct ChromeLikeText;

impl TextMeasure for ChromeLikeText {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        let advance = if style.font_family == "monospace" { 0.6 } else { 0.5 };
        let n = text.chars().count() as f32;
        (n * advance * style.font_size, style.line_height.unwrap_or((style.font_size * 1.15).round()))
    }
}

struct Setup {
    viewer: Viewer,
    ui: NgUi,
    _target: wgpu::Texture,
    view: wgpu::TextureView,
}

fn setup() -> Option<Setup> {
    let gpu = threers::gpu::testing::context("ng-ui interaction")?;
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/../ng-viewer/tests/fixtures");
    let state = serde_json::json!({
        "dimensions": { "x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"] },
        "position": [19.123, 14.777, 10.61],
        "crossSectionScale": 0.3171,
        "layers": [
            { "type": "image", "name": "em", "source": format!("precomputed://file://{fixtures}/image") },
            { "type": "segmentation", "name": "seg", "source": format!("precomputed://file://{fixtures}/seg"), "segments": ["75", "106"] }
        ],
        "layout": "4panel"
    });
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let mut viewer = Viewer::new(device, queue, FORMAT, W, H, 1.0, Arc::new(ThreadPool::new(2)), Arc::new(|| {}));
    viewer.open_state(&state).unwrap();
    let start = std::time::Instant::now();
    while viewer.is_loading() && start.elapsed().as_secs() < 30 {
        viewer.poll();
        viewer.render(&view);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let mut ui = NgUi::new();
    ui.update(&mut viewer, W as f32, H as f32, &mut ChromeLikeText);
    Some(Setup { viewer, ui, _target: target, view })
}

impl Setup {
    fn update(&mut self) {
        self.viewer.render(&self.view);
        self.ui.update(&mut self.viewer, W as f32, H as f32, &mut ChromeLikeText);
    }

    /// A primary click, as a shell routes it: to the UI, else the viewer.
    fn click(&mut self, x: f32, y: f32, modifiers: Modifiers) -> Option<Action> {
        self.ui.pointer_move(x, y);
        let action = self.ui.pointer_down(x, y, Button::Primary, modifiers, &self.viewer);
        if let Some(a) = &action {
            self.ui.apply(&mut self.viewer, a);
        }
        // A layer item's click comes with the release.
        let release = self.ui.pointer_up_action();
        if let Some(a) = &release {
            self.ui.apply(&mut self.viewer, a);
        }
        self.ui.pointer_up();
        self.update();
        release.or(action)
    }

    fn key(&mut self, key: &str) -> Option<Action> {
        let action = self.ui.key_down(key, Modifiers::default(), &self.viewer);
        if let Some(a) = &action {
            self.ui.apply(&mut self.viewer, a);
        }
        self.update();
        action
    }

    fn position(&self) -> Vec<f32> {
        self.viewer.navigation().position().to_vec()
    }
}

#[test]
fn bars_take_the_top_46_pixels() {
    let Some(s) = setup() else { return };
    // Top bar 20 + 3, layer bar 24 from y = 22.
    assert_eq!(s.viewer.viewport(), Some([0.0, 46.0, W as f64, H as f64 - 46.0]));
}

#[test]
fn layer_items_toggle_select_and_pick() {
    let Some(mut s) = setup() else { return };
    assert_eq!(s.click(30.0, 34.0, Modifiers::default()), Some(Action::ToggleLayerVisible(0)));
    assert!(!s.viewer.layers()[0].spec.visible);
    let alt = Modifiers { alt: true, ..Default::default() };
    assert_eq!(s.click(30.0, 34.0, alt), Some(Action::TogglePick(0)));
    assert_eq!(s.viewer.layers()[0].spec.json.get("pick"), Some(&false.into()));
    let control = Modifiers { control: true, ..Default::default() };
    assert_eq!(s.click(30.0, 34.0, control), Some(Action::ToggleLayerSidePanel(0)));
}

#[test]
fn coordinates_are_typed_stepped_and_picked() {
    let Some(mut s) = setup() else { return };
    // The x coordinate (at about 60–80 px).
    assert_eq!(s.click(70.0, 10.0, Modifiers::default()), Some(Action::Repaint));
    assert!(s.ui.has_keyboard_focus());
    // Typing replaces the selected value; Enter commits, to the voxel centre.
    s.key("5");
    s.key("Enter");
    assert_eq!(s.position()[0], 5.5);
    // ↓ steps by one voxel; the wheel does the same over the dimension.
    s.key("ArrowDown");
    assert_eq!(s.position()[0], 6.5);
    let action = s.ui.wheel(70.0, 10.0, -3.0, Modifiers::default());
    assert_eq!(action, Some(Action::AdjustCoordinate(0, -1.0)));
    s.ui.apply(&mut s.viewer, &action.unwrap());
    assert_eq!(s.position()[0], 5.5);
    // Tab moves on to y; typing "7," moves on again to z, committing
    // nothing until Enter (or focus leaving the widget) commits both.
    s.key("Tab");
    s.key("7");
    s.key(",");
    assert_eq!(s.position()[1], 14.777);
    s.key("Enter");
    assert_eq!(s.position()[1], 7.5);
    // Escape drops what was typed and leaves.
    s.key("Tab");
    s.key("Tab");
    s.key("3");
    s.key("Escape");
    assert!(!s.ui.has_keyboard_focus());
    assert_eq!(s.position()[1], 7.5);
    // A click elsewhere commits a typed value (`blur`).
    s.click(70.0, 10.0, Modifiers::default());
    s.key("1");
    s.key("2");
    s.click(600.0, 300.0, Modifiers::default());
    assert_eq!(s.position()[0], 12.5);
}

#[test]
fn the_dimension_plot_sets_the_position() {
    let Some(mut s) = setup() else { return };
    s.click(70.0, 10.0, Modifiers::default());
    // The plot's canvas: the dropdown's top (19), its border and padding
    // (3), the tool line (18), 0.5em (8).
    let top = 19.0 + 3.0 + 18.0 + 8.0;
    let x = s.ui.tree().rect(s.ui.tree().root()).width.min(40.0);
    s.ui.pointer_move(x, top + 1.0);
    let a = s.ui.pointer_down(x, top, Button::Primary, Modifiers::default(), &s.viewer).expect("plot press");
    s.ui.apply(&mut s.viewer, &a);
    assert_eq!(s.position()[0], 0.5, "{a:?}");
    // Dragging follows the pointer, past the canvas too; release ends it.
    let a = s.ui.pointer_move(x, top + 500.0).expect("drag");
    s.ui.apply(&mut s.viewer, &a);
    assert!(s.position()[0] > 10.0);
    assert!(s.ui.pointer_up());
    assert!(s.ui.pointer_move(x, top + 50.0).is_none_or(|a| a == Action::Repaint));
}

#[test]
fn layout_buttons_and_sections() {
    let Some(mut s) = setup() else { return };
    // The xz panel's ◫ button: top right of the top-right panel.
    let a = s.click(1185.0, 61.0, Modifiers::default());
    assert_eq!(a, Some(Action::SetLayout(Layout::SliceAnd3d(ng_core::panel::NamedAxes::Xz))));
    assert_eq!(s.viewer.layout, Layout::SliceAnd3d(ng_core::panel::NamedAxes::Xz));
    // The copy button copies `formatPositionText`.
    let effect = s.ui.apply(&mut s.viewer, &Action::CopyPosition);
    assert_eq!(effect, Effect::Clipboard("19, 14, 10".into()));
}

#[test]
fn layers_are_added_typed_and_retyped() {
    let Some(mut s) = setup() else { return };
    // `+`: a new layer, its URL focused.
    s.ui.apply(&mut s.viewer, &Action::AddLayer);
    s.update();
    let i = s.viewer.layers().len() - 1;
    assert_eq!((s.viewer.layers()[i].name(), s.viewer.layers()[i].spec.layer_type.as_str()), ("new layer", "new"));
    assert!(s.ui.has_keyboard_focus());
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/../ng-viewer/tests/fixtures");
    for c in format!("precomputed://file://{fixtures}/image").chars() {
        s.key(&c.to_string());
    }
    let action = s.ui.key_down("Enter", Modifiers::default(), &s.viewer).expect("commit");
    let Effect::LoadSources(pending) = s.ui.apply(&mut s.viewer, &action) else { panic!("{action:?}") };
    for p in pending {
        let result = futures::executor::block_on(ng_datasource::url::get(&p.url));
        s.viewer.source_loaded(p.layer, p.index, result);
    }
    s.update();
    // Named from the URL; its URL the canonical one; detected as an image.
    let layer = &s.viewer.layers()[i];
    assert_eq!(layer.name(), "image");
    assert!(layer.spec.sources[0].url.ends_with("|neuroglancer-precomputed:"), "{}", layer.spec.sources[0].url);
    assert_eq!(s.viewer.detected_layer_type(i), Some("image"));
    s.ui.apply(&mut s.viewer, &Action::ChangeLayerType(i, "image"));
    assert!(matches!(s.viewer.layers()[i].kind(), Some(ng_viewer::LayerKind::Image { .. })));
    // The layer type select: `seg`.
    s.ui.apply(&mut s.viewer, &Action::SetLayerOption(i, "type".into(), Some("seg".into())));
    assert!(matches!(s.viewer.layers()[i].kind(), Some(ng_viewer::LayerKind::Segmentation(_))));
}

#[test]
fn the_command_palette_finds_and_runs() {
    let Some(mut s) = setup() else { return };
    s.ui.apply(&mut s.viewer, &Action::OpenPalette);
    s.update();
    let zoom = s.viewer.navigation().zoom(ng_core::navigation::View::CrossSection);
    for c in ["z", "o", "o", "m"] {
        s.key(c);
    }
    // "Zoom In" first (a prefix match), run by Enter.
    let action = s.ui.key_down("Enter", Modifiers::default(), &s.viewer).expect("run");
    s.ui.apply(&mut s.viewer, &action);
    assert!(!s.ui.has_keyboard_focus());
    assert_eq!(s.viewer.navigation().zoom(ng_core::navigation::View::CrossSection), zoom * 0.5);
    // A group opens on its own list: Toggle Layer Visibility, then `em`.
    s.ui.apply(&mut s.viewer, &Action::OpenPalette);
    s.update();
    s.ui.apply(&mut s.viewer, &Action::PaletteRun(0));
    s.update();
    s.ui.apply(&mut s.viewer, &Action::PaletteRun(0));
    assert!(!s.viewer.layers()[0].spec.visible);
}

#[test]
fn playback_is_enabled_negated_and_played() {
    let Some(mut s) = setup() else { return };
    s.click(70.0, 10.0, Modifiers::default());
    let at = |s: &Setup, label: &str| s.ui.text_rects().into_iter().find(|(t, _)| t == label).map(|(_, r)| r).unwrap_or_else(|| panic!("no {label}"));
    // The video icon, left of "Playback", enables it (paused, reversing).
    let r = at(&s, "Playback");
    let a = s.click(r.x - 6.0, r.y + r.height / 2.0, Modifiers::default());
    assert!(matches!(a, Some(Action::SetVelocity(0, Some(_)))), "{a:?}");
    let v = s.viewer.velocity(0).expect("enabled");
    assert_eq!((v.velocity, v.paused), (10.0, true));
    // ± negates; the speed and boundary show.
    let r = at(&s, "±");
    s.click(r.x + r.width / 2.0, r.y + r.height / 2.0, Modifiers::default());
    assert_eq!(s.viewer.velocity(0).unwrap().velocity, -10.0);
    at(&s, "-10");
    at(&s, "reverse");
    // Alt+wheel over the dimension scales it.
    let alt = Modifiers { alt: true, ..Default::default() };
    let a = s.ui.wheel(70.0, 10.0, -200.0 * 2f64.ln(), alt).expect("velocity wheel");
    s.ui.apply(&mut s.viewer, &a);
    assert_eq!(s.viewer.velocity(0).unwrap().velocity, -5.0);
    // Unpaused, x advances at its speed.
    s.ui.apply(&mut s.viewer, &Action::TogglePlayback(0));
    let x = s.position()[0];
    assert!(s.viewer.step_playback(1000.0));
    assert!(s.viewer.step_playback(1200.0));
    assert!((s.position()[0] - (x - 1.0)).abs() < 1e-4, "{} from {x}", s.position()[0]);
    // The icon again disables it.
    let r = at(&s, "Playback");
    s.click(r.x - 6.0, r.y + r.height / 2.0, Modifiers::default());
    assert_eq!(s.viewer.velocity(0), None);
}

#[test]
fn tools_are_bound_activated_and_used() {
    let Some(mut s) = setup() else { return };
    use ng_viewer::tools::ToolOwner;
    // The dimension dropdown's `[ ]`: held, a letter binds the x tool.
    s.click(70.0, 10.0, Modifiers::default());
    let r = s.ui.text_rects().into_iter().find(|(t, _)| t == "[").map(|(_, r)| r).expect("binding widget");
    let (x, y) = (r.x + 3.0, r.y + r.height / 2.0);
    s.ui.pointer_move(x, y);
    s.ui.pointer_down(x, y, Button::Primary, Modifiers::default(), &s.viewer);
    assert!(s.ui.has_keyboard_focus());
    let a = s.ui.key_down("d", Modifiers::default(), &s.viewer).expect("bind");
    s.ui.apply(&mut s.viewer, &a);
    s.ui.pointer_up();
    s.update();
    assert_eq!(s.viewer.tool('D').map(|t| (t.owner, t.json)), Some((ToolOwner::Global, serde_json::json!({ "type": "dimension", "dimension": "x" }))));
    assert_eq!(s.viewer.state_json()["toolBindings"], serde_json::json!({ "D": { "type": "dimension", "dimension": "x" } }));
    // Binding D to a layer's control moves it there; Q binds the opacity.
    s.ui.apply(&mut s.viewer, &Action::BindTool('Q', ToolOwner::Layer(0), "opacity".into()));
    s.ui.apply(&mut s.viewer, &Action::BindTool('D', ToolOwner::Layer(0), "blend".into()));
    assert_eq!(s.viewer.state_json().get("toolBindings"), None);
    assert_eq!(s.viewer.state_json()["layers"][0]["toolBindings"], serde_json::json!({ "Q": "opacity", "D": "blend" }));
    // Shift+Q holds the tool: shift+wheel over a panel steps the opacity.
    let shift = Modifiers { shift: true, ..Default::default() };
    let a = s.ui.shortcut("KeyQ", shift, &s.viewer).expect("tool-Q");
    s.ui.apply(&mut s.viewer, &a);
    s.update();
    assert!(s.ui.text_rects().iter().any(|(t, _)| t == "shift+wheel→adjust-via-wheel"));
    let a = s.ui.wheel(300.0, 200.0, 100.0, shift).expect("tool wheel");
    s.ui.apply(&mut s.viewer, &a);
    assert_eq!(s.viewer.layers()[0].spec.json["opacity"], serde_json::json!(0.51));
    // Released, the wheel is the panel's again.
    assert!(s.ui.key_up("KeyQ"));
    s.update();
    assert_eq!(s.ui.wheel(300.0, 200.0, 100.0, shift), None);
    // A double click on the bound tool in the top bar unbinds it.
    let r = s.ui.text_rects().into_iter().find(|(t, _)| t == "Opacity (slice)").map(|(_, r)| r).expect("top bar tool");
    let a = s.ui.double_click(r.x + 2.0, r.y + 2.0).expect("unbind");
    s.ui.apply(&mut s.viewer, &a);
    assert_eq!(s.viewer.tool('Q'), None);
}

#[test]
fn tool_palettes_list_query_and_convert() {
    let Some(mut s) = setup() else { return };
    // The wrench opens its list; "All controls" makes a palette of `+`.
    s.click(992.0, 10.0, Modifiers::default());
    let r = s.ui.text_rects().into_iter().find(|(t, _)| t == "All controls").map(|(_, r)| r).expect("canned palette");
    let a = s.click(r.x + 5.0, r.y + 5.0, Modifiers::default());
    assert!(matches!(a, Some(Action::AddPalette { .. })), "{a:?}");
    // Renumbered after the (hidden) selected layer panel, as neuroglancer's.
    assert_eq!(s.viewer.state_json()["toolPalettes"], serde_json::json!({ "All controls": { "row": 1, "query": "+" } }));
    // Shown on the right: the dimensions, then each layer's controls.
    let texts: Vec<String> = s.ui.text_rects().into_iter().filter(|(_, r)| r.x > 900.0).map(|(t, _)| t).collect();
    for t in ["em", "Opacity (slice)", "normalized", "seg", "Saturation", "Highlight on hover"] {
        assert!(texts.iter().any(|x| x == t), "{t} in {texts:?}");
    }
    // The search icon makes the query's results its list.
    s.ui.apply(&mut s.viewer, &Action::ConvertPalette(0));
    let tools = s.viewer.state_json()["toolPalettes"]["All controls"]["tools"].clone();
    assert_eq!(tools[0], serde_json::json!({ "type": "dimension", "dimension": "x" }));
    assert_eq!(tools[5], serde_json::json!({ "layer": "em", "type": "opacity" }));
    assert_eq!(tools.as_array().unwrap().len(), 18);
    // A double click on a tool removes it; the bin deletes the palette.
    s.ui.apply(&mut s.viewer, &Action::RemovePaletteTool(0, 0));
    assert_eq!(s.viewer.state_json()["toolPalettes"]["All controls"]["tools"].as_array().unwrap().len(), 17);
    s.ui.apply(&mut s.viewer, &Action::DeletePalette(0));
    assert_eq!(s.viewer.state_json().get("toolPalettes"), None);
}

#[test]
fn dimensions_are_renamed_and_rescaled() {
    let Some(mut s) = setup() else { return };
    let at = |s: &Setup, label: &str| s.ui.text_rects().into_iter().find(|(t, _)| t == label).map(|(_, r)| r).unwrap_or_else(|| panic!("no {label}"));
    // A double click on the name makes it editable, all of it selected.
    let r = at(&s, "x");
    let (x, y) = (r.x + 2.0, r.y + 5.0);
    s.click(x, y, Modifiers::default());
    let a = s.ui.double_click(x, y).expect("edit name");
    s.ui.apply(&mut s.viewer, &a);
    s.update();
    for k in ["q", "Enter"] {
        s.key(k);
    }
    assert_eq!(s.viewer.dimensions().names, vec!["q", "y", "z"]);
    let state = s.viewer.state_json();
    assert!(state["dimensions"].get("q").is_some(), "{}", state["dimensions"]);
    // A name a layer's space has, or not a valid name, is refused.
    assert!(!s.viewer.rename_dimensions(&["y".into(), "y".into(), "z".into()]));
    assert!(!s.viewer.rename_dimensions(&["1a".into(), "y".into(), "z".into()]));
    // The scale: `8nm` doubles it.
    let r = at(&s, "4nm");
    let (x, y) = (r.x + 2.0, r.y + 5.0);
    s.click(x, y, Modifiers::default());
    let a = s.ui.double_click(x, y).expect("edit scale");
    s.ui.apply(&mut s.viewer, &a);
    s.update();
    for k in ["8", "n", "m", "Enter"] {
        s.key(k);
    }
    assert_eq!(s.viewer.dimensions().scales[0], 8e-9);
    assert_eq!(s.viewer.state_json()["dimensions"]["q"], serde_json::json!([8e-9, "m"]));
}

#[test]
fn layers_are_dragged_to_reorder_and_copy() {
    let Some(mut s) = setup() else { return };
    let names = |s: &Setup| s.viewer.layers().iter().map(|l| l.name().to_owned()).collect::<Vec<_>>();
    let drag = |s: &mut Setup, path: &[(f32, f32)], modifiers: Modifiers| {
        let (x, y) = path[0];
        s.ui.pointer_move(x, y);
        let a = s.ui.pointer_down(x, y, Button::Primary, modifiers, &s.viewer);
        if let Some(a) = a {
            s.ui.apply(&mut s.viewer, &a);
        }
        for &(x, y) in &path[1..] {
            if let Some(a) = s.ui.pointer_move(x, y) {
                if let Effect::LoadSources(pending) = s.ui.apply(&mut s.viewer, &a) {
                    for p in pending {
                        let result = futures::executor::block_on(ng_datasource::url::get(&p.url));
                        s.viewer.source_loaded(p.layer, p.index, result);
                    }
                }
            }
            s.update();
        }
        // The drag's status shows while it lasts.
        assert!(s.ui.text_rects().iter().any(|(t, _)| t.starts_with("Drop to") || t.starts_with("Drag layer")), "{:?}", s.ui.text_rects());
        let release = s.ui.pointer_up_action();
        s.ui.pointer_up();
        s.update();
        release
    };
    // `em` dragged over `seg`: after it; nothing toggled.
    let release = drag(&mut s, &[(30.0, 34.0), (40.0, 34.0), (120.0, 34.0)], Modifiers::default());
    assert_eq!(release, Some(Action::Repaint));
    assert_eq!(names(&s), ["seg", "em"]);
    assert!(s.viewer.layers()[1].spec.visible);
    // Shift copies: `seg` dragged onto the bar's end makes `seg1` there.
    let shift = Modifiers { shift: true, ..Default::default() };
    drag(&mut s, &[(30.0, 34.0), (40.0, 34.0), (400.0, 34.0)], shift);
    assert_eq!(names(&s), ["seg", "em", "seg1"]);
}

#[test]
fn layer_panels_are_pinned_and_unpinned() {
    let Some(mut s) = setup() else { return };
    let state = |s: &mut Setup| {
        let j = s.viewer.state_json();
        (j.get("selectedLayer").cloned(), j["layers"][0].get("tab").cloned(), j["layers"][0].get("panels").cloned())
    };
    s.ui.apply(&mut s.viewer, &Action::SelectLayer(0));
    s.update();
    // Pinned: the panel stays, as `em`'s; the selected layer's hides a row on.
    s.ui.apply(&mut s.viewer, &Action::PinLayerPanel);
    s.update();
    assert_eq!(state(&mut s), (Some(serde_json::json!({ "row": 1, "layer": "em" })), None, Some(serde_json::json!([{ "tab": "source" }]))));
    // Selecting `seg` shows its panel under the pinned one.
    s.ui.apply(&mut s.viewer, &Action::SelectLayer(1));
    s.update();
    assert_eq!(s.viewer.state_json()["selectedLayer"], serde_json::json!({ "row": 1, "visible": true, "layer": "seg" }));
    let titles: Vec<(String, f32)> = s.ui.text_rects().into_iter().filter(|(t, r)| (t == "em" || t == "seg") && r.x > 900.0).map(|(t, r)| (t, r.y)).collect();
    assert!(titles.len() >= 2 && titles[0].0 == "em" && titles[0].1 < titles[1].1, "{titles:?}");
    // Unpinning `em`: `seg`'s panel is pinned first, `em`'s is selected.
    s.ui.apply(&mut s.viewer, &Action::UnpinLayerPanel(0, 0));
    s.update();
    let j = s.viewer.state_json();
    assert_eq!(j["selectedLayer"]["layer"], "em");
    assert_eq!(j["layers"][0].get("panels"), None);
    assert_eq!(j["layers"][0]["tab"], "source");
    assert_eq!(j["layers"][1]["panels"].as_array().map(Vec::len), Some(1));
}

#[test]
fn the_colour_picker_sets_colours() {
    let Some(mut s) = setup() else { return };
    // Opened (as the native shell does for `Effect::PickColor`) for the
    // cross-section background, centred without its well showing.
    s.ui.open_color_picker(ng_ui::ColorSetting::CrossSectionBackground, [0.0, 0.0, 1.0]);
    s.update();
    assert!(s.ui.has_keyboard_focus());
    // The field's top right: saturation 1, value 1 of the hue (blue).
    let (x, y) = ((W as f32 - 218.0) / 2.0 + 9.0 + 199.0, (H as f32 - 196.0) / 2.0 + 9.0 + 1.0);
    let a = s.ui.pointer_down(x, y, Button::Primary, Modifiers::default(), &s.viewer).expect("pick");
    s.ui.apply(&mut s.viewer, &a);
    s.ui.pointer_up();
    let c = s.viewer.background;
    assert!(c[2] > 0.95 && c[0] < 0.05, "{c:?}");
    // Escape closes it.
    s.key("Escape");
    assert!(!s.ui.has_keyboard_focus());
}

#[test]
fn shaders_are_edited_and_their_controls_used() {
    let Some(mut s) = setup() else { return };
    // em's Rendering tab.
    s.ui.apply(&mut s.viewer, &Action::SelectLayer(0));
    s.ui.apply(&mut s.viewer, &Action::SelectLayerTab(0, "rendering"));
    s.ui.set_time(0.0);
    s.update();
    let at = |s: &Setup, label: &str| s.ui.text_rects().into_iter().find(|(t, _)| t == label).map(|(_, r)| r);
    // A click at the end of the code's last line, then a new line.
    let r = at(&s, "}").expect("the shader's closing brace");
    s.click(r.x + 30.0, r.y + 5.0, Modifiers::default());
    assert!(s.ui.has_keyboard_focus());
    let code = "#uicontrol float gain slider(min=0, max=2)";
    for k in ["Enter"].iter().copied().chain(std::iter::empty()) {
        s.key(k);
    }
    for c in code.chars() {
        s.key(&c.to_string());
    }
    // Set once the typing has rested 500 ms.
    assert_eq!(s.viewer.layers()[0].spec.json.get("shader"), None);
    s.ui.set_time(600.0);
    s.update();
    let shader = s.viewer.layers()[0].spec.json["shader"].as_str().unwrap().to_owned();
    assert!(shader.ends_with(&format!("}}\n{code}\n")), "{shader:?}");
    assert_eq!(s.viewer.layers()[0].shader_error(), None);
    // Its controls show: `normalized`, then `gain` at its default (1).
    s.key("Escape");
    assert!(!s.ui.has_keyboard_focus());
    assert!(at(&s, "gain").is_some(), "{:?}", s.ui.text_rects());
    // The slider's number takes a value.
    s.ui.apply(&mut s.viewer, &Action::SetShaderControl(0, "gain", Some(1.5.into())));
    assert_eq!(s.viewer.layers()[0].spec.json["shaderControls"]["gain"], 1.5);
    s.update();
    assert!(at(&s, "1.5").is_some());
    // A shader that does not compile: its error, the border red.
    s.ui.apply(&mut s.viewer, &Action::SetLayerOption(0, "shader".into(), Some("void main() { emitRGB(nothing); }\n".into())));
    s.update();
    let e = s.viewer.layers()[0].shader_error().expect("an error");
    assert_eq!(e.line, Some(0));
}

#[test]
fn shader_select_controls_choose_options() {
    let Some(mut s) = setup() else { return };
    s.ui.apply(&mut s.viewer, &Action::SelectLayer(0));
    s.ui.apply(&mut s.viewer, &Action::SelectLayerTab(0, "rendering"));
    let shader = "#uicontrol string_t mode select(options={\"Grayscale\": \"gray\", \"Jet\": \"jet\"})\nvoid main() {\n  float v = toNormalized(getDataValue());\n  if (mode == \"jet\") { emitRGB(colormapJet(v)); } else { emitGrayscale(v); }\n}\n";
    s.ui.apply(&mut s.viewer, &Action::SetLayerOption(0, "shader".into(), Some(shader.into())));
    s.update();
    assert_eq!(s.viewer.layers()[0].shader_error(), None);
    let at = |s: &Setup, label: &str| s.ui.text_rects().into_iter().filter(|(t, _)| t == label).map(|(_, r)| r).next_back();
    // The select shows the default's label; a click opens its options.
    let r = at(&s, "Grayscale (gray)").expect("the select");
    s.click(r.x + 5.0, r.y + 5.0, Modifiers::default());
    let r = at(&s, "Jet (jet)").expect("the open select's options");
    s.click(r.x + 5.0, r.y + 5.0, Modifiers::default());
    assert_eq!(s.viewer.layers()[0].spec.json["shaderControls"]["mode"], "jet");
    assert!(at(&s, "Jet (jet)").is_some());
    // Choosing the default again leaves it out.
    let r = at(&s, "Jet (jet)").unwrap();
    s.click(r.x + 5.0, r.y + 5.0, Modifiers::default());
    let r = at(&s, "Grayscale (gray)").expect("the options again");
    s.click(r.x + 5.0, r.y + 5.0, Modifiers::default());
    assert_eq!(s.viewer.layers()[0].spec.json.get("shaderControls"), None);
}

#[test]
fn double_clicks_select_and_star_segments() {
    let Some(mut s) = setup() else { return };
    // Loaded; then a point of the xy panel over some segment.
    let start = std::time::Instant::now();
    s.viewer.pointer_move(300.0, 200.0);
    s.update();
    while s.viewer.is_loading() && start.elapsed().as_secs() < 30 {
        s.viewer.poll();
        s.update();
    }
    let mut found = None;
    'scan: for y in (50..420).step_by(4) {
        for x in (4..596).step_by(4) {
            s.viewer.pointer_move(x as f64, y as f64);
            if let Some(Some(id)) = s.viewer.hovered_segments().get(1).copied() {
                if id != 0 {
                    found = Some(id);
                    break 'scan;
                }
            }
        }
    }
    let id = found.expect("a segment under some point");
    let segs = |s: &Setup| s.viewer.layers()[1].spec.json.get("segments").cloned();
    let toggle = |s: &mut Setup, shift: bool| {
        let a = s.ui.data_panel_double_click(Modifiers { shift, ..Default::default() }, &s.viewer).expect("a binding");
        s.ui.apply(&mut s.viewer, &a);
    };
    let before = segs(&s).unwrap();
    // Double-click: shown (and starred).
    toggle(&mut s, false);
    let mut shown = before.as_array().unwrap().clone();
    shown.push(id.to_string().into());
    assert_eq!(segs(&s), Some(serde_json::Value::Array(shown.clone())));
    // Again: hidden, still starred.
    toggle(&mut s, false);
    *shown.last_mut().unwrap() = format!("!{id}").into();
    assert_eq!(segs(&s), Some(serde_json::Value::Array(shown)));
    // Shift: unstarred.
    toggle(&mut s, true);
    assert_eq!(segs(&s), Some(before));
    // A layer that does not pick is left alone.
    s.viewer.layers_mut()[1].spec.json.insert("pick".into(), false.into());
    toggle(&mut s, false);
    assert_eq!(segs(&s).unwrap().as_array().unwrap().len(), 2);
}

#[test]
fn points_are_annotated_with_control_click() {
    let Some(mut s) = setup() else { return };
    let mut state = s.viewer.state_json();
    state["layers"].as_array_mut().unwrap().push(serde_json::json!({ "type": "annotation", "name": "ann", "source": "local://annotations" }));
    s.viewer.open_state(&state).unwrap();
    s.update();
    let control = Modifiers { control: true, ..Default::default() };
    // No layer selected: a message, nothing added.
    s.ui.pointer_move(300.0, 200.0);
    s.viewer.pointer_move(300.0, 200.0);
    let a = s.ui.data_panel_press(Button::Primary, control, &s.viewer).expect("annotate");
    assert_eq!(a, Action::Annotate);
    s.ui.apply(&mut s.viewer, &a);
    assert_eq!(s.viewer.layers()[2].spec.json.get("annotations"), None);
    // Selected, without a tool: still nothing; with the point tool, a point
    // at the pointer.
    s.ui.apply(&mut s.viewer, &Action::SelectLayer(2));
    s.ui.apply(&mut s.viewer, &Action::Annotate);
    assert_eq!(s.viewer.layers()[2].spec.json.get("annotations"), None);
    s.ui.apply(&mut s.viewer, &Action::SetLayerOption(2, "tool".into(), Some("annotatePoint".into())));
    s.ui.apply(&mut s.viewer, &Action::Annotate);
    let list = s.viewer.layers()[2].spec.json["annotations"].as_array().unwrap().clone();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["type"], "point");
    assert_eq!(list[0]["id"].as_str().unwrap().len(), 40);
    let point: Vec<f64> = list[0]["point"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let mouse = s.viewer.mouse_position().unwrap();
    for (p, m) in point.iter().zip(&mouse) {
        assert!((p - *m as f64).abs() < 1e-3, "{point:?} vs {mouse:?}");
    }
    assert_eq!(s.viewer.layers()[2].annotation_list(3).unwrap().len(), 1);
    // Hovered in the panel (picked within the pick radius): highlighted,
    // its list entry too; not once the pointer is away. (The side panel
    // has opened: the view, narrower, keeps its centre, now at x 224.)
    let id = list[0]["id"].as_str().unwrap().to_owned();
    s.ui.apply(&mut s.viewer, &Action::SetLayerOption(2, "tool".into(), None));
    for (x, hovered) in [(224.0, true), (228.0, true), (260.0, false)] {
        s.ui.pointer_move(x, 200.0);
        s.viewer.pointer_move(x as f64, 200.0);
        s.update();
        s.update();
        assert_eq!(s.viewer.annotation_hover(), hovered.then(|| (2, id.clone())), "at x {x}");
    }
    // Alt+drag (`move-annotation`) moves it with the pointer, 10 pixels
    // right: along x only.
    s.ui.pointer_move(224.0, 200.0);
    s.viewer.pointer_move(224.0, 200.0);
    s.update();
    let alt = Modifiers { alt: true, ..Default::default() };
    s.viewer.pointer_down(Button::Primary, alt);
    s.viewer.pointer_move(234.0, 200.0);
    s.viewer.pointer_up(Button::Primary);
    let moved: Vec<f64> = s.viewer.layers()[2].spec.json["annotations"][0]["point"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert!(moved[0] > point[0] + 1.0 && (moved[1] - point[1]).abs() < 1e-3 && moved[2] == point[2], "{point:?} → {moved:?}");
    // Ctrl+Alt+right-click (`delete-annotation`) over it deletes it; put
    // back for what follows.
    let kept = s.viewer.layers()[2].spec.json["annotations"].clone();
    s.update();
    s.update();
    assert!(s.viewer.picked_annotation().is_some(), "still under the pointer once moved");
    s.viewer.pointer_down(Button::Secondary, Modifiers { control: true, alt: true, ..Default::default() });
    assert_eq!(s.viewer.layers()[2].spec.json["annotations"].as_array().map(Vec::len), Some(0));
    s.ui.apply(&mut s.viewer, &Action::SetLayerOption(2, "annotations".into(), Some(kept)));
    s.ui.apply(&mut s.viewer, &Action::SetLayerOption(2, "tool".into(), Some("annotatePoint".into())));
    // A box: the first corner, the pointer moves, the second; the corners
    // differ in every dimension (z one more).
    s.ui.apply(&mut s.viewer, &Action::SetLayerOption(2, "tool".into(), Some("annotateBoundingBox".into())));
    s.ui.apply(&mut s.viewer, &Action::Annotate);
    s.viewer.pointer_move(340.0, 240.0);
    s.update();
    s.ui.apply(&mut s.viewer, &Action::Annotate);
    let list = s.viewer.layers()[2].spec.json["annotations"].as_array().unwrap().clone();
    assert_eq!(list.len(), 2);
    let a: Vec<f64> = list[1]["pointA"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let b: Vec<f64> = list[1]["pointB"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert!(b[0] > a[0] && b[1] > a[1], "{a:?} {b:?}");
    assert_eq!(b[2], ((a[2] as f32) + 1.0) as f64);
    // Moving on changes nothing once finished.
    s.viewer.pointer_move(100.0, 100.0);
    s.update();
    assert_eq!(s.viewer.layers()[2].spec.json["annotations"][1], list[1]);
    // The list: right-click moves to the box's middle; pinning selects it;
    // delete removes it.
    s.ui.apply(&mut s.viewer, &Action::MoveToAnnotation(2, 1));
    let p = s.position();
    assert!(((p[0] as f64) - (a[0] + b[0]) / 2.0).abs() < 1e-3, "{p:?}");
    s.ui.apply(&mut s.viewer, &Action::PinAnnotation(2, 0));
    let id0 = list[0]["id"].as_str().unwrap().to_owned();
    assert_eq!(s.viewer.ui_state("selection").unwrap()["layers"]["ann"]["annotationId"], serde_json::json!(id0));
    s.ui.apply(&mut s.viewer, &Action::DeleteAnnotation(2, id0));
    assert_eq!(s.viewer.layers()[2].spec.json["annotations"].as_array().unwrap().len(), 1);
    // A polyline: three clicks, then Enter keeps those three points.
    s.ui.apply(&mut s.viewer, &Action::SetLayerOption(2, "tool".into(), Some("annotatePolyline".into())));
    for (x, y) in [(250.0, 150.0), (300.0, 200.0), (350.0, 150.0)] {
        s.ui.pointer_move(x, y);
        s.viewer.pointer_move(x as f64, y as f64);
        s.update();
        s.ui.apply(&mut s.viewer, &Action::Annotate);
    }
    let enter = s.ui.shortcut("Enter", Modifiers::default(), &s.viewer).expect("finish-annotation");
    assert_eq!(enter, Action::AnnotationStep { finish: true });
    s.ui.apply(&mut s.viewer, &enter);
    let polyline = s.viewer.layers()[2].spec.json["annotations"][1].clone();
    assert_eq!(polyline["type"], "polyline");
    assert_eq!(polyline["points"].as_array().unwrap().len(), 3, "{polyline}");
    // Done: moving no longer changes it.
    s.viewer.pointer_move(100.0, 300.0);
    s.update();
    assert_eq!(s.viewer.layers()[2].spec.json["annotations"][1], polyline);
    // Prev/Next as bound tools: a step through the list each, moving there.
    use ng_viewer::tools::ToolOwner;
    s.ui.apply(&mut s.viewer, &Action::BindTool('N', ToolOwner::Layer(2), "selectNextAnnotation".into()));
    s.ui.apply(&mut s.viewer, &Action::BindTool('P', ToolOwner::Layer(2), "selectPreviousAnnotation".into()));
    s.ui.apply(&mut s.viewer, &Action::PinAnnotation(2, 0));
    let shift = Modifiers { shift: true, ..Default::default() };
    let selected = |s: &Setup| s.viewer.ui_state("selection").unwrap()["layers"]["ann"]["annotationId"].clone();
    let ids: Vec<serde_json::Value> = s.viewer.layers()[2].spec.json["annotations"].as_array().unwrap().iter().map(|a| a["id"].clone()).collect();
    for (code, expected) in [("KeyN", 1), ("KeyN", 1), ("KeyP", 0)] {
        let a = s.ui.shortcut(code, shift, &s.viewer).expect("tool");
        s.ui.apply(&mut s.viewer, &a);
        s.ui.key_up(code);
        assert_eq!(selected(&s), ids[expected], "{code}");
    }
}
