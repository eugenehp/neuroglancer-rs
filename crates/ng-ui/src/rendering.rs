//! An image layer's Rendering tab (`layer/image/index.ts`
//! `RenderingOptionsTab`): its layer controls (`widget/layer_control*.ts`,
//! `widget/render_scale_widget.ts`), the shader editor
//! (`widget/shader_code_widget.ts`, CodeMirror with
//! `third_party/codemirror-glsl.ts`) and the shader's controls
//! (`widget/shader_controls.ts`, `widget/invlerp.ts`).

use ng_core::data_type::DataType;
use ng_core::js::number_to_string;
use ng_core::render_scale::{RenderScaleHistogram, NUM_BINS};
use ng_viewer::{Interval, Viewer};
use threers::style;
use threers::ui::{controls, Edges, Length, NodeId};

use crate::panels::scroll_style;
use crate::view::Builder;
use crate::{icons, Action, Key, NgUi};

/// `10pt`.
use crate::tools::ToolRef;

const SMALL: f32 = 40.0 / 3.0;

/// `blendMode` options (`BLEND_MODES`).
pub(crate) const BLEND_MODES: [&str; 2] = ["default", "additive"];
/// `volumeRendering` options (`VolumeRenderingModes`).
pub(crate) const VOLUME_RENDERING_MODES: [&str; 4] = ["off", "on", "max", "min"];

impl NgUi {
    /// `RenderingOptionsTab` of image layer `i`.
    pub(crate) fn image_rendering_tab(&mut self, b: &mut Builder, content: NodeId, viewer: &mut Viewer, i: usize) {
        let scroll = b.ui.add_box(content, scroll_style(self.scroll.get(&self.layer_panel).copied().unwrap_or(0.0)));
        b.key(scroll, Key::Scroll(self.layer_panel), None);
        let layer = &viewer.layers()[i];
        let json = &layer.spec.json;

        // Resolution (slice).
        let row = control_row(b, scroll, "Resolution (slice)", ToolRef::layer(i, "crossSectionRenderScale"));
        render_scale_widget(b, row, i, "renderScale", layer.render_scale_histogram(), layer.render_scale_target(), "px");
        b.controls.insert(Key::Control(i, "renderScale"), Control::RenderScale { layer: i, key: "crossSectionRenderScale", target: layer.render_scale_target(), origin: ng_core::render_scale::ORIGIN, invert_wheel: false });

        // Blending (slice).
        let row = control_row(b, scroll, "Blending (slice)", ToolRef::layer(i, "blend"));
        let blend = json.get("blend").and_then(serde_json::Value::as_str).filter(|m| BLEND_MODES.contains(m)).unwrap_or("default");
        select(b, row, blend, &BLEND_MODES, Key::Control(i, "blend"), Action::Repaint);
        b.controls.insert(Key::Control(i, "blend"), Control::Select { layer: i, key: "blend", options: &BLEND_MODES, value: BLEND_MODES.iter().find(|m| **m == blend).unwrap(), default: "default" });

        // Opacity (slice).
        let row = control_row(b, scroll, "Opacity (slice)", ToolRef::layer(i, "opacity"));
        let opacity = json.get("opacity").and_then(serde_json::Value::as_f64).filter(|v| (0.0..=1.0).contains(v)).unwrap_or(0.5);
        range_slider(self, b, row, i, opacity, SliderSpec::alpha(0.5), Key::Control(i, "opacity"));
        b.controls.insert(Key::Control(i, "opacity"), Control::Slider { layer: i, key: "opacity", value: opacity, spec: SliderSpec::alpha(0.5) });

        // Volume rendering (experimental).
        let row = control_row(b, scroll, "Volume rendering (experimental)", ToolRef::layer(i, "volumeRendering"));
        let mode = json.get("volumeRendering").and_then(|v| match v {
            serde_json::Value::Bool(true) => Some("on"),
            serde_json::Value::Bool(false) => Some("off"),
            serde_json::Value::String(s) => VOLUME_RENDERING_MODES.iter().copied().find(|m| m == s),
            _ => None,
        });
        let mode = mode.unwrap_or("off");
        select(b, row, mode, &VOLUME_RENDERING_MODES, Key::Control(i, "volumeRendering"), Action::Repaint);
        b.controls.insert(Key::Control(i, "volumeRendering"), Control::Select { layer: i, key: "volumeRendering", options: &VOLUME_RENDERING_MODES, value: mode, default: "off" });
        // `Gain (3D)` while on; `Resolution (3D)` (depth samples) unless off.
        let (_, samples, gain) = ng_viewer::state::volume_rendering_settings(json);
        if mode == "on" {
            let row = control_row(b, scroll, "Gain (3D)", ToolRef::layer(i, "volumeRenderingGain"));
            let spec = SliderSpec { min: -10.0, max: 10.0, step: 0.1, default: 0.0 };
            range_slider(self, b, row, i, gain, spec, Key::Control(i, "volumeRenderingGain"));
            b.controls.insert(Key::Control(i, "volumeRenderingGain"), Control::Slider { layer: i, key: "volumeRenderingGain", value: gain, spec });
        }
        if mode != "off" {
            let row = control_row(b, scroll, "Resolution (3D)", ToolRef::layer(i, "volumeRenderingDepthSamples"));
            let layer = &viewer.layers()[i];
            render_scale_widget(b, row, i, "volumeRenderingDepthSamples", layer.volume_rendering_histogram(), samples, "samples");
            b.controls.insert(
                Key::Control(i, "volumeRenderingDepthSamples"),
                Control::RenderScale { layer: i, key: "volumeRenderingDepthSamples", target: samples, origin: ng_core::render_scale::VOLUME_RENDERING_ORIGIN, invert_wheel: true },
            );
        }

        // The shader's top row: label, show/hide, maximize, help.
        let top = b.ui.add_box(scroll, style! { display: flex; align-items: center; flex-shrink: 0; });
        b.text_box(top, "Shader", style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
        b.ui.add_box(top, style! { flex: 1; });
        let code_visible = json.get("codeVisible").and_then(serde_json::Value::as_bool).unwrap_or(true);
        // `codeVisible` (a `TrackableBoolean`, true unless said).
        let toggle_code = Action::SetLayerOption(i, "codeVisible".into(), code_visible.then_some(serde_json::Value::Bool(false)));
        let toggle = b.icon(top, 2700, Some(icons::CODE_ALT), "", Edges::symmetric(0.0, 1.0), toggle_code);
        b.title(Key::Icon(2700), if code_visible { "Hide code" } else { "Show code" });
        if code_visible {
            // `.neuroglancer-checkbox-icon.dark-background[data-checked]`,
            // which outranks `:hover`.
            b.ui.style_mut(toggle).background = Some([1.0, 1.0, 1.0, 0.2]);
        }
        b.icon(top, 2701, Some(icons::MAXIMISE), "", Edges::symmetric(0.0, 1.0), Action::OpenShaderOverlay(i));
        b.title(Key::Icon(2701), "Show larger editor view");
        b.icon(top, 2702, None, "?", Edges::symmetric(0.0, 1.0), Action::OpenUrl(IMAGE_RENDERING_DOCS.into()));
        b.title(Key::Icon(2702), "Documentation on image layer rendering");

        // Channel dimensions (`ChannelDimensionsWidget`).
        let channels = viewer.layers()[i].channel_dimensions();
        if !channels.is_empty() {
            channel_dimensions_widget(b, scroll, &channels);
        }

        // The shader code.
        let shader = json.get("shader").and_then(serde_json::Value::as_str).unwrap_or(ng_viewer::state::DEFAULT_IMAGE_SHADER).to_owned();
        if code_visible {
            self.shader_editor(b, scroll, viewer, i);
        }

        // The shader's controls.
        let hide = b.ui.add_box(scroll, style! { display: flex; align-items: center; font-family: sans-serif; font-size: {SMALL / 1.2}; color: white; flex-shrink: 0; height: 19px; });
        // `hideInactiveShaderControls`: no control is inactive here (the
        // shader is not compiled to find out), so none hides.
        let hide_inactive = json.get("hideInactiveShaderControls").and_then(serde_json::Value::as_bool).unwrap_or(false);
        let toggle = Action::SetLayerOption(i, "hideInactiveShaderControls".into(), (!hide_inactive).then_some(serde_json::Value::Bool(true)));
        light_checkbox(b, hide, hide_inactive, Key::Control(i, "hideInactive"), toggle.clone());
        b.key(hide, Key::Control(i, "hideInactiveRow"), Some(toggle));
        b.title(Key::Control(i, "hideInactiveRow"), "Hide #uicontrols that have no effect on the current render (their uniforms were eliminated by the GLSL compiler, e.g. controls only referenced inside an `if (checkbox)` branch that's currently false).");
        b.text_box(hide, "Hide inactive shader controls", style! { font-family: sans-serif; font-size: {SMALL / 1.2}; color: white; margin-left: 4px; });

        // Another shader's own controls.
        if shader != ng_viewer::state::DEFAULT_IMAGE_SHADER {
            self.shader_controls(b, scroll, viewer, i);
            return;
        }
        let layer = &viewer.layers()[i];
        let (Some(t), Some(range), Some(window)) = (layer.data_type(), layer.range(), layer.invlerp_window()) else { return };
        let json = json.clone();
        viewer.want_data_histogram(i, true);
        let histogram = viewer.layers()[i].data_histogram().copied();
        let row = control_row(b, scroll, "normalized", ToolRef::layer(i, serde_json::json!({ "type": "shaderControl", "control": "normalized" })));
        b.key(row, Key::Control(i, "invlerpRow"), None);
        invlerp_widget(self, b, row, i, &json, t, range, window, histogram.as_ref(), InvlerpCtl::normalized(t), true);
    }
}

/// `src/annotation/rendering.md`.
const ANNOTATION_RENDERING_DOCS: &str = "https://github.com/google/neuroglancer/blob/master/src/annotation/rendering.md";

impl NgUi {
    /// An annotation layer's Rendering tab (`layer/annotation`
    /// `RenderingOptionsTab`): the shader's top row and code (CodeMirror's
    /// 300 px), its controls, the annotation colour.
    pub(crate) fn annotation_rendering_tab(&mut self, b: &mut Builder, content: NodeId, viewer: &mut Viewer, i: usize) {
        let scroll = b.ui.add_box(content, scroll_style(self.scroll.get(&self.layer_panel).copied().unwrap_or(0.0)));
        b.key(scroll, Key::Scroll(self.layer_panel), None);
        let json = viewer.layers()[i].spec.json.clone();
        let code_visible = json.get("codeVisible").and_then(serde_json::Value::as_bool).unwrap_or(true);
        // `buildShaderPropertyList`: each property's type and accessor,
        // while the code shows.
        let properties = viewer.layers()[i].annotation_properties();
        if code_visible && !properties.is_empty() {
            let list = b.ui.add_box(scroll, style! { display: flex; flex-direction: column; max-height: {8.0 * SMALL}; overflow: hidden; flex-shrink: 0; });
            // `font-size: medium` monospace: Chrome's 13 px.
            let ch = b.ch("monospace", 13.0);
            for (k, (id, type_name, description)) in properties.iter().enumerate() {
                let line = b.ui.add_box(list, style! { display: flex; flex-shrink: 0; });
                b.text_box(line, type_name, style! { font-family: monospace; font-size: 13px; color: #c6c; margin-right: {ch}; white-space: pre; });
                b.text_box(line, &format!("prop_{id}"), style! { font-family: monospace; font-size: 13px; color: white; white-space: pre; });
                b.text_box(line, "()", style! { font-family: monospace; font-size: 13px; color: #999; white-space: pre; });
                if let Some(d) = description {
                    let key = Key::Control(i, crate::shader_ui::intern(&format!("shaderProperty{k}")));
                    b.key(line, key, None);
                    b.title(key, d);
                }
            }
        }
        self.shader_top_row(b, scroll, &json, i, ANNOTATION_RENDERING_DOCS);
        if code_visible {
            self.shader_editor(b, scroll, viewer, i);
        }
        let hide_inactive = json.get("hideInactiveShaderControls").and_then(serde_json::Value::as_bool).unwrap_or(false);
        self.hide_inactive_row(b, scroll, &json, i, "hideInactiveShaderControls", hide_inactive);
        self.shader_controls(b, scroll, viewer, i);

        // `LAYER_CONTROLS.annotationColor`: only while the shader (its
        // code without the directives) uses `defaultColor`.
        let code = ng_viewer::user_shader::parse(&crate::shader_ui::shader_text(viewer, i), ng_core::DataType::Float32).code;
        let uses_default = code.match_indices("defaultColor").any(|(at, m)| {
            let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
            !code[..at].ends_with(word) && !code[at + m.len()..].starts_with(word)
        });
        if !uses_default {
            return;
        }
        let row = control_row(b, scroll, "Annotation color", ToolRef::layer(i, "annotationColor"));
        b.ui.add_box(row, style! { flex: 1; });
        let color = viewer.layers()[i].annotation_color();
        let well = controls::color_well(&mut b.ui, row, color, 50.0);
        b.ui.style_mut(well).margin.left = 5.0;
        b.key(well, Key::Color(crate::ColorSetting::Annotation(i)), Some(Action::PickColor(crate::ColorSetting::Annotation(i))));
    }
}

impl NgUi {
    /// `makeShaderCodeWidgetTopRow`: "Shader", then the code visibility
    /// toggle, the larger editor and the help (`docs`).
    fn shader_top_row(&mut self, b: &mut Builder, parent: NodeId, json: &serde_json::Map<String, serde_json::Value>, i: usize, docs: &str) {
        let top = b.ui.add_box(parent, style! { display: flex; align-items: center; flex-shrink: 0; });
        b.text_box(top, "Shader", style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
        b.ui.add_box(top, style! { flex: 1; });
        let code_visible = json.get("codeVisible").and_then(serde_json::Value::as_bool).unwrap_or(true);
        let toggle_code = Action::SetLayerOption(i, "codeVisible".into(), code_visible.then_some(serde_json::Value::Bool(false)));
        let toggle = b.icon(top, 2700, Some(icons::CODE_ALT), "", Edges::symmetric(0.0, 1.0), toggle_code);
        b.title(Key::Icon(2700), if code_visible { "Hide code" } else { "Show code" });
        if code_visible {
            b.ui.style_mut(toggle).background = Some([1.0, 1.0, 1.0, 0.2]);
        }
        b.icon(top, 2701, Some(icons::MAXIMISE), "", Edges::symmetric(0.0, 1.0), Action::OpenShaderOverlay(i));
        b.title(Key::Icon(2701), "Show larger editor view");
        b.icon(top, 2702, None, "?", Edges::symmetric(0.0, 1.0), Action::OpenUrl(docs.into()));
        b.title(Key::Icon(2702), "Documentation on image layer rendering");
    }

    /// `ShaderControls`' "Hide inactive shader controls" checkbox, setting
    /// `key`.
    fn hide_inactive_row(&mut self, b: &mut Builder, parent: NodeId, _json: &serde_json::Map<String, serde_json::Value>, i: usize, key: &str, hide_inactive: bool) {
        let hide = b.ui.add_box(parent, style! { display: flex; align-items: center; font-family: sans-serif; font-size: {SMALL / 1.2}; color: white; flex-shrink: 0; height: 19px; });
        let toggle = Action::SetLayerOption(i, key.into(), (!hide_inactive).then_some(serde_json::Value::Bool(true)));
        light_checkbox(b, hide, hide_inactive, Key::Control(i, "hideInactive"), toggle.clone());
        b.key(hide, Key::Control(i, "hideInactiveRow"), Some(toggle));
        b.title(Key::Control(i, "hideInactiveRow"), "Hide #uicontrols that have no effect on the current render (their uniforms were eliminated by the GLSL compiler, e.g. controls only referenced inside an `if (checkbox)` branch that's currently false).");
        b.text_box(hide, "Hide inactive shader controls", style! { font-family: sans-serif; font-size: {SMALL / 1.2}; color: white; margin-left: 4px; });
    }
}

/// The segmentation layer checkboxes (`checkboxLayerControl`): label, JSON
/// member, default.
pub(crate) const SEGMENTATION_CHECKBOXES: [(&str, &str, bool); 4] = [
    ("Hide segment ID 0", "hideSegmentZero", true),
    ("Base segment coloring", "baseSegmentColoring", false),
    ("Show all by default", "ignoreNullVisibleSet", true),
    ("Highlight on hover", "hoverHighlight", true),
];

impl NgUi {
    /// A segmentation layer's Render tab (`DisplayOptionsTab`, its
    /// `LAYER_CONTROLS` for 2D data).
    pub(crate) fn segmentation_rendering_tab(&mut self, b: &mut Builder, content: NodeId, viewer: &mut Viewer, i: usize) {
        let scroll = b.ui.add_box(content, scroll_style(self.scroll.get(&self.layer_panel).copied().unwrap_or(0.0)));
        b.key(scroll, Key::Scroll(self.layer_panel), None);
        let layer = &viewer.layers()[i];
        let json = layer.spec.json.clone();
        let Some(ng_viewer::LayerKind::Segmentation(display)) = layer.kind().cloned() else { return };
        let fixed = display.default_color;

        // Color seed: a radio in the label; the seed and randomize, hidden
        // while a fixed colour is used.
        let row = control_row(b, scroll, "Color seed", ToolRef::layer(i, "colorSeed"));
        label_radio(b, row, fixed.is_none(), Key::Control(i, "colorSeedMode"), Action::SetLayerOption(i, "segmentDefaultColor".into(), None));
        let control = b.ui.add_box(row, style! { display: flex; align-items: center; justify-content: flex-end; flex: 1; max-width: 500px; margin-left: 5px; });
        let input = b.ui.add_box(control, style! { display: flex; position: relative; align-items: center; width: 149px; height: 17px; padding: 1px 2px; box-sizing: border-box; background: {[1.0, 1.0, 1.0, 0.3]}; flex-shrink: 0; });
        let id = crate::FieldId::ColorSeed(i);
        let editing = self.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t);
        let font = style! { font-family: Arial; font-size: {SMALL}; color: white; };
        self.text_contents(b, input, editing, &display.color_seed.to_string(), &font, 0.0);
        let icon = b.icon(control, 2900, Some(icons::ROTATE), "", Edges::symmetric(0.0, 1.0), Action::RandomizeSeed(i));
        if fixed.is_some() {
            b.ui.style_mut(control).opacity = 0.0;
            b.keys.remove(&icon);
        } else {
            b.key(input, Key::Field(id), None);
        }

        // Fixed color: a radio; the colour, hidden while hashing.
        let row = control_row(b, scroll, "Fixed color", ToolRef::layer(i, "segmentDefaultColor"));
        let choose = Action::Many(vec![
            Action::SetLayerOption(i, "segmentDefaultColor".into(), Some("#ff0000".into())),
            Action::PickColor(crate::ColorSetting::SegmentDefault(i)),
        ]);
        label_radio(b, row, fixed.is_some(), Key::Control(i, "fixedColorMode"), choose);
        b.ui.add_box(row, style! { flex: 1; });
        let well = threers::ui::controls::color_well(&mut b.ui, row, fixed.unwrap_or([1.0, 0.0, 0.0]), 50.0);
        b.ui.style_mut(well).margin.left = 5.0;
        if fixed.is_some() {
            b.key(well, Key::Color(crate::ColorSetting::SegmentDefault(i)), Some(Action::PickColor(crate::ColorSetting::SegmentDefault(i))));
        } else {
            b.ui.style_mut(well).opacity = 0.0;
        }

        for (label, key, default) in [("Saturation", "saturation", 1.0), ("Opacity (on)", "selectedAlpha", 0.5), ("Opacity (off)", "notSelectedAlpha", 0.0)] {
            let row = control_row(b, scroll, label, ToolRef::layer(i, key));
            // As stated (the display's copies are single precision).
            let value = json.get(key).and_then(serde_json::Value::as_f64).filter(|v| (0.0..=1.0).contains(v)).unwrap_or(default);
            range_slider(self, b, row, i, value, SliderSpec::alpha(default), Key::Control(i, key));
            b.controls.insert(Key::Control(i, key), Control::Slider { layer: i, key, value, spec: SliderSpec::alpha(default) });
        }

        let row = control_row(b, scroll, "Resolution (slice)", ToolRef::layer(i, "crossSectionRenderScale"));
        let layer = &viewer.layers()[i];
        render_scale_widget(b, row, i, "renderScale", layer.render_scale_histogram(), layer.render_scale_target(), "px");
        b.controls.insert(Key::Control(i, "renderScale"), Control::RenderScale { layer: i, key: "crossSectionRenderScale", target: layer.render_scale_target(), origin: ng_core::render_scale::ORIGIN, invert_wheel: false });

        // `has3dLayer`: the meshes' and skeletons' controls.
        let (has_3d, has_skeletons) = (layer.has_3d(), layer.has_skeletons());
        if has_3d {
            let row = control_row(b, scroll, "Resolution (mesh)", ToolRef::layer(i, "meshRenderScale"));
            render_scale_widget(b, row, i, "meshRenderScale", layer.mesh_render_scale_histogram(), layer.mesh_render_scale_target(), "px");
            b.controls.insert(Key::Control(i, "meshRenderScale"), Control::RenderScale { layer: i, key: "meshRenderScale", target: layer.mesh_render_scale_target(), origin: ng_core::render_scale::ORIGIN, invert_wheel: false });
            let silhouette = layer.silhouette_rendering();
            // `maxSilhouettePower`: 10, in steps of 0.1.
            for (label, key, value, spec) in [
                ("Opacity (3d)", "objectAlpha", json.get("objectAlpha").and_then(serde_json::Value::as_f64).filter(|v| (0.0..=1.0).contains(v)).unwrap_or(1.0), SliderSpec::alpha(1.0)),
                ("Silhouette (3d)", "meshSilhouetteRendering", silhouette, SliderSpec { min: 0.0, max: 10.0, step: 0.1, default: 0.0 }),
            ] {
                let row = control_row(b, scroll, label, ToolRef::layer(i, key));
                range_slider(self, b, row, i, value, spec, Key::Control(i, key));
                b.controls.insert(Key::Control(i, key), Control::Slider { layer: i, key, value, spec });
            }
        }

        for (label, key, default) in SEGMENTATION_CHECKBOXES {
            let row = control_row(b, scroll, label, ToolRef::layer(i, key));
            b.ui.add_box(row, style! { flex: 1; });
            let on = json.get(key).and_then(serde_json::Value::as_bool).unwrap_or(default);
            // `TrackableBoolean`: the default is left out.
            let value = (!on != default).then_some(serde_json::Value::Bool(!on));
            let node = threers::ui::controls::checkbox(&mut b.ui, row, on, 13.0, threers::ui::controls::BORDER, Edges::new(3.0, 3.0, 3.0, 5.0));
            b.key(node, Key::Control(i, key), Some(Action::SetLayerOption(i, key.into(), value.clone())));
            // The whole `<label>` toggles it.
            b.key(row, Key::Control(i, "checkboxRow"), Some(Action::SetLayerOption(i, key.into(), value)));
        }
        if has_skeletons {
            self.skeleton_controls(b, scroll, viewer, i);
        }
    }

    /// `hasSkeletonsLayer`: the view-specific skeleton options
    /// (`getViewSpecificSkeletonRenderingControl`), the vertex attributes
    /// (`buildShaderPropertyList`), the skeleton shader and its controls.
    fn skeleton_controls(&mut self, b: &mut Builder, scroll: NodeId, viewer: &mut Viewer, i: usize) {
        let json = viewer.layers()[i].spec.json.clone();
        let options = json.get("skeletonRendering").and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
        for (view, (mode_default, width_default)) in [("2d", ("lines_and_points", 2.0)), ("3d", ("lines", 1.0))] {
            let mode_key: &'static str = if view == "2d" { "skeletonRendering.mode2d" } else { "skeletonRendering.mode3d" };
            let width_key: &'static str = if view == "2d" { "skeletonRendering.lineWidth2d" } else { "skeletonRendering.lineWidth3d" };
            let row = control_row(b, scroll, &format!("Skeleton mode ({view})"), ToolRef::layer(i, mode_key));
            let mode = options
                .get(&format!("mode{view}"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_ascii_lowercase)
                .and_then(|m| SKELETON_MODES.iter().copied().find(|o| *o == m))
                .unwrap_or(mode_default);
            select(b, row, mode, &SKELETON_MODES, Key::Control(i, mode_key), Action::Repaint);
            b.controls.insert(Key::Control(i, mode_key), Control::Select { layer: i, key: mode_key, options: &SKELETON_MODES, value: mode, default: mode_default });
            let row = control_row(b, scroll, &format!("Line width ({view})"), ToolRef::layer(i, width_key));
            let spec = SliderSpec { min: 1.0, max: 40.0, step: 1.0, default: width_default };
            let width = options.get(&format!("lineWidth{view}")).and_then(serde_json::Value::as_f64).filter(|w| w.is_finite() && *w > 0.0).unwrap_or(width_default);
            range_slider(self, b, row, i, width, spec, Key::Control(i, width_key));
            b.controls.insert(Key::Control(i, width_key), Control::Slider { layer: i, key: width_key, value: width, spec });
        }
        // `.neuroglancer-annotation-shader-property-list`: monospace,
        // `medium`; the type `#c6c` then a space, the name, `()` in `#999`.
        let attributes = viewer.layers()[i].skeleton_vertex_attributes();
        if !attributes.is_empty() {
            let list = b.ui.add_box(scroll, style! { display: flex; flex-direction: column; max-height: {8.0 * 13.0}; overflow: hidden; flex-shrink: 0; });
            for (t, name) in attributes {
                let row = b.ui.add_box(list, style! { display: flex; flex-shrink: 0; });
                b.text_box(row, &t, style! { font-family: monospace; font-size: 13px; color: #c6c; white-space: pre; });
                b.text_box(row, " ", style! { font-family: monospace; font-size: 13px; color: white; white-space: pre; });
                b.text_box(row, &format!("prop_{name}"), style! { font-family: monospace; font-size: 13px; color: white; white-space: pre; });
                b.text_box(row, "()", style! { font-family: monospace; font-size: 13px; color: #999; white-space: pre; });
            }
        }
        self.shader_top_row(b, scroll, &json, i, IMAGE_RENDERING_DOCS);
        if json.get("codeVisible").and_then(serde_json::Value::as_bool).unwrap_or(true) {
            self.shader_editor(b, scroll, viewer, i);
        }
        self.hide_inactive_row(b, scroll, &json, i, "skeletonRendering.hideInactiveShaderControls", options.get("hideInactiveShaderControls").and_then(serde_json::Value::as_bool).unwrap_or(false));
    }
}

/// `SkeletonRenderMode`'s options, as `EnumSelectWidget` lists them.
pub(crate) const SKELETON_MODES: [&str; 2] = ["lines", "lines_and_points"];

/// A radio button at the start of a control's label
/// (`labelTextContainer.prepend`).
fn label_radio(b: &mut Builder, row: NodeId, checked: bool, key: Key, action: Action) {
    let label = b.ui.children(row)[0];
    let node = threers::ui::controls::radio(&mut b.ui, label, checked, Edges::new(3.0, 3.0, 0.0, 5.0));
    // After the key binding, before the text.
    let n = b.ui.children(label).len();
    b.ui.move_child(label, node, n - 2);
    b.ui.style_mut(label).align_items = threers::ui::AlignItems::FlexStart;
    b.key(row, key, Some(action));
}

/// `.neuroglancer-layer-control-container`: the label, with its tool key
/// binding (`[ ]` until bound), then the control the caller adds.
pub(crate) fn control_row_pub(b: &mut Builder, parent: NodeId, label: &str, tool: ToolRef) -> NodeId {
    control_row(b, parent, label, tool)
}

/// [`control_row_pub`] whose label container takes more below the label
/// (`options.labelContainer`): the row, and that column.
pub(crate) fn control_row_with_label_column(b: &mut Builder, parent: NodeId, label: &str, tool: ToolRef) -> (NodeId, NodeId) {
    let row = b.ui.add_box(parent, style! { display: flex; align-items: center; margin: 1.5px 0; flex-shrink: 0; });
    let column = b.ui.add_box(row, style! { display: flex; flex-direction: column; flex-shrink: 0; });
    let l = b.ui.add_box(column, style! { display: flex; align-items: center; flex-shrink: 0; });
    b.tool_binding(l, tool, None);
    b.text_box(l, label, style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
    (row, column)
}

/// `ChannelDimensionsWidget`: each channel dimension's name (`#ff6`) and
/// bounds (`[lower,` `upper)`, the brackets grey), a column each.
pub(crate) fn channel_dimensions_widget(b: &mut Builder, parent: NodeId, dims: &[(String, u32)]) {
    let ch = b.ch("monospace", 13.0);
    let grid = b.ui.add_box(parent, style! { display: flex; flex-direction: column; flex-shrink: 0; });
    let name_w = dims.iter().map(|(n, _)| n.chars().count().max(1)).max().unwrap_or(1);
    let upper_w = dims.iter().map(|(_, e)| e.to_string().len()).max().unwrap_or(1);
    let mono = |color: &str| style! { font-family: monospace; font-size: 13px; color: {color}; white-space: pre; };
    for (name, extent) in dims {
        let row = b.ui.add_box(grid, style! { display: flex; align-items: center; flex-shrink: 0; });
        crate::view::input(b, row, name, name_w, "#ff6", ch, false, None);
        let lower = b.ui.add_box(row, style! { display: flex; margin-left: {crate::view::lu(ch)}; flex-shrink: 0; });
        b.text_box(lower, "[", mono("#999"));
        b.text_box(lower, "0", mono("white"));
        b.text_box(lower, ",", mono("#999"));
        let upper = b.ui.add_box(row, style! { display: flex; justify-content: flex-end; width: {crate::view::lu((upper_w + 1) as f32 * ch)}; margin-left: {crate::view::lu(ch)}; flex-shrink: 0; });
        b.text_box(upper, &extent.to_string(), mono("white"));
        b.text_box(upper, ")", mono("#999"));
    }
}

fn control_row(b: &mut Builder, parent: NodeId, label: &str, tool: ToolRef) -> NodeId {
    let row = b.ui.add_box(parent, style! { display: flex; align-items: center; margin: 1.5px 0; flex-shrink: 0; });
    let l = b.ui.add_box(row, style! { display: flex; align-items: center; flex-shrink: 0; });
    b.tool_binding(l, tool, None);
    b.text_box(l, label, style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
    row
}

/// Layer controls drawn in the status bar (an active tool's) or a tool
/// palette are keyed apart from the layer panel's: their layer index with
/// a slot above [`LAYER_MASK`].
pub(crate) const STATUS_SLOT: usize = 1 << 20;
pub(crate) const LAYER_MASK: usize = STATUS_SLOT - 1;

/// The slot of palette `k`'s controls.
pub(crate) fn palette_slot(k: u16) -> usize {
    (2 + k as usize) << 20
}

impl NgUi {
    /// An active layer control tool's control in the status bar
    /// (`LayerControlTool.activate`): `.neuroglancer-layer-control-container`
    /// in `body`; returns its label container (for the header) and the
    /// input the tool takes.
    pub(crate) fn layer_control(&mut self, b: &mut Builder, body: NodeId, viewer: &mut Viewer, i: usize, tool: &ToolRef, slot: usize) -> Option<(NodeId, crate::tools::ControlKind)> {
        use crate::tools::ControlKind;
        let slot = i | slot;
        let layer = viewer.layers().get(i)?;
        let json = layer.spec.json.clone();
        let id = tool.id().to_owned();
        let label = if id == "shaderControl" { tool.json.get("control")?.as_str()?.to_owned() } else { crate::tools::control_label(&id)?.to_owned() };
        let row = b.ui.add_box(body, style! { display: flex; align-items: center; margin: 1.5px 0; flex: 1; });
        let l = b.ui.add_box(row, style! { display: flex; align-items: center; flex-shrink: 0; white-space: nowrap; });
        b.text_box(l, &label, style! { font-family: sans-serif; font-size: {SMALL}; color: white; });
        let key: &'static str = match id.as_str() {
            "opacity" => "opacity",
            "saturation" => "saturation",
            "selectedAlpha" => "selectedAlpha",
            "notSelectedAlpha" => "notSelectedAlpha",
            "blend" => "blend",
            "volumeRendering" => "volumeRendering",
            "crossSectionRenderScale" => "renderScale",
            "shaderControl" => "cdf",
            other => SEGMENTATION_CHECKBOXES.iter().find(|(_, k, _)| *k == other).map_or("", |(_, k, _)| k),
        };
        let kind = match id.as_str() {
            "opacity" | "saturation" | "selectedAlpha" | "notSelectedAlpha" => {
                let default = match key {
                    "opacity" => if layer.kind().is_some_and(|k| matches!(k, ng_viewer::LayerKind::Image { .. })) { 0.5 } else { 1.0 },
                    "saturation" => 1.0,
                    "selectedAlpha" => 0.5,
                    _ => 0.0,
                };
                let value = json.get(key).and_then(serde_json::Value::as_f64).filter(|v| (0.0..=1.0).contains(v)).unwrap_or(default);
                range_slider(self, b, row, slot, value, SliderSpec::alpha(default), Key::Control(slot, key));
                b.controls.insert(Key::Control(slot, key), Control::Slider { layer: i, key, value, spec: SliderSpec::alpha(default) });
                ControlKind::Wheel
            }
            "blend" | "volumeRendering" => {
                let (options, default): (&'static [&'static str], &'static str) = if key == "blend" { (&BLEND_MODES, "default") } else { (&VOLUME_RENDERING_MODES, "off") };
                let value = match json.get(key) {
                    Some(serde_json::Value::Bool(true)) => "on",
                    Some(serde_json::Value::String(v)) => options.iter().copied().find(|m| m == v).unwrap_or(default),
                    _ => default,
                };
                select(b, row, value, options, Key::Control(slot, key), Action::Repaint);
                b.controls.insert(Key::Control(slot, key), Control::Select { layer: i, key, options, value, default });
                ControlKind::Wheel
            }
            "crossSectionRenderScale" => {
                render_scale_widget(b, row, slot, "renderScale", layer.render_scale_histogram(), layer.render_scale_target(), "px");
                b.controls.insert(Key::Control(slot, "renderScale"), Control::RenderScale { layer: i, key: "crossSectionRenderScale", target: layer.render_scale_target(), origin: ng_core::render_scale::ORIGIN, invert_wheel: false });
                ControlKind::Wheel
            }
            "shaderControl" => {
                let (Some(t), Some(range), Some(window)) = (layer.data_type(), layer.range(), layer.invlerp_window()) else { return Some((l, ControlKind::Invlerp)) };
                viewer.want_data_histogram(i, true);
                let histogram = viewer.layers()[i].data_histogram().copied();
                invlerp_widget(self, b, row, slot, &json, t, range, window, histogram.as_ref(), InvlerpCtl::normalized(t), true);
                if let Some(Control::Cdf { layer, .. }) = b.controls.get_mut(&Key::Control(slot, "cdf")) {
                    *layer = i;
                }
                ControlKind::Invlerp
            }
            "colorSeed" | "segmentDefaultColor" => {
                let Some(ng_viewer::LayerKind::Segmentation(display)) = layer.kind().cloned() else { return Some((l, ControlKind::None)) };
                let fixed = display.default_color;
                // The mode's radio, first in the label.
                let seed = id == "colorSeed";
                let checked = if seed { fixed.is_none() } else { fixed.is_some() };
                let radio = threers::ui::controls::radio(&mut b.ui, l, checked, Edges::new(3.0, 3.0, 0.0, 5.0));
                b.ui.move_child(l, radio, 0);
                // `.neuroglancer-layer-control-label-text-container`: `align-items: start`.
                b.ui.style_mut(l).align_items = threers::ui::AlignItems::FlexStart;
                let mode = Key::Control(slot, if seed { "colorSeedMode" } else { "fixedColorMode" });
                let action = if seed {
                    Action::SetLayerOption(i, "segmentDefaultColor".into(), None)
                } else {
                    Action::Many(vec![Action::SetLayerOption(i, "segmentDefaultColor".into(), Some("#ff0000".into())), Action::PickColor(crate::ColorSetting::SegmentDefault(i))])
                };
                b.key(row, mode, Some(action));
                if seed {
                    let control = b.ui.add_box(row, style! { display: flex; align-items: center; justify-content: flex-end; flex: 1; max-width: 500px; margin-left: 5px; });
                    let input = b.ui.add_box(control, style! { display: flex; position: relative; align-items: center; width: 149px; height: 17px; padding: 1px 2px; box-sizing: border-box; background: {[1.0, 1.0, 1.0, 0.3]}; flex-shrink: 0; });
                    let fid = crate::FieldId::ColorSeed(slot);
                    let editing = self.field.as_ref().filter(|(f, _)| *f == fid).map(|(_, t)| t);
                    let font = style! { font-family: Arial; font-size: {SMALL}; color: white; };
                    self.text_contents(b, input, editing, &display.color_seed.to_string(), &font, 0.0);
                    let icon_id = 2900 + (slot >> 20) as u32;
                    let icon = b.icon(control, icon_id, Some(icons::ROTATE), "", Edges::symmetric(0.0, 1.0), Action::RandomizeSeed(i));
                    if fixed.is_some() {
                        b.ui.style_mut(control).opacity = 0.0;
                        b.keys.remove(&icon);
                    } else {
                        b.key(input, Key::Field(fid), None);
                    }
                } else {
                    b.ui.add_box(row, style! { flex: 1; });
                    let well = threers::ui::controls::color_well(&mut b.ui, row, fixed.unwrap_or([1.0, 0.0, 0.0]), 50.0);
                    b.ui.style_mut(well).margin.left = 5.0;
                    if fixed.is_some() {
                        b.key(well, Key::Color(crate::ColorSetting::SegmentDefault(i)), Some(Action::PickColor(crate::ColorSetting::SegmentDefault(i))));
                    } else {
                        b.ui.style_mut(well).opacity = 0.0;
                    }
                }
                ControlKind::None
            }
            _ if !key.is_empty() => {
                let default = SEGMENTATION_CHECKBOXES.iter().find(|(_, k, _)| *k == key).is_some_and(|c| c.2);
                b.ui.add_box(row, style! { flex: 1; });
                let on = json.get(key).and_then(serde_json::Value::as_bool).unwrap_or(default);
                let value = (!on != default).then_some(serde_json::Value::Bool(!on));
                let node = threers::ui::controls::checkbox(&mut b.ui, row, on, 13.0, threers::ui::controls::BORDER, Edges::new(3.0, 3.0, 3.0, 5.0));
                b.key(node, Key::Control(slot, key), Some(Action::SetLayerOption(i, key.into(), value)));
                ControlKind::None
            }
            _ => ControlKind::None,
        };
        Some((l, kind))
    }
}

/// `formatPixelNumber`.
fn format_pixel_number(x: f64) -> String {
    if !(1.0..=1024.0).contains(&x) {
        let exponent = x.log2().trunc();
        let coeff = x / exponent.exp2();
        return format!("{}p{}", ng_core::js::to_fixed(coeff, 1), number_to_string(exponent));
    }
    number_to_string(ng_core::js::round(x))
}

/// `RenderScaleWidget`: the histogram (log-scaled stacked bars per spatial
/// scale, present chunks solid, missing ones half bright), the target line,
/// and the legend.
fn render_scale_widget(b: &mut Builder, row: NodeId, i: usize, key: &'static str, histogram: &RenderScaleHistogram, target: f64, unit: &str) {
    let widget = b.ui.add_box(row, style! { display: flex; align-items: center; flex: 1; max-width: 500px; margin: 2px 0 2px 5px; });
    let _ = render_scale_body(b, widget, i, key, histogram, target, unit);
}

/// A `RenderScaleWidget` with its prompt (`label`) before the histogram,
/// as the Annotations tab shows its spacing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_scale_widget_with_prompt(b: &mut Builder, parent: NodeId, i: usize, key: &'static str, histogram: &RenderScaleHistogram, target: f64, unit: &str, prompt: &str, font: threers::ui::Style) {
    let widget = b.ui.add_box(parent, style! { display: flex; align-items: center; flex-shrink: 0; margin: 2px 0; });
    let label = b.text_box(widget, prompt, font);
    b.ui.style_mut(label).flex_shrink = 0.0;
    let canvas = render_scale_body(b, widget, i, key, histogram, target, unit);
    // After the prompt the canvas starts between pixels, and Chrome
    // resamples it there.
    b.ui.style_mut(canvas).subpixel = true;
}

/// The histogram canvas and its legend (the target, chunks present of
/// those wanted); the canvas.
fn render_scale_body(b: &mut Builder, widget: NodeId, i: usize, key: &'static str, histogram: &RenderScaleHistogram, target: f64, unit: &str) -> NodeId {
    let canvas = b.ui.add_box(widget, style! { position: relative; flex: 1; flex-basis: 0px; width: 0; height: 36px; overflow: hidden; });
    b.key(canvas, Key::Control(i, key), None);
    let legend_font = style! { font-family: sans-serif; font-size: 11px; color: white; };
    let ch = b.ch("sans-serif", 11.0);
    let legend = b.ui.add_box(widget, style! { display: flex; flex-direction: column; width: {crate::view::lu(15.0 * ch)}; flex-shrink: 0; });
    let (present, not_present) = histogram.totals();
    for text in [format!("{} {unit}", format_pixel_number(target)), format!("{present}/{}", present + not_present), String::new()] {
        let line = b.ui.add_box(legend, style! { display: flex; justify-content: flex-end; height: 12px; flex-shrink: 0; });
        if !text.is_empty() {
            b.text_box(line, &text, legend_font.clone());
        }
    }
    // The canvas is `offsetWidth` pixels, stretched over its box; drawn
    // after layout, as fractions of its width.
    let n = histogram.spatial_scales.len();
    // `sortedSpatialScales.sort()`: as strings.
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by_key(|&r| number_to_string(histogram.spatial_scales[r]));
    let max_count = (0..NUM_BINS)
        .map(|bin| histogram.rows.iter().map(|r| r[bin] + r[bin + NUM_BINS]).sum::<u32>())
        .fold(1, u32::max) as f64;
    let colors: Vec<([f32; 4], [f32; 4])> = sorted
        .iter()
        .map(|&r| {
            let s = histogram.spatial_scales[r];
            let hue = if s.is_finite() { ((s.log2() * 0.1) % 1.0 + 1.0) % 1.0 } else { 0.0 };
            let c = |v: f64| {
                let [r, g, b] = ng_core::segment_color::hsv_to_rgb(hue, 1.0, v);
                let q = |x: f32| (x * 255.0).round() / 255.0;
                [q(r), q(g), q(b), 1.0]
            };
            (c(1.0), c(0.5))
        })
        .collect();
    let mut bars = vec![];
    for bin in 0..NUM_BINS {
        let mut sum = 0u32;
        for k in (0..n).rev() {
            let r = &histogram.rows[sorted[k]];
            let (p, q) = (r[bin], r[bin + NUM_BINS]);
            let count = p + q;
            if count == 0 {
                continue;
            }
            bars.push(Bar { bin, start: sum, end: sum + count, present: p, colors: colors[k] });
            sum += count;
        }
    }
    b.deferred.push(Deferred::Bars(CanvasBars { node: canvas, bars, max_count, target_offset: ng_core::render_scale::histogram_offset_at(target, histogram.origin) }));
    canvas
}

/// A rectangle (x, y, width, height) and its colour.
pub(crate) type Fill = ([f64; 4], [f32; 4]);

/// One stacked bar segment of the render scale histogram.
#[derive(Debug, Clone)]
pub(crate) struct Bar {
    bin: usize,
    start: u32,
    end: u32,
    present: u32,
    colors: ([f32; 4], [f32; 4]),
}

/// A render scale canvas, drawn once its box is laid out.
#[derive(Debug, Clone)]
pub(crate) struct CanvasBars {
    pub node: NodeId,
    bars: Vec<Bar>,
    max_count: f64,
    target_offset: f64,
}

impl CanvasBars {
    /// The canvas's fills at its laid-out size: (x, y, w, h) in canvas
    /// pixels, and colour; then the bitmap's width.
    pub(crate) fn fills(&self, box_width: f32) -> (Vec<Fill>, f64) {
        // `canvas.width = canvas.offsetWidth`.
        let width = box_width.round() as f64;
        let height = 36.0;
        let x_of = |bin: f64| bin * width / NUM_BINS as f64;
        let y_scale = height / (1.0 + self.max_count).ln();
        let y_of = |count: u32| height - (1.0 + count as f64).ln() * y_scale;
        let mut out = vec![];
        for bar in &self.bars {
            let x0 = ng_core::js::round(x_of(bar.bin as f64));
            let x1 = ng_core::js::round(x_of(bar.bin as f64 + 1.0));
            let y_start = ng_core::js::round(y_of(bar.start));
            let y_end = ng_core::js::round(y_of(bar.end));
            let count = (bar.end - bar.start) as f64;
            let split = (bar.present as f64 * y_end + (count - bar.present as f64) * y_start) / count;
            out.push(([x0, y_end, x1 - x0, split - y_end], bar.colors.1));
            out.push(([x0, split, x1 - x0, y_start - split], bar.colors.0));
        }
        out.push(([x_of(self.target_offset).floor(), 0.0, 1.0, height], [1.0; 4]));
        (out, width)
    }
}

/// A closed `<select>` as Chrome draws it: white, rounded, a chevron; as
/// wide as its widest option.
pub(crate) fn select(b: &mut Builder, row: NodeId, value: &str, options: &[&str], key: Key, action: Action) -> NodeId {
    b.ui.add_box(row, style! { flex: 1; });
    let font = style! { font-family: Arial; font-size: {SMALL}; color: black; };
    let node = controls::select(&mut b.ui, row, value, options, &font, b.text);
    b.ui.style_mut(node).margin.left = 5.0;
    b.key(node, key, Some(action));
    node
}

/// `.range-slider`: an `<input type=range>` (Chrome's), then the number.
fn range_slider(ui: &NgUi, b: &mut Builder, row: NodeId, i: usize, value: f64, spec: SliderSpec, key: Key) {
    let Key::Control(_, key_name) = key else { return };
    let slider = b.ui.add_box(row, style! { display: flex; align-items: center; justify-content: flex-end; flex: 1; max-width: 500px; margin-left: 5px; });
    // `flex-basis: 0` without growing: Chrome's intrinsic 129px.
    let hovered = b.hover(key);
    let range = controls::range(&mut b.ui, slider, spec.fraction(value) as f32, hovered);
    b.ui.style_mut(range).margin = Edges::all(2.0);
    b.snapped.push(range);
    b.key(range, key, None);
    // The number: `maxNumberWidth + 2` characters of its font (7.412px
    // each), the longest of the bounds and a step in from each, and its
    // 2px side paddings.
    let len = |v: f64| number_to_string(v).len();
    let max_number_width = len(spec.min).max(len(spec.max)).max(len(spec.max.min(spec.min + spec.step))).max(len(spec.min.max(spec.max - spec.step)));
    let number_width = (max_number_width + 2) as f32 * ((48.47 - 4.0) / 6.0) + 4.0;
    let number = b.ui.add_box(slider, style! { display: flex; position: relative; align-items: center; width: {number_width}; height: 20px; padding: 1px 2px; box-sizing: border-box; background: {[1.0, 1.0, 1.0, 0.3]}; flex-shrink: 0; });
    let id = crate::FieldId::Slider(i, key_name);
    b.key(number, Key::Field(id), None);
    let editing = ui.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t);
    let text = number_to_string(value);
    let font = style! { font-family: Arial; font-size: {SMALL}; color: white; };
    ui.text_contents(b, number, editing, &text, &font, 0.0);
}

/// Chrome's unchecked checkbox in the light scheme: white, grey border.
fn light_checkbox(b: &mut Builder, parent: NodeId, checked: bool, key: Key, action: Action) -> NodeId {
    let node = controls::checkbox(&mut b.ui, parent, checked, 13.0, controls::BORDER, Edges::new(3.0, 3.0, 3.0, 4.0));
    b.key(node, key, Some(action));
    node
}

/// `image_layer_rendering.md`, the shader's help.
const IMAGE_RENDERING_DOCS: &str = "https://github.com/google/neuroglancer/blob/master/src/sliceview/image_layer_rendering.md";

/// CodeMirror's token classes and the default theme's colours.
pub(crate) fn token_color(token: Option<&str>) -> &'static str {
    match token {
        Some("keyword") => "#708",
        Some("atom") => "#219",
        Some("number") => "#164",
        Some("comment") => "#a50",
        Some("string") => "#a11",
        Some("meta") => "#555",
        Some("builtin") => "#30a",
        Some("bracket") => "#997",
        _ => "black",
    }
}

const GLSL_KEYWORDS: &str = "attribute const uniform varying break continue do for while if else in out inout float int void bool true false lowp mediump highp precision invariant discard return mat2 mat3 mat4 vec2 vec3 vec4 ivec2 ivec3 ivec4 bvec2 bvec3 bvec4 sampler2D samplerCube struct gl_FragCoord gl_FragColor";
const GLSL_BUILTINS: &str = "radians degrees sin cos tan asin acos atan pow exp log exp2 log2 sqrt inversesqrt abs sign floor ceil fract mod min max clamp mix step smoothstep length distance dot cross normalize faceforward reflect refract matrixCompMult lessThan lessThanEqual greaterThan greaterThanEqual equal notEqual any all not dFdx dFdy fwidth texture2D texture2DProj texture2DLod texture2DProjLod textureCube textureCubeLod require export";

/// `codemirror-glsl.ts`'s tokens, per line: text and class (`None` for
/// spaces and unstyled tokens). Adjacent tokens of one class merge, as
/// CodeMirror's spans do.
pub(crate) fn glsl_tokens(code: &str) -> Vec<Vec<(String, Option<&'static str>)>> {
    let mut in_comment = false;
    let mut out = vec![];
    for line in code.split('\n') {
        let c: Vec<char> = line.chars().collect();
        let mut tokens: Vec<(String, Option<&'static str>)> = vec![];
        let mut push = |s: String, t: Option<&'static str>| match tokens.last_mut() {
            Some((p, pt)) if *pt == t => p.push_str(&s),
            _ => tokens.push((s, t)),
        };
        let mut i = 0;
        let word = |ch: char| ch.is_alphanumeric() || ch == '_' || ch == '$';
        while i < c.len() {
            let start = i;
            if in_comment {
                let mut maybe_end = false;
                while i < c.len() {
                    let ch = c[i];
                    i += 1;
                    if ch == '/' && maybe_end {
                        in_comment = false;
                        break;
                    }
                    maybe_end = ch == '*';
                }
                push(c[start..i].iter().collect(), Some("comment"));
                continue;
            }
            if c[i].is_whitespace() {
                while i < c.len() && c[i].is_whitespace() {
                    i += 1;
                }
                push(c[start..i].iter().collect(), None);
                continue;
            }
            let ch = c[i];
            i += 1;
            let t: Option<&'static str> = if ch == '"' || ch == '\'' {
                let mut escaped = false;
                while i < c.len() {
                    let n = c[i];
                    i += 1;
                    if n == ch && !escaped {
                        break;
                    }
                    escaped = !escaped && n == '\\';
                }
                Some("string")
            } else if "[]{}(),;:.".contains(ch) {
                Some("bracket")
            } else if ch.is_ascii_digit() {
                while i < c.len() && (word(c[i]) || c[i] == '.') {
                    i += 1;
                }
                Some("number")
            } else if ch == '/' && c.get(i) == Some(&'*') {
                i += 1;
                in_comment = true;
                let mut maybe_end = false;
                while i < c.len() {
                    let n = c[i];
                    i += 1;
                    if n == '/' && maybe_end {
                        in_comment = false;
                        break;
                    }
                    maybe_end = n == '*';
                }
                Some("comment")
            } else if ch == '/' && c.get(i) == Some(&'/') {
                i = c.len();
                Some("comment")
            } else if ch == '#' {
                // `#`, a word, spaces, a word, spaces.
                for ws in [false, true, false, true] {
                    while i < c.len() && c[i].is_whitespace() == ws {
                        i += 1;
                    }
                }
                Some("comment")
            } else if "+-*&%=<>!?|/".contains(ch) {
                while i < c.len() && "+-*&%=<>!?|/".contains(c[i]) {
                    i += 1;
                }
                None
            } else {
                while i < c.len() && word(c[i]) {
                    i += 1;
                }
                let cur: String = c[start..i].iter().collect();
                if GLSL_KEYWORDS.split(' ').any(|k| k == cur) {
                    Some("keyword")
                } else if GLSL_BUILTINS.split(' ').any(|k| k == cur) {
                    Some("builtin")
                } else if cur == "null" {
                    Some("atom")
                } else {
                    None
                }
            };
            push(c[start..i].iter().collect(), t);
        }
        out.push(tokens);
    }
    out
}


/// `updateInputBoundValue`: integers as they are, others to 6 digits.
fn format_bound(v: f64) -> String {
    if v.fract() == 0.0 {
        number_to_string(v)
    } else {
        ng_core::js::to_precision(v, 6)
    }
}

/// `getIntervalBoundsEffectiveFraction`.
fn effective_fraction(t: DataType, window: [f64; 2]) -> f64 {
    if t == DataType::Float32 {
        return 1.0;
    }
    let diff = (window[0] - window[1]).abs();
    diff / (diff + 1.0)
}

/// `getIntervalBoundsEffectiveOffset`.
fn effective_offset(t: DataType, window: [f64; 2]) -> f64 {
    if t == DataType::Float32 {
        return 0.0;
    }
    0.5 / (window[0] - window[1]).abs()
}

/// `computeInvlerp` on numbers.
fn invlerp(interval: [f64; 2], v: f64) -> f64 {
    (v - interval[0]) / (interval[1] - interval[0])
}

/// `InvlerpWidget`: legend, range bounds (with the invert arrow), the CDF
/// panel, window bounds, range finder buttons.
#[allow(clippy::too_many_arguments)]
pub(crate) fn invlerp_widget(ui: &NgUi, b: &mut Builder, row: NodeId, i: usize, json: &serde_json::Map<String, serde_json::Value>, t: DataType, range: Interval, window: Interval, histogram: Option<&[u32; 256]>, ctl: InvlerpCtl, with_legend: bool) {
    let w = window.as_f64();
    let r = range.as_f64();
    let widget = b.ui.add_box(row, style! { display: flex; flex-direction: column; flex: 1; max-width: 500px; margin-left: 5px; });
    // The colour legend: the shader's output across the window (none for a
    // channel of several: `channelInvlerpLayerControl`).
    if with_legend {
        invlerp_legend(ui, b, widget, i, t, range, w, ctl);
    }
    invlerp_body(ui, b, widget, i, json, t, r, w, histogram, ctl);
}

#[allow(clippy::too_many_arguments)]
fn invlerp_legend(ui: &NgUi, b: &mut Builder, widget: NodeId, i: usize, t: DataType, range: Interval, w: [f64; 2], ctl: InvlerpCtl) {
    let legend = b.ui.add_box(widget, style! { position: relative; height: 15px; border: 1px solid #666; flex-shrink: 0; });
    // A user shader's legend: its colours as the viewer drew them (at the
    // width the legend had last), else the default shader's greys.
    let key = (i, ctl.name);
    let colors = ui.legend_colors.get(&key).cloned();
    b.deferred.push(Deferred::Legend(Legend { node: legend, t, range, window: w, colors, key }));
}

/// The rest of `InvlerpWidget`: range bounds (with the invert arrow), the
/// CDF panel, window bounds.
#[allow(clippy::too_many_arguments)]
fn invlerp_body(ui: &NgUi, b: &mut Builder, widget: NodeId, i: usize, json: &serde_json::Map<String, serde_json::Value>, t: DataType, r: [f64; 2], w: [f64; 2], histogram: Option<&[u32; 256]>, ctl: InvlerpCtl) {
    let ch = b.ch("monospace", 13.0);
    let bound_width = |s: &str| crate::view::lu((s.chars().count() as f32 + 0.1).max(1.0) * ch) + 4.0;
    let fraction = effective_fraction(t, w);
    let reversed = r[0] > r[1];
    let clamped = [r[0].clamp(w[0].min(w[1]), w[0].max(w[1])), r[1].clamp(w[0].min(w[1]), w[0].max(w[1]))];
    let left = invlerp(w, clamped[if reversed { 1 } else { 0 }]) * fraction;
    let right = invlerp(w, clamped[if reversed { 0 } else { 1 }]) * fraction + (1.0 - fraction);
    let bounds = b.ui.add_box(widget, style! { display: flex; height: 20px; flex-shrink: 0; });
    let spacer0 = b.ui.add_box(bounds, style! { width: {Length::Percent(left as f32 * 100.0)}; flex-shrink: 0; });
    // An input as wide as its text (`updateInputBoundWidth`, as typed).
    let input = |b: &mut Builder, parent: NodeId, id: crate::FieldId, text: &str, color: &str, height: f32| {
        let editing = ui.field.as_ref().filter(|(f, _)| *f == id).map(|(_, t)| t);
        let shown = editing.map_or(text, |t| t.text.as_str());
        let n = b.ui.add_box(parent, style! { display: flex; position: relative; align-items: center; width: {bound_width(shown)}; height: {height}; padding: 1px 2px; box-sizing: border-box; flex-shrink: 0; });
        b.key(n, Key::Field(id), None);
        let font = style! { font-family: monospace; font-size: 13px; color: {color}; };
        ui.text_contents(b, n, editing, text, &font, 0.0);
        if editing.is_some() {
            controls::focus_ring(&mut b.ui, n);
        }
        n
    };
    let (lo, hi) = (format_bound(r[0]), format_bound(r[1]));
    use crate::FieldId::Bound;
    // `row-reverse` when reversed: endpoint 1 shows first.
    let first = input(b, bounds, Bound(i, false, reversed as usize), if reversed { &hi } else { &lo }, "white", 20.0);
    let spacer1 = b.ui.add_box(bounds, style! { display: flex; justify-content: center; align-items: flex-start; flex: 1; min-width: 0; });
    let arrow = if reversed { icons::ARROW_LEFT } else { icons::ARROW_RIGHT };
    let layer = i & LAYER_MASK;
    let swapped = Action::SetLayerOption(layer, "shaderControls".into(), invlerp_controls_json(json.get("shaderControls"), t, ctl, [r[1], r[0]], w));
    // Apart per widget (a layer may show several).
    let icon = b.icon(spacer1, 0x8000_0000 ^ (i as u32), Some(arrow), "", Edges::symmetric(0.0, 1.0), swapped);
    b.ui.style_mut(icon).align_self = Some(threers::ui::AlignItems::FlexStart);
    input(b, bounds, Bound(i, false, !reversed as usize), if reversed { &lo } else { &hi }, "white", 20.0);
    b.ui.add_box(bounds, style! { width: {Length::Percent((1.0 - right) as f32 * 100.0)}; flex-shrink: 0; });
    let _ = (spacer0, first);

    // The CDF panel: the range's region (and, once data histograms are
    // computed, the CDF line).
    let cdf = b.ui.add_box(widget, style! { position: relative; height: 50px; border: 1px solid #666; flex-shrink: 0; });
    b.key(cdf, Key::Control(i, "cdf"), None);
    b.controls.insert(Key::Control(i, "cdf"), Control::Cdf { layer, t, range: r, window: w, controls: json.get("shaderControls").cloned(), ctl });
    let f0 = invlerp(w, r[0]);
    let f1 = invlerp(w, r[1]);
    let line = histogram.map(|h| ng_viewer::histogram::cdf_vertices(h, fraction));
    b.deferred.push(Deferred::Cdf(cdf, f0.min(f1) * fraction, f0.max(f1) * fraction + (1.0 - fraction), line));

    let window_bounds = b.ui.add_box(widget, style! { display: flex; justify-content: space-between; height: 17px; flex-shrink: 0; });
    input(b, window_bounds, Bound(i, true, 0), &format_bound(w[0]), "#0ff", 17.0);
    input(b, window_bounds, Bound(i, true, 1), &format_bound(w[1]), "#0ff", 17.0);

    let buttons = b.ui.add_box(widget, style! { display: flex; margin-top: 1px; flex-shrink: 0; });
    for (k, label) in ["Min-Max", "1-99%", "5-95%"].into_iter().enumerate() {
        let (lower, upper) = [(0.0, 1.0), (0.01, 0.99), (0.05, 0.95)][k];
        button(b, buttons, label, Key::Control(i, ["minMax", "p1", "p5"][k]), Action::AutoRange { layer, ctl, range: r, lower, upper });
    }
}

/// Chrome's `<button>`: light grey, rounded, 11pt Arial.
fn button(b: &mut Builder, parent: NodeId, label: &str, key: Key, action: Action) -> NodeId {
    let font = style! { font-family: Arial; font-size: {44.0 / 3.0}; color: black; };
    let node = controls::button(&mut b.ui, parent, label, &font, b.text);
    b.ui.style_mut(node).margin.right = 2.0;
    b.key(node, key, Some(action));
    node
}

/// What is drawn into a box once it is laid out (canvas contents).
#[derive(Debug, Clone)]
pub(crate) enum Deferred {
    Bars(CanvasBars),
    Legend(Legend),
    /// The CDF panel, the range region's fractions of it, and the CDF's
    /// vertices (fractions across, cumulative fractions).
    Cdf(NodeId, f64, f64, Option<Vec<(f64, f64)>>),
    /// A table laid out as `table-layout: auto` at full width: its
    /// columns' content widths, its cells (column, span); cells 2px apart.
    Table { table: NodeId, widths: Vec<f32>, cells: Vec<(NodeId, usize, usize)> },
    /// A dimension's plot in row orientation (the dimension tool's): the
    /// bound labels' box, the canvas box (its margins set here: half the
    /// wider label), the bounds, the position, the labels' `ch`.
    RowPlot { labels: NodeId, canvas: NodeId, space: Box<ng_core::CoordinateSpace>, dim: usize, position: f64, ch: f32 },
}

impl Deferred {
    /// Adds the drawing to the laid-out tree, as absolutely placed boxes.
    pub(crate) fn draw(&self, ui: &mut threers::ui::Ui) {
        let place = |ui: &mut threers::ui::Ui, parent: NodeId, [x, y, w, h]: [f64; 4], color: [f32; 4], radius: f32| {
            if w <= 0.0 || h <= 0.0 {
                return;
            }
            ui.add_box(
                parent,
                style! { position: absolute; left: {x as f32}; top: {y as f32}; width: {w as f32}; height: {h as f32}; background: {color}; border-radius: {radius}; },
            );
        };
        match self {
            Deferred::Bars(bars) => {
                let r = ui.rect(bars.node);
                let (fills, width) = bars.fills(r.width);
                let sx = r.width as f64 / width;
                for ([x, y, w, h], color) in fills {
                    place(ui, bars.node, [x * sx, y, w * sx, h], color, 0.0);
                }
            }
            Deferred::Legend(legend) => {
                let r = ui.rect(legend.node);
                let width = (r.width - 2.0).round() as usize;
                match legend.colors.as_ref().filter(|c| c.len() == width) {
                    // Over the panel's black: the colour times its alpha.
                    Some(colors) => {
                        for (x, c) in colors.iter().enumerate() {
                            let a = c[3] as f32 / 255.0;
                            let color = [c[0] as f32 / 255.0 * a, c[1] as f32 / 255.0 * a, c[2] as f32 / 255.0 * a, 1.0];
                            place(ui, legend.node, [x as f64, 0.0, 1.0, 15.0], color, 0.0);
                        }
                    }
                    None => {
                        for (x, g) in legend.columns(width).into_iter().enumerate() {
                            place(ui, legend.node, [x as f64, 0.0, 1.0, 15.0], [g, g, g, 1.0], 0.0);
                        }
                    }
                }
            }
            Deferred::Table { table, widths, cells } => {
                // The width left, shared in proportion to content widths.
                let n = widths.len() as f32;
                let available = ui.rect(*table).width - 2.0 - 2.0 * n;
                let sum: f32 = widths.iter().sum();
                let extra = (available - sum).max(0.0);
                let cols: Vec<f32> = widths.iter().map(|w| w + extra * w / sum.max(1.0)).collect();
                for &(node, c, span) in cells {
                    let w: f32 = cols[c..c + span].iter().sum::<f32>() + 2.0 * (span - 1) as f32;
                    ui.style_mut(node).width = Length::Px(w);
                }
            }
            Deferred::RowPlot { labels, canvas, space, dim, position, ch } => {
                let plot_width = ui.rect(*labels).width;
                let Some(probe) = crate::plot::dimension_bounds(space, *dim, 2.0) else { return };
                let (lower, upper) = (number_to_string(probe.lower), number_to_string(probe.upper));
                // `clientWidth`s: whole pixels.
                let (lw, uw) = ((lower.chars().count() as f32 * ch).round(), (upper.chars().count() as f32 * ch).round());
                let margin = lw.max(uw) / 2.0;
                // `canvas.width = canvas.clientWidth`.
                let css = (plot_width - 2.0 * margin).max(1.0);
                let length = css.round().max(1.0);
                // The canvas's pixels stretched over its box.
                let sx = (css / length) as f64;
                let scaled = |[x, y, w, h]: [f64; 4]| [x * sx, y, w * sx, h];
                let s = ui.style_mut(*canvas);
                s.margin.left = margin;
                s.margin.right = margin;
                let Some(bounds) = crate::plot::dimension_bounds(space, *dim, length as f64) else { return };
                let font = style! { position: absolute; top: 0; font-family: monospace; font-size: 13px; color: white; };
                for (text, x) in [(&lower, margin - lw / 2.0), (&upper, length + margin - uw / 2.0)] {
                    let mut f = font.clone();
                    f.left = Length::Px(x.max(0.0));
                    let node = ui.add_box(*labels, f);
                    ui.add_text(node, text.as_str());
                }
                let (ticks, bars) = crate::plot::row_bars(&bounds);
                for r in bars {
                    place(ui, *canvas, scaled(r), [0.8, 0.8, 0.8, 1.0], 0.0);
                }
                for r in ticks {
                    place(ui, *canvas, scaled(r), [1.0; 4], 0.0);
                }
                if let Some(j) = crate::plot::indicator(&bounds, *position, length as f64) {
                    // The line's canvas column, filtered (bilinear) onto
                    // the page's pixels: where the canvas will start.
                    let x0 = (ui.rect(*labels).x + margin) as f64;
                    let first = (x0 + (j - 1.0) * sx).floor() as i64;
                    for i in first..=first + 3 {
                        let u = (i as f64 + 0.5 - x0) / sx - 0.5;
                        let w = 1.0 - (u - j).abs();
                        if w > 0.0 {
                            place(ui, *canvas, [i as f64 - x0, 0.0, 1.0, crate::plot::ROW_CANVAS_HEIGHT], [1.0, 0.4, 0.4, w as f32], 0.0);
                        }
                    }
                }
            }
            Deferred::Cdf(node, lo, hi, line) => {
                let r = ui.rect(*node);
                let width = (r.width - 2.0).round() as f64;
                // Pixels whose centres the region covers, as GL fills.
                let a = (lo * width - 0.5).ceil().max(0.0);
                let b = (hi * width - 0.5).ceil().min(width);
                place(ui, *node, [a, 0.0, b - a, 50.0], [0.2, 0.2, 0.2, 1.0], 0.0);
                // The CDF: a 1px line, feathered by a pixel, bottom to top
                // (`cdf · (2 − w) − 1 + w/2` in clip space).
                if let Some(line) = line {
                    let height = 50.0;
                    let d: String = line
                        .iter()
                        .enumerate()
                        .map(|(k, (x, c))| format!("{}{:.3} {:.3}", if k == 0 { "M" } else { "L" }, x * width, height - (c * (height - 1.0) + 0.5)))
                        .collect();
                    ui.add_path(
                        *node,
                        style! { position: absolute; left: 0; top: 0; width: {width as f32}; height: 50px; color: #0ff; },
                        vec![d],
                        [0.0, 0.0, width as f32, height as f32],
                        Some(3.0),
                        false,
                    );
                }
            }
        }
    }
}

/// A colour legend to fill once laid out.
#[derive(Debug, Clone)]
pub(crate) struct Legend {
    pub node: NodeId,
    t: DataType,
    range: Interval,
    window: [f64; 2],
    /// A user shader's colours for it (RGBA over transparent), and the
    /// control it is (slot, name).
    pub colors: Option<Vec<[u8; 4]>>,
    pub key: (usize, &'static str),
}

/// The data value at each of `width` legend columns: the window's at the
/// column's centre (`ng_colorLegendLerp`), integers rounded.
pub(crate) fn legend_values(t: DataType, window: [f64; 2], width: usize) -> Vec<f64> {
    let off = effective_offset(t, window);
    let off = if off.is_finite() { off } else { 0.0 };
    (0..width)
        .map(|x| {
            let pos = -off + ((x as f64 + 0.5) / width as f64) * (1.0 + 2.0 * off);
            let v = window[0] + pos * (window[1] - window[0]);
            if t != DataType::Float32 {
                v.round()
            } else {
                v
            }
        })
        .collect()
}

impl Legend {
    /// The grey of each of `width` columns: the default shader
    /// (`emitGrayscale(normalized())`) of the value the window puts there.
    pub(crate) fn columns(&self, width: usize) -> Vec<f32> {
        legend_values(self.t, self.window, width).into_iter().map(|v| ng_viewer::expected_gray(self.t, self.range, v) as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glsl_tokens_as_codemirror() {
        let t = glsl_tokens(ng_viewer::state::DEFAULT_IMAGE_SHADER);
        let s = |v: &[(String, Option<&str>)]| v.iter().map(|(a, b)| format!("{a}|{}", b.unwrap_or("-"))).collect::<Vec<_>>().join(" ");
        assert_eq!(s(&t[0]), "#uicontrol invlerp |comment normalized|-");
        assert_eq!(s(&t[1]), "void|keyword  main|- ()|bracket  |- {|bracket");
        assert_eq!(s(&t[2]), "  emitGrayscale|- (|bracket normalized|- ());|bracket");
        assert_eq!(s(&t[3]), "}|bracket");
        assert_eq!(t.len(), 5);
        assert_eq!(s(&glsl_tokens("float x = 1.5; // c\n/* a\nb */ clamp")[2]), "b */|comment  |- clamp|builtin");
    }

    #[test]
    fn pixel_numbers_as_neuroglancer() {
        assert_eq!(format_pixel_number(1.0), "1");
        assert_eq!(format_pixel_number(0.5), "1.0p-1");
        assert_eq!(format_pixel_number(2048.0), "1.0p11");
        assert_eq!(format_pixel_number(1.49), "1");
        assert_eq!(format_bound(50.0), "50");
        assert_eq!(format_bound(0.25), "0.250000");
    }
}

/// What a layer control does with the pointer: its model as of the last
/// build (`Key::Control`).
#[derive(Debug, Clone)]
pub(crate) enum Control {
    /// `RenderScaleWidget`: its JSON member (`crossSectionRenderScale`,
    /// `meshRenderScale`, `volumeRenderingDepthSamples`), target, the
    /// histogram's origin, and whether the wheel runs the other way
    /// (`VolumeRenderingRenderScaleWidget`).
    RenderScale { layer: usize, key: &'static str, target: f64, origin: f64, invert_wheel: bool },
    /// A range input (`opacity`): its JSON member (dotted within an
    /// object), value, bounds, step and default.
    Slider { layer: usize, key: &'static str, value: f64, spec: SliderSpec },
    /// The invlerp CDF panel (`shaderControls.normalized`).
    Cdf { layer: usize, t: DataType, range: [f64; 2], window: [f64; 2], controls: Option<serde_json::Value>, ctl: InvlerpCtl },
    /// A `<select>`: the JSON member, its options, the current one, and
    /// the one written as absent.
    Select { layer: usize, key: &'static str, options: &'static [&'static str], value: &'static str, default: &'static str },
    /// A dimension's playback boundary behaviour (`EnumSelectWidget`).
    Boundary { dim: usize, value: &'static str },
    /// A user shader's slider: its layer and name, value, bounds and step,
    /// default, and whether it holds integers.
    ShaderSlider { layer: usize, name: &'static str, value: f64, min: f64, max: f64, step: f64, default: f64, int: bool },
    /// A user shader's `select`: its layer and name, the options' labels and
    /// values, the current one and the default.
    ShaderSelect { layer: usize, name: &'static str, labels: Vec<&'static str>, values: Vec<serde_json::Value>, current: usize, default: usize },
}

/// A range input's bounds and step, and its `TrackableValue`'s default
/// (left out of the state).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SliderSpec {
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub default: f64,
}

impl SliderSpec {
    /// `trackableAlphaValue`: 0 to 1 in steps of 0.01.
    pub(crate) const fn alpha(default: f64) -> Self {
        Self { min: 0.0, max: 1.0, step: 0.01, default }
    }

    /// The thumb's position: the value's fraction of the run.
    fn fraction(&self, value: f64) -> f64 {
        if self.max > self.min { ((value - self.min) / (self.max - self.min)).clamp(0.0, 1.0) } else { 0.0 }
    }

    /// The value at fraction `f` of the run, stepped as browsers step.
    fn at(&self, f: f64) -> f64 {
        crate::shader_ui::slider_value(f, self.min, self.max, self.step, false)
    }

    /// `stepUp` (`direction` 1) or `stepDown` (−1) from `value`.
    fn stepped(&self, value: f64, direction: f64) -> f64 {
        let span = (self.max - self.min).max(f64::EPSILON);
        self.at((value + direction * self.step - self.min) / span)
    }

    /// The state's member for `value`: none at the default.
    pub(crate) fn json(&self, value: f64) -> Option<serde_json::Value> {
        crate::shader_ui::slider_json(value, self.default)
    }
}

/// `VelocityBoundaryBehavior`'s options, as the select lists them.
pub(crate) const BOUNDARIES: [&str; 3] = ["stop", "loop", "reverse"];

/// `computeLerp` on numbers: integer types rounded and clamped to the
/// type's range.
fn lerp(t: DataType, interval: [f64; 2], x: f64) -> f64 {
    let v = interval[0] * (1.0 - x) + interval[1] * x;
    if t == DataType::Float32 {
        return v;
    }
    let [lo, hi] = ng_viewer::Interval::as_f64(&ng_viewer::default_data_type_range(t));
    ng_core::js::round(v).max(lo).min(hi)
}

/// A data type value as JSON (`dataTypeIntervalToJson`'s elements).
fn value_json(t: DataType, v: f64) -> serde_json::Value {
    if t == DataType::Uint64 {
        serde_json::Value::String(number_to_string(v))
    } else {
        ng_core::js::number(v)
    }
}

/// An invlerp control a widget edits: its name and its defaults (the
/// directive's, `range` and `normalizeDataTypeInterval(range)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvlerpCtl {
    pub name: &'static str,
    pub default_range: [f64; 2],
    pub default_window: [f64; 2],
}

impl InvlerpCtl {
    /// The default shader's `normalized`.
    pub(crate) fn normalized(t: DataType) -> Self {
        let d = ng_viewer::Interval::as_f64(&ng_viewer::default_data_type_range(t));
        InvlerpCtl { name: "normalized", default_range: d, default_window: d }
    }
}

/// `shaderControls` with invlerp control `ctl` as `toJSON` writes it:
/// `range` and `window` where they differ from its defaults, the control
/// left out if neither does, the member removed (`None`) if nothing is
/// left.
pub(crate) fn invlerp_controls_json(controls: Option<&serde_json::Value>, t: DataType, ctl: InvlerpCtl, range: [f64; 2], window: [f64; 2]) -> Option<serde_json::Value> {
    let mut o = controls.and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
    let mut n = o.get(ctl.name).and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
    for (k, v, default) in [("range", range, ctl.default_range), ("window", window, ctl.default_window)] {
        if v == default {
            n.remove(k);
        } else {
            n.insert(k.into(), serde_json::json!([value_json(t, v[0]), value_json(t, v[1])]));
        }
    }
    if n.is_empty() {
        o.remove(ctl.name);
    } else {
        o.insert(ctl.name.into(), serde_json::Value::Object(n));
    }
    (!o.is_empty()).then_some(serde_json::Value::Object(o))
}

impl Control {
    /// A press at `x` on the control's box `r`: the change, and (for drags)
    /// the endpoint the drag moves.
    pub(crate) fn press(&self, r: threers::ui::Rect, x: f32) -> Option<(Action, Option<usize>)> {
        match self {
            Control::RenderScale { layer, key, origin, .. } => {
                // `getTargetValue`: offsetX over the bitmap's width, in bins.
                let width = r.width.round() as f64;
                let position = (x - r.x) as f64 / width * NUM_BINS as f64;
                let target = ng_core::render_scale::scale_from_histogram_offset_at(position, *origin);
                Some((Action::SetLayerOption(*layer, (*key).into(), Some(ng_core::js::number(target))), None))
            }
            Control::Slider { .. } | Control::ShaderSlider { .. } => Some((self.drag(r, x, 0)?, Some(0))),
            Control::Cdf { range, window, .. } => {
                let value = self.cdf_value(r, x)?;
                // `getClosestEndpoint` of the range clamped to the window.
                let (lo, hi) = (window[0].min(window[1]), window[0].max(window[1]));
                let clamped = [range[0].clamp(lo, hi), range[1].clamp(lo, hi)];
                let endpoint = if (value - clamped[0]).abs() < (value - clamped[1]).abs() { 0 } else { 1 };
                Some((self.drag(r, x, endpoint)?, Some(endpoint)))
            }
            Control::Select { .. } | Control::Boundary { .. } | Control::ShaderSelect { .. } => None,
        }
    }

    /// The value under `x` on the CDF panel (`getWindowLerp`).
    fn cdf_value(&self, r: threers::ui::Rect, x: f32) -> Option<f64> {
        let Control::Cdf { t, window, .. } = self else { return None };
        let fraction = ((x - r.x) / r.width) as f64;
        fraction.is_finite().then(|| lerp(*t, *window, fraction))
    }

    /// The pointer dragged to `x`, moving `endpoint`.
    pub(crate) fn drag(&self, r: threers::ui::Rect, x: f32, endpoint: usize) -> Option<Action> {
        match self {
            Control::ShaderSlider { layer, name, min, max, step, default, int, .. } => {
                let f = controls::range_fraction(r, x) as f64;
                let v = crate::shader_ui::slider_value(f, *min, *max, *step, *int);
                Some(Action::SetShaderControl(*layer, name, crate::shader_ui::slider_json(v, *default)))
            }
            Control::Slider { layer, key, spec, .. } => {
                // Chrome's range input: the thumb's centre runs from 8px in
                // to 8px short of the end; values snap to the step.
                let f = controls::range_fraction(r, x) as f64;
                Some(Action::SetLayerOption(*layer, (*key).into(), spec.json(spec.at(f))))
            }
            Control::Cdf { layer, t, range, window, controls, ctl } => {
                // `getUpdatedRangeAndWindowParameters(bounds, "range", …)`.
                let value = self.cdf_value(r, x)?;
                let mut range = *range;
                range[endpoint] = value;
                Some(Action::SetLayerOption(*layer, "shaderControls".into(), invlerp_controls_json(controls.as_ref(), *t, *ctl, range, *window)))
            }
            _ => None,
        }
    }

    /// `adjust-via-wheel` on the render scale: doubled or halved, within
    /// the histogram's range.
    pub(crate) fn wheel(&self, delta_y: f64) -> Option<Action> {
        if delta_y == 0.0 {
            return None;
        }
        match self {
            // `stepUp`/`stepDown` by its step.
            Control::ShaderSlider { layer, name, value, min, max, step, default, int } => {
                let span = (max - min).max(f64::EPSILON);
                let v = crate::shader_ui::slider_value((value + delta_y.signum() * step - min) / span, *min, *max, *step, *int);
                return Some(Action::SetShaderControl(*layer, name, crate::shader_ui::slider_json(v, *default)));
            }
            // `RangeWidget.adjustViaWheel`: `stepUp` down the page.
            Control::Slider { layer, key, value, spec } => {
                return Some(Action::SetLayerOption(*layer, (*key).into(), spec.json(spec.stepped(*value, delta_y.signum()))));
            }
            // `EnumSelectWidget.adjustViaWheel`: the previous option down
            // the page, the next up it, wrapping.
            Control::Select { options, value, .. } => {
                let n = options.len();
                let k = options.iter().position(|o| o == value).unwrap_or(0);
                let k = if delta_y > 0.0 { (n + k - 1) % n } else { (k + 1) % n };
                return self.choose(k);
            }
            Control::ShaderSelect { labels, current, .. } => {
                let n = labels.len();
                let k = if delta_y > 0.0 { (n + current - 1) % n } else { (current + 1) % n };
                return self.choose(k);
            }
            _ => {}
        }
        let Control::RenderScale { layer, key, target, origin, invert_wheel } = self else { return None };
        let max = (origin + NUM_BINS as f64 * ng_core::render_scale::BIN_SIZE).round() - 1.0;
        let direction = if *invert_wheel { -delta_y.signum() } else { delta_y.signum() };
        let v = (target * direction.exp2()).clamp(origin.exp2(), max.exp2());
        Some(Action::SetLayerOption(*layer, (*key).into(), Some(ng_core::js::number(v))))
    }

    /// `reset` (double-click) on the render scale.
    pub(crate) fn reset(&self) -> Option<Action> {
        let Control::RenderScale { layer, key, .. } = self else { return None };
        Some(Action::SetLayerOption(*layer, (*key).into(), None))
    }

    /// A `<select>`'s options and current one.
    pub(crate) fn options(&self) -> Option<(Vec<&'static str>, &'static str)> {
        match self {
            Control::Select { options, value, .. } => Some((options.to_vec(), value)),
            Control::Boundary { value, .. } => Some((BOUNDARIES.to_vec(), value)),
            Control::ShaderSelect { labels, current, .. } => Some((labels.clone(), labels[*current])),
            _ => None,
        }
    }

    /// A `<select>`'s option `k` chosen.
    pub(crate) fn choose(&self, k: usize) -> Option<Action> {
        match self {
            Control::Boundary { dim, .. } => Some(Action::SetBoundary(*dim, BOUNDARIES.get(k)?.to_string())),
            Control::Select { layer, key, options, default, .. } => {
                let option = *options.get(k)?;
                let value = (option != *default).then(|| serde_json::Value::String(option.into()));
                Some(Action::SetLayerOption(*layer, (*key).into(), value))
            }
            // `TrackableValue`: the default left out.
            Control::ShaderSelect { layer, name, values, default, .. } => Some(Action::SetShaderControl(*layer, name, (k != *default).then(|| values.get(k).cloned()).flatten())),
            _ => None,
        }
    }
}

/// An open `<select>`'s options, under its box `r` (Chrome's popup is
/// the system's; this is a plain list): the option under the pointer, else
/// the current one, highlighted.
pub(crate) fn select_menu(b: &mut Builder, root: NodeId, r: threers::ui::Rect, options: &[&str], value: &str) {
    let hovered = (0..options.len()).find(|&k| b.hover(Key::MenuItem(k)));
    let current = options.iter().position(|o| *o == value);
    let font = style! { font-family: Arial; font-size: {SMALL}; color: black; };
    let (_, items) = controls::select_popup(&mut b.ui, root, r, options, current, hovered, &font);
    for (k, item) in items.into_iter().enumerate() {
        b.key(item, Key::MenuItem(k), None);
    }
}

impl Control {
    /// `adjustInvlerpContrast`: the range scaled about its middle.
    pub(crate) fn contrast(&self, scale: f64) -> Option<Action> {
        let Control::Cdf { layer, t, range, window, controls, ctl } = self else { return None };
        let r = [lerp(*t, *range, 0.5 - scale / 2.0), lerp(*t, *range, 0.5 + scale / 2.0)];
        Some(Action::SetLayerOption(*layer, "shaderControls".into(), invlerp_controls_json(controls.as_ref(), *t, *ctl, r, *window)))
    }

    /// An invlerp bound's text (`updateInputBoundValue`).
    pub(crate) fn bound_text(&self, window: bool, endpoint: usize) -> String {
        let Control::Cdf { range, window: w, .. } = self else { return String::new() };
        format_bound(if window { w[endpoint] } else { range[endpoint] })
    }

    /// An invlerp bound typed: `parseDataTypeValue`, then
    /// `getUpdatedRangeAndWindowParameters(…, fitRangeInWindow = true)`.
    /// `None` if the text is not a value of the type.
    pub(crate) fn set_bound(&self, window: bool, endpoint: usize, text: &str) -> Option<Action> {
        let Control::Cdf { layer, t, range, window: w, controls, ctl } = self else { return None };
        let value = ng_viewer::state::parse_data_type_value(*t, &serde_json::Value::String(text.into())).ok()?;
        let (mut range, mut w) = (*range, *w);
        if window {
            w[endpoint] = value;
            // Past the other bound: that one follows.
            let other = 1 - endpoint;
            if (value - w[other]) * (2.0 * endpoint as f64 - 1.0) < 0.0 {
                w[other] = value;
            }
        } else {
            range[endpoint] = value;
            // The window grows to hold it.
            if value < w[0] {
                w[0] = value;
            }
            if value > w[1] {
                w[1] = value;
            }
        }
        Some(Action::SetLayerOption(*layer, "shaderControls".into(), invlerp_controls_json(controls.as_ref(), *t, *ctl, range, w)))
    }
}

/// A colour member of layer `i`'s JSON (`#rrggbb`), if set.
pub(crate) fn layer_color(viewer: &Viewer, i: usize, key: &str) -> Option<[f32; 3]> {
    viewer.layers().get(i)?.spec.json.get(key)?.as_str().and_then(ng_viewer::state::parse_color)
}
