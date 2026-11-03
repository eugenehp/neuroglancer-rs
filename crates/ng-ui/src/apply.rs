//! Carrying out an [`Action`]: on the UI's own state, and on the viewer.

use super::*;

impl NgUi {
    /// Carries out `action` on `viewer`.
    pub fn apply(&mut self, viewer: &mut Viewer, action: &Action) -> Effect {
        match action {
            Action::Repaint | Action::Unported(_) => Effect::None,
            Action::TogglePanel(id) => {
                self.toggle_panel(viewer, *id);
                Effect::None
            }
            Action::ClosePanel(id) => {
                let mut l = id.location(viewer);
                l.visible = false;
                id.set_location(viewer, &l);
                Effect::None
            }
            Action::SetLocations(locations) => {
                for (id, l) in locations {
                    id.set_location(viewer, l);
                }
                Effect::None
            }
            Action::Toggle(setting) => {
                toggle_setting(viewer, *setting);
                Effect::None
            }
            Action::SetNumber(n, v) => {
                // `verifyPositiveInt`: anything else leaves the value.
                if *v > 0.0 && v.fract() == 0.0 {
                    match n {
                        NumberSetting::GpuMemoryLimit => viewer.settings.gpu_memory_limit = *v,
                        NumberSetting::SystemMemoryLimit => viewer.settings.system_memory_limit = *v,
                        NumberSetting::ConcurrentDownloads => viewer.settings.concurrent_downloads = *v,
                    }
                }
                if let Some((FieldId::Number(f), t)) = &mut self.field {
                    if f == n {
                        *t = TextField::new(number_to_string(number_setting(viewer, *n)));
                    }
                }
                Effect::None
            }
            Action::ToggleLayerBar => {
                let shown = panels::layer_bar_shown(viewer);
                let mut o = viewer.ui_state("uiControlVisibility").and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
                o.insert("showLayerPanel".into(), (!shown).into());
                viewer.set_ui_state("uiControlVisibility", Some(serde_json::Value::Object(o)));
                Effect::None
            }
            Action::SetArchived(i, archived) => {
                viewer.set_archived(*i, *archived);
                Effect::None
            }
            Action::ToggleLayerSidePanel(i) => {
                // Shown for this layer, or hidden if it already is.
                let shown = PanelId::Layer.location(viewer).visible && PanelId::Layer.layer(viewer) == Some(*i);
                if shown {
                    return self.apply(viewer, &Action::ClosePanel(PanelId::Layer));
                }
                if let Some(name) = viewer.layers().get(*i).map(|l| l.name().to_owned()) {
                    self.select_layer_named(viewer, &name, true);
                }
                Effect::None
            }
            Action::RenameLayer(i, name) => {
                let old = viewer.layers().get(*i).map(|l| l.name().to_owned());
                if let (Some(old), Some(new)) = (old, viewer.rename_layer(*i, name)) {
                    // `selectedLayer` names it.
                    if viewer.ui_state("selectedLayer").and_then(|v| v.get("layer")).and_then(serde_json::Value::as_str) == Some(old.as_str()) {
                        self.select_layer_named(viewer, &new, false);
                    }
                }
                if let Some((FieldId::LayerName(f), t)) = &mut self.field {
                    if f == i {
                        *t = TextField::new(viewer.layers().get(*i).map_or(String::new(), |l| l.name().to_owned()));
                    }
                }
                Effect::None
            }
            Action::SetTitle(t) => {
                viewer.settings.title = (!t.is_empty()).then(|| t.clone());
                Effect::None
            }
            Action::SetColor(setting, c) => {
                match setting {
                    ColorSetting::CrossSectionBackground => viewer.background = *c,
                    ColorSetting::ProjectionBackground => viewer.projection_background = *c,
                    ColorSetting::SegmentDefault(i) => viewer.set_layer_option(*i, "segmentDefaultColor", Some(ng_viewer::state::serialize_color(*c).into())),
                    ColorSetting::Annotation(i) => viewer.set_layer_option(*i, "annotationColor", Some(ng_viewer::state::serialize_color(*c).into())),
                    ColorSetting::ShaderColor(i, name) => {
                        // `TrackableRGB`: the default left out.
                        let Some(l) = viewer.layers().get(*i) else { return Effect::None };
                        let t = l.data_type().unwrap_or(ng_core::DataType::Uint8);
                        let text = l.spec.json.get("shader").and_then(serde_json::Value::as_str).unwrap_or(ng_viewer::state::DEFAULT_IMAGE_SHADER);
                        let parsed = ng_viewer::user_shader::parse_with_channels(text, t, l.channel_extents().len());
                        let default = parsed.controls.iter().find(|(n, _)| n == name).and_then(|(_, c)| match c {
                            ng_viewer::user_shader::Control::Color { default, .. } => Some(*default),
                            _ => None,
                        });
                        let hex = ng_viewer::state::serialize_color(*c);
                        let value = (default.map(ng_viewer::state::serialize_color).as_deref() != Some(hex.as_str())).then(|| hex.into());
                        let controls = shader_ui::set_control(l.spec.json.get("shaderControls"), name, value);
                        viewer.set_layer_option(*i, "shaderControls", controls);
                    }
                }
                Effect::None
            }
            Action::PickColor(setting) => Effect::PickColor {
                setting: *setting,
                current: match setting {
                    ColorSetting::CrossSectionBackground => viewer.background,
                    ColorSetting::ProjectionBackground => viewer.projection_background,
                    ColorSetting::SegmentDefault(i) => rendering::layer_color(viewer, *i, "segmentDefaultColor").unwrap_or([1.0, 0.0, 0.0]),
                    ColorSetting::Annotation(i) => viewer.layers().get(*i).map_or([1.0, 1.0, 0.0], |l| l.annotation_color()),
                    ColorSetting::ShaderColor(i, name) => viewer
                        .layers()
                        .get(*i)
                        .and_then(|l| {
                            let t = l.data_type()?;
                            let text = l.spec.json.get("shader").and_then(serde_json::Value::as_str).unwrap_or(ng_viewer::state::DEFAULT_IMAGE_SHADER);
                            let parsed = ng_viewer::user_shader::parse_with_channels(text, t, l.channel_extents().len());
                            let values = ng_viewer::user_shader::control_values(&parsed, t, l.spec.json.get("shaderControls"));
                            parsed.controls.iter().zip(values).find(|((n, _), _)| n == name).and_then(|(_, v)| match v {
                                ng_viewer::user_shader::ControlValue::Color(c) => Some(c),
                                _ => None,
                            })
                        })
                        .unwrap_or([1.0, 1.0, 1.0]),
                },
            },
            Action::Shortcut(name) => self.run_shortcut(viewer, name),
            Action::BindTool(key, owner, json) => {
                viewer.bind_tool(*key, *owner, json.clone());
                Effect::None
            }
            Action::UnbindTool(owner, json) => {
                viewer.unbind_tool(*owner, json);
                if self.active_tool.is_some_and(|k| viewer.tool(k).is_none()) {
                    self.active_tool = None;
                }
                Effect::None
            }
            Action::SelectSegments { layer, deselect, x, y } => {
                self.select_paint = Some(tools::SelectPaint { layer: *layer, deselect: *deselect, start: (*x, *y), painting: false, last: None });
                self.try_select_segment(viewer);
                Effect::None
            }
            Action::TogglePaletteList => {
                self.palette_list = !self.palette_list;
                Effect::None
            }
            Action::AddPalette { name, query } => {
                palettes::add_new(viewer, name, query);
                self.palette_list = false;
                Effect::None
            }
            Action::DeletePalette(k) => {
                palettes::remove(viewer, *k as usize);
                self.scroll.retain(|id, _| !matches!(id, PanelId::Palette(_)));
                Effect::None
            }
            Action::ConvertPalette(k) => {
                palettes::convert_to_explicit(viewer, *k as usize);
                Effect::None
            }
            Action::LayerDragOver(target) => self.layer_drag_over(viewer, *target),
            // `PositionWidget` over the channel space: the coordinate
            // floored, kept within the dimension; `channel` left out at
            // the directive's default.
            Action::SetShaderChannel(i, name, dim, text) => {
                let Some(l) = viewer.layers().get(*i) else { return Effect::None };
                let Some(t) = l.data_type() else { return Effect::None };
                let extents = l.channel_extents();
                let shader = l.spec.json.get("shader").and_then(serde_json::Value::as_str).unwrap_or(ng_viewer::state::DEFAULT_IMAGE_SHADER);
                let parsed = ng_viewer::user_shader::parse_with_channels(shader, t, extents.len());
                let values = ng_viewer::user_shader::control_values(&parsed, t, l.spec.json.get("shaderControls"));
                let Some(k) = parsed.controls.iter().position(|(n, _)| n == name) else { return Effect::None };
                let (ng_viewer::user_shader::Control::Invlerp { channel: default, .. }, ng_viewer::user_shader::ControlValue::Invlerp { channel, .. }) = (&parsed.controls[k].1, &values[k]) else { return Effect::None };
                let v = ng_core::js::parse_float(text.trim());
                let Some(&extent) = extents.get(*dim) else { return Effect::None };
                if !v.is_finite() || v < 0.0 || v >= extent as f64 {
                    return Effect::None;
                }
                let mut channel = channel.clone();
                channel[*dim] = v.floor() as u32;
                let controls = l.spec.json.get("shaderControls").cloned();
                let mut o = controls.as_ref().and_then(|c| c.get(*name)).and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
                if channel == *default {
                    o.remove("channel");
                } else {
                    o.insert("channel".into(), channel.iter().map(|&c| serde_json::Value::from(c)).collect());
                }
                let value = (!o.is_empty()).then_some(serde_json::Value::Object(o));
                let controls = shader_ui::set_control(controls.as_ref(), name, value);
                viewer.set_layer_option(*i, "shaderControls", controls);
                Effect::None
            }
            Action::SetShaderControl(i, name, value) => {
                let controls = viewer.layers().get(*i).and_then(|l| shader_ui::set_control(l.spec.json.get("shaderControls"), name, value.clone()));
                viewer.set_layer_option(*i, "shaderControls", controls);
                Effect::None
            }
            Action::RenameDimensions(names) => {
                viewer.rename_dimensions(names);
                Effect::None
            }
            Action::SetDimensionScales(scales) => {
                viewer.set_dimension_scales(scales);
                Effect::None
            }
            Action::RenamePalette(k, name) => {
                palettes::rename(viewer, *k as usize, name);
                Effect::None
            }
            Action::SetPaletteQuery(k, query) => {
                palettes::set_query(viewer, *k as usize, query);
                Effect::None
            }
            Action::RemovePaletteTool(k, index) => {
                palettes::remove_tool(viewer, *k as usize, *index);
                Effect::None
            }
            Action::ToolWheel { delta_y, modifiers } => {
                self.tool_wheel(viewer, *delta_y, *modifiers);
                Effect::None
            }
            Action::SelectLayerTab(i, tab) => {
                if let Some(l) = viewer.layers_mut().get_mut(*i) {
                    l.spec.json.insert("tab".into(), (*tab).into());
                }
                self.scroll.remove(&PanelId::Layer);
                Effect::None
            }
            Action::SelectPinnedTab(i, k, tab) => {
                layer_panel::set_pinned_tab(viewer, *i, *k, tab);
                self.scroll.remove(&PanelId::Pinned(*i as u16, *k as u16));
                Effect::None
            }
            Action::OpenUrl(url) => Effect::OpenUrl(url.clone()),
            Action::OpenShaderOverlay(i) => {
                self.shader_blur_pending = self.shader_focused();
                self.dialog = Some(dialogs::Dialog::ShaderCode(*i));
                Effect::None
            }
            Action::PinLayerPanel => {
                layer_panel::pin(viewer);
                Effect::None
            }
            Action::UnpinLayerPanel(i, k) => {
                layer_panel::unpin(viewer, *i, *k);
                Effect::None
            }
            Action::ToggleDefaultSubsources(i, s) => {
                if let Some(l) = viewer.layers_mut().get_mut(*i) {
                    let on = l.source_views().get(*s).is_some_and(|v| v.enable_default_subsources);
                    l.set_default_subsources(*s, !on);
                }
                Effect::None
            }
            Action::ToggleSubsource(i, s, k) => {
                if let Some(l) = viewer.layers_mut().get_mut(*i) {
                    let on = l.source_views().get(*s).and_then(|v| v.subsources.get(*k).map(|x| x.enabled)).unwrap_or(false);
                    l.set_subsource_enabled(*s, *k, !on);
                }
                Effect::None
            }
            Action::SelectPosition { pin } => {
                let mut l = PanelId::Selection.location(viewer);
                l.visible = true;
                PanelId::Selection.set_location(viewer, &l);
                self.selection_pinned = *pin;
                if *pin {
                    if let Some(sel) = panels::Selection::capture(viewer) {
                        sel.store(viewer);
                        // `history`: after the current entry.
                        self.selection_history.truncate(self.selection_index + 1);
                        self.selection_history.push(sel);
                        self.selection_index = self.selection_history.len() - 1;
                    }
                }
                Effect::None
            }
            Action::SelectionHistory(dir) => {
                let i = self.selection_index as i32 + dir;
                if i >= 0 && (i as usize) < self.selection_history.len() {
                    self.selection_index = i as usize;
                    self.selection_pinned = true;
                    self.selection_history[i as usize].clone().store(viewer);
                }
                Effect::None
            }
            Action::ToggleSelectionPin => {
                self.selection_pinned = !self.selection_pinned;
                if self.selection_pinned {
                    if let Some(sel) = panels::Selection::capture(viewer) {
                        sel.store(viewer);
                    }
                }
                Effect::None
            }
            Action::MoveTo(p) => {
                viewer.navigation_mut().set_position(p);
                Effect::None
            }
            Action::CopyText(t) => Effect::Clipboard(t.clone()),
            Action::ToggleSegment(layer, id) => {
                if let Some(l) = viewer.layers_mut().get_mut(*layer) {
                    let visible = panels::segment_visible(l, *id);
                    let mut segs: Vec<serde_json::Value> = l.spec.json.get("segments").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                    segs.retain(|v| {
                        let s = v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string());
                        s.trim_start_matches('!') != id.to_string()
                    });
                    // Hidden ones stay listed with `!` (the ordered set keeps them).
                    segs.push(if visible { format!("!{id}") } else { id.to_string() }.into());
                    l.spec.json.insert("segments".into(), serde_json::Value::Array(segs));
                }
                viewer.refresh_segments(*layer);
                Effect::None
            }
            Action::ToggleLayerVisible(i) => {
                if let Some(l) = viewer.layers_mut().get_mut(*i) {
                    l.spec.visible = !l.spec.visible;
                }
                Effect::None
            }
            Action::TogglePick(i) => {
                if let Some(l) = viewer.layers_mut().get_mut(*i) {
                    // `pickEnabled`, saved as `"pick": false` when off.
                    let enabled = l.spec.json.get("pick").and_then(serde_json::Value::as_bool).unwrap_or(true);
                    if enabled {
                        l.spec.json.insert("pick".into(), false.into());
                    } else {
                        l.spec.json.remove("pick");
                    }
                }
                Effect::None
            }
            Action::SetVelocity(dim, v) => {
                viewer.set_velocity(*dim, *v);
                Effect::None
            }
            Action::TogglePlayback(dim) => {
                // `togglePlayback`: enabled if it was not, then (un)paused.
                let v = viewer.velocity(*dim).unwrap_or_default();
                viewer.set_velocity(*dim, Some(ng_viewer::playback::DimensionVelocity { paused: !v.paused, ..v }));
                Effect::None
            }
            Action::MultiplyVelocity(dim, factor) => {
                let v = viewer.velocity(*dim).unwrap_or_default();
                viewer.set_velocity(*dim, Some(ng_viewer::playback::DimensionVelocity { velocity: ng_viewer::playback::multiply(v.velocity, *factor), ..v }));
                Effect::None
            }
            Action::SetBoundary(dim, name) => {
                if let Some(v) = viewer.velocity(*dim) {
                    let at_boundary = ng_viewer::playback::Boundary::ALL.into_iter().find(|b| b.name() == name).unwrap_or(v.at_boundary);
                    viewer.set_velocity(*dim, Some(ng_viewer::playback::DimensionVelocity { at_boundary, ..v }));
                }
                Effect::None
            }
            Action::AddLayer => {
                let i = viewer.add_new_layer();
                let name = viewer.layers()[i].name().to_owned();
                self.select_layer_named(viewer, &name, true);
                self.field = Some((FieldId::SourceUrl(i), TextField::new("")));
                self.caret_since = self.now;
                Effect::None
            }
            Action::SetLayerSource(i, url) => {
                // A new layer takes its name from the URL (`suggestLayerName`).
                if viewer.layers().get(*i).is_some_and(|l| l.spec.layer_type == "new") {
                    if let Some(name) = sources::suggest_layer_name(url) {
                        if let (Some(old), Some(new)) = (viewer.layers().get(*i).map(|l| l.name().to_owned()), viewer.rename_layer(*i, &name)) {
                            if viewer.ui_state("selectedLayer").and_then(|v| v.get("layer")).and_then(serde_json::Value::as_str) == Some(old.as_str()) {
                                self.select_layer_named(viewer, &new, false);
                            }
                        }
                    }
                }
                Effect::LoadSources(viewer.set_layer_source(*i, url))
            }
            Action::Complete(value) => {
                if let Some((FieldId::SourceUrl(_), field)) = &mut self.field {
                    *field = TextField::new(value.clone());
                }
                self.completion_active = None;
                Effect::None
            }
            Action::ChangeLayerType(i, t) => {
                viewer.change_layer_type(*i, t);
                Effect::None
            }
            Action::Annotate => self.annotate(viewer),
            Action::AnnotationStep { finish } => self.annotation_step(viewer, *finish),
            Action::PinAnnotation(i, n) => {
                let rank = viewer.dimensions().rank();
                if let Some(e) = viewer.layers().get(*i).and_then(|l| l.annotation_list(rank)).and_then(|l| l.into_iter().nth(*n)) {
                    self.select_annotation(viewer, *i, &e.id);
                }
                Effect::None
            }
            Action::MoveToAnnotation(i, n) => {
                // `getCenterPosition`: a point's or ellipsoid's first row; a
                // line's, box's or polyline's first two rows' middle.
                let rank = viewer.dimensions().rank();
                if let Some(e) = viewer.layers().get(*i).and_then(|l| l.annotation_list(rank)).and_then(|l| l.into_iter().nth(*n)) {
                    let two = matches!(e.icon, "ꕹ" | "❑" | "⤤") && e.rows.len() >= 2;
                    let mut position = viewer.navigation().position().to_vec();
                    for (d, p) in position.iter_mut().enumerate() {
                        let a = e.rows[0].get(d).copied().flatten();
                        let v = if two { a.zip(e.rows[1].get(d).copied().flatten()).map(|(a, b)| (a + b) / 2.0) } else { a };
                        if let Some(v) = v {
                            *p = v as f32;
                        }
                    }
                    viewer.navigation_mut().set_position(&position);
                }
                Effect::None
            }
            Action::DeleteAnnotation(i, id) => {
                viewer.delete_annotation(*i, id);
                Effect::None
            }
            Action::MoveToSegment(layer, id) => {
                if !viewer.move_to_segment(*layer, *id) {
                    // `showTemporaryMessage`'s default: 2 s.
                    self.temporary_messages.push((format!("No position information loaded for segment {id}"), self.now + 2000.0));
                }
                Effect::None
            }
            Action::ToggleHoveredSegments { star } => {
                // `LayerActionContext.segmentationToggleSegmentState`: once
                // any layer turns its segment on, only those turning one on
                // act.
                let hovered = viewer.hovered_segments();
                let mut toggles = vec![];
                let mut state = None;
                for (i, l) in viewer.layers().iter().enumerate() {
                    let picks = l.spec.json.get("pick").and_then(serde_json::Value::as_bool).unwrap_or(true);
                    let Some(Some(id)) = hovered.get(i).copied() else { continue };
                    let Ok((display, segs)) = ng_viewer::state::segmentation_settings(&l.spec.json) else { continue };
                    if !l.spec.visible || !picks || (id == 0 && display.hide_segment_zero) {
                        continue;
                    }
                    let on = if *star { !segs.iter().any(|s| s.0 == id) } else { !segs.iter().any(|s| s.0 == id && s.1) };
                    if on || state.is_none() {
                        state = Some(on);
                    }
                    toggles.push((i, id, on));
                }
                for (i, id, on) in toggles {
                    if Some(on) == state {
                        let change = if *star { segments::SegmentChange::Star(vec![id], on) } else { segments::SegmentChange::Show(vec![id], on) };
                        self.apply(viewer, &Action::SegmentChange(i, change));
                    }
                }
                Effect::None
            }
            Action::SegmentChange(i, change) => {
                if let Some(l) = viewer.layers().get(*i) {
                    let after = segments::apply_change(segments::starred(l), change);
                    viewer.set_layer_option(*i, "segments", (!after.is_empty()).then(|| segments::segments_json(&after)));
                    viewer.refresh_segments(*i);
                }
                Effect::None
            }
            Action::AutoRange { layer, ctl, range, lower, upper } => {
                let Some(l) = viewer.layers().get(*layer) else { return Effect::None };
                let Some(t) = l.data_type() else { return Effect::None };
                let type_range = ng_viewer::default_data_type_range(t).as_f64();
                let (range, window) = ng_viewer::histogram::auto_range(l.histogram_samples(), *lower, *upper, t, type_range, range[0] > range[1]);
                let controls = rendering::invlerp_controls_json(l.spec.json.get("shaderControls"), t, *ctl, range, window);
                viewer.set_layer_option(*layer, "shaderControls", controls);
                Effect::None
            }
            Action::EditJsonState
            | Action::CloseDialog
            | Action::ToggleFold(_)
            | Action::ApplyState
            | Action::DownloadState
            | Action::OpenScreenshot
            | Action::ScreenshotScale(_)
            | Action::ScreenshotKeepFov
            | Action::TakeScreenshot
            | Action::CancelScreenshot
            | Action::OpenPalette
            | Action::PaletteRun(_)
            | Action::PaletteBack
            | Action::PaletteCommand(_) => dialogs::apply(self, viewer, action),
            Action::RandomizeSeed(i) => {
                // A splitmix64 step of the time: a new 32-bit seed.
                let mut z = (self.now * 1e3) as u64 ^ 0x9e37_79b9_7f4a_7c15;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                viewer.set_layer_option(*i, "colorSeed", Some(((z ^ (z >> 31)) as u32).into()));
                Effect::None
            }
            // The layer type select: its abbreviation chosen.
            Action::SetLayerOption(i, key, Some(serde_json::Value::String(abbreviation))) if key == "type" => {
                if let Some((t, _)) = layer_panel::LAYER_TYPES.iter().find(|(_, a)| a == abbreviation) {
                    viewer.change_layer_type(*i, t);
                }
                Effect::None
            }
            Action::SetLayerOption(i, key, value) => {
                viewer.set_layer_option(*i, key, value.clone());
                Effect::None
            }
            Action::SelectLayer(i) => {
                if let Some(name) = viewer.layers().get(*i).map(|l| l.name().to_owned()) {
                    self.select_layer_named(viewer, &name, true);
                }
                Effect::None
            }
            Action::ArchiveLayer(i) => {
                viewer.set_archived(*i, true);
                Effect::None
            }
            Action::DeleteLayer(i) => {
                viewer.remove_layer(*i);
                Effect::None
            }
            Action::SetLayout(layout) => {
                // The button had focus; it goes with the old panels.
                self.refocus = viewer.layout != *layout;
                viewer.layout = *layout;
                Effect::None
            }
            Action::ToggleShowSlices => {
                viewer.show_slices = !viewer.show_slices;
                Effect::None
            }
            Action::CopyPosition => {
                if viewer.navigation().valid() {
                    Effect::Clipboard(format_position_text(viewer.navigation().position()))
                } else {
                    Effect::Clipboard(String::new())
                }
            }
            Action::CopyUrl => Effect::CopyUrl,
            Action::SetCoordinates(coordinates) => {
                for c in coordinates {
                    self.set_coordinate(viewer, c.dim, |space, _| {
                        if c.to_voxel_center && !space.bounds.voxel_center_at_integer_coordinates[c.dim] {
                            c.value + 0.5
                        } else {
                            c.value
                        }
                    });
                }
                Effect::None
            }
            Action::Many(actions) => {
                let mut effect = Effect::None;
                for a in actions {
                    let e = self.apply(viewer, a);
                    if e != Effect::None {
                        effect = e;
                    }
                }
                effect
            }
            Action::AdjustCoordinate(dim, delta) => {
                self.set_coordinate(viewer, *dim, |space, current| space.bounds.clamp_and_round_to_voxel_center(*dim, current + delta));
                Effect::None
            }
        }
    }

    /// A top-bar button: a panel shown or hidden. The layer panel shows the
    /// selected layer, or the first if none is (`selectedLayer.visible`).
    fn toggle_panel(&mut self, viewer: &mut Viewer, id: PanelId) {
        let mut l = id.location(viewer);
        l.visible = !l.visible;
        if id == PanelId::Layer && l.visible && id.layer(viewer).is_none() {
            let Some(first) = viewer.layers().iter().find(|l| !l.spec.archived).map(|l| l.name().to_owned()) else { return };
            self.select_layer_named(viewer, &first, true);
            return;
        }
        id.set_location(viewer, &l);
    }

    /// `selectedLayer.layer = …` (and shown, if `show`).
    fn select_layer_named(&mut self, viewer: &mut Viewer, name: &str, show: bool) {
        let mut o = viewer.ui_state("selectedLayer").and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
        o.insert("layer".into(), name.into());
        viewer.set_ui_state("selectedLayer", Some(serde_json::Value::Object(o)));
        if show {
            let mut l = PanelId::Layer.location(viewer);
            l.visible = true;
            PanelId::Layer.set_location(viewer, &l);
        }
    }

    /// The `n`th layer not archived (`getLayerByNonArchivedIndex`).
    fn nth_layer(viewer: &Viewer, n: usize) -> Option<usize> {
        viewer.layers().iter().enumerate().filter(|(_, l)| !l.spec.archived).nth(n).map(|(i, _)| i)
    }

    /// A global action (`Viewer.registerActionListeners`, the data panel
    /// layout's `toggle-layout`).
    fn run_shortcut(&mut self, viewer: &mut Viewer, name: &str) -> Effect {
        if let Some(key) = name.strip_prefix("tool-").and_then(|k| k.chars().next()) {
            self.activate_tool(viewer, key);
            return Effect::None;
        }
        let digit = |prefix: &str| name.strip_prefix(prefix).and_then(|d| d.parse::<usize>().ok());
        if let Some(n) = digit("toggle-layer-") {
            if let Some(i) = Self::nth_layer(viewer, n - 1) {
                return self.apply(viewer, &Action::ToggleLayerVisible(i));
            }
        } else if let Some(n) = digit("toggle-pick-layer-") {
            if let Some(i) = Self::nth_layer(viewer, n - 1) {
                return self.apply(viewer, &Action::TogglePick(i));
            }
        } else if let Some(n) = digit("select-layer-") {
            if let Some(i) = Self::nth_layer(viewer, n - 1) {
                let name = viewer.layers()[i].name().to_owned();
                self.select_layer_named(viewer, &name, true);
            }
        } else {
            match name {
                "toggle-axis-lines" => toggle_setting(viewer, Setting::ShowAxisLines),
                "toggle-scale-bar" => toggle_setting(viewer, Setting::ShowScaleBar),
                "toggle-default-annotations" => toggle_setting(viewer, Setting::ShowDefaultAnnotations),
                "toggle-show-slices" => toggle_setting(viewer, Setting::ShowSlices),
                "toggle-orthographic-projection" => toggle_setting(viewer, Setting::OrthographicProjection),
                "toggle-show-statistics" => toggle_setting(viewer, Setting::ShowStatistics),
                "help" => self.toggle_panel(viewer, PanelId::Help),
                "recolor" => {
                    let seed = || {
                        // `getRandomUint32`.
                        let mut b = [0u8; 4];
                        getrandom_fill(&mut b);
                        u32::from_le_bytes(b)
                    };
                    viewer.recolor(seed);
                }
                "clear-segments" => viewer.clear_segments(),
                // The focused panel's related layouts (`registerRelatedLayouts`).
                "toggle-layout" | "toggle-layout-alternative" => {
                    if let Some(p) = self.focused {
                        let panel = viewer.logical_panel_rects().get(p).map(|(panel, _)| *panel);
                        if let Some(panel) = panel {
                            let related = view::related_layouts(viewer.layout, panel);
                            let i = usize::from(name == "toggle-layout-alternative").min(related.len().saturating_sub(1));
                            if let Some(&target) = related.get(i) {
                                return self.apply(viewer, &Action::SetLayout(target));
                            }
                        }
                    }
                }
                "add-layer" => return self.apply(viewer, &Action::AddLayer),
                "open-command-palette" => return self.apply(viewer, &Action::OpenPalette),
                _ => {}
            }
        }
        Effect::None
    }

    fn set_coordinate(&mut self, viewer: &mut Viewer, dim: usize, f: impl FnOnce(&ng_core::CoordinateSpace, f64) -> f64) {
        if !viewer.navigation().valid() || dim >= viewer.dimensions().rank() {
            return;
        }
        let space = viewer.dimensions().clone();
        let mut p = viewer.navigation().position().to_vec();
        p[dim] = f(&space, p[dim] as f64) as f32;
        viewer.navigation_mut().set_position(&p);
        // The fields show the new position (`updateView` sets their values,
        // which leaves the caret at the end).
        self.pending.clear();
        if let Some(e) = &mut self.edit {
            e.text = coordinate_text(p[e.dim]);
            let n = e.text.chars().count();
            (e.start, e.end, e.modified) = (n, n, false);
        }
    }
}

/// A numeric setting's value.
pub(crate) fn number_setting(viewer: &Viewer, n: NumberSetting) -> f64 {
    match n {
        NumberSetting::GpuMemoryLimit => viewer.settings.gpu_memory_limit,
        NumberSetting::SystemMemoryLimit => viewer.settings.system_memory_limit,
        NumberSetting::ConcurrentDownloads => viewer.settings.concurrent_downloads,
    }
}

/// Flips a boolean setting.
fn toggle_setting(viewer: &mut Viewer, s: Setting) {
    match s {
        Setting::ShowAxisLines => viewer.show_axis_lines ^= true,
        Setting::ShowScaleBar => viewer.show_scale_bar ^= true,
        Setting::ShowSlices => viewer.show_slices ^= true,
        Setting::HideCrossSectionBackground3D => viewer.settings.hide_cross_section_background_3d ^= true,
        Setting::ShowDefaultAnnotations => viewer.settings.show_default_annotations ^= true,
        Setting::WireFrame => viewer.settings.wire_frame ^= true,
        Setting::Prefetch => viewer.settings.prefetch ^= true,
        Setting::AdaptiveDownsampling => viewer.settings.enable_adaptive_downsampling ^= true,
        Setting::OrthographicProjection => viewer.orthographic_projection ^= true,
        // The statistics panel's location (`statistics`).
        Setting::ShowStatistics => {
            let visible = viewer.ui_state("statistics").and_then(|v| v.get("visible")).and_then(serde_json::Value::as_bool).unwrap_or(false);
            let mut o = viewer.ui_state("statistics").and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
            if visible {
                o.remove("visible");
            } else {
                o.insert("visible".into(), true.into());
            }
            viewer.set_ui_state("statistics", (!o.is_empty()).then_some(serde_json::Value::Object(o)));
        }
    }
}

/// Random bytes (colour seeds, annotation IDs): `Math.random` in the
/// browser; natively the clock's nanoseconds, mixed with a fresh random
/// hasher per 8 bytes.
pub(crate) fn getrandom_fill(b: &mut [u8]) {
    #[cfg(target_arch = "wasm32")]
    for x in b.iter_mut() {
        *x = (js_sys::Math::random() * 256.0) as u8;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::hash::{BuildHasher, Hasher};
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        for (k, chunk) in b.chunks_mut(8).enumerate() {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u128(nanos);
            h.write_usize(k);
            for (x, y) in chunk.iter_mut().zip(h.finish().to_le_bytes()) {
                *x = y;
            }
        }
    }
}
