//! Pointer, wheel and key input over the bars and panels: what is hit, and
//! the [`Action`] it makes.

use super::*;

impl NgUi {
    /// The pointer moved to `(x, y)`, logical pixels. Returns an action
    /// when the UI changed: hover styles, the plot, focus — or the position,
    /// while dragging on a plot.
    pub fn pointer_move(&mut self, x: f32, y: f32) -> Option<Action> {
        self.pointer = Some((x, y));
        if let Some(field) = self.picker_drag {
            return self.picker_set(field, x, y);
        }
        // A layer dragged (past Chrome's 4px): over another item it moves
        // there, over the bar's empty end to the end (`dragenter`).
        if let Some(drag) = &mut self.layer_drag {
            let (dx, dy) = (x - drag.start.0, y - drag.start.1);
            if !drag.dragging && dx.abs().max(dy.abs()) > 4.0 {
                drag.dragging = true;
            }
            if drag.dragging {
                let keys = self.keys_at(x, y);
                // The drop zone: the bar past its items and `+`.
                let over_add = keys.iter().any(|k| matches!(k, Key::Icon(view::ADD_LAYER_ICON)));
                let target = keys.iter().copied().find(|k| matches!(k, Key::LayerItem(_))).or_else(|| keys.iter().copied().find(|k| matches!(k, Key::LayerPanel)).filter(|_| !over_add));
                if let Some(drag) = &mut self.layer_drag {
                    if target.is_some() && target != drag.target {
                        drag.target = target;
                        let to = match target {
                            Some(Key::LayerItem(i)) => Some(i),
                            _ => None,
                        };
                        return Some(Action::LayerDragOver(to));
                    }
                    if target.is_none() {
                        drag.target = None;
                    }
                }
                return Some(Action::Repaint);
            }
        }
        // The select tool paints once the pointer has moved 5px.
        if let Some(p) = &mut self.select_paint {
            let (dx, dy) = (x - p.start.0, y - p.start.1);
            if !p.painting && dx * dx + dy * dy > 25.0 {
                p.painting = true;
            }
        }
        if let Some(drag) = &self.gutter_drag {
            return Some(match drag {
                GutterDrag::Column { side, x0, size0, panels } => {
                    // `size -= outwards · delta`, at least the minimum.
                    let min = panels.iter().map(|(_, l)| l.min_size).fold(0.0, f64::max);
                    let at = if side.horizontal() { y } else { x };
                    let size = (size0 - (side.outwards() * (at - x0)) as f64).round().max(min);
                    Action::SetLocations(panels.iter().map(|(id, l)| (*id, Location { size, ..l.clone() })).collect())
                }
                GutterDrag::Split { first, second, top, bottom, across } => {
                    let at = if *across { x } else { y };
                    let fraction = ((at - top) / (bottom - top)).clamp(0.1, 0.9) as f64;
                    let sum = first.1.flex + second.1.flex;
                    let round = |v: f64| ng_core::js::round(v * 100.0) / 100.0;
                    Action::SetLocations(vec![
                        (first.0, Location { flex: round(fraction * sum), ..first.1.clone() }),
                        (second.0, Location { flex: round((1.0 - fraction) * sum), ..second.1.clone() }),
                    ])
                }
            });
        }
        if let Some((key, endpoint)) = self.control_drag {
            let (Some(node), Some(control)) = (self.node_of(key), self.controls.get(&key)) else { return None };
            return control.drag(self.ui.rect(node), x, endpoint);
        }
        let keys = self.keys_at(x, y);
        let mut changed = keys != self.hovered;
        if changed {
            // A tooltip shown goes; the next waits for the pointer to rest.
            changed |= self.tooltip().is_some();
            self.hover_since = self.now;
        }
        self.hovered = keys;
        // The dropdown is over the panels, not in them.
        let inside = if self.hovered.contains(&Key::Dropdown) { None } else { self.panel_at(x, y) };
        if inside != self.entered {
            self.entered = inside;
            if inside.is_some() && inside != self.focused && !self.input_focused {
                self.focused = inside;
                changed = true;
            }
        }
        if self.plot_drag {
            // `startRelativeMouseDrag`: the plot follows the pointer anywhere.
            self.plot_hover = None;
            self.plot_dragged = true;
            return self.plot_coordinate(y).map(|(dim, c)| Action::SetCoordinates(vec![Coordinate { dim, value: c, to_voxel_center: true }]));
        }
        let hover = if self.hovered.contains(&Key::Plot) { self.plot_coordinate(y).map(|(_, c)| c) } else { None };
        if hover != self.plot_hover {
            self.plot_hover = hover;
            changed = true;
        }
        changed.then_some(Action::Repaint)
    }

    pub fn pointer_leave(&mut self) -> bool {
        self.pointer = None;
        self.entered = None;
        self.plot_hover = None;
        !std::mem::take(&mut self.hovered).is_empty()
    }

    /// Whether `(x, y)` is over part of the UI rather than a bare panel: the
    /// viewer should then see the pointer leave (neuroglancer's panels track
    /// the mouse only when they are the event's target).
    pub fn covers(&self, x: f32, y: f32) -> bool {
        self.dialog.is_some() || !self.keys_at(x, y).is_empty()
    }

    /// Whether `(x, y)` is over a control, which takes presses and the wheel.
    pub fn captures(&self, x: f32, y: f32) -> bool {
        // A dialog's shade covers the viewer.
        if self.dialog.is_some() {
            return true;
        }
        self.keys_at(x, y).iter().any(|k| {
            self.actions.contains_key(k)
                || matches!(k, Key::Dimension(_) | Key::Dropdown | Key::LayerPanel | Key::Panel(_) | Key::Gutter(..) | Key::FlexGutter(..) | Key::Field(_))
        })
    }

    /// The node a key marks, as last laid out.
    pub(crate) fn node_of(&self, key: Key) -> Option<NodeId> {
        self.keys.iter().find(|(_, k)| **k == key).map(|(n, _)| *n)
    }

    /// A click on an invlerp row (`<label>`) off its inputs and buttons:
    /// the lower range bound focused, its text selected. Whether it was.
    fn label_click(&mut self, keys: &[Key]) -> bool {
        let Some(i) = keys.iter().find_map(|k| if let Key::Control(i, "invlerpRow") = k { Some(*i) } else { None }) else { return false };
        let interactive = keys.iter().any(|k| matches!(k, Key::Field(_) | Key::Control(_, "minMax" | "p1" | "p5")));
        if interactive {
            return false;
        }
        let Some(c) = self.controls.get(&Key::Control(i, "cdf")) else { return false };
        self.field = Some((FieldId::Bound(i, false, 0), TextField::selected(c.bound_text(false, 0))));
        self.caret_since = self.now;
        true
    }

    /// A press on a data panel that the UI acts on: `select-position` and
    /// `unpin-selected-position` (`getDefaultSelectBindings`). Call before
    /// handing the press to the viewer; `Some` means the UI took it.
    pub fn data_panel_press(&mut self, button: Button, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        // The active tool's bindings first (they are the panels' parents,
        // at the highest priority).
        if let Some(a) = self.tool_press(button, modifiers, viewer) {
            return Some(a);
        }
        let key = format!("mousedown{}", button as u8);
        match ng_viewer::input::lookup(ng_viewer::input::PanelKind::Slice, &key, modifiers, viewer.mac) {
            Some("select-position") => Some(Action::SelectPosition { pin: true }),
            Some("annotate") => Some(Action::Annotate),
            Some("unpin-selected-position") => Some(Action::SelectPosition { pin: false }),
            _ => None,
        }
    }

    /// A double-click on a data panel (`dblclick0`): `select` or `star`
    /// the segments under the pointer. Call when the UI's own
    /// [`NgUi::double_click`] did not take it and the pointer is over a
    /// panel.
    pub fn data_panel_double_click(&mut self, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        match ng_viewer::input::lookup(ng_viewer::input::PanelKind::Slice, "dblclick0", modifiers, viewer.mac) {
            Some("select") => Some(Action::ToggleHoveredSegments { star: false }),
            Some("star") => Some(Action::ToggleHoveredSegments { star: true }),
            _ => None,
        }
    }

    /// A button press at `(x, y)`; `None` leaves it to the viewer.
    pub fn pointer_down(&mut self, x: f32, y: f32, button: Button, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        // A press outside the data panels takes the focus from the one
        // that had it (`AutomaticallyFocusedElement` focuses again only on
        // the next `mouseenter`).
        let keys = self.keys_at(x, y);
        let blurred = self.focused.is_some() && (self.panel_at(x, y).is_none() || keys.contains(&Key::Dropdown));
        if blurred {
            self.focused = None;
        }
        self.input_focused = keys.iter().any(|k| matches!(k, Key::Field(_)) || matches!(self.controls.get(k), Some(rendering::Control::Slider { .. } | rendering::Control::ShaderSlider { .. })));
        // A press in a panel focuses it (`pointerdown`).
        if let Some(panel) = self.panel_at(x, y).filter(|_| !keys.contains(&Key::Dropdown) && self.focused.is_none()) {
            self.focused = Some(panel);
        }
        self.press(x, y, button, modifiers, viewer).or(blurred.then_some(Action::Repaint))
    }

    fn press(&mut self, x: f32, y: f32, button: Button, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        let keys = self.keys_at(x, y);
        // A segment entry's plain right-click (`onMousedown`): to the segment.
        let plain = !modifiers.control && !modifiers.alt && !modifiers.meta && !modifiers.shift;
        if button == Button::Secondary && plain && self.dialog.is_none() {
            if let Some((layer, id)) = keys.iter().find_map(|k| if let Key::SegmentEntry(l, id) = k { Some((*l, *id)) } else { None }) {
                return Some(Action::MoveToSegment(layer, id));
            }
            // An annotation layer's entry (`mousedown2`: move-to-annotation).
            if let Some((layer, n)) = keys.iter().find_map(|k| if let Key::AnnotationEntry(l, n) = k { Some((*l, *n)) } else { None }) {
                if viewer.layers().get(layer).is_some_and(|l| l.spec.layer_type == "annotation") {
                    return Some(Action::MoveToAnnotation(layer, n));
                }
            }
        }
        // The colour picker: its field and hue strip drag; a press outside
        // closes it.
        if self.color_picker.is_some() {
            for (key, field) in [(Key::ColorPickerField, true), (Key::ColorPickerHue, false)] {
                if keys.contains(&key) && button == Button::Primary {
                    self.picker_drag = Some(field);
                    return self.picker_set(field, x, y);
                }
            }
            if !keys.contains(&Key::ColorPicker) {
                self.color_picker = None;
            }
            return Some(Action::Repaint);
        }
        if self.dialog.is_some() {
            return if button == Button::Primary { self.dialog_press(viewer, &keys, x, y, modifiers.shift) } else { Some(Action::Repaint) };
        }
        // The shader editor: a press on it places the caret; elsewhere, it
        // loses the keyboard (what was typed is set).
        if button == Button::Primary && keys.iter().any(|k| matches!(k, Key::ShaderEditor(_) | Key::ShaderLine(..))) {
            if let Some(a) = self.shader_press(viewer, &keys, x, y, modifiers.shift) {
                return Some(a);
            }
        } else if self.shader_focused() {
            self.shader_blur_pending = true;
        }
        // The wrench's list closes on a press outside it and the wrench
        // (`focusout`).
        if self.palette_list && !keys.iter().any(|k| matches!(k, Key::PaletteDropdown | Key::Icon(0))) {
            self.palette_list = false;
        }
        // A tool's binding widget held: the next letter binds the tool.
        if button == Button::Primary {
            if let Some(tool) = keys.iter().find_map(|k| match k {
                Key::ToolBinding(n) | Key::ToolStatus(n) => self.tool_refs.get(*n as usize),
                _ => None,
            }) {
                self.binding_tool = Some(tool.clone());
                return Some(Action::Repaint);
            }
        }

        // A focused field loses focus to a press elsewhere (`change`).
        let mut committed = None;
        if let Some((id, _)) = &self.field {
            if !keys.contains(&Key::Field(*id)) {
                committed = self.commit_field();
                self.field = None;
            }
        }
        // A completion chosen (inside its field, which keeps the focus).
        if let Some(a) = keys.iter().find_map(|k| if matches!(k, Key::Completion(_)) { self.actions.get(k).cloned() } else { None }) {
            return Some(a);
        }
        // An open `<select>`: an option chosen, or the menu closed.
        if let Some(menu) = self.menu.take() {
            if let Some(k) = keys.iter().find_map(|k| if let Key::MenuItem(k) = k { Some(*k) } else { None }) {
                if self.controls.get(&menu).and_then(rendering::Control::options).is_some_and(|(options, _)| k < options.len()) {
                    return self.controls[&menu].choose(k).or(Some(Action::Repaint));
                }
            }
            if keys.contains(&menu) {
                return Some(Action::Repaint);
            }
        }
        // Layer controls: a `<select>` opens; the others take the press
        // (and maybe a drag).
        if button == Button::Primary {
            if let Some(key) = keys.iter().copied().find(|k| matches!(k, Key::Control(..)) && self.controls.contains_key(k)) {
                if self.controls.get(&key).and_then(rendering::Control::options).is_some() {
                    self.menu = Some(key);
                    return Some(Action::Repaint);
                }
                let r = self.ui.rect(self.node_of(key)?);
                if let Some((action, drag)) = self.controls[&key].press(r, x) {
                    self.control_drag = drag.map(|e| (key, e));
                    self.label_click(&keys);
                    return Some(action);
                }
            }
            // A click elsewhere on the invlerp control's `<label>` focuses its
            // first input, the range's lower bound, text selected.
            if self.label_click(&keys) {
                return Some(Action::Repaint);
            }
        }
        let early = self.panel_press(&keys, x, y, button, modifiers, viewer);
        match (committed, early) {
            (c, Some(a)) => Some(match c {
                Some(c) => Action::Many(vec![c, a]),
                None => a,
            }),
            (Some(c), None) if keys.iter().any(|k| matches!(k, Key::Panel(_))) => Some(c),
            (c, None) => {
                let rest = self.pointer_down_bars(&keys, x, y, button, modifiers, viewer);
                match (c, rest) {
                    (Some(c), Some(r)) => Some(Action::Many(vec![c, r])),
                    (c, r) => c.or(r),
                }
            }
        }
    }

    /// A press on the side panels: gutters, fields, colour inputs; `None`
    /// for the rest.
    fn panel_press(&mut self, keys: &[Key], x: f32, y: f32, button: Button, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        // A Layers panel row: right-click or Ctrl/⌘+click toggles the
        // layer's panel, Alt+click its picking (`LayerListItem`).
        if let Some(i) = keys.iter().find_map(|k| if let Key::ListItem(i) = k { Some(*i) } else { None }) {
            if button == Button::Secondary || (button == Button::Primary && (modifiers.control || modifiers.meta)) {
                return Some(Action::ToggleLayerSidePanel(i));
            }
            if button == Button::Primary && modifiers.alt {
                return Some(Action::TogglePick(i));
            }
        }
        if button != Button::Primary {
            return keys.iter().any(|k| matches!(k, Key::Panel(_))).then_some(Action::Repaint);
        }
        for k in keys {
            match *k {
                Key::Gutter(side, gi) => {
                    let group = side::groups(viewer, side).into_iter().nth(gi)?;
                    let at = if side.horizontal() { y } else { x };
                    self.gutter_drag = Some(GutterDrag::Column { side, x0: at, size0: group.size, panels: group.panels });
                    return Some(Action::Repaint);
                }
                Key::FlexGutter(side, gi, index) => {
                    let group = side::groups(viewer, side).into_iter().nth(gi)?;
                    let (first, second) = (group.panels.get(index)?.clone(), group.panels.get(index + 1)?.clone());
                    let a = self.ui.rect(self.node_of(Key::Panel(first.0))?);
                    let r = self.ui.rect(self.node_of(Key::Panel(second.0))?);
                    let across = side.horizontal();
                    let (top, bottom) = if across { (a.x, r.x + r.width) } else { (a.y, r.y + r.height) };
                    self.gutter_drag = Some(GutterDrag::Split { first, second, top, bottom, across });
                    let _ = y;
                    return Some(Action::Repaint);
                }
                Key::Field(id) => {
                    // Focus: the caret at the end (a click past the text).
                    if self.field.as_ref().is_none_or(|(f, _)| *f != id) {
                        let text = match id {
                            FieldId::Title => viewer.settings.title.clone().unwrap_or_default(),
                            FieldId::LayerName(i) => viewer.layers().get(i).map_or(String::new(), |l| l.name().to_owned()),
                            FieldId::Number(n) => number_to_string(number_setting(viewer, n)),
                            FieldId::Bound(i, window, endpoint) => match self.controls.get(&Key::Control(i, "cdf")) {
                                Some(c) => c.bound_text(window, endpoint),
                                None => String::new(),
                            },
                            FieldId::Slider(i, key) => match self.controls.get(&Key::Control(i, key)) {
                                Some(rendering::Control::Slider { value, .. }) => number_to_string(*value),
                                _ => String::new(),
                            },
                            FieldId::DimensionName(d) => self.dimension_texts.get(d).map_or(String::new(), |t| t.0.clone()),
                            FieldId::DimensionScale(d) => self.dimension_texts.get(d).map_or(String::new(), |t| t.1.clone()),
                            FieldId::ShaderChannel(i, name, dim) => {
                                let l = &viewer.layers()[i];
                                let t = l.data_type().unwrap_or(ng_core::DataType::Uint8);
                                let shader = l.spec.json.get("shader").and_then(serde_json::Value::as_str).unwrap_or(ng_viewer::state::DEFAULT_IMAGE_SHADER);
                                let parsed = ng_viewer::user_shader::parse_with_channels(shader, t, l.channel_extents().len());
                                let values = ng_viewer::user_shader::control_values(&parsed, t, l.spec.json.get("shaderControls"));
                                parsed
                                    .controls
                                    .iter()
                                    .position(|(n, _)| n == name)
                                    .and_then(|k| match &values[k] {
                                        ng_viewer::user_shader::ControlValue::Invlerp { channel, .. } => channel.get(dim).map(u32::to_string),
                                        _ => None,
                                    })
                                    .unwrap_or_default()
                            }
                            FieldId::ShaderSlider(i, name) => viewer
                                .layers()
                                .get(i)
                                .and_then(|l| l.spec.json.get("shaderControls")?.get(name)?.as_f64())
                                .or_else(|| self.controls.values().find_map(|c| match c {
                                    rendering::Control::ShaderSlider { layer, name: n, value, .. } if *layer == i && *n == name => Some(*value),
                                    _ => None,
                                }))
                                .map_or(String::new(), number_to_string),
                            FieldId::PaletteName(k) => palettes::entry(viewer, k as usize).map_or(String::new(), |e| e.0),
                            FieldId::PaletteQuery(k) => palettes::entry(viewer, k as usize).and_then(|e| e.1.get("query").and_then(serde_json::Value::as_str).map(str::to_owned)).unwrap_or_default(),
                            FieldId::Velocity(d) => self.velocities.get(&d).map_or(String::new(), |v| number_to_string(v.velocity)),
                            FieldId::SourceUrl(i) => viewer.layers().get(i).and_then(|l| l.spec.sources.first()).map_or(String::new(), |s| s.url.clone()),
                            FieldId::ScreenshotName => match &self.dialog {
                                Some(dialogs::Dialog::Screenshot(d)) => d.name.clone(),
                                _ => String::new(),
                            },
                            FieldId::SegmentQuery(i) => viewer.layers().get(i).and_then(|l| l.spec.json.get("segmentQuery")?.as_str().map(str::to_owned)).unwrap_or_default(),
                            FieldId::ColorSeed(i) => viewer
                                .layers()
                                .get(i & rendering::LAYER_MASK)
                                .and_then(|l| match l.kind() {
                                    Some(ng_viewer::LayerKind::Segmentation(d)) => Some(d.color_seed.to_string()),
                                    _ => None,
                                })
                                .unwrap_or_default(),
                        };
                        // The invlerp bounds select their text on focus.
                        let field = if matches!(id, FieldId::Bound(..)) { TextField::selected(text) } else { TextField::new(text) };
                        self.field = Some((id, field));
                    }
                    self.caret_since = self.now;
                    // A coordinate loses focus (`blur` commits it); its
                    // dropdown, holding this field, stays.
                    if self.edit.is_some() {
                        let commit = self.commit();
                        self.edit = None;
                        return Some(commit);
                    }
                    return Some(Action::Repaint);
                }
                _ => {}
            }
        }
        None
    }

    fn pointer_down_bars(&mut self, keys: &[Key], x: f32, y: f32, button: Button, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        let _ = x;
        let keys = keys.to_vec();
        // A press outside the open dimension closes its dropdown and blurs
        // its field (the document's capturing `pointerdown`).
        // (The dropdown is the open dimension's.)
        let in_dimension = |dim: usize| keys.contains(&Key::Dimension(dim)) || keys.contains(&Key::Dropdown);
        let mut result = None;
        if let Some(dim) = self.dropdown.or(self.edit.as_ref().map(|e| e.dim)) {
            if !in_dimension(dim) {
                result = self.blur();
            }
        }
        if keys.contains(&Key::Plot) && button == Button::Primary {
            if modifiers.control || modifiers.alt || modifiers.meta {
                return result.or(Some(Action::Repaint));
            }
            // The hover line stays until the pointer moves (a drag's move
            // clears it).
            self.plot_drag = true;
            return self.plot_coordinate(y).map(|(dim, c)| Action::SetCoordinates(vec![Coordinate { dim, value: c, to_voxel_center: true }]));
        }
        if keys.contains(&Key::Dropdown) {
            let action = if button == Button::Primary { keys.iter().find_map(|k| self.actions.get(k).cloned()) } else { None };
            return Some(action.or(result).unwrap_or(Action::Repaint));
        }
        // A press in a dimension (outside its inputs and dropdown) focuses
        // its coordinate.
        if let Some(dim) = keys.iter().find_map(|k| if let Key::Dimension(d) = k { Some(*d) } else { None }) {
            if button == Button::Primary && !keys.contains(&Key::Dropdown) {
                if self.edit.as_ref().is_some_and(|e| e.dim == dim) {
                    // Inside the focused field: a click moves the caret to
                    // the end (as a click past the text does).
                    if let Some(e) = &mut self.edit {
                        let n = e.text.chars().count();
                        (e.start, e.end) = (n, n);
                    }
                } else {
                    let space = viewer.dimensions();
                    let text = if viewer.navigation().valid() && dim < space.rank() {
                        coordinate_text(viewer.navigation().position()[dim])
                    } else {
                        String::new()
                    };
                    self.focus_coordinate(dim, text);
                }
                return Some(result.unwrap_or(Action::Repaint));
            }
            return result.or(Some(Action::Repaint));
        }
        let item = keys.iter().find_map(|k| if let Key::LayerItem(i) = k { Some(*i) } else { None });
        if let (Some(i), Button::Secondary) = (item, button) {
            // `contextmenu`: select the layer and show its panel.
            return Some(Action::SelectLayer(i));
        }
        if button != Button::Primary {
            return result;
        }
        let action = keys.iter().find_map(|k| self.actions.get(k).cloned());
        let action = match (action, item) {
            (Some(Action::ToggleLayerVisible(i)), _) if modifiers.control || modifiers.meta => Some(Action::ToggleLayerSidePanel(i)),
            (Some(Action::ToggleLayerVisible(i)), _) if modifiers.alt => Some(Action::TogglePick(i)),
            // ⌘/Ctrl+click on the wrench makes a palette.
            (Some(Action::TogglePaletteList), _) if modifiers.control || modifiers.meta => Some(Action::AddPalette { name: "Palette".into(), query: String::new() }),
            (a, _) => a,
        };
        // A layer item's click waits for the release: a press that moves
        // drags the layer instead (`draggable`; Shift copies it).
        if let (Some(i), Some(a @ (Action::ToggleLayerVisible(_) | Action::ToggleLayerSidePanel(_) | Action::TogglePick(_)))) = (item, &action) {
            if let Some(layer) = viewer.layers().get(i) {
                self.layer_drag = Some(LayerDrag { id: layer.id, start: (x, y), dragging: false, copy: modifiers.shift, click: Some(a.clone()), target: None, copy_id: None });
                return result.or(Some(Action::Repaint));
            }
        }
        action.or(result)
    }

    /// The release's action, before [`pointer_up`](Self::pointer_up): a
    /// layer item's click when the press did not drag.
    pub fn pointer_up_action(&mut self) -> Option<Action> {
        let drag = self.layer_drag.take()?;
        if drag.dragging {
            Some(Action::Repaint)
        } else {
            drag.click
        }
    }

    /// A button released: ends a drag along a plot or a gutter.
    pub fn pointer_up(&mut self) -> bool {
        self.plot_dragged = false;
        if self.picker_drag.take().is_some() {
            return true;
        }
        // The select tool's drag ends (back to idle).
        if self.select_paint.take().is_some() {
            return true;
        }
        // `addToolKeyBindHandlers`: binding lasts while the button is held.
        if self.binding_tool.take().is_some() {
            return true;
        }
        let gutter = self.gutter_drag.take().is_some() | self.control_drag.take().is_some();
        std::mem::take(&mut self.plot_drag) | gutter
    }

    /// Double-click at `(x, y)`: a dimension's name or scale becomes
    /// editable (not ported yet).
    pub fn double_click(&mut self, x: f32, y: f32) -> Option<Action> {
        let keys = self.keys_at(x, y);
        // A tool's binding widget, or the tool in the top bar: unbound; a
        // tool in an explicit palette (the event reaching its container
        // too): removed.
        let unbind = keys.iter().find_map(|k| match k {
            Key::ToolBinding(n) | Key::ToolStatus(n) => self.tool_refs.get(*n as usize),
            _ => None,
        });
        let unbind = unbind.map(|tool| Action::UnbindTool(tool.owner, tool.json.clone()));
        let remove = keys.iter().find_map(|k| if let Key::PaletteTool(p, i) = k { Some((*p, *i as usize)) } else { None }).map(|(p, i)| Action::RemovePaletteTool(p, i));
        // The selected layer's legacy tool in the top bar: unset.
        if let Some(i) = keys.iter().find_map(|k| if let Key::Control(i, "legacyTool") = k { Some(*i) } else { None }) {
            return Some(Action::SetLayerOption(i, "tool".into(), None));
        }
        match (unbind, remove) {
            (Some(a), Some(b)) => return Some(Action::Many(vec![a, b])),
            (Some(a), None) | (None, Some(a)) => return Some(a),
            (None, None) => {}
        }
        if let Some(a) = keys.iter().find_map(|k| self.controls.get(k)?.reset()) {
            return Some(a);
        }
        // A dimension's name or scale becomes editable, its text selected;
        // the coordinate loses the focus (and commits), the dropdown stays.
        let name = keys.iter().find_map(|k| if let Key::DimensionName(d) = k { Some(*d) } else { None });
        let scale = keys.iter().find_map(|k| if let Key::DimensionScale(d) = k { Some(*d) } else { None });
        if let Some((id, dim)) = name.map(|d| (FieldId::DimensionName(d), d)).or(scale.map(|d| (FieldId::DimensionScale(d), d))) {
            let commit = self.edit.is_some().then(|| self.commit());
            self.edit = None;
            self.dropdown = Some(dim);
            let text = match id {
                FieldId::DimensionName(d) => self.dimension_texts.get(d).map(|t| t.0.clone()),
                _ => self.dimension_texts.get(dim).map(|t| t.1.clone()),
            };
            self.field = Some((id, TextField::selected(text.unwrap_or_default())));
            self.caret_since = self.now;
            return Some(commit.unwrap_or(Action::Repaint));
        }
        None
    }

    /// The wheel at `(x, y)` (`delta_y` positive toward the user). Over a
    /// dimension it steps that coordinate (`adjust-via-wheel`); over another
    /// control it is swallowed.
    pub fn wheel(&mut self, x: f32, y: f32, delta_y: f64, modifiers: Modifiers) -> Option<Action> {
        let keys = self.keys_at(x, y);
        // The active tool's wheel bindings on the data panels.
        if keys.is_empty() && self.dialog.is_none() && self.panel_at(x, y).is_some() {
            if let Some(a) = self.tool_wheel_action(delta_y, modifiers) {
                return Some(a);
            }
        }
        if let Some(dialogs::Dialog::StateEditor(ed)) = &mut self.dialog {
            if keys.contains(&Key::DialogScroll) {
                ed.scroll = (ed.scroll + delta_y as f32).max(0.0);
            }
            return Some(Action::Repaint);
        }
        // The render scale's `adjust-via-wheel`, before its panel scrolls.
        if let Some(c) = keys.iter().find_map(|k| match self.controls.get(k)? {
            c @ rendering::Control::RenderScale { .. } => Some(c),
            _ => None,
        }) {
            return c.wheel(delta_y).or(Some(Action::Repaint));
        }
        // A range input's wheel: a step up (down the page) or down.
        // `EnumSelectWidget`'s own wheel listener: the previous or next
        // option.
        if let Some(c @ (rendering::Control::Select { .. } | rendering::Control::ShaderSelect { .. })) = keys.iter().find_map(|k| self.controls.get(k)) {
            if delta_y != 0.0 {
                return c.wheel(delta_y);
            }
        }
        if let Some(c @ rendering::Control::Slider { .. }) = keys.iter().find_map(|k| self.controls.get(k)) {
            if delta_y != 0.0 {
                return c.wheel(delta_y);
            }
        }
        // A scrolling panel scrolls (clamped when next laid out).
        if let Some(id) = keys.iter().find_map(|k| if let Key::Scroll(id) = k { Some(*id) } else { None }) {
            let s = self.scroll.entry(id).or_insert(0.0);
            *s = (*s + delta_y as f32).max(0.0);
            return Some(Action::Repaint);
        }
        if let Some(dim) = keys.iter().find_map(|k| if let Key::Dimension(d) = k { Some(*d) } else { None }) {
            if modifiers.alt {
                // `adjust-velocity-via-wheel`.
                return Some(Action::MultiplyVelocity(dim, ng_core::panel::wheel_zoom_amount(delta_y, 0)));
            }
            if delta_y == 0.0 {
                return Some(Action::Repaint);
            }
            return Some(Action::AdjustCoordinate(dim, delta_y.signum()));
        }
        self.captures(x, y).then_some(Action::Repaint)
    }

    /// A key pressed (`key` is the DOM `KeyboardEvent.key`: `"5"`,
    /// `"Enter"`, `"ArrowLeft"` …). `None` when the UI does not hold the
    /// keyboard, so the viewer gets it.
    pub fn key_down(&mut self, key: &str, modifiers: Modifiers, viewer: &Viewer) -> Option<Action> {
        // The shader editor's keys.
        if self.shader_focused() {
            return self.shader_key(key, KeyModifiers { shift: modifiers.shift, control: modifiers.control, alt: modifiers.alt, meta: modifiers.meta });
        }
        // The colour picker closes on Escape or Enter (its colour kept).
        if self.color_picker.is_some() {
            if matches!(key, "Escape" | "Enter") {
                self.color_picker = None;
            }
            return Some(Action::Repaint);
        }
        // A tool's binding widget held: A–Z binds (by key, any case).
        if let Some(tool) = &self.binding_tool {
            let mut chars = key.chars();
            return match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_ascii_alphabetic() => Some(tools::bind_action(c.to_ascii_uppercase(), tool)),
                _ => Some(Action::Repaint),
            };
        }
        self.caret_since = self.now;
        if matches!(self.dialog, Some(dialogs::Dialog::Palette(_))) {
            let m = KeyModifiers { shift: modifiers.shift, control: modifiers.control, alt: modifiers.alt, meta: modifiers.meta };
            return self.palette_key(viewer, key, m).or(Some(Action::Repaint));
        }
        if matches!(self.dialog, Some(dialogs::Dialog::StateEditor(_))) {
            let m = KeyModifiers { shift: modifiers.shift, control: modifiers.control, alt: modifiers.alt, meta: modifiers.meta };
            return self.dialog_key(key, m).or(Some(Action::Repaint));
        }
        // Other dialogs: their field takes keys; Escape closes them.
        if self.dialog.is_some() && self.field.is_none() {
            return Some(if key == "Escape" { Action::CloseDialog } else { Action::Repaint });
        }
        // A source URL's completions: ↑/↓ highlight one, Tab takes it.
        if let Some((FieldId::SourceUrl(_), field)) = &mut self.field {
            let completions = sources::completions(&field.text);
            if !completions.is_empty() {
                let n = completions.len();
                match key {
                    "ArrowDown" => {
                        self.completion_active = Some(self.completion_active.map_or(0, |k| (k + 1) % n));
                        return Some(Action::Repaint);
                    }
                    "ArrowUp" => {
                        self.completion_active = Some(self.completion_active.map_or(n - 1, |k| (k + n - 1) % n));
                        return Some(Action::Repaint);
                    }
                    "Tab" => {
                        let k = self.completion_active.unwrap_or(0).min(n - 1);
                        return Some(Action::Complete(completions[k].0.to_owned()));
                    }
                    _ => self.completion_active = None,
                }
            }
        }
        if let Some((id, field)) = &mut self.field {
            let numeric = matches!(id, FieldId::Number(_));
            let m = KeyModifiers { shift: modifiers.shift, control: modifiers.control, alt: modifiers.alt, meta: modifiers.meta };
            return match field.key(key, m, |c| !numeric || c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+') || !c.is_control()) {
                FieldKey::Commit => {
                    let a = self.commit_field();
                    // A source URL committed goes back to showing the source's.
                    if matches!(self.field, Some((FieldId::SourceUrl(_), _))) {
                        self.field = None;
                        self.completion_active = None;
                    }
                    a
                }
                FieldKey::Tab { .. } => {
                    let a = self.commit_field();
                    self.field = None;
                    a
                }
                FieldKey::Cancel => {
                    // A name field's `cancel`: back to the name, unfocused.
                    if matches!(id, FieldId::LayerName(_)) {
                        self.field = None;
                    }
                    Some(Action::Repaint)
                }
                // The number's `input`: each valid value applies at once.
                FieldKey::Edited => Some(match id {
                    FieldId::Slider(i, key) => slider_action(self.controls.get(&Key::Control(*i, key)), &field.text).unwrap_or(Action::Repaint),
                    // The list follows the query as typed.
                    FieldId::SegmentQuery(i) => Action::SetLayerOption(*i, "segmentQuery".into(), (!field.text.is_empty()).then(|| field.text.clone().into())),
                    _ => Action::Repaint,
                }),
                FieldKey::Moved => Some(Action::Repaint),
                FieldKey::Ignored => None,
            };
        }
        let edit = self.edit.as_mut()?;
        let n = edit.text.chars().count();
        let rank = viewer.dimensions().rank();
        let adjacent = |dim: usize, dir: i32| -> Option<usize> {
            let d = dim as i32 + dir;
            (d >= 0 && (d as usize) < rank).then_some(d as usize)
        };
        let move_to = |this: &mut Self, dim: usize| -> Action {
            // Focus moving to another coordinate commits nothing (its `blur`
            // skips that): what was typed waits, for the next commit.
            if let Some(p) = this.edit.take().filter(|p| p.modified) {
                this.pending.insert(p.dim, p.text);
            }
            let text = if viewer.navigation().valid() { coordinate_text(viewer.navigation().position()[dim]) } else { String::new() };
            this.focus_coordinate(dim, text);
            Action::Repaint
        };
        match key {
            "Enter" => Some(self.commit()),
            "Escape" => {
                // `cancel`: the fields show the position again; focus goes.
                self.edit = None;
                self.pending.clear();
                self.dropdown = None;
                self.plot = None;
                Some(Action::Repaint)
            }
            "ArrowUp" | "ArrowDown" => {
                // `adjustDimensionPosition` commits what was typed first.
                let dim = edit.dim;
                let commit = self.commit();
                Some(Action::Many(vec![commit, Action::AdjustCoordinate(dim, if key == "ArrowUp" { -1.0 } else { 1.0 })]))
            }
            "Tab" => {
                let dir = if modifiers.shift { -1 } else { 1 };
                // Past the last (or first) field focus stays: the binding
                // prevents the browser's tabbing.
                match adjacent(edit.dim, dir) {
                    Some(d) => Some(move_to(self, d)),
                    None => Some(Action::Repaint),
                }
            }
            "ArrowLeft" | "ArrowRight" => {
                let dir = if key == "ArrowRight" { 1 } else { -1 };
                let at_edge = edit.start == edit.end && edit.start == if dir == 1 { n } else { 0 };
                if at_edge {
                    if let Some(d) = adjacent(edit.dim, dir) {
                        return Some(move_to(self, d));
                    }
                }
                let caret = if edit.start != edit.end {
                    if dir == 1 {
                        edit.end
                    } else {
                        edit.start
                    }
                } else {
                    (edit.start as i32 + dir).clamp(0, n as i32) as usize
                };
                (edit.start, edit.end) = (caret, caret);
                Some(Action::Repaint)
            }
            "Home" | "End" => {
                let c = if key == "Home" { 0 } else { n };
                (edit.start, edit.end) = (c, c);
                Some(Action::Repaint)
            }
            "Backspace" | "Delete" => {
                if key == "Backspace" && edit.start == edit.end && edit.start == 0 {
                    // `delete-backward` at the start: the previous coordinate.
                    return Some(match adjacent(edit.dim, -1) {
                        Some(d) => move_to(self, d),
                        None => Action::Repaint,
                    });
                }
                let (mut a, mut b) = (edit.start, edit.end);
                if a == b {
                    if key == "Backspace" {
                        a -= 1;
                    } else if b < n {
                        b += 1;
                    }
                }
                let chars: Vec<char> = edit.text.chars().collect();
                edit.text = chars[..a].iter().chain(&chars[b..]).collect();
                (edit.start, edit.end) = (a, a);
                edit.modified = true;
                Some(Action::Repaint)
            }
            "a" if modifiers.control || modifiers.meta => {
                (edit.start, edit.end) = (0, n);
                Some(Action::Repaint)
            }
            k if k.chars().count() == 1 && !(modifiers.control || modifiers.meta) => {
                let c = k.chars().next().unwrap();
                // The `input` handler drops all but `-`, digits and `.`.
                if !(c == '-' || c == '.' || c.is_ascii_digit() || c == ' ' || c == ',') {
                    return Some(Action::Repaint);
                }
                let chars: Vec<char> = edit.text.chars().collect();
                let typed: String = chars[..edit.start].iter().chain(std::iter::once(&c)).chain(&chars[edit.end..]).collect();
                // `^(-?\d+(?:\.(?:\d+)?)?)((?:\s+(?![\s,]))|(?:\s*,\s*))$`,
                // typed at the end: on to the next coordinate.
                let at_end = edit.end == n;
                let separator = c == ' ' || c == ',';
                let number = typed.trim_end_matches([' ', ',']);
                let is_number = {
                    let s = number.strip_prefix('-').unwrap_or(number);
                    let (int, frac) = s.split_once('.').unwrap_or((s, ""));
                    !int.is_empty() && int.chars().all(|c| c.is_ascii_digit()) && frac.chars().all(|c| c.is_ascii_digit())
                };
                if separator {
                    edit.text = typed.chars().filter(|c| *c == '-' || *c == '.' || c.is_ascii_digit()).collect();
                    edit.start = edit.text.chars().count().min(edit.start);
                    edit.end = edit.start;
                    edit.modified = true;
                    if at_end && is_number {
                        if let Some(d) = adjacent(edit.dim, 1) {
                            return Some(move_to(self, d));
                        }
                    }
                    return Some(Action::Repaint);
                }
                edit.text = typed;
                edit.start += 1;
                edit.end = edit.start;
                edit.modified = true;
                Some(Action::Repaint)
            }
            _ => Some(Action::Repaint),
        }
    }
}
