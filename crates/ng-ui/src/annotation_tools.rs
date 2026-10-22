//! The annotation layers' placement tools (`ui/annotations.ts`
//! `PlacePointTool`, `PlaceLineTool`, `PlaceBoundingBoxTool`,
//! `PlaceEllipsoidTool`, `PlacePolylineTool`), run by `annotate`
//! (Ctrl+click on a panel) on the selected layer's `tool`. Points are placed
//! at once; the others follow the pointer from the first trigger until the
//! next finishes them (a polyline adds a point per trigger and finishes on
//! one at the same place).

use ng_viewer::Viewer;
use serde_json::{json, Value};

use crate::{Effect, NgUi};

/// An annotation being placed: its layer, ID and tool.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InProgress {
    pub layer: usize,
    pub id: String,
    pub tool: String,
}

fn numbers(p: &[f64]) -> Value {
    Value::Array(p.iter().map(|&v| ng_core::js::number(v)).collect())
}

fn point(v: &Value) -> Vec<f64> {
    v.as_array().map_or(vec![], |a| a.iter().filter_map(Value::as_f64).collect())
}

/// `makeAnnotationId`: 160 random bits as hex.
fn annotation_id() -> String {
    let mut b = [0u8; 20];
    crate::getrandom_fill(&mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The in-progress annotation updated to the pointer at `p`: `triggered`
/// by a click (else a preview). Returns whether it is finished.
fn update(a: &mut Value, tool: &str, p: &[f64], triggered: bool) -> bool {
    match tool {
        "annotateLine" => a["pointB"] = numbers(p),
        // `PlaceBoundingBoxTool`: a corner not to coincide with the first
        // in any dimension (one more where it would).
        "annotateBoundingBox" => {
            let first = point(&a["pointA"]);
            let b: Vec<f64> = p.iter().zip(&first).map(|(&x, &y)| if x == y { ((x as f32) + 1.0) as f64 } else { x }).collect();
            a["pointB"] = numbers(&b);
        }
        "annotateSphere" => {
            let center = point(&a["center"]);
            let radii: Vec<f64> = p.iter().zip(&center).map(|(&x, &c)| ((c - x).abs() as f32) as f64).collect();
            a["radii"] = numbers(&radii);
        }
        "annotatePolyline" => {
            let mut points: Vec<Value> = a["points"].as_array().cloned().unwrap_or_default();
            if !triggered {
                if let Some(last) = points.last_mut() {
                    *last = numbers(p);
                }
            } else {
                let n = points.len();
                let finished = n >= 2 && points[n - 1] == points[n - 2];
                if !(finished && n > 2) {
                    points.push(numbers(p));
                }
                a["points"] = Value::Array(points);
                return finished;
            }
            a["points"] = Value::Array(points);
        }
        _ => {}
    }
    triggered
}

impl NgUi {
    /// `selectAnnotation(…, pin = true)` (`captureSingleLayerState`): the
    /// annotation alone in a pinned selection, the panel shown.
    pub(crate) fn select_annotation(&mut self, viewer: &mut Viewer, layer: usize, id: &str) {
        let name = viewer.layers()[layer].name().to_owned();
        let selection = crate::panels::Selection {
            position: None,
            layers: vec![(name, Some(json!({ "annotationId": id, "annotationSource": 0, "annotationSubsource": "default" })))],
        };
        let mut l = crate::side::PanelId::Selection.location(viewer);
        l.visible = true;
        crate::side::PanelId::Selection.set_location(viewer, &l);
        self.selection_pinned = true;
        if let Some(o) = viewer.ui_state("selection").and_then(Value::as_object).cloned() {
            // A fresh value: no position.
            let mut o = o;
            o.remove("position");
            viewer.set_ui_state("selection", Some(Value::Object(o)));
        }
        selection.store(viewer);
        self.scroll_to_annotation = Some((layer, id.to_owned()));
        // Each selection a new value: into the history.
        self.selection_history.truncate(self.selection_index + 1);
        self.selection_history.push(selection);
        self.selection_index = self.selection_history.len() - 1;
    }

    /// `shiftSelectedIndexBy`: the annotation `offset` entries from the
    /// selected one (clamped to the list), selected and moved to.
    pub(crate) fn shift_selected_annotation(&mut self, viewer: &mut Viewer, layer: usize, offset: i64) {
        let name = viewer.layers()[layer].name().to_owned();
        let Some(id) = viewer.ui_state("selection").and_then(|s| s.get("layers")?.get(&name)?.get("annotationId")?.as_str().map(str::to_owned)) else { return };
        let rank = viewer.dimensions().rank();
        let Some(list) = viewer.layers()[layer].annotation_list(rank) else { return };
        let Some(index) = list.iter().position(|e| e.id == id) else { return };
        let target = (index as i64 + offset).clamp(0, list.len() as i64 - 1) as usize;
        if target == index {
            return;
        }
        let target_id = list[target].id.clone();
        self.select_annotation(viewer, layer, &target_id);
        self.apply(viewer, &crate::Action::MoveToAnnotation(layer, target));
    }

    /// The pinned selection's annotation of layer `layer`, if any.
    pub(crate) fn selected_annotation(&self, viewer: &Viewer, layer: usize) -> Option<String> {
        if !self.selection_pinned {
            return None;
        }
        let name = viewer.layers().get(layer)?.name();
        let v = viewer.ui_state("selection")?.get("layers")?.get(name)?;
        v.get("annotationId").and_then(Value::as_str).map(str::to_owned)
    }

    /// `annotate`: the selected layer's tool at the pointer.
    pub(crate) fn annotate(&mut self, viewer: &mut Viewer) -> Effect {
        let selected = viewer.ui_state("selectedLayer").and_then(|v| v.get("layer")).and_then(Value::as_str).map(str::to_owned);
        let Some(i) = selected.as_deref().and_then(|name| viewer.layers().iter().position(|l| l.name() == name)) else {
            self.temporary_messages.push(("The annotate command requires a layer to be selected.".into(), self.now + 2000.0));
            return Effect::None;
        };
        let Some(tool) = viewer.layers()[i].spec.json.get("tool").and_then(Value::as_str).map(str::to_owned) else {
            let name = viewer.layers()[i].name().to_owned();
            self.temporary_messages.push((format!("The selected layer ({}) does not have an active annotation tool.", serde_json::to_string(&name).unwrap()), self.now + 2000.0));
            return Effect::None;
        };
        let Some(p) = viewer.mouse_annotation_position(i) else { return Effect::None };
        // The next trigger of the tool placing an annotation finishes it.
        if let Some(progress) = self.annotating.clone() {
            if progress.layer == i && progress.tool == tool {
                let mut finished = false;
                let changed = viewer.update_annotation(i, &progress.id, |a| finished = update(a, &tool, &p, true));
                if changed == Some(true) {
                    self.select_annotation(viewer, i, &progress.id);
                }
                let found = changed.is_some();
                if !found || finished {
                    if found && tool == "annotatePolyline" {
                        // `complete`: the point the last click doubled goes.
                        viewer.update_annotation(i, &progress.id, |a| {
                            if let Some(points) = a["points"].as_array_mut() {
                                if points.len() > 2 {
                                    points.pop();
                                }
                            }
                        });
                    }
                    self.annotating = None;
                }
                return Effect::None;
            }
        }
        let id = annotation_id();
        let ps = numbers(&p);
        let annotation = match tool.as_str() {
            "annotatePoint" => json!({ "point": ps, "type": "point", "id": id }),
            "annotateLine" => json!({ "pointA": ps, "pointB": ps, "type": "line", "id": id }),
            "annotateBoundingBox" => json!({ "pointA": ps, "pointB": ps, "type": "axis_aligned_bounding_box", "id": id }),
            "annotateSphere" => json!({ "center": ps, "radii": numbers(&vec![0.0; p.len()]), "type": "ellipsoid", "id": id }),
            "annotatePolyline" => json!({ "points": [ps.clone(), ps], "type": "polyline", "id": id }),
            _ => return Effect::None,
        };
        viewer.add_annotation(i, annotation);
        self.select_annotation(viewer, i, &id);
        if tool != "annotatePoint" {
            self.annotating = Some(InProgress { layer: i, id, tool });
        }
        Effect::None
    }

    /// `complete` (`finish`) or `undo` of the selected layer's tool: only the
    /// polyline tool has them (dropping its last point, then, finishing,
    /// committing it); others say so.
    pub(crate) fn annotation_step(&mut self, viewer: &mut Viewer, finish: bool) -> Effect {
        let selected = viewer.ui_state("selectedLayer").and_then(|v| v.get("layer")).and_then(Value::as_str).map(str::to_owned);
        let Some(i) = selected.as_deref().and_then(|name| viewer.layers().iter().position(|l| l.name() == name)) else { return Effect::None };
        let Some(tool) = viewer.layers()[i].spec.json.get("tool").and_then(Value::as_str).map(str::to_owned) else { return Effect::None };
        if tool != "annotatePolyline" {
            let name = viewer.layers()[i].name().to_owned();
            self.temporary_messages.push((format!("The selected layer ({}) does not have annotation tool with complete step.", serde_json::to_string(&name).unwrap()), self.now + 2000.0));
            return Effect::None;
        }
        let Some(progress) = self.annotating.clone().filter(|p| p.layer == i) else { return Effect::None };
        // `removeLastPointFromAnnotation`.
        viewer.update_annotation(i, &progress.id, |a| {
            if let Some(points) = a["points"].as_array_mut() {
                if points.len() > 2 {
                    points.pop();
                }
            }
        });
        if finish {
            self.annotating = None;
        }
        Effect::None
    }

    /// The annotation being placed follows the pointer (each update).
    pub(crate) fn annotation_preview(&mut self, viewer: &mut Viewer) {
        let Some(progress) = self.annotating.clone() else { return };
        let Some(p) = viewer.mouse_annotation_position(progress.layer) else { return };
        match viewer.update_annotation(progress.layer, &progress.id, |a| {
            update(a, &progress.tool, &p, false);
        }) {
            // `updatePointB`: each change selects it again.
            Some(true) => self.select_annotation(viewer, progress.layer, &progress.id),
            Some(false) => {}
            None => self.annotating = None,
        }
    }
}
