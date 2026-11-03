//! What is picked and hovered (segments and annotations), annotation
//! editing, and the status line.

use super::*;

impl Viewer {
    /// Takes in annotation styles read back; whether any arrived.
    pub(crate) fn poll_styles(&mut self) -> bool {
        let mut pending = false;
        let mut any = false;
        if let Some(h) = &self.hashes {
            if h.poll() {
                self.hashes = None;
                any = true;
            } else {
                pending = true;
            }
        }
        for slot in self.annotation_styles.values_mut() {
            if let StyleState::Pending(r) = &slot.state {
                match r.read() {
                    Some(styles) => {
                        slot.state = StyleState::Ready(Arc::new(styles));
                        any = true;
                    }
                    None => pending = true,
                }
            }
        }
        // The mapping completes as the device is polled: natively and on
        // WebGL2 only then (a browser's WebGPU completes it by itself).
        // Asked again until it does.
        if pending {
            let _ = self.device.poll(wgpu::PollType::Poll);
            (self.wake)();
        }
        any
    }

    /// `completePickRequest`: once the pick window is read, the position
    /// under the pointer (from the nearest drawn sample's depth, through
    /// the inverse view-projection) and the object there. Whether that
    /// changed.
    pub(crate) fn poll_pick(&mut self) -> bool {
        let Some(pending) = self.pending_pick.as_ref().filter(|p| p.mapped) else { return false };
        // The mapping completes as the device is polled (natively, WebGL2).
        let _ = self.device.poll(wgpu::PollType::Poll);
        let Some((z, pick)) = pending.readback.read() else {
            // Not yet: asked again.
            (self.wake)();
            return false;
        };
        let pending = self.pending_pick.take().unwrap();
        let r = &pending.readback;
        let result = ng_render::pick::resolve_nearest(&z, &pick, r.radius).map(|sample| {
            let (w, h) = (r.size.0 as f64, r.size.1 as f64);
            let ndc = [
                2.0 * (r.x + sample.relative_x as i64 - r.radius as i64) as f64 / w - 1.0,
                2.0 * (r.y + sample.relative_y as i64 - r.radius as i64) as f64 / h - 1.0,
                2.0 * (1.0 - sample.depth as f64) - 1.0,
            ];
            let p = ng_core::glm::vec4_transform_mat4([ndc[0], ndc[1], ndc[2], 1.0], &pending.inv_vp);
            let mut position = pending.position.clone();
            for (k, &g) in pending.display.iter().enumerate() {
                if g >= 0 {
                    position[g as usize] = (p[k] / p[3]) as f32;
                }
            }
            let picked = (sample.pick as usize).checked_sub(1).and_then(|k| pending.ids.get(k).cloned());
            PerspectivePick { position, picked }
        });
        let changed = result != self.perspective_pick;
        self.perspective_pick = result;
        changed
    }

    /// Annotation `id` of layer `li`: a local one, or one of a loaded
    /// spatial index cell (the finest level holding it).
    pub(crate) fn find_annotation(&self, li: usize, id: &str) -> Option<annotations::Annotation> {
        let layer = self.layers.get(li)?;
        let ann = layer.annotation.as_ref()?;
        if let Some(a) = ann.annotations.iter().find(|a| a.id == id) {
            return Some(a.clone());
        }
        ann.spatial.as_ref()?;
        if let Some(Some(a)) = id.parse::<u64>().ok().and_then(|n| self.annotations_by_id.get(&(layer.id, n))).as_deref() {
            return Some(a.clone());
        }
        let mut found: Option<(usize, annotations::Annotation)> = None;
        self.annotation_cells.for_each(|(l, level, _), state, _, _, _| {
            if *l != layer.id || found.as_ref().is_some_and(|(f, _)| f <= level) {
                return;
            }
            if let State::Ready(list) = state {
                if let Some(a) = list.iter().find(|a| a.id == id) {
                    found = Some((*level, a.clone()));
                }
            }
        });
        found.map(|(_, a)| a)
    }

    /// Annotation `id` of layer `li` for the Selection panel; `None` until
    /// it is loaded.
    pub fn annotation_details(&self, li: usize, id: &str) -> Option<AnnotationDetails> {
        let layer = self.layers.get(li)?;
        let a = self.find_annotation(li, id)?;
        let entry = layer.annotation_entry(&a, self.global.rank())?;
        let spatial = layer.annotation.as_ref()?.spatial.as_ref();
        let properties = spatial.map_or(vec![], |info| info.properties.iter().zip(&a.properties).map(|(p, v)| (p.id.clone(), p.kind, *v)).collect());
        Some(AnnotationDetails { entry, readonly: spatial.is_some(), properties })
    }

    /// Asks for annotation `id` of a precomputed layer `li` from its `by_id`
    /// index (`getReference`); whether it may still come (`false`: there
    /// is no such annotation).
    pub fn request_annotation(&mut self, li: usize, id: &str) -> bool {
        let Some(layer) = self.layers.get(li) else { return false };
        if !layer.annotation.as_ref().is_some_and(|a| a.spatial.as_ref().is_some_and(|i| i.by_id.is_some())) {
            return false;
        }
        let Ok(n) = id.parse::<u64>() else { return false };
        let key = (layer.id, n);
        let mut known = None;
        self.annotations_by_id.for_each(|k, state, _, _, _| {
            if *k == key {
                known = Some(matches!(state, State::Queued | State::Downloading));
            }
        });
        match known {
            Some(pending) => pending,
            None => {
                self.annotation_wants.push(key);
                // A frame to ask for it in, once the caller (building the
                // UI, inside the app) is done.
                let wake = self.wake.clone();
                self.executor.spawn(Box::pin(async move { wake() }));
                true
            }
        }
    }

    /// Annotation `id`'s first property value of layer `li`, formatted.
    pub(crate) fn picked_annotation_value(&self, li: usize, id: &str) -> Option<String> {
        let spec = self.layers.get(li)?.annotation.as_ref()?.spatial.as_ref()?.properties.first()?;
        let a = self.find_annotation(li, id)?;
        Some(annotations::format_property_value(spec.kind, *a.properties.first()?))
    }

    /// The segments of a layer drawn as meshes, when capped
    /// (`meshes_max_segments`, a layer with skeletons): the last shown. None:
    /// every shown one.
    pub(crate) fn meshed_ids(&self, managed: &ManagedLayer, segments: &[(u64, bool)]) -> Option<std::collections::HashSet<u64>> {
        if self.objects.meshes_max_segments == 0 || managed.skeleton_sources().is_empty() {
            return None;
        }
        Some(segments.iter().filter(|(_, v)| *v).rev().take(self.objects.meshes_max_segments).map(|(id, _)| *id).collect())
    }

    /// Shown segments whose meshes are drawn: within the cap, and loaded (a
    /// single-resolution mesh, or a multi-resolution fragment on the GPU) —
    /// layer index and id. A host drawing skeletons leaves these out.
    pub fn scene_meshed_segments(&self) -> std::collections::HashSet<(usize, u64)> {
        let mut out = std::collections::HashSet::new();
        let with_fragments: std::collections::HashSet<(LayerId, u64)> = self.gpu_fragments.keys().map(|k| (k.0, k.2)).collect();
        for (li, managed) in self.layers.iter().enumerate() {
            if !managed.shown() || managed.mesh_sources().is_empty() {
                continue;
            }
            let segments = managed.segments();
            let meshed = self.meshed_ids(managed, &segments);
            for &(id, visible) in segments.iter() {
                if !visible || meshed.as_ref().is_some_and(|m| !m.contains(&id)) {
                    continue;
                }
                let loaded = with_fragments.contains(&(managed.id, id))
                    || (0..managed.mesh_sources().len()).any(|m| self.meshes.get(&(managed.id, m, id)).is_some_and(|d| !d.positions.is_empty()));
                if loaded {
                    out.insert((li, id));
                }
            }
        }
        out
    }

    /// What is picked under the pointer: in the 3D panel from its pick
    /// buffers, in a slice panel the nearest mark drawn within the pick
    /// radius (last drawn on top).
    pub(crate) fn picked(&self) -> Option<PickTarget> {
        let (panel, p) = self.pointer_panel()?;
        match panel {
            Panel::Perspective => self.perspective_pick.as_ref()?.picked.clone(),
            Panel::Slice(axes) => {
                let i = NamedAxes::ALL.iter().position(|&a| a == axes)?;
                self.slice_picks[i].pick(p.x * self.scale_factor, p.y * self.scale_factor)
            }
        }
    }

    /// The segment of layer `layer` picked under the pointer (a mesh or
    /// skeleton).
    pub(crate) fn picked_segment(&self, layer: usize) -> Option<u64> {
        let Some(PickTarget::Segment(id, segment)) = self.picked() else { return None };
        (self.layers.get(layer)?.id == id).then_some(segment)
    }

    /// The annotation picked under the pointer: its layer's index and ID.
    pub fn picked_annotation(&self) -> Option<(usize, String)> {
        self.picked_annotation_part().map(|(l, a, _)| (l, a))
    }

    /// The annotation picked under the pointer, and its part
    /// (`annotationPart`).
    pub fn picked_annotation_part(&self) -> Option<(usize, String, u32)> {
        let Some(PickTarget::Annotation(id, a, part)) = self.picked() else { return None };
        Some((self.layers.iter().position(|l| l.id == id)?, a, part.index()))
    }

    pub fn hovered_segments(&mut self) -> Vec<Option<u64>> {
        if let Some((layer, id)) = self.segment_hover {
            return (0..self.layers.len()).map(|i| (i == layer).then_some(id)).collect();
        }
        let pos = self.mouse_position();
        (0..self.layers.len())
            .map(|i| {
                if let Some(id) = self.picked_segment(i) {
                    return Some(id);
                }
                let pos = pos.as_ref()?;
                matches!(self.layers[i].kind(), Some(LayerKind::Segmentation(_))).then_some(())?;
                self.layer_value_at(i, pos)?.parse().ok()
            })
            .collect()
    }

    /// `getMousePositionInAnnotationCoordinates`: the pointer's (unsnapped)
    /// position in annotation layer `layer`'s source coordinates, as
    /// float32.
    pub fn mouse_annotation_position(&mut self, layer: usize) -> Option<Vec<f64>> {
        let position = self.mouse_position()?;
        self.global_to_annotation_model(layer, &position)
    }

    /// A global position in annotation layer `layer`'s source coordinates
    /// (float32).
    pub(crate) fn global_to_annotation_model(&self, layer: usize, position: &[f32]) -> Option<Vec<f64>> {
        global_to_model(&self.layers.get(layer)?.annotation.as_ref()?.layer, position)
    }

    /// Deletes annotation `id` of annotation layer `layer`
    /// (`LocalAnnotationSource.delete`).
    pub fn delete_annotation(&mut self, layer: usize, id: &str) {
        let Some(l) = self.layers.get_mut(layer) else { return };
        let Some(Value::Array(list)) = l.spec.json.get_mut("annotations") else { return };
        list.retain(|a| a.get("id").and_then(Value::as_str) != Some(id));
        self.setup_annotations(layer);
    }

    /// Adds an annotation (`LocalAnnotationSource.add`, its JSON) to
    /// annotation layer `layer`.
    pub fn add_annotation(&mut self, layer: usize, annotation: Value) {
        let Some(l) = self.layers.get_mut(layer) else { return };
        let mut list = l.spec.json.get("annotations").and_then(Value::as_array).cloned().unwrap_or_default();
        list.push(annotation);
        l.spec.json.insert("annotations".into(), Value::Array(list));
        self.setup_annotations(layer);
    }

    /// Changes annotation `id` of annotation layer `layer` in place
    /// (`LocalAnnotationSource.update`): `None` if it is not there, else
    /// whether it changed.
    pub fn update_annotation(&mut self, layer: usize, id: &str, f: impl FnOnce(&mut Value)) -> Option<bool> {
        let l = self.layers.get_mut(layer)?;
        let Some(Value::Array(list)) = l.spec.json.get_mut("annotations") else { return None };
        let a = list.iter_mut().find(|a| a.get("id").and_then(Value::as_str) == Some(id))?;
        let before = a.clone();
        f(a);
        let changed = *a != before;
        if changed {
            self.setup_annotations(layer);
        }
        Some(changed)
    }

    /// `moveToSegment`: the position moves to the loaded mesh vertex of
    /// segment `id` nearest it (`MeshLayer.getObjectPosition`, looking at
    /// about 100 000 vertices). Whether one was loaded.
    pub fn move_to_segment(&mut self, layer: usize, id: u64) -> bool {
        let Some(managed) = self.layers.get(layer) else { return false };
        let Some(state) = managed.render.as_ref() else { return false };
        let l = &state.layer;
        let (rank, n) = (l.rank, l.rank + 1);
        let model_rank = l.source_rank.min(3);
        for (k, source) in managed.mesh_sources().iter().enumerate() {
            // Multi-resolution: the centre of the manifest's bounds.
            let multiscale_center = if source.multiscale.is_some() {
                let Some(manifest) = self.manifests.get(&(managed.id, k, id)) else { continue };
                Some(std::array::from_fn::<f32, 3, _>(|i| ((manifest.clip_lower[i] as f64 + manifest.clip_upper[i] as f64) / 2.0) as f32))
            } else {
                None
            };
            let mesh = match multiscale_center {
                Some(_) => Arc::new(ng_datasource::mesh::MeshData::default()),
                None => {
                    let Some(mesh) = self.meshes.get(&(managed.id, k, id)) else { continue };
                    if mesh.positions.len() < 3 {
                        continue;
                    }
                    mesh
                }
            };
            // Mesh coordinates to the layer's: `to_model` in the model's
            // first dimensions, then model to layer.
            let mut to_model = ng_core::matrix::identity(n, n);
            for c in 0..model_rank {
                for r in 0..model_rank {
                    to_model[c * n + r] = source.to_model[c * 4 + r];
                }
                to_model[(n - 1) * n + c] = source.to_model[12 + c];
            }
            let a = ng_core::matrix::multiply(&l.model_to_layer, &to_model, n);
            let (inv, _) = ng_core::matrix::inverse(&a, n);
            let position = self.nav.position().to_vec();
            let mut layer_pos = vec![0.0; n];
            layer_pos[rank] = 1.0;
            for (d, g) in l.layer_to_global.iter().enumerate() {
                if let Some(g) = g {
                    layer_pos[d] = position.get(*g).copied().unwrap_or(0.0) as f64;
                }
            }
            let apply = |m: &[f64], p: &[f64]| -> Vec<f64> { (0..rank).map(|r| (0..n).map(|c| m[c * n + r] * p[c]).sum()).collect() };
            let near = apply(&inv, &layer_pos);
            let vertex_count = mesh.positions.len() / 3;
            let interval = ((vertex_count as f64 / 100_000.0).floor() as usize).max(1);
            let mut best: Option<(f64, [f32; 3])> = None;
            for v in mesh.positions.chunks_exact(3).step_by(interval) {
                let d: f64 = (0..3).map(|j| (v[j] as f64 - near.get(j).copied().unwrap_or(0.0)).powi(2)).sum();
                if best.is_none_or(|(b, _)| d < b) {
                    best = Some((d, [v[0], v[1], v[2]]));
                }
            }
            let Some((_, v)) = multiscale_center.map(|c| (0.0, c)).or(best) else { continue };
            let mut p = vec![0.0; n];
            p[..3.min(rank)].copy_from_slice(&[v[0] as f64, v[1] as f64, v[2] as f64][..3.min(rank)]);
            p[rank] = 1.0;
            let target = apply(&a, &p);
            let mut position = position;
            for (d, g) in l.layer_to_global.iter().enumerate() {
                if let Some(g) = g {
                    if let Some(x) = position.get_mut(*g) {
                        *x = target[d] as f32;
                    }
                }
            }
            self.nav.set_position(&position);
            return true;
        }
        false
    }

    /// The annotation hovered (`hoverState`): a list entry under the
    /// pointer, or one picked in a panel; whether that changed.
    pub fn set_annotation_hover(&mut self, hover: Option<(usize, String)>) -> bool {
        let hover = hover.and_then(|(l, id)| Some((self.layers.get(l)?.id, id)));
        let changed = self.annotation_hover != hover;
        self.annotation_hover = hover;
        changed
    }

    /// The annotation hovered: its layer's index and ID.
    pub fn annotation_hover(&self) -> Option<(usize, String)> {
        let (id, a) = self.annotation_hover.as_ref()?;
        Some((self.layers.iter().position(|l| l.id == *id)?, a.clone()))
    }

    /// The ID of annotation layer `layer`'s `n`th annotation.
    pub fn annotation_id(&self, layer: usize, n: usize) -> Option<String> {
        Some(self.layers.get(layer)?.annotation.as_ref()?.annotations.get(n)?.id.clone())
    }

    /// A segment hovered in a list (`None` once the pointer leaves it);
    /// whether that changed.
    pub fn set_segment_hover(&mut self, hover: Option<(usize, u64)>) -> bool {
        let changed = self.segment_hover != hover;
        self.segment_hover = hover;
        changed
    }

    /// One line describing the pointer's position and the values under it.
    pub fn status(&mut self) -> String {
        if !self.has_layer() {
            return "neuroglancer-rs — no layer".to_owned();
        }
        let space = self.nav.coordinate_space().clone();
        let fmt = |pos: &[f32]| {
            space
                .names
                .iter()
                .zip(pos)
                .map(|(n, c)| format!("{n} {}", c.floor()))
                .collect::<Vec<_>>()
                .join("  ")
        };
        match self.mouse_position() {
            Some(pos) => {
                let values: Vec<String> = (0..self.layers.len())
                    .filter(|&i| self.layers[i].render.is_some())
                    .map(|i| format!("{}: {}", self.layers[i].name(), self.layer_value_at(i, &pos).unwrap_or_default()))
                    .collect();
                format!("{}  |  {}", fmt(&pos), values.join("  "))
            }
            None => fmt(self.nav.position()),
        }
    }
}
