//! Adding, removing and changing layers, their sources and the dimensions
//! they bring.

use super::*;

impl Viewer {
    pub(crate) fn new_layer(&mut self, spec: LayerSpec) -> LayerId {
        let id = self.next_layer_id;
        self.next_layer_id += 1;
        let sources = spec.sources.iter().map(|s| SourceSlot { spec: s.clone(), load: SourceLoad::Pending, set_manually: false }).collect();
        let mut messages = vec![];
        if !matches!(spec.layer_type.as_str(), "image" | "segmentation" | "annotation" | "auto" | "new") {
            messages.push(format!("{} layers are not supported yet", spec.layer_type));
        }
        self.layers.push(ManagedLayer { id, spec, sources, render: None, messages, progress: (0, 0), render_scale_histogram: Default::default(), mesh_render_scale_histogram: Default::default(), annotation_histograms: Default::default(), volume_rendering_histogram: ng_core::render_scale::RenderScaleHistogram::volume_rendering(), volume_chunks: vec![], histogram_wanted: false, data_histogram: None, histogram_samples: vec![], histogram_windows: vec![], window_histograms: vec![], legend_wants: vec![], legend_jobs: vec![], legends: vec![], shader_error: None, annotation: None, annotation_shader: None, max_cross_section_scale: None, zoom_hidden: false, segments_memo: Default::default() });
        id
    }

    /// A data source finished loading (or failed): bind its transform, merge
    /// dimensions, and set the layer up to draw.
    pub fn source_loaded(&mut self, layer: LayerId, index: usize, result: anyhow::Result<DataSource>) {
        let Some(li) = self.layers.iter().position(|l| l.id == layer) else { return };
        let Some(slot) = self.layers[li].sources.get_mut(index) else { return };
        if !matches!(slot.load, SourceLoad::Pending) {
            return;
        }
        match result {
            Ok(source) => {
                // The URL typed (or of a layer whose type is still to be
                // found) becomes the source's canonical one
                // (`originalCanonicalUrl`).
                let layer_type = self.layers[li].spec.layer_type.clone();
                let slot = &mut self.layers[li].sources[index];
                if (matches!(layer_type.as_str(), "auto" | "new") || slot.set_manually) && source.canonical_url != slot.spec.url {
                    slot.spec.url = source.canonical_url.clone();
                    let url = slot.spec.url.clone();
                    if let Some(spec) = self.layers[li].spec.sources.get_mut(index) {
                        spec.url = url;
                    }
                }
                let slot = &mut self.layers[li].sources[index];
                // `local://annotations` without a state transform: the
                // identity onto the global space as it is now
                // (`LocalDataSourceProvider.get`).
                let local = source.subsources.iter().any(|e| matches!(e.subsource, ng_datasource::model::Subsource::LocalAnnotations));
                let transform = if local && slot.spec.transform.is_none() {
                    identity_over(&self.global)
                } else {
                    VolumeLayer::default_transform(&source).with_spec(slot.spec.transform.as_ref())
                };
                slot.load = SourceLoad::Loaded(Box::new(LoadedSource { source, transform, binding: Binding::default() }));
                self.bind_order.push((layer, index));
            }
            Err(e) => {
                let msg = format!("{}: {e:#}", slot.spec.url);
                slot.load = SourceLoad::Failed;
                self.layers[li].messages.push(msg);
            }
        }
        self.combiner.release();
        self.update_dimensions();
        if let Err(e) = self.setup_render(li) {
            let layer = &mut self.layers[li];
            layer.messages.push(format!("{e:#}"));
        }
    }

    /// Runs the global combiner over the bound sources and passes the result
    /// on: to navigation, and to each layer's transform.
    fn update_dimensions(&mut self) {
        let order = self.bind_order.clone();
        {
            let mut slots: Vec<(LayerId, usize, &mut SourceSlot)> = vec![];
            for layer in &mut self.layers {
                let id = layer.id;
                for (i, slot) in layer.sources.iter_mut().enumerate() {
                    slots.push((id, i, slot));
                }
            }
            let mut by_order: Vec<(usize, &mut Binding, &mut dyn BoundSpace)> = vec![];
            for (id, i, slot) in slots {
                if let SourceLoad::Loaded(loaded) = &mut slot.load {
                    let LoadedSource { transform, binding, .. } = &mut **loaded;
                    if let Some(rank) = order.iter().position(|&o| o == (id, i)) {
                        by_order.push((rank, binding, transform as &mut dyn BoundSpace));
                    }
                }
            }
            by_order.sort_by_key(|(r, _, _)| *r);
            let mut bound: Vec<(&mut Binding, &mut dyn BoundSpace)> = by_order.into_iter().map(|(_, b, t)| (b, t)).collect();
            self.combiner.update(&mut self.global, &mut bound);
        }
        if **self.nav.coordinate_space() != self.global {
            self.nav.set_coordinate_space(Arc::new(self.global.clone()));
        }
        let global = self.global.clone();
        for li in 0..self.layers.len() {
            if self.layers[li].annotation.is_some() {
                self.setup_annotations(li);
            }
        }
        for layer in &mut self.layers {
            let Some(render) = &mut layer.render else { continue };
            if let SourceLoad::Loaded(loaded) = &layer.sources[render.source_index].load {
                render.layer.set_transform(loaded.transform.clone(), &global);
            }
        }
    }

    /// An annotation layer's local annotations, once their source has
    /// loaded: its transform (the global space's identity if the state names
    /// none, as `LocalDataSourceProvider` makes it) and the parsed list.
    // (Shared with the loading tasks: across threads natively; on wasm,
    // where the kvstore is not `Send`, on the one thread.)
    #[allow(clippy::arc_with_non_send_sync)]
    pub(crate) fn setup_annotations(&mut self, li: usize) {
        let global = self.global.clone();
        let layer = &mut self.layers[li];
        if !matches!(layer.spec.layer_type.as_str(), "annotation" | "auto") {
            return;
        }
        // The local annotations, or a precomputed collection's.
        let Some((index, transform, spatial)) = layer.sources.iter().enumerate().find_map(|(i, s)| match &s.load {
            SourceLoad::Loaded(l) => l.source.subsources.iter().find_map(|e| match &e.subsource {
                ng_datasource::model::Subsource::LocalAnnotations => Some((i, l.transform.clone(), None)),
                ng_datasource::model::Subsource::Annotations(info) => Some((i, l.transform.clone(), Some(Arc::new((**info).clone())))),
                _ => None,
            }),
            _ => None,
        }) else {
            return;
        };
        if layer.spec.layer_type == "auto" {
            layer.spec.layer_type = "annotation".into();
        }
        let rank = transform.source_rank;
        let specs = annotations::parse_property_specs(layer.spec.json.get("annotationProperties"));
        let annotations = if spatial.is_some() { vec![] } else { annotations::parse_annotations_with(layer.spec.json.get("annotations"), rank, &specs) };
        layer.annotation = Some(AnnotationLayerState { source_index: index, layer: VolumeLayer::without_volume(transform, &global), annotations, spatial });
    }

    /// Sets up drawing for layer `li` once a volume source has loaded.
    fn setup_render(&mut self, li: usize) -> anyhow::Result<()> {
        self.setup_annotations(li);
        let global = self.global.clone();
        let layer = &mut self.layers[li];
        if layer.render.is_some() {
            return Ok(());
        }
        // the first source whose volume is on: not one whose `default`
        // subsource the spec turns off (a meshes-only source beside the
        // voxels, its info's volume a placeholder)
        let found = layer.sources.iter().enumerate().find_map(|(i, s)| match &s.load {
            SourceLoad::Loaded(l) if l.source.volume().is_some() && volume_enabled(s, &l.source) => Some((i, &l.source, &l.transform)),
            _ => None,
        });
        // no volume yet while another source is still loading: wait for it,
        // or a meshes-only source loaded first would make this a layer
        // without a volume for good
        if found.is_none() && layer.sources.iter().any(|s| matches!(s.load, SourceLoad::Pending)) {
            return Ok(());
        }
        let (source_index, mut volume) = match found {
            Some((i, source, transform)) => (i, VolumeLayer::new(source, transform.clone(), &global)?),
            None => {
                // Objects only (`precomputed://…` of meshes or skeletons): a
                // segmentation layer without a volume.
                let objects = layer.sources.iter().enumerate().find_map(|(i, s)| match &s.load {
                    SourceLoad::Loaded(l) if l.source.subsources.iter().any(|e| matches!(e.subsource, ng_datasource::model::Subsource::Mesh(_) | ng_datasource::model::Subsource::Skeleton(_))) => Some((i, &l.transform)),
                    _ => None,
                });
                match objects {
                    Some((i, transform)) if matches!(layer.spec.layer_type.as_str(), "segmentation" | "auto") => {
                        let mut v = VolumeLayer::without_volume(transform.clone(), &global);
                        v.volume_type = VolumeType::Segmentation;
                        (i, v)
                    }
                    _ => return Ok(()),
                }
            }
        };
        // `trimEdgeVoxels` (not neuroglancer's): `[x, y, z]` voxels of the
        // finest scale (or one number for all three) left off each face of
        // the volume, the same width at every scale. Some published
        // pyramids carry a band of stray values at their bounds (H01's EM:
        // ~20 µm of grey beside its white margin, from building it), which
        // draws a box around the data; the voxels stay where they are, only
        // the clip bounds move in.
        let trim = match layer.spec.json.get("trimEdgeVoxels") {
            Some(Value::Number(n)) => n.as_f64().map(|n| [n; 3]),
            Some(Value::Array(a)) if a.len() == 3 => {
                let v: Vec<f64> = a.iter().filter_map(Value::as_f64).collect();
                (v.len() == 3).then(|| [v[0], v[1], v[2]])
            }
            _ => None,
        };
        if let Some(trim) = trim.filter(|t| t.iter().any(|&v| v > 0.0)) {
            volume.scales = volume.scales.iter().map(|s| Arc::new(layer::trim_edges(s, trim))).collect();
        }
        let mut visible = None;
        let kind = match layer.spec.layer_type.as_str() {
            "image" => LayerKind::Image { opacity: 0.0 },
            "segmentation" => LayerKind::Segmentation(SegmentationDisplay::default()),
            "auto" => match volume.volume_type {
                VolumeType::Segmentation => LayerKind::Segmentation(SegmentationDisplay::default()),
                _ => LayerKind::Image { opacity: 0.0 },
            },
            _ => return Ok(()),
        };
        let kind = match kind {
            LayerKind::Image { .. } => {
                let (opacity, range, warning) = state::image_settings(&layer.spec.json, volume.data_type);
                volume.range = range;
                layer.messages.extend(warning);
                LayerKind::Image { opacity }
            }
            LayerKind::Segmentation(_) => {
                let (display, segments) = state::segmentation_settings(&layer.spec.json)?;
                let ids: Vec<u64> = segments.iter().filter(|&&(_, visible)| visible).map(|&(id, _)| id).collect();
                let mut set = VisibleSet::new(&self.device);
                set.set(&self.device, &self.queue, &ids);
                visible = Some((set, 0));
                LayerKind::Segmentation(display)
            }
        };
        let pipelines = match kind {
            LayerKind::Image { .. } => {
                let (sample, color) = ng_render::image::color_wgsl(volume.data_type);
                LayerPipelines::new(&self.device, sample, &color, false)
            }
            LayerKind::Segmentation(_) => LayerPipelines::new(
                &self.device,
                ng_render::segmentation::sample(),
                ng_render::segmentation::SEGMENTATION_WGSL,
                true,
            ),
        };
        let mut state = LayerState { source_index, layer: volume, kind, pipelines, textures: HashMap::new(), visible, epoch: crate::managed_layer::next_epoch(), shader: None, volume: None };
        layer.shader_error = state.update_shader(&self.device, &layer.spec.json);
        layer.render = Some(state);
        Ok(())
    }

    /// Sets (or, with `None`, removes) a member of layer `layer`'s JSON, and
    /// applies the display settings it holds: opacity, the `normalized`
    /// range, segment colours and alphas, render scale.
    pub fn set_layer_option(&mut self, layer: usize, key: &str, value: Option<Value>) {
        let Some(l) = self.layers.get_mut(layer) else { return };
        match key.split_once('.') {
            // A member of an object (`skeletonRendering.mode2d`): the object
            // left out once empty, as its `toJSON` does.
            Some((outer, inner)) => {
                let mut o = l.spec.json.get(outer).and_then(Value::as_object).cloned().unwrap_or_default();
                match value {
                    Some(v) => {
                        o.insert(inner.to_owned(), v);
                    }
                    None => {
                        o.remove(inner);
                    }
                }
                if o.is_empty() {
                    l.spec.json.remove(outer);
                } else {
                    l.spec.json.insert(outer.to_owned(), Value::Object(o));
                }
            }
            None => match value {
                Some(v) => {
                    l.spec.json.insert(key.to_owned(), v);
                }
                None => {
                    l.spec.json.remove(key);
                }
            },
        }
        let json = l.spec.json.clone();
        if let (Some(render), "shader" | "shaderControls") = (&mut l.render, key) {
            l.shader_error = render.update_shader(&self.device, &json);
        }
        if let Some(render) = &mut l.render {
            match &mut render.kind {
                LayerKind::Image { opacity } => {
                    let (o, range, _) = state::image_settings(&json, render.layer.data_type);
                    *opacity = o;
                    render.layer.range = range;
                }
                LayerKind::Segmentation(display) => {
                    if let Ok((d, _)) = state::segmentation_settings(&json) {
                        *display = d;
                    }
                }
            }
        }
    }

    /// `ManagedUserLayer.setArchived`: an archived layer is hidden and out
    /// of the layer bar, but stays in the state (and its sources bound).
    pub fn set_archived(&mut self, layer: usize, archived: bool) {
        let Some(l) = self.layers.get_mut(layer) else { return };
        if l.spec.archived == archived {
            return;
        }
        if archived {
            l.spec.visible = false;
        }
        l.spec.archived = archived;
    }

    /// Layer `from` moved to index `to` of the list without it (the layer
    /// bar's drag; `managedLayers.splice`).
    pub fn move_layer(&mut self, from: usize, to: usize) {
        if from >= self.layers.len() {
            return;
        }
        let layer = self.layers.remove(from);
        let to = to.min(self.layers.len());
        self.layers.insert(to, layer);
    }

    /// A copy of layer `i` (Shift+drag): its state, named uniquely
    /// (`getUniqueLayerName`), inserted at `at`. Returns its sources to load.
    pub fn copy_layer(&mut self, i: usize, at: usize) -> Vec<PendingSource> {
        let Some(json) = self.layers.get(i).map(|l| l.to_json()) else { return vec![] };
        let Ok((mut specs, _)) = state::parse_layers(&Value::Array(vec![json])) else { return vec![] };
        let Some(mut spec) = specs.pop() else { return vec![] };
        let base = spec.name.clone();
        let mut suffix = 0;
        while self.layers.iter().any(|l| l.name() == spec.name) {
            suffix += 1;
            spec.name = format!("{base}{suffix}");
        }
        let id = self.new_layer(spec);
        let li = self.layers.len() - 1;
        self.move_layer(li, at);
        let layer = self.layers.iter().find(|l| l.id == id).unwrap();
        let pending: Vec<PendingSource> = layer.sources.iter().enumerate().map(|(index, slot)| PendingSource { layer: id, index, url: slot.spec.url.clone() }).collect();
        for _ in &pending {
            self.combiner.retain();
        }
        pending
    }

    /// `addNewLayer`: a layer of type `new` named `new layer` (made
    /// unique), without sources. Returns its index.
    pub fn add_new_layer(&mut self) -> usize {
        let mut name = "new layer".to_owned();
        let mut suffix = 0;
        while self.layers.iter().any(|l| l.name() == name) {
            suffix += 1;
            name = format!("new layer{suffix}");
        }
        let spec = state::LayerSpec { name, layer_type: "new".into(), visible: true, archived: false, sources: vec![], json: Default::default() };
        self.new_layer(spec);
        self.layers.len() - 1
    }

    /// A layer's data source set to `url` (none if empty), replacing any;
    /// returns it to load (as [`restore_state`](Self::restore_state) does).
    pub fn set_layer_source(&mut self, layer: usize, url: &str) -> Vec<PendingSource> {
        let Some(l) = self.layers.get_mut(layer) else { return vec![] };
        l.spec.sources = if url.is_empty() {
            vec![]
        } else {
            vec![state::SourceSpec { url: url.to_owned(), transform: None, enable_default_subsources: true, subsources: Default::default() }]
        };
        l.sources = l.spec.sources.iter().map(|s| SourceSlot { spec: s.clone(), load: SourceLoad::Pending, set_manually: true }).collect();
        l.render = None;
        l.messages.clear();
        let id = l.id;
        let pending: Vec<PendingSource> = l.sources.iter().enumerate().map(|(index, slot)| PendingSource { layer: id, index, url: slot.spec.url.clone() }).collect();
        for _ in &pending {
            self.combiner.retain();
        }
        pending
    }

    /// `NewUserLayer.detectedLayerConstructor`: the type its loaded data
    /// suggests (`image` or `segmentation`).
    pub fn detected_layer_type(&self, layer: usize) -> Option<&'static str> {
        self.layers.get(layer)?.sources.iter().find_map(|s| match &s.load {
            SourceLoad::Loaded(l) => l.source.volume().map(|v| if matches!(v.volume_type, VolumeType::Segmentation) { "segmentation" } else { "image" }),
            _ => None,
        })
    }

    /// `changeLayerType`: the layer redone as `layer_type` from its JSON.
    pub fn change_layer_type(&mut self, layer: usize, layer_type: &str) {
        let Some(l) = self.layers.get_mut(layer) else { return };
        l.spec.layer_type = layer_type.to_owned();
        l.render = None;
        l.messages.retain(|m| !m.ends_with("layers are not supported yet"));
        if let Err(e) = self.setup_render(layer) {
            self.layers[layer].messages.push(format!("{e:#}"));
        }
    }

    /// `getRenameValidity`: each name valid (`/^[a-zA-Z][a-zA-Z_0-9]*['^]?$/`),
    /// unique, and (if new) not one a bound space already uses.
    pub fn dimension_rename_validity(&self, names: &[String]) -> Vec<bool> {
        let bound: Vec<String> = self
            .layers
            .iter()
            .flat_map(|l| l.sources.iter())
            .filter_map(|s| match &s.load {
                SourceLoad::Loaded(l) => Some(l.transform.space().names.clone()),
                _ => None,
            })
            .flatten()
            .collect();
        names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                ng_core::coordinate_space::is_valid_dimension_name(n, false)
                    && !names.iter().enumerate().any(|(j, m)| j != i && m == n)
                    && (self.global.names.contains(n) || !bound.contains(n))
            })
            .collect()
    }

    /// `PositionWidget.updateNames`: the global dimensions renamed (and the
    /// layers' transforms after them). `false` if a name is not valid or
    /// nothing changed.
    pub fn rename_dimensions(&mut self, names: &[String]) -> bool {
        if names.len() != self.global.rank() || self.dimension_rename_validity(names).contains(&false) || names == self.global.names.as_slice() {
            return false;
        }
        self.edit_counter += 1.0;
        let old = self.global.names.clone();
        let mut space = self.global.clone();
        for (i, n) in names.iter().enumerate() {
            if old[i] != *n {
                space.timestamps[i] = self.edit_counter;
            }
        }
        space.names = names.to_vec();
        self.global = space;
        self.update_dimensions();
        // What is keyed by name follows: playback, the dimension tools.
        for (o, n) in old.iter().zip(names) {
            if o != n {
                if let Some(v) = self.velocities.remove(o) {
                    self.velocities.insert(n.clone(), v);
                }
            }
        }
        self.rename_dimension_tools(&old, names);
        true
    }

    /// `PositionWidget.updateScales`: each dimension's scale and unit
    /// (`parseScale`) set. `false` if one does not parse or nothing changed.
    pub fn set_dimension_scales(&mut self, texts: &[String]) -> bool {
        if texts.len() != self.global.rank() {
            return false;
        }
        let Some(parsed) = texts.iter().map(|t| ng_core::si_units::parse_scale(t)).collect::<Option<Vec<_>>>() else { return false };
        let mut space = self.global.clone();
        let mut changed = false;
        self.edit_counter += 1.0;
        for (i, (scale, unit)) in parsed.into_iter().enumerate() {
            if space.scales[i] != scale || space.units[i] != unit {
                space.scales[i] = scale;
                space.units[i] = unit.to_owned();
                space.timestamps[i] = self.edit_counter;
                changed = true;
            }
        }
        if !changed {
            return false;
        }
        self.global = space;
        self.update_dimensions();
        true
    }

    /// `changeLayerName`: renames a layer, made unique
    /// (`getUniqueLayerName`: `name`, `name1`, `name2` …). Returns the name
    /// it got, or `None` if unchanged.
    pub fn rename_layer(&mut self, layer: usize, name: &str) -> Option<String> {
        if self.layers.get(layer)?.name() == name {
            return None;
        }
        let mut unique = name.to_owned();
        let mut suffix = 0;
        while self.layers.iter().any(|l| l.name() == unique) {
            suffix += 1;
            unique = format!("{name}{suffix}");
        }
        self.layers[layer].spec.name = unique.clone();
        Some(unique)
    }

    /// A segmentation layer's visible segments changed in its spec
    /// (`segments`): rebuilds the set it draws with.
    pub fn refresh_segments(&mut self, layer: usize) {
        let (device, queue) = (self.device.clone(), self.queue.clone());
        let Some(l) = self.layers.get_mut(layer) else { return };
        let Ok((_, segments)) = state::segmentation_settings(&l.spec.json) else { return };
        let Some(render) = l.render.as_mut() else { return };
        if let Some((set, _)) = &mut render.visible {
            let ids: Vec<u64> = segments.iter().filter(|s| s.1).map(|s| s.0).collect();
            set.set(&device, &queue, &ids);
        }
    }

    /// Whether segment `id` is shown in segmentation layer `layer` (listed in
    /// its `segments`, not hidden with `!`): drawn in 3D as its mesh.
    pub fn segment_visible(&self, layer: usize, id: u64) -> bool {
        self.layers.get(layer).is_some_and(|l| state::segmentation_settings(&l.spec.json).is_ok_and(|(_, s)| s.iter().any(|&(i, v)| i == id && v)))
    }

    /// Shows or hides segment `id` in segmentation layer `layer`, as a
    /// double-click does (a hidden one stays listed with `!`, keeping its
    /// place); its mesh loads or goes with the next frame. The new visibility.
    pub fn toggle_segment(&mut self, layer: usize, id: u64) -> bool {
        let visible = self.segment_visible(layer, id);
        let Some(l) = self.layers.get_mut(layer) else { return false };
        let mut segs: Vec<Value> = l.spec.json.get("segments").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        segs.retain(|v| {
            let s = v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string());
            s.trim_start_matches('!') != id.to_string()
        });
        segs.push(if visible { format!("!{id}") } else { id.to_string() }.into());
        l.spec.json.insert("segments".into(), Value::Array(segs));
        self.refresh_segments(layer);
        !visible
    }

    /// `recolor` on every segmentation layer
    /// (`segmentColorHash.randomize()`): a new colour seed.
    pub fn recolor(&mut self, seed: impl Fn() -> u32) {
        for l in &mut self.layers {
            if let Some(LayerKind::Segmentation(display)) = l.kind_mut() {
                let s = seed();
                display.color_seed = s;
                l.spec.json.insert("colorSeed".into(), s.into());
            }
        }
    }

    /// `clear-segments` on every segmentation layer that picks: no
    /// segments visible.
    pub fn clear_segments(&mut self) {
        for i in 0..self.layers.len() {
            let l = &mut self.layers[i];
            let picks = l.spec.json.get("pick").and_then(Value::as_bool).unwrap_or(true);
            if picks && matches!(l.kind(), Some(LayerKind::Segmentation(_))) {
                l.spec.json.remove("segments");
                self.refresh_segments(i);
            }
        }
    }

    /// `deleteLayer`: removes the layer, and its data sources from the
    /// global coordinate space.
    pub fn remove_layer(&mut self, layer: usize) {
        if layer >= self.layers.len() {
            return;
        }
        let removed = self.layers.remove(layer);
        self.bind_order.retain(|&(id, _)| id != removed.id);
        // Sources still loading held the dimensions (`markLoading`).
        for slot in &removed.sources {
            if matches!(slot.load, SourceLoad::Pending) {
                self.combiner.release();
            }
        }
        self.update_dimensions();
    }

    /// Shows `source` as the only layer (an `auto` layer named after it).
    pub fn add_layer(&mut self, source: DataSource) -> anyhow::Result<()> {
        self.reset();
        let url = source.canonical_url.clone();
        let name = url.trim_end_matches('/').rsplit(['/', ':']).next().filter(|s| !s.is_empty()).unwrap_or("layer").to_owned();
        let spec = LayerSpec {
            name,
            layer_type: "auto".into(),
            visible: true,
            archived: false,
            sources: vec![SourceSpec { url, transform: None, enable_default_subsources: true, subsources: Default::default() }],
            json: serde_json::Map::new(),
        };
        let id = self.new_layer(spec);
        self.combiner.retain();
        self.source_loaded(id, 0, Ok(source));
        let layer = self.layers.last().unwrap();
        match layer.messages.first() {
            Some(m) if layer.render.is_none() => anyhow::bail!("{m}"),
            _ if layer.render.is_none() => anyhow::bail!("the data source has no volume"),
            _ => Ok(()),
        }
    }

    /// Opens the data source at `url` and shows it. Natively this blocks while
    /// the metadata loads; in the browser use [`ng_datasource::url::get`] and
    /// [`add_layer`](Self::add_layer).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(&mut self, url: &str) -> anyhow::Result<()> {
        let source = futures::executor::block_on(ng_datasource::url::get(url))?;
        self.add_layer(source)
    }

    /// Restores `state` and loads its data sources, blocking (native).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_state(&mut self, state: &Value) -> anyhow::Result<()> {
        let pending = self.restore_state(state)?;
        for p in pending {
            let result = futures::executor::block_on(ng_datasource::url::get(&p.url));
            self.source_loaded(p.layer, p.index, result);
        }
        Ok(())
    }
}

/// `LocalDataSourceProvider`'s transform without a state transform: the
/// identity from dimensions named `0`, `1`, … to the global space's.
fn identity_over(global: &ng_core::CoordinateSpace) -> ng_core::transform::CoordinateSpaceTransform {
    let rank = global.rank();
    let input = ng_core::CoordinateSpace::new(true, (0..rank).map(|i| i.to_string()).collect(), global.units.clone(), global.scales.clone());
    // `makeCoordinateSpace` without bounds: unbounded.
    let output = ng_core::CoordinateSpace::new(true, global.names.clone(), global.units.clone(), global.scales.clone());
    ng_core::transform::CoordinateSpaceTransform::new(Arc::new(input), Arc::new(output), ng_core::matrix::identity(rank + 1, rank + 1))
}
