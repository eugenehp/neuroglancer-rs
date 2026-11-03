//! Data values: the value under the pointer, histograms of the data and the
//! window, and shader legends.

use super::*;

impl Viewer {
    fn layer_render(&self, layer: usize) -> Option<&LayerState> {
        self.layers.get(layer)?.render.as_ref()
    }

    /// Asks for layer `layer`'s data histogram from the next frame on, or
    /// (`false`) no longer: the histogram's widget shows. The last counts
    /// stay until replaced.
    pub fn want_data_histogram(&mut self, layer: usize, wanted: bool) {
        if let Some(l) = self.layers.get_mut(layer) {
            l.histogram_wanted = wanted;
            if !wanted {
                l.histogram_windows.clear();
            }
        }
    }

    /// The legend of a user shader for data `values` (one per pixel): its
    /// colours (RGBA, as drawn over transparent) once drawn — the last
    /// drawn as wide meanwhile; drawn from the next frame.
    pub fn shader_legend(&mut self, layer: usize, values: Vec<f64>) -> Option<Vec<[u8; 4]>> {
        let key = self.legend_key(layer, &values)?;
        let l = self.layers.get_mut(layer)?;
        if let Some((_, colors)) = l.legends.iter().find(|(k, _)| *k == key) {
            return Some(colors.clone());
        }
        if !l.legend_wants.contains(&values) {
            l.legend_wants.push(values.clone());
        }
        l.legends.iter().rev().find(|(k, _)| k.values.len() == values.len()).map(|(_, c)| c.clone())
    }

    /// What a legend over `values` is drawn for: the shader and its
    /// uniforms at opacity 1.
    fn legend_key(&self, layer: usize, values: &[f64]) -> Option<legend::LegendKey> {
        let l = self.layers.get(layer)?;
        let state = l.render.as_ref()?;
        let shader = state.shader.as_ref()?;
        let uniforms = ng_render::user_shader::uniform_bits(&shader.parsed, state.layer.data_type, &shader.values, 1.0);
        Some(legend::LegendKey { values: values.iter().map(|v| v.to_bits()).collect(), shader: shader.key.0.clone(), uniforms })
    }

    /// Whether a legend is being drawn (draw again for it to show).
    pub fn legend_pending(&self) -> bool {
        self.layers.iter().any(|l| !l.legend_jobs.is_empty() || !l.legend_wants.is_empty())
    }

    /// Starts the legends asked for, collects those drawn.
    pub(crate) fn process_legends(&mut self) {
        for li in 0..self.layers.len() {
            let wants = std::mem::take(&mut self.layers[li].legend_wants);
            for values in wants {
                let Some(key) = self.legend_key(li, &values) else { continue };
                let l = &self.layers[li];
                if l.legend_jobs.iter().any(|j| j.key == key) || l.legends.iter().any(|(k, _)| *k == key) {
                    continue;
                }
                let Some(state) = l.render.as_ref() else { continue };
                let Some(shader) = state.shader.as_ref() else { continue };
                let t = state.layer.data_type;
                // Opacity 1, as the legend's.
                shader.uniforms.set(&self.queue, &ng_render::user_shader::uniform_bits(&shader.parsed, t, &shader.values, 1.0));
                let job = legend::start(&self.device, &self.queue, &state.pipelines.first, &shader.uniforms.view, t, &values, key);
                self.layers[li].legend_jobs.push(job);
            }
            // Natively, a copy's mapping completes as the device is polled.
            if !self.layers[li].legend_jobs.is_empty() {
                let _ = self.device.poll(wgpu::PollType::Poll);
            }
            let l = &mut self.layers[li];
            let mut done = vec![];
            l.legend_jobs.retain(|j| match j.take() {
                Some(colors) => {
                    done.push((j.key.clone(), colors));
                    false
                }
                None => true,
            });
            for d in done {
                l.legends.push(d);
                // A few kept (several controls, a width changing).
                if l.legends.len() > 8 {
                    l.legends.remove(0);
                }
            }
        }
    }

    /// A histogram of layer `layer`'s data over `window` too (an invlerp
    /// control of a user shader), for the next frames: its values are
    /// sampled once for all.
    pub fn want_window_histogram(&mut self, layer: usize, window: [f64; 2]) {
        if let Some(l) = self.layers.get_mut(layer) {
            if !l.histogram_windows.contains(&window) {
                l.histogram_windows.push(window);
            }
        }
    }

    /// Each scale's layer-to-chunk transform (inverted `chunk_to_layer`).
    fn layer_to_chunk(layer: &VolumeLayer) -> Vec<Vec<f64>> {
        let n = layer.rank + 1;
        (0..layer.scales.len()).map(|s| ng_core::matrix::inverse(&layer.chunk_to_layer(s), n).0).collect()
    }

    /// Layer `layer`'s value at a global position, from the finest loaded
    /// chunk containing it, formatted as neuroglancer's layer panel shows it.
    pub fn layer_value_at(&self, layer: usize, position: &[f32]) -> Option<String> {
        // The object picked in the 3D panel (`pickedRenderLayer`'s layer:
        // `pickedValue`).
        if let Some(id) = self.picked_segment(layer) {
            return Some(id.to_string());
        }
        // A picked annotation's first property (`transformPickedValue`).
        if let Some(Some(v)) = self.layers.get(layer).and_then(|l| match self.picked() {
            Some(PickTarget::Annotation(id, annotation, _)) if id == l.id => Some(self.picked_annotation_value(layer, &annotation)),
            _ => None,
        }) {
            return Some(v);
        }
        let state = self.layer_render(layer)?;
        let (chunk, index) = self.locate(layer, &Self::layer_to_chunk(&state.layer), position)?;
        let l = &state.layer;
        if l.channels.is_empty() || !matches!(state.kind, LayerKind::Image { .. }) {
            return Some(l.data_type.read(&chunk.data, index).to_string());
        }
        // Every channel's value (`getChannelSpace` order), joined.
        let count: u32 = l.channels.iter().map(|c| c.extent).product();
        let values: Vec<String> = (0..count)
            .map(|flat| {
                let mut rest = flat;
                let coords: Vec<u32> = l
                    .channels
                    .iter()
                    .map(|c| {
                        let v = rest % c.extent;
                        rest /= c.extent;
                        v
                    })
                    .collect();
                channel_offset(l, &chunk, &coords).map_or(String::new(), |o| l.data_type.read(&chunk.data, index + o).to_string())
            })
            .collect();
        Some(values.join(","))
    }

    /// The finest loaded chunk of layer `layer` holding a global position,
    /// and the voxel's index in it.
    fn locate(&self, layer: usize, l2c_by_scale: &[Vec<f64>], position: &[f32]) -> Option<(std::sync::Arc<ng_datasource::chunk::ChunkData>, usize)> {
        let id = self.layers.get(layer)?.id;
        let state = self.layer_render(layer)?;
        let layer = &state.layer;
        let rank = layer.rank;
        let n = rank + 1;
        let layer_pos = layer.layer_position(position, 0.5)?;
        for (s, l2c) in l2c_by_scale.iter().enumerate() {
            let spec = &layer.scales[s].spec;
            let voxel: Vec<f64> = (0..rank).map(|c| (0..n).map(|k| l2c[k * n + c] * layer_pos[k]).sum::<f64>().floor()).collect();
            let sr = layer.source_rank;
            if (sr..rank).any(|c| voxel[c] != 0.0) || (0..sr).any(|c| voxel[c] < 0.0 || voxel[c] >= spec.upper_voxel_bound[c] as f64) {
                continue;
            }
            let grid: Vec<u64> = (0..sr).map(|c| (voxel[c] / spec.chunk_data_size[c] as f64).floor() as u64).collect();
            let Some(chunk) = self.chunks.get(&(id, s, grid.clone())) else { continue };
            let mut index = 0usize;
            let mut stride = 1usize;
            for c in 0..sr {
                let off = voxel[c] as usize - grid[c] as usize * spec.chunk_data_size[c] as usize;
                if off >= chunk.size[c] as usize {
                    return None;
                }
                index += off * stride;
                stride *= chunk.size[c] as usize;
            }
            return Some((chunk, index));
        }
        None
    }

    /// Fills the wanted data histograms from this frame's slice panels.
    pub(crate) fn compute_data_histograms(&mut self, slice_views: &[SliceViewFrame]) {
        let info = self.nav.render_info().clone();
        let position = self.nav.position().to_vec();
        for li in 0..self.layers.len() {
            let l = &self.layers[li];
            if !(l.histogram_wanted || !l.histogram_windows.is_empty()) || !l.shown() {
                continue;
            }
            let (Some(state), Some(window)) = (l.render.as_ref(), l.invlerp_window()) else { continue };
            let t = state.layer.data_type;
            let w = window.as_f64();
            let windows = l.histogram_windows.clone();
            // The invlerp's channel (`dataHistogramChannelSpecifications`).
            let channel = l.invlerp_channel();
            let l2c = Self::layer_to_chunk(&state.layer);
            let mut counts = [0u32; 256];
            let mut samples = Vec::with_capacity(histogram::SAMPLES * slice_views.len());
            // No slice view: volume rendering samples its chunks
            // (`needToDrawHistogram && !sliceViewsPresent`).
            if slice_views.is_empty() && !l.volume_chunks.is_empty() {
                let size_of = |c: &ChunkData, dims: &[Option<usize>; 3]| -> [u32; 3] { dims.map(|d| d.map_or(1, |d| c.size[d])) };
                let total: f64 = l.volume_chunks.iter().map(|(c, dims, _)| size_of(c, dims).iter().map(|&s| s as f64).product::<f64>()).sum();
                for (chunk, dims, fixed) in &l.volume_chunks {
                    let rank = chunk.size.len();
                    let mut strides = vec![1usize; rank];
                    for k in 1..rank {
                        strides[k] = strides[k - 1] * chunk.size[k - 1] as usize;
                    }
                    let mut base = 0usize;
                    for &(c, _, off) in fixed {
                        if !state.layer.channels.iter().any(|ch| ch.chunk_dim == c) {
                            base += off as usize * strides[c];
                        }
                    }
                    let Some(offset) = channel_offset(&state.layer, chunk, &channel) else { continue };
                    for p in histogram::volume_sample_points(size_of(chunk, dims), total) {
                        let mut index = base + offset;
                        for (k, d) in dims.iter().enumerate() {
                            if let Some(d) = d {
                                index += p[k] as usize * strides[*d];
                            }
                        }
                        let value = t.read(&chunk.data, index).as_f64();
                        counts[histogram::bin((value - w[0]) / (w[1] - w[0]))] += 1;
                        samples.push(value);
                    }
                }
            }
            for (_, inv_view, (pw, ph)) in slice_views {
                let (lw, lh) = (*pw as f64 / self.scale_factor, *ph as f64 / self.scale_factor);
                let mut global = position.clone();
                for &(u, v) in histogram::sample_points() {
                    let view = [((u as f64 - 0.5) * lw) as f32, ((v as f64 - 0.5) * lh) as f32, 0.0];
                    let plane = ng_core::glm::vec3_transform_mat4(view, *inv_view);
                    for i in 0..info.display_rank {
                        global[info.display_dimension_indices[i] as usize] = plane[i];
                    }
                    // Pixels no chunk was drawn on (depth 1) are skipped.
                    let Some((chunk, index)) = self.locate(li, &l2c, &global) else { continue };
                    let Some(offset) = channel_offset(&state.layer, &chunk, &channel) else { continue };
                    let value = t.read(&chunk.data, index + offset).as_f64();
                    counts[histogram::bin((value - w[0]) / (w[1] - w[0]))] += 1;
                    samples.push(value);
                }
            }
            let window_histograms = windows
                .into_iter()
                .map(|w| {
                    let mut c = [0u32; 256];
                    for v in &samples {
                        c[histogram::bin((v - w[0]) / (w[1] - w[0]))] += 1;
                    }
                    (w, Box::new(c))
                })
                .collect();
            let l = &mut self.layers[li];
            l.data_histogram = Some(Box::new(counts));
            l.histogram_samples = samples;
            l.window_histograms = window_histograms;
        }
    }

    /// The point under the pointer in display coordinates, all three (a
    /// display dimension the space lacks still has a coordinate on the
    /// panel's plane), if it is over a slice panel.
    pub fn mouse_display_point(&mut self) -> Option<[f32; 3]> {
        let (panel, p) = self.pointer_panel()?;
        let Panel::Slice(axes) = panel else { return None };
        let mut sp = self.panel_mut(axes);
        let inv = sp.inv_view_matrix(&mut self.nav);
        self.put_panel(axes, sp);
        let view = [(p.x - p.width / 2.0) as f32, (p.y - p.height / 2.0) as f32, 0.0];
        Some(ng_core::glm::vec3_transform_mat4(view, inv))
    }

    /// Continuous voxel coordinates of a global position in layer `layer`'s
    /// finest scale (chunk dimensions; voxel `i` spans `[i, i + 1)`).
    pub fn layer_voxel(&self, layer: usize, position: &[f32]) -> Option<Vec<f64>> {
        let layer = &self.layer_render(layer)?.layer;
        let rank = layer.rank;
        let n = rank + 1;
        let layer_pos = layer.layer_position(position, 0.5)?;
        let (l2c, _) = ng_core::matrix::inverse(&layer.chunk_to_layer(0), n);
        Some((0..layer.source_rank).map(|c| (0..n).map(|k| l2c[k * n + c] * layer_pos[k]).sum()).collect())
    }
}

/// The offset, in elements, of channel `coords` (one per channel
/// dimension) from channel 0's in a chunk; `None` if the chunk lacks it.
fn channel_offset(layer: &VolumeLayer, chunk: &ChunkData, coords: &[u32]) -> Option<usize> {
    let mut offset = 0;
    for (ch, &c) in layer.channels.iter().zip(coords) {
        if c as usize >= chunk.size.get(ch.chunk_dim).copied()? as usize {
            return None;
        }
        let stride: usize = chunk.size[..ch.chunk_dim].iter().map(|&n| n as usize).product();
        offset += c as usize * stride;
    }
    Some(offset)
}
