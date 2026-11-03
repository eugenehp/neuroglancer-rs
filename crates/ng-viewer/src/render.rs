//! Drawing a frame: the 3D panel (`render_perspective`) and the whole
//! layout (`render`), with the geometry and uploads they need.

use super::*;

impl Viewer {
    /// The 3D panel (`PerspectivePanel.drawWithPicking`, default content):
    /// the slice views as planes, the layers' data bounds, the axis lines.
    #[allow(clippy::too_many_arguments)]
    fn render_perspective(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: Option<&wgpu::TextureView>,
        phys: PixelRect,
        slice_views: &[SliceViewFrame],
        info: &ng_core::navigation::DisplayDimensionRenderInfo,
        position: &[f32],
        cross_section_background: [f32; 3],
    ) {
        use ng_core::glm::{mat4_invert, mat4_multiply, vec3_transform_quat, vec4_transform_mat4};
        let Some(camera) = self.perspective_camera() else { return };
        let hovered = self.hovered_segments();
        let skeletons = self.skeleton_draws(&hovered, info, false);
        self.pick_ids.clear();
        let (lw, lh) = self.logical_size();
        let logical = self.panel_logical_rect(Panel::Perspective, lw, lh).unwrap_or(phys);
        let vp = camera.view_projection;
        // Lighting: `lightDirection` is the view's −z — in scene mode the host
        // camera's, whose z axis (in display coordinates) is `inv_view`'s third
        // column; the planes' shading uses |light · normal|, so only the axis counts.
        let l = match &self.scene {
            Some(_) => {
                let z = [camera.inv_view[8], camera.inv_view[9], camera.inv_view[10]];
                let n = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2]).sqrt().max(1e-30);
                [z[0] / n, z[1] / n, z[2] / n]
            }
            None => vec3_transform_quat([0.0, 0.0, 1.0], self.nav.orientation(View::Projection)),
        };
        let light = [-(l[0] as f64), -(l[1] as f64), -(l[2] as f64)];
        // in scene mode the slices drawn are those the layout names (panel_rects)
        let show_planes = self.show_slices && (self.scene.is_some() || !matches!(self.layout, Layout::Perspective));
        let planes: Vec<ng_render::perspective::SlicePlane> = slice_views
            .iter()
            .filter(|_| show_planes)
            .filter_map(|&(i, inv_view, (sw, sh))| {
                let frame = self.panel_frames[i].as_ref()?;
                // `viewportNormalInCanonicalCoordinates`.
                let view = mat4_invert(&inv_view)?;
                let mut normal = [0f64; 3];
                for k in 0..3 {
                    normal[k] = (view[k * 4 + 2] as f64 / info.canonical_voxel_factors[k]) as f32 as f64;
                }
                let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                if len > 0.0 {
                    normal = normal.map(|x| (x * (1.0 / len)) as f32 as f64);
                }
                let scalar = (light[0] * normal[0] + light[1] * normal[1] + light[2] * normal[2]).abs();
                let factor = (0.2 + scalar * 0.8) as f32;
                // Maps (±1, ±1, 0) to the slice view's corners, then to clip.
                let mut m = [0f32; 16];
                m[0] = (sw as f64 / 2.0) as f32;
                m[5] = (-(sh as f64) / 2.0) as f32;
                m[10] = 1.0;
                m[15] = 1.0;
                let m = mat4_multiply(&vp, &mat4_multiply(&inv_view, &m));
                let corner = |x: f64, y: f64| vec4_transform_mat4([x, y, 0.0, 1.0], &m);
                Some(ng_render::perspective::SlicePlane {
                    texture: &frame.target.color_view,
                    corners: [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)],
                    factor,
                })
            })
            .collect();
        // Annotations: the data-bounds box edges.
        let inv_vp = mat4_invert(&vp).unwrap_or(vp);
        let mut lines = vec![];
        for managed in self.layers.iter().filter(|l| l.shown()) {
            let Some(state) = managed.render.as_ref() else { continue };
            let color = managed.annotation_color();
            for (lower, upper) in if self.settings.show_default_annotations { managed.data_bounds() } else { vec![] } {
                let Some((edges, clip)) = chunks::box_edges_3d(&state.layer, &lower, &upper, info, position, &inv_vp) else {
                    continue;
                };
                for (a, b) in edges {
                    lines.push(ng_render::perspective::Line3 {
                        a: vec4_transform_mat4([a[0], a[1], a[2], 1.0], &vp),
                        b: vec4_transform_mat4([b[0], b[1], b[2], 1.0], &vp),
                        width: 1.0,
                        feather: 1.0,
                        color: [color[0], color[1], color[2], clip as f32],
                        color_b: [color[0], color[1], color[2], clip as f32],
                        pick: 0,
                    });
                }
            }
        }
        // Annotation layers' annotations in 3D (`PerspectiveViewAnnotationLayer`).
        let mut circles = vec![];
        let mut triangles = vec![];
        let light_vec = [light[0] * 0.8, light[1] * 0.8, light[2] * 0.8, 0.2];
        let projection = camera.projection.map(|x| x as f64);
        let view = ng_core::matrix::inverse4(&camera.inv_view.map(|x| x as f64));
        let mut spatial: HashMap<LayerId, Vec<annotations::Annotation>> = HashMap::new();
        for li in 0..self.layers.len() {
            if !self.layers[li].shown() {
                continue;
            }
            if let Some(list) = spatial_annotations(&mut self.layers, &mut self.annotation_cells, li, AnnotationView::Projection, self.frame, info, position, &projection, &view, (phys.width as f64, phys.height as f64)) {
                spatial.insert(self.layers[li].id, list);
            }
        }
        // Each layer's styles (its shader's), once computed.
        let mut styles: HashMap<LayerId, Arc<Vec<ng_render::annotation_shader::AnnotationStyle>>> = HashMap::new();
        for managed in self.layers.iter().filter(|l| l.shown()) {
            let Some(ann) = managed.annotation.as_ref() else { continue };
            let list: &[annotations::Annotation] = spatial.get(&managed.id).map_or(&ann.annotations, |v| v);
            if let Some(s) = annotation_styles(&mut self.annotation_styles, &self.device, &self.queue, &self.wake, managed, list, true, self.frame) {
                styles.insert(managed.id, s);
            }
        }
        for managed in self.layers.iter().filter(|l| l.shown()) {
            let Some(ann) = managed.annotation.as_ref() else { continue };
            let list: &[annotations::Annotation] = spatial.get(&managed.id).map_or(&ann.annotations, |v| v);
            let Some(layer_styles) = styles.get(&managed.id) else { continue };
            let clip = |model: &[f64]| {
                let d = model_to_display(&ann.layer, info, model);
                vec4_transform_mat4([d[0], d[1], d[2], 1.0], &vp)
            };
            let hovered = self.annotation_hover.as_ref().filter(|(l, _)| *l == managed.id).map(|(_, id)| id.clone());
            for (k, a) in list.iter().enumerate() {
                let Some(s) = layer_styles.get(k).filter(|s| !s.discard) else { continue };
                let lit = |c: [f32; 4]| highlight(c, hovered.as_deref() == Some(a.id.as_str()));
                // The parts' pick IDs: the object, its endpoints, a corner.
                let [body, end0, end1] = [AnnotationPart::Body, AnnotationPart::Endpoint(0), AnnotationPart::Endpoint(1)].map(|part| {
                    self.pick_ids.push(PickTarget::Annotation(managed.id, a.id.clone(), part));
                    self.pick_ids.len() as u32
                });
                let corner_picks: Vec<u32> = if matches!(a.geometry, annotations::Geometry::AxisAlignedBoundingBox(..)) {
                    (0..8)
                        .map(|k| {
                            self.pick_ids.push(PickTarget::Annotation(managed.id, a.id.clone(), AnnotationPart::Corner(k)));
                            self.pick_ids.len() as u32
                        })
                        .collect()
                } else {
                    vec![]
                };
                let circle = |center: [f64; 4], diameter: f32, border_width: f32, feather: f64, interior: [f32; 4], border: [f32; 4], pick: u32| ng_render::annotations::Circle { center, diameter: diameter as f64, border_width: border_width as f64, feather, interior: lit(interior), border, pick };
                let line = |a: [f64; 4], b: [f64; 4], width: f32, colors: [[f32; 4]; 2]| ng_render::perspective::Line3 { a, b, width: width as f64, feather: 1.0, color: lit(colors[0]), color_b: lit(colors[1]), pick: body };
                match &a.geometry {
                    annotations::Geometry::Point(p) => circles.push(circle(clip(p), s.point_size, s.point_border_width, 1.0, s.point_color, s.point_border_color, body)),
                    annotations::Geometry::Line(pa, pb) => {
                        let (ca, cb) = (clip(pa), clip(pb));
                        lines.push(line(ca, cb, s.line_width, s.line_color));
                        for (e, (c, pick)) in [(ca, end0), (cb, end1)].into_iter().enumerate() {
                            circles.push(circle(c, s.marker_size[e], s.marker_border_width[e], 0.5, s.marker_color[e], s.marker_border_color[e], pick));
                        }
                    }
                    annotations::Geometry::Polyline(points) => {
                        let clips: Vec<[f64; 4]> = points.iter().map(|p| clip(p)).collect();
                        for w in clips.windows(2) {
                            lines.push(line(w[0], w[1], s.poly_line_width, s.poly_line_color));
                        }
                        if clips.len() >= 2 {
                            for (k, &c) in clips.iter().enumerate() {
                                let e = usize::from(k + 1 == clips.len());
                                circles.push(circle(c, s.poly_marker_size[e], s.poly_marker_border_width[e], 0.5, s.poly_marker_color[e], s.poly_marker_border_color[e], if e == 1 { end1 } else { end0 }));
                            }
                        }
                    }
                    annotations::Geometry::Ellipsoid { center, radii } => {
                        // `SphereRenderHelper(10, 10)`, lit per vertex.
                        let fill = lit(s.ellipsoid_fill_color);
                        let c = model_to_display(&ann.layer, info, center);
                        let r = linear_to_display(&ann.layer, info, radii);
                        for v in sphere_triangles(10, 10) {
                            let p = [v[0] * r[0] + c[0], v[1] * r[1] + c[1], v[2] * r[2] + c[2]];
                            let n = [v[0] / r[0].max(1e-6), v[1] / r[1].max(1e-6), v[2] / r[2].max(1e-6)];
                            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                            let n = n.map(|x| x / len);
                            let lighting = ((n[0] * light_vec[0] + n[1] * light_vec[1] + n[2] * light_vec[2]).abs() + light_vec[3]) as f32;
                            triangles.push(ng_render::perspective::ColoredVertex {
                                clip: vec4_transform_mat4([p[0], p[1], p[2], 1.0], &vp),
                                color: [fill[0] * lighting, fill[1] * lighting, fill[2] * lighting, fill[3]],
                            });
                        }
                    }
                    annotations::Geometry::AxisAlignedBoundingBox(pa, pb) => {
                        // Twelve edges (`vec4(vColor.rgb, getLineAlpha())`:
                        // opaque), and the corners as circles one line-width
                        // across without border or feather.
                        let border = s.box_border_color;
                        let opaque = [border[0], border[1], border[2], 1.0];
                        let corner = |bits: usize| -> Vec<f64> { (0..pa.len()).map(|d| if d < 3 && bits >> d & 1 == 1 { pa[d].max(pb[d]) } else { pa[d].min(pb[d]) }).collect() };
                        let corners: Vec<[f64; 4]> = (0..8).map(|b| clip(&corner(b))).collect();
                        for i in 0..8 {
                            for k in 0..3 {
                                let j = i | (1 << k);
                                if j != i {
                                    lines.push(line(corners[i], corners[j], s.box_border_width, [opaque; 2]));
                                }
                            }
                        }
                        for (k, c) in corners.into_iter().enumerate() {
                            circles.push(circle(c, s.box_border_width, 0.0, 0.0, opaque, opaque, corner_picks[k]));
                        }
                    }
                }
            }
        }
        // Segmentation layers' skeletons (`PerspectiveViewSkeletonLayer`),
        // drawn with the opaque layers, before the annotations.
        // Transparent ones (object alpha below 1) with the transparent
        // meshes.
        let (mut skeleton_lines, mut skeleton_circles) = (vec![], vec![]);
        let mut transparent = ng_render::perspective::TransparentMarks::default();
        let pick_ids = &mut self.pick_ids;
        for d in &skeletons {
            // `pickIDs.registerUint64`: each segment's ID, from 1.
            pick_ids.push(PickTarget::Segment(d.segment.0, d.segment.1));
            let pick = pick_ids.len() as u32;
            let line3 = |l: ng_render::annotations::Line| ng_render::annotations::Line { pick, ..l };
            let (mut l, mut c) = (vec![], vec![]);
            d.marks(&|p| vec4_transform_mat4([p[0], p[1], p[2], 1.0], &vp), 0.0, &mut l, &mut c);
            for circle in &mut c {
                circle.pick = pick;
            }
            if d.color[3] < 1.0 {
                transparent.lines.extend(l.into_iter().map(line3));
                transparent.circles.extend(c);
            } else {
                skeleton_lines.extend(l.into_iter().map(line3));
                skeleton_circles.extend(c);
            }
        }
        lines.splice(0..0, skeleton_lines);
        circles.splice(0..0, skeleton_circles);
        // Axis lines (`drawAxisLines`): `zoom · min(w, h) / h / 4` long.
        let axes = self.show_axis_lines.then(|| {
            let r = logical;
            let axis_length = self.nav.zoom(View::Projection) * ((r.width.min(r.height) as f64) / r.height as f64 / 4.0);
            let mut origin = [0.0; 3];
            for (k, o) in origin.iter_mut().enumerate() {
                let g = info.display_dimension_indices[k];
                *o = if g < 0 { 0.0 } else { position[g as usize] as f64 };
            }
            let clip = |p: [f64; 3]| vec4_transform_mat4([p[0], p[1], p[2], 1.0], &vp);
            let ends = std::array::from_fn(|k| {
                let mut p = origin;
                p[k] += axis_length / info.canonical_voxel_factors[k];
                clip(p)
            });
            (clip(origin), ends)
        });
        // Segmentation layers' meshes (`MeshLayer.draw`,
        // `MultiscaleMeshLayer.draw`): each visible segment's, in its colour
        // (`getObjectColor`), lit by `uLightDirection`.
        let light_vec = [(light[0] * 0.8) as f32, (light[1] * 0.8) as f32, (light[2] * 0.8) as f32, 0.2];
        let mut mesh_jobs: Vec<MeshJob> = vec![];
        // Segments' meshes or manifests wanted, by layer and source.
        // (layer, source, segment, mvp, normal matrix, colour, detail cutoff,
        // the source if multi-resolution).
        // Then the silhouette power and the scale multiplier (the model
        // matrix's physical scale, for the render scale histogram).
        type Wanted = (LayerId, usize, u64, [f32; 16], [f32; 9], [f32; 4], f64, Option<ng_datasource::mesh::MeshSource>, f32, f64);
        // Render scale histogram counts: (layer, spatial scale, render
        // scale, present, not present).
        let mut histogram_adds: Vec<(LayerId, f64, f64, u32, u32)> = vec![];
        let mut histogram_layers: Vec<LayerId> = vec![];
        let mut segments_wanted: Vec<Wanted> = vec![];
        for (li, managed) in self.layers.iter().enumerate() {
            if !managed.shown() {
                continue;
            }
            let (Some(state), Some(LayerKind::Segmentation(display))) = (managed.render.as_ref(), managed.kind()) else { continue };
            let sources = managed.mesh_sources();
            if sources.is_empty() {
                continue;
            }
            let alpha = managed.spec.json.get("objectAlpha").and_then(Value::as_f64).unwrap_or(1.0).min(1.0) as f32;
            if alpha <= 0.0 {
                continue;
            }
            // `meshRenderScale` (`renderScaleTarget`): the level-of-detail
            // cutoff, in pixels.
            let mut detail_cutoff = managed.mesh_render_scale_target();
            let silhouette = managed.silhouette_rendering() as f32;
            histogram_layers.push(managed.id);
            let segments = managed.segments();
            // many shown: each coarser, the triangles drawn about level
            // meshes for the last shown only, where skeletons stand for the rest
            let meshed = self.meshed_ids(managed, &segments);
            if self.objects.mesh_detail_segments > 0 {
                let shown = meshed.as_ref().map_or_else(|| segments.iter().filter(|(_, v)| *v).count(), |m| m.len());
                if shown > self.objects.mesh_detail_segments {
                    detail_cutoff *= (shown as f64 / self.objects.mesh_detail_segments as f64).sqrt();
                }
            }
            for (k, source) in sources.iter().enumerate() {
                let Some(model) = mesh_model_matrix(&state.layer, &source.to_model, info) else { continue };
                let mvp = mat4_multiply(&vp, &model);
                let normal_matrix = mesh_normal_matrix(&model, info.canonical_voxel_factors);
                let scale_multiplier = mesh_scale_multiplier(&model, info.voxel_physical_scales);
                for &(id, visible) in segments.iter() {
                    if !visible || meshed.as_ref().is_some_and(|m| !m.contains(&id)) {
                        continue;
                    }
                    let color = object_color(display, hovered.get(li).copied().flatten(), id, alpha);
                    let multiscale = source.multiscale.is_some().then(|| source.clone());
                    segments_wanted.push((managed.id, k, id, mvp, normal_matrix, color, detail_cutoff, multiscale, silhouette, scale_multiplier));
                }
            }
        }
        let (pw, ph) = (phys.width as f64, phys.height as f64);
        self.fragment_reads.clear();
        for (layer_id, k, id, mvp, normal_matrix, color, detail_cutoff, multiscale, silhouette, scale_multiplier) in segments_wanted {
            let key = (layer_id, k, id);
            let mut counted = (1u32, 0u32);
            let is_multiscale = multiscale.is_some();
            match multiscale {
                None => {
                    self.meshes.request(&key, 0.0);
                    let ready = self.meshes.get(&key);
                    counted.1 = u32::from(ready.is_some());
                    if ready.is_none() {
                        histogram_adds.push((layer_id, f64::INFINITY, f64::INFINITY, 0, 1));
                        if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                            l.progress.0 += counted.0;
                        }
                        continue;
                    }
                    if let Some(data) = ready {
                        // The manifest and its fragments.
                        histogram_adds.push((layer_id, f64::INFINITY, f64::INFINITY, 1 + data.fragments_loaded, data.fragments - data.fragments_loaded));
                        let fresh = !matches!(self.gpu_meshes.get(&key), Some((d, _, _)) if Arc::ptr_eq(d, &data));
                        if fresh && !data.slim && !data.indices.is_empty() {
                            let gpu = ng_render::perspective::GpuMesh::new(&self.device, &data.positions, &data.normals, &data.indices);
                            // on the GPU now: the cache keeps a sample of
                            // its vertices, not a second copy
                            let slim = Arc::new(data.slim(MESH_SLIM_POINTS));
                            self.meshes.replace(&key, slim.clone(), slim.bytes());
                            self.gpu_meshes.insert(key, (slim, gpu, data.bytes()));
                        } else if fresh && data.slim {
                            // its GPU copy gone (cannot happen while the cache
                            // holds it): loaded again
                            self.meshes.retain_keys(|k| *k != key);
                        }
                        if let Some((_, gpu, _)) = self.gpu_meshes.get(&key) {
                            self.pick_ids.push(PickTarget::Segment(layer_id, id));
                            let pick = self.pick_ids.len() as u32;
                            mesh_jobs.push(MeshJob { mesh: MeshRef::Whole(key), pick, mvp, normal_matrix, color, silhouette, indices: 0..gpu.count, origin: [0.0; 3], shape: [1.0; 3] });
                        }
                    }
                }
                Some(source) => {
                    let info_ms = source.multiscale.clone().unwrap();
                    self.manifests.request(&key, 1.0);
                    let Some(manifest) = self.manifests.get(&key) else {
                        if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                            l.progress.0 += 1;
                        }
                        histogram_adds.push((layer_id, f64::INFINITY, f64::INFINITY, 0, 1));
                        continue;
                    };
                    counted.1 = 1;
                    self.pick_ids.push(PickTarget::Segment(layer_id, id));
                    let segment_pick = self.pick_ids.len() as u32;
                    let planes = ng_datasource::multiscale_mesh::frustum_planes(&mvp);
                    // The backend's requests: every fragment wanted, coarser first.
                    let mut wanted = vec![];
                    ng_datasource::multiscale_mesh::desired_chunks(&manifest, &mvp, &planes, detail_cutoff, pw, ph, &mut |lod, row, _, empty| {
                        if !empty {
                            wanted.push((lod, row));
                        }
                    });
                    for &(lod, row) in &wanted {
                        let fkey = (layer_id, k, id, row as u32);
                        self.fragments.request(&fkey, lod as f64);
                        counted.0 += 1;
                        counted.1 += u32::from(self.fragments.get(&fkey).is_some());
                        let (path, begin, length) = manifest.fragment_range(id, row);
                        self.fragment_reads.insert(fkey, (source.location.clone(), path, begin, length, info_ms.vertex_quantization_bits, lod != 0));
                    }
                    let fragments = &self.fragments;
                    let mut draws = vec![];
                    ng_datasource::multiscale_mesh::chunks_to_draw(
                        &manifest,
                        &mvp,
                        &planes,
                        detail_cutoff,
                        pw,
                        ph,
                        &mut |lod, row, render_scale| {
                            let has = fragments.get(&(layer_id, k, id, row as u32)).is_some();
                            histogram_adds.push((layer_id, manifest.lod_scales[lod] as f64 * scale_multiplier, render_scale, u32::from(has), u32::from(!has)));
                            has
                        },
                        &mut |lod, row, b, e, _| draws.push((lod, row, b, e)),
                    );
                    for (lod, row, b, e) in draws {
                        let fkey = (layer_id, k, id, row as u32);
                        let Some(data) = self.fragments.get(&fkey) else { continue };
                        let fresh = !matches!(self.gpu_fragments.get(&fkey), Some((d, _)) if Arc::ptr_eq(d, &data));
                        if fresh && !data.indices.is_empty() {
                            let gpu = ng_render::perspective::GpuMesh::new(&self.device, &data.positions, &data.normals, &data.indices);
                            self.gpu_fragments.insert(fkey, (data.clone(), gpu));
                        }
                        let (Some(&ib), Some(&ie)) = (data.sub_chunk_offsets.get(b), data.sub_chunk_offsets.get(e)) else { continue };
                        if ib == ie || !self.gpu_fragments.contains_key(&fkey) {
                            continue;
                        }
                        // `uFragmentOrigin` and `uFragmentShape` (float32).
                        let scale = (1u64 << lod) as f64;
                        let o = &manifest.octree[row * 5..row * 5 + 3];
                        let origin = std::array::from_fn(|i| (manifest.grid_origin[i] as f64 + o[i] as f64 * manifest.chunk_shape[i] as f64 * scale + manifest.vertex_offsets[lod * 3 + i] as f64) as f32);
                        let shape = std::array::from_fn(|i| (manifest.chunk_shape[i] as f64 * scale) as f32);
                        mesh_jobs.push(MeshJob { mesh: MeshRef::Fragment(fkey), pick: segment_pick, mvp, normal_matrix, color, silhouette, indices: ib..ie, origin, shape });
                    }
                }
            }
            // The manifest at an infinite scale (single-resolution meshes'
            // with their fragments, above).
            if is_multiscale {
                histogram_adds.push((layer_id, f64::INFINITY, f64::INFINITY, 1, 0));
            }
            // Counted with the layer's visible chunks (the layer bar's
            // progress), as the manifests and fragments are.
            if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                l.progress.0 += counted.0;
                l.progress.1 += counted.1;
            }
        }
        // `renderScaleHistogram.begin` and `add` for each mesh layer drawn.
        let frame = self.frame;
        for l in self.layers.iter_mut().filter(|l| histogram_layers.contains(&l.id)) {
            l.mesh_render_scale_histogram.begin(frame);
        }
        for (layer_id, spatial, render, present, missing) in histogram_adds {
            if let Some(l) = self.layers.iter_mut().find(|l| l.id == layer_id) {
                l.mesh_render_scale_histogram.add(spatial, render, present, missing);
            }
        }
        let mesh_draws: Vec<ng_render::perspective::MeshDraw> = mesh_jobs
            .iter()
            .filter_map(|j| {
                let mesh = match j.mesh {
                    MeshRef::Whole(key) => &self.gpu_meshes.get(&key)?.1,
                    MeshRef::Fragment(key) => &self.gpu_fragments.get(&key)?.1,
                };
                Some(ng_render::perspective::MeshDraw {
                    mesh,
                    mvp: j.mvp,
                    normal_matrix: j.normal_matrix,
                    light: light_vec,
                    color: j.color,
                    indices: j.indices.clone(),
                    fragment_origin: j.origin,
                    fragment_shape: j.shape,
                    // `isTransparent`: object alpha below 1, or silhouettes.
                    transparent: j.color[3] < 1.0 || j.silhouette > 0.0,
                    silhouette: j.silhouette,
                    pick: j.pick,
                })
            })
            .collect();
        // Image layers' volume rendering (`VolumeRenderingRenderLayer`, "on"):
        // each layer's chunks within the frustum at the scale its depth
        // samples call for.
        let mut volume_jobs: Vec<(usize, TextureKey, ng_render::perspective::VolumeUniforms)> = vec![];
        for l in &mut self.layers {
            l.volume_chunks.clear();
        }
        for li in 0..self.layers.len() {
            let managed = &self.layers[li];
            if !managed.shown() {
                continue;
            }
            let (mode, samples, gain) = state::volume_rendering_settings(&managed.spec.json);
            let mode = match mode {
                "on" => ng_render::user_shader::VolumeMode::On,
                "max" => ng_render::user_shader::VolumeMode::Max,
                "min" => ng_render::user_shader::VolumeMode::Min,
                _ => continue,
            };
            if !matches!(managed.kind(), Some(LayerKind::Image { .. })) {
                continue;
            }
            let json = managed.spec.json.clone();
            let id = managed.id;
            let frame = self.frame;
            let Some(state) = self.layers[li].render.as_mut() else { continue };
            let t = state.layer.data_type;
            let Some(shader) = state.volume_shader(&self.device, &self.perspective, &json, mode) else { continue };
            shader.uniforms.set(&self.queue, &ng_render::user_shader::uniform_bits(&shader.parsed, t, &shader.values, 1.0));
            let sources = chunks::panel_sources(&state.layer, info, position);
            let Some(vrc) = chunks::volume_rendering_chunks(&state.layer, &sources, info, &camera.projection, &camera.inv_view, &vp, samples) else { continue };
            let ps = &vrc.source;
            let model = ps.chunk_to_display;
            let mvp64 = ng_core::matrix::multiply(&vp.map(|x| x as f64), &model, 4);
            let mvp: [f32; 16] = std::array::from_fn(|i| mvp64[i] as f32);
            let inv_mvp = ng_core::glm::mat4_invert(&mvp).unwrap_or(mvp);
            // `getVolumeRenderingNearFarBounds`.
            let planes = ng_datasource::multiscale_mesh::frustum_planes(&mvp);
            let (mut min_z, mut max_z) = (0.0f64, 0.0f64);
            for k in 0..3 {
                let c = planes[16 + k] as f64;
                let (a, b) = (c * vrc.lower_clip[k], c * vrc.upper_clip[k]);
                min_z += a.min(b);
                max_z += a.max(b);
            }
            let near = -(planes[19] as f64);
            let far = planes[23] as f64;
            let (adjusted_near, adjusted_far) = (near.max(min_z), far.min(max_z));
            let params = [((adjusted_near - near) / (far - near)) as f32, ((adjusted_far - near) / (far - near)) as f32, (vrc.optimal_samples / samples) as f32, gain.exp() as f32];
            let offsets: Vec<u32> = ps.fixed.iter().map(|f| f.2).collect();
            let (mut present, mut missing) = (0u32, 0u32);
            let mut drawn = vec![];
            for grid in &vrc.chunks {
                let key: ChunkKey = (id, ps.scale, grid.clone());
                let requested = self.chunks.request(&key, 0.0).clone();
                let State::Ready(chunk) = requested else {
                    missing += 1;
                    continue;
                };
                present += 1;
                drawn.push((chunk.clone(), ps.chunk_display_dims, ps.fixed.clone()));
                let tkey: TextureKey = (ps.scale, grid.clone(), offsets.clone());
                let Some(state) = self.layers[li].render.as_mut() else { break };
                if !state.textures.contains_key(&tkey) {
                    let draw = chunks::ChunkDraw { scale: ps.scale, grid: grid.clone(), chunk_display_dims: ps.chunk_display_dims, fixed: ps.fixed.clone(), rect: [0.0; 4], ndc_to_texture: [0.0; 16], clip: chunks::ChunkDraw::UNCLIPPED };
                    let (format, upload) = state.upload_format();
                    let volume = upload_slab(&self.device, &self.queue, &state.layer, &chunk, &draw, format, upload, true);
                    state.textures.insert(tkey.clone(), ChunkTexture { volume, bindings: Default::default(), last_used: frame });
                }
                if let Some(t) = state.textures.get_mut(&tkey) {
                    t.last_used = frame;
                }
                let mut translation = [0.0f32; 4];
                let mut size = [1.0f32; 4];
                for (k, cd) in ps.chunk_display_dims.iter().enumerate() {
                    if let Some(c) = cd {
                        translation[k] = (grid[*c] as f64 * vrc.chunk_size[k]) as f32;
                        size[k] = chunk.size[*c] as f32;
                    }
                }
                let lower = vrc.lower_clip.map(|x| x as f32);
                let upper = vrc.upper_clip.map(|x| x as f32);
                volume_jobs.push((
                    li,
                    tkey,
                    ng_render::perspective::VolumeUniforms {
                        mvp,
                        inv_mvp,
                        translation,
                        chunk_size: size,
                        lower_clip: [lower[0], lower[1], lower[2], 0.0],
                        upper_clip: [upper[0], upper[1], upper[2], 0.0],
                        params,
                        steps: [samples as u32, 0, 0, 0],
                    },
                ));
            }
            // The histogram: the other scales as indicator bars (each
            // samples count once), then the chunks drawn.
            self.layers[li].volume_chunks = drawn;
            // Counted with the layer's visible chunks (the layer bar).
            self.layers[li].progress.0 += present + missing;
            self.layers[li].progress.1 += present;
            let histogram = &mut self.layers[li].volume_rendering_histogram;
            histogram.begin(frame);
            if present + missing > 0 {
                use ng_core::render_scale::{scale_from_histogram_offset_at, BIN_SIZE, NUM_BINS, VOLUME_RENDERING_INDICATOR_BAR_HEIGHT, VOLUME_RENDERING_ORIGIN};
                // `clampAndRoundResolutionTargetValue`.
                let max = scale_from_histogram_offset_at(NUM_BINS as f64, VOLUME_RENDERING_ORIGIN).log2().round().exp2() - 1.0;
                let _ = BIN_SIZE;
                let round = |v: f64| v.round().clamp(VOLUME_RENDERING_ORIGIN.exp2(), max) as i64;
                let mut stored = vec![round(vrc.optimal_samples)];
                for (index, &(spacing, optimal)) in (0..vrc.scales.len()).rev().zip(&vrc.scales) {
                    let r = round(optimal);
                    if index != vrc.active && !stored.contains(&r) {
                        histogram.add_indicator(spacing, optimal, VOLUME_RENDERING_INDICATOR_BAR_HEIGHT);
                        stored.push(r);
                    }
                }
                histogram.add(vrc.physical_spacing, vrc.optimal_samples, present, missing);
            }
        }
        // scene mode hands the depth to the host, which binds it to composite
        let sampled = self.scene.is_some();
        if !matches!(&self.perspective_frame, Some(pf) if pf.target.width == phys.width && pf.target.height == phys.height && pf.target.sampled_depth == sampled) {
            let target = if sampled { PanelTarget::with_sampled_depth(&self.device, phys.width, phys.height) } else { PanelTarget::new(&self.device, phys.width, phys.height) };
            let to_panel = self.panel_compositor.create_binding(&self.device, &target.color_view);
            let to_frame = self.compositor.create_binding(&self.device, &target.panel_view);
            self.perspective_frame = Some(PanelFrame { target, to_panel, to_frame });
        }
        let pf = self.perspective_frame.as_ref().unwrap();
        self.perspective.draw(
            &self.device,
            encoder,
            &pf.target.flipped_view,
            &pf.target.depth_view,
            (phys.width, phys.height),
            // scene mode: transparent black outside the data (premultiplied,
            // so the host's frame shows through), and no slice background
            if self.scene.is_some() { [0.0; 3] } else { self.projection_background },
            (self.scene.is_none() && !self.settings.hide_cross_section_background_3d).then_some(cross_section_background),
            &mesh_draws,
            &planes,
            &lines,
            &circles,
            &triangles,
            axes,
            &transparent,
            &volume_jobs
                .iter()
                .filter_map(|(li, tkey, uniforms)| {
                    let state = self.layers[*li].render.as_ref()?;
                    let shader = state.volume.as_ref()?;
                    Some(ng_render::perspective::VolumeDraw { layer: *li, pipeline: &shader.pipeline, volume: state.textures.get(tkey)?.volume.view.as_wgpu()?, lookup: &shader.uniforms.view, uniforms: *uniforms })
                })
                .collect::<Vec<_>>(),
        );
        // `attemptToIssuePickRequest`: the window around the pointer, if
        // it is over this panel and no request is outstanding.
        if self.pending_pick.is_none() {
            if let Some((Panel::Perspective, p)) = self.pointer_panel() {
                let s = self.scale_factor;
                let x = (p.x * s).floor() as i64;
                let y = (phys.height as f64 - p.y * s).floor() as i64;
                if let Some(readback) = self.perspective.read_pick(&self.device, encoder, x, y, PICK_RADIUS) {
                    let display = info.display_dimension_indices;
                    self.pending_pick = Some(PendingPick { readback, inv_vp, position: position.to_vec(), display, ids: self.pick_ids.clone(), mapped: false });
                }
            }
        }
        // scene mode: the host composites the 3D panel itself (`scene_output`)
        if let Some(target) = target {
            self.flipper.flip(&self.device, encoder, &pf.target.flipped_view, &pf.target.panel_view);
            self.compositor.draw(&self.queue, encoder, target, phys, &pf.to_frame, cross_section_background);
        }
    }

    /// Draws a frame into `target`: `width` × `height` physical pixels in the
    /// format the viewer was created with. Requests the chunks it needs;
    /// [`poll`](Self::poll) and redraw as they arrive.
    pub fn render(&mut self, target: &wgpu::TextureView) {
        self.render_into(Some(target));
    }

    /// [`render`](Self::render), or with no target (scene mode): the panels
    /// are drawn offscreen and nothing is composited into a layout — the
    /// overlays drawn over a slice panel (annotations, axis lines, scale
    /// bar) only show there, so they are left out too.
    pub(crate) fn render_into(&mut self, target: Option<&wgpu::TextureView>) {
        self.frame += 1;
        self.process_legends();
        self.update_zoom_gates();
        // the downloads at once tuned as they download: pushback is chunks
        // failing and requests the server made us ask again (429, 5xx:
        // retried by ng-kvstore, so mostly not failed)
        if let (Some(a), Some(clock)) = (&mut self.adaptive_downloads, self.chunks.clock) {
            let mut stats = self.chunks.stats;
            stats.failed += ng_kvstore::retried().saturating_sub(self.retried_base);
            a.observe(clock(), stats);
            self.settings.concurrent_downloads = a.concurrency as f64;
        }
        self.chunks.max_concurrent_downloads = self.settings.concurrent_downloads.max(1.0) as usize;
        self.chunks.memory_budget = self.settings.system_memory_limit.max(0.0) as usize;
        // The other caches follow the same setting (each a quarter of it), so
        // `systemMemoryLimit` bounds the whole viewer: in a 32-bit wasm memory
        // (4 GB, shared with everything else) eight caches of 2 GB each could
        // outgrow it, and the allocation that fails aborts the module.
        let quarter = (self.settings.system_memory_limit.max(0.0) / 4.0) as usize;
        self.meshes.memory_budget = quarter;
        self.skeletons.memory_budget = quarter;
        self.annotation_cells.memory_budget = quarter;
        self.annotations_by_id.memory_budget = quarter;
        self.manifests.memory_budget = quarter;
        self.fragments.memory_budget = quarter;
        // a layer gone (the state replaced, a source switched): its downloads
        // are cancelled and its data freed, before the new layers' requests
        let live: Vec<LayerId> = self.layers.iter().map(|l| l.id).collect();
        if live != self.live_layers {
            let keep = |id: &LayerId| live.contains(id);
            self.chunks.retain_keys(|k| keep(&k.0));
            self.meshes.retain_keys(|k| keep(&k.0));
            self.skeletons.retain_keys(|k| keep(&k.0));
            self.annotation_cells.retain_keys(|k| keep(&k.0));
            self.annotations_by_id.retain_keys(|k| keep(&k.0));
            self.manifests.retain_keys(|k| keep(&k.0));
            self.fragments.retain_keys(|k| keep(&k.0));
            self.live_layers = live;
        }
        self.chunks.poll();
        self.chunks.begin_frame();
        self.meshes.poll();
        self.meshes.begin_frame();
        self.skeletons.poll();
        self.skeletons.begin_frame();
        self.annotation_cells.poll();
        self.annotation_cells.begin_frame();
        for layer in &mut self.layers {
            layer.update_annotation_shader(&self.device);
        }
        self.annotations_by_id.poll();
        self.annotations_by_id.begin_frame();
        for key in std::mem::take(&mut self.annotation_wants) {
            self.annotations_by_id.request(&key, 0.0);
        }
        self.manifests.poll();
        self.manifests.begin_frame();
        self.fragments.poll();
        self.fragments.begin_frame();
        let panel_rects = self.panel_rects();
        for l in &mut self.layers {
            l.progress = (0, 0);
        }

        // Black between panels (the page behind neuroglancer's canvas).
        let layout = self.layout;
        if let Some(target) = target {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            drop(threers::renderer::fullscreen::begin(&mut encoder, target, threers::renderer::fullscreen::Load::Clear));
            self.queue.submit([encoder.finish()]);
        }
        // Each slice panel's view, for the 3D panel: (axes, invViewMatrix,
        // physical size).
        let mut slice_views: Vec<SliceViewFrame> = vec![];

        let hovered = self.hovered_segments();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let (lw, lh) = self.logical_size();
        let position = self.nav.position().to_vec();
        let info = self.nav.render_info().clone();
        let valid = self.nav.valid();
        let frame = self.frame;
        let background = self.background;
        let has_slices = panel_rects.iter().any(|(p, _)| matches!(p, Panel::Slice(_)));
        let slice_skeletons = if has_slices && valid { self.skeleton_draws(&hovered, &info, true) } else { vec![] };
        for &(panel, phys) in &panel_rects {
            let Panel::Slice(axes) = panel else { continue };
            let i = NamedAxes::ALL.iter().position(|&a| a == axes).unwrap();
            let logical = self.panel_logical_rect(panel, lw, lh).unwrap();
            // a new target holds nothing yet: drawn whatever was drawn before
            let fresh = !matches!(&self.panel_frames[i], Some(pf) if pf.target.width == phys.width && pf.target.height == phys.height);
            let pf = match &mut self.panel_frames[i] {
                Some(pf) if !fresh => pf,
                slot => {
                    let target = PanelTarget::new(&self.device, phys.width, phys.height);
                    let to_panel = self.panel_compositor.create_binding(&self.device, &target.color_view);
                    let to_frame = self.compositor.create_binding(&self.device, &target.panel_view);
                    slot.insert(PanelFrame { target, to_panel, to_frame })
                }
            };
            let panel_rect_local = PixelRect { x: 0, y: 0, width: phys.width, height: phys.height };
            if !valid {
                pf.target.clear(&mut encoder);
                self.slice_drawn[i] = None;
                if let Some(target) = target {
                    self.panel_compositor.draw(&self.queue, &mut encoder, &pf.target.panel_view, panel_rect_local, &pf.to_panel, background);
                    self.compositor.draw(&self.queue, &mut encoder, target, phys, &pf.to_frame, background);
                }
                continue;
            }
            // `velocityEstimator.addSample` (with a clock to time it by).
            let prefetching = match (self.settings.prefetch, self.chunks.clock) {
                (true, Some(clock)) => {
                    self.velocity[i].add_sample(&position, clock());
                    true
                }
                _ => false,
            };
            let mut sp = self.panels[i].1.clone();
            let inv_view = sp.inv_view_matrix(&mut self.nav);
            self.panels[i].1 = sp;
            slice_views.push((i, inv_view, (phys.width, phys.height)));
            let camera = PanelCamera::new(inv_view, logical.width as f64, logical.height as f64);
            let sx = phys.width as f64 / logical.width as f64;
            let sy = phys.height as f64 / logical.height as f64;
            // Visible layers in order; the first drawn replaces, later ones
            // blend (`setGLBlendMode`). Their draws are collected first and
            // hashed with all that goes to the GPU (chunks, rects, uniforms,
            // shaders, the segment set): the same as this panel last drew,
            // its colour is still right and nothing is drawn — the chunks
            // are still requested (and their textures kept), as they must be
            // each frame. A host's camera orbiting the slices moves none.
            let mut drawn = 0;
            // Per layer drawn: its index, draw order, and chunk draws.
            let mut layer_draws: Vec<(usize, usize, Vec<ChunkOp>)> = vec![];
            let mut fingerprint = std::collections::hash_map::DefaultHasher::new();
            (phys.width, phys.height).hash(&mut fingerprint);
            for (li, managed) in self.layers.iter_mut().enumerate() {
                if !managed.shown() {
                    continue;
                }
                let id = managed.id;
                let Some(state) = managed.render.as_mut() else { continue };
                // A user shader's control values (the opacity first).
                if let (Some(shader), LayerKind::Image { opacity }) = (&state.shader, &state.kind) {
                    let values = ng_render::user_shader::uniform_bits(&shader.parsed, state.layer.data_type, &shader.values, *opacity);
                    values.hash(&mut fingerprint);
                    shader.uniforms.set(&self.queue, &values);
                }
                // A replaced visible-set texture invalidates the bindings.
                if let Some((set, seen)) = &mut state.visible {
                    if *seen != set.generation {
                        *seen = set.generation;
                        for t in state.textures.values_mut() {
                            t.bindings = Default::default();
                        }
                    }
                    (set.generation, set.revision).hash(&mut fingerprint);
                }
                (li, id, state.epoch).hash(&mut fingerprint);
                let sources = panel_sources(&state.layer, &info, &position);
                let pixel_size = camera.pixel_size(&info);
                let target = ng_core::render_scale::parse_target(managed.spec.json.get("crossSectionRenderScale"), 1.0);
                let scales = visible_scales(&sources, pixel_size, target);
                if !scales.is_empty() {
                    managed.render_scale_histogram.begin(frame);
                }
                let order = drawn;
                drawn += 1;
                order.hash(&mut fingerprint);
                let mut ops = vec![];
                // `scales` runs coarse to fine. Draw finest first: under the
                // depth test, coarser data only fills gaps.
                for (priority_index, &s) in scales.iter().enumerate().rev() {
                    let Some(ps) = sources[s].as_ref() else { continue };
                    let draws = visible_chunks(&state.layer, ps, &camera);
                    // `renderScaleHistogram.add`: the median voxel size
                    // (`effectiveVoxelSize` is a `Float32Array`).
                    let v = ps.effective_voxel_size.map(|x| x as f32 as f64);
                    let median = ng_core::render_scale::median_of_3(v[0], v[1], v[2]);
                    let present = draws.iter().filter(|d| self.chunks.get(&(id, s, d.grid.clone())).is_some()).count() as u32;
                    if !draws.is_empty() {
                        managed.render_scale_histogram.add(median, median / pixel_size, present, draws.len() as u32 - present);
                    }
                    // Chunks the moving position will reach, behind every
                    // visible one (the `PREFETCH` tier).
                    if prefetching && !draws.is_empty() {
                        let src = &state.layer.scales[s];
                        let display_dims: Vec<usize> = ps.chunk_display_dims.iter().flatten().copied().collect();
                        let sizes: Vec<u32> = src.spec.chunk_data_size.clone();
                        let rank = sizes.len().min(ps.position_voxel.len());
                        let position_chunk: Vec<i64> = (0..rank).map(|c| (ps.position_voxel[c] / sizes[c] as f64).floor() as i64).collect();
                        let within: Vec<f64> = (0..rank).map(|c| ps.position_voxel[c] - position_chunk[c] as f64 * sizes[c] as f64).collect();
                        let lower: Vec<f64> = (0..rank).map(|c| src.lower_clip_bound.as_ref().map_or(src.spec.lower_voxel_bound[c] as f64, |b| b[c] as f64)).collect();
                        let upper: Vec<f64> = (0..rank).map(|c| src.upper_clip_bound.as_ref().map_or(src.spec.upper_voxel_bound[c] as f64, |b| b[c] as f64)).collect();
                        let geometry = prefetch::SourceGeometry {
                            global_to_chunk: &ps.global_to_chunk[..rank],
                            display_dims: &display_dims,
                            chunk_data_size: &sizes[..rank],
                            position_chunk: &position_chunk,
                            position_in_chunk: &within,
                            lower_clip: &lower,
                            upper_clip: &upper,
                        };
                        let offsets = prefetch::prefetch_offsets(&self.velocity[i], &geometry);
                        if !offsets.is_empty() {
                            let visible: Vec<Vec<u64>> = draws.iter().map(|d| d.grid.clone()).collect();
                            for (grid, probability) in prefetch::prefetch_chunks(&visible, &offsets) {
                                self.chunks.request_prefetch(&(id, s, grid), PREFETCH_TIER + 1e9 * priority_index as f64 + probability);
                            }
                        }
                    }
                    for draw in draws {
                        // Finer scales first; within one, nearer the center first.
                        let cx = draw.rect[0] + draw.rect[2] / 2.0 - camera.width / 2.0;
                        let cy = draw.rect[1] + draw.rect[3] / 2.0 - camera.height / 2.0;
                        let priority = 1e9 * priority_index as f64 - (cx * cx + cy * cy).sqrt();
                        let key: ChunkKey = (id, s, draw.grid.clone());
                        let requested = self.chunks.request(&key, priority).clone();
                        managed.progress.0 += 1;
                        let State::Ready(chunk) = requested else { continue };
                        managed.progress.1 += 1;
                        // Panel-local pixels of the offscreen target.
                        let x0 = (draw.rect[0] * sx).floor().max(0.0);
                        let y0 = (draw.rect[1] * sy).floor().max(0.0);
                        let x1 = ((draw.rect[0] + draw.rect[2]) * sx).ceil().min(phys.width as f64);
                        let y1 = ((draw.rect[1] + draw.rect[3]) * sy).ceil().min(phys.height as f64);
                        if x1 <= x0 || y1 <= y0 {
                            continue;
                        }
                        let rect = PixelRect {
                            x: x0 as u32,
                            y: y0 as u32,
                            width: (x1 - x0) as u32,
                            height: (y1 - y0) as u32,
                        };
                        // The texture matrix was for the logical rectangle; the
                        // physical one is the same region of the panel, rounded
                        // outward, so re-derive its NDC mapping.
                        let ndc_to_voxel = Matrix4 {
                            elements: subrect_matrix(draw.ndc_to_texture, draw.rect, [x0 / sx, y0 / sy, (x1 - x0) / sx, (y1 - y0) / sy]),
                        };
                        let mut params = state.slice_params(ndc_to_voxel, hovered.get(li).copied().flatten());
                        // an image layer's free parameters: its clip bounds,
                        // checked per pixel by the shader (user_shader.rs)
                        if matches!(state.kind, LayerKind::Image { .. }) {
                            let [lo, hi] = draw.clip;
                            params.params[2] = [lo[0], lo[1], lo[2], 0.0];
                            params.params[3] = [hi[0], hi[1], hi[2], 1.0];
                        }
                        let offsets: Vec<u32> = draw.fixed.iter().map(|f| f.2).collect();
                        let tkey: TextureKey = (s, draw.grid.clone(), offsets);
                        let (format, upload) = state.upload_format();
                        let texture = state.textures.entry(tkey.clone()).or_insert_with(|| ChunkTexture {
                            volume: upload_slab(&self.device, &self.queue, &state.layer, &chunk, &draw, format, upload, matches!(state.kind, LayerKind::Image { .. })),
                            bindings: Default::default(),
                            last_used: frame,
                        });
                        texture.last_used = frame;
                        tkey.hash(&mut fingerprint);
                        (rect.x, rect.y, rect.width, rect.height).hash(&mut fingerprint);
                        params.ndc_to_voxel.elements.map(f32::to_bits).hash(&mut fingerprint);
                        params.params.map(|r| r.map(f32::to_bits)).hash(&mut fingerprint);
                        ops.push((tkey, rect, params));
                    }
                }
                layer_draws.push((li, order, ops));
            }
            let fingerprint = fingerprint.finish();
            if fresh || self.slice_drawn[i] != Some(fingerprint) {
                self.slice_drawn[i] = Some(fingerprint);
                self.slice_redraws += 1;
                pf.target.clear(&mut encoder);
                for (li, order, ops) in &layer_draws {
                    let Some(state) = self.layers[*li].render.as_mut() else { continue };
                    let pipeline = state.pipelines.for_layer(*order);
                    // The lookup (the fields, apart from the textures).
                    let lookup = state.visible.as_ref().map(|(set, _)| &set.view).or(state.shader.as_ref().map(|s| &s.uniforms.view));
                    let mut pass = pf.target.begin_layer(&mut encoder, *order);
                    for (tkey, rect, params) in ops {
                        let Some(texture) = state.textures.get_mut(tkey) else { continue };
                        // Bindings are made per pipeline: a layer's place in
                        // the drawing order picks it.
                        let slot = &mut texture.bindings[i][(*order > 0) as usize];
                        if slot.is_none() {
                            *slot = Some(match lookup {
                                Some(view) => pipeline.create_binding_with_lookup(&self.device, &texture.volume, view),
                                None => pipeline.create_binding(&self.device, &texture.volume),
                            });
                        }
                        let binding = slot.as_ref().unwrap();
                        pipeline.update(&self.queue, binding, params);
                        pipeline.draw_in_pass(&mut pass, *rect, binding);
                    }
                }
            }
            if target.is_some() {
                self.panel_compositor.draw(&self.queue, &mut encoder, &pf.target.panel_view, panel_rect_local, &pf.to_panel, background);
            }
            // Annotations over the panel: each visible layer's data bounds.
            let mut lines = vec![];
            for managed in self.layers.iter().filter(|l| l.shown()) {
                let Some(state) = managed.render.as_ref() else { continue };
                let color = managed.annotation_color();
                for (lower, upper) in if self.settings.show_default_annotations { managed.data_bounds() } else { vec![] } {
                    let Some((polygon, clip)) = chunks::box_cross_section(&state.layer, &lower, &upper, &info, &position, &camera) else {
                        continue;
                    };
                    for k in 0..polygon.len() {
                        let (a, b) = (polygon[k], polygon[(k + 1) % polygon.len()]);
                        lines.push(ng_render::lines::Line {
                            a: [a[0] * sx, a[1] * sy],
                            b: [b[0] * sx, b[1] * sy],
                            width: 1.0,
                            feather: 1.0,
                            color: [color[0], color[1], color[2], clip as f32],
                        });
                    }
                }
            }
            // Annotation layers' annotations (`SliceViewAnnotationLayer`):
            // boxes' cross-sections with the lines above; points and lines
            // in clip coordinates, fading with depth.
            let mut marks = AnnotationMarks::default();
            let relative_depth = self.nav.relative_depth_range(View::CrossSection);
            let vp = camera.view_projection();
            // Segmentation layers' skeletons (`SliceViewPanelSkeletonLayer`):
            // each segment's edges, then its nodes.
            // What can be picked here (`pickIDs`), ID `k + 1` entry `k`.
            let mut targets: Vec<PickTarget> = vec![];
            let mut skeleton_batches = vec![];
            for d in &slice_skeletons {
                let to_clip = |p: [f64; 3]| {
                    let c = ng_core::matrix::transform_point4(&vp, p);
                    [c[0], c[1], -c[2] / relative_depth, 1.0]
                };
                let (mut l, mut c) = (vec![], vec![]);
                d.marks(&to_clip, 1.0, &mut l, &mut c);
                targets.push(PickTarget::Segment(d.segment.0, d.segment.1));
                let pick = targets.len() as u32;
                l.iter_mut().for_each(|x| x.pick = pick);
                c.iter_mut().for_each(|x| x.pick = pick);
                skeleton_batches.push(ng_render::annotations::MarkBatch { lines: l, circles: c, ellipses: vec![] });
            }
            // Box cross-sections' lines: from which line, which pick.
            let first_box_line = lines.len();
            let mut box_picks: Vec<(usize, u32)> = vec![];
            // Precomputed sources' spatial index cells in view.
            let mut slice_projection = [0.0; 16];
            slice_projection[0] = 2.0 / camera.width;
            slice_projection[5] = -2.0 / camera.height;
            slice_projection[10] = -1.0 / relative_depth;
            slice_projection[15] = 1.0;
            let slice_view = ng_core::matrix::inverse4(&camera.inv_view);
            let mut spatial: HashMap<LayerId, Vec<annotations::Annotation>> = HashMap::new();
            for li in 0..self.layers.len() {
                if !self.layers[li].shown() {
                    continue;
                }
                if let Some(list) = spatial_annotations(&mut self.layers, &mut self.annotation_cells, li, AnnotationView::CrossSection, frame, &info, &position, &slice_projection, &slice_view, (phys.width as f64, phys.height as f64)) {
                    spatial.insert(self.layers[li].id, list);
                }
            }
            // Each layer's styles (its shader's), once computed.
            let mut styles: HashMap<LayerId, Arc<Vec<ng_render::annotation_shader::AnnotationStyle>>> = HashMap::new();
            for managed in self.layers.iter().filter(|l| l.shown()) {
                let Some(ann) = managed.annotation.as_ref() else { continue };
                let list: &[annotations::Annotation] = spatial.get(&managed.id).map_or(&ann.annotations, |v| v);
                if let Some(s) = annotation_styles(&mut self.annotation_styles, &self.device, &self.queue, &self.wake, managed, list, false, frame) {
                    styles.insert(managed.id, s);
                }
            }
            for managed in self.layers.iter().filter(|l| l.shown()) {
                let Some(ann) = managed.annotation.as_ref() else { continue };
                let list: &[annotations::Annotation] = spatial.get(&managed.id).map_or(&ann.annotations, |v| v);
                let Some(layer_styles) = styles.get(&managed.id) else { continue };
                let to_clip = |model: &[f64]| -> [f64; 4] {
                    let d = model_to_display(&ann.layer, &info, model);
                    let c = ng_core::matrix::transform_point4(&vp, d);
                    // `mat4.ortho(…, -relativeDepthRange, relativeDepthRange)`.
                    [c[0], c[1], -c[2] / relative_depth, 1.0]
                };
                let hovered = self.annotation_hover.as_ref().filter(|(l, _)| *l == managed.id).map(|(_, id)| id.as_str());
                let mut register = |id: &str, part: AnnotationPart| {
                    targets.push(PickTarget::Annotation(managed.id, id.to_owned(), part));
                    targets.len() as u32
                };
                let mut ellipses = vec![None; list.len()];
                for (k, a) in list.iter().enumerate() {
                    if let annotations::Geometry::Ellipsoid { center, radii } = &a.geometry {
                        let fill = highlight(layer_styles[k].ellipsoid_fill_color, hovered == Some(a.id.as_str()));
                        let to_display = |m: &[f64]| model_to_display(&ann.layer, &info, m);
                        let display_to_clip = |d: [f64; 3]| {
                            let c = ng_core::matrix::transform_point4(&vp, d);
                            [c[0], c[1], -c[2] / relative_depth, 1.0]
                        };
                        if let Some(corners) = ellipsoid_cross_section(&camera, to_display(center), linear_to_display(&ann.layer, &info, radii)) {
                            // `setEllipsoidFillColor` (`setColor`'s: half alpha here).
                            ellipses[k] = Some(ng_render::annotations::Ellipse { corners: corners.map(display_to_clip), color: fill, pick: 0 });
                        }
                    }
                }
                marks.add(list, &ellipses, layer_styles, hovered, &mut register, &to_clip, &mut |lower, upper, color, width, pick| {
                    let Some((polygon, clip)) = chunks::box_cross_section(&ann.layer, lower, upper, &info, &position, &camera) else { return };
                    box_picks.push((lines.len(), pick));
                    for k in 0..polygon.len() {
                        let (a, b) = (polygon[k], polygon[(k + 1) % polygon.len()]);
                        // `vColor.a * getLineAlpha() * vClipCoefficient`.
                        lines.push(ng_render::lines::Line { a: [a[0] * sx, a[1] * sy], b: [b[0] * sx, b[1] * sy], width, feather: 1.0, color: [color[0], color[1], color[2], color[3] * clip as f32] });
                    }
                });
            }
            // Overlays draw on the panel turned upside down, as in
            // neuroglancer's GL framebuffer, then it turns back.
            skeleton_batches.append(&mut marks.batches);
            if target.is_some() {
                self.flipper.flip(&self.device, &mut encoder, &pf.target.panel_view, &pf.target.flipped_view);
                self.lines.draw(&self.device, &mut encoder, &pf.target.flipped_view, panel_rect_local, &lines, true);
                self.annotation_renderer.draw_batches(&self.device, &mut encoder, &pf.target.flipped_view, panel_rect_local, &skeleton_batches, true);
            }
            // The pickable marks in drawing order, in panel pixels.
            let (w, h) = (phys.width as f64, phys.height as f64);
            let mut shapes = vec![];
            for (k, l) in lines.iter().enumerate().skip(first_box_line) {
                let Some(&(_, pick)) = box_picks.iter().rev().find(|(start, _)| *start <= k) else { continue };
                shapes.push(PickShape::segment(l.a, l.b, l.width, l.feather, pick));
            }
            for b in &skeleton_batches {
                shapes.extend(b.lines.iter().filter_map(|l| PickShape::from_line(l, w, h)));
                shapes.extend(b.ellipses.iter().filter_map(|e| PickShape::from_ellipse(e, w, h)));
                shapes.extend(b.circles.iter().filter_map(|c| PickShape::from_circle(c, w, h)));
            }
            self.slice_picks[i] = SlicePicks { size: (phys.width, phys.height), targets, shapes };
            self.last_lines[i] = lines;
            if self.show_axis_lines {
                // `computeAxisLineMatrix`: from the position, `axisLength`
                // pixels (¼ · 1.5 of the panel's smaller side) along each
                // display axis; `disableZProjection` keeps them unclipped.
                let axis_length = (logical.width.min(logical.height) as f64 / 4.0) * 1.5;
                let zoom = self.nav.zoom(View::CrossSection);
                let vp = camera.view_projection();
                let mut origin = [0.0; 3];
                for (k, o) in origin.iter_mut().enumerate() {
                    let g = info.display_dimension_indices[k];
                    *o = if g < 0 { 0.0 } else { position[g as usize] as f64 };
                }
                let ndc = |p: [f64; 3]| {
                    let q = ng_core::matrix::transform_point4(&vp, p);
                    [q[0] as f32, q[1] as f32]
                };
                let ends = std::array::from_fn(|k| {
                    let mut p = origin;
                    p[k] += axis_length * zoom / info.canonical_voxel_factors[k];
                    ndc(p)
                });
                let o = ndc(origin);
                if target.is_some() {
                    self.axis_lines.draw(&self.device, &mut encoder, &pf.target.flipped_view, panel_rect_local, o, ends, true);
                }
                // Recorded with the annotation lines (panel pixels).
                let px = |p: [f32; 2]| [(p[0] as f64 + 1.0) / 2.0 * phys.width as f64, (1.0 - p[1] as f64) / 2.0 * phys.height as f64];
                for e in ends {
                    self.last_lines[i].push(ng_render::lines::Line { a: px(o), b: px(e), width: 1.0, feather: 0.0, color: [1.0; 4] });
                }
            }
            // The scale bar: in `4panel` and `4panel-alt` on the xy panel only
            // (the others get `sliceViewerStateWithoutScaleBar`).
            let has_scale_bar = match layout {
                Layout::FourPanel | Layout::FourPanelAlt => axes == NamedAxes::Xy,
                _ => true,
            };
            if self.show_scale_bar && has_scale_bar && target.is_some() {
                let options = ng_render::scale_bar::ScaleBarOptions::default();
                let factors = self.nav.relative_display_factors();
                let zoom = self.nav.zoom(View::CrossSection);
                let bars = ng_render::scale_bar::scale_bars(&info, &factors, zoom, logical.width as f64, &options);
                let mut bottom = options.bottom_offset * options.scale_factor;
                for bar in &bars {
                    let key = (bar.text(), bar.length_in_pixels.to_bits());
                    let painter = &self.scale_bar_painter;
                    let image = self.scale_bar_cache.entry(key).or_insert_with(|| painter.paint(bar, &options));
                    // `gl.viewport(left, bottom, …)`, from the panel's bottom:
                    // the flipped panel's top-down y.
                    let x = (options.left_offset * options.scale_factor) as i32;
                    self.scale_bar_renderer.draw(&self.device, &self.queue, &mut encoder, &pf.target.flipped_view, image, x, bottom as i32, true);
                    let y = phys.height as i32 - bottom as i32 - image.height as i32;
                    // Recorded for tests, as one line as thick as the bar.
                    let (x0, y0, x1, y1) = (x as f64, y as f64, (x + image.width as i32) as f64, (y + image.height as i32) as f64);
                    self.last_lines[i].push(ng_render::lines::Line {
                        a: [x0, (y0 + y1) / 2.0],
                        b: [x1, (y0 + y1) / 2.0],
                        width: y1 - y0,
                        feather: 0.0,
                        color: [1.0; 4],
                    });
                    bottom += image.height as f64 + options.margin_between * options.scale_factor;
                }
            }
            if let Some(target) = target {
                self.flipper.flip(&self.device, &mut encoder, &pf.target.flipped_view, &pf.target.panel_view);
                self.compositor.draw(&self.queue, &mut encoder, target, phys, &pf.to_frame, background);
            }
        }
        if valid {
            if let Some(&(_, phys)) = panel_rects.iter().find(|(p, _)| *p == Panel::Perspective) {
                self.render_perspective(&mut encoder, target, phys, &slice_views, &info, &position, background);
            }
        }
        self.queue.submit([encoder.finish()]);
        if let Some(pick) = self.pending_pick.as_mut().filter(|p| !p.mapped) {
            pick.readback.map(self.wake.clone());
            pick.mapped = true;
            // Another frame polls for it: natively and on WebGL2 the mapping
            // only completes as the device is polled, and nothing else may
            // ask for one.
            (self.wake)();
        }
        if valid {
            self.compute_data_histograms(&slice_views);
        }

        // Start loads; drop textures of chunks no longer drawn or held.
        let scales: HashMap<LayerId, Vec<Arc<ng_datasource::model::VolumeSource>>> = self
            .layers
            .iter()
            .filter_map(|l| Some((l.id, l.render.as_ref()?.layer.scales.clone())))
            .collect();
        let mesh_sources: HashMap<LayerId, Vec<ng_datasource::mesh::MeshSource>> = self.layers.iter().map(|l| (l.id, l.mesh_sources())).collect();
        self.meshes.end_frame(|(id, k, segment)| {
            let source = mesh_sources.get(id).and_then(|v| v.get(*k)).cloned();
            let segment = *segment;
            Box::pin(async move {
                let source = source.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                let mesh = ng_datasource::mesh::load(&source, segment).await?;
                let bytes = mesh.bytes();
                Ok((mesh, bytes))
            })
        });
        let skeleton_sources: HashMap<LayerId, Vec<ng_datasource::skeleton::SkeletonSource>> = self.layers.iter().map(|l| (l.id, l.skeleton_sources())).collect();
        self.skeletons.end_frame(|(id, k, segment)| {
            let source = skeleton_sources.get(id).and_then(|v| v.get(*k)).cloned();
            let segment = *segment;
            Box::pin(async move {
                let source = source.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                let skeleton = ng_datasource::skeleton::load(&source, segment).await?.unwrap_or_default();
                let bytes = skeleton.bytes();
                Ok((skeleton, bytes))
            })
        });
        let annotation_infos: HashMap<LayerId, Arc<ng_datasource::annotation::AnnotationInfo>> = self.layers.iter().filter_map(|l| Some((l.id, l.annotation.as_ref()?.spatial.clone()?))).collect();
        self.annotation_cells.end_frame(|(id, level, cell)| {
            let info = annotation_infos.get(id).cloned();
            let (level, cell) = (*level, cell.clone());
            Box::pin(async move {
                let info = info.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                // Finest first here; the info lists the coarsest first.
                let decoded = ng_datasource::annotation::load_cell(&info, info.spatial.len() - 1 - level, &cell).await?;
                let list: Vec<annotations::Annotation> = decoded.iter().map(|a| annotations::from_precomputed(info.annotation_type, a)).collect();
                let bytes = list.len() * (64 + 8 * info.rank * 2);
                Ok((list, bytes))
            })
        });
        // Styles not drawn this frame (nor being computed) are dropped.
        let frame = self.frame;
        self.annotation_styles.retain(|_, slot| slot.frame == frame || matches!(slot.state, StyleState::Pending(_)));
        let by_id_infos = annotation_infos.clone();
        self.annotations_by_id.end_frame(|(id, annotation)| {
            let info = by_id_infos.get(id).cloned();
            let annotation = *annotation;
            Box::pin(async move {
                let info = info.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                let a = ng_datasource::annotation::load_by_id(&info, annotation).await?.map(|a| annotations::from_precomputed(info.annotation_type, &a));
                Ok((a, 64))
            })
        });
        let meshes = &self.meshes;
        self.gpu_meshes.retain(|k, (data, _, _)| meshes.get(k).is_some_and(|d| Arc::ptr_eq(&d, data)));
        let manifest_sources = mesh_sources;
        self.manifests.end_frame(|(id, k, segment)| {
            let source = manifest_sources.get(id).and_then(|v| v.get(*k)).cloned();
            let segment = *segment;
            Box::pin(async move {
                let source = source.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                let multiplier = source.multiscale.as_ref().map_or(1.0, |m| m.lod_scale_multiplier);
                // `<id>.index`, or the segment's entry in the shards.
                let (bytes, shard) = match source.multiscale.as_ref().and_then(|m| m.sharding) {
                    Some(sharding) => {
                        let (bytes, info) = ng_datasource::sharded::read_with_shard_info(&source.location, &sharding, segment as u128).await?.ok_or_else(|| anyhow::anyhow!("no manifest for segment {segment}"))?;
                        (bytes, Some(info))
                    }
                    None => (source.location.read(&format!("{segment}.index"), ng_kvstore::ByteRange::Full).await?.ok_or_else(|| anyhow::anyhow!("no manifest for segment {segment}"))?, None),
                };
                let mut manifest = ng_datasource::multiscale_mesh::parse_manifest(&bytes, multiplier)?;
                manifest.shard = shard;
                let size = manifest.octree.len() * 4 + manifest.offsets.len() * 8;
                Ok((manifest, size))
            })
        });
        let reads = std::mem::take(&mut self.fragment_reads);
        self.fragments.end_frame(|key| {
            let read = reads.get(key).cloned();
            Box::pin(async move {
                let (location, path, offset, length, bits, partition) = read.ok_or_else(|| anyhow::anyhow!("the fragment is not wanted"))?;
                let bytes = location.read(&path, ng_kvstore::ByteRange::Range { offset, length }).await?.ok_or_else(|| anyhow::anyhow!("no fragment data in {path}"))?;
                // `strictByteRange`.
                anyhow::ensure!(bytes.len() as u64 == length, "Received truncated response for {path}");
                let fragment = ng_datasource::multiscale_mesh::decode_fragment(&bytes, bits, partition)?;
                let size = fragment.bytes();
                Ok((fragment, size))
            })
        });
        let fragments = &self.fragments;
        self.gpu_fragments.retain(|k, (data, _)| fragments.get(k).is_some_and(|d| Arc::ptr_eq(&d, data)));
        // Segmentation scales neuroglancer transcodes, and into what blocks.
        let info = self.nav.render_info().clone();
        let mut blocks: HashMap<(LayerId, usize), [usize; 3]> = HashMap::new();
        for l in &self.layers {
            let Some(render) = l.render.as_ref() else { continue };
            for s in 0..render.layer.scales.len() {
                if let Some(b) = chunks::transcode_block(&render.layer, s, &info) {
                    blocks.insert((l.id, s), b);
                }
            }
        }
        self.chunks.end_frame(|(id, s, grid)| {
            let source = scales.get(id).and_then(|v| v.get(*s)).cloned();
            let block = blocks.get(&(*id, *s)).copied();
            let grid = grid.clone();
            Box::pin(async move {
                let source = source.ok_or_else(|| anyhow::anyhow!("the layer is gone"))?;
                let data = read_chunk(&source, &grid).await?;
                // As neuroglancer counts it (the statistics' memory): a
                // chunk it transcodes at its encoded size.
                let bytes = match block {
                    Some(block) if data.memory_bytes > 0 => {
                        let z = &data.size;
                        let shape = [z[0] as usize, z[1] as usize, z[2] as usize, z.get(3).copied().unwrap_or(1) as usize];
                        ng_datasource::chunk::compressed_segmentation_encoded_bytes(&data.data, shape, block, source.spec.data_type == DataType::Uint64)
                    }
                    _ => data.memory_bytes,
                };
                // The budget, though, is charged what this viewer holds — the
                // decoded data. A compressed segmentation's uint64 labels are
                // ~40× its encoded size: charged that, `systemMemoryLimit`
                // admits several times the memory it names, past what a 32-bit
                // wasm module has.
                let bytes = bytes.max(data.data.len());
                Ok((data, bytes))
            })
        });
        let chunks = &self.chunks;
        for layer in &mut self.layers {
            let id = layer.id;
            if let Some(state) = &mut layer.render {
                state
                    .textures
                    .retain(|(s, grid, _), t| t.last_used + 60 > frame && chunks.get(&(id, *s, grid.clone())).is_some());
            }
        }
    }
}

/// Which view a spatial index is drawn in, for its histogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnnotationView {
    CrossSection,
    Projection,
}

/// `SpatiallyIndexedAnnotationLayer.draw` for layer `li`: the annotations
/// of its precomputed source's spatial index cells within the view of
/// `projection · view` (`size` pixels), each cell's first
/// `round(count · drawFraction)` (the writer shuffles them), requesting the
/// cells. `None` for local annotations.
#[allow(clippy::too_many_arguments)]
fn spatial_annotations(
    layers: &mut [ManagedLayer],
    cells: &mut ChunkManager<AnnotationCellKey, Vec<annotations::Annotation>>,
    li: usize,
    view_kind: AnnotationView,
    frame: u64,
    info: &ng_core::navigation::DisplayDimensionRenderInfo,
    position: &[f32],
    projection: &[f64; 16],
    view: &[f64; 16],
    size: (f64, f64),
) -> Option<Vec<annotations::Annotation>> {
    let managed = layers.get_mut(li)?;
    let id = managed.id;
    let ann = managed.annotation.as_ref()?;
    let spatial = ann.spatial.as_ref()?;
    let key = match view_kind {
        AnnotationView::CrossSection => "crossSectionAnnotationSpacing",
        AnnotationView::Projection => "projectionAnnotationSpacing",
    };
    let target = ng_core::render_scale::parse_target(managed.spec.json.get(key), 8.0);
    let layouts = annotation_level_layouts(&ann.layer, spatial, info, position);
    let canonical_to_physical: f64 = info.voxel_physical_scales.iter().product();
    let visible = chunks::visible_annotation_cells(&layouts, projection, view, size, canonical_to_physical, target);
    let histogram = &mut managed.annotation_histograms[view_kind as usize];
    histogram.begin(frame);
    let mut out = vec![];
    for c in visible {
        let key = (id, c.level, c.cell.clone());
        cells.request(&key, -(c.level as f64));
        managed.progress.0 += 1;
        let present = match cells.get(&key) {
            Some(list) => {
                let count = (list.len() as f64 * c.draw_fraction).round() as usize;
                out.extend(list.iter().take(count).cloned());
                managed.progress.1 += 1;
                1
            }
            None => 0,
        };
        let histogram = &mut managed.annotation_histograms[view_kind as usize];
        histogram.add(c.physical_spacing, c.pixel_spacing, present, 1 - present);
    }
    Some(out)
}

/// A precomputed annotation source's spatial index levels, finest first,
/// as chunk layouts in the display (`getVolumetricTransformedSources` with
/// each level's `chunkToMultiscaleTransform`, a translation to the lower
/// bound).
fn annotation_level_layouts(layer: &VolumeLayer, spatial: &ng_datasource::annotation::AnnotationInfo, info: &ng_core::navigation::DisplayDimensionRenderInfo, position: &[f32]) -> Vec<chunks::AnnotationLevelLayout> {
    let rank = spatial.rank;
    let origin = model_to_display(layer, info, &spatial.lower_bound);
    let columns: Vec<[f64; 3]> = (0..rank)
        .map(|c| {
            let mut e = vec![0.0; rank];
            e[c] = 1.0;
            linear_to_display(layer, info, &e)
        })
        .collect();
    // Each display dimension's model dimension.
    let mut dims = [None; 3];
    for i in 0..3 {
        dims[i] = (0..rank).find(|&c| columns[c][i] != 0.0 && !dims.contains(&Some(c)));
    }
    let model_position = global_to_model(layer, position);
    spatial
        .spatial
        .iter()
        .rev()
        .map(|level| {
            let mut to_display = [0.0; 16];
            to_display[15] = 1.0;
            to_display[12..15].copy_from_slice(&origin);
            let mut chunk_size = [1.0; 3];
            let mut grid = [1.0; 3];
            for (i, d) in dims.iter().enumerate() {
                match *d {
                    Some(c) => {
                        to_display[4 * i..4 * i + 3].copy_from_slice(&columns[c]);
                        chunk_size[i] = level.chunk_size[c];
                        grid[i] = level.grid_shape[c] as f64;
                    }
                    None => to_display[5 * i] = 1.0,
                }
            }
            // The other dimensions: the cell at the position, and the
            // fraction of the level one slice through them holds.
            let mut base = Some(vec![0u64; rank]);
            let mut slice_fraction = 1.0;
            for c in (0..rank).filter(|c| !dims.contains(&Some(*c))) {
                slice_fraction /= level.grid_shape[c] as f64 * level.chunk_size[c];
                let cell = model_position.as_ref().map_or(-1.0, |p| ((p[c] - spatial.lower_bound[c]) / level.chunk_size[c]).floor());
                match base.as_mut() {
                    Some(b) if cell >= 0.0 && cell < level.grid_shape[c] as f64 => b[c] = cell as u64,
                    _ => base = None,
                }
            }
            chunks::AnnotationLevelLayout { to_display, dims, chunk_size, grid, base, limit: level.limit as f64, slice_fraction }
        })
        .collect()
}

/// Model coordinates of `layer` to the display dimensions' (global)
/// coordinates.
pub(crate) fn model_to_display(layer: &VolumeLayer, info: &ng_core::navigation::DisplayDimensionRenderInfo, model: &[f64]) -> [f64; 3] {
    let n = layer.rank + 1;
    let m2l = &layer.model_to_layer;
    let mut out = [0.0; 3];
    for (i, slot) in out.iter_mut().enumerate().take(info.display_rank) {
        let g = info.display_dimension_indices[i];
        if g < 0 {
            continue;
        }
        if let Some(ld) = layer.layer_to_global.iter().position(|&x| x == Some(g as usize)) {
            *slot = (0..layer.source_rank.min(model.len())).map(|c| m2l[c * n + ld] * model[c]).sum::<f64>() + m2l[layer.rank * n + ld];
        }
    }
    out
}

/// A model-space vector (no translation) in the display dimensions.
fn linear_to_display(layer: &VolumeLayer, info: &ng_core::navigation::DisplayDimensionRenderInfo, v: &[f64]) -> [f64; 3] {
    let a = model_to_display(layer, info, v);
    let o = model_to_display(layer, info, &vec![0.0; v.len()]);
    [a[0] - o[0], a[1] - o[1], a[2] - o[2]]
}

/// An axis-aligned ellipsoid's cross-section with the panel's plane
/// (`computeCrossSectionEllipse`, `computeCenterOrientEllipse`): the
/// corners of its bounding quad, in display coordinates, for offsets
/// (−1, −1), (1, −1), (1, 1), (−1, 1) along its axes; `None` if the plane
/// misses it.
fn ellipsoid_cross_section(camera: &chunks::PanelCamera, center: [f64; 3], radii: [f64; 3]) -> Option<[[f64; 3]; 4]> {
    let inv = &camera.inv_view;
    let e1 = [inv[0], inv[1], inv[2]];
    let e2 = [inv[4], inv[5], inv[6]];
    let c0 = [inv[12], inv[13], inv[14]];
    let r: [f64; 3] = std::array::from_fn(|i| radii[i].abs().max(1e-3));
    let d: [f64; 3] = std::array::from_fn(|i| c0[i] - center[i]);
    let sum = |f: &dyn Fn(usize) -> f64| (0..3).map(|i| f(i) / (r[i] * r[i])).sum::<f64>();
    let (a11, a12, a22) = (sum(&|i| e1[i] * e1[i]), sum(&|i| e1[i] * e2[i]), sum(&|i| e2[i] * e2[i]));
    let (b1, b2, c) = (sum(&|i| d[i] * e1[i]), sum(&|i| d[i] * e2[i]), sum(&|i| d[i] * d[i]));
    let det = a11 * a22 - a12 * a12;
    if det <= 0.0 {
        return None;
    }
    let k = [-(a22 * b1 - a12 * b2) / det, -(-a12 * b1 + a11 * b2) / det];
    let rhs = 1.0 - (c + b1 * k[0] + b2 * k[1]);
    if rhs <= 0.0 {
        return None;
    }
    let mean = (a11 + a22) / 2.0;
    let root = (((a11 - a22) / 2.0).powi(2) + a12 * a12).sqrt();
    let (l1, l2) = (mean - root, mean + root);
    let u1 = if a12.abs() > 1e-12 { let v = [l1 - a22, a12]; let n = (v[0] * v[0] + v[1] * v[1]).sqrt(); [v[0] / n, v[1] / n] } else if a11 <= a22 { [1.0, 0.0] } else { [0.0, 1.0] };
    let u2 = [-u1[1], u1[0]];
    let (a, b) = ((rhs / l1).sqrt(), (rhs / l2).sqrt());
    let corner = |cx: f64, cy: f64| -> [f64; 3] {
        let u = k[0] + u1[0] * cx * a + u2[0] * cy * b;
        let v = k[1] + u1[1] * cx * a + u2[1] * cy * b;
        std::array::from_fn(|i| c0[i] + u * e1[i] + v * e2[i])
    };
    Some([corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)])
}

/// `getSphereVertexArray` and `getSphereIndexArray`: the unit sphere's
/// triangles, vertex by vertex.
fn sphere_triangles(lat: usize, lon: usize) -> Vec<[f64; 3]> {
    let vertex = |li: usize, oi: usize| {
        let theta = li as f64 * std::f64::consts::PI / lat as f64;
        let phi = oi as f64 * 2.0 * std::f64::consts::PI / lon as f64;
        // Stored as float32.
        [(phi.cos() * theta.sin()) as f32 as f64, theta.cos() as f32 as f64, (phi.sin() * theta.sin()) as f32 as f64]
    };
    let mut out = vec![];
    for li in 0..lat {
        for oi in 0..lon {
            let (first, second) = ((li, oi), (li + 1, oi));
            for (a, b) in [first, second, (li, oi + 1), second, (li + 1, oi + 1), (li, oi + 1)] {
                out.push(vertex(a, b));
            }
        }
    }
    out
}

/// `getObjectColor`: segment `id`'s colour in `display`, highlighted if
/// `hovered`, premultiplied by `alpha`.
pub(crate) fn object_color(display: &ng_render::segmentation::SegmentationDisplay, hovered: Option<u64>, id: u64, alpha: f32) -> [f32; 4] {
    let base = display.default_color.unwrap_or_else(|| ng_core::SegmentColorHash::new(display.color_seed).compute(id));
    let mut saturation = display.saturation;
    if display.hover_highlight && hovered == Some(id) {
        saturation += if saturation > 0.5 { -0.5 } else { 0.5 };
    }
    let c = |x: f32| (x * saturation + (1.0 - saturation)) * alpha;
    [c(base[0]), c(base[1]), c(base[2]), alpha]
}

/// `update3dRenderLayerAttachment` for a mesh source: its coordinates
/// (`to_model`, column-major 4×4) through the layer's model-to-layer
/// transform to the display dimensions (as float32, gl-matrix's storage).
pub(crate) fn mesh_model_matrix(layer: &layer::VolumeLayer, to_model: &[f64; 16], info: &ng_core::navigation::DisplayDimensionRenderInfo) -> Option<[f32; 16]> {
    let n = layer.rank + 1;
    let m2l = &layer.model_to_layer;
    let model_rank = layer.source_rank.min(3);
    let mut out = [0f64; 16];
    out[15] = 1.0;
    for i in 0..3 {
        let g = info.display_dimension_indices[i];
        if g < 0 {
            continue;
        }
        let Some(ld) = layer.layer_to_global.iter().position(|&x| x == Some(g as usize)) else { continue };
        for c in 0..4 {
            let mut v = if c == 3 { m2l[(n - 1) * n + ld] } else { 0.0 };
            for d in 0..model_rank {
                v += m2l[d * n + ld] * to_model[c * 4 + d];
            }
            out[c * 4 + i] = v;
        }
    }
    out.iter().all(|v| v.is_finite()).then(|| out.map(|v| v as f32))
}

/// `MultiscaleMeshLayer.draw`'s `scaleMultiplier`: the cube root of the
/// model matrix's 3×3 determinant, its rows scaled to physical units
/// (`scaleMat3Output`, float32 as gl-matrix stores it).
fn mesh_scale_multiplier(model: &[f32; 16], scales: [f64; 3]) -> f64 {
    let m = |c: usize, r: usize| ((model[c * 4 + r] as f64) * scales[r]) as f32 as f64;
    let det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) - m(1, 0) * (m(0, 1) * m(2, 2) - m(2, 1) * m(0, 2)) + m(2, 0) * (m(0, 1) * m(1, 2) - m(1, 1) * m(0, 2));
    det.abs().powf(1.0 / 3.0)
}

/// `beginModel`'s normal matrix: the model matrix's 3×3, its rows scaled by
/// the canonical voxel factors (`scaleMat3Output`), inverted, transposed.
fn mesh_normal_matrix(model: &[f32; 16], factors: [f64; 3]) -> [f32; 9] {
    let mut m = [0f64; 9];
    for c in 0..3 {
        for r in 0..3 {
            // `mat3FromMat4`, stored as float32, then scaled (float32 again).
            m[c * 3 + r] = ((model[c * 4 + r] as f64) * factors[r]) as f32 as f64;
        }
    }
    // `mat3.invert`.
    let [a00, a01, a02, a10, a11, a12, a20, a21, a22] = m;
    let b01 = a22 * a11 - a12 * a21;
    let b11 = -a22 * a10 + a12 * a20;
    let b21 = a21 * a10 - a11 * a20;
    let det = a00 * b01 + a01 * b11 + a02 * b21;
    if det == 0.0 {
        return [0.0; 9];
    }
    let d = 1.0 / det;
    let inv = [
        b01 * d,
        (-a22 * a01 + a02 * a21) * d,
        (a12 * a01 - a02 * a11) * d,
        b11 * d,
        (a22 * a00 - a02 * a20) * d,
        (-a12 * a00 + a02 * a10) * d,
        b21 * d,
        (-a21 * a00 + a01 * a20) * d,
        (a11 * a00 - a01 * a10) * d,
    ]
    .map(|v| v as f32);
    // `mat3.transpose`.
    [inv[0], inv[3], inv[6], inv[1], inv[4], inv[7], inv[2], inv[5], inv[8]]
}

/// Re-maps an NDC → texture matrix for rectangle `from` (panel pixels) to the
/// same mapping expressed for rectangle `to`.
fn subrect_matrix(m: [f32; 16], from: [f64; 4], to: [f64; 4]) -> [f32; 16] {
    // ndc_to: NDC in `to` → NDC in `from`: x_from = a * x_to + b.
    let ax = to[2] / from[2];
    let bx = ((to[0] + to[2] / 2.0) - (from[0] + from[2] / 2.0)) / from[2] * 2.0;
    let ay = to[3] / from[3];
    let by = -((to[1] + to[3] / 2.0) - (from[1] + from[3] / 2.0)) / from[3] * 2.0;
    let mut t = [0.0f64; 16];
    t[0] = ax;
    t[5] = ay;
    t[10] = 1.0;
    t[15] = 1.0;
    t[12] = bx;
    t[13] = by;
    let m64 = m.map(|x| x as f64);
    let r = ng_core::matrix::multiply(&m64, &t, 4);
    std::array::from_fn(|i| r[i] as f32)
}

/// A chunk's displayed slab as a 3D texture: the chunk dimensions shown
/// along display x, y, z, at the fixed offsets of the others, cut at the
/// source's upper clip bound (neuroglancer discards beyond it: Zarr stores
/// edge chunks full-size, padded past the array's end).
#[allow(clippy::too_many_arguments)]
fn upload_slab(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layer: &VolumeLayer,
    chunk: &ChunkData,
    draw: &chunks::ChunkDraw,
    format: VolumeFormat,
    upload: UploadFn,
    stack_channels: bool,
) -> GpuVolume {
    let data_type = layer.data_type;
    let source = &layer.scales[draw.scale];
    let rank = chunk.size.len();
    let clip = |c: usize| -> usize {
        let upper = source.upper_clip_bound.as_ref().map_or(source.spec.upper_voxel_bound[c], |b| b[c]) as f64;
        let origin = draw.grid[c] as f64 * source.spec.chunk_data_size[c] as f64;
        ((upper - origin).ceil().max(0.0) as usize).min(chunk.size[c] as usize)
    };
    let mut strides = vec![1usize; rank];
    for c in 1..rank {
        strides[c] = strides[c - 1] * chunk.size[c - 1] as usize;
    }
    // Channel dimensions: every channel, each its own slab stacked along
    // z (first channel dimension fastest), when the layer reads them.
    let channels: &[layer::ChannelDim] = if stack_channels { &layer.channels } else { &[] };
    let mut base = 0usize;
    for &(c, _, off) in &draw.fixed {
        if channels.iter().any(|ch| ch.chunk_dim == c) {
            continue;
        }
        base += off as usize * strides[c];
    }
    let dims: [(usize, usize); 3] = draw.chunk_display_dims.map(|cd| match cd {
        Some(c) => (clip(c).max(1), strides[c]),
        None => (1, 0),
    });
    let elem = data_type.bytes();
    let short = chunk.data.len() < chunk.size.iter().map(|&s| s as usize).product::<usize>() * elem;
    let count: usize = channels.iter().map(|c| c.extent as usize).product();
    let mut slab = Vec::with_capacity(dims[0].0 * dims[1].0 * dims[2].0 * elem * count);
    for flat in 0..count {
        // The channel's offset in the chunk, or none if it is not there —
        // nor if the chunk's data falls short of its size (a missing chunk
        // decoded as empty): zeros, not a read past its end
        let mut offset = if short { None } else { Some(base) };
        let mut rest = flat;
        for ch in channels {
            let coordinate = rest % ch.extent as usize;
            rest /= ch.extent as usize;
            let origin = draw.grid[ch.chunk_dim] as usize * source.spec.chunk_data_size[ch.chunk_dim] as usize;
            offset = offset.and_then(|o| coordinate.checked_sub(origin).filter(|&k| k < chunk.size[ch.chunk_dim] as usize).map(|k| o + k * strides[ch.chunk_dim]));
        }
        for z in 0..dims[2].0 {
            for y in 0..dims[1].0 {
                // A row along the chunk's contiguous axis: copied whole.
                if let (Some(o), 1) = (offset, dims[0].1) {
                    let i = o + y * dims[1].1 + z * dims[2].1;
                    slab.extend_from_slice(&chunk.data[i * elem..(i + dims[0].0) * elem]);
                    continue;
                }
                for x in 0..dims[0].0 {
                    match offset {
                        Some(o) => {
                            let i = o + x * dims[0].1 + y * dims[1].1 + z * dims[2].1;
                            slab.extend_from_slice(&chunk.data[i * elem..(i + 1) * elem]);
                        }
                        None => slab.extend(std::iter::repeat_n(0u8, elem)),
                    }
                }
            }
        }
    }
    let bytes = upload(data_type, &slab);
    let texture = Data3DTexture::new(dims[0].0 as u32, dims[1].0 as u32, (dims[2].0 * count) as u32, format, bytes);
    GpuVolume::from_texture(device, queue, &texture)
}

/// An annotation list's styles: being computed, or ready.
pub(crate) enum StyleState {
    Pending(ng_render::annotation_shader::StyleReadback),
    Ready(Arc<Vec<ng_render::annotation_shader::AnnotationStyle>>),
}

pub(crate) struct StyleSlot {
    pub(crate) state: StyleState,
    /// The last frame they were asked for in.
    frame: u64,
}

/// The styles of layer `managed`'s annotations `list` in a view (3D:
/// `projection`): the default shader's at once; a user shader's once its
/// compute pass is read back (`None` meanwhile, the pass started).
#[allow(clippy::too_many_arguments)]
fn annotation_styles(
    cache: &mut HashMap<(LayerId, bool, u64), StyleSlot>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wake: &Arc<dyn Fn() + Send + Sync>,
    managed: &ManagedLayer,
    list: &[annotations::Annotation],
    projection: bool,
    frame: u64,
) -> Option<Arc<Vec<ng_render::annotation_shader::AnnotationStyle>>> {
    use std::hash::{Hash, Hasher};
    let color = managed.annotation_color();
    let shader = managed.annotation_shader.as_ref();
    let Some((shader, pipelines)) = shader.and_then(|s| Some((s, s.pipelines.as_ref()?))) else {
        return Some(Arc::new(vec![ng_render::annotation_shader::AnnotationStyle::default_shader(color, projection); list.len()]));
    };
    // Each annotation's property bits, in the shader's property order.
    let mut bits = Vec::with_capacity(list.len() * shader.properties.len());
    for a in list {
        for (k, (_, t)) in shader.properties.iter().enumerate() {
            bits.push(t.bits(a.properties.get(k).copied().unwrap_or(0.0)));
        }
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    shader.code.hash(&mut h);
    shader.controls.iter().for_each(|c| c.to_bits().hash(&mut h));
    color.iter().for_each(|c| c.to_bits().hash(&mut h));
    list.iter().for_each(|a| a.id.hash(&mut h));
    bits.hash(&mut h);
    let key = (managed.id, projection, h.finish());
    if let Some(slot) = cache.get_mut(&key) {
        slot.frame = frame;
        if let StyleState::Pending(r) = &slot.state {
            if let Some(styles) = r.read() {
                slot.state = StyleState::Ready(Arc::new(styles));
            }
        }
        return match &slot.state {
            StyleState::Ready(styles) => Some(styles.clone()),
            StyleState::Pending(_) => None,
        };
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    let readback = pipelines[usize::from(projection)].evaluate(device, &mut encoder, &shader.controls, color, &bits, list.len());
    queue.submit([encoder.finish()]);
    readback.map(wake.clone());
    // A frame to poll for it (as for picks).
    wake();
    cache.insert(key, StyleSlot { state: StyleState::Pending(readback), frame });
    None
}

/// A chunk drawn into a slice panel: its texture, the panel's pixels, and
/// the slice parameters.
type ChunkOp = (TextureKey, PixelRect, SliceParams);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subrect_matrix_is_identity_for_the_same_rect() {
        let m: [f32; 16] = std::array::from_fn(|i| i as f32 + 1.0);
        let r = [3.0, 4.0, 50.0, 60.0];
        assert_eq!(subrect_matrix(m, r, r), m);
    }
}
