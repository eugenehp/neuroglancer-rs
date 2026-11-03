//! `CoordinateSpaceTransform` (`src/coordinate_transform.ts`): a data
//! source's model space → the layer's space, and the output-space behaviour of
//! `WatchableCoordinateSpaceTransform` the combiner relies on.

use std::sync::Arc;

use crate::combiner::{convert_transform_output_scales, output_space_with_transformed_bounding_boxes, BoundSpace};
use crate::CoordinateSpace;

#[derive(Debug, Clone)]
pub struct CoordinateSpaceTransform {
    pub rank: usize,
    pub source_rank: usize,
    pub input_space: Arc<CoordinateSpace>,
    pub output_space: Arc<CoordinateSpace>,
    /// `(rank + 1)²`, column-major; rows are output dimensions. The linear
    /// part maps input to output coordinates as if both had unit scales; the
    /// translation is in output coordinates.
    pub transform: Vec<f64>,
}

impl CoordinateSpaceTransform {
    /// `input` → `output` by `transform` (`(rank + 1)²`, column-major).
    pub fn new(input_space: Arc<CoordinateSpace>, output_space: Arc<CoordinateSpace>, transform: Vec<f64>) -> Self {
        let (rank, source_rank) = (output_space.rank(), input_space.rank());
        debug_assert_eq!(transform.len(), (rank + 1) * (rank + 1));
        Self { rank, source_rank, input_space, output_space, transform }
    }

    /// `space` onto itself by `transform`.
    pub fn over(space: Arc<CoordinateSpace>, transform: Vec<f64>) -> Self {
        Self::new(space.clone(), space, transform)
    }

    /// The identity on `space`.
    pub fn identity(space: Arc<CoordinateSpace>) -> Self {
        let n = space.rank() + 1;
        Self::over(space, crate::matrix::identity(n, n))
    }

    /// `normalizeCoordinateSpaceTransform`: the output space's bounding boxes
    /// are the input space's, through the transform.
    pub fn normalized(mut self) -> Self {
        let out = output_space_with_transformed_bounding_boxes(&self.input_space, &self.transform, &self.output_space);
        if out != *self.output_space {
            self.output_space = Arc::new(out);
        }
        self
    }
}

impl BoundSpace for CoordinateSpaceTransform {
    fn space(&self) -> &CoordinateSpace {
        &self.output_space
    }

    /// `outputSpace`'s setter: the same rank only; the transform keeps its
    /// physical meaning in the new units, and the bounding boxes follow.
    fn set_space(&mut self, new_output: CoordinateSpace) {
        if *self.output_space == new_output || self.rank != new_output.rank() {
            return;
        }
        let transform = convert_transform_output_scales(&self.transform, &self.output_space.scales, &new_output.scales);
        let out = output_space_with_transformed_bounding_boxes(&self.input_space, &transform, &new_output);
        self.output_space = Arc::new(out);
        self.transform = transform;
    }
}

/// `CoordinateTransformSpecification`: a layer data source's `transform` in
/// viewer state, applied over the source's own transform.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformSpec {
    pub source_rank: usize,
    /// `(rank + 1)²`, column-major, if `matrix` was given.
    pub transform: Option<Vec<f64>>,
    pub input_space: Option<CoordinateSpace>,
    pub output_space: CoordinateSpace,
}

impl TransformSpec {
    /// `coordinateTransformSpecificationFromJson`; `None` for no spec.
    pub fn from_json(j: Option<&serde_json::Value>) -> anyhow::Result<Option<Self>> {
        use anyhow::{anyhow, bail, Context};
        let Some(j) = j else { return Ok(None) };
        let obj = j.as_object().ok_or_else(|| anyhow!("Expected object, but received: {j}"))?;
        let output_space = CoordinateSpace::from_json(obj.get("outputDimensions"), false)
            .context("Error parsing \"outputDimensions\" property")?;
        let rank = output_space.rank();
        let source_rank = match obj.get("sourceRank") {
            None => rank,
            Some(v) => match v.as_f64() {
                Some(r) if r.fract() == 0.0 && r >= 0.0 && r as usize <= rank => r as usize,
                _ => bail!("Expected integer in range [0, {rank}] but received: {v}"),
            },
        };
        let input_space = match obj.get("inputDimensions") {
            None => None,
            Some(v) => {
                let space = CoordinateSpace::from_json(Some(v), true)?;
                if space.rank() != rank {
                    bail!("Expected rank of {rank}, but received rank of: {}", space.rank());
                }
                Some(space)
            }
        };
        let transform = match obj.get("matrix") {
            None => None,
            Some(x) => {
                let rows = x.as_array().filter(|a| a.len() == rank).ok_or_else(|| anyhow!("Expected array of length {rank}, but received: {x}"))?;
                let mut t = vec![0.0; (rank + 1) * (rank + 1)];
                t[(rank + 1) * (rank + 1) - 1] = 1.0;
                for (i, row) in rows.iter().enumerate() {
                    let row = row
                        .as_array()
                        .filter(|r| r.len() == rank + 1)
                        .ok_or_else(|| anyhow!("Error in row {i}: Expected array of length {}, but received: {row}", rank + 1))?;
                    for (j, v) in row.iter().enumerate() {
                        t[(rank + 1) * j + i] = crate::js::verify_finite_float(v).with_context(|| format!("Error in row {i}"))?;
                    }
                }
                Some(t)
            }
        };
        Ok(Some(Self {
            source_rank,
            transform,
            input_space,
            output_space,
        }))
    }
}

impl CoordinateSpaceTransform {
    /// `WatchableCoordinateSpaceTransform`'s `spec` setter (immutable source
    /// rank): `spec` over this, the source's default transform. An
    /// incompatible spec (invalid output names) gives the default back.
    pub fn with_spec(&self, spec: Option<&TransformSpec>) -> Self {
        use crate::combiner::merge_coordinate_arrays;
        use crate::coordinate_space::{validate_dimension_names, BoundingBox, TransformedBoundingBox};
        let Some(spec) = spec else { return self.clone() };
        let d_in = &self.input_space;
        let d_out = &self.output_space;
        let d_src = self.source_rank;
        let d_rank = self.rank;
        let spec_rank = spec.output_space.rank();
        let spec_src = spec.source_rank;
        let spec_in_names = spec.input_space.as_ref().map_or(&d_in.names, |s| &s.names);
        let mut new_to_spec: Vec<Option<usize>> = (0..d_src)
            .map(|d| spec_in_names.iter().position(|n| *n == d_in.names[d]).filter(|&s| s < spec_src))
            .collect();
        let new_rank = spec_rank - spec_src + d_src;
        new_to_spec.extend((spec_src..spec_rank).map(Some));
        let merge_opt = |list: Vec<Option<&crate::CoordinateArray>>| {
            let l: Vec<_> = list.into_iter().flatten().cloned().collect();
            (!l.is_empty()).then(|| merge_coordinate_arrays(&l))
        };
        let mut in_scales = vec![0.0; new_rank];
        let mut in_units = vec![String::new(); new_rank];
        let mut in_arrays = vec![None; new_rank];
        for n in 0..d_src {
            match (new_to_spec[n], &spec.input_space) {
                (Some(s), Some(si)) => {
                    in_scales[n] = si.scales[s];
                    in_units[n] = si.units[s].clone();
                    in_arrays[n] = merge_opt(vec![d_in.coordinate_arrays[n].as_ref(), si.coordinate_arrays[s].as_ref()]);
                }
                _ => {
                    in_scales[n] = d_in.scales[n];
                    in_units[n] = d_in.units[n].clone();
                    in_arrays[n] = d_in.coordinate_arrays[n].clone();
                }
            }
        }
        let spec_in_or_out = spec.input_space.as_ref().unwrap_or(&spec.output_space);
        let mut in_names: Vec<String> = d_in.names[..d_src].to_vec();
        let mut out_names: Vec<String> = d_out.names[..d_src].to_vec();
        let mut out_arrays: Vec<Option<crate::CoordinateArray>> = d_out.coordinate_arrays[..d_src].to_vec();
        out_names.resize(new_rank, String::new());
        out_arrays.resize(new_rank, None);
        let mut out_scales = vec![0.0; new_rank];
        let mut out_units = vec![String::new(); new_rank];
        for n in 0..new_rank {
            match new_to_spec[n] {
                None => {
                    out_scales[n] = d_out.scales[n];
                    out_units[n] = d_out.units[n].clone();
                    out_arrays[n] = d_out.coordinate_arrays[n].clone();
                }
                Some(s) => {
                    out_names[n] = spec.output_space.names[s].clone();
                    out_units[n] = spec.output_space.units[s].clone();
                    out_scales[n] = spec.output_space.scales[s];
                    out_arrays[n] = spec.output_space.coordinate_arrays[s].clone();
                }
            }
        }
        if !validate_dimension_names(&out_names, false) {
            return self.clone();
        }
        for n in d_src..new_rank {
            let s = n - d_src + spec_src;
            in_scales[n] = spec_in_or_out.scales[s];
            in_units[n] = spec_in_or_out.units[s].clone();
            in_names.push(format!("{n}"));
        }
        let mut t = vec![0.0; (new_rank + 1) * (new_rank + 1)];
        t[(new_rank + 1) * (new_rank + 1) - 1] = 1.0;
        let spec_t = spec.transform.as_ref();
        for row in 0..new_rank {
            let spec_row = new_to_spec[row];
            let value = match (spec_row, spec_t) {
                (Some(sr), Some(st)) => st[spec_rank * (spec_rank + 1) + sr],
                _ if row >= d_src => 0.0,
                _ => self.transform[d_rank * (d_rank + 1) + row] * (d_out.scales[row] / out_scales[row]),
            };
            t[new_rank * (new_rank + 1) + row] = value;
            for col in 0..new_rank {
                let spec_col = new_to_spec[col];
                let value = if spec_row.is_none() != spec_col.is_none() {
                    0.0
                } else {
                    match (spec_row, spec_col, spec_t) {
                        (Some(sr), Some(sc), Some(st)) => st[sc * (spec_rank + 1) + sr],
                        // `specRow >= defaultSourceRank || ...` compares spec
                        // indices (or -1) with the default source rank.
                        (sr, sc, _) => {
                            let idx = |x: Option<usize>| x.map_or(-1, |v| v as i64);
                            if idx(sr) >= d_src as i64 || idx(sc) >= d_src as i64 {
                                if idx(sr) == idx(sc) { 1.0 } else { 0.0 }
                            } else {
                                self.transform[col * (d_rank + 1) + row]
                            }
                        }
                    }
                };
                t[col * (new_rank + 1) + row] = value;
            }
        }
        let mut boxes: Vec<TransformedBoundingBox> = d_in
            .bounding_boxes
            .iter()
            .map(|b| {
                // `extendTransformedBoundingBoxUpToRank`.
                if d_rank == new_rank {
                    return b.clone();
                }
                let input_rank = b.bounding_box.lower_bounds.len();
                let mut transform = vec![0.0; (input_rank + 1) * new_rank];
                for c in 0..=input_rank {
                    for r in 0..d_rank.min(new_rank) {
                        transform[c * new_rank + r] = b.transform[c * d_rank + r];
                    }
                }
                TransformedBoundingBox { bounding_box: b.bounding_box.clone(), transform }
            })
            .collect();
        for i in d_src..new_rank {
            let mut transform = vec![0.0; 2 * new_rank];
            transform[i] = 1.0;
            boxes.push(TransformedBoundingBox {
                bounding_box: BoundingBox { lower_bounds: vec![0.0], upper_bounds: vec![1.0] },
                transform,
            });
        }
        // Coordinate arrays follow identity-mapped dimensions to the output.
        for out in 0..new_rank {
            if t[new_rank * (new_rank + 1) + out] != 0.0 {
                continue;
            }
            let mut single: Option<Option<usize>> = None;
            for input in 0..new_rank {
                let f = t[input * (new_rank + 1) + out];
                if f == 0.0 {
                    continue;
                }
                if f == 1.0 && single.is_none() {
                    single = Some(Some(input));
                } else {
                    single = Some(None);
                    break;
                }
            }
            let Some(Some(input)) = single else { continue };
            let Some(mut array) = in_arrays[input].clone() else { continue };
            array.explicit = false;
            out_arrays[out] = merge_opt(vec![Some(&array), out_arrays[out].as_ref()]);
        }
        let mut output = CoordinateSpace::new(true, out_names, out_units, out_scales);
        output.coordinate_arrays = out_arrays;
        let mut input = CoordinateSpace::new(true, in_names, in_units, in_scales).with_bounding_boxes(boxes);
        input.coordinate_arrays = in_arrays;
        Self {
            rank: new_rank,
            source_rank: d_src,
            input_space: Arc::new(input),
            output_space: Arc::new(output),
            transform: t,
        }
        .normalized()
    }
}

/// `coordinateTransformSpecificationFromLegacyJson`: a layer's old top-level
/// `transform` (a 4×4 matrix, flat column-major or as rows, or an object of
/// `rotation`, `translation`, `scale`), over `x y z` in nanometres.
pub fn legacy_transform_spec(obj: Option<&serde_json::Value>) -> anyhow::Result<Option<TransformSpec>> {
    use anyhow::{anyhow, bail};
    let Some(obj) = obj else { return Ok(None) };
    let mut t = [0.0f64; 16];
    if let Some(a) = obj.as_array() {
        if a.len() == 16 {
            for i in 0..4 {
                for j in 0..4 {
                    t[i * 4 + j] = crate::js::verify_finite_float(&a[j * 4 + i])?;
                }
            }
        } else {
            if a.len() != 4 {
                bail!("Expected array of length 4, but received: {obj}");
            }
            for i in 0..4 {
                let row = a[i].as_array().filter(|r| r.len() == 4).ok_or_else(|| anyhow!("Expected array of length 4, but received: {}", a[i]))?;
                for j in 0..4 {
                    t[j * 4 + i] = crate::js::verify_finite_float(&row[j])?;
                }
            }
        }
    } else {
        let o = obj.as_object().ok_or_else(|| anyhow!("Expected object, but received: {obj}"))?;
        // `parseFiniteVec` into `Float32Array`s.
        let vec = |key: &str, n: usize, default: &[f32]| -> anyhow::Result<Vec<f32>> {
            let Some(v) = o.get(key) else { return Ok(default.to_vec()) };
            let a = v.as_array().filter(|a| a.len() == n).ok_or_else(|| anyhow!("Incompatible sizes"))?;
            let parsed: Vec<f64> = a.iter().map(|x| crate::js::parse_float(&crate::js::to_string(x))).collect();
            if parsed.iter().any(|x| !x.is_finite()) {
                bail!("Non-finite value.");
            }
            Ok(parsed.iter().map(|&x| x as f32).collect())
        };
        let r = vec("rotation", 4, &[0.0, 0.0, 0.0, 1.0])?;
        let r = if o.contains_key("rotation") { crate::glm::quat_normalize([r[0], r[1], r[2], r[3]]).to_vec() } else { r };
        let v = vec("translation", 3, &[0.0, 0.0, 0.0])?;
        let s = vec("scale", 3, &[1.0, 1.0, 1.0])?;
        // `mat4.fromRotationTranslationScale`: doubles from the f32 inputs,
        // each result stored back to f32.
        let (r, v, s): (Vec<f64>, Vec<f64>, Vec<f64>) =
            (r.iter().map(|&a| a as f64).collect(), v.iter().map(|&a| a as f64).collect(), s.iter().map(|&a| a as f64).collect());
        let (x, y, z, w) = (r[0], r[1], r[2], r[3]);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, xy, xz) = (x * x2, x * y2, x * z2);
        let (yy, yz, zz) = (y * y2, y * z2, z * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        let m: [f64; 16] = [
            (1.0 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0.0,
            (xy - wz) * s[1], (1.0 - (xx + zz)) * s[1], (yz + wx) * s[1], 0.0,
            (xz + wy) * s[2], (yz - wx) * s[2], (1.0 - (xx + yy)) * s[2], 0.0,
            v[0], v[1], v[2], 1.0,
        ];
        for (d, &v) in t.iter_mut().zip(&m) {
            *d = v as f32 as f64;
        }
    }
    let names = vec!["x".to_owned(), "y".to_owned(), "z".to_owned()];
    Ok(Some(TransformSpec {
        source_rank: 3,
        transform: Some(t.to_vec()),
        input_space: None,
        output_space: CoordinateSpace::new(true, names, vec!["m".into(); 3], vec![1e-9; 3]),
    }))
}

impl TransformSpec {
    /// `coordinateTransformSpecificationToJson`.
    pub fn to_json(&self) -> serde_json::Value {
        let rank = self.output_space.rank();
        let mut o = serde_json::Map::new();
        if self.source_rank != rank {
            o.insert("sourceRank".into(), self.source_rank.into());
        }
        if let Some(t) = &self.transform {
            let rows: Vec<serde_json::Value> = (0..rank)
                .map(|i| (0..=rank).map(|j| crate::js::number(t[(rank + 1) * j + i])).collect())
                .collect();
            o.insert("matrix".into(), rows.into());
        }
        if let Some(out) = self.output_space.to_json() {
            o.insert("outputDimensions".into(), out);
        }
        if let Some(input) = self.input_space.as_ref().and_then(CoordinateSpace::to_json) {
            o.insert("inputDimensions".into(), input);
        }
        serde_json::Value::Object(o)
    }
}

/// `isTransformDerivableFromDefault`.
fn derivable_from_default(
    default: &[f64],
    default_rank: usize,
    default_scales: &[f64],
    new: &[f64],
    new_rank: usize,
    new_scales: &[f64],
) -> bool {
    let (dn, nn) = (default_rank + 1, new_rank + 1);
    // `matrix.equal` over the default's linear part.
    for j in 0..default_rank {
        for i in 0..default_rank {
            if default[dn * j + i] != new[nn * j + i] {
                return false;
            }
        }
    }
    for i in 0..default_rank {
        if default[dn * default_rank + i] * (default_scales[i] / new_scales[i]) != new[nn * new_rank + i] {
            return false;
        }
    }
    for i in default_rank..new_rank {
        if new[nn * new_rank + i] != 0.0 {
            return false;
        }
    }
    for i in default_rank..new_rank {
        if (0..default_rank).any(|j| new[nn * j + i] != 0.0) {
            return false;
        }
        for j in 0..new_rank {
            let c = new[nn * i + j];
            if (i == j && c != 1.0) || (i != j && c != 0.0) {
                return false;
            }
        }
    }
    true
}

impl CoordinateSpaceTransform {
    /// `WatchableCoordinateSpaceTransform.spec`: this transform as a spec
    /// over `default` (the source's own), or `None` if it is the default.
    pub fn spec_over(&self, default: &CoordinateSpaceTransform) -> Option<TransformSpec> {
        let input = &self.input_space;
        let input_same = self.source_rank == self.rank
            && input.scales == default.input_space.scales
            && input.units == default.input_space.units;
        let transform_same = derivable_from_default(
            &default.transform,
            default.rank,
            &default.output_space.scales,
            &self.transform,
            self.rank,
            &self.output_space.scales,
        );
        let names_same = default.output_space.names == self.output_space.names;
        if transform_same && names_same && input_same {
            return None;
        }
        Some(TransformSpec {
            source_rank: self.source_rank,
            transform: (!transform_same).then(|| self.transform.clone()),
            output_space: (*self.output_space).clone(),
            input_space: (!input_same).then(|| (**input).clone()),
        })
    }
}
