//! Merging layer spaces into the global one: port of `CoordinateSpaceCombiner`
//! (`src/coordinate_transform.ts`), with what it needs of
//! `WatchableCoordinateSpaceTransform`'s output space.
//!
//! Each bound space (a layer's transform output) contributes its dimensions
//! by name; dimensions the combined space already has keep its units and
//! scales, which flow back into the bound spaces (a layer's transform is
//! rescaled to keep its physical meaning, `convertTransformOutputScales`). A
//! dimension edited more recently (a later timestamp) wins instead. Dimensions
//! no bound space uses are dropped, unless the combiner is retained (while
//! layers load).
//!
//! neuroglancer reacts to changes through signals: a bound space changing
//! re-runs the merge, and so does the merge rewriting a bound space, from
//! within the merge. Here [`CoordinateSpaceCombiner::update`] runs the merge
//! until the bound spaces stop changing, which reaches the same state.
//! Dimension IDs are opaque either way; their values differ from
//! neuroglancer's.

use std::cmp::Ordering;
use std::sync::atomic::{self, AtomicI64};

use crate::coordinate_space::{
    compute_combined_bounds, BoundingBox, CoordinateArray, CoordinateSpace, DimensionId, TransformedBoundingBox,
};

static NEXT_DIMENSION_ID: AtomicI64 = AtomicI64::new(0);

/// `newDimensionId`: positive, never reused.
pub fn new_dimension_id() -> DimensionId {
    NEXT_DIMENSION_ID.fetch_add(1, atomic::Ordering::Relaxed) + 1
}

/// Something whose space the combiner reads and rewrites.
pub trait BoundSpace {
    fn space(&self) -> &CoordinateSpace;
    /// Takes merged names, units, scales and timestamps. The holder may
    /// adjust further (a transform recomputes its bounding boxes).
    fn set_space(&mut self, space: CoordinateSpace);
}

/// The combiner's record of one bound space.
#[derive(Debug, Clone, Default)]
pub struct Binding {
    prev_value: Option<CoordinateSpace>,
    mapped_dimension_ids: Vec<Option<DimensionId>>,
}

pub struct CoordinateSpaceCombiner {
    include: fn(&str) -> bool,
    retain_count: usize,
    /// How many bound-space dimensions carry each name.
    pub dimension_ref_counts: Vec<(String, usize)>,
}

impl CoordinateSpaceCombiner {
    /// `include` picks the dimensions this combiner merges (global ones, or a
    /// layer's local and channel ones).
    pub fn new(include: fn(&str) -> bool) -> Self {
        Self {
            include,
            retain_count: 0,
            dimension_ref_counts: vec![],
        }
    }

    /// `retain`: keep dimensions nothing refers to, until [`release`](Self::release).
    pub fn retain(&mut self) {
        self.retain_count += 1;
    }

    /// The retain's disposer; call [`update`](Self::update) after.
    pub fn release(&mut self) {
        self.retain_count = self.retain_count.saturating_sub(1);
    }

    /// Merges `bound` into `combined` and the merged settings back into
    /// `bound`, until stable. Returns whether `combined` changed.
    pub fn update(&mut self, combined: &mut CoordinateSpace, bound: &mut [(&mut Binding, &mut dyn BoundSpace)]) -> bool {
        let mut changed = false;
        for _ in 0..16 {
            let (c, again) = self.update_once(combined, bound);
            changed |= c;
            if !again {
                break;
            }
        }
        changed
    }

    /// One `update()`; also returns whether a bound space ended up other than
    /// what it was given (the merge must run again).
    fn update_once(&mut self, combined: &mut CoordinateSpace, bound: &mut [(&mut Binding, &mut dyn BoundSpace)]) -> (bool, bool) {
        let retain_existing = self.retain_count > 0;
        if bound.is_empty() && !retain_existing {
            let empty = CoordinateSpace::empty_invalid();
            let changed = *combined != empty;
            *combined = empty;
            return (changed, false);
        }
        let include = self.include;
        let existing = combined.clone();
        let mut merged_names = existing.names.clone();
        let mut merged_units = existing.units.clone();
        let mut merged_scales = existing.scales.clone();
        let mut merged_ids = existing.ids.clone();
        let mut merged_timestamps = existing.timestamps.clone();
        let mut dimension_refs: Vec<usize> = existing.names.iter().map(|_| retain_existing as usize).collect();
        let mut binding_combined_indices: Vec<Vec<Option<usize>>> = vec![];
        let mut valid = false;
        for (binding, holder) in bound.iter_mut() {
            let space = holder.space().clone();
            valid = valid || space.valid;
            let prev_value = binding.prev_value.take();
            let mapped = std::mem::take(&mut binding.mapped_dimension_ids);
            let rank = space.rank();
            let mut new_mapped = vec![None; rank];
            let mut combined_indices = vec![None; rank];
            for i in 0..rank {
                let name = &space.names[i];
                if !include(name) {
                    continue;
                }
                if let Some(prev) = &prev_value {
                    if let Some(prev_index) = prev.ids.iter().position(|&id| id == space.ids[i]) {
                        if let Some(combined_id) = mapped.get(prev_index).copied().flatten() {
                            if let Some(ci) = merged_ids.iter().position(|&id| id == combined_id) {
                                new_mapped[i] = Some(combined_id);
                                dimension_refs[ci] += 1;
                                combined_indices[i] = Some(ci);
                                let timestamp = space.timestamps[i];
                                // `!(timestamp <= merged)`: NaN counts as newer.
                                if !matches!(timestamp.partial_cmp(&merged_timestamps[ci]), Some(Ordering::Less | Ordering::Equal)) {
                                    merged_names[ci] = name.clone();
                                    merged_scales[ci] = space.scales[i];
                                    merged_units[ci] = space.units[i].clone();
                                    merged_timestamps[ci] = timestamp;
                                }
                                continue;
                            }
                        }
                    }
                }
                if let Some(ci) = merged_names.iter().position(|n| n == name) {
                    new_mapped[i] = Some(merged_ids[ci]);
                    dimension_refs[ci] += 1;
                    combined_indices[i] = Some(ci);
                    continue;
                }
                let ci = merged_names.len();
                combined_indices[i] = Some(ci);
                dimension_refs.push(1 + retain_existing as usize);
                merged_names.push(name.clone());
                merged_units.push(space.units[i].clone());
                merged_scales.push(space.scales[i]);
                merged_timestamps.push(space.timestamps[i]);
                let id = new_dimension_id();
                merged_ids.push(id);
                new_mapped[i] = Some(id);
            }
            binding.mapped_dimension_ids = new_mapped;
            binding.prev_value = Some(space);
            binding_combined_indices.push(combined_indices);
        }

        // Propagate names, units, scales and timestamps back.
        let mut again = false;
        self.dimension_ref_counts.clear();
        for ((binding, holder), combined_indices) in bound.iter_mut().zip(&binding_combined_indices) {
            let space = holder.space().clone();
            let mut new_space = space.clone();
            for (i, ci) in combined_indices.iter().enumerate() {
                let Some(ci) = *ci else { continue };
                new_space.units[i] = merged_units[ci].clone();
                new_space.scales[i] = merged_scales[ci];
                new_space.timestamps[i] = merged_timestamps[ci];
                new_space.names[i] = merged_names[ci].clone();
            }
            for name in &new_space.names {
                match self.dimension_ref_counts.iter_mut().find(|(n, _)| n == name) {
                    Some((_, c)) => *c += 1,
                    None => self.dimension_ref_counts.push((name.clone(), 1)),
                }
            }
            if new_space.units != space.units
                || new_space.scales != space.scales
                || new_space.names != space.names
                || new_space.timestamps != space.timestamps
            {
                binding.prev_value = Some(new_space.clone());
                holder.set_space(new_space.clone());
                if *holder.space() != new_space {
                    // The holder changed it further: neuroglancer's signal
                    // would merge again.
                    again = true;
                }
            }
        }

        for (i, name) in merged_names.iter().enumerate() {
            if !include(name) {
                dimension_refs[i] = 0;
            }
        }
        let keep: Vec<bool> = dimension_refs.iter().map(|&r| r != 0).collect();
        fn filter<T>(v: &mut Vec<T>, keep: &[bool]) {
            let mut k = keep.iter();
            v.retain(|_| *k.next().unwrap());
        }
        filter(&mut merged_names, &keep);
        filter(&mut merged_units, &keep);
        filter(&mut merged_scales, &keep);
        filter(&mut merged_ids, &keep);
        filter(&mut merged_timestamps, &keep);
        let new_rank = merged_names.len();

        let mut merged_boxes = vec![];
        let mut all_arrays: Vec<Vec<CoordinateArray>> = vec![vec![]; new_rank];
        // Explicit coordinate arrays of the existing space stay.
        for (i, array) in existing.coordinate_arrays.iter().enumerate() {
            let Some(array) = array.as_ref().filter(|a| a.explicit) else { continue };
            if let Some(d) = merged_ids.iter().position(|&id| id == existing.ids[i]) {
                all_arrays[d] = vec![array.clone()];
            }
        }
        for (_, holder) in bound.iter() {
            let space = holder.space();
            let new_dims: Vec<Option<usize>> = space.names.iter().map(|n| merged_names.iter().position(|m| m == n)).collect();
            for b in &space.bounding_boxes {
                merged_boxes.push(extend_transformed_bounding_box(b, new_rank, &new_dims));
            }
            for (i, array) in space.coordinate_arrays.iter().enumerate() {
                if let (Some(array), Some(d)) = (array, new_dims[i]) {
                    all_arrays[d].push(array.clone());
                }
            }
        }
        let coordinate_arrays = all_arrays
            .iter()
            .map(|list| (!list.is_empty()).then(|| merge_coordinate_arrays(list)))
            .collect();
        // neuroglancer builds the merged space without its timestamps
        // (`makeCoordinateSpace` defaults them to -inf); the bound spaces keep
        // theirs, so a later merge still sees edits through them.
        let _ = merged_timestamps;
        let new_combined = CoordinateSpace {
            valid,
            timestamps: vec![f64::NEG_INFINITY; new_rank],
            names: merged_names,
            ids: merged_ids,
            units: merged_units,
            scales: merged_scales,
            coordinate_arrays,
            bounds: compute_combined_bounds(&merged_boxes, new_rank),
            bounding_boxes: merged_boxes,
        };
        let changed = existing != new_combined;
        if changed {
            *combined = new_combined;
        }
        (changed, again)
    }
}

/// `extendTransformedBoundingBox`: re-targets a box's output dimensions;
/// `None` drops one.
pub fn extend_transformed_bounding_box(
    b: &TransformedBoundingBox,
    new_output_rank: usize,
    new_output_dims: &[Option<usize>],
) -> TransformedBoundingBox {
    let old_rank = new_output_dims.len();
    let input_rank = b.bounding_box.lower_bounds.len();
    let mut transform = vec![0.0; (input_rank + 1) * new_output_rank];
    for (old, new) in new_output_dims.iter().enumerate() {
        let Some(new) = *new else { continue };
        for input in 0..=input_rank {
            transform[input * new_output_rank + new] = b.transform[input * old_rank + old];
        }
    }
    TransformedBoundingBox {
        bounding_box: b.bounding_box.clone(),
        transform,
    }
}

/// `mergeCoordinateArrays`.
pub fn merge_coordinate_arrays(arrays: &[CoordinateArray]) -> CoordinateArray {
    if arrays.len() == 1 {
        return arrays[0].clone();
    }
    // A JS `Map`: last label per coordinate wins.
    let mut entries: Vec<(i64, String)> = vec![];
    let mut explicit = false;
    for a in arrays {
        explicit |= a.explicit;
        for (&c, l) in a.coordinates.iter().zip(&a.labels) {
            match entries.iter_mut().find(|(k, _)| *k == c) {
                Some(e) => e.1 = l.clone(),
                None => entries.push((c, l.clone())),
            }
        }
    }
    entries.sort_by_key(|&(c, _)| c);
    let (coordinates, labels) = entries.into_iter().unzip();
    CoordinateArray {
        explicit,
        coordinates,
        labels,
    }
}

/// `convertTransformOutputScales`: the translation re-expressed in new output
/// units (the linear part is unit-free).
pub fn convert_transform_output_scales(transform: &[f64], old_scales: &[f64], new_scales: &[f64]) -> Vec<f64> {
    let rank = old_scales.len();
    let mut t = transform.to_vec();
    let base = (rank + 1) * rank;
    for i in 0..rank {
        t[base + i] *= old_scales[i] / new_scales[i];
    }
    t
}

/// `transformBoundingBox`: a box of the input space, through `transform`
/// (`(rank + 1)²`), into an output space with `target_scales`.
pub fn transform_bounding_box(
    b: &TransformedBoundingBox,
    transform: &[f64],
    source_scales: &[f64],
    target_scales: &[f64],
) -> TransformedBoundingBox {
    let input_rank = b.bounding_box.lower_bounds.len();
    let target_rank = target_scales.len();
    let old = &b.transform;
    let mut new = vec![0.0; (input_rank + 1) * target_rank];
    for target in 0..target_rank {
        let target_scale = target_scales[target];
        for input in 0..input_rank {
            let mut sum = 0.0;
            for source in 0..target_rank {
                sum += transform[(target_rank + 1) * source + target]
                    * old[target_rank * input + source]
                    * (source_scales[source] / target_scale);
            }
            new[target_rank * input + target] = sum;
        }
        let mut sum = transform[(target_rank + 1) * target_rank + target];
        for source in 0..target_rank {
            sum += transform[(target_rank + 1) * source + target]
                * old[target_rank * input_rank + source]
                * (source_scales[source] / target_scale);
        }
        new[target_rank * input_rank + target] = sum;
    }
    TransformedBoundingBox {
        bounding_box: BoundingBox {
            lower_bounds: b.bounding_box.lower_bounds.clone(),
            upper_bounds: b.bounding_box.upper_bounds.clone(),
        },
        transform: new,
    }
}

/// `getOutputSpaceWithTransformedBoundingBoxes`.
pub fn output_space_with_transformed_bounding_boxes(
    input_space: &CoordinateSpace,
    transform: &[f64],
    old_output: &CoordinateSpace,
) -> CoordinateSpace {
    let boxes = input_space
        .bounding_boxes
        .iter()
        .map(|b| transform_bounding_box(b, transform, &input_space.scales, &old_output.scales))
        .collect::<Vec<_>>();
    let mut s = old_output.clone();
    s.valid = input_space.valid;
    s.bounds = compute_combined_bounds(&boxes, s.rank());
    s.bounding_boxes = boxes;
    s
}
