//! Navigation: where the viewer is, which way it faces, how far it is zoomed.
//! Port of `src/navigation_state.ts` (`Position`, `OrientationState`,
//! `TrackableRelativeDisplayScales`, `TrackableDisplayDimensions`,
//! `DisplayDimensionRenderInfo`, `DisplayPose`, `TrackableZoom` and
//! `TrackableDepthRange`).
//!
//! neuroglancer's objects observe the coordinate space and adapt to it lazily,
//! by object identity. Here the space is an `Arc` and identity is
//! `Arc::ptr_eq`: [`Navigation`] owns the parts and brings them up to date
//! before each use, and [`Navigation::generation`] changes whenever anything
//! does, in place of change signals.
//!
//! Numerics follow the TS: positions and orientations are `Float32Array`s
//! (`f32`), gl-matrix does the vector math ([`crate::glm`]), everything else is
//! `f64`.
//!
//! Not ported yet: linked navigation (layer groups) and playback velocity.

use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use serde_json::{Map, Value};

use crate::coordinate_space::{get_center_bound, validate_dimension_names, CoordinateSpace};
use crate::glm::{self, Mat4, Quat, Vec3};
use crate::js;

pub type SpaceRef = Arc<CoordinateSpace>;

fn same(a: &SpaceRef, b: &SpaceRef) -> bool {
    Arc::ptr_eq(a, b)
}

/// `Array.prototype.indexOf` on dimension ids.
fn index_of(ids: &[i64], id: i64) -> Option<usize> {
    ids.iter().position(|&x| x == id)
}

// ── Position ───────────────────────────────────────────────────────────────

/// The global position, in voxels of the coordinate space. Port of `Position`.
#[derive(Debug, Clone, Default)]
pub struct Position {
    coordinates: Vec<f32>,
    cur: Option<SpaceRef>,
}

impl Position {
    /// `handleCoordinateSpaceChanged`. Returns whether the position changed.
    fn sync(&mut self, space: &SpaceRef) -> bool {
        let prev = match &self.cur {
            Some(prev) if same(prev, space) => return false,
            prev => prev.clone(),
        };
        self.cur = Some(space.clone());
        let rank = space.rank();
        if !space.valid {
            return false;
        }
        match prev {
            Some(prev) if prev.valid => {
                // Match dimensions by ID.
                let mut new = vec![0f32; rank];
                for (dim, slot) in new.iter_mut().enumerate() {
                    *slot = match index_of(&prev.ids, space.ids[dim]) {
                        None => get_center_bound(
                            space.bounds.lower_bounds[dim],
                            space.bounds.upper_bounds[dim],
                        ) as f32,
                        Some(old) => {
                            (self.coordinates[old] as f64 * (prev.scales[old] / space.scales[dim])) as f32
                        }
                    };
                }
                self.coordinates = new;
            }
            _ => {
                // Keep the existing coordinates if the rank matches; otherwise
                // start at the center of the bounds, on a voxel center.
                if self.coordinates.len() != rank {
                    let bounds = &space.bounds;
                    self.coordinates = (0..rank)
                        .map(|i| {
                            let c = get_center_bound(bounds.lower_bounds[i], bounds.upper_bounds[i]) as f32;
                            bounds.round_to_voxel_center(i, c as f64) as f32
                        })
                        .collect();
                }
            }
        }
        true
    }

    /// The position in voxels.
    pub fn value(&mut self, space: &SpaceRef) -> &[f32] {
        self.sync(space);
        &self.coordinates
    }

    /// `set value`: ignored unless the space is valid and the rank matches.
    pub fn set_value(&mut self, coordinates: &[f32]) -> bool {
        match &self.cur {
            Some(cur) if cur.valid && cur.rank() == coordinates.len() => {
                self.coordinates.copy_from_slice(coordinates);
                true
            }
            _ => false,
        }
    }

    pub fn reset(&mut self) {
        self.cur = None;
        self.coordinates.clear();
    }

    pub fn to_json(&mut self, space: &SpaceRef) -> Option<Value> {
        if !space.valid && self.coordinates.is_empty() {
            return None;
        }
        self.sync(space);
        if self.coordinates.is_empty() {
            return None;
        }
        Some(Value::Array(self.coordinates.iter().map(|&c| js::number(c as f64)).collect()))
    }

    /// `restoreState` for a value present in the JSON.
    pub fn restore_state(&mut self, json: &Value, space: &SpaceRef) -> Result<()> {
        let Some(items) = json.as_array() else {
            bail!("Expected array, but received: {json}.");
        };
        let coordinates = items
            .iter()
            .map(|x| js::verify_finite_float(x).map(|v| v as f32))
            .collect::<Result<Vec<_>>>()?;
        self.cur = None;
        self.coordinates = coordinates;
        self.sync(space);
        Ok(())
    }

    /// `snapToVoxel`.
    pub fn snap_to_voxel(&mut self, space: &SpaceRef) {
        self.sync(space);
        for (i, c) in self.coordinates.iter_mut().enumerate() {
            *c = space.bounds.round_to_voxel_center(i, *c as f64) as f32;
        }
    }
}

// ── OrientationState ───────────────────────────────────────────────────────

/// Port of `OrientationState`: a unit quaternion, from view axes to the
/// display dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orientation(pub Quat);

impl Default for Orientation {
    fn default() -> Self {
        Self(glm::QUAT_IDENTITY)
    }
}

impl Orientation {
    /// Normalizes in place, as `toJSON` does, and omits the identity.
    pub fn to_json(&mut self) -> Option<Value> {
        self.0 = glm::quat_normalize(self.0);
        if self.0 == glm::QUAT_IDENTITY {
            return None;
        }
        Some(Value::Array(self.0.iter().map(|&c| js::number(c as f64)).collect()))
    }

    /// Anything other than four finite numbers resets to the identity.
    pub fn restore_state(&mut self, json: &Value) {
        let parsed = json.as_array().filter(|a| a.len() == 4).and_then(|a| {
            let v: Vec<f64> = a.iter().map(|x| js::parse_float(&js::to_string(x))).collect();
            v.iter().all(|x| x.is_finite()).then_some(v)
        });
        self.0 = match parsed {
            Some(v) => glm::quat_normalize([v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]),
            None => glm::QUAT_IDENTITY,
        };
    }

    pub fn reset(&mut self) {
        self.0 = glm::QUAT_IDENTITY;
    }

    /// `snap`: to the nearest axis-aligned orientation.
    pub fn snap(&mut self) {
        let mut mat = glm::mat3_from_quat(self.0);
        let mut used = [false; 3];
        for i in 0..3 {
            let (mut max, mut argmax) = (0f32, 0usize);
            for j in 0..3 {
                let value = mat[i * 3 + j];
                mat[i * 3 + j] = 0.0;
                if used[j] {
                    continue;
                }
                if value.abs() > max.abs() {
                    max = value;
                    argmax = j;
                }
            }
            // `Math.sign`: 0 stays 0.
            mat[i * 3 + argmax] = if max > 0.0 { 1.0 } else if max < 0.0 { -1.0 } else { max };
            used[argmax] = true;
        }
        self.0 = glm::quat_from_mat3(mat);
    }
}

/// Port of `OrientationState.makeRelative`: an orientation that is always
/// `peer * peer_to_self`, for the xz and yz panels.
///
/// It keeps its own value the way the TS object does: when it is changed, the
/// peer is set to `self * inverse(peer_to_self)` and its own value is kept as
/// written; it is recomputed only when the peer *signals* a change. Following
/// the signal rather than the value matters: `toJSON` normalizes the peer in
/// place without signalling, and the TS panel keeps its value from before.
#[derive(Debug, Clone)]
pub struct RelativeOrientation {
    peer_to_self: Quat,
    self_to_peer: Quat,
    value: Quat,
    /// The peer's orientation version `value` was derived from.
    peer_version: Option<u64>,
}

impl RelativeOrientation {
    pub fn new(peer_to_self: Quat) -> Self {
        Self {
            peer_to_self,
            self_to_peer: glm::quat_invert(peer_to_self),
            value: glm::QUAT_IDENTITY,
            peer_version: None,
        }
    }

    /// The orientation, following the peer if it changed since `version`.
    fn get(&mut self, peer: Quat, version: u64) -> Quat {
        if self.peer_version != Some(version) {
            self.value = glm::quat_multiply(peer, self.peer_to_self);
            self.peer_version = Some(version);
        }
        self.value
    }

    /// Sets this orientation; returns the peer's new value. The caller bumps
    /// the peer's version to `peer_version`, which this then already knows.
    fn set(&mut self, value: Quat, peer_version: u64) -> Quat {
        self.value = value;
        self.peer_version = Some(peer_version);
        glm::quat_multiply(value, self.self_to_peer)
    }
}

// ── Relative display scales ────────────────────────────────────────────────

/// Port of `TrackableRelativeDisplayScales`: per-dimension factors on top of
/// the coordinate space's scales, for non-uniform zoom.
#[derive(Debug, Clone, Default)]
pub struct RelativeDisplayScales {
    factors: Vec<f64>,
    cur: Option<SpaceRef>,
}

impl RelativeDisplayScales {
    /// `update`: carry factors across a new space by dimension ID, defaulting
    /// to 1.
    fn sync(&mut self, space: &SpaceRef) -> bool {
        if let Some(cur) = &self.cur {
            if same(cur, space) {
                return false;
            }
        }
        let old_ids: &[i64] = self.cur.as_ref().map_or(&[], |c| &c.ids);
        let new: Vec<f64> = space
            .ids
            .iter()
            .map(|&id| index_of(old_ids, id).map_or(1.0, |old| self.factors[old]))
            .collect();
        if new == self.factors {
            // As in the TS, the space is not recorded here; the next access
            // compares again and finds the same factors.
            return false;
        }
        self.factors = new;
        self.cur = Some(space.clone());
        true
    }

    pub fn factors(&mut self, space: &SpaceRef) -> &[f64] {
        self.sync(space);
        &self.factors
    }

    pub fn reset(&mut self) {
        self.factors.clear();
        self.cur = None;
    }

    pub fn to_json(&mut self, space: &SpaceRef) -> Option<Value> {
        self.sync(space);
        let names: &[String] = self.cur.as_ref().map_or(&[], |c| &c.names);
        let mut json = Map::new();
        for (name, &factor) in names.iter().zip(&self.factors) {
            if factor != 1.0 {
                json.insert(name.clone(), js::number(factor));
            }
        }
        (!json.is_empty()).then_some(Value::Object(json))
    }

    /// `restoreState` for a value present in the JSON. (The TS fills factors
    /// with -1 when given `undefined`; state JSON never passes that, since
    /// `CompoundTrackable` skips absent keys.)
    pub fn restore_state(&mut self, json: &Value, space: &SpaceRef) -> Result<()> {
        let Some(obj) = json.as_object() else {
            bail!("Expected object, but received: {json}");
        };
        self.factors = space
            .names
            .iter()
            .map(|name| match obj.get(name) {
                None => Ok(1.0),
                Some(x) => js::verify_finite_positive_float(x),
            })
            .collect::<Result<_>>()?;
        self.cur = Some(space.clone());
        Ok(())
    }
}

// ── Display dimensions ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DisplayDimensionsValue {
    space: SpaceRef,
    rank: usize,
    /// The first `rank` are indices into the space; the rest are -1.
    indices: [i32; 3],
}

/// Port of `TrackableDisplayDimensions`: which (up to three) global
/// dimensions are shown.
#[derive(Debug, Clone)]
pub struct DisplayDimensions {
    default: bool,
    value: Option<DisplayDimensionsValue>,
}

impl Default for DisplayDimensions {
    fn default() -> Self {
        Self {
            default: true,
            value: None,
        }
    }
}

impl DisplayDimensions {
    fn sync(&mut self, space: &SpaceRef) -> bool {
        let Some(value) = &self.value else {
            self.set_to_default(space);
            return true;
        };
        if same(&value.space, space) {
            return false;
        }
        if self.default {
            self.set_to_default(space);
            return true;
        }
        let mut indices = [-1i32; 3];
        let mut rank = 0;
        for i in 0..value.rank {
            let old_id = value.space.ids[value.indices[i] as usize];
            if let Some(new_dim) = index_of(&space.ids, old_id) {
                indices[rank] = new_dim as i32;
                rank += 1;
            }
        }
        if rank == 0 {
            self.default = true;
            self.set_to_default(space);
            return true;
        }
        self.value = Some(DisplayDimensionsValue {
            space: space.clone(),
            rank,
            indices,
        });
        true
    }

    fn set_to_default(&mut self, space: &SpaceRef) {
        let rank = space.rank().min(3);
        let mut indices = [-1i32; 3];
        for (i, slot) in indices.iter_mut().enumerate().take(rank) {
            *slot = i as i32;
        }
        self.value = Some(DisplayDimensionsValue {
            space: space.clone(),
            rank,
            indices,
        });
    }

    /// `(displayRank, displayDimensionIndices)`.
    pub fn value(&mut self, space: &SpaceRef) -> (usize, [i32; 3]) {
        self.sync(space);
        let v = self.value.as_ref().unwrap();
        (v.rank, v.indices)
    }

    pub fn is_default(&mut self, space: &SpaceRef) -> bool {
        self.sync(space);
        self.default
    }

    pub fn reset(&mut self) {
        self.default = true;
        self.value = None;
    }

    pub fn set_dimension_indices(&mut self, space: &SpaceRef, rank: usize, indices: [i32; 3]) {
        self.default = false;
        self.value = Some(DisplayDimensionsValue {
            space: space.clone(),
            rank,
            indices,
        });
    }

    pub fn to_json(&mut self, space: &SpaceRef) -> Option<Value> {
        if self.default {
            return None;
        }
        self.sync(space);
        let v = self.value.as_ref().unwrap();
        if v.rank == 0 {
            return None;
        }
        Some(Value::Array(
            v.indices[..v.rank]
                .iter()
                .map(|&i| v.space.names[i as usize].clone().into())
                .collect(),
        ))
    }

    pub fn restore_state(&mut self, json: &Value, space: &SpaceRef) -> Result<()> {
        // `dimensionNamesFromJson`.
        let Some(items) = json.as_array() else {
            bail!("Expected array, but received: {json}.");
        };
        let names = items
            .iter()
            .map(|x| x.as_str().map(str::to_owned).ok_or_else(|| anyhow::anyhow!("Expected string, but received: {x}")))
            .collect::<Result<Vec<_>>>()?;
        ensure!(validate_dimension_names(&names, false), "Invalid dimensions: {names:?}");
        ensure!(names.len() <= 3, "Number of spatial dimensions must be <= 3");
        let mut indices = [-1i32; 3];
        let mut rank = 0;
        for name in &names {
            if let Some(index) = space.names.iter().position(|n| n == name) {
                indices[rank] = index as i32;
                rank += 1;
            }
        }
        if rank == 0 {
            self.reset();
            return Ok(());
        }
        self.default = false;
        self.value = Some(DisplayDimensionsValue {
            space: space.clone(),
            rank,
            indices,
        });
        Ok(())
    }
}

// ── Render info ────────────────────────────────────────────────────────────

/// Port of `DisplayDimensionRenderInfo`: what panels need to know about the
/// displayed dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayDimensionRenderInfo {
    pub global_rank: usize,
    pub global_dimension_names: Vec<String>,
    pub display_rank: usize,
    pub display_dimension_indices: [i32; 3],
    /// `relativeDisplayScales[d] * scales[d]` per displayed dimension, else 1.
    pub voxel_physical_scales: [f64; 3],
    pub display_dimension_units: [String; 3],
    pub display_dimension_scales: [f64; 3],
    /// The smallest of `voxel_physical_scales`, or 1 with nothing displayed.
    pub canonical_voxel_physical_size: f64,
    /// `voxel_physical_scales[i] / canonical_voxel_physical_size`.
    pub canonical_voxel_factors: [f64; 3],
}

/// Port of `getDisplayDimensionRenderInfo`.
pub fn display_dimension_render_info(
    space: &CoordinateSpace,
    display_rank: usize,
    indices: [i32; 3],
    factors: &[f64],
) -> DisplayDimensionRenderInfo {
    let mut voxel_physical_scales = [1.0; 3];
    let mut display_dimension_scales = [1.0; 3];
    let mut display_dimension_units: [String; 3] = Default::default();
    let mut canonical_voxel_factors = [1.0; 3];
    let canonical_voxel_physical_size = if display_rank == 0 {
        1.0
    } else {
        let mut min = f64::INFINITY;
        for i in 0..display_rank {
            let dim = indices[i] as usize;
            let s = factors[dim] * space.scales[dim];
            voxel_physical_scales[i] = s;
            min = if min.is_nan() || s.is_nan() { f64::NAN } else { min.min(s) };
            display_dimension_units[i] = space.units[dim].clone();
            display_dimension_scales[i] = space.scales[dim];
        }
        for i in 0..display_rank {
            canonical_voxel_factors[i] = voxel_physical_scales[i] / min;
        }
        min
    };
    DisplayDimensionRenderInfo {
        global_rank: space.rank(),
        global_dimension_names: space.names.clone(),
        display_rank,
        display_dimension_indices: indices,
        voxel_physical_scales,
        display_dimension_units,
        display_dimension_scales,
        canonical_voxel_physical_size,
        canonical_voxel_factors,
    }
}

// ── Zoom ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoomKind {
    /// Canonical voxels per viewport pixel; default 1.
    CrossSection,
    /// Canonical voxels per viewport height; default fits the bounds.
    Projection,
}

/// Port of `TrackableCrossSectionZoom` / `TrackableProjectionZoom`. NaN until
/// there is a valid space to derive a default from.
#[derive(Debug, Clone)]
pub struct Zoom {
    kind: ZoomKind,
    value: f64,
    legacy_value: f64,
    cur_canonical_voxel_physical_size: f64,
}

impl Zoom {
    pub fn new(kind: ZoomKind) -> Self {
        Self {
            kind,
            value: f64::NAN,
            legacy_value: f64::NAN,
            cur_canonical_voxel_physical_size: 0.0,
        }
    }

    /// `handleCoordinateSpaceChanged`.
    fn sync(&mut self, info: &DisplayDimensionRenderInfo, space: &CoordinateSpace) -> bool {
        let size = info.canonical_voxel_physical_size;
        if !self.value.is_nan() && size == self.cur_canonical_voxel_physical_size {
            return false;
        }
        if !self.value.is_nan() {
            if self.cur_canonical_voxel_physical_size != 0.0 {
                self.value *= self.cur_canonical_voxel_physical_size / size;
                self.cur_canonical_voxel_physical_size = size;
                return true;
            }
            return false;
        }
        if !space.valid || size == 0.0 {
            return false;
        }
        self.cur_canonical_voxel_physical_size = size;
        self.value = self.default_value(info, space);
        true
    }

    fn default_value(&mut self, info: &DisplayDimensionRenderInfo, space: &CoordinateSpace) -> f64 {
        let size = info.canonical_voxel_physical_size;
        match self.kind {
            ZoomKind::CrossSection => {
                if self.legacy_value.is_nan() {
                    1.0
                } else {
                    self.legacy_value * 1e-9 / size
                }
            }
            ZoomKind::Projection => {
                if !self.legacy_value.is_nan() {
                    let legacy = self.legacy_value;
                    self.legacy_value = f64::NAN;
                    return 2.0 * 100.0 * js::tan(std::f64::consts::PI / 8.0) * 1e-9 * legacy / size;
                }
                // `canonicalVoxelFactors.reduce(...)` over all three entries;
                // index -1 reads `undefined`, giving NaN, which `Math.max`
                // propagates and `isFinite` then catches.
                let mut value = 0.0f64;
                for i in 0..3 {
                    let dim = info.display_dimension_indices[i];
                    let extent = if dim < 0 {
                        f64::NAN
                    } else {
                        let d = dim as usize;
                        (space.bounds.upper_bounds[d] - space.bounds.lower_bounds[d])
                            * info.canonical_voxel_factors[i]
                    };
                    value = if value.is_nan() || extent.is_nan() { f64::NAN } else { value.max(extent) };
                }
                if !value.is_finite() {
                    // 1024 voxels when there is no bounds information.
                    1024.0
                } else {
                    2f64.powf(js::log2(value).ceil())
                }
            }
        }
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    /// `set value`.
    fn set(&mut self, value: f64, info: &DisplayDimensionRenderInfo) -> bool {
        let size = info.canonical_voxel_physical_size;
        if js::object_is(value, self.value) && size == self.cur_canonical_voxel_physical_size
        {
            return false;
        }
        self.cur_canonical_voxel_physical_size = size;
        self.legacy_value = f64::NAN;
        self.value = value;
        true
    }

    /// `set legacyValue`: the pre-2020 zoom units (`zoomFactor`,
    /// `perspectiveZoom`).
    pub fn set_legacy_value(&mut self, value: f64) {
        if js::object_is(value, self.legacy_value) {
            return;
        }
        self.value = f64::NAN;
        self.legacy_value = value;
        self.cur_canonical_voxel_physical_size = 0.0;
    }

    pub fn to_json(&self) -> Option<Value> {
        (!self.value.is_nan()).then(|| js::number(self.value))
    }

    pub fn restore_state(&mut self, json: &Value) -> Result<()> {
        self.cur_canonical_voxel_physical_size = 0.0;
        self.legacy_value = f64::NAN;
        self.value = js::verify_finite_positive_float(json)?;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.cur_canonical_voxel_physical_size = 0.0;
        self.value = f64::NAN;
        self.legacy_value = f64::NAN;
    }

    /// After a restore, what the viewer's linked copy of this zoom
    /// (`LinkedZoomState`, the layer group viewer's) does: it assigns the
    /// value back with `setPhysicalScale`, which ties it to the canonical
    /// voxel size of the space at hand (1 m when there is none yet), so it
    /// rescales when the space arrives or changes.
    fn adopt_canonical_voxel_physical_size(&mut self, size: f64) {
        if !self.value.is_nan() {
            self.cur_canonical_voxel_physical_size = size;
        }
    }
}

// ── Depth range ────────────────────────────────────────────────────────────

/// Port of `TrackableDepthRange`: how thick a slab the view draws. Positive
/// values are in canonical voxels, negative ones in viewport pixels.
#[derive(Debug, Clone)]
pub struct DepthRange {
    pub default_value: f64,
    value: f64,
    canonical_voxel_physical_size: f64,
}

impl DepthRange {
    fn new(default_value: f64, info: &DisplayDimensionRenderInfo) -> Self {
        Self {
            default_value,
            value: default_value,
            canonical_voxel_physical_size: info.canonical_voxel_physical_size,
        }
    }

    /// The `value` getter.
    ///
    /// When the canonical voxel size changes, a positive value becomes
    /// `previous size / new size` — the TS assigns the ratio rather than
    /// scaling the value by it. That is what neuroglancer does, so it is what
    /// this does.
    fn sync(&mut self, info: &DisplayDimensionRenderInfo) -> bool {
        if self.value > 0.0 {
            let size = info.canonical_voxel_physical_size;
            let prev = self.canonical_voxel_physical_size;
            if size != prev {
                self.canonical_voxel_physical_size = size;
                self.value = prev / size;
                return true;
            }
        }
        false
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    fn set(&mut self, value: f64, info: &DisplayDimensionRenderInfo) -> bool {
        self.sync(info);
        if value == self.value {
            return false;
        }
        self.value = value;
        self.canonical_voxel_physical_size = info.canonical_voxel_physical_size;
        true
    }

    pub fn to_json(&self) -> Option<Value> {
        (self.value != self.default_value).then(|| js::number(self.value))
    }

    fn restore_state(&mut self, json: &Value, info: &DisplayDimensionRenderInfo) {
        let value = match json.as_f64() {
            Some(x) if x.is_finite() && x != 0.0 => x,
            _ => self.default_value,
        };
        self.set(value, info);
    }
}

// ── Navigation ─────────────────────────────────────────────────────────────

/// The cross-section and 3D views each have an orientation, zoom and depth
/// range; they share the position and display dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    CrossSection,
    Projection,
}

#[derive(Debug, Clone)]
struct ViewState {
    orientation: Orientation,
    /// Bumped wherever the TS `OrientationState` dispatches `changed`.
    orientation_version: u64,
    /// The orientation as of that dispatch. TS listeners (linked panels)
    /// read the value when the signal fires; `toJSON` can normalize it in
    /// place afterwards without signalling, and they never see that.
    signalled: Quat,
    zoom: Zoom,
    depth: DepthRange,
}

/// The viewer's navigation: its coordinate space plus everything that moves
/// through it. The `position`, `crossSection*`, `projection*`,
/// `displayDimensions` and `relativeDisplayScales` members of the viewer state.
#[derive(Debug, Clone)]
pub struct Navigation {
    space: SpaceRef,
    position: Position,
    relative_display_scales: RelativeDisplayScales,
    display_dimensions: DisplayDimensions,
    render_info: DisplayDimensionRenderInfo,
    cross_section: ViewState,
    projection: ViewState,
    generation: u64,
}

impl Navigation {
    pub fn new(space: SpaceRef) -> Self {
        let render_info = display_dimension_render_info(&space, 0, [-1; 3], &[]);
        let view = |kind, depth| ViewState {
            orientation: Orientation::default(),
            orientation_version: 0,
            signalled: glm::QUAT_IDENTITY,
            zoom: Zoom::new(kind),
            depth: DepthRange::new(depth, &render_info),
        };
        let mut nav = Self {
            cross_section: view(ZoomKind::CrossSection, -10.0),
            projection: view(ZoomKind::Projection, -50.0),
            render_info,
            space,
            position: Position::default(),
            relative_display_scales: RelativeDisplayScales::default(),
            display_dimensions: DisplayDimensions::default(),
            generation: 0,
        };
        nav.sync();
        nav
    }

    /// Changes whenever anything about the navigation does.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn touch(&mut self, changed: bool) {
        if changed {
            self.generation += 1;
        }
    }

    pub fn coordinate_space(&self) -> &SpaceRef {
        &self.space
    }

    /// Replaces the coordinate space. Position, display dimensions, scales
    /// and zoom carry over by dimension ID, as in neuroglancer.
    pub fn set_coordinate_space(&mut self, space: SpaceRef) {
        if same(&self.space, &space) {
            return;
        }
        self.space = space;
        self.generation += 1;
        self.sync();
    }

    /// Brings every part up to date with the space, in dependency order.
    fn sync(&mut self) {
        let space = self.space.clone();
        let mut changed = self.position.sync(&space);
        changed |= self.relative_display_scales.sync(&space);
        changed |= self.display_dimensions.sync(&space);
        let (rank, indices) = self.display_dimensions.value(&space);
        let info = display_dimension_render_info(
            &space,
            rank,
            indices,
            self.relative_display_scales.factors(&space),
        );
        if info != self.render_info {
            self.render_info = info;
            changed = true;
        }
        for view in [&mut self.cross_section, &mut self.projection] {
            changed |= view.zoom.sync(&self.render_info, &space);
            changed |= view.depth.sync(&self.render_info);
        }
        self.touch(changed);
    }

    /// What the TS `orientation.changed.dispatch()` does here: linked panels
    /// see the value as it is now.
    fn signal_orientation(&mut self, v: View) {
        let view = self.view_mut(v);
        view.orientation_version += 1;
        view.signalled = view.orientation.0;
    }

    fn view(&self, v: View) -> &ViewState {
        match v {
            View::CrossSection => &self.cross_section,
            View::Projection => &self.projection,
        }
    }

    fn view_mut(&mut self, v: View) -> &mut ViewState {
        match v {
            View::CrossSection => &mut self.cross_section,
            View::Projection => &mut self.projection,
        }
    }

    pub fn valid(&self) -> bool {
        self.space.valid && !self.cross_section.zoom.value().is_nan()
    }

    pub fn position(&self) -> &[f32] {
        &self.position.coordinates
    }

    pub fn set_position(&mut self, coordinates: &[f32]) {
        let changed = self.position.set_value(coordinates);
        self.touch(changed);
    }

    pub fn render_info(&self) -> &DisplayDimensionRenderInfo {
        &self.render_info
    }

    pub fn orientation(&self, v: View) -> Quat {
        self.view(v).orientation.0
    }

    pub fn set_orientation(&mut self, v: View, q: Quat) {
        self.view_mut(v).orientation.0 = q;
        self.signal_orientation(v);
        self.generation += 1;
    }

    /// `relativeDisplayScales.value.factors`, per global dimension.
    pub fn relative_display_factors(&mut self) -> Vec<f64> {
        let space = self.space.clone();
        self.relative_display_scales.factors(&space).to_vec()
    }

    pub fn zoom(&self, v: View) -> f64 {
        self.view(v).zoom.value()
    }

    pub fn set_zoom(&mut self, v: View, value: f64) {
        let info = self.render_info.clone();
        let changed = self.view_mut(v).zoom.set(value, &info);
        self.touch(changed);
    }

    /// `NavigationState.zoomBy`.
    pub fn zoom_by(&mut self, v: View, factor: f64) {
        let value = self.zoom(v) * factor;
        self.set_zoom(v, value);
    }

    pub fn depth_range(&self, v: View) -> f64 {
        self.view(v).depth.value()
    }

    pub fn set_depth_range(&mut self, v: View, value: f64) {
        let info = self.render_info.clone();
        let changed = self.view_mut(v).depth.set(value, &info);
        self.touch(changed);
    }

    /// `NavigationState.relativeDepthRange`: in canonical voxels.
    pub fn relative_depth_range(&self, v: View) -> f64 {
        let depth = self.depth_range(v);
        if depth > 0.0 {
            depth / self.zoom(v)
        } else {
            -depth
        }
    }

    /// `DisplayPose.toMat4(mat, zoom)`: view coordinates to global voxel
    /// coordinates of the displayed dimensions.
    pub fn to_mat4(&self, v: View) -> Mat4 {
        self.to_mat4_zoom(v, self.zoom(v))
    }

    /// `DisplayPose.toMat4(mat, zoom)` with an explicit zoom.
    pub fn to_mat4_zoom(&self, v: View, zoom: f64) -> Mat4 {
        let mut mat = glm::mat4_from_quat(self.orientation(v));
        let info = &self.render_info;
        for i in 0..3 {
            let dim = info.display_dimension_indices[i];
            let scale = zoom / info.canonical_voxel_factors[i];
            for col in [0, 4, 8] {
                mat[col + i] = (mat[col + i] as f64 * scale) as f32;
            }
            // `voxelCoordinates[dim] || 0`: missing (dim -1) or NaN is 0.
            let c = if dim < 0 { 0.0 } else { self.position.coordinates.get(dim as usize).copied().unwrap_or(0.0) };
            mat[12 + i] = if c.is_nan() { 0.0 } else { c };
        }
        mat
    }

    /// `DisplayPose.updateDisplayPosition`: edit the displayed dimensions of
    /// the position as a vec3. `f` returning `false` cancels.
    pub fn update_display_position(&mut self, f: impl FnOnce(&mut Vec3) -> bool) -> bool {
        let (rank, indices) = (self.render_info.display_rank, self.render_info.display_dimension_indices);
        let coords = &mut self.position.coordinates;
        let mut temp = [0f32; 3];
        for i in 0..rank {
            temp[i] = coords[indices[i] as usize];
        }
        if !f(&mut temp) {
            return false;
        }
        for i in 0..rank {
            coords[indices[i] as usize] = temp[i];
        }
        self.generation += 1;
        true
    }

    /// `DisplayPose.translateDimensionRelative`: step one dimension, landing
    /// on a voxel center inside the bounds.
    pub fn translate_dimension_relative(&mut self, dim: usize, adjustment: f64) {
        if !self.space.valid {
            return;
        }
        let c = &mut self.position.coordinates[dim];
        *c = self.space.bounds.clamp_and_round_to_voxel_center(dim, *c as f64 + adjustment) as f32;
        self.generation += 1;
    }

    /// `DisplayPose.translateVoxelsRelative`: move by `translation` in view
    /// axes, each moved dimension landing on a voxel center.
    pub fn translate_voxels_relative(&mut self, v: View, translation: Vec3) {
        if !self.space.valid {
            return;
        }
        let temp = glm::vec3_transform_quat(translation, self.orientation(v));
        let info = &self.render_info;
        for (&adjustment, &dim) in temp.iter().zip(&info.display_dimension_indices).take(info.display_rank) {
            if adjustment == 0.0 {
                continue;
            }
            let dim = dim as usize;
            let c = &mut self.position.coordinates[dim];
            *c = self.space.bounds.clamp_and_round_to_voxel_center(dim, *c as f64 + adjustment as f64) as f32;
        }
        self.generation += 1;
    }

    /// `DisplayPose.rotateRelative`: rotate about `axis` in view coordinates.
    pub fn rotate_relative(&mut self, v: View, axis: Vec3, angle: f64) {
        let view = self.view_mut(v);
        view.orientation.0 = glm::quat_multiply(view.orientation.0, glm::quat_set_axis_angle(axis, angle));
        self.signal_orientation(v);
        self.generation += 1;
    }

    /// `DisplayPose.rotateAbsolute`: rotate about `axis` (display coordinates)
    /// through the global point `fixed_point`, which stays put on screen.
    pub fn rotate_absolute(&mut self, v: View, axis: Vec3, angle: f64, fixed_point: &[f32]) {
        let space = self.space.clone();
        let factors = self.relative_display_scales.factors(&space).to_vec();
        let (rank, indices) = (self.render_info.display_rank, self.render_info.display_dimension_indices);
        let temp = glm::quat_set_axis_angle(axis, angle);
        let mut orientation = self.orientation(v);

        // oldOrientation * local + oldPosition == fixedPoint, so
        // local = inverse(oldOrientation) * (fixedPoint - oldPosition).
        let mut local = [0f32; 3];
        let coords = &mut self.position.coordinates;
        for i in 0..rank {
            let dim = indices[i] as usize;
            let diff = fixed_point[dim] as f64 - coords[dim] as f64;
            local[i] = (diff * space.scales[dim] * factors[dim]) as f32;
        }
        local = glm::vec3_transform_quat(local, glm::quat_invert(orientation));
        // newPosition = fixedPoint - newOrientation * local.
        orientation = glm::quat_multiply(temp, orientation);
        local = glm::vec3_transform_quat(local, orientation);
        for i in 0..rank {
            let dim = indices[i] as usize;
            coords[dim] = (fixed_point[dim] as f64 - local[i] as f64 / (space.scales[dim] * factors[dim])) as f32;
        }
        self.view_mut(v).orientation.0 = orientation;
        self.signal_orientation(v);
        self.generation += 1;
    }

    /// `DisplayPose.snap`.
    pub fn snap(&mut self, v: View) {
        self.view_mut(v).orientation.snap();
        self.signal_orientation(v);
        let space = self.space.clone();
        self.position.snap_to_voxel(&space);
        self.generation += 1;
    }

    /// Runs `f` as a panel whose orientation is `relative` to the
    /// cross-section orientation, the way neuroglancer's xz and yz slice views
    /// hold their own `NavigationState`: inside `f`, `View::CrossSection` has
    /// the panel's orientation; afterwards, a changed orientation is carried
    /// back to the viewer's.
    pub fn with_relative_cross_section<R>(
        &mut self,
        relative: &mut RelativeOrientation,
        f: impl FnOnce(&mut Navigation) -> R,
    ) -> R {
        let base = self.cross_section.orientation.0;
        let version = self.cross_section.orientation_version;
        let panel = relative.get(self.cross_section.signalled, version);
        self.cross_section.orientation.0 = panel;
        let result = f(self);
        let view = &mut self.cross_section;
        if view.orientation_version == version {
            view.orientation.0 = base;
        } else {
            // The panel signalled a change: the TS listener recomputes the
            // peer from it, whether or not the value moved.
            view.orientation_version += 1;
            view.orientation.0 = relative.set(view.orientation.0, view.orientation_version);
            view.signalled = view.orientation.0;
        }
        result
    }

    // ── JSON ───────────────────────────────────────────────────────────────

    /// The viewer-state members this covers, under their neuroglancer keys.
    /// Absent members are omitted, as `toJSON` returning `undefined` is.
    pub fn to_json(&mut self) -> Map<String, Value> {
        let space = self.space.clone();
        let mut out = Map::new();
        let mut put = |k: &str, v: Option<Value>| {
            if let Some(v) = v {
                out.insert(k.to_owned(), v);
            }
        };
        put("dimensions", space.to_json());
        put("relativeDisplayScales", self.relative_display_scales.to_json(&space));
        put("displayDimensions", self.display_dimensions.to_json(&space));
        put("position", self.position.to_json(&space));
        put("crossSectionOrientation", self.cross_section.orientation.to_json());
        put("crossSectionScale", self.cross_section.zoom.to_json());
        put("crossSectionDepth", self.cross_section.depth.to_json());
        put("projectionOrientation", self.projection.orientation.to_json());
        put("projectionScale", self.projection.zoom.to_json());
        put("projectionDepth", self.projection.depth.to_json());
        out
    }

    /// Restores the members present in `json`, in the order
    /// `TrackableViewerState` registers them. Absent members are left as they
    /// are (`CompoundTrackable` skips them); `null` is passed on like any other
    /// value, so `"position": null` is an error, as in neuroglancer.
    ///
    /// The coordinate space itself (`dimensions`) is restored by the caller,
    /// since it is shared with the layers.
    pub fn restore_state(&mut self, json: &Map<String, Value>) -> Result<()> {
        let space = self.space.clone();
        let present = |k: &str| json.get(k);
        let ctx = |k: &str| format!("Error restoring property {k:?}");
        use anyhow::Context as _;
        if let Some(v) = present("relativeDisplayScales") {
            self.relative_display_scales.restore_state(v, &space).with_context(|| ctx("relativeDisplayScales"))?;
        }
        if let Some(v) = present("displayDimensions") {
            self.display_dimensions.restore_state(v, &space).with_context(|| ctx("displayDimensions"))?;
        }
        if let Some(v) = present("position") {
            self.position.restore_state(v, &space).with_context(|| ctx("position"))?;
        }
        if let Some(v) = present("crossSectionOrientation") {
            self.cross_section.orientation.restore_state(v);
            self.signal_orientation(View::CrossSection);
        }
        if let Some(v) = present("crossSectionScale") {
            self.sync();
            self.cross_section.zoom.restore_state(v).with_context(|| ctx("crossSectionScale"))?;
            let size = self.render_info.canonical_voxel_physical_size;
            self.cross_section.zoom.adopt_canonical_voxel_physical_size(size);
        }
        self.sync();
        let info = self.render_info.clone();
        if let Some(v) = present("crossSectionDepth") {
            self.cross_section.depth.restore_state(v, &info);
        }
        if let Some(v) = present("projectionOrientation") {
            self.projection.orientation.restore_state(v);
            self.signal_orientation(View::Projection);
        }
        if let Some(v) = present("projectionScale") {
            self.sync();
            self.projection.zoom.restore_state(v).with_context(|| ctx("projectionScale"))?;
            let size = self.render_info.canonical_voxel_physical_size;
            self.projection.zoom.adopt_canonical_voxel_physical_size(size);
        }
        if let Some(v) = present("projectionDepth") {
            self.projection.depth.restore_state(v, &info);
        }
        self.generation += 1;
        self.sync();
        Ok(())
    }
}
