//! The position widget's dimension plot (`src/widget/position_plot.ts`,
//! column orientation): each layer's extent along one dimension as a bar,
//! with ticks at its ends, and lines at the position and under the pointer.

use ng_core::coordinate_space::compute_combined_lower_upper_bound;
use ng_core::CoordinateSpace;

/// `tickWidth`, `barWidth`, `barRightMargin` (column orientation).
pub const TICK_WIDTH: f64 = 10.0;
pub const BAR_WIDTH: f64 = 15.0;
pub const BAR_RIGHT_MARGIN: f64 = 10.0;
/// The canvas: 35 × 100.
pub const CANVAS_WIDTH: f64 = TICK_WIDTH + BAR_WIDTH + BAR_RIGHT_MARGIN;
pub const CANVAS_HEIGHT: f64 = 100.0;

/// `getCanvasYFromCoordinate`.
pub fn canvas_y(coordinate: f64, lower: f64, upper: f64, height: f64) -> f64 {
    ((coordinate - lower) * (height - 1.0) / (upper - lower)).floor()
}

/// A dimension's plot: its integer bounds and each bounding box's extent in
/// canvas rows (`NormalizedDimensionBounds`).
#[derive(Debug, Clone, PartialEq)]
pub struct DimensionBounds {
    pub lower: f64,
    pub upper: f64,
    pub boxes: Vec<(f64, f64)>,
}

/// `getNormalizedDimensionBounds`; `None` where the plot is hidden (unbounded,
/// or a single position).
pub fn dimension_bounds(space: &CoordinateSpace, dim: usize, height: f64) -> Option<DimensionBounds> {
    let b = &space.bounds;
    // `getDisplayLowerUpperBounds`.
    let shift = if b.voxel_center_at_integer_coordinates[dim] { 0.5 } else { 0.0 };
    let lower = (b.lower_bounds[dim] + shift).floor();
    let upper = (b.upper_bounds[dim] + shift - 1.0).floor();
    if !lower.is_finite() || !upper.is_finite() || b.lower_bounds[dim] + 1.0 == b.upper_bounds[dim] {
        return None;
    }
    let normalize = |x: f64| canvas_y(x, lower, upper, height);
    let mut boxes: Vec<(f64, f64)> = space
        .bounding_boxes
        .iter()
        .filter_map(|bbox| compute_combined_lower_upper_bound(bbox, dim, space.rank()))
        .map(|(lo, hi)| (normalize(lo).max(0.0), normalize((hi - 1.0).ceil()).min(height - 1.0)))
        .collect();
    // By lower end (the tie-break compares `b` with itself: none), then
    // without repeats.
    boxes.sort_by(|a, b| a.0.total_cmp(&b.0));
    boxes.dedup();
    Some(DimensionBounds { lower, upper, boxes })
}

/// The canvas as filled rectangles `(x, y, width, height)` per colour:
/// white ticks, `#ccc` bars (`drawDimensionBounds`).
pub fn bars(bounds: &DimensionBounds) -> (Vec<[f64; 4]>, Vec<[f64; 4]>) {
    let mut ticks = vec![];
    for &(lo, hi) in &bounds.boxes {
        ticks.push([0.0, lo, TICK_WIDTH, 1.0]);
        ticks.push([0.0, hi, TICK_WIDTH, 1.0]);
    }
    let n = bounds.boxes.len() as f64;
    let bars = bounds
        .boxes
        .iter()
        .enumerate()
        .map(|(i, &(lo, hi))| {
            let x = (i as f64 * BAR_WIDTH / n).floor();
            [x + TICK_WIDTH, lo, (BAR_WIDTH / n).max(1.0), hi + 1.0 - lo]
        })
        .collect();
    (ticks, bars)
}

/// `tickWidth`, `barWidth`, `barRightMargin` in row orientation (the
/// dimension tool's plot): the canvas is as wide as its box, this tall.
pub const ROW_TICK_WIDTH: f64 = 5.0;
pub const ROW_BAR_WIDTH: f64 = 10.0;
pub const ROW_CANVAS_HEIGHT: f64 = ROW_TICK_WIDTH + ROW_BAR_WIDTH + 2.0;

/// [`bars`] in row orientation: the bounds along x, ticks down from the
/// top, the bars under them.
pub fn row_bars(bounds: &DimensionBounds) -> (Vec<[f64; 4]>, Vec<[f64; 4]>) {
    let mut ticks = vec![];
    for &(lo, hi) in &bounds.boxes {
        ticks.push([lo, 0.0, 1.0, ROW_TICK_WIDTH]);
        ticks.push([hi, 0.0, 1.0, ROW_TICK_WIDTH]);
    }
    let n = bounds.boxes.len() as f64;
    let bars = bounds
        .boxes
        .iter()
        .enumerate()
        .map(|(i, &(lo, hi))| {
            let y = (i as f64 * ROW_BAR_WIDTH / n).floor();
            [lo, y + ROW_TICK_WIDTH, hi + 1.0 - lo, (ROW_BAR_WIDTH / n).max(1.0)]
        })
        .collect();
    (ticks, bars)
}

/// The row of `position` (`drawPositionIndicator`), if within the bounds.
pub fn indicator(bounds: &DimensionBounds, position: f64, height: f64) -> Option<f64> {
    (position >= bounds.lower && position.floor() <= bounds.upper).then(|| canvas_y(position, bounds.lower, bounds.upper, height))
}

/// The coordinate at a fraction `t` (0 top, 1 bottom) of the canvas
/// (`getPositionFromMouseEvent`).
pub fn coordinate_at(bounds: &DimensionBounds, t: f64) -> f64 {
    ng_core::js::round(t.clamp(0.0, 1.0) * (bounds.upper - bounds.lower)) + bounds.lower
}

/// A rectangle as SVG path data.
pub fn rect_path(r: [f64; 4]) -> String {
    let [x, y, w, h] = r;
    format!("M{x} {y}h{w}v{h}h{}Z", -w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_coordinates_to_rows_as_neuroglancer() {
        let b = DimensionBounds { lower: 0.0, upper: 43.0, boxes: vec![(0.0, 99.0)] };
        assert_eq!(canvas_y(19.5, 0.0, 43.0, 100.0), 44.0);
        assert_eq!(indicator(&b, 19.5, 100.0), Some(44.0));
        assert_eq!(indicator(&b, 44.0, 100.0), None);
        assert_eq!(coordinate_at(&b, 0.5), 22.0);
        let (ticks, bars) = bars(&DimensionBounds { lower: 0.0, upper: 43.0, boxes: vec![(0.0, 92.0), (4.0, 99.0)] });
        assert_eq!(ticks.len(), 4);
        assert_eq!(bars, vec![[10.0, 0.0, 7.5, 93.0], [17.0, 4.0, 7.5, 96.0]]);
        assert_eq!(rect_path([10.0, 0.0, 7.5, 93.0]), "M10 0h7.5v93h-7.5Z");
        let (ticks, bars) = row_bars(&DimensionBounds { lower: 0.0, upper: 19.0, boxes: vec![(0.0, 1062.0)] });
        assert_eq!(ticks, vec![[0.0, 0.0, 1.0, 5.0], [1062.0, 0.0, 1.0, 5.0]]);
        assert_eq!(bars, vec![[0.0, 5.0, 1063.0, 10.0]]);
    }
}
