//! Playback (`navigation_state.ts` `CoordinateSpacePlaybackVelocity`,
//! `PlaybackManager`): a dimension's coordinate advancing at a velocity,
//! stopping, looping or reversing at the bounds.

use serde_json::{Map, Value};

/// `VelocityBoundaryBehavior`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    Stop,
    Loop,
    Reverse,
}

impl Boundary {
    pub const ALL: [Boundary; 3] = [Boundary::Stop, Boundary::Loop, Boundary::Reverse];

    pub fn name(self) -> &'static str {
        match self {
            Boundary::Stop => "stop",
            Boundary::Loop => "loop",
            Boundary::Reverse => "reverse",
        }
    }

    fn parse(s: &str) -> Option<Boundary> {
        Boundary::ALL.into_iter().find(|b| b.name() == s.to_ascii_lowercase())
    }
}

/// `DEFAULT_PLAYBACK_VELOCITY`.
pub const DEFAULT_VELOCITY: f64 = 10.0;

/// `DimensionPlaybackVelocity`: coordinates per second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DimensionVelocity {
    pub velocity: f64,
    pub at_boundary: Boundary,
    pub paused: bool,
}

impl Default for DimensionVelocity {
    /// `new DimensionPlaybackVelocity()`: reversing, paused.
    fn default() -> Self {
        DimensionVelocity { velocity: DEFAULT_VELOCITY, at_boundary: Boundary::Reverse, paused: true }
    }
}

impl DimensionVelocity {
    /// `dimensionVelocityFromJson`: an object; missing members their
    /// defaults (`atBoundary` then `stop`).
    pub fn from_json(v: &Value) -> Option<DimensionVelocity> {
        let o = v.as_object()?;
        Some(DimensionVelocity {
            velocity: o.get("velocity").and_then(Value::as_f64).filter(|v| v.is_finite()).unwrap_or(DEFAULT_VELOCITY),
            at_boundary: o.get("atBoundary").and_then(Value::as_str).and_then(Boundary::parse).unwrap_or(Boundary::Stop),
            paused: o.get("paused").and_then(Value::as_bool).unwrap_or(true),
        })
    }

    /// `dimensionVelocityToJson`.
    pub fn to_json(&self) -> Value {
        let mut o = Map::new();
        o.insert("velocity".into(), ng_core::js::number(self.velocity));
        if self.at_boundary != Boundary::Stop {
            o.insert("atBoundary".into(), self.at_boundary.name().into());
        }
        if !self.paused {
            o.insert("paused".into(), false.into());
        }
        Value::Object(o)
    }
}

/// A playing dimension's last step (`DimensionPlaybackState`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Running {
    pub prev_time: f64,
    pub prev_coordinate: f32,
}

/// `PlaybackManager.updateStarted` for one dimension: the coordinate after
/// `dt` ms, and the velocity as it then is (paused or reversed at a bound).
pub fn step(coordinate: f32, velocity: DimensionVelocity, dt: f64, lower: f64, upper: f64) -> (f32, DimensionVelocity) {
    let delta = dt * velocity.velocity / 1000.0;
    if delta == 0.0 {
        return (coordinate, velocity);
    }
    let mut v = velocity;
    let mut c = coordinate as f64 + delta;
    let upper = (upper - 1.0).ceil();
    let (limit, opposite) = if delta > 0.0 { (upper, lower) } else { (lower, upper) };
    let sign = delta.signum();
    if limit.is_finite() && c * sign >= limit * sign {
        match v.at_boundary {
            Boundary::Loop if opposite.is_finite() => c = opposite,
            Boundary::Loop | Boundary::Stop => {
                v.paused = true;
                c = limit;
            }
            Boundary::Reverse => {
                v.velocity = -v.velocity;
                c = limit;
            }
        }
    }
    (c as f32, v)
}

/// `multiplyVelocity`: rounded, never to 0.
pub fn multiply(velocity: f64, factor: f64) -> f64 {
    let v = ng_core::js::round(velocity * factor);
    if v == 0.0 {
        if velocity == 0.0 { 1.0 } else { velocity.signum() }
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_and_bounds_as_neuroglancer() {
        let v = DimensionVelocity { velocity: 10.0, at_boundary: Boundary::Reverse, paused: false };
        assert_eq!(step(5.0, v, 100.0, 0.0, 40.0).0, 6.0);
        let (c, v2) = step(38.5, v, 100.0, 0.0, 40.0);
        assert_eq!((c, v2.velocity), (39.0, -10.0));
        let (c, v3) = step(38.5, DimensionVelocity { at_boundary: Boundary::Stop, ..v }, 100.0, 0.0, 40.0);
        assert_eq!((c, v3.paused), (39.0, true));
        let (c, _) = step(38.5, DimensionVelocity { at_boundary: Boundary::Loop, ..v }, 100.0, 0.0, 40.0);
        assert_eq!(c, 0.0);
        assert_eq!(DimensionVelocity::from_json(&serde_json::json!({"velocity": 3, "paused": false})).unwrap().at_boundary, Boundary::Stop);
        assert_eq!(DimensionVelocity::default().to_json(), serde_json::json!({"velocity": 10, "atBoundary": "reverse"}));
        assert_eq!((multiply(10.0, 0.5), multiply(1.0, 0.1), multiply(-1.0, 0.1)), (5.0, 1.0, -1.0));
    }
}
