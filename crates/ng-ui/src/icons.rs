//! The icons neuroglancer's UI shows: ikonate's SVGs (`ikonate/icons/*.svg`),
//! their `rect`, `polyline`, `line` and `circle` elements written as path
//! data. All are 24 × 24, drawn at 16 px with `.neuroglancer-icon svg`'s
//! stroke: white, 2 wide, round caps and joins, no fill.

/// An icon: path data, and whether it is filled white (the delete button's
/// `svg.style.fill = "white"`).
#[derive(Debug, Clone, Copy)]
pub struct Icon {
    pub paths: &'static [&'static str],
    pub fill: bool,
}

const fn stroked(paths: &'static [&'static str]) -> Icon {
    Icon { paths, fill: false }
}

/// `<circle cx cy r>` as [`threers::ui::circle_path`] writes it (checked in
/// the tests), so painters draw it as an oval, as browsers do.
macro_rules! circle {
    ($s:literal) => {
        $s
    };
}

pub const COPY: Icon = stroked(&["M8 7h12v14H8z", "M16 3L4 3L4 17"]);
pub const TOOL: Icon = stroked(&["M9.74292939,13.7429294 C9.19135019,13.9101088 8.60617271,14 8,14 C4.6862915,14 2,11.3137085 2,8 C2,7.07370693 2.20990431,6.19643964 2.58474197,5.4131691 L6.94974747,9.77817459 L9.77817459,6.94974747 L5.4131691,2.58474197 C6.19643964,2.20990431 7.07370693,2 8,2 C11.3137085,2 14,4.6862915 14,8 C14,8.88040772 13.8103765,9.71652648 13.4697429,10.4697429 L20.5858636,17.5858636 C21.3669122,18.3669122 21.3669122,19.6332422 20.5858636,20.4142907 L19.9142907,21.0858636 C19.1332422,21.8669122 17.8669122,21.8669122 17.0858636,21.0858636 L9.74292939,13.7429294 Z"]);
pub const LAYERS: Icon = stroked(&["M12 4L20 8.00004L12 12L4 8.00004L12 4Z", "M20 12L12 16L4 12", "M20 16L12 20L4 16"]);
/// `<line x1="7" y1="7" x2="7" y2="7"/>`: a dot from the round caps. A 2D
/// canvas drops zero-length segments, so these are a hair long.
pub const LIST: Icon = stroked(&["M10 7L18 7M10 12L18 12M10 17L18 17", "M7 7L7.01 7", "M7 12L7.01 12", "M7 17L7.01 17"]);
pub const CONTROLS_ALT: Icon = stroked(&[
    circle!("M11 6A2 2 0 1 1 7 6A2 2 0 1 1 11 6Z"),
    "M4 6H7",
    "M11 6H20",
    circle!("M11 18A2 2 0 1 1 7 18A2 2 0 1 1 11 18Z"),
    "M4 18H7",
    "M11 18H20",
    circle!("M17 12A2 2 0 1 1 13 12A2 2 0 1 1 17 12Z"),
    "M4 12H13",
    "M17 12L20 12",
]);
pub const CAMERA: Icon = stroked(&["M8 7l.74-1.11A2 2 0 0 1 10.404 5h3.192a2 2 0 0 1 1.664.89L16 7h5v11H3V7h5z", circle!("M15 12A3 3 0 1 1 9 12A3 3 0 1 1 15 12Z")]);
pub const CLIPBOARD: Icon = stroked(&[
    "M15 3L19 3L19 21L5 21L5 3L5 3L9 3",
    "M14,4 L10,4 C9.44771525,4 9,3.55228475 9,3 C9,2.44771525 9.44771525,2 10,2 L14,2 C14.5522847,2 15,2.44771525 15,3 C15,3.55228475 14.5522847,4 14,4 Z",
]);
pub const SETTINGS: Icon = stroked(&[
    "M5.03506429,12.7050339 C5.01187484,12.4731696 5,12.2379716 5,12 C5,11.7620284 5.01187484,11.5268304 5.03506429,11.2949661 L3.20577137,9.23205081 L5.20577137,5.76794919 L7.9069713,6.32070904 C8.28729123,6.0461342 8.69629298,5.80882212 9.12862533,5.61412402 L10,3 L14,3 L14.8713747,5.61412402 C15.303707,5.80882212 15.7127088,6.0461342 16.0930287,6.32070904 L18.7942286,5.76794919 L20.7942286,9.23205081 L18.9649357,11.2949661 C18.9881252,11.5268304 19,11.7620284 19,12 C19,12.2379716 18.9881252,12.4731696 18.9649357,12.7050339 L20.7942286,14.7679492 L18.7942286,18.2320508 L16.0930287,17.679291 C15.7127088,17.9538658 15.303707,18.1911779 14.8713747,18.385876 L14,21 L10,21 L9.12862533,18.385876 C8.69629298,18.1911779 8.28729123,17.9538658 7.9069713,17.679291 L5.20577137,18.2320508 L3.20577137,14.7679492 L5.03506429,12.7050339 Z",
    circle!("M13 12A1 1 0 1 1 11 12A1 1 0 1 1 13 12Z"),
]);
pub const VIDEO: Icon = stroked(&["M18 12L9 16.9L9 7Z", circle!("M22 12A10 10 0 1 1 2 12A10 10 0 1 1 22 12Z")]);
pub const SEARCH: Icon = stroked(&["M14.4121122,14.4121122 L20,20", circle!("M16 10A6 6 0 1 1 4 10A6 6 0 1 1 16 10Z")]);
pub const PLAY: Icon = stroked(&["M20 12L5 21V3z"]);
pub const PAUSE: Icon = stroked(&["M5 4h4v16H5z", "M15 4h4v16h-4z"]);
pub const EYE: Icon = stroked(&["M22 12C22 12 19 18 12 18C5 18 2 12 2 12C2 12 5 6 12 6C19 6 22 12 22 12Z", circle!("M15 12A3 3 0 1 1 9 12A3 3 0 1 1 15 12Z")]);
pub const EYE_CROSSED: Icon = stroked(&["M22 12C22 12 19 18 12 18C5 18 2 12 2 12C2 12 5 6 12 6C19 6 22 12 22 12Z", circle!("M15 12A3 3 0 1 1 9 12A3 3 0 1 1 15 12Z"), "M3 21L20 4"]);
pub const ARROW_LEFT: Icon = stroked(&["M9 6l-6 6 6 6", "M21 12H4", "M3 12h1"]);
pub const ARROW_UP: Icon = stroked(&["M18 9l-6-6-6 6", "M12 21V4", "M12 3v1"]);
pub const ARROW_RIGHT: Icon = stroked(&["M15 18l6-6-6-6", "M3 12h17", "M21 12h-1"]);
pub const STAR: Icon = stroked(&["M12 17.844L6.183 20.902L7.294 14.425L2.588 9.838L9.092 8.893L12 3L14.908 8.893L21.412 9.838L16.706 14.425L17.817 20.902Z"]);
pub const CURSOR: Icon = stroked(&["M7 20L7 4L19 16L12 16L7 21Z"]);
/// ikonate `download`.
pub const DOWNLOAD: Icon = stroked(&["M12,3 L12,16", "M7 12L12 17L17 12", "M20,21 L4,21"]);
/// ikonate `upload`.
pub const UPLOAD: Icon = stroked(&["M12,4 L12,17", "M7 8L12 3L17 8", "M20,21 L4,21"]);
/// ikonate `help`.
pub const HELP: Icon = stroked(&[
    "M12 14C12 12 13.576002 11.6652983 14.1186858 11.1239516 14.663127 10.5808518 15 9.82976635 15 9 15 7.34314575 13.6568542 6 12 6 11.1040834 6 10.2998929 6.39272604 9.75018919 7.01541737 9.49601109 7.30334431 9.29624369 7.64043912 9.16697781 8.01061095",
    "M12 17L12.01 17",
    circle!("M22 12A10 10 0 1 1 2 12A10 10 0 1 1 22 12Z"),
]);
pub const PLUS: Icon = stroked(&["M20 12L4 12M12 4L12 20"]);
pub const CLOSE: Icon = stroked(&["M6.34314575 6.34314575L17.6568542 17.6568542M6.34314575 17.6568542L17.6568542 6.34314575"]);
pub const BIN: Icon = Icon {
    paths: &["M19 6L5 6M14 5L10 5M6 10L6 20C6 20.6666667 6.33333333 21 7 21 7.66666667 21 11 21 17 21 17.6666667 21 18 20.6666667 18 20 18 19.3333333 18 16 18 10"],
    fill: true,
};

pub const CODE_ALT: Icon = stroked(&["M7 8L3 12L7 16", "M17 16L21 12L17 8", "M9 19.5L14.5 5"]);
pub const MAXIMISE: Icon = stroked(&["M21 16L21 21L16 21", "M8 21L3 21L3 16", "M16 3L21 3L21 8", "M3 8L3 3L8 3"]);

pub const ROTATE: Icon = stroked(&[
    "M22 12l-3 3-3-3",
    "M2 12l3-3 3 3",
    "M19.016 14v-1.95A7.05 7.05 0 0 0 8 6.22",
    "M16.016 17.845A7.05 7.05 0 0 1 5 12.015V10",
    "M5 10V9",
    "M19 15v-1",
]);
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_icon_parses() {
        for icon in [COPY, TOOL, LAYERS, LIST, CONTROLS_ALT, CAMERA, CLIPBOARD, SETTINGS, VIDEO, EYE, EYE_CROSSED, ARROW_LEFT, ARROW_RIGHT, STAR, CURSOR, PLUS, CLOSE, BIN, CODE_ALT, MAXIMISE, ROTATE] {
            for d in icon.paths {
                threers::curves::parse_svg_subpaths(d).unwrap_or_else(|e| panic!("{d}: {e}"));
            }
        }
        // The `<circle>`s, in the form painters recognize.
        let circles: Vec<(&str, (f32, f32, f32))> =
            [CONTROLS_ALT, CAMERA, SETTINGS, VIDEO, EYE, EYE_CROSSED].iter().flat_map(|i| i.paths.iter()).filter_map(|d| Some((*d, threers::ui::parse_circle(d)?))).collect();
        assert_eq!(circles.len(), 8);
        for (d, (cx, cy, r)) in circles {
            assert_eq!(threers::ui::circle_path(cx, cy, r), d);
        }
    }
}
