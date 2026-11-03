//! The data panels and their layouts (`src/data_panel_layout.ts`).

use super::*;

/// A data panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Slice(NamedAxes),
    Perspective,
}

/// neuroglancer's data panel layouts (`LAYOUTS`, `src/data_panel_layout.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// xy and xz above, 3d and yz below (`FourPanelLayout`).
    FourPanel,
    /// xy and yz above, xz and 3d below (`FourPanelAltLayout`).
    FourPanelAlt,
    /// The 3D view alone (`SinglePerspectiveLayout`).
    Perspective,
    /// One cross-section (`SinglePanelLayout`).
    Single(NamedAxes),
    /// A cross-section left of the 3D view (`SliceViewPerspectiveTwoPanelLayout`, row).
    SliceAnd3d(NamedAxes),
}

impl Layout {
    pub const ALL: [Layout; 9] = [
        Layout::FourPanel,
        Layout::FourPanelAlt,
        Layout::Perspective,
        Layout::Single(NamedAxes::Xy),
        Layout::Single(NamedAxes::Xz),
        Layout::Single(NamedAxes::Yz),
        Layout::SliceAnd3d(NamedAxes::Xy),
        Layout::SliceAnd3d(NamedAxes::Xz),
        Layout::SliceAnd3d(NamedAxes::Yz),
    ];

    /// The state's name for it.
    pub fn name(self) -> String {
        let axes = |a: NamedAxes| match a {
            NamedAxes::Xy => "xy",
            NamedAxes::Xz => "xz",
            NamedAxes::Yz => "yz",
        };
        match self {
            Layout::FourPanel => "4panel".into(),
            Layout::FourPanelAlt => "4panel-alt".into(),
            Layout::Perspective => "3d".into(),
            Layout::Single(a) => axes(a).into(),
            Layout::SliceAnd3d(a) => format!("{}-3d", axes(a)),
        }
    }

    /// `getLayoutByName`.
    pub fn from_name(name: &str) -> Option<Layout> {
        Layout::ALL.into_iter().find(|l| l.name() == name)
    }

    /// The panels and their rectangles in a `width` × `height` area (pixels,
    /// top-left origin). Panels split the area evenly (`flex: 1`); with
    /// `borders` (neuroglancer's `showPanelBorders`, on by default) each has a
    /// 2 px border and draws inside it. Rounded as `RenderedDataPanel` does:
    /// the left/top edge `Math.round`ed, the size the element's integer
    /// `clientWidth`/`clientHeight`, clipped to the area.
    pub fn panels_with(self, width: u32, height: u32, borders: bool) -> Vec<(Panel, PixelRect)> {
        self.panels_scaled(width, height, if borders { 1.0 } else { 0.0 })
    }

    /// [`panels_with`](Self::panels_with) borders, in an area of physical
    /// pixels `scale` to a CSS pixel: the 2 px borders are `2 × scale`
    /// physical pixels (`scale` 0 for none).
    pub fn panels_scaled(self, width: u32, height: u32, scale: f64) -> Vec<(Panel, PixelRect)> {
        let inset = 2.0 * scale;
        let cell = |col: u32, row: u32, cols: u32, rows: u32| {
            let (w, h) = (width as f64 / cols as f64, height as f64 / rows as f64);
            let x = ng_core::js::round(col as f64 * w + inset);
            let y = ng_core::js::round(row as f64 * h + inset);
            let cw = ng_core::js::round(w - 2.0 * inset).max(1.0);
            let ch = ng_core::js::round(h - 2.0 * inset).max(1.0);
            PixelRect {
                x: x as u32,
                y: y as u32,
                width: (cw.min(width as f64 - x)).max(1.0) as u32,
                height: (ch.min(height as f64 - y)).max(1.0) as u32,
            }
        };
        self.cells().into_iter().map(|(p, [c, r, cs, rs])| (p, cell(c, r, cs, rs))).collect()
    }

    /// Each panel's cell: column, row, and the grid's columns and rows
    /// (panels share the area evenly, `flex: 1`).
    pub fn cells(self) -> Vec<(Panel, [u32; 4])> {
        let s = Panel::Slice;
        match self {
            Layout::FourPanel => vec![
                (s(NamedAxes::Xy), [0, 0, 2, 2]),
                (s(NamedAxes::Xz), [1, 0, 2, 2]),
                (Panel::Perspective, [0, 1, 2, 2]),
                (s(NamedAxes::Yz), [1, 1, 2, 2]),
            ],
            Layout::FourPanelAlt => vec![
                (s(NamedAxes::Xy), [0, 0, 2, 2]),
                (s(NamedAxes::Yz), [1, 0, 2, 2]),
                (s(NamedAxes::Xz), [0, 1, 2, 2]),
                (Panel::Perspective, [1, 1, 2, 2]),
            ],
            Layout::Perspective => vec![(Panel::Perspective, [0, 0, 1, 1])],
            Layout::Single(a) => vec![(s(a), [0, 0, 1, 1])],
            Layout::SliceAnd3d(a) => vec![(s(a), [0, 0, 2, 1]), (Panel::Perspective, [1, 0, 2, 1])],
        }
    }

    /// [`panels_with`](Self::panels_with) panel borders.
    pub fn panels(self, width: u32, height: u32) -> Vec<(Panel, PixelRect)> {
        self.panels_with(width, height, true)
    }

    /// The rectangle of `panel`, if the layout has it.
    pub fn rect(self, panel: Panel, width: u32, height: u32) -> Option<PixelRect> {
        self.panels(width, height).into_iter().find(|(p, _)| *p == panel).map(|(_, r)| r)
    }

    /// The panel containing `(x, y)`, if any (not a gap).
    pub fn panel_at(self, x: f64, y: f64, width: u32, height: u32) -> Option<Panel> {
        self.panels(width, height).into_iter().find_map(|(p, r)| {
            (x >= r.x as f64 && x < (r.x + r.width) as f64 && y >= r.y as f64 && y < (r.y + r.height) as f64).then_some(p)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_tile_the_target_without_overlap() {
        let (w, h) = (800, 600);
        for layout in Layout::ALL {
            assert_eq!(Layout::from_name(&layout.name()), Some(layout));
            let panels = layout.panels(w, h);
            for (i, &(p, r)) in panels.iter().enumerate() {
                assert!(r.x + r.width <= w && r.y + r.height <= h, "{layout:?} {p:?} {r:?}");
                let (cx, cy) = (r.x + r.width / 2, r.y + r.height / 2);
                assert_eq!(layout.panel_at(cx as f64, cy as f64, w, h), Some(p));
                for &(_, o) in &panels[i + 1..] {
                    let apart = r.x + r.width <= o.x || o.x + o.width <= r.x || r.y + r.height <= o.y || o.y + o.height <= r.y;
                    assert!(apart, "{layout:?}: {r:?} overlaps {o:?}");
                }
            }
        }
        assert_eq!(Layout::from_name("xy-3d"), Some(Layout::SliceAnd3d(NamedAxes::Xy)));
        assert_eq!(Layout::from_name("4panel-alt").unwrap().rect(Panel::Perspective, w, h).unwrap().x, 402);
        assert_eq!(Layout::from_name("bogus"), None);
    }

    /// Panel rectangles as neuroglancer computes them from its layout's
    /// elements (measured in Chrome: element boxes, `clientLeft`,
    /// `clientWidth`).
    #[test]
    fn panel_geometry_matches_neuroglancer() {
        let r = |x, y, width, height| PixelRect { x, y, width, height };
        let four = Layout::FourPanel;
        assert_eq!(four.rect(Panel::Slice(NamedAxes::Xz), 800, 600), Some(r(402, 2, 396, 296)));
        assert_eq!(four.rect(Panel::Slice(NamedAxes::Yz), 1000, 700), Some(r(502, 352, 496, 346)));
        // 801 × 599: elements 400.5 × 299.5, so x = round(402.5) = 403, and
        // the sizes round half up (396.5 → 397, 295.5 → 296).
        assert_eq!(four.rect(Panel::Slice(NamedAxes::Xz), 801, 599), Some(r(403, 2, 397, 296)));
        assert_eq!(four.rect(Panel::Perspective, 801, 599), Some(r(2, 302, 397, 296)));
        assert_eq!(Layout::SliceAnd3d(NamedAxes::Xy).rect(Panel::Perspective, 801, 599), Some(r(403, 2, 397, 595)));
        assert_eq!(Layout::Single(NamedAxes::Xy).rect(Panel::Slice(NamedAxes::Xy), 801, 599), Some(r(2, 2, 797, 595)));
        // Without borders: the element boxes themselves.
        assert_eq!(four.panels_with(800, 600, false)[1].1, r(400, 0, 400, 300));
    }
}
