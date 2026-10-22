# Reference comparisons against neuroglancer

Renders viewer states in real neuroglancer and in neuroglancer-rs, both in
headless Chrome at 800×600, and compares the screenshots pixel by pixel.

```sh
scripts/reference/setup.sh              # build neuroglancer (in /tmp) + ng-web; serve both and the data
(cd scripts/reference && npm i)         # playwright-core
python3 scripts/reference/run.py        # every state in states/; WEBGL=1 for the WebGL2 fallback
UI=1 python3 scripts/reference/run.py   # whole pages, both UIs, with the input in ui.json
UI=1 python3 scripts/reference/run.py /tmp/ng-reference real_fib25_xy real_fib25_4panel real_hemibrain_4panel   # real data from GCS
python3 scripts/reference/diff.py a.png b.png [tol] [x0,y0,x1,y1 ...]
python3 scripts/reference/analyze.py <scenario> [--dom capture.txt]   # what the differences are
python3 scripts/reference/analyze.py --all            # every scenario, totals by kind
python3 scripts/reference/positions.py <scenario>     # text runs where neuroglancer has them?
```

Finding out why a scenario differs:

- `analyze.py` groups the differing pixels into regions and classifies each
  by what would make it match — a whole-pixel `shift`, a `subpixel` one, a
  uniform `colour` offset, something `missing` or `extra`, `antialias`
  along edges, else `content` — with a report image beside the screenshots
  (`<scenario>-report.png`: both, the differences, each region magnified).
  With `--dom` (the output of `FULL=1 node capture.mjs states/x.json body 80`)
  each region is named by neuroglancer's element under it. `run.py` runs it
  on every failure.
- `positions.py` pairs every text run of neuroglancer's DOM with ours at full
  precision (`FULL=1` makes `capture.mjs` and `rsrects.mjs` print unrounded
  numbers). Chrome draws glyphs at quarter-pixel positions (`floor(4x + ½)`),
  so a run a hair off can land a quarter pixel over and rasterize
  differently even though layout "matches": those are flagged `QUARTER`.
  Text that differs but sits at the same position (`ok`) differs for another
  reason (shaping: runs Blink shapes apart, as `::before` content, must be
  separate runs here too).

- neuroglancer is built from a copy of `../neuroglancer`; the checkout is not
  touched. Its UI controls are hidden (panel borders kept, as by default);
  the canvas is what is compared.
- A scenario with `"ready": true` (the `real_*` ones, public data over the
  network) is screenshotted once the viewer has its data in: neuroglancer's
  `viewer.isReady()`, ng-web's `isReady()` (up to 90 s), not after a fixed wait.
- `states/*.json` reference data on the same origin
  (`http://127.0.0.1:8792/data/…` for `crates/ng-viewer/tests/fixtures`,
  `/testdata/…` for neuroglancer's test data).
- `ui.json` (with `UI=1`): whole-page scenarios at 1200×800 — a state, and
  `ACTIONS` for `compare.mjs` (`move:x,y`, `click:x,y`, `dblclick:x,y`,
  `key:Enter`, `wheel:x,y,dy`) — each with the pixels allowed to differ by more
  than 8 levels and why. neuroglancer-rs is opened with `?ui=0` otherwise.
- `states/index.json` gives each state's RGB tolerance, rectangles to skip
  (the 3D panel, a stand-in so far) and how many pixels may differ. Allowed
  differences are ties: a voxel boundary within float noise of a pixel centre,
  where neuroglancer's interpolated varying and ours round to different sides.
