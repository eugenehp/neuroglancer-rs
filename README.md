# neuroglancer-rs

A Rust port of [neuroglancer](https://github.com/google/neuroglancer), the
viewer for very large volumetric data: electron microscopy, connectomics
segmentations, light microscopy, MRI. It reads the same data sources, opens
the same viewer states and share links, and draws the same pixels. The
browser build is checked against neuroglancer itself, page for page, in
headless Chrome.

It runs natively (wgpu on Metal, Vulkan or DX12) and in the browser
(WebAssembly, WebGPU with a WebGL2 fallback). It is also a library: the whole
viewer, with or without its UI, renders into any wgpu texture you hand it.

By **Eugene Hauptmann**, © 2026, under MIT OR Apache-2.0:
<https://github.com/eugenehp/neuroglancer-rs>.

- **Live:** <https://eugenehp.github.io/neuroglancer-rs/>. See
  [Examples](#examples) for public datasets to open in it.
- **What is ported, and how each part is checked:** [PARITY.md](PARITY.md).

## Contents

- [Examples](#examples)
- [Running it](#running-it)
- [What it shows](#what-it-shows)
- [Data it reads](#data-it-reads)
- [Controls](#controls)
- [Loading: streaming, caching, parallelism](#loading-streaming-caching-parallelism)
- [Using it as a library](#using-it-as-a-library)
- [How it is built](#how-it-is-built)
- [Parity with neuroglancer](#parity-with-neuroglancer)
- [Development](#development)
- [Deployment](#deployment)
- [Platforms](#platforms)
- [License](#license)

## Examples

Public datasets, opened in the [live viewer](https://eugenehp.github.io/neuroglancer-rs/).
Each link carries a full viewer state in its `#!` hash, as neuroglancer's
share links do; edit the state in the JSON editor (`{}` in the top bar) or
copy the address bar to share where you are.

| Dataset | What you see | Format |
|---|---|---|
| [FIB-25 (fly medulla, Janelia FlyEM)](https://eugenehp.github.io/neuroglancer-rs/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B8e-09%2C%22m%22%5D%2C%22y%22%3A%5B8e-09%2C%22m%22%5D%2C%22z%22%3A%5B8e-09%2C%22m%22%5D%7D%2C%22position%22%3A%5B2914.5%2C3088.5%2C4045.5%5D%2C%22crossSectionScale%22%3A3.76%2C%22projectionScale%22%3A4096%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22name%22%3A%22image%22%2C%22source%22%3A%22precomputed%3A%2F%2Fgs%3A%2F%2Fneuroglancer-public-data%2Fflyem_fib-25%2Fimage%22%7D%2C%7B%22type%22%3A%22segmentation%22%2C%22name%22%3A%22ground-truth%22%2C%22source%22%3A%22precomputed%3A%2F%2Fgs%3A%2F%2Fneuroglancer-public-data%2Fflyem_fib-25%2Fground_truth%22%2C%22segments%22%3A%5B%22158571%22%2C%2221894%22%2C%2222060%22%2C%2224436%22%2C%222515%22%5D%7D%5D%2C%22layout%22%3A%224panel%22%7D) | EM with its ground-truth segmentation, five neurons with meshes in 3D, 4-panel | precomputed: JPEG image, compressed segmentation, legacy meshes |
| [Hemibrain v1.2 (fly brain, Janelia FlyEM)](https://eugenehp.github.io/neuroglancer-rs/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B8e-09%2C%22m%22%5D%2C%22y%22%3A%5B8e-09%2C%22m%22%5D%2C%22z%22%3A%5B8e-09%2C%22m%22%5D%7D%2C%22position%22%3A%5B17216.5%2C19776.5%2C20704.5%5D%2C%22crossSectionScale%22%3A4%2C%22projectionScale%22%3A40000%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22name%22%3A%22emdata%22%2C%22source%22%3A%22precomputed%3A%2F%2Fgs%3A%2F%2Fneuroglancer-janelia-flyem-hemibrain%2Femdata%2Fclahe_yz%2Fjpeg%22%7D%2C%7B%22type%22%3A%22segmentation%22%2C%22name%22%3A%22segmentation%22%2C%22source%22%3A%22precomputed%3A%2F%2Fgs%3A%2F%2Fneuroglancer-janelia-flyem-hemibrain%2Fv1.2%2Fsegmentation%22%2C%22segments%22%3A%5B%221001453586%22%2C%221002360103%22%5D%7D%5D%2C%22layout%22%3A%224panel%22%7D) | EM and the segmentation, two neurons in 3D | precomputed, sharded: JPEG, compressed segmentation, multi-resolution Draco meshes |
| [Kasthuri 2011 (mouse cortex)](https://eugenehp.github.io/neuroglancer-rs/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B6.000000000000001e-09%2C%22m%22%5D%2C%22y%22%3A%5B6.000000000000001e-09%2C%22m%22%5D%2C%22z%22%3A%5B3.0000000000000004e-08%2C%22m%22%5D%7D%2C%22position%22%3A%5B5523.5%2C8538.5%2C1198.5%5D%2C%22crossSectionScale%22%3A3%2C%22projectionScale%22%3A30000%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22name%22%3A%22image%22%2C%22source%22%3A%22precomputed%3A%2F%2Fgs%3A%2F%2Fneuroglancer-public-data%2Fkasthuri2011%2Fimage%22%7D%2C%7B%22type%22%3A%22segmentation%22%2C%22name%22%3A%22ground_truth%22%2C%22source%22%3A%22precomputed%3A%2F%2Fgs%3A%2F%2Fneuroglancer-public-data%2Fkasthuri2011%2Fground_truth%22%2C%22selectedAlpha%22%3A0.63%2C%22notSelectedAlpha%22%3A0.14%7D%5D%2C%22layout%22%3A%22xy-3d%22%7D) | EM with the dense ground-truth segmentation, xy + 3D | precomputed: JPEG image, compressed segmentation |
| [FAFB v14 (whole adult fly brain)](https://eugenehp.github.io/neuroglancer-rs/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B4e-09%2C%22m%22%5D%2C%22y%22%3A%5B4e-09%2C%22m%22%5D%2C%22z%22%3A%5B4e-08%2C%22m%22%5D%7D%2C%22position%22%3A%5B124416.5%2C67072.5%2C3531.5%5D%2C%22crossSectionScale%22%3A4%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22name%22%3A%22fafb_v14%22%2C%22source%22%3A%22precomputed%3A%2F%2Fgs%3A%2F%2Fneuroglancer-fafb-data%2Ffafb_v14%2Ffafb_v14_orig%22%7D%5D%2C%22layout%22%3A%22xy%22%7D) | EM, one cross-section | precomputed: JPEG, 9 scales |
| [H01 (human cortex, Google/Harvard)](https://eugenehp.github.io/neuroglancer-rs/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B4e-09%2C%22m%22%5D%2C%22y%22%3A%5B4e-09%2C%22m%22%5D%2C%22z%22%3A%5B3.3000000000000004e-08%2C%22m%22%5D%7D%2C%22position%22%3A%5B515892.5%2C356400.5%2C2646.5%5D%2C%22crossSectionScale%22%3A6%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22name%22%3A%224nm%20raw%22%2C%22source%22%3A%22precomputed%3A%2F%2Fgs%3A%2F%2Fh01-release%2Fdata%2F20210601%2F4nm_raw%22%7D%2C%7B%22type%22%3A%22segmentation%22%2C%22name%22%3A%22c3%22%2C%22source%22%3A%22precomputed%3A%2F%2Fgs%3A%2F%2Fh01-release%2Fdata%2F20210601%2Fc3%22%7D%5D%2C%22layout%22%3A%22xy%22%7D) | EM with the c3 segmentation | precomputed, sharded: JPEG image, compressed segmentation |

A single data source needs no state: `?url=` opens it centred, with
neuroglancer's defaults, for example
<https://eugenehp.github.io/neuroglancer-rs/?url=precomputed://gs://neuroglancer-public-data/flyem_fib-25/image>.
Any URL neuroglancer accepts works, including `zarr://…`, `n5://…` and
`nifti://…` sources on servers that allow cross-origin requests.

To open a state of your own, put its JSON after `#!` (URL-encoded), or a
neuroglancer share link's part after `#!`:

```
https://eugenehp.github.io/neuroglancer-rs/#!{"dimensions":{"x":[8e-9,"m"],"y":[8e-9,"m"],"z":[8e-9,"m"]},"layers":[{"type":"image","source":"precomputed://gs://neuroglancer-public-data/flyem_fib-25/image","name":"image"}],"layout":"xy"}
```

## Running it

### In the browser

Open the [live viewer](https://eugenehp.github.io/neuroglancer-rs/) with
either

- `#!{…}`: a neuroglancer viewer state, exactly as in neuroglancer's share
  links (also `#!gs://bucket/state.json`, a state stored in a bucket), or
- `?url=<data source>`: one data source, any URL neuroglancer accepts.

The URL hash follows the state as you navigate, as in neuroglancer, so the
address bar is always a share link, and editing the hash restores that
state. A neuroglancer share link opens here once its host is swapped.

Page options:

| Parameter | Effect |
|---|---|
| `?ui=0` | The panels alone, without neuroglancer's UI (its `showUIControls` off). |
| `?workers=0` | Decode chunks on the page instead of in Web Workers. |
| `?cache=1` | Keep downloads in the browser's Cache Storage, so a dataset seen before loads without the network (HTTPS or localhost only). |
| `?credits=0` | Without the port's credits (the footer, and the About section in Settings). |
| `?domtext=0` / `?domtext=1` | UI text all on the canvas / all as DOM (default: translucent text as DOM, as Chrome draws a page's). |

WebGPU needs a secure context, so over plain HTTP from another machine the
browser falls back to WebGL2. Data must be served with CORS, as for
neuroglancer (public buckets are).

### Natively

```sh
# Any data source neuroglancer accepts, or a local path or kvstore URL:
cargo run --release -p ng-native -- 'precomputed://gs://neuroglancer-public-data/flyem_fib-25/image'
cargo run --release -p ng-native -- ../neuroglancer/testdata/datasource/zarr/ome_zarr/simple_0.5
cargo run --release -p ng-native -- data/brain.nii.gz

# A viewer state: a share link (…#!{…} or #!gs://…), its JSON, or a file holding it
cargo run --release -p ng-native -- --state 'https://neuroglancer-demo.appspot.com/#!{"layers":[…]}'

# Keep downloads on disk, so a dataset seen before loads from there
cargo run --release -p ng-native -- --cache ~/.cache/neuroglancer-rs --state '…'
```

The window shows the same UI as the browser. The clipboard goes through the
platform's copy command, links open in the system browser, colours are
picked with the UI's own picker, and screenshots and downloads are saved in
the working directory.

### A local browser build

```sh
scripts/build-web.sh                         # into crates/ng-web/www/pkg; PROFILE=release-wasm for an optimized build
python3 -m http.server -d crates/ng-web/www 8080
# then http://localhost:8080/?url=…  or  http://localhost:8080/#!{…}
```

`build-web.sh` needs the `wasm32-unknown-unknown` target and a `wasm-bindgen`
CLI of the same version as the `wasm-bindgen` crate in `Cargo.lock`
(`cargo install wasm-bindgen-cli --version <that version>`).

## What it shows

**Layouts.** All of neuroglancer's data panel layouts:
- `4panel` and `4panel-alt` (xy, xz, yz and 3D);
- single cross-sections `xy`, `xz`, `yz`;
- `3d`;
- the `xy-3d`, `xz-3d` and `yz-3d` pairs.

Any orientation works. Each panel has its own layout buttons, axis lines,
scale bar and data bounds.

**Layers.**

- **Image.**
  - neuroglancer's default shader: `invlerp` over the data type's range,
    with its histogram widget and range finder.
  - User shaders in neuroglancer's GLSL dialect, with `#uicontrol` controls
    (sliders, colours, checkboxes, `invlerp`, transfer functions), translated
    to WGSL with naga.
  - Multiple channels.
  - Volume rendering in 3D: max, min, or the shader's emitted colours.
- **Segmentation.**
  - Segments coloured by neuroglancer's hash (same seed, same colours).
  - Selected and visible segment sets, `selectedAlpha`/`notSelectedAlpha`,
    saturation, hovering, and double-click to show or hide a segment.
  - In 3D: meshes (legacy precomputed and multi-resolution Draco, sharded or
    not) and skeletons, optionally transparent.
- **Annotation.**
  - Local annotations (points, lines, boxes, ellipsoids, polylines) and
    precomputed annotation sources (spatially indexed, sharded, `by_id`).
  - Annotation shaders and properties.
  - Drawing, hovering, selecting, moving and deleting.

**UI.** neuroglancer's interface, drawn by the port itself on
`threers::ui`:
- the position widget (type a coordinate, scroll it, pick from its plot);
- the layer bar;
- the layer side panels, with Source, Rendering, Seg. and Annotations tabs;
- the Settings, Help, Layers and Selection panels, docked and resizable as
  the state says;
- tool key bindings and tool palettes, and the command palette;
- the JSON state editor, the screenshot dialog and the chunk statistics
  panel;
- adding layers, with URL completion.

Not there yet:
- other datasources (graphene, DVID, BOSS, …);
- compresso, JPEG XL and crackle chunks;
- linked navigation between layer groups;
- parts of the panels above.

[PARITY.md](PARITY.md) has the details.

## Data it reads

| Format | Notes |
|---|---|
| Neuroglancer precomputed | Volumes (raw, `compressed_segmentation`, JPEG, PNG), sharded or not; legacy and multi-resolution meshes; skeletons; annotations. |
| Zarr v2 / v3 | All of neuroglancer's codecs, sharding (sub-chunks read by byte range), OME-NGFF 0.4–0.6 multiscales. |
| N5 | Attributes merged up the directory tree; raw, gzip, zlib, blosc and zstd blocks. |
| NIfTI-1 / NIfTI-2 | `.nii`, `.nii.gz`; voxel types kept exact (labels stay labels). |
| Deep Zoom | `.dzi` pyramids of PNG or JPEG tiles. |

The bytes can come from:
- local files;
- HTTP(S);
- Google Cloud Storage: `gs://` through its JSON API, and `gs+xml://`;
- S3: `s3://`;
- inside `zip` or `gzip` archives.

Data source URLs use neuroglancer's syntax:
- `precomputed://gs://bucket/path`;
- `zarr://https://host/array.zarr`, or `zarr2://…` / `zarr3://…` for a
  specific version;
- `n5://…` and `nifti://…`;
- a pipeline, such as `https://host/data.zip|zip:inner/path|zarr:`.

A plain path or URL is probed for its format.

## Controls

neuroglancer's default bindings (the Help panel, `h`, lists them all):

| Input | Action |
|---|---|
| Drag | Move in the plane |
| Shift-drag | Rotate |
| Wheel | Step through slices (Shift: by 10) |
| Ctrl/⌘-wheel | Zoom about the pointer |
| Right-click | Move to the pointer |
| Ctrl/⌘-right-click | Select the position (Selection panel) |
| Double-click a segment | Show/hide it |
| Arrows, `,` `.` | Step; `r` `e` and Shift-arrows rotate; `z` snaps to the axes |
| `1`–`9` | Toggle layers |
| `space` | Cycle layouts |
| `a` `s` `b` `v` `o` | Toggle axis lines, slices in 3D, scale bar, default annotations, orthographic 3D |
| `\` | Chunk statistics |
| Ctrl/⌘+P | Command palette |

On a Mac, ⌘ works wherever neuroglancer binds Ctrl.

## Loading: streaming, caching, parallelism

Data arrives in chunks. Every frame, the viewer asks for the chunks each
panel needs and shows each one as soon as it arrives.

- **Priorities.** Within a panel, finer scales load first, and within a
  scale the chunks nearer the centre. Coarser scales fill the gaps
  meanwhile; this is neuroglancer's ordering. At most 100 downloads are in
  flight (`concurrentDownloads`). Decoded chunks are kept up to
  `systemMemoryLimit` (2 GB), least recently used dropped first.
- **Prefetch.** While the position moves, the viewer estimates its velocity
  and loads the chunks likely needed within two seconds, after every visible
  chunk. This is neuroglancer's `VelocityEstimator` and
  `getPrefetchChunkOffsets`; the `prefetch` setting turns it off.
- **Sharded data.** A minishard index is read once, however many chunks share
  it, and kept (up to 8192 indexes).
- **In parallel.**
  - In the browser, JPEG, PNG and compressed-segmentation chunks decode in
    Web Workers (one per core, up to 8) running the same wasm module.
  - Natively, downloads wait on their own I/O threads through one keep-alive
    HTTP client, while a thread pool decodes.
- **Caching across sessions**, opt-in (neuroglancer keeps no such cache):
  - natively, a disk cache: `--cache <dir>` or `NG_CACHE_DIR`, 10 GB, oldest
    dropped first;
  - in the browser, Cache Storage: `?cache=1`.

  "Not found" answers are never cached. Data changed on the server since it
  was cached is shown as cached.

For a sense of scale, the hemibrain 4-panel view (sharded JPEG and
segmentation over the network) is ready in:

| | Network | Cache |
|---|---|---|
| Browser | about 3.5 s (as in neuroglancer) | about 1 s |
| Native | about 1.2 s | about 0.3 s |

## Using it as a library

The `neuroglancer` crate puts the viewer behind one platform-neutral
interface. You own the GPU device, the window or canvas, and the event loop.
The engine takes input in DOM terms and renders frames into a texture view.

```rust
use std::sync::Arc;
use neuroglancer::prelude::*;

let mut engine = Engine::new(EngineOptions {
    device, queue,                       // your wgpu device and queue
    format: wgpu::TextureFormat::Bgra8Unorm,
    width, height, scale_factor,
    executor: Arc::new(ThreadPool::new(8)), // `LocalExecutor` on wasm
    wake: Arc::new(|| request_redraw()), // background work finished
    ui: true,                            // neuroglancer's UI, or the panels alone
    now: clock_ms,                       // milliseconds, monotonic
});
engine.restore_url("https://neuroglancer-demo.appspot.com/#!{…}");

// For each input event (CSS pixels, DOM semantics):
let response = engine.pointer_move(x, y);
for request in response.requests { /* clipboard, open a link, save a file, pick a colour */ }
if response.redraw { request_redraw(); }

// For each frame:
let frame = engine.draw_frame(&texture_view, Some(&mut my_ui_canvas), || timestamp());
if let Some(shot) = frame.screenshot { save(&shot.name, &shot.png); }
if frame.animate { request_redraw(); }
```

- **`UiCanvas`** is where the UI paints. Implement it over any
  `threers::ui::Painter` (a CPU overlay, a 2D canvas), or pass `None` for the
  panels alone.
- **`Request`s** are what only the host can do: the clipboard, opening links,
  saving files, picking a colour.
- **Ready-made hosts:**
  - `neuroglancer::winit` (feature `winit`): a desktop window,
    `run(open, options)`.
  - `neuroglancer::web` (feature `web`, wasm32):
    `start(WebOptions { canvas, … })` on a full-window canvas or your own,
    and `with_engine` to reach the engine afterwards.
- **Your own adapter:** call
  `neuroglancer::render::capabilities::set_from_adapter(&adapter)` before
  `Engine::new`, so a downlevel GPU (WebGL2: no compute shaders, no
  independent blending) gets the fallback passes. The ready-made hosts do it.
- **The layers underneath** are re-exported for finer control: `viewer` (the
  viewer without UI), `render`, `datasource`, `kvstore`, `chunk`, `core`,
  `ui`, and `threers`.

`crates/neuroglancer/tests/engine.rs` drives the engine headless, as a
downstream crate would, and checks that its frames are byte-identical to the
viewer's own.

### Scene mode: volumes inside another renderer's 3D scene

An application with its own 3D scene (a brain atlas, a game engine) can have
neuroglancer's 3D content drawn from *its* camera and composite it into its
frame by depth. `ng_viewer::Viewer::render_scene` runs the normal frame with
three changes:

- the three cross-sections are drawn offscreen at a fixed size and shown as
  planes in 3D, as the `4panel` layout's 3D panel shows them;
- the 3D panel is the host's size and uses the host's camera
  (`scene::HostCamera`: its view and GL projection, and an affine from the
  host's world to physical metres of the displayed dimensions), so chunks,
  meshes and volume rendering are chosen for that view, and the planes are lit
  from it;
- nothing is composited into a layout or UI, and the slices' overlays
  (annotations, axis lines, scale bar), which only show in a layout, are not
  drawn.

The slices are drawn again only when what they show changes (position, scale,
a chunk arriving, a shader or segment set), so a host camera orbiting them
costs only the 3D panel (`slice_redraws()` counts the redraws).

```rust
use ng_viewer::scene::{HostCamera, SceneView};
use ng_viewer::{SceneObjects, Viewer};

let mut viewer = Viewer::new_scene(device, queue, executor, wake, SceneObjects::default());
// `projection` with WebGPU's depth range (0..1); `HostCamera { .. }` takes a GL one
let camera = HostCamera::webgpu(view, projection, physical_from_world);
// `bounds` (host world centre, radius): near/far fitted to it, for volume rendering
viewer.render_scene(SceneView { width, height, slice_size: 1024, camera, bounds: Some((centre, radius)) });
// drawn over the host's frame where nearer than its depth (or `on_top`)
viewer.composite_scene(&frame_view, frame_format, &host_depth_view, samples, false);
// or read it yourself: `scene_output()` — Rgba8Unorm premultiplied and
// Depth32Float, both upside down; `scene_depth_ranges()` the near/far used
```

`SceneObjects` tunes segments' objects for many neurons: skeletons thinned to
a shared edge budget, meshes coarsened or capped past a count, or skeletons
left to the host (`skeletons_drawn: false`, then
`scene_segment_skeleton_into` gives each in the host's frame as float32, and
`scene_segments_revision()` says when anything a host batches changed).

Navigation (where the slices cut, the cross-section scale), layers and
shaders stay the viewer's, set through its state. Two calls help a host's own
input:
- `scene_probe(physical_from_world, origin, direction, slice_size)`: where a
  ray from the host's camera meets the nearest cross-section within its field
  of view, and each layer's value there. That's what a hover or click on the
  slices means, since a host's own surfaces are hit first.
- `scene_position` / `scene_set_position`: where the cross-sections cross,
  physical metres;
- `toggle_segment` / `segment_visible`: a segment shown or hidden (its mesh
  in 3D), as a double-click does in the UI;
- `layer_index(name)`, `layer_progress()` (chunks wanted / ready /
  downloading / failed per layer), `ManagedLayer::live_settings()` (what
  `set_layer_option` changes in place), `set_zoom_gate(layer, max)` (a layer
  shown only at that cross-section scale or finer; `spec.visible` stays the
  reader's);
- `adaptive_downloads = Some(ng_chunk::Adaptive::new(start, min, max))`:
  the downloads at once tuned as they download, backing off when the server
  pushes back (`retried()`).

Volume rendering works from the host's camera too. It samples the
projection's whole near..far range, which `SceneView::bounds` fits to the
data (`composite_scene` converts the depth back). Without the UI, an embedder
uses `ng_viewer::host::Loader` for what `Engine` does in the background:
restore a state, fetch its sources, hand them over in `poll`. The `neuroglancer`
crate's `Engine` uses the same `Loader`.

`crates/ng-viewer/tests/scene.rs` checks it on the GPU: the cross-section lands
at the host camera's projected depth (anisotropic voxels included), a moving
camera redraws no slice and comes back to the same pixels, and the composite
hides behind a nearer host surface.

**Data handed over directly.** `ng_kvstore::memory().insert(name, bytes)` puts
bytes at `memory://<name>`, a location like any other:
`nifti://memory://scan.nii`.

**In a Web Worker.** Downloads and the browser cache go through the global
scope's `fetch` and `caches`, page or worker, so the viewer runs in a worker
that renders to an `OffscreenCanvas`.

`cargo run -p ng-viewer --example bounds -- state.json [name=file …]` prints
where a state's layers land in global coordinates: the first thing to check
when placing data in someone else's scene.

## How it is built

```
ng-core ─────────────┐   coordinate spaces, navigation, viewer state, segment colours
ng-kvstore ──────────┤   file, HTTP, GCS, S3, zip/gzip; disk and browser caches
ng-datasource ───────┤   precomputed, Zarr, N5, NIfTI, Deep Zoom; decoding (on exg)
ng-chunk ────────────┤   the chunk queue: priorities, budgets, executors
ng-render ───────────┤   GPU passes on threers' device: slices, meshes, skeletons,
                     │   annotations, volume rendering, user shaders (GLSL → WGSL)
ng-viewer ───────────┘   the viewer: layers, layouts, input bindings, frames
ng-ui                    neuroglancer's UI on threers::ui, over a Viewer
neuroglancer             Engine (viewer + UI), the winit and web hosts, prelude
ng-native, ng-web        the desktop and browser apps
```

- **Rendering** uses [threers](https://crates.io/crates/threers) (wgpu).
  Cross-sections are drawn chunk by chunk into offscreen targets and
  composited per panel, as neuroglancer's `SliceView` does. The 3D panel is a
  scene holding the slices, meshes, skeletons and annotations, with
  order-independent transparency.
- **Decoding** is partly this repository's (JPEG, compressed segmentation,
  sharding, N5) and partly [exg](https://github.com/eugenehp/exg)'s
  (`exg-nifti`, and `exg-zarr`'s codecs).
- **Threads.**
  - Natively, a thread pool polls the chunk tasks while I/O threads wait on
    the network. The render thread only takes in finished chunks.
  - In the browser, everything is async on the page's event loop, and
    decoding goes to workers.
  - The same code serves both: the chunk queue runs on an `Executor`, and a
    decode is a job that either runs in place or is sent to a worker.
- **Layering.** The viewer crates (`ng-core` … `ng-viewer`) depend on nothing
  above them and work without the UI; `crates/ng-ui/tests/layering.rs`
  checks this.

## Parity with neuroglancer

Behaviour is ported from neuroglancer's TypeScript and checked against
neuroglancer itself, not against expectations written by hand.

- **Golden data from neuroglancer's own code.**
  - `scripts/golden/` runs neuroglancer's TypeScript sources in node
    (`ng_register.mjs` compiles them on import) and records what they
    compute: navigation, coordinate-space merging, transforms, segment
    colours, Zarr metadata, compressed segmentation, JPEG decoding, sharding
    keys and mesh fragments. The Rust tests compare against those files.
  - Python scripts write the sample datasets (sharded volumes, meshes,
    skeletons, annotations, N5, Deep Zoom) with the libraries neuroglancer's
    own tests use.
- **Pixels against neuroglancer.**
  - `scripts/reference/` opens the same state in real neuroglancer (built
    from a checkout) and in neuroglancer-rs, both in headless Chrome. It
    applies the same input (pointer, keys, wheel, typing) and compares the
    screenshots.
  - There are 144 page scenarios: layouts, every layer type, the side panels
    and dialogs, tools, annotations, meshes, volume rendering, and real
    public datasets (FIB-25, hemibrain) over the network. Each one passes
    within its own allowance, and each allowance's reason is recorded in
    `ui.json`. Of the 960,000 pixels per page, 43 scenarios differ in none
    and 106 in 300 or fewer, mostly anti-aliasing at glyph edges. The largest
    differences are a few thousand pixels, in text-heavy views: the
    selection panel, the screenshot dialog and the JSON editor.
  - `analyze.py` classifies the differences, and `positions.py` finds text
    runs placed a fraction of a pixel off.
- **Offscreen renders.** Native tests render datasets headless and check
  sampled pixels against voxel values, segment colours and transforms
  computed independently on the CPU.

```sh
UI=1 python3 scripts/reference/run.py /tmp/ng-reference            # all page scenarios
UI=1 python3 scripts/reference/run.py /tmp/ng-reference help_panel # one
python3 scripts/reference/analyze.py help_panel --dir /tmp/ng-reference
```

See `scripts/reference/README.md` for setting up the harness: `setup.sh`
builds neuroglancer and serves both viewers and the data.

## Development

Requirements:
- Rust 1.87 or later.
- For the browser: the `wasm32-unknown-unknown` target and the matching
  `wasm-bindgen` CLI.
- For the tests' data and goldens: a neuroglancer checkout next to this one
  (`../neuroglancer`).

```sh
cargo test --workspace                 # unit, golden and offscreen render tests (GPU tests skip without a GPU)
cargo test --workspace -- --ignored    # also public datasets over the network
cargo clippy --workspace --all-targets --features neuroglancer/winit
cargo clippy -p ng-web -p neuroglancer --features neuroglancer/web --target wasm32-unknown-unknown
```

Dev builds optimize every crate and keep debug info. Unoptimized decoding
and drawing makes network datasets several times slower to show.

To work on threers alongside, add `[patch.crates-io] threers = { path =
"../threers" }` to the workspace `Cargo.toml`. Its version must match the
requirement there, or cargo ignores the patch and quietly builds the
crates.io release.

Regenerating goldens:

```sh
cd scripts/golden && npm install && cd ../..
node scripts/golden/segment_colors.mts > crates/ng-core/tests/fixtures/segment_colors.json
node --import ./scripts/golden/ng_register.mjs scripts/golden/navigation.mts > crates/ng-core/tests/fixtures/navigation.json
node scripts/golden/nifti_info.mjs > crates/ng-datasource/tests/fixtures/nifti_info.json
node --import ./scripts/golden/ng_register.mjs scripts/golden/zarr_metadata.mts > crates/ng-datasource/tests/fixtures/zarr_metadata.json
node --import ./scripts/golden/ng_register.mjs scripts/golden/compressed_segmentation.mts
python3 scripts/golden/snapshots.py    # neuroglancer's own metadata snapshots → JSON
```

## Deployment

`.github/workflows/pages.yml` builds the browser viewer
(`PROFILE=release-wasm scripts/build-web.sh`) and publishes
`crates/ng-web/www` to GitHub Pages. It runs on every push to `main`, or by
hand (Actions → Pages → Run workflow). Everything it needs comes from this
repository and crates.io. Once per repository, set Settings → Pages → Source
to GitHub Actions.

## Platforms

| Target | Status |
|---|---|
| macOS (Metal) | In use; all tests pass on an Apple M4 Pro, headless render tests included. |
| Linux (Vulkan) | All tests pass on an NVIDIA RTX 3080 Ti (Ubuntu), headless render tests and the public datasets included. |
| Browser, WebGPU | Compared with neuroglancer in headless Chrome on macOS. Headless Chrome on Linux with NVIDIA fails to present any WebGPU canvas (a Chrome/driver issue: a minimal WebGPU page fails alike). |
| Browser, WebGL2 fallback | Compared with neuroglancer in headless Chrome on macOS (the rendering states, and the 3D, annotation-shader and picking pages). Without compute shaders or independent blending, the 3D panel splits its annotation passes and annotation shaders run as a render pass, with the same results. Also checked with software WebGL2 (SwiftShader) on Linux. |
| Windows (DX12) | Should work through wgpu; not yet tested. |

## License

© 2026 Eugene Hauptmann. Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)), or
- MIT license ([LICENSE-MIT](LICENSE-MIT)),

at your option. The viewer shows the same credits in its footer and in the
Settings panel's About section, from the package metadata.

The NIfTI and Zarr decoders it depends on (`exg-nifti`, `exg-zarr`) are
licensed GPL-3.0-only.
