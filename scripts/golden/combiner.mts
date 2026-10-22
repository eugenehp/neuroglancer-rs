// neuroglancer's CoordinateSpaceCombiner over WatchableCoordinateSpaceTransforms,
// driven through scripted steps; records the global space and each layer's
// transform after every step.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  CoordinateSpaceCombiner,
  isGlobalDimension,
  makeCoordinateSpace,
  makeIdentityTransformedBoundingBox,
  TrackableCoordinateSpace,
  WatchableCoordinateSpaceTransform,
} from "#src/coordinate_transform.js";

type Layer = { names: string[]; units: string[]; scales: number[]; lower: number[]; upper: number[];
  outNames?: string[]; outUnits?: string[]; outScales?: number[]; transform?: number[] };
type Step = { op: string; layer?: string; spec?: Layer; transformSpec?: any; dims?: any; dim?: number; scale?: number; timestamp?: number };

const xyz = (s: number[], u = "m") => ({ names: ["x", "y", "z"], units: [u, u, u], scales: s });
const scenarios: Record<string, Step[]> = {
  two_layers_first_wins: [
    { op: "bind", layer: "a", spec: { ...xyz([4e-9, 4e-9, 40e-9]), lower: [0, 0, 0], upper: [100, 200, 30] } },
    { op: "bind", layer: "b", spec: { ...xyz([8e-9, 8e-9, 8e-9]), lower: [10, 0, 0], upper: [50, 50, 300] } },
    { op: "unbind", layer: "a" },
  ],
  state_dimensions_win: [
    { op: "restore", dims: { x: [8e-9, "m"], y: [8e-9, "m"], z: [8e-9, "m"] } },
    { op: "retain" },
    { op: "bind", layer: "a", spec: { ...xyz([4e-9, 4e-9, 40e-9]), lower: [0, 0, 0], upper: [100, 200, 30],
      transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1] } },
    { op: "release" },
  ],
  extra_state_dimension_dropped_after_load: [
    { op: "restore", dims: { x: [1e-6, "m"], y: [1e-6, "m"], t: [1, "s"] } },
    { op: "retain" },
    { op: "bind", layer: "a", spec: { names: ["x", "y"], units: ["m", "m"], scales: [2e-6, 3e-6], lower: [0, 0], upper: [10, 10] } },
    { op: "release" },
  ],
  channel_and_local_dims_excluded: [
    { op: "bind", layer: "a", spec: { names: ["x", "y", "c^", "t'"], units: ["m", "m", "", "s"], scales: [1e-9, 1e-9, 1, 0.5],
      lower: [0, 0, 0, 0], upper: [64, 64, 3, 10] } },
    { op: "bind", layer: "b", spec: { names: ["y", "x", "z"], units: ["m", "m", "m"], scales: [2e-9, 3e-9, 4e-9],
      lower: [0, 0, 0], upper: [8, 9, 10] } },
  ],
  renamed_output_and_rotation: [
    { op: "bind", layer: "a", spec: { names: ["d0", "d1", "d2"], units: ["m", "m", "m"], scales: [1e-9, 2e-9, 3e-9],
      lower: [0, 0, 0], upper: [10, 20, 30], outNames: ["z", "y", "x"], outUnits: ["m", "m", "m"], outScales: [1e-9, 1e-9, 1e-9],
      transform: [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 100, 0, -3, 1] } },
    { op: "bind", layer: "b", spec: { ...xyz([5e-10, 5e-10, 5e-10]), lower: [0, 0, 0], upper: [4, 4, 4] } },
  ],
  spec_then_rescaled_by_state_dimensions: [
    { op: "restore", dims: { x: [4e-9, "m"], y: [4e-9, "m"], z: [4e-8, "m"] } },
    { op: "retain" },
    { op: "bind", layer: "seg", spec: { ...xyz([8e-9, 8e-9, 80e-9]), lower: [0, 0, 0], upper: [20, 15, 10] },
      transformSpec: { outputDimensions: { x: [8e-9, "m"], y: [8e-9, "m"], z: [8e-8, "m"] }, matrix: [[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0]] } },
    { op: "release" },
  ],
  user_edit_with_timestamp: [
    { op: "bind", layer: "a", spec: { ...xyz([4e-9, 4e-9, 40e-9]), lower: [0, 0, 0], upper: [100, 200, 30] } },
    { op: "editGlobal", dim: 2, scale: 8e-9, timestamp: 1000 },
    { op: "bind", layer: "b", spec: { ...xyz([1e-9, 1e-9, 1e-9]), lower: [0, 0, 0], upper: [10, 10, 10] } },
  ],
};

const f64 = (a: ArrayLike<number>) => Array.from(a, (x) => (Number.isFinite(x) ? x : String(x)));
function spaceJson(s: any) {
  return { valid: s.valid, names: [...s.names], units: [...s.units], scales: f64(s.scales), timestamps: f64(s.timestamps),
    lower: f64(s.bounds.lowerBounds), upper: f64(s.bounds.upperBounds), voxelCenter: [...s.bounds.voxelCenterAtIntegerCoordinates] };
}

const out: any = {};
for (const [name, steps] of Object.entries(scenarios)) {
  const global = new TrackableCoordinateSpace();
  const combiner = new CoordinateSpaceCombiner(global, isGlobalDimension);
  const layers = new Map<string, { t: WatchableCoordinateSpaceTransform; dispose: () => void }>();
  const retainers: (() => void)[] = [];
  const results: any[] = [];
  for (const step of steps) {
    switch (step.op) {
      case "restore": global.restoreState(step.dims); break;
      case "retain": retainers.push(combiner.retain()); break;
      case "release": retainers.pop()!(); break;
      case "bind": {
        const l = step.spec!;
        const rank = l.names.length;
        const inputSpace = makeCoordinateSpace({ names: l.names, units: l.units, scales: Float64Array.from(l.scales),
          boundingBoxes: [makeIdentityTransformedBoundingBox({ lowerBounds: Float64Array.from(l.lower), upperBounds: Float64Array.from(l.upper) })] });
        const outputSpace = makeCoordinateSpace({ names: l.outNames ?? l.names, units: l.outUnits ?? l.units,
          scales: Float64Array.from(l.outScales ?? l.scales) });
        let transform = new Float64Array((rank + 1) ** 2);
        if (l.transform) transform = Float64Array.from(l.transform);
        else for (let i = 0; i <= rank; ++i) transform[i * (rank + 1) + i] = 1;
        const t = new WatchableCoordinateSpaceTransform({ rank, sourceRank: rank, inputSpace, outputSpace, transform });
        if (step.transformSpec !== undefined) t.restoreState(step.transformSpec);
        layers.set(step.layer!, { t, dispose: combiner.bind(t.outputSpace) });
        break;
      }
      case "unbind": layers.get(step.layer!)!.dispose(); layers.delete(step.layer!); break;
      case "editGlobal": {
        const s = global.value;
        const scales = Float64Array.from(s.scales); scales[step.dim!] = step.scale!;
        const timestamps = Array.from(s.timestamps); timestamps[step.dim!] = step.timestamp!;
        global.value = makeCoordinateSpace({ ...s, scales, timestamps, bounds: undefined });
        break;
      }
    }
    const layerOut: any = {};
    for (const [id, { t }] of layers) {
      layerOut[id] = { transform: f64(t.value.transform), output: spaceJson(t.value.outputSpace), json: t.toJSON() ?? null };
    }
    results.push({ global: spaceJson(global.value), layers: layerOut });
  }
  out[name] = { steps, results };
}
const file = fileURLToPath(new URL("../../crates/ng-core/tests/fixtures/combiner.json", import.meta.url));
writeFileSync(file, JSON.stringify(out, null, 1));
console.log(Object.keys(out).length, "scenarios");
