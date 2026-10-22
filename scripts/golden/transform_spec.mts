// neuroglancer's WatchableCoordinateSpaceTransform.restoreState: a data
// source's `transform` from viewer state over its default transform.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { coordinateTransformSpecificationFromLegacyJson, makeCoordinateSpace, makeIdentityTransformedBoundingBox, WatchableCoordinateSpaceTransform } from "#src/coordinate_transform.js";

const defaults: Record<string, any> = {
  xyz: { names: ["x", "y", "z"], units: ["m", "m", "m"], scales: [4e-9, 4e-9, 40e-9], lower: [0, 0, 0], upper: [100, 200, 30] },
  xyz_translated: { names: ["x", "y", "z"], units: ["m", "m", "m"], scales: [1e-3, 1e-3, 2e-3], lower: [0, 0, 0], upper: [10, 20, 30],
    transform: [0.5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, -4, 5, 1] },
  xyzc: { names: ["x", "y", "z", "c^"], units: ["m", "m", "m", ""], scales: [1e-9, 1e-9, 1e-9, 1], lower: [0, 0, 0, 0], upper: [8, 8, 8, 3] },
};
const nm = { names: ["x", "y", "z"], units: ["m", "m", "m"], scales: [4e-9, 4e-9, 40e-9], lower: [0, 0, 0], upper: [100, 200, 30] };
defaults.nm = nm;
const legacy: [string, string, any][] = [
  ["legacy_rows", "nm", [[2, 0, 0, 10], [0, 3, 0, -5], [0, 0, 1, 0.25], [0, 0, 0, 1]]],
  ["legacy_flat", "nm", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 7, 8, 9, 1]],
  ["legacy_object", "nm", { rotation: [0.1, 0.2, 0.3, 0.9], translation: [1.5, -2, 3], scale: [4, 4, 40] }],
  ["legacy_scale_only", "nm", { scale: [0.1, 0.3, 0.7] }],
];
const specs: [string, string, any][] = [
  ["rename", "xyz", { outputDimensions: { a: [4e-9, "m"], b: [4e-9, "m"], c: [40e-9, "m"] } }],
  ["rescale_output", "xyz_translated", { outputDimensions: { x: [1e-6, "m"], y: [2e-6, "m"], z: [1e-3, "m"] } }],
  ["matrix", "xyz", { outputDimensions: { x: [4e-9, "m"], y: [4e-9, "m"], z: [40e-9, "m"] },
    matrix: [[0, 1, 0, 10], [1, 0, 0, -20], [0, 0, 2, 0.5]] }],
  ["input_dimensions", "xyz", { inputDimensions: { x: [8e-9, "m"], y: [8e-9, "m"], z: [8e-9, "m"] },
    outputDimensions: { x: [8e-9, "m"], y: [8e-9, "m"], z: [8e-9, "m"] } }],
  ["extra_singleton", "xyz", { outputDimensions: { x: [4e-9, "m"], y: [4e-9, "m"], z: [40e-9, "m"], t: [1, "s"] }, sourceRank: 3,
    matrix: [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 7]] }],
  ["extra_singleton_no_matrix", "xyz_translated", { outputDimensions: { x: [1e-3, "m"], y: [1e-3, "m"], z: [2e-3, "m"], t: [2, "s"], u: [1, ""] }, sourceRank: 3 }],
  ["invalid_names_reset", "xyz", { outputDimensions: { x: [4e-9, "m"], y: [4e-9, "m"], "x'": [1, ""] }, matrix: undefined }],
  ["channel_kept", "xyzc", { outputDimensions: { x: [2e-9, "m"], y: [2e-9, "m"], z: [2e-9, "m"], "c'": [1, ""] } }],
  ["permuted_input_names", "xyz", { inputDimensions: { z: [40e-9, "m"], x: [4e-9, "m"], y: [4e-9, "m"] },
    outputDimensions: { p: [1e-9, "m"], q: [1e-9, "m"], r: [1e-9, "m"] }, matrix: [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]] }],
];

const f64 = (a: ArrayLike<number>) => Array.from(a, (x) => (Number.isFinite(x) ? x : String(x)));
const space = (s: any) => ({ names: [...s.names], units: [...s.units], scales: f64(s.scales),
  lower: f64(s.bounds.lowerBounds), upper: f64(s.bounds.upperBounds), voxelCenter: [...s.bounds.voxelCenterAtIntegerCoordinates] });
const out = [...specs, ...legacy].map(([name, d, spec]) => {
  const def = defaults[d];
  const rank = def.names.length;
  const inputSpace = makeCoordinateSpace({ names: def.names, units: def.units, scales: Float64Array.from(def.scales),
    boundingBoxes: [makeIdentityTransformedBoundingBox({ lowerBounds: Float64Array.from(def.lower), upperBounds: Float64Array.from(def.upper) })] });
  const outputSpace = makeCoordinateSpace({ names: def.names, units: def.units, scales: Float64Array.from(def.scales) });
  let transform = new Float64Array((rank + 1) ** 2);
  if (def.transform) transform = Float64Array.from(def.transform); else for (let i = 0; i <= rank; ++i) transform[i * (rank + 1) + i] = 1;
  const t = new WatchableCoordinateSpaceTransform({ rank, sourceRank: rank, inputSpace, outputSpace, transform });
  if (name.startsWith("legacy_")) t.spec = coordinateTransformSpecificationFromLegacyJson(spec);
  else t.restoreState(JSON.parse(JSON.stringify(spec)));
  const v = t.value;
  return { name, default: def, spec, result: { rank: v.rank, sourceRank: v.sourceRank, transform: f64(v.transform),
    input: space(v.inputSpace), output: space(v.outputSpace) }, json: t.toJSON() ?? null };
});
const file = fileURLToPath(new URL("../../crates/ng-core/tests/fixtures/transform_spec.json", import.meta.url));
writeFileSync(file, JSON.stringify(out, null, 1));
console.log(out.length, "cases");
