// Navigation restore with the viewer's full wiring: the root navigation
// state (viewer.ts) and the layer group viewer's linked copies
// (layer_group_viewer.ts), which rendering reads. Restores a state, then
// sets the coordinate space as the layers' arrival would, and records the
// root's JSON and zooms.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { makeCoordinateSpace, TrackableCoordinateSpace } from "#src/coordinate_transform.js";
import {
  DisplayPose, LinkedDepthRange, LinkedDisplayDimensions, LinkedOrientationState, LinkedPosition,
  LinkedRelativeDisplayScales, LinkedZoomState, NavigationState, OrientationState, Position, TrackableCrossSectionZoom,
  TrackableDepthRange, TrackableDisplayDimensions, TrackableProjectionZoom, TrackableRelativeDisplayScales,
  WatchableDisplayDimensionRenderInfo,
} from "#src/navigation_state.js";

function makeViewer() {
  const coordinateSpace = new TrackableCoordinateSpace();
  const position = new Position(coordinateSpace);
  const relativeDisplayScales = new TrackableRelativeDisplayScales(coordinateSpace);
  const displayDimensions = new TrackableDisplayDimensions(coordinateSpace);
  const info = new WatchableDisplayDimensionRenderInfo(relativeDisplayScales.addRef(), displayDimensions.addRef());
  const crossSectionOrientation = new OrientationState();
  const crossSectionScale = new TrackableCrossSectionZoom(info.addRef());
  const projectionOrientation = new OrientationState();
  const crossSectionDepthRange = new TrackableDepthRange(-10, info);
  const projectionDepthRange = new TrackableDepthRange(-50, info);
  const projectionScale = new TrackableProjectionZoom(info.addRef());
  const navigationState = new NavigationState(
    new DisplayPose(position.addRef(), info.addRef(), crossSectionOrientation.addRef()),
    crossSectionScale.addRef(), crossSectionDepthRange.addRef());
  const perspectiveNavigationState = new NavigationState(
    new DisplayPose(position.addRef(), info.addRef(), projectionOrientation.addRef()),
    projectionScale.addRef(), projectionDepthRange.addRef());
  // LayerGroupViewer's linked copies (default: linked).
  const lRel = new LinkedRelativeDisplayScales(navigationState.pose.relativeDisplayScales.addRef());
  const lDims = new LinkedDisplayDimensions(navigationState.pose.displayDimensions.addRef());
  const lPos = new LinkedPosition(navigationState.position.addRef());
  const lOrient = new LinkedOrientationState(navigationState.pose.orientation.addRef());
  const lInfo = new WatchableDisplayDimensionRenderInfo(lRel.value, lDims.value);
  const lScale = new LinkedZoomState(navigationState.zoomFactor.addRef() as TrackableCrossSectionZoom, lInfo.addRef());
  const lDepth = new LinkedDepthRange(navigationState.depthRange.addRef(), lInfo);
  new NavigationState(new DisplayPose(lPos.value, lInfo.addRef(), lOrient.value), lScale.value, lDepth.value);
  const lProjOrient = new LinkedOrientationState(perspectiveNavigationState.pose.orientation.addRef());
  const lProjScale = new LinkedZoomState(perspectiveNavigationState.zoomFactor.addRef() as TrackableProjectionZoom, lInfo.addRef());
  const lProjDepth = new LinkedDepthRange(perspectiveNavigationState.depthRange.addRef(), lInfo);
  new NavigationState(new DisplayPose(lPos.value.addRef(), lInfo.addRef(), lProjOrient.value), lProjScale.value, lProjDepth.value);
  const members: [string, any][] = [
    ["dimensions", coordinateSpace], ["relativeDisplayScales", relativeDisplayScales], ["displayDimensions", displayDimensions],
    ["position", position], ["crossSectionOrientation", crossSectionOrientation], ["crossSectionScale", crossSectionScale],
    ["crossSectionDepth", crossSectionDepthRange], ["projectionOrientation", projectionOrientation],
    ["projectionScale", projectionScale], ["projectionDepth", projectionDepthRange],
  ];
  return {
    coordinateSpace, crossSectionScale, projectionScale, position, lScale,
    restore(json: any) {
      for (const [k, v] of members) {
        if (Object.prototype.hasOwnProperty.call(json, k) && json[k] !== undefined) v.restoreState(json[k]);
      }
    },
    toJSON() {
      const out: any = {};
      for (const [k, v] of members) { const j = v.toJSON(); if (j !== undefined) out[k] = j; }
      return out;
    },
  };
}

const space = (names: string[], scales: number[], upper: number[]) => makeCoordinateSpace({
  names, units: names.map(() => "m"), scales: Float64Array.from(scales),
  boundingBoxes: [{ box: { lowerBounds: new Float64Array(names.length), upperBounds: Float64Array.from(upper) },
    transform: (() => { const r = names.length; const t = new Float64Array(r * (r + 1)); for (let i = 0; i < r; ++i) t[i * r + i] = 1; return t; })() }],
});

const scenarios: Record<string, { state: any; spaces: any[] }> = {
  zoom_without_dimensions: { state: { crossSectionScale: 0.037, projectionScale: 50, position: [5, 5] },
    spaces: [{ names: ["y", "x"], scales: [4e-9, 3e-8], upper: [10, 10] }] },
  zoom_with_dimensions: { state: { dimensions: { y: [4e-9, "m"], x: [3e-8, "m"] }, crossSectionScale: 0.037, projectionScale: 50 },
    spaces: [{ names: ["y", "x"], scales: [4e-9, 3e-8], upper: [10, 10] }] },
  dimensions_rescaled_later: { state: { dimensions: { x: [4e-9, "m"], y: [4e-9, "m"], z: [4e-8, "m"] }, crossSectionScale: 2 },
    spaces: [{ names: ["x", "y", "z"], scales: [4e-9, 4e-9, 4e-8], upper: [100, 100, 10] },
             { names: ["x", "y", "z"], scales: [8e-9, 8e-9, 4e-8], upper: [50, 50, 10] }] },
  defaults: { state: {}, spaces: [{ names: ["x", "y", "z"], scales: [4e-9, 4e-9, 4e-8], upper: [100, 51, 7] }] },
};
const out: any = {};
for (const [name, { state, spaces }] of Object.entries(scenarios)) {
  const v = makeViewer();
  if (state.dimensions) v.coordinateSpace.restoreState(state.dimensions);
  v.restore(state);
  const steps: any[] = [];
  for (const s of spaces) {
    v.coordinateSpace.value = space(s.names, s.scales, s.upper);
    steps.push({ json: v.toJSON(), crossSectionScale: v.crossSectionScale.value, projectionScale: v.projectionScale.value,
      position: Array.from(v.position.value) });
  }
  out[name] = { state, spaces, steps };
}
writeFileSync(fileURLToPath(new URL("../../crates/ng-core/tests/fixtures/navigation_linked.json", import.meta.url)), JSON.stringify(out, null, 1));
for (const [k, v] of Object.entries(out) as any) console.log(k, v.steps.map((s: any) => [s.crossSectionScale, s.projectionScale]));
