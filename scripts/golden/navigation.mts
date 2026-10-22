// Navigation goldens from neuroglancer's own navigation_state.ts, wired as
// viewer.ts wires it. The same scenario is replayed by
// crates/ng-core/tests/navigation_golden.rs.
//
//   node --import ./scripts/golden/ng_register.mjs scripts/golden/navigation.mts \
//     > crates/ng-core/tests/fixtures/navigation.json
import { makeCoordinateSpace, TrackableCoordinateSpace } from "#src/coordinate_transform.js";
import {
  DisplayPose, NavigationState, OrientationState, Position, TrackableCrossSectionZoom, TrackableDepthRange,
  TrackableDisplayDimensions, TrackableProjectionZoom, TrackableRelativeDisplayScales, WatchableDisplayDimensionRenderInfo,
} from "#src/navigation_state.js";
import { mat4, quat, vec3 } from "gl-matrix";
import { kAxes } from "#src/util/geom.js";

const coordinateSpace = new TrackableCoordinateSpace();
const position = new Position(coordinateSpace);
const relativeDisplayScales = new TrackableRelativeDisplayScales(coordinateSpace);
const displayDimensions = new TrackableDisplayDimensions(coordinateSpace);
const displayDimensionRenderInfo = new WatchableDisplayDimensionRenderInfo(relativeDisplayScales.addRef(), displayDimensions.addRef());
const crossSectionOrientation = new OrientationState();
const crossSectionScale = new TrackableCrossSectionZoom(displayDimensionRenderInfo.addRef());
const projectionOrientation = new OrientationState();
const crossSectionDepthRange = new TrackableDepthRange(-10, displayDimensionRenderInfo);
const projectionDepthRange = new TrackableDepthRange(-50, displayDimensionRenderInfo);
const projectionScale = new TrackableProjectionZoom(displayDimensionRenderInfo.addRef());
const cross = new NavigationState(
  new DisplayPose(position.addRef(), displayDimensionRenderInfo.addRef(), crossSectionOrientation.addRef()),
  crossSectionScale.addRef(), crossSectionDepthRange.addRef());
const proj = new NavigationState(
  new DisplayPose(position.addRef(), displayDimensionRenderInfo.addRef(), projectionOrientation.addRef()),
  projectionScale.addRef(), projectionDepthRange.addRef());

// TrackableViewerState's keys and order, for the members ported so far.
const members: [string, any][] = [
  ["dimensions", coordinateSpace], ["relativeDisplayScales", relativeDisplayScales], ["displayDimensions", displayDimensions],
  ["position", position], ["crossSectionOrientation", crossSectionOrientation], ["crossSectionScale", crossSectionScale],
  ["crossSectionDepth", crossSectionDepthRange], ["projectionOrientation", projectionOrientation],
  ["projectionScale", projectionScale], ["projectionDepth", projectionDepthRange],
];
function toJSON() {
  const out: any = {};
  for (const [k, v] of members) { const j = v.toJSON(); if (j !== undefined) out[k] = j; }
  return out;
}
function restore(json: any) {
  // CompoundTrackable.restoreState, minus "dimensions" (restored separately).
  for (const [k, v] of members) {
    if (k === "dimensions") continue;
    if (Object.prototype.hasOwnProperty.call(json, k) && json[k] !== undefined) v.restoreState(json[k]);
  }
}

const space = (scales: number[]) => makeCoordinateSpace({
  names: ["x", "y", "z"], units: ["m", "m", "m"], scales: Float64Array.from(scales),
  boundingBoxes: [{ box: { lowerBounds: Float64Array.of(0, 0, 0), upperBounds: Float64Array.of(100, 51, 7) },
    transform: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0) }],
});

const steps: any[] = [];
function record(name: string) {
  const info = displayDimensionRenderInfo.value;
  const m1 = mat4.create(), m2 = mat4.create();
  cross.toMat4(m1); proj.toMat4(m2);
  steps.push({
    name, json: toJSON(), position: Array.from(position.value),
    crossSectionScale: crossSectionScale.value, projectionScale: projectionScale.value,
    crossSectionDepth: crossSectionDepthRange.value, projectionDepth: projectionDepthRange.value,
    relativeDepth: [cross.relativeDepthRange, proj.relativeDepthRange],
    crossMat: Array.from(m1), projMat: Array.from(m2),
    canonicalVoxelPhysicalSize: info.canonicalVoxelPhysicalSize, canonicalVoxelFactors: Array.from(info.canonicalVoxelFactors),
    displayDimensionIndices: Array.from(info.displayDimensionIndices),
  });
}

coordinateSpace.value = space([4e-9, 4e-9, 40e-9]);
record("initial");
cross.pose.translateDimensionRelative(2, 1);
record("z step");
cross.pose.updateDisplayPosition((pos) => { vec3.add(pos, pos, vec3.fromValues(-3.25, 7.5, 0)); });
record("pan");
const fixed = Float32Array.from(position.value); fixed[0] += 10; fixed[1] -= 3;
cross.pose.rotateAbsolute(vec3.fromValues(0, 1, 0), 0.3, fixed);
record("rotate absolute");
proj.pose.rotateRelative(vec3.fromValues(1, 0, 0), -0.7);
record("rotate relative");
cross.zoomBy(1.1); cross.zoomBy(1.1); proj.zoomBy(0.5);
record("zoom");
cross.pose.translateVoxelsRelative(vec3.fromValues(0, 0, 1));
record("translate voxels");
crossSectionDepthRange.value = 5;
record("depth");
cross.pose.snap();
record("snap");
restore({ displayDimensions: ["z", "x"], relativeDisplayScales: { y: 2 } });
record("display dimensions");
coordinateSpace.value = space([8e-9, 4e-9, 40e-9]);
record("rescaled space");
restore({
  position: ["10.5", 20, "3.25px"], crossSectionOrientation: [0, 0, 1, 1], crossSectionScale: "2.5",
  projectionOrientation: "bad", projectionScale: 300, crossSectionDepth: 0, projectionDepth: 12,
  displayDimensions: ["x", "y", "z"], relativeDisplayScales: {},
});
record("restore");
// ── Slice panels ─────────────────────────────────────────────────────────
// makeSliceView (src/data_panel_layout.ts) with neuroglancer's real
// OrientationState.makeRelative; the handler bodies below are copied from
// src/sliceview/panel.ts and src/rendered_data_panel.ts, which need a DOM.
function makePanelNav(baseToSelf?: quat) {
  if (baseToSelf === undefined) return cross;
  return new NavigationState(
    new DisplayPose(position.addRef(), displayDimensionRenderInfo.addRef(),
      OrientationState.makeRelative(cross.pose.orientation, baseToSelf)),
    crossSectionScale.addRef(), crossSectionDepthRange.addRef());
}
const panels = {
  xy: makePanelNav(),
  xz: makePanelNav(quat.rotateX(quat.create(), quat.create(), Math.PI / 2)),
  yz: makePanelNav(quat.rotateY(quat.create(), quat.create(), Math.PI / 2)),
};
const W = 400, H = 300;
function invView(nav: NavigationState) { const m = mat4.create(); nav.toMat4(m); return m; }
function mousePosition(nav: NavigationState, mx: number, my: number) {
  const v = vec3.transformMat4(vec3.create(), vec3.fromValues(mx - W / 2, my - H / 2, 0), invView(nav));
  const out = Float32Array.from(position.value);
  const { displayRank, displayDimensionIndices } = displayDimensionRenderInfo.value;
  for (let i = 0; i < displayRank; ++i) out[displayDimensionIndices[i]] = v[i];
  return out;
}
function translateByViewportPixels(nav: NavigationState, deltaX: number, deltaY: number) {
  cross.pose.updateDisplayPosition((pos) => {
    vec3.set(pos, -deltaX, -deltaY, 0);
    vec3.transformMat4(pos, pos, invView(nav));
  });
}
function rotateViaMouseDrag(nav: NavigationState, deltaX: number, deltaY: number, initialPosition: Float32Array) {
  const { pose } = nav;
  const xAxis = vec3.transformQuat(vec3.create(), kAxes[0], pose.orientation.orientation);
  const yAxis = vec3.transformQuat(vec3.create(), kAxes[1], pose.orientation.orientation);
  cross.pose.rotateAbsolute(yAxis, ((-deltaX / 4.0) * Math.PI) / 180.0, initialPosition);
  cross.pose.rotateAbsolute(xAxis, ((-deltaY / 4.0) * Math.PI) / 180.0, initialPosition);
}
function zoomByMouse(nav: NavigationState, factor: number, mouseX: number, mouseY: number) {
  if (!nav.valid) return;
  const invViewMatrix = invView(nav);
  const { displayDimensionIndices, displayRank } = displayDimensionRenderInfo.value;
  mouseX -= W / 2;
  mouseY -= H / 2;
  const pos = nav.position.value;
  for (let i = 0; i < displayRank; ++i) {
    const dim = displayDimensionIndices[i];
    const f = invViewMatrix[i] * mouseX + invViewMatrix[4 + i] * mouseY;
    pos[dim] += f * (1 - factor);
  }
  nav.position.changed.dispatch();
  nav.zoomBy(factor);
}
function zViaWheel(nav: NavigationState, deltaY: number, deltaX: number, amount: number) {
  const offset = vec3.create();
  const delta = deltaY !== 0 ? deltaY : deltaX;
  offset[2] = (delta > 0 ? -1 : 1) * amount;
  nav.pose.translateVoxelsRelative(offset);
}
function getWheelZoomAmount(deltaY: number, deltaMode: number) {
  const multiplier = deltaMode === 0 ? 1 / 200.0 : deltaMode === 1 ? 1 / 10.0 : deltaMode === 2 ? 2 : 0;
  return Math.exp(deltaY * multiplier);
}
function recordPanels(name: string) {
  record(name);
  const last = steps[steps.length - 1];
  last.panels = {};
  for (const [k, nav] of Object.entries(panels)) {
    last.panels[k] = { orientation: Array.from(nav.pose.orientation.orientation), invView: Array.from(invView(nav)) };
  }
}

recordPanels("panels initial");
translateByViewportPixels(panels.xz, 13, -7);
recordPanels("xz drag");
zViaWheel(panels.yz, 120, 0, 1); zViaWheel(panels.yz, -3, 0, 10); zViaWheel(panels.xy, 0, 5, 1);
recordPanels("wheel steps");
zoomByMouse(panels.xy, getWheelZoomAmount(-240, 0), 310, 77);
zoomByMouse(panels.xz, getWheelZoomAmount(3, 1), 20, 260);
recordPanels("zoom by mouse");
{
  const initial = mousePosition(panels.xz, 250, 120);
  rotateViaMouseDrag(panels.xz, 8, -5, initial);
  rotateViaMouseDrag(panels.xz, 3, 11, initial);
}
recordPanels("xz rotate drag");
panels.yz.pose.rotateRelative(kAxes[0], 0.1);
panels.xz.pose.rotateRelative(kAxes[2], -0.1);
recordPanels("rotate relative keys");
panels.xz.pose.translateVoxelsRelative(vec3.fromValues(1, 0, 0));
panels.yz.pose.translateVoxelsRelative(vec3.fromValues(0, -1, 0));
recordPanels("arrow keys");
panels.yz.pose.snap();
recordPanels("snap from yz");
console.log(JSON.stringify(steps, null, 1));
