// neuroglancer's invlerp uniforms (enableLerpShaderFunction, src/webgl/lerp.ts)
// for a set of data types and ranges, captured from a mock GL.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// The WebGL constants modules read at import time: any distinct numbers.
let next = 1;
const constants = new Map<PropertyKey, number>();
(globalThis as any).WebGL2RenderingContext = new Proxy({}, {
  get: (_t, p) => constants.get(p) ?? (constants.set(p, next++), constants.get(p)),
});
const { DataType } = await import("#src/util/data_type.js");
const { enableLerpShaderFunction } = await import("#src/webgl/lerp.js");

const f32 = (x: number) => Math.fround(x);
const cases: [string, (number | string)[]][] = [
  ["UINT8", [0, 255]], ["UINT8", [128, 128]], ["INT16", [-32768, 32767]], ["FLOAT32", [0, 1]],
  ["FLOAT32", [1, 0]], ["FLOAT32", [0.5, 0.5]], ["FLOAT32", [-1e-30, 3e30]],
  ["UINT32", [0, 4294967295]], ["UINT32", [16777216, 16777217]], ["UINT32", [4294967295, 0]], ["UINT32", [7, 7]],
  ["UINT32", [100, 50000000]], ["INT32", [-2147483648, 2147483647]], ["INT32", [-5, 5]], ["INT32", [10, -100000000]],
  ["UINT64", ["0", "18446744073709551615"]], ["UINT64", ["1099511627776", "2199023255552"]],
  ["UINT64", ["18446744073709551615", "0"]], ["UINT64", ["5", "5"]], ["UINT64", ["3", "16777219"]],
  ["UINT64", ["123456789012345", "123456789012345678"]],
];
const out = cases.map(([type, interval]) => {
  const dataType = (DataType as any)[type];
  const captured: Record<string, number[]> = {};
  const rec = (name: string) => (loc: string, ...args: number[]) => { captured[loc] = args; };
  const gl = { uniform2f: rec("2f"), uniform2ui: rec("2ui"), uniform2i: rec("2i"), uniform3ui: rec("3ui"), uniform1f: rec("1f") };
  const shader: any = { gl, uniform: (n: string) => n };
  const iv = dataType === DataType.UINT64 ? interval.map((x) => BigInt(x)) : interval;
  enableLerpShaderFunction(shader, "f", dataType, iv as any);
  // As the uniforms hold them: floats rounded to f32, ints as uint32 bits.
  const r: any = { type, interval: interval.map(String) };
  if (captured["uLerpParams_f"]) r.params = captured["uLerpParams_f"].map(f32);
  if (captured["uLerpBounds_f"]) r.bounds = captured["uLerpBounds_f"].map((x) => x >>> 0);
  if (captured["uLerpScalar_f"]) r.scalar = f32(captured["uLerpScalar_f"][0]);
  return r;
});
const file = fileURLToPath(new URL("../../crates/ng-render/tests/fixtures/lerp_params.json", import.meta.url));
writeFileSync(file, JSON.stringify(out, (_k, v) => (typeof v === "number" && !Number.isFinite(v) ? String(v) : v), 1));
console.log(out.length, "cases");
