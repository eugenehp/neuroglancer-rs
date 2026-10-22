import { hashCombine } from "../../../neuroglancer/src/gpu_hash/hash_function.ts";
import { hsvToRgb } from "../../../neuroglancer/src/util/colorspace.ts";
// Mirrors SegmentColorHash.compute (src/segment_color.ts), which can't be
// imported directly because it pulls in WebGL modules.
function compute(seed: number, x: bigint) {
  let h = hashCombine(seed, Number(x & 0xffffffffn));
  h = hashCombine(h, Number(x >> 32n));
  const out = new Float32Array(3);
  hsvToRgb(out, (h & 0xff) / 255, 0.5 + 0.5 * (((h >> 8) & 0xff) / 255), 1.0);
  return Array.from(out);
}
const ids = [0n, 1n, 8n, 42n, 255n, 65536n, 4294967295n, 4294967296n, 123456789012345n, 18446744073709551615n];
const rows = [];
for (const seed of [0, 1, 12345, 4294967295]) for (const id of ids) rows.push([seed, id.toString(), compute(seed, id)]);
console.log(JSON.stringify({ hashCombine: [[0,0,hashCombine(0,0)],[1,2,hashCombine(1,2)],[4294967295,4294967295,hashCombine(4294967295,4294967295)]], colors: rows }));
