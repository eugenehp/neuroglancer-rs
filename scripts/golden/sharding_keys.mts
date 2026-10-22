// neuroglancer's chunk-key functions for sharded precomputed volumes:
// murmurHash3_x86_128Hash64Bits_Bigint and encodeZIndexCompressed3d.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { murmurHash3_x86_128Hash64Bits_Bigint } from "#src/util/hash.js";
import { encodeZIndexCompressed3d } from "#src/util/zorder.js";

const inputs: bigint[] = [0n, 1n, 2n, 255n, 0xffffffffn, 0x100000000n, 0xdeadbeefcafebaben, 0xffffffffffffffffn];
for (let i = 0; i < 40; ++i) inputs.push(BigInt(i * 7919) * 1000003n + BigInt(i));
const murmur = inputs.map((x) => [x.toString(), murmurHash3_x86_128Hash64Bits_Bigint(0, x).toString()]);
const zindex: [number[], number[], string][] = [];
for (const bits of [[0, 0, 0], [3, 2, 3], [1, 5, 0], [10, 10, 10], [21, 21, 21], [7, 0, 13]]) {
  for (const pos of [[0, 0, 0], [1, 2, 3], [5, 3, 7], [1023, 511, 77], [2097151, 1, 1048576]]) {
    const p = pos.map((v, d) => v % 2 ** bits[d]);
    zindex.push([bits, p, encodeZIndexCompressed3d(bits[0], bits[1], bits[2], p[0], p[1], p[2]).toString()]);
  }
}
const out = fileURLToPath(new URL("../../crates/ng-datasource/tests/fixtures/sharding_keys.json", import.meta.url));
writeFileSync(out, JSON.stringify({ murmur, zindex }, null, 1));
console.log(murmur.length, "hashes,", zindex.length, "z-indexes");
