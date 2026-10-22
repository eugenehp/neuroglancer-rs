// Compressed segmentation fixtures from neuroglancer's own encoder
// (src/sliceview/compressed_segmentation/encode_common.ts), over its raw
// 64³ uint64 test volume and sub-volumes of it with block sizes that do not
// divide the volume. The Rust decoder must give back the raw values.
//
//   node --import ./scripts/golden/ng_register.mjs scripts/golden/compressed_segmentation.mts
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { encodeChannels } from "#src/sliceview/compressed_segmentation/encode_common.js";
import { TypedArrayBuilder } from "#src/util/array.js";

const raw = readFileSync(fileURLToPath(new URL("../../../neuroglancer/testdata/codec/compressed_segmentation/64x64x64-raw-uint64-segmentation.dat", import.meta.url)));
const all = new BigUint64Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const out = fileURLToPath(new URL("../../crates/ng-datasource/tests/fixtures/compressed_segmentation/", import.meta.url));
mkdirSync(out, { recursive: true });

// A sub-volume [0, size) of the 64³ volume, x fastest, with `channels` copies
// (channel c adds c to every value so the channels differ).
function sub(size: number[], channels: number, uint32: boolean) {
  const n = size[0] * size[1] * size[2];
  // A typed array: the encoder picks its uint32 or uint64 path from it.
  const values: any = uint32 ? new Uint32Array(n * channels) : new BigUint64Array(n * channels);
  for (let c = 0; c < channels; ++c)
    for (let z = 0; z < size[2]; ++z)
      for (let y = 0; y < size[1]; ++y)
        for (let x = 0; x < size[0]; ++x) {
          const v = all[x + 64 * (y + 64 * z)] + BigInt(c);
          values[c * n + x + size[0] * (y + size[1] * z)] = uint32 ? Number(v & 0xffffffffn) : v;
        }
  return values;
}

const cases: [string, number[], number[], number, boolean][] = [
  ["u64_64x64x64_b8x8x8", [64, 64, 64], [8, 8, 8], 1, false],
  ["u64_13x10x7_b4x3x2", [13, 10, 7], [4, 3, 2], 1, false],
  ["u64_9x9x9_b8x8x8_2ch", [9, 9, 9], [8, 8, 8], 2, false],
  ["u32_17x11x5_b8x4x2", [17, 11, 5], [8, 4, 2], 1, true],
];
const index: any = {};
for (const [name, size, block, channels, uint32] of cases) {
  const values = sub(size, channels, uint32);
  const builder = new TypedArrayBuilder(Uint32Array, 1024);
  encodeChannels(builder, block, values, [...size, channels]);
  writeFileSync(out + name + ".bin", Buffer.from(builder.view.buffer, builder.view.byteOffset, builder.view.byteLength));
  index[name] = { size, block, channels, uint32 };
}
writeFileSync(out + "index.json", JSON.stringify(index, null, 1) + "\n");
console.log(Object.keys(index).join("\n"));
