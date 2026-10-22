// Decodes the JPEG fixtures with neuroglancer's own decoder
// (src/third_party/jpgjs, as src/async_computation/decode_jpeg.ts uses it:
// no RGB forcing, multi-component data made planar).
import { readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { JpegDecoder } from "#src/third_party/jpgjs/jpg.js";
import { transposeArray2d } from "#src/util/array.js";

const dir = fileURLToPath(new URL("../../crates/ng-datasource/tests/fixtures/images/", import.meta.url));
for (const name of readdirSync(dir).filter((f) => f.endsWith(".jpg"))) {
  const parser = new JpegDecoder();
  rmSync(dir + name + ".raw", { force: true });
  rmSync(dir + name + ".error", { force: true });
  try {
    parser.parse(new Uint8Array(readFileSync(dir + name)));
  } catch (e) {
    // What neuroglancer reports instead of showing the chunk.
    writeFileSync(dir + name + ".error", String(e));
    console.log(name, "error:", String(e));
    continue;
  }
  let result = parser.getData(parser.width, parser.height, false);
  if (parser.numComponents === 3) {
    result = transposeArray2d(result, parser.width * parser.height, 3);
  }
  writeFileSync(dir + name + ".raw", result);
  console.log(name, parser.width, parser.height, parser.numComponents);
}
