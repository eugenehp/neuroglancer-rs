// Zarr metadata goldens from neuroglancer's own parsers, run on its test data:
// metadata/parse.ts (parseV2Metadata / parseV3Metadata, with the codec
// resolvers registered) for every array, ome.ts (parseOmeMetadata) for every
// OME-NGFF group.
//
//   node --import ./scripts/golden/ng_register.mjs scripts/golden/zarr_metadata.mts \
//     > crates/ng-datasource/tests/fixtures/zarr_metadata.json
import { readdirSync, readFileSync, statSync, existsSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import "#src/datasource/zarr/codec/bytes/resolve.js";
import "#src/datasource/zarr/codec/transpose/resolve.js";
import "#src/datasource/zarr/codec/crc32c/resolve.js";
import "#src/datasource/zarr/codec/gzip/resolve.js";
import "#src/datasource/zarr/codec/zstd/resolve.js";
import "#src/datasource/zarr/codec/blosc/resolve.js";
import "#src/datasource/zarr/codec/sharding_indexed/resolve.js";
import { parseV2Metadata, parseV3Metadata } from "#src/datasource/zarr/metadata/parse.js";
import { ChunkKeyEncoding } from "#src/datasource/zarr/metadata/index.js";
import { parseOmeMetadata } from "#src/datasource/zarr/ome.js";
import { DataType } from "#src/util/data_type.js";

const root = fileURLToPath(new URL("../../../neuroglancer/testdata/datasource/zarr/", import.meta.url));
const BASE = "http://test/";
const json = (p: string) => JSON.parse(readFileSync(p, "utf8"));
const num = (x: number | bigint) => (typeof x === "bigint" ? x.toString() : x);

const arrays: any = {};
const ome: any = {};

function visit(dir: string) {
  const rel = relative(root, dir);
  if (/\.(zip|ocdbt|icechunk)$/.test(dir)) return;
  let meta: any;
  let attrs: any = undefined;
  let zarrVersion = 0;
  try {
    if (existsSync(join(dir, ".zarray"))) {
      zarrVersion = 2;
      const zattrs = existsSync(join(dir, ".zattrs")) ? json(join(dir, ".zattrs")) : {};
      meta = parseV2Metadata(json(join(dir, ".zarray")), zattrs, undefined);
    } else if (existsSync(join(dir, "zarr.json"))) {
      zarrVersion = 3;
      meta = parseV3Metadata(json(join(dir, "zarr.json")), undefined);
      if (meta.nodeType === "group") attrs = meta.userAttributes;
    } else if (existsSync(join(dir, ".zattrs"))) {
      zarrVersion = 2;
      attrs = json(join(dir, ".zattrs"));
    }
  } catch (e) {
    arrays[rel] = { error: String((e as Error).message) };
  }
  if (meta?.nodeType === "array") {
    const layout = meta.codecs.layoutInfo[0];
    arrays[rel] = {
      zarrVersion: meta.zarrVersion,
      shape: meta.shape,
      chunkShape: meta.chunkShape,
      dataType: DataType[meta.dataType],
      fillValue: num(meta.fillValue),
      dimensionNames: meta.dimensionNames,
      dimensionUnits: meta.dimensionUnits,
      dimensionSeparator: meta.dimensionSeparator,
      chunkKeyEncoding: ChunkKeyEncoding[meta.chunkKeyEncoding],
      physicalToLogicalDimension: Array.from(layout.physicalToLogicalDimension),
      readChunkShape: Array.from(layout.readChunkShape),
    };
  }
  if (attrs !== undefined) {
    try {
      // A directory URL, as neuroglancer passes it (kvstoreEnsureDirectoryPipelineUrl).
      const result = parseOmeMetadata(BASE + rel + "/", attrs, zarrVersion);
      if (result !== undefined) {
        const { multiscale } = result;
        ome[rel] = {
          zarrVersion,
          names: multiscale.coordinateSpace.names,
          units: multiscale.coordinateSpace.units,
          scales: Array.from(multiscale.coordinateSpace.scales),
          datasets: multiscale.scales.map((s) => ({ url: s.url, transform: Array.from(s.transform) })),
          baseScales: Array.from(multiscale.baseInfo.baseScales),
          baseTransform: Array.from(multiscale.baseInfo.baseTransform),
        };
      }
    } catch (e) {
      ome[rel] = { error: String((e as Error).message) };
    }
  }
  for (const entry of readdirSync(dir).sort()) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) visit(p);
  }
}

visit(root);
console.log(JSON.stringify({ arrays, ome }, null, 1));
