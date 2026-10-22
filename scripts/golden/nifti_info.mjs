// Expected NIfTI volume info for neuroglancer's test files, computed as
// neuroglancer computes it. The block marked VERBATIM is copied from
// neuroglancer/src/datasource/nifti/backend.ts (GET_NIFTI_VOLUME_INFO_RPC_ID);
// the file can't be imported directly outside neuroglancer's build, so the
// libraries it uses are: nifti-reader-js for the header, gl-matrix (Float32Array
// storage, which rounds) for the transform.
//
//   cd scripts/golden && npm install
//   node scripts/golden/nifti_info.mjs > crates/ng-datasource/tests/fixtures/nifti_info.json
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import { isCompressed, NIFTI1, readHeader, readImage } from "nifti-reader-js";
import { mat4, quat, vec3 } from "gl-matrix";

const testdata = fileURLToPath(new URL("../../../neuroglancer/testdata/datasource/nifti/", import.meta.url));
const kOneVec = vec3.fromValues(1, 1, 1);

// neuroglancer/src/util/geom.ts
function translationRotationScaleZReflectionToMat4(out, translation, rotation, scale, zReflection) {
  const temp = out;
  out[0] = scale[0];
  out[1] = scale[1];
  out[2] = scale[2] * zReflection;
  return mat4.fromRotationTranslationScale(out, rotation, translation, temp);
}

// neuroglancer/src/util/matrix.ts
function createIdentity(c, rows, cols = rows) {
  const m = new c(rows * cols);
  for (let i = 0; i < rows && i < cols; ++i) m[i * rows + i] = 1;
  return m;
}

const NiftiDataType = { 2: "UINT8", 4: "INT16", 8: "INT32", 16: "FLOAT32", 64: "FLOAT64", 256: "INT8", 512: "UINT16", 768: "UINT32", 1024: "INT64", 1280: "UINT64" };
const DATA_TYPE_CONVERSIONS = new Map([
  ["INT8", "INT8"], ["UINT8", "UINT8"], ["INT16", "INT16"], ["UINT16", "UINT16"],
  ["INT32", "INT32"], ["UINT32", "UINT32"], ["INT64", "UINT64"], ["UINT64", "UINT64"],
  ["FLOAT32", "FLOAT32"], ["FLOAT64", "FLOAT32"],
]);

function info(header) {
  const dataType = DATA_TYPE_CONVERSIONS.get(NiftiDataType[header.datatypeCode]);
  // ---- VERBATIM from backend.ts (DataType lookup above replaces the map) ----
  let spatialInvScale = 1;
  let spatialUnit = "";
  switch (header.xyzt_units & NIFTI1.SPATIAL_UNITS_MASK) {
    case NIFTI1.UNITS_METER: spatialInvScale = 1; spatialUnit = "m"; break;
    case NIFTI1.UNITS_MM: spatialInvScale = 1e3; spatialUnit = "m"; break;
    case NIFTI1.UNITS_MICRON: spatialInvScale = 1e6; spatialUnit = "m"; break;
  }
  let timeUnit = "";
  let timeInvScale = 1;
  switch (header.xyzt_units & NIFTI1.TEMPORAL_UNITS_MASK) {
    case NIFTI1.UNITS_SEC: timeUnit = "s"; timeInvScale = 1; break;
    case NIFTI1.UNITS_MSEC: timeUnit = "s"; timeInvScale = 1e3; break;
    case NIFTI1.UNITS_USEC: timeUnit = "s"; timeInvScale = 1e6; break;
    case NIFTI1.UNITS_HZ: timeUnit = "Hz"; timeInvScale = 1; break;
    case NIFTI1.UNITS_RADS: timeUnit = "rad/s"; timeInvScale = 1; break;
  }
  let units = [spatialUnit, spatialUnit, spatialUnit, timeUnit, "", "", ""];
  let sourceScales = Float64Array.of(
    header.pixDims[1] / spatialInvScale, header.pixDims[2] / spatialInvScale, header.pixDims[3] / spatialInvScale,
    header.pixDims[4] / timeInvScale, header.pixDims[5], header.pixDims[6], header.pixDims[7]);
  let viewScales = Float64Array.of(1 / spatialInvScale, 1 / spatialInvScale, 1 / spatialInvScale, 1 / timeInvScale, 1, 1, 1);
  let sourceNames = ["i", "j", "k", "m", "c^", "c1^", "c2^"];
  let viewNames = ["x", "y", "z", "t", "c^", "c1^", "c2^"];
  const rank = header.dims[0];
  sourceNames = sourceNames.slice(0, rank);
  viewNames = viewNames.slice(0, rank);
  units = units.slice(0, rank);
  sourceScales = sourceScales.slice(0, rank);
  viewScales = viewScales.slice(0, rank);
  const { quatern_b, quatern_c, quatern_d } = header;
  const quatern_a = Math.sqrt(1.0 - quatern_b * quatern_b - quatern_c * quatern_c - quatern_d * quatern_d);
  const qfac = header.pixDims[0] === -1 ? -1 : 1;
  const qoffset = vec3.fromValues(header.qoffset_x, header.qoffset_y, header.qoffset_z);
  const method2Transform = translationRotationScaleZReflectionToMat4(
    mat4.create(), qoffset, quat.fromValues(quatern_b, quatern_c, quatern_d, quatern_a), kOneVec, qfac);
  const transform = createIdentity(Float64Array, rank + 1);
  const copyRank = Math.min(3, rank);
  for (let row = 0; row < copyRank; ++row) {
    for (let col = 0; col < copyRank; ++col) {
      transform[col * (rank + 1) + row] = method2Transform[col * 4 + row];
    }
    transform[rank * (rank + 1) + row] = method2Transform[12 + row];
  }
  // ---- end VERBATIM ----
  return {
    rank, sourceNames, viewNames, units,
    sourceScales: Array.from(sourceScales), viewScales: Array.from(viewScales),
    description: header.description, transform: Array.from(transform), dataType,
    volumeSize: Array.from(header.dims.slice(1, 1 + rank)),
  };
}

// The voxels as the chunk decoder sees them: the stored type, float64
// narrowed to float32 (backend.ts `download`).
function voxels(header, buffer) {
  const image = readImage(header, buffer);
  const dv = new DataView(image);
  const le = header.littleEndian;
  const size = { 2: 1, 256: 1, 4: 2, 512: 2, 8: 4, 768: 4, 16: 4, 64: 8, 1024: 8, 1280: 8 }[header.datatypeCode];
  const read = {
    2: (o) => dv.getUint8(o), 256: (o) => dv.getInt8(o), 4: (o) => dv.getInt16(o, le), 512: (o) => dv.getUint16(o, le),
    8: (o) => dv.getInt32(o, le), 768: (o) => dv.getUint32(o, le), 16: (o) => dv.getFloat32(o, le),
    64: (o) => Math.fround(dv.getFloat64(o, le)),
    1024: (o) => dv.getBigUint64(o, le).toString(), 1280: (o) => dv.getBigUint64(o, le).toString(),
  }[header.datatypeCode];
  const n = image.byteLength / size;
  let sum = 0;
  const first = [];
  for (let i = 0; i < n; ++i) {
    const v = read(i * size);
    if (first.length < 16) first.push(v);
    sum += Number(v);
  }
  return { count: n, first, sum };
}

const out = {};
for (const name of ["standard.nii", "standard.nii.gz", "example_nifti2.nii.gz"]) {
  let bytes = readFileSync(testdata + name);
  let buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  if (isCompressed(buffer)) {
    const u = gunzipSync(bytes);
    buffer = u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength);
  }
  const header = readHeader(buffer);
  out[name] = { info: info(header), littleEndian: header.littleEndian, voxels: voxels(header, buffer) };
}
console.log(JSON.stringify(out, null, 1));
