// neuroglancer's urlSafeToJSON on assorted inputs.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { urlSafeToJSON } from "#src/util/json.js";

const inputs = [
  "{'a':1_'b':[1_2_3]}",
  `{"a":"x,y_z&w"_'b':'it\\'s'}`,
  `{'q':'say "hi"'}`,
  `{'e':'back\\\\slash'_'n':'\\n'}`,
  "{'unterminated:1}",
  `{"s":'mixed "a" \\'b\\''}`,
  "{'layers':[{'source':'precomputed://gs://bucket/path'_'type':'image'_'name':'em'}]_'position':[1.5_2_3]}",
  "a&b_c,d",
  `'only'`,
  `"dq"_'sq'`,
  "{'a':'x\\\ny'}",
  "{'k':'\\\\'}",
];
const out = inputs.map((x) => [x, urlSafeToJSON(x)]);
writeFileSync(fileURLToPath(new URL("../../crates/ng-core/tests/fixtures/url_safe.json", import.meta.url)), JSON.stringify(out, null, 1));
console.log(out.length);
