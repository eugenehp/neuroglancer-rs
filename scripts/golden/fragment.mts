// neuroglancer's share-link encoding: encodeStateAsFragment
// (JSON.stringify + encodeFragment) on assorted states.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Copied verbatim from src/ui/url_hash_binding.ts (which imports UI code and
// CSS, so it cannot load here).
function encodeFragment(fragment: string) {
  return encodeURI(fragment).replace(
    /[!'()*;,]/g,
    (c) => "%" + c.charCodeAt(0).toString(16).toUpperCase(),
  );
}
function encodeStateAsFragment(state: any): string {
  return encodeFragment(JSON.stringify(state));
}

const states = [
  { a: 1 },
  { dimensions: { x: [4e-9, "m"], "c^": [1, ""] }, position: [1.5, -0, 1e21, 1e-7, 0.1 + 0.2, 123456789012345680000] },
  { s: "quote \" back \\ tab \t nl \n ctl \u0001 \u001f del \u007f" },
  { s: "!'()*;, &=+$#/?:@ ~-_. ünïcødé 🧠 %20" },
  { layers: [{ type: "image", source: "precomputed://gs://bucket/path", name: "em" }], layout: "xy-3d" },
  { n: [5e-324, 1.7976931348623157e308, -1.5e-10, 100, 1e20, 2 ** 53] },
];
const out = states.map((s) => [s, encodeStateAsFragment(s)]);
writeFileSync(fileURLToPath(new URL("../../crates/ng-core/tests/fixtures/fragment.json", import.meta.url)), JSON.stringify(out, null, 1));
console.log(out.length);
