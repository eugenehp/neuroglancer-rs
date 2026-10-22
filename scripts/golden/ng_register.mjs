// Lets node run neuroglancer's TypeScript sources directly, so goldens come
// from neuroglancer's own code rather than a re-implementation of it:
//
//   node --import ./scripts/golden/ng_register.mjs <script.mts>
//
// `#src/foo.js` resolves to ../neuroglancer/src/foo.ts, which is compiled with
// the TypeScript compiler using neuroglancer's own options (tsconfig.json):
// node's built-in type stripping can't handle parameter properties or enums,
// and `useDefineForClassFields` changes class-field semantics. Bare package
// names resolve from this directory's node_modules.
import { existsSync, readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import ts from "typescript";

const src = new URL("../../../neuroglancer/src/", import.meta.url);
const here = new URL("./", import.meta.url);

const compilerOptions = {
  target: ts.ScriptTarget.ES2022,
  module: ts.ModuleKind.ESNext,
  useDefineForClassFields: true,
  experimentalDecorators: true,
  preserveConstEnums: true,
  esModuleInterop: true,
};

registerHooks({
  resolve(specifier, context, next) {
    if (specifier.startsWith("#src/")) {
      // `foo.js` names `foo.ts`, unless there really is a `foo.js`
      // (vendored third-party code).
      const ts = new URL(specifier.slice(5).replace(/\.js$/, ".ts"), src);
      const js = new URL(specifier.slice(5), src);
      return { url: (existsSync(ts) || !existsSync(js) ? ts : js).href, shortCircuit: true };
    }
    const bare = !/^(\.|\/|node:|file:)/.test(specifier);
    return next(specifier, bare ? { ...context, parentURL: here.href } : context);
  },
  load(url, context, next) {
    if (url.startsWith(src.href) && url.endsWith(".ts")) {
      const source = readFileSync(new URL(url), "utf8");
      const { outputText } = ts.transpileModule(source, { compilerOptions, fileName: url });
      return { format: "module", source: outputText, shortCircuit: true };
    }
    return next(url, context);
  },
});
