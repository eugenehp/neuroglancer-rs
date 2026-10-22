// A chunk decoder (`neuroglancer::decode_pool`): this page's wasm module,
// sent compiled, decoding the jobs posted to it with `ngDecodeJob`.
import init, { ngDecodeJob } from "./pkg/ng_web.js";

let ready;
self.onmessage = async (e) => {
  const data = e.data;
  if (data.module) {
    ready = init({ module_or_path: data.module });
    return;
  }
  await ready;
  try {
    const out = ngDecodeJob(data.words, data.bytes);
    self.postMessage({ id: data.id, out }, [out.buffer]);
  } catch (error) {
    self.postMessage({ id: data.id, error: String(error) });
  }
};
