//! Chunk decoding in Web Workers, as neuroglancer decodes in its workers:
//! each worker instantiates this same wasm module (compiled once, here) and
//! runs [`decode_job`] on what it is sent, so the bytes come out as they
//! would on the page. Jobs go to the least busy worker.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use futures::channel::oneshot;
use ng_datasource::decode::DecodeJob;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{MessageEvent, Worker, WorkerOptions, WorkerType};

type Reply = oneshot::Sender<anyhow::Result<Vec<u8>>>;

struct Pool {
    workers: Vec<Worker>,
    /// Jobs out with each worker.
    busy: Vec<Cell<usize>>,
    /// Replies waited for, by job id: the worker it went to, and where the
    /// result goes.
    pending: RefCell<HashMap<u32, (usize, Reply)>>,
    next: Cell<u32>,
}

/// Decodes one job: the worker's side (`ngDecodeJob(words, bytes)`).
#[wasm_bindgen(js_name = ngDecodeJob)]
pub fn decode_job(words: Vec<u32>, bytes: Vec<u8>) -> Result<Vec<u8>, JsValue> {
    let job = DecodeJob::from_words(&words).map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
    job.run(&bytes).map_err(|e| JsValue::from_str(&format!("{e:#}")))
}

fn set(object: &js_sys::Object, key: &str, value: &JsValue) {
    let _ = js_sys::Reflect::set(object, &JsValue::from_str(key), value);
}

fn get(object: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(object, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}

/// Starts `count` workers running `script` (a module worker importing this
/// module's JS and calling `ngDecodeJob`: see `ng-web`'s
/// `decode_worker.js`), and sends this thread's decode jobs to them.
pub fn start(script: &str, count: usize) -> Result<(), JsValue> {
    let options = WorkerOptions::new();
    options.set_type(WorkerType::Module);
    let module = wasm_bindgen::module();
    let mut workers = vec![];
    for _ in 0..count.max(1) {
        let worker = Worker::new_with_options(script, &options)?;
        let init = js_sys::Object::new();
        set(&init, "module", &module);
        worker.post_message(&init)?;
        workers.push(worker);
    }
    let pool = Rc::new(Pool { busy: workers.iter().map(|_| Cell::new(0)).collect(), workers, pending: Default::default(), next: Cell::new(0) });

    for (i, worker) in pool.workers.iter().enumerate() {
        let p = pool.clone();
        let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
            let data = e.data();
            let Some(id) = get(&data, "id").as_f64() else { return };
            let Some((w, reply)) = p.pending.borrow_mut().remove(&(id as u32)) else { return };
            p.busy[w].set(p.busy[w].get().saturating_sub(1));
            let error = get(&data, "error");
            let result = match error.as_string() {
                Some(message) => Err(anyhow::anyhow!(message)),
                None => Ok(js_sys::Uint8Array::new(&get(&data, "out")).to_vec()),
            };
            let _ = reply.send(result);
        });
        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        on_message.forget();
        // A worker that fails (its script not found, say): its jobs fail,
        // and from then on jobs decode on the page.
        let p = pool.clone();
        let on_error = Closure::<dyn FnMut(web_sys::Event)>::new(move |_e: web_sys::Event| {
            log::error!("a decode worker failed; decoding on the page from now on");
            ng_datasource::decode::clear_offload();
            let failed: Vec<u32> = p.pending.borrow().iter().filter(|(_, (w, _))| *w == i).map(|(id, _)| *id).collect();
            for id in failed {
                if let Some((_, reply)) = p.pending.borrow_mut().remove(&id) {
                    let _ = reply.send(Err(anyhow::anyhow!("the decode worker failed")));
                }
            }
        });
        worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        on_error.forget();
    }

    ng_datasource::decode::set_offload(Rc::new(move |job: DecodeJob, data: Vec<u8>| {
        let p = &pool;
        let w = (0..p.workers.len()).min_by_key(|&w| p.busy[w].get()).unwrap_or(0);
        let id = p.next.get();
        p.next.set(id.wrapping_add(1));
        let (send, receive) = oneshot::channel();
        p.pending.borrow_mut().insert(id, (w, send));
        p.busy[w].set(p.busy[w].get() + 1);
        let message = js_sys::Object::new();
        set(&message, "id", &JsValue::from(id));
        set(&message, "words", &js_sys::Uint32Array::from(&job.to_words()[..]));
        let bytes = js_sys::Uint8Array::from(&data[..]);
        set(&message, "bytes", &bytes);
        let posted = p.workers[w].post_message_with_transfer(&message, &js_sys::Array::of1(&bytes.buffer()));
        if let Err(e) = posted {
            p.pending.borrow_mut().remove(&id);
            p.busy[w].set(p.busy[w].get().saturating_sub(1));
            let error = anyhow::anyhow!("could not send a chunk to a decode worker: {e:?}");
            return Box::pin(async move { Err(error) });
        }
        Box::pin(async move { receive.await.unwrap_or_else(|_| Err(anyhow::anyhow!("the decode worker went away"))) })
    }));
    Ok(())
}
