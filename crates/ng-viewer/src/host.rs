//! Loading for a host: a viewer state restored, its data sources fetched in
//! the background and handed to the viewer as they arrive.
//!
//! [`Viewer::restore_state`](crate::Viewer::restore_state) only lays the
//! layers out; their sources (and states named by URL) come over the network.
//! [`Loader`] runs those reads on the host's executor and gives the results
//! to the viewer in [`poll`](Loader::poll), on the thread that owns it. The
//! `neuroglancer` crate's `Engine` uses it, and so can any embedder that
//! drives a [`Viewer`](crate::Viewer) directly (scene mode, an offscreen
//! renderer).
use std::sync::{Arc, Mutex};

use ng_chunk::Executor;
use ng_datasource::model::DataSource;
use serde_json::Value;

use crate::{PendingSource, Viewer};

/// Results arriving from background work, picked up by [`Loader::poll`].
#[derive(Default)]
struct Inbox {
    sources: Vec<(PendingSource, anyhow::Result<DataSource>)>,
    states: Vec<anyhow::Result<Value>>,
}

/// Background loading of states and data sources for one viewer.
#[derive(Clone)]
pub struct Loader {
    executor: Arc<dyn Executor>,
    wake: Arc<dyn Fn() + Send + Sync>,
    inbox: Arc<Mutex<Inbox>>,
}

impl Loader {
    /// `wake` is called (from any thread) when something arrived: the host
    /// then calls [`poll`](Self::poll), or draws a frame that does.
    pub fn new(executor: Arc<dyn Executor>, wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self { executor, wake, inbox: Default::default() }
    }

    /// Restores a viewer state now (`viewer.state.reset(); restoreState`);
    /// its data sources load in the background, each showing as it arrives.
    pub fn restore_state(&self, viewer: &mut Viewer, state: &Value) -> anyhow::Result<()> {
        let pending = viewer.restore_state(state)?;
        for m in &viewer.messages {
            log::warn!("{m}");
        }
        self.load_sources(pending);
        Ok(())
    }

    /// Fetches a state (a share link `…#!{…}`, `#!gs://…/state.json`, JSON)
    /// in the background; [`poll`](Self::poll) restores it.
    pub fn restore_url(&self, arg: &str) {
        let arg = arg.to_owned();
        let inbox = self.inbox.clone();
        let wake = self.wake.clone();
        self.executor.spawn(Box::pin(async move {
            let state = crate::state::fetch_state(&arg).await;
            inbox.lock().unwrap().states.push(state);
            wake();
        }));
    }

    /// Loads data sources in the background; [`poll`](Self::poll) hands
    /// each to its layer.
    pub fn load_sources(&self, pending: Vec<PendingSource>) {
        for p in pending {
            let inbox = self.inbox.clone();
            let wake = self.wake.clone();
            self.executor.spawn(Box::pin(async move {
                let result = ng_datasource::url::get(&p.url).await;
                inbox.lock().unwrap().sources.push((p, result));
                wake();
            }));
        }
    }

    /// Takes in what background work finished (states, sources, chunks);
    /// whether a frame would show something new.
    pub fn poll(&self, viewer: &mut Viewer) -> bool {
        let Inbox { sources, states } = std::mem::take(&mut *self.inbox.lock().unwrap());
        let mut changed = !sources.is_empty() || !states.is_empty();
        for state in states {
            match state.and_then(|s| self.restore_state(viewer, &s)) {
                Ok(()) => {}
                Err(e) => log::error!("could not restore the state: {e:#}"),
            }
        }
        for (p, result) in sources {
            viewer.source_loaded(p.layer, p.index, result);
            if let Some(l) = viewer.layers().iter().find(|l| l.id == p.layer) {
                for m in &l.messages {
                    log::warn!("{}: {m}", l.name());
                }
            }
        }
        changed |= viewer.poll();
        changed
    }

    /// Every layer's sources loaded (or failed) and nothing downloading
    /// (`viewer.isReady()`).
    pub fn is_ready(viewer: &Viewer) -> bool {
        viewer.layers().iter().all(|l| l.is_ready()) && !viewer.is_loading()
    }
}
