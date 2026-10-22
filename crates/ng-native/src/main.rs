//! Desktop shell: `neuroglancer::winit`'s window around the engine.
//!
//! ```text
//! ng-native <data source>
//! ng-native --state <neuroglancer URL | state JSON | file.json>
//! ```
//!
//! The data source is anything neuroglancer accepts — `precomputed://gs://…`,
//! `zarr://https://…`, `https://…/x.zarr|zarr3:`, `nifti://…` — or a local path
//! or kvstore URL whose format is detected (NIfTI, precomputed, Zarr). A state
//! is neuroglancer's viewer state: a share link (`…#!{…}` or `#!gs://…`), its
//! JSON, or a file holding it.
//!
//! `--cache <dir>` (or `NG_CACHE_DIR`) keeps what is downloaded in `dir`, so a
//! dataset seen before loads from disk; data changed on the server since is
//! shown as cached.

use neuroglancer::winit::{Open, WindowOptions};

fn main() {
    env_logger::init();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--cache") {
        if i + 1 < args.len() {
            let dir = args.remove(i + 1);
            neuroglancer::kvstore::set_disk_cache(Some(dir.into()), None);
        }
        args.remove(i);
    }
    let open = match args.as_slice() {
        [flag, state] if flag == "--state" => Some(Open::State(state.clone())),
        [arg] if arg.contains("#!") => Some(Open::State(arg.clone())),
        [source] => Some(Open::Source(source.clone())),
        _ => {
            eprintln!("usage: ng-native [--cache <dir>] <data source URL or path> | --state <neuroglancer URL, JSON or file>");
            None
        }
    };
    neuroglancer::winit::run(open, WindowOptions::default());
}
