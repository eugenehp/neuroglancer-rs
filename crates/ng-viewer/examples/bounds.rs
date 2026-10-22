//! Prints each layer's data bounds in global coordinates for a viewer state
//! (a file, or `-` for stdin): where neuroglancer puts the data.
//!
//!   cargo run -p ng-viewer --example bounds -- state.json [name=file …]
//!
//! Each `name=file` puts that file's bytes at `memory://name` first.
use std::io::Read;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let arg = std::env::args().nth(1).unwrap_or_else(|| "-".into());
    for put in std::env::args().skip(2) {
        let (name, path) = put.split_once('=').ok_or_else(|| anyhow::anyhow!("expected name=file, got {put}"))?;
        ng_kvstore::memory().insert(name, std::fs::read(path)?);
    }
    let mut text = String::new();
    if arg == "-" { std::io::stdin().read_to_string(&mut text)?; } else { text = std::fs::read_to_string(&arg)?; }
    let state: serde_json::Value = serde_json::from_str(&text)?;
    let gpu = threers::gpu::testing::context("bounds").ok_or_else(|| anyhow::anyhow!("no GPU"))?;
    let pool: Arc<dyn ng_chunk::Executor> = Arc::new(ng_chunk::ThreadPool::new(4));
    let mut viewer = ng_viewer::Viewer::new(gpu.device.clone(), gpu.queue.clone(), wgpu::TextureFormat::Rgba8Unorm, 64, 64, 1.0, pool.clone(), Arc::new(|| {}));
    let loader = ng_viewer::host::Loader::new(pool, Arc::new(|| {}));
    loader.restore_state(&mut viewer, &state)?;
    let start = std::time::Instant::now();
    while !ng_viewer::host::Loader::is_ready(&viewer) && start.elapsed().as_secs() < 60 {
        loader.poll(&mut viewer);
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let info = viewer.navigation().render_info().clone();
    println!("global dims {:?}, voxel physical scales {:?}", info.global_dimension_names, info.voxel_physical_scales);
    println!("position {:?}", viewer.navigation().position());
    for l in viewer.layers() {
        for (lo, hi) in l.data_bounds_global(info.global_rank) {
            println!("{}: {:?} .. {:?}", l.name(), lo.iter().map(|v| (v / 1e6 * 100.0).round() / 100.0).collect::<Vec<_>>(), hi.iter().map(|v| (v / 1e6 * 100.0).round() / 100.0).collect::<Vec<_>>());
        }
        for m in &l.messages { println!("  message: {m}"); }
    }
    Ok(())
}
