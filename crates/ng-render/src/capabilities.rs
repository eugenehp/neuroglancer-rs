//! What the GPU can do that the renderers choose by: a downlevel adapter
//! (WebGL2 above all) has no compute shaders and blends every colour target
//! alike. Recorded once, from the adapter, before the renderers are made
//! ([`set_from_adapter`]); without that, a full WebGPU/native adapter is
//! assumed.

use std::sync::atomic::{AtomicBool, Ordering};

static INDEPENDENT_BLEND: AtomicBool = AtomicBool::new(true);
static COMPUTE: AtomicBool = AtomicBool::new(true);

/// Records what `adapter` supports.
pub fn set_from_adapter(adapter: &wgpu::Adapter) {
    let flags = adapter.get_downlevel_capabilities().flags;
    set(flags.contains(wgpu::DownlevelFlags::INDEPENDENT_BLEND), flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS));
}

/// Records the capabilities directly (tests, or a host that knows).
pub fn set(independent_blend: bool, compute: bool) {
    INDEPENDENT_BLEND.store(independent_blend, Ordering::Relaxed);
    COMPUTE.store(compute, Ordering::Relaxed);
}

/// Colour targets of one pipeline may blend (and mask) differently.
pub fn independent_blend() -> bool {
    INDEPENDENT_BLEND.load(Ordering::Relaxed)
}

/// Compute shaders and storage buffers.
pub fn compute() -> bool {
    COMPUTE.load(Ordering::Relaxed)
}
