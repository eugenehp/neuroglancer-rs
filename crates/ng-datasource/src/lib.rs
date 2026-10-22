//! Data sources: turning stored data into volumes with neuroglancer's
//! semantics. Each module ports one of `src/datasource/*`.

pub mod annotation;
pub mod chunk;
pub mod decode;
pub mod deepzoom;
pub mod jpeg;
pub mod mesh;
pub mod multiscale_mesh;
pub mod n5;
pub mod model;
pub mod nifti;
pub mod precomputed;
pub mod sharded;
pub mod skeleton;
pub mod url;
pub mod zarr;
