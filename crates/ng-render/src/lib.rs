//! Rendering for neuroglancer-rs on threers' device. Volumes and
//! cross-sections are `threers::volume`; this crate adds what neuroglancer
//! needs on top: panel compositing, panel backgrounds, and its image- and
//! segmentation-layer shading.

pub mod annotation_shader;
pub mod capabilities;
pub mod annotations;
pub mod compose;
pub mod fill;
pub mod image;
pub mod lines;
pub mod perspective;
pub mod pick;
pub mod scale_bar;
pub mod segmentation;
pub mod transfer_function;
pub mod user_shader;
