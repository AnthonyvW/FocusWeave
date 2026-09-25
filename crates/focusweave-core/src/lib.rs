//! Focus stacking via Laplacian pyramid fusion.
//!
//! Image processing is OpenCV's `imgproc`; everything above it — registration,
//! pyramid fusion, culling, slabbing — is implemented here.
//!
//! The algorithm is a port of the original Python implementation; only the
//! primitives underneath it are shared with it, by calling the same library.

pub mod affine;
pub mod align;
pub mod border;
pub mod cli;
pub mod config;
pub mod cv;
pub mod ecc;
pub mod fft;
pub mod focus;
pub mod hooks;
pub mod image_source;
pub mod mat;
pub mod memory;
pub mod pyramid;
pub mod stack;
pub mod streaming;
