//! Focus stacking via Laplacian pyramid fusion.
//!
//! With the `opencv-backend` feature the native implementations are still
//! compiled but never called, so that both backends live in one tree and can
//! be benchmarked against each other without a second checkout.
#![cfg_attr(feature = "opencv-backend", allow(dead_code))]

//!
//! This crate is a self-contained port of the original Python/OpenCV
//! implementation. Every image-processing primitive it needs is implemented
//! here, so building it requires no system OpenCV installation.

#[cfg(feature = "opencv-backend")]
pub mod backend_opencv;

/// Which set of image-processing kernels this build was compiled with.
pub const BACKEND: &str = if cfg!(feature = "opencv-backend") {
    "opencv"
} else {
    "native"
};

pub mod affine;
pub mod align;
pub mod border;
pub mod clahe;
pub mod cli;
pub mod color;
pub mod config;
pub mod ecc;
pub mod fft;
pub mod filter;
pub mod focus;
pub mod hooks;
pub mod image_source;
pub mod mat;
pub mod memory;
pub mod pyramid;
pub mod resize;
pub mod simd;
pub mod stack;
pub mod streaming;
pub mod testio;
pub mod warp;
