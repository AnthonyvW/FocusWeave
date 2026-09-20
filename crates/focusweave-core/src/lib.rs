//! Focus stacking via Laplacian pyramid fusion.
//!
//! This crate is a self-contained port of the original Python/OpenCV
//! implementation. Every image-processing primitive it needs is implemented
//! here, so building it requires no system OpenCV installation.

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
pub mod pyramid;
pub mod resize;
pub mod stack;
pub mod streaming;
pub mod testio;
pub mod warp;
