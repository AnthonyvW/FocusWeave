//! Detects the OpenCV major version, so the call sites whose signatures differ
//! between OpenCV 4 and 5 can be written once for each.
//!
//! The `opencv` crate knows the version but keeps it to itself: its build
//! script's `ocvrs_opencv_branch_*` cfgs apply to that crate alone, and it
//! declares no `links` key, so there is no `DEP_OPENCV_*` metadata to read.
//! Probing again here is cheaper than the alternatives.

use std::path::PathBuf;
use std::process::Command;

fn from_headers() -> Option<u32> {
    let paths = std::env::var("OPENCV_INCLUDE_PATHS").ok()?;
    for dir in paths.split(',') {
        let header = PathBuf::from(dir.trim()).join("opencv2/core/version.hpp");
        let Ok(text) = std::fs::read_to_string(&header) else {
            continue;
        };
        for line in text.lines() {
            if let Some(rest) = line.trim().strip_prefix("#define CV_VERSION_MAJOR") {
                if let Ok(major) = rest.trim().parse() {
                    return Some(major);
                }
            }
        }
    }
    None
}

fn from_pkg_config() -> Option<u32> {
    for package in ["opencv5", "opencv4"] {
        let Ok(output) = Command::new("pkg-config")
            .args(["--modversion", package])
            .output()
        else {
            return None;
        };
        if !output.status.success() {
            continue;
        }
        let version = String::from_utf8_lossy(&output.stdout);
        if let Ok(major) = version.trim().split('.').next().unwrap_or("").parse() {
            return Some(major);
        }
    }
    None
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=OPENCV_INCLUDE_PATHS");
    println!("cargo::rustc-check-cfg=cfg(opencv_algorithm_hint)");

    // Falling back to 4 keeps the signature that has been stable since 4.0;
    // guessing 5 on an unreadable install would break a build that works.
    let major = from_headers().or_else(from_pkg_config).unwrap_or(4);
    println!("cargo::warning=building against OpenCV {major}.x");
    if major >= 5 {
        println!("cargo::rustc-cfg=opencv_algorithm_hint");
    }
}
