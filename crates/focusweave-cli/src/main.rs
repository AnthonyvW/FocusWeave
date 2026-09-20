//! Thin wrapper around the shared CLI implementation.

use std::process::ExitCode;

/// The pipeline allocates and frees multi-megabyte scratch buffers on every
/// pyramid level. glibc services those with `mmap` and returns them to the
/// kernel immediately, so each one is re-faulted page by page on first write.
/// An allocator that caches large blocks avoids paying that repeatedly.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match focusweave_core::cli::run_cli(&argv) {
        0 => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    }
}
