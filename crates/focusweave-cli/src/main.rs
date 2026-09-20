//! Thin wrapper around the shared CLI implementation.

use std::process::ExitCode;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match focusweave_core::cli::run_cli(&argv) {
        0 => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    }
}
