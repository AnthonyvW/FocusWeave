//! Progress reporting, cancellation and slab callbacks.

use crate::image_source::{ImageBuf, LoadError};

/// Which part of the pipeline is currently running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Loading,
    Culling,
    Aligning,
    Stacking,
    Slabbing,
    Complete,
}

impl Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Loading => "loading",
            Stage::Culling => "culling",
            Stage::Aligning => "aligning",
            Stage::Stacking => "stacking",
            Stage::Slabbing => "slabbing",
            Stage::Complete => "complete",
        }
    }
}

/// Reports how far through the run we are, which stage is active, and a
/// human-readable note about the step that just finished.
pub type ProgressFn<'a> = &'a dyn Fn(f64, Stage, &str);
/// Returns true to stop the run at the next checkpoint.
pub type InterruptFn<'a> = &'a dyn Fn() -> bool;
/// Receives each completed slab so the caller can persist it.
pub type SlabFn<'a> = &'a dyn Fn(&str, &ImageBuf);

/// Callbacks supplied by the caller.
///
/// All of them are invoked from the thread that drives the pipeline, never
/// from a worker, so implementations need not be thread safe.
#[derive(Clone, Copy, Default)]
pub struct Hooks<'a> {
    pub progress: Option<ProgressFn<'a>>,
    pub interrupt: Option<InterruptFn<'a>>,
    pub on_slab: Option<SlabFn<'a>>,
}

impl<'a> Hooks<'a> {
    pub fn report(&self, fraction: f64, stage: Stage, message: &str) {
        if let Some(f) = self.progress {
            f(fraction, stage, message);
        }
    }

    pub fn check(&self) -> Result<(), Error> {
        match self.interrupt {
            Some(f) if f() => Err(Error::Interrupted),
            _ => Ok(()),
        }
    }

    pub fn slab(&self, label: &str, image: &ImageBuf) {
        if let Some(f) = self.on_slab {
            f(label, image);
        }
    }
}

/// Anything that can stop a run.
#[derive(Debug)]
pub enum Error {
    Load(LoadError),
    Config(String),
    Interrupted,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Load(e) => write!(f, "{e}"),
            Error::Config(m) => f.write_str(m),
            Error::Interrupted => f.write_str("Interrupted"),
        }
    }
}

impl std::error::Error for Error {}

impl From<LoadError> for Error {
    fn from(e: LoadError) -> Self {
        Error::Load(e)
    }
}
