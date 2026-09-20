//! Run configuration and the top-level pipeline.

use crate::affine::{Affine, WarpConstraints};
use crate::align::{align_images, AlignOptions, AlignStrategy};
use crate::focus::cull_unfocused;
use crate::hooks::{Error, Hooks, Stage};
use crate::image_source::{image_size, list_folder, ImageBuf, Source};
use crate::pyramid::compute_levels;
use crate::stack::{compute_canvas, slab_images, stack_images, SlabOutcome, StackOptions};
use std::path::PathBuf;

/// Where the frames come from.
#[derive(Clone, Debug)]
pub enum Images {
    /// A folder to scan for supported image files.
    Folder(PathBuf),
    /// An explicit, ordered list of files.
    Paths(Vec<PathBuf>),
    /// Frames already in memory.
    Arrays(Vec<ImageBuf>),
}

#[derive(Clone, Debug)]
pub struct FocusStackConfig {
    pub images: Images,
    pub no_align: bool,
    pub keep_size: bool,
    pub crop: bool,
    pub no_fill: bool,
    /// Index of the alignment reference; negative selects the middle frame.
    pub reference: i64,
    pub cull: Option<f64>,
    pub global_align: bool,
    pub no_rotation: bool,
    pub no_scale: bool,
    pub no_shear: bool,
    pub no_translation: bool,
    pub full_res: bool,
    pub min_shift: f32,
    /// Pyramid levels; zero selects a value from the image size.
    pub levels: usize,
    pub sharpness: f32,
    pub workers: usize,
    pub slab: Option<(usize, usize)>,
    pub only_slab: bool,
    pub recursive_slab: bool,
}

impl FocusStackConfig {
    pub fn new(images: Images) -> Self {
        FocusStackConfig {
            images,
            no_align: false,
            keep_size: false,
            crop: false,
            no_fill: false,
            reference: -1,
            cull: None,
            global_align: false,
            no_rotation: false,
            no_scale: false,
            no_shear: false,
            no_translation: false,
            full_res: false,
            min_shift: 5.0,
            levels: 0,
            sharpness: 4.0,
            // Zero is automatic: every core, capped to fit in memory.
            workers: 0,
            slab: None,
            only_slab: false,
            recursive_slab: false,
        }
    }

    fn constraints(&self) -> WarpConstraints {
        WarpConstraints {
            no_rotation: self.no_rotation,
            no_scale: self.no_scale,
            no_shear: self.no_shear,
            no_translation: self.no_translation,
        }
    }
}

#[derive(Debug)]
pub struct RunResult {
    /// The stacked image, or `None` when only slabs were requested.
    pub image: Option<ImageBuf>,
    /// The intermediate slabs, present only when `only_slab` was set.
    pub slabs: Option<Vec<ImageBuf>>,
}

/// Resolve the configured images into sources plus the reference frame size.
pub fn resolve_images(images: &Images) -> Result<(Vec<Source>, (usize, usize)), Error> {
    let sources: Vec<Source> = match images {
        Images::Folder(folder) => {
            let paths = list_folder(folder)?;
            if paths.len() < 2 {
                return Err(Error::Config(format!(
                    "Need at least 2 images in '{}', found {}.",
                    folder.display(),
                    paths.len()
                )));
            }
            paths.into_iter().map(Source::Path).collect()
        }
        Images::Paths(paths) => paths.iter().cloned().map(Source::Path).collect(),
        Images::Arrays(arrays) => arrays.iter().cloned().map(Source::Array).collect(),
    };
    if sources.len() < 2 {
        return Err(Error::Config(format!(
            "Need at least 2 images, got {}.",
            sources.len()
        )));
    }
    let size = match &sources[0] {
        Source::Path(p) => image_size(p)?,
        Source::Array(a) => (a.width(), a.height()),
    };
    Ok((sources, size))
}

/// Run the full pipeline.
///
/// Progress is reported across the whole run: roughly 5% loading and culling,
/// 25% alignment and 65% stacking.
pub fn run(cfg: &FocusStackConfig, hooks: &Hooks) -> Result<RunResult, Error> {
    hooks.report(0.0, Stage::Loading, "Loading images...");
    let (mut sources, reference_size) = resolve_images(&cfg.images)?;

    if let Some(threshold) = cfg.cull {
        hooks.report(0.02, Stage::Culling, "Culling unfocused images...");
        let relay =
            |f: f64, stage: Stage, message: &str| hooks.report(0.02 + f * 0.03, stage, message);
        let cull_hooks = Hooks {
            progress: Some(&relay),
            interrupt: hooks.interrupt,
            on_slab: None,
        };
        let result = cull_unfocused(&sources, reference_size, threshold, &cull_hooks)?;
        for entry in &result.entries {
            let status = if entry.kept { "keep" } else { "CULL" };
            hooks.report(
                0.02,
                Stage::Culling,
                &format!(
                    "  [{status}] {}  (score={:.4}, cutoff={:.4})",
                    entry.label, entry.score, result.cutoff
                ),
            );
        }
        let kept = result.kept_indices();
        hooks.report(
            0.05,
            Stage::Culling,
            &format!(
                "Culled {}/{} image(s); {} frame(s) remaining.",
                result.n_culled,
                result.entries.len(),
                kept.len()
            ),
        );
        sources = kept.into_iter().map(|i| sources[i].clone()).collect();
    }

    let n_images = sources.len();
    let reference = if cfg.reference >= 0 {
        let r = cfg.reference as usize;
        if r >= n_images {
            return Err(Error::Config(format!(
                "reference {} is out of range (0\u{2013}{}).",
                cfg.reference,
                n_images - 1
            )));
        }
        r
    } else {
        n_images / 2
    };

    let levels = if cfg.levels > 0 {
        cfg.levels
    } else {
        compute_levels(reference_size.1, reference_size.0, 6)
    };

    let warps = if cfg.no_align {
        hooks.report(0.08, Stage::Aligning, "Skipping alignment.");
        vec![Affine::IDENTITY; n_images]
    } else {
        let strategy = if cfg.global_align {
            AlignStrategy::Global
        } else {
            AlignStrategy::NeighbourChained
        };
        let name = if cfg.global_align {
            "global"
        } else {
            "neighbour-chained"
        };
        hooks.report(
            0.08,
            Stage::Aligning,
            &format!("Aligning ({name}, reference image {})...", reference + 1),
        );
        let relay =
            |f: f64, stage: Stage, message: &str| hooks.report(0.08 + f * 0.22, stage, message);
        let align_hooks = Hooks {
            progress: Some(&relay),
            interrupt: hooks.interrupt,
            on_slab: None,
        };
        align_images(
            &sources,
            reference_size,
            reference,
            AlignOptions {
                strategy,
                constraints: cfg.constraints(),
                full_res: cfg.full_res,
                min_shift: cfg.min_shift,
                workers: cfg.workers,
            },
            &align_hooks,
        )?
    };

    let (canvas_size, adjusted) = compute_canvas(&warps, reference_size, cfg.keep_size, cfg.crop);
    let opts = StackOptions {
        levels,
        sharpness: cfg.sharpness,
        canvas_size: Some(canvas_size),
        no_fill: cfg.no_fill,
        workers: cfg.workers,
    };

    let relay = |f: f64, stage: Stage, message: &str| hooks.report(0.30 + f * 0.65, stage, message);
    let stack_hooks = Hooks {
        progress: Some(&relay),
        interrupt: hooks.interrupt,
        on_slab: hooks.on_slab,
    };

    if let Some((slab_size, overlap)) = cfg.slab {
        if slab_size < 2 {
            return Err(Error::Config("Slab SIZE must be at least 2.".into()));
        }
        if overlap >= slab_size {
            return Err(Error::Config(format!(
                "Slab OVERLAP must be >= 0 and < SIZE ({slab_size})."
            )));
        }
        hooks.report(0.30, Stage::Slabbing, "Slabbing...");
        let outcome = slab_images(
            &sources,
            &adjusted,
            slab_size,
            overlap,
            &opts,
            cfg.only_slab,
            cfg.recursive_slab && !cfg.only_slab,
            &stack_hooks,
        )?;
        hooks.report(1.0, Stage::Complete, "Complete");
        return Ok(match outcome {
            SlabOutcome::Slabs(slabs) => RunResult {
                image: None,
                slabs: Some(slabs),
            },
            SlabOutcome::Fused(image) => RunResult {
                image: Some(image),
                slabs: None,
            },
        });
    }

    hooks.report(0.30, Stage::Stacking, "Stacking...");
    let image = stack_images(&sources, &adjusted, &opts, &stack_hooks)?;
    hooks.report(1.0, Stage::Complete, "Complete");
    Ok(RunResult {
        image: Some(image),
        slabs: None,
    })
}
