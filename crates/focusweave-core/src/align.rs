//! Frame registration: per-pair ECC solves composed into per-image warps.

use crate::affine::{constrain_warp, Affine, WarpConstraints};
use crate::ecc::find_transform_ecc;
use crate::focus::{
    combine_masks, phase_correlation_translation, prepare_for_ecc, to_gray, Prepared,
};
use crate::hooks::{Error, Hooks, Stage};
use crate::image_source::Source;
use crate::mat::MatU8;
use rayon::prelude::*;

const FINE_RESOLUTION: usize = 1024;
const SEED_RESOLUTION: usize = 512;
const TRANSLATION_TOLERANCE: f64 = 0.15;
const AFFINE_DISTORTION_LIMIT: f64 = 0.02;

/// Termination settings for one ECC pass.
struct Pass {
    iterations: usize,
    eps: f64,
    gauss: usize,
}

impl Pass {
    fn fine(relaxed: bool) -> Pass {
        if relaxed {
            Pass {
                iterations: 100,
                eps: 0.005,
                gauss: 5,
            }
        } else {
            Pass {
                iterations: 50,
                eps: 0.001,
                gauss: 3,
            }
        }
    }
}

fn ecc_align(reference: &Prepared, source: &Prepared, init: Affine, pass: Pass) -> Option<Affine> {
    let mask = combine_masks(&reference.mask, &source.mask);
    let scale = reference.scale;
    let mut warp = init;
    if scale != 1.0 {
        warp.0[2] *= scale as f32;
        warp.0[5] *= scale as f32;
    }
    let solved = find_transform_ecc(
        &source.equalised,
        &reference.equalised,
        warp,
        pass.iterations,
        pass.eps,
        mask.as_ref(),
        pass.gauss,
    )
    .ok()?;
    let mut out = solved;
    out.0[2] /= scale as f32;
    out.0[5] /= scale as f32;
    Some(out)
}

/// Reject warps that are implausible for a focus stack frame.
///
/// Two independent checks: the ECC translation must not disagree wildly with
/// the phase-correlation seed, which is a far more reliable anchor on
/// low-texture subjects; and the linear block must stay close to the identity,
/// since the camera has not moved between frames.
fn validate_warp(warp: &Affine, seed: &Affine, image_long_edge: usize) -> bool {
    let dx = f64::from(warp.tx() - seed.tx());
    let dy = f64::from(warp.ty() - seed.ty());
    if dx.hypot(dy) > TRANSLATION_TOLERANCE * image_long_edge as f64 {
        return false;
    }
    warp.affine_distortion() <= AFFINE_DISTORTION_LIMIT
}

fn fine_resolution(gray: &MatU8, full_res: bool) -> usize {
    if full_res {
        usize::MAX
    } else {
        FINE_RESOLUTION.min(gray.h.max(gray.w))
    }
}

/// Register one pair, seeding ECC with a phase-correlation translation.
///
/// Returns the warp and whether it converged; a non-converged pair yields the
/// identity so the caller can fall back to the previous transform.
pub fn run_ecc(
    reference_gray: &MatU8,
    source_gray: &MatU8,
    reference_prepared: &Prepared,
    source_prepared: &Prepared,
) -> (Affine, bool) {
    let long_edge = reference_gray.h.max(reference_gray.w);
    let (tx, ty) = phase_correlation_translation(reference_gray, source_gray, SEED_RESOLUTION);
    let seed = Affine::translation(tx as f32, ty as f32);

    let warp = ecc_align(reference_prepared, source_prepared, seed, Pass::fine(false))
        .or_else(|| ecc_align(reference_prepared, source_prepared, seed, Pass::fine(true)));

    match warp {
        Some(w) if validate_warp(&w, &seed, long_edge) => (w, true),
        _ => (Affine::IDENTITY, false),
    }
}

/// How per-image warps are derived from the pairwise solves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignStrategy {
    /// Solve consecutive pairs and compose the results mathematically, so
    /// interpolation error never accumulates across the stack.
    NeighbourChained,
    /// Solve every frame directly against the reference. More robust when the
    /// frames are not ordered by similarity, more fragile across large gaps.
    Global,
}

#[derive(Clone, Copy, Debug)]
pub struct AlignOptions {
    pub strategy: AlignStrategy,
    pub constraints: WarpConstraints,
    pub full_res: bool,
    pub min_shift: f32,
    pub workers: usize,
}

impl Default for AlignOptions {
    fn default() -> Self {
        AlignOptions {
            strategy: AlignStrategy::NeighbourChained,
            constraints: WarpConstraints::default(),
            full_res: false,
            min_shift: 5.0,
            workers: 0,
        }
    }
}

fn warp_message(label: &str, warp: &Affine) -> String {
    format!(
        "{label}: rotation {:+.2}°  shift ({:+.1}, {:+.1}) px",
        warp.rotation_degrees(),
        warp.ty(),
        warp.tx()
    )
}

/// Compute an affine warp for every frame, relative to `reference_idx`.
///
/// The reference always receives the identity, and any frame whose cumulative
/// transform is negligible is snapped to the identity so it is copied rather
/// than resampled.
pub fn align_images(
    sources: &[Source],
    reference_size: (usize, usize),
    reference_idx: usize,
    options: AlignOptions,
    hooks: &Hooks,
) -> Result<Vec<Affine>, Error> {
    let n = sources.len();
    let to_align = n.saturating_sub(1).max(1);
    let mut warps = vec![Affine::IDENTITY; n];

    let reference_gray = to_gray(&sources[reference_idx], reference_size)?;
    let fine_res = fine_resolution(&reference_gray, options.full_res);

    // Loading and preparation carry no inter-frame dependencies, so they run
    // in parallel and keep the serial ECC chain off the critical path.
    let prepared: Vec<(MatU8, Prepared)> = {
        let results: Vec<Result<(MatU8, Prepared), Error>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let gray = if i == reference_idx {
                    reference_gray.clone()
                } else {
                    to_gray(&sources[i], reference_size)?
                };
                let prep = prepare_for_ecc(&gray, fine_res);
                Ok((gray, prep))
            })
            .collect();
        results.into_iter().collect::<Result<Vec<_>, Error>>()?
    };

    let constrain = |w: Affine| constrain_warp(&w, options.constraints);
    let is_negligible =
        |w: &Affine| w.translation_norm() < options.min_shift && w.is_pure_translation(1e-3);

    let mut aligned = 0usize;
    let notify = |aligned: usize, message: &str, hooks: &Hooks| -> Result<(), Error> {
        hooks.report(aligned as f64 / to_align as f64, Stage::Aligning, message);
        hooks.check()
    };

    match options.strategy {
        AlignStrategy::Global => {
            for i in 0..n {
                if i == reference_idx {
                    continue;
                }
                let (warp, converged) = run_ecc(
                    &reference_gray,
                    &prepared[i].0,
                    &prepared[reference_idx].1,
                    &prepared[i].1,
                );
                let warp = constrain(warp);
                let label = format!("Image {}", i + 1);
                let message = if !converged {
                    warps[i] = Affine::IDENTITY;
                    format!("{label}: ECC did not converge — using identity")
                } else if is_negligible(&warp) {
                    warps[i] = Affine::IDENTITY;
                    format!("{label}: transform negligible — skipped")
                } else {
                    warps[i] = warp;
                    warp_message(&label, &warp)
                };
                aligned += 1;
                notify(aligned, &message, hooks)?;
            }
        }
        AlignStrategy::NeighbourChained => {
            for direction in [1isize, -1] {
                let mut cumulative = Affine::IDENTITY;
                let mut prev = reference_idx;
                let mut i = reference_idx as isize + direction;
                while i >= 0 && (i as usize) < n {
                    let idx = i as usize;
                    let (warp, converged) = run_ecc(
                        &prepared[prev].0,
                        &prepared[idx].0,
                        &prepared[prev].1,
                        &prepared[idx].1,
                    );
                    let warp = constrain(warp);
                    prev = idx;
                    let label = format!("Image {}", idx + 1);
                    let message = if !converged {
                        warps[idx] = cumulative;
                        format!("{label}: ECC did not converge — using previous transform")
                    } else {
                        cumulative = constrain(cumulative.chain(&warp));
                        if options.constraints.no_rotation {
                            cumulative.0[0] = 1.0;
                            cumulative.0[1] = 0.0;
                            cumulative.0[3] = 0.0;
                            cumulative.0[4] = 1.0;
                        }
                        if is_negligible(&cumulative) {
                            warps[idx] = Affine::IDENTITY;
                            format!("{label}: transform negligible — skipped")
                        } else {
                            warps[idx] = cumulative;
                            warp_message(&label, &cumulative)
                        }
                    };
                    aligned += 1;
                    notify(aligned, &message, hooks)?;
                    i += direction;
                }
            }
        }
    }

    Ok(warps)
}
