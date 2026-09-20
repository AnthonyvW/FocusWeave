//! Incremental stacking for frames that arrive one at a time.

use crate::affine::{constrain_warp, Affine, WarpConstraints};
use crate::align::run_ecc;
use crate::border::Border;
use crate::color::{cv_round, rgb_to_gray_u8, rgb_to_lab_l_f32};
use crate::config::RunResult;
use crate::focus::{prepare_for_ecc, score_map_to_scalar, tenengrad_score_map, Prepared};
use crate::hooks::{Error, Hooks, Stage};
use crate::image_source::{load_u8, ImageBuf, Source};
use crate::mat::{Mat, MatU16, MatU8};
use crate::pyramid::{
    compute_levels, laplacian_pyramid, reconstruct, region_deviation, region_energy, region_entropy,
};
use crate::stack::{compute_canvas, slab_images, stack_images, SlabOutcome, StackOptions};
use crate::warp::{warp_affine_u16, warp_affine_u8, Interp};

#[derive(Clone, Debug)]
pub struct StreamingConfig {
    pub reference: i64,
    pub cull_threshold: Option<f64>,
    pub no_rotation: bool,
    pub no_scale: bool,
    pub no_shear: bool,
    pub no_translation: bool,
    pub full_res: bool,
    pub min_shift: f32,
    pub levels: usize,
    pub sharpness: f32,
    pub no_fill: bool,
    pub workers: usize,
    pub slab: Option<(usize, usize)>,
    pub only_slab: bool,
    pub recursive_slab: bool,
    /// Emit a downscaled partial stack after each frame, at this fraction of
    /// the reference size. `None` disables preview accumulation entirely.
    pub preview_scale: Option<f64>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        StreamingConfig {
            reference: -1,
            cull_threshold: None,
            no_rotation: false,
            no_scale: false,
            no_shear: false,
            no_translation: false,
            full_res: false,
            min_shift: 5.0,
            levels: 0,
            sharpness: 4.0,
            no_fill: false,
            workers: 0,
            slab: None,
            only_slab: false,
            recursive_slab: false,
            preview_scale: None,
        }
    }
}

impl StreamingConfig {
    fn constraints(&self) -> WarpConstraints {
        WarpConstraints {
            no_rotation: self.no_rotation,
            no_scale: self.no_scale,
            no_shear: self.no_shear,
            no_translation: self.no_translation,
        }
    }
}

/// Running sums for the incremental preview.
struct PreviewAccumulator {
    size: (usize, usize),
    scale: f64,
    levels: usize,
    depth: u32,
    energy: Vec<Option<Mat>>,
    weighted: Vec<Option<Mat>>,
    unweighted: Vec<Option<Mat>>,
    cumulative: Affine,
    count: usize,
}

/// Accept frames one at a time, scoring and registering each on arrival.
///
/// Frames are assumed to arrive in acquisition order. Culling and the pairwise
/// ECC solve against the previous frame both complete inside `add_image`, so
/// those costs overlap capture instead of landing in one block at the end.
pub struct StreamingFocusStacker {
    reference_size: (usize, usize),
    config: StreamingConfig,
    fine_resolution: usize,
    images: Vec<ImageBuf>,
    grays: Vec<MatU8>,
    prepared: Vec<Prepared>,
    pairwise: Vec<(Affine, bool)>,
    scores: Vec<f64>,
    preview: Option<PreviewAccumulator>,
}

impl StreamingFocusStacker {
    pub fn new(reference_size: (usize, usize), config: StreamingConfig) -> Self {
        let fine_resolution = if config.full_res {
            usize::MAX
        } else {
            1024.min(reference_size.0.max(reference_size.1))
        };
        StreamingFocusStacker {
            reference_size,
            config,
            fine_resolution,
            images: Vec::new(),
            grays: Vec::new(),
            prepared: Vec::new(),
            pairwise: Vec::new(),
            scores: Vec::new(),
            preview: None,
        }
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Add the next frame in the acquisition sequence.
    ///
    /// Returns the updated preview when preview accumulation is enabled.
    pub fn add_image(&mut self, image: ImageBuf) -> Result<Option<MatU8>, Error> {
        let (ref_w, ref_h) = self.reference_size;
        if image.width() != ref_w || image.height() != ref_h {
            return Err(Error::Config(format!(
                "Image size {}x{} does not match reference size {ref_w}x{ref_h}.",
                image.width(),
                image.height()
            )));
        }

        let source = Source::Array(image.clone());
        let score_map = tenengrad_score_map(&source, self.reference_size, 5, 1024)?;
        self.scores.push(score_map_to_scalar(&score_map));

        let gray = rgb_to_gray_u8(&load_u8(&source, self.reference_size)?);
        let prepared = prepare_for_ecc(&gray, self.fine_resolution);

        let mut step: Option<Affine> = None;
        if let Some(previous) = self.grays.last() {
            let (warp, converged) = run_ecc(
                previous,
                &gray,
                self.prepared.last().expect("prepared"),
                &prepared,
            );
            let warp = constrain_warp(&warp, self.config.constraints());
            self.pairwise.push((warp, converged));
            if converged {
                step = Some(warp);
            }
        }

        self.images.push(image);
        self.grays.push(gray);
        self.prepared.push(prepared);

        if self.config.preview_scale.is_some() {
            self.update_preview(step)?;
            return Ok(self.get_preview());
        }
        Ok(None)
    }

    /// The current partial stack, or `None` when previews are disabled.
    pub fn get_preview(&self) -> Option<MatU8> {
        self.preview.as_ref().and_then(reconstruct_preview)
    }

    /// Resolve the warp chain, apply culling, and run the final stack.
    pub fn finish(&self, keep_size: bool, crop: bool, hooks: &Hooks) -> Result<RunResult, Error> {
        let n = self.images.len();
        if n < 2 {
            return Err(Error::Config(format!("Need at least 2 images, got {n}.")));
        }

        let kept = self.apply_cull();
        let kept_indices: Vec<usize> = (0..n).filter(|i| kept[*i]).collect();

        let reference = if self.config.reference >= 0 {
            let r = self.config.reference as usize;
            if r >= n {
                return Err(Error::Config(format!(
                    "reference {} is out of range (0\u{2013}{}).",
                    self.config.reference,
                    n - 1
                )));
            }
            r
        } else {
            kept_indices[kept_indices.len() / 2]
        };

        let absolute = self.resolve_chain(reference);
        let sources: Vec<Source> = kept_indices
            .iter()
            .map(|i| Source::Array(self.images[*i].clone()))
            .collect();
        let warps: Vec<Affine> = kept_indices.iter().map(|i| absolute[*i]).collect();

        let levels = if self.config.levels > 0 {
            self.config.levels
        } else {
            compute_levels(self.reference_size.1, self.reference_size.0, 6)
        };
        let (canvas_size, adjusted) = compute_canvas(&warps, self.reference_size, keep_size, crop);
        let opts = StackOptions {
            levels,
            sharpness: self.config.sharpness,
            canvas_size: Some(canvas_size),
            no_fill: self.config.no_fill,
            workers: self.config.workers,
        };

        if let Some((slab_size, overlap)) = self.config.slab {
            if slab_size < 2 {
                return Err(Error::Config("Slab SIZE must be at least 2.".into()));
            }
            if overlap >= slab_size {
                return Err(Error::Config(format!(
                    "Slab OVERLAP must be >= 0 and < SIZE ({slab_size})."
                )));
            }
            let outcome = slab_images(
                &sources,
                &adjusted,
                slab_size,
                overlap,
                &opts,
                self.config.only_slab,
                self.config.recursive_slab && !self.config.only_slab,
                hooks,
            )?;
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

        let image = stack_images(&sources, &adjusted, &opts, hooks)?;
        Ok(RunResult {
            image: Some(image),
            slabs: None,
        })
    }

    fn apply_cull(&self) -> Vec<bool> {
        let n = self.images.len();
        let Some(threshold) = self.config.cull_threshold else {
            return vec![true; n];
        };
        let peak = self
            .scores
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if peak == 0.0 {
            return vec![true; n];
        }
        let mut kept: Vec<bool> = self.scores.iter().map(|s| *s >= threshold).collect();
        if kept.iter().filter(|k| **k).count() < 2 {
            let mut ranked: Vec<usize> = (0..n).collect();
            ranked.sort_by(|a, b| {
                self.scores[*b]
                    .partial_cmp(&self.scores[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            kept = vec![false; n];
            for idx in ranked.into_iter().take(2) {
                kept[idx] = true;
            }
        }
        kept
    }

    /// Turn the pairwise chain into warps relative to `reference`.
    ///
    /// The stored steps run forward, so frames before the reference are
    /// registered here rather than reusing an inverse.
    fn resolve_chain(&self, reference: usize) -> Vec<Affine> {
        let n = self.images.len();
        let mut absolute = vec![Affine::IDENTITY; n];
        let constraints = self.config.constraints();

        let mut cumulative = Affine::IDENTITY;
        for (i, slot) in absolute.iter_mut().enumerate().skip(reference + 1) {
            let (step, converged) = self.pairwise[i - 1];
            if !converged {
                *slot = cumulative;
                continue;
            }
            cumulative = cumulative.chain(&step);
            if self.config.no_rotation {
                cumulative.0[0] = 1.0;
                cumulative.0[1] = 0.0;
                cumulative.0[3] = 0.0;
                cumulative.0[4] = 1.0;
            }
            *slot = self.maybe_identity(cumulative);
        }

        cumulative = Affine::IDENTITY;
        for i in (0..reference).rev() {
            let (warp, converged) = run_ecc(
                &self.grays[i + 1],
                &self.grays[i],
                &self.prepared[i + 1],
                &self.prepared[i],
            );
            let warp = constrain_warp(&warp, constraints);
            if !converged {
                absolute[i] = cumulative;
                continue;
            }
            cumulative = cumulative.chain(&warp);
            if self.config.no_rotation {
                cumulative.0[0] = 1.0;
                cumulative.0[1] = 0.0;
                cumulative.0[3] = 0.0;
                cumulative.0[4] = 1.0;
            }
            absolute[i] = self.maybe_identity(cumulative);
        }

        absolute
    }

    fn maybe_identity(&self, warp: Affine) -> Affine {
        if warp.translation_norm() < self.config.min_shift && warp.is_pure_translation(1e-3) {
            Affine::IDENTITY
        } else {
            warp
        }
    }

    /// Fold one frame into the preview accumulator.
    ///
    /// Preview warps chain forward from the first frame rather than from the
    /// middle one `finish` picks. That is a different coordinate frame, but it
    /// yields a visually correct partial stack while frames are still arriving.
    fn update_preview(&mut self, step: Option<Affine>) -> Result<(), Error> {
        let scale = self.config.preview_scale.expect("preview enabled");
        let scale = scale.clamp(0.05, 1.0);
        let (ref_w, ref_h) = self.reference_size;
        let size = (
            (cv_round(ref_w as f64 * scale) as usize).max(2),
            (cv_round(ref_h as f64 * scale) as usize).max(2),
        );
        let image = self.images.last().expect("frame added").clone();
        let depth = image.depth();

        if self.preview.is_none() {
            let levels = if self.config.levels > 0 {
                self.config.levels
            } else {
                compute_levels(size.1, size.0, 6)
            };
            self.preview = Some(PreviewAccumulator {
                size,
                scale,
                levels,
                depth,
                energy: (0..=levels).map(|_| None).collect(),
                weighted: (0..=levels).map(|_| None).collect(),
                unweighted: (0..=levels).map(|_| None).collect(),
                cumulative: Affine::IDENTITY,
                count: 0,
            });
        }
        let accumulator = self.preview.as_mut().expect("initialised above");
        if let Some(step) = step {
            accumulator.cumulative = accumulator.cumulative.chain(&step);
        }

        let mut warp = accumulator.cumulative;
        warp.0[2] *= accumulator.scale as f32;
        warp.0[5] *= accumulator.scale as f32;

        let border = if self.config.no_fill {
            Border::Constant
        } else {
            Border::Reflect
        };
        let interp = if warp.is_pure_translation(1e-4) {
            Interp::Linear
        } else {
            Interp::Cubic
        };
        let source = Source::Array(image);

        let (rgb, lab_l) = if depth == 16 {
            let native = crate::image_source::load_native_f32(&source, accumulator.size)?;
            let native = if warp.is_identity() {
                native
            } else {
                let clamped = MatU16 {
                    h: native.h,
                    w: native.w,
                    c: native.c,
                    data: native
                        .data
                        .iter()
                        .map(|v| v.clamp(0.0, 65535.0) as u16)
                        .collect(),
                };
                let warped = warp_affine_u16(
                    &clamped,
                    &warp,
                    accumulator.size.0,
                    accumulator.size.1,
                    interp,
                    border,
                );
                Mat {
                    h: warped.h,
                    w: warped.w,
                    c: warped.c,
                    data: warped.data.iter().map(|v| f32::from(*v)).collect(),
                }
            };
            let as_u8 = MatU8 {
                h: native.h,
                w: native.w,
                c: native.c,
                data: native
                    .data
                    .iter()
                    .map(|v| (v * (255.0 / 65535.0)).clamp(0.0, 255.0) as u8)
                    .collect(),
            };
            let l = rgb_to_lab_l_f32(&as_u8);
            (native, l)
        } else {
            let img = load_u8(&source, accumulator.size)?;
            let img = if warp.is_identity() {
                img
            } else {
                warp_affine_u8(
                    &img,
                    &warp,
                    accumulator.size.0,
                    accumulator.size.1,
                    interp,
                    border,
                )
            };
            let l = rgb_to_lab_l_f32(&img);
            let f = Mat {
                h: img.h,
                w: img.w,
                c: img.c,
                data: img.data.iter().map(|v| f32::from(*v)).collect(),
            };
            (f, l)
        };

        let levels = accumulator.levels;
        let lab_bands = laplacian_pyramid(&lab_l, levels);
        let pixel_bands = laplacian_pyramid(&rgb, levels);

        for (i, band) in pixel_bands.into_iter().enumerate() {
            let energy = if i < levels {
                region_energy(&lab_bands[i], 3)
            } else {
                let residual = &lab_bands[i];
                let deviation = region_deviation(residual, 3);
                let entropy = region_entropy(residual, 8);
                let mut combined = Mat::new(residual.h, residual.w, 1);
                for j in 0..combined.data.len() {
                    combined.data[j] = (deviation.data[j] + entropy.data[j]) * 0.5;
                }
                combined
            };
            let energy = energy.map_into(|v| v.powf(self.config.sharpness));
            let mut weighted = Mat::new(band.h, band.w, band.c);
            for p in 0..band.h * band.w {
                let e = energy.data[p];
                for ch in 0..band.c {
                    weighted.data[p * band.c + ch] = band.data[p * band.c + ch] * e;
                }
            }
            add_into(&mut accumulator.energy[i], energy);
            add_into(&mut accumulator.weighted[i], weighted);
            add_into(&mut accumulator.unweighted[i], band);
        }
        accumulator.count += 1;
        Ok(())
    }
}

fn add_into(slot: &mut Option<Mat>, value: Mat) {
    match slot.as_mut() {
        Some(acc) => acc.add_assign(&value),
        None => *slot = Some(value),
    }
}

fn reconstruct_preview(accumulator: &PreviewAccumulator) -> Option<MatU8> {
    accumulator.energy.first()?.as_ref()?;
    let count = accumulator.count.max(1) as f32;
    let mut bands = Vec::with_capacity(accumulator.levels + 1);
    for i in 0..=accumulator.levels {
        let energy = accumulator.energy[i].as_ref()?;
        let weighted = accumulator.weighted[i].as_ref()?;
        let unweighted = accumulator.unweighted[i].as_ref()?;
        let mut band = Mat::new(weighted.h, weighted.w, weighted.c);
        let c = band.c;
        for p in 0..band.h * band.w {
            let e = energy.data[p];
            let alpha = (((e + 1e-40).log10() + 20.0) / 12.0).clamp(0.0, 1.0);
            let denom = e + 1e-40;
            for ch in 0..c {
                let idx = p * c + ch;
                band.data[idx] = alpha * (weighted.data[idx] / denom)
                    + (1.0 - alpha) * (unweighted.data[idx] / count);
            }
        }
        bands.push(band);
    }
    let image = reconstruct(&bands);
    let divisor = if accumulator.depth == 16 { 257.0 } else { 1.0 };
    Some(MatU8 {
        h: image.h,
        w: image.w,
        c: image.c,
        data: image
            .data
            .iter()
            .map(|v| (v / divisor).clamp(0.0, 255.0) as u8)
            .collect(),
    })
}

/// Stage label used when a streaming run reports progress.
pub const STREAMING_STAGE: Stage = Stage::Stacking;
