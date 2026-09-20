//! Canvas layout, Laplacian pyramid fusion, and slabbed stacking.

use crate::affine::Affine;
use crate::border::Border;
use crate::cv::{rgb_to_lab_l_f32, warp_affine_u16, warp_affine_u8, Interp};
use crate::hooks::{Error, Hooks, Stage};
use crate::image_source::{load_native_f32, load_u8, source_depth, source_size, ImageBuf, Source};
use crate::mat::{Mat, MatU16, MatU8};
use crate::pyramid::{
    laplacian_pyramid, reconstruct, region_deviation, region_energy, region_entropy,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

const ENERGY_WINDOW: usize = 3;
const ENTROPY_WINDOW: usize = 8;

/// Output canvas size and the warps adjusted to land on it.
///
/// By default the canvas grows to the full extent of every transformed corner,
/// so no pixel of any frame is discarded. `crop` instead tightens to the
/// intersection of all extents, removing border fill at the cost of size, and
/// `keep_size` leaves the canvas at the source size.
pub fn compute_canvas(
    warps: &[Affine],
    src_size: (usize, usize),
    keep_size: bool,
    crop: bool,
) -> ((usize, usize), Vec<Affine>) {
    if keep_size {
        return (src_size, warps.to_vec());
    }
    let (w, h) = (src_size.0 as f64, src_size.1 as f64);
    let corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];

    let transformed: Vec<[(f64, f64); 4]> = warps
        .iter()
        .map(|m| {
            let mut out = [(0.0, 0.0); 4];
            for (slot, (cx, cy)) in out.iter_mut().zip(corners.iter()) {
                *slot = m.apply(*cx, *cy);
            }
            out
        })
        .collect();

    let mut canvas_min = (f64::INFINITY, f64::INFINITY);
    let mut canvas_max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for pts in &transformed {
        for (x, y) in pts {
            canvas_min.0 = canvas_min.0.min(*x);
            canvas_min.1 = canvas_min.1.min(*y);
            canvas_max.0 = canvas_max.0.max(*x);
            canvas_max.1 = canvas_max.1.max(*y);
        }
    }

    if crop {
        let mut crop_min = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut crop_max = (f64::INFINITY, f64::INFINITY);
        for pts in &transformed {
            let mut lo = (f64::INFINITY, f64::INFINITY);
            let mut hi = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for (x, y) in pts {
                lo.0 = lo.0.min(*x);
                lo.1 = lo.1.min(*y);
                hi.0 = hi.0.max(*x);
                hi.1 = hi.1.max(*y);
            }
            crop_min.0 = crop_min.0.max(lo.0 - canvas_min.0);
            crop_min.1 = crop_min.1.max(lo.1 - canvas_min.1);
            crop_max.0 = crop_max.0.min(hi.0 - canvas_min.0);
            crop_max.1 = crop_max.1.min(hi.1 - canvas_min.1);
        }
        if crop_max.0 > crop_min.0 && crop_max.1 > crop_min.1 {
            canvas_max = (canvas_min.0 + crop_max.0, canvas_min.1 + crop_max.1);
            canvas_min = (canvas_min.0 + crop_min.0, canvas_min.1 + crop_min.1);
        }
    }

    let shift = Affine::translation(-canvas_min.0 as f32, -canvas_min.1 as f32);
    let adjusted: Vec<Affine> = warps.iter().map(|m| shift.chain(m)).collect();
    let canvas_w = ((canvas_max.0 - canvas_min.0).ceil() as usize).max(1);
    let canvas_h = ((canvas_max.1 - canvas_min.1).ceil() as usize).max(1);
    ((canvas_w, canvas_h), adjusted)
}

#[derive(Clone, Copy, Debug)]
pub struct StackOptions {
    pub levels: usize,
    pub sharpness: f32,
    pub canvas_size: Option<(usize, usize)>,
    pub no_fill: bool,
    pub workers: usize,
}

impl Default for StackOptions {
    fn default() -> Self {
        StackOptions {
            levels: 5,
            sharpness: 4.0,
            canvas_size: None,
            no_fill: false,
            // Zero is automatic; see resolve_workers.
            workers: 0,
        }
    }
}

/// Per-worker accumulation of the fusion sums.
struct Partial {
    energy: Vec<Option<Mat>>,
    weighted: Vec<Option<Mat>>,
    unweighted: Vec<Option<Mat>>,
    count: usize,
}

impl Partial {
    fn new(levels: usize) -> Partial {
        Partial {
            energy: (0..=levels).map(|_| None).collect(),
            weighted: (0..=levels).map(|_| None).collect(),
            unweighted: (0..=levels).map(|_| None).collect(),
            count: 0,
        }
    }

    fn merge(&mut self, other: Partial) {
        for i in 0..self.energy.len() {
            accumulate(&mut self.energy[i], other.energy[i].clone());
            accumulate(&mut self.weighted[i], other.weighted[i].clone());
            accumulate(&mut self.unweighted[i], other.unweighted[i].clone());
        }
        self.count += other.count;
    }
}

fn accumulate(slot: &mut Option<Mat>, value: Option<Mat>) {
    match (slot.as_mut(), value) {
        (Some(acc), Some(v)) => acc.add_assign(&v),
        (None, Some(v)) => *slot = Some(v),
        _ => {}
    }
}

/// Peak bytes a worker holds per canvas pixel while fusing one frame.
///
/// Measured, not derived: a worker carries its share of the accumulators plus
/// the transient pyramids for the frame in flight. 16-bit sources keep a wider
/// copy of the loaded frame on the way in.
fn bytes_per_pixel_per_worker(depth: u32) -> u64 {
    if depth == 16 {
        160
    } else {
        110
    }
}

/// Decide how many frames to fuse concurrently.
///
/// Zero means automatic: every core, but capped so the workers' buffers fit in
/// memory the machine actually has free. Coarse parallelism over frames is
/// what scales here — much of fusing a frame is per-pixel work that no
/// individual kernel parallelises — so the cap is the only thing standing
/// between a wide machine and a large image set going to swap.
fn resolve_workers(requested: usize, n: usize, canvas: (usize, usize), depth: u32) -> usize {
    if requested > 0 {
        return requested.min(n).max(1);
    }
    let cores = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(4);

    let per_worker = bytes_per_pixel_per_worker(depth) * (canvas.0 as u64) * (canvas.1 as u64);
    // Leave most of free memory alone: the caller may be holding the source
    // frames, and going to swap costs far more than a missing worker saves.
    const FALLBACK_BUDGET: u64 = 4 << 30;
    let budget = crate::memory::available_bytes()
        .map(|free| free / 2)
        .unwrap_or(FALLBACK_BUDGET);
    let by_memory = (budget / per_worker.max(1)).max(1) as usize;

    cores.min(by_memory).min(n).max(1)
}

/// Fuse one frame's pyramid bands into a worker's running sums.
fn fuse_one(
    src: &Source,
    warp: &Affine,
    canvas: (usize, usize),
    depth: u32,
    opts: &StackOptions,
    partial: &mut Partial,
) -> Result<(), Error> {
    let border = if opts.no_fill {
        Border::Constant
    } else {
        Border::Reflect
    };
    let interp = if warp.is_pure_translation(1e-4) {
        Interp::Linear
    } else {
        Interp::Cubic
    };
    let levels = opts.levels;

    let (rgb, lab_l) = if depth == 16 {
        let native = load_native_f32(src, canvas)?;
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
            let warped = warp_affine_u16(&clamped, warp, canvas.0, canvas.1, interp, border);
            Mat {
                h: warped.h,
                w: warped.w,
                c: warped.c,
                data: warped.data.iter().map(|v| f32::from(*v)).collect(),
            }
        };
        // OpenCV's Lab conversion is 8-bit only, so the frame is scaled into
        // that range first, truncating exactly as the reference does.
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
        let img = load_u8(src, canvas)?;
        let img = if warp.is_identity() {
            img
        } else {
            warp_affine_u8(&img, warp, canvas.0, canvas.1, interp, border)
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

    let lab_bands = laplacian_pyramid(&lab_l, levels);
    let pixel_bands = laplacian_pyramid(&rgb, levels);

    for (i, band) in pixel_bands.into_iter().enumerate() {
        let energy = if i < levels {
            region_energy(&lab_bands[i], ENERGY_WINDOW)
        } else {
            let residual = &lab_bands[i];
            let deviation = region_deviation(residual, ENERGY_WINDOW);
            let entropy = region_entropy(residual, ENTROPY_WINDOW);
            let mut combined = Mat::new(residual.h, residual.w, 1);
            for j in 0..combined.data.len() {
                combined.data[j] = (deviation.data[j] + entropy.data[j]) * 0.5;
            }
            combined
        };
        let energy = energy.map_into(|v| v.powf(opts.sharpness));

        let mut weighted = Mat::new(band.h, band.w, band.c);
        for p in 0..band.h * band.w {
            let e = energy.data[p];
            for ch in 0..band.c {
                weighted.data[p * band.c + ch] = band.data[p * band.c + ch] * e;
            }
        }

        accumulate(&mut partial.energy[i], Some(energy));
        accumulate(&mut partial.weighted[i], Some(weighted));
        accumulate(&mut partial.unweighted[i], Some(band));
    }
    partial.count += 1;
    Ok(())
}

/// Fuse a set of frames into a single image.
///
/// Energy-weighted bands and their energies are accumulated in one pass per
/// frame and divided at the end, which is equivalent to normalising per frame
/// but reads each image only once.
pub fn stack_images(
    sources: &[Source],
    warps: &[Affine],
    opts: &StackOptions,
    hooks: &Hooks,
) -> Result<ImageBuf, Error> {
    assert_eq!(sources.len(), warps.len(), "one warp per source");
    let n = sources.len();
    let depth = source_depth(&sources[0])?;
    let max_val = if depth == 16 { 65535.0f32 } else { 255.0 };
    let canvas = match opts.canvas_size {
        Some(size) => size,
        None => source_size(&sources[0])?,
    };
    let workers = resolve_workers(opts.workers, n, canvas, depth);
    let levels = opts.levels;

    hooks.report(
        0.0,
        Stage::Stacking,
        &format!("Fusing ({workers} workers)..."),
    );

    let chunk = n.div_ceil(workers);
    let batches: Vec<(usize, usize)> = (0..n)
        .step_by(chunk)
        .map(|s| (s, (s + chunk).min(n)))
        .collect();

    let cancelled = AtomicBool::new(false);
    let done = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel::<Result<(usize, Partial), Error>>();

    // Rayon tasks rather than OS threads. The filters these call parallelise
    // internally, and rayon composes nested parallelism from inside its own
    // pool; injecting it from foreign threads instead makes every inner
    // parallel_for a cross-thread handshake, which costs more the more cores
    // the machine has.
    rayon::in_place_scope(|scope| {
        for (start, end) in batches.iter().copied() {
            let tx = tx.clone();
            let cancelled = &cancelled;
            let done = &done;
            scope.spawn(move |_| {
                let mut partial = Partial::new(levels);
                for i in start..end {
                    if cancelled.load(Ordering::Relaxed) {
                        return;
                    }
                    if let Err(e) =
                        fuse_one(&sources[i], &warps[i], canvas, depth, opts, &mut partial)
                    {
                        let _ = tx.send(Err(e));
                        cancelled.store(true, Ordering::Relaxed);
                        return;
                    }
                    done.fetch_add(1, Ordering::Relaxed);
                }
                let _ = tx.send(Ok((end - start, partial)));
            });
        }
        drop(tx);

        let mut merged = Partial::new(levels);
        let mut result = Ok(());
        let mut fused_count = 0usize;
        for message in rx {
            match message {
                Ok((batch_len, partial)) => {
                    merged.merge(partial);
                    fused_count += batch_len;
                    hooks.report(
                        fused_count as f64 / n as f64 * 0.7,
                        Stage::Stacking,
                        &format!("Fused image {fused_count}/{n}"),
                    );
                    if let Err(e) = hooks.check() {
                        cancelled.store(true, Ordering::Relaxed);
                        result = Err(e);
                        break;
                    }
                }
                Err(e) => {
                    cancelled.store(true, Ordering::Relaxed);
                    result = Err(e);
                    break;
                }
            }
        }
        result?;
        finish(merged, levels, max_val, depth, hooks)
    })
}

/// Divide out the accumulated energies and collapse the pyramid.
fn finish(
    merged: Partial,
    levels: usize,
    max_val: f32,
    depth: u32,
    hooks: &Hooks,
) -> Result<ImageBuf, Error> {
    let count = merged.count.max(1) as f32;
    let mut bands = Vec::with_capacity(levels + 1);
    for i in 0..=levels {
        let energy = merged.energy[i].as_ref().expect("level accumulated");
        let weighted = merged.weighted[i].as_ref().expect("level accumulated");
        let unweighted = merged.unweighted[i].as_ref().expect("level accumulated");
        let mut band = Mat::new(weighted.h, weighted.w, weighted.c);
        let c = band.c;
        for p in 0..band.h * band.w {
            let e = energy.data[p];
            // Blend between the plain average and the energy-weighted result
            // using a confidence derived from total energy. Where no frame
            // carries sharpness — flat backgrounds, or weak texture that the
            // sharpness exponent compresses to nothing — the weighted ratio is
            // a quotient of two near-zero numbers and is pure noise, so the
            // average takes over. The midpoint sits just below any visually
            // meaningful texture and the ramp spans four decades, so the
            // transition never shows as a boundary.
            let alpha = (((e + 1e-40).log10() + 20.0) / 12.0).clamp(0.0, 1.0);
            // Divide rather than multiply by a reciprocal: the denominator can
            // be denormal, and its reciprocal would overflow to infinity where
            // the division stays finite.
            let denom = e + 1e-40;
            for ch in 0..c {
                let idx = p * c + ch;
                let w = weighted.data[idx] / denom;
                let a = unweighted.data[idx] / count;
                band.data[idx] = alpha * w + (1.0 - alpha) * a;
            }
        }
        bands.push(band);
    }

    hooks.report(0.8, Stage::Stacking, "Reconstructing pyramid...");
    let image = reconstruct(&bands);
    hooks.report(1.0, Stage::Stacking, "Stacking complete");

    Ok(if depth == 16 {
        ImageBuf::U16(MatU16 {
            h: image.h,
            w: image.w,
            c: image.c,
            data: image
                .data
                .iter()
                .map(|v| v.clamp(0.0, max_val) as u16)
                .collect(),
        })
    } else {
        ImageBuf::U8(MatU8 {
            h: image.h,
            w: image.w,
            c: image.c,
            data: image
                .data
                .iter()
                .map(|v| v.clamp(0.0, max_val) as u8)
                .collect(),
        })
    })
}

/// Index ranges for each overlapping sub-stack.
pub fn compute_slabs(n: usize, slab_size: usize, overlap: usize) -> Vec<(usize, usize)> {
    let step = slab_size.saturating_sub(overlap).max(1);
    let mut slabs = Vec::new();
    let mut s = 0usize;
    while s < n {
        let end = (s + slab_size).min(n);
        slabs.push((s, end));
        if end == n {
            break;
        }
        s += step;
    }
    slabs
}

pub enum SlabOutcome {
    Fused(ImageBuf),
    Slabs(Vec<ImageBuf>),
}

/// Stack in overlapping sub-stacks, then fuse the results.
///
/// Splitting the set reduces how many frames compete in any one fusion pass,
/// which sharpens the result on long stacks. With `recursive` the layer's
/// output is slabbed again until it fits a single pass.
#[allow(clippy::too_many_arguments)]
pub fn slab_images(
    sources: &[Source],
    warps: &[Affine],
    slab_size: usize,
    overlap: usize,
    opts: &StackOptions,
    only_slab: bool,
    recursive: bool,
    hooks: &Hooks,
) -> Result<SlabOutcome, Error> {
    let mut items: Vec<Source> = sources.to_vec();
    let mut current_warps: Vec<Affine> = warps.to_vec();
    let mut layer = 1usize;

    loop {
        let n = items.len();
        let slabs = compute_slabs(n, slab_size, overlap);
        let total = slabs.len();
        hooks.report(
            0.0,
            Stage::Slabbing,
            &format!("Layer {layer}: {total} slabs from {n} images"),
        );

        let mut results: Vec<ImageBuf> = Vec::with_capacity(total);
        for (idx, (start, end)) in slabs.iter().copied().enumerate() {
            let label = format!("slab_{layer}_{:03}", idx + 1);
            hooks.report(
                idx as f64 / total as f64,
                Stage::Slabbing,
                &format!("Stacking {label} (images {}–{end})", start + 1),
            );
            let quiet = Hooks {
                progress: None,
                interrupt: hooks.interrupt,
                on_slab: None,
            };
            let stacked =
                stack_images(&items[start..end], &current_warps[start..end], opts, &quiet)?;
            hooks.slab(&label, &stacked);
            results.push(stacked);
            hooks.check()?;
        }
        hooks.report(1.0, Stage::Slabbing, &format!("Layer {layer} complete"));

        if only_slab {
            return Ok(SlabOutcome::Slabs(results));
        }
        if results.len() == 1 {
            return Ok(SlabOutcome::Fused(results.pop().expect("one slab")));
        }
        if results.len() <= slab_size || !recursive {
            let sources: Vec<Source> = results.into_iter().map(Source::Array).collect();
            let identities = vec![Affine::IDENTITY; sources.len()];
            return Ok(SlabOutcome::Fused(stack_images(
                &sources,
                &identities,
                opts,
                hooks,
            )?));
        }

        items = results.into_iter().map(Source::Array).collect();
        current_warps = vec![Affine::IDENTITY; items.len()];
        layer += 1;
    }
}
