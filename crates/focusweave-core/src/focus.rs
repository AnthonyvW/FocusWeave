//! Sharpness scoring, culling, and the focus masks that steer registration.

use crate::clahe::clahe;
use crate::color::{cv_round, rgb_to_gray_u8};
use crate::filter::{dilate_u8, ellipse_kernel, gaussian_blur, laplacian3, percentile, sobel};
use crate::hooks::{Error, Hooks, Stage};
use crate::image_source::{load_u8, Source};
use crate::mat::{Mat, MatU8};
use crate::resize::{resize_area, resize_area_u8};

const CLAHE_CLIP: f64 = 2.0;
const CLAHE_TILES: usize = 8;

fn scaled_size(w: usize, h: usize, max_resolution: usize) -> Option<(usize, usize)> {
    let long_edge = w.max(h);
    if long_edge <= max_resolution {
        return None;
    }
    let scale = max_resolution as f64 / long_edge as f64;
    Some((
        cv_round(w as f64 * scale) as usize,
        cv_round(h as f64 * scale) as usize,
    ))
}

/// Raw Tenengrad response map for one frame.
///
/// The frame is downscaled to `max_resolution` on the long edge and CLAHE
/// normalised before the Sobel gradients are taken, so frames of differing
/// exposure compare fairly.
pub fn tenengrad_score_map(
    src: &Source,
    reference_size: (usize, usize),
    ksize: usize,
    max_resolution: usize,
) -> Result<Mat, Error> {
    let img = load_u8(src, reference_size)?;
    let img = match scaled_size(img.w, img.h, max_resolution) {
        Some((w, h)) => resize_area_u8(&img, w, h),
        None => img,
    };
    let gray = rgb_to_gray_u8(&img);
    let equalised = clahe(&gray, CLAHE_CLIP, CLAHE_TILES, CLAHE_TILES);
    let normed = Mat {
        h: equalised.h,
        w: equalised.w,
        c: 1,
        data: equalised
            .data
            .iter()
            .map(|v| f32::from(*v) / 255.0)
            .collect(),
    };
    let gx = sobel(&normed, 1, 0, ksize);
    let gy = sobel(&normed, 0, 1, ksize);
    let mut out = Mat::new(normed.h, normed.w, 1);
    for i in 0..out.data.len() {
        out.data[i] = gx.data[i] * gx.data[i] + gy.data[i] * gy.data[i];
    }
    Ok(out)
}

/// Summarise a score map as the ratio of high- to low-frequency energy.
///
/// Genuine fine texture shows up as compact bright spots (high HF relative to
/// LF), while halos and diffuse glows form broad blobs (high LF, low ratio).
/// Dividing by the LF energy makes the measure invariant to overall brightness.
pub fn score_map_to_scalar(score_map: &Mat) -> f64 {
    let peak = score_map.max();
    let denom = peak + 1e-8;
    let normed = Mat {
        h: score_map.h,
        w: score_map.w,
        c: 1,
        data: score_map.data.iter().map(|v| v / denom).collect(),
    };
    let lf = gaussian_blur(&normed, 31, 0.0);
    let mut lf_energy = 0.0f64;
    let mut hf_energy = 0.0f64;
    let mut count = 0usize;
    for (n, l) in normed.data.iter().zip(&lf.data) {
        if *n > 0.01 {
            let hf = *n - *l;
            lf_energy += f64::from(*l) * f64::from(*l);
            hf_energy += f64::from(hf) * f64::from(hf);
            count += 1;
        }
    }
    if count == 0 || lf_energy == 0.0 {
        return 0.0;
    }
    (hf_energy / count as f64) / (lf_energy / count as f64)
}

/// One frame's verdict from the culling stage.
#[derive(Clone, Debug)]
pub struct CullEntry {
    pub index: usize,
    pub label: String,
    pub score: f64,
    pub kept: bool,
}

#[derive(Clone, Debug)]
pub struct CullResult {
    pub entries: Vec<CullEntry>,
    pub cutoff: f64,
    pub n_culled: usize,
}

impl CullResult {
    pub fn kept_indices(&self) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|e| e.kept)
            .map(|e| e.index)
            .collect()
    }
}

/// Decide which frames carry enough detail to contribute to the stack.
///
/// The two sharpest frames are always retained so a stack can proceed even at
/// an aggressive threshold.
pub fn cull_unfocused(
    sources: &[Source],
    reference_size: (usize, usize),
    threshold: f64,
    hooks: &Hooks,
) -> Result<CullResult, Error> {
    let n = sources.len();
    let mut scores = Vec::with_capacity(n);
    for (i, src) in sources.iter().enumerate() {
        let map = tenengrad_score_map(src, reference_size, 5, 1024)?;
        let score = score_map_to_scalar(&map);
        scores.push(score);
        let label = src.label(i);
        hooks.report(
            (i + 1) as f64 / n as f64,
            Stage::Culling,
            &format!("Scored {label}  ({score:.4})"),
        );
        hooks.check()?;
    }

    let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut keep: Vec<bool> = if peak == 0.0 {
        vec![true; n]
    } else {
        scores.iter().map(|s| *s >= threshold).collect()
    };

    if keep.iter().filter(|k| **k).count() < 2 {
        let mut ranked: Vec<usize> = (0..n).collect();
        ranked.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for idx in ranked.into_iter().take(2) {
            keep[idx] = true;
        }
    }

    let entries: Vec<CullEntry> = (0..n)
        .map(|i| CullEntry {
            index: i,
            label: sources[i].label(i),
            score: scores[i],
            kept: keep[i],
        })
        .collect();
    let n_culled = entries.iter().filter(|e| !e.kept).count();
    if entries.len() - n_culled < 2 {
        return Err(Error::Config(
            "Fewer than 2 images survived culling. Lower the cull threshold or disable culling."
                .into(),
        ));
    }
    Ok(CullResult {
        entries,
        cutoff: threshold,
        n_culled,
    })
}

/// Mask of the sharpest pixels in a grayscale frame.
///
/// ECC is only shown pixels that carry real detail; out-of-focus regions,
/// which dominate macro frames and contain misleading intensity patterns, are
/// excluded from the cost function entirely.
pub fn focus_mask(gray: &MatU8, percentile_cut: f64) -> MatU8 {
    let grayf = gray.to_f32();
    let lap = laplacian3(&grayf);
    let squared = Mat {
        h: lap.h,
        w: lap.w,
        c: 1,
        data: lap.data.iter().map(|v| v * v).collect(),
    };
    let sharpness = gaussian_blur(&squared, 15, 0.0);
    let cut = percentile(&sharpness.data, percentile_cut);
    let mask = MatU8 {
        h: gray.h,
        w: gray.w,
        c: 1,
        data: sharpness
            .data
            .iter()
            .map(|v| if *v >= cut { 255u8 } else { 0 })
            .collect(),
    };
    dilate_u8(&mask, &ellipse_kernel(7, 7))
}

/// A frame prepared for ECC at one working resolution.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub equalised: Mat,
    pub mask: MatU8,
    pub scale: f64,
}

/// Downscale, CLAHE-equalise and mask a grayscale frame.
///
/// Callers cache the result per image and resolution so this never repeats
/// across ECC passes or across the forward and backward neighbour chains,
/// where each frame appears as both reference and source.
pub fn prepare_for_ecc(gray: &MatU8, max_resolution: usize) -> Prepared {
    let (small, scale) = match scaled_size(gray.w, gray.h, max_resolution) {
        Some((w, h)) => (
            resize_area_u8(gray, w, h),
            max_resolution as f64 / gray.w.max(gray.h) as f64,
        ),
        None => (gray.clone(), 1.0),
    };
    let equalised = clahe(&small, CLAHE_CLIP, CLAHE_TILES, CLAHE_TILES);
    let mask = focus_mask(&equalised, 30.0);
    Prepared {
        equalised: equalised.to_f32(),
        mask,
        scale,
    }
}

/// Estimate translation by phase correlation at reduced resolution.
///
/// Operating globally across all frequencies makes this robust on the
/// low-contrast, textureless regions where gradient-based methods drift.
pub fn phase_correlation_translation(
    ref_gray: &MatU8,
    src_gray: &MatU8,
    max_resolution: usize,
) -> (f64, f64) {
    let (ref_f, src_f, scale) = match scaled_size(ref_gray.w, ref_gray.h, max_resolution) {
        Some((w, h)) => (
            resize_area(&ref_gray.to_f32(), w, h),
            resize_area(&src_gray.to_f32(), w, h),
            max_resolution as f64 / ref_gray.w.max(ref_gray.h) as f64,
        ),
        None => (ref_gray.to_f32(), src_gray.to_f32(), 1.0),
    };
    let (tx, ty) = crate::fft::phase_correlate(&src_f, &ref_f);
    (tx / scale, ty / scale)
}

/// Combine two focus masks for a registration pair.
///
/// The intersection keeps ECC on pixels that are sharp in both frames; it
/// falls back to the union, then to no mask, when the overlap is too sparse.
pub fn combine_masks(a: &MatU8, b: &MatU8) -> Option<MatU8> {
    let mut inter = MatU8::new(a.h, a.w, 1);
    for i in 0..inter.data.len() {
        inter.data[i] = a.data[i] & b.data[i];
    }
    if inter.count_non_zero() >= 100 {
        return Some(inter);
    }
    let mut union = MatU8::new(a.h, a.w, 1);
    for i in 0..union.data.len() {
        union.data[i] = a.data[i] | b.data[i];
    }
    if union.count_non_zero() >= 100 {
        Some(union)
    } else {
        None
    }
}

/// Convert an 8-bit RGB frame to the grayscale used throughout alignment.
pub fn to_gray(src: &Source, size: (usize, usize)) -> Result<MatU8, Error> {
    Ok(rgb_to_gray_u8(&load_u8(src, size)?))
}
