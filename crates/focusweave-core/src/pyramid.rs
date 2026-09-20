//! Laplacian pyramid construction, reduction and the focus measures used to
//! weight each band.

use crate::border::Border;
use crate::filter::{box_filter, sep_filter, sqr_box_filter};
use crate::mat::Mat;

/// The 5-tap binomial kernel the pyramid is built from.
pub const K1D: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
/// The same kernel scaled for the expand step, which injects zeros.
pub const K1D_X2: [f32; 5] = [2.0 / 16.0, 8.0 / 16.0, 12.0 / 16.0, 8.0 / 16.0, 2.0 / 16.0];

/// Smooth and decimate by two.
pub fn reduce(image: &Mat) -> Mat {
    let smoothed = sep_filter(image, &K1D, &K1D, Border::Reflect);
    let (nh, nw) = (image.h.div_ceil(2), image.w.div_ceil(2));
    let c = image.c;
    let mut out = Mat::new(nh, nw, c);
    for y in 0..nh {
        for x in 0..nw {
            let src = (2 * y * image.w + 2 * x) * c;
            let dst = (y * nw + x) * c;
            out.data[dst..dst + c].copy_from_slice(&smoothed.data[src..src + c]);
        }
    }
    out
}

/// Upsample by two with zero insertion, smooth, and crop to `target`.
pub fn expand(image: &Mat, target: (usize, usize)) -> Mat {
    let c = image.c;
    let mut up = Mat::new(image.h * 2, image.w * 2, c);
    for y in 0..image.h {
        for x in 0..image.w {
            let src = (y * image.w + x) * c;
            let dst = (2 * y * up.w + 2 * x) * c;
            up.data[dst..dst + c].copy_from_slice(&image.data[src..src + c]);
        }
    }
    let expanded = sep_filter(&up, &K1D_X2, &K1D_X2, Border::Reflect);
    crop(&expanded, target.0, target.1)
}

fn crop(src: &Mat, h: usize, w: usize) -> Mat {
    let c = src.c;
    let mut out = Mat::new(h, w, c);
    for y in 0..h {
        let s = y * src.w * c;
        let d = y * w * c;
        out.data[d..d + w * c].copy_from_slice(&src.data[s..s + w * c]);
    }
    out
}

/// Build a Laplacian pyramid: `levels` detail bands followed by the residual.
pub fn laplacian_pyramid(image: &Mat, levels: usize) -> Vec<Mat> {
    let mut bands = Vec::with_capacity(levels + 1);
    let mut current = image.clone();
    for _ in 0..levels {
        let (h, w) = (current.h, current.w);
        let next = reduce(&current);
        let up = expand(&next, (h, w));
        for (a, b) in current.data.iter_mut().zip(&up.data) {
            *a -= *b;
        }
        bands.push(current);
        current = next;
    }
    bands.push(current);
    bands
}

/// Collapse a pyramid back into an image.
pub fn reconstruct(bands: &[Mat]) -> Mat {
    let mut image = bands[bands.len() - 1].clone();
    for band in bands[..bands.len() - 1].iter().rev() {
        let up = expand(&image, (band.h, band.w));
        image = up;
        for (a, b) in image.data.iter_mut().zip(&band.data) {
            *a += *b;
        }
    }
    image
}

/// Mean squared response over a window — the primary focus measure.
pub fn region_energy(level: &Mat, window: usize) -> Mat {
    sqr_box_filter(level, window, window, Border::Reflect)
}

/// Local standard deviation over a window.
pub fn region_deviation(image: &Mat, window: usize) -> Mat {
    let mean = box_filter(image, window, window, Border::Reflect);
    let sq_mean = sqr_box_filter(image, window, window, Border::Reflect);
    let mut out = Mat::new(image.h, image.w, image.c);
    for i in 0..out.data.len() {
        out.data[i] = (sq_mean.data[i] - mean.data[i] * mean.data[i])
            .max(0.0)
            .sqrt();
    }
    out
}

/// Local binary entropy over a window, used to score the pyramid residual.
pub fn region_entropy(image: &Mat, window: usize) -> Mat {
    let lo = image.min();
    let hi = image.max();
    let span = hi - lo + 1e-10;
    const EPS: f32 = 1e-10;
    let mut ent = Mat::new(image.h, image.w, image.c);
    for (dst, src) in ent.data.iter_mut().zip(&image.data) {
        let n = (*src - lo) / span;
        *dst = -n * (n + EPS).log2() - (1.0 - n) * (1.0 - n + EPS).log2();
    }
    box_filter(&ent, window, window, Border::Reflect)
}

/// Number of pyramid levels to use for an image of the given shape.
pub fn compute_levels(h: usize, w: usize, max_levels: usize) -> usize {
    let mut size = h.min(w);
    let mut levels = 0;
    while size > 16 && levels < max_levels {
        size /= 2;
        levels += 1;
    }
    levels
}
