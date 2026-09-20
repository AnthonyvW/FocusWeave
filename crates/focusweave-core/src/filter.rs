//! Convolution primitives mirroring the OpenCV routines the pipeline relies on.
//!
//! OpenCV performs *correlation*, not convolution, and anchors a kernel of
//! length `k` at `k / 2`. Both conventions are reproduced here so ported code
//! matches the reference implementation tap for tap.

use crate::border::{border_table, Border};
use crate::mat::Mat;
use rayon::prelude::*;

/// Longest separable kernel the column pass gathers on the stack. The widest
/// in use is the 31-tap Gaussian of the focus scorer.
const MAX_TAPS: usize = 64;

/// Separable correlation, equivalent to `cv2.sepFilter2D` with `CV_32F` output.
pub fn sep_filter(src: &Mat, kx: &[f32], ky: &[f32], border: Border) -> Mat {
    #[cfg(feature = "opencv-backend")]
    {
        crate::backend_opencv::sep_filter(src, kx, ky, border)
    }
    #[cfg(not(feature = "opencv-backend"))]
    {
        sep_filter_native(src, kx, ky, border)
    }
}

fn sep_filter_native(src: &Mat, kx: &[f32], ky: &[f32], border: Border) -> Mat {
    assert!(
        kx.len() <= MAX_TAPS && ky.len() <= MAX_TAPS,
        "kernel longer than {MAX_TAPS} taps"
    );
    let (h, w, c) = (src.h, src.w, src.c);
    let stride = w * c;
    let (kxs, kys) = (kx.len(), ky.len());
    let (x_anchor, y_anchor) = (kxs / 2, kys / 2);
    let x_table = border_table(w, kxs, x_anchor, border);
    let y_table = border_table(h, kys, y_anchor, border);

    // Columns in [lo, hi) have every horizontal tap inside the image, so they
    // skip the border table; only the two margins need index mapping.
    let lo = x_anchor.min(w);
    let hi = w.saturating_sub(kxs - 1 - x_anchor).max(lo);

    let mut dst = Mat::new(h, w, c);
    if h == 0 || w == 0 {
        return dst;
    }

    // The two passes are fused. Writing the horizontally filtered image out in
    // full and reading it back costs two extra trips through main memory per
    // filter, which is what dominates once several images are being fused at
    // once — the kernels are memory bound long before they are compute bound.
    // Instead each band of output rows keeps a ring of just `ky.len()`
    // filtered rows, small enough to stay in cache, and the intermediate never
    // exists in full.
    let threads = rayon::current_num_threads().max(1);
    let band = h.div_ceil(threads * 2).max(32).min(h);

    dst.data
        .par_chunks_mut(band * stride)
        .enumerate()
        .for_each(|(index, out)| {
            let first_row = index * band;
            let mut ring = vec![0.0f32; kys * stride];
            // Which source row currently occupies each ring slot.
            let mut resident = vec![usize::MAX; kys];

            for (offset, drow) in out.chunks_exact_mut(stride).enumerate() {
                let y = first_row + offset;
                let taps = &y_table[y * kys..(y + 1) * kys];

                // Fill any rows this output row needs that are not resident.
                // Advancing by one output row normally brings in exactly one.
                for source in taps.iter().copied() {
                    if source == usize::MAX {
                        continue;
                    }
                    let slot = source % kys;
                    if resident[slot] != source {
                        let (from, to) = (slot * stride, (slot + 1) * stride);
                        filter_row_into(
                            &mut ring[from..to],
                            &src.data[source * stride..(source + 1) * stride],
                            kx,
                            &x_table,
                            lo,
                            hi,
                            x_anchor,
                            c,
                        );
                        resident[slot] = source;
                    }
                }

                let mut rows: [(&[f32], f32); MAX_TAPS] = [(&[] as &[f32], 0.0f32); MAX_TAPS];
                let mut n = 0usize;
                for (j, source) in taps.iter().copied().enumerate() {
                    if source == usize::MAX {
                        continue;
                    }
                    let slot = source % kys;
                    rows[n] = (&ring[slot * stride..(slot + 1) * stride], ky[j]);
                    n += 1;
                }
                if n == 0 {
                    drow.fill(0.0);
                } else {
                    crate::simd::col_taps(drow, &rows[..n]);
                }
            }
        });
    dst
}

/// Horizontal pass for a single row.
#[allow(clippy::too_many_arguments)]
fn filter_row_into(
    out: &mut [f32],
    srow: &[f32],
    k: &[f32],
    x_table: &[usize],
    lo: usize,
    hi: usize,
    anchor: usize,
    c: usize,
) {
    let ks = k.len();
    let w = out.len() / c;
    let margin = |x: usize, out: &mut [f32]| {
        let taps = &x_table[x * ks..(x + 1) * ks];
        for ch in 0..c {
            let mut acc = 0.0f32;
            for (t, kv) in taps.iter().zip(k) {
                if *t != usize::MAX {
                    acc += srow[t * c + ch] * *kv;
                }
            }
            out[x * c + ch] = acc;
        }
    };
    for x in 0..lo {
        margin(x, out);
    }
    for x in hi..w {
        margin(x, out);
    }

    // An image narrower than the anchor has no interior; the margins above
    // have already covered every column.
    if hi <= lo || lo < anchor {
        return;
    }
    crate::simd::row_taps(&mut out[lo * c..hi * c], &srow[(lo - anchor) * c..], k, c);
}

/// Non-separable correlation, equivalent to `cv2.filter2D` with `CV_32F` output.
pub fn filter_2d(src: &Mat, kernel: &Mat, border: Border) -> Mat {
    assert_eq!(kernel.c, 1, "filter kernels are single channel");
    let (kh, kw) = (kernel.h, kernel.w);
    let (ay, ax) = (kh / 2, kw / 2);
    let xt = border_table(src.w, kw, ax, border);
    let yt = border_table(src.h, kh, ay, border);
    let mut dst = Mat::new(src.h, src.w, src.c);
    for y in 0..src.h {
        for x in 0..src.w {
            for ch in 0..src.c {
                let mut acc = 0.0f32;
                for ky in 0..kh {
                    let sy = yt[y * kh + ky];
                    if sy == usize::MAX {
                        continue;
                    }
                    for kx in 0..kw {
                        let sx = xt[x * kw + kx];
                        if sx == usize::MAX {
                            continue;
                        }
                        acc += src.data[(sy * src.w + sx) * src.c + ch] * kernel.data[ky * kw + kx];
                    }
                }
                *dst.at_mut(y, x, ch) = acc;
            }
        }
    }
    dst
}

/// Normalised box filter, equivalent to `cv2.boxFilter(..., normalize=True)`.
pub fn box_filter(src: &Mat, kw: usize, kh: usize, border: Border) -> Mat {
    #[cfg(feature = "opencv-backend")]
    {
        crate::backend_opencv::box_filter(src, kw, kh, border)
    }
    #[cfg(not(feature = "opencv-backend"))]
    {
        box_filter_native(src, kw, kh, border)
    }
}

fn box_filter_native(src: &Mat, kw: usize, kh: usize, border: Border) -> Mat {
    let kx = vec![1.0f32 / kw as f32; kw];
    let ky = vec![1.0f32 / kh as f32; kh];
    sep_filter(src, &kx, &ky, border)
}

/// Mean of squares over a window, equivalent to `cv2.sqrBoxFilter(..., normalize=True)`.
pub fn sqr_box_filter(src: &Mat, kw: usize, kh: usize, border: Border) -> Mat {
    #[cfg(feature = "opencv-backend")]
    {
        crate::backend_opencv::sqr_box_filter(src, kw, kh, border)
    }
    #[cfg(not(feature = "opencv-backend"))]
    {
        sqr_box_filter_native(src, kw, kh, border)
    }
}

fn sqr_box_filter_native(src: &Mat, kw: usize, kh: usize, border: Border) -> Mat {
    let squared = Mat {
        h: src.h,
        w: src.w,
        c: src.c,
        data: src.data.iter().map(|v| v * v).collect(),
    };
    box_filter(&squared, kw, kh, border)
}

/// OpenCV's `getGaussianKernel` for `CV_32F`, including its small-kernel table.
pub fn gaussian_kernel(n: usize, sigma: f64) -> Vec<f32> {
    const SMALL: [&[f32]; 4] = [
        &[1.0],
        &[0.25, 0.5, 0.25],
        &[0.0625, 0.25, 0.375, 0.25, 0.0625],
        &[
            0.03125, 0.109375, 0.21875, 0.3125, 0.21875, 0.109375, 0.03125,
        ],
    ];
    if n % 2 == 1 && n <= 7 && sigma <= 0.0 {
        return SMALL[n >> 1].to_vec();
    }
    let sigma_x = if sigma > 0.0 {
        sigma
    } else {
        ((n as f64 - 1.0) * 0.5 - 1.0) * 0.3 + 0.8
    };
    let scale = -0.5 / (sigma_x * sigma_x);
    let mut k: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f64 - (n as f64 - 1.0) * 0.5;
            (scale * x * x).exp() as f32
        })
        .collect();
    let sum: f32 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// `cv2.GaussianBlur` with a square kernel and OpenCV's default border.
pub fn gaussian_blur(src: &Mat, ksize: usize, sigma: f64) -> Mat {
    #[cfg(feature = "opencv-backend")]
    {
        crate::backend_opencv::gaussian_blur(src, ksize, sigma)
    }
    #[cfg(not(feature = "opencv-backend"))]
    {
        gaussian_blur_native(src, ksize, sigma)
    }
}

fn gaussian_blur_native(src: &Mat, ksize: usize, sigma: f64) -> Mat {
    if ksize == 1 {
        return src.clone();
    }
    let k = gaussian_kernel(ksize, sigma);
    sep_filter(src, &k, &k, Border::Reflect101)
}

/// OpenCV's `getDerivKernels` coefficients for a single axis (`normalize=false`).
pub fn deriv_kernel(order: usize, ksize: usize) -> Vec<f32> {
    if ksize == 1 {
        return vec![1.0];
    }
    if ksize == 3 {
        return match order {
            0 => vec![1.0, 2.0, 1.0],
            1 => vec![-1.0, 0.0, 1.0],
            _ => vec![1.0, -2.0, 1.0],
        };
    }
    let mut ker = vec![0i32; ksize + 1];
    ker[0] = 1;
    for _ in 0..(ksize - order - 1) {
        let mut oldval = ker[0];
        for j in 1..=ksize {
            let newval = ker[j] + ker[j - 1];
            ker[j - 1] = oldval;
            oldval = newval;
        }
    }
    for _ in 0..order {
        let mut oldval = -ker[0];
        for j in 1..=ksize {
            let newval = ker[j - 1] - ker[j];
            ker[j - 1] = oldval;
            oldval = newval;
        }
    }
    ker[..ksize].iter().map(|v| *v as f32).collect()
}

/// `cv2.Sobel` with `CV_32F` output and OpenCV's default border.
pub fn sobel(src: &Mat, dx: usize, dy: usize, ksize: usize) -> Mat {
    #[cfg(feature = "opencv-backend")]
    {
        crate::backend_opencv::sobel(src, dx, dy, ksize)
    }
    #[cfg(not(feature = "opencv-backend"))]
    {
        sobel_native(src, dx, dy, ksize)
    }
}

fn sobel_native(src: &Mat, dx: usize, dy: usize, ksize: usize) -> Mat {
    let kx = deriv_kernel(dx, ksize);
    let ky = deriv_kernel(dy, ksize);
    sep_filter(src, &kx, &ky, Border::Reflect101)
}

/// `cv2.Laplacian(..., ksize=3)`, which uses OpenCV's hard-coded 3x3 kernel.
pub fn laplacian3(src: &Mat) -> Mat {
    #[cfg(feature = "opencv-backend")]
    {
        crate::backend_opencv::laplacian3(src)
    }
    #[cfg(not(feature = "opencv-backend"))]
    {
        laplacian3_native(src)
    }
}

fn laplacian3_native(src: &Mat) -> Mat {
    let kernel = Mat::from_vec(3, 3, 1, vec![2.0, 0.0, 2.0, 0.0, -8.0, 0.0, 2.0, 0.0, 2.0]);
    filter_2d(src, &kernel, Border::Reflect101)
}

/// `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))`.
pub fn ellipse_kernel(w: usize, h: usize) -> Vec<Vec<bool>> {
    let r = (h / 2) as i32;
    let c = (w / 2) as i32;
    let inv_r2 = if r != 0 {
        1.0 / (r as f64 * r as f64)
    } else {
        0.0
    };
    let mut out = vec![vec![false; w]; h];
    for (i, row) in out.iter_mut().enumerate() {
        let dy = i as i32 - r;
        if dy.abs() > r {
            continue;
        }
        let inner = ((r * r - dy * dy) as f64 * inv_r2).max(0.0).sqrt();
        let dx = (c as f64 * inner).round() as i32;
        let j1 = (c - dx).max(0) as usize;
        let j2 = ((c + dx + 1) as usize).min(w);
        for cell in row.iter_mut().take(j2).skip(j1) {
            *cell = true;
        }
    }
    out
}

/// Binary dilation with an elliptical structuring element of the given size.
pub fn dilate_ellipse(src: &crate::mat::MatU8, kw: usize, kh: usize) -> crate::mat::MatU8 {
    #[cfg(feature = "opencv-backend")]
    return crate::backend_opencv::dilate_ellipse(src, kw, kh);
    #[cfg(not(feature = "opencv-backend"))]
    dilate_u8(src, &ellipse_kernel(kw, kh))
}

/// Binary dilation with an arbitrary structuring element.
///
/// Each structuring-element row is a contiguous run, so a per-row prefix sum
/// answers "is any pixel in this window set?" in constant time instead of
/// scanning the whole neighbourhood.
pub fn dilate_u8(src: &crate::mat::MatU8, se: &[Vec<bool>]) -> crate::mat::MatU8 {
    let (h, w) = (src.h, src.w);
    let kh = se.len();
    let kw = se[0].len();
    let (ay, ax) = ((kh / 2) as isize, (kw / 2) as isize);

    // Half-open column run of each structuring-element row.
    let runs: Vec<Option<(isize, isize)>> = se
        .iter()
        .map(|row| {
            let first = row.iter().position(|v| *v)?;
            let last = row.iter().rposition(|v| *v)?;
            Some((first as isize - ax, last as isize + 1 - ax))
        })
        .collect();

    let mut prefix = vec![0u32; w + 1];
    let mut row_any: Vec<Vec<bool>> = Vec::with_capacity(h);
    for y in 0..h {
        let row = &src.data[y * w..(y + 1) * w];
        prefix[0] = 0;
        for x in 0..w {
            prefix[x + 1] = prefix[x] + u32::from(row[x] != 0);
        }
        let mut any = vec![false; w * runs.len()];
        for (r, run) in runs.iter().enumerate() {
            let Some((from, to)) = *run else { continue };
            for x in 0..w {
                let a = (x as isize + from).clamp(0, w as isize) as usize;
                let b = (x as isize + to).clamp(0, w as isize) as usize;
                any[r * w + x] = prefix[b] > prefix[a];
            }
        }
        row_any.push(any);
    }

    let mut dst = crate::mat::MatU8::new(h, w, 1);
    for y in 0..h {
        for (r, run) in runs.iter().enumerate() {
            if run.is_none() {
                continue;
            }
            // cv2.dilate replicates the border, so rows outside clamp inward.
            let sy = (y as isize + r as isize - ay).clamp(0, h as isize - 1) as usize;
            let any = &row_any[sy][r * w..(r + 1) * w];
            let out = &mut dst.data[y * w..(y + 1) * w];
            for x in 0..w {
                if any[x] {
                    out[x] = 255;
                }
            }
        }
    }
    dst
}

/// `numpy.percentile` with the default linear interpolation method.
///
/// Uses selection rather than a full sort; only the two order statistics
/// bracketing the requested rank are needed.
pub fn percentile(values: &[f32], q: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut buf: Vec<f32> = values.to_vec();
    let n = buf.len();
    let pos = (q / 100.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

    let (_, lo_val, rest) = buf.select_nth_unstable_by(lo, cmp);
    let lo_val = *lo_val;
    if lo == hi {
        return lo_val;
    }
    let hi_val = *rest.iter().min_by(|a, b| cmp(a, b)).unwrap_or(&lo_val);
    lo_val + (hi_val - lo_val) * (pos - lo as f64) as f32
}
