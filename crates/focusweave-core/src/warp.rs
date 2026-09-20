//! Affine warping, equivalent to `cv2.warpAffine`.
//!
//! Source coordinates are evaluated in full floating point rather than
//! OpenCV's 1/32-pixel fixed-point tables, so the resampling is slightly more
//! accurate than the reference implementation it replaces.

use crate::affine::Affine;
use crate::border::{border_index, Border};
use crate::mat::{Mat, MatU16, MatU8};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Linear,
    Cubic,
    Nearest,
    /// Bilinear sampling with source coordinates snapped to 1/32 of a pixel.
    ///
    /// OpenCV's `warpAffine` builds its coordinate maps in fixed point and
    /// quantises to `INTER_TAB_SIZE` steps. Registration is sensitive to that
    /// quantisation around mask boundaries, so the ECC solver reproduces it
    /// instead of using the more accurate continuous coordinates.
    LinearQuantised,
    /// Nearest neighbour with OpenCV's fixed-point rounding (half away from
    /// zero rather than half to even).
    NearestQuantised,
}

/// Snap a coordinate to the 1/32-pixel grid OpenCV's warp maps use.
#[inline]
fn quantise(v: f64) -> f64 {
    (v * 32.0 + 0.5).floor() / 32.0
}

#[inline]
fn cubic_coeffs(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let mut c = [0.0f32; 4];
    c[0] = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    c[1] = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    c[2] = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    c[3] = 1.0 - c[0] - c[1] - c[2];
    c
}

#[inline]
fn fetch(src: &Mat, y: isize, x: isize, ch: usize, border: Border) -> f32 {
    match (
        border_index(y, src.h, border),
        border_index(x, src.w, border),
    ) {
        (Some(sy), Some(sx)) => src.data[(sy * src.w + sx) * src.c + ch],
        _ => 0.0,
    }
}

/// Sample `src` at `(sx, sy)` for every channel.
fn sample(src: &Mat, sx: f64, sy: f64, interp: Interp, border: Border, out: &mut [f32]) {
    match interp {
        Interp::LinearQuantised => {
            sample(src, quantise(sx), quantise(sy), Interp::Linear, border, out);
        }
        Interp::NearestQuantised => {
            sample(
                src,
                (sx + 0.5).floor(),
                (sy + 0.5).floor(),
                Interp::Nearest,
                border,
                out,
            );
        }
        Interp::Nearest => {
            let x = sx.round_ties_even() as isize;
            let y = sy.round_ties_even() as isize;
            for (ch, o) in out.iter_mut().enumerate() {
                *o = fetch(src, y, x, ch, border);
            }
        }
        Interp::Linear => {
            let x0 = sx.floor();
            let y0 = sy.floor();
            let fx = (sx - x0) as f32;
            let fy = (sy - y0) as f32;
            let (x0, y0) = (x0 as isize, y0 as isize);
            let wx = [1.0 - fx, fx];
            let wy = [1.0 - fy, fy];
            for (ch, o) in out.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (j, wyj) in wy.iter().enumerate() {
                    if *wyj == 0.0 {
                        continue;
                    }
                    let mut row = 0.0f32;
                    for (i, wxi) in wx.iter().enumerate() {
                        if *wxi == 0.0 {
                            continue;
                        }
                        row += *wxi * fetch(src, y0 + j as isize, x0 + i as isize, ch, border);
                    }
                    acc += *wyj * row;
                }
                *o = acc;
            }
        }
        Interp::Cubic => {
            let x0 = sx.floor();
            let y0 = sy.floor();
            let cx = cubic_coeffs((sx - x0) as f32);
            let cy = cubic_coeffs((sy - y0) as f32);
            let (x0, y0) = (x0 as isize, y0 as isize);
            for (ch, o) in out.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (j, cyj) in cy.iter().enumerate() {
                    let mut row = 0.0f32;
                    for (i, cxi) in cx.iter().enumerate() {
                        row +=
                            *cxi * fetch(src, y0 + j as isize - 1, x0 + i as isize - 1, ch, border);
                    }
                    acc += *cyj * row;
                }
                *o = acc;
            }
        }
    }
}

/// Warp a float image onto a `(dst_w, dst_h)` canvas.
///
/// `m` maps source coordinates to destination coordinates unless
/// `inverse_map` is set, matching OpenCV's `WARP_INVERSE_MAP` flag.
pub fn warp_affine(
    src: &Mat,
    m: &Affine,
    dst_w: usize,
    dst_h: usize,
    interp: Interp,
    border: Border,
    inverse_map: bool,
) -> Mat {
    #[cfg(feature = "opencv-backend")]
    {
        crate::backend_opencv::warp_affine(src, m, dst_w, dst_h, interp, border, inverse_map)
    }
    #[cfg(not(feature = "opencv-backend"))]
    {
        warp_affine_native(src, m, dst_w, dst_h, interp, border, inverse_map)
    }
}

#[allow(clippy::too_many_arguments)]
fn warp_affine_native(
    src: &Mat,
    m: &Affine,
    dst_w: usize,
    dst_h: usize,
    interp: Interp,
    border: Border,
    inverse_map: bool,
) -> Mat {
    let inv = if inverse_map { *m } else { m.invert() };
    let c = src.c;
    let mut dst = Mat::new(dst_h, dst_w, c);

    // How far the interpolation kernel reaches either side of the base sample.
    let (back, fwd) = match interp {
        Interp::Nearest | Interp::NearestQuantised => (0isize, 0isize),
        Interp::Linear | Interp::LinearQuantised => (0, 1),
        Interp::Cubic => (1, 2),
    };
    let inside_x = back..(src.w as isize - fwd);
    let inside_y = back..(src.h as isize - fwd);

    let stride = src.w * c;
    // Rows are independent; splitting them pays for itself on the large
    // buffers the ECC solver resamples every iteration.
    let rows = dst.data.par_chunks_exact_mut(dst_w * c);
    rows.enumerate().for_each(|(y, drow)| {
        let mut px = vec![0.0f32; c];
        for x in 0..dst_w {
            let (mut sx, mut sy) = inv.apply(x as f64, y as f64);
            if matches!(interp, Interp::LinearQuantised) {
                sx = quantise(sx);
                sy = quantise(sy);
            } else if matches!(interp, Interp::NearestQuantised) {
                sx = (sx + 0.5).floor();
                sy = (sy + 0.5).floor();
            } else if matches!(interp, Interp::Nearest) {
                sx = sx.round_ties_even();
                sy = sy.round_ties_even();
            }
            let fx = sx.floor();
            let fy = sy.floor();
            let ix = fx as isize;
            let iy = fy as isize;
            let base = x * c;

            if inside_x.contains(&ix) && inside_y.contains(&iy) {
                let origin = (iy as usize * src.w + ix as usize) * c;
                match interp {
                    Interp::Nearest | Interp::NearestQuantised => {
                        let o = origin;
                        drow[base..base + c].copy_from_slice(&src.data[o..o + c]);
                    }
                    Interp::Linear | Interp::LinearQuantised => {
                        // Weighted sums, not lerps: the rounding has to
                        // match the border path or ECC drifts between them.
                        let ax = (sx - fx) as f32;
                        let ay = (sy - fy) as f32;
                        let (wx0, wx1) = (1.0 - ax, ax);
                        let (wy0, wy1) = (1.0 - ay, ay);
                        for ch in 0..c {
                            let top = src.data[origin + ch] * wx0 + src.data[origin + c + ch] * wx1;
                            let bottom = src.data[origin + stride + ch] * wx0
                                + src.data[origin + stride + c + ch] * wx1;
                            drow[base + ch] = top * wy0 + bottom * wy1;
                        }
                    }
                    Interp::Cubic => {
                        let cx = cubic_coeffs((sx - fx) as f32);
                        let cy = cubic_coeffs((sy - fy) as f32);
                        let corner = origin - stride - c;
                        for ch in 0..c {
                            let mut acc = 0.0f32;
                            for (j, cyj) in cy.iter().enumerate() {
                                let row = corner + j * stride + ch;
                                let mut sum = 0.0f32;
                                for (i, cxi) in cx.iter().enumerate() {
                                    sum += *cxi * src.data[row + i * c];
                                }
                                acc += *cyj * sum;
                            }
                            drow[base + ch] = acc;
                        }
                    }
                }
            } else {
                let fallback = match interp {
                    Interp::LinearQuantised => Interp::Linear,
                    Interp::NearestQuantised => Interp::Nearest,
                    other => other,
                };
                sample(src, sx, sy, fallback, border, &mut px);
                drow[base..base + c].copy_from_slice(&px);
            }
        }
    });
    dst
}

/// Warp an 8-bit image, saturating on write the way OpenCV does.
pub fn warp_affine_u8(
    src: &MatU8,
    m: &Affine,
    dst_w: usize,
    dst_h: usize,
    interp: Interp,
    border: Border,
) -> MatU8 {
    #[cfg(feature = "opencv-backend")]
    {
        crate::backend_opencv::warp_affine_u8(src, m, dst_w, dst_h, interp, border)
    }
    #[cfg(not(feature = "opencv-backend"))]
    {
        warp_affine_u8_native(src, m, dst_w, dst_h, interp, border)
    }
}

fn warp_affine_u8_native(
    src: &MatU8,
    m: &Affine,
    dst_w: usize,
    dst_h: usize,
    interp: Interp,
    border: Border,
) -> MatU8 {
    let f = Mat {
        h: src.h,
        w: src.w,
        c: src.c,
        data: src.data.iter().map(|v| f32::from(*v)).collect(),
    };
    let r = warp_affine(&f, m, dst_w, dst_h, interp, border, false);
    MatU8 {
        h: r.h,
        w: r.w,
        c: r.c,
        data: r
            .data
            .iter()
            .map(|v| v.round_ties_even().clamp(0.0, 255.0) as u8)
            .collect(),
    }
}

/// Warp a 16-bit image, saturating on write the way OpenCV does.
pub fn warp_affine_u16(
    src: &MatU16,
    m: &Affine,
    dst_w: usize,
    dst_h: usize,
    interp: Interp,
    border: Border,
) -> MatU16 {
    #[cfg(feature = "opencv-backend")]
    {
        crate::backend_opencv::warp_affine_u16(src, m, dst_w, dst_h, interp, border)
    }
    #[cfg(not(feature = "opencv-backend"))]
    {
        warp_affine_u16_native(src, m, dst_w, dst_h, interp, border)
    }
}

fn warp_affine_u16_native(
    src: &MatU16,
    m: &Affine,
    dst_w: usize,
    dst_h: usize,
    interp: Interp,
    border: Border,
) -> MatU16 {
    let f = Mat {
        h: src.h,
        w: src.w,
        c: src.c,
        data: src.data.iter().map(|v| f32::from(*v)).collect(),
    };
    let r = warp_affine(&f, m, dst_w, dst_h, interp, border, false);
    MatU16 {
        h: r.h,
        w: r.w,
        c: r.c,
        data: r
            .data
            .iter()
            .map(|v| v.round_ties_even().clamp(0.0, 65535.0) as u16)
            .collect(),
    }
}
