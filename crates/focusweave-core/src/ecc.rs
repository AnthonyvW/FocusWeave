//! Affine image registration, equivalent to `cv2.findTransformECC` with
//! `MOTION_AFFINE`.
//!
//! Follows Evangelidis & Psarakis' enhanced correlation coefficient
//! maximisation as OpenCV implements it, including the illumination parameter
//! `lambda` and the masked zero-mean statistics.

use crate::affine::Affine;
use crate::border::Border;
use crate::filter::{filter_2d, gaussian_blur};
use crate::mat::{Mat, MatU8};
use crate::warp::{warp_affine, Interp};
use rayon::prelude::*;

const PARAMS: usize = 6;

/// The solver could not reach a correlation maximum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EccFailure(pub &'static str);

impl std::fmt::Display for EccFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for EccFailure {}

struct MaskedStats {
    mean: f64,
    std: f64,
    count: usize,
}

fn masked_stats(src: &Mat, mask: &MatU8) -> MaskedStats {
    let mut sum = 0.0f64;
    let mut sqsum = 0.0f64;
    let mut count = 0usize;
    for (v, m) in src.data.iter().zip(&mask.data) {
        if *m != 0 {
            let v = f64::from(*v);
            sum += v;
            sqsum += v * v;
            count += 1;
        }
    }
    if count == 0 {
        return MaskedStats {
            mean: 0.0,
            std: 0.0,
            count: 0,
        };
    }
    let mean = sum / count as f64;
    let var = (sqsum / count as f64 - mean * mean).max(0.0);
    MaskedStats {
        mean,
        std: var.sqrt(),
        count,
    }
}

/// Invert a small dense matrix by Gauss-Jordan elimination with partial
/// pivoting. Returns zeros when singular, as OpenCV's `Mat::inv` does.
fn invert_dense(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if m[r * n + col].abs() > m[pivot * n + col].abs() {
                pivot = r;
            }
        }
        if m[pivot * n + col].abs() < 1e-30 {
            return vec![0.0; n * n];
        }
        if pivot != col {
            for k in 0..n {
                m.swap(col * n + k, pivot * n + k);
                inv.swap(col * n + k, pivot * n + k);
            }
        }
        let d = m[col * n + col];
        for k in 0..n {
            m[col * n + k] /= d;
            inv[col * n + k] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * n + col];
            if f == 0.0 {
                continue;
            }
            for k in 0..n {
                m[r * n + k] -= f * m[col * n + k];
                inv[r * n + k] -= f * inv[col * n + k];
            }
        }
    }
    inv
}

fn mat_vec(m: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| m[i * n + j] * v[j]).sum())
        .collect()
}

/// Accumulated normal equations for one iteration.
///
/// The Jacobian of the warped image with respect to the six affine parameters
/// is never materialised. Each of its six planes is a warped gradient times a
/// coordinate weight, so every entry of the Hessian and of the projections can
/// be summed in a single pass over the gradients — 39 dot products over six
/// full-resolution planes collapse into one traversal.
#[derive(Clone)]
struct Normals {
    hessian: [f64; PARAMS * PARAMS],
    image_projection: [f64; PARAMS],
    template_projection: [f64; PARAMS],
}

impl Normals {
    fn zero() -> Normals {
        Normals {
            hessian: [0.0; PARAMS * PARAMS],
            image_projection: [0.0; PARAMS],
            template_projection: [0.0; PARAMS],
        }
    }

    fn merge(mut self, other: Normals) -> Normals {
        for i in 0..PARAMS * PARAMS {
            self.hessian[i] += other.hessian[i];
        }
        for i in 0..PARAMS {
            self.image_projection[i] += other.image_projection[i];
            self.template_projection[i] += other.template_projection[i];
        }
        self
    }
}

/// The six Jacobian values at one pixel, in OpenCV's parameter order.
#[inline]
fn jacobian_at(gx: f32, gy: f32, x: f32, y: f32) -> [f32; PARAMS] {
    [gx * x, gy * x, gx * y, gy * y, gx, gy]
}

fn accumulate_normals(gx: &Mat, gy: &Mat, image: &Mat, template: &Mat) -> Normals {
    let w = gx.w;
    (0..gx.h)
        .into_par_iter()
        .fold(Normals::zero, |mut acc, y| {
            let row = y * w;
            let yf = y as f32;
            for x in 0..w {
                let i = row + x;
                let j = jacobian_at(gx.data[i], gy.data[i], x as f32, yf);
                let iw = f64::from(image.data[i]);
                let tz = f64::from(template.data[i]);
                for (a, ja) in j.iter().map(|v| f64::from(*v)).enumerate() {
                    acc.image_projection[a] += ja * iw;
                    acc.template_projection[a] += ja * tz;
                    for (b, jb) in j.iter().enumerate().skip(a) {
                        acc.hessian[a * PARAMS + b] += ja * f64::from(*jb);
                    }
                }
            }
            acc
        })
        .reduce(Normals::zero, Normals::merge)
}

/// Project the ECC error image onto the Jacobian without materialising it.
///
/// `error = lambda * template_zm - image_warped`, evaluated in `f32` so the
/// rounding matches the reference implementation's intermediate buffer.
fn project_error(
    gx: &Mat,
    gy: &Mat,
    template_zm: &Mat,
    image_warped: &Mat,
    lambda: f32,
) -> [f64; PARAMS] {
    let w = gx.w;
    (0..gx.h)
        .into_par_iter()
        .fold(
            || [0.0f64; PARAMS],
            |mut acc, y| {
                let row = y * w;
                let yf = y as f32;
                for x in 0..w {
                    let i = row + x;
                    let j = jacobian_at(gx.data[i], gy.data[i], x as f32, yf);
                    let e = f64::from(lambda * template_zm.data[i] - image_warped.data[i]);
                    for (slot, jv) in acc.iter_mut().zip(j.iter()) {
                        *slot += f64::from(*jv) * e;
                    }
                }
                acc
            },
        )
        .reduce(
            || [0.0f64; PARAMS],
            |mut a, b| {
                for i in 0..PARAMS {
                    a[i] += b[i];
                }
                a
            },
        )
}

fn mirror_hessian(h: &mut [f64; PARAMS * PARAMS]) {
    for a in 0..PARAMS {
        for b in a + 1..PARAMS {
            h[b * PARAMS + a] = h[a * PARAMS + b];
        }
    }
}

/// Estimate the affine warp that best aligns `input` onto `template`.
///
/// `warp` is the initial guess and maps template coordinates into the input
/// image. `mask` marks the pixels of `input` that may participate.
#[allow(clippy::too_many_arguments)]
pub fn find_transform_ecc(
    template: &Mat,
    input: &Mat,
    warp: Affine,
    max_iterations: usize,
    termination_eps: f64,
    mask: Option<&MatU8>,
    gauss_filt_size: usize,
) -> Result<Affine, EccFailure> {
    find_transform_ecc_verbose(
        template,
        input,
        warp,
        max_iterations,
        termination_eps,
        mask,
        gauss_filt_size,
    )
    .map(|r| r.0)
}

/// As [`find_transform_ecc`], also reporting the final correlation and the
/// number of iterations executed.
#[allow(clippy::too_many_arguments)]
pub fn find_transform_ecc_verbose(
    template: &Mat,
    input: &Mat,
    warp: Affine,
    max_iterations: usize,
    termination_eps: f64,
    mask: Option<&MatU8>,
    gauss_filt_size: usize,
) -> Result<(Affine, f64, usize), EccFailure> {
    assert!(
        template.c == 1 && input.c == 1,
        "ECC operates on single-channel images"
    );
    let (hs, ws) = (template.h, template.w);

    let mut pre_mask = match mask {
        Some(m) => MatU8::from_vec(
            m.h,
            m.w,
            1,
            m.data.iter().map(|v| u8::from(*v > 0)).collect(),
        ),
        None => MatU8::filled(input.h, input.w, 1, 1u8),
    };
    if gauss_filt_size > 1 {
        let blurred = gaussian_blur(&pre_mask.to_f32(), gauss_filt_size, 0.0);
        pre_mask = MatU8 {
            h: blurred.h,
            w: blurred.w,
            c: 1,
            data: blurred
                .data
                .iter()
                .map(|v| v.round_ties_even().clamp(0.0, 255.0) as u8)
                .collect(),
        };
    }
    let pre_mask_f = pre_mask.to_f32();

    let template_f = gaussian_blur(template, gauss_filt_size, 0.0);
    let image_f = gaussian_blur(input, gauss_filt_size, 0.0);

    let dx = Mat::from_vec(1, 3, 1, vec![-0.5, 0.0, 0.5]);
    let dy = Mat::from_vec(3, 1, 1, vec![-0.5, 0.0, 0.5]);
    let mut gradient_x = filter_2d(&image_f, &dx, Border::Reflect101);
    let mut gradient_y = filter_2d(&image_f, &dy, Border::Reflect101);
    for (g, m) in gradient_x.data.iter_mut().zip(&pre_mask_f.data) {
        *g *= *m;
    }
    for (g, m) in gradient_y.data.iter_mut().zip(&pre_mask_f.data) {
        *g *= *m;
    }

    let mut map = warp;
    let mut rho = -1.0f64;
    let mut last_rho = -termination_eps;

    let mut iteration = 1;
    while iteration <= max_iterations && (rho - last_rho).abs() >= termination_eps {
        let mut image_warped = warp_affine(
            &image_f,
            &map,
            ws,
            hs,
            Interp::Linear,
            Border::Constant,
            true,
        );
        let gx_warped = warp_affine(
            &gradient_x,
            &map,
            ws,
            hs,
            Interp::Linear,
            Border::Constant,
            true,
        );
        let gy_warped = warp_affine(
            &gradient_y,
            &map,
            ws,
            hs,
            Interp::Linear,
            Border::Constant,
            true,
        );
        let mask_warped_f = warp_affine(
            &pre_mask_f,
            &map,
            ws,
            hs,
            Interp::Nearest,
            Border::Constant,
            true,
        );
        let image_mask = MatU8 {
            h: hs,
            w: ws,
            c: 1,
            data: mask_warped_f
                .data
                .iter()
                .map(|v| u8::from(*v != 0.0))
                .collect(),
        };

        let img = masked_stats(&image_warped, &image_mask);
        let tmpl = masked_stats(&template_f, &image_mask);

        // Zero-mean inside the mask only; outside it the reference
        // implementation leaves the warped values untouched, and every
        // quantity they feed is masked to zero anyway.
        let mut template_zm = Mat::new(hs, ws, 1);
        for i in 0..image_warped.data.len() {
            if image_mask.data[i] != 0 {
                image_warped.data[i] -= img.mean as f32;
                template_zm.data[i] = template_f.data[i] - tmpl.mean as f32;
            }
        }

        let n_mask = img.count as f64;
        let tmp_norm = (n_mask * tmpl.std * tmpl.std).sqrt();
        let img_norm = (n_mask * img.std * img.std).sqrt();

        let mut normals = accumulate_normals(&gx_warped, &gy_warped, &image_warped, &template_zm);
        mirror_hessian(&mut normals.hessian);
        let hess_inv = invert_dense(&normals.hessian, PARAMS);

        let correlation = template_zm.dot(&image_warped);

        last_rho = rho;
        rho = correlation / (img_norm * tmp_norm);
        if rho.is_nan() {
            return Err(EccFailure("NaN encountered during ECC iteration"));
        }

        let image_projection = normals.image_projection;
        let template_projection = normals.template_projection;
        let projection_hessian = mat_vec(&hess_inv, &image_projection, PARAMS);

        let lambda_n = img_norm * img_norm - dot(&image_projection, &projection_hessian);
        let lambda_d = correlation - dot(&template_projection, &projection_hessian);
        if lambda_d <= 0.0 {
            return Err(EccFailure(
                "ECC correlation is being minimised; images may be uncorrelated or non-overlapping",
            ));
        }
        let lambda = lambda_n / lambda_d;

        let error_projection = project_error(
            &gx_warped,
            &gy_warped,
            &template_zm,
            &image_warped,
            lambda as f32,
        );
        let delta = mat_vec(&hess_inv, &error_projection, PARAMS);

        map.0[0] += delta[0] as f32;
        map.0[3] += delta[1] as f32;
        map.0[1] += delta[2] as f32;
        map.0[4] += delta[3] as f32;
        map.0[2] += delta[4] as f32;
        map.0[5] += delta[5] as f32;

        iteration += 1;
    }

    Ok((map, rho, iteration - 1))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
