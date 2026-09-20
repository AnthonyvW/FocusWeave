//! Colour conversions reproducing OpenCV's 8-bit `cvtColor` paths.

use crate::mat::{Mat, MatU8};
use std::sync::OnceLock;

const YUV_SHIFT: i32 = 14;
const R2Y: i32 = 4899;
const G2Y: i32 = 9617;
const B2Y: i32 = 1868;

/// `cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)` for `CV_8U`.
pub fn rgb_to_gray_u8(src: &MatU8) -> MatU8 {
    assert_eq!(src.c, 3, "expected 3-channel RGB input");
    let half = 1 << (YUV_SHIFT - 1);
    let mut dst = MatU8::new(src.h, src.w, 1);
    for (out, px) in dst.data.iter_mut().zip(src.data.chunks_exact(3)) {
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        *out = ((r * R2Y + g * G2Y + b * B2Y + half) >> YUV_SHIFT) as u8;
    }
    dst
}

const LAB_SHIFT: i32 = 12;
const GAMMA_SHIFT: i32 = 3;
const LAB_SHIFT2: i32 = LAB_SHIFT + GAMMA_SHIFT;
const LAB_CBRT_TAB_SIZE_B: usize = 256 * 3 / 2 * (1 << GAMMA_SHIFT);

struct LabTabs {
    gamma: [u16; 256],
    cbrt: Vec<u16>,
    coeffs: [i32; 9],
}

fn lab_tabs() -> &'static LabTabs {
    static TABS: OnceLock<LabTabs> = OnceLock::new();
    TABS.get_or_init(|| {
        let mut gamma = [0u16; 256];
        for (i, slot) in gamma.iter_mut().enumerate() {
            let x = i as f32 / 255.0;
            let lin = if x <= 0.04045 {
                x / 12.92
            } else {
                ((x as f64 + 0.055) / 1.055).powf(2.4) as f32
            };
            *slot = saturate_u16(255.0 * (1 << GAMMA_SHIFT) as f32 * lin);
        }
        let mut cbrt = vec![0u16; LAB_CBRT_TAB_SIZE_B];
        for (i, slot) in cbrt.iter_mut().enumerate() {
            let x = i as f32 / (255.0 * (1 << GAMMA_SHIFT) as f32);
            let v = if x < 0.008856 {
                x * 7.787 + 0.137_931_03
            } else {
                x.cbrt()
            };
            *slot = saturate_u16((1 << LAB_SHIFT2) as f32 * v);
        }
        const SRGB2XYZ_D65: [f32; 9] = [
            0.412453, 0.357580, 0.180423, //
            0.212671, 0.715160, 0.072169, //
            0.019334, 0.119193, 0.950227,
        ];
        const WHITEPT: [f32; 3] = [0.950456, 1.0, 1.088754];
        let mut coeffs = [0i32; 9];
        for i in 0..3 {
            for j in 0..3 {
                let v = SRGB2XYZ_D65[i * 3 + j] * (1 << LAB_SHIFT) as f32 / WHITEPT[i];
                coeffs[i * 3 + j] = cv_round(v as f64);
            }
        }
        LabTabs {
            gamma,
            cbrt,
            coeffs,
        }
    })
}

#[inline]
fn saturate_u16(v: f32) -> u16 {
    let r = cv_round(v as f64);
    r.clamp(0, u16::MAX as i32) as u16
}

/// OpenCV's `cvRound`: round half away from zero is *not* used; it rounds half
/// to even like the underlying hardware instruction.
#[inline]
pub fn cv_round(v: f64) -> i32 {
    let r = v.round_ties_even();
    r as i32
}

#[inline]
fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// Lightness channel of `cv2.cvtColor(src, cv2.COLOR_RGB2Lab)` for `CV_8U`.
///
/// Only L is produced: the fusion weights are derived from luminance alone, so
/// the a/b channels of the reference implementation are never read.
pub fn rgb_to_lab_l_u8(src: &MatU8) -> MatU8 {
    assert_eq!(src.c, 3, "expected 3-channel RGB input");
    let t = lab_tabs();
    let l_scale = (116 * 255 + 50) / 100;
    let l_shift = -((16 * 255 * (1 << LAB_SHIFT2)) / 100);
    let (c3, c4, c5) = (t.coeffs[3], t.coeffs[4], t.coeffs[5]);
    let mut dst = MatU8::new(src.h, src.w, 1);
    for (out, px) in dst.data.iter_mut().zip(src.data.chunks_exact(3)) {
        let r = t.gamma[px[0] as usize] as i32;
        let g = t.gamma[px[1] as usize] as i32;
        let b = t.gamma[px[2] as usize] as i32;
        let yi =
            descale(r * c3 + g * c4 + b * c5, LAB_SHIFT).clamp(0, LAB_CBRT_TAB_SIZE_B as i32 - 1);
        let fy = t.cbrt[yi as usize] as i32;
        let l = descale(l_scale * fy + l_shift, LAB_SHIFT2);
        *out = l.clamp(0, 255) as u8;
    }
    dst
}

/// Lab lightness as `f32`, which is how the fusion code consumes it.
pub fn rgb_to_lab_l_f32(src: &MatU8) -> Mat {
    rgb_to_lab_l_u8(src).to_f32()
}
