//! Phase correlation, equivalent to `cv2.phaseCorrelate` without a window.

use crate::mat::Mat;
use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

/// `cv2.getOptimalDFTSize`: the smallest 5-smooth integer at or above `n`.
pub fn optimal_dft_size(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut candidate = n;
    loop {
        let mut m = candidate;
        for f in [2usize, 3, 5] {
            while m % f == 0 {
                m /= f;
            }
        }
        if m == 1 {
            return candidate;
        }
        candidate += 1;
    }
}

struct Fft2 {
    rows: std::sync::Arc<dyn rustfft::Fft<f64>>,
    cols: std::sync::Arc<dyn rustfft::Fft<f64>>,
    h: usize,
    w: usize,
}

impl Fft2 {
    fn new(h: usize, w: usize, inverse: bool) -> Fft2 {
        let mut planner = FftPlanner::new();
        let (rows, cols) = if inverse {
            (planner.plan_fft_inverse(w), planner.plan_fft_inverse(h))
        } else {
            (planner.plan_fft_forward(w), planner.plan_fft_forward(h))
        };
        Fft2 { rows, cols, h, w }
    }

    fn run(&self, buf: &mut [Complex64]) {
        for row in buf.chunks_exact_mut(self.w) {
            self.rows.process(row);
        }
        let mut column = vec![Complex64::new(0.0, 0.0); self.h];
        for x in 0..self.w {
            for (y, slot) in column.iter_mut().enumerate() {
                *slot = buf[y * self.w + x];
            }
            self.cols.process(&mut column);
            for (y, slot) in column.iter().enumerate() {
                buf[y * self.w + x] = *slot;
            }
        }
    }
}

fn zero_padded(src: &Mat, h: usize, w: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); h * w];
    for y in 0..src.h {
        for x in 0..src.w {
            out[y * w + x].re = f64::from(src.data[y * src.w + x]);
        }
    }
    out
}

/// OpenCV's quadrant swap, which rolls by `floor(n / 2)` on each axis for both
/// even and odd extents.
fn fft_shift(buf: &[f64], h: usize, w: usize) -> Vec<f64> {
    let dy = h / 2;
    let dx = w / 2;
    let mut out = vec![0.0f64; h * w];
    for y in 0..h {
        let ty = (y + dy) % h;
        for x in 0..w {
            out[ty * w + (x + dx) % w] = buf[y * w + x];
        }
    }
    out
}

/// Sub-pixel translation between two single-channel images.
///
/// Mirrors `cv2.phaseCorrelate(src1, src2)`, including its zero padding to an
/// optimal DFT size and the 5x5 weighted centroid around the correlation peak.
pub fn phase_correlate(src1: &Mat, src2: &Mat) -> (f64, f64) {
    assert!(
        src1.c == 1 && src2.c == 1,
        "phase correlation is single channel"
    );
    assert!(
        src1.h == src2.h && src1.w == src2.w,
        "inputs must match in size"
    );
    let m = optimal_dft_size(src1.h);
    let n = optimal_dft_size(src1.w);

    let mut f1 = zero_padded(src1, m, n);
    let mut f2 = zero_padded(src2, m, n);
    let fwd = Fft2::new(m, n, false);
    fwd.run(&mut f1);
    fwd.run(&mut f2);

    // Cross-power spectrum, normalised to unit magnitude.
    let mut cross: Vec<Complex64> = f1
        .iter()
        .zip(&f2)
        .map(|(a, b)| {
            let p = a * b.conj();
            let mag = p.norm();
            let denom = mag * mag + f64::EPSILON;
            p * (mag / denom)
        })
        .collect();

    Fft2::new(m, n, true).run(&mut cross);
    let real: Vec<f64> = cross.iter().map(|c| c.re).collect();
    let shifted = fft_shift(&real, m, n);

    let mut peak = 0usize;
    for (i, v) in shifted.iter().enumerate() {
        if *v > shifted[peak] {
            peak = i;
        }
    }
    let (py, px) = (peak / n, peak % n);

    let (cx, cy) = weighted_centroid(&shifted, m, n, py, px, 5, 5);
    (n as f64 / 2.0 - cx, m as f64 / 2.0 - cy)
}

fn weighted_centroid(
    src: &[f64],
    h: usize,
    w: usize,
    peak_y: usize,
    peak_x: usize,
    box_h: usize,
    box_w: usize,
) -> (f64, f64) {
    let minr = peak_y.saturating_sub(box_h >> 1);
    let minc = peak_x.saturating_sub(box_w >> 1);
    let maxr = (peak_y + (box_h >> 1)).min(h - 1);
    let maxc = (peak_x + (box_w >> 1)).min(w - 1);
    let mut sum = 0.0f64;
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    for y in minr..=maxr {
        for x in minc..=maxc {
            let v = src[y * w + x];
            cx += x as f64 * v;
            cy += y as f64 * v;
            sum += v;
        }
    }
    sum += f64::EPSILON;
    (cx / sum, cy / sum)
}
