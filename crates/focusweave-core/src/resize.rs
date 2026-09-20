//! Area-averaging resampling, equivalent to `cv2.resize(..., INTER_AREA)`.

use crate::mat::{Mat, MatU8};

struct Tap {
    src: usize,
    alpha: f32,
}

/// OpenCV's `computeResizeAreaTab`, grouped per destination index.
fn area_tabs(ssize: usize, dsize: usize, scale: f64) -> Vec<Vec<Tap>> {
    let mut out: Vec<Vec<Tap>> = (0..dsize).map(|_| Vec::new()).collect();
    for (dx, taps) in out.iter_mut().enumerate() {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as isize;
        let sx2 = (fsx2.floor() as isize).min(ssize as isize - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            taps.push(Tap {
                src: (sx1 - 1).max(0) as usize,
                alpha: ((sx1 as f64 - fsx1) / cell) as f32,
            });
        }
        for sx in sx1..sx2 {
            taps.push(Tap {
                src: sx.max(0) as usize,
                alpha: (1.0 / cell) as f32,
            });
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            let a = (fsx2 - sx2 as f64).min(1.0).min(cell) / cell;
            taps.push(Tap {
                src: sx2.max(0) as usize,
                alpha: a as f32,
            });
        }
    }
    out
}

/// OpenCV's "emulated area" coefficients, used when the axis is being upscaled.
fn upscale_tabs(ssize: usize, dsize: usize, scale: f64) -> Vec<Vec<Tap>> {
    let inv = 1.0 / scale;
    (0..dsize)
        .map(|dx| {
            let mut sx = (dx as f64 * scale).floor() as isize;
            let mut fx = (dx as f64 + 1.0) - (sx as f64 + 1.0) * inv;
            fx = if fx <= 0.0 { 0.0 } else { fx - fx.floor() };
            if sx < 0 {
                sx = 0;
                fx = 0.0;
            }
            if sx >= ssize as isize - 1 {
                sx = ssize as isize - 1;
                fx = 0.0;
            }
            let s0 = sx as usize;
            let s1 = (s0 + 1).min(ssize - 1);
            vec![
                Tap {
                    src: s0,
                    alpha: (1.0 - fx) as f32,
                },
                Tap {
                    src: s1,
                    alpha: fx as f32,
                },
            ]
        })
        .collect()
}

fn axis_tabs(ssize: usize, dsize: usize) -> (Vec<Vec<Tap>>, bool) {
    let scale = ssize as f64 / dsize as f64;
    (
        if scale >= 1.0 {
            area_tabs(ssize, dsize, scale)
        } else {
            upscale_tabs(ssize, dsize, scale)
        },
        scale >= 1.0,
    )
}

/// Resize a float image to `(dst_w, dst_h)` using area averaging.
pub fn resize_area(src: &Mat, dst_w: usize, dst_h: usize) -> Mat {
    if src.w == dst_w && src.h == dst_h {
        return src.clone();
    }
    let (xt, _) = axis_tabs(src.w, dst_w);
    let (yt, _) = axis_tabs(src.h, dst_h);
    let c = src.c;

    // Horizontal pass first so the vertical pass works on the smaller buffer.
    let mut mid = Mat::new(src.h, dst_w, c);
    for y in 0..src.h {
        let srow = src.row(y);
        let drow = mid.row_mut(y);
        for (dx, taps) in xt.iter().enumerate() {
            for ch in 0..c {
                let mut acc = 0.0f32;
                for t in taps {
                    acc += srow[t.src * c + ch] * t.alpha;
                }
                drow[dx * c + ch] = acc;
            }
        }
    }

    let mut dst = Mat::new(dst_h, dst_w, c);
    let stride = dst_w * c;
    for (dy, taps) in yt.iter().enumerate() {
        let drow = dst.row_mut(dy);
        drow.fill(0.0);
        for t in taps {
            let srow = &mid.data[t.src * stride..(t.src + 1) * stride];
            for (d, s) in drow.iter_mut().zip(srow) {
                *d += *s * t.alpha;
            }
        }
    }
    dst
}

/// Resize an 8-bit image, rounding the way OpenCV's integer path does.
pub fn resize_area_u8(src: &MatU8, dst_w: usize, dst_h: usize) -> MatU8 {
    if src.w == dst_w && src.h == dst_h {
        return src.clone();
    }
    let f = Mat {
        h: src.h,
        w: src.w,
        c: src.c,
        data: src.data.iter().map(|v| f32::from(*v)).collect(),
    };
    let r = resize_area(&f, dst_w, dst_h);
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
