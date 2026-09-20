//! Contrast-limited adaptive histogram equalisation, matching `cv2.createCLAHE`.

use crate::border::{border_index, Border};
use crate::mat::MatU8;

const HIST_SIZE: usize = 256;

/// `cv2.createCLAHE(clipLimit, (tiles_x, tiles_y)).apply(src)` for `CV_8UC1`.
pub fn clahe(src: &MatU8, clip_limit: f64, tiles_x: usize, tiles_y: usize) -> MatU8 {
    assert_eq!(src.c, 1, "CLAHE operates on single-channel images");

    // Pad so the tile grid divides the image exactly; OpenCV reflects the edges.
    let (lut_src, tile_w, tile_h) = if src.w % tiles_x == 0 && src.h % tiles_y == 0 {
        (src.clone(), src.w / tiles_x, src.h / tiles_y)
    } else {
        let pw = src.w + (tiles_x - src.w % tiles_x);
        let ph = src.h + (tiles_y - src.h % tiles_y);
        let mut ext = MatU8::new(ph, pw, 1);
        for y in 0..ph {
            let sy = border_index(y as isize, src.h, Border::Reflect101).unwrap_or(0);
            for x in 0..pw {
                let sx = border_index(x as isize, src.w, Border::Reflect101).unwrap_or(0);
                ext.data[y * pw + x] = src.data[sy * src.w + sx];
            }
        }
        (ext, pw / tiles_x, ph / tiles_y)
    };

    let tile_total = tile_w * tile_h;
    let lut_scale = (HIST_SIZE - 1) as f32 / tile_total as f32;
    let clip = if clip_limit > 0.0 {
        ((clip_limit * tile_total as f64 / HIST_SIZE as f64) as i32).max(1)
    } else {
        0
    };

    let mut luts = vec![0u8; tiles_x * tiles_y * HIST_SIZE];
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let mut hist = [0i32; HIST_SIZE];
            for y in ty * tile_h..(ty + 1) * tile_h {
                let row = &lut_src.data[y * lut_src.w..y * lut_src.w + lut_src.w];
                for v in &row[tx * tile_w..(tx + 1) * tile_w] {
                    hist[*v as usize] += 1;
                }
            }

            if clip > 0 {
                let mut clipped = 0i32;
                for bin in hist.iter_mut() {
                    if *bin > clip {
                        clipped += *bin - clip;
                        *bin = clip;
                    }
                }
                let redist = clipped / HIST_SIZE as i32;
                let mut residual = clipped - redist * HIST_SIZE as i32;
                for bin in hist.iter_mut() {
                    *bin += redist;
                }
                if residual != 0 {
                    let step = (HIST_SIZE as i32 / residual).max(1) as usize;
                    let mut i = 0usize;
                    while i < HIST_SIZE && residual > 0 {
                        hist[i] += 1;
                        residual -= 1;
                        i += step;
                    }
                }
            }

            let lut =
                &mut luts[(ty * tiles_x + tx) * HIST_SIZE..(ty * tiles_x + tx + 1) * HIST_SIZE];
            let mut sum = 0i32;
            for (slot, bin) in lut.iter_mut().zip(hist.iter()) {
                sum += *bin;
                *slot = (sum as f32 * lut_scale).round_ties_even().clamp(0.0, 255.0) as u8;
            }
        }
    }

    let inv_tw = 1.0f32 / tile_w as f32;
    let inv_th = 1.0f32 / tile_h as f32;

    // Per-column tile indices and blend weights are shared across rows.
    let mut col_lo = vec![0usize; src.w];
    let mut col_hi = vec![0usize; src.w];
    let mut col_a = vec![0f32; src.w];
    for x in 0..src.w {
        let txf = x as f32 * inv_tw - 0.5;
        let tx1 = txf.floor();
        col_a[x] = txf - tx1;
        col_lo[x] = (tx1.max(0.0) as usize).min(tiles_x - 1);
        col_hi[x] = ((tx1 as i64 + 1).max(0) as usize).min(tiles_x - 1);
    }

    let mut dst = MatU8::new(src.h, src.w, 1);
    for y in 0..src.h {
        let tyf = y as f32 * inv_th - 0.5;
        let ty1f = tyf.floor();
        let ya = tyf - ty1f;
        let ty1 = (ty1f.max(0.0) as usize).min(tiles_y - 1);
        let ty2 = ((ty1f as i64 + 1).max(0) as usize).min(tiles_y - 1);
        let plane1 = ty1 * tiles_x * HIST_SIZE;
        let plane2 = ty2 * tiles_x * HIST_SIZE;
        for x in 0..src.w {
            let v = src.data[y * src.w + x] as usize;
            let xa = col_a[x];
            let i1 = col_lo[x] * HIST_SIZE + v;
            let i2 = col_hi[x] * HIST_SIZE + v;
            let res = luts[plane1 + i1] as f32 * ((1.0 - xa) * (1.0 - ya))
                + luts[plane1 + i2] as f32 * (xa * (1.0 - ya))
                + luts[plane2 + i1] as f32 * ((1.0 - xa) * ya)
                + luts[plane2 + i2] as f32 * (xa * ya);
            dst.data[y * src.w + x] = res.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    dst
}
