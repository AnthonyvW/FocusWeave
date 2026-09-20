//! Runtime CPU dispatch for the kernels that dominate a run.
//!
//! The hot loops are written as plain Rust that the autovectoriser can widen;
//! the only thing missing from a portable binary is permission to emit AVX2
//! and FMA. Each kernel is therefore compiled twice — once for the baseline
//! target and once with those features enabled — and the right one is chosen
//! on first use. Accumulation order is identical in both, so results do not
//! depend on which one runs.

/// Number of f32 accumulated at a time. Sized so a chunk and the source
/// windows feeding it stay resident in L1 across every tap.
const CHUNK: usize = 1024;

/// `out[i] = sum_j base[i + j * stride] * k[j]`, for a separable row pass.
///
/// Chunking is what makes this cheap: accumulating tap by tap over a whole
/// row reads and rewrites the destination once per tap, which is bandwidth
/// bound, while doing it a chunk at a time keeps the running sum in cache and
/// still leaves each inner loop a contiguous multiply-add.
#[inline(always)]
fn row_taps_impl(out: &mut [f32], base: &[f32], k: &[f32], stride: usize) {
    for (index, chunk) in out.chunks_mut(CHUNK).enumerate() {
        let offset = index * CHUNK;
        let n = chunk.len();
        for (j, weight) in k.iter().enumerate() {
            let taps = &base[offset + j * stride..offset + j * stride + n];
            let weight = *weight;
            if j == 0 {
                for (d, s) in chunk.iter_mut().zip(taps) {
                    *d = *s * weight;
                }
            } else {
                for (d, s) in chunk.iter_mut().zip(taps) {
                    *d += *s * weight;
                }
            }
        }
    }
}

/// `out[i] = sum_j rows[j].0[i] * rows[j].1`, for a separable column pass.
#[inline(always)]
fn col_taps_impl(out: &mut [f32], rows: &[(&[f32], f32)]) {
    for (index, chunk) in out.chunks_mut(CHUNK).enumerate() {
        let offset = index * CHUNK;
        let n = chunk.len();
        for (j, (row, weight)) in rows.iter().enumerate() {
            let taps = &row[offset..offset + n];
            let weight = *weight;
            if j == 0 {
                for (d, s) in chunk.iter_mut().zip(taps) {
                    *d = *s * weight;
                }
            } else {
                for (d, s) in chunk.iter_mut().zip(taps) {
                    *d += *s * weight;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_taps_avx2(out: &mut [f32], base: &[f32], k: &[f32], stride: usize) {
    row_taps_impl(out, base, k, stride)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn col_taps_avx2(out: &mut [f32], rows: &[(&[f32], f32)]) {
    col_taps_impl(out, rows)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn row_taps_neon(out: &mut [f32], base: &[f32], k: &[f32], stride: usize) {
    row_taps_impl(out, base, k, stride)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn col_taps_neon(out: &mut [f32], rows: &[(&[f32], f32)]) {
    col_taps_impl(out, rows)
}

/// True when this CPU offers the widened path.
pub fn accelerated() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
    }
    #[cfg(target_arch = "aarch64")]
    {
        return std::arch::is_aarch64_feature_detected!("neon");
    }
    #[allow(unreachable_code)]
    false
}

pub fn row_taps(out: &mut [f32], base: &[f32], k: &[f32], stride: usize) {
    #[cfg(target_arch = "x86_64")]
    if accelerated() {
        // SAFETY: guarded by the feature detection immediately above.
        return unsafe { row_taps_avx2(out, base, k, stride) };
    }
    #[cfg(target_arch = "aarch64")]
    if accelerated() {
        // SAFETY: guarded by the feature detection immediately above.
        return unsafe { row_taps_neon(out, base, k, stride) };
    }
    row_taps_impl(out, base, k, stride)
}

pub fn col_taps(out: &mut [f32], rows: &[(&[f32], f32)]) {
    #[cfg(target_arch = "x86_64")]
    if accelerated() {
        // SAFETY: guarded by the feature detection immediately above.
        return unsafe { col_taps_avx2(out, rows) };
    }
    #[cfg(target_arch = "aarch64")]
    if accelerated() {
        // SAFETY: guarded by the feature detection immediately above.
        return unsafe { col_taps_neon(out, rows) };
    }
    col_taps_impl(out, rows)
}
