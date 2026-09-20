//! OpenCV border extrapolation modes.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Border {
    /// `fedcba|abcdefgh|hgfedcb` — OpenCV `BORDER_REFLECT`.
    Reflect,
    /// `gfedcb|abcdefgh|gfedcba` — OpenCV `BORDER_REFLECT_101` (a.k.a. `BORDER_DEFAULT`).
    Reflect101,
    /// Out-of-range reads yield zero — OpenCV `BORDER_CONSTANT` with value 0.
    Constant,
}

/// Map a possibly out-of-range coordinate onto `[0, n)`.
///
/// Returns `None` for [`Border::Constant`], where the caller must substitute zero.
#[inline]
pub fn border_index(mut i: isize, n: usize, mode: Border) -> Option<usize> {
    if n == 0 {
        return None;
    }
    if i >= 0 && (i as usize) < n {
        return Some(i as usize);
    }
    let n_i = n as isize;
    match mode {
        Border::Constant => None,
        Border::Reflect => {
            if n == 1 {
                return Some(0);
            }
            loop {
                if i < 0 {
                    i = -i - 1;
                } else if i >= n_i {
                    i = 2 * n_i - i - 1;
                } else {
                    return Some(i as usize);
                }
            }
        }
        Border::Reflect101 => {
            if n == 1 {
                return Some(0);
            }
            loop {
                if i < 0 {
                    i = -i;
                } else if i >= n_i {
                    i = 2 * n_i - i - 2;
                } else {
                    return Some(i as usize);
                }
            }
        }
    }
}

/// Precompute the source index for every tap position of a 1-D pass.
///
/// `table[i * ksize + k]` is the source index for output `i`, tap `k`, or
/// `usize::MAX` when the tap falls outside a [`Border::Constant`] image.
pub fn border_table(n: usize, ksize: usize, anchor: usize, mode: Border) -> Vec<usize> {
    let mut table = vec![usize::MAX; n * ksize];
    for i in 0..n {
        for k in 0..ksize {
            let src = i as isize + k as isize - anchor as isize;
            table[i * ksize + k] = border_index(src, n, mode).unwrap_or(usize::MAX);
        }
    }
    table
}
