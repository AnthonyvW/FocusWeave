//! Border extrapolation, as OpenCV names it.

/// How a filter or warp reads outside the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Border {
    /// `fedcba|abcdefgh|hgfedcb` — OpenCV `BORDER_REFLECT`.
    Reflect,
    /// `gfedcb|abcdefgh|gfedcba` — OpenCV `BORDER_REFLECT_101`, its default.
    Reflect101,
    /// Out-of-range reads yield zero — OpenCV `BORDER_CONSTANT` with value 0.
    Constant,
}
