//! The image-processing primitives, supplied by OpenCV's `imgproc`.
//!
//! Only `imgproc` is used. Registration stays on this crate's own ECC solver
//! and phase correlation: measured against OpenCV's they run at the same
//! speed, and `opencv_video` exists solely to supply `findTransformECC`, which
//! drags in dnn, calib3d, features2d and flann - 6.7 MB of linked code for no
//! gain.
//!
//! Conversion in both directions borrows rather than copies, so timings
//! reflect OpenCV's kernels and not marshalling overhead.

use crate::affine::Affine;
use crate::border::Border;
use crate::mat::{Mat, MatU16, MatU8};
use opencv::core::{BorderTypes, Mat as CvMat, Point, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;

/// Resampling kernel, mapped straight onto OpenCV's interpolation flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Linear,
    Cubic,
    Nearest,
}

/// OpenCV's `cvRound`, which rounds halves to even like the hardware does.
#[inline]
pub fn cv_round(v: f64) -> i32 {
    v.round_ties_even() as i32
}

fn border_flag(border: Border) -> i32 {
    match border {
        Border::Reflect => BorderTypes::BORDER_REFLECT as i32,
        Border::Reflect101 => BorderTypes::BORDER_REFLECT_101 as i32,
        Border::Constant => BorderTypes::BORDER_CONSTANT as i32,
    }
}

/// Borrow one of this crate's images as an OpenCV `Mat` of the same shape.
///
/// A `Mat` built over `w * c` single-channel columns reshapes into `c`
/// channels without touching the data, which is how interleaved buffers cross
/// the boundary for free. The reshaped view borrows from the flat one, so both
/// are bound in the caller's scope rather than returned.
macro_rules! as_cv {
    ($flat:ident, $view:ident, $src:expr, $ty:ty) => {
        let src = $src;
        let $flat =
            CvMat::new_rows_cols_with_data::<$ty>(src.h as i32, (src.w * src.c) as i32, &src.data)
                .expect("wrap buffer");
        let $view = $flat
            .reshape(src.c as i32, src.h as i32)
            .expect("reshape channels");
    };
}

/// Bind an already-sized destination buffer as an OpenCV `Mat`.
///
/// OpenCV writes into an output array that already has the right size and
/// type rather than reallocating, so the result lands straight in the
/// caller's buffer and nothing is copied back.
macro_rules! out_cv {
    ($flat:ident, $view:ident, $dst:expr, $ty:ty) => {
        let dst = $dst;
        let (dh, dw, dc) = (dst.h, dst.w, dst.c);
        let mut $flat =
            CvMat::new_rows_cols_with_data_mut::<$ty>(dh as i32, (dw * dc) as i32, &mut dst.data)
                .expect("wrap destination");
        let mut $view = $flat
            .reshape_mut(dc as i32, dh as i32)
            .expect("reshape destination");
    };
}

#[allow(dead_code)]
fn from_cv_f32(m: &CvMat) -> Mat {
    let c = m.channels() as usize;
    let (h, w) = (m.rows() as usize, m.cols() as usize);
    let flat = m.reshape(1, h as i32).expect("reshape to flat");
    let data = flat.data_typed::<f32>().expect("read f32").to_vec();
    Mat { h, w, c, data }
}

#[allow(dead_code)]
fn from_cv_u8(m: &CvMat) -> MatU8 {
    let c = m.channels() as usize;
    let (h, w) = (m.rows() as usize, m.cols() as usize);
    let flat = m.reshape(1, h as i32).expect("reshape to flat");
    let data = flat.data_typed::<u8>().expect("read u8").to_vec();
    MatU8 { h, w, c, data }
}

#[allow(dead_code)]
fn from_cv_u16(m: &CvMat) -> MatU16 {
    let c = m.channels() as usize;
    let (h, w) = (m.rows() as usize, m.cols() as usize);
    let flat = m.reshape(1, h as i32).expect("reshape to flat");
    let data = flat.data_typed::<u16>().expect("read u16").to_vec();
    MatU16 { h, w, c, data }
}

fn kernel_mat(k: &[f32]) -> opencv::boxed_ref::BoxedRef<'_, CvMat> {
    CvMat::new_rows_cols_with_data::<f32>(k.len() as i32, 1, k).expect("wrap kernel")
}

pub fn sep_filter(src: &Mat, kx: &[f32], ky: &[f32], border: Border) -> Mat {
    as_cv!(flat_input, input, src, f32);
    let mut out = Mat::new(src.h, src.w, src.c);
    out_cv!(flat_out, dst, &mut out, f32);
    imgproc::sep_filter_2d(
        &input,
        &mut dst,
        opencv::core::CV_32F,
        &kernel_mat(kx),
        &kernel_mat(ky),
        Point::new(-1, -1),
        0.0,
        border_flag(border),
    )
    .expect("sepFilter2D");
    drop(dst);
    out
}

/// `cv2.filter2D` with a single-channel kernel and `CV_32F` output.
pub fn filter_2d(src: &Mat, kernel: &Mat, border: Border) -> Mat {
    assert_eq!(kernel.c, 1, "filter kernels are single channel");
    as_cv!(flat_input, input, src, f32);
    as_cv!(flat_kernel, kernel_cv, kernel, f32);
    let mut out = Mat::new(src.h, src.w, src.c);
    out_cv!(flat_out, dst, &mut out, f32);
    imgproc::filter_2d(
        &input,
        &mut dst,
        opencv::core::CV_32F,
        &kernel_cv,
        Point::new(-1, -1),
        0.0,
        border_flag(border),
    )
    .expect("filter2D");
    drop(dst);
    out
}

pub fn gaussian_blur(src: &Mat, ksize: usize, sigma: f64) -> Mat {
    if ksize == 1 {
        return src.clone();
    }
    as_cv!(flat_input, input, src, f32);
    let mut out = Mat::new(src.h, src.w, src.c);
    out_cv!(flat_out, dst, &mut out, f32);
    // The `_def` form is used deliberately. OpenCV 4.11 added an
    // `AlgorithmHint` parameter to GaussianBlur and cvtColor, so the explicit
    // signatures differ between versions and will not compile against both.
    // The defaults it supplies are exactly what this call wants anyway:
    // sigmaY = 0 means "same as sigmaX", and BORDER_DEFAULT is
    // BORDER_REFLECT_101.
    imgproc::gaussian_blur_def(
        &input,
        &mut dst,
        Size::new(ksize as i32, ksize as i32),
        sigma,
    )
    .expect("GaussianBlur");
    drop(dst);
    out
}

pub fn sobel(src: &Mat, dx: usize, dy: usize, ksize: usize) -> Mat {
    as_cv!(flat_input, input, src, f32);
    let mut out = Mat::new(src.h, src.w, src.c);
    out_cv!(flat_out, dst, &mut out, f32);
    imgproc::sobel(
        &input,
        &mut dst,
        opencv::core::CV_32F,
        dx as i32,
        dy as i32,
        ksize as i32,
        1.0,
        0.0,
        BorderTypes::BORDER_REFLECT_101 as i32,
    )
    .expect("Sobel");
    drop(dst);
    out
}

pub fn laplacian3(src: &Mat) -> Mat {
    as_cv!(flat_input, input, src, f32);
    let mut out = Mat::new(src.h, src.w, src.c);
    out_cv!(flat_out, dst, &mut out, f32);
    imgproc::laplacian(
        &input,
        &mut dst,
        opencv::core::CV_32F,
        3,
        1.0,
        0.0,
        BorderTypes::BORDER_REFLECT_101 as i32,
    )
    .expect("Laplacian");
    drop(dst);
    out
}

pub fn box_filter(src: &Mat, kw: usize, kh: usize, border: Border) -> Mat {
    as_cv!(flat_input, input, src, f32);
    let mut out = Mat::new(src.h, src.w, src.c);
    out_cv!(flat_out, dst, &mut out, f32);
    imgproc::box_filter(
        &input,
        &mut dst,
        opencv::core::CV_32F,
        Size::new(kw as i32, kh as i32),
        Point::new(-1, -1),
        true,
        border_flag(border),
    )
    .expect("boxFilter");
    drop(dst);
    out
}

pub fn sqr_box_filter(src: &Mat, kw: usize, kh: usize, border: Border) -> Mat {
    as_cv!(flat_input, input, src, f32);
    let mut out = Mat::new(src.h, src.w, src.c);
    out_cv!(flat_out, dst, &mut out, f32);
    imgproc::sqr_box_filter(
        &input,
        &mut dst,
        opencv::core::CV_32F,
        Size::new(kw as i32, kh as i32),
        Point::new(-1, -1),
        true,
        border_flag(border),
    )
    .expect("sqrBoxFilter");
    drop(dst);
    out
}

pub fn dilate_ellipse(src: &MatU8, kw: usize, kh: usize) -> MatU8 {
    as_cv!(flat_input, input, src, u8);
    let element = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(kw as i32, kh as i32),
        Point::new(-1, -1),
    )
    .expect("structuring element");
    let mut out = MatU8::new(src.h, src.w, 1);
    out_cv!(flat_out, dst, &mut out, u8);
    imgproc::dilate(
        &input,
        &mut dst,
        &element,
        Point::new(-1, -1),
        1,
        BorderTypes::BORDER_CONSTANT as i32,
        imgproc::morphology_default_border_value().expect("border value"),
    )
    .expect("dilate");
    drop(dst);
    out
}

pub fn resize_area(src: &Mat, dst_w: usize, dst_h: usize) -> Mat {
    if src.w == dst_w && src.h == dst_h {
        return src.clone();
    }
    as_cv!(flat_input, input, src, f32);
    let mut out = Mat::new(dst_h, dst_w, src.c);
    out_cv!(flat_out, dst, &mut out, f32);
    imgproc::resize(
        &input,
        &mut dst,
        Size::new(dst_w as i32, dst_h as i32),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )
    .expect("resize");
    drop(dst);
    out
}

pub fn resize_area_u8(src: &MatU8, dst_w: usize, dst_h: usize) -> MatU8 {
    if src.w == dst_w && src.h == dst_h {
        return src.clone();
    }
    as_cv!(flat_input, input, src, u8);
    let mut out = MatU8::new(dst_h, dst_w, src.c);
    out_cv!(flat_out, dst, &mut out, u8);
    imgproc::resize(
        &input,
        &mut dst,
        Size::new(dst_w as i32, dst_h as i32),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )
    .expect("resize");
    drop(dst);
    out
}

fn interp_flag(interp: Interp) -> i32 {
    match interp {
        Interp::Linear => imgproc::INTER_LINEAR,
        Interp::Cubic => imgproc::INTER_CUBIC,
        Interp::Nearest => imgproc::INTER_NEAREST,
    }
}

fn warp_matrix(m: &Affine) -> CvMat {
    CvMat::from_slice_2d(&[[m.0[0], m.0[1], m.0[2]], [m.0[3], m.0[4], m.0[5]]])
        .expect("warp matrix")
}

fn warp_flags(interp: Interp, inverse_map: bool) -> i32 {
    let mut flags = interp_flag(interp);
    if inverse_map {
        flags |= opencv::imgproc::WARP_INVERSE_MAP;
    }
    flags
}

pub fn warp_affine(
    src: &Mat,
    m: &Affine,
    dst_w: usize,
    dst_h: usize,
    interp: Interp,
    border: Border,
    inverse_map: bool,
) -> Mat {
    as_cv!(flat_input, input, src, f32);
    let mut out = Mat::new(dst_h, dst_w, src.c);
    out_cv!(flat_out, dst, &mut out, f32);
    imgproc::warp_affine(
        &input,
        &mut dst,
        &warp_matrix(m),
        Size::new(dst_w as i32, dst_h as i32),
        warp_flags(interp, inverse_map),
        border_flag(border),
        Scalar::all(0.0),
    )
    .expect("warpAffine");
    drop(dst);
    out
}

pub fn warp_affine_u8(
    src: &MatU8,
    m: &Affine,
    dst_w: usize,
    dst_h: usize,
    interp: Interp,
    border: Border,
) -> MatU8 {
    as_cv!(flat_input, input, src, u8);
    let mut out = MatU8::new(dst_h, dst_w, src.c);
    out_cv!(flat_out, dst, &mut out, u8);
    imgproc::warp_affine(
        &input,
        &mut dst,
        &warp_matrix(m),
        Size::new(dst_w as i32, dst_h as i32),
        warp_flags(interp, false),
        border_flag(border),
        Scalar::all(0.0),
    )
    .expect("warpAffine");
    drop(dst);
    out
}

pub fn warp_affine_u16(
    src: &MatU16,
    m: &Affine,
    dst_w: usize,
    dst_h: usize,
    interp: Interp,
    border: Border,
) -> MatU16 {
    as_cv!(flat_input, input, src, u16);
    let mut out = MatU16::new(dst_h, dst_w, src.c);
    out_cv!(flat_out, dst, &mut out, u16);
    imgproc::warp_affine(
        &input,
        &mut dst,
        &warp_matrix(m),
        Size::new(dst_w as i32, dst_h as i32),
        warp_flags(interp, false),
        border_flag(border),
        Scalar::all(0.0),
    )
    .expect("warpAffine");
    drop(dst);
    out
}

pub fn rgb_to_gray_u8(src: &MatU8) -> MatU8 {
    as_cv!(flat_input, input, src, u8);
    let mut out = MatU8::new(src.h, src.w, 1);
    out_cv!(flat_out, dst, &mut out, u8);
    // `_def` for the same version-portability reason as GaussianBlur above;
    // its default dstCn of 0 is what this call passed explicitly.
    imgproc::cvt_color_def(&input, &mut dst, imgproc::COLOR_RGB2GRAY).expect("cvtColor gray");
    drop(dst);
    out
}

pub fn rgb_to_lab_l_u8(src: &MatU8) -> MatU8 {
    as_cv!(flat_input, input, src, u8);
    let mut full = MatU8::new(src.h, src.w, 3);
    out_cv!(flat_out, lab, &mut full, u8);
    imgproc::cvt_color_def(&input, &mut lab, imgproc::COLOR_RGB2Lab).expect("cvtColor Lab");
    drop(lab);
    MatU8 {
        h: full.h,
        w: full.w,
        c: 1,
        data: full.data.chunks_exact(3).map(|px| px[0]).collect(),
    }
}

pub fn clahe(src: &MatU8, clip_limit: f64, tiles_x: usize, tiles_y: usize) -> MatU8 {
    as_cv!(flat_input, input, src, u8);
    let mut clahe = imgproc::create_clahe(clip_limit, Size::new(tiles_x as i32, tiles_y as i32))
        .expect("createCLAHE");
    let mut out = MatU8::new(src.h, src.w, 1);
    out_cv!(flat_out, dst, &mut out, u8);
    clahe.apply(&input, &mut dst).expect("CLAHE apply");
    drop(dst);
    out
}

/// Lab lightness as `f32`, which is how the fusion code consumes it.
pub fn rgb_to_lab_l_f32(src: &MatU8) -> Mat {
    rgb_to_lab_l_u8(src).to_f32()
}
