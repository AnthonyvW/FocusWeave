//! Runs each ported primitive on a fixed input and dumps the result for the
//! cv2 cross-check in tests/compare_primitives.py.

use focusweave_core::affine::Affine;
use focusweave_core::border::Border;
use focusweave_core::cv::{self, Interp};
use focusweave_core::testio;
use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).expect("output directory"));
    let rgb = testio::read_u8(dir.join("in_rgb.bin"));
    let gray = testio::read_u8(dir.join("in_gray.bin"));
    let grayf = gray.to_f32();

    let k1d: Vec<f32> = [1.0f32, 4.0, 6.0, 4.0, 1.0]
        .iter()
        .map(|v| v / 16.0)
        .collect();
    let k1d_x2: Vec<f32> = k1d.iter().map(|v| v * 2.0).collect();

    testio::write_f32(
        dir.join("out_sep_reflect.bin"),
        &cv::sep_filter(&grayf, &k1d, &k1d, Border::Reflect),
    );
    let rgbf = rgb.to_f32();
    testio::write_f32(
        dir.join("out_sep_rgb.bin"),
        &cv::sep_filter(&rgbf, &k1d_x2, &k1d_x2, Border::Reflect),
    );
    testio::write_f32(
        dir.join("out_box3.bin"),
        &cv::box_filter(&grayf, 3, 3, Border::Reflect),
    );
    testio::write_f32(
        dir.join("out_box8.bin"),
        &cv::box_filter(&grayf, 8, 8, Border::Reflect),
    );
    testio::write_f32(
        dir.join("out_sqrbox3.bin"),
        &cv::sqr_box_filter(&grayf, 3, 3, Border::Reflect),
    );
    testio::write_f32(
        dir.join("out_gauss31.bin"),
        &cv::gaussian_blur(&grayf, 31, 0.0),
    );
    testio::write_f32(
        dir.join("out_gauss15.bin"),
        &cv::gaussian_blur(&grayf, 15, 0.0),
    );
    testio::write_f32(
        dir.join("out_gauss3.bin"),
        &cv::gaussian_blur(&grayf, 3, 0.0),
    );
    testio::write_f32(
        dir.join("out_gauss5.bin"),
        &cv::gaussian_blur(&grayf, 5, 0.0),
    );
    testio::write_f32(dir.join("out_sobel_x.bin"), &cv::sobel(&grayf, 1, 0, 5));
    testio::write_f32(dir.join("out_sobel_y.bin"), &cv::sobel(&grayf, 0, 1, 5));
    testio::write_f32(dir.join("out_laplacian.bin"), &cv::laplacian3(&grayf));

    let mask = focusweave_core::mat::MatU8::from_vec(
        gray.h,
        gray.w,
        1,
        gray.data
            .iter()
            .map(|v| if *v > 128 { 255u8 } else { 0 })
            .collect(),
    );
    testio::write_u8(dir.join("out_dilate.bin"), &cv::dilate_ellipse(&mask, 7, 7));

    testio::write_u8(dir.join("out_gray_from_rgb.bin"), &cv::rgb_to_gray_u8(&rgb));
    testio::write_u8(dir.join("out_lab_l.bin"), &cv::rgb_to_lab_l_u8(&rgb));
    testio::write_u8(dir.join("out_clahe.bin"), &cv::clahe(&gray, 2.0, 8, 8));

    testio::write_u8(
        dir.join("out_resize_small.bin"),
        &cv::resize_area_u8(&rgb, 41, 29),
    );
    testio::write_f32(
        dir.join("out_resize_f32.bin"),
        &cv::resize_area(&grayf, 53, 37),
    );

    let m = Affine([1.004, -0.013, 7.35, 0.011, 0.997, -4.2]);
    testio::write_u8(
        dir.join("out_warp_cubic.bin"),
        &cv::warp_affine_u8(&rgb, &m, rgb.w, rgb.h, Interp::Cubic, Border::Reflect),
    );
    let t = Affine([1.0, 0.0, 6.0, 0.0, 1.0, -3.0]);
    testio::write_u8(
        dir.join("out_warp_translate.bin"),
        &cv::warp_affine_u8(&rgb, &t, rgb.w, rgb.h, Interp::Linear, Border::Constant),
    );

    println!("wrote primitive dumps to {}", dir.display());
}
