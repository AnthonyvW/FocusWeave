//! Times the primitives that dominate a run, for comparison against OpenCV.

use focusweave_core::border::Border;
use focusweave_core::warp::Interp;
use focusweave_core::{affine::Affine, color, filter, mat::*, pyramid, resize, warp};
use std::time::Instant;

fn bench(label: &str, reps: usize, mut f: impl FnMut()) {
    f(); // warm up
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    println!(
        "{label:28} {:8.2} ms",
        t.elapsed().as_secs_f64() * 1000.0 / reps as f64
    );
}

fn main() {
    // Pin OpenCV to one thread so the per-primitive figures are per core on
    // both sides; the pure-Rust kernels are measured with RAYON_NUM_THREADS=1.
    #[cfg(feature = "opencv-backend")]
    opencv::core::set_num_threads(1).expect("set_num_threads");

    let backend = if cfg!(feature = "opencv-backend") {
        "opencv"
    } else {
        "native"
    };
    println!("backend: {backend}");

    let (h, w) = (1400usize, 2000usize);
    let rgb8 = MatU8::from_vec(
        h,
        w,
        3,
        (0..h * w * 3).map(|i| ((i * 37) % 251) as u8).collect(),
    );
    let rgb = rgb8.to_f32();
    let gray8 = color::rgb_to_gray_u8(&rgb8);
    let gray = gray8.to_f32();

    let k: Vec<f32> = [1.0f32, 4.0, 6.0, 4.0, 1.0]
        .iter()
        .map(|v| v / 16.0)
        .collect();
    let m = Affine([1.004, -0.013, 7.35, 0.011, 0.997, -4.2]);

    println!("single primitives at {w}x{h}\n");
    bench("sep_filter 5-tap rgb f32", 5, || {
        std::hint::black_box(filter::sep_filter(&rgb, &k, &k, Border::Reflect));
    });
    bench("sep_filter 5-tap gray f32", 5, || {
        std::hint::black_box(filter::sep_filter(&gray, &k, &k, Border::Reflect));
    });
    bench("sqr_box_filter 3x3 gray", 5, || {
        std::hint::black_box(filter::sqr_box_filter(&gray, 3, 3, Border::Reflect));
    });
    bench("gaussian_blur 15 gray", 5, || {
        std::hint::black_box(filter::gaussian_blur(&gray, 15, 0.0));
    });
    bench("warp_affine cubic rgb u8", 5, || {
        std::hint::black_box(warp::warp_affine_u8(
            &rgb8,
            &m,
            w,
            h,
            Interp::Cubic,
            Border::Reflect,
        ));
    });
    bench("warp_affine linear gray f32", 5, || {
        std::hint::black_box(warp::warp_affine(
            &gray,
            &m,
            w,
            h,
            Interp::Linear,
            Border::Constant,
            true,
        ));
    });
    bench("rgb_to_lab_l u8", 5, || {
        std::hint::black_box(color::rgb_to_lab_l_u8(&rgb8));
    });
    bench("resize_area rgb u8 -> 1024", 5, || {
        std::hint::black_box(resize::resize_area_u8(&rgb8, 1024, 716));
    });
    bench("laplacian_pyramid rgb 6", 3, || {
        std::hint::black_box(pyramid::laplacian_pyramid(&rgb, 6));
    });

    println!("\ncost of the zero-filled allocations sep_filter makes");
    bench("Mat::new rgb (x2 per call)", 10, || {
        std::hint::black_box(Mat::new(h, w, 3));
        std::hint::black_box(Mat::new(h, w, 3));
    });
    bench("Mat::new gray (x2 per call)", 10, || {
        std::hint::black_box(Mat::new(h, w, 1));
        std::hint::black_box(Mat::new(h, w, 1));
    });
}
