use focusweave_core::affine::Affine;
use focusweave_core::border::Border;
use focusweave_core::testio;
use focusweave_core::warp::{warp_affine, Interp};
use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap());
    let a = testio::read_f32(dir.join("reg_a.bin"));
    let m = Affine([1.0007, 0.0013, 5.37, -0.0009, 0.9994, -3.21]);
    testio::write_f32(
        dir.join("out_wf32_exact.bin"),
        &warp_affine(&a, &m, a.w, a.h, Interp::Linear, Border::Constant, true),
    );
    testio::write_f32(
        dir.join("out_wf32_quant.bin"),
        &warp_affine(
            &a,
            &m,
            a.w,
            a.h,
            Interp::LinearQuantised,
            Border::Constant,
            true,
        ),
    );
}
