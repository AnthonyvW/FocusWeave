//! Cross-check phase correlation and the ECC solver against OpenCV.

use focusweave_core::affine::Affine;
use focusweave_core::{ecc, fft, mat::MatU8, testio};
use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).expect("output directory"));
    let a = testio::read_f32(dir.join("reg_a.bin"));
    let b = testio::read_f32(dir.join("reg_b.bin"));
    let mask = testio::read_u8(dir.join("reg_mask.bin"));

    let (tx, ty) = fft::phase_correlate(&a, &b);
    let mut summary = format!("phasecorr {tx:.9} {ty:.9}\n");

    let mask_ref: Option<&MatU8> = Some(&mask);
    for (label, iters, eps, gauss, masked) in [
        ("ecc_plain", 50usize, 0.001f64, 5usize, false),
        ("ecc_masked", 50, 0.001, 5, true),
        ("ecc_rough", 25, 0.01, 1, true),
    ] {
        let seed = Affine([1.0, 0.0, 6.0, 0.0, 1.0, -4.0]);
        let result = ecc::find_transform_ecc_verbose(
            &a,
            &b,
            seed,
            iters,
            eps,
            if masked { mask_ref } else { None },
            gauss,
        );
        match result {
            Ok((m, rho, iters_run)) => {
                summary.push_str(label);
                for v in m.0 {
                    summary.push_str(&format!(" {v:.9}"));
                }
                summary.push_str(&format!(" rho={rho:.9} iters={iters_run}"));
                summary.push('\n');
            }
            Err(e) => summary.push_str(&format!("{label} ERROR {e}\n")),
        }
    }
    std::fs::write(dir.join("out_registration.txt"), summary).expect("write summary");
    println!("wrote registration summary");
}
