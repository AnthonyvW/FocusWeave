//! Unit tests for the pieces that stand alone from image data.

use focusweave_core::affine::{constrain_warp, svd2, Affine, WarpConstraints};
use focusweave_core::fft::optimal_dft_size;
use focusweave_core::mat::Mat;
use focusweave_core::pyramid::{compute_levels, laplacian_pyramid, reconstruct};
use focusweave_core::stack::{compute_canvas, compute_slabs};

#[test]
fn svd_reconstructs_the_matrix() {
    for m in [
        [1.004, -0.013, 0.011, 0.997],
        [0.87, 0.21, -0.19, 1.13],
        [1.0, 0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0, 2.0],
    ] {
        let (u, sv, vt) = svd2(m);
        assert!(
            sv[0] >= sv[1] && sv[1] >= 0.0,
            "singular values must be sorted and non-negative"
        );
        let us = [u[0] * sv[0], u[1] * sv[1], u[2] * sv[0], u[3] * sv[1]];
        let back = [
            us[0] * vt[0] + us[1] * vt[2],
            us[0] * vt[1] + us[1] * vt[3],
            us[2] * vt[0] + us[3] * vt[2],
            us[2] * vt[1] + us[3] * vt[3],
        ];
        for (a, b) in back.iter().zip(m.iter()) {
            assert!((a - b).abs() < 1e-12, "{back:?} != {m:?}");
        }
    }
}

#[test]
fn constraints_suppress_their_components() {
    let warp = Affine([1.004, -0.013, 7.35, 0.011, 0.997, -4.2]);

    let no_translation = constrain_warp(
        &warp,
        WarpConstraints {
            no_translation: true,
            ..Default::default()
        },
    );
    assert_eq!((no_translation.tx(), no_translation.ty()), (0.0, 0.0));

    let no_scale = constrain_warp(
        &warp,
        WarpConstraints {
            no_scale: true,
            ..Default::default()
        },
    );
    let det = f64::from(no_scale.0[0]) * f64::from(no_scale.0[4])
        - f64::from(no_scale.0[1]) * f64::from(no_scale.0[3]);
    assert!(
        (det.abs() - 1.0).abs() < 1e-6,
        "scale removal should leave |det| == 1, got {det}"
    );

    // With everything suppressed the warp keeps only an axis-aligned scale of
    // unit determinant — the singular values normalised to geometric mean one.
    // It is not the identity unless the two singular values already agreed.
    let all = constrain_warp(
        &warp,
        WarpConstraints {
            no_rotation: true,
            no_scale: true,
            no_shear: true,
            no_translation: true,
        },
    );
    assert_eq!(
        (all.0[1], all.0[3]),
        (0.0, 0.0),
        "off-diagonal terms are gone"
    );
    assert_eq!((all.tx(), all.ty()), (0.0, 0.0));
    let det = f64::from(all.0[0]) * f64::from(all.0[4]);
    assert!(
        (det - 1.0).abs() < 1e-6,
        "determinant should be 1, got {det}"
    );
}

#[test]
fn affine_inverse_round_trips() {
    let m = Affine([1.004, -0.013, 7.35, 0.011, 0.997, -4.2]);
    let back = m.chain(&m.invert());
    for (a, b) in back.0.iter().zip(Affine::IDENTITY.0.iter()) {
        assert!((a - b).abs() < 1e-5, "{back:?}");
    }
}

#[test]
fn pyramid_round_trips() {
    let (h, w) = (37usize, 53usize);
    let data: Vec<f32> = (0..h * w * 3).map(|i| ((i * 37) % 251) as f32).collect();
    let image = Mat::from_vec(h, w, 3, data);
    let bands = laplacian_pyramid(&image, 3);
    assert_eq!(bands.len(), 4);
    let back = reconstruct(&bands);
    assert_eq!((back.h, back.w, back.c), (h, w, 3));
    for (a, b) in back.data.iter().zip(image.data.iter()) {
        assert!((a - b).abs() < 1e-2, "reconstruction drifted: {a} vs {b}");
    }
}

#[test]
fn level_count_shrinks_to_sixteen_pixels() {
    assert_eq!(compute_levels(1400, 2000, 6), 6);
    assert_eq!(compute_levels(64, 64, 6), 2);
    assert_eq!(compute_levels(16, 16, 6), 0);
}

#[test]
fn canvas_expands_to_cover_every_frame() {
    let warps = vec![Affine::IDENTITY, Affine::translation(10.0, -6.0)];
    let ((w, h), adjusted) = compute_canvas(&warps, (100, 80), false, false);
    assert_eq!((w, h), (110, 86));
    // The origin shift puts the top-left of the union at (0, 0).
    assert_eq!((adjusted[0].tx(), adjusted[0].ty()), (0.0, 6.0));
    assert_eq!((adjusted[1].tx(), adjusted[1].ty()), (10.0, 0.0));

    let (kept, unchanged) = compute_canvas(&warps, (100, 80), true, false);
    assert_eq!(kept, (100, 80));
    assert_eq!(unchanged, warps);
}

#[test]
fn slabs_cover_every_index() {
    assert_eq!(
        compute_slabs(10, 4, 2),
        vec![(0, 4), (2, 6), (4, 8), (6, 10)]
    );
    assert_eq!(compute_slabs(5, 10, 2), vec![(0, 5)]);
    // Zero overlap still steps forward rather than looping.
    assert_eq!(compute_slabs(7, 3, 0), vec![(0, 3), (3, 6), (6, 7)]);
}

#[test]
fn optimal_dft_sizes_are_five_smooth() {
    assert_eq!(optimal_dft_size(160), 160);
    assert_eq!(optimal_dft_size(224), 225);
    assert_eq!(optimal_dft_size(1000), 1000);
    for n in [7usize, 11, 13, 224, 1021] {
        let mut m = optimal_dft_size(n);
        assert!(m >= n);
        for f in [2usize, 3, 5] {
            while m % f == 0 {
                m /= f;
            }
        }
        assert_eq!(m, 1, "optimal_dft_size({n}) is not 5-smooth");
    }
}
