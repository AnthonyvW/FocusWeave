use focusweave_core::affine::{constrain_warp, svd2, Affine, WarpConstraints};

fn main() {
    let mats: Vec<[f32; 6]> = vec![
        [1.004, -0.013, 7.35, 0.011, 0.997, -4.2],
        [0.998, 0.031, -3.1, -0.022, 1.011, 2.4],
        [1.0, 0.0, 5.0, 0.0, 1.0, -2.0],
        [0.87, 0.21, 1.0, -0.19, 1.13, -1.0],
        [1.02, 0.0, 0.0, 0.0, 0.98, 0.0],
    ];
    for m in &mats {
        let (u, sv, vt) = svd2([m[0] as f64, m[1] as f64, m[3] as f64, m[4] as f64]);
        println!(
            "svd {:?} {:?} {:?}",
            u.iter().map(|v| format!("{v:.9}")).collect::<Vec<_>>(),
            sv.iter().map(|v| format!("{v:.9}")).collect::<Vec<_>>(),
            vt.iter().map(|v| format!("{v:.9}")).collect::<Vec<_>>()
        );
        for flags in 0..16u8 {
            let c = WarpConstraints {
                no_rotation: flags & 1 != 0,
                no_scale: flags & 2 != 0,
                no_shear: flags & 4 != 0,
                no_translation: flags & 8 != 0,
            };
            let r = constrain_warp(&Affine(*m), c);
            println!(
                "con {flags} {}",
                r.0.iter()
                    .map(|v| format!("{v:.9}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
}
