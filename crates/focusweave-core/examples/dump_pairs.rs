use focusweave_core::align::run_ecc;
use focusweave_core::config::{resolve_images, Images};
use focusweave_core::focus::{prepare_for_ecc, to_gray};
use std::path::PathBuf;

fn main() {
    let folder = PathBuf::from(std::env::args().nth(1).expect("folder"));
    let (sources, size) = resolve_images(&Images::Folder(folder)).expect("resolve");
    let grays: Vec<_> = sources
        .iter()
        .map(|s| to_gray(s, size).expect("gray"))
        .collect();
    let fine = 1024.min(grays[0].h.max(grays[0].w));
    let prepared: Vec<_> = grays.iter().map(|g| prepare_for_ecc(g, fine)).collect();
    for i in 1..grays.len() {
        let (w, ok) = run_ecc(&grays[i - 1], &grays[i], &prepared[i - 1], &prepared[i]);
        println!(
            "{i} {} {}",
            u8::from(ok),
            w.0.iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}
