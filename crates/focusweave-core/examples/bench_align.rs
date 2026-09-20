use focusweave_core::align::run_ecc;
use focusweave_core::config::{resolve_images, Images};
use focusweave_core::focus::{phase_correlation_translation, prepare_for_ecc, to_gray};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let folder = PathBuf::from(std::env::args().nth(1).expect("folder"));
    let (sources, size) = resolve_images(&Images::Folder(folder)).expect("resolve");

    let t = Instant::now();
    let grays: Vec<_> = sources
        .iter()
        .map(|s| to_gray(s, size).expect("gray"))
        .collect();
    println!(
        "decode + gray  {:6.3}s  ({} frames at {}x{})",
        t.elapsed().as_secs_f64(),
        grays.len(),
        size.0,
        size.1
    );

    let fine = 1024.min(grays[0].h.max(grays[0].w));
    let t = Instant::now();
    let prepared: Vec<_> = grays.iter().map(|g| prepare_for_ecc(g, fine)).collect();
    println!("prepare_for_ecc{:6.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    for i in 1..grays.len() {
        phase_correlation_translation(&grays[i - 1], &grays[i], 512);
    }
    println!("phase corr     {:6.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    for i in 1..grays.len() {
        run_ecc(&grays[i - 1], &grays[i], &prepared[i - 1], &prepared[i]);
    }
    println!(
        "run_ecc total  {:6.3}s  ({} pairs)",
        t.elapsed().as_secs_f64(),
        grays.len() - 1
    );
}
