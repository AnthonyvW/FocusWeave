use focusweave_core::affine::WarpConstraints;
use focusweave_core::align::{align_images, AlignOptions, AlignStrategy};
use focusweave_core::config::{resolve_images, Images};
use focusweave_core::hooks::Hooks;
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let folder = PathBuf::from(args.next().expect("folder"));
    let reference: usize = args.next().unwrap_or_else(|| "0".into()).parse().unwrap();
    let (sources, size) = resolve_images(&Images::Folder(folder)).expect("resolve");
    let warps = align_images(
        &sources,
        size,
        reference,
        AlignOptions {
            strategy: AlignStrategy::NeighbourChained,
            constraints: WarpConstraints::default(),
            full_res: false,
            min_shift: 5.0,
            workers: 0,
        },
        &Hooks::default(),
    )
    .expect("align");
    for (i, w) in warps.iter().enumerate() {
        println!(
            "{i} {}",
            w.0.iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}
