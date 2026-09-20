//! Command line interface, shared by the native binary and the Python
//! `focusweave` entry point so both accept exactly the same flags.

use crate::config::{run, FocusStackConfig, Images, RunResult};
use crate::hooks::{Error, Hooks, Stage};
use crate::image_source::{save_image, ImageBuf, IMAGE_EXTENSIONS};
use std::cell::RefCell;
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

const HELP: &str = r#"Focus stack a folder of images using Laplacian pyramid fusion.

Usage: focusweave FOLDER [OPTIONS]

Output options
  --output PATH           Output file path (default: stacked.jpg inside the input
                          folder). Format is inferred from the extension.
  --quality N             JPEG output quality 1-95 (default: 95).

Alignment options
  --no-align              Skip alignment (use when images are already registered).
  --reference N           Index of the image to align all others to (default: the
                          middle image).
  --global-align          Align every image directly to the reference instead of
                          chaining through neighbours.
  --full-res              Run the fine alignment pass at full resolution instead of
                          the default 1024px cap.
  --min-shift PIXELS      Minimum shift in pixels before alignment is applied
                          (default: 5.0).
  --no-rotation           Suppress rotation correction during alignment.
  --no-scale              Suppress scale correction during alignment.
  --no-shear              Suppress shear correction during alignment.
  --no-translation        Suppress translation correction during alignment.

Canvas options
  --keep-size             Keep the output the same size as the inputs.
  --crop                  Crop to the intersection of all image extents.
  --no-fill               Fill border regions with black instead of reflecting
                          edge pixels.

Stacking options
  --levels N              Laplacian pyramid levels (default: auto from image size).
  --sharpness EXPONENT    Weight sharpness exponent (default: 4.0). Useful range is
                          roughly 1.0 (soft blend) to 8.0 (near-hard selection).
  --workers N             Number of frames fused concurrently. The default picks
                          one per core, capped so their buffers fit in free
                          memory; pass a number to override it. Each worker costs
                          roughly 110 MB per megapixel of output.

Culling options
  --cull [THRESHOLD]      Remove wholly out-of-focus images before stacking. Frames
                          scoring below THRESHOLD are dropped; THRESHOLD defaults to
                          0.6. At least the two sharpest frames are always kept.

Slabbing options
  --slab SIZE OVERLAP     Split the set into overlapping sub-stacks of SIZE images
                          sharing OVERLAP images, stack each, then fuse the results.
  --output-steps          Save each slab to a focusweave_slabs/ folder.
  --only-slab             Stop after producing slabs; implies --output-steps.
  --recursive-slab        Re-slab the layer's results until they fit a single pass.
  --slab-format EXT       File format for slab images (default: tiff).

Other
  --timings               Print how long each stage took, and what this build is,
                          after the run. Useful when reporting a slow run.
  --version               Show the version number and exit.
  --formats               List the supported image extensions and exit.
  --help                  Show this message and exit.
"#;

#[derive(Default)]
struct Args {
    folder: Option<PathBuf>,
    output: Option<PathBuf>,
    quality: Option<u8>,
    slab_format: Option<String>,
    output_steps: bool,
    only_slab: bool,
    timings: bool,
}

struct Parsed {
    cfg: FocusStackConfig,
    args: Args,
}

fn parse_number<T: std::str::FromStr>(name: &str, value: &str) -> Result<T, String> {
    value
        .parse::<T>()
        .map_err(|_| format!("--{name} expects a number, got '{value}'"))
}

fn parse(argv: &[String]) -> Result<Option<Parsed>, String> {
    if argv.is_empty() {
        print!("{HELP}");
        return Ok(None);
    }

    let mut args = Args::default();
    let mut cfg = FocusStackConfig::new(Images::Folder(PathBuf::new()));
    let mut i = 0usize;

    let next = |i: &mut usize, name: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("--{name} expects a value"))
    };

    while i < argv.len() {
        let arg = argv[i].clone();
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{HELP}");
                return Ok(None);
            }
            "--version" => {
                println!("focusweave {VERSION}");
                return Ok(None);
            }
            "--formats" => {
                println!("Supported image extensions:");
                println!("  {}", IMAGE_EXTENSIONS.join("  "));
                return Ok(None);
            }
            "--output" => args.output = Some(PathBuf::from(next(&mut i, "output")?)),
            "--quality" => args.quality = Some(parse_number("quality", &next(&mut i, "quality")?)?),
            "--slab-format" => args.slab_format = Some(next(&mut i, "slab-format")?),
            "--timings" => args.timings = true,
            "--output-steps" => args.output_steps = true,
            "--only-slab" => {
                args.only_slab = true;
                cfg.only_slab = true;
            }
            "--no-align" => cfg.no_align = true,
            "--keep-size" => cfg.keep_size = true,
            "--crop" => cfg.crop = true,
            "--no-fill" => cfg.no_fill = true,
            "--global-align" => cfg.global_align = true,
            "--no-rotation" => cfg.no_rotation = true,
            "--no-scale" => cfg.no_scale = true,
            "--no-shear" => cfg.no_shear = true,
            "--no-translation" => cfg.no_translation = true,
            "--full-res" => cfg.full_res = true,
            "--recursive-slab" => cfg.recursive_slab = true,
            "--reference" => {
                cfg.reference = parse_number("reference", &next(&mut i, "reference")?)?
            }
            "--min-shift" => {
                cfg.min_shift = parse_number("min-shift", &next(&mut i, "min-shift")?)?
            }
            "--levels" => cfg.levels = parse_number("levels", &next(&mut i, "levels")?)?,
            "--sharpness" => {
                cfg.sharpness = parse_number("sharpness", &next(&mut i, "sharpness")?)?
            }
            "--workers" => cfg.workers = parse_number("workers", &next(&mut i, "workers")?)?,
            "--cull" => {
                // The threshold is optional, exactly as in the Python CLI.
                let takes_value = argv
                    .get(i + 1)
                    .map(|v| !v.starts_with("--") && v.parse::<f64>().is_ok())
                    .unwrap_or(false);
                cfg.cull = Some(if takes_value {
                    i += 1;
                    parse_number("cull", &argv[i])?
                } else {
                    0.6
                });
            }
            "--slab" => {
                let size: usize = parse_number("slab", &next(&mut i, "slab")?)?;
                let overlap: usize = parse_number("slab", &next(&mut i, "slab")?)?;
                cfg.slab = Some((size, overlap));
            }
            other if other.starts_with('-') => {
                return Err(format!("unrecognised option '{other}'"))
            }
            other => {
                if args.folder.is_some() {
                    return Err(format!("unexpected argument '{other}'"));
                }
                args.folder = Some(PathBuf::from(other));
            }
        }
        i += 1;
    }

    let folder = args
        .folder
        .clone()
        .ok_or_else(|| "the following arguments are required: folder".to_string())?;
    if !folder.is_dir() {
        return Err(format!("'{}' is not a directory.", folder.display()));
    }
    cfg.images = Images::Folder(folder);
    Ok(Some(Parsed { cfg, args }))
}

/// Accumulates wall time per pipeline stage from the progress callback.
#[derive(Default)]
struct StageTimer {
    current: Option<(Stage, Instant)>,
    totals: Vec<(Stage, Duration)>,
}

impl StageTimer {
    fn observe(&mut self, stage: Stage) {
        match self.current {
            Some((previous, _)) if previous == stage => {}
            _ => {
                if let Some((previous, started)) = self.current.take() {
                    self.record(previous, started.elapsed());
                }
                self.current = Some((stage, Instant::now()));
            }
        }
    }

    fn finish(&mut self) {
        if let Some((stage, started)) = self.current.take() {
            self.record(stage, started.elapsed());
        }
    }

    fn record(&mut self, stage: Stage, elapsed: Duration) {
        match self.totals.iter_mut().find(|(s, _)| *s == stage) {
            Some((_, total)) => *total += elapsed,
            None => self.totals.push((stage, elapsed)),
        }
    }

    fn report(&self, total: Duration) {
        println!("\nTimings");
        println!("  build          {} kernels", crate::BACKEND);
        println!("  threads        {} available", available_threads());
        for (stage, elapsed) in &self.totals {
            let share = elapsed.as_secs_f64() / total.as_secs_f64() * 100.0;
            println!(
                "  {:<14} {:6.2}s  {share:4.1}%",
                stage.as_str(),
                elapsed.as_secs_f64()
            );
        }
        println!("  {:<14} {:6.2}s", "total", total.as_secs_f64());
    }
}

fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(0)
}

/// Run the command line interface. Returns the process exit code.
pub fn run_cli(argv: &[String]) -> i32 {
    let parsed = match parse(argv) {
        Ok(Some(p)) => p,
        Ok(None) => return 0,
        Err(message) => {
            eprintln!("Error: {message}");
            return 1;
        }
    };

    match execute(parsed) {
        Ok(()) => 0,
        Err(Error::Interrupted) => {
            println!("\nInterrupted.");
            0
        }
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

fn execute(parsed: Parsed) -> Result<(), Error> {
    let Parsed { cfg, args } = parsed;
    let folder = match &cfg.images {
        Images::Folder(f) => f.clone(),
        _ => unreachable!("the CLI only builds folder sources"),
    };
    let out_path = args
        .output
        .clone()
        .unwrap_or_else(|| folder.join("stacked.jpg"));
    let quality = args.quality.unwrap_or(95);
    let emit_steps = args.output_steps || args.only_slab;
    let steps_dir = out_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("focusweave_slabs");
    let slab_ext = args.slab_format.clone().unwrap_or_else(|| "tiff".into());
    let slab_ext = slab_ext.trim_start_matches('.').to_string();

    let timer = RefCell::new(StageTimer::default());
    let progress = |fraction: f64, stage: Stage, message: &str| {
        if args.timings {
            timer.borrow_mut().observe(stage);
        }
        if !message.is_empty() {
            println!("  {:5.1}%  {message}", fraction * 100.0);
        }
    };
    let on_slab = |label: &str, image: &ImageBuf| {
        if std::fs::create_dir_all(&steps_dir).is_err() {
            eprintln!("Warning: could not create {}", steps_dir.display());
            return;
        }
        let file = steps_dir.join(format!("{label}.{slab_ext}"));
        let t = Instant::now();
        match save_image(image, &file, quality) {
            Ok(()) => println!(
                "    Saved: {} ({:.2}s)",
                file.display(),
                t.elapsed().as_secs_f64()
            ),
            Err(e) => eprintln!("Warning: {e}"),
        }
    };

    let hooks = Hooks {
        progress: Some(&progress),
        interrupt: None,
        on_slab: if emit_steps { Some(&on_slab) } else { None },
    };

    let start = Instant::now();
    let result: RunResult = run(&cfg, &hooks)?;
    if args.timings {
        timer.borrow_mut().finish();
    }

    if let Some(slabs) = result.slabs {
        if emit_steps {
            println!("Slabs saved to: {}", steps_dir.display());
        }
        println!("Produced {} slab(s)", slabs.len());
        println!("Done ({:.2}s total)", start.elapsed().as_secs_f64());
        if args.timings {
            timer.borrow().report(start.elapsed());
        }
        return Ok(());
    }

    let image = result
        .image
        .expect("a non-slab run always produces an image");
    let t_save = Instant::now();
    save_image(&image, &out_path, quality)?;
    println!(
        "Saved: {} ({:.2}s)",
        out_path.display(),
        t_save.elapsed().as_secs_f64()
    );
    println!("Done ({:.2}s total)", start.elapsed().as_secs_f64());
    if args.timings {
        timer.borrow().report(start.elapsed());
    }
    Ok(())
}
