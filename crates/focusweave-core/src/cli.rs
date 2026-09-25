//! Command line interface, shared by the native binary and the Python
//! `focusweave` entry point so both accept exactly the same flags.

use crate::config::{run, FocusStackConfig, Images, RunResult};
use crate::hooks::{Error, Hooks, Stage};
use crate::image_source::{is_image_path, list_folder, save_image, ImageBuf, IMAGE_EXTENSIONS};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Where slab images go, relative to the output. Also the one subfolder a batch
/// never treats as an image set, since a previous run may have created it.
const SLAB_DIR: &str = "focusweave_slabs";

/// The `--batch-format` value that takes each set's format from its images.
const INHERIT: &str = "inherit";

const HELP: &str = r#"Focus stack a folder of images using Laplacian pyramid fusion.

Usage: focusweave FOLDER [OPTIONS]
       focusweave --batch FOLDER [OPTIONS]

Output options
  --output PATH           Output file path (default: stacked.jpg inside the input
                          folder). Format is inferred from the extension. With
                          --batch, a folder to write every result into instead.
  --quality N             JPEG output quality 1-95 (default: 95).

Batch options
  --batch FOLDER          Stack each subfolder of FOLDER as a separate set of
                          images. Each result is named after its subfolder and
                          saved in FOLDER itself unless --output names another
                          folder. Every other option applies to each set.
  --batch-format EXT      Format for batch results: inherit, or an extension such
                          as tiff, png or jpg (default: inherit). inherit uses the
                          most common extension among each set's images, so a
                          set of 16-bit TIFFs keeps its depth.

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
  --timings               Print how long each stage took after the run. Useful
                          when reporting a slow run.
  --version, -V           Show the version number and exit.
  --opencv-version        Show the version of the OpenCV library in use and exit.
  --formats               List the supported image extensions and exit.
  --help                  Show this message and exit.
"#;

#[derive(Default)]
struct Args {
    folder: Option<PathBuf>,
    batch: Option<PathBuf>,
    batch_format: Option<String>,
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
            "--version" | "-V" => {
                println!("focusweave {VERSION}");
                return Ok(None);
            }
            "--opencv-version" => {
                println!("OpenCV {}", crate::cv::opencv_version());
                return Ok(None);
            }
            "--formats" => {
                println!("Supported image extensions:");
                println!("  {}", IMAGE_EXTENSIONS.join("  "));
                return Ok(None);
            }
            "--batch" => args.batch = Some(PathBuf::from(next(&mut i, "batch")?)),
            "--batch-format" => args.batch_format = Some(next(&mut i, "batch-format")?),
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

    if let Some(raw) = args.batch_format.take() {
        if args.batch.is_none() {
            return Err("--batch-format only applies with --batch".into());
        }
        args.batch_format = Some(normalise_batch_format(&raw)?);
    }

    match (&args.folder, &args.batch) {
        (Some(_), Some(_)) => {
            return Err("give either a folder or --batch FOLDER, not both".into());
        }
        (None, None) => {
            return Err("the following arguments are required: folder (or --batch FOLDER)".into());
        }
        (Some(folder), None) => {
            if !folder.is_dir() {
                return Err(format!("'{}' is not a directory.", folder.display()));
            }
            cfg.images = Images::Folder(folder.clone());
        }
        (None, Some(batch)) => {
            if !batch.is_dir() {
                return Err(format!("'{}' is not a directory.", batch.display()));
            }
            // A name like results.tiff reads as a file, and would otherwise
            // quietly become a folder holding every result.
            if let Some(output) = args.output.as_deref().filter(|p| is_image_path(p)) {
                return Err(format!(
                    "with --batch, --output is the folder to write results into, not a file: '{}'",
                    output.display()
                ));
            }
        }
    }
    Ok(Some(Parsed { cfg, args }))
}

fn normalise_batch_format(raw: &str) -> Result<String, String> {
    let format = raw.trim_start_matches('.').to_ascii_lowercase();
    let supported: Vec<&str> = IMAGE_EXTENSIONS
        .iter()
        .map(|e| e.trim_start_matches('.'))
        .collect();
    if format == INHERIT || supported.contains(&format.as_str()) {
        return Ok(format);
    }
    Err(format!(
        "--batch-format expects {INHERIT} or one of {}, got '{raw}'",
        supported.join(", ")
    ))
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
    let timer = RefCell::new(StageTimer::default());
    let start = Instant::now();

    let failed = match &args.batch {
        Some(batch) => execute_batch(&cfg, &args, batch, &timer)?,
        None => {
            let folder = match &cfg.images {
                Images::Folder(f) => f.clone(),
                _ => unreachable!("the CLI only builds folder sources"),
            };
            let out_path = args
                .output
                .clone()
                .unwrap_or_else(|| folder.join("stacked.jpg"));
            let steps_dir = out_path.parent().unwrap_or(Path::new(".")).join(SLAB_DIR);
            stack_one(&cfg, &args, &out_path, &steps_dir, &timer)?;
            0
        }
    };

    println!("Done ({:.2}s total)", start.elapsed().as_secs_f64());
    if args.timings {
        timer.borrow().report(start.elapsed());
    }
    if failed > 0 {
        return Err(Error::Config(format!("{failed} set(s) failed")));
    }
    Ok(())
}

/// Stack each image-bearing subfolder of `batch` as its own set. A set that
/// fails is reported and skipped rather than abandoning the rest of the
/// batch; the return value is how many failed.
fn execute_batch(
    cfg: &FocusStackConfig,
    args: &Args,
    batch: &Path,
    timer: &RefCell<StageTimer>,
) -> Result<usize, Error> {
    let out_dir = args.output.clone().unwrap_or_else(|| batch.to_path_buf());
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| Error::Config(format!("could not create '{}': {e}", out_dir.display())))?;

    let sets = discover_sets(batch, &out_dir)?;
    if sets.is_empty() {
        return Err(Error::Config(format!(
            "no subfolder of '{}' contains images",
            batch.display()
        )));
    }

    let mut failed = Vec::new();
    for (index, (folder, images)) in sets.iter().enumerate() {
        let name = folder
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        println!("\n[{}/{}] {name}", index + 1, sets.len());

        let mut set_cfg = cfg.clone();
        set_cfg.images = Images::Folder(folder.clone());
        let extension = match args.batch_format.as_deref() {
            None | Some(INHERIT) => inherited_extension(images),
            Some(fixed) => fixed.to_string(),
        };
        let out_path = out_dir.join(format!("{name}.{extension}"));
        let steps_dir = out_dir.join(SLAB_DIR).join(&name);
        match stack_one(&set_cfg, args, &out_path, &steps_dir, timer) {
            Ok(()) => {}
            Err(Error::Interrupted) => return Err(Error::Interrupted),
            Err(e) => {
                eprintln!("Error in {name}: {e}");
                failed.push(name);
            }
        }
    }

    println!(
        "\nStacked {} of {} sets into {}",
        sets.len() - failed.len(),
        sets.len(),
        out_dir.display()
    );
    if !failed.is_empty() {
        println!("Failed: {}", failed.join(", "));
    }
    Ok(failed.len())
}

/// Subfolders of `batch` holding at least one image, with those images, sorted
/// by folder name.
///
/// The output folder and the slab folder are left out even when they sit
/// inside `batch`: both fill with images, and without this a second run of the
/// same batch would stack its own previous results as another set.
fn discover_sets(batch: &Path, out_dir: &Path) -> Result<Vec<(PathBuf, Vec<PathBuf>)>, Error> {
    let entries = std::fs::read_dir(batch)
        .map_err(|e| Error::Config(format!("could not read '{}': {e}", batch.display())))?;
    let out_dir = out_dir.canonicalize().ok();

    let mut sets = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == SLAB_DIR {
            continue;
        }
        if out_dir.is_some() && path.canonicalize().ok() == out_dir {
            continue;
        }
        match list_folder(&path) {
            Ok(images) if images.is_empty() => println!("Skipping {name}: no images"),
            Ok(images) => sets.push((path, images)),
            Err(e) => println!("Skipping {name}: {e}"),
        }
    }
    sets.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(sets)
}

/// The extension a set is saved with under `--batch-format inherit`: the most
/// common one among its images. A tie goes to whichever appears first by file
/// name, so a mixed folder still gets the same answer every run.
fn inherited_extension(images: &[PathBuf]) -> String {
    let mut counts: Vec<(String, usize)> = Vec::new();
    let extensions = images
        .iter()
        .filter_map(|p| p.extension())
        .map(|e| e.to_string_lossy().to_ascii_lowercase());
    for extension in extensions {
        match counts.iter_mut().find(|(e, _)| *e == extension) {
            Some((_, n)) => *n += 1,
            None => counts.push((extension, 1)),
        }
    }
    // max_by_key keeps the last of equal maxima; reversing makes that the
    // first seen.
    counts
        .into_iter()
        .rev()
        .max_by_key(|(_, n)| *n)
        .map(|(e, _)| e)
        .unwrap_or_else(|| "jpg".into())
}

/// Stack one folder and write the result, or its slabs, to disk.
fn stack_one(
    cfg: &FocusStackConfig,
    args: &Args,
    out_path: &Path,
    steps_dir: &Path,
    timer: &RefCell<StageTimer>,
) -> Result<(), Error> {
    let quality = args.quality.unwrap_or(95);
    let emit_steps = args.output_steps || args.only_slab;
    let slab_ext = args.slab_format.clone().unwrap_or_else(|| "tiff".into());
    let slab_ext = slab_ext.trim_start_matches('.').to_string();

    let progress = |fraction: f64, stage: Stage, message: &str| {
        if args.timings {
            timer.borrow_mut().observe(stage);
        }
        if !message.is_empty() {
            println!("  {:5.1}%  {message}", fraction * 100.0);
        }
    };
    let on_slab = |label: &str, image: &ImageBuf| {
        if std::fs::create_dir_all(steps_dir).is_err() {
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

    let result: Result<RunResult, Error> = run(cfg, &hooks);
    if args.timings {
        // Closed here rather than left open, so the time spent saving, and in
        // a batch the gap before the next set, is not billed to the last stage.
        timer.borrow_mut().finish();
    }
    let result = result?;

    if let Some(slabs) = result.slabs {
        if emit_steps {
            println!("Slabs saved to: {}", steps_dir.display());
        }
        println!("Produced {} slab(s)", slabs.len());
        return Ok(());
    }

    let image = result
        .image
        .expect("a non-slab run always produces an image");
    let t_save = Instant::now();
    save_image(&image, out_path, quality)?;
    println!(
        "Saved: {} ({:.2}s)",
        out_path.display(),
        t_save.elapsed().as_secs_f64()
    );
    Ok(())
}
