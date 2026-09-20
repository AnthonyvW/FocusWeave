//! Python bindings for FocusWeave.
//!
//! The module mirrors the surface the original pure-Python implementation
//! exposed, so existing callers keep working unchanged. Images cross the
//! boundary as numpy `uint8`/`uint16` RGB arrays and warps as 2x3 `float32`
//! arrays, exactly as before.

use focusweave_core::affine::{Affine, WarpConstraints};
use focusweave_core::align::{align_images, AlignOptions, AlignStrategy};
use focusweave_core::config::{run as core_run, FocusStackConfig, Images};
use focusweave_core::focus::cull_unfocused;
use focusweave_core::hooks::{Error as CoreError, Hooks, Stage};
use focusweave_core::image_source::{
    image_size, list_folder, save_image, ImageBuf, Source, IMAGE_EXTENSIONS,
};
use focusweave_core::mat::{Img, Mat, MatU8};
use focusweave_core::pyramid;
use focusweave_core::stack::{
    compute_canvas, slab_images, stack_images, SlabOutcome, StackOptions,
};
use focusweave_core::streaming::{StreamingConfig, StreamingFocusStacker as CoreStreamer};
use numpy::ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use std::path::PathBuf;
use std::sync::Mutex;

/// The pipeline allocates and frees multi-megabyte scratch buffers on every
/// pyramid level. glibc services those with `mmap` and returns them to the
/// kernel immediately, so each one is re-faulted page by page on first write.
/// An allocator that caches large blocks avoids paying that repeatedly.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// A stacked image and, when only slabs were requested, the slab list.
type StackOutput<'py> = (Option<Bound<'py, PyAny>>, Option<Bound<'py, PyAny>>);
/// A canvas size and the warps adjusted to land on it.
type CanvasOutput<'py> = ((usize, usize), Vec<Bound<'py, PyArray2<f32>>>);

create_exception!(
    _core,
    Interrupted,
    pyo3::exceptions::PyException,
    "Raised when an interrupt callback signals that the run should stop."
);

fn to_py_err(e: CoreError) -> PyErr {
    match e {
        CoreError::Interrupted => Interrupted::new_err("Interrupted"),
        CoreError::Config(m) => PyValueError::new_err(m),
        CoreError::Load(m) => PyValueError::new_err(m.0),
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

fn path_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Option<PathBuf>> {
    if obj.is_instance_of::<PyString>() {
        return Ok(Some(PathBuf::from(obj.extract::<String>()?)));
    }
    if obj.hasattr("__fspath__")? {
        let s: String = obj.call_method0("__fspath__")?.extract()?;
        return Ok(Some(PathBuf::from(s)));
    }
    Ok(None)
}

fn image_buf_from_any(obj: &Bound<'_, PyAny>) -> PyResult<ImageBuf> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray3<u8>>() {
        let view = arr.as_array();
        let (h, w, c) = (view.shape()[0], view.shape()[1], view.shape()[2]);
        if c != 3 {
            return Err(PyValueError::new_err(format!(
                "expected an RGB array with 3 channels, got {c}"
            )));
        }
        return Ok(ImageBuf::U8(Img::from_vec(
            h,
            w,
            3,
            view.iter().copied().collect(),
        )));
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray3<u16>>() {
        let view = arr.as_array();
        let (h, w, c) = (view.shape()[0], view.shape()[1], view.shape()[2]);
        if c != 3 {
            return Err(PyValueError::new_err(format!(
                "expected an RGB array with 3 channels, got {c}"
            )));
        }
        return Ok(ImageBuf::U16(Img::from_vec(
            h,
            w,
            3,
            view.iter().copied().collect(),
        )));
    }
    Err(PyTypeError::new_err(
        "expected a uint8 or uint16 RGB ndarray with shape (H, W, 3)",
    ))
}

fn source_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Source> {
    if let Some(path) = path_from_any(obj)? {
        return Ok(Source::Path(path));
    }
    Ok(Source::Array(image_buf_from_any(obj)?))
}

fn sources_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Source>> {
    if let Some(path) = path_from_any(obj)? {
        let paths = list_folder(&path).map_err(|e| PyValueError::new_err(e.0))?;
        if paths.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "Need at least 2 images in '{}', found {}.",
                path.display(),
                paths.len()
            )));
        }
        return Ok(paths.into_iter().map(Source::Path).collect());
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(source_from_any(&item?)?);
    }
    if out.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "Need at least 2 images, got {}.",
            out.len()
        )));
    }
    Ok(out)
}

fn image_to_py<'py>(py: Python<'py>, image: &ImageBuf) -> PyResult<Bound<'py, PyAny>> {
    match image {
        ImageBuf::U8(m) => {
            let arr = Array3::from_shape_vec((m.h, m.w, m.c), m.data.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyArray3::from_owned_array(py, arr).into_any())
        }
        ImageBuf::U16(m) => {
            let arr = Array3::from_shape_vec((m.h, m.w, m.c), m.data.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyArray3::from_owned_array(py, arr).into_any())
        }
    }
}

fn mat_u8_to_py<'py>(py: Python<'py>, m: &MatU8) -> PyResult<Bound<'py, PyAny>> {
    let arr = Array3::from_shape_vec((m.h, m.w, m.c), m.data.clone())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray3::from_owned_array(py, arr).into_any())
}

fn affine_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Affine> {
    let arr: PyReadonlyArray2<f32> = obj.extract()?;
    let view = arr.as_array();
    if view.shape() != [2, 3] {
        return Err(PyValueError::new_err("a warp must be a 2x3 float32 array"));
    }
    let v: Vec<f32> = view.iter().copied().collect();
    Ok(Affine([v[0], v[1], v[2], v[3], v[4], v[5]]))
}

fn affines_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Affine>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(affine_from_any(&item?)?);
    }
    Ok(out)
}

fn affine_to_py<'py>(py: Python<'py>, m: &Affine) -> Bound<'py, PyArray2<f32>> {
    let arr = Array2::from_shape_vec((2, 3), m.0.to_vec()).expect("2x3");
    PyArray2::from_owned_array(py, arr)
}

fn mat_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Mat> {
    let arr: PyReadonlyArray2<f32> = obj
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a 2-D float32 ndarray"))?;
    let view = arr.as_array();
    Ok(Mat::from_vec(
        view.shape()[0],
        view.shape()[1],
        1,
        view.iter().copied().collect(),
    ))
}

fn mat_to_py<'py>(py: Python<'py>, m: &Mat) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let arr = Array2::from_shape_vec((m.h, m.w), m.data.clone())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

// ---------------------------------------------------------------------------
// Callback plumbing
// ---------------------------------------------------------------------------

/// Bridges Python callables into the core's hook closures.
///
/// The core releases the GIL for the duration of a run, so every callback
/// reacquires it. An exception raised inside a callback is stored and used to
/// stop the run at the next checkpoint, then re-raised to the caller.
struct Callbacks {
    progress: Option<Py<PyAny>>,
    interrupt: Option<Py<PyAny>>,
    on_slab: Option<Py<PyAny>>,
    error: Mutex<Option<PyErr>>,
}

impl Callbacks {
    fn new(
        progress: Option<Py<PyAny>>,
        interrupt: Option<Py<PyAny>>,
        on_slab: Option<Py<PyAny>>,
    ) -> Self {
        Callbacks {
            progress,
            interrupt,
            on_slab,
            error: Mutex::new(None),
        }
    }

    fn record(&self, err: PyErr) {
        let mut slot = self.error.lock().expect("callback error slot");
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    fn failed(&self) -> bool {
        self.error.lock().expect("callback error slot").is_some()
    }

    fn take_error(&self) -> Option<PyErr> {
        self.error.lock().expect("callback error slot").take()
    }

    fn report(&self, fraction: f64, stage: Stage, message: &str) {
        let Some(cb) = &self.progress else { return };
        if self.failed() {
            return;
        }
        Python::attach(|py| {
            if let Err(e) = cb.call1(py, (fraction, stage.as_str(), message)) {
                self.record(e);
            }
        });
    }

    fn should_stop(&self) -> bool {
        if self.failed() {
            return true;
        }
        let Some(cb) = &self.interrupt else {
            return false;
        };
        Python::attach(|py| match cb.call0(py) {
            Ok(v) => v.is_truthy(py).unwrap_or(false),
            Err(e) => {
                self.record(e);
                true
            }
        })
    }

    fn slab(&self, label: &str, image: &ImageBuf) {
        let Some(cb) = &self.on_slab else { return };
        if self.failed() {
            return;
        }
        Python::attach(|py| {
            match image_to_py(py, image).and_then(|arr| cb.call1(py, (label, arr)).map(|_| ())) {
                Ok(()) => {}
                Err(e) => self.record(e),
            }
        });
    }
}

/// Run `body` with the GIL released, wiring `callbacks` into the core hooks.
fn with_hooks<T>(
    py: Python<'_>,
    callbacks: &Callbacks,
    body: impl FnOnce(&Hooks) -> Result<T, CoreError> + Send,
) -> PyResult<T>
where
    T: Send,
{
    let progress = |f: f64, s: Stage, m: &str| callbacks.report(f, s, m);
    let interrupt = || callbacks.should_stop();
    let on_slab = |label: &str, image: &ImageBuf| callbacks.slab(label, image);

    let result = py.detach(|| {
        let hooks = Hooks {
            progress: callbacks
                .progress
                .as_ref()
                .map(|_| &progress as &dyn Fn(f64, Stage, &str)),
            interrupt: if callbacks.interrupt.is_some() {
                Some(&interrupt as &dyn Fn() -> bool)
            } else {
                None
            },
            on_slab: callbacks
                .on_slab
                .as_ref()
                .map(|_| &on_slab as &dyn Fn(&str, &ImageBuf)),
        };
        body(&hooks)
    });

    if let Some(err) = callbacks.take_error() {
        return Err(err);
    }
    result.map_err(to_py_err)
}

fn constraints(
    no_rotation: bool,
    no_scale: bool,
    no_shear: bool,
    no_translation: bool,
) -> WarpConstraints {
    WarpConstraints {
        no_rotation,
        no_scale,
        no_shear,
        no_translation,
    }
}

// ---------------------------------------------------------------------------
// Module functions
// ---------------------------------------------------------------------------

#[pyfunction]
fn list_image_files(folder: &str) -> PyResult<Vec<String>> {
    let paths =
        list_folder(std::path::Path::new(folder)).map_err(|e| PyValueError::new_err(e.0))?;
    Ok(paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect())
}

#[pyfunction]
fn probe_size(path: &str) -> PyResult<(usize, usize)> {
    image_size(std::path::Path::new(path)).map_err(|e| PyValueError::new_err(e.0))
}

/// Read an image file as an RGB array at its native bit depth.
#[pyfunction]
fn read_image<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    let buf = focusweave_core::image_source::load_file(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(e.0))?;
    image_to_py(py, &buf)
}

#[pyfunction]
#[pyo3(signature = (image, path, quality=95))]
fn write_image(image: &Bound<'_, PyAny>, path: &str, quality: u8) -> PyResult<()> {
    let buf = image_buf_from_any(image)?;
    save_image(&buf, std::path::Path::new(path), quality).map_err(|e| PyValueError::new_err(e.0))
}

#[pyfunction]
#[pyo3(signature = (shape, max_levels=6))]
fn compute_levels(shape: (usize, usize), max_levels: usize) -> usize {
    pyramid::compute_levels(shape.0, shape.1, max_levels)
}

#[pyfunction]
fn reduce<'py>(py: Python<'py>, image: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    mat_to_py(py, &pyramid::reduce(&mat_from_any(image)?))
}

#[pyfunction]
fn expand<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    target_shape: (usize, usize),
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    mat_to_py(py, &pyramid::expand(&mat_from_any(image)?, target_shape))
}

#[pyfunction]
#[pyo3(signature = (level, window=3))]
fn region_energy<'py>(
    py: Python<'py>,
    level: &Bound<'py, PyAny>,
    window: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    mat_to_py(py, &pyramid::region_energy(&mat_from_any(level)?, window))
}

#[pyfunction]
#[pyo3(signature = (image, window=3))]
fn region_deviation<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    window: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    mat_to_py(
        py,
        &pyramid::region_deviation(&mat_from_any(image)?, window),
    )
}

#[pyfunction]
#[pyo3(signature = (image, window=8))]
fn region_entropy<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    window: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    mat_to_py(py, &pyramid::region_entropy(&mat_from_any(image)?, window))
}

#[pyfunction]
#[pyo3(signature = (warps, src_size, keep_size=false, crop=false))]
fn compute_canvas_py<'py>(
    py: Python<'py>,
    warps: &Bound<'py, PyAny>,
    src_size: (usize, usize),
    keep_size: bool,
    crop: bool,
) -> PyResult<CanvasOutput<'py>> {
    let warps = affines_from_any(warps)?;
    let (size, adjusted) = compute_canvas(&warps, src_size, keep_size, crop);
    Ok((size, adjusted.iter().map(|m| affine_to_py(py, m)).collect()))
}

#[pyfunction]
#[pyo3(signature = (
    images, reference_size, reference_idx=0, global_align=false, no_rotation=false,
    no_scale=false, no_shear=false, no_translation=false, full_res=false,
    min_shift=5.0, workers=0, progress=None, interrupt=None
))]
#[allow(clippy::too_many_arguments)]
fn align_images_py<'py>(
    py: Python<'py>,
    images: &Bound<'py, PyAny>,
    reference_size: (usize, usize),
    reference_idx: usize,
    global_align: bool,
    no_rotation: bool,
    no_scale: bool,
    no_shear: bool,
    no_translation: bool,
    full_res: bool,
    min_shift: f32,
    workers: usize,
    progress: Option<Py<PyAny>>,
    interrupt: Option<Py<PyAny>>,
) -> PyResult<Vec<Bound<'py, PyArray2<f32>>>> {
    let sources = sources_from_any(images)?;
    let options = AlignOptions {
        strategy: if global_align {
            AlignStrategy::Global
        } else {
            AlignStrategy::NeighbourChained
        },
        constraints: constraints(no_rotation, no_scale, no_shear, no_translation),
        full_res,
        min_shift,
        workers,
    };
    let callbacks = Callbacks::new(progress, interrupt, None);
    let warps = with_hooks(py, &callbacks, |hooks| {
        align_images(&sources, reference_size, reference_idx, options, hooks)
    })?;
    Ok(warps.iter().map(|m| affine_to_py(py, m)).collect())
}

#[pyfunction]
#[pyo3(signature = (
    images, warps, levels, sharpness, canvas_size=None, no_fill=false,
    workers=3, progress=None, interrupt=None
))]
#[allow(clippy::too_many_arguments)]
fn stack_images_py<'py>(
    py: Python<'py>,
    images: &Bound<'py, PyAny>,
    warps: &Bound<'py, PyAny>,
    levels: usize,
    sharpness: f32,
    canvas_size: Option<(usize, usize)>,
    no_fill: bool,
    workers: usize,
    progress: Option<Py<PyAny>>,
    interrupt: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let sources = sources_from_any(images)?;
    let warps = affines_from_any(warps)?;
    let options = StackOptions {
        levels,
        sharpness,
        canvas_size,
        no_fill,
        workers,
    };
    let callbacks = Callbacks::new(progress, interrupt, None);
    let image = with_hooks(py, &callbacks, |hooks| {
        stack_images(&sources, &warps, &options, hooks)
    })?;
    image_to_py(py, &image)
}

#[pyfunction]
#[pyo3(signature = (
    images, warps, slab_size, overlap, levels, sharpness, canvas_size,
    no_fill=false, workers=3, only_slab=false, recursive=false,
    on_slab=None, progress=None, interrupt=None
))]
#[allow(clippy::too_many_arguments)]
fn slab_images_py<'py>(
    py: Python<'py>,
    images: &Bound<'py, PyAny>,
    warps: &Bound<'py, PyAny>,
    slab_size: usize,
    overlap: usize,
    levels: usize,
    sharpness: f32,
    canvas_size: (usize, usize),
    no_fill: bool,
    workers: usize,
    only_slab: bool,
    recursive: bool,
    on_slab: Option<Py<PyAny>>,
    progress: Option<Py<PyAny>>,
    interrupt: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let sources = sources_from_any(images)?;
    let warps = affines_from_any(warps)?;
    let options = StackOptions {
        levels,
        sharpness,
        canvas_size: Some(canvas_size),
        no_fill,
        workers,
    };
    let callbacks = Callbacks::new(progress, interrupt, on_slab);
    let outcome = with_hooks(py, &callbacks, |hooks| {
        slab_images(
            &sources, &warps, slab_size, overlap, &options, only_slab, recursive, hooks,
        )
    })?;
    match outcome {
        SlabOutcome::Fused(image) => image_to_py(py, &image),
        SlabOutcome::Slabs(slabs) => {
            let items: PyResult<Vec<_>> = slabs.iter().map(|s| image_to_py(py, s)).collect();
            Ok(PyList::new(py, items?)?.into_any())
        }
    }
}

#[pyfunction]
#[pyo3(signature = (images, reference_size, threshold=0.05, progress=None, interrupt=None))]
fn cull_scores(
    py: Python<'_>,
    images: &Bound<'_, PyAny>,
    reference_size: (usize, usize),
    threshold: f64,
    progress: Option<Py<PyAny>>,
    interrupt: Option<Py<PyAny>>,
) -> PyResult<(Vec<f64>, Vec<bool>, f64, usize)> {
    let sources = sources_from_any(images)?;
    let callbacks = Callbacks::new(progress, interrupt, None);
    let result = with_hooks(py, &callbacks, |hooks| {
        cull_unfocused(&sources, reference_size, threshold, hooks)
    })?;
    Ok((
        result.entries.iter().map(|e| e.score).collect(),
        result.entries.iter().map(|e| e.kept).collect(),
        result.cutoff,
        result.n_culled,
    ))
}

#[pyfunction]
#[pyo3(signature = (config, progress=None, interrupt=None, on_slab=None))]
fn run_py<'py>(
    py: Python<'py>,
    config: &Bound<'py, PyAny>,
    progress: Option<Py<PyAny>>,
    interrupt: Option<Py<PyAny>>,
    on_slab: Option<Py<PyAny>>,
) -> PyResult<StackOutput<'py>> {
    let images_obj = config.get_item("images")?;
    let images = if let Some(path) = path_from_any(&images_obj)? {
        Images::Folder(path)
    } else {
        let sources = sources_from_any(&images_obj)?;
        if sources.iter().all(|s| matches!(s, Source::Path(_))) {
            Images::Paths(
                sources
                    .into_iter()
                    .map(|s| match s {
                        Source::Path(p) => p,
                        Source::Array(_) => unreachable!("checked above"),
                    })
                    .collect(),
            )
        } else {
            let mut arrays = Vec::with_capacity(sources.len());
            for source in sources {
                arrays.push(match source {
                    Source::Array(a) => a,
                    Source::Path(p) => focusweave_core::image_source::load_file(&p)
                        .map_err(|e| PyValueError::new_err(e.0))?,
                });
            }
            Images::Arrays(arrays)
        }
    };

    let get_bool = |name: &str| -> PyResult<bool> { config.get_item(name)?.extract() };
    let mut cfg = FocusStackConfig::new(images);
    cfg.no_align = get_bool("no_align")?;
    cfg.keep_size = get_bool("keep_size")?;
    cfg.crop = get_bool("crop")?;
    cfg.no_fill = get_bool("no_fill")?;
    cfg.reference = config.get_item("reference")?.extract()?;
    cfg.cull = config.get_item("cull")?.extract()?;
    cfg.global_align = get_bool("global_align")?;
    cfg.no_rotation = get_bool("no_rotation")?;
    cfg.no_scale = get_bool("no_scale")?;
    cfg.no_shear = get_bool("no_shear")?;
    cfg.no_translation = get_bool("no_translation")?;
    cfg.full_res = get_bool("full_res")?;
    cfg.min_shift = config.get_item("min_shift")?.extract()?;
    cfg.levels = config.get_item("levels")?.extract()?;
    cfg.sharpness = config.get_item("sharpness")?.extract()?;
    cfg.workers = config.get_item("workers")?.extract()?;
    cfg.slab = config.get_item("slab")?.extract()?;
    cfg.only_slab = get_bool("only_slab")?;
    cfg.recursive_slab = get_bool("recursive_slab")?;

    let callbacks = Callbacks::new(progress, interrupt, on_slab);
    let result = with_hooks(py, &callbacks, |hooks| core_run(&cfg, hooks))?;

    let image = match &result.image {
        Some(image) => Some(image_to_py(py, image)?),
        None => None,
    };
    let slabs = match &result.slabs {
        Some(slabs) => {
            let items: PyResult<Vec<_>> = slabs.iter().map(|s| image_to_py(py, s)).collect();
            Some(PyList::new(py, items?)?.into_any())
        }
        None => None,
    };
    Ok((image, slabs))
}

#[pyfunction]
fn cli_main(py: Python<'_>, argv: Vec<String>) -> i32 {
    py.detach(|| focusweave_core::cli::run_cli(&argv))
}

// ---------------------------------------------------------------------------
// Streaming stacker
// ---------------------------------------------------------------------------

#[pyclass(name = "StreamingFocusStacker", unsendable)]
struct PyStreamer {
    inner: CoreStreamer,
}

#[pymethods]
impl PyStreamer {
    #[new]
    #[pyo3(signature = (
        reference_size, reference=-1, cull_threshold=None, no_rotation=false,
        no_scale=false, no_shear=false, no_translation=false, full_res=false,
        min_shift=5.0, levels=0, sharpness=4.0, no_fill=false, workers=3,
        slab=None, only_slab=false, recursive_slab=false, preview_scale=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        reference_size: (usize, usize),
        reference: i64,
        cull_threshold: Option<f64>,
        no_rotation: bool,
        no_scale: bool,
        no_shear: bool,
        no_translation: bool,
        full_res: bool,
        min_shift: f32,
        levels: usize,
        sharpness: f32,
        no_fill: bool,
        workers: usize,
        slab: Option<(usize, usize)>,
        only_slab: bool,
        recursive_slab: bool,
        preview_scale: Option<f64>,
    ) -> Self {
        let config = StreamingConfig {
            reference,
            cull_threshold,
            no_rotation,
            no_scale,
            no_shear,
            no_translation,
            full_res,
            min_shift,
            levels,
            sharpness,
            no_fill,
            workers,
            slab,
            only_slab,
            recursive_slab,
            preview_scale,
        };
        PyStreamer {
            inner: CoreStreamer::new(reference_size, config),
        }
    }

    /// Add the next frame; returns a preview when one is configured.
    fn add_image<'py>(
        &mut self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let buf = image_buf_from_any(image)?;
        let preview = py.detach(|| self.inner.add_image(buf)).map_err(to_py_err)?;
        match preview {
            Some(p) => Ok(Some(mat_u8_to_py(py, &p)?)),
            None => Ok(None),
        }
    }

    fn get_preview<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.inner.get_preview() {
            Some(p) => Ok(Some(mat_u8_to_py(py, &p)?)),
            None => Ok(None),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[pyo3(signature = (keep_size=false, crop=false, progress=None, interrupt=None, on_slab=None))]
    fn finish<'py>(
        &self,
        py: Python<'py>,
        keep_size: bool,
        crop: bool,
        progress: Option<Py<PyAny>>,
        interrupt: Option<Py<PyAny>>,
        on_slab: Option<Py<PyAny>>,
    ) -> PyResult<StackOutput<'py>> {
        let callbacks = Callbacks::new(progress, interrupt, on_slab);
        let inner = &self.inner;
        let result = with_hooks(py, &callbacks, move |hooks| {
            inner.finish(keep_size, crop, hooks)
        })?;
        let image = match &result.image {
            Some(image) => Some(image_to_py(py, image)?),
            None => None,
        };
        let slabs = match &result.slabs {
            Some(slabs) => {
                let items: PyResult<Vec<_>> = slabs.iter().map(|s| image_to_py(py, s)).collect();
                Some(PyList::new(py, items?)?.into_any())
            }
            None => None,
        };
        Ok((image, slabs))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", focusweave_core::cli::VERSION)?;
    m.add("Interrupted", m.py().get_type::<Interrupted>())?;
    m.add("IMAGE_EXTENSIONS", PyTuple::new(m.py(), IMAGE_EXTENSIONS)?)?;
    m.add_function(wrap_pyfunction!(list_image_files, m)?)?;
    m.add_function(wrap_pyfunction!(probe_size, m)?)?;
    m.add_function(wrap_pyfunction!(read_image, m)?)?;
    m.add_function(wrap_pyfunction!(write_image, m)?)?;
    m.add_function(wrap_pyfunction!(compute_levels, m)?)?;
    m.add_function(wrap_pyfunction!(reduce, m)?)?;
    m.add_function(wrap_pyfunction!(expand, m)?)?;
    m.add_function(wrap_pyfunction!(region_energy, m)?)?;
    m.add_function(wrap_pyfunction!(region_deviation, m)?)?;
    m.add_function(wrap_pyfunction!(region_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(compute_canvas_py, m)?)?;
    m.add_function(wrap_pyfunction!(align_images_py, m)?)?;
    m.add_function(wrap_pyfunction!(stack_images_py, m)?)?;
    m.add_function(wrap_pyfunction!(slab_images_py, m)?)?;
    m.add_function(wrap_pyfunction!(cull_scores, m)?)?;
    m.add_function(wrap_pyfunction!(run_py, m)?)?;
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;
    m.add_class::<PyStreamer>()?;
    Ok(())
}
