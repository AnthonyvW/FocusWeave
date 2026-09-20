//! Image loading and the source abstraction shared by every stage.

use crate::cv::{resize_area, resize_area_u8};
use crate::mat::{Img, Mat, MatU16, MatU8};
use image::DynamicImage;
use std::path::{Path, PathBuf};

/// Extensions the loader recognises, matching the original `IMAGE_EXTENSIONS`.
pub const IMAGE_EXTENSIONS: [&str; 6] = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"];

pub fn is_image_path(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            let lower = format!(".{}", ext.to_ascii_lowercase());
            IMAGE_EXTENSIONS.contains(&lower.as_str())
        }
        None => false,
    }
}

/// An in-memory RGB image at its native bit depth.
#[derive(Clone, Debug)]
pub enum ImageBuf {
    U8(MatU8),
    U16(MatU16),
}

impl ImageBuf {
    pub fn width(&self) -> usize {
        match self {
            ImageBuf::U8(m) => m.w,
            ImageBuf::U16(m) => m.w,
        }
    }

    pub fn height(&self) -> usize {
        match self {
            ImageBuf::U8(m) => m.h,
            ImageBuf::U16(m) => m.h,
        }
    }

    pub fn depth(&self) -> u32 {
        match self {
            ImageBuf::U8(_) => 8,
            ImageBuf::U16(_) => 16,
        }
    }
}

/// Where a frame comes from: a file on disk or an array supplied by the caller.
#[derive(Clone, Debug)]
pub enum Source {
    Path(PathBuf),
    Array(ImageBuf),
}

impl Source {
    pub fn label(&self, index: usize) -> String {
        match self {
            Source::Path(p) => p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            Source::Array(_) => format!("image_{index}"),
        }
    }
}

#[derive(Debug)]
pub struct LoadError(pub String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LoadError {}

fn decode(path: &Path) -> Result<DynamicImage, LoadError> {
    image::open(path)
        .map_err(|e| LoadError(format!("could not read image '{}': {e}", path.display())))
}

/// Read a file as RGB at its native bit depth.
pub fn load_file(path: &Path) -> Result<ImageBuf, LoadError> {
    let img = decode(path)?;
    Ok(dynamic_to_buf(img))
}

fn dynamic_to_buf(img: DynamicImage) -> ImageBuf {
    let sixteen = matches!(
        img.color(),
        image::ColorType::L16
            | image::ColorType::La16
            | image::ColorType::Rgb16
            | image::ColorType::Rgba16
    );
    if sixteen {
        let rgb = img.to_rgb16();
        let (w, h) = rgb.dimensions();
        ImageBuf::U16(Img::from_vec(h as usize, w as usize, 3, rgb.into_raw()))
    } else {
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        ImageBuf::U8(Img::from_vec(h as usize, w as usize, 3, rgb.into_raw()))
    }
}

/// `(width, height)` of an image file.
pub fn image_size(path: &Path) -> Result<(usize, usize), LoadError> {
    let (w, h) = image::image_dimensions(path)
        .map_err(|e| LoadError(format!("could not read image '{}': {e}", path.display())))?;
    Ok((w as usize, h as usize))
}

pub fn source_size(src: &Source) -> Result<(usize, usize), LoadError> {
    match src {
        Source::Path(p) => image_size(p),
        Source::Array(a) => Ok((a.width(), a.height())),
    }
}

pub fn source_depth(src: &Source) -> Result<u32, LoadError> {
    match src {
        Source::Path(p) => Ok(load_file(p)?.depth()),
        Source::Array(a) => Ok(a.depth()),
    }
}

/// Load as 8-bit RGB, resizing to `size` when it differs.
///
/// Used by alignment and sharpness scoring, where 8 bits is enough; 16-bit
/// sources are shifted down the way `cv2.IMREAD_COLOR` does.
pub fn load_u8(src: &Source, size: (usize, usize)) -> Result<MatU8, LoadError> {
    let buf = match src {
        Source::Path(p) => load_file(p)?,
        Source::Array(a) => a.clone(),
    };
    let m = match buf {
        ImageBuf::U8(m) => m,
        ImageBuf::U16(m) => MatU8 {
            h: m.h,
            w: m.w,
            c: m.c,
            data: m.data.iter().map(|v| (v >> 8) as u8).collect(),
        },
    };
    Ok(if m.w != size.0 || m.h != size.1 {
        resize_area_u8(&m, size.0, size.1)
    } else {
        m
    })
}

/// Load as float RGB in the source's native range, resizing to `size`.
pub fn load_native_f32(src: &Source, size: (usize, usize)) -> Result<Mat, LoadError> {
    let buf = match src {
        Source::Path(p) => load_file(p)?,
        Source::Array(a) => a.clone(),
    };
    let m = match buf {
        ImageBuf::U8(m) => Mat {
            h: m.h,
            w: m.w,
            c: m.c,
            data: m.data.iter().map(|v| f32::from(*v)).collect(),
        },
        ImageBuf::U16(m) => Mat {
            h: m.h,
            w: m.w,
            c: m.c,
            data: m.data.iter().map(|v| f32::from(*v)).collect(),
        },
    };
    Ok(if m.w != size.0 || m.h != size.1 {
        resize_area(&m, size.0, size.1)
    } else {
        m
    })
}

/// Discover image files in a folder, sorted by name.
pub fn list_folder(folder: &Path) -> Result<Vec<PathBuf>, LoadError> {
    let entries = std::fs::read_dir(folder).map_err(|e| {
        LoadError(format!(
            "could not read directory '{}': {e}",
            folder.display()
        ))
    })?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_image_path(p))
        .collect();
    paths.sort();
    Ok(paths)
}

/// Write an RGB image, choosing the encoder from the path's extension.
///
/// JPEG and WebP cannot carry 16 bits, so 16-bit images are reduced before
/// being written to those formats; every other format keeps full depth.
pub fn save_image(img: &ImageBuf, path: &Path, quality: u8) -> Result<(), LoadError> {
    let suffix = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    let low_depth = matches!(suffix.as_str(), "jpg" | "jpeg" | "webp");

    let reduced;
    let img = match img {
        ImageBuf::U16(m) if low_depth => {
            reduced = ImageBuf::U8(MatU8 {
                h: m.h,
                w: m.w,
                c: m.c,
                data: m.data.iter().map(|v| (v >> 8) as u8).collect(),
            });
            &reduced
        }
        other => other,
    };

    let fail = |e: image::ImageError| {
        LoadError(format!("could not write image '{}': {e}", path.display()))
    };

    match img {
        ImageBuf::U8(m) => {
            let buf = image::RgbImage::from_raw(m.w as u32, m.h as u32, m.data.clone())
                .ok_or_else(|| LoadError("image buffer has the wrong length".into()))?;
            if matches!(suffix.as_str(), "jpg" | "jpeg") {
                let file = std::fs::File::create(path).map_err(|e| {
                    LoadError(format!("could not create '{}': {e}", path.display()))
                })?;
                let mut writer = std::io::BufWriter::new(file);
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, quality)
                    .encode_image(&buf)
                    .map_err(fail)
            } else {
                buf.save(path).map_err(fail)
            }
        }
        ImageBuf::U16(m) => {
            let buf = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_raw(
                m.w as u32,
                m.h as u32,
                m.data.clone(),
            )
            .ok_or_else(|| LoadError("image buffer has the wrong length".into()))?;
            buf.save(path).map_err(fail)
        }
    }
}
