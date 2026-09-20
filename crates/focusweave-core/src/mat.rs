//! Dense row-major image buffers with interleaved channels.

/// Row-major image with `c` interleaved channels.
#[derive(Clone, Debug, PartialEq)]
pub struct Img<T> {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub data: Vec<T>,
}

pub type Mat = Img<f32>;
pub type MatU8 = Img<u8>;
pub type MatU16 = Img<u16>;

impl<T: Copy + Default> Img<T> {
    pub fn new(h: usize, w: usize, c: usize) -> Self {
        Img {
            h,
            w,
            c,
            data: vec![T::default(); h * w * c],
        }
    }

    pub fn from_vec(h: usize, w: usize, c: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            h * w * c,
            "buffer length does not match {h}x{w}x{c}"
        );
        Img { h, w, c, data }
    }

    pub fn filled(h: usize, w: usize, c: usize, value: T) -> Self {
        Img {
            h,
            w,
            c,
            data: vec![value; h * w * c],
        }
    }
}

impl<T> Img<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// (width, height), matching OpenCV's `Size` ordering.
    #[inline]
    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.w * self.c
    }

    #[inline]
    pub fn row(&self, y: usize) -> &[T] {
        let s = self.stride();
        &self.data[y * s..(y + 1) * s]
    }

    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        let s = self.stride();
        &mut self.data[y * s..(y + 1) * s]
    }

    #[inline]
    pub fn at(&self, y: usize, x: usize, ch: usize) -> &T {
        &self.data[(y * self.w + x) * self.c + ch]
    }

    #[inline]
    pub fn at_mut(&mut self, y: usize, x: usize, ch: usize) -> &mut T {
        let c = self.c;
        let w = self.w;
        &mut self.data[(y * w + x) * c + ch]
    }

    pub fn same_shape<U>(&self, other: &Img<U>) -> bool {
        self.h == other.h && self.w == other.w && self.c == other.c
    }
}

impl Mat {
    /// Extract a single channel as a 1-channel image.
    pub fn channel(&self, ch: usize) -> Mat {
        if self.c == 1 {
            return self.clone();
        }
        let mut out = Mat::new(self.h, self.w, 1);
        for (dst, src) in out.data.iter_mut().zip(self.data.chunks_exact(self.c)) {
            *dst = src[ch];
        }
        out
    }

    pub fn max(&self) -> f32 {
        self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn min(&self) -> f32 {
        self.data.iter().copied().fold(f32::INFINITY, f32::min)
    }

    pub fn map_into(mut self, f: impl Fn(f32) -> f32) -> Mat {
        for v in &mut self.data {
            *v = f(*v);
        }
        self
    }

    pub fn add_assign(&mut self, other: &Mat) {
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += *b;
        }
    }

    pub fn scale_assign(&mut self, k: f32) {
        for a in &mut self.data {
            *a *= k;
        }
    }

    pub fn dot(&self, other: &Mat) -> f64 {
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum()
    }
}

impl MatU8 {
    pub fn count_non_zero(&self) -> usize {
        self.data.iter().filter(|v| **v != 0).count()
    }

    pub fn to_f32(&self) -> Mat {
        Mat {
            h: self.h,
            w: self.w,
            c: self.c,
            data: self.data.iter().map(|v| f32::from(*v)).collect(),
        }
    }
}
