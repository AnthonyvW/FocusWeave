//! 2x3 affine transforms and the decompositions the alignment stage needs.

/// Row-major 2x3 affine: `[m00, m01, m02, m10, m11, m12]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine(pub [f32; 6]);

impl Default for Affine {
    fn default() -> Self {
        Affine::IDENTITY
    }
}

impl Affine {
    pub const IDENTITY: Affine = Affine([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    #[inline]
    pub fn tx(&self) -> f32 {
        self.0[2]
    }

    #[inline]
    pub fn ty(&self) -> f32 {
        self.0[5]
    }

    pub fn translation(tx: f32, ty: f32) -> Affine {
        Affine([1.0, 0.0, tx, 0.0, 1.0, ty])
    }

    /// Compose so that `self` is applied after `other`.
    pub fn chain(&self, other: &Affine) -> Affine {
        let a = &self.0;
        let b = &other.0;
        Affine([
            a[0] * b[0] + a[1] * b[3],
            a[0] * b[1] + a[1] * b[4],
            a[0] * b[2] + a[1] * b[5] + a[2],
            a[3] * b[0] + a[4] * b[3],
            a[3] * b[1] + a[4] * b[4],
            a[3] * b[2] + a[4] * b[5] + a[5],
        ])
    }

    /// `cv2.invertAffineTransform`.
    pub fn invert(&self) -> Affine {
        let m = &self.0;
        let det = m[0] as f64 * m[4] as f64 - m[1] as f64 * m[3] as f64;
        let d = if det != 0.0 { 1.0 / det } else { 0.0 };
        let a11 = m[4] as f64 * d;
        let a22 = m[0] as f64 * d;
        let a12 = -(m[1] as f64) * d;
        let a21 = -(m[3] as f64) * d;
        let b1 = -a11 * m[2] as f64 - a12 * m[5] as f64;
        let b2 = -a21 * m[2] as f64 - a22 * m[5] as f64;
        Affine([
            a11 as f32, a12 as f32, b1 as f32, a21 as f32, a22 as f32, b2 as f32,
        ])
    }

    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.0;
        (
            m[0] as f64 * x + m[1] as f64 * y + m[2] as f64,
            m[3] as f64 * x + m[4] as f64 * y + m[5] as f64,
        )
    }

    pub fn is_identity(&self) -> bool {
        self.0 == Affine::IDENTITY.0
    }

    /// True when the linear block is the identity to within `tol`.
    pub fn is_pure_translation(&self, tol: f32) -> bool {
        let m = &self.0;
        (m[0] - 1.0).abs() <= tol
            && m[1].abs() <= tol
            && m[3].abs() <= tol
            && (m[4] - 1.0).abs() <= tol
    }

    pub fn translation_norm(&self) -> f32 {
        (self.0[2] * self.0[2] + self.0[5] * self.0[5]).sqrt()
    }

    /// Frobenius distance of the linear block from the identity.
    pub fn affine_distortion(&self) -> f64 {
        let m = &self.0;
        let d = [
            m[0] as f64 - 1.0,
            m[1] as f64,
            m[3] as f64,
            m[4] as f64 - 1.0,
        ];
        d.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Rotation angle in degrees implied by the linear block.
    pub fn rotation_degrees(&self) -> f64 {
        (self.0[3] as f64).atan2(self.0[0] as f64).to_degrees()
    }
}

/// SVD of a 2x2 matrix: returns `(u, sv, vt)` with `m = u * diag(sv) * vt`.
///
/// Singular values come back in descending order and non-negative, matching
/// `numpy.linalg.svd`.
pub fn svd2(m: [f64; 4]) -> ([f64; 4], [f64; 2], [f64; 4]) {
    let (a, b, c, d) = (m[0], m[1], m[2], m[3]);
    let e = (a + d) * 0.5;
    let f = (a - d) * 0.5;
    let g = (c + b) * 0.5;
    let h = (c - b) * 0.5;
    let q = e.hypot(h);
    let r = f.hypot(g);
    let sx = q + r;
    let mut sy = q - r;
    let theta = (h.atan2(e) - g.atan2(f)) * 0.5;
    let phi = (h.atan2(e) + g.atan2(f)) * 0.5;
    let (sin_phi, cos_phi) = phi.sin_cos();
    let (sin_theta, cos_theta) = theta.sin_cos();
    let u = [cos_phi, -sin_phi, sin_phi, cos_phi];
    let mut vt = [cos_theta, -sin_theta, sin_theta, cos_theta];
    if sy < 0.0 {
        // Absorb the sign into the second right singular vector so the
        // singular values come back non-negative, as numpy returns them.
        sy = -sy;
        vt[2] = -vt[2];
        vt[3] = -vt[3];
    }
    (u, [sx, sy], vt)
}

fn mat2_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

fn mat2_det(a: [f64; 4]) -> f64 {
    a[0] * a[3] - a[1] * a[2]
}

/// Which degrees of freedom to suppress when constraining a warp.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WarpConstraints {
    pub no_rotation: bool,
    pub no_scale: bool,
    pub no_shear: bool,
    pub no_translation: bool,
}

impl WarpConstraints {
    pub fn any(&self) -> bool {
        self.no_rotation || self.no_scale || self.no_shear || self.no_translation
    }

    fn any_linear(&self) -> bool {
        self.no_rotation || self.no_scale || self.no_shear
    }
}

/// Suppress selected degrees of freedom from an affine warp.
///
/// The linear part is factored by SVD as `M = U diag(sv) Vt`; the polar
/// rotation is `R = U Vt` and the symmetric scale/shear part is
/// `S = Vt' diag(sv) Vt`. Each flag drops one component before recomposing.
pub fn constrain_warp(warp: &Affine, c: WarpConstraints) -> Affine {
    if !c.any() {
        return *warp;
    }
    let mut out = *warp;
    if c.any_linear() {
        let m = [
            warp.0[0] as f64,
            warp.0[1] as f64,
            warp.0[3] as f64,
            warp.0[4] as f64,
        ];
        let (mut u, mut sv, vt) = svd2(m);
        if mat2_det(u) * mat2_det(vt) < 0.0 {
            u[1] = -u[1];
            u[3] = -u[3];
            sv[1] = -sv[1];
        }
        let r = mat2_mul(u, vt);
        if c.no_scale {
            let geomean = (sv[0] * sv[1]).abs().sqrt();
            let k = if geomean > 1e-10 { geomean } else { 1.0 };
            sv[0] /= k;
            sv[1] /= k;
        }
        let linear = if c.no_rotation && c.no_shear {
            [sv[0], 0.0, 0.0, sv[1]]
        } else if c.no_rotation {
            // Vt' diag(sv) Vt
            let v = [vt[0], vt[2], vt[1], vt[3]];
            mat2_mul(mat2_mul(v, [sv[0], 0.0, 0.0, sv[1]]), vt)
        } else if c.no_shear {
            mat2_mul(r, [sv[0], 0.0, 0.0, sv[1]])
        } else {
            mat2_mul([u[0] * sv[0], u[1] * sv[1], u[2] * sv[0], u[3] * sv[1]], vt)
        };
        out.0[0] = linear[0] as f32;
        out.0[1] = linear[1] as f32;
        out.0[3] = linear[2] as f32;
        out.0[4] = linear[3] as f32;
    }
    if c.no_translation {
        out.0[2] = 0.0;
        out.0[5] = 0.0;
    }
    out
}
