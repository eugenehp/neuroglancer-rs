//! The gl-matrix 3.1.0 functions neuroglancer's navigation uses, with its
//! numerics: values are stored in `Float32Array`s (`f32` here), arithmetic is
//! JavaScript's (`f64`), and each stored result is rounded to `f32`. Each
//! function is transcribed from gl-matrix's source so results agree bit for
//! bit.

use crate::js;

pub type Quat = [f32; 4];
pub type Vec3 = [f32; 3];
/// Column-major.
pub type Mat3 = [f32; 9];
/// Column-major.
pub type Mat4 = [f32; 16];

pub const QUAT_IDENTITY: Quat = [0.0, 0.0, 0.0, 1.0];

fn w(x: f32) -> f64 {
    x as f64
}
fn s(x: f64) -> f32 {
    x as f32
}

/// `quat.normalize` (= `vec4.normalize`).
pub fn quat_normalize(a: Quat) -> Quat {
    let [x, y, z, ww] = a.map(w);
    let mut len = x * x + y * y + z * z + ww * ww;
    if len > 0.0 {
        len = 1.0 / len.sqrt();
    }
    [s(x * len), s(y * len), s(z * len), s(ww * len)]
}

/// `quat.multiply(out, a, b)`: `a * b`.
pub fn quat_multiply(a: Quat, b: Quat) -> Quat {
    let [ax, ay, az, aw] = a.map(w);
    let [bx, by, bz, bw] = b.map(w);
    [
        s(ax * bw + aw * bx + ay * bz - az * by),
        s(ay * bw + aw * by + az * bx - ax * bz),
        s(az * bw + aw * bz + ax * by - ay * bx),
        s(aw * bw - ax * bx - ay * by - az * bz),
    ]
}

/// `quat.setAxisAngle`.
pub fn quat_set_axis_angle(axis: Vec3, rad: f64) -> Quat {
    let rad = rad * 0.5;
    let sn = js::sin(rad);
    [
        s(sn * w(axis[0])),
        s(sn * w(axis[1])),
        s(sn * w(axis[2])),
        s(js::cos(rad)),
    ]
}

/// `quat.invert`.
pub fn quat_invert(a: Quat) -> Quat {
    let [a0, a1, a2, a3] = a.map(w);
    let dot = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
    let inv_dot = if dot != 0.0 { 1.0 / dot } else { 0.0 };
    [s(-a0 * inv_dot), s(-a1 * inv_dot), s(-a2 * inv_dot), s(a3 * inv_dot)]
}

/// `quat.fromMat3`.
pub fn quat_from_mat3(m: Mat3) -> Quat {
    let m = m.map(w);
    let mut out = [0.0f32; 4];
    let trace = m[0] + m[4] + m[8];
    if trace > 0.0 {
        let mut root = (trace + 1.0).sqrt();
        out[3] = s(0.5 * root);
        root = 0.5 / root;
        out[0] = s((m[5] - m[7]) * root);
        out[1] = s((m[6] - m[2]) * root);
        out[2] = s((m[1] - m[3]) * root);
    } else {
        let mut i = 0;
        if m[4] > m[0] {
            i = 1;
        }
        if m[8] > m[i * 3 + i] {
            i = 2;
        }
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let mut root = (m[i * 3 + i] - m[j * 3 + j] - m[k * 3 + k] + 1.0).sqrt();
        out[i] = s(0.5 * root);
        root = 0.5 / root;
        out[3] = s((m[j * 3 + k] - m[k * 3 + j]) * root);
        out[j] = s((m[j * 3 + i] + m[i * 3 + j]) * root);
        out[k] = s((m[k * 3 + i] + m[i * 3 + k]) * root);
    }
    out
}

/// `vec3.transformQuat`.
pub fn vec3_transform_quat(a: Vec3, q: Quat) -> Vec3 {
    let [qx, qy, qz, qw] = q.map(w);
    let [x, y, z] = a.map(w);
    let (mut uvx, mut uvy, mut uvz) = (qy * z - qz * y, qz * x - qx * z, qx * y - qy * x);
    let (mut uuvx, mut uuvy, mut uuvz) = (
        qy * uvz - qz * uvy,
        qz * uvx - qx * uvz,
        qx * uvy - qy * uvx,
    );
    let w2 = qw * 2.0;
    uvx *= w2;
    uvy *= w2;
    uvz *= w2;
    uuvx *= 2.0;
    uuvy *= 2.0;
    uuvz *= 2.0;
    [s(x + uvx + uuvx), s(y + uvy + uuvy), s(z + uvz + uuvz)]
}

/// The products `mat3.fromQuat` and `mat4.fromQuat` share.
fn quat_products(q: Quat) -> [f64; 9] {
    let [x, y, z, ww] = q.map(w);
    let (x2, y2, z2) = (x + x, y + y, z + z);
    [
        x * x2,
        y * x2,
        y * y2,
        z * x2,
        z * y2,
        z * z2,
        ww * x2,
        ww * y2,
        ww * z2,
    ]
}

/// `mat3.fromQuat`.
pub fn mat3_from_quat(q: Quat) -> Mat3 {
    let [xx, yx, yy, zx, zy, zz, wx, wy, wz] = quat_products(q);
    [
        s(1.0 - yy - zz),
        s(yx + wz),
        s(zx - wy),
        s(yx - wz),
        s(1.0 - xx - zz),
        s(zy + wx),
        s(zx + wy),
        s(zy - wx),
        s(1.0 - xx - yy),
    ]
}

/// `mat4.fromQuat`.
pub fn mat4_from_quat(q: Quat) -> Mat4 {
    let [xx, yx, yy, zx, zy, zz, wx, wy, wz] = quat_products(q);
    [
        s(1.0 - yy - zz),
        s(yx + wz),
        s(zx - wy),
        0.0,
        s(yx - wz),
        s(1.0 - xx - zz),
        s(zy + wx),
        0.0,
        s(zx + wy),
        s(zy - wx),
        s(1.0 - xx - yy),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

/// `vec3.transformMat4`: `w` of 0 is treated as 1 (`w || 1.0`).
pub fn vec3_transform_mat4(a: Vec3, m: Mat4) -> Vec3 {
    let [x, y, z] = a.map(w);
    let m = m.map(w);
    let mut ww = m[3] * x + m[7] * y + m[11] * z + m[15];
    if ww == 0.0 || ww.is_nan() {
        ww = 1.0;
    }
    [
        s((m[0] * x + m[4] * y + m[8] * z + m[12]) / ww),
        s((m[1] * x + m[5] * y + m[9] * z + m[13]) / ww),
        s((m[2] * x + m[6] * y + m[10] * z + m[14]) / ww),
    ]
}

/// `kAxes` (`src/util/geom.ts`).
pub const AXES: [Vec3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// `quat.rotateX(out, a, rad)`.
pub fn quat_rotate_x(a: Quat, rad: f64) -> Quat {
    let rad = rad * 0.5;
    let [ax, ay, az, aw] = a.map(w);
    let (bx, bw) = (js::sin(rad), js::cos(rad));
    [s(ax * bw + aw * bx), s(ay * bw + az * bx), s(az * bw - ay * bx), s(aw * bw - ax * bx)]
}

/// `quat.rotateY(out, a, rad)`.
pub fn quat_rotate_y(a: Quat, rad: f64) -> Quat {
    let rad = rad * 0.5;
    let [ax, ay, az, aw] = a.map(w);
    let (by, bw) = (js::sin(rad), js::cos(rad));
    [s(ax * bw - az * by), s(ay * bw + aw * by), s(az * bw + ax * by), s(aw * bw - ay * by)]
}

// ── mat4, as gl-matrix computes it: f32 storage, doubles in between ───────

fn f(m: &Mat4, i: usize) -> f64 {
    m[i] as f64
}

/// `mat4.multiply(out, a, b)`: `a · b`.
pub fn mat4_multiply(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0f32; 16];
    for col in 0..4 {
        let (b0, b1, b2, b3) = (f(b, col * 4), f(b, col * 4 + 1), f(b, col * 4 + 2), f(b, col * 4 + 3));
        for row in 0..4 {
            out[col * 4 + row] = (b0 * f(a, row) + b1 * f(a, 4 + row) + b2 * f(a, 8 + row) + b3 * f(a, 12 + row)) as f32;
        }
    }
    out
}

/// `mat4.invert`; `None` for a singular matrix.
pub fn mat4_invert(a: &Mat4) -> Option<Mat4> {
    let [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = a.map(|x| x as f64);
    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if det == 0.0 || det.is_nan() {
        return None;
    }
    let det = 1.0 / det;
    Some(
        [
            (a11 * b11 - a12 * b10 + a13 * b09) * det,
            (a02 * b10 - a01 * b11 - a03 * b09) * det,
            (a31 * b05 - a32 * b04 + a33 * b03) * det,
            (a22 * b04 - a21 * b05 - a23 * b03) * det,
            (a12 * b08 - a10 * b11 - a13 * b07) * det,
            (a00 * b11 - a02 * b08 + a03 * b07) * det,
            (a32 * b02 - a30 * b05 - a33 * b01) * det,
            (a20 * b05 - a22 * b02 + a23 * b01) * det,
            (a10 * b10 - a11 * b08 + a13 * b06) * det,
            (a01 * b08 - a00 * b10 - a03 * b06) * det,
            (a30 * b04 - a31 * b02 + a33 * b00) * det,
            (a21 * b02 - a20 * b04 - a23 * b00) * det,
            (a11 * b07 - a10 * b09 - a12 * b06) * det,
            (a00 * b09 - a01 * b07 + a02 * b06) * det,
            (a31 * b01 - a30 * b03 - a32 * b00) * det,
            (a20 * b03 - a21 * b01 + a22 * b00) * det,
        ]
        .map(|x| x as f32),
    )
}

/// `mat4.perspective(out, fovy, aspect, near, far)` (finite `far`).
pub fn mat4_perspective(fovy: f64, aspect: f64, near: f64, far: f64) -> Mat4 {
    let f = 1.0 / crate::js::tan(fovy / 2.0);
    let nf = 1.0 / (near - far);
    let mut out = [0f32; 16];
    out[0] = (f / aspect) as f32;
    out[5] = f as f32;
    out[10] = ((far + near) * nf) as f32;
    out[11] = -1.0;
    out[14] = (2.0 * far * near * nf) as f32;
    out
}

/// `mat4.ortho(out, left, right, bottom, top, near, far)`.
pub fn mat4_ortho(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Mat4 {
    let lr = 1.0 / (left - right);
    let bt = 1.0 / (bottom - top);
    let nf = 1.0 / (near - far);
    let mut out = [0f32; 16];
    out[0] = (-2.0 * lr) as f32;
    out[5] = (-2.0 * bt) as f32;
    out[10] = (2.0 * nf) as f32;
    out[12] = ((left + right) * lr) as f32;
    out[13] = ((top + bottom) * bt) as f32;
    out[14] = ((far + near) * nf) as f32;
    out[15] = 1.0;
    out
}

/// `mat4.scale(out, a, v)`.
pub fn mat4_scale(a: &Mat4, v: [f64; 3]) -> Mat4 {
    let mut out = *a;
    for (col, &s) in v.iter().enumerate() {
        for row in 0..4 {
            out[col * 4 + row] = (f(a, col * 4 + row) * s) as f32;
        }
    }
    out
}

/// `mat4.translate(out, a, v)`.
pub fn mat4_translate(a: &Mat4, v: [f64; 3]) -> Mat4 {
    let mut out = *a;
    for row in 0..4 {
        out[12 + row] = (f(a, row) * v[0] + f(a, 4 + row) * v[1] + f(a, 8 + row) * v[2] + f(a, 12 + row)) as f32;
    }
    out
}

/// `vec4.transformMat4` of `(x, y, z, w)`.
pub fn vec4_transform_mat4(p: [f64; 4], m: &Mat4) -> [f64; 4] {
    std::array::from_fn(|r| f(m, r) * p[0] + f(m, 4 + r) * p[1] + f(m, 8 + r) * p[2] + f(m, 12 + r) * p[3])
}
