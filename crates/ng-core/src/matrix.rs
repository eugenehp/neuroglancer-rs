//! Dense column-major matrices over `f64`. Port of the parts of
//! `src/util/matrix.ts` that metadata parsing uses, with the same loop order
//! and pivoting, so results agree bit for bit.

/// `createIdentity`: an `rows × cols` identity, column-major.
pub fn identity(rows: usize, cols: usize) -> Vec<f64> {
    let mut m = vec![0.0; rows * cols];
    for i in 0..rows.min(cols) {
        m[i * rows + i] = 1.0;
    }
    m
}

/// `multiply(c, ldc, a, lda, b, ldb, m, n, k)`: `c = a · b` with `a` m×n and
/// `b` n×k, each with its own leading dimension.
#[allow(clippy::too_many_arguments)]
pub fn multiply_into(c: &mut [f64], ldc: usize, a: &[f64], lda: usize, b: &[f64], ldb: usize, m: usize, n: usize, k: usize) {
    for mi in 0..m {
        for ki in 0..k {
            let mut sum = 0.0;
            for ni in 0..n {
                sum += a[mi + lda * ni] * b[ni + ldb * ki];
            }
            c[mi + ldc * ki] = sum;
        }
    }
}

/// `a · b` for square `n × n` matrices.
pub fn multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    multiply_into(&mut c, n, a, n, b, n, n, n, n);
    c
}

/// `isIdentity(a, lda, n)`.
pub fn is_identity(a: &[f64], lda: usize, n: usize) -> bool {
    (0..n).all(|i| (0..n).all(|j| a[i * lda + j] == if i == j { 1.0 } else { 0.0 }))
}

/// `inverse` / `inverseInplace`: Gauss-Jordan with partial pivoting. Returns
/// the inverse and the determinant (a singular matrix gives infinities/NaN, as
/// in the TS).
pub fn inverse(a: &[f64], n: usize) -> (Vec<f64>, f64) {
    let lda = n;
    let mut a = a.to_vec();
    let mut determinant = 1.0;
    let mut pivots: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let k_col = lda * k;
        // Best pivot: the row >= k with the largest magnitude in column k.
        let mut pivot_row = k;
        let mut best = a[k_col + k].abs();
        for row in k + 1..n {
            let mag = a[k_col + row].abs();
            if mag > best {
                best = mag;
                pivot_row = row;
            }
        }
        if k != pivot_row {
            determinant *= -1.0;
            for col in 0..n {
                a.swap(lda * col + k, lda * col + pivot_row);
            }
            pivots.swap(k, pivot_row);
        }
        let pivot_value = a[k_col + k];
        let pivot_inv = 1.0 / pivot_value;
        determinant *= pivot_value;
        for j in 0..n {
            a[lda * j + k] *= pivot_inv;
        }
        a[k_col + k] = pivot_inv;
        for row in 0..n {
            if row == k {
                continue;
            }
            let factor = -a[lda * k + row];
            for j in 0..n {
                let j_col = lda * j;
                a[j_col + row] += factor * a[j_col + k];
            }
            a[lda * k + row] = factor * pivot_inv;
        }
    }
    // Permute columns back.
    for col in 0..n {
        let mut target = pivots[col];
        while target != col {
            for i in 0..n {
                a.swap(lda * col + i, lda * target + i);
            }
            let t = pivots[target];
            pivots[col] = t;
            pivots[target] = target;
            target = t;
        }
    }
    (a, determinant)
}

/// A column-major 4×4 over `f64`: cameras and model matrices.
pub type Mat4 = [f64; 16];

/// The 4×4 identity.
pub const IDENTITY4: Mat4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// `a · b` for 4×4s.
pub fn multiply4(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut c = [0.0; 16];
    multiply_into(&mut c, 4, a, 4, b, 4, 4, 4, 4);
    c
}

/// [`inverse`] of a 4×4: a singular matrix gives infinities/NaN.
pub fn inverse4(a: &Mat4) -> Mat4 {
    inverse(a, 4).0.try_into().unwrap()
}

/// The inverse of a 4×4, if it has a finite one.
pub fn try_inverse4(a: &Mat4) -> Option<Mat4> {
    let (inv, det) = inverse(a, 4);
    (det != 0.0 && inv.iter().all(|x| x.is_finite())).then(|| inv.try_into().unwrap())
}

/// Point `p` through `m`, divided by `w` (a zero `w` taken as 1).
pub fn transform_point4(m: &Mat4, p: [f64; 3]) -> [f64; 3] {
    let w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    let w = if w == 0.0 { 1.0 } else { w };
    [0, 1, 2].map(|r| (m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]) / w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse4_round_trips_and_rejects_singular() {
        let m = [2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 5.0, 6.0, 7.0, 1.0];
        let p = multiply4(&m, &try_inverse4(&m).unwrap());
        for (x, y) in p.iter().zip(IDENTITY4) {
            assert!((x - y).abs() < 1e-12, "{p:?}");
        }
        assert_eq!(try_inverse4(&[0.0; 16]), None);
    }

    #[test]
    fn inverse_round_trips() {
        // A permutation-and-scale matrix that needs pivoting.
        let m = [0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let (inv, det) = inverse(&m, 3);
        assert_eq!(multiply(&m, &inv, 3), identity(3, 3));
        assert_eq!(det, -6.0);
    }
}
