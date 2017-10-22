//
// This file is part of smallmath.
//
// smallmath is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// smallmath is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with smallmath. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright 2017 Chris Foster
//

#[cfg(feature = "with_blas")]
use blas;

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
use simd;

use vector::Vector;

#[cfg(feature = "with_blas")]
pub fn scale(a: &Vector, b: f32, c: &mut Vector) {
    unsafe { blas::c::saxpy(
        a.len() as i32, b, a, 1, c, 1,
    ) };
}

#[cfg(feature = "with_blas")]
pub fn scale_assign(a: &mut Vector, b: f32) {
    unsafe { blas::c::sscal(
        a.len() as i32, b, a, 1,
    ) };
}

#[cfg(not(feature = "with_blas"))]
pub fn scale(a: &Vector, b: f32, c: &mut Vector) {
    debug_assert!(a.len() == c.len(), "Source and destination vectors are different lengths!");

    use std::slice::from_raw_parts as s;
    use std::slice::from_raw_parts_mut as sm;

    if a.len() >= 16 {
        scale_16s(a, b, c);
    }

    if a.len() % 16 >= 8 {
        unsafe { scale_8(
            s(a.buffer.offset((a.len() as isize / 8 - 1) * 8), 8), b,
            sm(c.buffer.offset((a.len() as isize / 8 - 1) * 8), 8),
        ) };
    }

    if a.len() % 8 >= 4 {
        unsafe { scale_4(
            s(a.buffer.offset((a.len() as isize / 4 - 1) * 4), 4), b,
            sm(c.buffer.offset((a.len() as isize / 4 - 1) * 4), 4),
        ) };
    }

    if a.len() % 4 >= 2 {
        unsafe { scale_2(
            s(a.buffer.offset((a.len() as isize / 2 - 1) * 2), 2), b,
            sm(c.buffer.offset((a.len() as isize / 2 - 1) * 2), 2),
        ) };
    }

    if a.len() % 2 == 1 {
        c[a.len() - 1] = a[a.len() - 1] * b;
    }
}

#[cfg(not(feature = "with_blas"))]
pub fn scale_assign(a: &mut Vector, b: f32) {
    use std::slice::from_raw_parts_mut as sm;

    if a.len() >= 16 {
        scale_assign_16s(a, b);
    }

    if a.len() % 16 >= 8 {
        unsafe { scale_assign_8(
            sm(a.buffer.offset((a.len() as isize / 8 - 1) * 8), 8), b,
        ) };
    }

    if a.len() % 8 >= 4 {
        unsafe { scale_assign_4(
            sm(a.buffer.offset((a.len() as isize / 4 - 1) * 4), 4), b,
        ) };
    }

    if a.len() % 4 >= 2 {
        unsafe { scale_assign_2(
            sm(a.buffer.offset((a.len() as isize / 2 - 1) * 2), 2), b,
        ) };
    }

    if a.len() % 2 == 1 {
        let last_index = a.len() - 1;
        a[last_index] *= b;
    }
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn scale_16s(a: &[f32], b: f32, c: &mut [f32]) {
    let b_r = simd::x86::avx::f32x8::splat(b);

    for i in 0..(a.len() / 16) {
        let a_ra = simd::x86::avx::f32x8::load(a, i * 16 + 0);
        let a_rb = simd::x86::avx::f32x8::load(a, i * 16 + 8);

        let c_ra = a_ra * b_r;
        let c_rb = a_rb * b_r;

        c_ra.store(c, i * 16 + 0);
        c_rb.store(c, i * 16 + 8);
    }
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn scale_8(a: &[f32], b: f32, c: &mut [f32]) {
    let b_r = simd::x86::avx::f32x8::splat(b);
    let a_r = simd::x86::avx::f32x8::load(a, 0);

    let c_r = a_r * b_r;
    c_r.store(c, 0);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn scale_4(a: &[f32], b: f32, c: &mut [f32]) {
    let b_r = simd::f32x4::splat(b);
    let a_r = simd::f32x4::load(a, 0);

    let c_r = a_r * b_r;
    c_r.store(c, 0);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn scale_2(a: &[f32], b: f32, c: &mut [f32]) {
    let b_r = simd::f32x4::splat(b);
    let a_r = simd::f32x4::new(a[0], a[1], 0.0, 0.0);

    let c_r = a_r * b_r;
    c[0] = c_r.extract(0); c[1] = c_r.extract(1);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn scale_assign_16s(a: &mut[f32], b: f32) {
    let b_r = simd::x86::avx::f32x8::splat(b);

    for i in 0..(a.len() / 16) {
        let a_ra = simd::x86::avx::f32x8::load(a, i * 16 + 0) * b_r;
        let a_rb = simd::x86::avx::f32x8::load(a, i * 16 + 8) * b_r;

        a_ra.store(a, i * 16 + 0);
        a_rb.store(a, i * 16 + 8);
    }
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn scale_assign_8(a: &mut[f32], b: f32) {
    let b_r = simd::x86::avx::f32x8::splat(b);

    let a_r = simd::x86::avx::f32x8::load(a, 0) * b_r;

    a_r.store(a, 0);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn scale_assign_4(a: &mut[f32], b: f32) {
    let b_r = simd::f32x4::splat(b);

    let a_r = simd::f32x4::load(a, 0) * b_r;

    a_r.store(a, 0);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn scale_assign_2(a: &mut[f32], b: f32) {
    let b_r = simd::f32x4::splat(b);

    let a_r = simd::f32x4::new(a[0], a[1], 0.0, 0.0) * b_r;

    a[0] = a_r.extract(0); a[1] = a_r.extract(1);
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn scale_16s(a: &[f32], b: f32, c: &mut [f32]) {
    for i in 0..(a.len() / 16) {
        c[i * 16 +  0] = a[i * 16 +  0] * b;
        c[i * 16 +  1] = a[i * 16 +  1] * b;
        c[i * 16 +  2] = a[i * 16 +  2] * b;
        c[i * 16 +  3] = a[i * 16 +  3] * b;
        c[i * 16 +  4] = a[i * 16 +  4] * b;
        c[i * 16 +  5] = a[i * 16 +  5] * b;
        c[i * 16 +  6] = a[i * 16 +  6] * b;
        c[i * 16 +  7] = a[i * 16 +  7] * b;
        c[i * 16 +  8] = a[i * 16 +  8] * b;
        c[i * 16 +  9] = a[i * 16 +  9] * b;
        c[i * 16 + 10] = a[i * 16 + 10] * b;
        c[i * 16 + 11] = a[i * 16 + 11] * b;
        c[i * 16 + 12] = a[i * 16 + 12] * b;
        c[i * 16 + 13] = a[i * 16 + 13] * b;
        c[i * 16 + 14] = a[i * 16 + 14] * b;
        c[i * 16 + 15] = a[i * 16 + 15] * b;
    }
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn scale_8(a: &[f32], b: f32, c: &mut [f32]) {
    c[0] = a[0] * b;
    c[1] = a[1] * b;
    c[2] = a[2] * b;
    c[3] = a[3] * b;
    c[4] = a[4] * b;
    c[5] = a[5] * b;
    c[6] = a[6] * b;
    c[7] = a[7] * b;
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn scale_4(a: &[f32], b: f32, c: &mut [f32]) {
    c[0] = a[0] * b;
    c[1] = a[1] * b;
    c[2] = a[2] * b;
    c[3] = a[3] * b;
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn scale_2(a: &[f32], b: f32, c: &mut [f32]) {
    c[0] = a[0] * b;
    c[1] = a[1] * b;
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn scale_assign_16s(a: &mut[f32], b: f32) {
    for i in 0..(a.len() / 16) {
        a[i * 16 +  0] *= b;
        a[i * 16 +  1] *= b;
        a[i * 16 +  2] *= b;
        a[i * 16 +  3] *= b;
        a[i * 16 +  4] *= b;
        a[i * 16 +  5] *= b;
        a[i * 16 +  6] *= b;
        a[i * 16 +  7] *= b;
        a[i * 16 +  8] *= b;
        a[i * 16 +  9] *= b;
        a[i * 16 + 10] *= b;
        a[i * 16 + 11] *= b;
        a[i * 16 + 12] *= b;
        a[i * 16 + 13] *= b;
        a[i * 16 + 14] *= b;
        a[i * 16 + 15] *= b;
    }
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn scale_assign_8(a: &mut[f32], b: f32) {
    a[0] *= b;
    a[1] *= b;
    a[2] *= b;
    a[3] *= b;
    a[4] *= b;
    a[5] *= b;
    a[6] *= b;
    a[7] *= b;
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn scale_assign_4(a: &mut[f32], b: f32) {
    a[0] *= b;
    a[1] *= b;
    a[2] *= b;
    a[3] *= b;
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn scale_assign_2(a: &mut[f32], b: f32) {
    a[0] *= b;
    a[1] *= b;
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_scale(bencher: &mut Bencher) {
        let mut a = Vector::zeros(1000);
        for (i, x) in a.iter_mut().enumerate() {
            *x = (i % 16 + 1) as f32;
        }

        let b = 2.0;

        let mut c = Vector::zeros(1000);

        bencher.iter(|| {
            scale(&a, b, &mut c);
        });
    }

    #[bench]
    fn bench_scale_assign(bencher: &mut Bencher) {
        let mut a = Vector::zeros(1000);
        for (i, x) in a.iter_mut().enumerate() {
            *x = (i % 16 + 1) as f32;
        }

        let b = 2.0;

        bencher.iter(|| {
            scale_assign(&mut a, b);
        });
    }
}
