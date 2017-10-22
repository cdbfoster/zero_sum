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

#[cfg(feature = "with_simd")]
use simd;

use vector::Vector;

pub fn signum(a: &Vector, b: &mut Vector) {
    debug_assert!(a.len() == b.len(), "Operand vectors are different lengths!");

    use std::slice::from_raw_parts as s;
    use std::slice::from_raw_parts_mut as sm;

    if a.len() >= 16 {
        signum_16s(a, b);
    }

    if a.len() % 16 >= 8 {
        unsafe { signum_8(
            s(a.buffer.offset((a.len() as isize / 8 - 1) * 8), 8),
            sm(b.buffer.offset((a.len() as isize / 8 - 1) * 8), 8),
        ) };
    }

    if a.len() % 8 >= 4 {
        unsafe { signum_4(
            s(a.buffer.offset((a.len() as isize / 4 - 1) * 4), 4),
            sm(b.buffer.offset((a.len() as isize / 4 - 1) * 4), 4),
        ) };
    }

    if a.len() % 4 >= 2 {
        unsafe { signum_2(
            s(a.buffer.offset((a.len() as isize / 2 - 1) * 2), 2),
            sm(b.buffer.offset((a.len() as isize / 2 - 1) * 2), 2),
        ) };
    }

    if a.len() % 2 == 1 {
        b[a.len() - 1] = a[a.len() - 1].signum();
    }
}

pub fn signum_assign(a: &mut Vector) {
    use std::slice::from_raw_parts_mut as sm;

    if a.len() >= 16 {
        signum_assign_16s(a);
    }

    if a.len() % 16 >= 8 {
        unsafe { signum_assign_8(
            sm(a.buffer.offset((a.len() as isize / 8 - 1) * 8), 8),
        ) };
    }

    if a.len() % 8 >= 4 {
        unsafe { signum_assign_4(
            sm(a.buffer.offset((a.len() as isize / 4 - 1) * 4), 4),
        ) };
    }

    if a.len() % 4 >= 2 {
        unsafe { signum_assign_2(
            sm(a.buffer.offset((a.len() as isize / 2 - 1) * 2), 2),
        ) };
    }

    if a.len() % 2 == 1 {
        let last_index = a.len() - 1;
        a[last_index] = a[last_index].signum();
    }
}

#[cfg(feature = "with_simd")]
macro_rules! vandps {
    ($a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vandps $2, $1, $0" : "=x"($c) : "x"($a), "x"($b) ::) };
    )
}

#[cfg(feature = "with_simd")]
macro_rules! vorps {
    ($a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vorps $2, $1, $0" : "=x"($c) : "x"($a), "x"($b) ::) };
    )
}

#[cfg(feature = "with_simd")]
fn signum_16s(a: &[f32], b: &mut[f32]) {
    for i in 0..(a.len() / 16) {
        let a_ra = simd::x86::avx::f32x8::load(a, i * 16 + 0);
        let a_rb = simd::x86::avx::f32x8::load(a, i * 16 + 8);
        let sign = simd::x86::avx::f32x8::splat(-0.0);
        let one = simd::x86::avx::f32x8::splat(1.0);

        let mut b_ra: simd::x86::avx::f32x8; vandps!(sign, a_ra, b_ra); vorps!(one, b_ra, b_ra);
        let mut b_rb: simd::x86::avx::f32x8; vandps!(sign, a_rb, b_rb); vorps!(one, b_rb, b_rb);

        b_ra.store(b, i * 16 + 0);
        b_rb.store(b, i * 16 + 8);
    }
}

#[cfg(feature = "with_simd")]
fn signum_8(a: &[f32], b: &mut[f32]) {
    let a_r = simd::x86::avx::f32x8::load(a, 0);
    let sign = simd::x86::avx::f32x8::splat(-0.0);
    let one = simd::x86::avx::f32x8::splat(1.0);

    let mut b_r: simd::x86::avx::f32x8; vandps!(sign, a_r, b_r); vorps!(one, b_r, b_r);

    b_r.store(b, 0);
}

#[cfg(feature = "with_simd")]
fn signum_4(a: &[f32], b: &mut[f32]) {
    let a_r = simd::f32x4::load(a, 0);
    let sign = simd::f32x4::splat(-0.0);
    let one = simd::f32x4::splat(1.0);

    let mut b_r: simd::f32x4; vandps!(sign, a_r, b_r); vorps!(one, b_r, b_r);

    b_r.store(b, 0);
}

#[cfg(feature = "with_simd")]
fn signum_2(a: &[f32], b: &mut[f32]) {
    let a_r = simd::f32x4::new(a[0], a[1], 0.0, 0.0);
    let sign = simd::f32x4::splat(-0.0);
    let one = simd::f32x4::splat(1.0);

    let mut b_r: simd::f32x4; vandps!(sign, a_r, b_r); vorps!(one, b_r, b_r);

    b[0] = b_r.extract(0); b[1] = b_r.extract(1);
}

#[cfg(feature = "with_simd")]
fn signum_assign_16s(a: &mut [f32]) {
    for i in 0..(a.len() / 16) {
        let mut a_ra = simd::x86::avx::f32x8::load(a, i * 16 + 0);
        let mut a_rb = simd::x86::avx::f32x8::load(a, i * 16 + 8);
        let sign = simd::x86::avx::f32x8::splat(-0.0);
        let one = simd::x86::avx::f32x8::splat(1.0);

        vandps!(sign, a_ra, a_ra); vorps!(one, a_ra, a_ra);
        vandps!(sign, a_rb, a_rb); vorps!(one, a_rb, a_rb);

        a_ra.store(a, i * 16 + 0);
        a_rb.store(a, i * 16 + 8);
    }
}

#[cfg(feature = "with_simd")]
fn signum_assign_8(a: &mut [f32]) {
    let mut a_r = simd::x86::avx::f32x8::load(a, 0);
    let sign = simd::x86::avx::f32x8::splat(-0.0);
    let one = simd::x86::avx::f32x8::splat(1.0);

    vandps!(sign, a_r, a_r); vorps!(one, a_r, a_r);

    a_r.store(a, 0);
}

#[cfg(feature = "with_simd")]
fn signum_assign_4(a: &mut [f32]) {
    let mut a_r = simd::f32x4::load(a, 0);
    let sign = simd::f32x4::splat(-0.0);
    let one = simd::f32x4::splat(1.0);

    vandps!(sign, a_r, a_r); vorps!(one, a_r, a_r);

    a_r.store(a, 0);
}

#[cfg(feature = "with_simd")]
fn signum_assign_2(a: &mut [f32]) {
    let mut a_r = simd::f32x4::new(a[0], a[1], 0.0, 0.0);
    let sign = simd::f32x4::splat(-0.0);
    let one = simd::f32x4::splat(1.0);

    vandps!(sign, a_r, a_r); vorps!(one, a_r, a_r);

    a[0] = a_r.extract(0); a[1] = a_r.extract(1);
}

#[cfg(not(feature = "with_simd"))]
fn signum_16s(a: &[f32], b: &mut[f32]) {
    for i in 0..(a.len() / 16) {
        b[i * 16 +  0] = a[i * 16 +  0].signum();
        b[i * 16 +  1] = a[i * 16 +  1].signum();
        b[i * 16 +  2] = a[i * 16 +  2].signum();
        b[i * 16 +  3] = a[i * 16 +  3].signum();
        b[i * 16 +  4] = a[i * 16 +  4].signum();
        b[i * 16 +  5] = a[i * 16 +  5].signum();
        b[i * 16 +  6] = a[i * 16 +  6].signum();
        b[i * 16 +  7] = a[i * 16 +  7].signum();
        b[i * 16 +  8] = a[i * 16 +  8].signum();
        b[i * 16 +  9] = a[i * 16 +  9].signum();
        b[i * 16 + 10] = a[i * 16 + 10].signum();
        b[i * 16 + 11] = a[i * 16 + 11].signum();
        b[i * 16 + 12] = a[i * 16 + 12].signum();
        b[i * 16 + 13] = a[i * 16 + 13].signum();
        b[i * 16 + 14] = a[i * 16 + 14].signum();
        b[i * 16 + 15] = a[i * 16 + 15].signum();
    }
}

#[cfg(not(feature = "with_simd"))]
fn signum_8(a: &[f32], b: &mut[f32]) {
    b[0] = a[0].signum();
    b[1] = a[1].signum();
    b[2] = a[2].signum();
    b[3] = a[3].signum();
    b[4] = a[4].signum();
    b[5] = a[5].signum();
    b[6] = a[6].signum();
    b[7] = a[7].signum();
}

#[cfg(not(feature = "with_simd"))]
fn signum_4(a: &[f32], b: &mut[f32]) {
    b[0] = a[0].signum();
    b[1] = a[1].signum();
    b[2] = a[2].signum();
    b[3] = a[3].signum();
}

#[cfg(not(feature = "with_simd"))]
fn signum_2(a: &[f32], b: &mut[f32]) {
    b[0] = a[0].signum();
    b[1] = a[1].signum();
}

#[cfg(not(feature = "with_simd"))]
fn signum_assign_16s(a: &mut[f32]) {
    for i in 0..(a.len() / 16) {
        a[i * 16 +  0] = a[i * 16 +  0].signum();
        a[i * 16 +  1] = a[i * 16 +  1].signum();
        a[i * 16 +  2] = a[i * 16 +  2].signum();
        a[i * 16 +  3] = a[i * 16 +  3].signum();
        a[i * 16 +  4] = a[i * 16 +  4].signum();
        a[i * 16 +  5] = a[i * 16 +  5].signum();
        a[i * 16 +  6] = a[i * 16 +  6].signum();
        a[i * 16 +  7] = a[i * 16 +  7].signum();
        a[i * 16 +  8] = a[i * 16 +  8].signum();
        a[i * 16 +  9] = a[i * 16 +  9].signum();
        a[i * 16 + 10] = a[i * 16 + 10].signum();
        a[i * 16 + 11] = a[i * 16 + 11].signum();
        a[i * 16 + 12] = a[i * 16 + 12].signum();
        a[i * 16 + 13] = a[i * 16 + 13].signum();
        a[i * 16 + 14] = a[i * 16 + 14].signum();
        a[i * 16 + 15] = a[i * 16 + 15].signum();
    }
}

#[cfg(not(feature = "with_simd"))]
fn signum_assign_8(a: &mut[f32]) {
    a[0] = a[0].signum();
    a[1] = a[1].signum();
    a[2] = a[2].signum();
    a[3] = a[3].signum();
    a[4] = a[4].signum();
    a[5] = a[5].signum();
    a[6] = a[6].signum();
    a[7] = a[7].signum();
}

#[cfg(not(feature = "with_simd"))]
fn signum_assign_4(a: &mut[f32]) {
    a[0] = a[0].signum();
    a[1] = a[1].signum();
    a[2] = a[2].signum();
    a[3] = a[3].signum();
}

#[cfg(not(feature = "with_simd"))]
fn signum_assign_2(a: &mut[f32]) {
    a[0] = a[0].signum();
    a[1] = a[1].signum();
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_signum(bencher: &mut Bencher) {
        let mut a = Vector::zeros(1000);
        for (i, x) in a.iter_mut().enumerate() {
            *x = (1.0 - (i % 2) as f32 * 2.0) * (i % 16 + 1) as f32;
        }

        let mut b = Vector::zeros(1000);

        bencher.iter(|| {
            signum(&a, &mut b);
        });
    }

    #[bench]
    fn bench_signum_assign(bencher: &mut Bencher) {
        let mut a = Vector::zeros(1000);
        for (i, x) in a.iter_mut().enumerate() {
            *x = (1.0 - (i % 2) as f32 * 2.0) * (i % 16 + 1) as f32;
        }

        bencher.iter(|| {
            signum_assign(&mut a);
        });
    }
}
