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

use smallmath::Vector;

use activation_function::ActivationFunction;
use serialization::Identifiable;

#[derive(Clone, Debug)]
pub struct ReLuActivationFunction;

impl ActivationFunction for ReLuActivationFunction {
    fn f(x: f32) -> f32 {
        x.max(0.0)
    }

    fn f_prime(x: f32) -> f32 {
        x.signum().max(0.0)
    }

    fn f_vector(inputs: &Vector, outputs: &mut Vector) {
        relu(inputs, outputs);
    }

    fn f_prime_vector(inputs: &Vector, outputs: &mut Vector) {
        relu_prime(inputs, outputs);
    }
}

impl Identifiable for ReLuActivationFunction {
    fn identifier() -> String {
        String::from("ReLuActivationFunction")
    }

    fn get_identifier(&self) -> String {
        Self::identifier()
    }
}

fn relu(a: &Vector, b: &mut Vector) {
    debug_assert!(a.len() == b.len(), "Operand vectors are different lengths!");

    use std::slice::from_raw_parts as s;
    use std::slice::from_raw_parts_mut as sm;

    if a.len() >= 16 {
        relu_16s(a, b);
    }

    if a.len() % 16 >= 8 {
        unsafe { relu_8(
            s(a.as_ptr().offset((a.len() as isize / 8 - 1) * 8), 8),
            sm(b.as_mut_ptr().offset((a.len() as isize / 8 - 1) * 8), 8),
        ) };
    }

    if a.len() % 8 >= 4 {
        unsafe { relu_4(
            s(a.as_ptr().offset((a.len() as isize / 4 - 1) * 4), 4),
            sm(b.as_mut_ptr().offset((a.len() as isize / 4 - 1) * 4), 4),
        ) };
    }

    if a.len() % 4 >= 2 {
        unsafe { relu_2(
            s(a.as_ptr().offset((a.len() as isize / 2 - 1) * 2), 2),
            sm(b.as_mut_ptr().offset((a.len() as isize / 2 - 1) * 2), 2),
        ) };
    }

    if a.len() % 2 == 1 {
        b[a.len() - 1] = a[a.len() - 1].max(0.0);
    }
}

fn relu_prime(a: &Vector, b: &mut Vector) {
    debug_assert!(a.len() == b.len(), "Operand vectors are different lengths!");

    use std::slice::from_raw_parts as s;
    use std::slice::from_raw_parts_mut as sm;

    if a.len() >= 16 {
        relu_prime_16s(a, b);
    }

    if a.len() % 16 >= 8 {
        unsafe { relu_prime_8(
            s(a.as_ptr().offset((a.len() as isize / 8 - 1) * 8), 8),
            sm(b.as_mut_ptr().offset((a.len() as isize / 8 - 1) * 8), 8),
        ) };
    }

    if a.len() % 8 >= 4 {
        unsafe { relu_prime_4(
            s(a.as_ptr().offset((a.len() as isize / 4 - 1) * 4), 4),
            sm(b.as_mut_ptr().offset((a.len() as isize / 4 - 1) * 4), 4),
        ) };
    }

    if a.len() % 4 >= 2 {
        unsafe { relu_prime_2(
            s(a.as_ptr().offset((a.len() as isize / 2 - 1) * 2), 2),
            sm(b.as_mut_ptr().offset((a.len() as isize / 2 - 1) * 2), 2),
        ) };
    }

    if a.len() % 2 == 1 {
        b[a.len() - 1] = a[a.len() - 1].signum().max(0.0);
    }
}

#[cfg(feature = "with_simd")]
macro_rules! vandps {
    ($a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vandps $2, $1, $0" : "=x"($c) : "x"($a), "x"($b) ::) };
    )
}

#[cfg(feature = "with_simd")]
macro_rules! vmaxps {
    ($a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vmaxps $2, $1, $0" : "=x"($c) : "x"($a), "x"($b) ::) };
    )
}

#[cfg(feature = "with_simd")]
macro_rules! vorps {
    ($a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vorps $2, $1, $0" : "=x"($c) : "x"($a), "x"($b) ::) };
    )
}

#[cfg(feature = "with_simd")]
fn relu_16s(a: &[f32], b: &mut[f32]) {
    for i in 0..(a.len() / 16) {
        let a_ra = simd::x86::avx::f32x8::load(a, i * 16 + 0);
        let a_rb = simd::x86::avx::f32x8::load(a, i * 16 + 8);
        let zero = simd::x86::avx::f32x8::splat(0.0);

        let b_ra: simd::x86::avx::f32x8; vmaxps!(zero, a_ra, b_ra);
        let b_rb: simd::x86::avx::f32x8; vmaxps!(zero, a_rb, b_rb);

        b_ra.store(b, i * 16 + 0);
        b_rb.store(b, i * 16 + 8);
    }
}

#[cfg(feature = "with_simd")]
fn relu_8(a: &[f32], b: &mut[f32]) {
    let a_r = simd::x86::avx::f32x8::load(a, 0);
    let zero = simd::x86::avx::f32x8::splat(0.0);

    let b_r: simd::x86::avx::f32x8; vmaxps!(zero, a_r, b_r);

    b_r.store(b, 0);
}

#[cfg(feature = "with_simd")]
fn relu_4(a: &[f32], b: &mut[f32]) {
    let a_r = simd::f32x4::load(a, 0);
    let zero = simd::f32x4::splat(0.0);

    let b_r: simd::f32x4; vmaxps!(zero, a_r, b_r);

    b_r.store(b, 0);
}

#[cfg(feature = "with_simd")]
fn relu_2(a: &[f32], b: &mut[f32]) {
    let a_r = simd::f32x4::new(a[0], a[1], 0.0, 0.0);
    let zero = simd::f32x4::splat(0.0);

    let b_r: simd::f32x4; vmaxps!(zero, a_r, b_r);

    b[0] = b_r.extract(0); b[1] = b_r.extract(1);
}

#[cfg(feature = "with_simd")]
fn relu_prime_16s(a: &[f32], b: &mut[f32]) {
    for i in 0..(a.len() / 16) {
        let a_ra = simd::x86::avx::f32x8::load(a, i * 16 + 0);
        let a_rb = simd::x86::avx::f32x8::load(a, i * 16 + 8);
        let sign = simd::x86::avx::f32x8::splat(-0.0);
        let one = simd::x86::avx::f32x8::splat(1.0);
        let zero = simd::x86::avx::f32x8::splat(0.0);

        let mut b_ra: simd::x86::avx::f32x8; vandps!(sign, a_ra, b_ra); vorps!(one, b_ra, b_ra); vmaxps!(zero, b_ra, b_ra);
        let mut b_rb: simd::x86::avx::f32x8; vandps!(sign, a_rb, b_rb); vorps!(one, b_rb, b_rb); vmaxps!(zero, b_rb, b_rb);

        b_ra.store(b, i * 16 + 0);
        b_rb.store(b, i * 16 + 8);
    }
}

#[cfg(feature = "with_simd")]
fn relu_prime_8(a: &[f32], b: &mut[f32]) {
    let a_r = simd::x86::avx::f32x8::load(a, 0);
    let sign = simd::x86::avx::f32x8::splat(-0.0);
    let one = simd::x86::avx::f32x8::splat(1.0);
    let zero = simd::x86::avx::f32x8::splat(0.0);

    let mut b_r: simd::x86::avx::f32x8; vandps!(sign, a_r, b_r); vorps!(one, b_r, b_r); vmaxps!(zero, b_r, b_r);

    b_r.store(b, 0);
}

#[cfg(feature = "with_simd")]
fn relu_prime_4(a: &[f32], b: &mut[f32]) {
    let a_r = simd::f32x4::load(a, 0);
    let sign = simd::f32x4::splat(-0.0);
    let one = simd::f32x4::splat(1.0);
    let zero = simd::f32x4::splat(0.0);

    let mut b_r: simd::f32x4; vandps!(sign, a_r, b_r); vorps!(one, b_r, b_r); vmaxps!(zero, b_r, b_r);

    b_r.store(b, 0);
}

#[cfg(feature = "with_simd")]
fn relu_prime_2(a: &[f32], b: &mut[f32]) {
    let a_r = simd::f32x4::new(a[0], a[1], 0.0, 0.0);
    let sign = simd::f32x4::splat(-0.0);
    let one = simd::f32x4::splat(1.0);
    let zero = simd::f32x4::splat(0.0);


    let mut b_r: simd::f32x4; vandps!(sign, a_r, b_r); vorps!(one, b_r, b_r); vmaxps!(zero, b_r, b_r);

    b[0] = b_r.extract(0); b[1] = b_r.extract(1);
}

#[cfg(not(feature = "with_simd"))]
fn relu_16s(a: &[f32], b: &mut[f32]) {
    for i in 0..(a.len() / 16) {
        b[i * 16 +  0] = a[i * 16 +  0].max(0.0);
        b[i * 16 +  1] = a[i * 16 +  1].max(0.0);
        b[i * 16 +  2] = a[i * 16 +  2].max(0.0);
        b[i * 16 +  3] = a[i * 16 +  3].max(0.0);
        b[i * 16 +  4] = a[i * 16 +  4].max(0.0);
        b[i * 16 +  5] = a[i * 16 +  5].max(0.0);
        b[i * 16 +  6] = a[i * 16 +  6].max(0.0);
        b[i * 16 +  7] = a[i * 16 +  7].max(0.0);
        b[i * 16 +  8] = a[i * 16 +  8].max(0.0);
        b[i * 16 +  9] = a[i * 16 +  9].max(0.0);
        b[i * 16 + 10] = a[i * 16 + 10].max(0.0);
        b[i * 16 + 11] = a[i * 16 + 11].max(0.0);
        b[i * 16 + 12] = a[i * 16 + 12].max(0.0);
        b[i * 16 + 13] = a[i * 16 + 13].max(0.0);
        b[i * 16 + 14] = a[i * 16 + 14].max(0.0);
        b[i * 16 + 15] = a[i * 16 + 15].max(0.0);
    }
}

#[cfg(not(feature = "with_simd"))]
fn relu_8(a: &[f32], b: &mut[f32]) {
    b[0] = a[0].max(0.0);
    b[1] = a[1].max(0.0);
    b[2] = a[2].max(0.0);
    b[3] = a[3].max(0.0);
    b[4] = a[4].max(0.0);
    b[5] = a[5].max(0.0);
    b[6] = a[6].max(0.0);
    b[7] = a[7].max(0.0);
}

#[cfg(not(feature = "with_simd"))]
fn relu_4(a: &[f32], b: &mut[f32]) {
    b[0] = a[0].max(0.0);
    b[1] = a[1].max(0.0);
    b[2] = a[2].max(0.0);
    b[3] = a[3].max(0.0);
}

#[cfg(not(feature = "with_simd"))]
fn relu_2(a: &[f32], b: &mut[f32]) {
    b[0] = a[0].max(0.0);
    b[1] = a[1].max(0.0);
}

#[cfg(not(feature = "with_simd"))]
fn relu_prime_16s(a: &[f32], b: &mut[f32]) {
    for i in 0..(a.len() / 16) {
        b[i * 16 +  0] = a[i * 16 +  0].signum().max(0.0);
        b[i * 16 +  1] = a[i * 16 +  1].signum().max(0.0);
        b[i * 16 +  2] = a[i * 16 +  2].signum().max(0.0);
        b[i * 16 +  3] = a[i * 16 +  3].signum().max(0.0);
        b[i * 16 +  4] = a[i * 16 +  4].signum().max(0.0);
        b[i * 16 +  5] = a[i * 16 +  5].signum().max(0.0);
        b[i * 16 +  6] = a[i * 16 +  6].signum().max(0.0);
        b[i * 16 +  7] = a[i * 16 +  7].signum().max(0.0);
        b[i * 16 +  8] = a[i * 16 +  8].signum().max(0.0);
        b[i * 16 +  9] = a[i * 16 +  9].signum().max(0.0);
        b[i * 16 + 10] = a[i * 16 + 10].signum().max(0.0);
        b[i * 16 + 11] = a[i * 16 + 11].signum().max(0.0);
        b[i * 16 + 12] = a[i * 16 + 12].signum().max(0.0);
        b[i * 16 + 13] = a[i * 16 + 13].signum().max(0.0);
        b[i * 16 + 14] = a[i * 16 + 14].signum().max(0.0);
        b[i * 16 + 15] = a[i * 16 + 15].signum().max(0.0);
    }
}

#[cfg(not(feature = "with_simd"))]
fn relu_prime_8(a: &[f32], b: &mut[f32]) {
    b[0] = a[0].signum().max(0.0);
    b[1] = a[1].signum().max(0.0);
    b[2] = a[2].signum().max(0.0);
    b[3] = a[3].signum().max(0.0);
    b[4] = a[4].signum().max(0.0);
    b[5] = a[5].signum().max(0.0);
    b[6] = a[6].signum().max(0.0);
    b[7] = a[7].signum().max(0.0);
}

#[cfg(not(feature = "with_simd"))]
fn relu_prime_4(a: &[f32], b: &mut[f32]) {
    b[0] = a[0].signum().max(0.0);
    b[1] = a[1].signum().max(0.0);
    b[2] = a[2].signum().max(0.0);
    b[3] = a[3].signum().max(0.0);
}

#[cfg(not(feature = "with_simd"))]
fn relu_prime_2(a: &[f32], b: &mut[f32]) {
    b[0] = a[0].signum().max(0.0);
    b[1] = a[1].signum().max(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::{self, Bencher};

    #[bench]
    fn bench_relu(bencher: &mut Bencher) {
        bencher.iter(|| {
            ReLuActivationFunction::f(test::black_box(0.5))
        });
    }

    #[bench]
    fn bench_relu_prime(bencher: &mut Bencher) {
        bencher.iter(|| {
            ReLuActivationFunction::f_prime(test::black_box(0.5))
        });
    }

    #[bench]
    fn bench_relu_vector(bencher: &mut Bencher) {
        let a = Vector::from_vec(vec![0.5; 1000]);
        let mut b = Vector::zeros(1000);

        bencher.iter(|| {
            ReLuActivationFunction::f_vector(&a, &mut b);
        });
    }

    #[bench]
    fn bench_relu_prime_vector(bencher: &mut Bencher) {
        let a = Vector::from_vec(vec![0.5; 1000]);
        let mut b = Vector::zeros(1000);

        bencher.iter(|| {
            ReLuActivationFunction::f_prime_vector(&a, &mut b);
        });
    }
}
