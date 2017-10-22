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
#[cfg(feature = "with_simd")]
use simd::x86::avx::LowHigh128;

use matrix::Matrix;

pub fn transpose(a: &Matrix, b: &mut Matrix) {
    debug_assert!(a.rows == b.columns && a.columns == b.rows, "Invalid matrix dimensions for transpose!");

    use std::slice::from_raw_parts as s;
    use std::slice::from_raw_parts_mut as sm;

    for a_row_chunk in 0..(a.rows / 8) as isize {
        for a_column_chunk in 0..(a.columns / 8) as isize {
            unsafe { transpose_8x8(
                s(a.buffer.offset(a_row_chunk * 8 * a.columns as isize + a_column_chunk * 8), 7 * a.columns + 8), a.columns,
                sm(b.buffer.offset(a_column_chunk * 8 * a.rows as isize + a_row_chunk * 8), 7 * a.rows + 8), a.rows,
            ) };
        }

        if a.columns % 8 >= 4 {
            unsafe { transpose_8x4(
                s(a.buffer.offset(a_row_chunk * 8 * a.columns as isize + (a.columns as isize / 4 - 1) * 4), 7 * a.columns + 4), a.columns,
                sm(b.buffer.offset((a.columns as isize / 4 - 1) * 4 * a.rows as isize + a_row_chunk * 8), 3 * a.rows + 8), a.rows,
            ) };
        }

        if a.columns % 4 >= 2 {
            unsafe { transpose_8x2(
                s(a.buffer.offset(a_row_chunk * 8 * a.columns as isize + (a.columns as isize / 2 - 1) * 2), 7 * a.columns + 2), a.columns,
                sm(b.buffer.offset((a.columns as isize / 2 - 1) * 2 * a.rows as isize + a_row_chunk * 8), a.rows + 8), a.rows,
            ) };
        }

        if a.columns % 2 == 1 {
            unsafe { transpose_8x1(
                s(a.buffer.offset(a_row_chunk * 8 * a.columns as isize + a.columns as isize - 1), 7 * a.columns + 1), a.columns,
                sm(b.buffer.offset((a.columns as isize - 1) * a.rows as isize + a_row_chunk * 8), 8),
            ) };
        }
    }

    if a.rows % 8 >= 4 {
        for a_column_chunk in 0..(a.columns / 8) as isize {
            unsafe { transpose_4x8(
                s(a.buffer.offset((a.rows as isize / 4 - 1) * 4 * a.columns as isize + a_column_chunk * 8), 3 * a.columns + 8), a.columns,
                sm(b.buffer.offset(a_column_chunk * 8 * a.rows as isize + (a.rows as isize / 4 - 1) * 4), 7 * a.rows + 4), a.rows,
            ) };
        }

        if a.columns % 8 >= 4 {
            unsafe { transpose_4x4(
                s(a.buffer.offset((a.rows as isize / 4 - 1) * 4 * a.columns as isize + (a.columns as isize / 4 - 1) * 4), 3 * a.columns + 4), a.columns,
                sm(b.buffer.offset((a.columns as isize / 4 - 1) * 4 * a.rows as isize + (a.rows as isize / 4 - 1) * 4), 3 * a.rows + 4), a.rows,
            ) };
        }

        if a.columns % 4 >= 2 {
            unsafe { transpose_4x2(
                s(a.buffer.offset((a.rows as isize / 4 - 1) * 4 * a.columns as isize + (a.columns as isize / 2 - 1) * 2), 3 * a.columns + 2), a.columns,
                sm(b.buffer.offset((a.columns as isize / 2 - 1) * 2 * a.rows as isize + (a.rows as isize / 4 - 1) * 4), a.rows + 4), a.rows,
            ) };
        }

        if a.columns % 2 == 1 {
            unsafe { transpose_4x1(
                s(a.buffer.offset((a.rows as isize / 4 - 1) * 4 * a.columns as isize + a.columns as isize - 1), 3 * a.columns + 1), a.columns,
                sm(b.buffer.offset((a.columns as isize - 1) * a.rows as isize + (a.rows as isize / 4 - 1) * 4), 4),
            ) };
        }
    }

    if a.rows % 4 >= 2 {
        for a_column_chunk in 0..(a.columns / 8) as isize {
            unsafe { transpose_2x8(
                s(a.buffer.offset((a.rows as isize / 2 - 1) * 2 * a.columns as isize + a_column_chunk * 8), a.columns + 8), a.columns,
                sm(b.buffer.offset(a_column_chunk * 8 * a.rows as isize + (a.rows as isize / 2 - 1) * 2), 7 * a.rows + 2), a.rows,
            ) };
        }

        if a.columns % 8 >= 4 {
            unsafe { transpose_2x4(
                s(a.buffer.offset((a.rows as isize / 2 - 1) * 2 * a.columns as isize + (a.columns as isize / 4 - 1) * 4), a.columns + 4), a.columns,
                sm(b.buffer.offset((a.columns as isize / 4 - 1) * 4 * a.rows as isize + (a.rows as isize / 2 - 1) * 2), 3 * a.rows + 2), a.rows,
            ) };
        }

        if a.columns % 4 >= 2 {
            unsafe { transpose_2x2(
                s(a.buffer.offset((a.rows as isize / 2 - 1) * 2 * a.columns as isize + (a.columns as isize / 2 - 1) * 2), a.columns + 2), a.columns,
                sm(b.buffer.offset((a.columns as isize / 2 - 1) * 2 * a.rows as isize + (a.rows as isize / 2 - 1) * 2), a.rows + 2), a.rows,
            ) };
        }

        if a.columns % 2 == 1 {
            unsafe { transpose_2x1(
                s(a.buffer.offset((a.rows as isize / 2 - 1) * 2 * a.columns as isize + a.columns as isize - 1), a.columns + 1), a.columns,
                sm(b.buffer.offset((a.columns as isize - 1) * a.rows as isize + (a.rows as isize / 2 - 1) * 2), 2),
            ) };
        }
    }

    if a.rows % 2 == 1 {
        for a_column_chunk in 0..(a.columns / 8) as isize {
            unsafe { transpose_1x8(
                s(a.buffer.offset((a.rows as isize - 1) * a.columns as isize + a_column_chunk * 8), 8),
                sm(b.buffer.offset(a_column_chunk * 8 * a.rows as isize + a.rows as isize - 1), 7 * a.rows + 1), a.rows,
            ) };
        }

        if a.columns % 8 >= 4 {
            unsafe { transpose_1x4(
                s(a.buffer.offset((a.rows as isize - 1) * a.columns as isize + (a.columns as isize / 4 - 1) * 4), 4),
                sm(b.buffer.offset((a.columns as isize / 4 - 1) * 4 * a.rows as isize + a.rows as isize - 1), 3 * a.rows + 1), a.rows,
            ) };
        }

        if a.columns % 4 >= 2 {
            unsafe { transpose_1x2(
                s(a.buffer.offset((a.rows as isize - 1) * a.columns as isize + (a.columns as isize / 2 - 1) * 2), 2),
                sm(b.buffer.offset((a.columns as isize / 2 - 1) * 2 * a.rows as isize + a.rows as isize - 1), a.rows + 1), a.rows,
            ) };
        }

        if a.columns % 2 == 1 {
            b.as_vector_mut()[a.len - 1] = a.as_vector()[a.len - 1];
        }
    }
}

#[cfg(feature = "with_simd")]
macro_rules! vblendps {
    ($i:expr, $a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vblendps $3, $2, $1, $0" : "=x"($c) : "x"($a), "x"($b), "N"($i) ::) };
    )
}

#[cfg(feature = "with_simd")]
macro_rules! vperm2f128 {
    ($i:expr, $a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vperm2f128 $3, $2, $1, $0" : "=x"($c) : "x"($a), "x"($b), "N"($i) ::) };
    )
}

#[cfg(feature = "with_simd")]
macro_rules! vshufps {
    ($i:expr, $a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vshufps $3, $2, $1, $0" : "=x"($c) : "x"($a), "x"($b), "N"($i) ::) };
    )
}

#[cfg(feature = "with_simd")]
macro_rules! vunpcklps {
    ($a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vunpcklps $2, $1, $0" : "=x"($c) : "x"($a), "x"($b) ::) };
    )
}

#[cfg(feature = "with_simd")]
macro_rules! vunpckhps {
    ($a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vunpckhps $2, $1, $0" : "=x"($c) : "x"($a), "x"($b) ::) };
    )
}

#[cfg(feature = "with_simd")]
fn transpose_8x8(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    let mut r0 = simd::x86::avx::f32x8::load(&a, 0 * a_dim);
    let mut r1 = simd::x86::avx::f32x8::load(&a, 1 * a_dim);
    let mut r2 = simd::x86::avx::f32x8::load(&a, 2 * a_dim);
    let mut r3 = simd::x86::avx::f32x8::load(&a, 3 * a_dim);
    let mut r4 = simd::x86::avx::f32x8::load(&a, 4 * a_dim);
    let mut r5 = simd::x86::avx::f32x8::load(&a, 5 * a_dim);
    let mut r6 = simd::x86::avx::f32x8::load(&a, 6 * a_dim);
    let mut r7 = simd::x86::avx::f32x8::load(&a, 7 * a_dim);

    let mut t0: simd::x86::avx::f32x8; vunpcklps!(r0, r1, t0);
    let mut t1: simd::x86::avx::f32x8; vunpckhps!(r0, r1, t1);
    let mut t2: simd::x86::avx::f32x8; vunpcklps!(r2, r3, t2);
    let mut t3: simd::x86::avx::f32x8; vunpckhps!(r2, r3, t3);
    let mut t4: simd::x86::avx::f32x8; vunpcklps!(r4, r5, t4);
    let mut t5: simd::x86::avx::f32x8; vunpckhps!(r4, r5, t5);
    let mut t6: simd::x86::avx::f32x8; vunpcklps!(r6, r7, t6);
    let mut t7: simd::x86::avx::f32x8; vunpckhps!(r6, r7, t7);

    let mut v: simd::x86::avx::f32x8;

    vshufps!(0x4E, t0, t2, v);
    vblendps!(0xCC, t0, v, r0);
    vblendps!(0x33, t2, v, r1);

    vshufps!(0x4E, t1, t3, v);
    vblendps!(0xCC, t1, v, r2);
    vblendps!(0x33, t3, v, r3);

    vshufps!(0x4E, t4, t6, v);
    vblendps!(0xCC, t4, v, r4);
    vblendps!(0x33, t6, v, r5);

    vshufps!(0x4E, t5, t7, v);
    vblendps!(0xCC, t5, v, r6);
    vblendps!(0x33, t7, v, r7);

    vperm2f128!(0x20, r0, r4, t0);
    vperm2f128!(0x20, r1, r5, t1);
    vperm2f128!(0x20, r2, r6, t2);
    vperm2f128!(0x20, r3, r7, t3);
    vperm2f128!(0x31, r0, r4, t4);
    vperm2f128!(0x31, r1, r5, t5);
    vperm2f128!(0x31, r2, r6, t6);
    vperm2f128!(0x31, r3, r7, t7);

    t0.store(b, 0 * b_dim);
    t1.store(b, 1 * b_dim);
    t2.store(b, 2 * b_dim);
    t3.store(b, 3 * b_dim);
    t4.store(b, 4 * b_dim);
    t5.store(b, 5 * b_dim);
    t6.store(b, 6 * b_dim);
    t7.store(b, 7 * b_dim);
}

#[cfg(feature = "with_simd")]
fn transpose_8x4(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    let mut r0 = simd::f32x4::load(&a, 0 * a_dim);
    let mut r1 = simd::f32x4::load(&a, 1 * a_dim);
    let mut r2 = simd::f32x4::load(&a, 2 * a_dim);
    let mut r3 = simd::f32x4::load(&a, 3 * a_dim);
    let mut r4 = simd::f32x4::load(&a, 4 * a_dim);
    let mut r5 = simd::f32x4::load(&a, 5 * a_dim);
    let mut r6 = simd::f32x4::load(&a, 6 * a_dim);
    let mut r7 = simd::f32x4::load(&a, 7 * a_dim);

    let t0: simd::f32x4; vunpcklps!(r0, r1, t0);
    let t1: simd::f32x4; vunpckhps!(r0, r1, t1);
    let t2: simd::f32x4; vunpcklps!(r2, r3, t2);
    let t3: simd::f32x4; vunpckhps!(r2, r3, t3);
    let t4: simd::f32x4; vunpcklps!(r4, r5, t4);
    let t5: simd::f32x4; vunpckhps!(r4, r5, t5);
    let t6: simd::f32x4; vunpcklps!(r6, r7, t6);
    let t7: simd::f32x4; vunpckhps!(r6, r7, t7);

    let mut v: simd::f32x4;

    vshufps!(0x4E, t0, t2, v);
    vblendps!(0xCC, t0, v, r0);
    vblendps!(0x33, t2, v, r1);

    vshufps!(0x4E, t1, t3, v);
    vblendps!(0xCC, t1, v, r2);
    vblendps!(0x33, t3, v, r3);

    vshufps!(0x4E, t4, t6, v);
    vblendps!(0xCC, t4, v, r4);
    vblendps!(0x33, t6, v, r5);

    vshufps!(0x4E, t5, t7, v);
    vblendps!(0xCC, t5, v, r6);
    vblendps!(0x33, t7, v, r7);

    r0.store(b, 0 * b_dim + 0);
    r1.store(b, 1 * b_dim + 0);
    r2.store(b, 2 * b_dim + 0);
    r3.store(b, 3 * b_dim + 0);
    r4.store(b, 0 * b_dim + 4);
    r5.store(b, 1 * b_dim + 4);
    r6.store(b, 2 * b_dim + 4);
    r7.store(b, 3 * b_dim + 4);
}

#[cfg(feature = "with_simd")]
fn transpose_4x8(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    let mut r0 = simd::x86::avx::f32x8::load(&a, 0 * a_dim);
    let mut r1 = simd::x86::avx::f32x8::load(&a, 1 * a_dim);
    let mut r2 = simd::x86::avx::f32x8::load(&a, 2 * a_dim);
    let mut r3 = simd::x86::avx::f32x8::load(&a, 3 * a_dim);

    let t0: simd::x86::avx::f32x8; vunpcklps!(r0, r1, t0);
    let t1: simd::x86::avx::f32x8; vunpckhps!(r0, r1, t1);
    let t2: simd::x86::avx::f32x8; vunpcklps!(r2, r3, t2);
    let t3: simd::x86::avx::f32x8; vunpckhps!(r2, r3, t3);

    let mut v: simd::x86::avx::f32x8;

    vshufps!(0x4E, t0, t2, v);
    vblendps!(0xCC, t0, v, r0);
    vblendps!(0x33, t2, v, r1);

    vshufps!(0x4E, t1, t3, v);
    vblendps!(0xCC, t1, v, r2);
    vblendps!(0x33, t3, v, r3);

    r0.low().store(b, 0 * b_dim);
    r1.low().store(b, 1 * b_dim);
    r2.low().store(b, 2 * b_dim);
    r3.low().store(b, 3 * b_dim);
    r0.high().store(b, 4 * b_dim);
    r1.high().store(b, 5 * b_dim);
    r2.high().store(b, 6 * b_dim);
    r3.high().store(b, 7 * b_dim);
}

#[cfg(feature = "with_simd")]
fn transpose_4x4(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    let mut r0 = simd::f32x4::load(&a, 0 * a_dim);
    let mut r1 = simd::f32x4::load(&a, 1 * a_dim);
    let mut r2 = simd::f32x4::load(&a, 2 * a_dim);
    let mut r3 = simd::f32x4::load(&a, 3 * a_dim);

    let t0: simd::f32x4; vunpcklps!(r0, r1, t0);
    let t1: simd::f32x4; vunpckhps!(r0, r1, t1);
    let t2: simd::f32x4; vunpcklps!(r2, r3, t2);
    let t3: simd::f32x4; vunpckhps!(r2, r3, t3);

    let mut v: simd::f32x4;

    vshufps!(0x4E, t0, t2, v);
    vblendps!(0xCC, t0, v, r0);
    vblendps!(0x33, t2, v, r1);

    vshufps!(0x4E, t1, t3, v);
    vblendps!(0xCC, t1, v, r2);
    vblendps!(0x33, t3, v, r3);

    r0.store(b, 0 * b_dim);
    r1.store(b, 1 * b_dim);
    r2.store(b, 2 * b_dim);
    r3.store(b, 3 * b_dim);
}

#[cfg(not(feature = "with_simd"))]
fn transpose_8x8(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    b[0 * b_dim + 0] = a[0 * a_dim + 0]; b[1 * b_dim + 0] = a[0 * a_dim + 1]; b[2 * b_dim + 0] = a[0 * a_dim + 2]; b[3 * b_dim + 0] = a[0 * a_dim + 3]; b[4 * b_dim + 0] = a[0 * a_dim + 4]; b[5 * b_dim + 0] = a[0 * a_dim + 5]; b[6 * b_dim + 0] = a[0 * a_dim + 6]; b[7 * b_dim + 0] = a[0 * a_dim + 7];
    b[0 * b_dim + 1] = a[1 * a_dim + 0]; b[1 * b_dim + 1] = a[1 * a_dim + 1]; b[2 * b_dim + 1] = a[1 * a_dim + 2]; b[3 * b_dim + 1] = a[1 * a_dim + 3]; b[4 * b_dim + 1] = a[1 * a_dim + 4]; b[5 * b_dim + 1] = a[1 * a_dim + 5]; b[6 * b_dim + 1] = a[1 * a_dim + 6]; b[7 * b_dim + 1] = a[1 * a_dim + 7];
    b[0 * b_dim + 2] = a[2 * a_dim + 0]; b[1 * b_dim + 2] = a[2 * a_dim + 1]; b[2 * b_dim + 2] = a[2 * a_dim + 2]; b[3 * b_dim + 2] = a[2 * a_dim + 3]; b[4 * b_dim + 2] = a[2 * a_dim + 4]; b[5 * b_dim + 2] = a[2 * a_dim + 5]; b[6 * b_dim + 2] = a[2 * a_dim + 6]; b[7 * b_dim + 2] = a[2 * a_dim + 7];
    b[0 * b_dim + 3] = a[3 * a_dim + 0]; b[1 * b_dim + 3] = a[3 * a_dim + 1]; b[2 * b_dim + 3] = a[3 * a_dim + 2]; b[3 * b_dim + 3] = a[3 * a_dim + 3]; b[4 * b_dim + 3] = a[3 * a_dim + 4]; b[5 * b_dim + 3] = a[3 * a_dim + 5]; b[6 * b_dim + 3] = a[3 * a_dim + 6]; b[7 * b_dim + 3] = a[3 * a_dim + 7];
    b[0 * b_dim + 4] = a[4 * a_dim + 0]; b[1 * b_dim + 4] = a[4 * a_dim + 1]; b[2 * b_dim + 4] = a[4 * a_dim + 2]; b[3 * b_dim + 4] = a[4 * a_dim + 3]; b[4 * b_dim + 4] = a[4 * a_dim + 4]; b[5 * b_dim + 4] = a[4 * a_dim + 5]; b[6 * b_dim + 4] = a[4 * a_dim + 6]; b[7 * b_dim + 4] = a[4 * a_dim + 7];
    b[0 * b_dim + 5] = a[5 * a_dim + 0]; b[1 * b_dim + 5] = a[5 * a_dim + 1]; b[2 * b_dim + 5] = a[5 * a_dim + 2]; b[3 * b_dim + 5] = a[5 * a_dim + 3]; b[4 * b_dim + 5] = a[5 * a_dim + 4]; b[5 * b_dim + 5] = a[5 * a_dim + 5]; b[6 * b_dim + 5] = a[5 * a_dim + 6]; b[7 * b_dim + 5] = a[5 * a_dim + 7];
    b[0 * b_dim + 6] = a[6 * a_dim + 0]; b[1 * b_dim + 6] = a[6 * a_dim + 1]; b[2 * b_dim + 6] = a[6 * a_dim + 2]; b[3 * b_dim + 6] = a[6 * a_dim + 3]; b[4 * b_dim + 6] = a[6 * a_dim + 4]; b[5 * b_dim + 6] = a[6 * a_dim + 5]; b[6 * b_dim + 6] = a[6 * a_dim + 6]; b[7 * b_dim + 6] = a[6 * a_dim + 7];
    b[0 * b_dim + 7] = a[7 * a_dim + 0]; b[1 * b_dim + 7] = a[7 * a_dim + 1]; b[2 * b_dim + 7] = a[7 * a_dim + 2]; b[3 * b_dim + 7] = a[7 * a_dim + 3]; b[4 * b_dim + 7] = a[7 * a_dim + 4]; b[5 * b_dim + 7] = a[7 * a_dim + 5]; b[6 * b_dim + 7] = a[7 * a_dim + 6]; b[7 * b_dim + 7] = a[7 * a_dim + 7];
}

#[cfg(not(feature = "with_simd"))]
fn transpose_8x4(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    b[0 * b_dim + 0] = a[0 * a_dim + 0]; b[1 * b_dim + 0] = a[0 * a_dim + 1]; b[2 * b_dim + 0] = a[0 * a_dim + 2]; b[3 * b_dim + 0] = a[0 * a_dim + 3];
    b[0 * b_dim + 1] = a[1 * a_dim + 0]; b[1 * b_dim + 1] = a[1 * a_dim + 1]; b[2 * b_dim + 1] = a[1 * a_dim + 2]; b[3 * b_dim + 1] = a[1 * a_dim + 3];
    b[0 * b_dim + 2] = a[2 * a_dim + 0]; b[1 * b_dim + 2] = a[2 * a_dim + 1]; b[2 * b_dim + 2] = a[2 * a_dim + 2]; b[3 * b_dim + 2] = a[2 * a_dim + 3];
    b[0 * b_dim + 3] = a[3 * a_dim + 0]; b[1 * b_dim + 3] = a[3 * a_dim + 1]; b[2 * b_dim + 3] = a[3 * a_dim + 2]; b[3 * b_dim + 3] = a[3 * a_dim + 3];
    b[0 * b_dim + 4] = a[4 * a_dim + 0]; b[1 * b_dim + 4] = a[4 * a_dim + 1]; b[2 * b_dim + 4] = a[4 * a_dim + 2]; b[3 * b_dim + 4] = a[4 * a_dim + 3];
    b[0 * b_dim + 5] = a[5 * a_dim + 0]; b[1 * b_dim + 5] = a[5 * a_dim + 1]; b[2 * b_dim + 5] = a[5 * a_dim + 2]; b[3 * b_dim + 5] = a[5 * a_dim + 3];
    b[0 * b_dim + 6] = a[6 * a_dim + 0]; b[1 * b_dim + 6] = a[6 * a_dim + 1]; b[2 * b_dim + 6] = a[6 * a_dim + 2]; b[3 * b_dim + 6] = a[6 * a_dim + 3];
    b[0 * b_dim + 7] = a[7 * a_dim + 0]; b[1 * b_dim + 7] = a[7 * a_dim + 1]; b[2 * b_dim + 7] = a[7 * a_dim + 2]; b[3 * b_dim + 7] = a[7 * a_dim + 3];
}

fn transpose_8x2(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    b[0 * b_dim + 0] = a[0 * a_dim + 0]; b[1 * b_dim + 0] = a[0 * a_dim + 1];
    b[0 * b_dim + 1] = a[1 * a_dim + 0]; b[1 * b_dim + 1] = a[1 * a_dim + 1];
    b[0 * b_dim + 2] = a[2 * a_dim + 0]; b[1 * b_dim + 2] = a[2 * a_dim + 1];
    b[0 * b_dim + 3] = a[3 * a_dim + 0]; b[1 * b_dim + 3] = a[3 * a_dim + 1];
    b[0 * b_dim + 4] = a[4 * a_dim + 0]; b[1 * b_dim + 4] = a[4 * a_dim + 1];
    b[0 * b_dim + 5] = a[5 * a_dim + 0]; b[1 * b_dim + 5] = a[5 * a_dim + 1];
    b[0 * b_dim + 6] = a[6 * a_dim + 0]; b[1 * b_dim + 6] = a[6 * a_dim + 1];
    b[0 * b_dim + 7] = a[7 * a_dim + 0]; b[1 * b_dim + 7] = a[7 * a_dim + 1];
}

fn transpose_8x1(a: &[f32], a_dim: usize, b: &mut [f32]) {
    b[0] = a[0 * a_dim];
    b[1] = a[1 * a_dim];
    b[2] = a[2 * a_dim];
    b[3] = a[3 * a_dim];
    b[4] = a[4 * a_dim];
    b[5] = a[5 * a_dim];
    b[6] = a[6 * a_dim];
    b[7] = a[7 * a_dim];
}

#[cfg(not(feature = "with_simd"))]
fn transpose_4x8(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    b[0 * b_dim + 0] = a[0 * a_dim + 0]; b[1 * b_dim + 0] = a[0 * a_dim + 1]; b[2 * b_dim + 0] = a[0 * a_dim + 2]; b[3 * b_dim + 0] = a[0 * a_dim + 3]; b[4 * b_dim + 0] = a[0 * a_dim + 4]; b[5 * b_dim + 0] = a[0 * a_dim + 5]; b[6 * b_dim + 0] = a[0 * a_dim + 6]; b[7 * b_dim + 0] = a[0 * a_dim + 7];
    b[0 * b_dim + 1] = a[1 * a_dim + 0]; b[1 * b_dim + 1] = a[1 * a_dim + 1]; b[2 * b_dim + 1] = a[1 * a_dim + 2]; b[3 * b_dim + 1] = a[1 * a_dim + 3]; b[4 * b_dim + 1] = a[1 * a_dim + 4]; b[5 * b_dim + 1] = a[1 * a_dim + 5]; b[6 * b_dim + 1] = a[1 * a_dim + 6]; b[7 * b_dim + 1] = a[1 * a_dim + 7];
    b[0 * b_dim + 2] = a[2 * a_dim + 0]; b[1 * b_dim + 2] = a[2 * a_dim + 1]; b[2 * b_dim + 2] = a[2 * a_dim + 2]; b[3 * b_dim + 2] = a[2 * a_dim + 3]; b[4 * b_dim + 2] = a[2 * a_dim + 4]; b[5 * b_dim + 2] = a[2 * a_dim + 5]; b[6 * b_dim + 2] = a[2 * a_dim + 6]; b[7 * b_dim + 2] = a[2 * a_dim + 7];
    b[0 * b_dim + 3] = a[3 * a_dim + 0]; b[1 * b_dim + 3] = a[3 * a_dim + 1]; b[2 * b_dim + 3] = a[3 * a_dim + 2]; b[3 * b_dim + 3] = a[3 * a_dim + 3]; b[4 * b_dim + 3] = a[3 * a_dim + 4]; b[5 * b_dim + 3] = a[3 * a_dim + 5]; b[6 * b_dim + 3] = a[3 * a_dim + 6]; b[7 * b_dim + 3] = a[3 * a_dim + 7];
}

#[cfg(not(feature = "with_simd"))]
fn transpose_4x4(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    b[0 * b_dim + 0] = a[0 * a_dim + 0]; b[1 * b_dim + 0] = a[0 * a_dim + 1]; b[2 * b_dim + 0] = a[0 * a_dim + 2]; b[3 * b_dim + 0] = a[0 * a_dim + 3];
    b[0 * b_dim + 1] = a[1 * a_dim + 0]; b[1 * b_dim + 1] = a[1 * a_dim + 1]; b[2 * b_dim + 1] = a[1 * a_dim + 2]; b[3 * b_dim + 1] = a[1 * a_dim + 3];
    b[0 * b_dim + 2] = a[2 * a_dim + 0]; b[1 * b_dim + 2] = a[2 * a_dim + 1]; b[2 * b_dim + 2] = a[2 * a_dim + 2]; b[3 * b_dim + 2] = a[2 * a_dim + 3];
    b[0 * b_dim + 3] = a[3 * a_dim + 0]; b[1 * b_dim + 3] = a[3 * a_dim + 1]; b[2 * b_dim + 3] = a[3 * a_dim + 2]; b[3 * b_dim + 3] = a[3 * a_dim + 3];
}

fn transpose_4x2(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    b[0 * b_dim + 0] = a[0 * a_dim + 0]; b[1 * b_dim + 0] = a[0 * a_dim + 1];
    b[0 * b_dim + 1] = a[1 * a_dim + 0]; b[1 * b_dim + 1] = a[1 * a_dim + 1];
    b[0 * b_dim + 2] = a[2 * a_dim + 0]; b[1 * b_dim + 2] = a[2 * a_dim + 1];
    b[0 * b_dim + 3] = a[3 * a_dim + 0]; b[1 * b_dim + 3] = a[3 * a_dim + 1];
}

fn transpose_4x1(a: &[f32], a_dim: usize, b: &mut [f32]) {
    b[0] = a[0 * a_dim];
    b[1] = a[1 * a_dim];
    b[2] = a[2 * a_dim];
    b[3] = a[3 * a_dim];
}

fn transpose_2x8(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    b[0 * b_dim + 0] = a[0 * a_dim + 0]; b[0 * b_dim + 1] = a[1 * a_dim + 0];
    b[1 * b_dim + 0] = a[0 * a_dim + 1]; b[1 * b_dim + 1] = a[1 * a_dim + 1];
    b[2 * b_dim + 0] = a[0 * a_dim + 2]; b[2 * b_dim + 1] = a[1 * a_dim + 2];
    b[3 * b_dim + 0] = a[0 * a_dim + 3]; b[3 * b_dim + 1] = a[1 * a_dim + 3];
    b[4 * b_dim + 0] = a[0 * a_dim + 4]; b[4 * b_dim + 1] = a[1 * a_dim + 4];
    b[5 * b_dim + 0] = a[0 * a_dim + 5]; b[5 * b_dim + 1] = a[1 * a_dim + 5];
    b[6 * b_dim + 0] = a[0 * a_dim + 6]; b[6 * b_dim + 1] = a[1 * a_dim + 6];
    b[7 * b_dim + 0] = a[0 * a_dim + 7]; b[7 * b_dim + 1] = a[1 * a_dim + 7];
}

fn transpose_2x4(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    b[0 * b_dim + 0] = a[0 * a_dim + 0]; b[0 * b_dim + 1] = a[1 * a_dim + 0];
    b[1 * b_dim + 0] = a[0 * a_dim + 1]; b[1 * b_dim + 1] = a[1 * a_dim + 1];
    b[2 * b_dim + 0] = a[0 * a_dim + 2]; b[2 * b_dim + 1] = a[1 * a_dim + 2];
    b[3 * b_dim + 0] = a[0 * a_dim + 3]; b[3 * b_dim + 1] = a[1 * a_dim + 3];
}

fn transpose_2x2(a: &[f32], a_dim: usize, b: &mut [f32], b_dim: usize) {
    b[0 * b_dim + 0] = a[0 * a_dim + 0]; b[0 * b_dim + 1] = a[1 * a_dim + 0];
    b[1 * b_dim + 0] = a[0 * a_dim + 1]; b[1 * b_dim + 1] = a[1 * a_dim + 1];
}

fn transpose_2x1(a: &[f32], a_dim: usize, b: &mut [f32]) {
    b[0] = a[0 * a_dim]; b[1] = a[1 * a_dim];
}

fn transpose_1x8(a: &[f32], b: &mut [f32], b_dim: usize) {
    b[0 * b_dim] = a[0];
    b[1 * b_dim] = a[1];
    b[2 * b_dim] = a[2];
    b[3 * b_dim] = a[3];
    b[4 * b_dim] = a[4];
    b[5 * b_dim] = a[5];
    b[6 * b_dim] = a[6];
    b[7 * b_dim] = a[7];
}

fn transpose_1x4(a: &[f32], b: &mut [f32], b_dim: usize) {
    b[0 * b_dim] = a[0];
    b[1 * b_dim] = a[1];
    b[2 * b_dim] = a[2];
    b[3 * b_dim] = a[3];
}

fn transpose_1x2(a: &[f32], b: &mut [f32], b_dim: usize) {
    b[0 * b_dim] = a[0];
    b[1 * b_dim] = a[1];
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_transpose(bencher: &mut Bencher) {
        let (r, c) = (264, 100);

        let mut a = Matrix::zeros(r, c);
        for (i, x) in a.iter_mut().enumerate() {
            *x = (i % 16 + 1) as f32;
        }

        let mut b = Matrix::zeros(c, r);

        bencher.iter(|| {
            transpose(&a, &mut b);
        });
    }
}
