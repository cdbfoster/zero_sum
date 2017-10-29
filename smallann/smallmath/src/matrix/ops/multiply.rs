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

use matrix::Matrix;

#[cfg(feature = "with_blas")]
pub fn multiply(a: &Matrix, b: &Matrix, c: &mut Matrix) {
    debug_assert!(a.columns == b.rows, "Invalid matrix dimensions for multiplication!");
    debug_assert!(a.rows == c.rows && b.columns == c.columns, "Invalid destination matrix dimensions for multiplication!");

    let (m, k, n) = (a.rows as i32, b.rows as i32, b.columns as i32);
    unsafe { blas::c::sgemm(
        blas::c::Layout::RowMajor, blas::c::Transpose::None, blas::c::Transpose::None,
        m, n, k,
        1.0, a, k,
        b, n,
        0.0, c, n,
    ) };
}

#[cfg(not(feature = "with_blas"))]
pub fn multiply(a: &Matrix, b: &Matrix, c: &mut Matrix) {
    debug_assert!(a.columns == b.rows, "Invalid matrix dimensions for multiplication!");
    debug_assert!(a.rows == c.rows && b.columns == c.columns, "Invalid destination matrix dimensions for multiplication!");

    use std::slice::from_raw_parts as s;
    use std::slice::from_raw_parts_mut as sm;

    for a_row_chunk in 0..(a.rows / 4) as isize {
        for b_column_chunk in 0..(b.columns / 16) as isize {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_4x16(
                    a.columns, c.columns,
                    s(a.buffer.offset(a_row_chunk * 4 * a.columns as isize + a_column), 3 * a.columns + 1),
                    s(b.buffer.offset(a_column * b.columns as isize + b_column_chunk * 16), 16),
                    sm(c.buffer.offset(a_row_chunk * 4 * b.columns as isize + b_column_chunk * 16), 3 * b.columns + 16),
                ) };
            }
        }
    }

    if b.columns % 16 >= 8 {
        for a_row_chunk in 0..(a.rows / 4) as isize {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_4x8(
                    a.columns, c.columns,
                    s(a.as_ptr().offset(a_row_chunk * 4 * a.columns as isize + a_column), 3 * a.columns + 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + (b.columns as isize / 8 - 1) * 8), 8),
                    sm(c.as_mut_ptr().offset(a_row_chunk * 4 * b.columns as isize + (b.columns as isize / 8 - 1) * 8), 3 * b.columns + 8),
                ) };
            }
        }
    }

    if b.columns % 8 >= 4 {
        for a_row_chunk in 0..(a.rows / 4) as isize {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_4x4(
                    a.columns, c.columns,
                    s(a.as_ptr().offset(a_row_chunk * 4 * a.columns as isize + a_column), 3 * a.columns + 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + (b.columns as isize / 4 - 1) * 4), 4),
                    sm(c.as_mut_ptr().offset(a_row_chunk * 4 * b.columns as isize + (b.columns as isize / 4 - 1) * 4), 3 * b.columns + 4),
                ) };
            }
        }
    }

    if b.columns % 4 >= 2 {
        for a_row_chunk in 0..(a.rows / 4) as isize {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_4x2(
                    a.columns, c.columns,
                    s(a.as_ptr().offset(a_row_chunk * 4 * a.columns as isize + a_column), 3 * a.columns + 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + (b.columns as isize / 2 - 1) * 2), 2),
                    sm(c.as_mut_ptr().offset(a_row_chunk * 4 * b.columns as isize + (b.columns as isize / 2 - 1) * 2), 3 * b.columns + 2),
                ) };
            }
        }
    }

    if b.columns % 2 == 1 {
        for a_row_chunk in 0..(a.rows / 4) as isize {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_4x1(
                    a.columns, c.columns,
                    s(a.as_ptr().offset(a_row_chunk * 4 * a.columns as isize + a_column), 3 * a.columns + 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + b.columns as isize - 1), 1),
                    sm(c.as_mut_ptr().offset(a_row_chunk * 4 * b.columns as isize + b.columns as isize - 1), 3 * b.columns + 1),
                ) };
            }
        }
    }

    if a.rows % 4 >= 2 {
        for b_column_chunk in 0..(b.columns / 16) as isize {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_2x16(
                    a.columns, c.columns,
                    s(a.buffer.offset((a.rows as isize / 2 - 1) * 2 * a.columns as isize + a_column), a.columns + 1),
                    s(b.buffer.offset(a_column * b.columns as isize + b_column_chunk * 16), 16),
                    sm(c.buffer.offset((a.rows as isize / 2 - 1) * 2 * b.columns as isize + b_column_chunk * 16), b.columns + 16),
                ) };
            }
        }

        if b.columns % 16 >= 8 {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_2x8(
                    a.columns, c.columns,
                    s(a.as_ptr().offset((a.rows as isize / 2 - 1) * 2 * a.columns as isize + a_column), a.columns + 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + (b.columns as isize / 8 - 1) * 8), 8),
                    sm(c.as_mut_ptr().offset((a.rows as isize / 2 - 1) * 2 * b.columns as isize + (b.columns as isize / 8 - 1) * 8), b.columns + 8),
                ) };
            }
        }

        if b.columns % 8 >= 4 {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_2x4(
                    a.columns, c.columns,
                    s(a.as_ptr().offset((a.rows as isize / 2 - 1) * 2 * a.columns as isize + a_column), a.columns + 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + (b.columns as isize / 4 - 1) * 4), 4),
                    sm(c.as_mut_ptr().offset((a.rows as isize / 2 - 1) * 2 * b.columns as isize + (b.columns as isize / 4 - 1) * 4), b.columns + 4),
                ) };
            }
        }

        if b.columns % 4 >= 2 {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_2x2(
                    a.columns, c.columns,
                    s(a.as_ptr().offset((a.rows as isize / 2 - 1) * 2 * a.columns as isize + a_column), a.columns + 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + (b.columns as isize / 2 - 1) * 2), 2),
                    sm(c.as_mut_ptr().offset((a.rows as isize / 2 - 1) * 2 * b.columns as isize + (b.columns as isize / 2 - 1) * 2), b.columns + 2),
                ) };
            }
        }

        if b.columns % 2 == 1 {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_2x1(
                    a.columns, c.columns,
                    s(a.as_ptr().offset((a.rows as isize / 2 - 1) * 2 * a.columns as isize + a_column), a.columns + 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + b.columns as isize - 1), 1),
                    sm(c.as_mut_ptr().offset((a.rows as isize / 2 - 1) * 2 * b.columns as isize + b.columns as isize - 1), b.columns + 1),
                ) };
            }
        }
    }

    if a.rows % 2 == 1 {
        for b_column_chunk in 0..(b.columns / 16) as isize {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_1x16(
                    s(a.buffer.offset((a.rows as isize - 1) * a.columns as isize + a_column), 1),
                    s(b.buffer.offset(a_column * b.columns as isize + b_column_chunk * 16), 16),
                    sm(c.buffer.offset((a.rows as isize - 1) * b.columns as isize + b_column_chunk * 16), 16),
                ) };
            }
        }

        if b.columns % 16 >= 8 {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_1x8(
                    s(a.as_ptr().offset((a.rows as isize - 1) * a.columns as isize + a_column), 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + (b.columns as isize / 8 - 1) * 8), 8),
                    sm(c.as_mut_ptr().offset((a.rows as isize - 1) * b.columns as isize + (b.columns as isize / 8 - 1) * 8), 8),
                ) };
            }
        }

        if b.columns % 8 >= 4 {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_1x4(
                    s(a.as_ptr().offset((a.rows as isize - 1) * a.columns as isize + a_column), 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + (b.columns as isize / 4 - 1) * 4), 4),
                    sm(c.as_mut_ptr().offset((a.rows as isize - 1) * b.columns as isize + (b.columns as isize / 4 - 1) * 4), 4),
                ) };
            }
        }

        if b.columns % 4 >= 2 {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_1x2(
                    s(a.as_ptr().offset((a.rows as isize - 1) * a.columns as isize + a_column), 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + (b.columns as isize / 2 - 1) * 2), 2),
                    sm(c.as_mut_ptr().offset((a.rows as isize - 1) * b.columns as isize + (b.columns as isize / 2 - 1) * 2), 2),
                ) };
            }
        }

        if b.columns % 2 == 1 {
            for a_column in 0..a.columns as isize {
                unsafe { multiply_1x1(
                    s(a.as_ptr().offset((a.rows as isize - 1) * a.columns as isize + a_column), 1),
                    s(b.as_ptr().offset(a_column * b.columns as isize + b.columns as isize - 1), 1),
                    sm(c.as_mut_ptr().offset((a.rows as isize - 1) * b.columns as isize + b.columns as isize - 1), 1),
                ) };
            }
        }
    }
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_4x16(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_ra = simd::x86::avx::f32x8::load(b, 0);
    let b_rb = simd::x86::avx::f32x8::load(b, 8);
    let a_c1 = simd::x86::avx::f32x8::splat(a[0 * a_columns]);
    let a_c2 = simd::x86::avx::f32x8::splat(a[1 * a_columns]);
    let a_c3 = simd::x86::avx::f32x8::splat(a[2 * a_columns]);
    let a_c4 = simd::x86::avx::f32x8::splat(a[3 * a_columns]);

    let c_r1a = simd::x86::avx::f32x8::load(c, 0 * c_columns + 0) + b_ra * a_c1;
    let c_r1b = simd::x86::avx::f32x8::load(c, 0 * c_columns + 8) + b_rb * a_c1;
    let c_r2a = simd::x86::avx::f32x8::load(c, 1 * c_columns + 0) + b_ra * a_c2;
    let c_r2b = simd::x86::avx::f32x8::load(c, 1 * c_columns + 8) + b_rb * a_c2;
    let c_r3a = simd::x86::avx::f32x8::load(c, 2 * c_columns + 0) + b_ra * a_c3;
    let c_r3b = simd::x86::avx::f32x8::load(c, 2 * c_columns + 8) + b_rb * a_c3;
    let c_r4a = simd::x86::avx::f32x8::load(c, 3 * c_columns + 0) + b_ra * a_c4;
    let c_r4b = simd::x86::avx::f32x8::load(c, 3 * c_columns + 8) + b_rb * a_c4;

    c_r1a.store(c, 0 * c_columns + 0);
    c_r1b.store(c, 0 * c_columns + 8);
    c_r2a.store(c, 1 * c_columns + 0);
    c_r2b.store(c, 1 * c_columns + 8);
    c_r3a.store(c, 2 * c_columns + 0);
    c_r3b.store(c, 2 * c_columns + 8);
    c_r4a.store(c, 3 * c_columns + 0);
    c_r4b.store(c, 3 * c_columns + 8);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_4x8(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_r = simd::x86::avx::f32x8::load(b, 0);
    let a_c1 = simd::x86::avx::f32x8::splat(a[0 * a_columns]);
    let a_c2 = simd::x86::avx::f32x8::splat(a[1 * a_columns]);
    let a_c3 = simd::x86::avx::f32x8::splat(a[2 * a_columns]);
    let a_c4 = simd::x86::avx::f32x8::splat(a[3 * a_columns]);

    let c_r1 = simd::x86::avx::f32x8::load(c, 0 * c_columns) + b_r * a_c1;
    let c_r2 = simd::x86::avx::f32x8::load(c, 1 * c_columns) + b_r * a_c2;
    let c_r3 = simd::x86::avx::f32x8::load(c, 2 * c_columns) + b_r * a_c3;
    let c_r4 = simd::x86::avx::f32x8::load(c, 3 * c_columns) + b_r * a_c4;

    c_r1.store(c, 0 * c_columns);
    c_r2.store(c, 1 * c_columns);
    c_r3.store(c, 2 * c_columns);
    c_r4.store(c, 3 * c_columns);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_4x4(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_r = simd::f32x4::load(b, 0);
    let a_c1 = simd::f32x4::splat(a[0 * a_columns]);
    let a_c2 = simd::f32x4::splat(a[1 * a_columns]);
    let a_c3 = simd::f32x4::splat(a[2 * a_columns]);
    let a_c4 = simd::f32x4::splat(a[3 * a_columns]);

    let c_r1 = simd::f32x4::load(c, 0 * c_columns) + b_r * a_c1;
    let c_r2 = simd::f32x4::load(c, 1 * c_columns) + b_r * a_c2;
    let c_r3 = simd::f32x4::load(c, 2 * c_columns) + b_r * a_c3;
    let c_r4 = simd::f32x4::load(c, 3 * c_columns) + b_r * a_c4;

    c_r1.store(c, 0 * c_columns);
    c_r2.store(c, 1 * c_columns);
    c_r3.store(c, 2 * c_columns);
    c_r4.store(c, 3 * c_columns);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_4x2(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_r = simd::f32x4::new(b[0], b[1], 0.0, 0.0);
    let a_c1 = simd::f32x4::splat(a[0 * a_columns]);
    let a_c2 = simd::f32x4::splat(a[1 * a_columns]);
    let a_c3 = simd::f32x4::splat(a[2 * a_columns]);
    let a_c4 = simd::f32x4::splat(a[3 * a_columns]);

    let c_r1 = simd::f32x4::new(c[0 * c_columns + 0], c[0 * c_columns + 1], 0.0, 0.0) + b_r * a_c1;
    let c_r2 = simd::f32x4::new(c[1 * c_columns + 0], c[1 * c_columns + 1], 0.0, 0.0) + b_r * a_c2;
    let c_r3 = simd::f32x4::new(c[2 * c_columns + 0], c[2 * c_columns + 1], 0.0, 0.0) + b_r * a_c3;
    let c_r4 = simd::f32x4::new(c[3 * c_columns + 0], c[3 * c_columns + 1], 0.0, 0.0) + b_r * a_c4;

    c[0 * c_columns + 0] = c_r1.extract(0); c[0 * c_columns + 1] = c_r1.extract(1);
    c[1 * c_columns + 0] = c_r2.extract(0); c[1 * c_columns + 1] = c_r2.extract(1);
    c[2 * c_columns + 0] = c_r3.extract(0); c[2 * c_columns + 1] = c_r3.extract(1);
    c[3 * c_columns + 0] = c_r4.extract(0); c[3 * c_columns + 1] = c_r4.extract(1);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_4x1(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_r = simd::f32x4::splat(b[0]);
    let a_c = simd::f32x4::new(a[0 * a_columns], a[1 * a_columns], a[2 * a_columns], a[3 * a_columns]);

    let c_c = b_r * a_c;

    c[0 * c_columns] += c_c.extract(0);
    c[1 * c_columns] += c_c.extract(1);
    c[2 * c_columns] += c_c.extract(2);
    c[3 * c_columns] += c_c.extract(3);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_2x16(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_ra = simd::x86::avx::f32x8::load(b, 0);
    let b_rb = simd::x86::avx::f32x8::load(b, 8);
    let a_c1 = simd::x86::avx::f32x8::splat(a[0 * a_columns]);
    let a_c2 = simd::x86::avx::f32x8::splat(a[1 * a_columns]);

    let c_r1a = simd::x86::avx::f32x8::load(c, 0 * c_columns + 0) + b_ra * a_c1;
    let c_r1b = simd::x86::avx::f32x8::load(c, 0 * c_columns + 8) + b_rb * a_c1;
    let c_r2a = simd::x86::avx::f32x8::load(c, 1 * c_columns + 0) + b_ra * a_c2;
    let c_r2b = simd::x86::avx::f32x8::load(c, 1 * c_columns + 8) + b_rb * a_c2;

    c_r1a.store(c, 0 * c_columns + 0);
    c_r1b.store(c, 0 * c_columns + 8);
    c_r2a.store(c, 1 * c_columns + 0);
    c_r2b.store(c, 1 * c_columns + 8);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_2x8(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_r = simd::x86::avx::f32x8::load(b, 0);
    let a_c1 = simd::x86::avx::f32x8::splat(a[0 * a_columns]);
    let a_c2 = simd::x86::avx::f32x8::splat(a[1 * a_columns]);

    let c_r1 = simd::x86::avx::f32x8::load(c, 0 * c_columns) + b_r * a_c1;
    let c_r2 = simd::x86::avx::f32x8::load(c, 1 * c_columns) + b_r * a_c2;

    c_r1.store(c, 0 * c_columns);
    c_r2.store(c, 1 * c_columns);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_2x4(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_r = simd::f32x4::load(b, 0);
    let a_c1 = simd::f32x4::splat(a[0 * a_columns]);
    let a_c2 = simd::f32x4::splat(a[1 * a_columns]);

    let c_r1 = simd::f32x4::load(c, 0 * c_columns) + b_r * a_c1;
    let c_r2 = simd::f32x4::load(c, 1 * c_columns) + b_r * a_c2;

    c_r1.store(c, 0 * c_columns);
    c_r2.store(c, 1 * c_columns);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_2x2(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_r = simd::f32x4::new(b[0], b[1], 0.0, 0.0);
    let a_c1 = simd::f32x4::splat(a[0 * a_columns]);
    let a_c2 = simd::f32x4::splat(a[1 * a_columns]);

    let c_r1 = simd::f32x4::new(c[0 * c_columns + 0], c[0 * c_columns + 1], 0.0, 0.0) + b_r * a_c1;
    let c_r2 = simd::f32x4::new(c[1 * c_columns + 0], c[1 * c_columns + 1], 0.0, 0.0) + b_r * a_c2;

    c[0 * c_columns + 0] = c_r1.extract(0); c[0 * c_columns + 1] = c_r1.extract(1);
    c[1 * c_columns + 0] = c_r2.extract(0); c[1 * c_columns + 1] = c_r2.extract(1);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_2x1(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_r = simd::f32x4::splat(b[0]);
    let a_c = simd::f32x4::new(a[0 * a_columns], a[1 * a_columns], 0.0, 0.0);

    let c_c = b_r * a_c;

    c[0 * c_columns] += c_c.extract(0);
    c[1 * c_columns] += c_c.extract(1);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_1x16(a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_ra = simd::x86::avx::f32x8::load(b, 0);
    let b_rb = simd::x86::avx::f32x8::load(b, 8);
    let a_c1 = simd::x86::avx::f32x8::splat(a[0]);

    let c_r1a = simd::x86::avx::f32x8::load(c, 0) + b_ra * a_c1;
    let c_r1b = simd::x86::avx::f32x8::load(c, 8) + b_rb * a_c1;

    c_r1a.store(c, 0);
    c_r1b.store(c, 8);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_1x8(a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_r = simd::x86::avx::f32x8::load(b, 0);
    let a_c1 = simd::x86::avx::f32x8::splat(a[0]);

    let c_r1 = simd::x86::avx::f32x8::load(c, 0) + b_r * a_c1;

    c_r1.store(c, 0);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_1x4(a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_r = simd::f32x4::load(b, 0);
    let a_c1 = simd::f32x4::splat(a[0]);

    let c_r1 = simd::f32x4::load(c, 0) + b_r * a_c1;

    c_r1.store(c, 0);
}

#[cfg(all(feature = "with_simd", not(feature = "with_blas")))]
fn multiply_1x2(a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_r = simd::f32x4::new(b[0], b[1], 0.0, 0.0);
    let a_c1 = simd::f32x4::splat(a[0]);

    let c_r1 = simd::f32x4::new(c[0], c[1], 0.0, 0.0) + b_r * a_c1;

    c[0] = c_r1.extract(0); c[1] = c_r1.extract(1);
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_4x16(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0 * c_columns +  0] += a[0 * a_columns] * b[ 0];
    c[0 * c_columns +  1] += a[0 * a_columns] * b[ 1];
    c[0 * c_columns +  2] += a[0 * a_columns] * b[ 2];
    c[0 * c_columns +  3] += a[0 * a_columns] * b[ 3];
    c[0 * c_columns +  4] += a[0 * a_columns] * b[ 4];
    c[0 * c_columns +  5] += a[0 * a_columns] * b[ 5];
    c[0 * c_columns +  6] += a[0 * a_columns] * b[ 6];
    c[0 * c_columns +  7] += a[0 * a_columns] * b[ 7];
    c[0 * c_columns +  8] += a[0 * a_columns] * b[ 8];
    c[0 * c_columns +  9] += a[0 * a_columns] * b[ 9];
    c[0 * c_columns + 10] += a[0 * a_columns] * b[10];
    c[0 * c_columns + 11] += a[0 * a_columns] * b[11];
    c[0 * c_columns + 12] += a[0 * a_columns] * b[12];
    c[0 * c_columns + 13] += a[0 * a_columns] * b[13];
    c[0 * c_columns + 14] += a[0 * a_columns] * b[14];
    c[0 * c_columns + 15] += a[0 * a_columns] * b[15];

    c[1 * c_columns +  0] += a[1 * a_columns] * b[ 0];
    c[1 * c_columns +  1] += a[1 * a_columns] * b[ 1];
    c[1 * c_columns +  2] += a[1 * a_columns] * b[ 2];
    c[1 * c_columns +  3] += a[1 * a_columns] * b[ 3];
    c[1 * c_columns +  4] += a[1 * a_columns] * b[ 4];
    c[1 * c_columns +  5] += a[1 * a_columns] * b[ 5];
    c[1 * c_columns +  6] += a[1 * a_columns] * b[ 6];
    c[1 * c_columns +  7] += a[1 * a_columns] * b[ 7];
    c[1 * c_columns +  8] += a[1 * a_columns] * b[ 8];
    c[1 * c_columns +  9] += a[1 * a_columns] * b[ 9];
    c[1 * c_columns + 10] += a[1 * a_columns] * b[10];
    c[1 * c_columns + 11] += a[1 * a_columns] * b[11];
    c[1 * c_columns + 12] += a[1 * a_columns] * b[12];
    c[1 * c_columns + 13] += a[1 * a_columns] * b[13];
    c[1 * c_columns + 14] += a[1 * a_columns] * b[14];
    c[1 * c_columns + 15] += a[1 * a_columns] * b[15];

    c[2 * c_columns +  0] += a[2 * a_columns] * b[ 0];
    c[2 * c_columns +  1] += a[2 * a_columns] * b[ 1];
    c[2 * c_columns +  2] += a[2 * a_columns] * b[ 2];
    c[2 * c_columns +  3] += a[2 * a_columns] * b[ 3];
    c[2 * c_columns +  4] += a[2 * a_columns] * b[ 4];
    c[2 * c_columns +  5] += a[2 * a_columns] * b[ 5];
    c[2 * c_columns +  6] += a[2 * a_columns] * b[ 6];
    c[2 * c_columns +  7] += a[2 * a_columns] * b[ 7];
    c[2 * c_columns +  8] += a[2 * a_columns] * b[ 8];
    c[2 * c_columns +  9] += a[2 * a_columns] * b[ 9];
    c[2 * c_columns + 10] += a[2 * a_columns] * b[10];
    c[2 * c_columns + 11] += a[2 * a_columns] * b[11];
    c[2 * c_columns + 12] += a[2 * a_columns] * b[12];
    c[2 * c_columns + 13] += a[2 * a_columns] * b[13];
    c[2 * c_columns + 14] += a[2 * a_columns] * b[14];
    c[2 * c_columns + 15] += a[2 * a_columns] * b[15];

    c[3 * c_columns +  0] += a[3 * a_columns] * b[ 0];
    c[3 * c_columns +  1] += a[3 * a_columns] * b[ 1];
    c[3 * c_columns +  2] += a[3 * a_columns] * b[ 2];
    c[3 * c_columns +  3] += a[3 * a_columns] * b[ 3];
    c[3 * c_columns +  4] += a[3 * a_columns] * b[ 4];
    c[3 * c_columns +  5] += a[3 * a_columns] * b[ 5];
    c[3 * c_columns +  6] += a[3 * a_columns] * b[ 6];
    c[3 * c_columns +  7] += a[3 * a_columns] * b[ 7];
    c[3 * c_columns +  8] += a[3 * a_columns] * b[ 8];
    c[3 * c_columns +  9] += a[3 * a_columns] * b[ 9];
    c[3 * c_columns + 10] += a[3 * a_columns] * b[10];
    c[3 * c_columns + 11] += a[3 * a_columns] * b[11];
    c[3 * c_columns + 12] += a[3 * a_columns] * b[12];
    c[3 * c_columns + 13] += a[3 * a_columns] * b[13];
    c[3 * c_columns + 14] += a[3 * a_columns] * b[14];
    c[3 * c_columns + 15] += a[3 * a_columns] * b[15];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_4x8(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0 * c_columns + 0] += a[0 * a_columns] * b[0];
    c[0 * c_columns + 1] += a[0 * a_columns] * b[1];
    c[0 * c_columns + 2] += a[0 * a_columns] * b[2];
    c[0 * c_columns + 3] += a[0 * a_columns] * b[3];
    c[0 * c_columns + 4] += a[0 * a_columns] * b[4];
    c[0 * c_columns + 5] += a[0 * a_columns] * b[5];
    c[0 * c_columns + 6] += a[0 * a_columns] * b[6];
    c[0 * c_columns + 7] += a[0 * a_columns] * b[7];

    c[1 * c_columns + 0] += a[1 * a_columns] * b[0];
    c[1 * c_columns + 1] += a[1 * a_columns] * b[1];
    c[1 * c_columns + 2] += a[1 * a_columns] * b[2];
    c[1 * c_columns + 3] += a[1 * a_columns] * b[3];
    c[1 * c_columns + 4] += a[1 * a_columns] * b[4];
    c[1 * c_columns + 5] += a[1 * a_columns] * b[5];
    c[1 * c_columns + 6] += a[1 * a_columns] * b[6];
    c[1 * c_columns + 7] += a[1 * a_columns] * b[7];

    c[2 * c_columns + 0] += a[2 * a_columns] * b[0];
    c[2 * c_columns + 1] += a[2 * a_columns] * b[1];
    c[2 * c_columns + 2] += a[2 * a_columns] * b[2];
    c[2 * c_columns + 3] += a[2 * a_columns] * b[3];
    c[2 * c_columns + 4] += a[2 * a_columns] * b[4];
    c[2 * c_columns + 5] += a[2 * a_columns] * b[5];
    c[2 * c_columns + 6] += a[2 * a_columns] * b[6];
    c[2 * c_columns + 7] += a[2 * a_columns] * b[7];

    c[3 * c_columns + 0] += a[3 * a_columns] * b[0];
    c[3 * c_columns + 1] += a[3 * a_columns] * b[1];
    c[3 * c_columns + 2] += a[3 * a_columns] * b[2];
    c[3 * c_columns + 3] += a[3 * a_columns] * b[3];
    c[3 * c_columns + 4] += a[3 * a_columns] * b[4];
    c[3 * c_columns + 5] += a[3 * a_columns] * b[5];
    c[3 * c_columns + 6] += a[3 * a_columns] * b[6];
    c[3 * c_columns + 7] += a[3 * a_columns] * b[7];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_4x4(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0 * c_columns + 0] += a[0 * a_columns] * b[0];
    c[0 * c_columns + 1] += a[0 * a_columns] * b[1];
    c[0 * c_columns + 2] += a[0 * a_columns] * b[2];
    c[0 * c_columns + 3] += a[0 * a_columns] * b[3];

    c[1 * c_columns + 0] += a[1 * a_columns] * b[0];
    c[1 * c_columns + 1] += a[1 * a_columns] * b[1];
    c[1 * c_columns + 2] += a[1 * a_columns] * b[2];
    c[1 * c_columns + 3] += a[1 * a_columns] * b[3];

    c[2 * c_columns + 0] += a[2 * a_columns] * b[0];
    c[2 * c_columns + 1] += a[2 * a_columns] * b[1];
    c[2 * c_columns + 2] += a[2 * a_columns] * b[2];
    c[2 * c_columns + 3] += a[2 * a_columns] * b[3];

    c[3 * c_columns + 0] += a[3 * a_columns] * b[0];
    c[3 * c_columns + 1] += a[3 * a_columns] * b[1];
    c[3 * c_columns + 2] += a[3 * a_columns] * b[2];
    c[3 * c_columns + 3] += a[3 * a_columns] * b[3];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_4x2(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0 * c_columns + 0] += a[0 * a_columns] * b[0];
    c[0 * c_columns + 1] += a[0 * a_columns] * b[1];

    c[1 * c_columns + 0] += a[1 * a_columns] * b[0];
    c[1 * c_columns + 1] += a[1 * a_columns] * b[1];

    c[2 * c_columns + 0] += a[2 * a_columns] * b[0];
    c[2 * c_columns + 1] += a[2 * a_columns] * b[1];

    c[3 * c_columns + 0] += a[3 * a_columns] * b[0];
    c[3 * c_columns + 1] += a[3 * a_columns] * b[1];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_4x1(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0 * c_columns] += a[0 * a_columns] * b[0];
    c[1 * c_columns] += a[1 * a_columns] * b[0];
    c[2 * c_columns] += a[2 * a_columns] * b[0];
    c[3 * c_columns] += a[3 * a_columns] * b[0];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_2x16(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0 * c_columns +  0] += a[0 * a_columns] * b[ 0];
    c[0 * c_columns +  1] += a[0 * a_columns] * b[ 1];
    c[0 * c_columns +  2] += a[0 * a_columns] * b[ 2];
    c[0 * c_columns +  3] += a[0 * a_columns] * b[ 3];
    c[0 * c_columns +  4] += a[0 * a_columns] * b[ 4];
    c[0 * c_columns +  5] += a[0 * a_columns] * b[ 5];
    c[0 * c_columns +  6] += a[0 * a_columns] * b[ 6];
    c[0 * c_columns +  7] += a[0 * a_columns] * b[ 7];
    c[0 * c_columns +  8] += a[0 * a_columns] * b[ 8];
    c[0 * c_columns +  9] += a[0 * a_columns] * b[ 9];
    c[0 * c_columns + 10] += a[0 * a_columns] * b[10];
    c[0 * c_columns + 11] += a[0 * a_columns] * b[11];
    c[0 * c_columns + 12] += a[0 * a_columns] * b[12];
    c[0 * c_columns + 13] += a[0 * a_columns] * b[13];
    c[0 * c_columns + 14] += a[0 * a_columns] * b[14];
    c[0 * c_columns + 15] += a[0 * a_columns] * b[15];

    c[1 * c_columns +  0] += a[1 * a_columns] * b[ 0];
    c[1 * c_columns +  1] += a[1 * a_columns] * b[ 1];
    c[1 * c_columns +  2] += a[1 * a_columns] * b[ 2];
    c[1 * c_columns +  3] += a[1 * a_columns] * b[ 3];
    c[1 * c_columns +  4] += a[1 * a_columns] * b[ 4];
    c[1 * c_columns +  5] += a[1 * a_columns] * b[ 5];
    c[1 * c_columns +  6] += a[1 * a_columns] * b[ 6];
    c[1 * c_columns +  7] += a[1 * a_columns] * b[ 7];
    c[1 * c_columns +  8] += a[1 * a_columns] * b[ 8];
    c[1 * c_columns +  9] += a[1 * a_columns] * b[ 9];
    c[1 * c_columns + 10] += a[1 * a_columns] * b[10];
    c[1 * c_columns + 11] += a[1 * a_columns] * b[11];
    c[1 * c_columns + 12] += a[1 * a_columns] * b[12];
    c[1 * c_columns + 13] += a[1 * a_columns] * b[13];
    c[1 * c_columns + 14] += a[1 * a_columns] * b[14];
    c[1 * c_columns + 15] += a[1 * a_columns] * b[15];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_2x8(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0 * c_columns + 0] += a[0 * a_columns] * b[0];
    c[0 * c_columns + 1] += a[0 * a_columns] * b[1];
    c[0 * c_columns + 2] += a[0 * a_columns] * b[2];
    c[0 * c_columns + 3] += a[0 * a_columns] * b[3];
    c[0 * c_columns + 4] += a[0 * a_columns] * b[4];
    c[0 * c_columns + 5] += a[0 * a_columns] * b[5];
    c[0 * c_columns + 6] += a[0 * a_columns] * b[6];
    c[0 * c_columns + 7] += a[0 * a_columns] * b[7];

    c[1 * c_columns + 0] += a[1 * a_columns] * b[0];
    c[1 * c_columns + 1] += a[1 * a_columns] * b[1];
    c[1 * c_columns + 2] += a[1 * a_columns] * b[2];
    c[1 * c_columns + 3] += a[1 * a_columns] * b[3];
    c[1 * c_columns + 4] += a[1 * a_columns] * b[4];
    c[1 * c_columns + 5] += a[1 * a_columns] * b[5];
    c[1 * c_columns + 6] += a[1 * a_columns] * b[6];
    c[1 * c_columns + 7] += a[1 * a_columns] * b[7];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_2x4(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0 * c_columns + 0] += a[0 * a_columns] * b[0];
    c[0 * c_columns + 1] += a[0 * a_columns] * b[1];
    c[0 * c_columns + 2] += a[0 * a_columns] * b[2];
    c[0 * c_columns + 3] += a[0 * a_columns] * b[3];

    c[1 * c_columns + 0] += a[1 * a_columns] * b[0];
    c[1 * c_columns + 1] += a[1 * a_columns] * b[1];
    c[1 * c_columns + 2] += a[1 * a_columns] * b[2];
    c[1 * c_columns + 3] += a[1 * a_columns] * b[3];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_2x2(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0 * c_columns + 0] += a[0 * a_columns] * b[0];
    c[0 * c_columns + 1] += a[0 * a_columns] * b[1];

    c[1 * c_columns + 0] += a[1 * a_columns] * b[0];
    c[1 * c_columns + 1] += a[1 * a_columns] * b[1];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_2x1(a_columns: usize, c_columns: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0 * c_columns] += a[0 * a_columns] * b[0];
    c[1 * c_columns] += a[1 * a_columns] * b[0];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_1x16(a: &[f32], b: &[f32], c: &mut [f32]) {
    c[ 0] += a[0] * b[ 0];
    c[ 1] += a[0] * b[ 1];
    c[ 2] += a[0] * b[ 2];
    c[ 3] += a[0] * b[ 3];
    c[ 4] += a[0] * b[ 4];
    c[ 5] += a[0] * b[ 5];
    c[ 6] += a[0] * b[ 6];
    c[ 7] += a[0] * b[ 7];
    c[ 8] += a[0] * b[ 8];
    c[ 9] += a[0] * b[ 9];
    c[10] += a[0] * b[10];
    c[11] += a[0] * b[11];
    c[12] += a[0] * b[12];
    c[13] += a[0] * b[13];
    c[14] += a[0] * b[14];
    c[15] += a[0] * b[15];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_1x8(a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0] += a[0] * b[0];
    c[1] += a[0] * b[1];
    c[2] += a[0] * b[2];
    c[3] += a[0] * b[3];
    c[4] += a[0] * b[4];
    c[5] += a[0] * b[5];
    c[6] += a[0] * b[6];
    c[7] += a[0] * b[7];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_1x4(a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0] += a[0] * b[0];
    c[1] += a[0] * b[1];
    c[2] += a[0] * b[2];
    c[3] += a[0] * b[3];
}

#[cfg(all(not(feature = "with_simd"), not(feature = "with_blas")))]
fn multiply_1x2(a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0] += a[0] * b[0];
    c[1] += a[0] * b[1];
}

#[cfg(not(feature = "with_blas"))]
fn multiply_1x1(a: &[f32], b: &[f32], c: &mut [f32]) {
    c[0] += b[0] * a[0];
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_multiply(bencher: &mut Bencher) {
        let (m, k, n) = (1, 264, 100);

        let mut a = Matrix::zeros(m, k);
        for (i, x) in a.iter_mut().enumerate() {
            *x = (i % 16 + 1) as f32;
        }

        let mut b = Matrix::zeros(k, n);
        for (i, x) in b.iter_mut().enumerate() {
            *x = (i % 16 + 1) as f32;
        }

        let mut c = Matrix::zeros(m, n);

        bencher.iter(|| {
            c.zero();
            multiply(&a, &b, &mut c);
        });
    }
}
