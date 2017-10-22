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

use matrix::Matrix;

pub fn clone_region(a: &Matrix, a_start_row: usize, a_start_column: usize, b: &mut Matrix, b_start_row: usize, b_start_column: usize, rows: usize, columns: usize) {
    debug_assert!(a.rows >= a_start_row + rows && a.columns >= a_start_column + columns, "Target region is too big to clone out of source matrix!");
    debug_assert!(b.rows >= b_start_row + rows && b.columns >= b_start_column + columns, "Target region is too big to clone into destination matrix!");

    use std::slice::from_raw_parts as s;
    use std::slice::from_raw_parts_mut as sm;

    for row in 0..rows as isize {
        for column_chunk in 0..(columns / 16) as isize {
            unsafe { clone_16(
                s(a.buffer.offset((a_start_row as isize + row) * a.columns as isize + a_start_column as isize + column_chunk * 16), 16),
                sm(b.buffer.offset((b_start_row as isize + row) * b.columns as isize + b_start_column as isize + column_chunk * 16), 16),
            ) };
        }

        if columns % 16 >= 8 {
            unsafe { clone_8(
                s(a.buffer.offset((a_start_row as isize + row) * a.columns as isize + a_start_column as isize + (columns as isize / 8 - 1) * 8), 8),
                sm(b.buffer.offset((b_start_row as isize + row) * b.columns as isize + b_start_column as isize + (columns as isize / 8 - 1) * 8), 8),
            ) };
        }

        if columns % 8 >= 4 {
            unsafe { clone_4(
                s(a.buffer.offset((a_start_row as isize + row) * a.columns as isize + a_start_column as isize + (columns as isize / 4 - 1) * 4), 4),
                sm(b.buffer.offset((b_start_row as isize + row) * b.columns as isize + b_start_column as isize + (columns as isize / 4 - 1) * 4), 4),
            ) };
        }

        if columns % 4 >= 2 {
            unsafe { clone_2(
                s(a.buffer.offset((a_start_row as isize + row) * a.columns as isize + a_start_column as isize + (columns as isize / 2 - 1) * 2), 2),
                sm(b.buffer.offset((b_start_row as isize + row) * b.columns as isize + b_start_column as isize + (columns as isize / 2 - 1) * 2), 2),
            ) };
        }

        if columns % 2 == 1 {
            b[(b_start_row + row as usize, b_start_column + columns - 1)] = a[(a_start_row + row as usize, a_start_column + columns - 1)];
        }
    }
}

#[cfg(feature = "with_simd")]
fn clone_16(a: &[f32], b: &mut [f32]) {
    let d1 = simd::x86::avx::f32x8::load(a, 0);
    let d2 = simd::x86::avx::f32x8::load(a, 8);
    d1.store(b, 0);
    d2.store(b, 8);
}

#[cfg(feature = "with_simd")]
fn clone_8(a: &[f32], b: &mut [f32]) {
    let d1 = simd::x86::avx::f32x8::load(a, 0);
    d1.store(b, 0);
}

#[cfg(feature = "with_simd")]
fn clone_4(a: &[f32], b: &mut [f32]) {
    let d1 = simd::f32x4::load(a, 0);
    d1.store(b, 0);
}

#[cfg(not(feature = "with_simd"))]
fn clone_16(a: &[f32], b: &mut [f32]) {
    b[ 0] = a[ 0];
    b[ 1] = a[ 1];
    b[ 2] = a[ 2];
    b[ 3] = a[ 3];
    b[ 4] = a[ 4];
    b[ 5] = a[ 5];
    b[ 6] = a[ 6];
    b[ 7] = a[ 7];
    b[ 8] = a[ 8];
    b[ 9] = a[ 9];
    b[10] = a[10];
    b[11] = a[11];
    b[12] = a[12];
    b[13] = a[13];
    b[14] = a[14];
    b[15] = a[15];
}

#[cfg(not(feature = "with_simd"))]
fn clone_8(a: &[f32], b: &mut [f32]) {
    b[ 0] = a[ 0];
    b[ 1] = a[ 1];
    b[ 2] = a[ 2];
    b[ 3] = a[ 3];
    b[ 4] = a[ 4];
    b[ 5] = a[ 5];
    b[ 6] = a[ 6];
    b[ 7] = a[ 7];
}

#[cfg(not(feature = "with_simd"))]
fn clone_4(a: &[f32], b: &mut [f32]) {
    b[ 0] = a[ 0];
    b[ 1] = a[ 1];
    b[ 2] = a[ 2];
    b[ 3] = a[ 3];
}

fn clone_2(a: &[f32], b: &mut [f32]) {
    b[0] = a[0];
    b[1] = a[1];
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_clone_region(bencher: &mut Bencher) {
        let mut a = Matrix::zeros(100, 264);
        for (i, x) in a.iter_mut().enumerate() {
            *x = (i % 16 + 1) as f32;
        }

        let mut b = Matrix::zeros(100, 200);

        bencher.iter(|| {
            clone_region(&a, 0, 14, &mut b, 0, 0, 100, 200);
        });
    }
}
