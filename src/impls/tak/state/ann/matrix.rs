//
// This file is part of zero_sum.
//
// zero_sum is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// zero_sum is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with zero_sum. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright 2016-2017 Chris Foster
//

use std::ptr::write_bytes;
use std::ops::{Index, IndexMut};

/// Column-major matrix
#[derive(Clone, Debug)]
pub struct MatrixCm {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f32>,
}

impl MatrixCm {
    pub fn zeros(rows: usize, columns: usize) -> MatrixCm {
        MatrixCm {
            rows: rows,
            columns: columns,
            values: vec![0.0; rows * columns],
        }
    }

    pub fn ones(rows: usize, columns: usize) -> MatrixCm {
        MatrixCm {
            rows: rows,
            columns: columns,
            values: vec![1.0; rows * columns],
        }
    }

    pub fn from_vec(rows: usize, columns: usize, values: Vec<f32>) -> MatrixCm {
        assert!(values.len() == rows * columns, "Dimension mismatch in matrix creation!");

        MatrixCm {
            rows: rows,
            columns: columns,
            values: values,
        }
    }

    pub fn from_row_major_vec(rows: usize, columns: usize, values: Vec<f32>) -> MatrixCm {
        assert!(values.len() == rows * columns, "Dimension mismatch in matrix creation!");

        let mut transposed = Vec::with_capacity(rows * columns);
        for j in 0..columns {
            for i in 0..rows {
                transposed.push(values[i * columns + j]);
            }
        }
        MatrixCm {
            rows: rows,
            columns: columns,
            values: transposed,
        }
    }

    pub fn same_size(&self, other: &MatrixCm) -> bool {
        self.rows == other.rows && self.columns == other.columns
    }

    pub fn zero(&mut self) {
        unsafe {
            let values = self.values.as_mut_ptr();
            write_bytes(values, 0x00, self.values.len());
        }
    }
}

impl Index<usize> for MatrixCm {
    type Output = [f32];

    fn index(&self, index: usize) -> &[f32] {
        debug_assert!(index < self.columns, "Invalid matrix index!");

        &self.values[index * self.rows..(index + 1) * self.rows]
    }
}

impl IndexMut<usize> for MatrixCm {
    fn index_mut(&mut self, index: usize) -> &mut [f32] {
        debug_assert!(index < self.columns, "Invalid matrix index!");

        &mut self.values[index * self.rows..(index + 1) * self.rows]
    }
}

/// Row-major matrix
#[derive(Clone, Debug)]
pub struct MatrixRm {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f32>,
}

impl MatrixRm {
    pub fn zeros(rows: usize, columns: usize) -> MatrixRm {
        MatrixRm {
            rows: rows,
            columns: columns,
            values: vec![0.0; rows * columns],
        }
    }

    pub fn ones(rows: usize, columns: usize) -> MatrixRm {
        MatrixRm {
            rows: rows,
            columns: columns,
            values: vec![1.0; rows * columns],
        }
    }

    pub fn from_vec(rows: usize, columns: usize, values: Vec<f32>) -> MatrixRm {
        assert!(values.len() == rows * columns, "Dimension mismatch in matrix creation!");

        MatrixRm {
            rows: rows,
            columns: columns,
            values: values,
        }
    }

    pub fn same_size(&self, other: &MatrixRm) -> bool {
        self.rows == other.rows && self.columns == other.columns
    }

    pub fn zero(&mut self) {
        unsafe {
            let values = self.values.as_mut_ptr();
            write_bytes(values, 0x00, self.values.len());
        }
    }

    pub fn resize(&mut self, rows: usize, columns: usize) {
        self.rows = rows;
        self.columns = columns;
        if self.rows * self.columns > self.values.capacity() {
            let additional = self.rows * self.columns - self.values.capacity();
            self.values.reserve(additional);
        }
        unsafe { self.values.set_len(self.rows * self.columns); }
    }
}

impl Index<usize> for MatrixRm {
    type Output = [f32];

    fn index(&self, index: usize) -> &[f32] {
        debug_assert!(index < self.rows, "Invalid matrix index!");

        &self.values[index * self.columns..(index + 1) * self.columns]
    }
}

impl IndexMut<usize> for MatrixRm {
    fn index_mut(&mut self, index: usize) -> &mut [f32] {
        debug_assert!(index < self.rows, "Invalid matrix index!");

        &mut self.values[index * self.columns..(index + 1) * self.columns]
    }
}
