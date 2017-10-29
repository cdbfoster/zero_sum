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

use std::mem;
use std::ops::{Deref, DerefMut, Drop, Index, IndexMut, Mul};
use std::slice;

use vector::Vector;

/// A row-major matrix.  `Deref`s into a [`&Vector`](../vector/struct.Vector.html), and can be converted into a `Vec<f32>` via `into()`.
#[repr(C)]
pub struct Matrix {
    rows: usize,
    columns: usize,

    buffer: *mut f32,
    capacity: usize,
    len: usize,
}

impl Matrix {
    /// Constructs a matrix from a `Vec<f32>`.
    ///
    /// # Panics
    /// Panics if the length of `values` isn't equal to `rows` times `columns`.
    pub fn from_vec(rows: usize, columns: usize, mut values: Vec<f32>) -> Matrix {
        assert!(values.len() == rows * columns, "Incorrect number of values for matrix!");

        let buffer = values.as_mut_ptr();
        let capacity = values.capacity();
        let len = values.len();
        mem::forget(values);

        Matrix {
            rows: rows,
            columns: columns,
            buffer: buffer,
            capacity: capacity,
            len: len,
        }
    }

    pub fn zeros(rows: usize, columns: usize) -> Matrix {
        Matrix::from_vec(rows, columns, vec![0.0; rows * columns])
    }

    /// Constructs an identity matrix of the specified size.
    pub fn identity(size: usize) -> Matrix {
        let mut matrix = Matrix::zeros(size, size);
        for i in 0..size {
            matrix[(i, i)] = 1.0;
        }
        matrix
    }

    /// Constructs a matrix from a buffer and a capacity.
    ///
    /// # Safety
    /// This method requires the same precautions as [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html)'s
    /// [`from_raw_parts`](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.from_raw_parts) method.
    pub unsafe fn from_raw_parts(rows: usize, columns: usize, buffer: *mut f32, capacity: usize) -> Matrix {
        Matrix {
            rows: rows,
            columns: columns,
            buffer: buffer,
            capacity: capacity,
            len: rows * columns,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Borrows the matrix as a vector of length `self.rows()` * `self.columns()`.
    pub fn as_vector(&self) -> &Vector {
        self
    }

    /// Mutably borrows the matrix as a vector of length `self.rows()` * `self.columns()`.
    pub fn as_vector_mut(&mut self) -> &mut Vector {
        self
    }

    /// Resizes the matrix.  This incurs a reallocation only if the new size exceeds the matrix's internal capacity.  This method resizes the matrix's internal buffer
    /// and does no reinterpretation of the data; values will no longer be at the same indices.  To preserve the data in the matrix, instead create a new matrix and then
    /// use [`matrix::ops::clone_region`](ops/fn.clone_region.html) to copy the original data into the correct position.
    pub fn resize(&mut self, rows: usize, columns: usize) {
        self.rows = rows;
        self.columns = columns;

        self.as_vector_mut().resize(rows * columns);
    }

    /// Returns the transpose of the matrix.  This will allocate and return a new matrix.  To avoid an additional allocation, use [`matrix::ops::tranpose`](ops/fn.transpose.html).
    pub fn transpose(&self) -> Matrix {
        let mut transposed = Matrix::zeros(self.columns, self.rows);
        ops::transpose(self, &mut transposed);
        transposed
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        let vec = unsafe { Vec::from_raw_parts(self.buffer, self.len, self.capacity) };
        let new_vec = vec.clone();
        mem::forget(vec);

        Matrix::from_vec(self.rows, self.columns, new_vec)
    }

    fn clone_from(&mut self, source: &Matrix) {
        let mut self_vec = unsafe { Vec::from_raw_parts(self.buffer, self.len, self.capacity) };
        let source_vec = unsafe { Vec::from_raw_parts(source.buffer, source.len, source.capacity) };

        self_vec.clone_from(&source_vec);

        self.rows = source.rows;
        self.columns = source.columns;
        self.buffer = self_vec.as_mut_ptr();
        self.capacity = self_vec.capacity();
        self.len = self_vec.len();

        mem::forget(self_vec);
        mem::forget(source_vec);
    }
}

impl Deref for Matrix {
    type Target = Vector;

    fn deref(&self) -> &Vector {
        let ptr_matrix = self as *const Matrix;

        unsafe {
            let ptr_usize: *const usize = mem::transmute(ptr_matrix);
            let ptr_vector: *const Vector = mem::transmute(ptr_usize.offset(2));
            &*ptr_vector
        }
    }
}

impl DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut Vector {
        let ptr_matrix = self as *mut Matrix;

        unsafe {
            let ptr_usize: *mut usize = mem::transmute(ptr_matrix);
            let ptr_vector: *mut Vector = mem::transmute(ptr_usize.offset(2));
            &mut *ptr_vector
        }
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        mem::drop(unsafe { Vec::from_raw_parts(self.buffer, self.len, self.capacity) });
    }
}

impl From<Matrix> for Vec<f32> {
    fn from(matrix: Matrix) -> Vec<f32> {
        let buffer = matrix.buffer;
        let capacity = matrix.capacity;
        let len = matrix.len;
        mem::forget(matrix);

        unsafe { Vec::from_raw_parts(buffer, len, capacity) }
    }
}

impl Index<usize> for Matrix {
    type Output = [f32];

    /// Returns the specified row of the matrix.
    fn index(&self, index: usize) -> &[f32] {
        unsafe {
            &slice::from_raw_parts(self.buffer, self.len)[index * self.columns..(index + 1) * self.columns]
        }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    /// Returns the specified element of the matrix.
    fn index(&self, (row, column): (usize, usize)) -> &f32 {
        unsafe {
            &slice::from_raw_parts(self.buffer, self.len)[row * self.columns + column]
        }
    }
}

impl IndexMut<usize> for Matrix {
    /// Returns the specified row of the matrix.
    fn index_mut(&mut self, index: usize) -> &mut [f32] {
        unsafe {
            &mut slice::from_raw_parts_mut(self.buffer, self.len)[index * self.columns..(index + 1) * self.columns]
        }
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    /// Returns the specified element of the matrix.
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut f32 {
        unsafe {
            &mut slice::from_raw_parts_mut(self.buffer, self.len)[row * self.columns + column]
        }
    }
}

impl<'a> Mul<&'a Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        let mut result = Matrix::zeros(self.rows, rhs.columns);
        ops::multiply(self, rhs, &mut result);
        result
    }
}

impl<'a> Mul<&'a Vector> for &'a Matrix {
    type Output = Vector;

    /// Treats `rhs` as a `rhs.len()`x1 matrix.
    fn mul(self, rhs: &Vector) -> Vector {
        let mut result = Matrix::zeros(self.columns, 1);

        let temp_matrix = Matrix {
            rows: rhs.len(),
            columns: 1,

            buffer: unsafe { mem::transmute(rhs.as_ptr()) },
            capacity: 0, // Unneeded
            len: rhs.len(),
        };
        ops::multiply(self, &temp_matrix, &mut result);
        mem::forget(temp_matrix);

        Vector::from_vec(result.into())
    }
}

impl<'a> Mul<&'a Matrix> for &'a Vector {
    type Output = Vector;

    /// Treats `self` as a 1x`self.len()` matrix.
    fn mul(self, rhs: &Matrix) -> Vector {
        let mut result = Matrix::zeros(1, rhs.columns());

        let temp_matrix = Matrix {
            rows: 1,
            columns: self.len(),

            buffer: unsafe { mem::transmute(self.as_ptr()) },
            capacity: 0, // Unneeded, because we'll never resize temp_matrix while it's alive
            len: self.len(),
        };
        ops::multiply(&temp_matrix, rhs, &mut result);
        mem::forget(temp_matrix);

        Vector::from_vec(result.into())
    }
}

unsafe impl Send for Matrix { }

pub mod ops;
