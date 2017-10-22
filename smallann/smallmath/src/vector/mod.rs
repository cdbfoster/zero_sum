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
use std::ops::{Add, AddAssign, Div, DivAssign, Drop, Deref, DerefMut, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use std::ptr;
use std::slice;

/// A vector.  `Derefs` into `&[f32]`, and can be converted into a `Vec<f32>` via `into()`.
#[repr(C)]
pub struct Vector {
    buffer: *mut f32,
    capacity: usize,
    len: usize,
}

impl Vector {
    /// Constructs a vector from a `Vec<f32>`.
    pub fn from_vec(mut values: Vec<f32>) -> Vector {
        let buffer = values.as_mut_ptr();
        let capacity = values.capacity();
        let len = values.len();
        mem::forget(values);

        Vector {
            buffer: buffer,
            capacity: capacity,
            len: len,
        }
    }

    pub fn zeros(size: usize) -> Vector {
        Vector::from_vec(vec![0.0; size])
    }

    pub fn resize(&mut self, size: usize) {
        if size > self.capacity {
            let mut vec = unsafe { Vec::from_raw_parts(self.buffer, self.len, self.capacity) };
            vec.reserve(size - self.capacity);

            self.buffer = vec.as_mut_ptr();
            let new_capacity = vec.capacity();

            mem::forget(vec);

            unsafe {
                ptr::write_bytes(self.buffer.offset(self.capacity as isize), 0, new_capacity - self.capacity);
            }
            self.capacity = new_capacity;
        }
        self.len = size;
    }

    /// Clears the vector to all zeros.
    pub fn zero(&mut self) {
        unsafe {
            ptr::write_bytes(self.buffer, 0, self.len);
        }
    }
}

impl<'a> Add<f32> for &'a Vector {
    type Output = Vector;

    /// Offsets each value in the vector by `rhs`.  This allocates a new vector.
    fn add(self, rhs: f32) -> Vector {
        let mut result = Vector::zeros(self.len());

        ops::offset(self, rhs, &mut result);

        result
    }
}

impl<'a> Add<&'a Vector> for &'a Vector {
    type Output = Vector;

    /// Performs the element-wise addition of the vector and `rhs`.  The vectors must be the same length.  This allocates a new vector.
    fn add(self, rhs: &Vector) -> Vector {
        let mut result = Vector::zeros(self.len());

        ops::add(self, rhs, &mut result);

        result
    }
}

impl<'a> Add<&'a [f32]> for &'a Vector {
    type Output = Vector;

    /// Performs the element-wise addition of the vector and the slice `rhs`.  The vector and the slice must be the same length.  This allocates a new vector.
    fn add(self, rhs: &[f32]) -> Vector {
        let mut result = Vector::zeros(self.len());
        let temp_vector = Vector {
            buffer: unsafe { mem::transmute(rhs.as_ptr()) },
            capacity: 0,
            len: rhs.len(),
        };

        ops::add(self, &temp_vector, &mut result);

        mem::forget(temp_vector);

        result
    }
}

impl AddAssign<f32> for Vector {
    /// Offsets each value in the vector by `rhs`.
    fn add_assign(&mut self, rhs: f32) {
        ops::offset_assign(self, rhs);
    }
}

impl<'a> AddAssign<&'a Vector> for Vector {
    /// Performs the element-wise addition of the vector and `rhs`.  The vectors must be the same length.
    fn add_assign(&mut self, rhs: &Vector) {
        ops::add_assign(self, rhs);
    }
}

impl<'a> AddAssign<&'a [f32]> for Vector {
    /// Performs the element-wise addition of the vector and the slice `rhs`.  The vector and the slice must be the same length.
    fn add_assign(&mut self, rhs: &[f32]) {
        let temp_vector = Vector {
            buffer: unsafe { mem::transmute(rhs.as_ptr()) },
            capacity: 0,
            len: rhs.len(),
        };

        ops::add_assign(self, &temp_vector);

        mem::forget(temp_vector);
    }
}

impl Clone for Vector {
    fn clone(&self) -> Vector {
        let vec = unsafe { Vec::from_raw_parts(self.buffer, self.len, self.capacity) };
        let new_vec = vec.clone();
        mem::forget(vec);

        Vector::from_vec(new_vec)
    }

    fn clone_from(&mut self, source: &Vector) {
        let mut self_vec = unsafe { Vec::from_raw_parts(self.buffer, self.len, self.capacity) };
        let source_vec = unsafe { Vec::from_raw_parts(source.buffer, source.len, source.capacity) };

        self_vec.clone_from(&source_vec);

        self.buffer = self_vec.as_mut_ptr();
        self.capacity = self_vec.capacity();
        self.len = self_vec.len();

        mem::forget(self_vec);
        mem::forget(source_vec);
    }
}

impl Deref for Vector {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        unsafe {
            slice::from_raw_parts(self.buffer, self.len)
        }
    }
}

impl DerefMut for Vector {
    fn deref_mut(&mut self) -> &mut [f32] {
        unsafe {
            slice::from_raw_parts_mut(self.buffer, self.len)
        }
    }
}

impl<'a> Div<f32> for &'a Vector {
    type Output = Vector;

    fn div(self, rhs: f32) -> Vector {
        let mut result = Vector::zeros(self.len());

        ops::scale(self, 1.0 / rhs, &mut result);

        result
    }
}

impl<'a> Div<&'a Vector> for &'a Vector {
    type Output = Vector;

    fn div(self, rhs: &Vector) -> Vector {
        let mut result = Vector::zeros(self.len());

        ops::divide(self, rhs, &mut result);

        result
    }
}

impl<'a> Div<&'a [f32]> for &'a Vector {
    type Output = Vector;

    fn div(self, rhs: &[f32]) -> Vector {
        let mut result = Vector::zeros(self.len());
        let temp_vector = Vector {
            buffer: unsafe { mem::transmute(rhs.as_ptr()) },
            capacity: 0,
            len: rhs.len(),
        };

        ops::divide(self, &temp_vector, &mut result);

        mem::forget(temp_vector);

        result
    }
}

impl DivAssign<f32> for Vector {
    fn div_assign(&mut self, rhs: f32) {
        ops::scale_assign(self, 1.0 / rhs);
    }
}

impl<'a> DivAssign<&'a Vector> for Vector {
    fn div_assign(&mut self, rhs: &Vector) {
        ops::divide_assign(self, rhs);
    }
}

impl<'a> DivAssign<&'a [f32]> for Vector {
    fn div_assign(&mut self, rhs: &[f32]) {
        let temp_vector = Vector {
            buffer: unsafe { mem::transmute(rhs.as_ptr()) },
            capacity: 0,
            len: rhs.len(),
        };

        ops::divide_assign(self, &temp_vector);

        mem::forget(temp_vector);
    }
}

impl Drop for Vector {
    fn drop(&mut self) {
        mem::drop(unsafe { Vec::from_raw_parts(self.buffer, self.len, self.capacity) });
    }
}

impl From<Vector> for Vec<f32> {
    fn from(vector: Vector) -> Vec<f32> {
        let buffer = vector.buffer;
        let capacity = vector.capacity;
        let len = vector.len;
        mem::forget(vector);

        unsafe { Vec::from_raw_parts(buffer, len, capacity) }
    }
}

impl Index<usize> for Vector {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        unsafe {
            &slice::from_raw_parts(self.buffer, self.len)[index]
        }
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        unsafe {
            &mut slice::from_raw_parts_mut(self.buffer, self.len)[index]
        }
    }
}

impl<'a> Mul<f32> for &'a Vector {
    type Output = Vector;

    fn mul(self, rhs: f32) -> Vector {
        let mut result = Vector::zeros(self.len());

        ops::scale(self, rhs, &mut result);

        result
    }
}

impl<'a> Mul<&'a Vector> for &'a Vector {
    type Output = Vector;

    fn mul(self, rhs: &Vector) -> Vector {
        let mut result = Vector::zeros(self.len());

        ops::multiply(self, rhs, &mut result);

        result
    }
}

impl<'a> Mul<&'a [f32]> for &'a Vector {
    type Output = Vector;

    fn mul(self, rhs: &[f32]) -> Vector {
        let mut result = Vector::zeros(self.len());
        let temp_vector = Vector {
            buffer: unsafe { mem::transmute(rhs.as_ptr()) },
            capacity: 0,
            len: rhs.len(),
        };

        ops::multiply(self, &temp_vector, &mut result);

        mem::forget(temp_vector);

        result
    }
}

impl MulAssign<f32> for Vector {
    fn mul_assign(&mut self, rhs: f32) {
        ops::scale_assign(self, rhs);
    }
}

impl<'a> MulAssign<&'a Vector> for Vector {
    fn mul_assign(&mut self, rhs: &Vector) {
        ops::multiply_assign(self, rhs);
    }
}

impl<'a> MulAssign<&'a [f32]> for Vector {
    fn mul_assign(&mut self, rhs: &[f32]) {
        let temp_vector = Vector {
            buffer: unsafe { mem::transmute(rhs.as_ptr()) },
            capacity: 0,
            len: rhs.len(),
        };

        ops::multiply_assign(self, &temp_vector);

        mem::forget(temp_vector);
    }
}

impl<'a> Sub<f32> for &'a Vector {
    type Output = Vector;

    fn sub(self, rhs: f32) -> Vector {
        let mut result = Vector::zeros(self.len());

        ops::offset(self, -rhs, &mut result);

        result
    }
}

impl<'a> Sub<&'a Vector> for &'a Vector {
    type Output = Vector;

    fn sub(self, rhs: &Vector) -> Vector {
        let mut result = Vector::zeros(self.len());

        ops::subtract(self, rhs, &mut result);

        result
    }
}

impl<'a> Sub<&'a [f32]> for &'a Vector {
    type Output = Vector;

    fn sub(self, rhs: &[f32]) -> Vector {
        let mut result = Vector::zeros(self.len());
        let temp_vector = Vector {
            buffer: unsafe { mem::transmute(rhs.as_ptr()) },
            capacity: 0,
            len: rhs.len(),
        };

        ops::subtract(self, &temp_vector, &mut result);

        mem::forget(temp_vector);

        result
    }
}

impl SubAssign<f32> for Vector {
    fn sub_assign(&mut self, rhs: f32) {
        ops::offset_assign(self, -rhs);
    }
}

impl<'a> SubAssign<&'a Vector> for Vector {
    fn sub_assign(&mut self, rhs: &Vector) {
        ops::subtract_assign(self, rhs);
    }
}

impl<'a> SubAssign<&'a [f32]> for Vector {
    fn sub_assign(&mut self, rhs: &[f32]) {
        let temp_vector = Vector {
            buffer: unsafe { mem::transmute(rhs.as_ptr()) },
            capacity: 0,
            len: rhs.len(),
        };

        ops::subtract_assign(self, &temp_vector);

        mem::forget(temp_vector);
    }
}

pub mod ops;
