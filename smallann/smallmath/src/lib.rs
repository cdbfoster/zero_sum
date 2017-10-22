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

#![feature(asm)]
#![feature(test)]

#[cfg(feature = "with_blas")]
extern crate blas;

#[cfg(feature = "with_simd")]
extern crate simd;

#[cfg(test)]
extern crate test;

pub use matrix::Matrix;
pub use vector::Vector;

pub mod matrix;
pub mod vector;
