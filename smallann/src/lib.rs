//
// This file is part of smallann.
//
// smallann is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// smallann is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with smallann. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright 2017 Chris Foster
//

#![feature(asm)]
#![feature(test)]

extern crate rand;

#[cfg(feature = "with_simd")]
extern crate simd;

extern crate smallmath;

pub use smallmath::{Matrix, matrix, Vector, vector};

#[cfg(test)]
extern crate test;

pub use ann::Ann;
pub use serialization::{File, Serializable};

pub mod activation_function;
mod ann;
pub mod gradient_descent;
pub mod layer;
pub mod loss_function;
mod serialization;
