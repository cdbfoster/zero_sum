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
// Copyright 2016 Chris Foster
//

//! An analysis engine for zero-sum games.
//!
//! This crate provides a number of traits that can be used to facilitate the
//! implementation of a zero-sum game, and to allow the analysis thereof.
//!
//! # Usage
//!
//! This crate is [on crates.io](https://crates.io/crates/zero_sum) and can be
//! used by adding `zero_sum` to the dependencies in your project's `Cargo.toml`.
//!
//! ```toml
//! [dependencies]
//! zero_sum = "0.2"
//! ```
//!
//! and add this to your crate root:
//!
//! ```rust
//! #[macro_use]
//! extern crate zero_sum;
//! # fn main() { }
//! ```
//!
//! # Implementation
//!
//! The three basic traits are `Ply`, `Resolution`, and `State`.  These form
//! the basic building blocks of any zero-sum game.
//!
//! In order to provide analysis, one must also create an evaluation type
//! (usually a tuple wrapper around a numeric type, i.e. `struct Eval(i32);`)
//! with `analysis::Evaluation`, and implement `analysis::Evaluatable` and
//! `analysis::Extrapolatable` on the `State` type.
//!
//! # Example
//!
//! A working example can be found in [examples/tic_tac_toe.rs](https://github.com/cdbfoster/zero_sum/blob/master/examples/tic_tac_toe.rs).

#![feature(test)]

extern crate fnv;
extern crate test;

#[cfg(feature = "with_tak")]
#[macro_use]
extern crate lazy_static;

#[macro_use]
pub mod analysis;

pub use self::ply::Ply;
pub use self::resolution::Resolution;
pub use self::state::State;

#[cfg(feature = "with_tak")]
pub mod impls;

mod ply;
mod resolution;
mod state;
