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

//! An analysis engine for zero-sum games.
//!
//! This crate provides a number of traits that can be used to facilitate the
//! implementation of a zero-sum game, and to allow the analysis thereof.
//!
//! Also provided through the use of optional features are implementations
//! for tic-tac-toe and the game of [tak](http://cheapass.com/tak/).
//!
//! # Usage
//!
//! This crate is [on crates.io](https://crates.io/crates/zero_sum) and can be
//! used by adding `zero_sum` to the dependencies in your project's `Cargo.toml`.
//!
//! ```toml
//! [dependencies]
//! zero_sum = "1.2"
//! ```
//!
//! and add this to your crate root:
//!
//! ```rust
//! extern crate zero_sum;
//! # fn main() { }
//! ```
//!
//! If you want to implement the library, you'll need to include a `#[macro_use]`
//! line before `extern crate zero_sum;`
//!
//! If you want to use one of the implementations provided inside the `zero_sum::impls`
//! module, you'll need to specify the appropriate features in your project's `Cargo.toml`:
//!
//! ```toml
//! [features]
//! default = ["zero_sum/with_tak"]
//! ```
//!
//! for instance, to include the `tak` module.
//!
//! # Implementation
//!
//! The three basic traits are `Ply`, `Resolution`, and `State`.  These form
//! the basic building blocks of any zero-sum game.
//!
//! In order to provide analysis, one must also create an evaluator type with
//! `analysis::Evaluator` that has an associated evaluation type that implements
//! `analysis::Evaluation` (usually a tuple wrapper around a numeric type, i.e.
//! `struct Eval(i32);`).  Finally, implement `analysis::Extrapolatable` on the
//! `State` type.
//!
//! # Example
//!
//! The provided tic-tac-toe implementation is very simple and a usage example can
//! be found in [examples/tic_tac_toe.rs](https://github.com/cdbfoster/zero_sum/blob/master/examples/tic_tac_toe.rs).

#![feature(test)]

extern crate fnv;

#[cfg(test)]
extern crate test;

#[cfg(feature = "with_tak")]
#[macro_use]
extern crate lazy_static;

#[cfg(feature = "with_tak")]
extern crate rand;

#[cfg(feature = "with_tak_ann")]
extern crate cblas;
#[cfg(feature = "with_tak_ann")]
extern crate openblas_src;
#[cfg(feature = "with_tak_ann")]
extern crate smallann;

#[macro_use]
pub mod analysis;

pub use self::ply::Ply;
pub use self::resolution::Resolution;
pub use self::state::State;

#[cfg(any(feature = "with_tak", feature = "with_tic_tac_toe"))]
pub mod impls;

mod ply;
mod resolution;
mod state;
mod util;
