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

use std::fmt::{Debug, Display};
use std::hash::Hash;

/// This trait marks a ply.
///
/// Implementors should implement Hash manually, writing only a single u64 to the hasher.
///
/// # Example
///
/// For tic-tac-toe, we might have:
///
/// ```rust
/// # extern crate zero_sum;
/// # use zero_sum::Ply;
/// # #[derive(Clone, Debug, Hash, PartialEq)]
/// enum Mark { X, O }
///
/// # #[derive(Clone, Debug, Hash, PartialEq)]
/// struct Move {
///     mark: Mark,
///     coords: (usize, usize),
/// }
///
/// impl Ply for Move { }
/// # impl std::fmt::Display for Move { fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { Ok(()) } }
/// # fn main() { }
/// ```
pub trait Ply: Clone + Debug + Display + Hash + PartialEq { }
