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

use std::fmt::Debug;

/// A game's resolution.
///
/// This is often an `enum` that represents each ending a game can have.
///
/// # Example
///
/// For tic-tac-toe, we might have:
///
/// ```rust
/// # extern crate zero_sum;
/// # use zero_sum::Resolution;
/// # #[derive(PartialEq)]
/// enum Mark { X, O }
///
/// # #[derive(PartialEq)]
/// enum End {
///     Win(Mark),
///     CatsGame,
/// }
///
/// impl Resolution for End {
///     fn get_winner(&self) -> Option<u8> {
///         match *self {
///             End::Win(Mark::X) => Some(0),
///             End::Win(Mark::O) => Some(1),
///             _ => None,
///         }
///     }
///     fn is_draw(&self) -> bool { if *self == End::CatsGame { true } else { false } }
/// }
/// # fn main() { }
/// ```
pub trait Resolution: Debug {
    /// Returns the index of the winning player if this `Resolution` represents a win.
    fn get_winner(&self) -> Option<u8>;
    fn is_draw(&self) -> bool;
}
