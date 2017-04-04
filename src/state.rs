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

use std::fmt::Display;
use std::hash::Hash;

use ply::Ply;
use resolution::Resolution;

/// The state of the game.
///
/// This should represent everything that makes up a single moment of the game, i.e. in chess,
/// this would be the board and all of its pieces, the turn number, etc.
///
/// However, if the implementor of this trait does store data that changes or increments every turn,
/// like a turn number, it is recommended to implement `Hash` manually and to exclude that data from
/// the hash, perhaps simplifying it into the next player to move.  This is in order to allow
/// the struct to benefit from certain search optimization techniques -- primarily a transposition
/// table.
///
/// # Example
///
/// For tic-tac-toe, we might have:
///
/// ```rust
/// # extern crate zero_sum;
/// # use std::hash::{Hash, Hasher};
/// # use zero_sum::{Ply, Resolution, State};
/// # #[derive(Clone, Copy, Eq, Hash, PartialEq)]
/// enum Mark { X, O }
/// # #[derive(Clone, Debug, Hash, PartialEq)]
/// struct Move { /* ... */ }
/// enum End { /* ... */ }
///
/// # #[derive(Clone, Eq, PartialEq)]
/// struct Board([Option<Mark>; 9], u8); // The board and the turn number
///
/// impl State for Board {
///     type Ply = Move;
///     type Resolution = End;
///
///     fn get_ply_count(&self) -> usize {
///         self.1 as usize
///     }
///
///     fn execute_ply_preallocated(&self, ply: &Move, next: &mut Board) -> Result<(), String> {
///         // ...
///         # Ok(())
///     }
///
///     fn check_resolution(&self) -> Option<End> {
///         // ...
///         # None
///     }
/// }
///
/// impl Hash for Board {
///     fn hash<H>(&self, state: &mut H) where H: Hasher {
///         self.0.hash(state);
///         if self.1 % 2 == 0 {
///             Mark::X.hash(state);
///         } else {
///             Mark::O.hash(state);
///         }
///     }
/// }
/// # impl Ply for Move { }
/// # impl Resolution for End { fn get_winner(&self) -> Option<u8> { None } fn is_draw(&self) -> bool { false } }
/// # impl std::fmt::Display for Move { fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { Ok(()) } }
/// # impl std::fmt::Display for Board { fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { Ok(()) } }
/// # fn main() { }
/// ```
pub trait State: Clone + Display + Eq + Hash + PartialEq {
    type Ply: Ply;
    type Resolution: Resolution;

    /// Returns the number of plies that have passed in the game.
    fn get_ply_count(&self) -> usize;

    /// Executes the given ply on this state.  Pass `None` to execute a null move.
    fn execute_ply(&mut self, ply: Option<&Self::Ply>) -> Result<(), String>;

    /// Reverts the given ply from the state.  Pass `None` to revert a null move.
    fn revert_ply(&mut self, ply: Option<&Self::Ply>) -> Result<(), String>;

    /// Returns `None` if the game has not reached a conclusion.
    fn check_resolution(&self) -> Option<Self::Resolution>;

    /// Returns true if the state is in a good place to allow the null move search optimization.
    /// This is optional to implement, returning a default of `false`.
    fn null_move_allowed(&self) -> bool {
        false
    }

    /// Executes each ply in `plies` on the result of the previous ply.
    fn execute_plies(&mut self, plies: &[Self::Ply]) -> Result<(), String> {
        for ply in plies {
            if let Err(error) = self.execute_ply(Some(ply)) {
                return Err(format!("Error executing plies: {}, {}", ply, error));
            }
        }
        Ok(())
    }
}
