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

use std::fmt::Display;
use std::hash::Hash;

use super::{Ply, Resolution};

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
/// impl State<Move, End> for Board {
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
/// # impl Resolution for End { fn is_win(&self) -> bool { false } fn is_draw(&self) -> bool { false } }
/// # impl std::fmt::Display for Move { fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { Ok(()) } }
/// # impl std::fmt::Display for Board { fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { Ok(()) } }
/// # fn main() { }
/// ```
pub trait State<P, R>: Clone + Display + Eq + Hash + PartialEq where
    P: Ply,
    R: Resolution {
    /// Executes a ply on the state, storing the resultant state in the preallocated `next`.
    /// It is recommended to implement `Clone` on the `State` implementor manually,
    /// to take advantage of `Clone`'s `clone_from` method in order to avoid costly
    /// allocations during a speed-critical search.
    fn execute_ply_preallocated(&self, ply: &P, next: &mut Self)-> Result<(), String>;

    /// Returns `None` if the game has not reached a conclusion.
    fn check_resolution(&self) -> Option<R>;

    /// Clones the state and then calls `execute_ply_preallocated`.
    fn execute_ply(&self, ply: &P) -> Result<Self, String> {
        let mut next = self.clone();
        match self.execute_ply_preallocated(ply, &mut next) {
            Ok(_) => Ok(next),
            Err(error) => Err(error),
        }
    }

    /// Executes each ply in `plies` on the result of the previous ply.
    fn execute_plies(&self, plies: &[P]) -> Result<Self, String> {
        let mut state = self.clone();
        for ply in plies {
            match state.execute_ply(ply) {
                Ok(next) => state = next,
                Err(error) => return Err(format!("Error executing plies: {}, {}", ply, error)),
            }
        }
        Ok(state)
    }
}
