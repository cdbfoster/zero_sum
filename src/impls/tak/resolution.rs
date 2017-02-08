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

use impls::tak::Color;
use resolution;

/// The ways a game can end.
pub enum Resolution {
    /// One player has completed a road.
    Road(Color),
    /// A player has run out of stones or the board is completely full,
    /// and one player has more flatstones.
    Flat(Color),
    /// A player has run out of stones or the board is completely full,
    /// and neither player has more flatstones.
    Draw,
}

impl resolution::Resolution for Resolution {
    /// Returns true if the value is either a `Road` or a `Flat`; false otherwise.
    fn is_win(&self) -> bool {
        match *self {
            Resolution::Road(_) |
            Resolution::Flat(_) => true,
            Resolution::Draw => false,
        }
    }

    /// Returns true if the value is a `Draw`.
    fn is_draw(&self) -> bool {
        match *self {
            Resolution::Draw => true,
            _ => false,
        }
    }
}
