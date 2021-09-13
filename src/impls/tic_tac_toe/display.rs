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

use std::fmt;

use impls::tic_tac_toe::{Board, Mark, Ply};

impl fmt::Display for Mark {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match *self {
            Mark::X => "X",
            Mark::O => "O",
        })
    }
}

impl fmt::Display for Ply {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}, {:?}", self.mark, (self.coordinates.0 + 1, self.coordinates.1 + 1))
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\n   1  2  3")?;
        for y in 0..3 {
            write!(f, "\n{} ", y + 1)?;
            for x in 0..3 {
                write!(f, "[{}]", match self.0[x + 3 * y] {
                    Some(mark) => Box::new(mark) as Box<fmt::Display>,
                    None => Box::new(" "),
                })?;
            }

        }
        Ok(())
    }
}
