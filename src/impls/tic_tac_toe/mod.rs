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

use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum Mark {
    X,
    O,
}

#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Ply {
    pub mark: Mark,
    pub coordinates: (usize, usize),
}

#[derive(Debug)]
pub enum Resolution {
    Win(Mark),
    CatsGame,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Board(pub [Option<Mark>; 9], pub u8);

impl Board {
    pub fn new() -> Board {
        Board([None; 9], 0)
    }

    pub fn next_mark(&self) -> Mark {
        if self.1 % 2 == 0 {
            Mark::X
        } else {
            Mark::O
        }
    }
}

impl Hash for Board {
    fn hash<H>(&self, state: &mut H) where H: Hasher {
        self.0.hash(state);
        if self.1 % 2 == 0 {
            Mark::X.hash(state);
        } else {
            Mark::O.hash(state);
        }
    }
}

pub use self::zero_sum::Evaluation;

mod display;
mod zero_sum;
