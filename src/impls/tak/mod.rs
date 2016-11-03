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

//! The game of [tak](http://cheapass.com/tak/).

/// The colors of the players.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn flip(&self) -> Color {
        match *self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

/// The types of pieces.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Piece {
    Flatstone(Color),
    StandingStone(Color),
    Capstone(Color),
}

impl Piece {
    pub fn get_color(&self) -> Color {
        match *self {
            Piece::Flatstone(color) |
            Piece::StandingStone(color) |
            Piece::Capstone(color) => color,
        }
    }
}

/// The slidable directions.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    pub fn to_offset(&self) -> (i8, i8) {
        match *self {
            Direction::North => (0, 1),
            Direction::East => (1, 0),
            Direction::South => (0, -1),
            Direction::West => (-1, 0),
        }
    }
}

pub use self::ply::Ply;
pub use self::resolution::Resolution;
pub use self::state::{Evaluation, State};

mod ply;
mod resolution;
mod state;
