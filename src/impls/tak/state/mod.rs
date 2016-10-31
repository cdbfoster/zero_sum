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

use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use impls::tak::{Color, Piece};

use self::metadata::Metadata;

#[derive(Debug, Eq, PartialEq)]
pub struct State {
    pub p1_flatstones: u8,
    pub p1_capstones: u8,

    pub p2_flatstones: u8,
    pub p2_capstones: u8,

    pub board: Vec<Vec<Vec<Piece>>>,
    pub ply_count: u16,

    metadata: Metadata,
}

impl State {
    pub fn new(board_size: usize) -> State {
        let (flatstone_count, capstone_count) = match board_size {
            3 => (10, 0),
            4 => (15, 0),
            5 => (21, 1),
            6 => (30, 1),
            7 => (40, 1),
            8 => (50, 2),
            s => panic!("Illegal board size: {}", s),
        };

        State {
            p1_flatstones: flatstone_count,
            p1_capstones: capstone_count,
            p2_flatstones: flatstone_count,
            p2_capstones: capstone_count,
            board: vec![vec![Vec::new(); board_size]; board_size],
            ply_count: 0,
            metadata: Metadata::new(board_size),
        }
    }

    pub fn from_tps(tps: &str) -> Option<State> { // XXX Return Result<State, String>
        if &tps[0..6] != "[TPS \"" || &tps[(tps.len() - 2)..] != "\"]" {
            return None;
        }

        let mut chars = tps[6..(tps.len() - 2)].chars();

        let mut x = 0;
        let mut y = 0;
        let mut board: Vec<Vec<Vec<Piece>>> = Vec::new();
        let mut piece_color = None;

        let mut p1_used_flatstones = 0;
        let mut p1_used_capstones = 0;

        let mut p2_used_flatstones = 0;
        let mut p2_used_capstones = 0;

        fn ensure_dimensions(board: &mut Vec<Vec<Vec<Piece>>>, x: usize, y: usize) {
            if x >= board.len() {
                for _ in board.len()..(x + 1) {
                    board.push(Vec::new());
                }
            }

            for column in board.iter_mut() {
                if y >= column.len() {
                    for _ in column.len()..(y + 1) {
                        column.push(Vec::new());
                    }
                }
            }
        }

        let mut next = chars.next();
        while next.is_some() {
            ensure_dimensions(&mut board, x, y);

            if let Some(color) = piece_color {
                let piece = match next {
                    Some('S') => Piece::StandingStone(color),
                    Some('C') => Piece::Capstone(color),
                    _ => Piece::Flatstone(color),
                };

                let (used_flatstones, used_capstones) = match color {
                    Color::White => (&mut p1_used_flatstones, &mut p1_used_capstones),
                    Color::Black => (&mut p2_used_flatstones, &mut p2_used_capstones),
                };

                match piece {
                    Piece::Capstone(_) => *used_capstones += 1,
                    _ => *used_flatstones += 1,
                }

                board[x][y].push(piece);

                piece_color = None;
                match next {
                    Some('S') |
                    Some('C') => next = chars.next(),
                    _ => (),
                }
            }

            match next {
                Some('x') => match chars.next() {
                    Some(c) => if c.is_digit(10) {
                        x += (c as u8 - 49) as usize;
                    } else if c == ',' {
                        x += 1;
                    } else if c == '/' {
                        x = 0;
                        y += 1;
                    } else if c == ' ' {
                        break;
                    } else {
                        return None;
                    },
                    _ => return None,
                },
                Some(',') => {
                    x += 1;
                },
                Some('/') => {
                    x = 0;
                    y += 1;
                },
                Some(' ') => break,
                Some('1') => piece_color = Some(Color::White),
                Some('2') => piece_color = Some(Color::Black),
                _ => return None,
            }

            next = chars.next();
        }

        let ply_count = {
            let player = match chars.next() {
                Some('1') => 0,
                Some('2') => 1,
                _ => return None,
            };

            chars.next();

            let turn_count = match u16::from_str(chars.as_str()) {
                Ok(c) => if c > 0 {
                    c - 1
                } else {
                    return None
                },
                _ => return None,
            };

            turn_count * 2 + player
        };

        for column in &mut board {
            column.reverse();
        }

        let mut state = State::new(board.len());
        state.p1_flatstones -= p1_used_flatstones;
        state.p1_capstones -= p1_used_capstones;
        state.p2_flatstones -= p2_used_flatstones;
        state.p2_capstones -= p2_used_capstones;
        state.board = board;
        state.ply_count = ply_count;
        state.metadata = Metadata::from_state(&state);

        Some(state)
    }
}

impl Clone for State {
    fn clone(&self) -> State {
        State {
            p1_flatstones: self.p1_flatstones,
            p1_capstones: self.p1_capstones,
            p2_flatstones: self.p2_flatstones,
            p2_capstones: self.p2_capstones,
            board: self.board.clone(),
            ply_count: self.ply_count,
            metadata: self.metadata.clone(),
        }
    }

    fn clone_from(&mut self, source: &State) {
        self.p1_flatstones = source.p1_flatstones;
        self.p1_capstones = source.p1_capstones;
        self.p2_flatstones = source.p2_flatstones;
        self.p2_capstones = source.p2_capstones;
        self.board.clone_from(&source.board);
        self.ply_count = source.ply_count;
        self.metadata.clone_from(&source.metadata);
    }
}

impl Hash for State {
    fn hash<H>(&self, state: &mut H) where H: Hasher {
        if self.ply_count % 2 == 0 {
            Color::White.hash(state);
        } else {
            Color::Black.hash(state);
        }
        self.metadata.p1_flatstones.hash(state);
        self.metadata.p2_flatstones.hash(state);
        self.metadata.standing_stones.hash(state);
        self.metadata.capstones.hash(state);
        self.metadata.p1_pieces.hash(state);
        self.metadata.p2_pieces.hash(state);
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let board_size = self.board.len();

        let column_widths = self.board.iter().map(|column| {
            column.iter().fold(6, |max, stack| {
                let stack_width = stack.iter().fold(0, |acc, piece| {
                    match *piece {
                        Piece::Flatstone(_) => acc + 1,
                        _ => acc + 2,
                    }
                }) + 3 + if !stack.is_empty() {
                    stack.len() - 1
                } else {
                    0
                };

                if max > stack_width { max } else { stack_width }
            })
        }).collect::<Vec<_>>();

        try!(write!(f, "\n Player 1: {:>2} flatstone{}", self.p1_flatstones,
            if self.p1_flatstones != 1 { "s" } else { "" }
        ));

        if self.p1_capstones > 0 {
            try!(write!(f, ", {} capstone{}", self.p1_capstones,
                if self.p1_capstones != 1 { "s" } else { "" }
            ));
        }

        try!(write!(f, "\n Player 2: {:>2} flatstone{}", self.p2_flatstones,
            if self.p2_flatstones != 1 { "s" } else { "" }
        ));

        if self.p2_capstones > 0 {
            try!(write!(f, ", {} capstone{}\n\n", self.p2_capstones,
                if self.p2_capstones != 1 { "s" } else { "" }
            ));
        } else {
            try!(write!(f, "\n\n"));
        }

        for row in (0..board_size).rev() {
            try!(write!(f, " {}   ", row + 1));

            for column in 0..board_size {
                let mut c = String::new();
                try!(write!(c, "["));

                for (index, piece) in self.board[column][row].iter().rev().enumerate() {
                    if index > 0 {
                        try!(write!(c, " "));
                    }

                    try!(write!(c, "{}", match piece.get_color() {
                        Color::White => "W",
                        Color::Black => "B",
                    }));

                    match *piece {
                        Piece::StandingStone(_) => { try!(write!(c, "S")); },
                        Piece::Capstone(_) => { try!(write!(c, "C")); },
                        _ => (),
                    }
                }

                try!(write!(c, "]"));

                try!(write!(f, "{:<width$}", c, width = column_widths[column]));
            }

            try!(write!(f, "\n"));
        }

        try!(write!(f, "\n     "));

        for (index, column_width) in column_widths.iter().enumerate() {
            try!(write!(f, "{:<width$}", (index as u8 + 97) as char, width = column_width));
        }

        write!(f, "\n")
    }
}

pub mod evaluation;
mod extrapolation;
mod metadata;
mod state;
