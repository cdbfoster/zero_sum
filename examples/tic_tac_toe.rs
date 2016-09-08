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

#![feature(box_syntax)]
#![feature(step_by)]

#[macro_use]
extern crate zero_sum;

use std::fmt;
use std::i8;
use std::io::{self, Write};
use std::ops::{Add, Div, Mul, Neg, Sub};

use zero_sum::ai::Evaluation;
use zero_sum::ai::search::Search;
use zero_sum::State;

// Tic-Tac-Toe types

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
enum Mark {
    X,
    O,
}

#[derive(Clone, Debug, Hash, PartialEq)]
struct Ply {
    mark: Mark,
    coordinates: (usize, usize),
}

#[derive(Debug)]
enum Resolution {
    Win(Mark),
    CatsGame,
}

#[derive(Clone, Debug)]
struct Board([Option<Mark>; 9], u8);

impl Board {
    fn new() -> Board {
        Board([None; 9], 0)
    }

    fn next_mark(&self) -> Mark {
        if self.1 % 2 == 0 {
            Mark::X
        } else {
            Mark::O
        }
    }
}

// Display implementations

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
        try!(write!(f, "\n   1  2  3"));
        for y in 0..3 {
            try!(write!(f, "\n{} ", y + 1));
            for x in 0..3 {
                try!(write!(f, "[{}]", match self.0[x + 3 * y] {
                    Some(mark) => box mark as Box<fmt::Display>,
                    None => box " ",
                }));
            }

        }
        Ok(())
    }
}

// zero_sum implementations

impl zero_sum::Ply for Ply { }

impl zero_sum::Resolution for Resolution {
    fn is_win(&self) -> bool {
        if let Resolution::Win(_) = *self { true } else { false }
    }

    fn is_draw(&self) -> bool {
        if let Resolution::CatsGame = *self { true } else { false }
    }
}

impl zero_sum::State<Ply, Resolution> for Board {
    fn execute_ply_preallocated(&self, ply: &Ply, next: &mut Board) -> Result<(), String> {
        if ply.coordinates.0 >= 3 || ply.coordinates.1 >= 3 {
            return Err(String::from("Coordinates out of bounds"));
        }

        let index = ply.coordinates.0 + 3 * ply.coordinates.1;

        if self.0[index].is_some() {
            return Err(String::from("Space already occupied"));
        }

        next.0 = self.0;
        next.0[index] = Some(ply.mark);
        next.1 = self.1 + 1;
        Ok(())
    }

    fn check_resolution(&self) -> Option<Resolution> {
        fn get_lines(b: &[Option<Mark>; 9]) -> [[&Option<Mark>; 3]; 8] {
            [[&b[0], &b[1], &b[2]],
             [&b[3], &b[4], &b[5]],
             [&b[6], &b[7], &b[8]],
             [&b[0], &b[3], &b[6]],
             [&b[1], &b[4], &b[7]],
             [&b[2], &b[5], &b[8]],
             [&b[0], &b[4], &b[8]],
             [&b[2], &b[4], &b[6]]]
        }

        if let Some(Some(mark)) = get_lines(&self.0)
            .iter()
            .map(|line| if line[0] == line[1] && line[0] == line[2] { *line[0] } else { None })
            .find(|r| r.is_some()) {
            return Some(Resolution::Win(mark))
        }

        if self.0.iter().find(|space| space.is_none()).is_none() {
            Some(Resolution::CatsGame)
        } else {
            None
        }
    }
}

impl zero_sum::ai::Extrapolatable<Ply> for Board {
    fn extrapolate(&self) -> Vec<Ply> {
        let next_mark = self.next_mark();
        self.0.iter().enumerate().filter_map(|(index, space)| if space.is_none() {
            Some(Ply {
                mark: next_mark,
                coordinates: (index % 3, index / 3),
            })
        } else {
            None
        }).collect::<Vec<Ply>>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct Eval(i8);

prepare_evaluation_tuple!(Eval); // Implements arithmetic operators and display in terms of the inner type

impl zero_sum::ai::Evaluation for Eval {
    fn null() -> Eval { Eval(0) }
    fn epsilon() -> Eval { Eval(1) }
    fn win() -> Eval { Eval(14) }
    fn max() -> Eval { Eval(i8::MAX) }
    fn is_win(&self) -> bool { self.0.abs() >= 5 }
}

impl zero_sum::ai::Evaluatable<Eval> for Board {
    fn evaluate(&self) -> Eval {
        let next_mark = self.next_mark();

        if let Some(Resolution::Win(mark)) = self.check_resolution() {
            if mark == next_mark {
                Eval(Eval::win().0 - self.1 as i8)
            } else {
                -Eval(Eval::win().0 - self.1 as i8)
            }
        } else {
            // Weight the corners.
            // This doesn't matter at all with regards to perfect play,
            // but there are more ways a human can mess up if we start in the corner.
            let x_corners = (0..9).step_by(2).filter(|&x| x != 4 && self.0[x] == Some(Mark::X)).count() as i8;
            let o_corners = (0..9).step_by(2).filter(|&x| x != 4 && self.0[x] == Some(Mark::O)).count() as i8;

            if next_mark == Mark::X {
                Eval(x_corners - o_corners)
            } else {
                Eval(o_corners - x_corners)
            }
        }
    }
}

fn main() {
    let mut game = 1;

    loop {
        let mut board = Board::new();
        let mut ai = zero_sum::ai::search::PvSearch::<Eval, Board, Ply, Resolution>::with_depth(9);

        println!("--------------------");

        if game % 2 == 1 {
            println!("Human goes first!");
        } else {
            println!("Computer goes first!");
        }

        'game: loop {
            println!("{}\n", board);

            match board.check_resolution() {
                Some(Resolution::Win(mark)) => {
                    println!("{} wins!\n", mark);
                    break 'game;
                },
                Some(Resolution::CatsGame) => {
                    println!("Cat's game!\n");
                    break 'game;
                },
                None => (),
            }

            let ply = if (game + board.1 as usize) % 2 == 1 {
                println!("Human's turn:");

                fn get_coordinate(prompt: &str) -> usize {
                    loop {
                        print!("{}", prompt);
                        io::stdout().flush().ok();
                        let mut input = String::new();
                        if let Ok(_) = io::stdin().read_line(&mut input) {
                            if let Ok(coordinate) = input.trim().parse::<usize>() {
                                if coordinate > 0 && coordinate <= 3 {
                                    return coordinate;
                                }
                            }
                        }
                    }
                }

                let x = get_coordinate("X coordinate (1 - 3): ");
                let y = get_coordinate("Y coordinate (1 - 3): ");

                Ply {
                    mark: board.next_mark(),
                    coordinates: (x - 1, y - 1),
                }
            } else {
                println!("Computer's turn:");

                ai.search(&board, None).principal_variation[0].clone()
            };

            match board.execute_ply(&ply) {
                Ok(next) => board = next,
                Err(error) => println!("Error: {}", error),
            }
        }

        game += 1;
    }
}
