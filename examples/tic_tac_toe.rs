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

extern crate zero_sum;

use std::io::{self, Write};

use zero_sum::analysis::search::Search;
use zero_sum::impls::tic_tac_toe::*;
use zero_sum::State;

fn main() {
    let mut game = 1;

    loop {
        let mut board = Board::new();
        let mut ai = zero_sum::analysis::search::PvSearch::<Evaluation, Board, Ply, Resolution>::new();

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
