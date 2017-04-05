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

extern crate rand;
extern crate rusqlite;
extern crate zero_sum;

use std::cmp;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::str::FromStr;

use rand::{Rng, thread_rng};

use zero_sum::State as StateTrait;
use zero_sum::analysis::{Evaluation, Evaluator, Extrapolatable};
use zero_sum::impls::tak::{Color, Direction, Piece, Ply, State};
use zero_sum::impls::tak::evaluator::{AnnEvaluator, StaticEvaluator};

fn main() {
    let players = vec![
        "TakticianBot",
        "TakkerusBot",
        "fwwwwibib",
        "NohatCoder",
        "SultanPepper",
        "Ally",
        "Turing",
        "Simmon",
        "Syme",
        "applemonkeyman",
        "Abyss",
        "Gray_Mouser",
        "UnSweet",
        "Tayacan",
    ];
    let sample_spacing = 2;
    let samples_per_base = 10;
    let label = true;
    let normalize_range = true;
    let file_prefix = String::from("training");

    let connection = rusqlite::Connection::open("games_anon.db").unwrap();

    println!("Reading games...");
    let games = get_games(&connection, 5, Some(&players));
    println!("  Done. Read {} games.", games.len());

    println!("Generating training positions...");
    let mut states = HashSet::new();
    for (number, game) in games.iter().enumerate() {
        let sample_size = cmp::max(game.plies.len() / sample_spacing, 1);
        let start_index = thread_rng().gen_range(0, cmp::min(sample_spacing, game.plies.len()));

        for i in 0..sample_size {
            let base = game.state_at((start_index + i * sample_spacing) % game.plies.len());

            if base.check_resolution().is_some() {
                continue;
            }

            let mut next_plies = base.extrapolate();
            thread_rng().shuffle(&mut next_plies);

            for _ in 0..samples_per_base {
                let mut next = base.clone();
                if next.execute_ply(next_plies.pop().as_ref()).is_ok() && next.check_resolution().is_none() {
                    states.insert(next);
                }
            }
        }

        if number % (games.len() / 10) == 0 {
            println!("  {}%", 10 * number / (games.len() / 10));
        }
    }
    println!("  Done. Generated {} training positions.", states.len());

    println!("Writing to file...");
    if let Ok(mut file) = OpenOptions::new().write(true).truncate(true).create(true).open(format!("{}_positions", file_prefix)) {
        for state in &states {
            write!(&mut file, "{}\n", state.to_tps()).ok();
        }
    } else {
        println!("Cannot write file: {}", format!("{}_positions", file_prefix));
    }
    println!("  Done.");

    if label {
        println!("Labeling training positions...");
        let evaluator = StaticEvaluator;
        let mut labels = states.iter().map(|s| evaluator.evaluate(s)).collect::<Vec<_>>();

        if normalize_range {
            let extent = labels.iter().map(|evaluation| evaluation.0.abs()).max().unwrap() as f32;
            println!("  Normalizing to {}.", extent);

            for label in &mut labels {
                *label = <StaticEvaluator as Evaluator>::Evaluation::new((label.0 as f32 / extent * <AnnEvaluator as Evaluator>::Evaluation::win().0 as f32) as i32);
            }
        }
        println!("  Done. Labeled {} training positions.", labels.len());

        println!("Writing to file...");
        if let Ok(mut file) = OpenOptions::new().write(true).truncate(true).create(true).open(format!("{}_labels", file_prefix)) {
            for label in &labels {
                write!(&mut file, "{}\n", label).ok();
            }
        } else {
            println!("Cannot write file: {}", format!("{}_labels", file_prefix));
        }
        println!("  Done.");
    }
}

#[derive(Clone, Debug)]
struct Game {
    id: usize,
    date: u64,
    size: usize,
    p1: String,
    p2: String,
    plies: Vec<Ply>,
    result: String,
}

impl Game {
    fn state_at(&self, ply_number: usize) -> State {
        State::from_plies(self.size, &self.plies[0..ply_number + 1]).unwrap()
    }
}

/// Grabs games of the specified size, from specified players from the database.  Skips empty games.
fn get_games(connection: &rusqlite::Connection, size: usize, from: Option<&[&str]>) -> Vec<Game> {
    let mut query = String::from("select id, date, size, player_white, player_black, notation, result from games");

    query += &format!(" where size = {}", size);

    if let Some(from) = from {
        if !from.is_empty() {
            for (index, player) in from.iter().enumerate() {
                if index == 0 {
                    query += " and (";
                } else {
                    query += " or ";
                }
                query += &format!("player_white = \"{0}\" or player_black = \"{0}\"", player);
            }
            query += ")";
        }
    }

    query += ";";

    let mut statement = connection.prepare(&query).unwrap();

    let games = statement.query_map(&[], |row| Game {
        id: row.get::<i32, u32>(0) as usize,
        date: row.get::<i32, i64>(1) as u64,
        size: row.get::<i32, u32>(2) as usize,
        p1: row.get(3),
        p2: row.get(4),
        plies: row.get::<i32, String>(5).split(",").enumerate().filter_map(|(i, ply_string)| {
            let mut color = if i % 2 == 0 {
                Color::White
            } else {
                Color::Black
            };
            if i < 2 {
                color = color.flip();
            }

            if ply_string.is_empty() {
                None
            } else {
                ply(ply_string, color)
            }
        }).collect::<Vec<_>>(),
        result: row.get(6),
    }).unwrap().filter_map(|game| {
        if let Ok(game) = game {
            if game.plies.is_empty() {
                None
            } else {
                Some(game)
            }
        } else {
            None
        }
    }).collect::<Vec<_>>();

    games
}

/// Converts between the PlayTak server's notation and our Ply.
fn ply(string: &str, color: Color) -> Option<Ply> {
    fn parse_square(square: &str) -> Option<(usize, usize)> {
        let mut chars = square.chars();

        let x = if let Some(x) = chars.next() {
            (x as u8 - 65) as usize
        } else {
            return None;
        };

        let y = if let Some(y) = chars.next() {
            (y as u8 - 49) as usize
        } else {
            return None;
        };

        Some((x, y))
    }

    let parts = string.split_whitespace().collect::<Vec<_>>();

    if parts[0] == "P" {
        if parts.len() < 2 {
            return None;
        }

        let (x, y) = if let Some(coordinates) = parse_square(parts[1]) {
            coordinates
        } else {
            return None;
        };

        let piece = if parts.len() >= 3 {
            if parts[2] == "W" {
                Piece::StandingStone(color)
            } else if parts[2] == "C" {
                Piece::Capstone(color)
            } else {
                return None;
            }
        } else {
            Piece::Flatstone(color)
        };

        Some(Ply::Place {
            x: x,
            y: y,
            piece: piece
        })
    } else if parts[0] == "M" {
        if parts.len() < 4 {
            return None;
        }

        let (x, y) = if let Some(coordinates) = parse_square(parts[1]) {
            coordinates
        } else {
            return None;
        };

        let (tx, ty) = if let Some(coordinates) = parse_square(parts[2]) {
            coordinates
        } else {
            return None;
        };

        let direction = {
            let (dx, dy) = (
                tx as i8 - x as i8,
                ty as i8 - y as i8,
            );

            if dx < 0 && dy == 0 {
                Direction::West
            } else if dx > 0 && dy == 0 {
                Direction::East
            } else if dy < 0 && dx == 0 {
                Direction::South
            } else if dy > 0 && dx == 0 {
                Direction::North
            } else {
                return None;
            }
        };

        let drops = parts[3..].iter().map(|drop| u8::from_str(drop).unwrap()).collect::<Vec<_>>();

        Some(Ply::Slide {
            x: x,
            y: y,
            direction: direction,
            drops: drops,
        })
    } else {
        None
    }
}
