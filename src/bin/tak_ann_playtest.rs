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

use std::sync::{Arc, mpsc, Mutex};
use std::thread;

use zero_sum::analysis::Extrapolatable;
use zero_sum::analysis::search::Search;
use zero_sum::analysis::search::pvsearch::PvSearch;
use zero_sum::impls::tak::evaluator::{AnnEvaluator, StaticEvaluator};
use zero_sum::impls::tak::{Resolution, State};
use zero_sum::Resolution as ResolutionTrait;
use zero_sum::State as StateTrait;

fn main() {
    let network_file = String::from("evaluator");
    let games = 50;
    let threads = 4;
    let search_depth = 3;

    let ann_evaluator = if let Ok(evaluator) = AnnEvaluator::from_file(&format!("{}", &network_file)) {
        evaluator
    } else {
        panic!("Cannot read network file: {}", network_file);
    };

    let static_evaluator = StaticEvaluator;

    let ann_wins = Arc::new(Mutex::new(0));
    let static_wins = Arc::new(Mutex::new(0));
    let white_wins = Arc::new(Mutex::new(0));
    let black_wins = Arc::new(Mutex::new(0));
    let road_wins = Arc::new(Mutex::new(0));
    let flat_wins = Arc::new(Mutex::new(0));
    let draws = Arc::new(Mutex::new(0));
    let loops = Arc::new(Mutex::new(0));

    let remaining = Arc::new(Mutex::new(games));
    let (finished_sender, finished_receiver) = mpsc::channel();

    for thread in 0..threads {
        let ann_wins = ann_wins.clone();
        let static_wins = static_wins.clone();
        let white_wins = white_wins.clone();
        let black_wins = black_wins.clone();
        let road_wins = road_wins.clone();
        let flat_wins = flat_wins.clone();
        let draws = draws.clone();
        let loops = loops.clone();

        let ann_evaluator = ann_evaluator.clone();
        let static_evaluator = static_evaluator.clone();

        let remaining = remaining.clone();
        let finished_sender = finished_sender.clone();

        thread::spawn(move || {
            loop {
                let game = {
                    let mut remaining = remaining.lock().unwrap();
                    if *remaining == 0 {
                        break;
                    }
                    *remaining -= 1;
                    games - *remaining
                };

                println!("Thread {}: Start game {}", thread, game);

                let mut state = State::new(5);
                for _ in 0..4 {
                    let mut plies = state.extrapolate();
                    loop {
                        match state.execute_ply(&plies.pop().unwrap()) {
                            Ok(next) => {
                                state = next;
                                break;
                            },
                            _ => (),
                        }
                    }
                }

                let mut ann_search = PvSearch::with_depth(ann_evaluator.clone(), search_depth);
                let mut static_search = PvSearch::with_depth(static_evaluator.clone(), search_depth);

                loop {
                    let ply = if (game + state.ply_count) % 2 == 0 {
                        let result = ann_search.search(&state, None);
                        result.principal_variation[0].clone()
                    } else {
                        let result = static_search.search(&state, None);
                        result.principal_variation[0].clone()
                    };

                    match state.execute_ply(&ply) {
                        Ok(next) => state = next,
                        Err(error) => panic!("Illegal move: {}", error),
                    }

                    if let Some(resolution) = state.check_resolution() {
                        if let Some(winner) = resolution.get_winner() {
                            println!("Thread {}: Game {}: {} wins. {:?}", thread, game,
                                if (game + winner as u16) % 2 == 0 {
                                    *ann_wins.lock().unwrap() += 1;
                                    "ANN"
                                } else {
                                    *static_wins.lock().unwrap() += 1;
                                    "Static"
                                },
                                resolution,
                            );

                            if winner == 0 {
                                *white_wins.lock().unwrap() += 1;
                            } else {
                                *black_wins.lock().unwrap() += 1;
                            }
                        } else {
                            println!("Thread {}: Game {}: Draw.", thread, game);
                            *draws.lock().unwrap() += 1;
                        }

                        match resolution {
                            Resolution::Road(_) => *road_wins.lock().unwrap() += 1,
                            Resolution::Flat(_) => *flat_wins.lock().unwrap() += 1,
                            _ => (),
                        }

                        break;
                    }

                    if state.ply_count > 150 {
                        println!("Thread {}: Game {}: Loop.", thread, game);
                        *loops.lock().unwrap() += 1;
                        break;
                    }
                }
            }

            finished_sender.send(()).ok();
        });
    }

    for _ in 0..threads {
        finished_receiver.recv().ok();
    }

    println!("");

    let ann_wins = *ann_wins.lock().unwrap();
    let static_wins = *static_wins.lock().unwrap();
    let white_wins = *white_wins.lock().unwrap();
    let black_wins = *black_wins.lock().unwrap();
    let road_wins = *road_wins.lock().unwrap();
    let flat_wins = *flat_wins.lock().unwrap();
    let draws = *draws.lock().unwrap();
    let loops = *loops.lock().unwrap();
    println!("ANN wins:    {:3} / {:3} {:6.2}%", ann_wins, games, ann_wins as f32 / games as f32 * 100.0);
    println!("Static wins: {:3} / {:3} {:6.2}%", static_wins, games, static_wins as f32 / games as f32 * 100.0);
    println!("White wins:  {:3} / {:3} {:6.2}%", white_wins, games, white_wins as f32 / games as f32 * 100.0);
    println!("Black wins:  {:3} / {:3} {:6.2}%", black_wins, games, black_wins as f32 / games as f32 * 100.0);
    println!("Road wins:   {:3} / {:3} {:6.2}%", road_wins, games, road_wins as f32 / games as f32 * 100.0);
    println!("Flat wins:   {:3} / {:3} {:6.2}%", flat_wins, games, flat_wins as f32 / games as f32 * 100.0);
    println!("Draws:       {:3} / {:3} {:6.2}%", draws, games, draws as f32 / games as f32 * 100.0);
    println!("Loops:       {:3} / {:3} {:6.2}%", loops, games, loops as f32 / games as f32 * 100.0);
}
