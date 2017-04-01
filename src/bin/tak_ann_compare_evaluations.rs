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

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};

use zero_sum::analysis::Evaluator;
use zero_sum::impls::tak::evaluator::{AnnEvaluator, StaticEvaluator};
use zero_sum::impls::tak::State;

fn main() {
    let network_file = String::from("evaluator_000230");
    let positions_file = String::from("training_positions");
    let positions_count = 3000;
    let output_file = String::from("comparison");

    println!("Reading positions...");
    let positions = if let Ok(file) = OpenOptions::new().read(true).open(&positions_file) {
        let reader = BufReader::new(file);
        reader.lines().take(positions_count).map(|line| State::from_tps(line.unwrap().trim()).unwrap()).collect::<Vec<_>>()
    } else {
        panic!("Cannot read file: {}", positions_file);
    };
    println!("  Done. Read {} positions.", positions.len());

    let static_evaluator = StaticEvaluator;

    let ann_evaluator = if let Ok(evaluator) = AnnEvaluator::from_file(&format!("{}", &network_file)) {
        evaluator
    } else {
        panic!("Cannot read file: {}", network_file);
    };

    println!("Writing evaluations...");
    if let Ok(mut file) = OpenOptions::new().write(true).truncate(true).create(true).open(&output_file) {
        for position in &positions {
            write!(&mut file, "{} {}\n", static_evaluator.evaluate(position), ann_evaluator.evaluate(position)).ok();
        }
    } else {
        println!("Cannot write file: {}", output_file);
    }
    println!("  Done.");
}
