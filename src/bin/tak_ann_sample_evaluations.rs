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

use std::env;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};

use zero_sum::analysis::Evaluator;
use zero_sum::impls::tak::evaluator::{AnnEvaluator, StaticEvaluator};
use zero_sum::impls::tak::State;

fn main() {
    let args: Vec<String> = env::args().collect();

    let positions_file = if args.len() < 2 {
        String::from("sample_positions")
    } else {
        args[1].clone()
    };

    let network_file = if args.len() < 3 {
        None
    } else {
        Some(args[2].clone())
    };

    let output_file = String::from("sample");

    println!("Reading positions...");
    let positions = if let Ok(file) = OpenOptions::new().read(true).open(&positions_file) {
        let reader = BufReader::new(file);
        reader.lines().map(|line| State::from_tps(line.unwrap().trim()).unwrap()).collect::<Vec<_>>()
    } else {
        panic!("Cannot read file: {}", positions_file);
    };
    println!("  Done. Read {} positions.", positions.len());

    if let Some(network_file) = network_file {
        let evaluator = if let Ok(evaluator) = AnnEvaluator::from_file(&format!("{}", &network_file)) {
            evaluator
        } else {
            panic!("Cannot read file: {}", network_file);
        };

        println!("Writing evaluations...");
        if let Ok(mut file) = OpenOptions::new().write(true).truncate(true).create(true).open(&output_file) {
            for position in &positions {
                write!(&mut file, "{}\n", evaluator.evaluate(position)).ok();
            }
        } else {
            println!("Cannot write file: {}", output_file);
        }
    } else {
        let evaluator = StaticEvaluator;

        println!("Writing evaluations...");
        if let Ok(mut file) = OpenOptions::new().write(true).truncate(true).create(true).open(&output_file) {
            for position in &positions {
                write!(&mut file, "{}\n", evaluator.evaluate(position)).ok();
            }
        } else {
            println!("Cannot write file: {}", output_file);
        }
    }
    println!("  Done.");
}
