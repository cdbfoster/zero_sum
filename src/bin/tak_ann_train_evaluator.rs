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
use std::path::Path;
use std::time::Instant;

use zero_sum::impls::tak::evaluator::AnnEvaluator;
use zero_sum::impls::tak::State;

fn main() {
    let network_prefix = String::from("evaluator");
    let positions_file = String::from("training_positions");
    let progress_file = String::from("progress");
    let batch_size = 30;
    let serialize_interval = 10;
    let progress_interval = 1;
    let resume_iteration = None;
    let threads = 4;

    println!("Reading positions...");
    let mut positions = if let Ok(file) = OpenOptions::new().read(true).open(&positions_file) {
        BufReader::new(file)
    } else {
        panic!("Cannot read file: {}", positions_file);
    };
    println!("  Done.");

    println!("Searching for resume network file...");
    let (start_iteration, mut evaluator) = {
        let mut resume = 0;
        let mut evaluator = None;

        if let Some(resume_iteration) = resume_iteration {
            if let Ok(read) = AnnEvaluator::from_file(&format!("{}_{:06}", &network_prefix, resume_iteration)) {
                resume = resume_iteration;
                evaluator = Some(read);
            }
        } else {
            for iteration in (0..).map(|i| i * serialize_interval) {
                if Path::new(&format!("{}_{:06}", &network_prefix, iteration)).exists() {
                    resume = iteration;
                } else {
                    if let Ok(read) = AnnEvaluator::from_file(&format!("{}_{:06}", &network_prefix, resume)) {
                        evaluator = Some(read);
                    }
                    break;
                }
            }
        }

        if let Some(evaluator) = evaluator {
            (resume + 1, evaluator)
        } else {
            (0, AnnEvaluator::new())
        }
    };

    // If we're starting at the beginning, look for a bootstrapped network file
    if start_iteration == 0 {
        if let Ok(read) = AnnEvaluator::from_file(&format!("{}_bootstrap", &network_prefix)) {
            println!("  Done. No resume network file found. Found bootstrap network file.");
            evaluator = read;
        } else {
            println!("  Done. No resume network file found. Starting from random initialization.");
        }
    } else {
        println!("  Done. Resuming from iteration {}.", start_iteration - 1);
    }

    // Skip previous positions
    for _ in 0..start_iteration * batch_size {
        let mut line = String::new();
        if positions.read_line(&mut line).unwrap() == 0 {
            panic!("Too few positions in {} to continue training!", &positions_file);
        }
    }

    println!("Training...");
    for iteration in start_iteration.. {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let mut line = String::new();
            if positions.read_line(&mut line).unwrap() == 0 {
                break;
            }

            batch.push(State::from_tps(line.trim()).unwrap());
        }

        if batch.is_empty() {
            break;
        }

        let start_batch = Instant::now();

        let mut error = 0.0;
        evaluator.train_batch_tdleaf(
            &batch,
            Some(&mut error),
            threads,
        );

        let elapsed_batch = start_batch.elapsed();
        let elapsed_batch = elapsed_batch.as_secs() as f32 + elapsed_batch.subsec_nanos() as f32 / 1_000_000_000.0;

        println!("Iteration {}: Error: {:.6}, Time: {:.2}s/position", iteration, error, elapsed_batch / batch.len() as f32);

        if iteration % serialize_interval == 0 {
            evaluator.to_file(&format!("{}_{:06}", &network_prefix, iteration));
        }

        if iteration % progress_interval == 0 {
            if let Ok(mut file) = OpenOptions::new().append(true).create(true).open(&progress_file) {
                write!(&mut file, "{} {:.6}\n", iteration, error).ok();
            } else {
                println!("Cannot write file: {}", progress_file);
            }
        }
    }

    evaluator.to_file(&network_prefix);
}
