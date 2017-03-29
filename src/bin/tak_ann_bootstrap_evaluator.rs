extern crate rand;
extern crate zero_sum;

use std::cmp;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::str::FromStr;

use zero_sum::analysis::Evaluator;
use zero_sum::impls::tak::evaluator::AnnEvaluator;
use zero_sum::impls::tak::State;

fn main() {
    let network_file = String::from("evaluator");
    let positions_file = String::from("training_positions");
    let labels_file = String::from("training_labels");
    let progress_file = String::from("progress");
    let batch_size = 30;

    let mut evaluator = if let Ok(evaluator) = AnnEvaluator::from_file(&network_file) {
        evaluator
    } else {
        AnnEvaluator::new()
    };

    println!("Reading positions...");
    let positions = if let Ok(file) = OpenOptions::new().read(true).open(&positions_file) {
        let reader = BufReader::new(file);
        reader.lines().map(|line| State::from_tps(line.unwrap().trim()).unwrap()).collect::<Vec<_>>()
    } else {
        panic!("Cannot read file: {}", positions_file);
    };
    println!("  Done. Read {} positions.", positions.len());

    println!("Reading labels...");
    let labels = if let Ok(file) = OpenOptions::new().read(true).open(&labels_file) {
        let reader = BufReader::new(file);
        reader.lines().map(|line| <AnnEvaluator as Evaluator>::Evaluation::new(i32::from_str(line.unwrap().trim()).unwrap())).collect::<Vec<_>>()
    } else {
        panic!("Cannot read file: {}", labels_file);
    };
    println!("  Done. Read {} labels.", labels.len());

    assert!(positions.len() == labels.len(), "Mismatched number of positions and labels!");

    for iteration in 0..(positions.len() / batch_size + 1) {
        let start_index = iteration * batch_size;
        let end_index = cmp::min((iteration + 1) * batch_size, positions.len());

        if start_index == positions.len() {
            break;
        }

        if iteration % 1000 == 0 {
            let mut error = 0.0;

            evaluator.train_batch(
                &positions[start_index..end_index],
                &labels[start_index..end_index],
                Some(&mut error),
            );

            println!("{:4}: {:.6}", iteration, error);

            if let Ok(mut file) = OpenOptions::new().append(true).create(true).open(&progress_file) {
                write!(&mut file, "{} {:.6}\n", iteration, error).ok();
            } else {
                println!("Cannot write file: {}", progress_file);
            }
        } else {
            evaluator.train_batch(
                &positions[start_index..end_index],
                &labels[start_index..end_index],
                None,
            );
        }
    }

    evaluator.to_file(&network_file);
}
