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

use std::fs::OpenOptions;
use std::i32;
use std::io::BufReader;
use std::sync::{Arc, mpsc, Mutex};
use std::thread;

use analysis::{self, Evaluation as EvaluationTrait};
use analysis::search::Search;
use analysis::search::pvsearch::PvSearch;
use impls::tak::{Color, Resolution, State};
use impls::tak::state::ann::*;
use state::State as StateTrait;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Evaluation(pub i32);

impl Evaluation {
    pub fn new(value: i32) -> Evaluation {
        Evaluation(value)
    }
}

prepare_evaluation_tuple!(Evaluation);

impl analysis::Evaluation for Evaluation {
    fn null() -> Evaluation { Evaluation(0) }
    fn epsilon() -> Evaluation { Evaluation(1) }
    fn win() -> Evaluation { Evaluation(100_000) }
    fn max() -> Evaluation { Evaluation(i32::MAX) }
    fn is_win(&self) -> bool { self.0 >= 99_000 }
}

/// [0 - common) => [0.0 - 0.9), [common - win] => [0.9 - 1.0]
fn scale_evaluation(evaluation: f32, common: f32, win: f32) -> f32 {
    if evaluation.abs() < common {
        evaluation / common * 0.9
    } else {
        evaluation.signum() * (0.9 + (evaluation.abs() - common) / (win - common) * 0.1)
    }
}

/// [0.0 - 0.9) => [0 - common), [0.9 - 1.0] => [common - win]
fn unscale_evaluation(evaluation: f32, common: f32, win: f32) -> f32 {
    if evaluation.abs() < 0.9 {
        evaluation / 0.9 * common
    } else {
        evaluation.signum() * ((evaluation.abs() - 0.9) / 0.1 * (win - common) + common)
    }
}

#[derive(Clone)]
pub struct AnnEvaluator {
    ann: Ann<ReLuActivationFunction, TanHActivationFunction, AdadeltaGradientDescent>,
}

impl AnnEvaluator {
    pub fn new() -> AnnEvaluator {
        let input_count = 339;
        let hidden_layers = [150, 64, 32];
        let output_count = 1;

        let mut weight_mask = MatrixCm::zeros(input_count, hidden_layers[0]);

        // Global group
        for column in 0..10 {
            for row in 0..14 {
                weight_mask[column][row] = 1.0;
            }
        }

        // Stack positions and configuration
        for column in 10..125 {
            for row in 14..289 {
                weight_mask[column][row] = 1.0;
            }
        }

        // Influence
        for column in 125..150 {
            for row in 289..339 {
                weight_mask[column][row] = 1.0;
            }
        }

        AnnEvaluator {
            ann: Ann::<ReLuActivationFunction, TanHActivationFunction, AdadeltaGradientDescent>::new(
                input_count,
                &hidden_layers,
                output_count,
                &[Some(weight_mask)],
                AdadeltaGradientDescent::new(
                    input_count,
                    &hidden_layers,
                    output_count,
                    0.000001,
                ),
            ),
        }
    }

    pub fn from_file(filename: &str) -> Result<AnnEvaluator, String> {
        let mut evaluator = AnnEvaluator::new();

        if let Ok(file) = OpenOptions::new().read(true).open(filename) {
            let mut reader = BufReader::new(file);
            match read_network(&mut reader, &mut evaluator.ann) {
                Ok(_) => Ok(evaluator),
                Err(error) => Err(error),
            }
        } else {
            Err(format!("Cannot open file: {}", filename))
        }
    }

    pub fn to_file(&self, filename: &str) {
        if let Ok(mut file) = OpenOptions::new().write(true).truncate(true).create(true).open(filename) {
            write_network(&mut file, &self.ann).ok();
        } else {
            println!("Cannot write file: {}", filename);
        }
    }

    pub fn train_batch(&mut self, positions: &[State], labels: &[Evaluation], error: Option<&mut f32>) {
        let mut inputs = MatrixRm::zeros(positions.len(), 339);
        for i in 0..positions.len() {
            inputs[i].clone_from_slice(&gather_features(&positions[i]));
        }

        let mut targets = MatrixRm::zeros(labels.len(), 1);
        for i in 0..labels.len() {
            targets[i].clone_from_slice(&[scale_evaluation(labels[i].0 as f32, 12_000.0, Evaluation::win().0 as f32)]);
        }

        if let Some(error) = error {
            let mut outputs = MatrixRm::zeros(targets.rows, targets.columns);
            self.ann.propagate_forward_simple(&inputs, &mut outputs);

            let mut error_matrix = MatrixRm::zeros(outputs.rows, outputs.columns);
            *error = calculate_error::<TanHActivationFunction>(&outputs, &targets, &mut error_matrix) / inputs.rows as f32;
        }

        self.ann.train(&inputs, &targets, 0.5);
    }

    /// Use temporal difference learning to train the system through self-play.
    pub fn train_batch_tdleaf(&mut self, positions: &[State], error: Option<&mut f32>, thread_count: usize) {
        let search_depth = 3;

        let total_error = Arc::new(Mutex::new(0.0));

        let mut inputs = MatrixRm::zeros(positions.len(), 339);
        for i in 0..positions.len() {
            inputs[i].clone_from_slice(&gather_features(&positions[i]));
        }

        let targets = Arc::new(Mutex::new(MatrixRm::zeros(positions.len(), 1)));

        let remaining = Arc::new(Mutex::new(positions.len()));
        let (finished_sender, finished_receiver) = mpsc::channel();

        for _ in 0..thread_count {
            let total_error = total_error.clone();
            let positions = positions.to_vec();
            let targets = targets.clone();
            let remaining = remaining.clone();
            let finished_sender = finished_sender.clone();
            let mut search = PvSearch::with_depth(self.clone(), search_depth);

            thread::spawn(move || {
                loop {
                    let i = {
                        let mut remaining = remaining.lock().unwrap();
                        if *remaining == 0 {
                            break;
                        }
                        *remaining -= 1;
                        *remaining
                    };

                    let result = search.search(&positions[i], None);
                    let leaf_score = scale_evaluation(result.evaluation.0 as f32, 12_000.0, Evaluation::win().0 as f32);

                    if !result.principal_variation.is_empty() {
                        let mut state = match positions[i].execute_ply(&result.principal_variation[0]) {
                            Ok(next) => next,
                            Err(error) => panic!("Invalid principal variation: {}", error),
                        };

                        let mut accumulated_error = 0.0;
                        let mut last_score = leaf_score;
                        let mut td_discount = 0.83;
                        let mut absolute_discount = 0.995;

                        for j in 0..10 {
                            let result = search.search(&state, None);

                            let sign = if j % 2 == 0 { -1.0 } else { 1.0 };
                            let next_score = scale_evaluation(result.evaluation.0 as f32, 12_000.0, Evaluation::win().0 as f32)
                                * absolute_discount * sign;

                            accumulated_error += td_discount * (next_score - last_score);
                            td_discount *= 0.83;
                            last_score = next_score;

                            absolute_discount *= 0.995;

                            if state.check_resolution().is_some() || result.principal_variation.is_empty() {
                                break;
                            }

                            match state.execute_ply(&result.principal_variation[0]) {
                                Ok(next) => state = next,
                                Err(error) => panic!("Invalid principal variation: {}", error),
                            }
                        }

                        *total_error.lock().unwrap() += accumulated_error.abs();

                        // Clamp error
                        accumulated_error = accumulated_error.max(-1.0).min(1.0);

                        targets.lock().unwrap()[i].clone_from_slice(&[leaf_score + accumulated_error]);
                    }
                }

                finished_sender.send(()).ok();
            });
        }

        for _ in 0..thread_count {
            finished_receiver.recv().ok();
        }

        if let Some(error) = error {
            *error = *total_error.lock().unwrap() / positions.len() as f32;
        }

        self.ann.train(&inputs, &*targets.lock().unwrap(), 0.5);
    }
}

impl analysis::Evaluator for AnnEvaluator {
    type State = State;
    type Evaluation = Evaluation;

    fn evaluate(&self, state: &State) -> Evaluation {
        let next_color = if state.ply_count % 2 == 0 {
            Color::White
        } else {
            Color::Black
        };

        match state.check_resolution() {
            None => (),
            Some(Resolution::Road(win_color)) |
            Some(Resolution::Flat(win_color)) => {
                if win_color == next_color {
                    return Evaluation::win() - Evaluation(state.ply_count as i32);
                } else {
                    return -Evaluation::win() + Evaluation(state.ply_count as i32);
                }
            },
            Some(Resolution::Draw) => return Evaluation::null(),
        }

        let features = gather_features(state);
        let input = MatrixRm::from_vec(1, features.len(), features);
        let mut output = MatrixRm::zeros(1, 1);

        self.ann.propagate_forward_simple(&input, &mut output);

        Evaluation(unscale_evaluation(output.values[0], 12_000.0, Evaluation::win().0 as f32) as i32)
    }
}
