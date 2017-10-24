//
// This file is part of smallann.
//
// smallann is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// smallann is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with smallann. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright 2017 Chris Foster
//

use std::io::{BufReader, Result, Write};
use std::str::FromStr;

use smallmath::Matrix;
use smallmath::vector::{self, Vector};

use gradient_descent::GradientDescent;
use serialization::{File, Identifiable, read_error, read_line, Serializable};

#[derive(Clone)]
pub struct SimpleGradientDescent {
    weights_buffer: Matrix,
    biases_buffer: Vector,
}

impl SimpleGradientDescent {
    pub fn new(inputs: usize, outputs: usize) -> SimpleGradientDescent {
        SimpleGradientDescent {
            weights_buffer: Matrix::zeros(inputs, outputs),
            biases_buffer: Vector::zeros(outputs),
        }
    }
}

impl GradientDescent for SimpleGradientDescent {
    fn descend(
        &mut self,
        weights: &mut Matrix,
        biases: &mut Vector,
        weight_gradients: &Matrix,
        bias_gradients: &Vector,
        rate: f32,
    ) {
        // weights -= weight_gradients * rate
        vector::ops::scale(weight_gradients, rate, &mut self.weights_buffer);
        vector::ops::subtract_assign(weights, &self.weights_buffer);

        // biases -= bias_gradients * rate
        vector::ops::scale(bias_gradients, rate, &mut self.biases_buffer);
        vector::ops::subtract_assign(biases, &self.biases_buffer);
    }
}

impl Identifiable for SimpleGradientDescent {
    fn identifier() -> String {
        String::from("SimpleGradientDescent")
    }

    fn get_identifier(&self) -> String {
        Self::identifier()
    }
}

impl Serializable for SimpleGradientDescent {
    fn read_from_file(file: &mut BufReader<File>) -> Result<SimpleGradientDescent> {
        let (inputs, outputs) = {
            let strings = read_line(file)?;

            if strings.len() < 2 {
                return read_error(file, "Cannot read inputs/outputs!");
            }

            (
                if let Ok(inputs) = usize::from_str(&strings[0]) {
                    inputs
                } else {
                    return read_error(file, "Cannot parse inputs!");
                },
                if let Ok(outputs) = usize::from_str(&strings[1]) {
                    outputs
                } else {
                    return read_error(file, "Cannot parse outputs!");
                },
            )
        };

        Ok(SimpleGradientDescent::new(inputs, outputs))
    }

    fn write_to_file(&self, file: &mut File) -> Result<()> {
        let indentation = file.indentation();
        write!(file, "{}{}  {}\n", indentation, self.weights_buffer.rows(), self.weights_buffer.columns())
    }
}
