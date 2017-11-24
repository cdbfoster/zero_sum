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

use smallmath::Matrix;
use smallmath::vector::{self, Vector};

use gradient_descent::GradientDescent;

#[derive(Clone)]
pub struct SimpleGradientDescent {
    pub(crate) inputs:usize,
    pub(crate) outputs: usize,

    weights_buffer: Matrix,
    biases_buffer: Vector,
}

impl SimpleGradientDescent {
    pub fn new(inputs: usize, outputs: usize) -> SimpleGradientDescent {
        SimpleGradientDescent {
            inputs: inputs,
            outputs: outputs,
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
