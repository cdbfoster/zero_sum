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
pub struct MomentumGradientDescent {
    pub(crate) momentum: f32,

    pub(crate) weights_v: Matrix,
    pub(crate) biases_v: Vector,

    weights_temp: Matrix,
    biases_temp: Vector,
}

impl MomentumGradientDescent {
    pub fn new(inputs: usize, outputs: usize, momentum: f32) -> MomentumGradientDescent {
        assert!(inputs > 0, "Invalid number of inputs!");
        assert!(outputs > 0, "Invalid number of outputs!");

        MomentumGradientDescent {
            momentum: momentum,
            weights_v: Matrix::zeros(inputs, outputs),
            biases_v: Vector::zeros(outputs),
            weights_temp: Matrix::zeros(inputs, outputs),
            biases_temp: Vector::zeros(outputs),
        }
    }

    pub(crate) fn construct(
        inputs: usize, outputs: usize,
        momentum: f32,
        weights_v: Matrix,
        biases_v: Vector,
    ) -> MomentumGradientDescent {
        MomentumGradientDescent {
            momentum: momentum,
            weights_v: weights_v,
            biases_v: biases_v,
            weights_temp: Matrix::zeros(inputs, outputs),
            biases_temp: Vector::zeros(outputs),
        }
    }
}

impl GradientDescent for MomentumGradientDescent {
    fn descend(
        &mut self,
        weights: &mut Matrix,
        biases: &mut Vector,
        weight_gradients: &Matrix,
        bias_gradients: &Vector,
        rate: f32,
    ) {
        // weights_v = momentum * weights_v + rate * weight_gradients
        vector::ops::scale_assign(&mut self.weights_v, self.momentum);
        vector::ops::scale(weight_gradients, rate, &mut self.weights_temp);
        vector::ops::add_assign(&mut self.weights_v, &self.weights_temp);

        // biases_v = momentum * biases_v + rate * weight_gradients
        vector::ops::scale_assign(&mut self.biases_v, self.momentum);
        vector::ops::scale(bias_gradients, rate, &mut self.biases_temp);
        vector::ops::add_assign(&mut self.biases_v, &self.biases_temp);

        // weights -= weights_v
        vector::ops::subtract_assign(weights, &self.weights_v);

        // biases -= biases_v
        vector::ops::subtract_assign(biases, &self.biases_v);
    }
}
