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

use smallmath::Vector;

pub use self::relu::ReLuActivationFunction;
//pub use self::sigmoid::SigmoidActivationFunction;
//pub use self::softmax::SoftmaxActivationFunction;
pub use self::tanh::TanHActivationFunction;

pub trait ActivationFunction: Clone {
    fn f(x: f32) -> f32;
    fn f_prime(x: f32) -> f32;

    fn f_vector(inputs: &Vector, outputs: &mut Vector) {
        debug_assert!(inputs.len() == outputs.len(), "inputs.len() doesn't match outputs.len()!");
        for i in 0..inputs.len() {
            outputs[i] = Self::f(inputs[i]);
        }
    }

    fn f_prime_vector(inputs: &Vector, outputs: &mut Vector) {
        debug_assert!(inputs.len() == outputs.len(), "inputs.len() doesn't match outputs.len()!");
        for i in 0..inputs.len() {
            outputs[i] = Self::f_prime(inputs[i]);
        }
    }
}

mod relu;
//mod sigmoid;
//mod softmax;
mod tanh;
