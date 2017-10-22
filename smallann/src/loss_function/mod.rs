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

//pub use self::cross_entropy::CrossEntropyLossFunction;
pub use self::mean_squared_error::MeanSquaredErrorLossFunction;

pub trait LossFunction {
    fn l(x: f32, y: f32) -> f32;
    fn l_prime(x: f32, y: f32) -> f32;

    fn l_vector(outputs: &Vector, targets: &Vector, loss: &mut Vector) {
        debug_assert!(outputs.len() == targets.len() && outputs.len() == loss.len(), "Vector dimensions don't match!");
        for i in 0..outputs.len() {
            loss[i] = Self::l(outputs[i], targets[i]);
        }
    }

    fn l_prime_vector(outputs: &Vector, targets: &Vector, loss_prime: &mut Vector) {
        debug_assert!(outputs.len() == targets.len() && outputs.len() == loss_prime.len(), "Vector dimensions don't match!");
        for i in 0..outputs.len() {
            loss_prime[i] = Self::l_prime(outputs[i], targets[i]);
        }
    }
}

//mod cross_entropy;
mod mean_squared_error;
