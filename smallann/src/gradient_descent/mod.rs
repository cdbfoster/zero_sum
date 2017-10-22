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

use smallmath::{Matrix, Vector};
pub use self::adadelta::AdadeltaGradientDescent;
pub use self::momentum::MomentumGradientDescent;
pub use self::simple::SimpleGradientDescent;

pub trait GradientDescent {
    fn descend(
        &mut self,
        weights: &mut Matrix,
        biases: &mut Vector,
        weight_gradients: &Matrix,
        bias_gradients: &Vector,
        rate: f32,
    );
}

mod adadelta;
mod momentum;
mod simple;
