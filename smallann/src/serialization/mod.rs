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

use std::io::{BufReader, Result};

use smallmath::{Matrix, Vector};

use activation_function::*;
use gradient_descent::*;
use layer::*;

pub use self::file::File;
pub use self::identifiable::Identifiable;
pub use self::serializable::{read_error, read_line, Serializable, write_matrix};

mod file;
#[macro_use]
mod identifiable;
#[macro_use]
mod serializable;

read_layer_types! {
    activation_layers: [
      ActivationLayer<F>,
    ],
    activation_functions: [
      ReLuActivationFunction,
      TanHActivationFunction,
    ],
    gradient_descent_layers: [
      ConvolutionalLayer<G>,
      FullyConnectedLayer<G>,
    ],
    gradient_descent_algorithms: [
      AdadeltaGradientDescent,
      MomentumGradientDescent,
      SimpleGradientDescent,
    ],
    other_layers: [
      //BlockLayer,
      CompositeLayer,
      PassThroughLayer,
      SplitLayer,
    ],
}

identifiable!(
    Matrix,
    Vector,
);
