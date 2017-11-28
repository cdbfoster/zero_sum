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

use serialization::Serializable;

pub use self::activation::ActivationLayer;
//pub use self::block::BlockLayer;
pub use self::composite::CompositeLayer;
pub use self::convolutional::ConvolutionalLayer;
pub use self::fully_connected::FullyConnectedLayer;
pub use self::pass_through::PassThroughLayer;
pub use self::split::SplitLayer;

pub trait Layer: Send + Serializable {
    fn inputs(&self) -> usize;
    fn outputs(&self) -> usize;
    fn feed_forward(&self, inputs: &Matrix, outputs: &mut Matrix);
    fn propagate_backward(&mut self, gradients: &Matrix, previous_inputs: &Matrix, previous_gradients: &mut Matrix, rate: f32);
    fn boxed_clone(&self) -> Box<Layer>;
}

impl Clone for Box<Layer> {
    fn clone(&self) -> Box<Layer> {
        self.boxed_clone()
    }
}

mod activation;
//mod block;
mod composite;
mod convolutional;
mod fully_connected;
mod pass_through;
mod split;
