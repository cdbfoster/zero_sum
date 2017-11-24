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

use layer::Layer;

#[derive(Clone)]
pub struct PassThroughLayer {
    pub(crate) size: usize,
}

impl PassThroughLayer {
    pub fn new(size: usize) -> PassThroughLayer {
        PassThroughLayer {
            size: size,
        }
    }
}

impl Layer for PassThroughLayer {
    fn inputs(&self) -> usize {
        self.size
    }

    fn outputs(&self) -> usize {
        self.size
    }

    fn feed_forward(&self, inputs: &Matrix, outputs: &mut Matrix) {
        outputs.clone_from(inputs);
    }

    fn propagate_backward(&mut self, gradients: &Matrix, _: &Matrix, previous_gradients: &mut Matrix, _: f32) {
        previous_gradients.clone_from(gradients);
    }

    fn boxed_clone(&self) -> Box<Layer> {
        Box::new(self.clone())
    }
}
