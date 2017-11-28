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

use std::mem;

use smallmath::matrix::Matrix;
use smallmath::vector;

use layer::Layer;

#[derive(Clone)]
pub struct SplitLayer {
    outputs: usize,
    pub(crate) layers: Vec<Box<Layer>>,
    previous_gradients_buffer: Matrix,
}

impl SplitLayer {
    pub fn new(layers: Vec<Box<Layer>>) -> SplitLayer {
        let inputs = layers.first().unwrap().inputs();
        let mut outputs = 0;

        for layer in &layers {
            assert!(layer.inputs() == inputs, "Input size mismatch in SplitLayer!");
            outputs += layer.outputs();
        }

        SplitLayer {
            outputs: outputs,
            layers: layers,
            previous_gradients_buffer: Matrix::zeros(10, inputs),
        }
    }
}

impl Layer for SplitLayer {
    fn inputs(&self) -> usize {
        self.layers.first().unwrap().inputs()
    }

    fn outputs(&self) -> usize {
        self.outputs
    }

    fn feed_forward(&self, inputs: &Matrix, outputs: &mut Matrix) {
        // XXX Add debug asserts on the sizes of the matrices
        let mut output_position = 0;

        for i in 0..self.layers.len() {
            let mut layer_outputs = unsafe { Matrix::from_raw_parts(
                inputs.rows(), self.layers[i].outputs(),
                outputs.as_mut_ptr().offset((output_position * inputs.rows()) as isize),
                0,
            ) };

            self.layers[i].feed_forward(inputs, &mut layer_outputs);

            output_position += self.layers[i].outputs();

            mem::forget(layer_outputs);
        }
    }

    fn propagate_backward(&mut self, gradients: &Matrix, previous_inputs: &Matrix, previous_gradients: &mut Matrix, rate: f32) {
        // XXX Add debug asserts on the sizes of the matrices
        let mut output_position = 0;

        if self.layers.len() > 1 {
            self.previous_gradients_buffer.resize(previous_inputs.rows(), previous_inputs.columns());
        }

        for i in 0..self.layers.len() {
           let layer_gradients = unsafe { Matrix::from_raw_parts(
                gradients.rows(), self.layers[i].outputs(),
                mem::transmute(gradients.as_ptr().offset((output_position * gradients.rows()) as isize)),
                0,
            ) };

            if i == 0 {
                self.layers[i].propagate_backward(&layer_gradients, previous_inputs, previous_gradients, rate);
            } else {
                self.layers[i].propagate_backward(&layer_gradients, previous_inputs, &mut self.previous_gradients_buffer, rate);
                vector::ops::add_assign(previous_gradients, &self.previous_gradients_buffer);
            }

            output_position += self.layers[i].outputs();

            mem::forget(layer_gradients);
        }
    }

    fn boxed_clone(&self) -> Box<Layer> {
        Box::new(self.clone())
    }
}
