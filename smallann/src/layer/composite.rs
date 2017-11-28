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

use layer::Layer;

#[derive(Clone)]
pub struct CompositeLayer {
    inputs: usize,
    outputs: usize,
    pub(crate) layers: Vec<Box<Layer>>,
}

impl CompositeLayer {
    pub fn new(layers: Vec<Box<Layer>>) -> CompositeLayer {
        let mut inputs = 0;
        let mut outputs = 0;

        for layer in &layers {
            inputs += layer.inputs();
            outputs += layer.outputs();
        }

        CompositeLayer {
            inputs: inputs,
            outputs: outputs,
            layers: layers,
        }
    }
}

impl Layer for CompositeLayer {
    fn inputs(&self) -> usize {
        self.inputs
    }

    fn outputs(&self) -> usize {
        self.outputs
    }

    fn feed_forward(&self, inputs: &Matrix, outputs: &mut Matrix) {
        // XXX Add debug asserts on the sizes of the matrices
        let mut input_position = 0;
        let mut output_position = 0;

        for i in 0..self.layers.len() {
            let layer_inputs = unsafe { Matrix::from_raw_parts(
                inputs.rows(), self.layers[i].inputs(),
                mem::transmute(inputs.as_ptr().offset((input_position * inputs.rows()) as isize)),
                0,
            ) };
            let mut layer_outputs = unsafe { Matrix::from_raw_parts(
                inputs.rows(), self.layers[i].outputs(),
                outputs.as_mut_ptr().offset((output_position * inputs.rows()) as isize),
                0,
            ) };

            self.layers[i].feed_forward(&layer_inputs, &mut layer_outputs);

            input_position += self.layers[i].inputs();
            output_position += self.layers[i].outputs();

            mem::forget(layer_inputs);
            mem::forget(layer_outputs);
        }
    }

    fn propagate_backward(&mut self, gradients: &Matrix, previous_inputs: &Matrix, previous_gradients: &mut Matrix, rate: f32) {
        // XXX Add debug asserts on the sizes of the matrices
        let mut input_position = 0;
        let mut output_position = 0;

        for i in 0..self.layers.len() {
           let layer_gradients = unsafe { Matrix::from_raw_parts(
                gradients.rows(), self.layers[i].outputs(),
                mem::transmute(gradients.as_ptr().offset((output_position * gradients.rows()) as isize)),
                0,
            ) };

            let layer_previous_inputs = unsafe { Matrix::from_raw_parts(
                gradients.rows(), self.layers[i].inputs(),
                mem::transmute(previous_inputs.as_ptr().offset((input_position * gradients.rows()) as isize)),
                0,
            ) };

            let mut layer_previous_gradients = unsafe { Matrix::from_raw_parts(
                gradients.rows(), self.layers[i].inputs(),
                previous_gradients.as_mut_ptr().offset((input_position * gradients.rows()) as isize),
                0,
            ) };

            self.layers[i].propagate_backward(&layer_gradients, &layer_previous_inputs, &mut layer_previous_gradients, rate);

            input_position += self.layers[i].inputs();
            output_position += self.layers[i].outputs();

            mem::forget(layer_gradients);
            mem::forget(layer_previous_inputs);
            mem::forget(layer_previous_gradients);
        }
    }

    fn boxed_clone(&self) -> Box<Layer> {
        Box::new(self.clone())
    }
}
