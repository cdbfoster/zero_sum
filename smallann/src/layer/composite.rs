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

use std::cell::RefCell;
use std::io::{BufReader, Result, Write};
use std::mem;
use std::str::FromStr;

use smallmath::matrix::{self, Matrix};

use layer::Layer;
use serialization::{File, Identifiable, read_error, read_layer, read_line, Serializable};

pub struct CompositeLayer {
    inputs: usize,
    outputs: usize,
    layers: Vec<Box<Layer>>,

    input_buffers: RefCell<Vec<Matrix>>,
    output_buffers: RefCell<Vec<Matrix>>,
    previous_gradient_buffers: RefCell<Vec<Matrix>>,
}

impl CompositeLayer {
    pub fn new(layers: Vec<Box<Layer>>) -> CompositeLayer {
        let mut inputs = 0;
        let mut outputs = 0;
        let mut input_buffers = Vec::new();
        let mut output_buffers = Vec::new();
        let mut previous_gradient_buffers = Vec::new();

        for layer in &layers {
            inputs += layer.inputs();
            outputs += layer.outputs();
            input_buffers.push(Matrix::zeros(10, layer.inputs()));
            output_buffers.push(Matrix::zeros(10, layer.outputs()));
            previous_gradient_buffers.push(Matrix::zeros(10, layer.inputs()));
        }

        CompositeLayer {
            inputs: inputs,
            outputs: outputs,
            layers: layers,
            input_buffers: RefCell::new(input_buffers),
            output_buffers: RefCell::new(output_buffers),
            previous_gradient_buffers: RefCell::new(previous_gradient_buffers),
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
        let mut input_buffers = self.input_buffers.borrow_mut();
        let mut output_buffers = self.output_buffers.borrow_mut();

        let mut input_position = 0;
        let mut output_position = 0;

        for i in 0..self.layers.len() {
            let single_input = unsafe { Matrix::from_raw_parts(
                1, self.layers[i].inputs(),
                mem::transmute(inputs.as_ptr().offset(input_position as isize)),
                0,
            ) };
            let mut single_output = unsafe { Matrix::from_raw_parts(
                1, self.layers[i].outputs(),
                outputs.as_mut_ptr().offset(output_position as isize),
                0,
            ) };

            {
                let layer_inputs = if inputs.rows() == 1 {
                    &single_input
                } else {
                    input_buffers[i].resize(inputs.rows(), self.layers[i].inputs());
                    matrix::ops::clone_region(inputs, 0, input_position, &mut input_buffers[i], 0, 0, inputs.rows(), self.layers[i].inputs());
                    &input_buffers[i]
                };

                let layer_outputs = if inputs.rows() == 1 {
                    &mut single_output
                } else {
                    output_buffers[i].resize(inputs.rows(), self.layers[i].outputs());
                    &mut output_buffers[i]
                };

                self.layers[i].feed_forward(layer_inputs, layer_outputs);

                if inputs.rows() > 1 {
                    matrix::ops::clone_region(layer_outputs, 0, 0, outputs, 0, output_position, inputs.rows(), self.layers[i].outputs());
                }
            }

            input_position += self.layers[i].inputs();
            output_position += self.layers[i].outputs();

            mem::forget(single_input);
            mem::forget(single_output);
        }
    }

    fn propagate_backward(&mut self, gradients: &Matrix, previous_inputs: &Matrix, previous_gradients: &mut Matrix, rate: f32) {
        let mut input_buffers = self.input_buffers.borrow_mut();
        let mut output_buffers = self.output_buffers.borrow_mut();
        let mut previous_gradient_buffers = self.previous_gradient_buffers.borrow_mut();

        let mut input_position = 0;
        let mut output_position = 0;

        for i in 0..self.layers.len() {
            let layer_gradients = {
                output_buffers[i].resize(gradients.rows(), self.layers[i].outputs());
                matrix::ops::clone_region(gradients, 0, output_position, &mut output_buffers[i], 0, 0, gradients.rows(), self.layers[i].outputs());
                &output_buffers[i]
            };

            let layer_previous_inputs = {
                input_buffers[i].resize(gradients.rows(), self.layers[i].inputs());
                matrix::ops::clone_region(previous_inputs, 0, input_position, &mut input_buffers[i], 0, 0, gradients.rows(), self.layers[i].inputs());
                &input_buffers[i]
            };

            let layer_previous_gradients = {
                previous_gradient_buffers[i].resize(gradients.rows(), self.layers[i].inputs());
                &mut previous_gradient_buffers[i]
            };

            self.layers[i].propagate_backward(layer_gradients, layer_previous_inputs, layer_previous_gradients, rate);

            matrix::ops::clone_region(layer_previous_gradients, 0, 0, previous_gradients, 0, input_position, gradients.rows(), self.layers[i].inputs());

            input_position += self.layers[i].inputs();
            output_position += self.layers[i].outputs();
        }
    }
}

impl Identifiable for CompositeLayer {
    fn identifier() -> String {
        String::from("CompositeLayer")
    }

    fn get_identifier(&self) -> String {
        Self::identifier()
    }
}

impl Serializable for CompositeLayer {
    fn read_from_file(file: &mut BufReader<File>) -> Result<CompositeLayer> {
        let strings = read_line(file)?;

        if strings.len() < 1 {
            return read_error(file, "Cannot read layer count!");
        }

        let layer_count = if let Ok(layer_count) = usize::from_str(&strings[0]) {
            layer_count
        } else {
            return read_error(file, "Cannot parse layer count!");
        };

        let mut layers: Vec<Box<Layer>> = Vec::new();
        for _ in 0..layer_count {
            let layer = read_layer(file)?;
            layers.push(layer);
        }

        Ok(CompositeLayer::new(layers))
    }

    fn write_to_file(&self, file: &mut File) -> Result<()> {
        let indentation = file.indentation();
        write!(file, "{}{}\n", indentation, self.layers.len())?;
        for layer in &self.layers {
            write!(file, "{}{}\n", indentation, layer.get_identifier())?;
            file.indent();
            layer.write_to_file(file)?;
            file.unindent();
        }
        Ok(())
    }
}
