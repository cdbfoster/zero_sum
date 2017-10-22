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
use std::io::{BufReader, Error, ErrorKind, Result, Write};
use std::str::FromStr;

use smallmath::Matrix;

use layer::Layer;
use loss_function::LossFunction;
use serialization::{File, Identifiable, read_error, read_layer, read_line, Serializable};

pub struct Ann {
    layers: Vec<Box<Layer>>,
    layer_buffers: RefCell<Vec<Matrix>>,
    loss_derivatives: Matrix,
}

impl Ann {
    pub fn new(layers: Vec<Box<Layer>>) -> Ann {
        assert!(!layers.is_empty(), "Ann must have at least one layer!");
        assert!((|| {
            for i in 0..(layers.len() - 1) {
                if layers[i].outputs() != layers[i + 1].inputs() {
                    return false;
                }
            }
            true
        })(), "Layer outputs don't match the next layer's inputs!");

        let layer_buffers = layers.iter().map(|l| Matrix::zeros(1, l.outputs())).collect::<Vec<_>>();
        let loss_derivatives = Matrix::zeros(1, layers.last().unwrap().outputs());

        Ann {
            layers: layers,
            layer_buffers: RefCell::new(layer_buffers),
            loss_derivatives: loss_derivatives,
        }
    }

    pub fn from_file(filename: &str) -> Result<Ann> {
        let mut file = BufReader::new(File::open(filename)?);

        let strings = read_line(&mut file)?;

        if strings.len() < 1 || strings[0] != "Ann" {
            return read_error(&file, "Cannot read network!");
        }

        Ann::read_from_file(&mut file)
    }

    pub fn to_file(&self, filename: &str) -> Result<()> {
        let mut file = File::create(filename)?;

        write!(file, "Ann\n")?;
        file.indent();
        self.write_to_file(&mut file)
    }

    pub fn classify(&self, inputs: &Matrix, outputs: &mut Matrix) {
        debug_assert!(inputs.rows() == outputs.rows(), "Input and output matrices have different numbers of rows!");
        debug_assert!(inputs.columns() == self.layers.first().unwrap().inputs(), "Input matrix has the wrong number of columns!");
        debug_assert!(outputs.columns() == self.layers.last().unwrap().outputs(), "Output matrix has the wrong number of columns!");

        let layer_count = self.layers.len();
        let mut layer_buffers = self.layer_buffers.borrow_mut();

        for layer_buffer in layer_buffers[0..layer_count - 1].iter_mut() {
            let columns = layer_buffer.columns();
            layer_buffer.resize(inputs.rows(), columns);
        }

        for i in 0..self.layers.len() {
            let (previous_buffer, next_buffer) = layer_buffers.split_at_mut(i);

            next_buffer[0].resize(inputs.rows(), self.layers[i].outputs());

            self.layers[i].feed_forward(
                if i > 0 {
                    previous_buffer.last().unwrap()
                } else {
                    inputs
                },
                if i < self.layers.len() - 1 {
                    next_buffer.first_mut().unwrap()
                } else {
                    outputs
                },
            );
        }
    }

    pub fn train<L>(&mut self, inputs: &Matrix, targets: &Matrix, rate: f32) where L: LossFunction {
        debug_assert!(inputs.rows() == targets.rows(), "Input and target matrices have different numbers of rows!");
        debug_assert!(inputs.columns() == self.layers.first().unwrap().inputs(), "Input matrix has the wrong number of columns!");
        debug_assert!(targets.columns() == self.layers.last().unwrap().outputs(), "Target matrix has the wrong number of columns!");

        let layer_count = self.layers.len();
        let mut layer_buffers = self.layer_buffers.borrow_mut();

        for layer_buffer in layer_buffers.iter_mut() {
            let columns = layer_buffer.columns();
            layer_buffer.resize(inputs.rows(), columns);
        }

        for i in 0..layer_count {
            let (previous_buffer, next_buffer) = layer_buffers.split_at_mut(i);

            next_buffer[0].resize(inputs.rows(), self.layers[i].outputs());

            self.layers[i].feed_forward(
                if i > 0 {
                    previous_buffer.last().unwrap()
                } else {
                    inputs
                },
                next_buffer.first_mut().unwrap(),
            );
        }

        self.loss_derivatives.resize(targets.rows(), targets.columns());
        <L as LossFunction>::l_prime_vector(layer_buffers.last().unwrap(), targets, &mut self.loss_derivatives);

        for i in (0..layer_count).rev() {
            let (previous_inputs_buffer, gradients) = layer_buffers.split_at_mut(i);
            let (previous_gradients_buffer, gradients_buffer) = gradients.split_at_mut(1);

            previous_gradients_buffer[0].resize(inputs.rows(), self.layers[i].inputs());

            self.layers[i].propagate_backward(
                if i < layer_count - 1 {
                    gradients_buffer.first().unwrap()
                } else {
                    &self.loss_derivatives
                },
                if i > 0 {
                    previous_inputs_buffer.last().unwrap()
                } else {
                    inputs
                },
                previous_gradients_buffer.first_mut().unwrap(),
                rate,
            );
        }
    }
}

impl Identifiable for Ann {
    fn identifier() -> String {
        String::from("Ann")
    }

    fn get_identifier(&self) -> String {
        Self::identifier()
    }
}

impl Serializable for Ann {
    fn read_from_file(file: &mut BufReader<File>) -> Result<Ann> {
        let strings = read_line(file)?;

        if strings.len() < 1 {
            return Err(Error::new(ErrorKind::Other, "Cannot read layer count!"));
        }

        let layer_count = if let Ok(layer_count) = usize::from_str(&strings[0]) {
            layer_count
        } else {
            return Err(Error::new(ErrorKind::Other, "Cannot parse layer count!"));
        };

        let mut layers: Vec<Box<Layer>> = Vec::new();
        for _ in 0..layer_count {
            let layer = read_layer(file)?;

            if let Some(last) = layers.last() {
                if layer.inputs() != last.outputs() {
                    return Err(Error::new(ErrorKind::Other, "Layer inputs doesn't match the previous layer's outputs!"));
                }
            }

            layers.push(layer);
        }

        Ok(Ann::new(layers))
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
