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

use std::io::{BufReader, Result, Write};

use rand::Rng;
use rand::distributions::{Normal, Sample};

use smallmath::Vector;
use smallmath::matrix::{self, Matrix};

use gradient_descent::GradientDescent;
use layer::Layer;
use serialization::{File, Identifiable, read_error, read_line, Serializable};

#[derive(Clone)]
pub struct FullyConnectedLayer<G> {
    weights: Matrix,
    biases: Vector,

    gradient_descent: G,
    weight_gradients: Matrix,
    bias_gradients: Vector,

    transpose_buffer: Matrix,
}

impl<G> FullyConnectedLayer<G> where G: GradientDescent {
    pub fn new<R>(inputs: usize, outputs: usize, gradient_descent: G, rng: &mut R) -> FullyConnectedLayer<G> where R: Rng {
        FullyConnectedLayer {
            weights: Matrix::from_vec(inputs, outputs, {
                let mut distribution = Normal::new(0.0, (2.0 / outputs as f64).sqrt());
                (0..inputs * outputs).map(|_| distribution.sample(rng) as f32).collect::<Vec<_>>()
            }),
            biases: Vector::zeros(outputs),
            gradient_descent: gradient_descent,
            weight_gradients: Matrix::zeros(inputs, outputs),
            bias_gradients: Vector::zeros(outputs),
            transpose_buffer: Matrix::zeros(outputs, inputs),
        }
    }
}

impl<G> Layer for FullyConnectedLayer<G> where G: 'static + GradientDescent + Serializable {
    fn inputs(&self) -> usize {
        self.weights.rows()
    }

    fn outputs(&self) -> usize {
        self.weights.columns()
    }

    fn feed_forward(&self, inputs: &Matrix, outputs: &mut Matrix) {
        for i in 0..inputs.rows() {
            outputs[i].clone_from_slice(&self.biases);
        }

        matrix::ops::multiply(inputs, &self.weights, outputs);
    }

    fn propagate_backward(&mut self, gradients: &Matrix, previous_inputs: &Matrix, previous_gradients: &mut Matrix, rate: f32) {
        // bias_gradients = gradients, all inputs summed
        self.bias_gradients.clone_from_slice(&gradients[0]);
        for i in 1..gradients.rows() {
            self.bias_gradients += &gradients[i];
        }

        // weight_gradients = previous_inputs.transpose() * gradients
        self.transpose_buffer.resize(previous_inputs.columns(), previous_inputs.rows());
        matrix::ops::transpose(previous_inputs, &mut self.transpose_buffer);
        self.weight_gradients.zero();
        matrix::ops::multiply(&self.transpose_buffer, gradients, &mut self.weight_gradients);

        // previous_gradients = gradients * weights.transpose()
        self.transpose_buffer.resize(self.weights.columns(), self.weights.rows());
        matrix::ops::transpose(&self.weights, &mut self.transpose_buffer);
        previous_gradients.zero();
        matrix::ops::multiply(gradients, &self.transpose_buffer, previous_gradients);

        self.gradient_descent.descend(
            &mut self.weights, &mut self.biases,
            &self.weight_gradients, &self.bias_gradients,
            rate,
        );
    }

    fn boxed_clone(&self) -> Box<Layer> {
        Box::new(self.clone())
    }
}

impl<G> Identifiable for FullyConnectedLayer<G> where G: Identifiable {
    fn identifier() -> String {
        format!("FullyConnectedLayer<{}>", G::identifier())
    }

    fn get_identifier(&self) -> String {
        Self::identifier()
    }
}

impl<G> Serializable for FullyConnectedLayer<G> where G: Serializable {
    fn read_from_file(file: &mut BufReader<File>) -> Result<FullyConnectedLayer<G>> {
        let strings = read_line(file)?;

        if strings.len() < 1 || strings[0] != "Weights" {
            return read_error(file, "Cannot read layer weights!");
        }

        let weights = Matrix::read_from_file(file)?;
        let inputs = weights.rows();
        let outputs = weights.columns();

        let strings = read_line(file)?;

        if strings.len() < 1 || strings[0] != "Biases" {
            return read_error(file, "Cannot read layer biases!");
        }

        let biases = Vector::read_from_file(file)?;

        if biases.len() != outputs {
            return read_error(file, "Biases don't match layer outputs!");
        }

        let strings = read_line(file)?;

        if strings.len() < 1 || strings[0] != G::identifier() {
            return read_error(file, "Cannot read layer gradient descent!");
        }

        let gradient_descent = G::read_from_file(file)?;

        Ok(FullyConnectedLayer {
            weights: weights,
            biases: biases,
            gradient_descent: gradient_descent,
            weight_gradients: Matrix::zeros(inputs, outputs),
            bias_gradients: Vector::zeros(outputs),
            transpose_buffer: Matrix::zeros(outputs, inputs),
        })
    }

    fn write_to_file(&self, file: &mut File) -> Result<()> {
        let indentation = file.indentation();
        write!(file, "{}Weights\n", indentation)?;
        file.indent();
        self.weights.write_to_file(file)?;
        write!(file, "{}Biases\n", indentation)?;
        self.biases.write_to_file(file)?;
        write!(file, "{}{}\n", indentation, G::identifier())?;
        self.gradient_descent.write_to_file(file)?;
        file.unindent();
        Ok(())
    }
}
