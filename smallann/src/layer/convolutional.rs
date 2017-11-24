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

use rand::Rng;
use rand::distributions::{Normal, Sample};

use smallmath::Vector;
use smallmath::matrix::{self, Matrix};

use gradient_descent::GradientDescent;
use layer::Layer;
use serialization::Serializable;

#[derive(Clone)]
pub struct ConvolutionalLayer<G> {
    pub(crate) weights: Matrix,
    pub(crate) weights_mask: Matrix,
    pub(crate) biases: Vector,

    pub(crate) gradient_descent: G,
    weight_gradients: Matrix,
    bias_gradients: Vector,

    transpose_buffer: Matrix,
}

impl<G> ConvolutionalLayer<G> {
    pub fn new<R>(
        input_width: usize,
        input_height: usize,
        input_padding: usize,
        input_depth: usize,
        filter_width: usize,
        filter_height: usize,
        stride: usize,
        output_depth: usize,
        gradient_descent: G,
        rng: &mut R,
    ) -> ConvolutionalLayer<G> where R: Rng {
        let output_width = (input_width + 2 * input_padding - filter_width) / stride + 1;
        let output_height = (input_height + 2 * input_padding - filter_height) / stride + 1;

        let inputs = input_width * input_height * input_depth;
        let outputs = output_width * output_height * output_depth;

        let mut weights = Matrix::from_vec(
            inputs,
            outputs,
            {
                let mut distribution = Normal::new(0.0, (2.0 / outputs as f64).sqrt());
                (0..inputs * outputs).map(|_| distribution.sample(rng) as f32).collect::<Vec<_>>()
            },
        );

        let mut weights_mask = Matrix::zeros(inputs, outputs);
        for iz in 0..input_depth {
            for fy in 0..filter_height {
                for fx in 0..filter_width {
                    for oz in 0..output_depth {
                        for oy in 0..output_height {
                            for ox in 0..output_width {
                                let ix = (ox * stride + fx) as isize - input_padding as isize;
                                let iy = (oy * stride + fy) as isize - input_padding as isize;
                                if ix >= 0 && ix < input_width as isize && iy >= 0 && iy < input_height as isize {
                                    weights_mask[(
                                        ix as usize + iy as usize * input_width + iz as usize * input_width * input_height,
                                        ox + oy * output_width + oz * output_width * output_height,
                                    )] = 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        *weights.as_vector_mut() *= weights_mask.as_vector();

        let biases = Vector::zeros(outputs);

        ConvolutionalLayer {
            weights: weights,
            weights_mask: weights_mask,
            biases: biases,
            gradient_descent: gradient_descent,
            weight_gradients: Matrix::zeros(inputs, outputs),
            bias_gradients: Vector::zeros(outputs),
            transpose_buffer: Matrix::zeros(outputs, inputs),
        }
    }

    pub(crate) fn construct(
        inputs:usize, outputs: usize,
        weights: Matrix,
        weights_mask: Matrix,
        biases: Vector,
        gradient_descent: G,
    ) -> ConvolutionalLayer<G> {
        ConvolutionalLayer {
            weights: weights,
            weights_mask: weights_mask,
            biases: biases,
            gradient_descent: gradient_descent,
            weight_gradients: Matrix::zeros(inputs, outputs),
            bias_gradients: Vector::zeros(outputs),
            transpose_buffer: Matrix::zeros(outputs, inputs),
        }
    }
}

impl<G> Layer for ConvolutionalLayer<G> where G: 'static + GradientDescent + Serializable {
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

        *self.weights.as_vector_mut() *= self.weights_mask.as_vector();
    }

    fn boxed_clone(&self) -> Box<Layer> {
        Box::new(self.clone())
    }
}
