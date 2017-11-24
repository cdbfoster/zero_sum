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

use std::marker::PhantomData;

use smallmath::Matrix;

use activation_function::ActivationFunction;
use layer::Layer;
use serialization::Identifiable;

#[derive(Clone)]
pub struct ActivationLayer<F> {
    pub(crate) size: usize,
    activation_function: PhantomData<F>,
}

impl<F> ActivationLayer<F> {
    pub fn new(size: usize) -> ActivationLayer<F> {
        ActivationLayer {
            size: size,
            activation_function: PhantomData,
        }
    }
}

impl<F> Layer for ActivationLayer<F> where F: 'static + ActivationFunction + Identifiable {
    fn inputs(&self) -> usize {
        self.size
    }

    fn outputs(&self) -> usize {
        self.size
    }

    fn feed_forward(&self, inputs: &Matrix, outputs: &mut Matrix) {
        if inputs.rows() == 1 && inputs.columns() == 1 {
            outputs[(0, 0)] = <F as ActivationFunction>::f(inputs[(0, 0)]);
        } else {
            <F as ActivationFunction>::f_vector(inputs, outputs);
        }
        assert!(outputs.iter().find(|x| x.is_nan()).is_none(), "NaN here!");
    }

    fn propagate_backward(&mut self, gradients: &Matrix, previous_inputs: &Matrix, previous_gradients: &mut Matrix, _: f32) {
        // previous_gradients = f'(previous_inputs) .* gradients
        <F as ActivationFunction>::f_prime_vector(previous_inputs, previous_gradients);
        *previous_gradients.as_vector_mut() *= gradients.as_vector();
    }

    fn boxed_clone(&self) -> Box<Layer> {
        Box::new(self.clone())
    }
}
