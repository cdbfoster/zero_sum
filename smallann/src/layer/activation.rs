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
use std::marker::PhantomData;
use std::str::FromStr;

use smallmath::Matrix;

use activation_function::ActivationFunction;
use layer::Layer;
use serialization::{File, Identifiable, read_error, read_line, Serializable};

pub struct ActivationLayer<F> where F: ActivationFunction {
    size: usize,
    activation_function: PhantomData<F>,
}

impl<F> ActivationLayer<F> where F: ActivationFunction {
    pub fn new(size: usize) -> ActivationLayer<F> {
        ActivationLayer {
            size: size,
            activation_function: PhantomData,
        }
    }
}

impl<F> Layer for ActivationLayer<F> where F: ActivationFunction + Identifiable {
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
}

impl<F> Identifiable for ActivationLayer<F> where F: ActivationFunction + Identifiable {
    fn identifier() -> String {
        format!("ActivationLayer<{}>", F::identifier())
    }

    fn get_identifier(&self) -> String {
        Self::identifier()
    }
}

impl<F> Serializable for ActivationLayer<F> where F: ActivationFunction + Identifiable {
    fn read_from_file(file: &mut BufReader<File>) -> Result<ActivationLayer<F>> {
        let strings = read_line(file)?;

        if strings.len() < 1 {
            return read_error(file, "Cannot read layer weights!");
        }

        let size = if let Ok(size) = usize::from_str(&strings[0]) {
            size
        } else {
            return read_error(file, "Cannot parse layer size!");
        };

        Ok(ActivationLayer::<F>::new(size))
    }

    fn write_to_file(&self, file: &mut File) -> Result<()> {
        let indentation = file.indentation();
        write!(file, "{}{}\n", indentation, self.size)
    }
}
