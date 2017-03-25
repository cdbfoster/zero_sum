//
// This file is part of zero_sum.
//
// zero_sum is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// zero_sum is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with zero_sum. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright 2016-2017 Chris Foster
//

use std::cell::RefCell;
use std::cmp;
use std::marker::PhantomData;

use blas::c as blas;
use rand::distributions::{Normal, Sample};
use rand::{thread_rng};

pub use self::activation_function::{ActivationFunction, ReLuActivationFunction, TanHActivationFunction};
pub use self::feature_representation::gather_features;
pub use self::gradient_descent::{AdadeltaGradientDescent, GradientDescent, SimpleGradientDescent};
pub use self::matrix::{MatrixCm, MatrixRm};
pub use self::serialization::{read_network, write_network};

/// Artificial neural network
#[derive(Clone, Debug)]
pub struct Ann<A, F, G> where
    A: ActivationFunction,
    F: ActivationFunction,
    G: GradientDescent {
    activation_function: PhantomData<A>,
    final_activation_function: PhantomData<F>,

    weights: Vec<MatrixCm>,
    weight_masks: Vec<Option<MatrixCm>>,
    biases: Vec<MatrixRm>,

    gradient_descent: G,

    pre_activations_buffer: RefCell<MatrixRm>,
    activations_buffer: RefCell<MatrixRm>,
}

impl<A, F, G> Ann<A, F, G> where
    A: ActivationFunction,
    F: ActivationFunction,
    G: GradientDescent {
    pub fn new(
        inputs: usize,
        hidden_layers: &[usize],
        outputs: usize,
        weight_masks: &[Option<MatrixCm>],
        gradient_descent: G,
    ) -> Ann<A, F, G> {
        assert!(inputs > 0, "Invalid number of inputs!");
        assert!(hidden_layers.iter().find(|&&l| l == 0).is_none(), "Invalid number of hidden-layer neurons!");
        assert!(outputs > 0, "Invalid number of outputs!");

        // Used in propagate_forward_simple
        let activations_buffer = RefCell::new(MatrixRm::zeros(1, cmp::max(
            if let Some(max) = hidden_layers.iter().max() {
                *max
            } else {
                0
            },
            outputs,
        )));

        let mut ann = Ann {
            activation_function: PhantomData,
            final_activation_function: PhantomData,
            weights: Vec::new(),
            weight_masks: Vec::new(),
            biases: Vec::new(),
            gradient_descent: gradient_descent,
            pre_activations_buffer: activations_buffer.clone(),
            activations_buffer: activations_buffer,
        };

        for layer in 0..hidden_layers.len() + 1 {
            let input_size = if layer == 0 {
                inputs
            } else {
                hidden_layers[layer - 1]
            };

            let output_size = if layer < hidden_layers.len() {
                hidden_layers[layer]
            } else {
                outputs
            };

            let weight_mask = if layer < weight_masks.len() {
                weight_masks[layer].clone()
            } else {
                None
            };

            ann.weights.push(MatrixCm::from_vec(
                input_size, output_size,
                if A::new().as_any().is::<ReLuActivationFunction>() {
                    let mut distribution = Normal::new(0.0, (2.0 / output_size as f64).sqrt());
                    (0..input_size * output_size).map(|i|
                        distribution.sample(&mut thread_rng()) as f32 * if let Some(ref mask) = weight_mask {
                            mask.values[i]
                        } else {
                            1.0
                        }
                    ).collect::<Vec<_>>()
                } else {
                    vec![0.0; input_size * output_size]
                },
            ));

            ann.weight_masks.push(weight_mask);

            ann.biases.push(MatrixRm::zeros(1, output_size));
        }

        ann
    }

    pub fn allocate_activation_buffers(&self, inputs: usize) -> (Vec<MatrixRm>, Vec<MatrixRm>) {
        let buffer = self.weights.iter().map(|l| MatrixRm::zeros(inputs, l.columns)).collect::<Vec<_>>();
        (buffer.clone(), buffer)
    }

    pub fn propagate_forward(&self, inputs: &MatrixRm, pre_activations: &mut [MatrixRm], activations: &mut [MatrixRm]) {
        debug_assert!(inputs.columns == self.weights[0].rows, "Incorrect number of inputs!");
        debug_assert!(pre_activations.len() == self.weights.len(), "pre_activations.len() doesn't match the number of layers!");
        debug_assert!(activations.len() == self.weights.len(), "activations.len() doesn't match the number of layers!");
        debug_assert!((|| {
            for i in 0..self.weights.len() {
                if pre_activations[i].rows != inputs.rows || pre_activations[i].columns != self.weights[i].columns {
                    return false;
                }
            }
            true
        })(), "Incorrect matrix dimensions in pre_activations!");
        debug_assert!((|| {
            for i in 0..self.biases.len() {
                if activations[i].rows != inputs.rows || activations[i].columns != self.biases[i].columns {
                    return false;
                }
            }
            true
        })(), "Incorrect matrix dimensions in activations!");

        for layer in 0..self.weights.len() {
            let pre_activations = &mut pre_activations[layer];
            let weights = &self.weights[layer];

            for i in 0..pre_activations.rows {
                pre_activations[i].clone_from_slice(&self.biases[layer].values);
            }

            {
                let layer_inputs = if layer == 0 {
                    inputs
                } else {
                    &activations[layer - 1]
                };

                // pre_activations = layer_inputs * weights + biases
                let (m, n, k) = (layer_inputs.rows as i32, weights.columns as i32, layer_inputs.columns as i32);
                if n > 1 {
                    blas::sgemm(
                        blas::Layout::RowMajor, blas::Transpose::None, blas::Transpose::Ordinary,
                        m, n, k,
                        1.0, layer_inputs.values.as_slice(), k,
                        weights.values.as_slice(), weights.rows as i32,
                        1.0, pre_activations.values.as_mut_slice(), n,
                    );
                } else {
                    blas::sgemv(
                        blas::Layout::RowMajor, blas::Transpose::None,
                        m, k,
                        1.0, layer_inputs.values.as_slice(), k,
                        weights.values.as_slice(), 1,
                        1.0, pre_activations.values.as_mut_slice(), 1,
                    );
                }
            }

            // activations = f(pre_activations)
            for i in 0..pre_activations.values.len() {
                activations[layer].values[i] = if layer != self.weights.len() - 1 {
                    A::f(pre_activations.values[i])
                } else {
                    F::f(pre_activations.values[i])
                };
            }
        }
    }

    pub fn propagate_forward_simple(&self, inputs: &MatrixRm, outputs: &mut MatrixRm) {
        debug_assert!(inputs.columns == self.weights[0].rows, "Incorrect number of inputs!");
        debug_assert!(outputs.columns == self.weights.last().unwrap().columns, "Incorrect number of outputs!");
        debug_assert!(inputs.rows == outputs.rows, "Row mismatch between inputs and outputs!");

        for layer in 0..self.weights.len() {
            let mut pre_activations = self.pre_activations_buffer.borrow_mut();
            let mut activations = self.activations_buffer.borrow_mut();
            let weights = &self.weights[layer];

            pre_activations.resize(inputs.rows, weights.columns);

            for i in 0..pre_activations.rows {
                pre_activations[i].clone_from_slice(&self.biases[layer].values);
            }

            {
                let layer_inputs = if layer == 0 {
                    inputs
                } else {
                    &*activations
                };

                // pre_activations = layer_inputs * weights + biases
                let (m, n, k) = (layer_inputs.rows as i32, weights.columns as i32, layer_inputs.columns as i32);
                if n == 1 {
                    blas::sgemv(
                        blas::Layout::RowMajor, blas::Transpose::None,
                        m, k,
                        1.0, layer_inputs.values.as_slice(), k,
                        weights.values.as_slice(), 1,
                        1.0, pre_activations.values.as_mut_slice(), 1,
                    );
                } else {
                    blas::sgemm(
                        blas::Layout::RowMajor, blas::Transpose::None, blas::Transpose::Ordinary,
                        m, n, k,
                        1.0, layer_inputs.values.as_slice(), k,
                        weights.values.as_slice(), weights.rows as i32,
                        1.0, pre_activations.values.as_mut_slice(), n,
                    );
                }
            }

            if layer < self.weights.len() - 1 {
                activations.resize(pre_activations.rows, pre_activations.columns);

                for i in 0..pre_activations.values.len() {
                    activations.values[i] = A::f(pre_activations.values[i]);
                }
            } else {
                for i in 0..pre_activations.values.len() {
                    outputs.values[i] = F::f(pre_activations.values[i]);
                }
            };
        }
    }

    pub fn allocate_gradient_buffers(&self) -> (Vec<MatrixCm>, Vec<MatrixRm>) {
        (
            self.weights.iter().map(|l| MatrixCm::zeros(l.rows, l.columns)).collect::<Vec<_>>(),
            self.biases.iter().map(|l| MatrixRm::zeros(l.rows, l.columns)).collect::<Vec<_>>(),
        )
    }

    pub fn propagate_backward(&self,
        error_derivatives: &MatrixRm,
        inputs: &MatrixRm, pre_activations: &[MatrixRm], activations: &[MatrixRm],
        weight_gradients: &mut [MatrixCm], bias_gradients: &mut [MatrixRm],
    ) {
        debug_assert!(inputs.rows == error_derivatives.rows && inputs.columns == self.weights[0].rows, "Incorrect number of inputs!");
        debug_assert!(pre_activations.len() == self.weights.len(), "pre_activations.len() does not match the number of layers!");
        debug_assert!(activations.len() == self.weights.len(), "activations.len() does not match the number of layers!");
        debug_assert!(weight_gradients.len() == self.weights.len(), "weight_gradients.len() does not match the number of layers!");
        debug_assert!(bias_gradients.len() == self.weights.len(), "bias_gradients.len() does not match the number of layers!");
        debug_assert!((|| {
            for i in 0..self.weights.len() {
                if pre_activations[i].rows != error_derivatives.rows || pre_activations[i].columns != self.weights[i].columns {
                    return false;
                }
            }
            true
        })(), "Incorrect dimensions in pre_activations!");
        debug_assert!((|| {
            for i in 0..self.weights.len() {
                if activations[i].rows != error_derivatives.rows || activations[i].columns != self.weights[i].columns {
                    return false;
                }
            }
            true
        })(), "Incorrect dimensions in activations!");
        debug_assert!((|| {
            for i in 0..self.weights.len() {
                if !weight_gradients[i].same_size(&self.weights[i]) {
                    return false;
                }
            }
            true
        })(), "Incorrect dimensions in weight_gradients!");
        debug_assert!((|| {
            for i in 0..self.biases.len() {
                if !bias_gradients[i].same_size(&self.biases[i]) {
                    return false;
                }
            }
            true
        })(), "Incorrect dimensions in bias_gradients!");

        let mut delta = error_derivatives.clone();
        let mut delta_tmp = delta.clone();

        for layer in (0..self.weights.len()).rev() {
            // bias_gradients[layer] = delta, all inputs summed
            bias_gradients[layer].values.clone_from_slice(&delta[0]);
            for i in 1..delta.rows {
                blas::saxpy(
                    delta.columns as i32,
                    1.0,
                    &delta[i], 1,
                    bias_gradients[layer].values.as_mut_slice(), 1,
                );
            }

            // weight_gradients[layer] = previous_activations.transpose() * delta
            let previous_activations = if layer == 0 {
                inputs
            } else {
                &activations[layer - 1]
            };
            blas::sgemm(
                blas::Layout::ColumnMajor, blas::Transpose::None, blas::Transpose::Ordinary,
                previous_activations.columns as i32, delta.columns as i32, previous_activations.rows as i32,
                1.0, previous_activations.values.as_slice(), previous_activations.columns as i32,
                delta.values.as_slice(), delta.columns as i32,
                0.0, weight_gradients[layer].values.as_mut_slice(), weight_gradients[layer].rows as i32,
            );

            if layer > 0 {
                // delta = delta * self.weights[layer].transpose() .* f_prime(pre_activations[layer - 1])
                let (m, n, k) = (delta.rows as i32, self.weights[layer].rows as i32, delta.columns as i32);
                delta_tmp.resize(m as usize, n as usize);

                blas::sgemm(
                    blas::Layout::RowMajor, blas::Transpose::None, blas::Transpose::None,
                    m, n, k,
                    1.0, delta.values.as_slice(), k,
                    self.weights[layer].values.as_slice(), n,
                    0.0, delta_tmp.values.as_mut_slice(), n,
                );

                delta.resize(delta_tmp.rows, delta_tmp.columns);

                for i in 0..delta_tmp.values.len() {
                    delta.values[i] = delta_tmp.values[i] * A::f_prime(pre_activations[layer - 1].values[i]);
                }
            }
        }
    }

    pub fn train(&mut self, inputs: &MatrixRm, targets: &MatrixRm, rate: f32) {
        debug_assert!(inputs.columns == self.weights[0].rows, "Incorrect number of inputs!");
        debug_assert!(inputs.rows == targets.rows, "Incorrect number of targets!");
        debug_assert!(targets.columns == self.weights.last().unwrap().columns, "Incorrect number of outputs!");

        let (mut pre_activations, mut activations) = self.allocate_activation_buffers(inputs.rows);
        self.propagate_forward(
            inputs,
            &mut pre_activations,
            &mut activations,
        );

        let mut error_derivatives = MatrixRm::zeros(inputs.rows, self.weights.last().unwrap().columns);
        calculate_error_derivatives::<F>(activations.last().unwrap(), targets, &mut error_derivatives);

        let (mut weight_gradients, mut bias_gradients) = self.allocate_gradient_buffers();
        self.propagate_backward(
            &error_derivatives,
            inputs,
            &pre_activations, &activations,
            &mut weight_gradients, &mut bias_gradients,
        );

        self.gradient_descent.descend(
            &mut self.weights,
            &self.weight_masks,
            &mut self.biases,
            &weight_gradients,
            &bias_gradients,
            rate,
        );
    }
}

/// Returns the total sum of the error.
pub fn calculate_error<F>(outputs: &MatrixRm, targets: &MatrixRm, error: &mut MatrixRm) -> f32 where F: ActivationFunction {
    debug_assert!(outputs.same_size(targets), "outputs's dimensions are different than targets's!");
    debug_assert!(outputs.same_size(error), "outputs's dimensions are different than error's!");

    if F::new().as_any().is::<TanHActivationFunction>() {
        // error = (outputs - targets) * (outputs - targets)
        error.values.clone_from(&outputs.values);
        blas::saxpy((error.rows * error.columns) as i32, -1.0, &targets.values, 1, &mut error.values, 1);

        for value in &mut error.values {
            *value *= *value;
        }
    } else { // Linear
        // error = outputs - targets
        error.values.clone_from(&outputs.values);
        blas::saxpy((error.rows * error.columns) as i32, -1.0, &targets.values, 1, &mut error.values, 1);
    }

    error.values.iter().sum()
}

pub fn calculate_error_derivatives<F>(outputs: &MatrixRm, targets: &MatrixRm, error_derivatives: &mut MatrixRm) where F: ActivationFunction {
    debug_assert!(outputs.same_size(targets), "outputs's dimensions are different than targets's!");
    debug_assert!(outputs.same_size(error_derivatives), "outputs's dimensions are different than error_derivatives's!");

    if F::new().as_any().is::<TanHActivationFunction>() {
        // error_derivatives = (outputs - targets) .* (1.0 - (outputs ^ 2))
        error_derivatives.values.clone_from(&outputs.values);
        blas::saxpy(
            (error_derivatives.rows * error_derivatives.columns) as i32,
            -1.0,
            &targets.values, 1,
            &mut error_derivatives.values, 1,
        );

        for i in 0..error_derivatives.values.len() {
            error_derivatives.values[i] *= 1.0 - (outputs.values[i] * outputs.values[i]);
        }
    } else { // Linear
        // error_derivatives = (outputs - targets) > 0.0 ? 1.0 : -1.0
        error_derivatives.values.clone_from(&outputs.values);
        blas::saxpy(
            (error_derivatives.rows * error_derivatives.columns) as i32,
            -1.0,
            &targets.values, 1,
            &mut error_derivatives.values, 1,
        );

        for value in &mut error_derivatives.values {
            *value = if *value > 0.0 {
                1.0
            } else {
                -1.0
            };
        }
    }
}

mod activation_function;
mod feature_representation;
mod gradient_descent;
mod matrix;
mod serialization;
