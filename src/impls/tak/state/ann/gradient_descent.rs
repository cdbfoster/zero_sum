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

use blas::c as blas;

use impls::tak::state::ann::{MatrixCm, MatrixRm};

pub trait GradientDescent {
    fn descend(
        &mut self,
        weights: &mut [MatrixCm],
        weight_masks: &[Option<MatrixCm>],
        biases: &mut [MatrixRm],
        weight_gradients: &[MatrixCm],
        bias_gradients: &[MatrixRm],
        rate: f32,
    );
}

#[derive(Clone, Debug)]
pub struct SimpleGradientDescent;

impl GradientDescent for SimpleGradientDescent {
    fn descend(
        &mut self,
        weights: &mut [MatrixCm],
        weight_masks: &[Option<MatrixCm>],
        biases: &mut [MatrixRm],
        weight_gradients: &[MatrixCm],
        bias_gradients: &[MatrixRm],
        rate: f32,
    ) {
        debug_assert!(weights.len() == weight_masks.len(), "weights.len() doesn't match weight_masks.len()!");
        debug_assert!(weights.len() == weight_gradients.len(), "weights.len() doesn't match weight_gradients.len()!");
        debug_assert!(biases.len() == bias_gradients.len(), "biases.len() doesn't match bias_gradients.len()!");
        debug_assert!((|| {
            for i in 0..weights.len() {
                if let Some(ref mask) = weight_masks[i] {
                    if !weights[i].same_size(mask) {
                        return false;
                    }
                }
            }
            true
        })(), "Dimension mismatch between weights and weight_masks!");
        debug_assert!((|| {
            for i in 0..weights.len() {
                if !weights[i].same_size(&weight_gradients[i]) {
                    return false;
                }
            }
            true
        })(), "Dimension mismatch between weights and weight_gradients!");
        debug_assert!((|| {
            for i in 0..biases.len() {
                if !biases[i].same_size(&bias_gradients[i]) {
                    return false;
                }
            }
            true
        })(), "Dimension mismatch between biases and bias_gradients!");
        debug_assert!(rate > 0.0, "Invalid learning rate!");

        for layer in 0..weights.len() {
            let weights = &mut weights[layer];
            let biases = &mut biases[layer];

            // weights -= rate * weight_gradients
            blas::saxpy(
                (weights.rows * weights.columns) as i32,
                -rate,
                &weight_gradients[layer].values, 1,
                &mut weights.values, 1,
            );

            // weights .*= weight_masks
            if let Some(ref mask) = weight_masks[layer] {
                for i in 0..weights.values.len() {
                    weights.values[i] *= mask.values[i];
                }
            }

            // biases -= rate * bias_gradients
            blas::saxpy(
                (biases.columns) as i32,
                -rate,
                &bias_gradients[layer].values, 1,
                &mut biases.values, 1,
            );
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdadeltaGradientDescent {
    weights_e: Vec<MatrixCm>,
    weights_rms: Vec<MatrixCm>,
    weights_temp: Vec<MatrixCm>,
    biases_e: Vec<MatrixRm>,
    biases_rms: Vec<MatrixRm>,
    biases_temp: Vec<MatrixRm>,
}

impl AdadeltaGradientDescent {
    pub fn new(inputs: usize, hidden_layers: &[usize], outputs: usize) -> AdadeltaGradientDescent {
        assert!(inputs > 0, "Invalid number of inputs!");
        assert!(hidden_layers.iter().find(|&&l| l == 0).is_none(), "Invalid number of hidden-layer neurons!");
        assert!(outputs > 0, "Invalid number of outputs!");

        let mut weights = Vec::new();
        let mut biases = Vec::new();

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

            weights.push(MatrixCm::zeros(input_size, output_size));
            biases.push(MatrixRm::zeros(1, output_size));
        }

        AdadeltaGradientDescent {
            weights_e: weights.clone(),
            weights_rms: weights.clone(),
            weights_temp: weights,
            biases_e: biases.clone(),
            biases_rms: biases.clone(),
            biases_temp: biases,
        }
    }
}

impl GradientDescent for AdadeltaGradientDescent {
    fn descend(
        &mut self,
        weights: &mut [MatrixCm],
        weight_masks: &[Option<MatrixCm>],
        biases: &mut [MatrixRm],
        weight_gradients: &[MatrixCm],
        bias_gradients: &[MatrixRm],
        rate: f32,
    ) {
        debug_assert!(weights.len() == weight_masks.len(), "weights.len() doesn't match weight_masks.len()!");
        debug_assert!(weights.len() == weight_gradients.len(), "weights.len() doesn't match weight_gradients.len()!");
        debug_assert!(weights.len() == self.weights_e.len(), "weights.len() doesn't match self.weights_e.len()!");
        debug_assert!(biases.len() == bias_gradients.len(), "biases.len() doesn't match bias_gradients.len()!");
        debug_assert!(biases.len() == self.biases_e.len(), "biases.len() doesn't match self.biases_e.len()!");
        debug_assert!((|| {
            for i in 0..weights.len() {
                if let Some(ref mask) = weight_masks[i] {
                    if !weights[i].same_size(mask) {
                        return false;
                    }
                }
            }
            true
        })(), "Dimension mismatch between weights and weight_masks!");
        debug_assert!((|| {
            for i in 0..weights.len() {
                if !weights[i].same_size(&weight_gradients[i]) {
                    return false;
                }
            }
            true
        })(), "Dimension mismatch between weights and weight_gradients!");
        debug_assert!((|| {
            for i in 0..weights.len() {
                if !weights[i].same_size(&self.weights_e[i]) {
                    return false;
                }
            }
            true
        })(), "Dimension mismatch between weights and self.weights_e!");
        debug_assert!((|| {
            for i in 0..biases.len() {
                if !biases[i].same_size(&bias_gradients[i]) {
                    return false;
                }
            }
            true
        })(), "Dimension mismatch between biases and bias_gradients!");
        debug_assert!((|| {
            for i in 0..biases.len() {
                if !biases[i].same_size(&self.biases_e[i]) {
                    return false;
                }
            }
            true
        })(), "Dimension mismatch between biases and self.biases_e!");
        debug_assert!(rate > 0.0, "Invalid learning rate!");

        for layer in 0..weights.len() {
            let weights = &mut weights[layer];
            let weights_e = &mut self.weights_e[layer];
            let weights_rms = &mut self.weights_rms[layer];
            let weights_temp = &mut self.weights_temp[layer];
            let biases = &mut biases[layer];
            let biases_e = &mut self.biases_e[layer];
            let biases_rms = &mut self.biases_rms[layer];
            let biases_temp = &mut self.biases_temp[layer];

            let weight_gradients = &weight_gradients[layer];
            let bias_gradients = &bias_gradients[layer];

            let decay = 0.99;
            // weights_e = weights_e * decay + weight_gradients ^ 2 * (1.0 - decay)
            blas::sscal((weights_e.rows * weights_e.columns) as i32, decay, &mut weights_e.values, 1);
            for i in 0..weight_gradients.values.len() {
                weights_temp.values[i] = weight_gradients.values[i] * weight_gradients.values[i];
            }
            blas::saxpy(
                (weights_e.rows * weights_e.columns) as i32,
                1.0 - decay,
                &weights_temp.values, 1,
                &mut weights_e.values, 1,
            );
            // biases_e = biases_e * decay + bias_gradients ^ 2 * (1.0 - decay)
            blas::sscal((biases_e.rows * biases_e.columns) as i32, decay, &mut biases_e.values, 1);
            for i in 0..bias_gradients.values.len() {
                biases_temp.values[i] = bias_gradients.values[i] * bias_gradients.values[i];
            }
            blas::saxpy(
                (biases_e.rows * biases_e.columns) as i32,
                1.0 - decay,
                &biases_temp.values, 1,
                &mut biases_e.values, 1,
            );

            let epsilon = 1e-8;
            // weights_delta = weight_gradients .* (weights_rms + epsilon).sqrt() / (weights_e + epsilon).sqrt()
            for i in 0..weight_gradients.values.len() {
                weights_temp.values[i] = weight_gradients.values[i] * (weights_rms.values[i] + epsilon).sqrt() / (weights_e.values[i] + epsilon).sqrt();
            }
            // biases_delta = bias_gradients .* (biases_rms + epsilon).sqrt() / (biases_e + epsilon).sqrt()
            for i in 0..bias_gradients.values.len() {
                biases_temp.values[i] = bias_gradients.values[i] * (biases_rms.values[i] + epsilon).sqrt() / (biases_e.values[i] + epsilon).sqrt();
            }

            // weights -= rate * weights_delta
            blas::saxpy(
                (weights.rows * weights.columns) as i32,
                -rate,
                &weights_temp.values, 1,
                &mut weights.values, 1,
            );

            // weights .*= weight_masks
            if let Some(ref mask) = weight_masks[layer] {
                for i in 0..weights.values.len() {
                    weights.values[i] *= mask.values[i];
                }
            }

            // biases -= rate * biases_delta
            blas::saxpy(
                (biases.columns) as i32,
                -rate,
                &biases_temp.values, 1,
                &mut biases.values, 1,
            );

            // weights_rms = weights_rms * decay + weights_delta ^ 2 * (1.0 - decay)
            blas::sscal((weights_rms.rows * weights_rms.columns) as i32, decay, &mut weights_rms.values, 1);
            for i in 0..weights_temp.values.len() {
                weights_temp.values[i] *= weights_temp.values[i];
            }
            blas::saxpy(
                (weights_rms.rows * weights_rms.columns) as i32,
                1.0 - decay,
                &weights_temp.values, 1,
                &mut weights_rms.values, 1,
            );
            // biases_rms = biases_rms * decay + biases_delta ^ 2 * (1.0 - decay)
            blas::sscal((biases_rms.rows * biases_rms.columns) as i32, decay, &mut biases_rms.values, 1);
            for i in 0..biases_temp.values.len() {
                biases_temp.values[i] *= biases_temp.values[i];
            }
            blas::saxpy(
                (biases_rms.rows * biases_rms.columns) as i32,
                1.0 - decay,
                &biases_temp.values, 1,
                &mut biases_rms.values, 1,
            );
        }
    }
}
