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

extern crate rand;
extern crate smallann;
extern crate smallmath;

use std::f32::consts::FRAC_PI_2;

use smallann::Ann;
use smallann::activation_function::{ReLuActivationFunction, TanHActivationFunction};
use smallann::gradient_descent::SimpleGradientDescent;
use smallann::layer::{ActivationLayer, FullyConnectedLayer, Layer};
use smallann::loss_function::{LossFunction, MeanSquaredErrorLossFunction};

use smallmath::Matrix;

fn main() {
    // Creates a matrix that represents 100 inputs of 1 feature each, with values spanning [-pi / 2, pi / 2)
    let inputs = Matrix::from_vec(
        100, 1,                                                                             // rows, columns,
        (0..100).map(|x| FRAC_PI_2 * (2.0 * x as f32 / 100.0 - 1.0)).collect::<Vec<f32>>(), // values,
    );
    // Our targets are the sine of each input
    let targets = Matrix::from_vec(
        100, 1,
        inputs.iter().map(|x| x.sin()).collect::<Vec<f32>>(),
    );

    // Used to initialize each layer with random weights
    let mut rng = rand::thread_rng();

    // Our network consists of a 1-neuron input layer, followed by a 100-neuron hidden layer (ReLU activation), concluded by a 1-neuron output layer (TanH activation)
    let layers: Vec<Box<Layer>> = vec![
        Box::new(FullyConnectedLayer::new(
            1, 100,                             // inputs, outputs,
            SimpleGradientDescent::new(1, 100), // gradient descent algorithm,
            &mut rng,
        )),
        Box::new(ActivationLayer::<ReLuActivationFunction>::new(
            100,                                // inputs/outputs,
        )),
        Box::new(FullyConnectedLayer::new(
            100, 1,
            SimpleGradientDescent::new(100, 1),
            &mut rng,
        )),
        Box::new(ActivationLayer::<TanHActivationFunction>::new(
            1,
        )),
    ];

    // Build our network from our layer descriptions
    let mut ann = Ann::new(layers);

    println!("Before training:");
    test(&ann);

    // Train using mean squared error loss
    for _epoch in 0..10000 {
        ann.train::<MeanSquaredErrorLossFunction>(&inputs, &targets, 0.01);
    }

    println!("After training:");
    test(&ann);

    println!("Writing network to file: sin_net");
    ann.to_file("sin_net").unwrap();
}

fn test(ann: &Ann) {
    // The training inputs are spaced every 0.01 * pi.  We shift these over by 0.005 * pi to test how well the network generalizes the gaps.
    let inputs = Matrix::from_vec(
        100, 1,
        (0..100).map(|x| FRAC_PI_2 * (2.0 * x as f32 / 100.0 - 1.0 + 0.01)).collect::<Vec<f32>>(),
    );
    let targets = Matrix::from_vec(
        100, 1,
        inputs.iter().map(|x| x.sin()).collect::<Vec<f32>>(),
    );

    // Get the outputs for our test inputs
    let mut outputs = Matrix::zeros(100, 1);
    ann.classify(&inputs, &mut outputs);

    // Calculate the error per input and then sum it
    let mut error = Matrix::zeros(100, 1);
    MeanSquaredErrorLossFunction::l_vector(&outputs, &targets, &mut error);
    println!("  Error: {}", error.iter().sum::<f32>());

    // Print a sample of the outputs to compare to the targets
    print!("  Output: ");
    for i in 0..20 {
        print!("{:7.4} ", outputs[(i * 5, 0)]);
    }
    println!();

    print!("  Target: ");
    for i in 0..20 {
        print!("{:7.4} ", targets[(i * 5, 0)]);
    }
    println!();
}
