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

use std::any::Any;

pub trait ActivationFunction {
    fn new() -> Self;

    /// Calculates the activation function at `x`.
    fn f(x: f32) -> f32;

    /// Calculates the derivative of the activation function at `x`.
    fn f_prime(x: f32) -> f32;

    fn as_any(&self) -> &Any;
}

#[derive(Debug)]
pub struct ReLuActivationFunction;

impl ActivationFunction for ReLuActivationFunction {
    fn new() -> ReLuActivationFunction {
        ReLuActivationFunction
    }

    fn f(x: f32) -> f32 {
        x.max(0.0)
    }

    fn f_prime(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    fn as_any(&self) -> &Any {
        self
    }
}

#[derive(Debug)]
pub struct TanHActivationFunction;

impl ActivationFunction for TanHActivationFunction {
    fn new() -> TanHActivationFunction {
        TanHActivationFunction
    }

    fn f(x: f32) -> f32 {
        x.tanh()
    }

    fn f_prime(mut x: f32) -> f32 {
        x = x.tanh();
        1.0 - x * x
    }

    fn as_any(&self) -> &Any {
        self
    }
}
