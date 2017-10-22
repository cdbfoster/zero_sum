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

use smallmath::vector::{self, Vector};

use activation_function::ActivationFunction;
use serialization::Identifiable;

#[derive(Debug)]
pub struct TanHActivationFunction;

impl ActivationFunction for TanHActivationFunction {
    fn f(x: f32) -> f32 {
        // f = x.tanh() ..kinda
        let e = 1.0 + x.abs() + x * x * (0.5658 + x * x * 0.1430);
        x.signum() * (e - 1.0 / e) / (e + 1.0 / e)
    }

    fn f_prime(x: f32) -> f32 {
        // f' = 1.0 - f * f
        let y = Self::f(x);
        1.0 - (y * y)
    }

    fn f_vector(inputs: &Vector, outputs: &mut Vector) {
        // e = 1.0 + |x| + x * x * (0.5658 + x * x * 0.1430);
        let mut temp = inputs.clone();
        vector::ops::square_assign(&mut temp);
        vector::ops::scale(&temp, 0.1430, outputs);
        vector::ops::offset_assign(outputs, 0.5658);
        vector::ops::multiply_assign(outputs, &temp);
        let mut temp2 = inputs.clone();
        vector::ops::abs_assign(&mut temp2);
        vector::ops::add_assign(outputs, &temp2);
        vector::ops::offset_assign(outputs, 1.0);

        // f = x.signum() * (e - 1.0 / e) / (e + 1.0 / e)
        temp.clone_from(outputs);
        vector::ops::reciprocal_assign(&mut temp);
        temp2.clone_from(outputs);
        vector::ops::add_assign(&mut temp2, &temp);
        vector::ops::subtract_assign(outputs, &temp);
        vector::ops::divide_assign(outputs, &temp2);
        vector::ops::signum(inputs, &mut temp);
        vector::ops::multiply_assign(outputs, &temp);

        assert!(outputs.iter().find(|x| x.is_nan()).is_none(), "NaN here!");
    }

    fn f_prime_vector(inputs: &Vector, outputs: &mut Vector) {
        // f' = 1.0 - f * f
        Self::f_vector(inputs, outputs);
        vector::ops::square_assign(outputs);
        vector::ops::scale_assign(outputs, -1.0);
        vector::ops::offset_assign(outputs, 1.0);
    }
}

impl Identifiable for TanHActivationFunction {
    fn identifier() -> String {
        String::from("TanHActivationFunction")
    }

    fn get_identifier(&self) -> String {
        Self::identifier()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::{self, Bencher};

    #[bench]
    fn bench_tanh(bencher: &mut Bencher) {
        bencher.iter(|| {
            TanHActivationFunction::f(test::black_box(0.5))
        });
    }

    #[bench]
    fn bench_tanh_prime(bencher: &mut Bencher) {
        bencher.iter(|| {
            TanHActivationFunction::f_prime(test::black_box(0.5))
        });
    }

    #[bench]
    fn bench_tanh_vector(bencher: &mut Bencher) {
        let a = Vector::from_vec(vec![0.5; 1000]);
        let mut b = Vector::zeros(1000);

        bencher.iter(|| {
            TanHActivationFunction::f_vector(&a, &mut b);
        });
    }

    #[bench]
    fn bench_tanh_prime_vector(bencher: &mut Bencher) {
        let a = Vector::from_vec(vec![0.5; 1000]);
        let mut b = Vector::zeros(1000);

        bencher.iter(|| {
            TanHActivationFunction::f_prime_vector(&a, &mut b);
        });
    }
}
