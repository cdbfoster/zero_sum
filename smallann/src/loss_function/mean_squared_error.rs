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

use loss_function::LossFunction;

pub struct MeanSquaredErrorLossFunction;

impl LossFunction for MeanSquaredErrorLossFunction {
    fn l(x: f32, y: f32) -> f32 {
        let difference = x - y;
        difference * difference / 2.0
    }

    fn l_prime(x: f32, y: f32) -> f32 {
        x - y
    }

    fn l_vector(outputs: &Vector, targets: &Vector, loss: &mut Vector) {
        vector::ops::subtract(outputs, targets, loss);
        vector::ops::square_assign(loss);
        vector::ops::scale_assign(loss, 0.5);
    }

    fn l_prime_vector(outputs: &Vector, targets: &Vector, loss_prime: &mut Vector) {
        vector::ops::subtract(outputs, targets, loss_prime);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::{self, Bencher};

    #[bench]
    fn bench_mean_squared_error(bencher: &mut Bencher) {
        bencher.iter(|| {
            MeanSquaredErrorLossFunction::l(test::black_box(0.5), 0.6)
        });
    }

    #[bench]
    fn bench_mean_squared_error_prime(bencher: &mut Bencher) {
        bencher.iter(|| {
            MeanSquaredErrorLossFunction::l_prime(test::black_box(0.5), 0.6)
        });
    }

    #[bench]
    fn bench_mean_squared_error_vector(bencher: &mut Bencher) {
        let a = Vector::from_vec(vec![0.5; 1000]);
        let b = Vector::from_vec(vec![0.6; 1000]);
        let mut c = Vector::zeros(1000);

        bencher.iter(|| {
            MeanSquaredErrorLossFunction::l_vector(&a, &b, &mut c);
        });
    }

    #[bench]
    fn bench_mean_squared_error_prime_vector(bencher: &mut Bencher) {
        let a = Vector::from_vec(vec![0.5; 1000]);
        let b = Vector::from_vec(vec![0.6; 1000]);
        let mut c = Vector::zeros(1000);

        bencher.iter(|| {
            MeanSquaredErrorLossFunction::l_prime_vector(&a, &b, &mut c);
        });
    }
}
