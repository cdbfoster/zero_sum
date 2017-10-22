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
use std::str::FromStr;

#[cfg(feature = "with_simd")]
use simd;

use smallmath::Matrix;
use smallmath::vector::{self, Vector};

use gradient_descent::GradientDescent;
use serialization::{File, Identifiable, read_error, read_line, Serializable};

pub struct AdadeltaGradientDescent {
    decay: f32,
    regularization: f32,

    weights_eg2: Matrix,
    biases_eg2: Vector,
    weights_edx2: Matrix,
    biases_edx2: Vector,

    weights_temp: Matrix,
    weights_temp2: Matrix,
    biases_temp: Vector,
    biases_temp2: Vector,
}

impl AdadeltaGradientDescent {
    pub fn new(inputs: usize, outputs: usize, decay: f32, regularization: f32) -> AdadeltaGradientDescent {
        assert!(inputs > 0, "Invalid number of inputs!");
        assert!(outputs > 0, "Invalid number of outputs!");

        AdadeltaGradientDescent {
            decay: decay,
            regularization: regularization,
            weights_eg2: Matrix::zeros(inputs, outputs),
            biases_eg2: Vector::zeros(outputs),
            weights_edx2: Matrix::zeros(inputs, outputs),
            biases_edx2: Vector::zeros(outputs),
            weights_temp: Matrix::zeros(inputs, outputs),
            weights_temp2: Matrix::zeros(inputs, outputs),
            biases_temp: Vector::zeros(outputs),
            biases_temp2: Vector::zeros(outputs),
        }
    }
}

impl GradientDescent for AdadeltaGradientDescent {
    fn descend(
        &mut self,
        weights: &mut Matrix,
        biases: &mut Vector,
        weight_gradients: &Matrix,
        bias_gradients: &Vector,
        rate: f32,
    ) {
        // weights_eg2 = weights_eg2 * decay + weight_gradients ^ 2 * (1.0 - decay)
        vector::ops::square(weight_gradients, &mut self.weights_temp);
        vector::ops::scale_assign(&mut self.weights_temp, 1.0 - self.decay);
        vector::ops::scale_assign(&mut self.weights_eg2, self.decay);
        vector::ops::add_assign(&mut self.weights_eg2, &self.weights_temp);

        // biases_eg2 = biases_eg2 * decay + bias_gradients ^ 2 * (1.0 - decay)
        vector::ops::square(bias_gradients, &mut self.biases_temp);
        vector::ops::scale_assign(&mut self.biases_temp, 1.0 - self.decay);
        vector::ops::scale_assign(&mut self.biases_eg2, self.decay);
        vector::ops::add_assign(&mut self.biases_eg2, &self.biases_temp);

        let epsilon = 1e-8;

        // weights_reg = max(min(weights, -regularization), regularization)
        // weights_dx = weight_gradients .* (weights_edx2 + epsilon).sqrt() / (weights_eg2 + epsilon).sqrt() + weights_reg
        vector::ops::offset(&self.weights_edx2, epsilon, &mut self.weights_temp);
        vector::ops::sqrt_assign(&mut self.weights_temp);
        vector::ops::multiply_assign(&mut self.weights_temp, weight_gradients);
        vector::ops::offset(&self.weights_eg2, epsilon, &mut self.weights_temp2);
        vector::ops::sqrt_assign(&mut self.weights_temp2);
        vector::ops::divide_assign(&mut self.weights_temp, &self.weights_temp2);
        max(weights, -self.regularization, &mut self.weights_temp2);
        min_assign(&mut self.weights_temp2, self.regularization);
        vector::ops::add_assign(&mut self.weights_temp, &self.weights_temp2);

        // biases_dx = bias_gradients .* (biases_edx2 + epsilon).sqrt() / (biases_eg2 + epsilon).sqrt()
        vector::ops::offset(&self.biases_edx2, epsilon, &mut self.biases_temp);
        vector::ops::sqrt_assign(&mut self.biases_temp);
        vector::ops::multiply_assign(&mut self.biases_temp, bias_gradients);
        vector::ops::offset(&self.biases_eg2, epsilon, &mut self.biases_temp2);
        vector::ops::sqrt_assign(&mut self.biases_temp2);
        vector::ops::divide_assign(&mut self.biases_temp, &self.biases_temp2);

        // weights -= weights_dx * rate
        vector::ops::scale(&self.weights_temp, rate, &mut self.weights_temp2);
        vector::ops::subtract_assign(weights, &self.weights_temp2);

        // biases -= biases_dx * rate
        vector::ops::scale(&self.biases_temp, rate, &mut self.biases_temp2);
        vector::ops::subtract_assign(biases, &self.biases_temp2);

        // weights_edx2 = weights_edx2 * decay + weights_dx ^ 2 * (1.0 - decay)
        vector::ops::square_assign(&mut self.weights_temp);
        vector::ops::scale_assign(&mut self.weights_temp, 1.0 - self.decay);
        vector::ops::scale_assign(&mut self.weights_edx2, self.decay);
        vector::ops::add_assign(&mut self.weights_edx2, &self.weights_temp);

        // biases_edx2 = biases_edx2 * decay + biases_dx ^ 2 * (1.0 - decay)
        vector::ops::square_assign(&mut self.biases_temp);
        vector::ops::scale_assign(&mut self.biases_temp, 1.0 - self.decay);
        vector::ops::scale_assign(&mut self.biases_edx2, self.decay);
        vector::ops::add_assign(&mut self.biases_edx2, &self.biases_temp);
    }
}

impl Identifiable for AdadeltaGradientDescent {
    fn identifier() -> String {
        String::from("AdadeltaGradientDescent")
    }

    fn get_identifier(&self) -> String {
        Self::identifier()
    }
}

impl Serializable for AdadeltaGradientDescent {
    fn read_from_file(file: &mut BufReader<File>) -> Result<AdadeltaGradientDescent> {
        let (decay, regularization) = {
            let strings = read_line(file)?;

            if strings.len() < 2 {
                return read_error(file, "Cannot read decay/regularization!");
            }

            (
                if let Ok(decay) = f32::from_str(&strings[0]) {
                    decay
                } else {
                    return read_error(file, "Cannot parse decay!");
                },
                if let Ok(regularization) = f32::from_str(&strings[1]) {
                    regularization
                } else {
                    return read_error(file, "Cannot parse regularization!");
                },
            )
        };

        let strings = read_line(file)?;

        if strings.len() < 1 || strings[0] != "WeightsE[g^2]" {
            println!("{}", strings[0]);
            return read_error(file, "Cannot read weights E[g^2]!");
        }

        let weights_eg2 = Matrix::read_from_file(file)?;
        let inputs = weights_eg2.rows();
        let outputs = weights_eg2.columns();

        let strings = read_line(file)?;

        if strings.len() < 1 || strings[0] != "BiasesE[g^2]" {
            return read_error(file, "Cannot read biases E[g^2]!");
        }

        let biases_eg2 = Vector::read_from_file(file)?;

        if biases_eg2.len() != outputs {
            return read_error(file, "Biases E[g^2] doesn't match gradient descent outputs!");
        }

        let strings = read_line(file)?;

        if strings.len() < 1 || strings[0] != "WeightsE[dx^2]" {
            return read_error(file, "Cannot read weights E[dx^2]!");
        }

        let weights_edx2 = Matrix::read_from_file(file)?;

        if weights_edx2.rows() != inputs || weights_edx2.columns() != outputs {
            return read_error(file, "Weights E[dx^2] doesn't match weights E[g^2] dimensions!");
        }

        let strings = read_line(file)?;

        if strings.len() < 1 || strings[0] != "BiasesE[dx^2]" {
            return read_error(file, "Cannot read biases E[dx^2]!");
        }

        let biases_edx2 = Vector::read_from_file(file)?;

        if biases_edx2.len() != outputs {
            return read_error(file, "Biases E[dx^2] doesn't match gradient descent outputs!");
        }

        Ok(AdadeltaGradientDescent {
            decay: decay,
            regularization: regularization,
            weights_eg2: weights_eg2,
            biases_eg2: biases_eg2,
            weights_edx2: weights_edx2,
            biases_edx2: biases_edx2,
            weights_temp: Matrix::zeros(inputs, outputs),
            weights_temp2: Matrix::zeros(inputs, outputs),
            biases_temp: Vector::zeros(outputs),
            biases_temp2: Vector::zeros(outputs),
        })
    }

    fn write_to_file(&self, file: &mut File) -> Result<()> {
        let indentation = file.indentation();
        write!(file, "{}{}  {}\n", indentation, self.decay, self.regularization)?;
        write!(file, "{}WeightsE[g^2]\n", indentation)?;
        file.indent();
        self.weights_eg2.write_to_file(file)?;
        write!(file, "{}BiasesE[g^2]\n", indentation)?;
        self.biases_eg2.write_to_file(file)?;
        write!(file, "{}WeightsE[dx^2]\n", indentation)?;
        self.weights_edx2.write_to_file(file)?;
        write!(file, "{}BiasesE[dx^2]\n", indentation)?;
        self.biases_edx2.write_to_file(file)?;
        file.unindent();
        Ok(())
    }
}

fn max(a: &Vector, b: f32, c: &mut Vector) {
    debug_assert!(a.len() == c.len(), "Operand vectors are different lengths!");

    use std::slice::from_raw_parts as s;
    use std::slice::from_raw_parts_mut as sm;

    if a.len() >= 16 {
        max_16s(a, b, c);
    }

    if a.len() % 16 >= 8 {
        unsafe { max_8(
            s(a.as_ptr().offset((a.len() as isize / 8 - 1) * 8), 8), b,
            sm(c.as_mut_ptr().offset((a.len() as isize / 8 - 1) * 8), 8),
        ) };
    }

    if a.len() % 8 >= 4 {
        unsafe { max_4(
            s(a.as_ptr().offset((a.len() as isize / 4 - 1) * 4), 4), b,
            sm(c.as_mut_ptr().offset((a.len() as isize / 4 - 1) * 4), 4),
        ) };
    }

    if a.len() % 4 >= 2 {
        unsafe { max_2(
            s(a.as_ptr().offset((a.len() as isize / 2 - 1) * 2), 2), b,
            sm(c.as_mut_ptr().offset((a.len() as isize / 2 - 1) * 2), 2),
        ) };
    }

    if a.len() % 2 == 1 {
        c[a.len() - 1] = a[a.len() - 1].max(b);
    }
}

#[cfg(feature = "with_simd")]
macro_rules! vmaxps {
    ($a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vmaxps $2, $1, $0" : "=x"($c) : "x"($a), "x"($b) ::) };
    )
}

#[cfg(feature = "with_simd")]
fn max_16s(a: &[f32], b: f32, c: &mut[f32]) {
    for i in 0..(a.len() / 16) {
        let a_ra = simd::x86::avx::f32x8::load(a, i * 16 + 0);
        let a_rb = simd::x86::avx::f32x8::load(a, i * 16 + 8);
        let b_r = simd::x86::avx::f32x8::splat(b);

        let c_ra: simd::x86::avx::f32x8; vmaxps!(a_ra, b_r, c_ra);
        let c_rb: simd::x86::avx::f32x8; vmaxps!(a_rb, b_r, c_rb);

        c_ra.store(c, i * 16 + 0);
        c_rb.store(c, i * 16 + 8);
    }
}

#[cfg(feature = "with_simd")]
fn max_8(a: &[f32], b: f32, c: &mut[f32]) {
    let a_r = simd::x86::avx::f32x8::load(a, 0);
    let b_r = simd::x86::avx::f32x8::splat(b);

    let c_r: simd::x86::avx::f32x8; vmaxps!(a_r, b_r, c_r);

    c_r.store(c, 0);
}

#[cfg(feature = "with_simd")]
fn max_4(a: &[f32], b: f32, c: &mut[f32]) {
    let a_r = simd::f32x4::load(a, 0);
    let b_r = simd::f32x4::splat(b);

    let c_r: simd::f32x4; vmaxps!(a_r, b_r, c_r);

    c_r.store(c, 0);
}

#[cfg(feature = "with_simd")]
fn max_2(a: &[f32], b: f32, c: &mut[f32]) {
    let a_r = simd::f32x4::new(a[0], a[1], 0.0, 0.0);
    let b_r = simd::f32x4::splat(b);

    let c_r: simd::f32x4; vmaxps!(a_r, b_r, c_r);

    c[0] = c_r.extract(0); c[1] = c_r.extract(1);
}

#[cfg(not(feature = "with_simd"))]
fn max_16s(a: &[f32], b: f32, c: &mut[f32]) {
    for i in 0..(a.len() / 16) {
        c[i * 16 +  0] = a[i * 16 +  0].max(b);
        c[i * 16 +  1] = a[i * 16 +  1].max(b);
        c[i * 16 +  2] = a[i * 16 +  2].max(b);
        c[i * 16 +  3] = a[i * 16 +  3].max(b);
        c[i * 16 +  4] = a[i * 16 +  4].max(b);
        c[i * 16 +  5] = a[i * 16 +  5].max(b);
        c[i * 16 +  6] = a[i * 16 +  6].max(b);
        c[i * 16 +  7] = a[i * 16 +  7].max(b);
        c[i * 16 +  8] = a[i * 16 +  8].max(b);
        c[i * 16 +  9] = a[i * 16 +  9].max(b);
        c[i * 16 + 10] = a[i * 16 + 10].max(b);
        c[i * 16 + 11] = a[i * 16 + 11].max(b);
        c[i * 16 + 12] = a[i * 16 + 12].max(b);
        c[i * 16 + 13] = a[i * 16 + 13].max(b);
        c[i * 16 + 14] = a[i * 16 + 14].max(b);
        c[i * 16 + 15] = a[i * 16 + 15].max(b);
    }
}

#[cfg(not(feature = "with_simd"))]
fn max_8(a: &[f32], b: f32, c: &mut[f32]) {
    c[0] = a[0].max(b);
    c[1] = a[1].max(b);
    c[2] = a[2].max(b);
    c[3] = a[3].max(b);
    c[4] = a[4].max(b);
    c[5] = a[5].max(b);
    c[6] = a[6].max(b);
    c[7] = a[7].max(b);
}

#[cfg(not(feature = "with_simd"))]
fn max_4(a: &[f32], b: f32, c: &mut[f32]) {
    c[0] = a[0].max(b);
    c[1] = a[1].max(b);
    c[2] = a[2].max(b);
    c[3] = a[3].max(b);
}

#[cfg(not(feature = "with_simd"))]
fn max_2(a: &[f32], b: f32, c: &mut[f32]) {
    c[0] = a[0].max(b);
    c[1] = a[1].max(b);
}

fn min_assign(a: &mut Vector, b: f32) {
    use std::slice::from_raw_parts_mut as sm;

    if a.len() >= 16 {
        min_assign_16s(a, b);
    }

    if a.len() % 16 >= 8 {
        unsafe { min_assign_8(
            sm(a.as_mut_ptr().offset((a.len() as isize / 8 - 1) * 8), 8), b,
        ) };
    }

    if a.len() % 8 >= 4 {
        unsafe { min_assign_4(
            sm(a.as_mut_ptr().offset((a.len() as isize / 4 - 1) * 4), 4), b,
        ) };
    }

    if a.len() % 4 >= 2 {
        unsafe { min_assign_2(
            sm(a.as_mut_ptr().offset((a.len() as isize / 2 - 1) * 2), 2), b,
        ) };
    }

    if a.len() % 2 == 1 {
        let last_index = a.len() - 1;
        a[last_index] = a[last_index].min(b);
    }
}

#[cfg(feature = "with_simd")]
macro_rules! vminps {
    ($a:ident, $b:ident, $c:ident) => (
        unsafe { asm!("vminps $2, $1, $0" : "=x"($c) : "x"($a), "x"($b) ::) };
    )
}

#[cfg(feature = "with_simd")]
fn min_assign_16s(a: &mut [f32], b: f32) {
    for i in 0..(a.len() / 16) {
        let mut a_ra = simd::x86::avx::f32x8::load(a, i * 16 + 0);
        let mut a_rb = simd::x86::avx::f32x8::load(a, i * 16 + 8);
        let b_r = simd::x86::avx::f32x8::splat(b);

        vminps!(a_ra, b_r, a_ra);
        vminps!(a_rb, b_r, a_rb);

        a_ra.store(a, i * 16 + 0);
        a_rb.store(a, i * 16 + 8);
    }
}

#[cfg(feature = "with_simd")]
fn min_assign_8(a: &mut [f32], b: f32) {
    let mut a_r = simd::x86::avx::f32x8::load(a, 0);
    let b_r = simd::x86::avx::f32x8::splat(b);

    vminps!(a_r, b_r, a_r);

    a_r.store(a, 0);
}

#[cfg(feature = "with_simd")]
fn min_assign_4(a: &mut [f32], b: f32) {
    let mut a_r = simd::f32x4::load(a, 0);
    let b_r = simd::f32x4::splat(b);

    vminps!(a_r, b_r, a_r);

    a_r.store(a, 0);
}

#[cfg(feature = "with_simd")]
fn min_assign_2(a: &mut [f32], b: f32) {
    let mut a_r = simd::f32x4::new(a[0], a[1], 0.0, 0.0);
    let b_r = simd::f32x4::splat(b);

    vminps!(a_r, b_r, a_r);

    a[0] = a_r.extract(0); a[1] = a_r.extract(1);
}

#[cfg(not(feature = "with_simd"))]
fn min_assign_16s(a: &mut [f32], b: f32) {
    for i in 0..(a.len() / 16) {
        a[i * 16 +  0] = a[i * 16 +  0].min(b);
        a[i * 16 +  1] = a[i * 16 +  1].min(b);
        a[i * 16 +  2] = a[i * 16 +  2].min(b);
        a[i * 16 +  3] = a[i * 16 +  3].min(b);
        a[i * 16 +  4] = a[i * 16 +  4].min(b);
        a[i * 16 +  5] = a[i * 16 +  5].min(b);
        a[i * 16 +  6] = a[i * 16 +  6].min(b);
        a[i * 16 +  7] = a[i * 16 +  7].min(b);
        a[i * 16 +  8] = a[i * 16 +  8].min(b);
        a[i * 16 +  9] = a[i * 16 +  9].min(b);
        a[i * 16 + 10] = a[i * 16 + 10].min(b);
        a[i * 16 + 11] = a[i * 16 + 11].min(b);
        a[i * 16 + 12] = a[i * 16 + 12].min(b);
        a[i * 16 + 13] = a[i * 16 + 13].min(b);
        a[i * 16 + 14] = a[i * 16 + 14].min(b);
        a[i * 16 + 15] = a[i * 16 + 15].min(b);
    }
}

#[cfg(not(feature = "with_simd"))]
fn min_assign_8(a: &mut [f32], b: f32) {
    a[0] = a[0].min(b);
    a[1] = a[1].min(b);
    a[2] = a[2].min(b);
    a[3] = a[3].min(b);
    a[4] = a[4].min(b);
    a[5] = a[5].min(b);
    a[6] = a[6].min(b);
    a[7] = a[7].min(b);
}

#[cfg(not(feature = "with_simd"))]
fn min_assign_4(a: &mut [f32], b: f32) {
    a[0] = a[0].min(b);
    a[1] = a[1].min(b);
    a[2] = a[2].min(b);
    a[3] = a[3].min(b);
}

#[cfg(not(feature = "with_simd"))]
fn min_assign_2(a: &mut [f32], b: f32) {
    a[0] = a[0].min(b);
    a[1] = a[1].min(b);
}
