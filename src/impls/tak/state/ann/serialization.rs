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

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::str::FromStr;

use impls::tak::state::ann::{ActivationFunction, Ann, GradientDescent, MatrixCm, MatrixRm};

pub trait Matrix {
    fn from_vec(rows: usize, columns: usize, values: Vec<f32>) -> Self;
    fn get_dimensions(&self) -> (usize, usize);
    fn get_row(&self, row: usize) -> Vec<f32>;
}

impl Matrix for MatrixCm {
    fn from_vec(rows: usize, columns: usize, values: Vec<f32>) -> MatrixCm { MatrixCm::from_row_major_vec(rows, columns, values) }
    fn get_dimensions(&self) -> (usize, usize) { (self.rows, self.columns) }
    fn get_row(&self, row: usize) -> Vec<f32> { let mut values = Vec::new(); for i in 0..self.columns { values.push(self.values[i * self.rows + row]); } values }
}

impl Matrix for MatrixRm {
    fn from_vec(rows: usize, columns: usize, values: Vec<f32>) -> MatrixRm { MatrixRm::from_vec(rows, columns, values) }
    fn get_dimensions(&self) -> (usize, usize) { (self.rows, self.columns) }
    fn get_row(&self, row: usize) -> Vec<f32> { self[row].to_vec() }
}

fn read_line(file: &mut BufReader<File>) -> io::Result<Vec<String>> {
    let mut line = String::new();
    file.read_line(&mut line)?;
    Ok(
        line.trim().split(" ").filter_map(|s| if !s.is_empty() {
            Some(s.to_string())
        } else {
            None
        }).collect::<Vec<String>>()
    )
}

fn read_matrix<M: Matrix>(file: &mut BufReader<File>) -> Result<M, String> {
    let (rows, columns) = {
        let strings = match read_line(file) {
            Ok(strings) => strings,
            _ => return Err(String::from("Cannot read matrix dimensions!")),
        };

        (
            match usize::from_str(&strings[0]) {
                Ok(rows) => rows,
                _ => return Err(String::from("Cannot parse matrix rows!")),
            },
            match usize::from_str(&strings[1]) {
                Ok(columns) => columns,
                _ => return Err(String::from("Cannot parse matrix columns!")),
            },
        )
    };

    let mut values = Vec::new();
    for _ in 0..rows {
        let strings = match read_line(file) {
            Ok(strings) => strings,
            _ => return Err(String::from("Cannot read matrix row!")),
        };

        for column in 0..columns {
            match f32::from_str(&strings[column]) {
                Ok(value) => values.push(value),
                _ => return Err(String::from("Cannot parse matrix value!")),
            }
        }
    }

    Ok(M::from_vec(rows, columns, values))
}

fn write_matrix<M: Matrix>(file: &mut File, matrix: &M, precision: usize) -> io::Result<()> {
    let (rows, columns) = matrix.get_dimensions();
    write!(file, "{}  {}\n", rows, columns)?;
    for row in 0..rows {
        for value in &matrix.get_row(row) {
            write!(file, "{:total$.precision$} ", value, total = if precision > 0 {
                precision + 4
            } else {
                1
            }, precision = precision)?;
        }
        write!(file, "\n")?;
    }
    Ok(())
}

pub fn read_network<A, F, G>(file: &mut BufReader<File>, network: &mut Ann<A, F, G>) -> Result<(), String> where
    A: ActivationFunction,
    F: ActivationFunction,
    G: GradientDescent {
    let strings = match read_line(file) {
        Ok(strings) => strings,
        _ => return Err(String::from("Cannot read network dimensions!")),
    };
    match usize::from_str(&strings[0]) {
        Ok(value) => if network.weights[0].rows != value {
            return Err(String::from("Incorrect number of inputs!"));
        },
        _ => return Err(String::from("Cannot parse number of inputs!")),
    }
    for (layer, string) in strings[1..].iter().enumerate() {
        match usize::from_str(string) {
            Ok(value) => if network.weights[layer].columns != value {
                return Err(String::from("Incorrect layer size!"));
            },
            _ => return Err(String::from("Cannot parse layer size!")),
        }
    }

    for layer in 0..network.weights.len() {
        let weights = read_matrix::<MatrixCm>(file)?;
        if weights.same_size(&network.weights[layer]) {
            network.weights[layer] = weights;
        } else {
            return Err(String::from("Incorrect matrix dimensions!"));
        }

        let mut weight_mask_present = String::new();
        match file.read_line(&mut weight_mask_present) {
            Err(_) => return Err(String::from("Unable to read weight_mask_present!")),
            _ => (),
        }
        if weight_mask_present.trim() == "+" {
            let weight_mask = read_matrix::<MatrixCm>(file)?;
            if weight_mask.same_size(&network.weights[layer]) {
                network.weight_masks[layer] = Some(weight_mask);
            } else {
                return Err(String::from("Incorrect matrix dimensions!"));
            }
        } else {
            network.weight_masks[layer] = None;
        }

        let biases = read_matrix::<MatrixRm>(file)?;
        if biases.same_size(&network.biases[layer]) {
            network.biases[layer] = biases;
        } else {
            return Err(String::from("Incorrect matrix dimensions!"));
        }
    }

    Ok(())
}

pub fn write_network<A, F, G>(file: &mut File, network: &Ann<A, F, G>) -> io::Result<()> where
    A: ActivationFunction,
    F: ActivationFunction,
    G: GradientDescent {
    for weights in &network.weights {
        write!(file, "{}  ", weights.rows)?;
    }
    write!(file, "{}\n", network.weights.last().unwrap().columns)?;

    for layer in 0..network.weights.len() {
        write_matrix(file, &network.weights[layer], 9)?;

        if let Some(ref mask) = network.weight_masks[layer] {
            write!(file, "+\n")?;
            write_matrix(file, mask, 0)?;
        } else {
            write!(file, "-\n")?;
        }

        write_matrix(file, &network.biases[layer], 9)?;
    }
    Ok(())
}

