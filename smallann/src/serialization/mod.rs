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

use std::fs::{File as StdFile, OpenOptions};
use std::io::{self, BufRead, BufReader, Error, ErrorKind, Read, Result, Write};
use std::str::FromStr;

use smallmath::{Matrix, Vector};

use activation_function::*;
use gradient_descent::*;
use layer::*;

pub use self::identifiable::Identifiable;
pub use self::serializable::Serializable;

#[macro_use]
mod identifiable;
#[macro_use]
mod serializable;

identifiable!(
    ReLuActivationFunction,
    TanHActivationFunction,
);

pub fn read_error<T>(file: &BufReader<File>, message: &str) -> Result<T> {
    Err(Error::new(ErrorKind::Other, format!("Line {}: {}", file.get_ref().line, message)))
}

pub fn read_line(file: &mut BufReader<File>) -> Result<Vec<String>> {
    let mut line = String::new();
    file.read_line(&mut line)?;
    file.get_mut().line += 1;
    Ok(
        line.trim().split(" ").filter_map(|s| if !s.is_empty() {
            Some(s.to_string())
        } else {
            None
        }).collect::<Vec<String>>()
    )
}

pub fn write_matrix(file: &mut File, matrix: &Matrix, precision: usize) -> Result<()> {
    let (rows, columns) = (matrix.rows(), matrix.columns());
    let indentation = file.indentation();
    write!(file, "{}{}  {}\n", indentation, rows, columns)?;
    for row in 0..rows {
        write!(file, "{}", indentation)?;
        for value in &matrix[row] {
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

impl Identifiable for Matrix {
    fn identifier() -> String {
        String::from("Matrix")
    }

    fn get_identifier(&self) -> String {
        Self::identifier()
    }
}

impl Serializable for Matrix {
    fn read_from_file(file: &mut BufReader<File>) -> Result<Matrix> {
        let (rows, columns) = {
            let strings = read_line(file)?;

            if strings.len() < 2 {
                return read_error(file, "Cannot read matrix dimensions!");
            }

            (
                if let Ok(rows) = usize::from_str(&strings[0]) {
                    rows
                } else {
                    return read_error(file, "Cannot parse matrix rows!");
                },
                if let Ok(columns) = usize::from_str(&strings[1]) {
                    columns
                } else {
                    return read_error(file, "Cannot parse matrix columns!");
                },
            )
        };

        let mut values = Vec::new();
        for _ in 0..rows {
            let strings = read_line(file)?;

            if strings.len() < columns {
                return read_error(file, "Cannot read matrix row!");
            }

            for column in 0..columns {
                if let Ok(value) = f32::from_str(&strings[column]) {
                    values.push(value);
                } else {
                    return read_error(file, "Cannot parse matrix value!");
                }
            }
        }

        Ok(Matrix::from_vec(rows, columns, values))
    }

    fn write_to_file(&self, file: &mut File) -> Result<()> {
        write_matrix(file, self, 8)
    }
}

impl Identifiable for Vector {
    fn identifier() -> String {
        String::from("Vector")
    }

    fn get_identifier(&self) -> String {
        Self::identifier()
    }
}

impl Serializable for Vector {
    fn read_from_file(file: &mut BufReader<File>) -> Result<Vector> {
        let size = {
            let strings = read_line(file)?;

            if strings.len() < 1 {
                return read_error(file, "Cannot read vector size!");
            }

            if let Ok(rows) = usize::from_str(&strings[0]) {
                rows
            } else {
                return read_error(file, "Cannot parse vector size!");
            }
        };

        let mut values = Vec::new();
        let strings = read_line(file)?;

        if strings.len() < size {
            return read_error(file, "Cannot read vector values!");
        }

        for i in 0..size {
            if let Ok(value) = f32::from_str(&strings[i]) {
                values.push(value)
            } else {
                return read_error(file, "Cannot parse vector value!");
            }
        }

        Ok(Vector::from_vec(values))
    }

    fn write_to_file(&self, file: &mut File) -> Result<()> {
        let size = self.len();
        let indentation = file.indentation();
        write!(file, "{0}{1}\n{0}", indentation, size)?;
        for value in self.iter() {
            write!(file, "{:12.8} ", value)?;
        }
        write!(file, "\n")
    }
}

pub fn read_layer(file: &mut BufReader<File>) -> Result<Box<Layer>> {
    let strings = read_line(file)?;

    if strings.len() < 1 {
        return read_error(file, "Cannot read layer type!");
    }

    match &strings[0] as &str {
        "ActivationLayer<ReLuActivationFunction>" => ActivationLayer::<ReLuActivationFunction>::read_from_file(file).map(|l| Box::new(l) as Box<Layer>),
        "ActivationLayer<TanHActivationFunction>" => ActivationLayer::<TanHActivationFunction>::read_from_file(file).map(|l| Box::new(l) as Box<Layer>),
        "CompositeLayer" => CompositeLayer::read_from_file(file).map(|l| Box::new(l) as Box<Layer>),
        "ConvolutionalLayer<AdadeltaGradientDescent>" => ConvolutionalLayer::<AdadeltaGradientDescent>::read_from_file(file).map(|l| Box::new(l) as Box<Layer>),
        "ConvolutionalLayer<MomentumGradientDescent>" => ConvolutionalLayer::<MomentumGradientDescent>::read_from_file(file).map(|l| Box::new(l) as Box<Layer>),
        "ConvolutionalLayer<SimpleGradientDescent>" => ConvolutionalLayer::<SimpleGradientDescent>::read_from_file(file).map(|l| Box::new(l) as Box<Layer>),
        "FullyConnectedLayer<AdadeltaGradientDescent>" => FullyConnectedLayer::<AdadeltaGradientDescent>::read_from_file(file).map(|l| Box::new(l) as Box<Layer>),
        "FullyConnectedLayer<MomentumGradientDescent>" => FullyConnectedLayer::<MomentumGradientDescent>::read_from_file(file).map(|l| Box::new(l) as Box<Layer>),
        "FullyConnectedLayer<SimpleGradientDescent>" => FullyConnectedLayer::<SimpleGradientDescent>::read_from_file(file).map(|l| Box::new(l) as Box<Layer>),
        "PassThroughLayer" => PassThroughLayer::read_from_file(file).map(|l| Box::new(l) as Box<Layer>),
        _ => read_error(file, "Unknown layer type!"),
    }
}

pub use self::file::File;

mod file;
