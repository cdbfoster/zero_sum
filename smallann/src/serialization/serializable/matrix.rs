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

use smallmath::Matrix;

use serialization::{File, read_error, read_line, Serializable};

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
