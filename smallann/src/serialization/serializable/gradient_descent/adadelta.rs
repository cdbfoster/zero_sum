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

use smallmath::{Matrix, Vector};

use gradient_descent::AdadeltaGradientDescent;
use serialization::{File, read_error, read_line, Serializable};

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

        Ok(AdadeltaGradientDescent::construct(
            inputs, outputs,
            decay,
            regularization,
            weights_eg2,
            biases_eg2,
            weights_edx2,
            biases_edx2,
        ))
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
