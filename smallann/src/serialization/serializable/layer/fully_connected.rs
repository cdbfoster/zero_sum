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

use smallmath::{Matrix, Vector};

use layer::FullyConnectedLayer;
use serialization::{File, read_error, read_line, Serializable};

impl<G> Serializable for FullyConnectedLayer<G> where G: Serializable {
    fn read_from_file(file: &mut BufReader<File>) -> Result<FullyConnectedLayer<G>> {
        let strings = read_line(file)?;

        if strings.len() < 1 || strings[0] != "Weights" {
            return read_error(file, "Cannot read layer weights!");
        }

        let weights = Matrix::read_from_file(file)?;
        let inputs = weights.rows();
        let outputs = weights.columns();

        let strings = read_line(file)?;

        if strings.len() < 1 || strings[0] != "Biases" {
            return read_error(file, "Cannot read layer biases!");
        }

        let biases = Vector::read_from_file(file)?;

        if biases.len() != outputs {
            return read_error(file, "Biases don't match layer outputs!");
        }

        let strings = read_line(file)?;

        if strings.len() < 1 || strings[0] != G::identifier() {
            return read_error(file, "Cannot read layer gradient descent!");
        }

        let gradient_descent = G::read_from_file(file)?;

        Ok(FullyConnectedLayer::construct(
            inputs, outputs,
            weights,
            biases,
            gradient_descent,
        ))
    }

    fn write_to_file(&self, file: &mut File) -> Result<()> {
        let indentation = file.indentation();
        write!(file, "{}Weights\n", indentation)?;
        file.indent();
        self.weights.write_to_file(file)?;
        write!(file, "{}Biases\n", indentation)?;
        self.biases.write_to_file(file)?;
        write!(file, "{}{}\n", indentation, G::identifier())?;
        self.gradient_descent.write_to_file(file)?;
        file.unindent();
        Ok(())
    }
}
