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

use smallmath::Vector;

use serialization::{File, read_error, read_line, Serializable};

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
