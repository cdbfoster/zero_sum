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

use layer::{SplitLayer, Layer};
use serialization::{File, read_error, read_layer, read_line, Serializable};

impl Serializable for SplitLayer {
    fn read_from_file(file: &mut BufReader<File>) -> Result<SplitLayer> {
        let strings = read_line(file)?;

        if strings.len() < 1 {
            return read_error(file, "Cannot read layer count!");
        }

        let layer_count = if let Ok(layer_count) = usize::from_str(&strings[0]) {
            layer_count
        } else {
            return read_error(file, "Cannot parse layer count!");
        };

        let mut layers: Vec<Box<Layer>> = Vec::new();
        for _ in 0..layer_count {
            let layer = read_layer(file)?;
            layers.push(layer);
        }

        Ok(SplitLayer::new(layers))
    }

    fn write_to_file(&self, file: &mut File) -> Result<()> {
        let indentation = file.indentation();
        write!(file, "{}{}\n", indentation, self.layers.len())?;
        for layer in &self.layers {
            write!(file, "{}{}\n", indentation, layer.get_identifier())?;
            file.indent();
            layer.write_to_file(file)?;
            file.unindent();
        }
        Ok(())
    }
}
