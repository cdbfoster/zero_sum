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

use std::io::{BufRead, BufReader, Error, ErrorKind, Result};

use serialization::{File, Identifiable};

pub trait Serializable: Identifiable {
    fn read_from_file(file: &mut BufReader<File>) -> Result<Self> where Self: Sized;
    fn write_to_file(&self, file: &mut File) -> Result<()>;
}

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

pub use self::matrix::write_matrix;

mod gradient_descent;
#[macro_use]
mod layer;
mod matrix;
mod vector;
