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

pub struct File {
    handle: StdFile,
    pub(super) line: usize,
    indentation: usize,
}

impl File {
    pub fn new(handle: StdFile, indentation: usize) -> File {
        File {
            handle: handle,
            line: 0,
            indentation: indentation,
        }
    }

    pub fn open(filename: &str) -> Result<File> {
        OpenOptions::new().read(true).open(filename).map(|f| File { handle: f, line: 0, indentation: 0 })
    }

    pub fn create(filename: &str) -> Result<File> {
        OpenOptions::new().write(true).truncate(true).create(true).open(filename).map(|f| File { handle: f, line: 0, indentation: 0 })
    }

    pub fn append(filename: &str, indentation: usize) -> Result<File> {
        OpenOptions::new().append(true).open(filename).map(|f| File { handle: f, line: 0, indentation: indentation })
    }

    pub fn indent(&mut self) {
        self.indentation += 4;
    }

    pub fn unindent(&mut self) {
        self.indentation -= 4;
    }

    pub fn indentation(&mut self) -> String {
        (0..self.indentation).map(|_| " ").collect::<String>()
    }
}

impl Read for File {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        self.handle.read(buf)
    }
}

impl Write for File {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        self.handle.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.handle.flush()
    }
}
