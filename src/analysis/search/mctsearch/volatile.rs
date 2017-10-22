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

use std::ptr;

#[derive(Clone)]
pub struct Volatile<T: Copy>(T);

impl<T: Copy> Volatile<T> {
    pub fn new(value: T) -> Volatile<T> {
        Volatile(value)
    }

    pub fn read(&self) -> T {
        unsafe { ptr::read_volatile(&self.0) }
    }

    pub fn write(&mut self, value: T) {
        unsafe { ptr::write_volatile(&mut self.0, value); }
    }

    pub fn update<F>(&mut self, f: F) where F: FnOnce(&mut T) {
        let mut value = self.read();
        f(&mut value);
        self.write(value);
    }
}
