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
// Copyright 2016 Chris Foster
//

use std::fmt::Display;

use super::{Ply, Resolution};

pub trait State<P, R>: Clone + Display where
    P: Ply,
    R: Resolution {
    fn execute_ply_preallocated(&self, ply: &P, next: &mut Self)-> Result<(), String>;
    fn check_resolution(&self) -> Option<R>;

    fn execute_ply(&self, ply: &P) -> Result<Self, String> {
        let mut next = self.clone();
        match self.execute_ply_preallocated(ply, &mut next) {
            Ok(_) => Ok(next),
            Err(error) => Err(error),
        }
    }
}
