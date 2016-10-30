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

use impls::tak::Color;
use resolution;

pub enum Resolution {
    Road(Color),
    Flat(Color),
    Draw,
}

impl resolution::Resolution for Resolution {
    fn is_win(&self) -> bool {
        match *self {
            Resolution::Road(_) |
            Resolution::Flat(_) => true,
            Resolution::Draw => false,
        }
    }

    fn is_draw(&self) -> bool {
        match *self {
            Resolution::Draw => true,
            _ => false,
        }
    }
}
