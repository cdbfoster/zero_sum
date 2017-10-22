//
// This file is part of smallmath.
//
// smallmath is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// smallmath is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with smallmath. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright 2017 Chris Foster
//

pub use self::abs::{abs, abs_assign};
pub use self::add::{add, add_assign};
//pub use self::cross::cross;
pub use self::divide::{divide, divide_assign};
//pub use self::dot::dot;
pub use self::multiply::{multiply, multiply_assign};
pub use self::offset::{offset, offset_assign};
pub use self::reciprocal::{reciprocal, reciprocal_assign};
pub use self::scale::{scale, scale_assign};
pub use self::signum::{signum, signum_assign};
pub use self::sqrt::{sqrt, sqrt_assign};
pub use self::square::{square, square_assign};
pub use self::subtract::{subtract, subtract_assign};

mod abs;
mod add;
//mod cross;
mod divide;
//mod dot;
mod multiply;
mod offset;
mod reciprocal;
mod scale;
mod signum;
mod sqrt;
mod square;
mod subtract;
