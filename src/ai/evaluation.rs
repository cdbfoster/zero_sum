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
use std::ops::{Add, Div, Mul, Neg, Sub};

use ply::Ply;
use resolution::Resolution;
use state::State;

pub trait Evaluation:
    Sized +
    Clone +
    Copy +
    Display +
    Add<Output = Self> +
    Sub<Output = Self> +
    Mul<Output = Self> +
    Neg<Output = Self> +
    Div<Output = Self> +
    PartialEq +
    PartialOrd {
    fn null() -> Self;
    fn epsilon() -> Self;
    fn win() -> Self;
    fn max() -> Self;
    fn is_win(&self) -> bool;
}

pub trait Evaluatable<E> where
    E: Evaluation {
    fn evaluate(&self) -> E;

    fn evaluate_plies<P, R>(&self, plies: &[P]) -> E where P: Ply, R: Resolution, Self: State<P, R> {
        let mut temp_state = self.clone();
        for ply in plies.iter() {
            match temp_state.execute_ply(ply) {
                Ok(next) => temp_state = next,
                Err(error) => panic!("Error calculating evaluation: {}, {:?}", error, ply),
            }
        }
        if plies.len() % 2 == 0 {
            temp_state.evaluate()
        } else {
            -temp_state.evaluate()
        }
    }
}

#[macro_export]
macro_rules! prepare_evaluation_tuple {
    ($type_: ident) => {
        impl_tuple_operation! { impl Add for $type_ { fn add } }
        impl_tuple_operation! { impl Sub for $type_ { fn sub } }
        impl_tuple_operation! { impl Mul for $type_ { fn mul } }
        impl_tuple_operation! { impl Div for $type_ { fn div } }

        impl Neg for $type_ {
            type Output = $type_;
            fn neg(self) -> $type_ {
                let $type_(a) = self;
                $type_(-a)
            }
        }

        impl fmt::Display for $type_ {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let $type_(a) = *self;
                write!(f, "{}", a)
            }
        }
    }
}

#[macro_export]
macro_rules! impl_tuple_operation {
    (impl $trait_: ident for $type_: ident { fn $method: ident }) => {
        impl $trait_ for $type_ {
            type Output = $type_;
            fn $method(self, $type_(b): $type_) -> $type_ {
                let $type_(a) = self;
                $type_(a.$method(&b))
            }
        }
    }
}
