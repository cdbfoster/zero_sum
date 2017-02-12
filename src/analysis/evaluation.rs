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

use std::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Sub};

use state::State;

/// An evaluation type.
///
/// This is usually a tuple around a signed numeric type.
///
/// # Example
///
/// There is a [helper macro](../macro.prepare_evaluation_tuple.html) to facilitate the implementation of tuple structs:
///
/// ```rust
/// #[macro_use]
/// extern crate zero_sum;
/// # use zero_sum::analysis::Evaluation;
/// # use std::fmt;
/// use std::i32;
/// use std::ops::{Add, Div, Mul, Neg, Sub};
///
/// #[derive(Clone, Copy, PartialEq, PartialOrd)]
/// struct Eval(i32);
///
/// prepare_evaluation_tuple!(Eval); // impl Add, Div, Mul, Neg, Sub, and Display
///
/// impl Evaluation for Eval {
///     fn null() -> Eval { Eval(0) }
///     fn epsilon() -> Eval { Eval(1) }
///     fn win() -> Eval { Eval(100000) }
///     fn max() -> Eval { Eval(i32::MAX) }
///     fn is_win(&self) -> bool { self.0.abs() > 99000 }
/// }
/// # fn main() { }
/// ```
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
    /// An empty, or zero evaluation.
    fn null() -> Self;
    /// The smallest step to consider.
    fn epsilon() -> Self;
    /// The base value of a win.  The evaluation system may add or subtract to it in
    /// in order to promote it or discourage it in favor of others in the search.
    fn win() -> Self;
    /// The maximum value representable.  This must be safely negatable.
    fn max() -> Self;
    /// Returns `true` if this evaluation contains a win.  This is usually a check to
    /// see if the absolute value is above a certain threshold.
    fn is_win(&self) -> bool;
}

/// Provides evaluation capabilities.
///
/// This is usually implemented on a `State`.
pub trait Evaluatable<E> where
    E: Evaluation {
    /// Returns the evaluation of the current state.
    fn evaluate(&self) -> E;

    /// Returns the evaluation of the state after executing `plies`.
    ///
    /// # Panics
    /// Will panic if the execution of any ply in `plies` causes an error.
    fn evaluate_plies(&self, plies: &[Self::Ply]) -> E where Self: State {
        match self.execute_plies(plies) {
            Ok(state) => if plies.len() % 2 == 0 {
                state.evaluate()
            } else {
                -state.evaluate()
            },
            Err(error) => panic!("Error calculating evaluation: {}", error),
        }
    }
}

/// Implement arithmetic operators (`Add`, `Sub`, `Mul`, `Neg`, `Div`) and `Display` for a tuple
/// struct in terms of the enclosed type.
///
/// # Example
///
/// ```rust
/// #[macro_use]
/// extern crate zero_sum;
/// # use zero_sum::analysis::Evaluation;
/// # use std::fmt;
/// use std::i32;
/// use std::ops::{Add, Div, Mul, Neg, Sub};
///
/// #[derive(Clone, Copy, PartialEq, PartialOrd)]
/// struct Eval(i32);
///
/// prepare_evaluation_tuple!(Eval); // impl Add, Div, Mul, Neg, Sub, and Display
///
/// impl Evaluation for Eval {
///     fn null() -> Eval { Eval(0) }
///     fn epsilon() -> Eval { Eval(1) }
///     fn win() -> Eval { Eval(100000) }
///     fn max() -> Eval { Eval(i32::MAX) }
///     fn is_win(&self) -> bool { self.0.abs() > 99000 }
/// }
/// # fn main() { }
/// ```
#[macro_export]
macro_rules! prepare_evaluation_tuple {
    ($type_: ident) => {
        impl ::std::ops::Add for $type_ {
            type Output = $type_;
            fn add(self, $type_(b): $type_) -> $type_ {
                let $type_(a) = self;
                $type_(a + b)
            }
        }

        impl ::std::ops::Sub for $type_ {
            type Output = $type_;
            fn sub(self, $type_(b): $type_) -> $type_ {
                let $type_(a) = self;
                $type_(a - b)
            }
        }

        impl ::std::ops::Mul for $type_ {
            type Output = $type_;
            fn mul(self, $type_(b): $type_) -> $type_ {
                let $type_(a) = self;
                $type_(a * b)
            }
        }

        impl ::std::ops::Div for $type_ {
            type Output = $type_;
            fn div(self, $type_(b): $type_) -> $type_ {
                let $type_(a) = self;
                $type_(a / b)
            }
        }

        impl ::std::ops::Neg for $type_ {
            type Output = $type_;
            fn neg(self) -> $type_ {
                let $type_(a) = self;
                $type_(-a)
            }
        }

        impl ::std::fmt::Display for $type_ {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                let $type_(a) = *self;
                write!(f, "{}", a)
            }
        }
    }
}
