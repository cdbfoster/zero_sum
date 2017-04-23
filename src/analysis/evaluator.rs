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
///     fn shift(self, steps: i32) -> Eval { Eval(self.0 + steps) }
///     fn win() -> Eval { Eval(100000) }
///     fn max() -> Eval { Eval(i32::MAX) }
///     fn is_win(&self) -> bool { self.0 > 99000 }
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
    /// Shift the evaluation by the smallest representable amount `steps` times in the positive or negative direction.
    fn shift(self, steps: i32) -> Self;
    /// The base value of a win.  The evaluator may add or subtract to it in
    /// in order to promote it or discourage it in favor of others in the search.
    fn win() -> Self;
    /// The base value of a loss.  The evaluator may add or subtract to it in
    /// in order to promote it or discourage it in favor of others in the search.
    fn lose() -> Self { -Self::win() }
    /// The maximum value representable.  This must be safely negatable.
    fn max() -> Self;
    /// The minimum value representable.
    fn min() -> Self { -Self::max() }
    /// Returns `true` if this evaluation contains a win.  This is usually a check to
    /// see if the value is above a certain threshold.
    fn is_win(&self) -> bool;
    /// Returns `true` if this evaluation contains a loss.
    fn is_lose(&self) -> bool { (-*self).is_win() }
    /// Returns `true` if this evaluation is either a win or a loss.
    fn is_end(&self) -> bool { self.is_win() || self.is_lose() }
}

/// Evaluates a State.
pub trait Evaluator {
    type State: State;
    type Evaluation: Evaluation;

    /// Returns the evaluation of `state`.
    fn evaluate(&self, state: &Self::State) -> Self::Evaluation;

    /// Returns the evaluation of `state` after executing `plies`.
    ///
    /// # Panics
    /// Will panic if the execution of any ply in `plies` causes an error.
    fn evaluate_plies(&self, state: &Self::State, plies: &[<Self::State as State>::Ply]) -> Self::Evaluation {
        let mut state = state.clone();
        if let Err(error) = state.execute_plies(plies) {
            panic!("Error calculating evaluation: {}", error);
        }
        if plies.len() % 2 == 0 {
            self.evaluate(&state)
        } else {
            -self.evaluate(&state)
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
///     fn shift(self, steps: i32) -> Eval { Eval(self.0 + steps) }
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
