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

//! Tools for searching the game tree.

use std::any::Any;
use std::fmt::Display;
use std::sync::mpsc::Receiver;

use analysis::Extrapolatable;
use state::State;

/// The results of the search.
///
/// The search returns a boxed `Analysis`, which can either be printed as-is, or downcast into
/// a concrete analysis type from a particular search.
///
/// # Example
///
/// ```rust
/// # #[macro_use] extern crate zero_sum;
/// use zero_sum::analysis::search::{PvSearch, PvSearchAnalysis, Search};
/// # use std::ops::{Add, Div, Mul, Neg, Sub};
/// # #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)] struct Eval(i8);
/// # prepare_evaluation_tuple!(Eval);
/// # impl zero_sum::analysis::Evaluation for Eval { fn null() -> Eval { Eval(0) } fn shift(self, _: i32) -> Eval { Eval(0) } fn win() -> Eval { Eval(0) } fn max() -> Eval { Eval(0) } fn is_win(&self) -> bool { false } }
/// # #[derive(Clone, Debug, Hash, PartialEq)] struct Ply(i8);
/// # impl zero_sum::Ply for Ply { }
/// # impl std::fmt::Display for Ply { fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { Ok(()) } }
/// # struct Resolution(i8);
/// # impl zero_sum::Resolution for Resolution { fn get_winner(&self) -> Option<u8> { None } fn is_draw(&self) -> bool { false } }
/// # #[derive(Clone, Eq, Hash, PartialEq)] struct State(i8);
/// # impl State { fn new() -> State { State(0) } }
/// # impl zero_sum::State for State { type Ply = Ply; type Resolution = Resolution; fn get_ply_count(&self) -> usize { 0 } fn execute_ply(&mut self, _: Option<&Ply>) -> Result<(), String> { Ok(()) } fn revert_ply(&mut self, _: Option<&Ply>) -> Result<(), String> { Ok(()) } fn check_resolution(&self) -> Option<Resolution> { None } }
/// # struct Evaluator;
/// # impl zero_sum::analysis::Evaluator for Evaluator { type State = State; type Evaluation = Eval; fn evaluate(&self, _: &State) -> Eval { Eval(0) } }
/// # impl zero_sum::analysis::Extrapolatable<Ply> for State { fn extrapolate(&self) -> Vec<Ply> { Vec::new() } }
/// # impl std::fmt::Display for State { fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { Ok(()) } }
/// # fn main() {
///
/// let state = State::new();
/// let (interrupt_sender, interrupt_receiver) = std::sync::mpsc::channel();
///
/// let evaluator = Evaluator;
/// let mut search = PvSearch::with_depth(evaluator, 5);
/// let analysis = search.search(&state, Some(interrupt_receiver));
///
/// println!("{}", analysis);
///
/// let pvsearch_analysis = analysis.as_any().downcast_ref::<PvSearchAnalysis<State, Evaluator>>().unwrap();
/// println!("{}", pvsearch_analysis.evaluation);
/// # }
/// ```
pub trait Analysis: Display {
    fn as_any(&self) -> &Any;
}

/// Provides search capabilities.
pub trait Search<S> where
    S: State + Extrapolatable<<S as State>::Ply> {
    /// Generates an analysis of `state`.  `interrupt` is optionally provided to interrupt long searches.
    fn search(&mut self, state: &S, interrupt: Option<Receiver<()>>) -> Box<Analysis>;
}

#[cfg(feature = "with_mctsearch")]
pub use self::mctsearch::{MctSearch, MctSearchAnalysis};

pub use self::pvsearch::{PvSearch, PvSearchAnalysis};

#[cfg(feature = "with_mctsearch")]
mod mctsearch;

mod pvsearch;
