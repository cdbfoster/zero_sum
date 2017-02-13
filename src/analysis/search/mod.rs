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

use std::fmt;
use std::sync::mpsc::Receiver;

use analysis::{Evaluation, Evaluator, Extrapolatable};
use state::State;

/// The result of a search.
pub struct Analysis<'a, S, E> where
    S: 'a + State + Extrapolatable<<S as State>::Ply>,
    E: Evaluator<State = S> {
    /// A reference to the state on which the search was performed.
    pub state: &'a S,
    /// The evaluation of the state after applying the principal variation.
    pub evaluation: <E as Evaluator>::Evaluation,
    /// The principal variation of the state.
    pub principal_variation: Vec<<S as State>::Ply>,
    /// Optional statistics from the search may be available for printing.
    pub stats: Option<Box<fmt::Display>>,
}

/// Provides search capabilities
pub trait Search<S, E> where
    S: State + Extrapolatable<<S as State>::Ply>,
    E: Evaluator<State = S> {
    /// Generates an analysis of `state`.  `interrupt` is optionally provided to interrupt long searches.
    fn search<'a>(&mut self, state: &'a S, interrupt: Option<Receiver<()>>) -> Analysis<'a, S, E>;
}

impl<'a, S, E> fmt::Display for Analysis<'a, S, E> where
    S: 'a + State + Extrapolatable<<S as State>::Ply>,
    E: Evaluator<State = S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "State: {}\n", self.state));
        if let Ok(result) = self.state.execute_plies(&self.principal_variation) {
            try!(write!(f, "Resultant State: {}\n", result));
            // XXX Make Resolution require Display and print the resolution if any
        }
        try!(write!(f, "Evaluation: {}{}", self.evaluation, if self.evaluation.is_end() {
            if self.evaluation.is_win() {
                " (Win)\n"
            } else {
                " (Lose)\n"
            }
        } else {
            "\n"
        }));
        try!(write!(f, "Principal Variation:"));
        for ply in &self.principal_variation {
            try!(write!(f, "\n  {}", ply));
        }
        if let Some(ref stats) = self.stats {
            try!(write!(f, "\nStatistics:\n{}", stats));
        }

        Ok(())
    }
}

pub use self::pvsearch::PvSearch;

mod pvsearch;
