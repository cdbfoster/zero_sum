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

use std::sync::mpsc::Receiver;

use analysis::{Evaluator, Extrapolatable};
use state::State;

/// Provides search capabilities.
pub trait Search<S, E> where
    S: State + Extrapolatable<<S as State>::Ply>,
    E: Evaluator<State = S> {
    type Analysis;

    /// Generates an analysis of `state`.  `interrupt` is optionally provided to interrupt long searches.
    fn search(&mut self, state: &S, interrupt: Option<Receiver<()>>) -> Self::Analysis;
}

pub mod pvsearch;
