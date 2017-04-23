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

pub trait Analysis: Display {
    fn as_any(&self) -> &Any;
}

/// Provides search capabilities.
pub trait Search<S> where
    S: State + Extrapolatable<<S as State>::Ply> {
    /// Generates an analysis of `state`.  `interrupt` is optionally provided to interrupt long searches.
    fn search(&mut self, state: &S, interrupt: Option<Receiver<()>>) -> Box<Analysis>;
}

pub use self::pvsearch::{PvSearch, PvSearchAnalysis};

mod pvsearch;
