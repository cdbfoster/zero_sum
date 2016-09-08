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

use std::fmt;
use std::marker::PhantomData;
use std::sync::mpsc::Receiver;

use ai::{Evaluatable, Evaluation, Extrapolatable};
use ply::Ply;
use resolution::Resolution;
use state::State;

pub struct Analysis<'a, E, S, P, R> where
    E: Evaluation,
    S: 'a + State<P, R> + Evaluatable<E> + Extrapolatable<P>,
    P: Ply,
    R: Resolution {
    pub state: &'a S,
    pub evaluation: E,
    pub principal_variation: Vec<P>,
    pub stats: Option<Box<fmt::Display>>,
    _phantom: PhantomData<R>,
}

pub trait Search<E, S, P, R> where
    E: Evaluation,
    S: State<P, R> + Evaluatable<E> + Extrapolatable<P>,
    P: Ply,
    R: Resolution {
    fn search<'a>(&mut self, state: &'a S, interrupt: Option<Receiver<()>>) -> Analysis<'a, E, S, P, R>;
}

impl<'a, E, S, P, R> fmt::Display for Analysis<'a, E, S, P, R> where
    E: Evaluation,
    S: 'a + State<P, R> + Evaluatable<E> + Extrapolatable<P>,
    P: Ply,
    R: Resolution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "State: {}\n", self.state));
        try!(write!(f, "Evaluation: {}\n", self.evaluation));
        try!(write!(f, "Principal Variation:"));
        for ply in &self.principal_variation {
            try!(write!(f, "\n  {}", ply));
        }
        if let Some(ref stats) = self.stats {
            write!(f, "\nStatistics:\n{}", stats)
        } else {
            Ok(())
        }
    }
}

pub use self::pvsearch::PvSearch;

mod pvsearch;
