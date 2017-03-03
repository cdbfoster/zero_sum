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

use std::cell::{RefCell, RefMut};
use std::marker::PhantomData;

use analysis::Extrapolatable;
use analysis::search::prob_pvsearch::estimator::ProbabilityEstimator;
use resolution::Resolution;
use state::State;

pub struct NaiveProbabilityEstimator<S> where
    S: State + Extrapolatable<<S as State>::Ply> {
    temp: RefCell<Option<S>>,
    temp_next: RefCell<Option<S>>,
}

impl<S> NaiveProbabilityEstimator<S> where
    S: State + Extrapolatable<<S as State>::Ply> {
    pub fn new() -> NaiveProbabilityEstimator<S> {
        NaiveProbabilityEstimator {
            temp: RefCell::new(None),
            temp_next: RefCell::new(None),
        }
    }

    fn ensure_temp_states(&self, state: &S) {
        let mut temp = self.temp.borrow_mut();
        if temp.is_none() {
            *temp = Some(state.clone());
        }

        let mut temp_next = self.temp_next.borrow_mut();
        if temp_next.is_none() {
            *temp_next = Some(state.clone());
        }
    }
}

impl<S> ProbabilityEstimator for NaiveProbabilityEstimator<S> where
    S: State + Extrapolatable<<S as State>::Ply> {
    type State = S;

    fn estimate_probability(&self, state: &S, ply: &<S as State>::Ply) -> f32 {
        return 1.0;
    }
}
