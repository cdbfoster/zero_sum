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

use state::State;

pub trait ProbabilityEstimator {
    type State: State;
    fn estimate_probability(&self, state: &Self::State, ply: &<Self::State as State>::Ply) -> f32;
    fn estimate_probabilities(&self, state: &Self::State, plies: &[<Self::State as State>::Ply]) -> (Vec<(<Self::State as State>::Ply, f32)>, f32) {
        let mut plies = plies.iter().map(|ply| {(
            ply.clone(),
            self.estimate_probability(state, ply),
        )}).collect::<Vec<_>>();

        let sum = plies.iter().fold(0.0, |sum, &(_, probability)| sum + probability);
        let sum_inv = 1.0 / sum;

        for &mut (_, ref mut probability) in &mut plies {
            *probability *= sum_inv;
        }

        (plies, sum)
    }
}

pub use self::naive::NaiveProbabilityEstimator;

mod naive;
