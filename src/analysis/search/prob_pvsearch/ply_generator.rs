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

use std::marker::PhantomData;
use std::rc::Rc;

use analysis::Extrapolatable;
use analysis::search::prob_pvsearch::estimator::ProbabilityEstimator;
use state::State;

pub struct PlyGenerator<'a, S, P> where
    S: 'a + State + Extrapolatable<<S as State>::Ply>,
    P: ProbabilityEstimator<State = S> {
    state: &'a S,
    principal_ply: Option<<S as State>::Ply>,
    plies: Vec<(<S as State>::Ply, f32)>,
    scale: f32,
    operation: u8,
    estimator: Rc<P>,
}

impl<'a, S, P> PlyGenerator<'a, S, P> where
    S: 'a + State + Extrapolatable<<S as State>::Ply>,
    P: ProbabilityEstimator<State = S> {
    pub fn new(state: &'a S, principal_ply: Option<<S as State>::Ply>, estimator: &Rc<P>) -> PlyGenerator<'a, S, P> {
        let (plies, scale) = estimator.estimate_probabilities(state, &state.extrapolate());

        PlyGenerator {
            state: state,
            principal_ply: principal_ply,
            plies: plies,
            scale: scale,
            operation: 0,
            estimator: estimator.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.plies.len()
    }
}

impl<'a, S, P> Iterator for PlyGenerator<'a, S, P> where
    S: 'a + State + Extrapolatable<<S as State>::Ply>,
    P: ProbabilityEstimator<State = S> {
    type Item = (<S as State>::Ply, f32);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.operation == 0 {
                self.operation += 1;

                if let Some(ref principal_ply) = self.principal_ply {
                    return Some((
                        principal_ply.clone(),
                        self.estimator.estimate_probability(self.state, principal_ply) / self.scale,
                    ));
                }
            }

            if self.operation == 1 {
                self.operation += 1;
                self.plies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }

            if self.operation == 2 {
                let ply = self.plies.pop();

                if let Some(ref ply) = ply {
                    if Some(&ply.0) != self.principal_ply.as_ref() {
                        return Some(ply.clone());
                    }
                } else {
                    return ply;
                }
            }
        }
    }
}
