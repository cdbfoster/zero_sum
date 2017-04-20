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
use std::sync::{Arc, Mutex};

use analysis::Extrapolatable;
use ply::Ply;
use super::history::History;

pub struct PlyGenerator<X, P> where
    X: Extrapolatable<P>,
    P: Ply {
    principal_ply: Option<P>,
    history: Arc<Mutex<History>>,
    plies: Vec<P>,
    operation: u8,
    phantom: PhantomData<X>,
}

impl<X, P> PlyGenerator<X, P> where
    X: Extrapolatable<P>,
    P: Ply {
    pub fn new(state: &X, principal_ply: Option<P>, history: Arc<Mutex<History>>) -> PlyGenerator<X, P> {
        PlyGenerator {
            principal_ply: principal_ply,
            history: history,
            plies: state.extrapolate(),
            operation: 0,
            phantom: PhantomData,
        }
    }
}

impl<X, P> Iterator for PlyGenerator<X, P> where
    X: Extrapolatable<P>,
    P: Ply {
    type Item = P;

    fn next(&mut self) -> Option<P> {
        loop {
            if self.operation == 0 {
                self.operation += 1;

                if self.principal_ply.is_some() {
                    return self.principal_ply.clone();
                }
            }

            if self.operation == 1 {
                self.operation += 1;

                {
                    let history = self.history.lock().unwrap();

                    history.sort_plies(&mut self.plies);
                }
            }

            if self.operation == 2 {
                let ply = self.plies.pop();

                if ply != self.principal_ply || ply.is_none() {
                    return ply;
                }
            }
        }
    }
}
