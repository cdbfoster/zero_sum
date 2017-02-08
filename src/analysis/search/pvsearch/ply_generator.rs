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

use std::sync::{Arc, Mutex};

use analysis::Extrapolatable;
use ply::Ply;
use super::history::History;

pub struct PlyGenerator<'a, X, P> where
    X: 'a + Extrapolatable<P>,
    P: Ply {
    state: &'a X,
    principal_ply: Option<P>,
    history: Arc<Mutex<History>>,
    plies: Vec<P>,
    operation: u8,
}

impl<'a, X, P> PlyGenerator<'a, X, P> where
    X: 'a + Extrapolatable<P>,
    P: Ply {
    pub fn new(state: &'a X, principal_ply: Option<P>, history: Arc<Mutex<History>>) -> PlyGenerator<'a, X, P> {
        PlyGenerator {
            state: state,
            principal_ply: principal_ply,
            history: history,
            plies: Vec::new(),
            operation: 0,
        }
    }
}

impl<'a, X, P> Iterator for PlyGenerator<'a, X, P> where
    X: 'a + Extrapolatable<P>,
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

                self.plies = self.state.extrapolate();

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

#[cfg(all(test, feature = "with_tak"))]
mod test_tak {
    use test::Bencher;

    use super::*;
    use analysis::search::{Search, PvSearch};
    use impls::tak::*;

    #[bench]
    #[ignore]
    fn bench_ply_generator_drain(b: &mut Bencher) {
        let mut search = PvSearch::<Evaluation, State, Ply, Resolution>::with_depth(5);
        let state = State::new(5);
        search.search(&state, None);

        b.iter(|| {
            let mut r = Ply::Place { x: 0, y: 0, piece: Piece::Flatstone(Color::White) };
            let ply_generator = PlyGenerator::new(&state, None, search.history.clone());
            for ply in ply_generator {
                r = ply;
            }
            r
        });
    }
}
