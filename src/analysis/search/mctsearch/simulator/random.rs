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

use std::thread;
use std::time::Duration;

use rand::Rng;

use analysis::Extrapolatable;
use analysis::search::mctsearch::Simulator;
use analysis::search::mctsearch::jkiss32::JKiss32Rng;
use resolution::Resolution;
use state::State;

#[derive(Clone)]
pub struct RandomSimulator<S> where
    S: State + Extrapolatable<<S as State>::Ply>,
    <S as State>::Ply: Send {
    rng: JKiss32Rng,
    current_state: Option<S>,
    plies: Vec<<S as State>::Ply>,
}

impl<S> RandomSimulator<S> where
    S: State + Extrapolatable<<S as State>::Ply>,
    <S as State>::Ply: Send {
    pub fn new() -> RandomSimulator<S> {
        RandomSimulator {
            rng: JKiss32Rng::new(),
            current_state: None,
            plies: Vec::new(),
        }
    }
}

impl<S> Simulator<S> for RandomSimulator<S> where
    S: Send + State + Extrapolatable<<S as State>::Ply>,
    <S as State>::Ply: Send {
    fn simulate(&mut self, state: &S) -> f32 {
        let mut current_state = if let Some(ref mut allocated_state) = self.current_state {
            allocated_state.clone_from(state);
            allocated_state
        } else {
            self.current_state = Some(state.clone());
            self.current_state.as_mut().unwrap()
        };

        let mut steps = 0;

        loop {
            self.plies.clear();
            current_state.extrapolate_into(&mut self.plies);
            if self.plies.is_empty() {
                return 0.0;
            }

            while current_state.execute_ply(self.rng.choose(&self.plies)).is_err() {
            }

            steps += 1;

            if let Some(resolution) = current_state.check_resolution() {
                //println!("{}: Simulation finished in {} steps: {:?}", thread::current().name().unwrap(), steps, resolution);
                if let Some(winner) = resolution.get_winner() {
                    //println!("Winner is {}.", winner);
                    if state.get_ply_count() % 2 == winner as usize {
                        // A win doesn't go to state, but the state before it
                        return -1.0;
                    } else {
                        return 1.0;
                    }
                } else {
                    //println!("Draw");
                    return 0.0;
                }
            }
        }
    }

    fn split(&self) -> RandomSimulator<S> {
        RandomSimulator {
            rng: JKiss32Rng::new(),
            current_state: self.current_state.clone(),
            plies: self.plies.clone(),
        }
    }
}
