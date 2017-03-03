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

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use std::sync::mpsc::Receiver;
use std::time::Instant;
use std::usize;

use analysis::{Evaluation, Evaluator, Extrapolatable};
use analysis::search::Search;
use state::State;

use self::estimator::ProbabilityEstimator;
use self::ply_generator::PlyGenerator;
use self::statistics::{Statistics, StatisticsLevel};
use self::transposition_table::{Bound, TranspositionTable, TranspositionTableEntry};

pub struct Analysis<S, E> where
    S: State + Extrapolatable<<S as State>::Ply>,
    E: Evaluator<State = S> {
    /// The state on which the search was performed.
    pub state: S,
    /// The evaluation of the state after applying the principal variation.
    pub evaluation: <E as Evaluator>::Evaluation,
    /// The principal variation of the state.
    pub principal_variation: Vec<<S as State>::Ply>,
    pub statistics: Statistics,
}

pub struct ProbabilisticPvSearch<S, E, P> where
    S: State + Extrapolatable<<S as State>::Ply>,
    E: Evaluator<State = S>,
    P: ProbabilityEstimator<State = S> {
    evaluator: E,
    estimator: Rc<P>,
    node_budget: u64,
    branching_factor: f32,
    deepen_rate: f32,
    transposition_table: TranspositionTable<S, <E as Evaluator>::Evaluation>,
    interrupted: bool,
}

impl<S, E, P> ProbabilisticPvSearch<S, E, P> where
    S: State + Extrapolatable<<S as State>::Ply>,
    E: Evaluator<State = S>,
    P: ProbabilityEstimator<State = S> {
    pub fn with_iterations(evaluator: E, estimator: P, iterations: u8, branching_factor: f32, deepen_rate: f32) -> ProbabilisticPvSearch<S, E, P> {
        ProbabilisticPvSearch {
            evaluator: evaluator,
            estimator: Rc::new(estimator),
            node_budget: branching_factor.powf(iterations as f32) as u64,
            branching_factor: branching_factor,
            deepen_rate: deepen_rate,
            transposition_table: TranspositionTable::new(),
            interrupted: false,
        }
    }

    fn minimax(
        &mut self,
        state: &S,
        principal_variation: &mut Vec<<S as State>::Ply>,
        node_budget: u64,
        depth: u8,
        mut alpha: <E as Evaluator>::Evaluation,
        beta: <E as Evaluator>::Evaluation,
        states_preallocated: &[RefCell<S>],
        stats: &mut Statistics,
        interrupt: Option<&Receiver<()>>,
    ) -> <E as Evaluator>::Evaluation {
        {
            let mut iteration_stats = stats.iteration.last_mut().unwrap();
            if iteration_stats.len() < depth as usize + 1 {
                iteration_stats.push(StatisticsLevel::new());
            }

            if node_budget == 0 || state.check_resolution().is_some() {
                principal_variation.clear();
                iteration_stats[depth as usize].evaluated += 1;
                return self.evaluator.evaluate(state);
            }

            iteration_stats[depth as usize].visited += 1;
        }

        if let Some(entry) = self.transposition_table.get(state) {
            let mut iteration_stats = stats.iteration.last_mut().unwrap();
            iteration_stats[depth as usize].tt_hits += 1;

            let mut usable = false;

            if entry.node_budget >= node_budget &&
              (entry.bound == Bound::Exact ||
              (entry.bound == Bound::Upper && entry.value < alpha) ||
              (entry.bound == Bound::Lower && entry.value >= beta)) {
                usable = true;
            }

            if entry.bound == Bound::Exact && entry.value.is_end() {
                usable = true;
            }

            if usable {
                if let Ok(_) = state.execute_ply_preallocated(
                    &entry.principal_variation[0],
                    &mut states_preallocated[depth as usize].borrow_mut(),
                ) {
                    iteration_stats[depth as usize].tt_saves += 1;

                    principal_variation.clear();
                    principal_variation.append(&mut entry.principal_variation.clone());

                    return entry.value;
                }
            }
        }

        let ply_generator = PlyGenerator::new(
            state,
            principal_variation.first().cloned(),
            &self.estimator,
        );

        stats.iteration.last_mut().unwrap()[depth as usize].branch += ply_generator.len() as f32;

        let mut next_principal_variation = if !principal_variation.is_empty() {
            principal_variation[1..].to_vec()
        } else {
            Vec::new()
        };

        let mut first_iteration = true;
        let mut raised_alpha = false;

        for (iteration, (ref ply, probability)) in ply_generator.enumerate() {
            let next_state = {
                if let Err(_) = state.execute_ply_preallocated(
                    ply,
                    &mut states_preallocated[depth as usize].borrow_mut(),
                ) {
                    continue;
                }
                states_preallocated[depth as usize].borrow()
            };

            let next_node_budget = ((node_budget - 1) as f32 * probability) as u64; // Subtract 1 for the current node

            let next_eval = if first_iteration {
                -self.minimax(
                    &next_state, &mut next_principal_variation,
                    next_node_budget, depth + 1,
                    -beta, -alpha,
                    states_preallocated,
                    stats,
                    interrupt,
                )
            } else {
                let mut npv = next_principal_variation.clone();
                let next_eval = -self.minimax(
                    &next_state, &mut npv,
                    next_node_budget, depth + 1,
                    -alpha - <E as Evaluator>::Evaluation::epsilon(), -alpha,
                    states_preallocated,
                    stats,
                    interrupt,
                );

                if next_eval > alpha && next_eval < beta {
                    -self.minimax(
                        &next_state, &mut next_principal_variation,
                        next_node_budget, depth + 1,
                        -beta, -alpha,
                        states_preallocated,
                        stats,
                        interrupt,
                    )
                } else {
                    next_principal_variation = npv;
                    next_eval
                }
            };

            if next_eval > alpha {
                alpha = next_eval;
                raised_alpha = true;

                principal_variation.clear();
                principal_variation.push(ply.clone());
                principal_variation.append(&mut next_principal_variation.clone());

                if alpha >= beta {
                    break;
                }
            }

            first_iteration = false;

            if self.is_interrupted(&interrupt) {
                return alpha;
            }
        }

        if let Some(ply) = principal_variation.first() {
            if let Ok(_) = state.execute_ply_preallocated(
                ply,
                &mut *states_preallocated[depth as usize].borrow_mut(),
            ) {
                self.transposition_table.insert(state.clone(),
                    TranspositionTableEntry {
                        node_budget: node_budget,
                        value: alpha,
                        bound: if !raised_alpha {
                            Bound::Upper
                        } else if alpha >= beta {
                            Bound::Lower
                        } else {
                            Bound::Exact
                        },
                        principal_variation: principal_variation.clone(),
                        lifetime: 2,
                    }
                );
                stats.iteration.last_mut().unwrap()[depth as usize].tt_stores += 1;
            }
        }

        alpha
    }

    fn is_interrupted(&mut self, interrupt: &Option<&Receiver<()>>) -> bool {
        if self.interrupted {
            return true;
        } else if let Some(interrupt) = *interrupt {
            if let Ok(_) = interrupt.try_recv() {
                self.interrupted = true;
                return true;
            }
        }

        false
    }
}

impl<S, E, P> Search<S> for ProbabilisticPvSearch<S, E, P> where
    S: State + Extrapolatable<<S as State>::Ply>,
    E: Evaluator<State = S>,
    P: ProbabilityEstimator<State = S> {
    type Analysis = Analysis<S, E>;

    fn search(&mut self, state: &S, interrupt: Option<Receiver<()>>) -> Self::Analysis {
        let mut eval = <E as Evaluator>::Evaluation::null();
        let mut principal_variation = Vec::new();

        let start_move = Instant::now();

        let mut states = vec![RefCell::new(state.clone()); (self.node_budget as f32).log(self.branching_factor) as usize * 2];
        let mut stats = Statistics::new();

        let mut node_budget = 1;
        while node_budget <= self.node_budget {
            stats.iteration.push(Vec::new());

            let start_search = Instant::now();

            eval = self.minimax(
                state, &mut principal_variation,
                node_budget, 0,
                <E as Evaluator>::Evaluation::min(), <E as Evaluator>::Evaluation::max(),
                &states,
                &mut stats,
                interrupt.as_ref(),
            );

            let elapsed_search = start_search.elapsed();
            let elapsed_search = elapsed_search.as_secs() as f32 + elapsed_search.subsec_nanos() as f32 / 1_000_000_000.0;
            let elapsed_move = start_move.elapsed();
            let elapsed_move = elapsed_move.as_secs() as f32 + elapsed_move.subsec_nanos() as f32 / 1_000_000_000.0;

            stats.iteration.last_mut().unwrap()[0].time = elapsed_search;

            node_budget = (node_budget as f32 * self.branching_factor.powf(self.deepen_rate)) as u64;
        }

        for iteration in &mut stats.iteration {
            for depth in iteration {
                if depth.visited > 0 {
                    depth.branch /= (depth.visited - depth.tt_saves) as f32;
                }
            }
        }

        Analysis {
            state: state.clone(),
            evaluation: eval,
            principal_variation: principal_variation,
            statistics: stats,
        }
    }
}

impl<S, E> fmt::Display for Analysis<S, E> where
    S: State + Extrapolatable<<S as State>::Ply>,
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
        try!(write!(f, "\nStatistics: {}", self.statistics));
        Ok(())
    }
}

pub mod estimator;
mod ply_generator;
mod statistics;
mod transposition_table;
