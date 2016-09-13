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

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::mpsc::Receiver;
use std::u8;

use time;

use analysis::{Evaluatable, Evaluation, Extrapolatable};
use analysis::search::{Analysis, Search};
use ply::Ply;
use resolution::Resolution;
use state::State;

use self::history::History;
use self::ply_generator::PlyGenerator;
use self::statistics::{StatisticPrinter, Statistics};
use self::transposition_table::{Bound, TranspositionTable, TranspositionTableEntry};

/// A PVS implementation of `Search` with a few common optimizations.
///
/// # Example
///
/// ```rust
/// # #[macro_use] extern crate zero_sum;
/// # use zero_sum::analysis::search::{Search, PvSearch};
/// # use std::ops::{Add, Div, Mul, Neg, Sub};
/// # #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)] struct Eval(i8);
/// # prepare_evaluation_tuple!(Eval);
/// # impl zero_sum::analysis::Evaluation for Eval { fn null() -> Eval { Eval(0) } fn epsilon() -> Eval { Eval(0) } fn win() -> Eval { Eval(0) } fn max() -> Eval { Eval(0) } fn is_win(&self) -> bool { false } }
/// # #[derive(Clone, Debug, Hash, PartialEq)] struct Ply(i8);
/// # impl zero_sum::Ply for Ply { }
/// # impl std::fmt::Display for Ply { fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { Ok(()) } }
/// # struct Resolution(i8);
/// # impl zero_sum::Resolution for Resolution { fn is_win(&self) -> bool { false } fn is_draw(&self) -> bool { false } }
/// # #[derive(Clone, Eq, Hash, PartialEq)] struct State(i8);
/// # impl State { fn new() -> State { State(0) } }
/// # impl zero_sum::State<Ply, Resolution> for State { fn execute_ply_preallocated(&self, _: &Ply, _: &mut State) -> Result<(), String> { Ok(()) } fn check_resolution(&self) -> Option<Resolution> { None } }
/// # impl zero_sum::analysis::Evaluatable<Eval> for State { fn evaluate(&self) -> Eval { Eval(0) } }
/// # impl zero_sum::analysis::Extrapolatable<Ply> for State { fn extrapolate(&self) -> Vec<Ply> { Vec::new() } }
/// # impl std::fmt::Display for State { fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { Ok(()) } }
/// # fn main() {
/// let state = State::new();
/// let (interrupt_sender, interrupt_receiver) = std::sync::mpsc::channel();
///
/// let mut search = PvSearch::<Eval, State, Ply, Resolution>::with_depth(5);
/// let analysis = search.search(&state, Some(interrupt_receiver));
/// # }
/// ```
pub struct PvSearch<E, S, P, R> where
    E: Evaluation,
    S: State<P, R> + Evaluatable<E> + Extrapolatable<P>,
    P: Ply,
    R: Resolution {
    depth: u8,
    goal: u16,
    branching_factor: f32,
    history: Rc<RefCell<History>>,
    transposition_table: TranspositionTable<E, S, P, R>,
}

impl<E, S, P, R> PvSearch<E, S, P, R> where
    E: Evaluation,
    S: State<P, R> + Evaluatable<E> + Extrapolatable<P>,
    P: Ply,
    R: Resolution {
    /// Creates a `PvSearch` without a target depth or time goal.  It will search until
    /// it finds a favorable resolution, or until the search is interrupted.
    pub fn new() -> PvSearch<E, S, P, R> {
        PvSearch {
            depth: 0,
            goal: 0,
            branching_factor: 0.0,
            history: Rc::new(RefCell::new(History::new())),
            transposition_table: TranspositionTable::new(),
        }
    }

    /// Creates a `PvSearch` that will search to a maximum depth of `depth`.
    pub fn with_depth(depth: u8) -> PvSearch<E, S, P, R> {
        let mut search = PvSearch::new();
        search.depth = depth;
        search
    }

    /// Creates a `PvSearch` that will search until it predicts that it will exceed
    /// `goal` seconds with the next depth.  `branching_factor` is used to predict
    /// the required time to search at the next depth.
    pub fn with_goal(goal: u16, branching_factor: f32) -> PvSearch<E, S, P, R> {
        let mut search = PvSearch::new();
        search.goal = goal;
        search.branching_factor = if branching_factor <= 0.0 || branching_factor.is_nan() || branching_factor.is_infinite() {
            1.0
        } else {
            branching_factor
        };
        search
    }

    fn minimax(
        &mut self,
        state: &S,
        principal_variation: &mut Vec<P>,
        depth: u8,
        max_depth: u8,
        mut alpha: E,
        beta: E,
        states_preallocated: &[RefCell<S>],
        stats: &mut [Statistics],
        interrupt: Option<&Receiver<()>>,
    ) -> E {
        let search_iteration = (max_depth - depth) as usize;

        if depth == 0 || state.check_resolution().is_some() {
            stats[search_iteration - 1].evaluated += 1;
            principal_variation.clear();
            return state.evaluate();
        }

        stats[search_iteration].visited += 1;

        if let Some(entry) = self.transposition_table.get(state) {
            stats[search_iteration].tt_hits += 1;

            let mut usable = false;

            if entry.depth >= depth &&
              (entry.bound == Bound::Exact ||
              (entry.bound == Bound::Upper && entry.value < alpha) ||
              (entry.bound == Bound::Lower && entry.value >= beta)) {
                usable = true;
            }

            if entry.bound == Bound::Exact && entry.value.is_win() {
                usable = true;
            }

            if usable {
                if let Ok(_) = state.execute_ply_preallocated(
                    &entry.principal_variation[0],
                    &mut states_preallocated[search_iteration].borrow_mut(),
                ) {
                    stats[search_iteration].tt_saves += 1;

                    principal_variation.clear();
                    principal_variation.append(&mut entry.principal_variation.clone());

                    return entry.value;
                }
            }
        }

        let ply_generator = PlyGenerator::new(
            state,
            match principal_variation.first() {
                Some(ply) => Some(ply.clone()),
                None => None,
            },
            self.history.clone(),
        );

        let mut next_principal_variation = if !principal_variation.is_empty() {
            principal_variation[1..].to_vec()
        } else {
            Vec::new()
        };

        let mut first_iteration = true;
        let mut raised_alpha = false;

        for ply in ply_generator {
            let next_state = {
                if let Err(_) = state.execute_ply_preallocated(
                    &ply,
                    &mut *states_preallocated[search_iteration].borrow_mut(),
                ) {
                    continue;
                }
                states_preallocated[search_iteration].borrow()
            };

            let next_eval = if first_iteration {
                -self.minimax(
                    &next_state, &mut next_principal_variation, depth - 1, max_depth,
                    -beta, -alpha,
                    states_preallocated,
                    stats,
                    interrupt,
                )
            } else {
                let mut npv = next_principal_variation.clone();
                let next_eval = -self.minimax(
                    &next_state, &mut npv, depth - 1, max_depth,
                    -alpha - E::epsilon(), -alpha,
                    states_preallocated,
                    stats,
                    interrupt,
                );

                if next_eval > alpha && next_eval < beta {
                    -self.minimax(
                        &next_state, &mut next_principal_variation, depth - 1, max_depth,
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
                    {
                        let mut history = self.history.borrow_mut();
                        let entry = history.entry(&ply).or_insert(0);
                        *entry += 1 << depth;
                    }
                    break;
                }
            }

            first_iteration = false;

            if let Some(interrupt) = interrupt {
                if let Ok(_) = interrupt.try_recv() {
                    return E::null();
                }
            }
        }

        if let Some(ply) = principal_variation.first() {
            if let Ok(_) = state.execute_ply_preallocated(
                ply,
                &mut *states_preallocated[search_iteration].borrow_mut(),
            ) {
                self.transposition_table.insert(state.clone(),
                    TranspositionTableEntry {
                        depth: depth, // XXX Are we cutting the depth short from previous searches?
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
                stats[search_iteration].tt_stores += 1;
            }
        }

        alpha
    }
}

impl<E, S, P, R> Search<E, S, P, R> for PvSearch<E, S, P, R> where
    E: Evaluation,
    S: State<P, R> + Evaluatable<E> + Extrapolatable<P>,
    P: Ply,
    R: Resolution {
    fn search<'a>(&mut self, state: &'a S, interrupt: Option<Receiver<()>>) -> Analysis<'a, E, S, P, R> {
        let mut eval = E::null();
        let mut principal_variation = Vec::new();
        let mut statistics = Vec::new();

        let max_depth = if self.depth == 0 {
            u8::MAX - 1
        } else {
            self.depth
        };

        let start_move = time::precise_time_ns();

        self.history.borrow_mut().clear();

        let precalculated = match self.transposition_table.get(state) {
            Some(entry) => {
                if entry.bound == Bound::Exact {
                    principal_variation.append(&mut entry.principal_variation.clone());
                    entry.depth
                } else {
                    0
                }
            },
            None => 0,
        };

        for depth in 1..precalculated + 1 {
            statistics.push(vec![Statistics::new(); depth as usize]);
        }

        // Purge transposition table
        {
            let mut forget = Vec::with_capacity(self.transposition_table.len() / 5);

            for (key, entry) in self.transposition_table.iter_mut() {
                if entry.lifetime > 0 {
                    entry.lifetime -= 1;
                } else {
                    forget.push(key.clone());
                }
            }

            for key in forget {
                self.transposition_table.remove(&key);
            }
        }

        for depth in 1..max_depth + 1 - precalculated {
            let search_depth = depth + precalculated;

            let states_preallocated = vec![RefCell::new(state.clone()); depth as usize];
            statistics.push(vec![Statistics::new(); search_depth as usize]);

            let start_search = time::precise_time_ns();

            eval = self.minimax(
                state,
                &mut principal_variation,
                search_depth, search_depth,
                -E::max(), E::max(),
                &states_preallocated,
                &mut statistics[search_depth as usize - 1],
                interrupt.as_ref(),
            );

            let elapsed_search = (time::precise_time_ns() - start_search) as f32 / 1000000000.0;
            let elapsed_move = (time::precise_time_ns() - start_move) as f32 / 1000000000.0;

            statistics[search_depth as usize - 1][0].time = elapsed_search;

            if let Ok(eval_state) = state.execute_plies(&principal_variation) {
                if eval_state.check_resolution().is_some() {
                    break;
                }
            }

            if self.goal != 0 && elapsed_move + elapsed_search * self.branching_factor > self.goal as f32 {
                break;
            }
        }

        Analysis {
            state: state,
            evaluation: eval,
            principal_variation: principal_variation,
            stats: Some(Box::new(StatisticPrinter(statistics))),
            _phantom: PhantomData,
        }
    }
}

mod history;
mod ply_generator;
mod statistics;
mod transposition_table;
