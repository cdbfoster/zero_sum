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

use std::f32;
use std::marker::PhantomData;
use std::slice;
use std::thread;

use rand::Rng;

use analysis::Extrapolatable;
use analysis::search::mctsearch::{Selector, TreeNode};
use analysis::search::mctsearch::jkiss32::JKiss32Rng;
use state::State;

static EXPLORATION_BIAS: f32 = 0.7;

#[derive(Clone)]
pub struct PuctSelector<S> where
    S: State + Extrapolatable<<S as State>::Ply> {
    rng: JKiss32Rng,
    best_children_indices: Vec<usize>,
    _marker: PhantomData<S>,
}

#[derive(Clone, Default)]
pub struct PuctSelectorData;

impl<S> PuctSelector<S> where
    S: State + Extrapolatable<<S as State>::Ply> {
    pub fn new() -> PuctSelector<S> {
        PuctSelector {
            rng: JKiss32Rng::new(),
            best_children_indices: Vec::with_capacity(100),
            _marker: PhantomData,
        }
    }
}

impl<S> Selector<S> for PuctSelector<S> where
    S: Send + State + Extrapolatable<<S as State>::Ply> {
    type Data = PuctSelectorData;

    fn initialize_policy_data(&self) -> Self::Data {
        PuctSelectorData
    }

    fn select_descendant_index(&mut self, tree_node: &TreeNode<S, Self>) -> Option<usize> where Self: Sized {
        let children_buffer = tree_node.children.read();
        if children_buffer.is_null() {
            return None;
        }

        let tree_exploration_value = EXPLORATION_BIAS * (tree_node.get_visits() as f32).sqrt();

        let mut best_score = f32::MIN;

        let children = unsafe { slice::from_raw_parts(children_buffer, tree_node.child_count.read() as usize) };
        for (index, &(_, ref child)) in children.iter().enumerate() {
            if child.is_invalid() {
                continue;
            }

            let score = {
                let value = child.get_value();
                let virtual_loss = child.get_virtual_loss() as f32;
                let visits = child.get_visits() as f32 + virtual_loss;

                value
                    - virtual_loss * (1.0 - value) / visits.max(1.0) // Virtual loss
                    + tree_exploration_value / (1.0 + visits)  // Exploration incentive
            };

            if score > best_score {
                best_score = score;
                self.best_children_indices.clear();
                self.best_children_indices.push(index);
            } else if score == best_score {
                self.best_children_indices.push(index);
            }
        }

        //println!("{}: Selecting a child from {} best candidates. Score: {:.3}", thread::current().name().unwrap(), self.best_children_indices.len(), best_score);

        self.rng.choose(&self.best_children_indices).map(|i| *i)
    }

    fn get_current_pv(&mut self, tree_node: &TreeNode<S, Self>) -> Vec<(<S as State>::Ply, TreeNode<S, Self>)> where Self: Sized {
        fn append_pv<S>(rng: &mut JKiss32Rng, pv: &mut Vec<(<S as State>::Ply, TreeNode<S, PuctSelector<S>>)>, tree_node: &TreeNode<S, PuctSelector<S>>) where S: Send + State + Extrapolatable<<S as State>::Ply> {
            let children_buffer = tree_node.children.read();
            if children_buffer.is_null() {
                return;
            }

            let children = unsafe { slice::from_raw_parts(children_buffer, tree_node.child_count.read() as usize) };
            let max_visits = children.iter().map(|c| c.1.get_visits()).max().unwrap();
            if let Some(&most_visited) = rng.choose(&children.iter().filter(|c| c.1.get_visits() == max_visits).collect::<Vec<_>>()) {
                pv.push(most_visited.clone());
                append_pv(rng, pv, &most_visited.1);
            }
        };
        let mut pv = Vec::new();
        append_pv(&mut self.rng, &mut pv, tree_node);
        pv
    }

    fn split(&self) -> Self {
        PuctSelector {
            rng: JKiss32Rng::new(),
            best_children_indices: Vec::with_capacity(100),
            _marker: PhantomData,
        }
    }
}
