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

use std::collections::HashMap;
use std::collections::hash_map::IterMut;
use std::hash::BuildHasherDefault;

use fnv::FnvHasher;

use analysis::Evaluation;
use ply::Ply;
use state::State;

#[derive(PartialEq)]
pub enum Bound {
    Lower,
    Exact,
    Upper,
}

pub struct TranspositionTableEntry<P, E> where
    P: Ply,
    E: Evaluation {
    pub node_budget: u64,
    pub value: E,
    pub bound: Bound,
    pub principal_variation: Vec<P>,
    pub lifetime: u8,
}

pub struct TranspositionTable<S, E> where
    S: State,
    E: Evaluation {
    map: HashMap<S, TranspositionTableEntry<<S as State>::Ply, E>, BuildHasherDefault<FnvHasher>>,
}

impl<S, E> TranspositionTable<S, E> where
    S: State,
    E: Evaluation {
    pub fn new() -> TranspositionTable<S, E> {
        TranspositionTable {
            map: HashMap::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn get(&self, state: &S) -> Option<&TranspositionTableEntry<<S as State>::Ply, E>> {
        self.map.get(state)
    }

    pub fn insert(&mut self, state: S, entry: TranspositionTableEntry<<S as State>::Ply, E>) -> Option<TranspositionTableEntry<<S as State>::Ply, E>> {
        self.map.insert(state, entry)
    }

    pub fn remove(&mut self, state: &S) -> Option<TranspositionTableEntry<<S as State>::Ply, E>> {
        self.map.remove(state)
    }

    pub fn iter_mut(&mut self) -> IterMut<S, TranspositionTableEntry<<S as State>::Ply, E>> {
        self.map.iter_mut()
    }
}
