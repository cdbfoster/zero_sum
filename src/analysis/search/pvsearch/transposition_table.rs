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

use std::collections::HashMap;
use std::collections::hash_map::IterMut;
use std::hash::BuildHasherDefault;
use std::marker::PhantomData;

use fnv::FnvHasher;

use analysis::Evaluation;
use ply::Ply;
use resolution::Resolution;
use state::State;

#[derive(PartialEq)]
pub enum Bound {
    Lower,
    Exact,
    Upper,
}

pub struct TranspositionTableEntry<E, P> where
    E: Evaluation,
    P: Ply {
    pub depth: u8,
    pub value: E,
    pub bound: Bound,
    pub principal_variation: Vec<P>,
    pub lifetime: u8,
}

pub struct TranspositionTable<E, S, P, R> where
    E: Evaluation,
    S: State<P, R>,
    P: Ply,
    R: Resolution {
    map: HashMap<S, TranspositionTableEntry<E, P>, BuildHasherDefault<FnvHasher>>,
    _phantom: PhantomData<R>,
}

impl<E, S, P, R> TranspositionTable<E, S, P, R> where
    E: Evaluation,
    S: State<P, R>,
    P: Ply,
    R: Resolution {
    pub fn new() -> TranspositionTable<E, S, P, R> {
        TranspositionTable {
            map: HashMap::default(),
            _phantom: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn get(&self, state: &S) -> Option<&TranspositionTableEntry<E, P>> {
        self.map.get(state)
    }

    pub fn insert(&mut self, state: S, entry: TranspositionTableEntry<E, P>) -> Option<TranspositionTableEntry<E, P>> {
        self.map.insert(state, entry)
    }

    pub fn remove(&mut self, state: &S) -> Option<TranspositionTableEntry<E, P>> {
        self.map.remove(state)
    }

    pub fn iter_mut(&mut self) -> IterMut<S, TranspositionTableEntry<E, P>> {
        self.map.iter_mut()
    }
}
