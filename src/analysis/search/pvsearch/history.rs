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
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::hash::Hasher;

use ply::Ply;

struct SingleHasher(u64);

impl Hasher for SingleHasher {
    fn write(&mut self, _: &[u8]) { }

    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

pub struct History {
    hasher: RefCell<SingleHasher>,
    map: BTreeMap<u64, u32>,
}

impl History {
    pub fn new() -> History {
        History {
            hasher: RefCell::new(SingleHasher(0)),
            map: BTreeMap::new(),
        }
    }

    pub fn get<P>(&self, ply: &P) -> Option<&u32> where P: Ply {
        let hash = self.hash(ply);
        self.map.get(&hash)
    }

    pub fn entry<P>(&mut self, ply: &P) -> Entry<u64, u32> where P: Ply {
        let hash = self.hash(ply);
        self.map.entry(hash)
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn sort_plies<P>(&self, plies: &mut [P]) where P: Ply {
        if self.is_empty() {
            return;
        }

        let mut histories = Vec::with_capacity(plies.len());
        for (index, ply) in plies.iter().enumerate() {
            histories.push((*self.get(ply).unwrap_or(&0), index));
        }

        histories.sort_by(|a, b| {
            a.0.cmp(&b.0)
        });

        let mut final_indices = vec![0; plies.len()];
        for (index, starting_index) in histories.iter().enumerate() {
            final_indices[starting_index.1] = index;
        }

        apply_permutation(plies, &final_indices);
    }

    fn hash<P>(&self, ply: &P) -> u64 where P: Ply {
        let mut hasher = self.hasher.borrow_mut();
        ply.hash(&mut *hasher);
        hasher.finish()
    }
}

fn apply_permutation<T>(v: &mut [T], p: &[usize]) {
    let mut done = vec![false; v.len()];

    for i in 0..v.len() {
        if done[i] {
            continue;
        }

        done[i] = true;

        let mut j = p[i];
        while i != j {
            v.swap(i, j);
            done[j] = true;
            j = p[j];
        }
    }
}
