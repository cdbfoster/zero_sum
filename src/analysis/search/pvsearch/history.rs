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

    fn hash<P>(&self, ply: &P) -> u64 where P: Ply {
        let mut hasher = self.hasher.borrow_mut();
        ply.hash(&mut *hasher);
        hasher.finish()
    }
}

#[cfg(all(test, feature = "with_tak"))]
mod test_tak {
    use test::{self, Bencher};

    use super::*;
    use analysis::search::{Search, PvSearch};
    use impls::tak::*;

    #[bench]
    fn bench_history_empty_add(b: &mut Bencher) {
        let mut history = History::new();
        let ply = Ply::from_ptn("a1", Color::Black).unwrap();

        b.iter(|| {
            history.entry(test::black_box(&ply)).or_insert(0);
        });
    }

    #[bench]
    #[ignore]
    fn bench_history_full_add(b: &mut Bencher) {
        let mut search = PvSearch::<Evaluation, State, Ply, Resolution>::with_depth(5);
        let state = State::new(5);
        search.search(&state, None);

        let mut history = search.history.borrow_mut();
        let ply = Ply::from_ptn("a1", Color::Black).unwrap();

        b.iter(|| {
            history.entry(test::black_box(&ply)).or_insert(0);
        });
    }

    #[bench]
    fn bench_history_hash(b: &mut Bencher) {
        let history = History::new();
        let ply = Ply::from_ptn("2a2>11", Color::White).unwrap();

        b.iter(|| {
            history.hash(test::black_box(&ply))
        });
    }
}
