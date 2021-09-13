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

use rand::{self, Rng, RngCore};

#[derive(Clone, Copy)]
pub struct JKiss32Rng {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
    c: bool,
}

impl JKiss32Rng {
    pub fn new() -> JKiss32Rng {
        let mut rng = rand::thread_rng();
        JKiss32Rng {
            x: rng.gen(),
            y: rng.gen(),
            z: rng.gen(),
            w: rng.gen(),
            c: false,
        }
    }
}

impl RngCore for JKiss32Rng {
    fn next_u32(&mut self) -> u32 {
        self.y ^= self.y << 5;
        self.y ^= self.y >> 7;
        self.y ^= self.y << 22;
        let t = self.z.wrapping_add(self.w).wrapping_add(self.c as u32) as i32;
        self.z = self.w;
        self.c = t < 0;
        self.w = (t & 0x7FFFFFFF) as u32;
        self.x = self.x.wrapping_add(1411392427);
        self.x.wrapping_add(self.y).wrapping_add(self.w)
    }

    fn next_u64(&mut self) -> u64 {
        rand_core::impls::next_u64_via_u32(self)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        rand_core::impls::fill_bytes_via_next(self, dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        Ok(rand_core::impls::fill_bytes_via_next(self, dest))
    }
}
