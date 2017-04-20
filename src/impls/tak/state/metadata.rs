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

use std::num::Wrapping;

use impls::tak::{Color, Piece, State};

pub type Bitmap = u64;

lazy_static! {
    pub static ref EDGE: [[Bitmap; 4]; 9] = generate_edge_masks();
    pub static ref BOARD: [Bitmap; 9] = generate_board_masks();
}

#[derive(Debug, Eq, PartialEq)]
pub struct Metadata {
    pub board_size: usize,

    // The number of flatstones on the board for each player
    pub p1_flatstone_count: u8,
    pub p2_flatstone_count: u8,

    // The maps of the flatstones at each layer of the board for each player
    pub p1_flatstones: Vec<Bitmap>,
    pub p2_flatstones: Vec<Bitmap>,

    // The maps of all standing stones and capstones on the board
    pub standing_stones: Bitmap,
    pub capstones: Bitmap,

    // The map of all top pieces for each player
    pub p1_pieces: Bitmap,
    pub p2_pieces: Bitmap,

    // The maps of each discrete island of road-contributing pieces for each player
    pub p1_road_groups: Vec<Bitmap>,
    pub p2_road_groups: Vec<Bitmap>,
}

impl Metadata {
    pub fn new(board_size: usize) -> Metadata {
        Metadata {
            board_size: board_size,
            p1_flatstone_count: 0,
            p2_flatstone_count: 0,
            p1_flatstones: Vec::new(),
            p2_flatstones: Vec::new(),
            standing_stones: 0,
            capstones: 0,
            p1_pieces: 0,
            p2_pieces: 0,
            p1_road_groups: Vec::new(),
            p2_road_groups: Vec::new(),
        }
    }

    pub fn from_state(state: &State) -> Metadata {
        let board_size = state.board.len();

        let mut metadata = Metadata::new(board_size);

        for x in 0..board_size {
            for y in 0..board_size {
                for z in 0..state.board[x][y].len() {
                    if z > 0 {
                        match state.board[x][y][z - 1] {
                            Piece::Flatstone(color) => metadata.cover_flatstone(
                                color, x, y,
                            ),
                            ref block => metadata.remove_blocking_stone(block, x, y),
                        }
                    }

                    match state.board[x][y][z] {
                        Piece::Flatstone(color) => metadata.add_flatstone(
                            color, x, y, z,
                        ),
                        ref block => metadata.add_blocking_stone(block, x, y),
                    }
                }
            }
        }

        metadata.calculate_road_groups();

        metadata
    }

    pub fn add_flatstone(&mut self, color: Color, x: usize, y: usize, z: usize) {
        match color {
            Color::White => {
                self.p1_flatstone_count += 1;

                if z >= self.p1_flatstones.len() {
                    for _ in 0..(z - self.p1_flatstones.len() + 1) {
                        self.p1_flatstones.push(0);
                    }
                }

                self.p1_flatstones[z].set(x, y, self.board_size);
                self.p1_pieces.set(x, y, self.board_size);
            },
            Color::Black => {
                self.p2_flatstone_count += 1;

                if z >= self.p2_flatstones.len() {
                    for _ in 0..(z - self.p2_flatstones.len() + 1) {
                        self.p2_flatstones.push(0);
                    }
                }

                self.p2_flatstones[z].set(x, y, self.board_size);
                self.p2_pieces.set(x, y, self.board_size);
            },
        }
    }

    pub fn remove_flatstone(&mut self, color: Color, x: usize, y: usize, z: usize) {
        match color {
            Color::White => {
                self.p1_flatstone_count -= 1;
                self.p1_flatstones[z].clear(x, y, self.board_size);
                self.p1_pieces.clear(x, y, self.board_size);
            },
            Color::Black => {
                self.p2_flatstone_count -= 1;
                self.p2_flatstones[z].clear(x, y, self.board_size);
                self.p2_pieces.clear(x, y, self.board_size);
            },
        }
    }

    pub fn reveal_flatstone(&mut self, color: Color, x: usize, y: usize) {
        match color {
            Color::White => {
                self.p1_flatstone_count += 1;
                self.p1_pieces.set(x, y, self.board_size);
            },
            Color::Black => {
                self.p2_flatstone_count += 1;
                self.p2_pieces.set(x, y, self.board_size);
            },
        }
    }

    pub fn cover_flatstone(&mut self, color: Color, x: usize, y: usize) {
        match color {
            Color::White => {
                self.p1_flatstone_count -= 1;
                self.p1_pieces.clear(x, y, self.board_size);
            },
            Color::Black => {
                self.p2_flatstone_count -= 1;
                self.p2_pieces.clear(x, y, self.board_size);
            },
        }
    }

    pub fn add_blocking_stone(&mut self, piece: &Piece, x: usize, y: usize) {
        match *piece {
            Piece::StandingStone(color) => if color == Color::White {
                self.standing_stones.set(x, y, self.board_size);
                self.p1_pieces.set(x, y, self.board_size);
            } else {
                self.standing_stones.set(x, y, self.board_size);
                self.p2_pieces.set(x, y, self.board_size);
            },
            Piece::Capstone(color) => if color == Color::White {
                self.capstones.set(x, y, self.board_size);
                self.p1_pieces.set(x, y, self.board_size);
            } else {
                self.capstones.set(x, y, self.board_size);
                self.p2_pieces.set(x, y, self.board_size);
            },
            _ => panic!("StateAnalysis.add_blocking_stone was passed a flatstone!"),
        }
    }

    pub fn remove_blocking_stone(&mut self, piece: &Piece, x: usize, y: usize) {
        match *piece {
            Piece::StandingStone(color) => if color == Color::White {
                self.standing_stones.clear(x, y, self.board_size);
                self.p1_pieces.clear(x, y, self.board_size);
            } else {
                self.standing_stones.clear(x, y, self.board_size);
                self.p2_pieces.clear(x, y, self.board_size);
            },
            Piece::Capstone(color) => if color == Color::White {
                self.capstones.clear(x, y, self.board_size);
                self.p1_pieces.clear(x, y, self.board_size);
            } else {
                self.capstones.clear(x, y, self.board_size);
                self.p2_pieces.clear(x, y, self.board_size);
            },
            _ => panic!("StateAnalysis.remove_blocking_stone was passed a flatstone!"),
        }
    }

    pub fn calculate_road_groups(&mut self) {
        self.p1_road_groups = (self.p1_pieces & !self.standing_stones).get_groups(self.board_size);
        self.p2_road_groups = (self.p2_pieces & !self.standing_stones).get_groups(self.board_size);
    }
}

impl Clone for Metadata {
    fn clone(&self) -> Metadata {
        Metadata {
            board_size: self.board_size,
            p1_flatstone_count: self.p1_flatstone_count,
            p2_flatstone_count: self.p2_flatstone_count,
            p1_flatstones: self.p1_flatstones.clone(),
            p2_flatstones: self.p2_flatstones.clone(),
            standing_stones: self.standing_stones,
            capstones: self.capstones,
            p1_pieces: self.p1_pieces,
            p2_pieces: self.p2_pieces,
            p1_road_groups: self.p1_road_groups.clone(),
            p2_road_groups: self.p2_road_groups.clone(),
        }
    }

    fn clone_from(&mut self, source: &Metadata) {
        self.board_size = source.board_size;
        self.p1_flatstone_count = source.p1_flatstone_count;
        self.p2_flatstone_count = source.p2_flatstone_count;
        self.p1_flatstones.clone_from(&source.p1_flatstones);
        self.p2_flatstones.clone_from(&source.p2_flatstones);
        self.standing_stones = source.standing_stones;
        self.capstones = source.capstones;
        self.p1_pieces = source.p1_pieces;
        self.p2_pieces = source.p2_pieces;
        self.p1_road_groups.clone_from(&source.p1_road_groups);
        self.p2_road_groups.clone_from(&source.p2_road_groups);
    }
}

pub trait BitmapInterface {
    fn set(&mut self, x: usize, y: usize, stride: usize);
    fn clear(&mut self, x: usize, y: usize, stride: usize);
    fn get(&self, x: usize, y: usize, stride: usize) -> bool;
    fn get_groups(&self, stride: usize) -> Vec<Bitmap>;
    fn get_population(&self) -> u8;
    fn get_dimensions(&self, stride: usize) -> (usize, usize);
    fn grow(&self, bounds: Bitmap, stride: usize) -> Bitmap;
}

impl BitmapInterface for Bitmap {
    fn set(&mut self, x: usize, y: usize, stride: usize) {
        *self |= 1 << ((stride - 1 - x) + y * stride);
    }

    fn clear(&mut self, x: usize, y: usize, stride: usize) {
        *self &= !(1 << ((stride - 1 - x) + y * stride));
    }

    fn get(&self, x: usize, y: usize, stride: usize) -> bool {
        (self >> ((stride - 1 - x) + y * stride)) & 1 == 1
    }

    // Returns a map of each discrete group
    fn get_groups(&self, stride: usize) -> Vec<Bitmap> {
        fn pop_bit(map: Bitmap) -> (Bitmap, Bitmap) {
            let remainder = map & (map - 1);
            let bit = map & !remainder;
            (bit, remainder)
        }

        fn flood(bit: Bitmap, bounds: Bitmap, stride: usize) -> Bitmap {
            let mut total = bit;

	        loop {
	            let next = total.grow(bounds, stride);

	            if next == total {
	                break;
                }

                total = next;
            }

            total
        }

        if *self == 0 {
            return Vec::new();
        }

        let mut groups = Vec::new();
        let mut map = *self;

        loop {
            let (bit, mut remainder) = pop_bit(map);

            let group = flood(bit, map, stride);

            groups.push(group);

            remainder &= !group;
            if remainder == 0 {
                break;
            }

            map = remainder;
        }

        groups
    }

    fn get_population(&self) -> u8 { // XXX use count_ones if we can target a native cpu
        // Utter magic.
        // https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

        let mut x = self - ((self >> 1) as u64 & 0x555555555555555);
        x = ((x >> 2) & 0x3333333333333333) + (x & 0x3333333333333333);
        x += x >> 4;
        x &= 0x0F0F0F0F0F0F0F0F;
        let overflow_x = Wrapping(x) * Wrapping(0x0101010101010101); // This can overflow, hence Wrapping
        let Wrapping(population) = overflow_x >> 56;
        population as u8
    }

    fn get_dimensions(&self, stride: usize) -> (usize, usize) {
        use impls::tak::Direction::*;

        if *self != 0 {
            let width = {
                let mut mask = EDGE[stride][West as usize];
                while *self & mask == 0 {
                    mask >>= 1;
                }

                let mut width = 0;
                while mask != 0 && *self & mask != 0 && width < stride {
                    mask >>= 1;
                    width += 1;
                }

                width
            };

            let height = {
                let mut mask = EDGE[stride][North as usize];
                while *self & mask == 0 {
                    mask >>= stride;
                }

                let mut height = 0;
                while mask != 0 && *self & mask != 0 {
                    mask >>= stride;
                    height += 1;
                }

                height
            };

            (width, height)
        } else {
            (0, 0)
        }
    }

    fn grow(&self, bounds: Bitmap, stride: usize) -> Bitmap {
        use impls::tak::Direction::*;

        let mut next = *self;
        next |= (*self << 1) & !EDGE[stride][East as usize];
        next |= (*self >> 1) & !EDGE[stride][West as usize];
        next |= *self << stride;
        next |= *self >> stride;
        next & bounds
    }
}

fn generate_edge_masks() -> [[Bitmap; 4]; 9] {
    use impls::tak::Direction::*;

    let mut edge = [[0; 4]; 9];

    for size in 3..(8 + 1) {
        for y in 0..size {
            edge[size][East as usize] |= 1 << (y * size);
        }
        edge[size][West as usize] = edge[size][East as usize] << (size - 1);

        edge[size][South as usize] = (1 << size) - 1;
        edge[size][North as usize] = edge[size][South as usize] << (size * (size - 1));
    }

    edge
}

fn generate_board_masks() -> [Bitmap; 9] {
    let mut board = [0; 9];

    for size in 3..(8 + 1) {
        board[size] = 0xFFFFFFFFFFFFFFFF >> (64 - size * size);
    }

    board
}
