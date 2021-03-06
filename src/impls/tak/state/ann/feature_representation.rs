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

use std::cmp;

use impls::tak::{Color, Direction, Piece, State};
use impls::tak::state::metadata::{Bitmap, BitmapInterface, EDGE};

pub fn gather_features(state: &State) -> Vec<f32> {
    let mut features = Vec::with_capacity(264);

    // 1 - Side to move
    features.push((state.ply_count % 2) as f32);

    // 1 - White reserve flatstones
    features.push(state.p1_flatstones as f32 / 21.0);

    // 1 - Black reserve flatstones
    features.push(state.p2_flatstones as f32 / 21.0);

    // 1 - Empty board spaces
    let mut stacks = state.board.iter().flat_map(|column| column.iter()).enumerate().map(|(i, stack)| (stack, (i / 5) as f32 / 4.0, (i % 5) as f32 / 4.0)).collect::<Vec<_>>();
    features.push(stacks.iter().filter(|&&(stack, _, _)| !stack.is_empty()).count() as f32 / 25.0);

    // 1 - White played capstone
    features.push(1.0 - state.p1_capstones as f32);

    // 2 - White capstone position
    let (x, y) = get_position(state.metadata.capstones & state.metadata.p1_pieces);
    features.push(x);
    features.push(y);

    // 1 - Black played capstone
    features.push(1.0 - state.p2_capstones as f32);

    // 2 - Black capstone position
    let (x, y) = get_position(state.metadata.capstones & state.metadata.p2_pieces);
    features.push(x);
    features.push(y);

    // 1 - White played standing stones
    features.push((state.metadata.standing_stones & state.metadata.p1_pieces).get_population() as f32 / 5.0);

    // 1 - Black played standing stones
    features.push((state.metadata.standing_stones & state.metadata.p2_pieces).get_population() as f32 / 5.0);

    // 1 - Largest white road group
    let p1_road_group_sizes = state.metadata.p1_road_groups.iter().map(|group| group.get_population()).collect::<Vec<_>>();
    features.push(8.0f32.min(*p1_road_group_sizes.iter().max().unwrap_or(&0) as f32) / 8.0);

    // 1 - Largest black road group
    let p2_road_group_sizes = state.metadata.p2_road_groups.iter().map(|group| group.get_population()).collect::<Vec<_>>();
    features.push(8.0f32.min(*p2_road_group_sizes.iter().max().unwrap_or(&0) as f32) / 8.0);

    // 200 - Stack positions and configurations, ordered tallest to shortest
    stacks.sort_by(|a, b| {
        b.0.len().cmp(&a.0.len())
    });
    for &(stack, x, y) in &stacks {
        features.push(x);
        features.push(y);
        for i in (cmp::max(stack.len() as i32 - 6, 0) as usize..stack.len()).rev() {
            features.push(match stack[i] {
                Piece::Flatstone(Color::White) => 1.0,
                Piece::Flatstone(Color::Black) => 0.0,
                Piece::StandingStone(Color::White) => 0.8,
                Piece::StandingStone(Color::Black) => 0.2,
                Piece::Capstone(Color::White) => 0.9,
                Piece::Capstone(Color::Black) => 0.1,
            });
        }
        for _ in 0..(6 - stack.len() as i32) {
            features.push(0.5);
        }
    }

    // 50 - Influence
    let blocks = state.metadata.standing_stones | state.metadata.capstones;
    evaluate_influence(&mut features,
        blocks,
        state.metadata.p1_pieces,
        &state.metadata.p1_flatstones,
    );
    evaluate_influence(&mut features,
        blocks,
        state.metadata.p2_pieces,
        &state.metadata.p2_flatstones,
    );

    features
}

fn get_position(mut bitmap: Bitmap) -> (f32, f32) {
    if bitmap == 0 {
        return (0.0, 0.0);
    }

    let mut x = 0;
    while bitmap & EDGE[5][Direction::West as usize] == 0 {
        bitmap <<= 1;
        x += 1;
    }

    let mut y = 0;
    while bitmap & EDGE[5][Direction::South as usize] == 0 {
        bitmap >>= 5;
        y += 1;
    }

    (x as f32 / 4.0, y as f32 / 4.0)
}

fn evaluate_influence(features: &mut Vec<f32>, blocks: Bitmap, own_pieces: Bitmap, own_stacks: &[Bitmap]) {
    fn add_bitmap(accumulator: &mut Vec<Bitmap>, mut bitmap: Bitmap) {
        if accumulator.is_empty() {
            accumulator.push(bitmap);
            return;
        }

        let len = accumulator.len();
        let carry = bitmap & accumulator[len - 1];
        if carry != 0 {
            accumulator[len - 1] ^= carry;
            accumulator.push(carry);
            bitmap ^= carry;
        }

        for level in (0..len - 1).rev() {
            let carry = bitmap & accumulator[level];
            accumulator[level] ^= carry;
            accumulator[level + 1] |= carry;
            bitmap ^= carry;
        }

        accumulator[0] |= bitmap;
    }

    let own_blocks = blocks & own_pieces;

    let mut stack_levels = Vec::new();
    add_bitmap(&mut stack_levels, own_blocks);
    for &level in own_stacks {
        add_bitmap(&mut stack_levels, level & own_pieces);
    }

    let add_influence = |influence: &mut Vec<Bitmap>, pieces: Bitmap, cast: usize| {
        use impls::tak::Direction::*;

        let mut cast_map = [pieces; 4];
        for _ in 0..cast {
            cast_map[North as usize] <<= 5;
            cast_map[East as usize] = (cast_map[East as usize] >> 1) & !EDGE[5][West as usize];
            cast_map[South as usize] >>= 5;
            cast_map[West as usize] = (cast_map[West as usize] << 1) & !EDGE[5][East as usize];

            add_bitmap(influence, cast_map[North as usize]);
            add_bitmap(influence, cast_map[East as usize]);
            add_bitmap(influence, cast_map[South as usize]);
            add_bitmap(influence, cast_map[West as usize]);

            cast_map[North as usize] &= !blocks;
            cast_map[East as usize] &= !blocks;
            cast_map[South as usize] &= !blocks;
            cast_map[West as usize] &= !blocks;
        }
    };

    let mut influence = Vec::new();
    for (cast, &level) in stack_levels.iter().enumerate() {
        add_influence(&mut influence, level, cast + 1);
    }

    let start_index = features.len();
    for _ in 0..25 {
        features.push(0.0);
    }

    for (level, map) in influence.iter().enumerate() {
        for i in 0..25 {
            if map.get(i % 5, i / 5, 5) == true {
                features[start_index + i] = (level + 1) as f32 / 10.0;
            }
        }
    }
}
