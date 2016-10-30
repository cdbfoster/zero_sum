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

use std::cmp;
use std::i32;

use analysis::{self, Evaluation as EvaluationTrait};
use impls::tak::{Color};
use impls::tak::resolution::Resolution;
use impls::tak::state::State;
use impls::tak::state::metadata::{Bitmap, BitmapInterface, BOARD, EDGE, Metadata};
use state::State as StateTrait;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Evaluation(i32);

prepare_evaluation_tuple!(Evaluation);

impl analysis::Evaluation for Evaluation {
    fn null() -> Evaluation { Evaluation(0) }
    fn epsilon() -> Evaluation { Evaluation(1) }
    fn win() -> Evaluation { Evaluation(100_000) }
    fn max() -> Evaluation { Evaluation(i32::MAX) }
    fn is_win(&self) -> bool { self.0.abs() >= 99_000 }
}

const END_GAME_FLATSTONE_THRESHOLD: [i32; 9] = [0, 0, 0, 5, 8, 10, 15, 20, 25];

struct Weights {
    flatstone: (i32, i32),
    standing_stone: i32,
    capstone: i32,

    hard_flat: (i32, i32, i32),
    soft_flat: (i32, i32, i32),

    threat: i32,

    influence: (i32, i32, i32),

    group: [i32; 8],
}

const WEIGHT: Weights = Weights {
    flatstone:         (400, 800),
    standing_stone:     200,
    capstone:           300,

    hard_flat:         (125, 125, 150),
    soft_flat:         (-75, -50, -25),

    threat:             200,

    influence:         (20, 15, -5),

    group: [0, 0, 100, 200, 400, 600, 0, 0],
};

impl analysis::Evaluatable<Evaluation> for State {
    fn evaluate(&self) -> Evaluation {
        let next_color = if self.ply_count % 2 == 0 {
            Color::White
        } else {
            Color::Black
        };

        match self.check_resolution() {
            None => (),
            Some(Resolution::Road(win_color)) |
            Some(Resolution::Flat(win_color)) => {
                if win_color == next_color {
                    return Evaluation::max() - Evaluation(self.ply_count as i32);
                } else {
                    return -Evaluation::max() + Evaluation(self.ply_count as i32);
                }
            },
            Some(Resolution::Draw) => return Evaluation::null(),
        }

        let mut p1_eval = 0;
        let mut p2_eval = 0;

        let m = &self.metadata;

        let total_pieces = m.p1_pieces | m.p2_pieces;

        let p1_flatstones = m.p1_pieces & !m.standing_stones & !m.capstones;
        let p2_flatstones = m.p2_pieces & !m.standing_stones & !m.capstones;

        let p1_standing_stones = m.p1_pieces & m.standing_stones;
        let p2_standing_stones = m.p2_pieces & m.standing_stones;

        let p1_capstones = m.p1_pieces & m.capstones;
        let p2_capstones = m.p2_pieces & m.capstones;

        let (p1_flatstone_weight, p2_flatstone_weight) = {
            let flatstone_threshold = END_GAME_FLATSTONE_THRESHOLD[m.board_size];

            let p1_position = cmp::min(self.p1_flatstones as i32, flatstone_threshold);
            let p2_position = cmp::min(self.p2_flatstones as i32, flatstone_threshold);

            (
                WEIGHT.flatstone.0 * p1_position / flatstone_threshold +
                WEIGHT.flatstone.1 * (flatstone_threshold - p1_position) / flatstone_threshold,
                WEIGHT.flatstone.0 * p2_position / flatstone_threshold +
                WEIGHT.flatstone.1 * (flatstone_threshold - p2_position) / flatstone_threshold,
            )
        };

        // Top-level pieces
        p1_eval += evaluate_top_pieces(m.p1_flatstone_count as i32, p1_flatstone_weight, p1_standing_stones, p1_capstones);
        p2_eval += evaluate_top_pieces(m.p2_flatstone_count as i32, p2_flatstone_weight, p2_standing_stones, p2_capstones);

        // Stacked flatstones
        {
            let stacked_flatstones_eval = evaluate_stacked_flatstones(
                m,
                p1_flatstones,
                p2_flatstones,
                p1_standing_stones,
                p2_standing_stones,
                p1_capstones,
                p2_capstones,
            );
            p1_eval += stacked_flatstones_eval.0;
            p2_eval += stacked_flatstones_eval.1;
        }

        // Road groups
        p1_eval += evaluate_road_groups(m, &m.p1_road_groups);
        p2_eval += evaluate_road_groups(m, &m.p2_road_groups);

        // Threats
        p1_eval += evaluate_threats(m, total_pieces, &m.p1_road_groups);
        p2_eval += evaluate_threats(m, total_pieces, &m.p2_road_groups);

        // Influence
        p1_eval += evaluate_influence(
            m,
            total_pieces,
            m.p1_pieces,
            &m.p1_flatstones,
            p1_flatstones,
            m.p2_pieces,
        );
        p2_eval += evaluate_influence(
            m,
            total_pieces,
            m.p2_pieces,
            &m.p2_flatstones,
            p2_flatstones,
            m.p1_pieces,
        );

        match next_color {
            Color::White => Evaluation(p1_eval - p2_eval),
            Color::Black => Evaluation(p2_eval - p1_eval),
        }
    }
}

fn evaluate_top_pieces(flatstone_count: i32, flatstone_weight: i32, standing_stones: Bitmap, capstones: Bitmap) -> i32 {
    flatstone_count as i32 * flatstone_weight +
    standing_stones.get_population() as i32 * WEIGHT.standing_stone +
    capstones.get_population() as i32 * WEIGHT.capstone
}

fn evaluate_stacked_flatstones(
    m: &Metadata,
    p1_flatstones: Bitmap,
    p2_flatstones: Bitmap,
    p1_standing_stones: Bitmap,
    p2_standing_stones: Bitmap,
    p1_capstones: Bitmap,
    p2_capstones: Bitmap,
) -> (i32, i32) {
    let mut p1_flatstone_hard_flats = -(m.p1_flatstone_count as i32); // Top-level flatstones don't count
    let mut p1_flatstone_soft_flats = 0;

    let mut p1_standing_stone_hard_flats = 0;
    let mut p1_standing_stone_soft_flats = 0;

    let mut p1_capstone_hard_flats = 0;
    let mut p1_capstone_soft_flats = 0;

    for level in &m.p1_flatstones {
        if *level != 0 {
            p1_flatstone_hard_flats += (level & p1_flatstones).get_population() as i32;
            p1_flatstone_soft_flats += (level & p2_flatstones).get_population() as i32;

            p1_standing_stone_hard_flats += (level & p1_standing_stones).get_population() as i32;
            p1_standing_stone_soft_flats += (level & p2_standing_stones).get_population() as i32;

            p1_capstone_hard_flats += (level & p1_capstones).get_population() as i32;
            p1_capstone_soft_flats += (level & p2_capstones).get_population() as i32;
        }
    }

    let mut p2_flatstone_hard_flats = -(m.p2_flatstone_count as i32);
    let mut p2_flatstone_soft_flats = 0;

    let mut p2_standing_stone_hard_flats = 0;
    let mut p2_standing_stone_soft_flats = 0;

    let mut p2_capstone_hard_flats = 0;
    let mut p2_capstone_soft_flats = 0;

    for level in &m.p2_flatstones {
        if *level != 0 {
            p2_flatstone_hard_flats += (level & p2_flatstones).get_population() as i32;
            p2_flatstone_soft_flats += (level & p1_flatstones).get_population() as i32;

            p2_standing_stone_hard_flats += (level & p2_standing_stones).get_population() as i32;
            p2_standing_stone_soft_flats += (level & p1_standing_stones).get_population() as i32;

            p2_capstone_hard_flats += (level & p2_capstones).get_population() as i32;
            p2_capstone_soft_flats += (level & p1_capstones).get_population() as i32;
        }
    }

    let mut p1_eval = 0;
    p1_eval += p1_flatstone_hard_flats * WEIGHT.hard_flat.0 + p2_flatstone_soft_flats * WEIGHT.soft_flat.0;
    p1_eval += p1_standing_stone_hard_flats * WEIGHT.hard_flat.1 + p2_standing_stone_soft_flats * WEIGHT.soft_flat.1;
    p1_eval += p1_capstone_hard_flats * WEIGHT.hard_flat.2 + p2_capstone_soft_flats * WEIGHT.soft_flat.2;

    let mut p2_eval = 0;
    p2_eval += p2_flatstone_hard_flats * WEIGHT.hard_flat.0 + p1_flatstone_soft_flats * WEIGHT.soft_flat.0;
    p2_eval += p2_standing_stone_hard_flats * WEIGHT.hard_flat.1 + p1_standing_stone_soft_flats * WEIGHT.soft_flat.1;
    p2_eval += p2_capstone_hard_flats * WEIGHT.hard_flat.2 + p1_capstone_soft_flats * WEIGHT.soft_flat.2;

    (p1_eval, p2_eval)
}

fn evaluate_road_groups(m: &Metadata, groups: &[Bitmap]) -> i32 {
    let mut eval = 0;

    for group in groups {
        let (width, height) = group.get_dimensions(m.board_size);

        eval += WEIGHT.group[width] + WEIGHT.group[height];
    }

    eval
}

fn evaluate_threats(m: &Metadata, total_pieces: Bitmap, groups: &[Bitmap]) -> i32 {
    let mut expanded_groups = vec![0; groups.len()];
    let mut threats = 0;

    let is_road = |group: Bitmap| {
        use impls::tak::Direction::*;

        if (group & EDGE[m.board_size][North as usize] != 0 &&
            group & EDGE[m.board_size][South as usize] != 0) ||
           (group & EDGE[m.board_size][West as usize] != 0 &&
            group & EDGE[m.board_size][East as usize] != 0) {
            return true;
        }

        false
    };

    for i in 0..groups.len() {
        expanded_groups[i] = groups[i].grow(BOARD[m.board_size], m.board_size);
    }

    for l in 0..groups.len() {
        for r in l..groups.len() {
            if l != r {
                let overlap = expanded_groups[l] & expanded_groups[r] & !total_pieces;

                if overlap == 0 {
                    continue;
                }

                if is_road(groups[l] | groups[r] | overlap) {
                    threats += 1;
                }
            }
        }
    }

    threats * WEIGHT.threat
}

fn evaluate_influence(
    m: &Metadata,
    total_pieces: Bitmap,
    own_pieces: Bitmap,
    own_stacks: &[Bitmap],
    own_flatstones: Bitmap,
    enemy_pieces: Bitmap,
) -> i32 {
    fn add_bitmap(mut bitmap: Bitmap, accumulator: &mut Vec<Bitmap>) {
        let mut bit = 0;
        if accumulator.is_empty() {
            accumulator.push(0);
        }

        while bitmap != 0 {
            let carry = accumulator[bit] & bitmap;
            accumulator[bit] ^= bitmap;
            bitmap = carry;
            bit += 1;
            if accumulator.len() <= bit {
                accumulator.push(0);
            }
        }
    }

    fn convert_accumulator(accumulator: &[Bitmap]) -> Vec<Bitmap> {
        let mut result = vec![0; (1 << accumulator.len()) - 1];

        for (bit, bitmap) in accumulator.iter().enumerate() {
            let mut bitmap = *bitmap;

            for level in 1..1 << bit {
                let advance = result[level - 1] & bitmap;
                bitmap &= !advance;
                result[level - 1] &= !advance;
                result[level + (1 << bit) - 1] |= advance;
            }

            result[(1 << bit) - 1] |= bitmap;
        }

        result
    }

    let blocks = total_pieces & (m.standing_stones | m.capstones);
    let own_blocks = blocks & own_pieces;

    let stack_levels = {
        let mut stack_counts = Vec::with_capacity(3);
        add_bitmap(own_blocks, &mut stack_counts);
        for level in own_stacks {
            add_bitmap(*level & own_pieces, &mut stack_counts);
        }
        convert_accumulator(&stack_counts)
    };

    let add_influence = |pieces: Bitmap, cast: usize, influence: &mut Vec<Bitmap>| {
        use impls::tak::Direction::*;

        let mut cast_map = [pieces; 4];
        for _ in 0..cast {
            cast_map[North as usize] <<= m.board_size;
            cast_map[East as usize] = (cast_map[East as usize] >> 1) & !EDGE[m.board_size][West as usize];
            cast_map[South as usize] >>= m.board_size;
            cast_map[West as usize] = (cast_map[West as usize] << 1) & !EDGE[m.board_size][East as usize];

            add_bitmap(cast_map[North as usize], influence);
            add_bitmap(cast_map[East as usize], influence);
            add_bitmap(cast_map[South as usize], influence);
            add_bitmap(cast_map[West as usize], influence);

            cast_map[North as usize] &= !blocks;
            cast_map[East as usize] &= !blocks;
            cast_map[South as usize] &= !blocks;
            cast_map[West as usize] &= !blocks;
        }
    };

    let influence = {
        let mut influence_accumulator = Vec::with_capacity(4);
        for (cast, level) in stack_levels.iter().enumerate() {
            add_influence(*level, cast + 1, &mut influence_accumulator);
        }
        convert_accumulator(&influence_accumulator)
    };

    {
        let mut eval = 0;
        for (level, map) in influence.iter().enumerate() {
            eval += (map & own_flatstones).get_population() as i32 * (WEIGHT.influence.0 * (level as i32 + 1));
            eval += (map & !total_pieces).get_population() as i32 * (WEIGHT.influence.1 * (level as i32 + 1));
            eval += (map & enemy_pieces).get_population() as i32 * (WEIGHT.influence.2 >> level);
        }
        eval
    }
}
