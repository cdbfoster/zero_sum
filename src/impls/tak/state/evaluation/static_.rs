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
use std::i32;

use analysis::{self, Evaluation as EvaluationTrait};
use impls::tak::{Color};
use impls::tak::resolution::Resolution;
use impls::tak::state::State;
use impls::tak::state::metadata::{Bitmap, BitmapInterface, BOARD, EDGE, Metadata};
use state::State as StateTrait;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Evaluation(pub i32);

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

    influence:         ( 20,  15,  -5),

    group: [0, 0, 100, 200, 400, 600, 0, 0],
};

/// Provides a static evaluation of a tak state.  This evaluator considers
/// top-level pieces, stacked flatstones, road-group size, one-away threats,
/// and stack influence.
pub struct StaticEvaluator;

impl analysis::Evaluator for StaticEvaluator {
    type State = State;
    type Evaluation = Evaluation;

    fn evaluate(&self, state: &State) -> Evaluation {
        let next_color = if state.ply_count % 2 == 0 {
            Color::White
        } else {
            Color::Black
        };

        match state.check_resolution() {
            None => (),
            Some(Resolution::Road(win_color)) |
            Some(Resolution::Flat(win_color)) => {
                if win_color == next_color {
                    return Evaluation::win() - Evaluation(state.ply_count as i32);
                } else {
                    return -Evaluation::win() + Evaluation(state.ply_count as i32);
                }
            },
            Some(Resolution::Draw) => return Evaluation::null(),
        }

        let mut p1_eval = 0;
        let mut p2_eval = 0;

        let m = &state.metadata;

        let total_pieces = m.p1_pieces | m.p2_pieces;

        let p1_flatstones = m.p1_pieces & !m.standing_stones & !m.capstones;
        let p2_flatstones = m.p2_pieces & !m.standing_stones & !m.capstones;

        let p1_standing_stones = m.p1_pieces & m.standing_stones;
        let p2_standing_stones = m.p2_pieces & m.standing_stones;

        let p1_capstones = m.p1_pieces & m.capstones;
        let p2_capstones = m.p2_pieces & m.capstones;

        let (p1_flatstone_weight, p2_flatstone_weight) = {
            let flatstone_threshold = END_GAME_FLATSTONE_THRESHOLD[m.board_size];

            let p1_position = cmp::min(state.p1_flatstones as i32, flatstone_threshold);
            let p2_position = cmp::min(state.p2_flatstones as i32, flatstone_threshold);

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

    let blocks = m.standing_stones | m.capstones;
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
            cast_map[North as usize] <<= m.board_size;
            cast_map[East as usize] = (cast_map[East as usize] >> 1) & !EDGE[m.board_size][West as usize];
            cast_map[South as usize] >>= m.board_size;
            cast_map[West as usize] = (cast_map[West as usize] << 1) & !EDGE[m.board_size][East as usize];

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

#[cfg(test)]
mod test {
    use std::cmp;
    use test::{self, Bencher};

    use analysis::Evaluator;
    use impls::tak::*;
    use super::{
        END_GAME_FLATSTONE_THRESHOLD,
        evaluate_influence,
        evaluate_road_groups,
        evaluate_stacked_flatstones,
        evaluate_threats,
        evaluate_top_pieces,
        WEIGHT,
    };

    lazy_static! {
        static ref STATE: State = State::from_tps("[TPS \"21,22221C,1,12212S,x/2121,2S,2,1S,2/x2,2,2,x/1,2111112C,2,x,21/x,1,21,x2 1 32\"]").unwrap();
    }

    #[bench]
    fn bench_evaluate(b: &mut Bencher) {
        b.iter(|| {
            let evaluator = evaluator::StaticEvaluator;
            evaluator.evaluate(test::black_box(&STATE))
        });
    }

    #[bench]
    fn bench_evaluate_top_pieces(b: &mut Bencher) {
        let mut p1_eval = 0;
        let mut p2_eval = 0;

        let m = &STATE.metadata;

        let p1_standing_stones = m.p1_pieces & m.standing_stones;
        let p2_standing_stones = m.p2_pieces & m.standing_stones;

        let p1_capstones = m.p1_pieces & m.capstones;
        let p2_capstones = m.p2_pieces & m.capstones;

        let (p1_flatstone_weight, p2_flatstone_weight) = {
            let flatstone_threshold = END_GAME_FLATSTONE_THRESHOLD[m.board_size];

            let p1_position = cmp::min(STATE.p1_flatstones as i32, flatstone_threshold);
            let p2_position = cmp::min(STATE.p2_flatstones as i32, flatstone_threshold);

            (
                WEIGHT.flatstone.0 * p1_position / flatstone_threshold +
                WEIGHT.flatstone.1 * (flatstone_threshold - p1_position) / flatstone_threshold,
                WEIGHT.flatstone.0 * p2_position / flatstone_threshold +
                WEIGHT.flatstone.1 * (flatstone_threshold - p2_position) / flatstone_threshold,
            )
        };

        b.iter(|| {
            p1_eval += test::black_box(evaluate_top_pieces(m.p1_flatstone_count as i32, p1_flatstone_weight, p1_standing_stones, p1_capstones));
            p2_eval += test::black_box(evaluate_top_pieces(m.p2_flatstone_count as i32, p2_flatstone_weight, p2_standing_stones, p2_capstones));
        });
    }

    #[bench]
    fn bench_evaluate_stacked_flatstones(b: &mut Bencher) {
        let mut p1_eval = 0;
        let mut p2_eval = 0;

        let m = &STATE.metadata;

        let p1_flatstones = m.p1_pieces & !m.standing_stones & !m.capstones;
        let p2_flatstones = m.p2_pieces & !m.standing_stones & !m.capstones;

        let p1_standing_stones = m.p1_pieces & m.standing_stones;
        let p2_standing_stones = m.p2_pieces & m.standing_stones;

        let p1_capstones = m.p1_pieces & m.capstones;
        let p2_capstones = m.p2_pieces & m.capstones;

        b.iter(|| {
            let stacked_flatstones_eval = test::black_box(evaluate_stacked_flatstones(
                m,
                p1_flatstones,
                p2_flatstones,
                p1_standing_stones,
                p2_standing_stones,
                p1_capstones,
                p2_capstones,
            ));
            p1_eval += stacked_flatstones_eval.0;
            p2_eval += stacked_flatstones_eval.1;
        });
    }

    #[bench]
    fn bench_evaluate_road_groups(b: &mut Bencher) {
        let mut p1_eval = 0;
        let mut p2_eval = 0;

        let m = &STATE.metadata;

        b.iter(|| {
            p1_eval += test::black_box(evaluate_road_groups(m, &m.p1_road_groups));
            p2_eval += test::black_box(evaluate_road_groups(m, &m.p2_road_groups));
        });
    }

    #[bench]
    fn bench_evaluate_threats(b: &mut Bencher) {
        let mut p1_eval = 0;
        let mut p2_eval = 0;

        let m = &STATE.metadata;

        let total_pieces = m.p1_pieces | m.p2_pieces;

        b.iter(|| {
            p1_eval += test::black_box(evaluate_threats(m, total_pieces, &m.p1_road_groups));
            p2_eval += test::black_box(evaluate_threats(m, total_pieces, &m.p2_road_groups));
        });
    }

    #[bench]
    fn bench_evaluate_influence(b: &mut Bencher) {
        let mut p1_eval = 0;
        let mut p2_eval = 0;

        let m = &STATE.metadata;

        let total_pieces = m.p1_pieces | m.p2_pieces;

        let p1_flatstones = m.p1_pieces & !m.standing_stones & !m.capstones;
        let p2_flatstones = m.p2_pieces & !m.standing_stones & !m.capstones;

        b.iter(|| {
            p1_eval += test::black_box(evaluate_influence(
                m,
                total_pieces,
                m.p1_pieces,
                &m.p1_flatstones,
                p1_flatstones,
                m.p2_pieces,
            ));
            p2_eval += test::black_box(evaluate_influence(
                m,
                total_pieces,
                m.p2_pieces,
                &m.p2_flatstones,
                p2_flatstones,
                m.p1_pieces,
            ));
        });
    }
}
