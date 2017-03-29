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

use impls::tak::{Color, Direction, Piece, State};
use impls::tak::state::metadata::{Bitmap, BitmapInterface, EDGE};

pub fn gather_features(state: &State) -> Vec<f32> {
    let mut features = Vec::new();

    // 1 - Side to move
    features.push((state.ply_count % 2) as f32);

    // 1 - White reserve flatstones
    features.push(state.p1_flatstones as f32 / 21.0);

    // 1 - Black reserve flatstones
    features.push(state.p2_flatstones as f32 / 21.0);

    // 1 - White played capstone
    features.push(1.0 - state.p1_capstones as f32);

    // 1 - Black played capstone
    features.push(1.0 - state.p2_capstones as f32);

    // 2 - White capstone position
    let (x, y) = get_position(state.metadata.capstones & state.metadata.p1_pieces);
    features.push(x);
    features.push(y);

    // 2 - Black capstone position
    let (x, y) = get_position(state.metadata.capstones & state.metadata.p2_pieces);
    features.push(x);
    features.push(y);

    // 1 - White played standing stones
    features.push((state.metadata.standing_stones & state.metadata.p1_pieces).get_population() as f32 / 5.0);

    // 1 - Black played standing stones
    features.push((state.metadata.standing_stones & state.metadata.p2_pieces).get_population() as f32 / 5.0);

    // 275 - Stack positions and configurations, ordered tallest to shortest
    let mut stacks = state.board.iter().flat_map(|column| column.iter()).enumerate().map(|(i, stack)| (stack, (i / 5) as f32 / 4.0, (i % 5) as f32 / 4.0)).collect::<Vec<_>>();
    stacks.sort_by(|a, b| {
        b.0.len().cmp(&a.0.len())
    });
    for &(stack, x, y) in &stacks {
        features.push(x);
        features.push(y);
        for i in 0..9 {
            if i < stack.len() {
                features.push(match stack[i] {
                    Piece::Flatstone(Color::White) => 1.0,
                    Piece::Flatstone(Color::Black) => 0.0,
                    Piece::StandingStone(Color::White) => 0.75,
                    Piece::StandingStone(Color::Black) => 0.25,
                    _ => 0.5,
                });
            } else {
                features.push(0.5);
            }
        }
    }

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

