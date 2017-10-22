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

use std::i8;

use analysis::{self, Evaluation as EvaluationTrait};
use impls::tic_tac_toe::{Board, Mark, Ply, Resolution};
use ply;
use resolution;
use state::{self, State};

impl ply::Ply for Ply { }

impl resolution::Resolution for Resolution {
    fn get_winner(&self) -> Option<u8> {
        match *self {
            Resolution::Win(Mark::X) => Some(0),
            Resolution::Win(Mark::O) => Some(1),
            _ => None,
        }
    }

    fn is_draw(&self) -> bool {
        if let Resolution::CatsGame = *self { true } else { false }
    }
}

impl state::State for Board {
    type Ply = Ply;
    type Resolution = Resolution;

    fn get_ply_count(&self) -> usize {
        self.1 as usize
    }

    fn execute_ply(&mut self, ply: Option<&Ply>) -> Result<(), String> {
        let ply = if let Some(ply) = ply {
            ply
        } else {
            self.1 += 1;
            return Ok(());
        };

        if ply.coordinates.0 >= 3 || ply.coordinates.1 >= 3 {
            return Err(String::from("Coordinates out of bounds"));
        }

        let index = ply.coordinates.0 + 3 * ply.coordinates.1;

        if self.0[index].is_some() {
            return Err(String::from("Space already occupied"));
        }

        self.0[index] = Some(ply.mark);
        self.1 += 1;
        Ok(())
    }

    fn revert_ply(&mut self, ply: Option<&Ply>) -> Result<(), String> {
        let ply = if let Some(ply) = ply {
            ply
        } else {
            self.1 -= 1;
            return Ok(());
        };

        if ply.coordinates.0 >= 3 || ply.coordinates.1 >= 3 {
            return Err(String::from("Coordinates out of bounds"));
        }

        let index = ply.coordinates.0 + 3 * ply.coordinates.1;

        if self.0[index].is_none() {
            return Err(String::from("Space is empty"));
        }

        if self.0[index] != Some(ply.mark) {
            return Err(String::from("Space has the wrong mark"));
        }

        self.0[index] = None;
        self.1 -= 1;
        Ok(())
    }

    fn check_resolution(&self) -> Option<Resolution> {
        fn get_lines(b: &[Option<Mark>; 9]) -> [[&Option<Mark>; 3]; 8] {
            [[&b[0], &b[1], &b[2]],
             [&b[3], &b[4], &b[5]],
             [&b[6], &b[7], &b[8]],
             [&b[0], &b[3], &b[6]],
             [&b[1], &b[4], &b[7]],
             [&b[2], &b[5], &b[8]],
             [&b[0], &b[4], &b[8]],
             [&b[2], &b[4], &b[6]]]
        }

        if let Some(Some(mark)) = get_lines(&self.0)
            .iter()
            .map(|line| if line[0] == line[1] && line[0] == line[2] { *line[0] } else { None })
            .find(|r| r.is_some()) {
            return Some(Resolution::Win(mark))
        }

        if self.0.iter().find(|space| space.is_none()).is_none() {
            Some(Resolution::CatsGame)
        } else {
            None
        }
    }
}

impl analysis::Extrapolatable<Ply> for Board {
    fn extrapolate_into(&self, plies: &mut Vec<Ply>) {
        let next_mark = self.next_mark();
        for ply in self.0.iter().enumerate().filter_map(|(index, space)| if space.is_none() {
            Some(Ply {
                mark: next_mark,
                coordinates: (index % 3, index / 3),
            })
        } else {
            None
        }) {
            plies.push(ply)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Evaluation(pub i8);

prepare_evaluation_tuple!(Evaluation); // Implements arithmetic operators and display in terms of the inner type

impl analysis::Evaluation for Evaluation {
    fn null() -> Evaluation { Evaluation(0) }
    fn shift(self, steps: i32) -> Evaluation { Evaluation(self.0 + steps as i8) }
    fn win() -> Evaluation { Evaluation(14) }
    fn max() -> Evaluation { Evaluation(i8::MAX) }
    fn is_win(&self) -> bool { self.0 >= 5 }
}

/// A very simple evaluator.  Gives extra weight to the corners.
pub struct Evaluator;

impl analysis::Evaluator for Evaluator {
    type State = Board;
    type Evaluation = Evaluation;

    fn evaluate(&self, state: &Board) -> Evaluation {
        let next_mark = state.next_mark();

        if let Some(Resolution::Win(mark)) = state.check_resolution() {
            if mark == next_mark {
                Evaluation(Evaluation::win().0 - state.1 as i8)
            } else {
                -Evaluation(Evaluation::win().0 - state.1 as i8)
            }
        } else {
            // Weight the corners.
            // This doesn't matter at all with regards to perfect play,
            // but there are more ways a human can mess up if we start in the corner.
            let x_corners = (0..9).filter(|&x| x % 2 == 0 && x != 4 && state.0[x] == Some(Mark::X)).count() as i8;
            let o_corners = (0..9).filter(|&x| x % 2 == 0 && x != 4 && state.0[x] == Some(Mark::O)).count() as i8;

            if next_mark == Mark::X {
                Evaluation(x_corners - o_corners)
            } else {
                Evaluation(o_corners - x_corners)
            }
        }
    }
}
