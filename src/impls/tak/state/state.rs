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

use impls::tak::{Color, Piece};
use impls::tak::ply::Ply;
use impls::tak::resolution::Resolution;
use impls::tak::state::State;
use impls::tak::state::metadata::{Bitmap, BOARD, EDGE};
use state;

impl state::State for State {
    type Ply = Ply;
    type Resolution = Resolution;

    fn execute_ply_preallocated(&self, ply: &Ply, next: &mut State) -> Result<(), String> {
        next.clone_from(self);
        next.ply_count += 1;

        let board_size = next.board.len();

        match *ply {
            Ply::Place { x, y, ref piece } => {
                if !next.board[x][y].is_empty() {
                    return Err(String::from("Cannot place piece in an occupied space."));
                }

                let count = match *piece {
                    Piece::Flatstone(color) |
                    Piece::StandingStone(color) => if color == Color::White {
                        &mut next.p1_flatstones
                    } else {
                        &mut next.p2_flatstones
                    },
                    Piece::Capstone(color) => if color == Color::White {
                        &mut next.p1_capstones
                    } else {
                        &mut next.p2_capstones
                    },
                };

                if *count > 0 {
                    *count -= 1;
                } else {
                    return Err(String::from("Insufficient pieces for placement."));
                }

                next.board[x][y].push(piece.clone());

                match *piece {
                    Piece::Flatstone(color) => next.metadata.add_flatstone(
                        color, x, y, next.board[x][y].len() - 1,
                    ),
                    ref block => next.metadata.add_blocking_stone(block, x, y),
                }

                match *piece {
                    Piece::Flatstone(_) |
                    Piece::Capstone(_) => next.metadata.calculate_road_groups(),
                    _ => (),
                }
            },
            Ply::Slide { x, y, direction, ref drops } => {
                let next_color = if self.ply_count % 2 == 0 {
                    Color::White
                } else {
                    Color::Black
                };

                match next.board[x][y].last() {
                    Some(&Piece::Flatstone(color)) |
                    Some(&Piece::StandingStone(color)) |
                    Some(&Piece::Capstone(color)) => if color != next_color {
                        return Err(String::from("Cannot move an opponent's piece."));
                    },
                    _ => (),
                }

                let grab = drops.iter().fold(0, |acc, x| acc + x) as usize;

                if grab > board_size || next.board[x][y].len() < grab {
                    return Err(String::from("Illegal carry amount."));
                }

                let mut stack = Vec::new();
	            for _ in 0..grab {
	                let piece = next.board[x][y].pop().unwrap();

	                match piece {
	                    Piece::Flatstone(color) => next.metadata.remove_flatstone(
	                        color, x, y, next.board[x][y].len(),
                        ),
                        ref block => next.metadata.remove_blocking_stone(block, x, y),
                    }

                    if let Some(revealed) = next.board[x][y].last() {
                        next.metadata.reveal_flatstone(
                            revealed.get_color(), x, y,
                        );
                    }

	                stack.push(piece);
                }

                let (dx, dy) = direction.to_offset();

                let mut nx = x as i8;
                let mut ny = y as i8;

                {
                    let (tx, ty) = (
                        nx + dx * drops.len() as i8,
                        ny + dy * drops.len() as i8,
                    );

                    if tx < 0 || tx >= board_size as i8 ||
                       ty < 0 || ty >= board_size as i8 {
                        return Err(String::from("Slide out of bounds."));
                    }
                }

                for drop in drops {
                    nx += dx;
                    ny += dy;

                    if !next.board[nx as usize][ny as usize].is_empty() {
                        let target_top = next.board[nx as usize][ny as usize].last().unwrap().clone();
                        match target_top {
                            Piece::Capstone(_) => return Err(String::from("Cannot slide onto a capstone.")),
                            Piece::StandingStone(color) => if stack.len() == 1 {
                                match stack[0] {
                                    Piece::Capstone(_) => {
                                        *next.board[nx as usize][ny as usize].last_mut().unwrap() = Piece::Flatstone(color);
                                        next.metadata.remove_blocking_stone(&Piece::StandingStone(color), nx as usize, ny as usize);
                                        next.metadata.add_flatstone(
                                            color, nx as usize, ny as usize,
                                            next.board[nx as usize][ny as usize].len() - 1,
                                        )
                                    },
                                    _ => return Err(String::from("Cannot slide onto a standing stone.")),
                                }
                            } else {
                                return Err(String::from("Cannot slide onto a standing stone."));
                            },
                            _ => (),
                        }
                    }

                    for _ in 0..*drop {
                        if let Some(covered) = next.board[nx as usize][ny as usize].last() {
                            next.metadata.cover_flatstone(
                                covered.get_color(), nx as usize, ny as usize,
                            );
                        }

                        let piece = stack.pop().unwrap();

                        match piece {
                            Piece::Flatstone(color) => next.metadata.add_flatstone(
                                color, nx as usize, ny as usize,
                                next.board[nx as usize][ny as usize].len(),
                            ),
                            ref block => next.metadata.add_blocking_stone(
                                block, nx as usize, ny as usize,
                            ),
                        }

                        next.board[nx as usize][ny as usize].push(piece);
                    }
                }

                next.metadata.calculate_road_groups();
            },
        }

        Ok(())
    }

    fn check_resolution(&self) -> Option<Resolution> {
        let board_size = self.board.len();
        let m = &self.metadata;

        let has_road = |groups: &Vec<Bitmap>| {
            use impls::tak::Direction::*;

            for group in groups.iter() {
                if (group & EDGE[board_size][North as usize] != 0 &&
                    group & EDGE[board_size][South as usize] != 0) ||
                   (group & EDGE[board_size][West as usize] != 0 &&
                    group & EDGE[board_size][East as usize] != 0) {
                    return true;
                }
            }

            false
        };

        let p1_has_road = has_road(&m.p1_road_groups);
        let p2_has_road = has_road(&m.p2_road_groups);

        if p1_has_road && p2_has_road {
            if self.ply_count % 2 == 1 {
                Some(Resolution::Road(Color::White))
            } else {
                Some(Resolution::Road(Color::Black))
            }
        } else if p1_has_road {
            Some(Resolution::Road(Color::White))
        } else if p2_has_road {
            Some(Resolution::Road(Color::Black))
        } else if (self.p1_flatstones + self.p1_capstones) == 0 ||
                  (self.p2_flatstones + self.p2_capstones) == 0 ||
                  (m.p1_pieces | m.p2_pieces) == BOARD[board_size] {
            if m.p1_flatstone_count > m.p2_flatstone_count {
                Some(Resolution::Flat(Color::White))
            } else if m.p2_flatstone_count > m.p1_flatstone_count {
                Some(Resolution::Flat(Color::Black))
            } else {
                Some(Resolution::Draw)
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use test::{self, Bencher};

    use impls::tak::*;
    use state::State as StateTrait;

    #[bench]
    fn bench_state_execute_ply_place(b: &mut Bencher) {
        let state = State::from_tps("[TPS \"21,2,x,1212221C,x/21,2,2,21,x/x2,2,x2/x4,21/1112C,1,21,x2 2 22\"]").unwrap();
        let mut next = State::new(5);
        let ply = Ply::from_ptn("c2", Color::Black).unwrap();

        b.iter(|| {
            test::black_box(&state).execute_ply_preallocated(&ply, &mut next).unwrap()
        });
    }

    #[bench]
    fn bench_state_execute_ply_slide(b: &mut Bencher) {
        let state = State::from_tps("[TPS \"21,2,x,1212,x/21,2,2,2,x/x2,2,1221C,x/x4,21/1112C,1,21,x2 1 22\"]").unwrap();
        let mut next = State::new(5);
        let ply = Ply::from_ptn("4d3+13", Color::White).unwrap();

        b.iter(|| {
            test::black_box(&state).execute_ply_preallocated(&ply, &mut next).unwrap()
        });
    }

    #[bench]
    fn bench_state_check_resolution(b: &mut Bencher) {
        let state = State::from_tps("[TPS \"2112S,22221C,112S,122,x/212,x,2,1S,2/1S,x,2,2,1/1,2111112C,x,1,21/x,1,212,1,1 2 36\"]").unwrap();

        b.iter(|| {
            test::black_box(&state).check_resolution().unwrap()
        });
    }
}
