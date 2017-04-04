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
use impls::tak::state::metadata::{Bitmap, BitmapInterface, BOARD, EDGE};
use state;

impl state::State for State {
    type Ply = Ply;
    type Resolution = Resolution;

    fn get_ply_count(&self) -> usize {
        self.ply_count as usize
    }

    fn execute_ply(&mut self, ply: Option<&Ply>) -> Result<(), String> {
        // Null move
        if ply.is_none() {
            self.ply_count += 1;
            return Ok(());
        }

        let board_size = self.board.len();

        match *ply.unwrap() {
            Ply::Place { x, y, ref piece } => {
                if !self.board[x][y].is_empty() {
                    return Err(String::from("Cannot place piece in an occupied space."));
                }

                let count = match *piece {
                    Piece::Flatstone(color) |
                    Piece::StandingStone(color) => if color == Color::White {
                        &mut self.p1_flatstones
                    } else {
                        &mut self.p2_flatstones
                    },
                    Piece::Capstone(color) => if color == Color::White {
                        &mut self.p1_capstones
                    } else {
                        &mut self.p2_capstones
                    },
                };

                if *count > 0 {
                    *count -= 1;
                } else {
                    return Err(String::from("Insufficient pieces for placement."));
                }

                self.board[x][y].push(piece.clone());

                match *piece {
                    Piece::Flatstone(color) => self.metadata.add_flatstone(
                        color, x, y, self.board[x][y].len() - 1,
                    ),
                    ref block => self.metadata.add_blocking_stone(block, x, y),
                }

                match *piece {
                    Piece::Flatstone(_) |
                    Piece::Capstone(_) => self.metadata.calculate_road_groups(),
                    _ => (),
                }

                self.ply_crushes.push(false);
            },
            Ply::Slide { x, y, direction, ref drops } => {
                // First, verify that the slide is okay
                let next_color = if self.ply_count % 2 == 0 {
                    Color::White
                } else {
                    Color::Black
                };

                if let Some(piece) = self.board[x][y].last() {
                    if piece.get_color() != next_color {
                        return Err(String::from("Cannot move an opponent's piece."));
                    }
                }

                let grab: usize = drops.iter().sum::<u8>() as usize;

                if grab > board_size || self.board[x][y].len() < grab {
                    return Err(String::from("Illegal carry amount."));
                }

                let (dx, dy) = direction.to_offset();

                {
                    let (tx, ty) = (
                        x as i8 + dx * drops.len() as i8,
                        y as i8 + dy * drops.len() as i8,
                    );

                    if tx < 0 || tx >= board_size as i8 ||
                       ty < 0 || ty >= board_size as i8 {
                        return Err(String::from("Slide out of bounds."));
                    }
                }

                let mut nx = x as i8;
                let mut ny = y as i8;

                for drop in drops {
                    nx += dx;
                    ny += dy;

                    if !self.board[nx as usize][ny as usize].is_empty() {
                        match *self.board[nx as usize][ny as usize].last().unwrap() {
                            Piece::Capstone(_) => return Err(String::from("Cannot slide onto a capstone.")),
                            Piece::StandingStone(_) => if *drop == 1 {
                                match *self.board[x][y].last().unwrap() {
                                    Piece::Capstone(_) => (),
                                    _ => return Err(String::from("Cannot slide onto a standing stone.")),
                                }
                            } else {
                                return Err(String::from("Cannot slide onto a standing stone."));
                            },
                            _ => (),
                        }
                    }
                }

                // If everything checks out, execute the slide
                let mut stack = Vec::new();
	            for _ in 0..grab {
	                let piece = self.board[x][y].pop().unwrap();

	                match piece {
	                    Piece::Flatstone(color) => self.metadata.remove_flatstone(
	                        color, x, y, self.board[x][y].len(),
                        ),
                        ref block => self.metadata.remove_blocking_stone(block, x, y),
                    }

                    if let Some(revealed) = self.board[x][y].last() {
                        self.metadata.reveal_flatstone(
                            revealed.get_color(), x, y,
                        );
                    }

	                stack.push(piece);
                }

                self.ply_crushes.push(false);

                let mut nx = x as i8;
                let mut ny = y as i8;

                for drop in drops {
                    nx += dx;
                    ny += dy;

                    if !self.board[nx as usize][ny as usize].is_empty() {
                        let target_top = self.board[nx as usize][ny as usize].last().unwrap().clone();
                        match target_top {
                            Piece::StandingStone(color) => {
                                *self.board[nx as usize][ny as usize].last_mut().unwrap() = Piece::Flatstone(color);
                                self.metadata.remove_blocking_stone(&Piece::StandingStone(color), nx as usize, ny as usize);
                                self.metadata.add_flatstone(
                                    color, nx as usize, ny as usize,
                                    self.board[nx as usize][ny as usize].len() - 1,
                                );
                                *self.ply_crushes.last_mut().unwrap() = true;
                            },
                            _ => (),
                        }
                    }

                    for _ in 0..*drop {
                        if let Some(covered) = self.board[nx as usize][ny as usize].last() {
                            self.metadata.cover_flatstone(
                                covered.get_color(), nx as usize, ny as usize,
                            );
                        }

                        let piece = stack.pop().unwrap();

                        match piece {
                            Piece::Flatstone(color) => self.metadata.add_flatstone(
                                color, nx as usize, ny as usize,
                                self.board[nx as usize][ny as usize].len(),
                            ),
                            ref block => self.metadata.add_blocking_stone(
                                block, nx as usize, ny as usize,
                            ),
                        }

                        self.board[nx as usize][ny as usize].push(piece);
                    }
                }

                self.metadata.calculate_road_groups();
            },
        }

        self.ply_count += 1;

        Ok(())
    }

    fn revert_ply(&mut self, ply: Option<&Ply>) -> Result<(), String> {
        if self.ply_count == 0 {
            return Err(String::from("No more plies to revert."));
        }

        // Null move
        if ply.is_none() {
            self.ply_count -= 1;
            return Ok(())
        }

        let board_size = self.board.len();

        match *ply.unwrap() {
            Ply::Place { x, y, ref piece } => {
                if self.board[x][y].is_empty() {
                    return Err(String::from("Cannot remove piece from an empty space."));
                }

                if self.board[x][y].len() != 1 {
                    return Err(String::from("Cannot remove piece from a stack."));
                }

                if self.board[x][y][0] != *piece {
                    return Err(String::from("Top piece is the wrong piece to remove."));
                }

                let count = match *piece {
                    Piece::Flatstone(color) |
                    Piece::StandingStone(color) => if color == Color::White {
                        &mut self.p1_flatstones
                    } else {
                        &mut self.p2_flatstones
                    },
                    Piece::Capstone(color) => if color == Color::White {
                        &mut self.p1_capstones
                    } else {
                        &mut self.p2_capstones
                    },
                };

                *count += 1;

                self.board[x][y].pop();

                match *piece {
                    Piece::Flatstone(color) => self.metadata.remove_flatstone(
                        color, x, y, 0,
                    ),
                    ref block => self.metadata.remove_blocking_stone(block, x, y),
                }

                match *piece {
                    Piece::Flatstone(_) |
                    Piece::Capstone(_) => self.metadata.calculate_road_groups(),
                    _ => (),
                }

                self.ply_crushes.pop();
            },
            Ply::Slide { x, y, direction, ref drops } => {
                let previous_color = if self.ply_count % 2 == 1 {
                    Color::White
                } else {
                    Color::Black
                };

                let (dx, dy) = direction.to_offset();

                let (tx, ty) = (
                    x as i8 + dx * drops.len() as i8,
                    y as i8 + dy * drops.len() as i8,
                );

                if tx < 0 || tx >= board_size as i8 ||
                   ty < 0 || ty >= board_size as i8 {
                    return Err(String::from("Slide out of bounds."));
                }

                if self.board[tx as usize][ty as usize].is_empty() {
                    return Err(String::from("Target space is empty."));
                }

                if self.board[tx as usize][ty as usize].last().unwrap().get_color() != previous_color {
                    return Err(String::from("Cannot revert move with an opponent's piece"));
                }

                if let Some(&crush) = self.ply_crushes.last() {
                    if crush {
                        if self.board[tx as usize][ty as usize].len() < 2 {
                            return Err(String::from("Not enough pieces to revert standing stone crush."));
                        }
                        if *drops.last().unwrap() != 1 {
                            return Err(String::from("Move crushed a standing stone, but dropped more than one stone."));
                        }
                    }
                }

                let grab: usize = drops.iter().sum::<u8>() as usize;

                if grab > board_size {
                    return Err(String::from("Illegal carry amount."));
                }

                let mut px = tx;
                let mut py = ty;

                for drop in drops.iter().rev() {
                    if self.board[px as usize][py as usize].len() < *drop as usize {
                        return Err(String::from("Insufficient pieces in stack to revert."));
                    }

                    px -= dx;
                    py -= dy;
                }

                // If everything checks out, revert the slide
                let mut stack = Vec::new();
                let mut px = tx;
                let mut py = ty;

                for drop in drops.iter().rev() {
                    for _ in 0..*drop {
                        let piece = self.board[px as usize][py as usize].pop().unwrap();

                        match piece {
	                        Piece::Flatstone(color) => self.metadata.remove_flatstone(
	                            color, px as usize, py as usize, self.board[px as usize][py as usize].len(),
                            ),
                            ref block => self.metadata.remove_blocking_stone(block, px as usize, py as usize),
                        }

                        if let Some(revealed) = self.board[px as usize][py as usize].last() {
                            self.metadata.reveal_flatstone(revealed.get_color(), px as usize, py as usize);
                        }

                        stack.push(piece);
                    }

                    px -= dx;
                    py -= dy;
                }

                for _ in 0..grab {
                    if let Some(covered) = self.board[x][y].last() {
                        self.metadata.cover_flatstone(covered.get_color(), x, y);
                    }

                    let piece = stack.pop().unwrap();

                    match piece {
                        Piece::Flatstone(color) => self.metadata.add_flatstone(
                            color, x, y,
                            self.board[x][y].len(),
                        ),
                        ref block => self.metadata.add_blocking_stone(block, x, y),
                    }

                    self.board[x][y].push(piece);
                }

                if let Some(crush) = self.ply_crushes.pop() {
                    if crush {
                        let color = self.board[tx as usize][ty as usize].pop().unwrap().get_color();
                        self.board[tx as usize][ty as usize].push(Piece::StandingStone(color));
                        self.metadata.remove_flatstone(
                            color, tx as usize, ty as usize,
                            self.board[tx as usize][ty as usize].len() - 1,
                        );
                        self.metadata.add_blocking_stone(&Piece::StandingStone(color), tx as usize, ty as usize);
                    }
                }

                self.metadata.calculate_road_groups();
            },
        }

        self.ply_count -= 1;

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

    fn null_move_allowed(&self) -> bool {
        self.p1_flatstones > 3 &&
        self.p2_flatstones > 3 &&
        (self.metadata.p1_pieces | self.metadata.p2_pieces).get_population() < (self.board.len() * self.board.len() - 3) as u8
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
