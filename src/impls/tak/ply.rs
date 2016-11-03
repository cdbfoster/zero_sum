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

use std::fmt;
use std::hash::{Hash, Hasher};

use impls::tak::{Color, Direction, Piece};
use ply;

/// Represents either a piece placement or a slide.
#[derive(Clone, Debug, PartialEq)]
pub enum Ply {
    /// Placement of a stone in an empty space.
    Place {
        x: usize,
        y: usize,
        piece: Piece
    },
    /// Slide of a stone or stack of stones already on the board.
    Slide {
        x: usize,
        y: usize,
        direction: Direction,
        drops: Vec<u8>
    },
}

impl Ply {
    pub fn from_ptn(ptn: &str, color: Color) -> Option<Ply> { // XXX Return Result<Ply, String>
        let mut chars = ptn.chars();

        let mut next = chars.next();

        let mut new_piece = match next {
            Some('S') => {
                next = chars.next();
                Some(Piece::StandingStone(color))
            },
            Some('C') => {
                next = chars.next();
                Some(Piece::Capstone(color))
            },
            Some('F') => {
                next = chars.next();
                Some(Piece::Flatstone(color))
            },
            None => return None,
            _ => None,
        };

        let grab = match next {
            Some(c) => if c.is_digit(10) {
                next = chars.next();
                Some(c as u8 - 48)
            } else {
                None
            },
            None => return None,
        };

        let x = match next {
            Some(c) => if c.is_alphabetic() && c.is_lowercase() {
                (c as u8 - 97) as usize
            } else {
                return None;
            },
            None => return None,
        };

        let y = match chars.next() {
            Some(c) => if c.is_digit(10) && c != '0' {
                (c as u8 - 49) as usize
            } else {
                return None;
            },
            None => return None,
        };

        let direction = match chars.next() {
            Some('+') => Some(Direction::North),
            Some('>') => Some(Direction::East),
            Some('-') => Some(Direction::South),
            Some('<') => Some(Direction::West),
            None => {
                if new_piece.is_none() {
                    new_piece = Some(Piece::Flatstone(color));
                }
                None
            },
            _ => return None,
        };

        let mut drops = Vec::new();
        for c in chars {
            if c.is_digit(10) {
                drops.push(c as u8 - 48);
            } else {
                return None;
            }
        }

        if new_piece.is_some() {
            if grab.is_some() || direction.is_some() || !drops.is_empty() {
                return None;
            }

            Some(Ply::Place {
                x: x,
                y: y,
                piece: new_piece.unwrap(),
            })
        } else if direction.is_some() {
            if drops.is_empty() {
                if grab.is_some() {
                    drops.push(grab.unwrap());
                } else {
                    drops.push(1);
                }
            } else {
                if grab.is_none() {
                    return None;
                }
                if grab.unwrap() != drops.iter().fold(0, |acc, x| acc + x) {
                    return None;
                }
            }

            Some(Ply::Slide {
                x: x,
                y: y,
                direction: direction.unwrap(),
                drops: drops
            })
        } else {
            None
        }
    }

    pub fn to_ptn(&self) -> String {
        let mut ptn = String::new();

        match *self {
            Ply::Place { x, y, ref piece } => {
                match *piece {
                    Piece::StandingStone(_) => ptn.push('S'),
                    Piece::Capstone(_) => ptn.push('C'),
                    _ => (),
                }

                ptn.push((x as u8 + 97) as char);
                ptn.push((y as u8 + 49) as char);
            },
            Ply::Slide { x, y, direction, ref drops } => {
                let grab = drops.iter().fold(0, |acc, x| acc + x);

                if grab > 1 {
                    ptn.push((grab as u8 + 48) as char);
                }

                ptn.push((x as u8 + 97) as char);
                ptn.push((y as u8 + 49) as char);

                match direction {
                    Direction::North => ptn.push('+'),
                    Direction::East => ptn.push('>'),
                    Direction::South => ptn.push('-'),
                    Direction::West => ptn.push('<'),
                }

                if drops.len() > 1 {
                    for drop in drops.iter() {
                        ptn.push((*drop as u8 + 48) as char);
                    }
                }
            },
        }

        ptn
    }
}

impl ply::Ply for Ply { }

impl fmt::Display for Ply {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_ptn())
    }
}

impl Hash for Ply {
    fn hash<H>(&self, state: &mut H) where H: Hasher {
        let mut hash = 0;
        match *self {
            Ply::Place { x, y, ref piece } => {
                hash = x as u64;
                hash = (hash << 3) | y as u64;
                match *piece {
                    Piece::Flatstone(color) =>     hash = (hash << 3) | color as u64,
                    Piece::StandingStone(color) => hash = (hash << 3) | 2 | color as u64,
                    Piece::Capstone(color) =>      hash = (hash << 3) | 4 | color as u64,
                }
            },
            Ply::Slide { x, y, direction, ref drops } => {
                for drop in drops {
                    hash = (hash << 3) | *drop as u64;
                }
                hash = (hash << 3) | x as u64;
                hash = (hash << 3) | y as u64;
                hash = (hash << 3) | direction as u64;
            },
        }
        state.write_u64(hash);
    }
}
