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
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct Statistics {
    pub visited: u32,
    pub evaluated: u32,
    pub tt_saves: u32,
    pub tt_hits: u32,
    pub tt_stores: u32,
    pub time: f32,
}

impl Statistics {
    pub fn new() -> Statistics {
        Statistics {
            visited: 0,
            evaluated: 0,
            tt_saves: 0,
            tt_hits: 0,
            tt_stores: 0,
            time: 0.0,
        }
    }
}

pub struct StatisticPrinter(pub Vec<Vec<Statistics>>);

impl fmt::Display for StatisticPrinter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let titles = [
            "Visited:",
            "Evaluated:",
            "TT Saves:",
            "TT Hits:",
            "TT Stores:",
            "Time:",
        ];

        let title_width = titles.iter().map(|title| title.len()).max().unwrap() + 1;

        let mut data = vec![vec![vec![String::new(); titles.len()]; self.0.len()]; self.0.len() + 1];

        for (i, depth) in self.0.iter().enumerate() {
            for j in 0..i + 1 {
                data[i][j][0] = format!("{}", depth[j].visited);
                data[i][j][1] = format!("{}", depth[j].evaluated);
                data[i][j][2] = format!("{}", depth[j].tt_saves);
                data[i][j][3] = format!("{}", depth[j].tt_hits);
                data[i][j][4] = format!("{}", depth[j].tt_stores);
                data[i][j][5] = if depth[j].time != 0.0 {
                    format!("{:.2}", depth[j].time)
                } else {
                    String::new()
                };
            }
        }

        let totals = {
            let mut totals = vec![Statistics::new(); self.0.len()];

            for (i, max_depth) in self.0.iter().enumerate() {
                for depth in max_depth {
                    totals[i].visited += depth.visited;
                    totals[i].evaluated += depth.evaluated;
                    totals[i].tt_saves += depth.tt_saves;
                    totals[i].tt_hits += depth.tt_hits;
                    totals[i].tt_stores += depth.tt_stores;
                    totals[i].time += depth.time;
                }
            }

            totals
        };

        {
            let i = data.len() - 1;
            for j in 0..i {
                data[i][j][0] = format!("{}", totals[j].visited);
                data[i][j][1] = format!("{}", totals[j].evaluated);
                data[i][j][2] = format!("{}", totals[j].tt_saves);
                data[i][j][3] = format!("{}", totals[j].tt_hits);
                data[i][j][4] = format!("{}", totals[j].tt_stores);
                data[i][j][5] = format!("{:.2}", totals[j].time);
            }
        }

        let column_widths = {
            let widths = data.iter().map(|max_depth|
                max_depth.iter().map(|depth|
                    depth.iter().map(|stat| stat.len()).collect::<Vec<_>>()
                ).collect::<Vec<_>>()
            ).collect::<Vec<_>>();

            (0..data.len() - 1).map(|depth|
                (0..(data.len() * titles.len())).map(|i| {
                    let max_depth = i / titles.len();
                    let stat = i % titles.len();
                    widths[max_depth][depth][stat]
                }).max().unwrap()
            ).collect::<Vec<_>>()
        };

        for (i, max_depth) in data[..data.len() - 1].iter().enumerate() {
            try!(write!(f, "\n  {0:1$}", format!("Max Depth {}:", i + 1), title_width + 2));
            for j in 0..max_depth.len() {
                try!(write!(f, "  {0:>1$}", j + 1, column_widths[j]));
            }
            for (j, title) in titles.iter().enumerate() {
                try!(write!(f, "\n    {0:1$}", title, title_width));
                for (k, depth) in max_depth.iter().enumerate() {
                    try!(write!(f, "  {0:>1$}", depth[j], column_widths[k]));
                }
            }
            try!(write!(f, "\n"));
        }

        let final_totals = {
            let mut final_totals = Statistics::new();
            for total in &totals {
                final_totals.visited += total.visited;
                final_totals.evaluated += total.evaluated;
                final_totals.tt_saves += total.tt_saves;
                final_totals.tt_hits += total.tt_hits;
                final_totals.tt_stores += total.tt_stores;
                final_totals.time += total.time;
            }

            vec![
                format!("{}", final_totals.visited),
                format!("{}", final_totals.evaluated),
                format!("{}", final_totals.tt_saves),
                format!("{}", final_totals.tt_hits),
                format!("{}", final_totals.tt_stores),
                format!("{:.2}", final_totals.time),
            ]
        };

        let final_totals_width = cmp::max(
            final_totals.iter().map(|total| total.len()).max().unwrap(),
            "Total".len(),
        );

        let totals_strings = data.last().unwrap();
        try!(write!(f, "\n  {0:1$}", "Totals:", title_width + 2));
        for j in 0..totals_strings.len() {
            try!(write!(f, "  {0:>1$}", j + 1, column_widths[j]));
        }
        try!(write!(f, "  {0:>1$}", "Total", final_totals_width));
        for (j, title) in titles.iter().enumerate() {
            try!(write!(f, "\n    {0:1$}", title, title_width));
            for (k, depth) in totals_strings.iter().enumerate() {
                try!(write!(f, "  {0:>1$}", depth[j], column_widths[k]));
            }
            try!(write!(f, "  {0:>1$}", final_totals[j], final_totals_width));
        }

        Ok(())
    }
}
