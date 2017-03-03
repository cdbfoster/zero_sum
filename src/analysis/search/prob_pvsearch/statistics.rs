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

/// Represents statistics for the search at a single depth.
#[derive(Clone, Copy, Debug)]
pub struct StatisticsLevel {
    /// The number of nodes visited on this level.
    pub visited: u32,
    /// The number of nodes evaluated on this level.
    pub evaluated: u32,
    /// The number of usable transposition table hits.
    pub tt_saves: u32,
    /// The number of states we visited that were already in the transposition table.
    pub tt_hits: u32,
    /// The number of states we put into the transposition table.
    pub tt_stores: u32,
    /// The amount of time we spent searching this depth.
    pub time: f32,
    /// The average branching factor.
    pub branch: f32,
}

impl StatisticsLevel {
    pub fn new() -> StatisticsLevel {
        StatisticsLevel {
            visited: 0,
            evaluated: 0,
            tt_saves: 0,
            tt_hits: 0,
            tt_stores: 0,
            time: 0.0,
            branch: 0.0,
        }
    }
}

/// Full statistics for this search.
pub struct Statistics {
    /// Statistics for each depth of the search.
    pub iteration: Vec<Vec<StatisticsLevel>>,
}

impl Statistics {
    pub fn new() -> Statistics {
        Statistics {
            iteration: Vec::new(),
        }
    }

    /// Calculate the totals for each iteration of the search's iterative deepening.
    pub fn calculate_iteration_totals(&self) -> Vec<StatisticsLevel> {
        let mut totals = vec![StatisticsLevel::new(); self.iteration.len()];
        for (i, iteration) in self.iteration.iter().enumerate() {
            for depth in iteration {
                totals[i].visited += depth.visited;
                totals[i].evaluated += depth.evaluated;
                totals[i].tt_saves += depth.tt_saves;
                totals[i].tt_hits += depth.tt_hits;
                totals[i].tt_stores += depth.tt_stores;
                totals[i].time += depth.time;
                totals[i].branch += depth.branch;
            }
            totals[i].branch /= (iteration.len() - 1) as f32;
        }
        totals
    }

    /// Calculate the totals for all search iterations.
    pub fn calculate_totals(&self) -> StatisticsLevel {
        let mut final_totals = StatisticsLevel::new();
        for total in &self.calculate_iteration_totals() {
            final_totals.visited += total.visited;
            final_totals.evaluated += total.evaluated;
            final_totals.tt_saves += total.tt_saves;
            final_totals.tt_hits += total.tt_hits;
            final_totals.tt_stores += total.tt_stores;
            final_totals.time += total.time;
            final_totals.branch += total.branch;
        }
        final_totals.branch /= self.iteration.len() as f32;
        final_totals
    }
}

impl fmt::Display for Statistics {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let titles = [
            "Visited:",
            "Evaluated:",
            "TT Saves:",
            "TT Hits:",
            "TT Stores:",
            "Time:",
            "Branching Factor:",
        ];

        let title_width = titles.iter().map(|title| title.len()).max().unwrap() + 1;
        let max_depth = self.iteration.iter().map(|iteration| iteration.len()).max().unwrap();
        let max_iteration = self.iteration.len();

        let mut data = vec![vec![vec![String::new(); titles.len()]; max_depth]; max_iteration];
        let mut data_totals = vec![vec![String::new(); titles.len()]; max_iteration];

        for (i, iteration) in self.iteration.iter().enumerate() {
            for j in 0..iteration.len() {
                data[i][j][0] = format!("{}", iteration[j].visited);
                data[i][j][1] = format!("{}", iteration[j].evaluated);
                data[i][j][2] = format!("{}", iteration[j].tt_saves);
                data[i][j][3] = format!("{}", iteration[j].tt_hits);
                data[i][j][4] = format!("{}", iteration[j].tt_stores);
                data[i][j][5] = if iteration[j].time != 0.0 {
                    format!("{:.2}", iteration[j].time)
                } else {
                    String::new()
                };
                data[i][j][6] = format!("{:.2}", iteration[j].branch);
            }
        }

        let totals = self.calculate_iteration_totals();

        for i in 0..max_iteration {
            data_totals[i][0] = format!("{}", totals[i].visited);
            data_totals[i][1] = format!("{}", totals[i].evaluated);
            data_totals[i][2] = format!("{}", totals[i].tt_saves);
            data_totals[i][3] = format!("{}", totals[i].tt_hits);
            data_totals[i][4] = format!("{}", totals[i].tt_stores);
            data_totals[i][5] = format!("{:.2}", totals[i].time);
            data_totals[i][6] = format!("{:.2}", totals[i].branch);
        }

        let column_widths = {
            let widths = data.iter().map(|iteration|
                iteration.iter().map(|depth|
                    depth.iter().map(|stat| stat.len()).collect::<Vec<_>>()
                ).collect::<Vec<_>>()
            ).collect::<Vec<_>>();

            let totals_widths = data_totals.iter().map(|iteration|
                iteration.iter().map(|stat| stat.len()).collect::<Vec<_>>()
            ).collect::<Vec<_>>();

            (0..cmp::max(max_depth, max_iteration)).map(|depth|
                (0..(max_iteration * titles.len())).map(|i| {
                    let iteration = i / titles.len();
                    let stat = i % titles.len();
                    cmp::max(
                        if depth < max_depth {
                            widths[iteration][depth][stat]
                        } else {
                            0
                        },
                        totals_widths[iteration][stat],
                    )
                }).max().unwrap()
            ).collect::<Vec<_>>()
        };

        for (i, iteration) in data.iter().enumerate() {
            try!(write!(f, "\n  {0:1$}", format!("Iteration {}:", i + 1), title_width + 2));
            for j in 0..iteration.len() {
                try!(write!(f, "  {0:>1$}", j + 1, column_widths[j]));
            }
            for (j, title) in titles.iter().enumerate() {
                try!(write!(f, "\n    {0:1$}", title, title_width));
                for (k, depth) in iteration.iter().enumerate() {
                    try!(write!(f, "  {0:>1$}", depth[j], column_widths[k]));
                }
            }
            try!(write!(f, "\n"));
        }

        let final_totals = {
            let final_totals = self.calculate_totals();

            vec![
                format!("{}", final_totals.visited),
                format!("{}", final_totals.evaluated),
                format!("{}", final_totals.tt_saves),
                format!("{}", final_totals.tt_hits),
                format!("{}", final_totals.tt_stores),
                format!("{:.2}", final_totals.time),
                format!("{:.2}", final_totals.branch),
            ]
        };

        let final_totals_width = cmp::max(
            final_totals.iter().map(|total| total.len()).max().unwrap(),
            "Total".len(),
        );

        try!(write!(f, "\n  {0:1$}", "Totals:", title_width + 2));
        for j in 0..max_iteration {
            try!(write!(f, "  {0:>1$}", j + 1, column_widths[j]));
        }
        try!(write!(f, "  {0:>1$}", "Total", final_totals_width));
        for (stat, title) in titles.iter().enumerate() {
            try!(write!(f, "\n    {0:1$}", title, title_width));
            for (k, iteration) in data_totals.iter().enumerate() {
                try!(write!(f, "  {0:>1$}", iteration[stat], column_widths[k]));
            }
            try!(write!(f, "  {0:>1$}", final_totals[stat], final_totals_width));
        }

        Ok(())
    }
}
