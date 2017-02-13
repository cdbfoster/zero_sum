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

extern crate zero_sum;

use zero_sum::analysis::search::Search;
use zero_sum::impls::tak::*;

fn main() {
    let state = State::new(5);
    let evaluator = evaluator::StaticEvaluator;
    let mut search = zero_sum::analysis::search::pvsearch::PvSearch::with_goal(evaluator, 60, 12.0);

    println!("Searching for opening move...");

    let analysis = search.search(&state, None);

    println!("{}", analysis.statistics);
}
