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

use analysis::Extrapolatable;
use analysis::search::mctsearch::TreeNode;
use state::State;

pub trait Selector<S>: Clone + Send where
    S: State + Extrapolatable<<S as State>::Ply> {
    type Data: Clone + Default;

    fn initialize_policy_data(&self) -> Self::Data;

    fn select_descendant_index(&mut self, tree_node: &TreeNode<S, Self>) -> Option<usize> where Self: Sized;
    fn get_current_pv(&mut self, tree_node: &TreeNode<S, Self>) -> Vec<(<S as State>::Ply, TreeNode<S, Self>)> where Self: Sized;

    fn split(&self) -> Self;
}

pub use self::puct::PuctSelector;

mod puct;
