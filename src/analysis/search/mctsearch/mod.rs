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

use std::any::Any;
use std::f32;
use std::fmt;
use std::ptr;

use analysis::Extrapolatable;
use analysis::search::{Analysis, Search};
use state::State;

use self::volatile::Volatile;
use self::worker::Worker;

pub struct MctSearchAnalysis<S> where
    S: State + Extrapolatable<<S as State>::Ply> {
    pub state: S,
    pub principal_variation: Vec<<S as State>::Ply>,
}

pub struct MctSearch<S, P, Q> where
    S: 'static + State + Extrapolatable<<S as State>::Ply>,
    P: 'static + Selector<S>,
    Q: Simulator<S> {
    selector: P,
    simulator: Q,

    limit: f32,

    worker_count: usize,
    workers: Vec<Worker<S, P>>,
    persist: bool,

    tree: *mut TreeNode<S, P>,
}

impl<S, P, Q> MctSearch<S, P, Q> where
    S: 'static + State + Extrapolatable<<S as State>::Ply>,
    P: 'static + Selector<S>,
    Q: Simulator<S> {
    pub fn new(selector: P, simulator: Q) -> MctSearch<S, P, Q> {
        MctSearch {
            selector: selector,
            simulator: simulator,
            limit: 20.0,
            worker_count: 1,
            workers: Vec::new(),
            persist: false,
            tree: ptr::null_mut(),
        }
    }

    //fn initialize
}

#[derive(Clone)]
pub struct TreeNode<S, P> where
    S: State + Extrapolatable<<S as State>::Ply>,
    P: Selector<S> {
    pub policy_data: <P as Selector<S>>::Data,

    pub child_count: Volatile<u16>,
    pub children: Volatile<*mut (<S as State>::Ply, TreeNode<S, P>)>,

    value: Volatile<f32>,
    visits: Volatile<u32>,
    virtual_loss: Volatile<u8>,

    terminal: Volatile<bool>, // XXX Needed?
    proven: Volatile<bool>,
}

impl<S, P> TreeNode<S, P> where
    S: State + Extrapolatable<<S as State>::Ply>,
    P: Selector<S> {
    fn new() -> TreeNode<S, P> {
        TreeNode {
            policy_data: <P as Selector<S>>::Data::default(),
            child_count: Volatile::new(0),
            children: Volatile::new(ptr::null_mut()),
            value: Volatile::new(0.0),
            visits: Volatile::new(0),
            virtual_loss: Volatile::new(0),
            terminal: Volatile::new(false),
            proven: Volatile::new(false),
        }
    }

    pub fn get_value(&self) -> f32 {
        self.value.read()
    }

    pub fn set_value(&mut self, value: f32, count: u32) {
        self.value.write(value);
        self.visits.write(count);
    }

    pub fn add_value(&mut self, value: f32, count: u32) {
        let visits = self.visits.read() + count;
        let old_value = self.value.read();
        self.value.write(old_value + count as f32 * (value - old_value) / visits as f32);
        self.visits.write(visits);
    }

    pub fn remove_value(&mut self, value: f32, count: u32) {
        let mut visits = self.visits.read();
        if visits > count {
            visits -= count;
            let old_value = self.value.read();
            self.value.write(old_value - count as f32 * (value - old_value) / visits as f32);
            self.visits.write(visits);
        } else {
            self.value.write(0.0);
            self.visits.write(0);
        }
    }

    pub fn get_visits(&self) -> u32 {
        self.visits.read()
    }

    pub fn get_virtual_loss(&self) -> u8 {
        self.virtual_loss.read()
    }

    pub fn add_virtual_loss(&mut self, count: u8) {
        self.virtual_loss.update(|v| *v += count);
    }

    pub fn remove_virtual_loss(&mut self, count: u8) {
        let virtual_loss = self.virtual_loss.read();
        if virtual_loss > count {
            self.virtual_loss.write(virtual_loss - count);
        } else {
            self.virtual_loss.write(0);
        }
    }

    pub fn mark_invalid(&mut self) {
        self.value.write(f32::MIN);
        self.visits.write(0);
    }

    pub fn is_invalid(&self) -> bool {
        self.value.read() == f32::MIN
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal.read()
    }

    pub fn set_terminal(&mut self, value: bool) {
        self.terminal.write(value);
    }

    pub fn is_proven(&self) -> bool {
        self.proven.read()
    }

    pub fn set_proven(&mut self, value: bool) {
        self.proven.write(value);
    }
}

pub use self::selector::{PuctSelector, Selector};
pub use self::simulator::{RandomSimulator, Simulator};

mod jkiss32;
mod selector;
mod simulator;
mod volatile;
mod worker;
