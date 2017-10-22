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

use std::mem;
use std::ops::Deref;
use std::ptr;
use std::slice;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use analysis::Extrapolatable;
use analysis::search::mctsearch::{Selector, Simulator, TreeNode};
use resolution::Resolution;
use state::State;

static BUFFER_SIZE: usize = 4_000_000;
static EXPAND_THRESHOLD: u32 = 1;
static VIRTUAL_LOSS: u8 = 4;
static PLAYOUT_COUNT: u32 = 1;

pub struct Worker<S, P> where
    S: 'static + State + Extrapolatable<<S as State>::Ply>,
    P: 'static + Selector<S> {
    index: usize,
    thread: thread::JoinHandle<()>,
    interrupt: mpsc::Sender<Option<(S, SendPointer<TreeNode<S, P>>)>>,
}

fn pause() {
    thread::sleep(Duration::from_millis(250));
}

fn print_tree<S, P>(root: *const TreeNode<S, P>, depth: Option<usize>) where
    S: 'static + State + Extrapolatable<<S as State>::Ply> + Send,
    P: 'static + Selector<S> {
    fn print_branch<S, P>(branch: &TreeNode<S, P>, number: Option<(usize, <S as State>::Ply)>, indent: usize, depth: Option<usize>) where
        S: 'static + State + Extrapolatable<<S as State>::Ply> + Send,
        P: 'static + Selector<S> {
        println!("{: <1$}{2} {3:0.6}, {4}", "", indent, if let Some((number, ref ply)) = number {
            format!("Child {}, {}:", number, ply)
        } else {
            String::from("Root:")
        }, branch.get_value(), branch.get_visits());

        if depth.is_none() || depth.unwrap() > 0 {
            if branch.children.read().is_null() {
                println!("{: <1$}No children", "", indent + 2);
            } else {
                let mut visited_children = 0;
                for (index, &(ref ply, ref child)) in unsafe { slice::from_raw_parts(branch.children.read(), branch.child_count.read() as usize) }.iter().enumerate() {
                    if child.get_visits() != 0 {
                        visited_children += 1;
                        print_branch(child, Some((index, ply.clone())), indent + 2, Some(depth.unwrap() - 1));
                    }
                }

                if branch.child_count.read() - visited_children > 0 {
                    println!("{: <1$}{2} unvisited children", "", indent + 2, branch.child_count.read() - visited_children);
                }
            }
        }

        println!();
    }

    let root = unsafe { &*root };
    print_branch(root, None, 2, depth);
}

impl<S, P> Worker<S, P> where
    S: 'static + State + Extrapolatable<<S as State>::Ply> + Send,
    P: 'static + Selector<S> {
    pub fn new<Q>(index: usize, mut selector: P, mut simulator: Q) -> Worker<S, P> where
        Q: 'static + Simulator<S> {
        let worker = thread::Builder::new().name(format!("Worker {}", index));

        let (interrupt, receiver) = mpsc::channel::<Option<(S, SendPointer<TreeNode<S, P>>)>>();
        let thread = worker.spawn(move || {
            let mut buffer_vec: Vec<(<S as State>::Ply, TreeNode<S, P>)> = Vec::with_capacity(BUFFER_SIZE);
            let mut buffer = buffer_vec.as_mut_ptr();

            let mut expansion_buffer = Vec::new();

            let mut proven_count = 0;

            println!("{}: Waiting for root.", thread::current().name().unwrap());

            while let Ok(root) = receiver.recv() {
                if root.is_none() {
                    continue;
                }

                println!("{}: Accepted root.", thread::current().name().unwrap());

                let (root_state, SendPointer(root_node)) = root.unwrap();
                let mut visited_nodes = Vec::new();

                'sample: loop {
                    //println!("{}: Beginning sample.", thread::current().name().unwrap()); pause();
                    //println!("PV: [{}]", selector.get_current_pv(unsafe { &*root_node }).iter().map(|&(ref ply, _)| format!("{}", ply)).collect::<Vec<_>>().join(", "));
                    //print_tree(root_node, None);
                    visited_nodes.clear();

                    // Descend the tree
                    let mut current_state = root_state.clone();
                    let mut current_node = unsafe { &mut *root_node };

                    //println!("{}: Beginning descend.", thread::current().name().unwrap()); pause();
                    'descend: loop {
                        if current_node.is_proven() {
                            //println!("{}: Node is proven.", thread::current().name().unwrap()); pause();
                            break 'descend;
                        } else if let Some(index) = selector.select_descendant_index(current_node) {
                            //pause();
                            // Add a virtual loss to discourage other threads from following us
                            current_node.add_virtual_loss(VIRTUAL_LOSS);

                            let ref mut pair = unsafe { slice::from_raw_parts_mut(current_node.children.read(), current_node.child_count.read() as usize) }[index];

                            //println!("{}: Selected Child {}, {}.", thread::current().name().unwrap(), index, pair.0);

                            // Advance the state
                            let next_ply = &pair.0;
                            if current_state.execute_ply(Some(next_ply)).is_err() {
                                //println!("{}: Bad ply. Reselecting.", thread::current().name().unwrap()); pause();
                                let bad_node = &mut pair.1;
                                bad_node.mark_invalid();
                                continue 'descend;
                            }

                            visited_nodes.push(current_node as *mut TreeNode<S, P>);
                            current_node = &mut pair.1;
                        } else if current_node.get_visits() >= EXPAND_THRESHOLD {
                            //println!("{}: Expanding node.", thread::current().name().unwrap()); pause();
                            // Expand the current node
                            expansion_buffer.clear();
                            current_state.extrapolate_into(&mut expansion_buffer);
                            let child_count = expansion_buffer.len();
                            if buffer_vec.len() + child_count < buffer_vec.capacity() {
                                for ply in &expansion_buffer {
                                    buffer_vec.push((ply.clone(), TreeNode::new()));
                                }
                            } else {
                                panic!("{} / {}, Not enough memory for node expansion!", buffer_vec.len(), buffer_vec.capacity());
                            }
                            current_node.child_count.write(child_count as u16);
                            current_node.children.write(buffer);
                            buffer = unsafe { buffer.offset(child_count as isize) };
                        } else {
                            // Simulate from the current node
                            break 'descend;
                        }
                    }
                    //println!("{}: Exiting descend.", thread::current().name().unwrap()); pause();

                    //println!("{}: Assessing newly proven.", thread::current().name().unwrap()); pause();
                    let newly_proven = if current_node.get_visits() > 0 {
                        None
                    } else if let Some(resolution) = current_state.check_resolution() {
                        //println!("{}: This node is newly proven!", thread::current().name().unwrap()); pause();
                        // This will only happen if this is the first visit to the node.  Repeat visits would be caught by the above condition.
                        current_node.set_terminal(true);
                        current_node.set_proven(true);
                        if let Some(winner) = resolution.get_winner() {
                            Some(if current_state.get_ply_count() % 2 == winner as usize {
                                // A win here doesn't go to current_state, but the one before it
                                -1.0
                            } else {
                                1.0
                            })
                        } else {
                            // Draw
                            Some(0.0)
                        }
                    } else {
                        None
                    };

                    // Simulate from the leaf
                    let value = if !current_node.is_proven() {
                        //println!("{}: Beginning simulation.", thread::current().name().unwrap()); pause();
                        // Simulate
                        let mut total_value = 0.0;
                        for i in 0..PLAYOUT_COUNT {
                            //println!("{}: Beginning playout {} / {}.", thread::current().name().unwrap(), i, PLAYOUT_COUNT); pause();
                            let value = simulator.simulate(&current_state);
                            //println!("{}: Ending playout {} with value {:.2}.", thread::current().name().unwrap(), i, value);
                            total_value += value;
                        }
                        //println!("{}: Ending simulation.", thread::current().name().unwrap()); pause();
                        let value = total_value / PLAYOUT_COUNT as f32;
                        current_node.add_value(value, PLAYOUT_COUNT);
                        value
                    } else {
                        //println!("{}: \"Simulating\".", thread::current().name().unwrap()); pause();
                        // "Simulate"
                        let value = newly_proven.unwrap_or(current_node.get_value());
                        let visits = current_node.get_visits() + PLAYOUT_COUNT;
                        current_node.set_value(value, visits);
                        value
                    };

                    //println!("{}: Updating tree with value {:.2}.", thread::current().name().unwrap(), value); pause();
                    // Update the tree with the results
                    backpropagate_results(&visited_nodes, value, PLAYOUT_COUNT);
                    if newly_proven.is_some() {
                        backpropagate_proven(&visited_nodes);
                    }
                    //println!("{}: Ending sample.", thread::current().name().unwrap()); pause();
                }
            }
        }).unwrap();

        Worker {
            index: index,
            thread: thread,
            interrupt: interrupt,
        }
    }

    pub fn update_root(&mut self, root_state: S, root_node: *mut TreeNode<S, P>) {
        self.interrupt.send(Some((root_state, SendPointer(root_node)))).ok();
    }
}

fn backpropagate_proven<S, P>(ancestors: &[*mut TreeNode<S, P>]) where
    S: 'static + State + Extrapolatable<<S as State>::Ply>,
    P: 'static + Selector<S> {
    for ancestor in ancestors {
        let mut ancestor = unsafe { &mut **ancestor };

        let mut proven = Some(1.0);
        for &(_, ref child) in unsafe { slice::from_raw_parts(ancestor.children.read(), ancestor.child_count.read() as usize) } {
            if !child.is_proven() {
                proven = None;
            } else {
                let child_value = child.get_value();
                if child_value == 1.0 {
                    proven = Some(-1.0);
                    break;
                } else if child_value == 0.0 && proven.is_some() { // The opponent will take a draw over a loss
                    proven = Some(0.0);
                }
            }
        }

        if let Some(value) = proven {
            let visits = ancestor.get_visits();
            ancestor.set_value(value, visits);
            ancestor.set_proven(true);
        } else {
            break;
        }
    }
}

fn backpropagate_results<S, P>(ancestors: &[*mut TreeNode<S, P>], mut value: f32, count: u32) where
    S: 'static + State + Extrapolatable<<S as State>::Ply>,
    P: 'static + Selector<S> {
    for ancestor in ancestors.iter().rev() {
        let mut ancestor = unsafe { &mut **ancestor };

        value *= -1.0;

        if !ancestor.is_proven() {
            ancestor.add_value(value, count);
        } else {
            value = ancestor.get_value();
            let visits = ancestor.get_visits() + count;
            ancestor.set_value(value, visits);
        }
        ancestor.remove_virtual_loss(VIRTUAL_LOSS);
    }
}

struct SendPointer<T>(*mut T);

unsafe impl<T> Send for SendPointer<T> { }

impl<T> Deref for SendPointer<T> {
    type Target = *mut T;

    fn deref(&self) -> &*mut T {
        &self.0
    }
}

#[cfg(test)]
mod test {
    use analysis::search::mctsearch::*;
    use impls::tak;
    //use impls::tic_tac_toe;
    use super::*;

    #[test]
    fn test_worker() {
        let mut selector = PuctSelector::new();
        let mut simulator = RandomSimulator::<tak::State>::new();
        //let mut simulator = RandomSimulator::<tic_tac_toe::Board>::new();
        let mut worker_0 = Worker::new(0, selector.split(), simulator.split());
        let mut worker_1 = Worker::new(1, selector.split(), simulator.split());
        let mut worker_2 = Worker::new(2, selector.split(), simulator.split());
        let mut worker_3 = Worker::new(3, selector.split(), simulator.split());
        //let mut worker_4 = Worker::new(4, selector.split(), simulator.split());
        //let mut worker_5 = Worker::new(5, selector.split(), simulator.split());

        pause();

        let root_state = tak::State::new(5);
        //let root_state = tic_tac_toe::Board::new();
        let mut root_node = TreeNode::<tak::State, PuctSelector<tak::State>>::new();
        //let mut root_node = TreeNode::<tic_tac_toe::Board, PuctSelector<tic_tac_toe::Board>>::new();

        {
            let root_node = SendPointer(&mut root_node as *mut TreeNode<tak::State, PuctSelector<tak::State>>);
            //let root_node = SendPointer(&mut root_node as *mut TreeNode<tic_tac_toe::Board, PuctSelector<tic_tac_toe::Board>>);
            thread::spawn(move || {
                let mut last_visits = 0;

                let mut root_node = unsafe { &**root_node };

                loop {
                    println!("PV: [{}]", selector.get_current_pv(root_node).iter().map(|&(ref ply, _)| format!("{}", ply)).collect::<Vec<_>>().join(", "));
                    println!("Root: {:.6}, {}, {:.0}/s", root_node.get_value(), root_node.get_visits(), (root_node.get_visits() - last_visits) as f32 / 2.0);
                    print_tree(root_node as *const TreeNode<tak::State, PuctSelector<tak::State>>, Some(1));
                    //print_tree(root_node as *const TreeNode<tic_tac_toe::Board, PuctSelector<tic_tac_toe::Board>>, Some(1));
                    last_visits = root_node.get_visits();
                    thread::sleep(Duration::from_secs(2));
                }
            });
        }

        worker_0.update_root(root_state.clone(), &mut root_node as *mut TreeNode<tak::State, PuctSelector<tak::State>>);
        worker_1.update_root(root_state.clone(), &mut root_node as *mut TreeNode<tak::State, PuctSelector<tak::State>>);
        worker_2.update_root(root_state.clone(), &mut root_node as *mut TreeNode<tak::State, PuctSelector<tak::State>>);
        worker_3.update_root(root_state.clone(), &mut root_node as *mut TreeNode<tak::State, PuctSelector<tak::State>>);
        //worker_4.update_root(root_state.clone(), &mut root_node as *mut TreeNode<tak::State, PuctSelector<tak::State>>);
        //worker_5.update_root(root_state.clone(), &mut root_node as *mut TreeNode<tak::State, PuctSelector<tak::State>>);

        worker_0.thread.join();
    }
}
