#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::collections::{BTreeMap, HashMap};
use std::ops::CoroutineState;

use nohash_hasher::{IntMap, IntSet};
use petgraph::algo::tarjan_scc;
use petgraph::graph::NodeIndex;
use petgraph::prelude::StableGraph;
use petgraph::stable_graph::StableUnGraph;
use petgraph::Undirected;
use rapidhash::fast::RapidHashSet;
use rapidhash::v3::rapidhash_v3;
use serde::Deserialize;

use crate::generator_bridge::BridgeGenerator;
use crate::generator_weighted_combo::WeightedRangeComboGenerator;
use crate::helpers_common::sort_pair;
use crate::idtree::IDTree;

const SUPER_ROOT: usize = 99_999;
const DO_DBG: bool = false;

pub type ExplorationGraphData = BTreeMap<usize, ExplorationNodeData>;

#[derive(Debug, Deserialize, Clone)]
pub struct ExplorationNodeData {
    pub waypoint_key: usize,
    pub need_exploration_point: usize,
    pub is_base_town: bool,
    pub link_list: Vec<usize>,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Clone)]
pub struct DynamicState {
    combo_gen_direction: bool,
    has_super_terminal: bool,
    idtree: IDTree,
    idtree_active_indices: IntSet<usize>,
    terminal_to_root: IntMap<usize, usize>,
    untouchables: IntSet<usize>,
    connected_pairs: RapidHashSet<(usize, usize)>,
    bridge_affected_base_towns: IntSet<usize>,
    bridge_affected_indices: IntSet<usize>,
    bridge_affected_terminals: RapidHashSet<(usize, usize)>,
}

/// Solves Node-Weighted Steiner Forest using primal-dual and bridge heuristics.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "python", pyclass(unsendable))]
pub struct NodeRouter {
    // Used in the _combinations_with_weight_range_generator to limit combo generation.
    max_node_weight: usize,
    // Bridge heuristics
    // min 350 => 1.5x the max iter of test cases for 2-direction pass
    max_removal_attempts: usize,
    // Used for controlling the sort order of removal set generator.
    combo_gen_direction: bool, // 'true' for descending
    // Used for controlling special case handlers for super terminals
    has_super_terminal: bool,

    // Static Mappings
    base_towns: IntSet<usize>,
    index_to_neighbors: IntMap<usize, IntSet<usize>>,
    index_to_waypoint: IntMap<usize, usize>,
    index_to_weight: IntMap<usize, usize>,
    waypoint_to_index: IntMap<usize, usize>,

    // The main workhorse of the PD Approximation
    ref_graph: StableUnGraph<usize, usize>,

    // The main workhorse of the Bridge Heuristic
    idtree: IDTree,
    idtree_active_indices: IntSet<usize>,

    // Contains all terminal, root pairs
    terminal_to_root: IntMap<usize, usize>,
    // Contains all terminals, fixed roots, leaf terminal parents
    untouchables: IntSet<usize>,
    // Used in approximation to reduce violated set connectivity checks.
    connected_pairs: RapidHashSet<(usize, usize)>,
    // Used in reverse deletion to filter deletion and connection checks.
    bridge_affected_base_towns: IntSet<usize>,
    bridge_affected_indices: IntSet<usize>,
    bridge_affected_terminals: RapidHashSet<(usize, usize)>,
}

#[cfg(feature = "python")]
#[pymethods]
impl NodeRouter {
    #[new]
    pub fn py_new(exploration_json: &str) -> NodeRouter {
        use std::str::FromStr;
        let str_map: BTreeMap<String, ExplorationNodeData> =
            serde_json::from_str(exploration_json).unwrap();
        let exploration_data: ExplorationGraphData = str_map
            .into_iter()
            .map(|(k, v)| usize::from_str(&k).map(|k| (k, v)).unwrap())
            .collect();
        Self::new(&exploration_data)
    }
    #[pyo3(name = "solve_for_terminal_pairs")]
    pub fn py_solve_for_terminal_pairs(
        &mut self,
        terminal_pairs: Vec<(usize, usize)>,
    ) -> Vec<usize> {
        self.solve_for_terminal_pairs(terminal_pairs)
    }
}

impl NodeRouter {
    pub fn new(exploration_data: &ExplorationGraphData) -> Self {
        let mut max_node_weight = 0;
        let mut base_towns: IntSet<usize> = IntSet::default();
        let mut waypoint_to_index: IntMap<usize, usize> = IntMap::default();
        let mut index_to_waypoint: IntMap<usize, usize> = IntMap::default();
        let mut index_to_weight: IntMap<usize, usize> = IntMap::default();
        let mut index_to_neighbors: IntMap<usize, IntSet<usize>> = IntMap::default();
        let mut ref_graph = StableUnGraph::<usize, usize>::default();

        // First we populate the translation mappings...
        if DO_DBG {
            println!("Populating translation mappings...");
        }
        for (i, (&waypoint_key, node_data)) in exploration_data.iter().enumerate() {
            let ref_i = ref_graph.add_node(node_data.need_exploration_point);
            assert_eq!(i, ref_i.index());
            index_to_waypoint.insert(i, waypoint_key);
            waypoint_to_index.insert(waypoint_key, i);
            index_to_waypoint.insert(i, waypoint_key);
            if node_data.is_base_town {
                base_towns.insert(i);
            }
            index_to_weight.insert(i, node_data.need_exploration_point);
            if node_data.need_exploration_point > max_node_weight {
                max_node_weight = node_data.need_exploration_point;
            }
        }
        // Then we populate the neighbor cache...
        if DO_DBG {
            println!("Populating neighbor cache...");
        }
        for (waypoint_key, node_data) in exploration_data.iter() {
            let waypoint_idx = waypoint_to_index.get(waypoint_key).unwrap();
            index_to_neighbors.insert(*waypoint_idx, IntSet::default());
            for &neighbor in &node_data.link_list {
                index_to_neighbors
                    .get_mut(waypoint_idx)
                    .unwrap()
                    .insert(*waypoint_to_index.get(&neighbor).unwrap());
            }
        }

        // Then we generate all of the edges for the ref_graph...
        if DO_DBG {
            println!("Generating ref_graph edges...");
        }
        index_to_neighbors
            .iter()
            .flat_map(|(i, neighbors)| {
                neighbors
                    .iter()
                    .map(move |&neighbor| sort_pair(*i as u32, neighbor as u32))
            })
            .for_each(|(u, v)| {
                _ = ref_graph.add_edge(
                    NodeIndex::new(u as usize),
                    NodeIndex::new(v as usize),
                    *index_to_weight.get(&(v as usize)).unwrap(),
                );
            });

        // Lastly, initialize the IDTree...
        if DO_DBG {
            println!("Initializing IDTree with isolated nodes...");
        }

        let mut initialization_adj_dict = IntMap::default();
        for i in ref_graph.node_indices() {
            initialization_adj_dict.insert(i.index(), IntSet::default());
        }
        let idtree = IDTree::new(&initialization_adj_dict);
        let idtree_active_indices = IntSet::default();

        Self {
            max_node_weight,           // static
            max_removal_attempts: 350, // static
            combo_gen_direction: true,
            has_super_terminal: false,
            base_towns,         // static
            index_to_neighbors, // static
            index_to_waypoint,  // static
            index_to_weight,    // static
            waypoint_to_index,  // static
            ref_graph,          // static
            idtree,
            idtree_active_indices,
            terminal_to_root: IntMap::default(), // static per solve run
            untouchables: IntSet::default(),     // static per solve run
            connected_pairs: RapidHashSet::default(),
            bridge_affected_base_towns: IntSet::default(),
            bridge_affected_indices: IntSet::default(),
            bridge_affected_terminals: RapidHashSet::default(),
        }
    }

    fn clear_dynamic_state(&mut self) {
        self.combo_gen_direction = true;
        self.has_super_terminal = false;
        for node in self.idtree.active_nodes() {
            self.idtree.isolate_node(node);
        }
        self.idtree_active_indices.clear();
        self.terminal_to_root.clear();
        self.untouchables.clear();
        self.connected_pairs.clear();
        self.bridge_affected_base_towns.clear();
        self.bridge_affected_indices.clear();
        self.bridge_affected_terminals.clear();
    }

    pub fn get_dynamic_state(&self) -> DynamicState {
        DynamicState {
            combo_gen_direction: self.combo_gen_direction,
            has_super_terminal: self.has_super_terminal,
            idtree: self.idtree.clone(),
            idtree_active_indices: self.idtree_active_indices.clone(),
            terminal_to_root: self.terminal_to_root.clone(),
            untouchables: self.untouchables.clone(),
            connected_pairs: self.connected_pairs.clone(),
            bridge_affected_base_towns: self.bridge_affected_base_towns.clone(),
            bridge_affected_indices: self.bridge_affected_indices.clone(),
            bridge_affected_terminals: self.bridge_affected_terminals.clone(),
        }
    }

    pub fn restore_dynamic_state(&mut self, state: DynamicState) {
        self.combo_gen_direction = state.combo_gen_direction.clone();
        self.has_super_terminal = state.has_super_terminal.clone();
        self.idtree = state.idtree.clone();
        self.idtree_active_indices = state.idtree_active_indices.clone();
        self.terminal_to_root = state.terminal_to_root.clone();
        self.untouchables = state.untouchables.clone();
        self.connected_pairs = state.connected_pairs.clone();
        self.bridge_affected_base_towns = state.bridge_affected_base_towns.clone();
        self.bridge_affected_indices = state.bridge_affected_indices.clone();
        self.bridge_affected_terminals = state.bridge_affected_terminals.clone();
    }

    fn init_terminal_pairs(&mut self, terminal_pairs: Vec<(usize, usize)>) {
        for (t, r) in terminal_pairs {
            let t_idx = *self.waypoint_to_index.get(&t).unwrap();
            let r_idx = if r == SUPER_ROOT {
                self.has_super_terminal = true;
                SUPER_ROOT
            } else {
                *self.waypoint_to_index.get(&r).unwrap()
            };
            self.terminal_to_root.insert(t_idx, r_idx);
        }
    }

    fn idtree_weight(&self) -> (IntSet<usize>, usize) {
        let active_nodes = self.idtree_active_indices.clone();
        let total_weight: usize = active_nodes.iter().map(|&i| self.index_to_weight[&i]).sum();

        if DO_DBG {
            println!(
                "  pass completed... terminals connected: {:?} weight: {}",
                self.terminal_pairs_connected(),
                total_weight
            );
        }

        (active_nodes, total_weight)
    }

    /// Induce subgraph from self.ref_graph using node indices
    fn ref_subgraph_stable(&self, indices: &IntSet<usize>) -> StableGraph<(), usize, Undirected> {
        let subgraph = self.ref_graph.filter_map(
            |node_idx, _| {
                if indices.contains(&node_idx.index()) {
                    Some(())
                } else {
                    None
                }
            },
            |_, edge_idx| Some(*edge_idx),
        );
        subgraph
    }

    /// Solve for a list of terminal pairs [(terminal, root), ...]
    /// where root is an exploration data waypoint with attribute `is_base_town`
    /// or `99999` to indicate a super-terminal that can connect to any base town.
    pub fn solve_for_terminal_pairs(&mut self, terminal_pairs: Vec<(usize, usize)>) -> Vec<usize> {
        if DO_DBG {
            println!("solve for terminal pairs: {:?}", terminal_pairs);
        }

        self.clear_dynamic_state();
        self.init_terminal_pairs(terminal_pairs);
        self.generate_untouchables();

        let ordered_removables = self.approximate();
        let post_approximation_state = self.get_dynamic_state();

        if DO_DBG {
            println!("Forward pass...");
        }
        self.bridge_heuristics(&mut ordered_removables.clone());
        let (fwd_indices, fwd_weight) = self.idtree_weight();

        self.restore_dynamic_state(post_approximation_state);
        self.combo_gen_direction = false;

        if DO_DBG {
            println!("Reverse pass...");
        }
        self.bridge_heuristics(&mut ordered_removables.clone());
        let (rev_indices, rev_weight) = self.idtree_weight();

        // Convert idtree active nodes of winning pass to waypoints and return
        if DO_DBG {
            println!("Translating winning results...");
        }
        let winner = if fwd_weight < rev_weight {
            fwd_indices
        } else {
            rev_indices
        };

        winner.iter().map(|&i| self.index_to_waypoint[&i]).collect()
    }

    /// Set of all terminals, fixed roots and leaf terminal parents
    fn generate_untouchables(&mut self) {
        if DO_DBG {
            println!("Generating untouchables...");
        }

        self.untouchables.clear();
        self.untouchables.extend(self.terminal_to_root.keys());
        self.untouchables.extend(self.terminal_to_root.values());
        self.untouchables.remove(&SUPER_ROOT);

        // Add unambigous connected nodes (degree 1)...
        for &node in self.untouchables.clone().iter() {
            if let Some(neighbors) = self.index_to_neighbors.get(&node) {
                if neighbors.len() == 1 {
                    self.untouchables.extend(neighbors);
                }
            }
        }
        if DO_DBG {
            let untouchable_waypoints = self
                .untouchables
                .iter()
                .map(|&i| self.index_to_waypoint[&i])
                .collect::<Vec<_>>();
            println!(
                "  untouchables (w/unambiguous connections): {:?}",
                untouchable_waypoints
            );
        }
    }

    // MARK: PD Approximation

    /// Solves the routing problem and returns a list of active nodes in solution
    fn approximate(&mut self) -> Vec<usize> {
        if DO_DBG {
            println!("Starting PD Approximation...");
        }

        let mut x = self.untouchables.clone();
        if DO_DBG {
            let mut x_waypoints = x
                .iter()
                .map(|&i| self.index_to_waypoint[&i])
                .collect::<Vec<_>>();
            x_waypoints.sort();
            println!("  initializing x to {:?}", x_waypoints);
        }

        if self.has_super_terminal {
            self.augment_superterminal_roots(&mut x);
            if DO_DBG {
                let mut x_waypoints = x
                    .iter()
                    .map(|&i| self.index_to_waypoint[&i])
                    .collect::<Vec<_>>();
                x_waypoints.sort();
                println!("  augmented x to {:?}", x_waypoints);
            }
        }

        let (x, mut ordered_removables) = self.primal_dual_approximation(x);
        if DO_DBG {
            assert!(self.terminal_pairs_connected());
            println!("  ... ok")
        }
        self.populate_idtree(&x);

        // Ordered removables are in temporal order of going tight, sub ordered structurally
        // by sorting by waypoint key.  When removing the nodes they should be processed in
        // reverse order to facilitate the removal of the latest nodes to 'go tight' first.
        // The list is reversed here and processed in forward order thoughout the remainder
        // of the algorithm and bridge heuristic processing.
        ordered_removables = ordered_removables.iter().rev().cloned().collect();

        // remove_removables is setup to primarily handle 'bridged' components in the Bridge
        // Heuristic. To simplify the code the bridge related variables are set here to
        // cover all removables, terminals and base towns in the graph.
        self.update_bridge_affected_nodes(self.idtree_active_indices.clone());

        if DO_DBG {
            println!("Removing removables...");
        }
        let (freed, _freed_edges) = self.remove_removables(&ordered_removables);
        if DO_DBG {
            println!("  freed {} nodes", freed.len());
            assert!(self.terminal_pairs_connected());
            println!("  ... ok")
        }

        self.idtree_active_indices.retain(|&v| !freed.contains(&v));
        ordered_removables.retain(|&v| !freed.contains(&v));
        if DO_DBG {
            println!(
                "Result: num active nodes is {} num active indices is {} num removables is {}",
                self.idtree.active_nodes().len(),
                self.idtree_active_indices.len(),
                ordered_removables.len()
            );
        }

        ordered_removables
    }

    /// Augments initial approximation set with super-terminal potential roots when
    /// that root is nearer than any existing rooted node in the current approximation set.
    fn augment_superterminal_roots(&mut self, x: &mut IntSet<usize>) {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut working_roots = self.terminal_to_root.clone();
        let mut pending_super_terminals: IntSet<usize> = working_roots
            .iter()
            .filter_map(|(&t, &r)| if r == SUPER_ROOT { Some(t) } else { None })
            .collect();

        // Processes the super terminals such that the super-terminal nearest a fixed
        // terminal or any base town is completed first and then becomes available to
        // be a potential root until all super terminals have a potential root in x.
        while !pending_super_terminals.is_empty() {
            // (super_terminal, target_node, cost)
            let mut super_terminal_distances: Vec<(usize, usize, usize)> = Vec::new();

            for &terminal in &pending_super_terminals {
                let mut heap = BinaryHeap::new();
                let mut visited = IntSet::default();
                heap.push((Reverse(0), terminal));

                while let Some((Reverse(cost), node)) = heap.pop() {
                    if !visited.insert(node) {
                        continue;
                    }

                    if node != terminal {
                        let is_rooted = match working_roots.get(&node) {
                            Some(root) if root != &SUPER_ROOT => true,
                            None => self.base_towns.contains(&node),
                            _ => false,
                        };

                        if is_rooted {
                            super_terminal_distances.push((terminal, node, cost));
                            break;
                        }
                    }

                    for &neighbor in &self.index_to_neighbors[&node] {
                        if visited.contains(&neighbor) {
                            continue;
                        }
                        let next_cost = if x.contains(&neighbor) {
                            cost
                        } else {
                            cost + self.index_to_weight[&neighbor]
                        };
                        heap.push((Reverse(next_cost), neighbor));
                    }
                }
            }

            super_terminal_distances.sort_by_key(|&(_, _, cost)| cost);
            let (terminal, target, cost) = super_terminal_distances[0];
            x.insert(target);
            working_roots.insert(terminal, target);
            pending_super_terminals.remove(&terminal);

            if DO_DBG {
                println!(
                    "  super-terminal {} should connect to {} (cost = {})",
                    self.index_to_waypoint[&terminal], self.index_to_waypoint[&target], cost
                );
            }
        }
    }

    /// Node Weighted Primal Dual Approximation (Demaine et al.)
    fn primal_dual_approximation(&mut self, mut x: IntSet<usize>) -> (IntSet<usize>, Vec<usize>) {
        // The main loop operations and frontier node calculations are set based.
        // The violated sets identification requires subgraphing the ref_graph and running
        // connected_components. The loop usually only iterates a half dozen times.
        if DO_DBG {
            let mut x_waypoints = x
                .iter()
                .map(|&i| self.index_to_waypoint[&i])
                .collect::<Vec<_>>();
            x_waypoints.sort();
            println!(
                "Running primal-dual approximation for x of length {}: {:?}...",
                x.len(),
                x_waypoints
            );
        }

        let mut y = vec![0; self.ref_graph.node_count()];
        let mut ordered_removables = Vec::new();

        while let Some(violated_sets) = self.violated_sets(&x) {
            let num_violated_sets = violated_sets.len();
            let violated: IntSet<_> = violated_sets.into_iter().flatten().collect();

            if DO_DBG {
                let mut violated_waypoints = violated
                    .iter()
                    .map(|&i| self.index_to_waypoint[&i])
                    .collect::<Vec<_>>();
                violated_waypoints.sort();
                println!(
                    "  violated waypoints from {} components: {:?}",
                    num_violated_sets, violated_waypoints
                );
            }

            let frontier_nodes = self.find_frontier_nodes(&violated, None);

            for &v in &frontier_nodes {
                y[v] += 1;
            }

            let tight_nodes: Vec<_> = frontier_nodes
                .iter()
                .cloned()
                .filter(|&v| y[v] >= self.index_to_weight[&v])
                .collect();

            if DO_DBG {
                let tight_node_waypoints: Vec<usize> = tight_nodes
                    .iter()
                    .map(|&i| self.index_to_waypoint[&i])
                    .collect();
                println!("tight_node_waypoints {:?}", tight_node_waypoints);
            }

            x.extend(&tight_nodes);
            ordered_removables.extend(self.sort_by_weights(&tight_nodes, None));
        }

        (x, ordered_removables)
    }

    /// Returns connected components violating connectivity constraints.
    fn violated_sets(&mut self, x: &IntSet<usize>) -> Option<Vec<IntSet<usize>>> {
        if DO_DBG {
            println!(
                "  checking for violated sets for x of length {}...",
                x.len()
            );
        }

        // Compute connected components (undirected graph)
        let subgraph = self.ref_subgraph_stable(x);
        let components: Vec<IntSet<usize>> = tarjan_scc(&subgraph)
            .into_iter()
            .map(|comp| comp.iter().map(|nidx| nidx.index()).collect())
            .collect();
        if DO_DBG {
            let components_as_waypoints = components
                .iter()
                .map(|comp| {
                    comp.iter()
                        .map(|&i| self.index_to_waypoint[&i])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            println!("components_as_waypoints {:?}", components_as_waypoints);
        }

        let mut violated: Vec<IntSet<usize>> = Vec::new();
        for cc in &components {
            // Since the pd approximation is additive we can safely avoid duplicate checks.
            let tmp_connected_pairs = self.connected_pairs.clone();
            let active_terminals = self
                .terminal_to_root
                .iter()
                .filter(|p| !tmp_connected_pairs.contains(&(*p.0, *p.1)));

            for (&terminal, &root) in active_terminals {
                let terminal_in_cc = cc.contains(&terminal);

                let root_in_cc = if root == SUPER_ROOT {
                    cc.intersection(&self.base_towns).next().is_some()
                } else {
                    cc.contains(&root)
                };

                if !terminal_in_cc && !root_in_cc {
                    continue;
                }
                if terminal_in_cc && root_in_cc {
                    self.connected_pairs.insert((terminal, root));
                } else {
                    violated.push(cc.clone());
                    break;
                }
            }
        }

        if violated.is_empty() {
            None
        } else {
            Some(violated)
        }
    }

    /// Finds and returns nodes not in settlement with neighbors in settlement.
    fn find_frontier_nodes(
        &self,
        settlement: &IntSet<usize>,
        min_degree: Option<usize>,
    ) -> IntSet<usize> {
        if DO_DBG {
            println!("Finding frontier nodes ...",);
        }
        let mut frontier = settlement
            .iter()
            .flat_map(|v| &self.index_to_neighbors[v])
            .filter(|n| !settlement.contains(n))
            .copied()
            .collect::<IntSet<_>>();

        if let Some(min_degree) = min_degree {
            frontier.retain(|v| self.index_to_neighbors[v].len() >= min_degree);
        }
        if DO_DBG {
            println!(
                "  found {} frontier nodes for settlement of size {}",
                frontier.len(),
                settlement.len()
            );
        }
        frontier
    }

    fn sort_by_weights(&self, numbers: &[usize], weights: Option<&[usize]>) -> Vec<usize> {
        // NOTE:
        // The sorting here was purely for deterministic testing while refactoring from networkx.
        // But amazingly enough this _dramatically_ improves results with PD!
        // Considering this is sorting by 'waypoint_key' it doesn't really represent a weight
        // that _should_ have anything to do with the component growth but since the keys
        // are somewhat geographically ordered the structural ordering of the bridge
        // and removal candidates during the bridge heuristics is altered.
        let effective_weights: Vec<usize> = match weights {
            Some(ws) => ws.to_vec(),
            None => numbers
                .iter()
                .map(|&i| self.index_to_waypoint[&i])
                .collect(),
        };

        let mut pairs: Vec<(usize, usize)> = effective_weights
            .into_iter()
            .zip(numbers.iter().cloned())
            .collect();

        pairs.sort_unstable();

        pairs.into_iter().map(|(_, number)| number).collect()
    }

    /// Populates the IDTree and initializes idtree_active_indices
    fn populate_idtree(&mut self, x: &IntSet<usize>) {
        if DO_DBG {
            println!("Populating IDTree...");
        }
        self.ref_subgraph_stable(x)
            .edge_indices()
            .for_each(|edge_idx| {
                let (u, v) = self.ref_graph.edge_endpoints(edge_idx).unwrap();
                self.idtree.insert_edge(u.index(), v.index());
            });
        self.idtree_active_indices = self.idtree.active_nodes().into_iter().collect();
    }

    /// Updates self._bridge_* variables with relevant bridged component nodes.
    fn update_bridge_affected_nodes(&mut self, affected_component: IntSet<usize>) {
        if DO_DBG {
            println!("Updating bridge affected nodes...");
        }
        self.bridge_affected_indices = affected_component;
        self.bridge_affected_terminals = self
            .terminal_to_root
            .iter()
            .filter(|p| self.bridge_affected_indices.contains(p.0))
            .map(|p| (*p.0, *p.1))
            .collect();
        self.bridge_affected_base_towns = self
            .base_towns
            .intersection(&self.bridge_affected_indices)
            .cloned()
            .collect();
    }

    /////
    // MARK: Connectivity Testing
    /////

    /// Attempt removals of each node in ordered_removables.
    ///
    /// NOTE: This function should only be entered after terminal pairs connected check succeeds.
    fn remove_removables(
        &mut self,
        ordered_removables: &Vec<usize>,
    ) -> (IntSet<usize>, Vec<(usize, usize)>) {
        if DO_DBG {
            println!(
                "Performing removals on {} added nodes...",
                ordered_removables.len()
            );
        }
        let mut freed = IntSet::default();
        let mut freed_edges = Vec::new();

        for &v in ordered_removables {
            if !self.bridge_affected_indices.contains(&v) {
                continue;
            }

            // Simulate removal by isolating the node
            let mut deleted_edges = Vec::new();
            for &u in self.index_to_neighbors[&v].intersection(&self.bridge_affected_indices) {
                if self.idtree.delete_edge(v, u) != -1 {
                    deleted_edges.push((v, u));
                }
            }
            if self.terminal_pairs_connected() {
                // Finalize removal
                self.bridge_affected_indices.remove(&v);
                self.bridge_affected_base_towns.remove(&v);
                freed.insert(v);
                freed_edges.extend(deleted_edges);
            } else {
                // Restore broken connectivity
                for &(v, u) in &deleted_edges {
                    self.idtree.insert_edge(v, u);
                }
            }
        }
        (freed, freed_edges)
    }

    /// Check all `bridge_affected_terminals` pair connectivity.
    fn terminal_pairs_connected(&self) -> bool {
        for &(terminal, root) in &self.bridge_affected_terminals {
            if !self.terminal_is_connected(terminal, root) {
                return false;
            }
        }
        true
    }

    /// Check terminal pair connectivity.
    fn terminal_is_connected(&self, terminal: usize, root: usize) -> bool {
        if root == SUPER_ROOT {
            self.bridge_affected_base_towns
                .iter()
                .any(|&b| self.idtree.query(terminal, b))
        } else {
            self.idtree.query(terminal, root)
        }
    }

    // MARK: Bridge Heuristic

    /// Bridge heuristic: find and utilize potential bridges to _increase_
    /// cycle counts and then identify removable non-articulation points
    /// that can improve the solution.
    fn bridge_heuristics(&mut self, ordered_removables: &mut Vec<usize>) {
        let mut incumbent_indices: IntSet<usize> = self.idtree_active_indices.clone();
        let mut seen_before_cache: IntSet<u64> = IntSet::default();

        let mut improved = true;
        while improved {
            improved = false;

            let bridge_generator =
                BridgeGenerator::new(self.ref_graph.clone(), self.index_to_neighbors.clone());
            let mut bridge_gen = bridge_generator.generate_bridges(incumbent_indices.clone());

            while let CoroutineState::Yielded(bridge) = bridge_gen.as_mut().resume(()) {
                let reisolate_bridge_nodes: Vec<usize> = bridge
                    .iter()
                    .filter(|&&v| !incumbent_indices.contains(&v))
                    .copied()
                    .collect();

                self.connect_bridge(&bridge);

                let Some(bridge_rooted_cycles) = self.bridge_rooted_cycles(&bridge) else {
                    self.idtree.isolate_nodes(reisolate_bridge_nodes);
                    self.idtree_active_indices = incumbent_indices.clone();
                    continue;
                };

                if self.was_seen_before(&bridge, &bridge_rooted_cycles, &mut seen_before_cache) {
                    self.idtree.isolate_nodes(reisolate_bridge_nodes);
                    self.idtree_active_indices = incumbent_indices.clone();
                    if DO_DBG {
                        println!("  skipping as seen before...")
                    }
                    continue;
                }

                let Some(removal_candidates) =
                    self.removal_candidates(&bridge, &bridge_rooted_cycles)
                else {
                    self.idtree.isolate_nodes(reisolate_bridge_nodes);
                    self.idtree_active_indices = incumbent_indices.clone();
                    if DO_DBG {
                        println!("  skipping as no removal candidates...")
                    }
                    continue;
                };

                let (is_improved, _removal_attempts, freed) =
                    self.improve_component(&bridge, &removal_candidates, ordered_removables);

                if is_improved {
                    incumbent_indices = self.idtree._active_nodes();
                    self.idtree_active_indices = incumbent_indices.clone();
                    improved = true;

                    ordered_removables.retain(|v| !freed.contains(v));
                    let tmp: Vec<usize> = bridge.iter().copied().collect();
                    ordered_removables.extend(self.sort_by_weights(tmp.as_slice(), None));
                    break;
                }

                if DO_DBG {
                    println!("  failed to improve...");
                }

                self.idtree.isolate_nodes(reisolate_bridge_nodes);
                self.idtree_active_indices = incumbent_indices.clone();
            }
        }
    }

    /// Applies bridge to idtree active nodes
    fn connect_bridge(&mut self, bridge: &IntSet<usize>) {
        // Insert edges connecting bridge nodes to their active neighbors.
        // Use tmp to store the whole bridge, deplete by moving from tmp to dtree
        // when the node in tmp has an active neighbor in dtree.
        let mut tmp = bridge.clone();
        let mut moved_node = true;
        while !tmp.is_empty() && moved_node {
            moved_node = false;
            for v in tmp.clone().drain() {
                let active_neighbors = self
                    .index_to_neighbors
                    .get(&v)
                    .unwrap()
                    .intersection(&self.idtree_active_indices);
                let mut inserted_active_neighbor = false;
                for &u in active_neighbors {
                    if self.idtree.insert_edge(v, u) != -1 {
                        inserted_active_neighbor = true;
                    }
                }
                if inserted_active_neighbor {
                    self.idtree_active_indices.insert(v);
                    tmp.remove(&v);
                    moved_node = true;
                }
            }
        }

        if DO_DBG {
            let bridge_node = *bridge.iter().next().unwrap();
            let component_len = self.idtree.node_connected_component(bridge_node).len();
            println!(
                "Bridged component of len {} generated from bridge {:?}...",
                component_len, bridge
            );
        }
    }

    fn bridge_rooted_cycles(&mut self, bridge: &IntSet<usize>) -> Option<Vec<IntSet<usize>>> {
        if DO_DBG {
            let bridge_waypoints: Vec<_> =
                bridge.iter().map(|&v| self.index_to_waypoint[&v]).collect();
            println!("Processing bridge {:?}... {:?}", bridge, bridge_waypoints);
        }

        let root = *bridge.iter().next()?;
        let all_cycles = self.idtree.cycle_basis(Some(root));
        if DO_DBG {
            let all_cycle_waypoints: Vec<_> = all_cycles
                .iter()
                .map(|cycle| {
                    cycle
                        .iter()
                        .map(|&v| self.index_to_waypoint[&v])
                        .collect::<Vec<_>>()
                })
                .collect();
            println!("all_cycle waypoints {:?}", all_cycle_waypoints);
        }

        let filtered: Vec<IntSet<usize>> = all_cycles
            .into_iter()
            .filter(|cycle| {
                cycle.len() >= (2 + bridge.len()) && cycle.iter().any(|v| bridge.contains(v))
            })
            .map(|v| IntSet::from_iter(v.iter().copied()))
            .collect();

        if filtered.is_empty() {
            None
        } else {
            if DO_DBG {
                let filtered_waypoints: Vec<_> = filtered
                    .iter()
                    .map(|cycle| {
                        cycle
                            .iter()
                            .map(|&v| self.index_to_waypoint[&v])
                            .collect::<Vec<_>>()
                    })
                    .collect();
                println!("filtered_waypoints {:?}", filtered_waypoints);
            }
            Some(filtered)
        }
    }

    fn was_seen_before(
        &mut self,
        bridge: &IntSet<usize>,
        cycles: &[IntSet<usize>],
        seen_before: &mut IntSet<u64>,
    ) -> bool {
        let mut all = Vec::from_iter(bridge.iter().copied());
        for cycle in cycles {
            all.extend(cycle.iter().copied());
        }
        all.sort_unstable();
        let all_hash = rapidhash_v3(
            &all.iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        if seen_before.contains(&all_hash) {
            true
        } else {
            seen_before.insert(all_hash);
            false
        }
    }

    fn removal_candidates(
        &mut self,
        bridge: &IntSet<usize>,
        cycles: &[IntSet<usize>],
    ) -> Option<Vec<(usize, usize)>> {
        if DO_DBG {
            let bridge_waypoints: Vec<_> =
                bridge.iter().map(|&v| self.index_to_waypoint[&v]).collect();
            println!(
                "Finding removal candidates for bridge={:?}... {:?}",
                bridge, bridge_waypoints
            );
        }
        let threshold = cycles.len() + 1;
        let mut candidates = IntSet::default();
        for cycle in cycles {
            candidates.extend(cycle.iter().copied());
        }
        candidates.retain(|&v| !self.untouchables.contains(&v) && !bridge.contains(&v));
        let idtree_candidates: Vec<(usize, usize)> = candidates
            .iter()
            .filter(|&&v| self.idtree.degree(v) as usize <= threshold)
            .map(|&v| (v, self.index_to_weight[&v]))
            .collect();

        if idtree_candidates.is_empty() {
            None
        } else {
            Some(idtree_candidates)
        }
    }

    fn improve_component(
        &mut self,
        bridge: &IntSet<usize>,
        removal_candidates: &[(usize, usize)],
        ordered_removables: &[usize],
    ) -> (bool, usize, IntSet<usize>) {
        if DO_DBG {
            let bridge_waypoints = bridge
                .iter()
                .map(|&v| self.index_to_waypoint[&v])
                .collect::<Vec<_>>();
            let removal_set_waypoints = removal_candidates
                .iter()
                .map(|&(v, _)| self.index_to_waypoint[&v])
                .collect::<Vec<_>>();
            println!(
                "Improving component for bridge {:?} => {:?}...\n  using removal candidates {:?} => {:?}...",
                bridge, bridge_waypoints, removal_candidates, removal_set_waypoints
            );
        }

        let mut ordered_removables = ordered_removables.to_owned();
        let mut removal_attempts = 0;
        let max_removal_attempts = self.max_removal_attempts;
        let bridged_component = self.idtree_active_indices.clone();
        self.update_bridge_affected_nodes(bridged_component.clone());
        let bridge_weight: usize = bridge.iter().map(|&v| self.index_to_weight[&v]).sum();
        let incumbent_weight: usize = self
            .idtree_active_indices
            .iter()
            .map(|&v| self.index_to_weight[&v])
            .sum::<usize>()
            - bridge_weight;
        let incumbent_component_count = self.idtree.num_connected_components();

        let removal_set_generator = WeightedRangeComboGenerator::new(
            removal_candidates,
            bridge_weight,
            bridge.len(),
            self.max_node_weight,
            self.combo_gen_direction,
        );

        for removal_set in removal_set_generator.generate() {
            removal_attempts += 1;
            if removal_attempts > max_removal_attempts {
                break;
            }

            if DO_DBG {
                let removal_set_waypoints: Vec<_> = removal_set
                    .iter()
                    .map(|&v| self.index_to_waypoint[&v])
                    .collect();
                println!("  removing removal set {:?}", removal_set_waypoints);
            }

            let mut deleted_edges = Vec::new();
            for &v in &removal_set {
                let neighbors = self.index_to_neighbors[&v].intersection(&bridged_component);
                for &u in neighbors {
                    if self.idtree.delete_edge(v, u) != -1 {
                        deleted_edges.push((v, u));
                    }
                }
            }

            if !self.terminal_pairs_connected() {
                for &(v, u) in &deleted_edges {
                    self.idtree.insert_edge(v, u);
                }
                if DO_DBG {
                    let removal_set_waypoints = removal_set
                        .iter()
                        .map(|&v| self.index_to_waypoint[&v])
                        .collect::<Vec<_>>();
                    println!(
                        "  skipping removal set as not connected... {:?}...",
                        removal_set_waypoints
                    );
                }
                continue;
            }

            let mut active_component_indices = bridged_component.clone();
            active_component_indices.retain(|&v| !removal_set.contains(&v));
            self.update_bridge_affected_nodes(active_component_indices.clone());
            ordered_removables.retain(|&v| !removal_set.contains(&v));

            // let mut tmp_removables = ordered_removables.clone();
            // tmp_removables.extend(bridge.iter().copied());
            let (freed, freed_edges) = self.remove_removables(&ordered_removables);
            active_component_indices.retain(|&v| !freed.contains(&v));
            let new_weight = active_component_indices
                .iter()
                .map(|&v| self.index_to_weight[&v])
                .sum();

            if incumbent_weight < new_weight
                || (incumbent_weight == new_weight
                    && self.idtree.num_connected_components() == incumbent_component_count)
            {
                for &(v, u) in &deleted_edges {
                    self.idtree.insert_edge(v, u);
                }
                for &(v, u) in &freed_edges {
                    self.idtree.insert_edge(v, u);
                }
                continue;
            }

            if DO_DBG {
                println!(
                    "  Improved Component! cost:{}, components: {}",
                    new_weight,
                    self.idtree.num_connected_components()
                );
            }

            let mut ret_val = freed.clone();
            ret_val.extend(removal_set.clone());
            return (true, removal_attempts, ret_val);
        }

        (false, removal_attempts, IntSet::default())
    }
}
