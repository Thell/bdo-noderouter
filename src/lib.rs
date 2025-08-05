use pyo3::prelude::*;
use wasm_bindgen::prelude::*;

use std::collections::{HashMap, HashSet, VecDeque};
use std::vec;

use nohash_hasher::{IntMap, IntSet};
use rapidhash::fast::{RandomState, RapidHashSet};
use rapidhash::v3::rapidhash_v3;

use petgraph::algo::{all_simple_paths, has_path_connecting, tarjan_scc};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::IntoNodeIdentifiers;

use serde::Deserialize;

const SUPER_ROOT: usize = 99_999;

const DO_DBG: bool = false;

fn sort_pair<T: Ord>(a: T, b: T) -> (T, T) {
    if a > b {
        (b, a)
    } else {
        (a, b)
    }
}

fn hash_intset(set: &IntSet<usize>) -> u64 {
    let mut sorted: Vec<_> = set.iter().copied().collect();
    sorted.sort_unstable();

    let mut buf = Vec::with_capacity(sorted.len() * std::mem::size_of::<usize>());
    for x in &sorted {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    rapidhash_v3(&buf)
}

pub(crate) struct BridgeState {
    pub rings: Vec<IntSet<usize>>,
    pub yielded_hashes: IntSet<u64>,
    pub search_state: VecDeque<SearchState>,
    pub seen_nodes: IntSet<usize>,
    pub current_ring_idx: usize,
}

pub(crate) struct SearchState {
    ring_idx: usize,
    current_nodes: IntSet<usize>,
    bridge: IntSet<usize>,
}

pub type ExplorationGraphData = HashMap<usize, ExplorationNodeData>;

#[derive(Debug, Deserialize, Clone)]
pub struct ExplorationNodeData {
    pub waypoint_key: usize,
    pub need_exploration_point: usize,
    pub is_base_town: bool,
    pub link_list: Vec<usize>,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

//////
// MARK: NodeRouter
//////

/// Solves Node-Weighted Steiner Forest using primal-dual and bridge heuristics.
#[derive(Clone, Debug)]
#[pyclass(unsendable)]
#[wasm_bindgen]
pub struct NodeRouter {
    // Used in the _combinations_with_weight_range_generator to limit combo generation.
    max_node_weight: usize,
    // Bridge heuristics
    // min 350 => 1.5x the max iter of test cases for 2-direction pass
    max_removal_attempts: usize,
    // Used for controlling the sort order of removal set generator.
    combo_gen_direction: bool, // 'true' for descending

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
    connected_pairs: HashSet<(usize, usize)>,
    // Used in reverse deletion to filter deletion and connection checks.
    bridge_affected_base_towns: IntSet<usize>,
    bridge_affected_indices: IntSet<usize>,
    bridge_affected_terminals: HashSet<(usize, usize)>,
    ordered_removables: Vec<usize>,
}

#[wasm_bindgen]
impl NodeRouter {
    #[wasm_bindgen(constructor)]
    pub fn wasm_new(exploration_json: &str) -> NodeRouter {
        let exploration_data: ExplorationGraphData =
            serde_json::from_str(exploration_json).unwrap();
        Self::new(&exploration_data)
    }
}

#[pymethods]
impl NodeRouter {
    #[new]
    pub fn py_new(exploration_json: &str) -> NodeRouter {
        use std::str::FromStr;
        let str_map: HashMap<String, ExplorationNodeData> =
            serde_json::from_str(exploration_json).unwrap();
        let exploration_data: ExplorationGraphData = str_map
            .into_iter()
            .map(|(k, v)| usize::from_str(&k).map(|k| (k, v)).unwrap())
            .collect();
        Self::new(&exploration_data)
    }
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

        // First we populate the translation mappings...
        if DO_DBG {
            println!("Populating translation mappings...");
        }
        for (i, (&waypoint_key, node_data)) in exploration_data.iter().enumerate() {
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
            let waypoint_idx = waypoint_to_index.get(&waypoint_key).unwrap();
            index_to_neighbors.insert(*waypoint_idx, IntSet::default());
            for &neighbor in &node_data.link_list {
                index_to_neighbors
                    .get_mut(&waypoint_idx)
                    .unwrap()
                    .insert(*waypoint_to_index.get(&neighbor).unwrap());
            }
        }

        // Testing... print the neighbors of waypoint 1
        let waypoint = 1;
        let tmp_index = waypoint_to_index.get(&waypoint).unwrap();
        let tmp_neighbors = index_to_neighbors.get(&tmp_index).unwrap();
        let tmp_neighbor_waypoints: Vec<_> = tmp_neighbors
            .into_iter()
            .map(|n| index_to_waypoint.get(n).unwrap())
            .collect();
        if DO_DBG {
            println!(
                "  neighbors of waypoint 1 at index {:?} are {:?} at indices {:?}",
                tmp_index, tmp_neighbor_waypoints, tmp_neighbors
            );
        }
        // Then we generate all of the edges for the ref_graph...
        if DO_DBG {
            println!("Generating ref_graph...");
        }
        let edges: HashSet<(u32, u32)> = index_to_neighbors
            .iter()
            .flat_map(|(i, neighbors)| {
                neighbors
                    .iter()
                    .map(move |&neighbor| sort_pair(*i as u32, neighbor as u32))
            })
            .collect();
        let ref_graph = StableUnGraph::<usize, usize>::from_edges(edges);

        // Lastly, initialize the IDTree...
        if DO_DBG {
            println!("Initializing IDTree...");
        }
        // self.dtree: IDTree = IDTree({v: [] for v in self.ref_graph.node_indices()})
        let mut initialization_adj_dict = IntMap::default();
        for i in ref_graph.node_indices() {
            initialization_adj_dict.insert(i.index(), IntSet::default());
        }
        let idtree = IDTree::new(&initialization_adj_dict);
        let idtree_active_indices = IntSet::default();

        Self {
            max_node_weight,
            max_removal_attempts: 350,
            combo_gen_direction: true,
            base_towns,
            index_to_neighbors,
            index_to_waypoint,
            index_to_weight,
            waypoint_to_index,
            ref_graph,
            idtree,
            idtree_active_indices,
            terminal_to_root: IntMap::default(),
            untouchables: IntSet::default(),
            connected_pairs: HashSet::default(),
            bridge_affected_base_towns: IntSet::default(),
            bridge_affected_indices: IntSet::default(),
            bridge_affected_terminals: HashSet::default(),
            ordered_removables: Vec::default(),
        }
    }

    /// Solve for a list of terminal pairs [(terminal, root), ...]
    /// where root is an exploration data waypoint with attribute `is_base_town`
    /// or `99999` to indicate a super-terminal that can connect to any base town.
    pub fn solve_for_terminal_pairs(&mut self, terminal_pairs: Vec<(usize, usize)>) -> Vec<usize> {
        self.terminal_to_root.clear();
        self.connected_pairs.clear();
        self.bridge_affected_base_towns.clear();
        self.bridge_affected_indices.clear();
        self.bridge_affected_terminals.clear();

        if DO_DBG {
            println!("solve for terminal pairs: {:?}", terminal_pairs);
        }
        let tmp_indices: Vec<_> = terminal_pairs
            .iter()
            .map(|(t, r)| {
                let t_idx = self.waypoint_to_index.get(&t).unwrap();
                let r_idx = if r == &SUPER_ROOT {
                    SUPER_ROOT
                } else {
                    *self.waypoint_to_index.get(&r).unwrap()
                };
                (t_idx, r_idx)
            })
            .collect();
        if DO_DBG {
            println!("at indices: {:?}", tmp_indices);
        }

        for (t, r) in terminal_pairs {
            let t_idx = *self.waypoint_to_index.get(&t).unwrap();
            let r_idx = if r == SUPER_ROOT {
                SUPER_ROOT
            } else {
                *self.waypoint_to_index.get(&r).unwrap()
            };
            self.terminal_to_root.insert(t_idx, r_idx);
        }

        if DO_DBG {
            println!("Generating untouchables...");
        }
        self.generate_untouchables();
        if DO_DBG {
            println!("  untouchables len {}", self.untouchables.len());
        }

        if DO_DBG {
            println!("Approximating...");
        }
        self.approximate();

        self.bridge_heuristics(self.ordered_removables.clone().as_mut());

        // Convert all remaining active indices in idtree to waypoints and return
        if DO_DBG {
            println!("Translating results...");
        }
        self.idtree_active_indices
            .iter()
            .map(|&i| self.index_to_waypoint[&i])
            .collect()
    }

    /// Set of all terminals, fixed roots and leaf terminal parents
    fn generate_untouchables(&mut self) {
        self.untouchables.clear();
        self.untouchables.extend(self.terminal_to_root.keys());
        self.untouchables.extend(self.terminal_to_root.values());
        self.untouchables.remove(&SUPER_ROOT);

        // Add unambigous connected nodes (degree 1)...
        if DO_DBG {
            println!("  untouchables: {:?}", self.untouchables);
        }
        for &node in self.untouchables.clone().iter() {
            if let Some(neighbors) = self.index_to_neighbors.get(&node) {
                if neighbors.len() == 1 {
                    self.untouchables.extend(neighbors);
                }
            }
        }
    }

    /////
    /// MARK: PD Approximation
    /////

    /// Solves the routing problem and returns a list of active nodes in solution
    fn approximate(&mut self) {
        let x = self.untouchables.clone();

        if DO_DBG {
            println!("Starting PD Approximation...");
        }
        let (x, mut ordered_removables) = self.primal_dual_approximation(x);

        // Transfer the solution to IDTree using a ref_graph subgraph's edges.
        if DO_DBG {
            println!("Transferring solution of {} nodes to IDTree...", x.len());
        }
        self.ref_graph
            .filter_map(
                |node_idx, _| {
                    if x.contains(&node_idx.index()) {
                        Some(())
                    } else {
                        None
                    }
                },
                |_, edge_idx| Some(*edge_idx),
            )
            .edge_indices()
            .for_each(|edge_idx| {
                let (u, v) = self.ref_graph.edge_endpoints(edge_idx).unwrap();
                self.idtree.insert_edge(u.index(), v.index());
            });
        self.idtree_active_indices = self.idtree.active_nodes().into_iter().collect();

        // Pruning is simply to reduce the ordered_removables in bulk on unambiguous nodes.
        if DO_DBG {
            println!("Pruning...");
        }
        self.prune_approximation(&mut ordered_removables);

        // Ordered removables are in temporal order of going tight, sub ordered structurally
        // by sorting by waypoint key.  When removing the nodes they should be processed in
        // reverse order to facilitate the removal of the latest nodes to 'go tight' first.
        // The list is reversed here and processed in forward order thoughout the remainder
        // of the algorithm and bridge heuristic processing.
        self.ordered_removables = ordered_removables.iter().rev().cloned().collect();

        // remove_removables is setup to primarily handle 'bridged' components in the Bridge
        // Heuristic. To simplify the code the bridge related variables are set here to
        // cover all removables, terminals and base towns in the graph.
        self.update_bridge_affected_nodes(self.idtree_active_indices.clone());

        if DO_DBG {
            println!("Removing removables...");
        }
        let (freed, _freed_edges) = self.remove_removables();
        if DO_DBG {
            println!("  freed {} nodes", freed.len());
        }

        self.idtree_active_indices.retain(|&v| !freed.contains(&v));
        self.ordered_removables.retain(|&v| !freed.contains(&v));
        if DO_DBG {
            println!(
                "Result: num active nodes is {} num active indices is {} num removables is {}",
                self.idtree.active_nodes().len(),
                self.idtree_active_indices.len(),
                self.ordered_removables.len()
            );
        }
    }

    /// Node Weighted Primal Dual Approximation (Demaine et al.)
    fn primal_dual_approximation(&mut self, mut x: IntSet<usize>) -> (IntSet<usize>, Vec<usize>) {
        // The main loop operations and frontier node calculations are set based.
        // The violated sets identification requires subgraphing the ref_graph and running
        // connected_components. The loop usually only iterates a half dozen times.
        let mut y = vec![0; self.ref_graph.node_count()];
        let mut ordered_removables = Vec::new();

        while let Some(violated_sets) = self.violated_sets(&x) {
            let violated: IntSet<_> = violated_sets.into_iter().flatten().collect();
            let frontier_nodes = self.find_frontier_nodes(&violated, None);
            let tight_nodes: Vec<_> = frontier_nodes
                .iter()
                .cloned()
                .filter(|&v| y[v] == self.index_to_weight[&v])
                .collect();

            x.extend(&tight_nodes);
            ordered_removables.extend(self.sort_by_weights(&tight_nodes, None));

            for &v in &frontier_nodes {
                y[v] += 1;
            }
        }

        (x, ordered_removables)
    }

    /// Returns connected components violating connectivity constraints.
    fn violated_sets(&mut self, x: &IntSet<usize>) -> Option<Vec<IntSet<usize>>> {
        // NOTE:It is fairly common in the smaller test incidents that a super terminal can
        // be placed in a Terminal Cluster Set of a component with a higher cost simply
        // because another (cheaper) component near it is non-violated. To avoid these
        // situations reset the connected pairs cache and mark all components as violated
        // until all super terminals are connected.

        // NOTE: Another scenario where a super terminal can be put in a higher cost component
        // is when it is nearer to a basetown that is a 'potential' super-terminal root but
        // that basetown isn't a fixed root for any other terminal while the super-terminal
        // is also near enough to a higher cost component's settlement that _is_ violated.
        // Meaning the violated component and super-terminal component both grow while the
        // potential super-terminal root basetown doesn't; leading to the super-terminal
        // joining with the higher cost component prior to joining with the lower costing
        // basetown. A possible fix for this is to include potential roots in the violated
        // components until all super terminals are not violated where the potential roots
        // are those that are nearest to the super-terminals, and if that basetown is already
        // a fixed basetown for a terminal then nothing special needs to be done.

        if DO_DBG {
            println!("Checking for violated sets...");
        }

        // Induce subgraph from self.ref_graph using node indices in x
        let subgraph = self.ref_graph.filter_map(
            |node_idx, _| {
                if x.contains(&node_idx.index()) {
                    Some(())
                } else {
                    None
                }
            },
            |_, edge_idx| Some(*edge_idx),
        );

        // Compute connected components (undirected graph)
        let components: Vec<IntSet<usize>> = tarjan_scc(&subgraph)
            .into_iter()
            .map(|comp| comp.iter().map(|nidx| nidx.index()).collect())
            .collect();

        let mut violated = Vec::new();
        let mut has_violated_super_terminal = false;

        // Here we test each iteration's current active terminals located with
        // the current component. This allows short-circuiting of testing once
        // all terminals are connected as well as per iteration reduction of
        // testable pairs.
        // The only real downside is that when super terminals are last in the
        // connected_pairs list the whole thing is cleared (as per the note above)
        // and the work will be repeated on the next iteration. While there are
        // alternatives the book keeping complexity doesn't provide an improvement
        // in efficiency for our input graph (<1000 nodes and ~ 100 terminal pairs).
        for cc in &components {
            let tmp_connected_pairs = self.connected_pairs.clone();
            let active_terminals = self
                .terminal_to_root
                .iter()
                .filter(|p| !tmp_connected_pairs.contains(&(*p.0, *p.1)));

            let mut cc_violated = false;

            for (&terminal, &root) in active_terminals {
                let terminal_in_cc = cc.contains(&terminal);

                // Test for any base town in the component if terminal is a super terminal.
                let root_in_cc = if root == SUPER_ROOT {
                    if DO_DBG {
                        let tmp: Vec<_> = cc.intersection(&self.base_towns).collect();
                        println!("cc: {:?}", cc);
                        println!("base_towns: {:?}", self.base_towns);
                        println!("intersection: {:?}", tmp);
                    }
                    cc.intersection(&self.base_towns).next().is_some()
                } else {
                    cc.contains(&root)
                };

                // Neither of the pair are in cc... skip...
                if !terminal_in_cc && !root_in_cc {
                    if root == SUPER_ROOT {
                        has_violated_super_terminal = true;
                        break;
                    }
                    continue;
                }

                // Both are in the cc, record and continue...
                if terminal_in_cc && root_in_cc {
                    self.connected_pairs.insert((terminal, root));
                } else {
                    // The pair is violated...
                    if root == SUPER_ROOT {
                        // violated super terminals violate all components...
                        has_violated_super_terminal = true;
                        break;
                    }
                    // Just this cc is violated...
                    cc_violated = true;
                    break;
                }
            }

            if has_violated_super_terminal {
                break;
            }

            if cc_violated {
                violated.push(cc.clone());
            }
        }

        if has_violated_super_terminal {
            self.connected_pairs.clear();
            Some(components)
        } else if violated.is_empty() {
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
            println!(
                "Finding frontier nodes for settlement of size {}...",
                settlement.len()
            );
        }
        let mut frontier = IntSet::default();
        for &v in settlement {
            if let Some(neighbors) = self.index_to_neighbors.get(&v) {
                frontier.extend(neighbors);
            }
        }
        frontier.retain(|v| !settlement.contains(v));

        if let Some(min_degree) = min_degree {
            frontier.retain(|v| {
                self.index_to_neighbors
                    .get(v)
                    .map_or(false, |nbrs| nbrs.len() >= min_degree)
            });
        }
        if DO_DBG {
            println!("  num frontier nodes {}", frontier.len());
        }
        frontier
    }

    fn sort_by_weights(&self, numbers: &[usize], weights: Option<&[usize]>) -> Vec<usize> {
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

        // NOTE:
        // The sorting here was purely for deterministic testing while refactoring from networkx.
        // But amazingly enough this _dramatically_ improves results with PD!
        // Considering this is sorting by 'waypoint_key' it doesn't really represent a weight
        // that _should_ have anything to do with the component growth but since the keys
        // are somewhat geographically ordered the structural ordering of the bridge
        // and removal candidates during the bridge heuristics is altered.
        pairs.sort_unstable();

        pairs.into_iter().map(|(_, number)| number).collect()
    }

    /// Simple straight forward recursive pruning of non-terminal degree 1 nodes from the graph.
    fn prune_approximation(&mut self, ordered_removables: &mut Vec<usize>) {
        if DO_DBG {
            println!(
                "Pruning idtree with {} nodes using {} active nodes and {} removables...",
                self.idtree.active_nodes().len(),
                self.idtree_active_indices.len(),
                ordered_removables.len()
            );
        }
        let mut untouchable_indices = self.untouchables.clone();
        untouchable_indices.retain(|&i| self.idtree_active_indices.contains(&i));
        untouchable_indices.extend(self.terminal_to_root.keys());
        loop {
            let extracted = self
                .idtree_active_indices
                .extract_if(|&i| self.idtree.degree(i) == 1 && !untouchable_indices.contains(&i))
                .collect::<Vec<_>>();
            if extracted.is_empty() {
                break;
            }
            self.idtree.isolate_nodes(extracted);
        }
        ordered_removables.retain(|&v| self.idtree.active_nodes().contains(&v));
        if DO_DBG {
            println!(
                "  num active nodes {} and removables {}",
                self.idtree.active_nodes().len(),
                ordered_removables.len()
            );
        }
    }

    /// Updates self._bridge_* variables with relevant bridged component nodes.
    fn update_bridge_affected_nodes(&mut self, affected_component: IntSet<usize>) {
        self.bridge_affected_indices = affected_component;
        self.bridge_affected_terminals = self
            .terminal_to_root
            .iter()
            .filter(|p| self.bridge_affected_indices.contains(&p.0))
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
    fn remove_removables(&mut self) -> (IntSet<usize>, Vec<(usize, usize)>) {
        let mut freed = IntSet::default();
        let mut freed_edges = Vec::new();
        for &v in &self.ordered_removables {
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

    /////
    /// MARK: Bridge Heuristic
    /////

    /// Bridge heuristic: find and utilize potential bridges to _increase_
    /// cycle counts and then identify removable articulation points that can
    /// improve the solution.
    fn bridge_heuristics(&mut self, ordered_removables: &mut Vec<usize>) {
        let mut incumbent_indices: IntSet<usize> = self.idtree_active_indices.clone();
        let mut seen_before_cache: IntSet<u64> = IntSet::default();

        let mut improved = true;
        while improved {
            improved = false;

            let mut bridge_state = self.init_bridge_generator(&incumbent_indices);
            while let Some(bridge) = self.next_bridge(&mut bridge_state) {
                let tmp_bridge_as_waypoints = bridge
                    .iter()
                    .map(|&v| self.index_to_waypoint[&v])
                    .collect::<Vec<_>>();
                println!(
                    "Processing bridge: {:?} => {:?}",
                    bridge, tmp_bridge_as_waypoints
                );

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

                // We create a single hash for the full set of the rooted cycles since
                // that's the closest we can get to Python's Frozenset testing...
                let bridge_rooted_cycles_hash = rapidhash_v3(
                    &bridge_rooted_cycles
                        .iter()
                        .flat_map(|cycle| cycle.iter().copied())
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                );
                if !seen_before_cache.insert(bridge_rooted_cycles_hash) {
                    println!("  skipping as seen before...");
                    self.idtree.isolate_nodes(reisolate_bridge_nodes);
                    self.idtree_active_indices = incumbent_indices.clone();
                    continue;
                }

                // let Some(removal_candidates) =
                //     self.removal_candidates(&bridge, &bridge_rooted_cycles)
                // else {
                //     self.idtree.isolate_nodes(reisolate_bridge_nodes);
                //     self.idtree_active_indices = incumbent_indices.clone();
                //     num_skipped_no_candidates += 1;
                //     continue;
                // };

                // let (is_improved, removal_attempts, freed) =
                //     self.improve_component(&bridge, &removal_candidates, ordered_removables);

                // Testing...
                let is_improved = false;
                let freed: IntSet<usize> = IntSet::default();

                if is_improved {
                    incumbent_indices = self.idtree._active_nodes();
                    self.idtree_active_indices = incumbent_indices.clone();
                    bridge_state = self.init_bridge_generator(&incumbent_indices);
                    improved = true;

                    ordered_removables.retain(|v| !freed.contains(v));
                    let tmp: Vec<usize> = bridge.iter().copied().collect();
                    ordered_removables.extend(self.sort_by_weights(tmp.as_slice(), None));
                    break;
                }

                self.idtree.isolate_nodes(reisolate_bridge_nodes);
                self.idtree_active_indices = incumbent_indices.clone();
            }

            if !improved {
                break;
            }
        }
    }

    fn init_bridge_generator(&self, settlement: &IntSet<usize>) -> BridgeState {
        let rings: Vec<IntSet<usize>> = vec![settlement.clone()];
        let seen_nodes = settlement.clone();
        let current_ring_idx = 0;

        if DO_DBG {
            println!("Initializing bridge generator...");
            let settlement_waypoints = settlement
                .iter()
                .map(|&v| self.index_to_waypoint[&v])
                .collect::<Vec<_>>();
            println!("  settlement: {:?}", settlement_waypoints);
        }
        BridgeState {
            rings,
            yielded_hashes: IntSet::default(),
            search_state: VecDeque::new(),
            seen_nodes,
            current_ring_idx,
        }
    }

    fn next_bridge(&self, state: &mut BridgeState) -> Option<IntSet<usize>> {
        let max_frontier_rings = 3;
        let ring_combo_cutoff = [0, 3, 2, 2];

        // Process existing search states
        while let Some(search) = state.search_state.pop_front() {
            let ring_idx = search.ring_idx;
            if ring_idx == 1 {
                let nodes_to_check = &search.current_nodes;
                let s_neighbors: IntSet<usize> = nodes_to_check
                    .iter()
                    .flat_map(|v| self.index_to_neighbors.get(v).unwrap().iter())
                    .copied()
                    .filter(|v| state.rings[0].contains(v))
                    .collect();

                let mut s_neighbors_waypoints: Vec<usize> = s_neighbors
                    .iter()
                    .map(|&v| self.index_to_waypoint.get(&v).copied().unwrap_or(v))
                    .collect();
                s_neighbors_waypoints.sort();

                if DO_DBG {
                    let mut bridge_waypoints: Vec<usize> = search
                        .bridge
                        .iter()
                        .map(|&v| self.index_to_waypoint.get(&v).copied().unwrap_or(v))
                        .collect();
                    bridge_waypoints.sort();
                    println!(
                        "Checking bridge {:?} (waypoints {:?}) at ring_idx 1 with inner-ring neighbors {:?} (waypoints {:?})",
                        search.bridge, bridge_waypoints, s_neighbors, s_neighbors_waypoints
                    );
                }

                if s_neighbors.len() >= 2 {
                    let bridge_hash = hash_intset(&search.bridge);
                    if state.yielded_hashes.insert(bridge_hash) {
                        return Some(search.bridge);
                    } else if DO_DBG {
                        println!("Bridge {:?} discarded: already yielded", search.bridge);
                    }
                } else if DO_DBG {
                    println!(
                        "Bridge {:?} discarded: insufficient settlement neighbors ({})",
                        search.bridge,
                        s_neighbors.len()
                    );
                }
                continue;
            }

            let inner_ring = &state.rings[ring_idx - 1];
            let candidates: Vec<usize> = search
                .current_nodes
                .iter()
                .flat_map(|n| self.index_to_neighbors.get(n).unwrap().iter())
                .copied()
                .filter(|v| inner_ring.contains(v))
                .collect::<RapidHashSet<_>>()
                .into_iter()
                .collect();

            let mut candidates_waypoints: Vec<usize> = candidates
                .iter()
                .map(|&v| self.index_to_waypoint.get(&v).copied().unwrap_or(v))
                .collect();
            candidates_waypoints.sort();

            if DO_DBG {
                let mut bridge_waypoints: Vec<usize> = search
                    .bridge
                    .iter()
                    .map(|&v| self.index_to_waypoint.get(&v).copied().unwrap_or(v))
                    .collect();
                bridge_waypoints.sort();
                println!(
                    "Ring_idx {}: Candidates for bridge {:?} (waypoints {:?}): {:?} (waypoints {:?})",
                    ring_idx, search.bridge, bridge_waypoints, candidates, candidates_waypoints
                );
            }

            let mut seen_candidate_pairs = RapidHashSet::default();
            for i in 0..(candidates.len() - 1) {
                let u = candidates[i];
                for j in (i + 1)..candidates.len() {
                    let v = candidates[j];
                    let candidate_pair = sort_pair(u, v);
                    if seen_candidate_pairs.contains(&candidate_pair) {
                        if DO_DBG {
                            let u_waypoint = self.index_to_waypoint.get(&u).copied().unwrap_or(u);
                            let v_waypoint = self.index_to_waypoint.get(&v).copied().unwrap_or(v);
                            println!(
                                "  skipping candidate pair {} {} => {} {}",
                                u, u_waypoint, v, v_waypoint
                            );
                        }
                        continue;
                    }
                    seen_candidate_pairs.insert(candidate_pair);

                    let mut new_bridge = search.bridge.clone();
                    new_bridge.insert(u);
                    new_bridge.insert(v);

                    let new_current_nodes = IntSet::from_iter([u, v]);
                    state.search_state.push_back(SearchState {
                        ring_idx: ring_idx - 1,
                        current_nodes: new_current_nodes,
                        bridge: new_bridge,
                    });
                }
            }
        }

        // Generate new bridges by advancing to the next ring
        if state.current_ring_idx < max_frontier_rings {
            state.current_ring_idx += 1;
            let new_nodes = self.find_frontier_nodes(&state.seen_nodes, Some(2));
            if !new_nodes.is_empty() {
                state.seen_nodes.extend(new_nodes.iter());
                state.rings.push(new_nodes.clone());
                let ring_idx = state.current_ring_idx;
                let inner_ring = &state.rings[ring_idx - 1];
                let mut seen_endpoints = RapidHashSet::default();

                // Phase 1: Single-node bridges
                for &node in new_nodes.iter() {
                    let neighbors: IntSet<usize> = self
                        .index_to_neighbors
                        .get(&node)
                        .unwrap()
                        .intersection(inner_ring)
                        .copied()
                        .collect();

                    let mut neighbors_waypoints: Vec<usize> = neighbors
                        .iter()
                        .map(|&v| self.index_to_waypoint.get(&v).copied().unwrap_or(v))
                        .collect();
                    neighbors_waypoints.sort();

                    if neighbors.len() < 2 {
                        continue;
                    }

                    let neighbors_vec: Vec<_> = neighbors.iter().copied().collect();
                    let first = neighbors_vec[0];
                    let rest = &neighbors_vec[1..];
                    let inner_ring_neighbors: IntSet<usize> = self
                        .index_to_neighbors
                        .get(&first)
                        .unwrap()
                        .intersection(inner_ring)
                        .copied()
                        .collect();

                    if !rest.iter().any(|n| {
                        inner_ring_neighbors
                            .intersection(self.index_to_neighbors.get(n).unwrap())
                            .next()
                            .is_some()
                    }) {
                        let bridge = IntSet::from_iter([node]);
                        state.search_state.push_back(SearchState {
                            ring_idx,
                            current_nodes: IntSet::from_iter([node]),
                            bridge,
                        });
                    }
                }

                // Phase 2: Multi-node bridges
                let subgraph = self.ref_graph.filter_map(
                    |node_idx, _| {
                        if new_nodes.contains(&node_idx.index()) {
                            Some(())
                        } else {
                            None
                        }
                    },
                    |_, edge_idx| Some(*edge_idx),
                );

                let node_indentifiers: Vec<_> = subgraph.node_identifiers().collect();
                let mut node_indices: Vec<usize> =
                    node_indentifiers.iter().map(|&v| v.index()).collect();
                node_indices.sort_unstable();

                if DO_DBG {
                    let node_indices_waypoints = node_indentifiers
                        .iter()
                        .map(|&v| self.index_to_waypoint.get(&v.index()))
                        .collect::<Vec<_>>();
                    println!(
                        "  subgraph nodes {:?} (waypoints {:?})",
                        node_indentifiers, node_indices_waypoints
                    );
                }

                for u in node_indices.iter() {
                    for v in node_indices.iter().skip(1) {
                        let u_identifier = NodeIndex::new(*u);
                        let v_identifier = NodeIndex::new(*v);
                        if !has_path_connecting(&subgraph, u_identifier, v_identifier, None) {
                            continue;
                        }
                        let u_waypoint = self.index_to_waypoint.get(&u).unwrap();
                        let v_waypoint = self.index_to_waypoint.get(&v).unwrap();

                        let u_neighbors = self.index_to_neighbors.get(&u).unwrap();
                        let v_neighbors = self.index_to_neighbors.get(&v).unwrap();
                        if u_neighbors
                            .intersection(&v_neighbors)
                            .any(|n| inner_ring.contains(n))
                        {
                            if DO_DBG {
                                println!(
                                    "  phase2 neighbor check: skipping {} {} => {} {}",
                                    u, u_waypoint, v, v_waypoint
                                );
                            }
                            continue;
                        }

                        let key = sort_pair(u, v);
                        if seen_endpoints.contains(&key) {
                            if DO_DBG {
                                println!(
                                    "  pase2 seen_endpoints: skipping {} {} => {} {}",
                                    u, u_waypoint, v, v_waypoint
                                );
                            }
                            continue;
                        }

                        for path_len in 2..=ring_combo_cutoff[ring_idx] {
                            let mut path_iter = all_simple_paths::<Vec<NodeIndex>, _, RandomState>(
                                &subgraph,
                                u_identifier,
                                v_identifier,
                                path_len - 2,
                                Some(path_len),
                            )
                            .take(1);

                            if let Some(path) = path_iter.next() {
                                let combo = IntSet::from_iter(path.iter().map(|n| n.index()));
                                let combo_waypoints: Vec<usize> = combo
                                    .iter()
                                    .map(|&v| self.index_to_waypoint.get(&v).copied().unwrap_or(v))
                                    .collect();
                                if seen_endpoints.contains(&key) {
                                    if DO_DBG {
                                        println!(
                                            "  pase2 seen_endpoints: skipping {} {} => {} {}",
                                            u, u_waypoint, v, v_waypoint
                                        );
                                    }
                                    continue;
                                }
                                seen_endpoints.insert(key);
                                println!(
                                    "  phase2 pushing state on {} {} combo {:?} => {} {} [{:?}])",
                                    u, v, combo, u_waypoint, v_waypoint, combo_waypoints
                                );

                                state.search_state.push_back(SearchState {
                                    ring_idx,
                                    current_nodes: combo.clone(),
                                    bridge: combo,
                                });
                            }
                        }
                    }
                }

                if DO_DBG {
                    println!(
                        "Search_state size after ring {}: {}",
                        ring_idx,
                        state.search_state.len()
                    );
                }
            }
            // Re-run to process new states
            return self.next_bridge(state);
        }

        None
    }
    // fn next_bridge(&self, state: &mut BridgeState) -> Option<IntSet<usize>> {
    //     let max_frontier_rings = 3;
    //     let ring_combo_cutoff = [0, 3, 2, 2]; // Match Python's bridge size limits

    //     // Process existing search states
    //     while let Some(search) = state.search_state.pop_back() {
    //         let ring_idx = search.ring_idx;
    //         if ring_idx == 1 {
    //             let nodes_to_check = &search.current_nodes;
    //             let s_neighbors: IntSet<usize> = nodes_to_check
    //                 .iter()
    //                 .flat_map(|v| self.index_to_neighbors.get(v).unwrap().iter())
    //                 .copied()
    //                 .filter(|v| state.rings[0].contains(v))
    //                 .collect();

    //             if DO_DBG {
    //                 let s_neighbors_waypoints: Vec<usize> = s_neighbors
    //                     .iter()
    //                     .map(|&v| self.index_to_waypoint.get(&v).copied().unwrap_or(v))
    //                     .collect();
    //                 println!(
    //                     "Checking bridge {:?} at ring_idx 1 with inner-ring neighbors {:?} (waypoints {:?})",
    //                     search.bridge, s_neighbors, s_neighbors_waypoints
    //                 );
    //             }

    //             if s_neighbors.len() >= 2 {
    //                 let bridge_hash = hash_intset(&search.bridge);
    //                 if state.yielded_hashes.insert(bridge_hash) {
    //                     return Some(search.bridge);
    //                 }
    //             } else if DO_DBG {
    //                 println!(
    //                     "Bridge {:?} discarded: insufficient settlement neighbors ({})",
    //                     search.bridge, s_neighbors.len()
    //                 );
    //             }
    //             continue;
    //         }

    //         let inner_ring = &state.rings[ring_idx - 1];
    //         let candidates: Vec<usize> = search
    //             .current_nodes
    //             .iter()
    //             .flat_map(|n| self.index_to_neighbors.get(n).unwrap().iter())
    //             .copied()
    //             .filter(|v| inner_ring.contains(v))
    //             .collect::<RapidHashSet<_>>()
    //             .into_iter()
    //             .collect();

    //         if DO_DBG {
    //             let candidates_waypoints: Vec<usize> = candidates
    //                 .iter()
    //                 .map(|&v| self.index_to_waypoint.get(&v).copied().unwrap_or(v))
    //                 .collect();

    //             println!(
    //                 "Ring_idx {}: Candidates for bridge {:?} (waypoints {:?}): {:?} (waypoints {:?})",
    //                 ring_idx, search.bridge,
    //                 search.bridge.iter().map(|&v| self.index_to_waypoint.get(&v).copied().unwrap_or(v)).collect::<Vec<_>>(),
    //                 candidates, candidates_waypoints
    //             );
    //         }

    //         let mut seen_candidate_pairs = RapidHashSet::default();
    //         for i in 0..(candidates.len() - 1) {
    //             let u = candidates[i];
    //             for j in (i + 1)..candidates.len() {
    //                 let v = candidates[j];
    //                 let candidate_pair = sort_pair(u, v);
    //                 if seen_candidate_pairs.contains(&candidate_pair) {
    //                     continue;
    //                 }
    //                 seen_candidate_pairs.insert(candidate_pair);

    //                 let mut new_bridge = search.bridge.clone();
    //                 new_bridge.insert(u);
    //                 new_bridge.insert(v);

    //                 let new_current_nodes = IntSet::from_iter([u, v]);
    //                 state.search_state.push_back(SearchState {
    //                     ring_idx: ring_idx - 1,
    //                     current_nodes: new_current_nodes,
    //                     bridge: new_bridge,
    //                 });
    //             }
    //         }
    //     }

    //     // Generate new bridges by advancing to the next ring
    //     if state.current_ring_idx < max_frontier_rings {
    //         state.current_ring_idx += 1;
    //         let new_nodes = self.find_frontier_nodes(&state.seen_nodes, Some(2));
    //         if !new_nodes.is_empty() {
    //             if DO_DBG {
    //                 println!("Ring {}: {:?}", state.current_ring_idx, new_nodes);
    //             }
    //             state.seen_nodes.extend(new_nodes.iter());
    //             state.rings.push(new_nodes.clone());
    //             let ring_idx = state.current_ring_idx;
    //             let inner_ring = &state.rings[ring_idx - 1];

    //             // Phase 1: Single-node bridges
    //             for &node in new_nodes.iter() {
    //                 let neighbors: IntSet<usize> = self
    //                     .index_to_neighbors
    //                     .get(&node)
    //                     .unwrap()
    //                     .intersection(inner_ring)
    //                     .copied()
    //                     .collect();

    //                 if DO_DBG {
    //                     let neighbors_waypoints: Vec<usize> = neighbors
    //                         .iter()
    //                         .map(|&v| self.index_to_waypoint.get(&v).copied().unwrap_or(v))
    //                         .collect();

    //                     println!(
    //                         "Node {} has inner-ring neighbors {:?} (waypoints {:?})",
    //                         node, neighbors, neighbors_waypoints
    //                     );
    //                 }

    //                 if neighbors.len() < 2 {
    //                     continue;
    //                 }

    //                 let neighbors_vec: Vec<_> = neighbors.iter().copied().collect();
    //                 let first = neighbors_vec[0];
    //                 let rest = &neighbors_vec[1..];
    //                 let inner_ring_neighbors: IntSet<usize> = self
    //                     .index_to_neighbors
    //                     .get(&first)
    //                     .unwrap()
    //                     .intersection(inner_ring)
    //                     .copied()
    //                     .collect();

    //                 if !rest.iter().any(|n| {
    //                     inner_ring_neighbors
    //                         .intersection(self.index_to_neighbors.get(n).unwrap())
    //                         .next()
    //                         .is_some()
    //                 }) {
    //                     let bridge = IntSet::from_iter([node]);
    //                     state.search_state.push_back(SearchState {
    //                         ring_idx,
    //                         current_nodes: IntSet::from_iter([node]),
    //                         bridge,
    //                     });
    //                 } else if DO_DBG {
    //                     println!("Node {} discarded: inner-ring neighbors are adjacent", node);
    //                 }
    //             }

    //             // Phase 2: Multi-node bridges
    //             let subgraph = self.ref_graph.filter_map(
    //                 |node_idx, _| {
    //                     if new_nodes.contains(&node_idx.index()) {
    //                         Some(())
    //                     } else {
    //                         None
    //                     }
    //                 },
    //                 |_, edge_idx| Some(*edge_idx),
    //             );

    //             let node_indices: Vec<_> = subgraph.node_identifiers().collect();
    //             let mut seen_endpoints = RapidHashSet::default();

    //             for i in 0..node_indices.len() {
    //                 let u = node_indices[i].index();
    //                 for j in (i + 1)..node_indices.len() {
    //                     let v = node_indices[j].index();

    //                     let u_neighbors = self.index_to_neighbors.get(&u).unwrap();
    //                     let v_neighbors = self.index_to_neighbors.get(&v).unwrap();
    //                     if u_neighbors.intersection(&v_neighbors).any(|n| inner_ring.contains(n)) {
    //                         if DO_DBG {
    //                             let u_waypoint = self.index_to_waypoint.get(&u).unwrap();
    //                             let v_waypoint = self.index_to_waypoint.get(&v).unwrap();
    //                             println!("  skipping {} {} => {} {}", u, u_waypoint, v, v_waypoint);
    //                         }
    //                         continue;
    //                     }

    //                     let key = sort_pair(u, v);
    //                     if seen_endpoints.contains(&key) {
    //                         if DO_DBG {
    //                             let u_waypoint = self.index_to_waypoint.get(&u).unwrap();
    //                             let v_waypoint = self.index_to_waypoint.get(&v).unwrap();
    //                             println!("  skipping {} {} => {} {}", u, u_waypoint, v, v_waypoint);
    //                         }
    //                         continue;
    //                     }
    //                     seen_endpoints.insert(key);

    //                     let mut path_iter = all_simple_paths::<Vec<NodeIndex>, _, RandomState>(
    //                         &subgraph,
    //                         node_indices[i],
    //                         node_indices[j],
    //                         0,
    //                         Some(ring_combo_cutoff[ring_idx]),
    //                     ).take(1);

    //                     if let Some(path) = path_iter.next() {
    //                         let combo = IntSet::from_iter(path.iter().map(|n| n.index()));
    //                         state.search_state.push_back(SearchState {
    //                             ring_idx,
    //                             current_nodes: combo.clone(),
    //                             bridge: combo,
    //                         });
    //                     } else {
    //                         if DO_DBG {
    //                             let u_waypoint = self.index_to_waypoint.get(&u).unwrap();
    //                             let v_waypoint = self.index_to_waypoint.get(&v).unwrap();
    //                             println!("  skipping {} {} => {} {}", u, u_waypoint, v, v_waypoint);
    //                         }
    //                     }
    //                 }
    //             }

    //             if DO_DBG {
    //                 println!("Search_state size after ring {}: {}", ring_idx, state.search_state.len());
    //             }
    //         }
    //         // Re-run to process new states
    //         return self.next_bridge(state);
    //     }

    //     None
    // }

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
    }

    fn bridge_rooted_cycles(&mut self, bridge: &IntSet<usize>) -> Option<Vec<IntSet<usize>>> {
        let root = *bridge.iter().next()?;
        let all_cycles = self.idtree.cycle_basis(Some(root));

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
}

//////
// MARK: IDTree
//////
#[derive(Clone, Debug, PartialEq, Eq)]
struct Node {
    parent: i32,
    subtree_size: usize,
    adj: IntSet<usize>,
}

impl Node {
    fn new() -> Self {
        Node {
            parent: -1,
            subtree_size: 1,
            adj: IntSet::default(),
        }
    }

    fn insert_adj(&mut self, u: usize) {
        self.adj.insert(u);
    }

    fn delete_adj(&mut self, u: usize) {
        self.adj.remove(&u);
    }
}

#[derive(Clone, Debug)]
#[pyclass(unsendable)]
struct IDTree {
    n: usize,
    nodes: Vec<Node>,
    used: Vec<bool>, // scratch area
    q: Vec<usize>,   // scratch area
    l: Vec<usize>,   // scratch area
}

#[pymethods]
impl IDTree {
    #[new]
    fn py_new(adj_dict: HashMap<usize, Vec<usize>>) -> Self {
        let adj_dict: IntMap<usize, IntSet<usize>> = adj_dict
            .into_iter()
            .map(|(k, v)| (k, IntSet::from_iter(v)))
            .collect();
        let mut instance = Self::new(&adj_dict);
        instance.initialize();
        instance
    }

    fn py_clone(&self) -> Self {
        self.clone()
    }

    // MARK: Core
    pub fn insert_edge(&mut self, u: usize, v: usize) -> i32 {
        if !self.insert_edge_in_graph(u, v) {
            return -1;
        }
        self.insert_edge_balanced(u, v)
    }

    pub fn delete_edge(&mut self, u: usize, v: usize) -> i32 {
        if !self.delete_edge_in_graph(u, v) {
            return -1;
        }
        self.delete_edge_balanced(u, v)
    }

    pub fn query(&self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n {
            return false;
        }
        let mut root_u = u;
        while self.nodes[root_u].parent != -1 {
            root_u = self.nodes[root_u].parent as usize;
        }
        let mut root_v = v;
        while self.nodes[root_v].parent != -1 {
            root_v = self.nodes[root_v].parent as usize;
        }
        root_u == root_v
    }

    // MARK: Extensions

    /// Rooted Tree-Based Fundamental Cycle Basis
    pub fn cycle_basis(&mut self, root: Option<usize>) -> Vec<Vec<usize>> {
        // Constructs a fundamental cycle basis for the connected component containing `root`,
        // using the ID-Tree structure as its spanning tree. A fundamental cycle is formed
        // each time a non-tree edge is encountered during DFS from the `root`.
        //
        // Each cycle:
        // - Includes the `root` node
        // - Is simple (no repeated nodes)
        // - Is formed by combining paths from both endpoints of the non-tree edge up to their
        //   lowest common ancestor (LCA) in the tree
        //
        // Assumes:
        // - The underlying graph is undirected and simple (no self-loops or multi-edges)
        // - The `parent` field in each node defines a rooted spanning tree (ID-Tree)
        // - The `root` is a valid node within a connected component
        if root.is_none() {
            return vec![];
        }
        let root = root.unwrap();
        let mut cycles = Vec::with_capacity(self.n / 2);
        let mut seen_edges = IntSet::default();
        let mut in_component = IntSet::default();
        let mut stack = vec![root];

        in_component.insert(root);

        while let Some(u) = stack.pop() {
            for &v in &self.nodes[u].adj {
                // Skip if already handled from other end
                if seen_edges.contains(&(v * self.n + u)) || u == v {
                    continue;
                }

                seen_edges.insert(u * self.n + v);

                // First-time discovery of a node in this component
                if in_component.insert(v) {
                    stack.push(v);
                }

                // Skip tree edges
                let pu = self.nodes[u].parent;
                let pv = self.nodes[v].parent;
                if pu == v as i32 || pv == u as i32 {
                    continue;
                }

                // Found a fundamental cycle via (u, v)
                let mut path_u = vec![u];
                let mut path_v = vec![v];
                let mut a = u;
                let mut b = v;

                let mut visited_u = IntSet::default();
                visited_u.insert(a);
                let mut visited_v = IntSet::default();
                visited_v.insert(b);

                while a != b {
                    if self.nodes[a].parent != -1 {
                        a = self.nodes[a].parent as usize;
                        if !visited_u.insert(a) {
                            break;
                        }
                        path_u.push(a);
                        if visited_v.contains(&a) {
                            break;
                        }
                    }
                    if self.nodes[b].parent != -1 && a != b {
                        b = self.nodes[b].parent as usize;
                        if !visited_v.insert(b) {
                            break;
                        }
                        path_v.push(b);
                        if visited_u.contains(&b) {
                            break;
                        }
                    }
                }

                let lca = *path_u.iter().rev().find(|x| path_v.contains(x)).unwrap();
                while path_u.last() != Some(&lca) {
                    path_u.pop();
                }
                while path_v.last() != Some(&lca) {
                    path_v.pop();
                }

                path_v.pop(); // avoid repeating lca
                path_v.reverse();
                path_u.extend(path_v);
                cycles.push(path_u);
            }
        }

        cycles
    }

    pub fn node_connected_component(&mut self, v: usize) -> Vec<usize> {
        let mut stack = vec![v];
        let mut visited = IntSet::from_iter([v]);
        while let Some(node) = stack.pop() {
            for &neighbor in self.nodes[node].adj.iter() {
                if visited.insert(neighbor) {
                    stack.push(neighbor);
                }
            }
        }
        visited.into_iter().collect()
    }

    pub fn num_connected_components(&mut self) -> usize {
        (0..self.n)
            .filter(|&i| self.nodes[i].parent == -1 && !self.is_isolated(i))
            .count()
    }

    pub fn connected_components(&mut self) -> Vec<Vec<usize>> {
        let roots: Vec<_> = (0..self.n)
            .filter(|&i| self.nodes[i].parent == -1 && !self.is_isolated(i))
            .collect();
        roots
            .into_iter()
            .map(|i| self.node_connected_component(i))
            .collect()
    }

    pub fn active_nodes(&mut self) -> Vec<usize> {
        (0..self.n).filter(|&i| !self.is_isolated(i)).collect()
    }

    pub fn _active_nodes(&mut self) -> IntSet<usize> {
        (0..self.n).filter(|&i| !self.is_isolated(i)).collect()
    }

    pub fn isolate_node(&mut self, v: usize) {
        self.nodes[v].adj.clone().iter().for_each(|neighbor| {
            self.delete_edge(v, *neighbor);
        });
    }

    pub fn isolate_nodes(&mut self, nodes: Vec<usize>) {
        nodes.iter().for_each(|&v| self.isolate_node(v));
    }

    pub fn is_isolated(&mut self, v: usize) -> bool {
        self.nodes[v].adj.is_empty()
    }

    pub fn degree(&mut self, v: usize) -> i32 {
        self.nodes[v].adj.len() as i32
    }

    pub fn neighbors(&mut self, v: usize) -> Vec<usize> {
        self.nodes[v].adj.iter().cloned().collect()
    }

    pub fn retain_active_nodes_from(&mut self, from_indices: Vec<usize>) -> Vec<usize> {
        from_indices
            .into_iter()
            .filter(|&neighbor| !self.is_isolated(neighbor))
            .collect()
    }
}

impl IDTree {
    fn new(adj_dict: &IntMap<usize, IntSet<usize>>) -> Self {
        Self::setup(adj_dict)
    }

    fn setup(adj_dict: &IntMap<usize, IntSet<usize>>) -> Self {
        let n = adj_dict.len();
        let nodes: Vec<Node> = (0..n)
            .map(|i| {
                let mut node = Node::new();
                for &j in adj_dict.get(&i).unwrap_or(&IntSet::default()) {
                    node.insert_adj(j);
                }
                node
            })
            .collect();
        Self {
            n,
            nodes,
            used: vec![false; n],
            q: vec![],
            l: vec![],
        }
    }

    fn initialize(&mut self) {
        for &node_index in self.sort_nodes_by_degree().iter() {
            if self.used[node_index] {
                continue;
            }
            self.bfs_setup_subtrees(node_index);

            if let Some(centroid_node) = self.find_centroid_in_q() {
                self.reroot(centroid_node);
            }
        }
        self.used.fill(false);
    }

    fn sort_nodes_by_degree(&self) -> Vec<usize> {
        // Sort nodes by degree in descending order.
        let mut node_indices: Vec<usize> = (0..self.n).collect();
        node_indices
            .sort_unstable_by(|&a, &b| self.nodes[b].adj.len().cmp(&self.nodes[a].adj.len()));
        node_indices
    }

    fn bfs_setup_subtrees(&mut self, root: usize) {
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(root);

        self.q.clear();
        self.q.push(root);
        self.used[root] = true;

        while let Some(node_index) = queue.pop_front() {
            for j in 0..self.nodes[node_index].adj.len() {
                let neighbor_index = *self.nodes[node_index].adj.get(&j).unwrap();
                if !self.used[neighbor_index] {
                    self.used[neighbor_index] = true;
                    self.nodes[neighbor_index].parent = node_index as i32;
                    self.q.push(neighbor_index);
                    queue.push_back(neighbor_index);
                }
            }
        }

        // Propagate subtree sizes up the tree, skipping the root
        for &child_index in self.q.iter().skip(1).rev() {
            let parent_index = self.nodes[child_index].parent as usize;
            self.nodes[parent_index].subtree_size += self.nodes[child_index].subtree_size;
        }
    }

    fn find_centroid_in_q(&self) -> Option<usize> {
        let num_nodes = self.q.len();
        let half_num_nodes = num_nodes / 2;

        self.q.iter().rev().find_map(|&node_index| {
            if self.nodes[node_index].subtree_size > half_num_nodes {
                Some(node_index)
            } else {
                None
            }
        })
    }

    fn insert_edge_in_graph(&mut self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n || u == v {
            return false;
        }
        self.nodes[u].insert_adj(v);
        self.nodes[v].insert_adj(u);
        true
    }

    fn insert_edge_balanced(&mut self, mut u: usize, mut v: usize) -> i32 {
        // Algorithm 1: ID-Insert

        let (mut root_u, mut root_v, mut p, mut pp);

        // 1 𝑟𝑜𝑜𝑡𝑢 ← compute the root of 𝑢;
        root_u = u;
        while self.nodes[root_u].parent != -1 {
            root_u = self.nodes[root_u].parent as usize;
        }
        // 2 𝑟𝑜𝑜𝑡𝑣 ← compute the root of 𝑣;
        root_v = v;
        while self.nodes[root_v].parent != -1 {
            root_v = self.nodes[root_v].parent as usize;
        }

        //  /* non-tree edge insertion */
        // 3 if 𝑟𝑜𝑜𝑡𝑢 = 𝑟𝑜𝑜𝑡𝑣 then
        if root_u == root_v {
            let mut reshape = false;
            let mut depth = 0;
            p = self.nodes[u].parent;
            pp = self.nodes[v].parent;

            // 4 if 𝑑𝑒𝑝𝑡ℎ(𝑢) < 𝑑𝑒𝑝𝑡ℎ(𝑣) then swap(𝑢,𝑣);
            while depth < self.n {
                if p == -1 {
                    if pp != -1 && self.nodes[pp as usize].parent == -1 {
                        std::mem::swap(&mut u, &mut v);
                        std::mem::swap(&mut p, &mut pp);
                        reshape = true;
                    }
                    break;
                } else if pp == -1 {
                    if p == -1 && self.nodes[p as usize].parent == -1 {
                        reshape = true;
                    }
                    break;
                }
                p = self.nodes[p as usize].parent;
                pp = self.nodes[pp as usize].parent;
                depth += 1;
            }

            if reshape {
                // Find new centroid...
                // depth u is greater than or equal to depth v from step 4
                // p and pp are at depth v; count levels to depth u for difference from depth v
                // for 1 ≤ 𝑖 < (𝑑𝑒𝑝𝑡ℎ(𝑢)−𝑑𝑒𝑝𝑡ℎ(𝑣))/2
                let mut w = p;
                depth = 0;
                while w != -1 {
                    depth += 1;
                    w = self.nodes[w as usize].parent;
                }
                if depth <= 1 {
                    return 0;
                }
                // split depth in half and set w to the split point
                depth = depth / 2 - 1;
                w = u as i32;
                while depth > 0 {
                    w = self.nodes[w as usize].parent;
                    depth -= 1;
                }

                // 9 Unlink(𝑤);
                let (root_v, _subtree_u_size) = self.unlink(w as usize, v);

                // 10 Link(ReRoot(𝑢),𝑣,𝑟𝑜𝑜𝑡𝑣);
                self.reroot(u);
                if let Some(new_root) = self.link_non_tree_edge(u, v, root_v) {
                    if new_root != root_v {
                        self.reroot(new_root);
                    }
                }
            }

            // 11 return;
            return 0;
        }

        // /* tree edge insertion */
        // 12 if 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑟𝑜𝑜𝑡𝑢) > 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑟𝑜𝑜𝑡𝑣) then
        if self.nodes[root_u].subtree_size > self.nodes[root_v].subtree_size {
            // 13 swap(𝑢,𝑣);
            std::mem::swap(&mut u, &mut v);
            // 14 swap(𝑟𝑜𝑜𝑡𝑢,𝑟𝑜𝑜𝑡𝑣);
            std::mem::swap(&mut root_u, &mut root_v);
        }

        // 15 Link(ReRoot(𝑢),𝑣,𝑟𝑜𝑜𝑡𝑣);
        self.reroot_tree_edge(u, v);
        if let Some(new_root) = self.link_tree_edge(root_u, v, root_v) {
            if new_root != root_v {
                self.reroot(new_root);
            }
        }
        1
    }

    fn delete_edge_in_graph(&mut self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n || u == v {
            return false;
        }
        self.nodes[u].delete_adj(v);
        self.nodes[v].delete_adj(u);
        true
    }

    fn delete_edge_balanced(&mut self, mut u: usize, mut v: usize) -> i32 {
        // 1 if 𝑝𝑎𝑟𝑒𝑛𝑡(𝑢) ≠ 𝑣 ∧ 𝑝𝑎𝑟𝑒𝑛𝑡(𝑣) ≠ 𝑢 then return;
        if (self.nodes[u].parent != v as i32 && self.nodes[v].parent != u as i32) || u == v {
            return 0;
        }

        // 2 if 𝑝𝑎𝑟𝑒𝑛𝑡(𝑣) = 𝑢 then swap(𝑢,𝑣);
        if self.nodes[v].parent == u as i32 {
            std::mem::swap(&mut u, &mut v);
        }

        // 3 𝑟𝑜𝑜𝑡𝑣 ← Unlink(𝑢);
        let (mut root_v, subtree_u_size) = self.unlink(u, v);

        // 4 if 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑟𝑜𝑜𝑡𝑣) < 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑢) then swap(𝑢,𝑟𝑜𝑜𝑡𝑣);
        if self.nodes[root_v].subtree_size < subtree_u_size {
            std::mem::swap(&mut u, &mut root_v);
        }

        // /* search subtree rooted in 𝑢 */
        if self.find_replacement(u, root_v) {
            return 1;
        }
        2
    }

    fn find_replacement(&mut self, u: usize, f: usize) -> bool {
        // 5 𝑄 ← an empty queue, 𝑄.𝑝𝑢𝑠ℎ(𝑢);
        // let mut queue: VecDeque<usize> = VecDeque::new();
        // queue.push_back(u);

        self.q.clear();
        self.l.clear();

        self.q.push(u);
        self.l.push(u);
        self.used[u] = true;

        //  7 while 𝑄 ≠ ∅ do
        let mut i = 0;
        while i < self.q.len() {
            let mut x = self.q[i];
            i += 1;

            //  9 foreach 𝑦 ∈ 𝑁(𝑥) do
            for &y in &self.nodes[x].adj {
                // 10 if 𝑦 = 𝑝𝑎𝑟𝑒𝑛𝑡(𝑥) then continue;
                if y as i32 == self.nodes[x].parent {
                    continue;
                }
                // 11 else if 𝑥 = 𝑝𝑎𝑟𝑒𝑛𝑡(𝑦) then
                if self.nodes[y].parent == x as i32 {
                    // 12 𝑄.𝑝𝑢𝑠ℎ(𝑦);
                    self.q.push(y);

                    if !self.used[y] {
                        // 13 𝑆 ← 𝑆 ∪ {𝑦};
                        self.used[y] = true;
                        self.l.push(y);
                    }
                    continue;
                }

                // Try to build a new path from y upward
                // 15 𝑠𝑢𝑐𝑐 ← true;
                let mut replacement_found = true;

                // 16 foreach 𝑤 from 𝑦 to the root do
                let mut w = y as i32;
                while w != -1 {
                    // 17 if 𝑤 ∈ 𝑆 then
                    if self.used[w as usize] {
                        // 18 𝑠𝑢𝑐𝑐 ← false;
                        replacement_found = false;
                        // 19 break;
                        break;
                    }
                    // 20 else
                    // 21 𝑆 ← 𝑆 ∪ {𝑤};
                    self.used[w as usize] = true;
                    self.l.push(w as usize);

                    w = self.nodes[w as usize].parent;
                }
                if !replacement_found {
                    continue;
                }

                // 22 if 𝑠𝑢𝑐𝑐 then

                // 23 𝑟𝑜𝑜𝑡𝑣 ← Link(ReRoot(𝑥),𝑦,𝑟𝑜𝑜𝑡𝑣);

                // Link
                // ReRoot(𝑥)
                let mut p = self.nodes[x].parent;
                self.nodes[x].parent = y as i32;
                while p != -1 {
                    let pp = self.nodes[p as usize].parent;
                    self.nodes[p as usize].parent = x as i32;
                    x = p as usize;
                    p = pp;
                }

                // Compute new root => update subtree sizes and find new root
                let subtree_u_size = self.nodes[u].subtree_size;
                let s = (self.nodes[f].subtree_size + subtree_u_size) / 2;
                let mut new_root = None;
                let mut p = y as i32;
                while p != -1 {
                    self.nodes[p as usize].subtree_size += subtree_u_size;
                    if new_root.is_none() && self.nodes[p as usize].subtree_size > s {
                        new_root = Some(p as usize);
                    }
                    p = self.nodes[p as usize].parent;
                }

                // Fix subtree sizes
                let mut p = self.nodes[x].parent;
                while p != y as i32 {
                    self.nodes[x].subtree_size -= self.nodes[p as usize].subtree_size;
                    self.nodes[p as usize].subtree_size += self.nodes[x].subtree_size;
                    x = p as usize;
                    p = self.nodes[p as usize].parent;
                }

                for &k in &self.l {
                    self.used[k] = false;
                }

                if let Some(new_root) = new_root {
                    if new_root != f {
                        self.reroot(new_root);
                    }
                }
                return true;
            }
        }

        for &k in &self.l {
            self.used[k] = false;
        }

        false
    }

    fn reroot_tree_edge(&mut self, mut u: usize, v: usize) {
        let mut p = self.nodes[u].parent;
        self.nodes[u].parent = v as i32;
        while p != -1 {
            let temp = self.nodes[p as usize].parent;
            self.nodes[p as usize].parent = u as i32;
            u = p as usize;
            p = temp;
        }
    }

    fn reroot(&mut self, mut u: usize) {
        // - rotates the tree and makes 𝑢 as the new root by updating the parent-child
        //   relationship and the subtree size attribute from 𝑢 to the original root.
        //   The time complexity of ReRoot() is 𝑂(𝑑𝑒𝑝𝑡ℎ(𝑢)).

        // Rotate tree
        // Set parents of nodes between u and the old root.
        let mut p = self.nodes[u].parent;
        let mut pp;
        self.nodes[u].parent = -1;
        while p != -1 {
            pp = self.nodes[p as usize].parent;
            self.nodes[p as usize].parent = u as i32;
            u = p as usize;
            p = pp;
        }

        // Fix subtree sizes of nodes between u and the old root.
        p = self.nodes[u].parent;
        while p != -1 {
            self.nodes[u].subtree_size -= self.nodes[p as usize].subtree_size;
            self.nodes[p as usize].subtree_size += self.nodes[u].subtree_size;
            u = p as usize;
            p = self.nodes[p as usize].parent;
        }
    }

    fn link_non_tree_edge(&mut self, u: usize, v: usize, root_v: usize) -> Option<usize> {
        // Link
        self.nodes[u].parent = v as i32;
        self.link(u, v, root_v)
    }

    fn link_tree_edge(&mut self, u: usize, v: usize, root_v: usize) -> Option<usize> {
        let new_root = self.link(u, v, root_v);

        // Fix subtree sizes between u and the old root
        let mut p = self.nodes[u].parent;
        let mut u = u;
        while p != v as i32 {
            self.nodes[u].subtree_size -= self.nodes[p as usize].subtree_size;
            self.nodes[p as usize].subtree_size += self.nodes[u].subtree_size;
            u = p as usize;
            p = self.nodes[u].parent;
        }

        new_root
    }

    fn link(&mut self, u: usize, v: usize, root_v: usize) -> Option<usize> {
        // - Link(𝑢, 𝑣,𝑟𝑜𝑜𝑡 𝑣) adds a tree 𝑇𝑢 rooted in 𝑢 to the children of 𝑣.
        //     𝑟𝑜𝑜𝑡 𝑣 is the root of 𝑣.
        //     Given that the subtree size of 𝑣 is changed, it updates the subtree size for each
        //     vertex from 𝑣 to the root.
        //     We apply the centroid heuristic by recording the first vertex with a subtree size
        //     larger than 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑟𝑜𝑜𝑡𝑣)/2.
        //     If such a vertex is found, we reroot the tree, and the operator returns the new root.
        //     The time complexity of Link() is 𝑂(𝑑𝑒𝑝𝑡ℎ(𝑣)).

        // Compute new root => update subtree sizes and find new root
        let subtree_u_size = self.nodes[u].subtree_size;
        let s = (self.nodes[root_v].subtree_size + subtree_u_size) / 2;
        let mut new_root = None;
        let mut p = v as i32;
        while p != -1 {
            self.nodes[p as usize].subtree_size += subtree_u_size;
            if new_root.is_none() && self.nodes[p as usize].subtree_size > s {
                new_root = Some(p as usize);
            }
            p = self.nodes[p as usize].parent;
        }
        new_root
    }

    fn unlink(&mut self, u: usize, v: usize) -> (usize, usize) {
        let mut root_v: usize = 0;
        let mut w = v as i32;
        let subtree_u_size = self.nodes[u].subtree_size;
        while w != -1 {
            self.nodes[w as usize].subtree_size -= subtree_u_size;
            root_v = w as usize;
            w = self.nodes[w as usize].parent;
        }
        self.nodes[u].parent = -1;
        (root_v, subtree_u_size)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn nwsf_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IDTree>()?;
    m.add_class::<NodeRouter>()?;
    Ok(())
}
