use smallvec::SmallVec;

use std::collections::{BTreeMap, HashMap};
use std::hash::Hasher;
use std::ops::CoroutineState;
use std::rc::Rc;

use fixedbitset::FixedBitSet;
use idtree::IDTree;
use nohash_hasher::{BuildNoHashHasher, IntMap, IntSet};
use petgraph::stable_graph::StableUnGraph;
use rapidhash::fast::RapidHasher;
use rapidhash::{HashSetExt, RapidHashSet};
use serde::Deserialize;

use crate::exploration_data::ExplorationData;
use crate::generator_bridge::BridgeGenerator;
use crate::generator_weighted_combo::WeightedRangeComboGenerator;
use crate::gssp::DialsRouter;
use crate::primal_dual::approximate;

pub const SUPER_ROOT: usize = 99_999;

pub type ExplorationGraphData = BTreeMap<usize, ExplorationNodeData>;
pub type SharedExplorationData = Rc<ExplorationData>;

#[derive(Debug, Deserialize, Clone)]
pub struct ExplorationNodeData {
    pub waypoint_key: usize,
    pub need_exploration_point: usize,
    pub is_base_town: bool,
    pub link_list: Vec<usize>,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Debug)]
pub struct DynamicState {
    combo_gen_direction: bool,
    has_super_terminal: bool,
    idtree: IDTree,
    idtree_active_indices: FixedBitSet,
    terminal_to_root: IntMap<usize, usize>,
    terminal_root_pairs: RapidHashSet<(usize, usize)>,
    untouchables: IntSet<usize>,
    bridge_affected_base_towns: IntSet<usize>,
    bridge_affected_indices: FixedBitSet,
    bridge_affected_terminals: RapidHashSet<(usize, usize)>,
}

/// Solves Node-Weighted Steiner Forest using primal-dual and bridge heuristics.
#[derive(Clone, Debug)]
pub struct NodeRouter {
    /// All exploration node centric data.
    pub(crate) exploration: SharedExplorationData,

    // Static Mappings (hoisted from exploration data)
    pub(crate) neighbors: Vec<SmallVec<[usize; 4]>>,
    pub(crate) weights: Vec<u32>,

    /// Used for controlling special case handlers for super terminals
    pub(crate) has_super_terminal: bool,

    // Bridge heuristics
    /// min 350 => 1.5x the max iter of test cases for 2-direction pass
    max_removal_attempts: usize,
    /// Controls the number of frontier rings for the bridge generator
    max_frontier_rings: usize,
    /// Contols the width of the bridge within each frontier ring
    ring_combo_cutoff: Vec<usize>,
    /// Used for controlling the sort order of removal set generator.
    /// (false => ascending)
    combo_gen_direction: bool,
    /// Used to control if terminal->root intermediate node betweenness is used
    /// to control the sort order of the removal set generator.
    use_betweenness: bool,
    /// Limit inclusion of nodes with degree into removal candidates
    cycle_degree_threshold: usize,

    // Greedy Shortest Shared Paths Approximation
    gssp_router: DialsRouter,

    // The main workhorse of the Bridge Heuristic
    idtree: IDTree,
    pub(crate) idtree_active_indices: FixedBitSet,
    bridge_generator: BridgeGenerator,

    // Contains all terminal, root pairs
    pub(crate) terminal_to_root: IntMap<usize, usize>,
    terminal_root_pairs: RapidHashSet<(usize, usize)>,

    // Contains all terminals, fixed roots, leaf terminal parents
    pub(crate) untouchables: IntSet<usize>,

    // Used in reverse deletion to filter deletion and connection checks.
    bridge_affected_base_towns: IntSet<usize>,
    bridge_affected_indices: FixedBitSet,
    bridge_affected_terminals: RapidHashSet<(usize, usize)>,

    bridge_all_cycle_nodes: Vec<usize>,
    hash_buf: Vec<u8>,
    scratch_nodes: Vec<usize>,
}

impl NodeRouter {
    pub fn new(exploration_data: &ExplorationGraphData) -> Self {
        let exploration = Rc::new(ExplorationData::new(exploration_data));

        let gssp_router = DialsRouter::new(exploration.clone());

        let mut initialization_adj_dict = IntMap::with_capacity_and_hasher(
            exploration.ref_ungraph.node_count(),
            BuildNoHashHasher::default(),
        );
        for i in exploration.ref_ungraph.node_indices() {
            initialization_adj_dict.insert(i.index(), IntSet::default());
        }

        let node_count = exploration.ref_ungraph.node_count();
        let max_frontier_rings = 3;
        let ring_combo_cutoff = vec![3, 2, 2];

        Self {
            exploration: exploration.clone(),
            neighbors: exploration.index_to_neighbors_ungraph.clone(),
            weights: exploration.index_to_weight.clone(),

            max_removal_attempts: 350, // option (9_000 for full bridge testing, 1_800 for single pass)
            max_frontier_rings,
            ring_combo_cutoff: ring_combo_cutoff.clone(),
            combo_gen_direction: false,
            use_betweenness: false,
            has_super_terminal: false,

            // TODO: Determine if this still serves any purpose since we now have betweenness...
            cycle_degree_threshold: 5, // max intermediate usage degree on ref_graph is 6

            gssp_router,
            idtree: IDTree::from_adj(&initialization_adj_dict),
            idtree_active_indices: FixedBitSet::with_capacity(node_count),
            bridge_generator: BridgeGenerator::new(
                &exploration,
                max_frontier_rings,
                ring_combo_cutoff,
            ),

            terminal_to_root: IntMap::default(), // static per solve run
            terminal_root_pairs: RapidHashSet::default(), // static per solve run
            untouchables: IntSet::with_capacity_and_hasher(
                node_count,
                BuildNoHashHasher::default(),
            ), // static per solve run

            bridge_affected_base_towns: IntSet::with_capacity_and_hasher(
                node_count,
                BuildNoHashHasher::default(),
            ),
            bridge_affected_indices: FixedBitSet::with_capacity(node_count),
            bridge_affected_terminals: RapidHashSet::with_capacity(node_count),
            bridge_all_cycle_nodes: Vec::with_capacity(node_count),
            hash_buf: Vec::with_capacity(node_count * core::mem::size_of::<usize>()),
            scratch_nodes: Vec::with_capacity(64),
        }
    }

    /// Set node router options by string, value
    /// Options:
    /// "max_removal_attempts" => usize
    /// "max_frontier_rings" => usize
    /// "ring_combo_cutoff" => usize
    pub fn set_option(&mut self, option: &str, value: &str) -> Result<(), String> {
        match option {
            "max_removal_attempts" => {
                let parsed: usize = value
                    .parse()
                    .map_err(|_| format!("invalid integer for {option}"))?;
                if !(1..=1_000_000).contains(&parsed) {
                    return Err(format!(
                        "value {parsed} out of range for {option} (1–1_000_000)"
                    ));
                }
                self.max_removal_attempts = parsed;
            }

            "max_frontier_rings" => {
                let parsed: usize = value
                    .parse()
                    .map_err(|_| format!("invalid integer for {option}"))?;
                if !(1..=100).contains(&parsed) {
                    return Err(format!("value {parsed} out of range for {option} (1–100)"));
                }

                self.max_frontier_rings = parsed;
                // Store the last combo cutoff value
                let current_cutoff: usize = self.ring_combo_cutoff.last().copied().unwrap_or(0);
                self.ring_combo_cutoff = vec![0; parsed];

                // Set all cutoffs to the current 'last' cutoff
                self.ring_combo_cutoff
                    .iter_mut()
                    .for_each(|c| *c = current_cutoff);
                // Increment the first position to be +1
                self.ring_combo_cutoff[0] += 1;

                println!("ring_combo_cutoff: {:?}", self.ring_combo_cutoff);
            }

            "ring_combo_cutoff" => {
                let cutoff: usize = value
                    .parse()
                    .map_err(|_| format!("invalid integer for {option}"))?;
                if cutoff > 100 {
                    return Err(format!("value {cutoff} out of range for {option} (≤ 100)"));
                }

                // Build vector of exactly `max_frontier_rings` elements
                self.ring_combo_cutoff = vec![cutoff; self.max_frontier_rings];
                // Increase the first by 1
                if let Some(first) = self.ring_combo_cutoff.first_mut() {
                    *first += 1;
                }
            }
            _ => return Err(format!("unknown option: {option}")),
        }

        if matches!(option, "max_frontier_rings" | "ring_combo_cutoff") {
            self.bridge_generator = BridgeGenerator::new(
                &self.exploration,
                self.max_frontier_rings,
                self.ring_combo_cutoff.clone(),
            );
        }

        Ok(())
    }

    /// Solve for a list of terminal pairs [(terminal, root), ...]
    /// where root is an exploration data waypoint with attribute `is_base_town`
    /// or `99999` to indicate a super-terminal that can connect to any base town.
    pub fn solve_for_terminal_pairs(
        &mut self,
        terminal_pairs: Vec<(usize, usize)>,
    ) -> (Vec<usize>, usize) {
        const EXPORT_TO_CSV: bool = false;
        let mut csv_hdr = if EXPORT_TO_CSV {
            Some(Vec::new())
        } else {
            None
        };
        let mut csv_row = if EXPORT_TO_CSV {
            Some(Vec::new())
        } else {
            None
        };
        let csv_filename = "results.csv";

        const DO_IOP_APPROXIMATION: bool = true;
        const DO_IOP_PBS_FWD: bool = true;
        const DO_IOP_PBS_REV: bool = true;
        const DO_IOP_PBS_BET: bool = true;

        const DO_GSSP_APPROXIMATION: bool = true;
        const DO_GSSP_PBS_FWD: bool = true;
        const DO_GSSP_PBS_REV: bool = true;
        const DO_GSSP_PBS_BET: bool = true;

        const DO_PD_APPROXIMATION: bool = true;
        const DO_PD_PBS_FWD: bool = true;
        const DO_PD_PBS_REV: bool = true;
        const DO_PD_PBS_BET: bool = true;

        // At least one approximation must be enabled
        const {
            assert!(DO_IOP_APPROXIMATION || DO_GSSP_APPROXIMATION || DO_PD_APPROXIMATION);
        }

        // --------------------------------------------------------------------
        // Terminal Pairs preprocessing
        // --------------------------------------------------------------------
        // TODO: This section will apply transformations and reductions to the
        // terminal pairs based on the static reductions applied to the
        // exploration data during the graph construction phase.
        //
        // If the result of the preprocessing is a single terminal pair, then
        // approximation can be skipped since it is a shortest path.
        //
        // If the result of the preprocessing is a single component (tree), then
        // approximation could be skipped and the component could be solved
        // directly using an exact Steiner tree algorithm.

        // --------------------------------------------------------------------
        // Input Ordered Paths (IOP) Approximation
        // --------------------------------------------------------------------
        let iop_winner = if DO_IOP_APPROXIMATION {
            // Prepare
            self.clear_dynamic_state();
            let terminal_idx_pairs = self.init_terminal_pairs(terminal_pairs.clone());
            self.generate_untouchables();
            self.gssp_router.reset();
            self.use_betweenness = false;

            // Approximate
            let (iop_visited, iop_ordered_removables) =
                self.gssp_router.input_ordered_paths(&terminal_idx_pairs);
            self.populate_idtree(&iop_visited);
            let (iop_approximation, iop_approximation_weight) = self.idtree_weight();

            let post_iop_state = self.get_dynamic_state();

            // Improve
            let (iop_fwd_indices, iop_fwd_weight) = if DO_IOP_PBS_FWD {
                self.bridge_heuristics(&mut iop_ordered_removables.clone());
                let (iop_fwd_indices, iop_fwd_weight) = self.idtree_weight();
                self.restore_dynamic_state(post_iop_state);
                (iop_fwd_indices, iop_fwd_weight)
            } else {
                (iop_approximation.clone(), iop_approximation_weight)
            };

            let (iop_rev_indices, iop_rev_weight) = if DO_IOP_PBS_REV {
                self.combo_gen_direction = true;
                self.bridge_heuristics(&mut iop_ordered_removables.clone());
                let (iop_rev_indices, iop_rev_weight) = self.idtree_weight();
                (iop_rev_indices, iop_rev_weight)
            } else {
                (FixedBitSet::new(), usize::MAX)
            };

            let (iop_bet_indices, iop_bet_weight) = if DO_IOP_PBS_BET {
                self.combo_gen_direction = false;
                self.use_betweenness = true;

                self.bridge_heuristics(&mut iop_ordered_removables.clone());
                let (iop_bet_indices, iop_bet_weight) = self.idtree_weight();
                (iop_bet_indices, iop_bet_weight)
            } else {
                (FixedBitSet::new(), usize::MAX)
            };

            // Export
            if let Some(csv_row) = &mut csv_row {
                csv_hdr.as_mut().unwrap().push("iop_base".to_string());
                csv_hdr.as_mut().unwrap().push("iop_fwd".to_string());
                csv_hdr.as_mut().unwrap().push("iop_rev".to_string());
                csv_hdr.as_mut().unwrap().push("iop_bet".to_string());
                csv_row.push(iop_approximation_weight);
                csv_row.push(iop_fwd_weight);
                csv_row.push(iop_rev_weight);
                csv_row.push(iop_bet_weight);
            }

            // The overall iop winner
            let iop_solutions = [
                (iop_fwd_indices, iop_fwd_weight),
                (iop_rev_indices, iop_rev_weight),
                (iop_bet_indices, iop_bet_weight),
            ];
            iop_solutions
                .iter()
                .min_by_key(|(_, weight)| *weight)
                .unwrap()
                .clone()
        } else {
            if let Some(csv_row) = &mut csv_row {
                csv_hdr.as_mut().unwrap().push("iop_base".to_string());
                csv_hdr.as_mut().unwrap().push("iop_fwd".to_string());
                csv_hdr.as_mut().unwrap().push("iop_rev".to_string());
                csv_hdr.as_mut().unwrap().push("iop_bet".to_string());
                csv_row.push(usize::MAX);
                csv_row.push(usize::MAX);
                csv_row.push(usize::MAX);
                csv_row.push(usize::MAX);
            }
            (FixedBitSet::new(), usize::MAX)
        };

        // --------------------------------------------------------------------
        // Greedy Shortest Shared Paths Approximation
        // --------------------------------------------------------------------
        let gssp_winner = if DO_GSSP_APPROXIMATION {
            // Prepare
            self.clear_dynamic_state();
            let terminal_idx_pairs = self.init_terminal_pairs(terminal_pairs.clone());
            self.generate_untouchables();
            self.gssp_router.reset();
            self.use_betweenness = false;

            // Approximate
            let (gssp_visited, gssp_ordered_removables) = self
                .gssp_router
                .greedy_shortest_shared_paths(&terminal_idx_pairs);
            self.populate_idtree(&gssp_visited);
            let (gssp_approximation, gssp_approximation_weight) = self.idtree_weight();

            let post_gssp_state = self.get_dynamic_state();

            // Improve
            let (gssp_fwd_indices, gssp_fwd_weight) = if DO_GSSP_PBS_FWD {
                self.bridge_heuristics(&mut gssp_ordered_removables.clone());
                let (gssp_fwd_indices, gssp_fwd_weight) = self.idtree_weight();
                self.restore_dynamic_state(post_gssp_state);
                (gssp_fwd_indices, gssp_fwd_weight)
            } else {
                (gssp_approximation.clone(), gssp_approximation_weight)
            };

            let (gssp_rev_indices, gssp_rev_weight) = if DO_GSSP_PBS_REV {
                self.combo_gen_direction = true;
                self.bridge_heuristics(&mut gssp_ordered_removables.clone());
                let (gssp_rev_indices, gssp_rev_weight) = self.idtree_weight();
                (gssp_rev_indices, gssp_rev_weight)
            } else {
                (FixedBitSet::new(), usize::MAX)
            };

            let (gssp_bet_indices, gssp_bet_weight) = if DO_GSSP_PBS_BET {
                self.combo_gen_direction = false;
                self.use_betweenness = true;

                self.bridge_heuristics(&mut gssp_ordered_removables.clone());
                let (gssp_bet_indices, gssp_bet_weight) = self.idtree_weight();
                (gssp_bet_indices, gssp_bet_weight)
            } else {
                (FixedBitSet::new(), usize::MAX)
            };

            // Export
            if let Some(csv_row) = &mut csv_row {
                csv_hdr.as_mut().unwrap().push("gssp_base".to_string());
                csv_hdr.as_mut().unwrap().push("gssp_fwd".to_string());
                csv_hdr.as_mut().unwrap().push("gssp_rev".to_string());
                csv_hdr.as_mut().unwrap().push("gssp_bet".to_string());
                csv_row.push(gssp_approximation_weight);
                csv_row.push(gssp_fwd_weight);
                csv_row.push(gssp_rev_weight);
                csv_row.push(gssp_bet_weight);
            }

            // The overall gssp winner
            let gssp_solutions = [
                (gssp_fwd_indices, gssp_fwd_weight),
                (gssp_rev_indices, gssp_rev_weight),
                (gssp_bet_indices, gssp_bet_weight),
            ];
            gssp_solutions
                .iter()
                .min_by_key(|(_, weight)| *weight)
                .unwrap()
                .clone()
        } else {
            if let Some(csv_row) = &mut csv_row {
                csv_hdr.as_mut().unwrap().push("gssp_base".to_string());
                csv_hdr.as_mut().unwrap().push("gssp_fwd".to_string());
                csv_hdr.as_mut().unwrap().push("gssp_rev".to_string());
                csv_hdr.as_mut().unwrap().push("gssp_bet".to_string());
                csv_row.push(usize::MAX);
                csv_row.push(usize::MAX);
                csv_row.push(usize::MAX);
                csv_row.push(usize::MAX);
            }
            (FixedBitSet::new(), usize::MAX)
        };

        // --------------------------------------------------------------------
        // Primal Dual Approximation
        // --------------------------------------------------------------------
        let pd_winner = if DO_PD_APPROXIMATION {
            // Prepare
            self.clear_dynamic_state();
            let _terminal_idx_pairs = self.init_terminal_pairs(terminal_pairs.clone());
            self.generate_untouchables();
            self.use_betweenness = false;

            // Approximate
            let pd_ordered_removables = approximate(self);
            let (pd_approximation, pd_approximation_weight) = self.idtree_weight();

            let post_pd_state = self.get_dynamic_state();

            // Improve
            let (pd_fwd_indices, pd_fwd_weight) = if DO_PD_PBS_FWD {
                self.bridge_heuristics(&mut pd_ordered_removables.clone());
                let (pd_fwd_indices, pd_fwd_weight) = self.idtree_weight();
                self.restore_dynamic_state(post_pd_state);
                (pd_fwd_indices, pd_fwd_weight)
            } else {
                (pd_approximation.clone(), pd_approximation_weight)
            };

            let (pd_rev_indices, pd_rev_weight) = if DO_PD_PBS_REV {
                self.combo_gen_direction = true;
                self.bridge_heuristics(&mut pd_ordered_removables.clone());
                let (pd_rev_indices, pd_rev_weight) = self.idtree_weight();
                (pd_rev_indices, pd_rev_weight)
            } else {
                (FixedBitSet::new(), usize::MAX)
            };

            let (pd_bet_indices, pd_bet_weight) = if DO_PD_PBS_BET {
                self.combo_gen_direction = false;
                self.use_betweenness = true;

                self.bridge_heuristics(&mut pd_ordered_removables.clone());
                let (pd_bet_indices, pd_bet_weight) = self.idtree_weight();
                (pd_bet_indices, pd_bet_weight)
            } else {
                (FixedBitSet::new(), usize::MAX)
            };

            // Export
            if let Some(csv_row) = &mut csv_row {
                csv_hdr.as_mut().unwrap().push("pd_base".to_string());
                csv_hdr.as_mut().unwrap().push("pd_fwd".to_string());
                csv_hdr.as_mut().unwrap().push("pd_rev".to_string());
                csv_hdr.as_mut().unwrap().push("pd_bet".to_string());
                csv_row.push(pd_approximation_weight);
                csv_row.push(pd_fwd_weight);
                csv_row.push(pd_rev_weight);
                csv_row.push(pd_bet_weight);
            }

            // The overall pd winner
            let pd_solutions = [
                (pd_fwd_indices, pd_fwd_weight),
                (pd_rev_indices, pd_rev_weight),
                (pd_bet_indices, pd_bet_weight),
            ];
            pd_solutions
                .iter()
                .min_by_key(|(_, weight)| *weight)
                .unwrap()
                .clone()
        } else {
            if let Some(csv_row) = &mut csv_row {
                csv_hdr.as_mut().unwrap().push("pd_base".to_string());
                csv_hdr.as_mut().unwrap().push("pd_fwd".to_string());
                csv_hdr.as_mut().unwrap().push("pd_rev".to_string());
                csv_hdr.as_mut().unwrap().push("pd_bet".to_string());
                csv_row.push(usize::MAX);
                csv_row.push(usize::MAX);
                csv_row.push(usize::MAX);
                csv_row.push(usize::MAX);
            }
            (FixedBitSet::new(), usize::MAX)
        };

        // --------------------------------------------------------------------
        // CSV Export - appends to the file
        // --------------------------------------------------------------------
        if let Some(csv_row) = &mut csv_row {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(csv_filename)
                .unwrap();
            let is_empty = file.metadata().unwrap().len() == 0;
            let mut writer = csv::Writer::from_writer(file);

            if is_empty {
                writer.write_record(csv_hdr.as_ref().unwrap()).unwrap();
            }

            // Write out the row
            writer.serialize(csv_row).unwrap();
            writer.flush().unwrap();
        }

        // --------------------------------------------------------------------
        // Overall Winner
        // --------------------------------------------------------------------
        let approximations = [iop_winner, gssp_winner, pd_winner];
        let (winner, weight) = approximations
            .iter()
            .min_by_key(|(_, weight)| *weight)
            .unwrap()
            .clone();

        let winner = winner
            .ones()
            .map(|i| self.exploration.index_to_waypoint[i])
            .collect();

        (winner, weight)
    }
}

impl NodeRouter {
    fn clear_dynamic_state(&mut self) {
        self.combo_gen_direction = false;
        self.has_super_terminal = false;
        for node in self.idtree.active_nodes_vec() {
            self.idtree.isolate_node(node);
        }
        self.idtree_active_indices.clear();
        self.terminal_to_root.clear();
        self.terminal_root_pairs.clear();
        self.untouchables.clear();
        self.bridge_affected_base_towns.clear();
        self.bridge_affected_indices.clear();
        self.bridge_affected_terminals.clear();
    }

    fn get_dynamic_state(&self) -> DynamicState {
        DynamicState {
            combo_gen_direction: self.combo_gen_direction,
            has_super_terminal: self.has_super_terminal,
            idtree: self.idtree.clone(),
            idtree_active_indices: self.idtree_active_indices.clone(),
            terminal_to_root: self.terminal_to_root.clone(),
            terminal_root_pairs: self.terminal_root_pairs.clone(),
            untouchables: self.untouchables.clone(),
            bridge_affected_base_towns: self.bridge_affected_base_towns.clone(),
            bridge_affected_indices: self.bridge_affected_indices.clone(),
            bridge_affected_terminals: self.bridge_affected_terminals.clone(),
        }
    }

    fn restore_dynamic_state(&mut self, state: DynamicState) {
        self.combo_gen_direction = state.combo_gen_direction;
        self.has_super_terminal = state.has_super_terminal;
        self.idtree = state.idtree.clone();
        self.idtree_active_indices = state.idtree_active_indices.clone();
        self.terminal_to_root = state.terminal_to_root.clone();
        self.terminal_root_pairs = state.terminal_root_pairs.clone();
        self.untouchables = state.untouchables.clone();
        self.bridge_affected_base_towns = state.bridge_affected_base_towns.clone();
        self.bridge_affected_indices = state.bridge_affected_indices.clone();
        self.bridge_affected_terminals = state.bridge_affected_terminals.clone();
    }

    fn init_terminal_pairs(&mut self, terminal_pairs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
        let mut terminal_idx_pairs = Vec::with_capacity(terminal_pairs.len());
        let waypoint_to_index = &self.exploration.waypoint_to_index;
        for (t, r) in terminal_pairs {
            let t_idx = *waypoint_to_index.get(&t).unwrap();
            let r_idx = if r == SUPER_ROOT {
                self.has_super_terminal = true;
                SUPER_ROOT
            } else {
                *waypoint_to_index.get(&r).unwrap()
            };
            terminal_idx_pairs.push((t_idx, r_idx));
            self.terminal_to_root.insert(t_idx, r_idx);
            self.terminal_root_pairs.insert((t_idx, r_idx));
        }
        terminal_idx_pairs
    }

    /// Induce subgraph from self.ref_graph using node indices
    pub(crate) fn ref_subgraph_stable(&self, indices: &IntSet<usize>) -> StableUnGraph<(), usize> {
        self.exploration.ref_ungraph.filter_map(
            |node_idx, _| {
                if indices.contains(&node_idx.index()) {
                    Some(())
                } else {
                    None
                }
            },
            |_, edge_idx| Some(*edge_idx),
        )
    }

    /// Set of all terminals, fixed roots and leaf terminal parents
    fn generate_untouchables(&mut self) {
        // TODO: Rewrite this to take advantage of reduced graph transforms/pendants
        //       so that all fixed nodes are untouchable.
        self.untouchables.clear();
        self.untouchables.extend(self.terminal_to_root.keys());
        self.untouchables.extend(self.terminal_to_root.values());
        self.untouchables.remove(&SUPER_ROOT);

        // Add unambigous connected nodes (degree 1)...
        for &node in self.untouchables.clone().iter() {
            if self.neighbors[node].len() == 1 {
                self.untouchables.insert(self.neighbors[node][0]);
            }
        }
    }

    fn idtree_weight(&self) -> (FixedBitSet, usize) {
        let active_nodes = self.idtree_active_indices.clone();
        let total_weight: u32 = active_nodes.ones().map(|i| self.weights[i]).sum();
        (active_nodes, total_weight as usize)
    }

    /// Populates the IDTree and initializes idtree_active_indices
    pub(crate) fn populate_idtree(&mut self, x: &IntSet<usize>) {
        self.ref_subgraph_stable(x)
            .edge_indices()
            .for_each(|edge_idx| {
                let (u, v) = self
                    .exploration
                    .ref_ungraph
                    .edge_endpoints(edge_idx)
                    .unwrap();
                self.idtree.insert_edge(u.index(), v.index());
            });

        self.idtree_active_indices = self.idtree.active_nodes_bitset();
    }

    /////
    // MARK: Connectivity Testing
    /////

    /// Attempt removals of each node in ordered_removables.
    ///
    /// NOTE: This function should only be entered after terminal pairs connected check succeeds.
    pub(crate) fn remove_removables(
        &mut self,
        ordered_removables: &Vec<usize>,
    ) -> (Vec<usize>, Vec<(usize, usize)>) {
        let mut freed = Vec::new();
        let mut freed_edges = Vec::new();
        let mut active_neighbors = Vec::<(usize, usize)>::with_capacity(4);

        for &u in ordered_removables {
            active_neighbors.clear();
            let mut need_check = false;

            // Simulate removal by isolating the idtree node
            // NOTE: Ordered removables, during the improve_component stage, are filtered
            //       to only include active nodes from the component being tested.
            for &v in &self.idtree.neighbors_smallvec(u) {
                match self.idtree.delete_edge(u, v) {
                    // nothing removed
                    -1 => continue,
                    // adjacency removed OR replacement found
                    0 | 1 => active_neighbors.push((u, v)),
                    // new component
                    2 => {
                        active_neighbors.push((u, v));
                        need_check = true;
                    }
                    _ => unreachable!(),
                }
            }

            if active_neighbors.is_empty() {
                continue;
            }

            // NOTE: Removal of non base town leaf nodes can not disconnect terminal pairs
            //       since removables is already disjoint from untouchables.
            if active_neighbors.len() == 1 && !self.bridge_affected_base_towns.contains(&u) {
                self.bridge_affected_indices.remove(u);
                freed.push(u);
                freed_edges.extend_from_slice(&active_neighbors);
                continue;
            }

            if need_check && self.terminal_pairs_connected() {
                // Finalize removal
                self.bridge_affected_indices.remove(u);
                self.bridge_affected_base_towns.remove(&u);
                freed.push(u);
                freed_edges.extend_from_slice(&active_neighbors);
            } else if need_check {
                // Restore connectivity
                for &(u, v) in &active_neighbors {
                    self.idtree.insert_edge(u, v);
                }
            }
        }

        (freed, freed_edges)
    }

    /// Check all `bridge_affected_terminals` pair connectivity.
    fn terminal_pairs_connected(&self) -> bool {
        self.bridge_affected_terminals
            .iter()
            .all(|(terminal, root)| self.terminal_is_connected(*terminal, *root))
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

    /// Updates self._bridge_* variables with relevant bridged component nodes.
    pub(crate) fn update_bridge_affected_nodes(&mut self, affected_component: FixedBitSet) {
        self.bridge_affected_terminals = self.terminal_root_pairs.clone();
        self.bridge_affected_terminals
            .retain(|p| affected_component.contains(p.0));

        self.bridge_affected_base_towns = self.exploration.base_town_indices.clone();
        self.bridge_affected_base_towns
            .retain(|&b| affected_component.contains(b));

        self.bridge_affected_indices = affected_component;
    }

    /// Bridge heuristic: find and utilize potential bridges to _increase_
    /// cycle counts and then identify removable non-articulation points
    /// that can improve the solution.
    fn bridge_heuristics(&mut self, ordered_removables: &mut Vec<usize>) {
        let mut incumbent_indices = self.idtree_active_indices.clone();
        let mut seen_before_cache: IntSet<u64> =
            IntSet::with_capacity_and_hasher(128, BuildNoHashHasher::default());

        let mut improved = true;
        while improved {
            improved = false;

            let bridge_generator = std::mem::take(&mut self.bridge_generator);
            {
                let mut bridge_gen = bridge_generator.generate_bridges(incumbent_indices.clone());
                while let CoroutineState::Yielded(bridge) = bridge_gen.as_mut().resume(()) {
                    let reisolate_bridge_nodes: Vec<usize> = bridge
                        .iter()
                        .filter(|&&v| !incumbent_indices.contains(v))
                        .copied()
                        .collect();

                    self.connect_bridge(&bridge);

                    let Some(bridge_rooted_cycles) = self.bridge_rooted_cycles(&bridge) else {
                        self.idtree.isolate_nodes(reisolate_bridge_nodes);
                        self.idtree_active_indices = incumbent_indices.clone();
                        continue;
                    };

                    self.bridge_all_cycle_nodes.clear();
                    self.bridge_all_cycle_nodes
                        .extend(bridge_rooted_cycles.iter().flat_map(|c| c.iter().copied()));

                    if self.was_seen_before(&bridge, &mut seen_before_cache) {
                        self.idtree.isolate_nodes(reisolate_bridge_nodes);
                        self.idtree_active_indices = incumbent_indices.clone();
                        continue;
                    }

                    let Some(removal_candidates) = self.removal_candidates(&bridge) else {
                        self.idtree.isolate_nodes(reisolate_bridge_nodes);
                        self.idtree_active_indices = incumbent_indices.clone();
                        continue;
                    };

                    let (is_improved, _removal_attempts, freed) =
                        self.improve_component(&bridge, &removal_candidates, ordered_removables);

                    if is_improved {
                        incumbent_indices = self.idtree.active_nodes_bitset();
                        self.idtree_active_indices = incumbent_indices.clone();
                        improved = true;

                        ordered_removables.retain(|v| !freed.contains(v));
                        ordered_removables.extend(self.sort_by_weights(&bridge));
                        break;
                    }

                    self.idtree.isolate_nodes(reisolate_bridge_nodes);
                    self.idtree_active_indices = incumbent_indices.clone();
                }
            }
            self.bridge_generator = bridge_generator;
        }
    }

    fn connect_bridge(&mut self, bridge: &[usize]) {
        let mut tmp: Vec<_> = bridge.to_vec();
        let mut moved_node = true;
        let neighbors = &self.neighbors;
        while !tmp.is_empty() && moved_node {
            moved_node = false;
            let mut i = 0;
            while i < tmp.len() {
                let v = tmp[i];
                let mut inserted_active_neighbor = false;

                for &u in neighbors[v]
                    .iter()
                    .filter(|&&n| self.idtree_active_indices.contains(n))
                {
                    if self.idtree.insert_edge(v, u) != -1 {
                        inserted_active_neighbor = true;
                    }
                }

                if inserted_active_neighbor {
                    self.idtree_active_indices.insert(v);
                    tmp.swap_remove(i);
                    moved_node = true;
                } else {
                    i += 1;
                }
            }
        }
    }

    fn bridge_rooted_cycles(&mut self, bridge: &[usize]) -> Option<Vec<Vec<usize>>> {
        let root = *bridge.first()?;
        let all_cycles = self.idtree.cycle_basis(Some(root));
        let filtered: Vec<Vec<usize>> = all_cycles
            .into_iter()
            .filter(|c| c.len() >= (2 + bridge.len()) && c.iter().any(|v| bridge.contains(v)))
            .collect();
        (!filtered.is_empty()).then_some(filtered)
    }

    fn was_seen_before(&mut self, bridge: &[usize], seen_before: &mut IntSet<u64>) -> bool {
        self.scratch_nodes.clear();
        self.scratch_nodes.extend_from_slice(bridge);
        self.scratch_nodes
            .extend_from_slice(&self.bridge_all_cycle_nodes);
        self.scratch_nodes.sort_unstable();

        self.hash_buf.clear();
        for &x in &self.scratch_nodes {
            self.hash_buf.extend_from_slice(&x.to_le_bytes());
        }
        let mut hasher = RapidHasher::default();
        hasher.write(&self.hash_buf);
        let all_hash = hasher.finish();
        !seen_before.insert(all_hash)
    }

    fn removal_candidates(&mut self, bridge: &[usize]) -> Option<Vec<(usize, usize)>> {
        let cycle_degree_threshold = self.cycle_degree_threshold;
        self.scratch_nodes.clear();
        self.scratch_nodes
            .extend_from_slice(&self.bridge_all_cycle_nodes);
        self.scratch_nodes.sort_unstable();
        self.scratch_nodes.dedup();

        let mut idtree_candidates: Vec<(usize, usize)> =
            Vec::with_capacity(self.scratch_nodes.len());

        // Filter out untouchables and bridge members.
        for &v in &self.scratch_nodes {
            if self.untouchables.contains(&v) || bridge.contains(&v) {
                continue;
            }
            if self.idtree.degree(v) as usize <= cycle_degree_threshold {
                idtree_candidates.push((v, self.weights[v] as usize));
            }
        }

        (!idtree_candidates.is_empty()).then_some(idtree_candidates)
    }

    fn improve_component(
        &mut self,
        bridge: &[usize],
        removal_candidates: &[(usize, usize)],
        ordered_removables: &[usize],
    ) -> (bool, usize, Vec<usize>) {
        let mut active_ordered_removables = ordered_removables.to_owned();
        let mut removal_attempts = 0;
        let max_removal_attempts = self.max_removal_attempts;

        let bridged_component = self.idtree.node_connected_component_bitset(bridge[0]);
        self.update_bridge_affected_nodes(bridged_component.clone());

        let bridge_weight: u32 = bridge.iter().map(|&v| self.weights[v]).sum();
        let incumbent_component_count = self.idtree.num_connected_components();

        let removal_candidates = self.sort_removal_candidates(removal_candidates);
        let removal_set_generator = WeightedRangeComboGenerator::new(
            &removal_candidates,
            bridge_weight as usize,
            bridge.len(),
            self.exploration.max_node_weight as usize,
        );

        for (removal_set, removal_set_weight) in removal_set_generator.generate() {
            removal_attempts += 1;
            if removal_attempts > max_removal_attempts {
                break;
            }

            // Isolate the removal_set nodes
            let mut deleted_edges = Vec::new();
            for &v in &removal_set {
                for &u in self.neighbors[v]
                    .iter()
                    .filter(|&&u| bridged_component.contains(u))
                {
                    if self.idtree.delete_edge(v, u) != -1 {
                        deleted_edges.push((v, u));
                    }
                }
            }

            if !self.terminal_pairs_connected() {
                // Reconnect and try the next bridge...
                for &(v, u) in &deleted_edges {
                    self.idtree.insert_edge(v, u);
                }
                continue;
            }

            // A viable challenger has been found.
            // Update the active component and sync the bridge affected
            // collections for removal attempts of ordered_removables.
            let mut active_component_indices = bridged_component.clone();

            removal_set
                .iter()
                .for_each(|&v| active_component_indices.remove(v));

            self.update_bridge_affected_nodes(active_component_indices.clone());

            active_ordered_removables.retain(|&v| !removal_set.contains(&v));
            active_ordered_removables.retain(|&v| self.bridge_affected_indices.contains(v));

            let (mut freed, freed_edges) = self.remove_removables(&active_ordered_removables);

            if !freed.is_empty()
                || removal_set_weight > bridge_weight as usize
                || self.idtree.num_connected_components() > incumbent_component_count
            {
                freed.extend(removal_set);
                return (true, removal_attempts, freed);
            }

            // Challenger is not an improvement, replace all isolated/freed nodes
            // and continue to the next removal_set...
            for &(v, u) in &deleted_edges {
                self.idtree.insert_edge(v, u);
            }
            for &(v, u) in &freed_edges {
                self.idtree.insert_edge(v, u);
            }
            active_ordered_removables = ordered_removables.to_owned();
            continue;
        }

        (false, removal_attempts, vec![])
    }

    fn sort_by_weights(&self, numbers: &[usize]) -> Vec<usize> {
        let mut pairs: Vec<(usize, usize)> = numbers
            .iter()
            .map(|&i| self.weights[i] as usize)
            .zip(numbers.iter().cloned())
            .collect();
        pairs.sort_unstable();
        pairs.into_iter().map(|(_, number)| number).collect()
    }

    /// Sorts removal candidates by betweenness and/or weight.
    ///
    /// SAFETY: This function will not return valid betweenness values if
    ///         called prior to `update_bridge_affected_nodes`.
    fn sort_removal_candidates(
        &mut self,
        removal_candidates: &[(usize, usize)],
    ) -> Vec<(usize, usize)> {
        let mut removal_candidates = removal_candidates.to_vec();
        if self.use_betweenness {
            // NOTE: Primary order _must_ be betweenness in _ascending_ order!
            // Secondary order of weight determined by self.combo_gen_direction
            let betweenness = self.idtree.compute_subset_betweenness(
                &removal_candidates,
                &self.bridge_affected_terminals,
                &self.bridge_affected_base_towns,
                Some(SUPER_ROOT),
            );
            if self.combo_gen_direction {
                removal_candidates.sort_by_key(|&(v, w)| (betweenness[&v], -(w as i32)));
            } else {
                removal_candidates.sort_by_key(|&(v, w)| (betweenness[&v], w as i32));
            }
        } else {
            if self.combo_gen_direction {
                removal_candidates.sort_by_key(|b| std::cmp::Reverse(b.1)); // Descending
            } else {
                removal_candidates.sort_by_key(|a| a.1); // Ascending
            }
        }

        removal_candidates
    }
}
