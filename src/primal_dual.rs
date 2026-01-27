// gssp_data.rs

use nohash_hasher::{BuildNoHashHasher, IntMap, IntSet};
use petgraph::algo::tarjan_scc;
use petgraph::prelude::StableDiGraph;
use rapidhash::{RapidHashMap, RapidHashSet};
use smallvec::SmallVec;

use crate::node_router::SharedExplorationData;

// MARK: - PDBatchGenerator

/// Primal-dual batch generator
///
/// SAFETY: This is safe to use on both ref_graph and reduced_ref_graph.
///         Since reduced_ref_graph is a subgraph of ref_graph,
///         any pair connectivity violation on reduced_ref_graph is also a violation on ref_graph.
///         Since only leaf nodes are removed from ref_graph,
///         any pair connectivity violation on ref_graph is also a violation on reduced_ref_graph
///         if and only if the pair is not a leaf and its' parent. (leaf, parent) is not a violation
///         by definition.
#[derive(Clone, Debug)]
pub struct PDBatchGenerator {
    exploration: SharedExplorationData,

    settled_nodes: IntSet<usize>, // nodes
    paid_weight: Vec<u32>,        // weights
    connected_pairs: RapidHashSet<(usize, usize)>,
    terminal_to_root: Vec<(usize, usize)>,
}

impl PDBatchGenerator {
    pub fn new(exploration: SharedExplorationData) -> Self {
        let num_nodes = exploration.num_nodes;
        Self {
            exploration,
            settled_nodes: IntSet::default(),
            paid_weight: vec![0; num_nodes],
            connected_pairs: RapidHashSet::default(),
            terminal_to_root: Vec::new(),
        }
    }

    /// Induce subgraph from reference graph using node indices.
    fn ref_subgraph_stable(&self, indices: &IntSet<usize>) -> StableDiGraph<(), usize> {
        self.exploration.ref_digraph.filter_map(
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

    /// Finds and returns nodes not in settlement with neighbors in settlement.
    fn find_frontier_nodes(&self, settlement: &IntSet<usize>) -> IntSet<usize> {
        let mut frontier = IntSet::with_capacity_and_hasher(
            self.exploration.num_nodes,
            BuildNoHashHasher::default(),
        );
        frontier.extend(
            settlement
                .iter()
                .flat_map(|&v| &self.exploration.index_to_out_neighbors[v])
                .filter(|n| !settlement.contains(n)),
        );
        frontier
    }

    /// Returns true if any violated sets are found
    ///
    /// SAFETY: This function drives the PDBatchGenerator via side effects!
    /// - Mutates `violated` such that it contains all violated sets
    /// - Mutates `self.connected_pairs` such that it contains all connected pairs
    ///
    /// Requires:
    /// - exploration_data ensuring that if super root is present it is weakly connected
    ///   from all base towns (base_town -> super_root).
    /// - base_town_indices
    ///
    /// SAFETY: Since ref_graph's super_root is inbound connections only, it is assumed
    ///         that the super_root is always isolated. Therefore we need to utilize
    ///         the base_town_indices to check if a super terminal is in the current set
    ///         with a base town, meaning it connects to the super_root.
    fn update_violations(&mut self, violated: &mut IntSet<usize>) -> bool {
        let super_root_index = self.exploration.super_root_index;

        let x = &self.settled_nodes;
        let subgraph = self.ref_subgraph_stable(x);

        let components: Vec<IntSet<usize>> = tarjan_scc(&subgraph)
            .into_iter()
            .map(|comp| comp.iter().map(|nidx| nidx.index()).collect())
            .collect();

        for cc in &components {
            // NOTE: PD Approximation is purely additive so we can safely avoid duplicated checks.
            let tmp_connected_pairs = self.connected_pairs.clone();
            let active_pairs = self
                .terminal_to_root
                .iter()
                .filter(|&(s, t)| !tmp_connected_pairs.contains(&(*s, *t)));

            for (s, t) in active_pairs {
                let terminal_in_cc = cc.contains(s);

                let root_in_cc = if t == &super_root_index {
                    cc.intersection(&self.exploration.base_town_indices)
                        .next()
                        .is_some()
                } else {
                    cc.contains(t)
                };

                if !terminal_in_cc && !root_in_cc {
                    continue;
                }
                if terminal_in_cc && root_in_cc {
                    self.connected_pairs.insert((*s, *t));
                } else {
                    violated.extend(cc.iter().cloned());
                    break;
                }
            }
        }

        !violated.is_empty()
    }

    /// Accumulate connected pairs of each tightening of violated pairs into batches.
    ///
    /// # Requires
    /// - super_root must not be a source node in any pair
    ///
    ///
    /// Returns a vector of batches.
    fn primal_dual_batch_generator(
        &mut self,
        pairs: &[(usize, usize)],
    ) -> Vec<SmallVec<[(usize, usize); 4]>> {
        let num_nodes = self.exploration.num_nodes;
        let super_root_index = self.exploration.super_root_index;
        let index_to_weight = self.exploration.index_to_weight.clone();

        // Initialize terminal_to_root mappings for violated pairs
        self.terminal_to_root.clear();
        for &(s, t) in pairs {
            assert!(
                s != super_root_index,
                "super root must not be a source node!"
            );
            self.terminal_to_root.push((s, t));
        }

        // Initialize settlement with all terminals from pairs _excluding_ super root.
        self.settled_nodes.clear();
        self.settled_nodes
            .extend(pairs.iter().flat_map(|&(s, t)| [s, t]));
        self.settled_nodes.remove(&super_root_index);

        // Initialize paid weights where all terminals from pairs have pre-paid weight
        self.paid_weight = vec![0; num_nodes];
        for &idx in self.settled_nodes.iter() {
            self.paid_weight[idx] = index_to_weight[idx];
        }

        // Initialize connected pairs
        self.connected_pairs.clear();
        let mut previously_connected_pairs = self.connected_pairs.clone();

        let mut batches = Vec::new();
        let mut violated =
            IntSet::with_capacity_and_hasher(num_nodes, BuildNoHashHasher::default());

        while self.update_violations(&mut violated) {
            let violated_frontier = self.find_frontier_nodes(&violated);
            for &v in violated_frontier.iter() {
                self.paid_weight[v] += 1;
                if self.paid_weight[v] >= index_to_weight[v] {
                    self.settled_nodes.insert(v);
                }
            }
            violated.clear();

            if previously_connected_pairs.len() != self.connected_pairs.len() {
                let mut batch = SmallVec::new();
                for &(s, t) in self.connected_pairs.difference(&previously_connected_pairs) {
                    batch.push((s, t));
                }
                batch.sort();
                batches.push(batch);
                previously_connected_pairs = self.connected_pairs.clone();
            }
        }

        // Capture the last batch
        if previously_connected_pairs.len() != self.connected_pairs.len() {
            let mut batch = SmallVec::new();
            for &(s, t) in self.connected_pairs.difference(&previously_connected_pairs) {
                batch.push((s, t));
            }
            batch.sort();
            batches.push(batch);
        }

        batches
    }

    /// Generates batches of original pair indices (0..pairs.len()).
    ///
    /// Requires that:
    /// - super_root is not a source node in any pair.
    ///
    /// When pairs.len() ≤ batching_threshold, returns a single batch of all indices.
    pub fn generate_pair_index_batches(
        &mut self,
        pairs: &[(usize, usize)],
        pair_index_to_pair_key: &IntMap<usize, (usize, usize)>,
        batching_threshold: usize,
    ) -> Vec<SmallVec<[usize; 4]>> {
        if pairs.len() <= batching_threshold {
            return vec![(0..pairs.len()).collect::<SmallVec<[usize; 4]>>()];
        }

        let pair_key_to_index: RapidHashMap<(usize, usize), usize> = pair_index_to_pair_key
            .iter()
            .map(|(&idx, &key)| (key, idx))
            .collect();

        self.primal_dual_batch_generator(pairs)
            .into_iter()
            .map(|batch| batch.iter().map(|key| pair_key_to_index[key]).collect())
            .collect()
    }
}

#[allow(unused)]
pub fn canonicalize_pairs(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    pairs
        .iter()
        .map(|&(s, t)| if s <= t { (s, t) } else { (t, s) })
        .collect()
}
