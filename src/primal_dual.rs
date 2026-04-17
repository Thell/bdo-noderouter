// gssp_data.rs

use nohash_hasher::{BuildNoHashHasher, IntMap, IntSet};
use petgraph::algo::tarjan_scc;
use petgraph::prelude::StableDiGraph;
use rapidhash::{RapidHashMap, RapidHashSet};
use smallvec::SmallVec;

use crate::exploration_data::SUPER_ROOT;
use crate::{NodeRouter, node_router::SharedExplorationData};

// MARK: PD Approximation

/// Solves the routing problem for a given NodeRouter and returns a list of active nodes in solution
///
/// NOTE: Mutates the NodeRouter to populate required data structures for post processing.
pub(crate) fn approximate(nr: &mut NodeRouter) -> Vec<usize> {
    let mut x = nr.untouchables.clone();
    if nr.has_super_terminal {
        augment_superterminal_roots(nr, &mut x);
    }

    let (x, mut ordered_removables) = primal_dual_approximation(nr, x);
    nr.populate_idtree(&x);

    // Ordered removables are in temporal order of going tight, sub ordered structurally
    // by sorting by waypoint key.  When removing the nodes they should be processed in
    // reverse order to facilitate the removal of the latest nodes to 'go tight' first.
    // The list is reversed here and processed in forward order thoughout the remainder
    // of the algorithm and bridge heuristic processing.
    ordered_removables = ordered_removables.iter().rev().cloned().collect();

    // remove_removables is setup to primarily handle 'bridged' components in the Bridge
    // Heuristic. To simplify the code for the approximation removal testings the bridge
    // related variables are set here to cover all removables, terminals and base towns
    // in the graph.
    nr.update_bridge_affected_nodes(nr.idtree_active_indices.clone());

    let (freed, _freed_edges) = nr.remove_removables(&ordered_removables);

    freed
        .iter()
        .for_each(|&v| nr.idtree_active_indices.remove(v));
    ordered_removables.retain(|&v| !freed.contains(&v));

    ordered_removables
}

/// Augments initial approximation set with super-terminal potential roots when
/// that root is nearer than any existing rooted node in the current approximation set.
fn augment_superterminal_roots(nr: &mut NodeRouter, x: &mut IntSet<usize>) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut working_roots = nr.terminal_to_root.clone();
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
            heap.push((Reverse(0), Reverse(terminal)));

            while let Some((Reverse(cost), Reverse(node))) = heap.pop() {
                if !visited.insert(node) {
                    continue;
                }

                if node != terminal {
                    let is_rooted = match working_roots.get(&node) {
                        Some(root) if root != &SUPER_ROOT => true,
                        None => nr.exploration.base_town_indices.contains(&node),
                        _ => false,
                    };

                    if is_rooted {
                        super_terminal_distances.push((terminal, node, cost));
                        break;
                    }
                }

                for &neighbor in &nr.neighbors[node] {
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    let next_cost = if x.contains(&neighbor) {
                        cost
                    } else {
                        cost + nr.weights[neighbor] as usize
                    };
                    heap.push((Reverse(next_cost), Reverse(neighbor)));
                }
            }
        }

        super_terminal_distances.sort_by_key(|&(_, _, cost)| cost);
        let (terminal, target, _cost) = super_terminal_distances[0];
        x.insert(target);
        working_roots.insert(terminal, target);
        pending_super_terminals.remove(&terminal);
    }
}

/// Node Weighted Primal Dual Approximation (Demaine et al.)
fn primal_dual_approximation(
    nr: &mut NodeRouter,
    mut x: IntSet<usize>,
) -> (IntSet<usize>, Vec<usize>) {
    // The main loop operations and frontier node calculations are set based.
    // The violated sets identification requires subgraphing the ref_graph and running
    // connected_components. The loop usually only iterates a half dozen times.
    let mut y = vec![0; nr.exploration.ref_ungraph.node_count()];
    let mut ordered_removables = Vec::new();
    let mut violated = IntSet::with_capacity_and_hasher(
        nr.exploration.ref_ungraph.node_count(),
        BuildNoHashHasher::default(),
    );

    let mut connected_pairs: RapidHashSet<(usize, usize)> = RapidHashSet::default();

    while violated_sets(nr, &x, &mut violated, &mut connected_pairs) {
        for v in find_frontier_nodes(nr, &violated) {
            y[v] += 1;
            if y[v] >= nr.weights[v] {
                x.insert(v);
                ordered_removables.push(v);
            }
        }
        violated.clear();
    }

    (x, ordered_removables)
}

/// Returns connected components violating connectivity constraints.    
fn violated_sets(
    nr: &mut NodeRouter,
    x: &IntSet<usize>,
    violated: &mut IntSet<usize>,
    connected_pairs: &mut RapidHashSet<(usize, usize)>,
) -> bool {
    // Compute connected components (undirected graph)
    let subgraph = nr.ref_subgraph_stable(x);
    let components: Vec<IntSet<usize>> = tarjan_scc(&subgraph)
        .into_iter()
        .map(|comp| comp.iter().map(|nidx| nidx.index()).collect())
        .collect();

    for cc in &components {
        // Since the pd approximation is additive we can safely avoid duplicate checks.
        let tmp_connected_pairs = connected_pairs.clone();
        let active_terminals = nr
            .terminal_to_root
            .iter()
            .filter(|p| !tmp_connected_pairs.contains(&(*p.0, *p.1)));

        for (&terminal, &root) in active_terminals {
            let terminal_in_cc = cc.contains(&terminal);

            let root_in_cc = if root == SUPER_ROOT {
                cc.intersection(&nr.exploration.base_town_indices)
                    .next()
                    .is_some()
            } else {
                cc.contains(&root)
            };

            if !terminal_in_cc && !root_in_cc {
                continue;
            }
            if terminal_in_cc && root_in_cc {
                connected_pairs.insert((terminal, root));
            } else {
                violated.extend(cc.iter().cloned());
                break;
            }
        }
    }

    !violated.is_empty()
}

/// Finds and returns nodes not in settlement with neighbors in settlement.
fn find_frontier_nodes(nr: &mut NodeRouter, settlement: &IntSet<usize>) -> IntSet<usize> {
    let mut frontier = IntSet::with_capacity_and_hasher(
        nr.exploration.ref_ungraph.node_count(),
        BuildNoHashHasher::default(),
    );
    frontier.extend(
        settlement
            .iter()
            .flat_map(|&v| &nr.neighbors[v])
            .filter(|n| !settlement.contains(n)),
    );
    frontier
}

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
