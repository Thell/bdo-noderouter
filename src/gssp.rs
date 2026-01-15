// gssp.rs

/// Implementation of Bi-directional Dial's algorithm.
///
/// Runs the 'greedy shortest shared path' algorithm on the reference graph
/// to build an approximation to the Steiner forest problem and returns all
/// intermediate nodes on all paths in 'as added' path order.
///
/// Base reference graph details:
/// * Node weighted planar graph node weights in range 0..=3
/// * 947 nodes and 2149 edges
/// * 472 leaf nodes; 386 degree 2 to 4 nodes; 89 >= 5 degree nodes
/// * Bi-directional anti-parallel edges with head node weight as edge weight
///
/// Reduced reference graph has all leaf nodes removed.
/// Super root node is added with in edges only from base towns.
///
/// Key components:
/// * Bi-directional Dial's algorithm
/// * Batching via Primal-Dual algorithm
/// * Contraction Hierarchy cutoff generation
/// * DSTree constant connectivity query
///
use std::cmp::Reverse;
use std::collections::BTreeMap;

use fixedbitset::FixedBitSet;
use nohash_hasher::{IntMap, IntSet};
use smallvec::SmallVec;

use crate::dstree::DSTree;
use crate::gssp_data::{ExplorationData, PDBatchGenerator, canonicalize_pairs};
use crate::node_router::ExplorationNodeData;

#[derive(Clone, Debug)]
pub struct DialsRouter {
    pub exploration_data: ExplorationData,

    // reference graph
    index_to_weight: Vec<u32>,
    reduced_index_to_in_neighbors: Vec<SmallVec<[usize; 4]>>,
    reduced_index_to_out_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>>,
    reduced_index_to_in_neighbors_pos: Vec<IntMap<usize, usize>>,

    batch_generator: PDBatchGenerator,
    dstree: DSTree,

    // Dial's BiDir
    forward_data: Vec<(u32, u32)>, // (distance, predecessor)
    reverse_data: Vec<(u32, u32)>,
    forward_buckets: [Vec<usize>; 4],
    reverse_buckets: [Vec<usize>; 4],
    dials_touched_indices: FixedBitSet,
}

impl DialsRouter {
    pub fn new(nodes_map: BTreeMap<usize, ExplorationNodeData>) -> Self {
        let ed = ExplorationData::new(nodes_map);
        let index_to_weight = ed.index_to_weight.clone();

        // We need to keep both full and reduced maps since the single pair query is
        // done on the full graph and the batch query is done on the reduced graph.
        let reduced_index_to_in_neighbors = ed.reduced_index_to_in_neighbors.clone();
        let reduced_index_to_out_neighbors_weighted =
            ed.reduced_index_to_out_neighbors_weighted.clone();
        let reduced_index_to_in_neighbors_pos = ed.reduced_index_to_in_neighbors_pos.clone();

        let batch_generator = PDBatchGenerator::new(&ed);

        // Initialize with empty adj to populate only the nodes of the DSTree.
        let empty_adj: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::with_capacity(4); ed.num_nodes];
        let dstree = DSTree::new(&empty_adj, true);

        let num_nodes = ed.num_nodes;

        DialsRouter {
            exploration_data: ed,
            index_to_weight,
            reduced_index_to_in_neighbors,
            reduced_index_to_out_neighbors_weighted,
            reduced_index_to_in_neighbors_pos,
            batch_generator,
            dstree,
            forward_data: vec![(u32::MAX, u32::MAX); num_nodes],
            reverse_data: vec![(u32::MAX, u32::MAX); num_nodes],
            forward_buckets: std::array::from_fn(|_| Vec::with_capacity(32)),
            reverse_buckets: std::array::from_fn(|_| Vec::with_capacity(32)),
            dials_touched_indices: FixedBitSet::with_capacity(num_nodes),
        }
    }

    fn rebuild_path_bidir_from(
        &self,
        start_node: usize,
        goal_node: usize,
        meet_node: usize,
        forward_data: &[(u32, u32)],
        reverse_data: &[(u32, u32)],
    ) -> Vec<usize> {
        let mut start_to_meet_path = Vec::new();
        let mut current_node = meet_node;

        while current_node != start_node {
            start_to_meet_path.push(current_node);
            let predecessor = forward_data[current_node].1;
            if predecessor == u32::MAX {
                return Vec::new();
            }
            current_node = predecessor as usize;
        }
        start_to_meet_path.push(start_node);
        start_to_meet_path.reverse();

        let mut meet_to_goal_path = Vec::new();
        current_node = meet_node;

        while current_node != goal_node {
            let predecessor = reverse_data[current_node].1;
            if predecessor == u32::MAX {
                return Vec::new();
            }
            current_node = predecessor as usize;
            meet_to_goal_path.push(current_node);
        }

        start_to_meet_path
            .into_iter()
            .chain(meet_to_goal_path)
            .collect()
    }

    pub fn dials_path(
        &mut self,
        inbound_neighbors: &[SmallVec<[usize; 4]>],
        weights: &[u32],
        outbound_neighbors_weighted: &[SmallVec<[(usize, u32); 4]>],
        start_node: usize,
        goal_node: usize,
        cutoff_distance: u32,
    ) -> Option<(u32, usize)> {
        const MAX_WEIGHT: usize = 3;

        let n = outbound_neighbors_weighted.len();
        if start_node >= n || goal_node >= n {
            return None;
        }
        if start_node == goal_node {
            return Some((0, start_node));
        }

        for idx in self.dials_touched_indices.ones() {
            self.forward_data[idx] = (u32::MAX, u32::MAX); // (distance, predecessor)
            self.reverse_data[idx] = (u32::MAX, u32::MAX);
        }
        for bucket in &mut self.forward_buckets {
            bucket.clear();
        }
        for bucket in &mut self.reverse_buckets {
            bucket.clear();
        }
        self.dials_touched_indices.clear();

        self.forward_data[start_node] = (0, u32::MAX);
        self.reverse_data[goal_node] = (0, u32::MAX);

        self.forward_buckets[0].push(start_node);
        self.reverse_buckets[0].push(goal_node);

        self.dials_touched_indices.insert(start_node);
        self.dials_touched_indices.insert(goal_node);

        let mut forward_layer: u32 = 0;
        let mut reverse_layer: u32 = 0;
        let mut forward_bucket_index: usize = 0;
        let mut reverse_bucket_index: usize = 0;

        let mut best_distance = cutoff_distance;
        let mut best_meet_node: Option<usize> = None;

        let mut have_forward = false;
        let mut have_reverse = false;

        loop {
            let mut made_progress = false;

            if forward_layer < best_distance {
                let next_forward_node = loop {
                    if let Some(current_node) = self.forward_buckets[forward_bucket_index].pop() {
                        break Some(current_node);
                    }
                    forward_layer += 1;
                    if forward_layer >= best_distance {
                        break None;
                    }
                    forward_bucket_index = (forward_bucket_index + 1) & MAX_WEIGHT;
                };

                if let Some(current_node) = next_forward_node {
                    made_progress = true;
                    have_forward = true;

                    let current_distance = self.forward_data[current_node].0;

                    if self.reverse_data[current_node].0 != u32::MAX {
                        let candidate = current_distance + self.reverse_data[current_node].0;
                        if candidate < best_distance {
                            best_distance = candidate;
                            best_meet_node = Some(current_node);
                        }
                    }

                    for &(neighbor, weight) in &outbound_neighbors_weighted[current_node] {
                        let new_distance = current_distance + weight;
                        if new_distance < self.forward_data[neighbor].0
                            && new_distance < best_distance
                        {
                            self.dials_touched_indices.insert(neighbor);
                            self.forward_data[neighbor] = (new_distance, current_node as u32);
                            self.forward_buckets[(new_distance as usize) & MAX_WEIGHT]
                                .push(neighbor);
                        }
                    }
                }
            }

            if reverse_layer < best_distance {
                let next_reverse_node = loop {
                    if let Some(node) = self.reverse_buckets[reverse_bucket_index].pop() {
                        break Some(node);
                    }
                    reverse_layer += 1;
                    if reverse_layer >= best_distance {
                        break None;
                    }
                    reverse_bucket_index = (reverse_bucket_index + 1) & MAX_WEIGHT;
                };

                if let Some(current_node) = next_reverse_node {
                    made_progress = true;
                    have_reverse = true;

                    let current_distance = self.reverse_data[current_node].0;

                    if self.forward_data[current_node].0 != u32::MAX {
                        let candidate = current_distance + self.forward_data[current_node].0;
                        if candidate < best_distance {
                            best_distance = candidate;
                            best_meet_node = Some(current_node);
                        }
                    }

                    let current_node_weight = weights[current_node];

                    for &neighbor in &inbound_neighbors[current_node] {
                        let new_distance = current_distance + current_node_weight;
                        if new_distance < self.reverse_data[neighbor].0
                            && new_distance < best_distance
                        {
                            self.dials_touched_indices.insert(neighbor);
                            self.reverse_data[neighbor] = (new_distance, current_node as u32);
                            self.reverse_buckets[(new_distance as usize) & MAX_WEIGHT]
                                .push(neighbor);
                        }
                    }
                }
            }

            if !made_progress {
                break;
            }

            if have_forward && have_reverse && (forward_layer + reverse_layer >= best_distance) {
                break;
            }
        }

        best_meet_node.map(|m| (best_distance, m))
    }

    pub fn greedy_shortest_shared_paths(
        &mut self,
        pairs: &[(usize, usize)],
    ) -> (IntSet<usize>, Vec<usize>) {
        let super_root_index = self.exploration_data.waypoint_to_index[&99999];
        let mut ordered_removables = Vec::with_capacity(self.exploration_data.num_nodes);

        // Replace all instances of SUPER_ROOT with super_root_index
        let pairs = pairs
            .iter()
            .map(|&(s, t)| {
                assert!(s != 99999, "super root must not be a source node!");
                if t == 99999 {
                    (s, super_root_index)
                } else {
                    (s, t)
                }
            })
            .collect::<Vec<_>>();

        // NOTE: Sorting pairs first by target then by source gives better approximations.
        let mut pairs = canonicalize_pairs(&pairs);
        pairs.sort_by_key(|&(s, t)| (t, s));

        // Pair reduction...
        // NOTE: We need to maintain the original pairs for the deduplication step before returning.
        let (working_pairs, removables) = self
            .exploration_data
            .transform_pairs_to_reduced_pairs(&pairs);
        ordered_removables.extend(removables);

        let inbound_neighbors = self.reduced_index_to_in_neighbors.clone();
        let out_nbr_pos_maps = self.reduced_index_to_in_neighbors_pos.clone();
        // zero-weight augmentation occurs in these two data structures
        let mut outbound_neighbors_weighted = self.reduced_index_to_out_neighbors_weighted.clone();
        let mut weights = self.index_to_weight.clone();

        // All terminals of all pairs are paid for in advance with zero-weight augmentation
        working_pairs.iter().for_each(|pair| {
            for v in [pair.0, pair.1] {
                weights[v] = 0;
                let pos_map = &out_nbr_pos_maps[v];
                for &n in inbound_neighbors[v].iter() {
                    outbound_neighbors_weighted[n][pos_map[&n]].1 = 0;
                }
            }
        });

        let mut cutoffs_heap = self
            .exploration_data
            .generate_cutoff_heap(&working_pairs, Some(true));
        self.dstree.reset_all_edges();
        let mut used_pairs = FixedBitSet::with_capacity(working_pairs.len());

        // Path rebuilding for the best pair of each round of augmentation.
        // SAFETY: Can not rely on struct's predecessor arrays as they are scratch space.
        let mut best_forward_data = vec![(u32::MAX, u32::MAX); self.forward_data.len()];
        let mut best_reverse_data = vec![(u32::MAX, u32::MAX); self.reverse_data.len()];
        let mut best_meet: usize = usize::MAX;

        let pair_index_to_pair_key: IntMap<usize, (usize, usize)> = working_pairs
            .iter()
            .enumerate()
            .map(|(i, &(s, t))| (i, (s, t)))
            .collect();

        let batches = self.batch_generator.generate_pair_index_batches(
            &working_pairs,
            &pair_index_to_pair_key,
            20,
        );

        for batch in batches {
            let mut use_cutoff = true;
            let expected_ones_count = batch.len() + used_pairs.count_ones(..);

            loop {
                // Extract tightest remaining cutoff from heap (skipping used pairs)
                // This serves as the starting weight upper bound for the current batch.
                let mut initial_cutoff = u32::MAX;
                if use_cutoff {
                    while let Some(Reverse((w, idx))) = cutoffs_heap.pop() {
                        if !used_pairs.contains(idx) {
                            initial_cutoff = w;
                            cutoffs_heap.push(Reverse((w, idx)));
                            break;
                        }
                    }
                }

                let mut best_index = usize::MAX;
                let mut best_weight = initial_cutoff;

                for &pair_index in &batch {
                    if used_pairs.contains(pair_index) {
                        continue;
                    }

                    let &(s, t) = &pair_index_to_pair_key[&pair_index];

                    // Capture pairs connected as a side-effect of connecting other pairs.
                    if t != super_root_index && self.dstree.query(s, t) {
                        if let Some(original_idx) = working_pairs.iter().position(|&p| p == (s, t))
                        {
                            used_pairs.insert(original_idx);
                        }
                        continue;
                    }

                    if let Some((w, meet)) = self.dials_path(
                        &inbound_neighbors,
                        &weights,
                        &outbound_neighbors_weighted,
                        s,
                        t,
                        best_weight,
                    ) && w < best_weight
                    {
                        best_index = pair_index;
                        best_weight = w;

                        // SAFETY: Struct's predecessor arrays are scratch arrays so capture
                        // them before we overwrite them with the next pair's predecessors
                        best_forward_data.copy_from_slice(&self.forward_data);
                        best_reverse_data.copy_from_slice(&self.reverse_data);
                        best_meet = meet;

                        if w == 0 {
                            break;
                        }
                    }
                }

                if best_index == usize::MAX {
                    let cutoff_failed = used_pairs.count_ones(..) != expected_ones_count;
                    if use_cutoff && cutoff_failed {
                        use_cutoff = false;
                        continue;
                    }
                    break;
                }
                used_pairs.insert(best_index);

                let (s, t) = pair_index_to_pair_key[&best_index];
                let path = self.rebuild_path_bidir_from(
                    s,
                    t,
                    best_meet,
                    &best_forward_data,
                    &best_reverse_data,
                );

                ordered_removables.extend(&path);

                // Zero-weight augmentation to encourage shared paths.
                // Path nodes are settled (incoming edges are free) and intermediates are connected.
                let mut u = s;
                for v in path {
                    weights[v] = 0;
                    let pos_map = &out_nbr_pos_maps[v];
                    for &n in inbound_neighbors[v].iter() {
                        outbound_neighbors_weighted[n][pos_map[&n]].1 = 0;
                    }
                    if v != super_root_index {
                        self.dstree.insert_edge(u, v);
                    }
                    u = v;
                }
            }
        }

        let mut visited = IntSet::from_iter(ordered_removables.iter().copied());
        for &(s, t) in &pairs {
            visited.insert(s);
            visited.insert(t);
        }
        // SAFETY: DO NOT LEAK SUPER ROOT!
        visited.remove(&super_root_index);

        let mut seen = IntSet::default();
        for &(s, t) in &working_pairs {
            seen.insert(s);
            seen.insert(t);
        }
        ordered_removables.retain(|&i| seen.insert(i));

        (visited, ordered_removables)
    }
}
