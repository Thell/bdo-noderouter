// gssp_data.rs

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap},
    fmt,
};

use fast_paths::{FastGraph, InputGraph, Params, PathCalculator};
use nohash_hasher::{BuildNoHashHasher, IntMap, IntSet};
use petgraph::algo::{astar, tarjan_scc};
use petgraph::{Direction, graph::NodeIndex, prelude::StableDiGraph};
use rapidhash::{RapidHashMap, RapidHashSet};
use smallvec::SmallVec;

use crate::node_router::ExplorationNodeData;

#[derive(serde::Deserialize)]
pub struct ExplorationNodeDataInternal {
    pub need_exploration_point: u32,
    pub is_base_town: bool,
    pub link_list: Vec<usize>,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
#[allow(unused)]
pub struct NodeData {
    pub waypoint_key: usize,
    pub need_exploration_point: u32,
    pub is_base_town: bool,
    pub in_neighbors: Vec<usize>,
    pub out_neighbors: Vec<usize>,
    pub extra: HashMap<String, serde_json::Value>,
}

pub struct ExplorationData {
    all_pairs: Vec<(usize, usize)>,
    base_town_indices: IntSet<usize>,
    pub num_nodes: usize,
    pub max_weight: u32,
    pub ref_graph: StableDiGraph<usize, usize>,
    pub index_to_waypoint: Vec<usize>,
    pub index_to_weight: Vec<u32>,
    pub waypoint_to_index: IntMap<usize, usize>,
    pub waypoint_to_weight: IntMap<usize, u32>,
    pub index_to_in_neighbors: Vec<SmallVec<[usize; 4]>>,
    pub index_to_in_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>>,
    pub index_to_out_neighbors: Vec<SmallVec<[usize; 4]>>,
    pub index_to_out_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>>,
    pub index_to_in_neighbors_pos: Vec<IntMap<usize, usize>>,
    pub input_graph: InputGraph,
    pub fast_graph: FastGraph,
    pub calc: PathCalculator, // doesn't implement Clone or Debug so we have to make our own.
    pub leaf_index_to_parent_index: IntMap<usize, usize>,
    pub reduced_ref_graph: StableDiGraph<usize, usize>,
    pub reduced_index_to_in_neighbors: Vec<SmallVec<[usize; 4]>>,
    pub reduced_index_to_in_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>>,
    pub reduced_index_to_out_neighbors: Vec<SmallVec<[usize; 4]>>,
    pub reduced_index_to_out_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>>,
    pub reduced_index_to_in_neighbors_pos: Vec<IntMap<usize, usize>>,
    pub reduced_input_graph: InputGraph,
    pub reduced_fast_graph: FastGraph,
    pub reduced_calc: PathCalculator, // doesn't implement Clone or Debug so we have to make our own.
}

impl fmt::Debug for ExplorationData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExplorationData")
            .field("all_pairs", &self.all_pairs)
            .field("base_town_indices", &self.base_town_indices)
            .field("num_nodes", &self.num_nodes)
            .field("max_weight", &self.max_weight)
            .field("ref_graph", &self.ref_graph)
            .field("index_to_waypoint", &self.index_to_waypoint)
            .field("index_to_weight", &self.index_to_weight)
            .field("waypoint_to_index", &self.waypoint_to_index)
            .field("waypoint_to_weight", &self.waypoint_to_weight)
            .field("index_to_in_neighbors", &self.index_to_in_neighbors)
            .field(
                "index_to_in_neighbors_weighted",
                &self.index_to_in_neighbors_weighted,
            )
            .field("index_to_out_neighbors", &self.index_to_out_neighbors)
            .field(
                "index_to_out_neighbors_weighted",
                &self.index_to_out_neighbors_weighted,
            )
            .field("index_to_in_neighbors_pos", &self.index_to_in_neighbors_pos)
            .field("input_graph", &self.input_graph)
            .field("fast_graph", &self.fast_graph)
            .field("calc", &"PathCalculator(...)")
            .field(
                "leaf_index_to_parent_index",
                &self.leaf_index_to_parent_index,
            )
            .field("reduced_ref_graph", &self.reduced_ref_graph)
            .field(
                "reduced_index_to_in_neighbors",
                &self.reduced_index_to_in_neighbors,
            )
            .field(
                "reduced_index_to_in_neighbors_weighted",
                &self.reduced_index_to_in_neighbors_weighted,
            )
            .field(
                "reduced_index_to_out_neighbors",
                &self.reduced_index_to_out_neighbors,
            )
            .field(
                "reduced_index_to_out_neighbors_weighted",
                &self.reduced_index_to_out_neighbors_weighted,
            )
            .field(
                "reduced_index_to_in_neighbors_pos",
                &self.reduced_index_to_in_neighbors_pos,
            )
            .field("reduced_input_graph", &self.reduced_input_graph)
            .field("reduced_fast_graph", &self.reduced_fast_graph)
            .field("reduced_calc", &"PathCalculator(...)")
            .finish()
    }
}

impl Clone for ExplorationData {
    /// Fastpaths doesn't implement Clone so we have to make our own.
    fn clone(&self) -> Self {
        let cloned_fast_graph = self.fast_graph.clone();
        let reduced_cloned_fast_graph = self.reduced_fast_graph.clone();
        Self {
            all_pairs: self.all_pairs.clone(),
            base_town_indices: self.base_town_indices.clone(),
            num_nodes: self.num_nodes,
            max_weight: self.max_weight,
            ref_graph: self.ref_graph.clone(),
            index_to_waypoint: self.index_to_waypoint.clone(),
            index_to_weight: self.index_to_weight.clone(),
            waypoint_to_index: self.waypoint_to_index.clone(),
            waypoint_to_weight: self.waypoint_to_weight.clone(),
            index_to_in_neighbors: self.index_to_in_neighbors.clone(),
            index_to_in_neighbors_weighted: self.index_to_in_neighbors_weighted.clone(),
            index_to_out_neighbors: self.index_to_out_neighbors.clone(),
            index_to_out_neighbors_weighted: self.index_to_out_neighbors_weighted.clone(),
            index_to_in_neighbors_pos: self.index_to_in_neighbors_pos.clone(),
            input_graph: self.input_graph.clone(),
            fast_graph: cloned_fast_graph.clone(),
            calc: fast_paths::create_calculator(&cloned_fast_graph),
            leaf_index_to_parent_index: self.leaf_index_to_parent_index.clone(),
            reduced_ref_graph: self.reduced_ref_graph.clone(),
            reduced_index_to_in_neighbors: self.reduced_index_to_in_neighbors.clone(),
            reduced_index_to_in_neighbors_weighted: self
                .reduced_index_to_in_neighbors_weighted
                .clone(),
            reduced_index_to_out_neighbors: self.reduced_index_to_out_neighbors.clone(),
            reduced_index_to_out_neighbors_weighted: self
                .reduced_index_to_out_neighbors_weighted
                .clone(),
            reduced_index_to_in_neighbors_pos: self.reduced_index_to_in_neighbors_pos.clone(),
            reduced_input_graph: self.reduced_input_graph.clone(),
            reduced_fast_graph: reduced_cloned_fast_graph.clone(),
            reduced_calc: fast_paths::create_calculator(&cloned_fast_graph),
        }
    }
}

impl ExplorationData {
    pub fn new(nodes_map: BTreeMap<usize, ExplorationNodeData>) -> Self {
        // ---- Prepare Exploration Data ----
        let mut nodes: IntMap<usize, NodeData> = nodes_map
            .into_iter()
            .map(|(waypoint_key, data)| {
                let node_data = NodeData {
                    waypoint_key,
                    need_exploration_point: data.need_exploration_point as u32,
                    is_base_town: data.is_base_town,
                    in_neighbors: data.link_list.clone(),
                    out_neighbors: data.link_list,
                    extra: data.extra,
                };
                (waypoint_key, node_data)
            })
            .collect();

        // SAFETY: Recursively remove any invalid link_list entries and any nodes without a link_list
        //         from both the in and out neighbors.
        let mut valid_node_ids = IntSet::from_iter(nodes.keys().copied());
        loop {
            for n in nodes.values_mut() {
                n.in_neighbors.retain(|l| valid_node_ids.contains(l));
                n.out_neighbors.retain(|l| valid_node_ids.contains(l));
            }

            let mut removables = Vec::new();
            let tmp_nodes = nodes.clone();
            for (wp_key, n) in tmp_nodes {
                if n.in_neighbors.is_empty() || !valid_node_ids.contains(&wp_key) {
                    removables.push(wp_key);
                }
            }
            removables.iter().for_each(|wp_key| {
                nodes.remove(wp_key);
                valid_node_ids.remove(wp_key);
            });
            if removables.is_empty() {
                break;
            } else {
                valid_node_ids = IntSet::from_iter(nodes.keys().copied());
            }
        }
        let mut nodes: Vec<NodeData> = nodes.into_values().collect();

        // SAFETY: Ensure all remaining links are bi-directional and out == in
        let nodes_by_key: HashMap<_, _> = nodes.iter().map(|n| (n.waypoint_key, n)).collect();
        for n in &nodes {
            for &neighbor in &n.in_neighbors {
                debug_assert!(
                    nodes_by_key[&neighbor]
                        .out_neighbors
                        .contains(&n.waypoint_key),
                    "Asymmetric link: {} -> {}",
                    neighbor,
                    n.waypoint_key
                );
            }
            for &neighbor in &n.out_neighbors {
                debug_assert!(
                    nodes_by_key[&neighbor]
                        .in_neighbors
                        .contains(&n.waypoint_key),
                    "Asymmetric link: {} -> {}",
                    neighbor,
                    n.waypoint_key
                );
            }
        }

        // Inject SUPER_ROOT node with in_neighbors entries for all base towns.
        debug_assert!(
            !nodes.iter().any(|n| n.waypoint_key == 99999),
            "Duplicate waypoint key found (reserved for SUPER_ROOT): 99999"
        );

        let base_town_nodes: Vec<_> = nodes.iter().filter(|n| n.is_base_town).cloned().collect();
        nodes.push(NodeData {
            waypoint_key: 99999,
            need_exploration_point: 0,
            is_base_town: true,
            in_neighbors: base_town_nodes.iter().map(|n| n.waypoint_key).collect(),
            out_neighbors: vec![],
            extra: HashMap::new(),
        });
        nodes.sort_by_key(|n| n.waypoint_key);
        for node in nodes.iter_mut() {
            if node.is_base_town {
                node.out_neighbors.push(99999);
            }
        }

        // SAFETY: nodes should not me mutated after this point!
        let nodes = nodes;

        // ---- Build Reference Graph ----
        let num_nodes = nodes.len();
        let mut ref_graph = StableDiGraph::<usize, usize>::with_capacity(num_nodes, num_nodes * 2);

        let mut max_weight = 0;
        let mut index_to_waypoint = Vec::new();
        let mut index_to_weight = Vec::new();
        let mut waypoint_to_index =
            IntMap::with_capacity_and_hasher(num_nodes, BuildNoHashHasher::default());
        let mut waypoint_to_weight =
            IntMap::with_capacity_and_hasher(num_nodes, BuildNoHashHasher::default());

        for (n_idx, node_data) in nodes.iter().enumerate() {
            let waypoint_key = node_data.waypoint_key;
            let weight = node_data.need_exploration_point;

            // SAFETY: There must be a 1:1 mapping between node index and ref_graph node index!
            let idx = ref_graph.add_node(weight as usize).index();
            assert_eq!(idx, n_idx, "node index mismatch");

            index_to_waypoint.insert(idx, waypoint_key);
            index_to_weight.insert(idx, weight);
            waypoint_to_index.insert(waypoint_key, idx);
            waypoint_to_weight.insert(waypoint_key, weight);
            max_weight = max_weight.max(weight);
        }

        // SAFETY: Dial's algorithm is built with the assumption of max weight of 3
        debug_assert!(max_weight <= 3);

        // Insert edges (tail -> head => head weight)
        // NOTE: All exploration links are bidirectional and populated into both in_neighbors and
        // out_neighbors, except for the SUPER_ROOT node with in_neighbors only
        for (head_idx, &head_weight) in index_to_weight.iter().enumerate() {
            for &tail_wp in &nodes[head_idx].in_neighbors {
                let tail_idx = waypoint_to_index[&tail_wp];
                ref_graph.add_edge(
                    NodeIndex::new(tail_idx),
                    NodeIndex::new(head_idx),
                    head_weight as usize,
                );
            }
        }

        // SAFETY: The reference graph is built with the assumption that all nodes are reachable
        //         from both the in and out neighbors except the super root.
        let mut cc = tarjan_scc(&ref_graph);
        cc.sort_by_key(|c| c.len());
        debug_assert!(cc.len() == 2);
        debug_assert!(
            cc[1].len() == num_nodes - 1,
            "{} != {}",
            cc[0].len(),
            num_nodes - 1
        );
        debug_assert!(cc[0].len() == 1);
        debug_assert!(cc[0][0].index() == waypoint_to_index[&99999]); // super root

        // SAFETY: reference_graph should not be modified after this point!
        let ref_graph = ref_graph;
        debug_assert!(ref_graph.node_count() == num_nodes);
        debug_assert!(ref_graph.edge_count() * (max_weight as usize) < u32::MAX as usize);

        // ---- Build Neighbor Mappings ----
        let mut index_to_in_neighbors: Vec<SmallVec<[usize; 4]>> = Vec::with_capacity(num_nodes);
        let mut index_to_out_neighbors: Vec<SmallVec<[usize; 4]>> = Vec::with_capacity(num_nodes);
        let mut index_to_in_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>> =
            Vec::with_capacity(num_nodes);
        let mut index_to_out_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>> =
            Vec::with_capacity(num_nodes);

        // SAFETY: There must be a 1:1 mapping between node index and ref_graph node index
        //         to ensure vec pos matches node index.
        //         petgraph's node_indices() is an enumeration of all node indices
        //         (self.raw_nodes().iter().enumerate())
        for node_idx in ref_graph.node_indices() {
            let mut in_neighbors: SmallVec<[usize; 4]> = ref_graph
                .neighbors_directed(node_idx, Direction::Incoming)
                .map(|n| n.index())
                .collect();
            let mut out_neighbors: SmallVec<[usize; 4]> = ref_graph
                .neighbors_directed(node_idx, Direction::Outgoing)
                .map(|n| n.index())
                .collect();

            // Sorted in ascending order by node weight
            in_neighbors.sort_by_key(|&i| index_to_weight[i]);
            out_neighbors.sort_by_key(|&i| index_to_weight[i]);

            // [(neighbor_index, neighbor_weight), ...]
            let current_node_weight = index_to_weight[node_idx.index()];
            let in_neighbors_weighted: SmallVec<[(usize, u32); 4]> = in_neighbors
                .iter()
                .map(|&i| (i, current_node_weight))
                .collect();
            let out_neighbors_weighted: SmallVec<[(usize, u32); 4]> = out_neighbors
                .iter()
                .map(|&i| (i, index_to_weight[i]))
                .collect();

            index_to_in_neighbors.push(in_neighbors);
            index_to_out_neighbors.push(out_neighbors);
            index_to_in_neighbors_weighted.push(in_neighbors_weighted);
            index_to_out_neighbors_weighted.push(out_neighbors_weighted);
        }

        // NOTE: When doing bidir Dial's for paths the forward and reverse paths are built separately.
        // - forward is built using the out_neighbors / out_neighbors_weighted.
        // - reverse is built using the in_neighbors / in_neighbors_weighted.
        //
        // When zero-weight augmentation is done, all inbound edge weights to v must be set to 0 to
        // encourage path sharing.
        // - From v's perspective set the inbound edge to v to weight 0 for each in_neighbor of v.
        //   Since you are at node v the weight of any incoming edge is the same as the weight of v
        //   itself in the index_to_weight collection.
        //   Therefore we need to update the outbound edges on all neighbors in the in_neighbors
        //   of v collection.
        // - From the neighbor's perspective set the outbound edge to v to weight 0.
        //
        // The index_to_in_neighbors_out_neighbor_pos_map is used to find the position of v in the
        // out_neighbors of v's neighbors.
        // That position is the entry position to index in out_neighbors_weighted to update.
        //
        // ie:
        //   for v in path:
        //     pos_map = index_to_in_neighbors_out_neighbor_pos_map[v]
        //     for n in index_to_in_neighbors[v]:
        //       v_pos = pos_map[&n]
        //       out_neighbors_weighted[n][v_pos] = 0
        //
        let mut index_to_in_neighbors_out_neighbor_pos_map: Vec<IntMap<usize, usize>> =
            Vec::with_capacity(num_nodes);

        for head in ref_graph.node_indices() {
            let mut pos_map = IntMap::<usize, usize>::default();
            for &tail in &index_to_in_neighbors[head.index()] {
                if let Some(pos) = index_to_out_neighbors[tail]
                    .iter()
                    .position(|&idx| idx == head.index())
                {
                    pos_map.insert(tail, pos);
                } else {
                    panic!("No pos found in head: {} for tail: {}", head.index(), tail);
                }
            }
            index_to_in_neighbors_out_neighbor_pos_map.push(pos_map);
        }

        // Populate all_pairs for random pair generation
        // reverse pair order allowed to stress test routers
        // no self loops and no super-root as source
        let super_root_idx = waypoint_to_index[&99999];
        let mut all_pairs = Vec::with_capacity((num_nodes - 1).pow(2));
        for i in 0..nodes.len() {
            if i == super_root_idx {
                continue;
            }
            for j in 0..nodes.len() {
                if i != j {
                    all_pairs.push((i, j));
                }
            }
        }

        // ---- Fast Graph ----
        // NOTE: This is only used for ref_graph shortest path distances.
        // NOTE: fast_paths does not allow zero weight edges.
        // Scaling by a weight factor >= max(w(i)) * ∑w(i) allows a nominal weight of 1 to be
        // the reduced 'zero weight' edge cost ensuring the results are correct.
        // let weight_factor = ref_graph.edge_weights().sum::<usize>() * max_weight as usize;
        let weight_factor = 10_000;
        debug_assert!(weight_factor > 0);
        debug_assert!(weight_factor * (num_nodes - 1) <= u32::MAX as usize);

        let mut input_graph = InputGraph::new();
        for arc_idx in ref_graph.edge_indices() {
            if let Some((u_idx, v_idx)) = ref_graph.edge_endpoints(arc_idx) {
                let mut w = ref_graph.edge_weight(arc_idx).unwrap() * weight_factor;
                if w == 0 {
                    w = 1;
                }
                input_graph.add_edge(u_idx.index(), v_idx.index(), w);
            }
        }
        input_graph.freeze();

        // These fastpaths params provide the fastest query time.
        let params = Params::new(0.1, 200, 200, 200); // default: 0.1, 500, 100, 500
        let fast_graph = fast_paths::prepare_with_params(&input_graph, &params);
        let calc = fast_paths::create_calculator(&fast_graph);

        // ---- Reduced Reference Graph ----

        // We essentially redo everything we did above but on the reduced reference graph...
        // NOTE: All leaf nodes of ref_graph are removed and a mapping is created from the leaf
        //       node to its parent node.
        // NOTE: Since ref_graph was previously guaranteed to have all edges bi-directed except
        //       the super root, the reduced ref_graph is also guaranteed to have all edges
        //       bi-directed except the super root.
        let mut leaf_index_to_parent_index =
            IntMap::with_capacity_and_hasher(num_nodes - 1, BuildNoHashHasher::default());
        for node in ref_graph.node_indices() {
            let nbr_count = ref_graph
                .neighbors_directed(node, Direction::Incoming)
                .count();
            if nbr_count == 1 {
                let parent = ref_graph
                    .neighbors_directed(node, Direction::Incoming)
                    .next()
                    .unwrap();
                leaf_index_to_parent_index.insert(node.index(), parent.index());
            }
        }
        let removables = IntSet::from_iter(leaf_index_to_parent_index.keys().copied());
        let mut reduced_ref_graph = ref_graph.clone();
        reduced_ref_graph.retain_nodes(|_g, n| !removables.contains(&n.index()));

        // SAFETY: The reduced reference graph is built with the same assumption as the ref_graph
        //         that all nodes are reachable from the in and out neighbors except the super root.
        let mut cc = tarjan_scc(&reduced_ref_graph);
        cc.sort_by_key(|c| c.len());
        debug_assert!(cc.len() == 2, "cc.len(): {}", cc.len());
        debug_assert!(
            cc[1].len() == num_nodes - removables.len() - 1,
            "cc[1].len(): {}",
            cc[1].len()
        );
        debug_assert!(cc[0].len() == 1, "cc[0].len(): {}", cc[0].len());
        debug_assert!(
            cc[0][0].index() == waypoint_to_index[&99999],
            "super root not isolated"
        );

        // SAFETY: reference_graph should not be modified after this point!
        let reduced_ref_graph = reduced_ref_graph;

        // ---- Build Reduced Neighbor Mappings ----
        // SAFETY: **We need to ensure a 1:1 mapping between ref_graph and reduced_ref_graph
        //         indices within the lookup vectors!**
        //
        // Stable Graphs guarantee that the removal of nodes does not change indices and our vec
        // indices must do the same thing.
        let mut reduced_index_to_in_neighbors: Vec<SmallVec<[usize; 4]>> =
            Vec::with_capacity(num_nodes);
        let mut reduced_index_to_out_neighbors: Vec<SmallVec<[usize; 4]>> =
            Vec::with_capacity(num_nodes);
        let mut reduced_index_to_in_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>> =
            Vec::with_capacity(num_nodes);
        let mut reduced_index_to_out_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>> =
            Vec::with_capacity(num_nodes);

        for node_idx in ref_graph.node_indices() {
            let is_removed_leaf = leaf_index_to_parent_index.contains_key(&node_idx.index());
            let mut in_neighbors: SmallVec<[usize; 4]> = if is_removed_leaf {
                SmallVec::new()
            } else {
                reduced_ref_graph
                    .neighbors_directed(node_idx, Direction::Incoming)
                    .map(|n| n.index())
                    .collect()
            };
            let mut out_neighbors: SmallVec<[usize; 4]> = if is_removed_leaf {
                SmallVec::new()
            } else {
                reduced_ref_graph
                    .neighbors_directed(node_idx, Direction::Outgoing)
                    .map(|n| n.index())
                    .collect()
            };

            // Sorted in ascending order by node weight
            in_neighbors.sort_by_key(|&i| index_to_weight[i]);
            out_neighbors.sort_by_key(|&i| index_to_weight[i]);

            // [(neighbor_index, neighbor_weight), ...]
            let current_node_weight = index_to_weight[node_idx.index()];
            let in_neighbors_weighted: SmallVec<[(usize, u32); 4]> = in_neighbors
                .iter()
                .map(|&i| (i, current_node_weight))
                .collect();
            let out_neighbors_weighted: SmallVec<[(usize, u32); 4]> = out_neighbors
                .iter()
                .map(|&i| (i, index_to_weight[i]))
                .collect();

            reduced_index_to_in_neighbors.push(in_neighbors);
            reduced_index_to_out_neighbors.push(out_neighbors);
            reduced_index_to_in_neighbors_weighted.push(in_neighbors_weighted);
            reduced_index_to_out_neighbors_weighted.push(out_neighbors_weighted);
        }

        // NOTE: When doing bidir Dial's for paths the forward and reverse paths are built separately.
        // - forward is built using the out_neighbors / out_neighbors_weighted.
        // - reverse is built using the in_neighbors / in_neighbors_weighted.
        //
        // Refer to the comments in the ref_graph section for more details.
        let mut reduced_index_to_in_neighbors_out_neighbor_pos_map: Vec<IntMap<usize, usize>> =
            Vec::with_capacity(num_nodes);

        for head in ref_graph.node_indices() {
            let is_removed_leaf = leaf_index_to_parent_index.contains_key(&head.index());
            let pos_map: IntMap<usize, usize> = if is_removed_leaf {
                IntMap::default()
            } else {
                let mut pos_map = IntMap::<usize, usize>::default();
                for &tail in &reduced_index_to_in_neighbors[head.index()] {
                    if let Some(pos) = reduced_index_to_out_neighbors[tail]
                        .iter()
                        .position(|&idx| idx == head.index())
                    {
                        pos_map.insert(tail, pos);
                    } else {
                        panic!("No pos found in head: {} for tail: {}", head.index(), tail);
                    }
                }
                pos_map
            };
            reduced_index_to_in_neighbors_out_neighbor_pos_map.push(pos_map);
        }

        // NOTE: Unlike the main ref_graph building section we skip the population of the
        // all_pairs for random pair generation since that is used by testing/benching to
        // simulate the external inputs and the ref_graph is used for that.

        // ---- Fast Graph ----
        // NOTE: This is only used for reduced_ref_graph shortest path distances.
        // See notes in the ref_graph section for more details.

        // We re-use the weight_factor from the ref_graph to scale the weights.
        let mut reduced_input_graph = InputGraph::new();
        for arc_idx in reduced_ref_graph.edge_indices() {
            if let Some((u_idx, v_idx)) = reduced_ref_graph.edge_endpoints(arc_idx) {
                let mut w = reduced_ref_graph.edge_weight(arc_idx).unwrap() * weight_factor;
                if w == 0 {
                    w = 1;
                }
                reduced_input_graph.add_edge(u_idx.index(), v_idx.index(), w);
            }
        }
        reduced_input_graph.freeze();

        // We re-use the params from the ref_graph to prepare the fast_graph.
        let reduced_fast_graph = fast_paths::prepare_with_params(&reduced_input_graph, &params);
        let reduced_calc = fast_paths::create_calculator(&reduced_fast_graph);

        // ---- Base Towns ----
        // Primal Dual Basetowns - Exclude super root
        let base_town_indices: IntSet<usize> = IntSet::from_iter(
            base_town_nodes
                .iter()
                .map(|base_town_node| waypoint_to_index[&base_town_node.waypoint_key])
                .filter(|&idx| idx != super_root_idx),
        );

        // NOTE: Each of the following is safely shared between the ref_graph and the reduced_ref_graph
        // as long as the usage is for its intended mapping and not for munging of the graph membership.
        // - index_to_waypoint,
        // - index_to_weight,
        // - waypoint_to_index,
        // - waypoint_to_weight,

        assert!(index_to_in_neighbors.len() == num_nodes);
        assert!(index_to_out_neighbors.len() == num_nodes);
        assert!(index_to_in_neighbors_weighted.len() == num_nodes);
        assert!(index_to_out_neighbors_weighted.len() == num_nodes);

        // ---- Return ----
        Self {
            all_pairs,
            base_town_indices,
            max_weight,
            num_nodes,
            ref_graph,
            index_to_waypoint,
            index_to_weight,
            waypoint_to_index,
            waypoint_to_weight,
            index_to_in_neighbors,
            index_to_out_neighbors,
            index_to_in_neighbors_weighted,
            index_to_out_neighbors_weighted,
            index_to_in_neighbors_pos: index_to_in_neighbors_out_neighbor_pos_map,
            input_graph,
            fast_graph,
            calc,
            leaf_index_to_parent_index,
            reduced_ref_graph,
            reduced_index_to_in_neighbors,
            reduced_index_to_out_neighbors,
            reduced_index_to_in_neighbors_weighted,
            reduced_index_to_out_neighbors_weighted,
            reduced_index_to_in_neighbors_pos: reduced_index_to_in_neighbors_out_neighbor_pos_map,
            reduced_input_graph,
            reduced_fast_graph,
            reduced_calc,
        }
    }

    /// Returns a subgraph of the reference graph with only the nodes in `nodes`.
    ///
    /// NOTE: Edge indices are invalidated as they would be following the removal
    ///       of each edge with an endpoint in a removed node.
    pub fn subgraph_stable(
        &self,
        nodes: &[usize],
        use_reduced: Option<bool>,
    ) -> StableDiGraph<usize, usize> {
        let nodes = IntSet::from_iter(nodes.iter().copied());
        let mut subgraph = if use_reduced.unwrap_or(false) {
            self.reduced_ref_graph.clone()
        } else {
            self.ref_graph.clone()
        };
        subgraph.retain_nodes(|_g, n| nodes.contains(&n.index()));
        subgraph
    }

    /// Returns whether all `pairs` have paths.
    ///
    /// NOTE: All valid paths within reduced_ref_graph are used are valid in ref_graph.
    pub fn all_have_paths(&self, nodes: &[usize], pairs: &[(usize, usize)]) -> bool {
        // NOTE: It is assumed that nodes are intermediate nodes and pairs are endpoints.
        let mut nodes = nodes.to_owned();
        for &(s, t) in pairs {
            nodes.push(s);
            nodes.push(t);
        }

        let subgraph = self.subgraph_stable(&nodes, None);

        let mut final_result = true;
        for (s, t) in pairs {
            let start_idx = NodeIndex::new(*s);
            let goal_idx = NodeIndex::new(*t);
            let result = astar(
                &subgraph,
                start_idx,
                |finish| finish == goal_idx,
                |e| *e.weight(),
                |_| 0, // zero heuristic → pure Dijkstra
            );
            if result.is_none() {
                println!(
                    "missing path from {} to {}",
                    self.index_to_waypoint[*s], self.index_to_waypoint[*t]
                );
                final_result = false;
            }
        }
        final_result
    }

    /// Dijkstra path finder for a single waypoint pair
    ///
    /// # Returns
    /// Option<(cost, path)>
    pub fn dijkstra_path(
        &self,
        start: usize,
        goal: usize,
        use_reduced: Option<bool>,
    ) -> Option<(usize, Vec<usize>)> {
        let start_idx = petgraph::graph::NodeIndex::new(start);
        let goal_idx = petgraph::graph::NodeIndex::new(goal);

        let graph = if use_reduced.unwrap_or(false) {
            &self.reduced_ref_graph
        } else {
            &self.ref_graph
        };
        astar(
            graph,
            start_idx,
            |finish| finish == goal_idx,
            |e| *e.weight(),
            |_| 0, // zero heuristic → pure Dijkstra
        )
        .map(|(cost, nodes)| (cost, nodes.into_iter().map(|n| n.index()).collect()))
    }

    /// Transforms original ref_graph index pairs to reduced_ref_graph index pairs
    /// by replacing all terminals in all pairs that are removed leaf nodes with their parent.
    ///
    /// NOTE: Retains original order.
    ///
    /// If transformed source == target then the pair is omitted.
    pub fn transform_pairs_to_reduced_pairs(
        &self,
        pairs: &[(usize, usize)],
    ) -> (Vec<(usize, usize)>, Vec<usize>) {
        // We need to track any single hop pairs that are removed leaf nodes
        // for addition to the ordered_removables.
        let mut sibling_parents = Vec::new();
        let mut transformed_pairs: Vec<(usize, usize)> = Vec::new();
        let leaf_to_parent = &self.leaf_index_to_parent_index;
        for (orig_s, orig_t) in pairs {
            let trans_s = leaf_to_parent.get(orig_s).copied().unwrap_or(*orig_s);
            let trans_t = leaf_to_parent.get(orig_t).copied().unwrap_or(*orig_t);
            if trans_s != trans_t {
                transformed_pairs.push((trans_s, trans_t));
            } else if orig_s != orig_t {
                sibling_parents.push(trans_s);
            }
        }
        (transformed_pairs, sibling_parents)
    }

    /// Returns the distance between two nodes in the reference graph
    ///
    /// SAFETY: **The path length of ref_graph and reduced_ref_graph will not match
    ///         for any pair containing a leaf node! All other pairs should match.**
    ///
    /// NOTE: This is a fast path finder using fastpaths and the fast graph which
    ///       uses the same nodes as the reference graph but the edge weights are
    ///       augmented to be * 10_000 the original weight, except for zero cost
    ///       edges which are set to a nominal value of 1.
    ///
    /// NOTE: mut reference required because fastgraph's calc_path is a mutable method.
    pub fn query_fp_distance(
        &mut self,
        start_idx: usize,
        goal_idx: usize,
        use_reduced: Option<bool>,
    ) -> u32 {
        let path = if use_reduced.unwrap_or(false) {
            self.reduced_calc
                .calc_path(&self.reduced_fast_graph, start_idx, goal_idx)
        } else {
            self.calc.calc_path(&self.fast_graph, start_idx, goal_idx)
        };

        if let Some(path) = path {
            path.get_weight() as u32
        } else {
            println!(
                "Check reference graph and fast graph for mismatch, ref_graph should be fully connected."
            );
            panic!(
                "fastpath failed to find path from {} to {}",
                start_idx, goal_idx
            );
        }
    }

    /// Returns the distance between two nodes in the reference graph
    ///
    /// Decodes FastPaths integer distance as:
    /// fp_dist = true_weight * weight_factor + hop_count
    /// => true_weight = fp_dist / weight_factor;
    ///
    /// SAFETY: **The path length of ref_graph and reduced_ref_graph will not match
    ///         for any pair containing a leaf node! All other pairs should match.**
    ///
    /// NOTE: This is a fast path finder using fastpaths and the fast graph which
    ///       uses the same nodes as the reference graph but the edge weights are
    ///       augmented to be * 10_000 the original weight, except for zero cost
    ///       edges which are set to a nominal value of 1.
    ///
    /// NOTE: mut reference required because fastgraph's calc_path is a mutable method.
    pub fn query_fp_ref_graph_distance(
        &mut self,
        start_idx: usize,
        goal_idx: usize,
        use_reduced: Option<bool>,
    ) -> u32 {
        let fp_dist = self.query_fp_distance(start_idx, goal_idx, use_reduced);
        let weight_factor = 10_000u32; // or self.weight_factor if stored
        fp_dist / weight_factor
    }

    /// Generate a cutoff heap (priority queue) for the given pairs.
    ///
    /// Uses fastpath to find the distance between each pair.
    ///
    /// Returns a BinaryHeap of Reverse<(u32, usize)>
    pub fn generate_cutoff_heap(
        &mut self,
        pairs: &[(usize, usize)],
        use_reduced: Option<bool>,
    ) -> BinaryHeap<Reverse<(u32, usize)>> {
        let mut cutoffs_heap = BinaryHeap::new();

        for (i, &(s, t)) in pairs.iter().enumerate() {
            let weight = self.query_fp_ref_graph_distance(s, t, use_reduced);
            cutoffs_heap.push(Reverse((weight, i)));
        }

        cutoffs_heap
    }

    /// NOTE: This is safe to use on both ref_graph and reduced_ref_graph.
    pub fn index_to_waypoint_many(&self, path: &[usize]) -> Vec<usize> {
        path.iter()
            .map(|&idx| self.index_to_waypoint[idx])
            .collect()
    }

    /// NOTE: This is safe to use on both ref_graph and reduced_ref_graph.
    pub fn index_to_weight_many(&self, indices: &[usize]) -> Vec<u32> {
        indices
            .iter()
            .map(|&idx| self.index_to_weight[idx])
            .collect()
    }

    /// NOTE: This is safe to use on both ref_graph and reduced_ref_graph.
    pub fn waypoint_to_weight_many(&self, waypoints: &[usize]) -> Vec<u32> {
        waypoints
            .iter()
            .map(|wp| self.waypoint_to_weight[wp])
            .collect()
    }

    pub fn sort_by_ref_graph_distance(
        &mut self,
        wp_pairs: &mut [(usize, usize)],
        use_reduced: Option<bool>,
    ) {
        wp_pairs.sort_by_key(|(s, t)| self.query_fp_distance(*s, *t, use_reduced));
    }
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
    exploration_data: ExplorationData,

    settled_nodes: IntSet<usize>, // nodes
    paid_weight: Vec<u32>,        // weights
    connected_pairs: RapidHashSet<(usize, usize)>,
    terminal_to_root: Vec<(usize, usize)>,
}

impl PDBatchGenerator {
    pub fn new(exploration_data: &ExplorationData) -> Self {
        let exploration_data = exploration_data.clone();
        let connected_pairs = RapidHashSet::default();
        let terminal_to_root = Vec::new();
        let pd_x = IntSet::default();
        let pd_y = vec![0; exploration_data.ref_graph.node_count()];

        Self {
            exploration_data,
            settled_nodes: pd_x,
            paid_weight: pd_y,
            connected_pairs,
            terminal_to_root,
        }
    }

    /// Induce subgraph from reference graph using node indices.
    fn ref_subgraph_stable(&self, indices: &IntSet<usize>) -> StableDiGraph<(), usize> {
        self.exploration_data.ref_graph.filter_map(
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
            self.exploration_data.ref_graph.node_count(),
            BuildNoHashHasher::default(),
        );
        frontier.extend(
            settlement
                .iter()
                .flat_map(|&v| &self.exploration_data.index_to_out_neighbors[v])
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
        let super_root_index = self.exploration_data.waypoint_to_index[&99999];

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
                    cc.intersection(&self.exploration_data.base_town_indices)
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
        let num_nodes = self.exploration_data.num_nodes;
        let super_root_index = self.exploration_data.waypoint_to_index[&99999];
        let index_to_weight = self.exploration_data.index_to_weight.clone();

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

    /// Generates batches of original pairs.
    ///
    /// Requires that:
    /// - super_root is not a source node in any pair.
    ///
    /// When pairs.len() ≤ batching_threshold, returns a single batch of all indices.
    pub fn generate_pair_batches(
        &mut self,
        pairs: &[(usize, usize)],
        batching_threshold: usize,
    ) -> Vec<SmallVec<[(usize, usize); 4]>> {
        if pairs.len() <= batching_threshold {
            vec![
                pairs
                    .iter()
                    .cloned()
                    .collect::<SmallVec<[(usize, usize); 4]>>(),
            ]
        } else {
            self.primal_dual_batch_generator(pairs)
        }
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

    /// Generates batches of original pair indices (0..pairs.len()).
    ///
    /// The batched pairs are those that went tight in the same iteration
    /// of the Node Weighted Primal-Dual approximation.
    ///
    /// Requires that:
    /// - `pair_index_to_pair_key` maps index → pair.
    ///
    /// When pairs.len() ≤ batching_threshold, returns a single batch of all indices.
    pub fn generate_pair_batches_with_indices<'a>(
        &self,
        pairs: &'a [(usize, usize)],
        batch_size: usize,
    ) -> impl Iterator<Item = Vec<((usize, usize), usize)>> + 'a {
        (0..pairs.len()).step_by(batch_size).map(move |start| {
            let end = (start + batch_size).min(pairs.len());
            (start..end)
                .map(|i| (pairs[i], i)) // pair + its stable index in `pairs`
                .collect()
        })
    }
}

pub fn canonicalize_pairs(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    pairs
        .iter()
        .map(|&(s, t)| if s <= t { (s, t) } else { (t, s) })
        .collect()
}
