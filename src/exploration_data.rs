// exploration_data.rs

/// Base reference graph details:
/// * Node weighted planar graph node weights in range 0..=3
/// * 947 nodes and 2149 edges
/// * 472 leaf nodes; 386 degree 2 to 4 nodes; 89 >= 5 degree nodes
///
/// Reduced graph has all leaf nodes (and edges) recursively removed.
/// * 411 nodes and 1076 edges
/// * degree count: {2: 239, 3: 126, 4: 39, 5: 4, 6: 2, 32: 1}
///   (the single 32 degree node is the SUPER_ROOT node)
///
use nohash_hasher::{IntMap, IntSet};
use smallvec::SmallVec;

use petgraph::prelude::{StableDiGraph, StableUnGraph};

use crate::node_router::{ExplorationGraphData, SUPER_ROOT};

#[derive(Clone)]
struct NodeData {
    pub waypoint_key: usize,
    pub need_exploration_point: u32,
    pub is_base_town: bool,
    pub in_neighbors: Vec<usize>,
    pub out_neighbors: Vec<usize>,
}

fn exploration_nodes_to_graph_nodes(nodes_map: &ExplorationGraphData) -> Vec<NodeData> {
    // ---- Prepare Exploration Data ----
    let mut nodes: IntMap<usize, NodeData> = nodes_map
        .iter()
        .map(|(&waypoint_key, data)| {
            let node_data = NodeData {
                waypoint_key,
                need_exploration_point: data.need_exploration_point as u32,
                is_base_town: data.is_base_town,
                in_neighbors: data.link_list.clone(),
                out_neighbors: data.link_list.clone(),
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
    let nodes_by_key: IntMap<usize, &NodeData> =
        nodes.iter().map(|n| (n.waypoint_key, n)).collect();
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
    });
    nodes.sort_by_key(|n| n.waypoint_key);
    for node in nodes.iter_mut() {
        if node.is_base_town {
            node.out_neighbors.push(99999);
        }
    }

    nodes
}

/// Exploration Data
///
/// IDTree, DNDTree, and NodeRouter depend on the undirected graph structure.
///
/// SAFETY: DiGraphs contain SUPER_ROOT node with inbound edges only.
///         This must not be leaked!
///
#[derive(Clone, Debug)]
pub struct ExplorationData {
    pub num_nodes: usize,
    pub max_node_weight: u32,
    pub super_root_index: usize,

    pub base_town_indices: IntSet<usize>,
    pub index_to_waypoint: Vec<usize>,
    pub index_to_weight: Vec<u32>,
    pub waypoint_to_index: IntMap<usize, usize>,

    pub ref_ungraph: StableUnGraph<usize, usize>,
    pub index_to_neighbors_ungraph: Vec<SmallVec<[usize; 4]>>,

    pub ref_digraph: StableDiGraph<usize, usize>,
    pub index_to_out_neighbors: Vec<SmallVec<[usize; 4]>>,

    pub reduced_ref_digraph: StableDiGraph<usize, usize>,
    pub leaf_index_to_parent_index: IntMap<usize, usize>,
    pub reduced_index_to_in_neighbors: Vec<SmallVec<[usize; 4]>>,
    pub reduced_index_to_out_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>>,
    pub reduced_index_to_in_neighbors_pos: Vec<IntMap<usize, usize>>,
}

impl ExplorationData {
    pub fn new(nodes_map: &ExplorationGraphData) -> Self {
        use petgraph::Direction;
        use petgraph::algo::tarjan_scc;
        use petgraph::graph::NodeIndex;

        let nodes = exploration_nodes_to_graph_nodes(nodes_map);

        // ---- Build Reference Graph ----
        let num_nodes = nodes.len();

        let mut ref_ungraph =
            StableUnGraph::<usize, usize>::with_capacity(nodes.len(), nodes.len());
        let mut ref_digraph =
            StableDiGraph::<usize, usize>::with_capacity(num_nodes, num_nodes * 2);

        let mut max_node_weight = 0;
        let mut index_to_waypoint = Vec::new();
        let mut index_to_weight = Vec::new();
        let mut waypoint_to_index = IntMap::with_capacity_and_hasher(
            num_nodes,
            nohash_hasher::BuildNoHashHasher::default(),
        );
        let mut waypoint_to_weight = IntMap::with_capacity_and_hasher(
            num_nodes,
            nohash_hasher::BuildNoHashHasher::default(),
        );

        for (n_idx, node_data) in nodes.iter().enumerate() {
            let waypoint_key = node_data.waypoint_key;
            let weight = node_data.need_exploration_point;

            // SAFETY: There must be a 1:1 mapping between node index and ref graph node indices!
            let un_idx = ref_ungraph.add_node(weight as usize).index();
            let idx = ref_digraph.add_node(weight as usize).index();
            assert_eq!(un_idx, idx, "node index mismatch");
            assert_eq!(idx, n_idx, "node index mismatch");

            index_to_waypoint.insert(idx, waypoint_key);
            index_to_weight.insert(idx, weight);
            waypoint_to_index.insert(waypoint_key, idx);
            waypoint_to_weight.insert(waypoint_key, weight);
            max_node_weight = max_node_weight.max(weight);
        }

        // ---- Base Towns ----
        // Primal Dual Basetowns - Exclude super root
        let super_root_index = waypoint_to_index[&SUPER_ROOT];
        let base_town_nodes: Vec<_> = nodes.iter().filter(|n| n.is_base_town).cloned().collect();
        let base_town_indices: IntSet<usize> = IntSet::from_iter(
            base_town_nodes
                .iter()
                .map(|base_town_node| waypoint_to_index[&base_town_node.waypoint_key])
                .filter(|&idx| idx != super_root_index),
        );

        // Insert edges (tail -> head => head weight)
        // NOTE: All exploration links are bidirectional and populated into both in_neighbors and
        // out_neighbors, except for the SUPER_ROOT node with in_neighbors only
        //
        for (head_idx, &head_weight) in index_to_weight.iter().enumerate() {
            for &tail_wp in &nodes[head_idx].in_neighbors {
                let tail_idx = waypoint_to_index[&tail_wp];

                // SAFETY: StableGraph allows the addition of parallel edges
                //         and we do not want those in the undirected graph.
                if tail_idx < head_idx {
                    ref_ungraph.add_edge(NodeIndex::new(tail_idx), NodeIndex::new(head_idx), 0);
                }

                ref_digraph.add_edge(
                    NodeIndex::new(tail_idx),
                    NodeIndex::new(head_idx),
                    head_weight as usize,
                );
            }
        }

        // SAFETY: The SUPER_ROOT must not be in the undirected graph!
        ref_ungraph.remove_node(NodeIndex::new(super_root_index));

        // ---- Directed Graph Validation ----
        // SAFETY: The reference graph is built with the assumption that all nodes are reachable
        //         from both the in and out neighbors except the super root.
        // SAFETY: Dial's algorithm is built with the assumption of max weight of 3
        debug_assert!(ref_digraph.node_count() == num_nodes);
        debug_assert!(ref_digraph.edge_count() * (max_node_weight as usize) < u32::MAX as usize);

        let mut cc = tarjan_scc(&ref_digraph);
        cc.sort_by_key(|c| c.len());
        debug_assert!(cc.len() == 2);
        debug_assert!(
            cc[1].len() == num_nodes - 1,
            "{} != {}",
            cc[0].len(),
            num_nodes - 1
        );
        debug_assert!(cc[0].len() == 1);
        debug_assert!(cc[0][0].index() == super_root_index);

        assert!(
            max_node_weight <= 3,
            "max_weight error! Got {} and expected 3.",
            max_node_weight
        );

        // SAFETY: reference_graph should not be modified after this point!
        let ref_digraph = ref_digraph;

        // ---- Build Neighbor Mappings ----
        // SAFETY: There must be a 1:1 mapping between node index and ref_graph node index
        //         to ensure vec pos matches node index.
        //         petgraph's node_indices() is an enumeration of all node indices
        //         (self.raw_nodes().iter().enumerate())
        let mut index_to_neighbors_ungraph: Vec<SmallVec<[usize; 4]>> =
            Vec::with_capacity(num_nodes);

        for node_idx in ref_ungraph.node_indices() {
            let mut neighbors: SmallVec<[usize; 4]> =
                ref_ungraph.neighbors(node_idx).map(|n| n.index()).collect();

            // Sorts in ascending order
            neighbors.sort_by_key(|&i| index_to_weight[i]);

            index_to_neighbors_ungraph.push(neighbors);
        }

        let mut index_to_out_neighbors: Vec<SmallVec<[usize; 4]>> = Vec::with_capacity(num_nodes);
        let mut index_to_out_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>> =
            Vec::with_capacity(num_nodes);

        for node_idx in ref_digraph.node_indices() {
            let mut out_neighbors: SmallVec<[usize; 4]> = ref_digraph
                .neighbors_directed(node_idx, Direction::Outgoing)
                .map(|n| n.index())
                .collect();

            // Sorts in ascending order
            out_neighbors.sort_by_key(|&i| index_to_weight[i]);

            // [(neighbor_index, neighbor_weight), ...]
            let out_neighbors_weighted: SmallVec<[(usize, u32); 4]> = out_neighbors
                .iter()
                .map(|&i| (i, index_to_weight[i]))
                .collect();

            index_to_out_neighbors.push(out_neighbors);
            index_to_out_neighbors_weighted.push(out_neighbors_weighted);
        }

        // ---- Reduced Reference Graph ----

        // We essentially redo everything we did above but on the reduced reference graph...
        // NOTE: All leaf nodes of ref_graph are removed and a mapping is created from the leaf
        //       node to its parent node.
        // NOTE: Since ref_graph was previously guaranteed to have all edges bi-directed except
        //       the super root, the reduced ref_graph is also guaranteed to have all edges
        //       bi-directed except the super root.
        let mut leaf_index_to_parent_index = IntMap::with_capacity_and_hasher(
            num_nodes - 1,
            nohash_hasher::BuildNoHashHasher::default(),
        );
        for node in ref_digraph.node_indices() {
            let nbr_count = ref_digraph
                .neighbors_directed(node, Direction::Incoming)
                .count();
            if nbr_count == 1 {
                let parent = ref_digraph
                    .neighbors_directed(node, Direction::Incoming)
                    .next()
                    .unwrap();
                leaf_index_to_parent_index.insert(node.index(), parent.index());
            }
        }
        let removables = IntSet::from_iter(leaf_index_to_parent_index.keys().copied());
        let mut reduced_ref_graph = ref_digraph.clone();
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
            cc[0][0].index() == super_root_index,
            "super root not isolated"
        );

        // SAFETY: reduced ref graph should not be modified after this point!
        let reduced_ref_digraph = reduced_ref_graph;

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
        let mut reduced_index_to_out_neighbors_weighted: Vec<SmallVec<[(usize, u32); 4]>> =
            Vec::with_capacity(num_nodes);

        for node_idx in ref_digraph.node_indices() {
            let is_removed_leaf = leaf_index_to_parent_index.contains_key(&node_idx.index());
            let mut in_neighbors: SmallVec<[usize; 4]> = if is_removed_leaf {
                SmallVec::new()
            } else {
                reduced_ref_digraph
                    .neighbors_directed(node_idx, Direction::Incoming)
                    .map(|n| n.index())
                    .collect()
            };
            let mut out_neighbors: SmallVec<[usize; 4]> = if is_removed_leaf {
                SmallVec::new()
            } else {
                reduced_ref_digraph
                    .neighbors_directed(node_idx, Direction::Outgoing)
                    .map(|n| n.index())
                    .collect()
            };

            // Sorted in ascending order by node weight
            in_neighbors.sort_by_key(|&i| index_to_weight[i]);
            out_neighbors.sort_by_key(|&i| index_to_weight[i]);

            // [(neighbor_index, neighbor_weight), ...]
            let out_neighbors_weighted: SmallVec<[(usize, u32); 4]> = out_neighbors
                .iter()
                .map(|&i| (i, index_to_weight[i]))
                .collect();

            reduced_index_to_in_neighbors.push(in_neighbors);
            reduced_index_to_out_neighbors.push(out_neighbors);
            reduced_index_to_out_neighbors_weighted.push(out_neighbors_weighted);
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
        let mut reduced_index_to_in_neighbors_out_neighbor_pos_map: Vec<IntMap<usize, usize>> =
            Vec::with_capacity(num_nodes);

        for head in ref_digraph.node_indices() {
            let is_removed_leaf = leaf_index_to_parent_index.contains_key(&head.index());
            let mut pos_map = IntMap::<usize, usize>::default();
            if !is_removed_leaf {
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
            };
            reduced_index_to_in_neighbors_out_neighbor_pos_map.push(pos_map);
        }

        // NOTE: Each of the following is safely shared between the ref_graph and the reduced_ref_graph
        // as long as the usage is for its intended mapping and not for munging of the graph membership.
        // - index_to_weight,

        assert!(index_to_out_neighbors.len() == num_nodes);
        assert!(index_to_out_neighbors_weighted.len() == num_nodes);

        Self {
            num_nodes,
            max_node_weight,
            super_root_index,
            base_town_indices,
            index_to_waypoint,
            index_to_weight,
            waypoint_to_index,

            ref_ungraph,
            index_to_neighbors_ungraph,

            ref_digraph,
            index_to_out_neighbors,

            reduced_ref_digraph,
            leaf_index_to_parent_index,
            reduced_index_to_in_neighbors,
            reduced_index_to_out_neighbors_weighted,
            reduced_index_to_in_neighbors_pos: reduced_index_to_in_neighbors_out_neighbor_pos_map,
        }
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
}
