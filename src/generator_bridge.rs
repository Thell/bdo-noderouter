use std::ops::Coroutine;
use std::pin::Pin;

use nohash_hasher::{BuildNoHashHasher, IntMap, IntSet};
use petgraph::algo::astar;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::{Dfs, IntoNodeIdentifiers};
use rapidhash::fast::{HashSetExt, RapidHashSet};

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum NodeState {
    WildFrontier = 0,
    Frontier = 1,
    Settled = 2,
}

/// Finds bridging spans of settlement border nodes within frontier and wild frontier.
#[derive(Debug, Clone)]
pub struct BridgeGenerator {
    ref_graph: StableUnGraph<usize, usize>,
    index_to_neighbors: IntMap<usize, IntSet<usize>>,
}

impl BridgeGenerator {
    pub fn new(
        ref_graph: StableUnGraph<usize, usize>,
        index_to_neighbors: IntMap<usize, IntSet<usize>>,
    ) -> Self {
        Self {
            ref_graph,
            index_to_neighbors,
        }
    }

    // /// Finds and returns nodes not in settlement with neighbors in settlement.
    // fn find_frontier_nodes(&self, settlement: &IntSet<usize>) -> IntSet<usize> {
    //     // This will result in a sorted frontier set.
    //     let mut frontier = IntSet::with_capacity_and_hasher(
    //         self.ref_graph.node_count(),
    //         BuildNoHashHasher::default(),
    //     );
    //     frontier.extend(
    //         settlement
    //             .iter()
    //             .flat_map(|v| &self.index_to_neighbors[v])
    //             .filter(|n| !settlement.contains(n)),
    //     );
    //     frontier.retain(|v| self.index_to_neighbors[v].len() > 1);
    //     frontier
    // }

    // pub fn generate_bridges<'a>(&'a self, settlement: IntSet<usize>) -> BridgeCoroutine<'a> {
    //     let num_nodes = self.ref_graph.node_count();
    //     let max_frontier_rings = 3;
    //     let ring_combo_cutoff = [0, 3, 2, 2];
    //     let mut seen_candidate_pairs = RapidHashSet::with_capacity(num_nodes);

    //     // Populate ring0 (Settlement border)
    //     let mut rings: Vec<IntSet<usize>> = Vec::with_capacity(ring_combo_cutoff.len() + 1);
    //     rings.push(settlement.clone());

    //     // This will result in a sorted set because of load% !
    //     let mut seen_nodes =
    //         IntSet::with_capacity_and_hasher(num_nodes, BuildNoHashHasher::default());
    //     seen_nodes.extend(&settlement);

    //     // This will result in a sorted set because of load% !
    //     let mut combo = IntSet::with_capacity_and_hasher(
    //         self.ref_graph.node_count(),
    //         BuildNoHashHasher::default(),
    //     );

    //     let mut reachable = Vec::with_capacity(num_nodes);

    //     Box::pin(
    //         #[coroutine]
    //         static move |_: ()| {
    //             while rings.len() <= max_frontier_rings {
    //                 // Populate the new outermost ring with a sorted set!
    //                 let nodes = self.find_frontier_nodes(&seen_nodes);

    //                 // This extend will keep seen_nodes sorted.
    //                 seen_nodes.extend(&nodes);

    //                 let ring_idx = rings.len();
    //                 rings.push(nodes.clone());

    //                 // Phase 1: Yield single node bridges
    //                 // outermost ring connecting with >=2 neighbors in inner ring.
    //                 let inner_ring = &rings[ring_idx - 1];

    //                 // This will result in a sorted set!
    //                 let mut bridge =
    //                     IntSet::with_capacity_and_hasher(num_nodes, BuildNoHashHasher::default());

    //                 for &node in &nodes {
    //                     // Singletons must connect to at least 2 distinct inner ring nodes.
    //                     if self.index_to_neighbors[&node]
    //                         .intersection(inner_ring)
    //                         .nth(1)
    //                         .is_none()
    //                     {
    //                         continue;
    //                     }

    //                     bridge.clear();
    //                     bridge.insert(node);

    //                     for bridge in self.descend_to_yield_bridges(
    //                         ring_idx,
    //                         &bridge,
    //                         bridge.clone(),
    //                         &rings,
    //                         &mut seen_candidate_pairs,
    //                     ) {
    //                         yield bridge;
    //                     }
    //                 }

    //                 // Phase 2: Yield multi-node bridges from outermost ring.
    //                 let subgraph = self.ref_graph.filter_map(
    //                     |node_idx, _| {
    //                         if nodes.contains(&node_idx.index()) {
    //                             Some(())
    //                         } else {
    //                             None
    //                         }
    //                     },
    //                     |_, edge_idx| Some(*edge_idx),
    //                 );

    //                 for u_identifier in subgraph.node_identifiers() {
    //                     // Stop isolates _in this ring_ from creating a dfs
    //                     if subgraph.neighbors_undirected(u_identifier).next().is_none() {
    //                         continue;
    //                     }

    //                     let u = u_identifier.index();

    //                     let mut dfs = Dfs::new(&subgraph, u_identifier);
    //                     reachable.clear();
    //                     while let Some(node) = dfs.next(&subgraph) {
    //                         let node_idx = node.index();
    //                         if node_idx > u {
    //                             reachable.push(node_idx);
    //                         }
    //                     }

    //                     for &v in &reachable {
    //                         // u, v are the endpoints of the path and should not share a
    //                         // common neighbor on the inner ring.
    //                         let u_neighbors = &self.index_to_neighbors[&u];
    //                         let v_neighbors = &self.index_to_neighbors[&v];
    //                         if u_neighbors
    //                             .intersection(v_neighbors)
    //                             .any(|n| inner_ring.contains(n))
    //                         {
    //                             continue;
    //                         }

    //                         // astar with no heuristic will reduce to a Dijkstra but the path
    //                         // is generated during traversal instead of rebuilt in post-processing.
    //                         if let Some((path_len, path)) = astar(
    //                             &subgraph,
    //                             u_identifier,
    //                             |f| f == NodeIndex::new(v),
    //                             |_| 1,
    //                             |_| 0,
    //                         ) {
    //                             if path_len + 1 > ring_combo_cutoff[ring_idx] {
    //                                 continue;
    //                             }

    //                             combo.clear();
    //                             combo.extend(path.iter().map(|n| n.index()));
    //                             for bridge in self.descend_to_yield_bridges(
    //                                 ring_idx,
    //                                 &combo,
    //                                 combo.clone(),
    //                                 &rings,
    //                                 &mut seen_candidate_pairs,
    //                             ) {
    //                                 yield bridge;
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //             IntSet::default()
    //         },
    //     )
    // }

    pub fn generate_bridges<'a>(&'a self, settlement: IntSet<usize>) -> BridgeCoroutine<'a> {
        let num_nodes = self.ref_graph.node_count();
        let max_frontier_rings = 3;
        let ring_combo_cutoff = [0, 3, 2, 2];
        let mut seen_candidate_pairs = RapidHashSet::with_capacity(num_nodes);

        // Populate ring0 (Settlement border)
        let mut rings: Vec<IntSet<usize>> = Vec::with_capacity(ring_combo_cutoff.len() + 1);
        rings.push(settlement.clone());

        // Initialize tri-state membership
        let mut node_state = vec![NodeState::WildFrontier; num_nodes];
        for &v in &settlement {
            node_state[v] = NodeState::Settled;
        }

        // This will result in a sorted set because of load% !
        let mut combo = IntSet::with_capacity_and_hasher(
            self.ref_graph.node_count(),
            BuildNoHashHasher::default(),
        );

        let mut reachable = Vec::with_capacity(num_nodes);

        Box::pin(
            #[coroutine]
            static move |_: ()| {
                while rings.len() <= max_frontier_rings {
                    // Build next frontier strictly from the current outer ring -----
                    let previous_frontier = rings.last().unwrap();

                    // This will result in a sorted set because of load% !
                    let mut frontier =
                        IntSet::with_capacity_and_hasher(num_nodes, BuildNoHashHasher::default());

                    for &v in previous_frontier {
                        for &n in &self.index_to_neighbors[&v] {
                            if node_state[n] == NodeState::WildFrontier {
                                if self.index_to_neighbors[&n].len() > 1 {
                                    node_state[n] = NodeState::Frontier;
                                    frontier.insert(n);
                                }
                            }
                        }
                    }
                    if frontier.is_empty() {
                        break;
                    }
                    for &n in &frontier {
                        node_state[n] = NodeState::Settled;
                    }

                    let ring_idx = rings.len();
                    rings.push(frontier.clone());

                    // Phase 1: Yield single node bridges
                    // outermost ring connecting with >=2 neighbors in inner ring.
                    let inner_ring = &rings[ring_idx - 1];

                    // This will result in a sorted set!
                    let mut bridge =
                        IntSet::with_capacity_and_hasher(num_nodes, BuildNoHashHasher::default());

                    for &node in &frontier {
                        // Singletons must connect to at least 2 distinct inner ring.
                        if self.index_to_neighbors[&node]
                            .intersection(inner_ring)
                            .nth(1)
                            .is_none()
                        {
                            continue;
                        }

                        bridge.clear();
                        bridge.insert(node);

                        for bridge in self.descend_to_yield_bridges(
                            ring_idx,
                            &bridge,
                            bridge.clone(),
                            &rings,
                            &mut seen_candidate_pairs,
                        ) {
                            yield bridge;
                        }
                    }

                    // Phase 2: Yield multi-node bridges from outermost ring.
                    let subgraph = self.ref_graph.filter_map(
                        |node_idx, _| {
                            if frontier.contains(&node_idx.index()) {
                                Some(())
                            } else {
                                None
                            }
                        },
                        |_, edge_idx| Some(*edge_idx),
                    );

                    for u_identifier in subgraph.node_identifiers() {
                        // Stop isolates _in this ring_ from creating a dfs
                        if subgraph.neighbors_undirected(u_identifier).next().is_none() {
                            continue;
                        }

                        let u = u_identifier.index();

                        let mut dfs = Dfs::new(&subgraph, u_identifier);
                        reachable.clear();
                        while let Some(node) = dfs.next(&subgraph) {
                            let node_idx = node.index();
                            if node_idx > u {
                                reachable.push(node_idx);
                            }
                        }

                        for &v in &reachable {
                            // u, v are the endpoints of the path and should not share a
                            // common neighbor on the inner ring.
                            let u_neighbors = &self.index_to_neighbors[&u];
                            let v_neighbors = &self.index_to_neighbors[&v];
                            if u_neighbors
                                .intersection(v_neighbors)
                                .any(|n| inner_ring.contains(n))
                            {
                                continue;
                            }

                            // astar with no heuristic will reduce to a Dijkstra but the path
                            // is generated during traversal instead of rebuilt in post-processing.
                            if let Some((path_len, path)) = astar(
                                &subgraph,
                                u_identifier,
                                |f| f == NodeIndex::new(v),
                                |_| 1,
                                |_| 0,
                            ) {
                                if path_len + 1 > ring_combo_cutoff[ring_idx] {
                                    continue;
                                }

                                combo.clear();
                                combo.extend(path.iter().map(|n| n.index()));

                                for bridge in self.descend_to_yield_bridges(
                                    ring_idx,
                                    &combo,
                                    combo.clone(),
                                    &rings,
                                    &mut seen_candidate_pairs,
                                ) {
                                    yield bridge;
                                }
                            }
                        }
                    }
                }
                IntSet::default()
            },
        )
    }

    /// Descend from current ring to settlement frontier (F0)
    /// yielding bridges connecting ≥2 nodes.
    fn descend_to_yield_bridges(
        &self,
        ring_idx: usize,
        current_nodes: &IntSet<usize>,
        bridge: IntSet<usize>,
        rings: &Vec<IntSet<usize>>,
        seen_candidate_pairs: &mut RapidHashSet<(usize, usize)>,
    ) -> impl Iterator<Item = IntSet<usize>> {
        let mut output = Vec::new();

        // Base case (Settlement Frontier): yield and return
        if ring_idx == 1 {
            output.push(bridge);

        // Descending case:
        } else {
            // Collect and process pairwise combinations of current_nodes candidates...
            // Candidates are current_nodes neighbors in ring_idx - 1
            let inner_ring = &rings[ring_idx - 1];
            let mut candidates: Vec<_> = current_nodes
                .iter()
                .flat_map(|&n| &self.index_to_neighbors[&n])
                .filter(|x| inner_ring.contains(x))
                .copied()
                .collect();
            candidates.sort_unstable();
            candidates.dedup();

            for i in 0..(candidates.len() - 1) {
                let u = candidates[i];
                for &v in candidates.iter().skip(i + 1) {
                    if seen_candidate_pairs.insert((u, v)) {
                        let new_current = IntSet::from_iter([u, v]);
                        let mut new_bridge = bridge.clone();
                        new_bridge.extend([u, v]);
                        output.extend(self.descend_to_yield_bridges(
                            ring_idx - 1,
                            &new_current,
                            new_bridge,
                            rings,
                            seen_candidate_pairs,
                        ));
                    }
                }
            }
        }
        output.into_iter()
    }
}

impl Default for BridgeGenerator {
    fn default() -> Self {
        Self {
            ref_graph: StableUnGraph::default(),
            index_to_neighbors: IntMap::default(),
        }
    }
}

pub type BridgeCoroutine<'a> =
    Pin<Box<dyn Coroutine<(), Yield = IntSet<usize>, Return = IntSet<usize>> + 'a>>;
