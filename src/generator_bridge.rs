use crate::helpers_common::hash_intset;
use nohash_hasher::{IntMap, IntSet};
use petgraph::algo::astar;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableUnGraph;
use rapidhash::fast::RapidHashSet;
use std::ops::Coroutine;
use std::pin::Pin;

// ref_graph => full reference graph of all nodes and edges.
// Contains potentially topologically constrained expansion of Settlement
// Settlement potentially contains disjoint connected components
//
// S => Settlement (induced subgraph of currently 'active' nodes of ref_graph)
// B => Border (nodes in Settlement with ref_graph neighbors in Frontier)
// F => Frontier (non-settled nodes with ref_graph neighbors in Settlement)
// W => Wild Frontier (non-settled nodes with no settled ref_graph neighbors)
//
// 𝓕 => Fringe edges (edges in ref_graph connecting border and frontier nodes)
// 𝓑 => bridging span of nodes not in S that connect distinct nodes in S.
//
// Let ring 0 => B (in most cases B == S since S is most commonly a Steiner Forest/Tree)
// Let ring 1 => F0 be an eccentric ring around {B}
// Let ring 2 => F1 be an eccentric ring around {F0|B}
// Let ring 3 => F2 be an eccentric ring around {F1|F0|B}
// ...

/// Finds bridging spans of settlement border nodes within frontier and wild frontier.
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

    /// Finds and returns nodes not in settlement with neighbors in settlement.
    fn find_frontier_nodes(
        &self,
        settlement: &IntSet<usize>,
        min_degree: Option<usize>,
    ) -> IntSet<usize> {
        let mut frontier = settlement
            .iter()
            .flat_map(|v| &self.index_to_neighbors[v])
            .filter(|n| !settlement.contains(n))
            .copied()
            .collect::<IntSet<_>>();
        if let Some(min_degree) = min_degree {
            frontier.retain(|v| self.index_to_neighbors[v].len() >= min_degree);
        }
        frontier
    }

    pub fn generate_bridges<'a>(&'a self, settlement: IntSet<usize>) -> BridgeCoroutine<'a> {
        let max_frontier_rings = 3;
        let ring_combo_cutoff = [0, 3, 2, 2];
        let mut seen_candidate_pairs = RapidHashSet::default();
        let mut yielded_hashes: IntSet<u64> = IntSet::default();

        // Populate ring0 (Settlement border)
        let mut rings: Vec<IntSet<usize>> = vec![settlement.clone()];
        let mut seen_nodes = settlement;

        Box::pin(
            #[coroutine]
            static move |_: ()| {
                while rings.len() <= max_frontier_rings {
                    // Populate the new outermost ring
                    let nodes = self.find_frontier_nodes(&seen_nodes, Some(2));
                    seen_nodes.extend(&nodes);
                    let ring_idx = rings.len();
                    rings.push(nodes.clone());

                    // Phase 1:
                    // Yield single node bridges from outermost ring connecting with >=2 neighbors in inner ring.
                    let inner_ring = &rings[ring_idx - 1];
                    for &node in &nodes {
                        if self.index_to_neighbors[&node]
                            .intersection(inner_ring)
                            .nth(1)
                            .is_none()
                        {
                            continue;
                        }
                        let bridge = IntSet::from_iter([node]);
                        for bridge in self.descend_to_yield_bridges(
                            ring_idx,
                            &bridge,
                            bridge.clone(),
                            &rings,
                            &mut yielded_hashes,
                            &mut seen_candidate_pairs,
                        ) {
                            yield bridge;
                        }
                    }

                    // Phase 2:
                    // Yield multi-node bridges from outermost ring.

                    // NOTE: Each ring is only a single node 'thick' and in the python implementation
                    // the `rustworkx.all_pairs_all_simple_paths` is efficient at obtaining all size
                    // constrained connected runs within the ring along with the endpoint nodes.
                    // This Rust implementation utilizes petgraph's A* algorithm on a per pair
                    // basis for because with the simple heuristic of node cost 1 and edge cost 0
                    // A* behaves like Dijkstra but will return the path len and ordered path while
                    // the path would need to be built by processing the Dijkstra returned values.
                    // We get away with the node cost of 1 because of the single node 'thick'
                    // ring, so we are simply counting hops to control the length of connected runs.
                    let subgraph = self.ref_graph.filter_map(
                        |node_idx, _| {
                            if nodes.contains(&node_idx.index()) {
                                Some(())
                            } else {
                                None
                            }
                        },
                        |_, edge_idx| Some(*edge_idx),
                    );
                    let mut node_indices: Vec<_> = nodes.iter().copied().collect();
                    node_indices.sort_unstable();

                    for i in 0..(node_indices.len() - 1) {
                        let u = node_indices[i];
                        let u_identifier = NodeIndex::new(u);

                        for &v in node_indices.iter().skip(i + 1) {
                            let u_neighbors = &self.index_to_neighbors[&u];
                            let v_neighbors = &self.index_to_neighbors[&v];
                            if u_neighbors
                                .intersection(v_neighbors)
                                .any(|n| inner_ring.contains(n))
                            {
                                continue;
                            }

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

                                let combo = IntSet::from_iter(path.iter().map(|n| n.index()));
                                for bridge in self.descend_to_yield_bridges(
                                    ring_idx,
                                    &combo,
                                    combo.clone(),
                                    &rings,
                                    &mut yielded_hashes,
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

    /// Descend from current ring to settlement frontier (F0), yielding bridges connecting ≥2 S nodes.
    fn descend_to_yield_bridges(
        &self,
        ring_idx: usize,
        current_nodes: &IntSet<usize>,
        bridge: IntSet<usize>,
        rings: &Vec<IntSet<usize>>,
        yielded: &mut IntSet<u64>,
        seen_candidate_pairs: &mut RapidHashSet<(usize, usize)>,
    ) -> impl Iterator<Item = IntSet<usize>> {
        let mut output = Vec::new();

        // Base case (Settlement Frontier): yield and return
        if ring_idx == 1 {
            let bridge_hash = hash_intset(&bridge);
            if !yielded.contains(&bridge_hash) {
                if current_nodes
                    .iter()
                    .flat_map(|&v| &self.index_to_neighbors[&v])
                    .filter(|n| rings[0].contains(n))
                    .nth(1)
                    .is_some()
                {
                    yielded.insert(bridge_hash);
                    output.push(bridge);
                }
            }

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
                            yielded,
                            seen_candidate_pairs,
                        ));
                    }
                }
            }
        }
        output.into_iter()
    }
}

pub type BridgeCoroutine<'a> =
    Pin<Box<dyn Coroutine<(), Yield = IntSet<usize>, Return = IntSet<usize>> + 'a>>;
