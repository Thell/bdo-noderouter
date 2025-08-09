use crate::helpers_common::{hash_intset, sort_pair};
use nohash_hasher::{IntMap, IntSet};
use petgraph::algo::astar;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::IntoNodeIdentifiers;
use rapidhash::fast::RapidHashSet;
use std::ops::Coroutine;
use std::pin::Pin;

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

    fn find_frontier_nodes(
        &self,
        settlement: &IntSet<usize>,
        min_degree: Option<usize>,
    ) -> IntSet<usize> {
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
                    .is_some_and(|nbrs| nbrs.len() >= min_degree)
            });
        }
        frontier
    }

    pub fn generate_bridges<'a>(&'a self, settlement: IntSet<usize>) -> BridgeCoroutine<'a> {
        let max_frontier_rings = 3;
        let ring_combo_cutoff = [0, 3, 2, 2];
        let mut seen_candidate_pairs = RapidHashSet::default();
        let mut yielded_hashes: IntSet<u64> = IntSet::default();
        let mut rings: Vec<IntSet<usize>> = vec![settlement.clone()];
        let mut seen_nodes = settlement.clone();

        Box::pin(
            #[coroutine]
            static move |_: ()| {
                while rings.len() <= max_frontier_rings {
                    // Populate the new outermost ring
                    let nodes = self.find_frontier_nodes(&seen_nodes, Some(2));
                    seen_nodes.extend(&nodes);
                    let ring_idx = rings.len();
                    rings.push(nodes.clone());

                    // Phase 1: Single-node bridges
                    let inner_ring = &rings[ring_idx - 1];
                    for node in nodes.clone() {
                        let neighbors: IntSet<usize> = self
                            .index_to_neighbors
                            .get(&node)
                            .unwrap()
                            .intersection(inner_ring)
                            .copied()
                            .collect();

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
                    }

                    // Phase 2: Multi-node bridges
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
                    let mut node_identifiers: Vec<_> = subgraph.node_identifiers().collect();
                    node_identifiers.sort_unstable();
                    let node_indices: Vec<_> =
                        node_identifiers.iter().map(|&v| v.index()).collect();

                    let mut seen_endpoints = RapidHashSet::default();
                    for i in 0..(node_indices.len() - 1) {
                        let u = node_indices[i];
                        let u_identifier = NodeIndex::new(u);

                        for &v in node_indices.iter().skip(i + 1) {
                            let v_identifier = NodeIndex::new(v);

                            let key = sort_pair(u, v);
                            if seen_endpoints.contains(&key) {
                                continue;
                            }
                            seen_endpoints.insert(key);

                            let u_neighbors = self.index_to_neighbors.get(&u).unwrap();
                            let v_neighbors = self.index_to_neighbors.get(&v).unwrap();
                            if u_neighbors
                                .intersection(v_neighbors)
                                .any(|n| inner_ring.contains(n))
                            {
                                continue;
                            }

                            if let Some((path_len, path)) =
                                astar(&subgraph, u_identifier, |f| f == v_identifier, |_| 1, |_| 0)
                            {
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
        if ring_idx == 1 {
            let bridge_hash = hash_intset(&bridge);
            if !yielded.contains(&bridge_hash) {
                let s_neighbors: IntSet<usize> = current_nodes
                    .iter()
                    .flat_map(|&v| self.index_to_neighbors.get(&v).unwrap())
                    .filter(|n| rings[0].contains(n))
                    .copied()
                    .collect();

                if s_neighbors.len() >= 2 {
                    yielded.insert(bridge_hash);
                    output.push(bridge);
                }
            }
        } else {
            let inner_ring = &rings[ring_idx - 1];
            let mut candidates: Vec<usize> = current_nodes
                .iter()
                .flat_map(|&n| self.index_to_neighbors.get(&n).unwrap())
                .filter(|x| inner_ring.contains(x))
                .copied()
                .collect::<IntSet<_>>()
                .into_iter()
                .collect();
            candidates.sort_unstable();

            for i in 0..(candidates.len() - 1) {
                let u = candidates[i];
                for &v in candidates.iter().skip(i + 1) {
                    let candidate_pair = sort_pair(u, v);
                    if !seen_candidate_pairs.insert(candidate_pair) {
                        continue;
                    }

                    let mut new_current = IntSet::default();
                    new_current.insert(u);
                    new_current.insert(v);
                    let mut new_bridge = bridge.clone();
                    new_bridge.insert(u);
                    new_bridge.insert(v);

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
        output.into_iter()
    }
}

pub type BridgeCoroutine<'a> =
    Pin<Box<dyn Coroutine<(), Yield = IntSet<usize>, Return = IntSet<usize>> + 'a>>;
