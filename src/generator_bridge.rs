use std::ops::Coroutine;
use std::pin::Pin;

use nohash_hasher::{BuildNoHashHasher, IntMap, IntSet};
use petgraph::stable_graph::StableUnGraph;
use rapidhash::fast::{HashSetExt, RapidHashSet};
use smallvec::SmallVec;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum NodeState {
    WildFrontier = 0,
    Frontier = 1,
    Settled = 2,
}

/// Finds bridging spans of settlement border nodes within frontier and wild frontier.
#[derive(Debug, Default, Clone)]
pub struct BridgeGenerator {
    ref_graph: StableUnGraph<usize, usize>,
    index_to_neighbors: IntMap<usize, SmallVec<[usize; 4]>>,
}

impl BridgeGenerator {
    pub fn new(
        ref_graph: StableUnGraph<usize, usize>,
        index_to_neighbors: IntMap<usize, SmallVec<[usize; 4]>>,
    ) -> Self {
        Self {
            ref_graph,
            index_to_neighbors,
        }
    }

    /// Bounded BFS restricted to `allowed` set.
    /// Returns path if <= cutoff, else None.
    fn bounded_path(
        &self,
        start: usize,
        goal: usize,
        allowed: &IntSet<usize>,
        cutoff: usize,
    ) -> Option<Vec<usize>> {
        use std::collections::VecDeque;

        let mut queue = VecDeque::new();
        let mut parent: IntMap<usize, usize> = IntMap::default();

        queue.push_back(start);
        parent.insert(start, start);

        while let Some(u) = queue.pop_front() {
            if u == goal {
                // Reconstruct path
                let mut path = Vec::new();
                let mut cur = goal;
                while cur != start {
                    path.push(cur);
                    cur = parent[&cur];
                }
                path.push(start);
                return Some(path);
            }

            let depth = {
                // depth = distance from start
                let mut d = 0;
                let mut cur = u;
                while cur != start {
                    cur = parent[&cur];
                    d += 1;
                }
                d
            };

            if depth >= cutoff {
                continue;
            }

            for &n in &self.index_to_neighbors[&u] {
                if allowed.contains(&n) && !parent.contains_key(&n) {
                    parent.insert(n, u);
                    queue.push_back(n);
                }
            }
        }

        None
    }

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

        let mut reachable = Vec::with_capacity(num_nodes);

        Box::pin(
            #[coroutine]
            static move |_: ()| {
                while rings.len() <= max_frontier_rings {
                    // Build next frontier strictly from the current outer ring
                    let previous_frontier = rings.last().unwrap();

                    // This will result in a sorted set because of load% !
                    let mut frontier =
                        IntSet::with_capacity_and_hasher(num_nodes, BuildNoHashHasher::default());

                    for &v in previous_frontier {
                        for &n in &self.index_to_neighbors[&v] {
                            if node_state[n] == NodeState::WildFrontier
                                && self.index_to_neighbors[&n].len() > 1
                            {
                                node_state[n] = NodeState::Frontier;
                                frontier.insert(n);
                            }
                        }
                    }
                    if frontier.is_empty() {
                        break;
                    }

                    let ring_idx = rings.len();
                    rings.push(frontier.clone());

                    // Phase 1: Yield single node bridges
                    // outermost ring connecting with >=2 neighbors in inner ring.
                    let inner_ring = &rings[ring_idx - 1];

                    for &node in &frontier {
                        // Singletons must connect to at least 2 distinct inner ring.
                        if self.index_to_neighbors[&node]
                            .iter()
                            .filter(|n| inner_ring.contains(n))
                            .nth(1)
                            .is_none()
                        {
                            continue;
                        }

                        for bridge in self.descend_to_yield_bridges(
                            ring_idx,
                            vec![node],
                            &rings,
                            &mut seen_candidate_pairs,
                        ) {
                            yield bridge;
                        }
                    }

                    for &u in &frontier {
                        // Skip isolates inside this ring
                        if self.index_to_neighbors[&u]
                            .iter()
                            .all(|n| !frontier.contains(n))
                        {
                            continue;
                        }

                        // Collect reachable nodes inside the frontier (u < v ordering)
                        reachable.clear();
                        let mut dfs = vec![u];
                        let mut seen = IntSet::default();
                        seen.insert(u);

                        while let Some(x) = dfs.pop() {
                            for &n in &self.index_to_neighbors[&x] {
                                if n > u && frontier.contains(&n) && seen.insert(n) {
                                    dfs.push(n);
                                    reachable.push(n);
                                }
                            }
                        }

                        for &v in &reachable {
                            // u, v should not share a common neighbor in the inner ring
                            let u_neighbors = &self.index_to_neighbors[&u];
                            let v_neighbors = &self.index_to_neighbors[&v];
                            if u_neighbors
                                .iter()
                                .filter(|n| v_neighbors.contains(n))
                                .any(|n| inner_ring.contains(n))
                            {
                                continue;
                            }

                            if let Some(path) =
                                self.bounded_path(u, v, &frontier, ring_combo_cutoff[ring_idx])
                            {
                                if path.len() > ring_combo_cutoff[ring_idx] {
                                    continue;
                                }

                                for bridge in self.descend_to_yield_bridges(
                                    ring_idx,
                                    path,
                                    &rings,
                                    &mut seen_candidate_pairs,
                                ) {
                                    yield bridge;
                                }
                            }
                        }
                    }
                }
                Vec::default()
            },
        )
    }

    /// Descend from current ring to settlement frontier (F0)
    /// yielding bridges connecting ≥2 nodes.
    fn descend_to_yield_bridges(
        &self,
        ring_idx: usize,
        mut bridge: Vec<usize>,
        rings: &Vec<IntSet<usize>>,
        seen_candidate_pairs: &mut RapidHashSet<(usize, usize)>,
    ) -> impl Iterator<Item = Vec<usize>> {
        let mut output = Vec::new();

        // Base case (Settlement Frontier): yield and return
        if ring_idx == 1 {
            output.push(bridge);

        // Descending case:
        } else {
            // Collect and process pairwise combinations of current_nodes candidates...
            //
            // Current frontier is the working contents of `bridge` but we only want
            // to consider the *last step*'s additions to expand from.
            // So we collect candidate next-layer nodes by scanning neighbors of the
            // *current frontier subset*.
            //
            // That subset is: bridge ∩ rings[ring_idx]
            // Candidates are neighbors in inner ring (ring_idx - 1)
            let inner_ring = &rings[ring_idx - 1];
            let mut candidates: Vec<usize> = bridge
                .iter()
                .copied()
                .filter(|n| rings[ring_idx].contains(n))
                .flat_map(|n| self.index_to_neighbors[&n].iter().copied())
                .filter(|x| inner_ring.contains(x))
                .collect();

            candidates.sort_unstable();
            candidates.dedup();

            for i in 0..(candidates.len().saturating_sub(1)) {
                let u = candidates[i];
                for &v in candidates.iter().skip(i + 1) {
                    if seen_candidate_pairs.insert((u, v)) {
                        // Extend bridge in-place, recurse, then backtrack
                        bridge.push(u);
                        bridge.push(v);

                        output.extend(self.descend_to_yield_bridges(
                            ring_idx - 1,
                            bridge.clone(),
                            rings,
                            seen_candidate_pairs,
                        ));

                        // // Backtrack: remove u,v so outer loop stays clean
                        bridge.pop();
                        bridge.pop();
                    }
                }
            }
        }

        output.into_iter()
    }
}

pub type BridgeCoroutine<'a> =
    Pin<Box<dyn Coroutine<(), Yield = Vec<usize>, Return = Vec<usize>> + 'a>>;
