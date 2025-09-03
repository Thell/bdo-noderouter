use std::collections::VecDeque;
use std::pin::Pin;
use std::{cell::RefCell, ops::Coroutine};

use fixedbitset::FixedBitSet;
use nohash_hasher::IntMap;
use petgraph::stable_graph::StableUnGraph;
use rapidhash::{HashSetExt, RapidHashSet};
use smallvec::SmallVec;

#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeState {
    WildFrontier = 0,
    Frontier = 1,
    Settled = 2,
}

/// Finds bridging spans of settlement border nodes within frontier and wild frontier.
#[derive(Debug, Default, Clone)]
pub struct BridgeGenerator {
    ref_graph: StableUnGraph<usize, usize>,
    index_to_neighbors: Vec<SmallVec<[usize; 4]>>,
    stack: RefCell<Vec<usize>>,
    node_scratch0: RefCell<FixedBitSet>,
    frontier_buffer: RefCell<FixedBitSet>,
    parent_map: RefCell<IntMap<usize, usize>>,
    queue_buffer: RefCell<VecDeque<(usize, usize)>>,
}

impl BridgeGenerator {
    pub fn new(
        ref_graph: StableUnGraph<usize, usize>,
        index_to_neighbors: Vec<SmallVec<[usize; 4]>>,
    ) -> Self {
        let node_count = ref_graph.node_count();
        Self {
            ref_graph,
            index_to_neighbors,
            stack: RefCell::new(Vec::with_capacity(node_count)),
            node_scratch0: RefCell::new(FixedBitSet::with_capacity(node_count)),
            frontier_buffer: RefCell::new(FixedBitSet::with_capacity(node_count)),
            parent_map: RefCell::new(IntMap::default()),
            queue_buffer: RefCell::new(VecDeque::new()),
        }
    }

    /// Bounded BFS restricted to `allowed` set.
    fn bounded_path(
        &self,
        start: usize,
        goal: usize,
        allowed: &FixedBitSet,
        cutoff: usize,
    ) -> Option<Vec<usize>> {
        let mut queue = self.queue_buffer.borrow_mut();
        let mut parent = self.parent_map.borrow_mut();

        queue.clear();
        parent.clear();

        queue.push_back((start, 0));
        parent.insert(start, start);

        while let Some((u, depth)) = queue.pop_front() {
            if depth >= cutoff {
                continue;
            }

            if u == goal {
                let mut path = Vec::with_capacity(depth + 1);
                let mut cur = goal;
                while cur != start {
                    path.push(cur);
                    cur = parent[&cur];
                }
                path.push(start);
                return Some(path);
            }

            for &n in &self.index_to_neighbors[u] {
                if allowed.contains(n) && !parent.contains_key(&n) {
                    parent.insert(n, u);
                    queue.push_back((n, depth + 1));
                }
            }
        }

        None
    }

    fn reachable<'a>(
        &'a self,
        start: usize,
        frontier: &'a FixedBitSet,
    ) -> impl Iterator<Item = usize> + 'a {
        let mut stack = self.stack.borrow_mut();
        let mut visited = self.node_scratch0.borrow_mut();
        stack.clear();
        visited.clear();
        stack.push(start);
        visited.insert(start);

        std::iter::from_fn(move || {
            if let Some(x) = stack.pop() {
                for &n in &self.index_to_neighbors[x] {
                    if n > start && frontier.contains(n) && !visited.put(n) {
                        stack.push(n);
                    }
                }
                return Some(x);
            }
            None
        })
    }

    pub fn generate_bridges<'a>(&'a self, settlement: FixedBitSet) -> BridgeCoroutine<'a> {
        let num_nodes = self.ref_graph.node_count();
        let max_frontier_rings = 3;
        let ring_combo_cutoff = [0, 3, 2, 2];
        let mut seen_candidate_pairs = RapidHashSet::with_capacity(num_nodes);

        // Initialize tri-state membership
        let mut node_state = vec![NodeState::WildFrontier; num_nodes];
        for v in settlement.ones() {
            node_state[v] = NodeState::Settled;
        }

        // Populate ring0 (Settlement border)
        let mut rings: Vec<FixedBitSet> = Vec::with_capacity(max_frontier_rings + 1);
        rings.push(settlement);

        Box::pin(
            #[coroutine]
            static move |_: ()| {
                while rings.len() <= max_frontier_rings {
                    // Build next frontier strictly from the current outer ring
                    let previous_frontier = rings.last().unwrap();
                    let mut frontier = self.frontier_buffer.borrow_mut();
                    frontier.clear();

                    for v in previous_frontier.ones() {
                        for &n in &self.index_to_neighbors[v] {
                            if node_state[n] == NodeState::WildFrontier
                                && self.index_to_neighbors[n].len() > 1
                            {
                                node_state[n] = NodeState::Frontier;
                                frontier.insert(n);
                            }
                        }
                    }

                    let ring_idx = rings.len();
                    rings.push(frontier.clone());

                    // Phase 1: Yield single node bridges
                    // outermost ring connecting with >=2 neighbors in inner ring.
                    let inner_ring = &rings[ring_idx - 1];

                    for node in frontier.ones() {
                        // Singletons must connect to at least 2 distinct inner ring.
                        if self.index_to_neighbors[node]
                            .iter()
                            .filter(|&&n| inner_ring.contains(n))
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

                    // Phase 2: Yield multi-node bridges from outermost ring.
                    for u in frontier.ones() {
                        // Skip isolates inside this ring
                        if self.index_to_neighbors[u]
                            .iter()
                            .all(|&n| !frontier.contains(n))
                        {
                            continue;
                        }

                        for v in self.reachable(u, &frontier) {
                            let u_neighbors = &self.index_to_neighbors[u];
                            let v_neighbors = &self.index_to_neighbors[v];
                            if u_neighbors
                                .iter()
                                .filter(|n| v_neighbors.contains(n))
                                .any(|&n| inner_ring.contains(n))
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
        rings: &[FixedBitSet],
        seen_candidate_pairs: &mut RapidHashSet<(usize, usize)>,
    ) -> impl Iterator<Item = Vec<usize>> {
        let mut output = Vec::with_capacity(bridge.len() * (bridge.len() + 1) / 2);

        // Base case (Settlement Frontier): yield and return
        if ring_idx == 1 {
            output.push(bridge);

        // Descending case:
        } else {
            // Collect and process pairwise combinations of current_nodes candidates...
            //
            // Current frontier is the working contents of `bridge` but we only want
            // to consider the *last step*'s additions to expand from.
            // That subset is: bridge ∩ rings[ring_idx]

            // Candidates are neighbors in inner ring (ring_idx - 1)
            let inner_ring = &rings[ring_idx - 1];
            let mut candidates: Vec<usize> = bridge
                .iter()
                .copied()
                .filter(|&n| rings[ring_idx].contains(n))
                .flat_map(|n| self.index_to_neighbors[n].iter().copied())
                .filter(|&x| inner_ring.contains(x))
                .collect();

            candidates.sort_unstable();
            candidates.dedup();

            for i in 0..(candidates.len().saturating_sub(1)) {
                let u = candidates[i];
                for &v in candidates.iter().skip(i + 1) {
                    if seen_candidate_pairs.insert((u, v)) {
                        bridge.push(u);
                        bridge.push(v);

                        output.extend(self.descend_to_yield_bridges(
                            ring_idx - 1,
                            bridge.clone(),
                            rings,
                            seen_candidate_pairs,
                        ));

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
