use std::collections::VecDeque;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use fixedbitset::FixedBitSet;
use nohash_hasher::{BuildNoHashHasher, IntMap, IntSet};

use rapidhash::RapidHashSet;
use smallvec::SmallVec;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Node {
    parent: i32,
    subtree_size: usize,
    adj: SmallVec<[usize; 4]>,
}

impl Node {
    #[inline]
    fn new() -> Self {
        Node {
            parent: -1,
            subtree_size: 1,
            adj: SmallVec::new(),
        }
    }

    #[inline]
    fn insert_adj(&mut self, u: usize) {
        // preserve set semantics (no duplicates)
        if !self.adj.contains(&u) {
            self.adj.push(u);
        }
    }

    #[inline]
    fn delete_adj(&mut self, u: usize) {
        if let Some(i) = self.adj.iter().position(|&x| x == u) {
            self.adj.swap_remove(i);
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "python", pyclass(unsendable))]
#[allow(unused)]
pub struct IDTree {
    n: usize,
    nodes: Vec<Node>,
    distance_generations: Vec<u32>,    // (used for betweenness)
    distances: Vec<i32>,               // (used for betweenness)
    current_distance_generation: u32,  // (used for betweenness)
    deque_scratch: VecDeque<usize>,    // scratch area (used by shortest path)
    node_vec_scratch: Vec<usize>,      // |nodes| len scratch area
    vec_bool_scratch: Vec<bool>,       // scratch area
    vec_scratch_stack: Vec<usize>,     // scratch area
    node_bitset_scratch0: FixedBitSet, // |nodes| len scratch area
    node_bitset_scratch1: FixedBitSet, // |nodes| len scratch area
    node_bitset_scratch2: FixedBitSet, // |nodes| len scratch area
}

#[cfg(feature = "python")]
#[pymethods]
impl IDTree {
    #[new]
    fn py_new(adj_dict: std::collections::HashMap<usize, Vec<usize>>) -> Self {
        let adj_dict: IntMap<usize, IntSet<usize>> = adj_dict
            .into_iter()
            .map(|(k, v)| (k, IntSet::from_iter(v)))
            .collect();
        let mut instance = Self::new(&adj_dict);
        instance.initialize();
        instance
    }

    #[pyo3(name = "clone")]
    fn py_clone(&self) -> Self {
        self.clone()
    }

    #[pyo3(name = "insert_edge")]
    pub fn py_insert_edge(&mut self, u: usize, v: usize) -> i32 {
        self.insert_edge(u, v)
    }

    #[pyo3(name = "delete_edge")]
    pub fn py_delete_edge(&mut self, u: usize, v: usize) -> i32 {
        self.delete_edge(u, v)
    }

    #[pyo3(name = "query")]
    pub fn py_query(&self, u: usize, v: usize) -> bool {
        self.query(u, v)
    }

    #[pyo3(name = "cycle_basis")]
    pub fn py_cycle_basis(&mut self, root: Option<usize>) -> Vec<Vec<usize>> {
        self.cycle_basis(root)
    }

    #[pyo3(name = "node_connected_component")]
    pub fn py_node_connected_component(&mut self, v: usize) -> Vec<usize> {
        self.node_connected_component(v)
    }

    #[pyo3(name = "num_connected_components")]
    pub fn py_num_connected_components(&mut self) -> usize {
        self.num_connected_components()
    }

    #[pyo3(name = "connected_components")]
    pub fn py_connected_components(&mut self) -> Vec<Vec<usize>> {
        self.connected_components()
    }

    #[pyo3(name = "active_nodes")]
    pub fn py_active_nodes(&mut self) -> Vec<usize> {
        self.active_nodes_vec()
    }

    #[pyo3(name = "isolate_node")]
    pub fn py_isolate_node(&mut self, v: usize) {
        self.isolate_node(v)
    }

    #[pyo3(name = "isolate_nodes")]
    pub fn py_isolate_nodes(&mut self, nodes: Vec<usize>) {
        self.isolate_nodes(nodes)
    }

    #[pyo3(name = "is_isolated")]
    pub fn py_is_isolated(&mut self, v: usize) -> bool {
        self.is_isolated(v)
    }

    #[pyo3(name = "degree")]
    pub fn py_degree(&mut self, v: usize) -> i32 {
        self.degree(v)
    }

    #[pyo3(name = "neighbors")]
    pub fn py_neighbors(&mut self, v: usize) -> Vec<usize> {
        self.neighbors(v)
    }

    #[pyo3(name = "retain_active_nodes_from")]
    pub fn py_retain_active_nodes_from(&mut self, from_indices: Vec<usize>) -> Vec<usize> {
        self.retain_active_nodes_from(from_indices)
    }
}

impl IDTree {
    // MARK: Core

    pub fn insert_edge(&mut self, u: usize, v: usize) -> i32 {
        if !self.insert_edge_in_graph(u, v) {
            return -1;
        }
        self.insert_edge_balanced(u, v)
    }

    pub fn delete_edge(&mut self, u: usize, v: usize) -> i32 {
        if !self.delete_edge_in_graph(u, v) {
            return -1;
        }
        self.delete_edge_balanced(u, v)
    }

    pub fn query(&self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n {
            return false;
        }
        let mut root_u = u;
        while self.nodes[root_u].parent != -1 {
            root_u = self.nodes[root_u].parent as usize;
        }
        let mut root_v = v;
        while self.nodes[root_v].parent != -1 {
            root_v = self.nodes[root_v].parent as usize;
        }
        root_u == root_v
    }

    // MARK: Extensions

    /// Rooted Tree-Based Fundamental Cycle Basis
    pub fn cycle_basis(&mut self, root: Option<usize>) -> Vec<Vec<usize>> {
        // Constructs a fundamental cycle basis for the connected component containing `root`,
        // using the ID-Tree structure as its spanning tree. A fundamental cycle is formed
        // each time a non-tree edge is encountered during DFS from the `root`.
        if root.is_none() {
            return vec![];
        }
        let root = root.unwrap();

        let mut cycles = Vec::with_capacity(self.n / 2);

        let stack = &mut self.vec_scratch_stack;
        let in_component = &mut self.node_bitset_scratch0;

        stack.clear();
        in_component.clear();

        stack.push(root);
        in_component.set(root, true);

        while let Some(u) = stack.pop() {
            for &v in &self.nodes[u].adj {
                if !in_component[v] {
                    stack.push(v);
                    in_component.set(v, true);
                }

                let pu = self.nodes[u].parent;
                let pv = self.nodes[v].parent;
                if pu == v as i32 || pv == u as i32 {
                    continue;
                }

                if u >= v {
                    continue;
                }

                // Found a fundamental cycle via (u, v)
                let mut path_u = Vec::with_capacity(self.n);
                let mut path_v = Vec::with_capacity(self.n);
                path_u.push(u);
                path_v.push(v);

                let visited_u = &mut self.node_bitset_scratch1;
                let visited_v = &mut self.node_bitset_scratch2;
                visited_u.clear();
                visited_v.clear();
                visited_u.set(u, true);
                visited_v.set(v, true);

                let mut a = u;
                let mut b = v;

                while a != b {
                    if self.nodes[a].parent != -1 {
                        a = self.nodes[a].parent as usize;
                        if visited_u[a] {
                            break;
                        }
                        visited_u.set(a, true);

                        path_u.push(a);

                        if visited_v[a] {
                            break;
                        }
                    }
                    if self.nodes[b].parent != -1 && a != b {
                        b = self.nodes[b].parent as usize;
                        if visited_v[b] {
                            break;
                        }
                        visited_v.set(b, true);

                        path_v.push(b);

                        if visited_u[b] {
                            break;
                        }
                    }
                }

                let lca = *path_u.iter().rev().find(|x| path_v.contains(x)).unwrap();
                while path_u.last() != Some(&lca) {
                    path_u.pop();
                }
                while path_v.last() != Some(&lca) {
                    path_v.pop();
                }
                path_v.pop(); // avoid repeating lca

                path_v.reverse();
                path_u.extend(path_v);
                cycles.push(path_u);
            }
        }

        cycles
    }

    pub fn node_connected_component(&mut self, v: usize) -> Vec<usize> {
        let mut stack = vec![v];
        let mut visited = IntSet::from_iter([v]);
        while let Some(node) = stack.pop() {
            for &neighbor in self.nodes[node].adj.iter() {
                if visited.insert(neighbor) {
                    stack.push(neighbor);
                }
            }
        }
        visited.into_iter().collect()
    }

    pub fn node_connected_component_bitset(&mut self, v: usize) -> FixedBitSet {
        let stack = &mut self.vec_scratch_stack;
        let visited = &mut self.node_bitset_scratch0;

        stack.clear();
        visited.clear();

        stack.push(v);
        visited.insert(v);

        while let Some(node) = stack.pop() {
            stack.extend(
                self.nodes[node]
                    .adj
                    .iter()
                    .filter(|&v| !visited.put(*v))
                    .copied(),
            )
        }

        visited.clone()
    }

    pub fn num_connected_components(&mut self) -> usize {
        (0..self.n)
            .filter(|&i| self.nodes[i].parent == -1 && !self.is_isolated(i))
            .count()
    }

    pub fn connected_components(&mut self) -> Vec<Vec<usize>> {
        let roots: Vec<_> = (0..self.n)
            .filter(|&i| self.nodes[i].parent == -1 && !self.is_isolated(i))
            .collect();
        roots
            .into_iter()
            .map(|i| self.node_connected_component(i))
            .collect()
    }

    pub fn active_nodes_vec(&mut self) -> Vec<usize> {
        (0..self.n).filter(|&i| !self.is_isolated(i)).collect()
    }

    pub fn active_nodes_set(&mut self) -> IntSet<usize> {
        let mut active_nodes =
            IntSet::with_capacity_and_hasher(self.n, BuildNoHashHasher::default());
        for i in 0..self.n {
            if !self.is_isolated(i) {
                active_nodes.insert(i);
            }
        }
        active_nodes
    }

    pub fn active_nodes_bitset(&mut self) -> FixedBitSet {
        let mut active_nodes = FixedBitSet::with_capacity(self.n);
        for i in 0..self.n {
            if !self.is_isolated(i) {
                active_nodes.insert(i);
            }
        }
        active_nodes
    }

    pub fn isolate_node(&mut self, v: usize) {
        self.nodes[v].adj.clone().iter().for_each(|neighbor| {
            self.delete_edge(v, *neighbor);
        });
    }

    pub fn isolate_nodes(&mut self, nodes: Vec<usize>) {
        nodes.iter().for_each(|&v| self.isolate_node(v));
    }

    pub fn is_isolated(&mut self, v: usize) -> bool {
        self.nodes[v].adj.is_empty()
    }

    pub fn degree(&mut self, v: usize) -> i32 {
        self.nodes[v].adj.len() as i32
    }

    pub fn neighbors(&mut self, v: usize) -> Vec<usize> {
        self.nodes[v].adj.iter().cloned().collect()
    }

    pub fn neighbors_smallvec(&mut self, v: usize) -> SmallVec<[usize; 4]> {
        self.nodes[v].adj.clone()
    }

    pub fn retain_active_nodes_from(&mut self, from_indices: Vec<usize>) -> Vec<usize> {
        from_indices
            .into_iter()
            .filter(|&neighbor| !self.is_isolated(neighbor))
            .collect()
    }

    /// Returns the shortest path from `start` to `target` in the undirected graph,
    /// using idtree adjacency graph.
    ///
    /// The path is returned as a vector of node indices from `start` to `target`,
    /// inclusive. If no path exists, returns `None`.
    pub fn shortest_path(&mut self, start: usize, target: usize) -> Option<Vec<usize>> {
        if start >= self.n || target >= self.n {
            return None;
        }
        if start == target {
            return Some(vec![start]);
        }

        let queue = &mut self.deque_scratch;
        queue.clear();

        let parents = &mut self.node_vec_scratch;
        let visited = &mut self.distance_generations;
        self.current_distance_generation += 1;

        queue.push_back(start);
        visited[start] = self.current_distance_generation;
        parents[start] = usize::MAX;

        let mut found = false;
        while let Some(u) = queue.pop_front() {
            if u == target {
                found = true;
                break;
            }

            for &v in &self.nodes[u].adj {
                if visited[v] != self.current_distance_generation {
                    visited[v] = self.current_distance_generation;
                    parents[v] = u;
                    queue.push_back(v);
                }
            }
        }

        if !found {
            return None;
        }

        let mut path = Vec::with_capacity(32);
        let mut current = target;
        while current != usize::MAX {
            path.push(current);
            current = parents[current];
        }
        path.reverse();
        Some(path)
    }

    /// Computes betweenness for candidate nodes via idtree adjacency graph.
    ///
    /// NOTE: This is an undirected, unweighted betweenness result.
    pub fn compute_subset_betweenness(
        &mut self,
        removal_candidates: &[(usize, usize)],
        affected_terminals: &RapidHashSet<(usize, usize)>,
        affected_base_towns: &IntSet<usize>,
    ) -> IntMap<usize, usize> {
        if removal_candidates.is_empty() || affected_terminals.is_empty() {
            return removal_candidates.iter().map(|&(v, _)| (v, 0)).collect();
        }

        // Group terminals by root
        let mut root_to_terminals: IntMap<usize, SmallVec<[usize; 16]>> = IntMap::default();
        for &(terminal, pair_root) in affected_terminals {
            root_to_terminals
                .entry(pair_root)
                .or_default()
                .push(terminal);
        }

        let num_terminals = affected_terminals.len();
        let num_roots = root_to_terminals.len();
        let num_candidates = removal_candidates.len();

        // Decision: grouped is cheaper if #roots + #candidates < #terminals
        // TODO: Validate this threshold on a larger variety of test instances.
        let use_grouped = (num_roots + num_candidates) < num_terminals;
        if use_grouped {
            if num_terminals < num_candidates {
                self.compute_subset_betweenness_grouped_terminal_centric(
                    removal_candidates,
                    root_to_terminals,
                    affected_base_towns,
                )
            } else {
                self.compute_subset_betweenness_grouped_candidate_centric(
                    removal_candidates,
                    root_to_terminals,
                    affected_base_towns,
                )
            }
        } else {
            self.compute_subset_betweenness_pairwise(
                removal_candidates,
                root_to_terminals,
                affected_base_towns,
            )
        }
    }

    /// Betweenness via idtree adjacency graph using BFS per pair.
    fn compute_subset_betweenness_pairwise(
        &mut self,
        removal_candidates: &[(usize, usize)],
        root_to_terminals: IntMap<usize, SmallVec<[usize; 16]>>,
        affected_base_towns: &IntSet<usize>,
    ) -> IntMap<usize, usize> {
        use crate::node_router::SUPER_ROOT;

        let mut index_to_betweenness: IntMap<usize, usize> =
            removal_candidates.iter().map(|&(v, _)| (v, 0)).collect();

        for (pair_root, terminals_for_root) in root_to_terminals {
            if pair_root == SUPER_ROOT {
                // Accumulate all paths to each base town for the super terminal
                for terminal in terminals_for_root {
                    for &base_town in affected_base_towns {
                        if let Some(path) = self.shortest_path(terminal, base_town) {
                            for &node in &path {
                                if let Some(count) = index_to_betweenness.get_mut(&node) {
                                    *count += 1;
                                }
                            }
                        }
                    }
                }
            } else {
                for terminal in terminals_for_root {
                    if let Some(path) = self.shortest_path(pair_root, terminal) {
                        for &node in &path {
                            if let Some(count) = index_to_betweenness.get_mut(&node) {
                                *count += 1;
                            }
                        }
                    }
                }
            }
        }

        index_to_betweenness
    }

    /// Betweenness via idtree adjacency graph using triangle equality via BFS per root
    /// and per removal candidate.
    fn compute_subset_betweenness_grouped_candidate_centric(
        &mut self,
        removal_candidates: &[(usize, usize)],
        mut root_to_terminals: IntMap<usize, SmallVec<[usize; 16]>>,
        _affected_base_towns: &IntSet<usize>,
    ) -> IntMap<usize, usize> {
        use crate::node_router::SUPER_ROOT;

        let mut betweenness_counts = vec![0usize; self.n];

        let mut candidate_filter = vec![false; self.n];
        for &(candidate_index, _) in removal_candidates {
            candidate_filter[candidate_index] = true;
        }

        let _super_root_terminals = root_to_terminals.remove(&SUPER_ROOT);

        // Phase 1: Cache distances from each root.
        let mut dist_from_root_cache: IntMap<usize, Vec<i32>> = IntMap::default();
        for &pair_root in root_to_terminals.keys() {
            self.compute_distances_from_internal(pair_root);
            dist_from_root_cache.insert(pair_root, self.distances.clone());
        }

        // Phase 2: Triangle Equality Check.
        for &(candidate, _) in removal_candidates {
            self.compute_distances_from_internal(candidate);
            let mut current_candidate_betweenness = 0;

            for (&pair_root, terminals) in &root_to_terminals {
                let distances_from_root = &dist_from_root_cache[&pair_root];
                let distance_root_to_candidate = distances_from_root[candidate];

                // If the root cannot reach the candidate, it cannot be on a path to terminals.
                if distance_root_to_candidate < 0 {
                    continue;
                }

                for &terminal in terminals {
                    let distance_root_to_terminal = distances_from_root[terminal];
                    let distance_candidate_to_terminal = self.distances[terminal];

                    // Check if candidate is on the shortest path between root and terminal.
                    if distance_root_to_terminal >= 0
                        && self.distance_generations[terminal] == self.current_distance_generation
                        && distance_root_to_terminal
                            == distance_root_to_candidate + distance_candidate_to_terminal
                    {
                        current_candidate_betweenness += 1;
                    }
                }
            }
            betweenness_counts[candidate] = current_candidate_betweenness;
        }

        removal_candidates
            .iter()
            .map(|&(v, _)| (v, betweenness_counts[v]))
            .collect()
    }

    fn compute_subset_betweenness_grouped_terminal_centric(
        &mut self,
        removal_candidates: &[(usize, usize)],
        mut root_to_terminals: IntMap<usize, SmallVec<[usize; 16]>>,
        _affected_base_towns: &IntSet<usize>,
    ) -> IntMap<usize, usize> {
        use crate::node_router::SUPER_ROOT;

        let mut betweenness_counts = vec![0usize; self.n];

        let mut candidate_filter = vec![false; self.n];
        for &(candidate_index, _) in removal_candidates {
            candidate_filter[candidate_index] = true;
        }

        let _super_root_terminals = root_to_terminals.remove(&SUPER_ROOT);

        // Phase 1: Cache distances from each root.
        let mut dist_from_root_cache: IntMap<usize, Vec<i32>> = IntMap::default();
        for &pair_root in root_to_terminals.keys() {
            self.compute_distances_from_internal(pair_root);
            dist_from_root_cache.insert(pair_root, self.distances.clone());
        }

        // Phase 2: Triangle Equality Check (Terminal-Centric Inversion).
        // Instead of BFS per candidate, we BFS once per unique terminal.
        for (&pair_root, terminals) in &root_to_terminals {
            let distances_from_root = &dist_from_root_cache[&pair_root];

            for &terminal in terminals {
                let distance_root_to_terminal = distances_from_root[terminal];

                if distance_root_to_terminal < 0 {
                    continue;
                }

                self.compute_distances_from_internal(terminal);

                for &(candidate, _) in removal_candidates {
                    let distance_root_to_candidate = distances_from_root[candidate];
                    let distance_candidate_to_terminal = self.distances[candidate];

                    if distance_root_to_candidate >= 0
                        && self.distance_generations[candidate] == self.current_distance_generation
                        && distance_root_to_terminal
                            == distance_root_to_candidate + distance_candidate_to_terminal
                    {
                        betweenness_counts[candidate] += 1;
                    }
                }
            }
        }

        removal_candidates
            .iter()
            .map(|&(v, _)| (v, betweenness_counts[v]))
            .collect()
    }

    /// Internal helper to populate distance and generation arrays for a source node.
    fn compute_distances_from_internal(&mut self, source: usize) {
        self.current_distance_generation += 1;
        if self.current_distance_generation == 0 {
            self.distance_generations.fill(0);
            self.current_distance_generation = 1;
        }

        let queue = &mut self.deque_scratch;
        queue.clear();

        self.distances[source] = 0;
        self.distance_generations[source] = self.current_distance_generation;
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            let distance_to_u = self.distances[u];
            for &v in &self.nodes[u].adj {
                if self.distance_generations[v] != self.current_distance_generation {
                    self.distance_generations[v] = self.current_distance_generation;
                    self.distances[v] = distance_to_u + 1;
                    queue.push_back(v);
                }
            }
        }
    }
}

impl IDTree {
    pub(crate) fn new(adj_dict: &IntMap<usize, IntSet<usize>>) -> Self {
        Self::setup(adj_dict)
    }

    fn setup(adj_dict: &IntMap<usize, IntSet<usize>>) -> Self {
        let n = adj_dict.len();
        let nodes: Vec<Node> = (0..n)
            .map(|i| {
                let mut node = Node::new();
                for &j in adj_dict.get(&i).unwrap_or(&IntSet::default()) {
                    node.insert_adj(j);
                }
                node
            })
            .collect();
        Self {
            n,
            nodes,
            distance_generations: vec![0; n],
            distances: vec![0; n],
            current_distance_generation: 0,
            deque_scratch: VecDeque::with_capacity(n),
            node_vec_scratch: vec![0; n],
            vec_bool_scratch: vec![false; n],
            vec_scratch_stack: vec![],
            node_bitset_scratch0: FixedBitSet::with_capacity(n),
            node_bitset_scratch1: FixedBitSet::with_capacity(n),
            node_bitset_scratch2: FixedBitSet::with_capacity(n),
        }
    }

    // Used via the PyO3 binding
    #[cfg(feature = "python")]
    fn initialize(&mut self) {
        for &node_index in self.sort_nodes_by_degree().iter() {
            if self.vec_bool_scratch[node_index] {
                continue;
            }
            self.bfs_setup_subtrees(node_index);

            if let Some(centroid_node) = self.find_centroid_in_q() {
                self.reroot(centroid_node);
            }
        }
        self.vec_bool_scratch.fill(false);
    }

    #[cfg(feature = "python")]
    fn sort_nodes_by_degree(&self) -> Vec<usize> {
        // Sort nodes by degree in descending order.
        let mut node_indices: Vec<usize> = (0..self.n).collect();
        node_indices
            .sort_unstable_by(|&a, &b| self.nodes[b].adj.len().cmp(&self.nodes[a].adj.len()));
        node_indices
    }

    #[cfg(feature = "python")]
    fn bfs_setup_subtrees(&mut self, root: usize) {
        use std::collections::VecDeque;

        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(root);

        self.vec_scratch_stack.clear();
        self.vec_scratch_stack.push(root);
        self.vec_bool_scratch[root] = true;

        while let Some(node_index) = queue.pop_front() {
            for j in 0..self.nodes[node_index].adj.len() {
                let neighbor_index = self.nodes[node_index].adj[j];
                if !self.vec_bool_scratch[neighbor_index] {
                    self.vec_bool_scratch[neighbor_index] = true;
                    self.nodes[neighbor_index].parent = node_index as i32;
                    self.vec_scratch_stack.push(neighbor_index);
                    queue.push_back(neighbor_index);
                }
            }
        }

        // Propagate subtree sizes up the tree, skipping the root
        for &child_index in self.vec_scratch_stack.iter().skip(1).rev() {
            let parent_index = self.nodes[child_index].parent as usize;
            self.nodes[parent_index].subtree_size += self.nodes[child_index].subtree_size;
        }
    }

    #[cfg(feature = "python")]
    fn find_centroid_in_q(&self) -> Option<usize> {
        let num_nodes = self.vec_scratch_stack.len();
        let half_num_nodes = num_nodes / 2;

        self.vec_scratch_stack.iter().rev().find_map(|&node_index| {
            if self.nodes[node_index].subtree_size > half_num_nodes {
                Some(node_index)
            } else {
                None
            }
        })
    }

    fn insert_edge_in_graph(&mut self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n || u == v {
            return false;
        }
        self.nodes[u].insert_adj(v);
        self.nodes[v].insert_adj(u);
        true
    }

    fn insert_edge_balanced(&mut self, mut u: usize, mut v: usize) -> i32 {
        // Algorithm 1: ID-Insert

        let (mut root_u, mut root_v, mut p, mut pp);

        // 1 𝑟𝑜𝑜𝑡𝑢 ← compute the root of 𝑢;
        root_u = u;
        while self.nodes[root_u].parent != -1 {
            root_u = self.nodes[root_u].parent as usize;
        }
        // 2 𝑟𝑜𝑜𝑡𝑣 ← compute the root of 𝑣;
        root_v = v;
        while self.nodes[root_v].parent != -1 {
            root_v = self.nodes[root_v].parent as usize;
        }

        //  /* non-tree edge insertion */
        // 3 if 𝑟𝑜𝑜𝑡𝑢 = 𝑟𝑜𝑜𝑡𝑣 then
        if root_u == root_v {
            let mut reshape = false;
            let mut depth = 0;
            p = self.nodes[u].parent;
            pp = self.nodes[v].parent;

            // 4 if 𝑑𝑒𝑝𝑡ℎ(𝑢) < 𝑑𝑒𝑝𝑡ℎ(𝑣) then swap(𝑢,𝑣);
            while depth < self.n {
                if p == -1 {
                    if pp != -1 && self.nodes[pp as usize].parent == -1 {
                        std::mem::swap(&mut u, &mut v);
                        std::mem::swap(&mut p, &mut pp);
                        reshape = true;
                    }
                    break;
                } else if pp == -1 {
                    if p == -1 && self.nodes[p as usize].parent == -1 {
                        reshape = true;
                    }
                    break;
                }
                p = self.nodes[p as usize].parent;
                pp = self.nodes[pp as usize].parent;
                depth += 1;
            }

            if reshape {
                // Find new centroid...
                // depth u is greater than or equal to depth v from step 4
                // p and pp are at depth v; count levels to depth u for difference from depth v
                // for 1 ≤ 𝑖 < (𝑑𝑒𝑝𝑡ℎ(𝑢)−𝑑𝑒𝑝𝑡ℎ(𝑣))/2
                let mut w = p;
                depth = 0;
                while w != -1 {
                    depth += 1;
                    w = self.nodes[w as usize].parent;
                }
                if depth <= 1 {
                    return 0;
                }
                // split depth in half and set w to the split point
                depth = depth / 2 - 1;
                w = u as i32;
                while depth > 0 {
                    w = self.nodes[w as usize].parent;
                    depth -= 1;
                }

                // 9 Unlink(𝑤);
                let (root_v, _subtree_u_size) = self.unlink(w as usize, v);

                // 10 Link(ReRoot(𝑢),𝑣,𝑟𝑜𝑜𝑡𝑣);
                self.reroot(u);
                if let Some(new_root) = self.link_non_tree_edge(u, v, root_v)
                    && new_root != root_v
                {
                    self.reroot(new_root);
                }
            }

            // 11 return;
            return 0;
        }

        // /* tree edge insertion */
        // 12 if 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑟𝑜𝑜𝑡𝑢) > 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑟𝑜𝑜𝑡𝑣) then
        if self.nodes[root_u].subtree_size > self.nodes[root_v].subtree_size {
            // 13 swap(𝑢,𝑣);
            std::mem::swap(&mut u, &mut v);
            // 14 swap(𝑟𝑜𝑜𝑡𝑢,𝑟𝑜𝑜𝑡𝑣);
            std::mem::swap(&mut root_u, &mut root_v);
        }

        // 15 Link(ReRoot(𝑢),𝑣,𝑟𝑜𝑜𝑡𝑣);
        self.reroot_tree_edge(u, v);
        if let Some(new_root) = self.link_tree_edge(root_u, v, root_v)
            && new_root != root_v
        {
            self.reroot(new_root);
        }
        1
    }

    fn delete_edge_in_graph(&mut self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n || u == v {
            return false;
        }
        self.nodes[u].delete_adj(v);
        self.nodes[v].delete_adj(u);
        true
    }

    fn delete_edge_balanced(&mut self, mut u: usize, mut v: usize) -> i32 {
        // 1 if 𝑝𝑎𝑟𝑒𝑛𝑡(𝑢) ≠ 𝑣 ∧ 𝑝𝑎𝑟𝑒𝑛𝑡(𝑣) ≠ 𝑢 then return;
        if (self.nodes[u].parent != v as i32 && self.nodes[v].parent != u as i32) || u == v {
            return 0;
        }

        // 2 if 𝑝𝑎𝑟𝑒𝑛𝑡(𝑣) = 𝑢 then swap(𝑢,𝑣);
        if self.nodes[v].parent == u as i32 {
            std::mem::swap(&mut u, &mut v);
        }

        // 3 𝑟𝑜𝑜𝑡𝑣 ← Unlink(𝑢);
        let (mut root_v, subtree_u_size) = self.unlink(u, v);

        // 4 if 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑟𝑜𝑜𝑡𝑣) < 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑢) then swap(𝑢,𝑟𝑜𝑜𝑡𝑣);
        if self.nodes[root_v].subtree_size < subtree_u_size {
            std::mem::swap(&mut u, &mut root_v);
        }

        // /* search subtree rooted in 𝑢 */
        if self.find_replacement(u, root_v) {
            return 1;
        }
        2
    }

    fn find_replacement(&mut self, u: usize, root_v: usize) -> bool {
        let nodes = &mut self.nodes;
        let stack = &mut self.vec_scratch_stack;
        let used = &mut self.node_bitset_scratch0;

        // 5 𝑄 ← an empty queue, 𝑄.𝑝𝑢𝑠ℎ(𝑢);
        stack.clear();
        used.clear();

        stack.push(u);
        used.insert(u);

        //  7 while 𝑄 ≠ ∅ do
        while let Some(mut node) = stack.pop() {
            //  9 foreach 𝑦 ∈ 𝑁(𝑥) do
            'neighbors: for &neighbor in nodes[node]
                .adj
                .iter()
                // 10 if 𝑦 = 𝑝𝑎𝑟𝑒𝑛𝑡(𝑥) then continue;
                .filter(|&&n| n != nodes[node].parent as usize)
            {
                // 11 else if 𝑥 = 𝑝𝑎𝑟𝑒𝑛𝑡(𝑦) then
                // 12 𝑄.𝑝𝑢𝑠ℎ(𝑦);
                // 13 𝑆 ← 𝑆 ∪ {𝑦};
                if node as i32 == nodes[neighbor].parent {
                    stack.push(neighbor);
                    used.insert(neighbor);
                    continue;
                }

                // Try to build a new path from y upward
                // 15 𝑠𝑢𝑐𝑐 ← true;
                // 16 foreach 𝑤 from 𝑦 to the root do
                // 17 if 𝑤 ∈ 𝑆 then
                // 18  𝑠𝑢𝑐𝑐 ← false;
                // 19  break
                // 20 else
                // 21  𝑆 ← 𝑆 ∪ {𝑤};
                let mut w = neighbor as i32;
                while w != -1 {
                    if used.put(w as usize) {
                        continue 'neighbors;
                    } else {
                        w = nodes[w as usize].parent;
                    }
                }

                // 22 if 𝑠𝑢𝑐𝑐 then
                // 23   𝑟𝑜𝑜𝑡𝑣 ← Link(ReRoot(𝑥),𝑦,𝑟𝑜𝑜𝑡𝑣);
                // Compute new root => update subtree sizes and find new root
                let mut p = nodes[node].parent;
                nodes[node].parent = neighbor as i32;
                while p != -1 {
                    let pp = nodes[p as usize].parent;
                    nodes[p as usize].parent = node as i32;
                    node = p as usize;
                    p = pp;
                }

                let subtree_u_size = nodes[u].subtree_size;
                let s = (nodes[root_v].subtree_size + subtree_u_size) / 2;
                let mut new_root = None;
                let mut p = neighbor as i32;
                while p != -1 {
                    nodes[p as usize].subtree_size += subtree_u_size;
                    if new_root.is_none() && nodes[p as usize].subtree_size > s {
                        new_root = Some(p as usize);
                    }
                    p = nodes[p as usize].parent;
                }

                // Fix subtree sizes
                let mut p = nodes[node].parent;
                while p != neighbor as i32 {
                    nodes[node].subtree_size -= nodes[p as usize].subtree_size;
                    nodes[p as usize].subtree_size += nodes[node].subtree_size;
                    node = p as usize;
                    p = nodes[p as usize].parent;
                }

                if let Some(new_root) = new_root
                    && new_root != root_v
                {
                    self.reroot(new_root);
                }
                return true;
            }
        }
        false
    }

    fn reroot_tree_edge(&mut self, mut u: usize, v: usize) {
        let mut p = self.nodes[u].parent;
        self.nodes[u].parent = v as i32;
        while p != -1 {
            let temp = self.nodes[p as usize].parent;
            self.nodes[p as usize].parent = u as i32;
            u = p as usize;
            p = temp;
        }
    }

    fn reroot(&mut self, mut u: usize) {
        // - rotates the tree and makes 𝑢 as the new root by updating the parent-child
        //   relationship and the subtree size attribute from 𝑢 to the original root.
        //   The time complexity of ReRoot() is 𝑂(𝑑𝑒𝑝𝑡ℎ(𝑢)).

        // Rotate tree
        // Set parents of nodes between u and the old root.
        let mut p = self.nodes[u].parent;
        let mut pp;
        self.nodes[u].parent = -1;
        while p != -1 {
            pp = self.nodes[p as usize].parent;
            self.nodes[p as usize].parent = u as i32;
            u = p as usize;
            p = pp;
        }

        // Fix subtree sizes of nodes between u and the old root.
        p = self.nodes[u].parent;
        while p != -1 {
            self.nodes[u].subtree_size -= self.nodes[p as usize].subtree_size;
            self.nodes[p as usize].subtree_size += self.nodes[u].subtree_size;
            u = p as usize;
            p = self.nodes[p as usize].parent;
        }
    }

    fn link_non_tree_edge(&mut self, u: usize, v: usize, root_v: usize) -> Option<usize> {
        // Link
        self.nodes[u].parent = v as i32;
        self.link(u, v, root_v)
    }

    fn link_tree_edge(&mut self, u: usize, v: usize, root_v: usize) -> Option<usize> {
        let new_root = self.link(u, v, root_v);

        // Fix subtree sizes between u and the old root
        let mut p = self.nodes[u].parent;
        let mut u = u;
        while p != v as i32 {
            self.nodes[u].subtree_size -= self.nodes[p as usize].subtree_size;
            self.nodes[p as usize].subtree_size += self.nodes[u].subtree_size;
            u = p as usize;
            p = self.nodes[u].parent;
        }

        new_root
    }

    fn link(&mut self, u: usize, v: usize, root_v: usize) -> Option<usize> {
        // - Link(𝑢, 𝑣,𝑟𝑜𝑜𝑡 𝑣) adds a tree 𝑇𝑢 rooted in 𝑢 to the children of 𝑣.
        //     𝑟𝑜𝑜𝑡 𝑣 is the root of 𝑣.
        //     Given that the subtree size of 𝑣 is changed, it updates the subtree size for each
        //     vertex from 𝑣 to the root.
        //     We apply the centroid heuristic by recording the first vertex with a subtree size
        //     larger than 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑟𝑜𝑜𝑡𝑣)/2.
        //     If such a vertex is found, we reroot the tree, and the operator returns the new root.
        //     The time complexity of Link() is 𝑂(𝑑𝑒𝑝𝑡ℎ(𝑣)).

        // Compute new root => update subtree sizes and find new root
        let subtree_u_size = self.nodes[u].subtree_size;
        let s = (self.nodes[root_v].subtree_size + subtree_u_size) / 2;
        let mut new_root = None;
        let mut p = v as i32;
        while p != -1 {
            self.nodes[p as usize].subtree_size += subtree_u_size;
            if new_root.is_none() && self.nodes[p as usize].subtree_size > s {
                new_root = Some(p as usize);
            }
            p = self.nodes[p as usize].parent;
        }
        new_root
    }

    fn unlink(&mut self, u: usize, v: usize) -> (usize, usize) {
        let mut root_v: usize = 0;
        let mut w = v as i32;
        let subtree_u_size = self.nodes[u].subtree_size;
        while w != -1 {
            self.nodes[w as usize].subtree_size -= subtree_u_size;
            root_v = w as usize;
            w = self.nodes[w as usize].parent;
        }
        self.nodes[u].parent = -1;
        (root_v, subtree_u_size)
    }
}
