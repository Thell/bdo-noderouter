// fast_paths.rs

use fast_paths::{FastGraph, PathCalculator};

use crate::node_router::SharedExplorationData;

/// Fast paths calculator
///
/// This is only used for shortest path distances.
///
/// SAFETY: The fastpaths graph does not allow zero weight edges!
///         We scale all edges by a weight factor of 10_000 and give
///         zero weight edges are set to a nominal value of 1.
///
#[derive(Clone, Debug)]
pub struct FastPathsCalc {
    /// Fast graph
    fast_graph: FastGraph,
    reduced_fast_graph: FastGraph,

    /// Fast path calculator against the full reference graph
    calc: PathCalculator,
    /// Fast path calculator against the reduced reference graph
    reduced_calc: PathCalculator,
}

impl FastPathsCalc {
    pub fn new(exploration: SharedExplorationData) -> Self {
        // SAFETY: fast_paths does not allow zero weight edges.
        //
        // Scaling by a weight factor >= max(w(i)) * ∑w(i) allows a nominal weight of 1 to be
        // the reduced 'zero weight' edge cost ensuring the results are correct. Since we know
        //
        //   let weight_factor = ref_graph.edge_weights().sum::<usize>() * max_weight as usize;
        //   debug_assert!(weight_factor > 0);
        //   debug_assert!(weight_factor * (num_nodes - 1) <= u32::MAX as usize);
        //
        // NOTE: This is only used for ref_graph shortest path distances. Since we know the
        // ref graph parameters we use a weight factor of 10_000 which allows accurately
        // scaled ref_graph results to be calculated efficiently.
        let weight_factor = 10_000;

        let ref_graph = &exploration.ref_digraph;
        let reduced_ref_graph = &exploration.reduced_ref_digraph;

        // These empirically determined params provide the fastest query time.
        let params = fast_paths::Params::new(0.1, 200, 200, 200); // default: 0.1, 500, 100, 500

        // Ref Graph -> Fast Graph
        let mut input_graph = fast_paths::InputGraph::new();
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

        let fast_graph = fast_paths::prepare_with_params(&input_graph, &params);
        let calc = fast_paths::create_calculator(&fast_graph);

        // Reduced Ref Graph -> Fast Graph
        let mut reduced_input_graph = fast_paths::InputGraph::new();
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

        let reduced_fast_graph = fast_paths::prepare_with_params(&reduced_input_graph, &params);
        let reduced_calc = fast_paths::create_calculator(&reduced_fast_graph);

        Self {
            fast_graph,
            calc,
            reduced_fast_graph,
            reduced_calc,
        }
    }

    /// Returns the distance between two nodes in the reference graph
    ///
    /// SAFETY: **The path length of ref_graph and reduced_ref_graph will not match
    ///         for any pair containing a leaf node! All other pairs should match.**
    ///
    /// SAFETY: This is a fast path finder using fastpaths and the fast graph which
    ///         uses the same nodes as the reference graph but the edge weights are
    ///         augmented to be * 10_000 the original weight, except for zero cost
    ///         edges which are set to a nominal value of 1.
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
    /// PERFORMANCE: Directly use the `query_fp_distance` method when weights are
    ///              used strictly for relative ordering.
    ///
    /// SAFETY: **The path length of ref_graph and reduced_ref_graph will not match
    ///         for any pair containing reduced/contracted node!**
    ///
    /// SAFETY: This is a fast path finder using fastpaths and the fast graph which
    ///         uses the same nodes as the reference graph but the edge weights are
    ///         augmented to be * 10_000 the original weight, except for zero cost
    ///         edges which are set to a nominal value of 1.
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

    /// Generate a priority queue for the given pairs.
    ///
    /// NOTE: Uses `std::cmp::Reverse`.
    pub fn generate_cutoff_heap(
        &mut self,
        pairs: &[(usize, usize)],
        use_reduced: Option<bool>,
    ) -> std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> {
        use std::cmp::Reverse;
        let mut cutoffs_heap = std::collections::BinaryHeap::new();

        for (i, &(s, t)) in pairs.iter().enumerate() {
            let weight = self.query_fp_ref_graph_distance(s, t, use_reduced);
            cutoffs_heap.push(Reverse((weight, i)));
        }

        cutoffs_heap
    }
}
