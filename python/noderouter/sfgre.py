from __future__ import annotations

import heapq
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations
from time import time

import networkx as nx
import nwst_dw
import rustworkx as rx
from bidict import bidict
from loguru import logger
from rustworkx import PyDiGraph
from sfpgre_cch import get_interactivity_edges

import api_nwst as nwst
from api_common import HYPERNODE_CONTENTS_KEY, INT_INF, PAYLOAD_WEIGHT_KEY
from api_exploration_data import get_exploration_data
from api_nwst_types import (
    BlockCosts,
    BlockedInteractionEdges,
    BlockResults,
    BlockTask,
    ConnectedComponent_BlockKeys,
    Cost,
    CoverageSets,
    NodeIndex,
    SolutionSet,
)
from api_nwstp_problem import TreeProblem
from api_nwstp_solver import solve_tree
from api_rx_pydigraph import subgraph_stable

# When reducing the enclosed steiner clusters they are recursively decomposed
# until the interface count becomes tractable for exact Dreyfus-Wagner.
MAX_ENCLOSED_STEINER_INTERFACES = 6

# 64-bit entropy guarantees a 2^-64 collision probability for randomized 2-edge cut detection
# for finding and splitting 2-edge components.
RANDOM_EDGE_XOR_SEED_BITS = 64


r"""
# Steiner Forest Reduction Grammar (v1.0)

The purpose of this document is to facilitate the communication of various
reductions that can be utilized during pre-processing of a Steiner forest
problem prior to approximation or MIP solving that retains the optimal solution.

In the absence of any super root the utilized graph is a node weighted,
undirected graph with edges represented as bi-directional arcs with no weight.
When a super root is present it is expected that the resulting problem will be
modelled or solved in such a way that a super root is a sink and super terminals
are normal nodes within the graph acting as sources that can satisfy super root
demand via any potential root. As such, a super root only has inbound edges and
potential roots have an outbound edge to the super root. All other connections
are symmetric.

## 1. Symbol Glossary

### Nodes & Sets

| Symbol | Meaning | Notes |
| :--- | :--- | :--- |
| **○** | Neutral Placeholder | Used for general neighbors in abstract rules. |
| **Ⓢ** | Steiner Node | An optional node; no inherent weight/demand. |
| **Ⓣ** | Terminal Node | A node that *must* be connected to its partner. |
| **Ⓡ** | Root Node | The root of an Arborescence (**𝓐**). |
| **𝕡** | Potential Root Node | An individual node member of **𝓟**. |
| **𝕣** | Super Root Node | A super root sink for super terminal demands. |
| **𝕥** | Super Terminal Node | A terminal whose demand is met by any node in **𝔹**. |
| **ⓤ, ⓥ, ⓦ** | Variable Nodes | Generic nodes often consumed in reductions. |
| **⦿** | Hyper-Node | A contracted component or cluster of nodes. |
| **𝓐𐞪** | Arborescence Set | The cluster of nodes rooted at **Ⓡ**. |
| **𝓟** | Potential Roots Set | Set of nodes with edges to a Super Root. |
| **𝓡** | Roots Set | Set of nodes rooting the forest of trees in 𝓐. |
| **𝕊** | Super Root Set | A "sink" for potential root demands. |
| **𝐆** | Graph Set | Active nodes and edges. |
| **𝐃** | Demand Set | Unsatisfied terminal pairs. |
| **𝑻** | Terminal Sets | Unsatisfied `root: {terminal}` sets. |
| **𝓢** | Solution Set | Nodes/Edges selected for the forest. |

### Connectivity & Topology

| Symbol | Meaning | Notes |
| :--- | :--- | :--- |
| **⇋** | Bi-directional Edge | Parallel edges or symmetric relationships. |
| **🡒** | Specific Path | A single, defined path between two nodes. |
| **↠** | Existence of Path | Represents any or all paths between nodes. |
| **↠ⓥ↠** | Existence of Path Traversing Variable Node | Represents any or all such paths between nodes. |
| **↠∣ⓥ∣↠** | Existence of Path Avoiding Variable Node | Represents any or all such paths between nodes. |
| **∣** | Boundary | Absolute boundary. |
| **⁝** | Universe Boundary | The "Tricolon" isolation cut. |
| **⋯** | Continuity | Indicates a path extends further into the graph. |
| **∞** | Infinite Set | The global "Universe" beyond a node. |

### Logic & Operators

| Symbol | Meaning | Role |
| :--- | :--- | :--- |
| **⭆** | Transform | The multi-operation reduction arrow. |
| **:** | Guard Delimiter | "Where / Such that" for local node properties. |
| **⇔** | Invariant | "If and only if" for global/attribute constraints. |
| **∈ / ∉** | Membership | Relative to an Arborescence **𝓐** or Set. |
| **∖ / ⋃** | Set Ops | Exclusion (subtraction) and Union (addition). |
| **𝔀( )** | Weight Function | The cost of a node or path. |
| **( )** | Degree Function | The degree of a node. |

---

## 2. Grammar Structure
Rules follow a "Pattern-First" sequence:

**`Pattern : Local Guards ⭆ Transform ⇔ Global/Attribute Guards`**

---

## Definitions

### Steiner node

Any node which exists in the graph and is not a part of the solution.
In the presence of a super root node specified Steiner nodes may be classified
as a potential root for super terminals.

> Ⓢ ≔ ⓥ ∈ {𝐆 ∖ { 𝓢 ⋃ 𝓟 }}

### Terminal node

Any node which exists in the graph and is a part of the solution.
A specially designated terminal is classified as a root for each terminal
set cluster which will span the arborsence rooted at that specified node.

> Ⓣ ≔ ⓥ ∈ { 𝓢 ∖ 𝓡 }

### Root node

Any node designated as the root of an arborescence (rooted tree) of the
active demand sets as denoted by 𝓐𐞪 excluding any super root.

> Ⓡ ≔ ⓥ ∈ { 𝓡 ∖ 𝕊 }

### Super Terminal (𝕥)

A terminal belonging to a super root’s arbor. Its demand is not bound to a
single fixed root in 𝓡; it is satisfied by reachability to any admissible
sink in 𝓟 ∪ 𝓡 that can reach its super root.

> 𝕥 ∈ 𝓐_𝕊          (reference: |𝕊| = 1)
> 𝕥 ∈ 𝓐_s, s ∈ 𝕊   (general)

### Super Root (𝕣 ∈ 𝕊)

A virtual sink vertex whose arbor 𝓐_𝕣 collects every super terminal assigned
to it. Every potential root in 𝓟 that is eligible for 𝕣 has an outbound edge
to 𝕣; 𝕣 itself has no outbound edges. This lets each super-terminal demand be
treated as ordinary single-sink reachability (𝕥 ↠ 𝕣) during reduction and DP
partitioning, while still allowing any admissible sink in 𝓟 ∪ 𝓡 to serve
that demand.

When |𝕊| = 1 this recovers the single global sink of the reference
implementation; |𝕊| > 1 permits regional super roots.

> 𝕊 ∩ 𝓡 = ∅,  𝕊 ∩ 𝓟 = ∅
> 𝓐_𝕊 ≔ ⋃_{s ∈ 𝕊} 𝓐_𝕣

### Max DRT

The global upper bound on any root–terminal shortest-path distance that can
appear in an optimal forest. It is computed once on the original instance
and never increases under reduction; subsequent local maxima are clipped to it.

> **max_drt ≔ max { 𝔀(Ⓡ ↠ Ⓣ) | (Ⓡ, Ⓣ) ∈ 𝐃 }**
> **∀ subsequent states : max_drt' ≤ max_drt**

Used throughout as a dominance/containment radius (directly, or as 2·max_drt
for collision-envelope certificates); see individual reduction handles for
specific test conditions.

"""


@dataclass
class SFGraphReductionEngine:
    """
    A reduction engine for the Steiner Forest problem.

    Reduces the problem space of the Steiner Forest problem solution for a multi-commodity flow problem
    focusing on reducing the problem graph which reduces MIP model variables, constraints and non-zeros.

    ** All reductions and transformations guarantee that the remaining problem space contains an optimal solution. **

    With that in mind, only certified inclusion/exclusion/redundancy is a consideration.
    """

    instance_id: str
    graph: PyDiGraph
    node_to_key: bidict[int, int]
    super_root_index: int | None = None

    fixed_nodes: set[int] = field(default_factory=set)
    potential_roots: set[int] = field(default_factory=set)
    terminal_sets: CoverageSets = field(default_factory=dict)
    super_candidate_sink_sets: CoverageSets = field(default_factory=dict)

    # Global maximum distance between a root and its' furthest terminal
    _global_min_max_drt: float = float("inf")
    _steiner_distance_prev_max_drt: float = float("inf")

    # solve_as_tree is a witness of a non-reducible graph and should only execute once
    _solved_as_tree: bool = False

    do_debug: bool = False
    call_counts: Counter = field(default_factory=Counter)
    solved_trees: int = 0

    # MARK: Common Helpers

    # Logging/Debugging helpers
    def dump_graph(self, msg: str, graph: PyDiGraph | None = None, override_debug: bool = False):
        if not self.do_debug and not override_debug:
            return
        print(f"=== BEGIN: {msg} ===")
        G = graph if graph is not None else self.graph
        print(f"    num_nodes = {G.num_nodes()}, num_edges = {G.num_edges()}\n")
        terminals = {
            self.get_node_key(t): self.get_node_key(r) for r, ts in self.terminal_sets.items() for t in ts
        }
        edges = [(self.get_node_key(s), self.get_node_key(t)) for s, t in G.edge_list()]
        weights = {self.get_node_key(n): G[n][PAYLOAD_WEIGHT_KEY] for n in G.node_indices()}

        print(f"    terminals = {terminals}")
        print(f"    actual = { {self.get_node_key(n) for n in G.node_indices()} }")
        print(f"    actual_edges = {edges}")
        print(f"    actual_weights = {weights}")
        print(f"=== END: {msg} ===")

    def dump_state(self, msg: str, trigger_count: int = 0):
        self.call_counts[msg] += trigger_count
        logger.debug(f"Post {msg}:  trigger count: {trigger_count}")
        if trigger_count == 0:
            return

        sets = self.terminal_sets
        logger.debug(
            f"  Nodes: {self.graph.num_nodes():4}, Edges: {self.graph.num_edges():4}, Roots: {len(sets):4}, Terminals: {sum(len(ts) for ts in sets.values()):4}"
        )
        sets_wp = {self.get_node_key(r): {self.get_node_key(t) for t in ts} for r, ts in sets.items()}
        logger.debug(f"  Demand sets: {sets_wp}")

    def dump_final_report(
        self,
        reduced_root_pairs_wp: dict[int, int],
        fixed_nodes_wp: set[int],
    ):
        print("=== Post final reduction ===")
        logger.debug(f"  {self.graph.num_nodes()} nodes, {self.graph.num_edges()} edges")
        logger.debug(f"  {len(reduced_root_pairs_wp)} pairs")
        logger.trace(f"  {reduced_root_pairs_wp=}")
        logger.trace(f"  {fixed_nodes_wp=}")
        logger.trace(f"  num components: {rx.number_strongly_connected_components(self.graph)}")

        node_weights = [n[PAYLOAD_WEIGHT_KEY] for n in self.graph.nodes()]
        nodes_by_weight = Counter(node_weights)
        nodes_by_weight = sorted(nodes_by_weight.items())
        logger.debug(f"  nodes by weight: {nodes_by_weight}")

        node_degrees = [self.graph.out_degree(n) for n in self.graph.node_indices()]
        nodes_by_degree = Counter(node_degrees)
        nodes_by_degree = sorted(nodes_by_degree.items())
        logger.debug(f"  nodes by degree: {nodes_by_degree}")

        # Breakdown by degree into fixed, basetown, and steiner
        towns = get_exploration_data().towns
        for d, c in nodes_by_degree:
            sub_counter = Counter()
            for i in self.graph.node_indices():
                is_town = self.get_node_key(i) in towns
                if self.graph.out_degree(i) == d:
                    if i in self.fixed_nodes:
                        sub_counter["fixed"] += 1
                    if is_town:
                        sub_counter["basetown"] += 1
                    if i not in self.fixed_nodes and not is_town:
                        sub_counter["steiner"] += 1
            logger.trace(f"    {d}: {sub_counter}")

        logger.trace("=== PATHS ===")
        reduced_root_pairs: dict[int, int] = {}
        for r, comp in self.terminal_sets.items():
            for t in comp:
                reduced_root_pairs[t] = r
        for u, v in reduced_root_pairs.items():
            has_u = self.graph.has_node(u)
            has_v = self.graph.has_node(v)
            has_path = rx.has_path(self.graph, u, v) if has_u and has_v else False
            logger.trace(
                f"  {self.get_node_key(u)} ({has_u}) -> {self.get_node_key(v)} ({has_v}) => {has_path}"
            )

        logger.debug("\n=== REDUCTIONS ===")
        assert self.call_counts
        logger.debug(f"Total reduction trigger count: {sum(self.call_counts.values())}")
        logger.debug("Trigger counts per reduction step:")
        for msg, count in sorted(self.call_counts.items()):
            logger.debug(f"  {msg}: {count}")

        logger.debug("=== STEINERS ===")
        nodes = set(self.graph.node_indices())
        non_steiner_nodes = set(self.non_steiner_nodes())
        steiner_nodes = nodes - non_steiner_nodes
        logger.debug(f"  {len(steiner_nodes)} steiner nodes")

        critical_steiner_nodes = self.critical_steiner_nodes()
        if critical_steiner_nodes:
            logger.warning(f"  {len(critical_steiner_nodes)} critical steiner nodes")
            logger.warning(f"  {critical_steiner_nodes=}")

    def dump_reduction_results(
        self,
        msg: str,
        num_edges_start: int,
        num_nodes_start: int,
        num_terminal_roots_start: int,
        num_terminals_start: int,
    ):
        # Reduction percentages
        num_edges_end = self.graph.num_edges()
        num_nodes_end = self.graph.num_nodes()
        num_terminal_roots_end = len(self.terminal_sets)
        num_terminals_end = sum(len(ts) for ts in self.terminal_sets.values()) + num_terminal_roots_end
        per_edges = (num_edges_end - num_edges_start) / num_edges_start
        per_nodes = (num_nodes_end - num_nodes_start) / num_nodes_start
        per_terminals = (num_terminals_end - num_terminals_start) / num_terminals_start
        per_roots = (num_terminal_roots_end - num_terminal_roots_start) / num_terminal_roots_start
        num_super_terminals = (
            0 if self.super_root_index is None else len(self.terminal_sets.get(self.super_root_index, []))
        )
        print(
            f"  {msg}: Reduction Percentages: Edges ({num_edges_start} -> {num_edges_end}): {per_edges * 100:.2f}%, Nodes ({num_nodes_start} -> {num_nodes_end}): {per_nodes * 100:.2f}%, Terminals ({num_terminals_start} -> {num_terminals_end}): {per_terminals * 100:.2f}%, Roots ({num_terminal_roots_start} -> {num_terminal_roots_end}): {per_roots * 100:.2f}% ({num_super_terminals} super terminals) [Trees solved: {self.solved_trees}]"
        )

    def critical_steiner_nodes(self):
        """Identifies all Steiner nodes that are critical to the graph."""
        nodes = set(self.graph.node_indices())
        non_steiner_nodes = set(self.non_steiner_nodes())
        steiner_nodes = nodes - non_steiner_nodes
        critical_nodes = set()
        for steiner_node in steiner_nodes:
            # Demand separation test
            tmp = self.graph.copy()
            tmp.remove_node(steiner_node)
            violates_demand = any(
                # Traverse from terminal to root to ensure super terminal reachability
                not rx.has_path(tmp, t, r)
                for r, terminals in self.terminal_sets.items()
                for t in terminals
            )
            if violates_demand:
                critical_nodes.add(steiner_node)
        critical_nodes = {self.get_node_key(n) for n in critical_nodes}
        return critical_nodes

    def get_node_key(self, node: int) -> int:
        """Logging helper."""
        key = self.node_to_key.get(node, None)
        if key is None:
            raise RuntimeError(f"Node {node} not found in index <-> key map.")
        return key

    # Operation helpers
    def get_terminal_set_roots(self, terminals: Iterable[int]) -> list[int]:
        return [next(r for r, ts in self.terminal_sets.items() if t in ts) for t in terminals]

    def non_steiner_nodes(self) -> set[int]:
        """Returns the set of nodes that are not steiner nodes in the graph."""
        all_active = set(self.terminal_sets.keys()) | set().union(*self.terminal_sets.values())

        if self.super_root_index is not None and self.super_root_index in self.terminal_sets:
            potential_roots = {i for i in self.potential_roots if self.graph.has_node(i)}
            return potential_roots | all_active

        return all_active

    def reduction_degree(self, node: int) -> int:
        """Returns the degree of the given node pertaining to reduction semantics."""
        # NOTE: In the node weighted setting all edges are treated as unweighted and undirected
        #       reresentated by bi-directional anti-parallel arcs with the exception of a super root.
        #       If present, the super root has incoming edges only from each potential root
        #       that can satisfy an associated super terminal demand.
        #       As such, for the purposes of graph reduction the SUPER_ROOT degree is zero,
        #       potential roots use in_degree and out_degree is the determinant.
        return min(self.graph.in_degree(node), self.graph.out_degree(node))

    def reduction_neighbors(self, node: int) -> set[int]:
        """Returns the set of neighbors of the given node that are used in the reduction semantics."""
        # NOTE: In the node weighted setting all edges are treated as unweighted and undirected
        #       reresentated by bi-directional anti-parallel arcs with the exception of a super root.
        #       If present, the super root has incoming edges only from each potential root
        #       that can satisfy an associated super terminal demand.
        #       As such, for the purposes of graph reduction neighbors can be considered in the undirected
        #       sense and taken from the predecessor set.
        return set(self.graph.predecessor_indices(node))

    def set_edge_weights(self):
        """Sets the weights of all edges to the weight of the destination node."""
        non_steiner_nodes = self.non_steiner_nodes()
        for u, v in self.graph.edge_list():
            weight = self.graph[v][PAYLOAD_WEIGHT_KEY]
            if v in non_steiner_nodes - self.potential_roots:
                weight = 0
            self.graph.update_edge(u, v, {PAYLOAD_WEIGHT_KEY: weight})

    def validate_reachability(self):
        logger.trace("validate_reachability...")

        all_ts_reachable = True
        for r, terminals in self.terminal_sets.items():
            for t in terminals:
                # NOTE: traverse from terminal to root to ensure super terminal reachability
                if not rx.has_path(self.graph, t, r):
                    logger.error(f"Unreachable ts pair: {self.get_node_key(t)} → {self.get_node_key(r)}")
                    all_ts_reachable = False
        if not all_ts_reachable:
            raise RuntimeError("Unreachable pairs")
        logger.trace("    ...successful")

    def choose_minimax_root(self, candidate_roots: set[int], terminals: set[int]) -> int:
        """
        Given a set of candidate roots and a merged terminal set,
        return the root r that minimizes max_t d(r, t).

        This is the minimax root selection used when trees collide.
        """
        best_root = next(iter(candidate_roots))
        best_value = float("inf")

        for r in candidate_roots:
            # Compute distances from r to all terminals
            # NOTE: since demand endpoints are all zero-weight order does not matter
            #       but since all other r -> t paths are taken as terminal -> root
            #       we do the same here.
            path_lengths = [self.shortest_path_length(t, r) for t in terminals]

            # Compute the maximum distance to terminals
            max_rt = max(path_lengths)
            if max_rt < best_value:
                best_value = max_rt
                best_root = r

        return best_root

    def dw_remaining_complexity(self):
        complexity = 0  # 3^t * n + 2^t * n^2 for Dreyfus Wagner

        num_nodes = self.graph.num_nodes()
        num_roots = len(self.terminal_sets)
        num_terminals = len(set().union(*self.terminal_sets.values()))
        num_super_terminals = (
            len(self.terminal_sets[self.super_root_index]) if self.super_root_index is not None else 0
        )

        for choose_count in range(1, num_roots + 1):
            for comb in combinations(self.terminal_sets.keys(), choose_count):
                num_terms = sum(len(self.terminal_sets[c]) for c in comb) + choose_count
                complexity += (3**num_terms * num_nodes) + (2**num_terms) * (num_nodes**2)

        # TODO: Change back to info level after seeing how well scipstp scales...
        logger.debug(
            f"    remaining complexity: {complexity}, |r|: {num_roots}, |t|: {num_terminals}, |n|: {num_nodes} (st: {num_super_terminals})"
        )

        return complexity

    def shortest_path_length(self, source_node, target_node) -> float:
        """Returns the length of the shortest path from source_node to target_node."""
        path_lengths = rx.dijkstra_shortest_path_lengths(
            self.graph,
            node=source_node,
            edge_cost_fn=lambda e: e[PAYLOAD_WEIGHT_KEY],  # Maps edge head weight
            goal=target_node,
        )
        length = path_lengths[target_node] if target_node in path_lengths else float("inf")
        return length

    def _get_maximum_rt_distance(self) -> float:
        """Calculates and returns the maximum root to terminal distance from shortest paths."""
        max_rt_distance = 1
        for r, terminals in self.terminal_sets.items():
            for t in terminals:
                # Ensure path goes from terminal to root to handle super-root reachability
                max_rt_distance = max(max_rt_distance, self.shortest_path_length(t, r))
        return max_rt_distance

    def _get_maximum_rt_distances(self) -> dict[int, float]:
        """Calculates and returns the maximum root to terminal distance from shortest paths for each arbor."""
        max_rt_distances = defaultdict(float)
        for r, terminals in self.terminal_sets.items():
            max_rt_distances[r] = max(self.shortest_path_length(t, r) for t in terminals)
        return max_rt_distances

    @property
    def min_max_rt_distance(self) -> float:
        """Updates and returns the maximum root to terminal distance."""
        self.set_edge_weights()
        max_rt_distance = min(self._global_min_max_drt, self._get_maximum_rt_distance())
        if max_rt_distance != self._global_min_max_drt and max_rt_distance > 0:
            logger.trace(f"  max_rt_distance changed from {self._global_min_max_drt} to {max_rt_distance}")
            self._global_min_max_drt = int(max_rt_distance)
        logger.trace(f"  max_rt_distance: {max_rt_distance}")

        return max_rt_distance

    @property
    def node_index_map(self) -> dict[int, int]:
        return {u: i for i, u in enumerate(self.graph.node_indices())}

    @property
    def node_weight_map(self) -> dict[int, int]:
        return {u: self.graph[u][PAYLOAD_WEIGHT_KEY] for u in self.graph.node_indices()}

    # MARK: Primitive Terminal Set Actions

    def consume_terminal_set(self, terminal_sets: dict[int, set[int]], r_j: int, r_k: int):
        """𝓐ᵏ consumes 𝓐ʲ, 𝓐ᵏ survives containing all terminals of 𝓐ʲ and Ⓡʲ as a terminal.
        ** 𝓐ʲ ↣ 𝓐ᵏ **
        ```
        𝓐ᵏ ≔ 𝓐ᵏ ∪ 𝓐ʲ ∪ Ⓡʲ
        ```
        """
        if self.do_debug:
            assert r_j in terminal_sets, f"terminal set {self.get_node_key(r_j)} not in terminal sets"
            assert r_k in terminal_sets, f"terminal set {self.get_node_key(r_k)} not in terminal sets"
        terminal_sets[r_j].update(terminal_sets[r_k])
        terminal_sets[r_j].add(r_k)
        del terminal_sets[r_k]

    # MARK: Primitive Graph Actions

    def remove_node(self, u: int):
        """Removes the given node from the graph.
        `G ≔ G ∖ ⓤ`
        """
        if self.do_debug:
            assert self.graph.has_node(u), f"node {self.get_node_key(u)} not in graph"
        self.graph.remove_node(u)

    def remove_nodes_from(self, nodes: Iterable[int]):
        """Removes the given node from the graph.
        `∀ x : x ∈ nodes ⇒ G ≔ G ∖ ⓤ`
        """
        if self.do_debug:
            for x in nodes:
                assert self.graph.has_node(x), f"node {self.get_node_key(x)} not in graph"
        self.graph.remove_nodes_from(nodes)

    def move_edges(self, u: int, v: int):
        """Move all incident edges of ⓤ to ⓥ, avoiding self-loops and parallel edges, isolates ⓤ.
        ```
        ∀ x : (ⓤ ⇋ x ∈ G ∧ x ≠ ⓥ) ⇒ (ⓥ ⇋ x ∈ G), ∣ⓤ∣
        ```
        """
        if self.do_debug:
            assert self.graph.has_node(u), f"node {self.get_node_key(u)} not in graph"
            assert self.graph.has_node(v), f"node {self.get_node_key(v)} not in graph"

        # NOTE: We do not use self.reduction_neighbors here because a node can potentially have edges
        #       to the super root, and this just seems clearer...
        preds = list(self.graph.predecessor_indices(u))
        succs = list(self.graph.successor_indices(u))

        for w in preds:
            self.graph.remove_edge(w, u)
            if w != v and not self.graph.has_edge(w, v):
                self.graph.add_edge(w, v, {})

        for w in succs:
            self.graph.remove_edge(u, w)
            if w != v and not self.graph.has_edge(v, w):
                self.graph.add_edge(v, w, {})

    def absorb(self, u: int, v: int):
        """ⓥ absorbs ⓤ, ⓥ survives combining weights and embedded collapsed nodes.
        **ⓤ ⇴ ⓥ**
        ```
        move_edges(ⓤ, ⓥ)
        𝔀(ⓥ) ≔ 𝔀(ⓥ) + 𝔀(ⓤ)
        ⓥ(⦿) ≔ ⓥ(⦿) ⋃ ⓤ(⦿) ⋃ {ⓤ}
        remove(ⓤ)
        ```
        """
        self.move_edges(u, v)
        self.graph[v][PAYLOAD_WEIGHT_KEY] += self.graph[u][PAYLOAD_WEIGHT_KEY]
        self.graph[v][HYPERNODE_CONTENTS_KEY].update(self.graph[u][HYPERNODE_CONTENTS_KEY])
        self.graph[v][HYPERNODE_CONTENTS_KEY].add(u)
        self.remove_node(u)

    def consume(self, u: int, v: int):
        """ⓥ consumes ⓤ, ⓥ survives.
        **ⓤ ↣ ⓥ**
        ```
        𝓢 ≔ 𝓢 ⋃ ⓤ(⦿) ⋃ {ⓤ}
        move_edges(ⓤ, ⓥ)
        remove(ⓤ)
        ```
        """
        # Absorb ⓤ(⦿) into solution, ignoring the weight - since we are buying it.
        self.fixed_nodes.update(self.graph[u][HYPERNODE_CONTENTS_KEY])
        self.fixed_nodes.add(u)

        self.move_edges(u, v)
        self.remove_node(u)

    # MARK: Reductions

    def reduce_potential_roots(self) -> int:
        """
        Reduces potential roots by state demotion based on candidacy.

        A potential root is only viable if it is a candidate sink for some super terminal,
        per super candidate sink sets.
        If not, it is removed from the set of potential roots and becomes a normal Steiner node.

        > **𝕡 ∈ 𝓟 : ∄ 𝕥 : 𝕡 ∈ candidate_sinks(𝕥) ⭆ 𝓟 ⭆ 𝓟 ∖ {𝕡}, Ⓢ ⭆ Ⓢ ⋃ {𝕡} ⇔ 𝓢 ≡ 𝓢'**
        """
        logger.trace("reduce_potential_roots...")

        # Nothing to do...
        if self.super_root_index is None or self._solved_as_tree:
            return 0

        sr_index = self.super_root_index

        # All super demands are satisfied. Just do cleanup...
        if not self.terminal_sets.get(sr_index):
            if self.terminal_sets.pop(sr_index, None) is not None:
                logger.warning("      reduce_potential_roots: removing super root...")
            self.graph.remove_node(sr_index)
            self.potential_roots = set()
            self.super_root_index = None
            return 0

        logger.trace("reduce_potential_roots...")

        old_roots = set(self.potential_roots)
        num_old_roots = len(self.potential_roots)
        real_roots = set(self.terminal_sets.keys())

        # drt factor is 1 because we are only interested in the single terminal's perspective not arbor collisions
        candidate_sink_sets, _ = self._super_candidate_sink_sets(max_drt_factor=1)
        used_sinks = set().union(*candidate_sink_sets.values()) - {sr_index}

        self.potential_roots = used_sinks | real_roots - {sr_index}

        # This can potentially leave a leaf root node that was demoted to potential root
        # and then further demoted to Steiner node in the graph and in that case there could
        # potentially be a cascading pipeline reduction that doesn't trigger.
        # This is not a correctness issue. (A state could be tracked for all original roots
        # and then we could check if any of them were demoted to a Steiner node but that is
        # out of scope of a pure "potential root" reduction and state tracking for terminals
        # and roots is a performance sink for every reduction when they are written as self
        # contained reference implementations.)
        demoted = old_roots - self.potential_roots - self.fixed_nodes
        if self.do_debug:
            logger.debug(f"    demoted: {sorted(demoted)}")

        # A demoted potential root that is not already a fixed root no longer
        # needs to be connected to the super root.
        for pr in demoted:
            if self.graph.has_edge(pr, sr_index):
                self.graph.remove_edge(pr, sr_index)

        num_removed = num_old_roots - len(self.potential_roots)
        if num_removed > 0:
            logger.warning(f"  reduced potential roots from {num_old_roots} to {len(self.potential_roots)}")
            self.dump_state("reduce_potential_roots", num_removed)
            self.validate_reachability()

        return num_removed

    def _zero_increment_root_merge_groups(self) -> list[set[int]]:
        """Certify real-root groups connected at zero incremental cost.

        Every active demand endpoint is selected in every feasible solution.
        Optional representatives with zero residual payload can be added for
        free. Real demand edges are added as virtual antiparallel arcs because
        feasibility already connects each such class. Therefore, real roots in
        one resulting strongly connected component can be required to share one
        class without increasing cost.

        ``fixed_nodes`` alone is deliberately not a zero-increment certificate:
        an active fixed representative can contain unpaid absorbed hypernodes.

        The directed super-root is excluded: its synthetic inbound arcs encode a
        choice of floating sink, not an ordinary zero-cost connection.
        """
        sr_index = self.super_root_index
        real_roots = {
            root for root, terminals in self.terminal_sets.items() if root != sr_index and terminals
        }
        if len(real_roots) < 2:
            return []

        active_nodes = set(self.graph.node_indices())
        demand_nodes = real_roots | {
            terminal for terminals in self.terminal_sets.values() for terminal in terminals
        }
        zero_increment_nodes = (
            demand_nodes | {node for node in active_nodes if self.graph[node][PAYLOAD_WEIGHT_KEY] == 0}
        ) - {sr_index}

        closure = subgraph_stable(self.graph, zero_increment_nodes)

        # A virtual arc pair represents connectivity already required by a real
        # demand class; it carries no assertion about which paid path realizes it.
        for root, terminals in self.terminal_sets.items():
            if root == sr_index or not terminals:
                continue
            for terminal in terminals:
                if not closure.has_edge(root, terminal):
                    closure.add_edge(root, terminal, {})
                if not closure.has_edge(terminal, root):
                    closure.add_edge(terminal, root, {})

        groups = [set(component) & real_roots for component in rx.strongly_connected_components(closure)]
        groups = [group for group in groups if len(group) > 1]
        return sorted(groups, key=lambda group: tuple(sorted(group)))

    def _merge_zero_increment_root_groups(self, groups: list[set[int]]) -> int:
        """Apply certified root-class merges from one immutable snapshot."""
        merges = 0
        for certified_roots in groups:
            roots = {root for root in certified_roots if self.terminal_sets.get(root)}
            if len(roots) < 2:
                continue

            merged_terminals = set().union(*(self.terminal_sets[root] for root in roots))
            representative = self.choose_minimax_root(roots, merged_terminals)
            merged_terminals.update(roots - {representative})
            self.terminal_sets[representative] = merged_terminals

            for root in roots - {representative}:
                self.terminal_sets.pop(root)

            merges += len(roots) - 1

        return merges

    def reduce_demand_roots(self) -> int:
        """
        Reduces roots by merging arbor sets based on arbor adjacency.

        Handles:
        > **`𝑻 ⭆ 𝑻', |𝑻'𐞪| ≤ |𝑻𐞪|, 𝐃 ≡ 𝐃'`**

        A subset of G consisting of the roots and terminals of all active demands with any
        connecting edges between them is taken and augmented with direct edges between each
        terminal and its root. The resulting connected components then define the reduced
        terminal sets.

        > | 𝐆[𝓡 ⋃ 𝑻] : ∃ (Ⓡ_i ⇋ Ⓡ_j) ∨ ∃ (Ⓣ_i ⇋ Ⓣ_j) ⭆ Ⓡ_i ⇴ Ⓡ_j'

        Super Terminals are treated as independent of their root (since any super terminal
        can be connected to any potential root).

        NOTE: This does not reduce the number of terminal-root pairs. It canonicalizes the root
        side of each pair and adds root->root pairs for collapsed roots which potentially
        increases the number of terminal-root pairs.
        """
        logger.trace("reduce_roots...")

        # NOTE: SUPER_ROOT directly connects to any base-town and any super terminal can
        #       be connected to any base-town, so it is excluded from the fixed set, and
        #       treated separately as isolated nodes during mapping.
        graph = self.graph
        super_root_index = self.super_root_index

        roots = set(self.terminal_sets.keys())
        num_roots = len(roots)
        roots.discard(self.super_root_index)
        tmp_fixed = roots | {t for ts in self.terminal_sets.values() for t in ts}

        self.set_edge_weights()

        # This acts as a union-find data structure for the terminal set clusters.
        sub = subgraph_stable(graph, tmp_fixed)
        for r, ts in self.terminal_sets.items():
            if r == self.super_root_index:
                continue
            for t in ts:
                if self.do_debug and not sub.has_node(r):
                    logger.error(
                        f"Terminal {self.get_node_key(t)} not connected to root {self.get_node_key(r)} missing {self.get_node_key(r)}:"
                    )
                if self.do_debug and not sub.has_node(t):
                    logger.error(
                        f"Root {self.get_node_key(r)} not connected to terminal {self.get_node_key(t)} missing {self.get_node_key(t)}:"
                    )
                sub.add_edge(r, t, {})
                sub.add_edge(t, r, {})
        components = rx.strongly_connected_components(sub)

        terminal_sets: dict[int, set[int]] = {}
        if super_root_index is not None:
            terminal_sets[super_root_index] = set()

        for comp in components:
            comp_roots = set(comp) & roots

            # Collect roots contained in component and select the one that minimizes the maximum d(root, terminal)
            # of the graph.
            rep = next(iter(comp_roots), None)
            if rep is not None:
                comp = set(comp)
                if len(comp_roots) > 1 and len(comp - comp_roots) > 0:
                    merged_terminals = comp - comp_roots
                    rep = self.choose_minimax_root(comp_roots, merged_terminals)
                terminal_sets[rep] = comp - {rep}
            elif super_root_index is not None:
                terminal_sets[super_root_index].update(comp)
            else:
                raise RuntimeError(
                    f"Component has no roots and super root is missing: {[self.get_node_key(n) for n in comp]}"
                )

        if self.super_root_index is not None and terminal_sets[self.super_root_index]:
            # Remove any terminal in the set that has a direct edge to super root
            # which happens when other reductions cascade into the super root.
            orig_set = set(terminal_sets[self.super_root_index])
            terminal_sets[self.super_root_index] = {
                t
                for t in terminal_sets[self.super_root_index]
                if not self.graph.has_edge(t, self.super_root_index)
            }
            for t in orig_set - terminal_sets[self.super_root_index]:
                # We need to ensure that any hyper node contents are added to fixed nodes
                # then remove this node from the terminal set but leave it available to any
                # other routing.
                self.fixed_nodes.update(self.graph[t][HYPERNODE_CONTENTS_KEY])
                self.fixed_nodes.add(t)
                self.graph[t][HYPERNODE_CONTENTS_KEY].clear()
                terminal_sets[self.super_root_index].discard(t)
                logger.trace(f"Removed direct connected super terminal: {self.get_node_key(t)}")

        if self.super_root_index is not None and not terminal_sets[self.super_root_index]:
            logger.warning("      reduce_demand_roots: removing super root...")
            terminal_sets.pop(self.super_root_index)
            self.graph.remove_node(self.super_root_index)
            self.super_root_index = None
            self.potential_roots = set()

        self.terminal_sets = terminal_sets

        zero_increment_groups = self._zero_increment_root_merge_groups()
        zero_increment_merges = self._merge_zero_increment_root_groups(zero_increment_groups)
        if zero_increment_merges:
            self.call_counts["reduce_zero_increment_root_closure"] += zero_increment_merges
            logger.warning(f"  merged {zero_increment_merges} roots through zero-increment closure")

        merges = num_roots - len(self.terminal_sets)

        if merges > 0:
            logger.info(f"  merged {merges} roots")
            if self.do_debug:
                self.dump_state("reduce_demand_roots", merges)
                self.validate_reachability()

        return merges

    def reduce_roots_via_articulation(self) -> int:
        """Reduces roots by merging arbor sets based reachability violations by cut articulation points.

        Handles:
        > | 𝓐_i ↠ ⓥ ↠ 𝑻_i : (ⓥ ∈ Cut(𝐆)) ∧ (𝓐_i ↠∣ⓥ∣↠ 𝑻_i = ∅) ⭆ 𝓐_i ⋃ 𝓐_j ⭆ 𝓐_{ij} ⇔ 𝐃 ≡ 𝐃'

        All arbors that must traverse through a cut articulation point must collide and be merged.
        """
        logger.trace("reduce_via_articulation_points...")

        if self.graph.num_nodes() <= 2:
            return 0

        # --- Begin Undirected Graph ---

        # We only need subgraph for articulation point identification but we need to compress
        # the sparse indices before conversion to undirected.
        self.set_edge_weights()
        sub, node_map = self.graph.subgraph_with_nodemap(
            list(set(self.graph.node_indices()) - {self.super_root_index})
        )
        sub_undir = sub.to_undirected()

        articulations = rx.articulation_points(sub_undir)
        if not articulations:
            return 0

        # Convert back to original indices
        cuts = [node_map[a] for a in articulations]

        # --- End Undirected Graph ---

        merges = 0

        g_prime = self.graph.copy()
        active_roots = set(self.terminal_sets.keys())

        if self.super_root_index is not None:
            g_prime.remove_node(self.super_root_index)
            active_roots.discard(self.super_root_index)

        for cut in cuts:
            if cut == self.super_root_index:
                continue

            g_tmp = g_prime.copy()
            g_tmp.remove_node(cut)
            components = list(rx.strongly_connected_components(g_tmp))

            # Map each component to the set of active roots that have nodes in it
            comp_to_roots: dict[int, set[int]] = {}
            for comp_id, comp_nodes in enumerate(components):
                comp_set = set(comp_nodes)
                for r in active_roots:
                    # A root "touches" this component if root or any terminal is here
                    touches = comp_set & (self.terminal_sets[r] | {r})
                    if touches:
                        comp_to_roots.setdefault(comp_id, set()).add(r)

            # Find groups of roots that are split across multiple components
            root_to_comps: dict[int, set[int]] = {}
            for comp_id, roots_in_comp in comp_to_roots.items():
                for r in roots_in_comp:
                    root_to_comps.setdefault(r, set()).add(comp_id)

            # These trees _must_ collide; meaning we can merge them.
            multi_comp_roots = {r for r, comps in root_to_comps.items() if len(comps) > 1}
            if len(multi_comp_roots) < 2:
                continue

            # Merge them
            to_merge = list(multi_comp_roots)

            merged_ts = set()
            for r in to_merge:
                merged_ts.update(self.terminal_sets[r])

            to_merge = set(to_merge)
            if to_merge:
                rep = self.choose_minimax_root(to_merge, merged_ts)

                # The merged roots become terminals in the new cluster.
                self.terminal_sets[rep] = merged_ts
                self.terminal_sets[rep].update(to_merge)
                self.terminal_sets[rep].discard(rep)

                # Clear out the old roots
                for r in to_merge - {rep}:
                    self.terminal_sets.pop(r, None)
                    active_roots.discard(r)

                if rep in self.terminal_sets[rep]:
                    logger.error(f"Duplicate rep {rep} root: {self.get_node_key(rep)}")

                merges += len(to_merge) - 1

        if merges > 0:
            logger.info(f"  reduced {merges} bottleneck merged roots")
            if self.do_debug:
                self.dump_state("reduce_roots_via_articulation_points", merges)
                self.validate_reachability()

        return merges

    def reduce_adjacent_terminals(self) -> int:
        """Reduces the number of distinct terminals by collapsing adjacent terminals per root.

        NOTE: Terminal_sets must be disjoint by root, which is guaranteed by reduce_demand_roots.

        Handles:
        > **`⁝Ⓣʲ¹ ⇋ Ⓣʲ²⁝  ⭆  ⁝Ⓣʲ¹⁝, 𝐆 ∖ Ⓣʲ², 𝓢 ⋃ Ⓣʲ²(⦿)`**
        > **`⁝Ⓡʲ ⇋ Ⓣʲ⁝  ⭆  ⁝Ⓡʲ⁝, 𝐆 ∖ Ⓣʲ, 𝓢 ⋃ Ⓡʲ(⦿)`**
        """
        logger.trace("reduce_adjacent_terminals...")

        def consume_adjacent_terminals(survivor: int, terminals: set[int]) -> bool:
            # NOTE: rustworkx adjacency queries of node indices not present in the graph return empty sets.
            neighbors = self.reduction_neighbors(survivor)
            consumables = neighbors & terminals
            for c in list(consumables):
                self.consume(c, survivor)
                terminals.discard(c)
            return bool(consumables)

        reduced_terminal_sets: dict[int, set[int]] = {r: set(ts) for r, ts in self.terminal_sets.items()}

        num_terminals = sum(len(ts) for ts in reduced_terminal_sets.values()) + len(reduced_terminal_sets)

        for r, terminals in reduced_terminal_sets.items():
            # Collapse terminal adjacent terminals
            made_progress = True
            while made_progress:
                made_progress = False
                for t in list(terminals):
                    made_progress |= consume_adjacent_terminals(t, terminals)

            # Collapse root adjacent terminals
            if r != self.super_root_index:
                consume_adjacent_terminals(r, terminals)

        # Drop empty sets
        self.terminal_sets = {r: ts for r, ts in reduced_terminal_sets.items() if ts}

        merges = num_terminals - (
            sum(len(ts) for ts in self.terminal_sets.values()) + len(self.terminal_sets)
        )

        if merges > 0:
            logger.info(f"  merged {merges} adjacent terminals")
            if self.do_debug:
                self.dump_state("reduce_adjacent_terminals", merges)
                self.validate_reachability()

        return merges

    def reduce_degree1_steiners(self) -> int:
        """Reduces Steiner nodes by removal based on degree.

        If a Steiner node is a leaf, it cannot satisfy or bridge a demand.

        Handles:
        > **`|Ⓢ ⇋ ○⁝  ⭆  ○⁝, 𝐆 ∖ Ⓢ`**
        """
        logger.trace("reduce_degree1_steiner_nodes...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        removables = []
        # A true steiner node leaf has a strict degree of 1 not just a reduction degree of 1 because
        # a root/potential root demotion to a Steiner node may have a reduction degree of 1 but a true
        # degree of 2 because of the presence of a super root and its removal breaks the directionality
        # of reductions by moving towards the terminals instead of towards the roots.
        while deg1_nodes := {i for i in self.graph.node_indices() if self.graph.out_degree(i) == 1}:
            deg1_nodes.difference_update(non_steiners)
            if not deg1_nodes:
                break
            self.graph.remove_nodes_from(deg1_nodes)
            removables.extend(deg1_nodes)

        removals = len(removables)
        if removals > 0:
            logger.info(f"  removed {removals} degree-1 steiner nodes")
            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(i) for i in removables)}")
                self.dump_state("reduce_degree1_steiner_nodes", removals)
                self.validate_reachability()

        return removals

    def reduce_degree1_terminals(self) -> int:
        """Reduces terminal adjacent Steiner nodes by consumption based on terminal degree.

        If a terminal node is a leaf it must use its neighbor to satisfy a demand.

        Handles:
        > **`|Ⓣ ⇋ Ⓢ⁝  ⭆  Ⓣ ⇋ ⁝, 𝐆 ∖ Ⓢ, 𝓢 ⋃ {Ⓢ, Ⓢ(⦿)}`**
        > **`|𝕥 ⇋ Ⓢ⁝  ⭆  𝕥 ⇋ ⁝, 𝐆 ∖ Ⓢ, 𝓢 ⋃ {Ⓢ, Ⓢ(⦿)}`**

        """
        logger.trace("reduce_degree1_terminals...")

        terminal_sets = self.terminal_sets
        removals = 0

        for ts in terminal_sets.values():
            for t in list(ts):
                if self.reduction_degree(t) != 1:
                    continue

                succs = list(self.graph.successor_indices(t))
                s = succs[0]

                # Don't consume fixed nodes or SUPER_ROOT
                # (adjacent terminals are handled in reduce_adjacent_terminals)
                if s in self.fixed_nodes or s == self.super_root_index:
                    if self.do_debug:
                        roots = set(terminal_sets.keys())
                        logger.debug(
                            f"  ...skipping degree-1 reduction: Root: {s in roots} "
                            f"Super Root: {s == self.super_root_index} "
                            f"Adjacent Terminal: {s in ts} Fixed: {s in self.fixed_nodes}"
                        )
                    continue

                self.consume(s, t)
                removals += 1

        if removals > 0:
            logger.info(f"  removed {removals} degree-1 terminals")
            if self.do_debug:
                self.dump_state("reduce_degree1_terminals", removals)
                self.validate_reachability()

        return removals

    def reduce_degree1_roots(self) -> int:
        """Reduces root adjacent Steiner and terminal nodes by consumption based on root degree.

        If a root node with unsatisfied demand is a leaf it must use its neighbor to satisfy a demand.

        Handles:
        > **`|Ⓡ ⇋ Ⓢ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ Ⓢ, 𝓢 ⋃ {Ⓢ, Ⓢ(⦿)}`**
        > **`|Ⓡᵏ ⇋ Ⓣᵏ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ Ⓣᵏ, 𝓢 ⋃ Ⓣᵏ(⦿), 𝓐ᵏ ∖ Ⓣᵏ`**
        > **`|Ⓡᵏ ⇋ Ⓣʲ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ Ⓣʲ, 𝓢 ⋃ Ⓣʲ(⦿), 𝓐ᵏ ⋃ {𝓐ʲ ∖ Ⓣʲ}`**
        > **`|Ⓡᵏ ⇋ 𝕥ʲ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ 𝕥ʲ, 𝓢 ⋃ 𝕥ʲ(⦿), 𝕊 ∖ 𝕥`**
        """
        logger.trace("reduce_degree1_roots...")

        # NOTE: Potential roots are not considered as roots here because a potential
        #       root may not be part of the optimal solution. If a potential root was
        #       to absorb its neighbor into its hypernode contents then it would only
        #       be able to do so if the neighbor was degree-2 since it could end up
        #       in a path used in the optimal solution after the absorption while
        #       containing hypernode contents thereby making what would have been an
        #       optimal solution no longer optimal.

        terminal_sets = self.terminal_sets

        roots = set(terminal_sets.keys())
        roots.discard(self.super_root_index)
        # Only active roots are considered
        roots = roots - {r for r, ts in terminal_sets.items() if not ts}
        all_terminals = set(terminal_sets.keys()) | roots

        removables = []

        for r_k in list(roots):
            # NOTE: If r_k is the root of a previously consumed terminal set then it is now a terminal.
            if r_k not in terminal_sets:
                continue

            # A root, in the presence of a super_root, can have 2 successors
            # but only ever 1 predecessor when reduction_degree == 1
            if self.reduction_degree(r_k) != 1:
                continue
            neighbor = next(iter(self.reduction_neighbors(r_k)))
            if neighbor in all_terminals:
                # Don't consume adjacent roots or terminals, they are handled during adjacent root reduction
                continue

            self.consume(neighbor, r_k)
            removables.append(neighbor)

            r_j = next((r for r, ts in terminal_sets.items() if neighbor in ts), None)
            if r_j is not None:
                # The neighbor was not a Steiner node, fix terminal sets...
                terminal_sets[r_j].remove(neighbor)
                if r_k != r_j and r_j != self.super_root_index:
                    self.consume_terminal_set(terminal_sets, r_j, r_k)

        removals = len(removables)
        if removals > 0:
            # Drop empty sets
            self.terminal_sets = {r: ts for r, ts in terminal_sets.items() if ts}
            logger.info(f"  removed {removals} degree-1 root Steiner nodes")
            if self.do_debug:
                logger.trace(f"    {[self.get_node_key(i) for i in removables]}")
                self.dump_state("reduce_degree1_roots", removals)
                self.validate_reachability()

        return removals

    def reduce_adjacent_degree2_steiners(self) -> int:
        """Reduces adjacent degree-2 Steiner nodes by absorption.

        If a degree-2 Steiner node is adjacent to another degree-2 Steiner node they
        can only satisfy a demand by bridging both Steiner nodes.

        Handles:
        > **`⁝⇋ Ⓢ¹ ⇋ Ⓢ² ⇋⁝  ⭆  ⁝⇋ Ⓢ¹ ⇋⁝, Ⓢ² ⇴ Ⓢ¹`**

        """
        logger.trace("merge_adjacent_degree2_steiner_chains...")

        non_steiners = self.non_steiner_nodes()
        removables = []

        made_progress = True
        while made_progress:
            made_progress = False

            for node in set(self.graph.node_indices()) - non_steiners:
                # A stale entry will have reduction_degree == 0
                if self.reduction_degree(node) != 2:
                    continue

                neighbor = next(
                    (
                        n
                        for n in self.reduction_neighbors(node)
                        if self.reduction_degree(n) == 2 and n not in non_steiners
                    ),
                    None,
                )
                if neighbor:
                    self.absorb(neighbor, node)
                    # logger.trace(f"    {self.get_node_key(neighbor)} ⇴ {self.get_node_key(node)}")
                    removables.append(neighbor)
                    made_progress = True
                    break

        removals = len(removables)
        if removals > 0:
            logger.info(f"  absorbed {removals} degree-2 Steiner chain nodes")
            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(i) for i in removables)}")
                self.dump_state("merge_adjacent_degree2_steiner_chains", removals)
                self.validate_reachability()

        return removals

    def reduce_degree2_steiners_by_dominance(self) -> int:
        """Reduces Steiner nodes of degree 2 that are dominated by alternate paths.

        A Steiner node v of degree 2 is dominated if there is a path from its adjacent
        node u to its adjacent node v that is shorter than the direct path traversing v.

        Handles:
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝ ⭆  ⁝ⓤ ⇋ ⓥ⁝, 𝐆 ∖ Ⓢ  ⇔ 𝔀(ⓤ ↠∣Ⓢ∣↠ ⓥ) ≤ 𝔀(ⓤ ⇋ Ⓢ ⇋ ⓥ)`**
        """
        logger.trace("reduce_degree2_steiner_dominance...")

        graph = self.graph
        non_steiner_nodes = self.non_steiner_nodes()
        removables = []

        made_progress = True
        while made_progress:
            made_progress = False

            # Each iteration recomputes the edge weights and steiner nodes...
            self.set_edge_weights()
            for node in set(graph.node_indices()) - non_steiner_nodes:
                if self.reduction_degree(node) != 2:
                    continue
                neighbors = list(self.reduction_neighbors(node))
                u, v = neighbors
                if u == v:
                    continue

                try:
                    paths = rx.all_shortest_paths(graph, u, v, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY])
                except rx.NoPathFound:
                    continue

                dominated = any(node not in P for P in paths)
                if not dominated:
                    continue

                self.remove_node(node)

                removables.append(node)
                made_progress = True

                break

        removals = len(removables)
        if removals > 0:
            logger.info(f"  removed {removals} degree-2 dominated nodes")
            if self.do_debug:
                logger.trace(f"    {[self.get_node_key(i) for i in removables]}")
                self.dump_state("reduce_degree2_steiner_dominance", removals)
                self.validate_reachability()

        return removals

    def reduce_degree2_steiners_via_articulation(self) -> int:
        """
        Processes degree-2 articulation point / 2-bridge nodes for inclusion/exclusion.

        If an articulation node's removal violates any demands it must be included in the solution, if
        its removal does not violate any demands and no super terminal would prefer it in an optimal
        solution then it is safe to be excluded from the solution.

        Handles:
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝  ⭆  ⁝ⓤ ∣ ⓥ⁝, 𝐆 ∖ Ⓢ  ⇔  Ⓡ (↠∣Ⓢ∣↠) Ⓣ ∀ 𝐃  ∧  (𝕊 = ∅ ∨ ∀ 𝕥 ∈ 𝕊 : Env(𝕥) ⊆ SCC(𝐆 ∖ Ⓢ, 𝕥))`**
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝  ⭆  ⁝ⓤ ⇋ ⓥ⁝, Ⓢ ↣ ○  ⇔  ∃ 𝐃 : Ⓡ ¬(↠∣Ⓢ∣↠) Ⓣ`**
        """
        logger.trace("reduce_degree2_articulation...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        fixes = 0
        removals = 0

        # Get articulation points
        tmpG = self.graph.copy()
        if self.super_root_index is not None:
            tmpG.remove_node(self.super_root_index)
        tmp_undir = tmpG.to_undirected(multigraph=False)
        articulations = rx.articulation_points(tmp_undir)
        if not articulations:
            return 0

        # Map back to self.graph indices
        tmp_map = {i: u for i, u in enumerate(self.graph.node_indices())}
        articulations = {tmp_map[i] for i in articulations}

        articulations = {u for u in articulations if u not in non_steiners and self.reduction_degree(u) == 2}
        if not articulations:
            return 0

        # Compute collision envelopes
        sr_index = self.super_root_index
        st_collisions: dict[int, set[int]] = {}
        if sr_index is not None:
            radius_cap = 2 * self.min_max_rt_distance
            weight_map = self.node_weight_map
            for st in self.terminal_sets[sr_index]:
                st_collisions[st] = self._collision_envelope(st, non_steiners, weight_map, radius_cap)

        for u in articulations:
            if not self.graph.has_node(u):
                continue

            neighbors = list(self.graph.successor_indices(u))
            if len(neighbors) != 2:
                continue
            a, b = neighbors

            # Demand separation test
            tmp = self.graph.copy()
            tmp.remove_node(u)

            # NOTE: Traverse from terminal to root (for super-root presence)
            violates_demand = any(
                not rx.has_path(tmp, t, r) for r, terminals in self.terminal_sets.items() for t in terminals
            )

            if violates_demand:
                # Must include
                self.consume(u, a if a in self.fixed_nodes else b)
                fixes += 1
                continue

            if sr_index is None:
                continue

            # Super terminal violation test
            components = rx.strongly_connected_components(tmp)
            potential_violation = False
            for st in self.terminal_sets[sr_index]:
                for c in components:
                    if st in c and not st_collisions[st].issubset(set(c)):
                        potential_violation = True
                        break
                if potential_violation:
                    break

            if not potential_violation:
                # Safe to remove
                self.remove_node(u)
                removals += 1
            # else:
            #     # Can not certify because of ambiguity with super terminal optimality
            #     pass

        if removals > 0 or fixes > 0:
            logger.info(f"  fixed {fixes} degree-2 bridges, removed {removals} redundant degree-2 nodes")
            if self.do_debug:
                self.dump_state("reduce_degree2_articulation", removals + fixes)
                self.validate_reachability()

        return removals + fixes

    def reduce_steiner_bridges(self) -> int:
        """Bridge reduction for Steiner Forest.

        A bridge edge (ⓤ, ⓥ) whose removal severs no active demand path is a candidate for
        exclusion. Exclusion is certified without consuming either endpoint only when, of the
        two components the cut produces, at least one is either (a) free of any fixed node, or
        (b) touched by exactly one root and contains that root's entire terminal set — in either
        case the edge was never load-bearing for more than one arborescence. When a super root
        is present, exclusion additionally requires every super terminal's collision envelope to
        stay contained within its own resulting component; the super root's own trivial singleton
        component is never treated as a candidate piece. If the cut does violate an active demand,
        the edge must be traversed in any optimal solution and the Steiner endpoint is consumed
        into the other.

        Handles:
        > **`⁝ⓤ ⇋ ⓥ⁝ : (ⓤ,ⓥ) ∈ Bridges(𝐆), (Ⓢ ∈ {ⓤ,ⓥ})  ⭆  𝐆 ∖ {(ⓤ,ⓥ)}  ⇔  Ⓡ (↠|(ⓤ,ⓥ)|↠) Ⓣ ∀ 𝐃 ∖ 𝕊`**
        > **`  ∧  (𝕊 = ∅ ∨ ∀ 𝕥 ∈ 𝕊 : Env(𝕥) ⊆ SCC(𝐆 ∖ {(ⓤ,ⓥ)}, 𝕥))`**
        > **`  ∧  ∃ 𝓒 ∈ π₀(𝐆 ∖ {(ⓤ,ⓥ)}) ∖ {𝕊} : (𝓒 ∩ 𝓢 = ∅) ∨ (|𝓒 ∩ 𝓡| = 1 ∧ 𝓐ʳ ⊆ 𝓒, r ∈ 𝓒 ∩ 𝓡)`**
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝ : (ⓤ,ⓥ) ∈ Bridges(𝐆)  ⭆  ⁝ⓤ ⇋ ⓥ⁝, Ⓢ ↣ ○  ⇔  ∃ 𝐃 ∖ 𝕊 : Ⓡ ¬(↠|(ⓤ,ⓥ)|↠) Ⓣ`**
        """
        logger.trace("reduce_steiner_bridges...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        removals = 0
        edge_removals = 0

        # Get bridges
        tmpG = self.graph.copy()
        sr_index = self.super_root_index
        if sr_index is not None:
            tmpG.remove_node(sr_index)

        tmp_undir = tmpG.to_undirected(multigraph=False)
        tmp_map = {i: u for i, u in enumerate(self.graph.node_indices())}

        bridges = rx.bridges(tmp_undir)
        if not bridges:
            return 0

        bridges = sorted((tmp_map[min(u, v)], tmp_map[max(u, v)]) for u, v in bridges)  # ty:ignore[invalid-assignment]
        logger.trace(f"  {bridges=}")

        # Precompute super terminal collision envelopes once, up front -- same construction
        # and staleness caveat as reduce_degree2_articulation: computed against the graph
        # state at loop start, so a removal earlier in this same pass can only make a later
        # check MORE conservative (a stale, wider envelope is harder to satisfy issubset
        # against), never unsound. Self-corrects on the pipeline's next iteration.
        st_collisions: dict[int, set[int]] = {}
        if sr_index is not None:
            radius_cap = 2 * self.min_max_rt_distance
            weight_map = self.node_weight_map
            for st in self.terminal_sets[sr_index]:
                st_collisions[st] = self._collision_envelope(st, non_steiners, weight_map, radius_cap)

        for u, v in bridges:
            if not (self.graph.has_node(u) and self.graph.has_node(v)):
                continue

            deg_u = self.reduction_degree(u)
            deg_v = self.reduction_degree(v)
            if not ((deg_u >= 2 and deg_v > 2) or (deg_v >= 2 and deg_u > 2)):
                continue

            steiner_u = u not in non_steiners
            steiner_v = v not in non_steiners
            if not (steiner_u or steiner_v):
                continue

            # Demand separation test
            tmp = self.graph.copy()
            tmp.remove_edge(u, v)
            tmp.remove_edge(v, u)

            violates_demand = any(
                not rx.has_path(tmp, t, r)
                for r, terminals in self.terminal_sets.items()
                for t in terminals
                if r != self.super_root_index
            )

            if not violates_demand:
                sccs = list(rx.strongly_connected_components(tmp))

                if sr_index is not None:
                    # Super terminal collision certification. If every super terminal's full
                    # collision envelope stays inside its own resulting component, this edge
                    # was never on a route any optimal solution needed -- safe to consider for
                    # removal. If any envelope now spans the cut, we can't certify: skip.
                    scc_of: dict[int, int] = {}
                    for scc_i, scc in enumerate(sccs):
                        for n in scc:
                            scc_of[n] = scc_i

                    violates_st = False
                    for st, envelope in st_collisions.items():
                        scc_i = scc_of.get(st)
                        if scc_i is None:
                            continue  # resolved by an earlier iteration of this same pass
                        if not envelope.issubset(sccs[scc_i]):
                            violates_st = True
                            break

                    if violates_st:
                        continue

                    # sr_index's own SCC is always a content-free trivial singleton (pure sink,
                    # never in fixed_nodes) -- it would vacuously satisfy "no fixed nodes" below
                    # regardless of what the real split looks like, so it must not be considered
                    # as one of the candidate pieces.
                    sccs = [scc for scc in sccs if sr_index not in scc]

                # NOTE: We can't certify exclusion of either endpoint of a single bridge because it says nothing
                #       about the two endpoints in an optimal Solution. And in a node weighted problem if both
                #       end up being selected then the edge is naturally selected.
                #       But, if the created component from removing the bridge contains no terminals or roots
                #       or only one arborescence then it is safe to remove the bridge edge.
                #       In that case we can remove the edge but not consume either endpoint.
                for scc in sccs:
                    set_scc = set(scc)

                    if not set_scc & self.fixed_nodes:
                        self.graph.remove_edge(u, v)
                        self.graph.remove_edge(v, u)
                        edge_removals += 1
                        logger.trace(
                            f"  removed bridge edge: no fixed nodes: ({self.get_node_key(u)}, {self.get_node_key(v)})"
                        )
                        break

                    scc_roots = set_scc & set(self.terminal_sets.keys())
                    if len(scc_roots) == 1:
                        scc_root = next(iter(scc_roots))
                        if set_scc & self.terminal_sets[scc_root]:
                            self.graph.remove_edge(u, v)
                            self.graph.remove_edge(v, u)
                            edge_removals += 1
                            logger.trace(
                                f"  removed bridge edge: single root: ({self.get_node_key(u)}, {self.get_node_key(v)})"
                            )
                            break
                continue

            # The edge must be traversed in an optimal Solution...
            if steiner_u:
                self.consume(u, v)
                logger.trace(f"  consumed Steiner bridge node: {self.get_node_key(u)}")
            else:
                self.consume(v, u)
                logger.trace(f"  consumed Steiner bridge node: {self.get_node_key(v)}")
            removals += 1

        if removals > 0 or edge_removals > 0:
            logger.info(
                f"  consumed {removals} Steiner bridge nodes and removed {edge_removals} bridge edges"
            )
            if self.do_debug:
                self.dump_state("reduce_bridge_steiner_consumption", removals + edge_removals)
                self.validate_reachability()

        return removals

    def reduce_degreek_steiner_dominance(self, max_degree: int = 4) -> int:
        """
        Reduces Steiner nodes of small degree k that are dominated by alternate
        paths by removal. Strict dominance is used in the presence of fixed nodes.

        Reduction Rule:
        ```
        Ⓢ : N(Ⓢ) = {○₁, ○₂, …, ○ₖ}, 2 < k ≤ K
        ∧ ∀ {○ᵢ, ○ⱼ} ⊆ N(Ⓢ):
            (○ᵢ ↠∣Ⓢ∣↠ ○ⱼ) exists
            if N(Ⓢ) ∩ 𝓢:
                ∧ 𝔀(○ᵢ ↠∣Ⓢ∣↠ ○ⱼ) < 𝔀(○ᵢ 🡒 Ⓢ 🡒 ○ⱼ)
            else:
                ∧ 𝔀(○ᵢ ↠∣Ⓢ∣↠ ○ⱼ) ≤ 𝔀(○ᵢ 🡒 Ⓢ 🡒 ○ⱼ)

        ⭆  𝐆 ≔ 𝐆 ∖ Ⓢ
        ```
        """

        logger.trace("reduce_degreek_steiner_dominance...")

        self.set_edge_weights()
        graph = self.graph
        non_steiners = self.non_steiner_nodes()
        removals = 0

        def dijkstra_avoiding(start: int, forbidden: int) -> dict[int, float]:
            """Dijkstra that skips a single forbidden node."""

            dist: dict[int, float] = {start: 0.0}
            heap = [(0.0, start)]

            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue

                for v in graph.successor_indices(u):
                    if v == forbidden:
                        continue

                    w = graph.get_edge_data(u, v)[PAYLOAD_WEIGHT_KEY]
                    nd = d + w

                    if v not in dist or nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))

            return dist

        made_progress = True
        while made_progress:
            made_progress = False

            for s in set(graph.node_indices()) - non_steiners:
                deg = graph.out_degree(s)

                if deg <= 2 or deg > max_degree:
                    continue

                neighbors = list(graph.successor_indices(s))
                if len(neighbors) != deg:
                    continue

                dominated = True
                has_fixed_neighbor = set(neighbors) & self.fixed_nodes

                for i in range(len(neighbors)):
                    if not dominated:
                        break

                    u = neighbors[i]

                    # NOTE: Since both dijkstra_shortest_path_lengths and
                    #       dijkstra_avoiding are based on Dijkstra's algorithm, they
                    #       are guaranteed to return the same result.
                    #       (Which omits the initial tail weight.)

                    # Full distances
                    try:
                        dist_full = rx.dijkstra_shortest_path_lengths(
                            graph, u, lambda e: e[PAYLOAD_WEIGHT_KEY]
                        )
                    except Exception:  # noqa: BLE001
                        dominated = False
                        break

                    # Distances avoiding s
                    dist_avoid = dijkstra_avoiding(u, s)

                    for j in range(i + 1, len(neighbors)):
                        v = neighbors[j]

                        if v not in dist_full or v not in dist_avoid:
                            dominated = False
                            break

                        if dist_avoid[v] > dist_full[v] or (
                            dist_avoid[v] == dist_full[v] and has_fixed_neighbor
                        ):
                            dominated = False
                            break

                if not dominated:
                    continue

                self.remove_node(s)
                made_progress = True

                removals += 1
                break

        if removals > 0:
            logger.info(f"  removed {removals} degree-k dominated nodes")
            if self.do_debug:
                self.dump_state("reduce_degreek_steiner_dominance", removals)
                self.validate_reachability()

        return removals

    def reduce_steiner_triangle_degree2_legs(self):
        """Reduces degree-2 Steiner nodes adjacent to degree-3 Steiner triangles by absorption.

        Any degree-2 Steiner node adjacent to a degree-3 Steiner triangle can
        only satisfy a demand by bridging the triangle.

        Handles:
        > **`⁝⋯Ⓢᵃ⇋Ⓢᵘ⇋Ⓢᵇ⋯⁝ ∧ ⋯Ⓢᵃ⇋Ⓢᵇ⋯ ∧ ⋯Ⓢᵘ⇋Ⓢᵛ⇋⁝  ⭆  ⓥ ⇴ ⓤ`**
        """
        logger.trace("reduce_degree3_steiner_triangle_legs...")

        non_steiners = self.non_steiner_nodes()
        removals = 0

        # Single pass iteration...
        d3_steiners = {
            v for v in set(self.graph.node_indices()) - non_steiners if self.reduction_degree(v) == 3
        }
        for node in d3_steiners:
            if self.reduction_degree(node) != 3:
                continue
            neighbors = self.reduction_neighbors(node)
            if neighbors & non_steiners:
                continue

            # One must be degree-2 and the other two must be adjacent to each-other (for the triangle)
            degree2_neighbor = next((n for n in neighbors if self.reduction_degree(n) == 2), None)
            if degree2_neighbor is None:
                continue

            neighbors = neighbors - {degree2_neighbor}
            a, b = neighbors
            if not self.graph.has_edge(a, b):
                continue

            # It is safe to absorb the degree-2 node into the degree-3 triangle
            self.absorb(degree2_neighbor, node)
            removals += 1

        if removals > 0:
            logger.info(f"  removed {removals} degree-3 steiner triangle legs")
            if self.do_debug:
                self.dump_state("reduce_degree3_steiner_triangle_legs", removals)
                self.validate_reachability()

        return removals

    def reduce_blocked_roots(self):
        """Reduces roots whose demands are blocked by strictly one other root's arbor.

        Any root whose demands are blocked by strictly one other root's arbor can
        only satisfy a demand by collision with that other root's arbor.

        Handles:
        > **` ⁝ Ⓣʲ ¬(↠∣𝓐ᵏ∣↠) Ⓡʲ ⁝  ⭆  𝓐ʲ ↣ 𝓐ᵏ`**
        """
        logger.trace("reduce_blocked_roots...")

        def is_blocking_cluster(t: int, r: int, blockers: set[int]) -> bool:
            visited = set()
            stack = [t]

            while stack:
                u = stack.pop()

                if u == r:
                    return False

                if u in visited or u in blockers:
                    continue

                visited.add(u)

                # NOTE: We do not use self.reduction_neighbors here because the super root could potentially
                #       be blocked by another root, and this allows for super-terminal to regular root merges
                stack.extend(self.graph.successor_indices(u))
                stack.extend(self.graph.predecessor_indices(u))

            return True

        terminal_sets = self.terminal_sets
        merges = 0
        made_progress = True

        while made_progress:
            made_progress = False

            # 𝓐𐞪 lookup
            # Blocker-first ordering: larger clusters first
            cluster_nodes = {r: set(ts) | {r} for r, ts in terminal_sets.items()}
            sorted_roots = sorted(terminal_sets.keys(), key=lambda r: len(terminal_sets[r]), reverse=True)

            # Try each root as a potential blocker
            for r_k in sorted_roots:
                if r_k == self.super_root_index:
                    continue

                Ck = cluster_nodes[r_k]

                # Test whether r_j blocks any other root r_k
                for r_j, terminals in terminal_sets.items():
                    if r_j == r_k or r_j == self.super_root_index:
                        continue

                    if any(is_blocking_cluster(t, r_j, Ck) for t in terminals):
                        # Perform merge: 𝓐ʲ ↣ 𝓐ᵏ
                        self.consume_terminal_set(terminal_sets, r_j, r_k)
                        merges += 1
                        made_progress = True
                        break

                if made_progress:
                    break

        # Drop empty sets
        self.terminal_sets = {r: ts for r, ts in terminal_sets.items() if ts}

        if merges > 0:
            logger.info(f"  merged {merges} blocked roots")
            if self.do_debug:
                self.dump_state("reduce_blocked_roots", merges)
                self.validate_reachability()

        return merges

    def reduce_degreek_enclosed_steiner(self) -> int:
        """Reduces Steiner nodes surrounded by degree-2 terminals of
        the same arborescence by consumption.

        When there exists at least one unambiguous outer Steiner endpoint u*
        such that w(e) <= w(u*), then e must be purchased in any optimal
        solution (even if all other outer endpoints are "free" due to
        external sharing).

        Handles:
        > **`⁝Ⓢᵘᵢ ⇋ Ⓣᵏᵢ ⇋ Ⓢᵉ ⇋ Ⓣᵏⱼ ⇋ Ⓢᵘⱼ⁝  ⭆  Ⓢᵉ ↣ Ⓣᵏ  ⇔  ∃ Ⓢᵘ* : 𝔀(Ⓢᵉ) ≤ 𝔀(Ⓢᵘ*)`**

        NOTE: This generalizes both the degree-2 and degree-3 enclosed cases,
              yet is more expensive to compute than the degree-2 case.
        """
        logger.trace("reduce_enclosed_same_root_steiner...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        def is_unambiguous_outer_endpoint(u: int, t: int) -> bool:
            if u in non_steiners or u == self.super_root_index:
                return False

            deg = self.reduction_degree(u)
            if deg == 2:
                return True
            if deg != 3:
                return False

            # Degree 3 triangle case
            neighbors = self.reduction_neighbors(u)
            neighbors.discard(t)

            if len(neighbors) != 2:
                return False

            x, y = neighbors
            if x in non_steiners or y in non_steiners:
                return False

            return self.graph.has_edge(x, y) or self.graph.has_edge(y, x)

        consumptions = 0
        made_progress = True

        while made_progress:
            made_progress = False

            for e in list(self.graph.node_indices()):
                if e == self.super_root_index or e in non_steiners:
                    continue
                if self.reduction_degree(e) < 2:
                    continue

                # Find all degree-2 terminal neighbors of e belonging to same root
                terminal_neighbors = []
                root = None

                for t in list(self.graph.successor_indices(e)):
                    if t not in non_steiners:
                        continue
                    if self.reduction_degree(t) != 2:
                        continue

                    t_root = (
                        t
                        if t in self.terminal_sets
                        else next((r for r, ts in self.terminal_sets.items() if t in ts), None)
                    )
                    if t_root is None:
                        continue

                    if root is None:
                        root = t_root
                    elif t_root != root:
                        root = None
                        break

                    terminal_neighbors.append(t)

                if len(terminal_neighbors) < 2 or root is None:
                    continue

                # Find unambiguous outer endpoints
                outer_endpoints = []
                for t in terminal_neighbors:
                    t_neighbors = self.reduction_neighbors(t)
                    outers = [n for n in t_neighbors if n != e]
                    if len(outers) != 1:
                        # terminal wasn't degree-2
                        continue

                    u = outers[0]
                    if is_unambiguous_outer_endpoint(u, t):
                        outer_endpoints.append(u)

                if not outer_endpoints:
                    continue

                # Certification check: exists at least one u* where w(e) <= w(u*)
                w_e = self.graph[e][PAYLOAD_WEIGHT_KEY]
                for u in outer_endpoints:
                    w_u = self.graph[u][PAYLOAD_WEIGHT_KEY]
                    if w_e <= w_u:
                        # Safe to consume
                        self.consume(e, terminal_neighbors[0])
                        consumptions += 1
                        made_progress = True
                        break

                if made_progress:
                    break

        if consumptions > 0:
            logger.info(f"  consumed {consumptions} enclosed (degree k) same-root Steiners")
            if self.do_debug:
                self.dump_state("reduce_degreek_enclosed_same_root_steiner", consumptions)
                self.validate_reachability()

        return consumptions

    def reduce_enclosed_steiner_clusters(self) -> int:
        """
        Reduces Steiner nodes provably absent from every optimal Steiner tree
        realization over every interface subset of recursively decomposed
        Steiner components by removal.

        NOTE: Large interface components are recursively decomposed across bridge edges
        until the interface count becomes tractable for exact Dreyfus-Wagner certification.

        Handles:
        ```
        ∀ ℂ ∈ π₀(G ∖ S)
        I ≔ N(S) ∩ ℂ
        ∀ X ⊆ I, |X| ≥ 2 :
          Wₓ ∈ OPT_ST(X, ℂ)
        k ≔ ℂ ∖ ( ⋃_{X ⊆ I, |X| ≥ 2} Wₓ )
        G ≔ G ∖ k
        ```
        Lemma 1 (Structural Invariance of Enclosed Steiner Cluster Reduction).
        Let S be the set of all non‑Steiner nodes. For each connected component
        ℂ of G ∖ S, let I ≔ N(S) ∩ ℂ be its structural interface.

        For every interface subset X ⊆ I with |X| ≥ 2, let Wₓ ∈ OPT_ST(X, ℂ)
        be an arbitrary optimal Steiner-tree realization inside ℂ. Define:

            k ≔ ℂ ∖ ( ⋃_{X ⊆ I, |X| ≥ 2} Wₓ )

        Then no element of k is required for the existence of at least one
        globally optimal realization of G. Therefore the reduction:

            G ≔ G ∖ k

        preserves:
        • the exact global optimum objective value, and
        • the existence of at least one globally optimal realization.
        """
        logger.trace("reduce_enclosed_steiner_clusters...")

        self.set_edge_weights()

        non_steiners = self.non_steiner_nodes()
        steiners = set(self.graph.node_indices()) - non_steiners
        non_steiners.discard(self.super_root_index)
        if len(steiners) < 3:
            return 0

        g_prime: PyDiGraph = subgraph_stable(self.graph, steiners)
        interfaces = {i for i in steiners if any(n not in steiners for n in self.reduction_neighbors(i))}
        if len(interfaces) < 2:
            return 0

        # self.dump_graph("reduce_enclosed_steiner_clusters", g_prime)

        # This is an node weighted undirected graph...
        g_steiner_undir = nx.Graph()
        for i in g_prime.node_indices():
            g_steiner_undir.add_node(i, weight=g_prime[i][PAYLOAD_WEIGHT_KEY])
        g_steiner_undir.add_edges_from(g_prime.edge_list())  # ignores multi-edges

        removables = set()

        comps = nx.connected_components(g_steiner_undir)
        for comp_nodes in comps:
            comp = set(comp_nodes)
            comp_interfaces = comp & interfaces
            if len(comp_interfaces) < 2:
                continue

            removable = self._reduce_steiner_cluster_dreyfus_wagner(
                component=comp,
                interfaces=comp_interfaces,
                interface_limit=MAX_ENCLOSED_STEINER_INTERFACES,
            )
            removables.update(removable)

        if removables:
            # self.dump_graph("pre-remove enclosed steiner cluster nodes", self.graph)
            self.graph.remove_nodes_from(removables)
            logger.info(f"  removed {len(removables)} Steiner cluster nodes")
            # self.dump_graph("post-remove enclosed steiner cluster nodes", self.graph)
            if self.do_debug:
                logger.trace(f"  {sorted(self.get_node_key(n) for n in removables)}")
                self.dump_state("reduce_enclosed_steiner_clusters_new", len(removables))
                self.validate_reachability()

        return len(removables)

    def _reduce_steiner_cluster_dreyfus_wagner(
        self,
        component: set[int],
        interfaces: set[int],
        interface_limit: int,
    ) -> set[int]:
        """
        Recursive bounded-interface exact certification.

        Components exceeding the interface limit are recursively split.
        Split endpoints become interfaces in their respective child components.
        """
        # logger.trace("  reduce_steiner_cluster_dreyfus_wagner...")

        k = len(interfaces)
        if k <= 1:
            return set()

        # Recursive decomposition phase
        if k > interface_limit:
            split = self._split_steiner_component(component, interfaces)
            if split is None:
                # No valid decomposition found - refuse certification.
                return set()

            left_nodes, right_nodes, bridges_u, bridges_v = split

            left_interfaces = (interfaces | bridges_u | bridges_v) & left_nodes
            right_interfaces = (interfaces | bridges_u | bridges_v) & right_nodes

            removable = set()
            removable |= self._reduce_steiner_cluster_dreyfus_wagner(
                component=left_nodes,
                interfaces=left_interfaces,
                interface_limit=interface_limit,
            )
            removable |= self._reduce_steiner_cluster_dreyfus_wagner(
                component=right_nodes,
                interfaces=right_interfaces,
                interface_limit=interface_limit,
            )
            return removable

        # Preparation phase for Dreyfus-Wagner algorithm
        local_nodes = list(component)
        adj_map: dict[int, set[int]] = {n: set() for n in local_nodes}
        node_weight_map: dict[int, int] = {}

        # Build adjacency map
        for u in local_nodes:
            nbrs = self.reduction_neighbors(u)
            for v in nbrs:
                if v in component:
                    adj_map[u].add(v)
            node_weight_map[u] = int(self.graph[u][PAYLOAD_WEIGHT_KEY])

        def steiner_tree_nodes(terminals: tuple[int, ...]) -> set[int]:
            """Returns the exact optimal Steiner tree topology nodes."""
            terminals_list = list(terminals)
            # Use the first terminal arbitrarily as the root/source node context
            cost, _d_nodes, solution_nodes = nwst_dw.solve_nwst(
                adj_map, node_weight_map, terminals_list, terminals_list[0]
            )
            if cost == INT_INF:
                return set(terminals)
            return set(solution_nodes)

        # Build KEEP set: nodes that appear in some optimal routing
        keep: set[int] = set()
        interface_list = sorted(interfaces)

        # Steiner trees for all interface subsets up to interface_limit
        for subset_mask in range(1, 1 << len(interface_list)):
            subset = tuple(interface_list[i] for i in range(len(interface_list)) if subset_mask & (1 << i))
            if len(subset) <= interface_limit:
                nodes = steiner_tree_nodes(subset)
                keep.update(nodes)
            else:
                logger.warning(f"    skipping interface subset with {len(subset)} interfaces...")

        # Final component removals: unneeded internal nodes
        return (component - interfaces) - keep

    def _split_steiner_component(
        self,
        component: set[int],
        interfaces: set[int],
    ) -> tuple[set[int], set[int], set[int], set[int]] | None:
        """
        Split a Steiner cluster component into two child components.
        """
        # logger.trace("      split_steiner_component...")

        # If there are less than 6 nodes or less than 2 interfaces, we don't need to split
        if len(component) < 6 or len(interfaces) < 2:
            return None

        result = self._find_and_split_bridge_edge_components(component)
        if result is not None:
            # logger.trace("      found single edge bridge split")
            left_component, right_component, bridge_u, bridge_v = result
            return left_component, right_component, {bridge_u}, {bridge_v}

        rxG = subgraph_stable(self.graph, component)
        result = self._find_and_split_2_edge_components(rxG, PAYLOAD_WEIGHT_KEY)
        if result is None:
            return None

        # logger.trace("      found 2-edge bridge split")

        left_gsub, right_gsub, bridges_u, bridges_v = result
        left_component = set(left_gsub.node_indices())
        right_component = set(right_gsub.node_indices())

        return left_component, right_component, set(bridges_u), set(bridges_v)

    def _find_and_split_bridge_edge_components(
        self, component: set[int]
    ) -> tuple[set[int], set[int], int, int] | None:
        """
        Finds a bridge edge suitable for recursive decomposition.

        Returns:
            (
                left_component,
                right_component,
                bridge_u,
                bridge_v,
            )

        where removing edge (u, v) disconnects the component.
        """

        # Build undirected adjacency.
        adj = {n: self.reduction_neighbors(n) & component for n in component}

        # Naive bridge search: Correctness first.
        for u in component:
            for v in adj[u]:
                if u > v:
                    continue

                # Temporarily remove edge.
                adj[u].remove(v)
                adj[v].remove(u)

                # Connectivity test.
                seen = set()
                stack = [u]

                while stack:
                    x = stack.pop()
                    if x in seen:
                        continue

                    seen.add(x)
                    stack.extend(adj[x] - seen)

                # Restore edge.
                adj[u].add(v)
                adj[v].add(u)

                if len(seen) == len(component):
                    continue

                left = seen
                right = component - seen
                if not left or not right:
                    continue

                return (left, right, u, v)

        return None

    def _find_and_split_2_edge_components(
        self, digraph: rx.PyDiGraph, weight_attr: str = "weight"
    ) -> tuple[rx.PyDiGraph, rx.PyDiGraph, set[int], set[int]] | None:
        """
        Finds 2-edge cuts using index-aligned local subgraphs, which maps endpoints
        back to the original DiGraph space using scalar lookups, and returns two valid
        subgraphs for the re-entrant DP states.
        """
        # 1. Isolate structural connectivity with a tracked NodeMap
        g_sub, node_map = digraph.subgraph_with_nodemap(digraph.node_indices())
        undirected_g = g_sub.to_undirected(multigraph=False)

        # 2. Extract node weights using local subgraph keys
        node_weights = {i: g_sub[i].get(weight_attr, 1) for i in undirected_g.node_indices()}
        total_graph_weight = sum(node_weights.values())

        # 3. Construct local spanning tree
        tree_edges_list = rx.graph_dfs_edges(undirected_g)
        tree_edges = set()
        adj_tree = defaultdict(list)
        for u, v in tree_edges_list:
            edge_key = (min(u, v), max(u, v))
            tree_edges.add(edge_key)
            adj_tree[u].append(v)
            adj_tree[v].append(u)

        all_edges = [(min(u, v), max(u, v)) for u, v in undirected_g.edge_list()]
        non_tree_edges = [e for e in all_edges if e not in tree_edges]

        # 4. Generate random 64-bit structural hashes
        edge_hashes = {}
        node_xor_accumulator = defaultdict(int)
        for edge in non_tree_edges:
            h = random.getrandbits(RANDOM_EDGE_XOR_SEED_BITS)
            edge_hashes[edge] = h
            u, v = edge
            node_xor_accumulator[u] ^= h
            node_xor_accumulator[v] ^= h

        # 5. Accumulate values up the spanning tree
        visited = set()
        tree_edge_hashes = {}
        subtree_weights = {}
        node_parent = {}

        def dfs_accumulate(u, p=None):
            visited.add(u)
            node_parent[u] = p
            current_xor = node_xor_accumulator[u]
            current_weight = node_weights[u]

            for v in adj_tree[u]:
                if v != p and v not in visited:
                    child_xor, child_weight = dfs_accumulate(v, u)
                    edge_key = (min(u, v), max(u, v))
                    tree_edge_hashes[edge_key] = child_xor
                    current_xor ^= child_xor
                    current_weight += child_weight

            subtree_weights[u] = current_weight
            return current_xor, current_weight

        for node in undirected_g.node_indices():
            if node not in visited:
                dfs_accumulate(node)

        edge_hashes.update(tree_edge_hashes)

        # 6. Group by structural hash
        hash_to_edges = defaultdict(list)
        for edge, h in edge_hashes.items():
            if h != 0:
                hash_to_edges[h].append(edge)

        valid_cuts = [edge_list for edge_list in hash_to_edges.values() if len(edge_list) == 2]

        best_left_graph = None
        best_right_graph = None
        best_cut_left_endpoints = None
        best_cut_right_endpoints = None
        min_balance_diff = float("inf")

        # 7. Evaluate and extract cuts using translated original IDs
        for e1, e2 in valid_cuts:
            # FIX: e1 and e2 are tuples of (u_sub, v_sub). Extract and map individual integers!
            u1_sub, v1_sub = e1
            u2_sub, v2_sub = e2

            u1_orig = node_map[u1_sub]
            v1_orig = node_map[v1_sub]
            u2_orig = node_map[u2_sub]
            v2_orig = node_map[v2_sub]

            # Clone original input graph for validation testing
            test_graph = digraph.copy()

            # Identify all matching anti-parallel directed arcs using original node coordinates
            edges_to_remove = []
            for src, dst in [(u1_orig, v1_orig), (v1_orig, u1_orig), (u2_orig, v2_orig), (v2_orig, u2_orig)]:
                if test_graph.has_edge(src, dst):
                    edges_to_remove.append((src, dst))
            test_graph.remove_edges_from(edges_to_remove)

            # Verify if a real structural partition occurred
            components = rx.weakly_connected_components(test_graph)
            if len(components) < 2:
                continue  # Rejects false hash collisions or edge shortcuts

            # Sort components by their aggregate node weight to isolate the two largest parts
            components = sorted(
                components, key=lambda c: sum(digraph[n].get(weight_attr, 1) for n in c), reverse=True
            )

            left_nodes_orig = next(iter(components))
            right_nodes_orig = []
            for other_comp in components[1:]:
                right_nodes_orig.extend(list(other_comp))

            # FIX: Construct subgraphs using node indices from the ORIGINAL graph coordinate space
            # This keeps the internal edge wiring and planarity perfectly intact
            left_graph = subgraph_stable(digraph, left_nodes_orig)
            right_graph = subgraph_stable(digraph, right_nodes_orig)

            # Evaluate balance
            left_weight = sum(digraph[n].get(weight_attr, 1) for n in left_nodes_orig)
            right_weight = total_graph_weight - left_weight
            diff = abs(left_weight - right_weight)

            if diff < min_balance_diff:
                min_balance_diff = diff
                best_left_graph = left_graph
                best_right_graph = right_graph
                best_cut_left_endpoints = {u1_orig, u2_orig}
                best_cut_right_endpoints = {v1_orig, v2_orig}

        if (
            best_left_graph is None
            or best_right_graph is None
            or best_cut_left_endpoints is None
            or best_cut_right_endpoints is None
        ):
            return None

        return best_left_graph, best_right_graph, best_cut_left_endpoints, best_cut_right_endpoints

    def reduce_steiner_nodes_by_distance(self) -> int:
        """Reduces Steiner nodes based on weight or reachability distance bounds by removal.

        If a Steiner node's weight exceeds min_max_drt, or if its minimum distance to
        any terminal exceeds min_max_drt, it is dominated and cannot participate in an
        optimal solution.

        Handles:
            > **` 𝔀(Ⓢ) > min_max_drt  ⭆  𝐆 ≔ 𝐆 ∖ Ⓢ`**
            > **`¬( 𝔀(Ⓢ ↠ Ⓣ) ≤ min_max_drt )  ⭆  𝐆 ≔ 𝐆 ∖ Ⓢ`**
        """
        logger.trace("reduce_steiner_nodes_by_distance...")

        self.set_edge_weights()
        non_steiner_nodes = self.non_steiner_nodes()
        min_max_drt = self.min_max_rt_distance

        # This can be a fairly expensive reduction so if min_max_drt has not changed,
        # we can skip it knowing that it would have only reduced Steiner nodes not
        # roots/terminals which is a lessor impact.
        if min_max_drt >= self._steiner_distance_prev_max_drt:
            return 0
        self._steiner_distance_prev_max_drt = min_max_drt

        steiners = set(self.graph.node_indices()) - non_steiner_nodes - {self.super_root_index}
        terminals = (set(self.terminal_sets.keys()) | set().union(*self.terminal_sets.values())) - {
            self.super_root_index
        }

        removables = []
        removed_by_weight = 0
        removed_by_distance = 0

        for v in steiners:
            if self.graph[v][PAYLOAD_WEIGHT_KEY] > min_max_drt:
                logger.trace(f"  removing Steiner node {self.get_node_key(v)} by weight")
                removables.append(v)
                removed_by_weight += 1
                continue

            # NOTE: Since all edge weights are head node weighted and all t are terminals
            #       with zero weight, t -> v via dijkstra is always the full path cost.
            if all(self.shortest_path_length(t, v) > min_max_drt for t in terminals):
                logger.trace(f"  removing Steiner node {self.get_node_key(v)} by distance")
                removables.append(v)
                removed_by_distance += 1

        if removables:
            self.remove_nodes_from(removables)
            logger.debug(
                f"    removed {len(removables)} Steiner nodes [max_drt: ({min_max_drt}) by weight: {removed_by_weight}, by distance: {removed_by_distance}]"
            )
            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(n) for n in removables)}")
                self.dump_state("reduce_steiner_nodes_by_distance", len(removables))
                self.validate_reachability()

        return len(removables)

    def reduce_steiner_nodes_by_distance_recursive(self) -> int:
        """Reduces Steiner nodes based on weight or reachability distance bounds by removal.

        If a Steiner node's weight exceeds min_max_drt, or if its minimum distance to
        any terminal exceeds min_max_drt, it is dominated and cannot participate in an
        optimal solution. Where the max drt is the max of the nearest reachable arbor
        within the bound of the currently applied max drt.

        Handles:
            > **` 𝔀(Ⓢ) > min_max_drt  ⭆  𝐆 ≔ 𝐆 ∖ Ⓢ`**
            > **`¬( 𝔀(Ⓢ ↠ Ⓣ) ≤ min_max_drt )  ⭆  𝐆 ≔ 𝐆 ∖ Ⓢ`**
        """
        logger.trace("reduce_steiner_nodes_by_distance_recursive...")

        self.set_edge_weights()
        non_steiner_nodes = self.non_steiner_nodes()

        # This can be a fairly expensive reduction so if min_max_drt has not changed,
        # we can skip it knowing that it would have only reduced Steiner nodes not
        # roots/terminals which is a lessor impact.
        global_min_max_drt = self.min_max_rt_distance
        # if global_min_max_drt >= self._steiner_distance_prev_max_drt:
        #     return 0
        # self._steiner_distance_prev_max_drt = global_min_max_drt

        steiners = set(self.graph.node_indices()) - non_steiner_nodes - {self.super_root_index}

        drt_distances = self._get_maximum_rt_distances()
        terminal_to_root_map = {t: r for r, terminals in self.terminal_sets.items() for t in terminals}
        terminal_to_root_map.update({r: r for r in self.terminal_sets})

        effective_terminal_drt = {
            t: min(global_min_max_drt, drt_distances[r]) for t, r in terminal_to_root_map.items()
        }
        # populate the root entries as well, using any of the terminal drts
        effective_terminal_drt.update({
            r: min(global_min_max_drt, drt_distances[r]) for r in self.terminal_sets
        })

        def is_safe_to_remove(v: int, drt: float, blockers: set[int]) -> bool:
            if nearby_terminals := self._collect_local_demand_neighborhood(
                v, blockers, self.node_weight_map, int(drt)
            ):
                nearby_terminals -= {self.super_root_index}
                if not nearby_terminals:
                    return True
                # recurse using max effective distance
                max_effective_distance = max(effective_terminal_drt.get(t, drt) for t in nearby_terminals)
                if max_effective_distance == drt:
                    return False
                return is_safe_to_remove(v, max_effective_distance, nearby_terminals)
            return True

        removables = [
            v
            for v in steiners
            if self.graph[v][PAYLOAD_WEIGHT_KEY] > global_min_max_drt
            or is_safe_to_remove(v, global_min_max_drt, self.non_steiner_nodes())
        ]

        if removables:
            self.remove_nodes_from(removables)
            logger.warning(f"    removed {len(removables)} Steiner nodes via recursive drt reduction")
            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(n) for n in removables)}")
                self.dump_state("reduce_steiner_nodes_by_distance_recursive", len(removables))
                self.validate_reachability()

        return len(removables)

    def _local_dijkstra(self, source: int, radius: int, weight: dict[int, int]) -> dict[int, int]:
        import heapq

        dist: dict[int, int] = {source: weight[source]}
        heap = [(weight[source], source)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if d > radius:
                break  # stop expanding beyond radius

            for v in self.graph.neighbors(u):
                nd = d + weight[v]
                if nd < dist.get(v, float("inf")) and nd <= radius:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        return dist

    def _collect_local_demand_neighborhood(
        self,
        center: int,
        demand_nodes: set[int],
        weight: dict[int, int],
        radius: int,
    ) -> set[int]:
        dist = self._local_dijkstra(center, radius, weight)
        return {u for u in dist if u in demand_nodes}

    def _collision_envelope(
        self, source: int, non_steiners: set[int], weight_map: dict[int, int], radius_cap: float
    ) -> set[int]:
        """Dijkstra from source, expanding only through Steiner nodes and freezing each
        frontier branch on its first collision with an existing arbor (any node already
        committed to some root's demand). Mirrors is_potential_violation's stop-at-non-
        Steiner rule, but accumulates the touched envelope and collision points instead
        of returning a bool.

        Returns:
          collisions -- the first non-Steiner node hit along each frontier branch
        """
        dist: dict[int, float] = {source: weight_map[source]}
        heap = [(weight_map[source], source)]
        envelope: set[int] = set()
        collisions: set[int] = set()

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if d > radius_cap:
                break
            if u != source and u in non_steiners:
                collisions.add(u)
                continue  # frozen: don't expand past a collision

            envelope.add(u)
            for v in self.reduction_neighbors(u):
                nd = d + weight_map[v]
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        return collisions

    def _zero_increment_superterminal_assignments(self) -> dict[int, int]:
        """Certify floating demands that can join a real class at zero incremental cost.

        Every active demand endpoint is selected in every feasible solution, and
        optional representatives with zero residual payload can be added for free.
        Virtual antiparallel arcs represent connectivity already required inside
        each real demand class.  Consequently, a superterminal in the same
        strongly connected closure as a real root can be assigned to that root
        when the root has an explicit current admissibility arc to the directed
        super-root.

        ``fixed_nodes`` alone is deliberately not a zero-increment certificate:
        an active fixed representative can contain unpaid absorbed hypernodes.
        The artificial super-root is excluded because its inbound arcs represent
        alternative sink choices rather than ordinary graph connectivity.
        """
        sr_index = self.super_root_index
        if sr_index is None or not self.terminal_sets.get(sr_index):
            return {}

        admissible_real_roots = {
            root
            for root, terminals in self.terminal_sets.items()
            if root != sr_index and terminals and self.graph.has_edge(root, sr_index)
        }
        if not admissible_real_roots:
            return {}

        active_nodes = set(self.graph.node_indices())
        demand_nodes = {
            node for root, terminals in self.terminal_sets.items() if terminals for node in {root, *terminals}
        }
        zero_increment_nodes = (
            demand_nodes | {node for node in active_nodes if self.graph[node][PAYLOAD_WEIGHT_KEY] == 0}
        ) - {sr_index}

        closure = subgraph_stable(self.graph, zero_increment_nodes)

        # Each virtual pair is backed by connectivity that every feasible
        # solution already supplies; it does not choose or charge a concrete path.
        for root, terminals in self.terminal_sets.items():
            if root == sr_index or not terminals:
                continue
            for terminal in terminals:
                if not closure.has_edge(root, terminal):
                    closure.add_edge(root, terminal, {})
                if not closure.has_edge(terminal, root):
                    closure.add_edge(terminal, root, {})

        assignments: dict[int, int] = {}
        superterminals = self.terminal_sets[sr_index]
        for component in rx.strongly_connected_components(closure):
            component_nodes = set(component)
            roots = sorted(
                component_nodes & admissible_real_roots,
                key=self.get_node_key,
            )
            if not roots:
                continue

            for superterminal in sorted(component_nodes & superterminals, key=self.get_node_key):
                root = next((candidate for candidate in roots if candidate != superterminal), None)
                if root is not None:
                    assignments[superterminal] = root

        return assignments

    def _assign_zero_increment_superterminals(self, assignments: dict[int, int]) -> int:
        """Apply one immutable snapshot of certified floating-demand assignments."""
        sr_index = self.super_root_index
        if sr_index is None:
            return 0

        assigned = 0
        for superterminal, root in assignments.items():
            if (
                superterminal not in self.terminal_sets.get(sr_index, set())
                or root == superterminal
                or not self.terminal_sets.get(root)
                or not self.graph.has_edge(root, sr_index)
            ):
                continue

            self.terminal_sets[root].add(superterminal)
            self.terminal_sets[sr_index].remove(superterminal)
            self.super_candidate_sink_sets.pop(superterminal, None)
            self.ambiguous_super_terminals.discard(superterminal)
            assigned += 1

            logger.warning(
                f"      assigned zero-increment super-terminal {self.get_node_key(superterminal)} "
                f"to real root {self.get_node_key(root)}"
            )

        return assigned

    def reduce_isolated_super_terminals(self) -> int:
        """Reduces super root set by promoting super terminals to real roots based on single sink
        collision envelope over 2 * global_min_max_drt.

        When there exists only a single potential sink for a super terminal within 2 * global_min_max_drt
        then that must be the super terminal's root in an optimal solution.

        Handles:
        > **`| 𝕥 : card({○ ∈ (𝓟 ⋃ 𝑻 ⋃ 𝓡) : dist(𝕥, ○) ≤ 2 * min_max_drt}) = 1  ⭆  𝕥 ⭆ Ⓣ, ○ ⭆ Ⓡ, 𝕊 ≔ 𝕊 ∖ 𝕥`**
        """
        num_moved = 0
        candidate_sink_sets = self.super_candidate_sink_sets
        ambiguous_super_terminals = self.ambiguous_super_terminals
        sr_index = self.super_root_index

        if sr_index is None:
            return 0

        for st, sinks in list(candidate_sink_sets.items()):
            if len(sinks) != 1 or st in ambiguous_super_terminals:
                continue

            sink = next(iter(sinks))
            num_moved += 1

            if sink in self.fixed_nodes and sink in self.terminal_sets:
                # Moving super-terminal to real root
                self.terminal_sets[sink].add(st)
                sink_type = "r"
            else:
                # Moving to promoted potential root
                self.terminal_sets[sink] = {st}
                self.fixed_nodes.add(sink)
                sink_type = "pr"

            self.terminal_sets[sr_index].discard(st)
            del candidate_sink_sets[st]

            logger.warning(
                f"      promoted super-terminal {self.get_node_key(st)} "
                f"to root {self.get_node_key(sink)} (was: {sink_type})"
            )

        zero_increment_assignments = self._zero_increment_superterminal_assignments()
        zero_increment_moved = self._assign_zero_increment_superterminals(zero_increment_assignments)
        if zero_increment_moved:
            self.call_counts["reduce_zero_increment_superterminal_assignment"] += zero_increment_moved
            num_moved += zero_increment_moved

        # super-root may now be empty
        if not self.terminal_sets.get(sr_index):
            logger.debug("      reduce_isolated_super_terminals: removing super root")
            self.terminal_sets.pop(sr_index, None)
            self.graph.remove_node(sr_index)
            self.potential_roots = set()
            self.super_root_index = None
            self.super_candidate_sink_sets = {}

        if num_moved:
            logger.warning(f"      promoted {num_moved} isolated super terminals...")
            if self.do_debug:
                self.dump_state("reduce_isolated_super_terminals", num_moved)
                self.validate_reachability()

        return num_moved

    def solve_isolated_roots_as_trees(self) -> int:
        """Reduces globally isolated singleton root arborescences as exact sub-trees by consumption.

        When an arborescence 𝓐ᵏ is provably isolated from all other demand sets
        in the global potential interaction graph, its optimal realization W_k
        can be computed independently. The nodes of W_k (excluding Ⓡᵏ) are consumed
        outward into the root, fixing them into the solution set 𝓢.

        Handles:
        > | Ⓡᵏ : Isol(𝓐ᵏ) ⭆ ⓥ ↣ Ⓡᵏ ∀ ⓥ ∈ (W_k ∖ Ⓡᵏ), 𝓢 ≔ 𝓢 ⋃ W_k  ⇔  W_k ∈ OPT_ST(𝓐ᵏ, 𝐆)

        NOTE: Global isolation (`Isol(𝓐ᵏ)`) is distinct from local isolation within a
        filtered interactivity graph block. Consuming nodes based on filtered local
        isolation is unsound because fixing nodes in 𝓢 introduces zero-cost paths that
        may distort global sub-optimality for disjoint arborescences. Only global
        envelope isolation guarantees that consuming W_k preserves global optimality.

        NOTE:
        This function does not process super terminals as roots.
        """
        logger.trace("solve_isolated_roots_as_trees...")

        # NOTE: super terminals can not be handled as isolates,
        # but they can be part of an isolates solution or promoted to real terminals...

        sr_index = self.super_root_index
        self.set_edge_weights()

        if sr_index is None:
            interactivity_graph, _blocked_edges = self._interactivity_graph(
                self.graph, self.graph, self.terminal_sets
            )
        else:
            candidate_sink_sets = self.super_candidate_sink_sets
            node_weight_map = self.node_weight_map

            non_super_coverage_sets = self.terminal_sets.copy()
            non_super_coverage_sets.pop(sr_index, None)

            non_super_G = self.graph.copy()
            non_super_G.remove_node(sr_index)

            interactivity_graph, _, _ = self._interactivity_graph_super(
                non_super_G,
                node_weight_map,
                candidate_sink_sets,
                non_super_coverage_sets,
            )

        isolates = rx.isolates(interactivity_graph)
        isolates = set(isolates) & set(self.terminal_sets) - {sr_index}
        if not isolates:
            return 0

        all_consumed = set()

        for root in isolates:
            witnessed_nodes = self.solve_root_as_tree(root)

            # Consume all witnessed nodes, from the root outward; leaving the root node in the graph.
            consumed = set()
            witnessed_to_consume = set(witnessed_nodes)
            while len(witnessed_to_consume) > 0:
                neighbors = self.reduction_neighbors(root)
                neighbors = neighbors & witnessed_to_consume
                if len(neighbors) == 0:
                    break

                for n in neighbors:
                    self.consume(n, root)
                    consumed.add(n)
                    witnessed_to_consume.discard(n)

            all_consumed.update(consumed)

            # Handle any naturally settled super-terminals
            if sr_index is not None:
                settled_super_terminals = self.terminal_sets[sr_index] & consumed
                self.terminal_sets[sr_index] -= settled_super_terminals
                if len(self.terminal_sets[sr_index]) == 0:
                    logger.warning("      solve_isolated_roots_as_trees: removing super root")
                    self.terminal_sets.pop(sr_index)
                    self.graph.remove_node(sr_index)
                    self.potential_roots = set()
                    self.super_root_index = None
                    sr_index = None

            # Lastly remove the terminal set cluster
            self.terminal_sets.pop(root)

        if all_consumed:
            logger.warning(f"      solved {len(isolates)} isolated trees...")
            if self.do_debug:
                logger.trace(f"    consumed: {sorted(self.get_node_key(n) for n in all_consumed)}")
                self.dump_state("solve_isolated_roots_as_trees", len(all_consumed))
                self.validate_reachability()

        return len(all_consumed)

    def solve_as_path(self):
        """Solves a single remaining terminal -> root path and resolves the full problem space."""
        logger.trace("solve_as_path...")

        assert len(self.terminal_sets) == 1
        root = next(iter(self.terminal_sets.keys()))
        assert len(self.terminal_sets[root]) == 1
        terminal = next(iter(self.terminal_sets[root]))

        self.set_edge_weights()
        paths = rx.all_shortest_paths(self.graph, terminal, root, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY])

        # NOTE: We retain any ambiguity in parallel paths since MIP presolve will resolve
        #       it with the benefit of the fixed_nodes structure more efficiently.
        witnessed = set().union(*paths)
        removables = set(self.graph.node_indices()) - witnessed
        removals = len(removables)

        if removables:
            self.graph.remove_nodes_from(removables)
            logger.info(f"    solve_as_path: removed {removals} dead solution nodes")

            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(i) for i in removables)}")
                self.dump_state("solve_as_path", removals)
                self.validate_reachability()

        return removals

    def solve_as_tree(self) -> int:
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances.

        For any optimal global Forest solution that Forest can be found in one of the solutions to
        all of the partitions of the terminal sets. The partitions themselves can be represented as
        independent composite instances over all subsets of the terminal sets. This function utilizes
        a set cover DP solver to find the best solution to each composite instance and then takes
        the best solution from these composite instances as a witnessed optimal solution to the remaining
        stable reduced state graph and removes all non-witnessed nodes from the graph.

        Handles:
        > **`𝐏* ≔ argmin_{𝐏 ∈ Partition(𝒞)} ∑_{B ∈ 𝐏} OPT_ST(B, 𝐆)  ⭆  𝐆 ≔ 𝐆 ∖ ({ⓥ ∈ 𝐆 : ⓥ ∉ ⋃_{B ∈ 𝐏*} W_B} ∖ 𝕊)  ⇔  𝐆 stable ∧ ¬𝓦 ∧ 𝑻 ≠ ∅`**
        >
        > where: 𝒞 ≔ 𝓡 ∖ 𝕊, or 𝓡 ∖ 𝕊 augmented with 𝕥 ∈ dom(𝕊) when 𝕊 is active

        Several optimizations take place at the block level, the task level, and the mask level to
        speed up the solving process; each is independently certified.

        Block level optimization lemmas (prep, applied while generating candidate blocks):
            1. A basic structurally valid block can only consist of reachable coverage representatives
               and the potential interactivity graph (IG) represents potential interactions with edges
               between the coverage representatives, thus all representatives of a block are strongly
               connected in the IG.
            2. An IG edge admitted by proximity (gap/drt) but lacking a genuine Steiner-only path is
               illusory as a standalone pair -- some third representative's own required territory is
               doing the connecting, and that representative's cost is paid in full regardless of block
               membership, so the bare pair buys no sharing. The pair itself is excluded from block
               generation, but nothing else is: {A,B} being illusory says nothing about {A,B,C}, so any
               superset still generated using that same IG edge is left untouched and judged on its own
               solve, never pruned by inheritance from the pair's exclusion.
            3. The same illusory-sharing question, asked directly rather than through the IG's proximity
               edges: for any candidate block (of any size), union-find the block's own representatives
               together with the Steiner components each one's own territory touches. A block whose
               representatives don't all land in one component has no representative acting as the real
               bridge for at least one pair inside it -- the same node would then be priced by more than
               one independently-solved block in the DP sum, which is unsound regardless of how the IG's
               edges happened to admit the block structurally. This is the tightest sound necessary
               condition available; it subsumes lemma 2 but is more expensive, so lemma 2 still prunes
               first wherever it can.
            4. Pre-solve dominance by max terminal-to-terminal distance: no Steiner tree spanning a set
               of terminals can cost less than the single longest pairwise shortest-path distance among
               them (that path must be covered by the tree in some form). If that distance already
               exceeds the sum of the block's already-solved singleton costs, the composite cannot beat
               the decomposition and is dropped before ever reaching the solver.
            5. Pre-solve dominance by MST lower bound: `OPT >= W_MST / (2 - 2/k)` (KMB) is a lower bound
               on any Steiner tree over k terminals. If that bound already exceeds the sum of the block's
               singleton costs, no joint solve can beat the decomposition, so the block is dropped before
               solving. Strict '>' only -- a tie falls through to the solver, since a tying composite may
               still share more structure with the rest of the forest than any decomposition would.
            6. Post-solve dominance by singletons: once solved, a block whose direct cost is not lower
               than the sum of its members' singleton costs is dominated by the decomposition and is
               excluded from the mask space entirely -- it can never be part of an optimal partition,
               since replacing it with its singletons is always at least as good.
            7. Super terminal pathway only: any non-super block already proven dominated during the
               non-super solve stays excluded from the augmented (super-terminal-inclusive) candidate
               space -- augmenting a block with a super terminal cannot rescue a decomposition that was
               already proven no better than its primitives before the super terminal was even considered.
            8. Co-occurrence dominance: when a block's non-super members were already solved as their own
               block B, B's solution may cover super-terminals. If so B's result is dominant at equality --
               reuse B's solution and cost for the super-composite instead of issuing a solve.
        Mask level optimization lemmas (DP, applied over solved block costs before/during partitioning):
            8. Composite mask dominance: for a solved block's mask, if any decomposition into two
               disjoint, already-solved sub-masks costs no more than the block's own direct solve, the
               block is not a primitive realization -- it is dropped from the mask space the same as
               lemma 6, just re-checked at the mask level where the full combinatorial decomposition
               space (not just the block's own singleton members) is available to compare against.
            9. Co-occurrence partitioning: two coverage representatives only ever need a joint DP state
               if some surviving mask contains both, directly or transitively via a chain of shared
               masks. Any representative outside that union-find grouping is provably independent of
               the rest -- the global optimum decomposes exactly into each component's independent
               optimum, so the DP is solved once per independent component instead of once over the
               full state space, which a single combined n-bit DP would pay for needlessly.
        """
        logger.trace("solve_as_tree...")

        start_time = time()

        non_super_G = self.graph.copy()
        sr_index = self.super_root_index
        if sr_index is not None:
            non_super_G.remove_node(sr_index)
        non_super_coverage_sets = {r: v for r, v in self.terminal_sets.items() if r != sr_index}

        (
            solution,
            _cost,
            non_super_valid_blocks,
            non_super_valid_block_results,
            non_super_valid_block_costs,
            non_super_dp_block_results,
            non_super_dp_block_costs,
        ) = self.solve_as_tree_composite(non_super_G, non_super_coverage_sets)

        if sr_index is not None:
            # Pass thru to MIP (or, let the solve as path deal with this)
            if self.super_root_index not in self.terminal_sets or (
                len(self.terminal_sets) == 1 and len(self.terminal_sets[self.super_root_index]) == 1
            ):
                solution = set(self.graph.node_indices())
            else:
                solution = self.solve_as_tree_composite_super(
                    non_super_G,
                    non_super_coverage_sets,
                    non_super_valid_blocks,
                    non_super_valid_block_results,
                    non_super_valid_block_costs,
                    non_super_dp_block_results,
                    non_super_dp_block_costs,
                )

        end_time = time()

        removables = set(self.graph.node_indices()) - set(solution)
        removals = len(removables)

        # Regardless of the disposition of the super root we don't want to remove it from the graph
        # as that is handled during root reduction.
        removables.discard(self.super_root_index)

        if removals > 0:
            duration = end_time - start_time

            complexity = self.dw_remaining_complexity()
            num_nodes = self.graph.num_nodes()
            num_roots = len(self.terminal_sets)
            num_terminals = len(set().union(*self.terminal_sets.values()))
            num_super_terminals = (
                len(self.terminal_sets[self.super_root_index]) if self.super_root_index is not None else 0
            )
            logger.warning(
                f"    solved as tree in {duration:.2f}s with complexity {complexity}, |r|: {num_roots}, |t|: {num_terminals}, |st|: {num_super_terminals}, |n|: {num_nodes}"
            )

            if self.do_debug:
                print("pre-tree-solution graph dead node removal...")
                self.validate_reachability()
                self.dump_graph("before solving as tree")

            self.graph.remove_nodes_from(removables)

            logger.info(f"    removed {removals} dead tree solution nodes")
            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(i) for i in removables)}")
                self.dump_state("solve_as_tree", removals)
                self.dump_graph("after solving as tree")
                self.validate_reachability()

        else:
            if len(self.terminal_sets) == 1 and self.super_root_index in self.terminal_sets:
                logger.warning("    did not solve as tree, passing to MIP...")
            else:
                logger.error("    failed to solve as tree")

        return removals

    # MARK: Tree Solvers

    def solve_root_as_tree(self, root: int) -> set[int]:
        """Solves a single root from the remaining reduced state graph.

        NOTE: An isolated root (per the interactivity-graph isolation check) has
        no possible interaction with any other root, so there is no partition
        or decomposition question to answer here -- unlike solve_as_tree_composite,
        this is always exactly one block: the root's own flattened terminal set,
        solved as a single tree.
        """
        logger.trace("solve_root_as_tree...")

        terminals_list = sorted(self.terminal_sets[root] | {root})
        logger.trace(f"    solving isolated root {root} with {len(terminals_list)} terminals...")

        self.solved_trees += 1

        root_component_index_map, component_data = nwst._connected_component_mappings(
            self.graph, set(self.terminal_sets.keys())
        )
        comp_map = component_data[root_component_index_map[root]]

        # NOTE: solve_tree auto switches between DW and scipstp based on tree complexity,
        _, cost, mask = solve_tree(
            TreeProblem(
                instance_id=self.instance_id,
                block_key=(root,),
                terminals=terminals_list,
                adj_map=comp_map["adj_map"],
                node_weight_map=self.node_weight_map,
                node_index_map=self.node_index_map,
                dimacs_id_map=comp_map["dimacs_id_map"],
                inv_dimacs_id_map=comp_map["inv_dimacs_id_map"],
                enable_super_root_index=0,
                do_debug=self.do_debug,
                mip_validation=False,
            )
        )

        logger.trace(f"    solved isolated root {root} with cost {cost}")
        best_solution = nwst._unmask_solution(self.graph, mask)

        if best_solution:
            logger.debug(
                f"    solved for terminals: {self.get_node_key(root)}: "
                f"{ {self.get_node_key(n) for n in self.terminal_sets[root]} }"
            )
            logger.debug(
                f"    found best solution (cost: {cost}): {sorted(self.get_node_key(n) for n in best_solution)}"
            )

        return best_solution

    def solve_as_tree_composite(
        self, G: PyDiGraph, coverage_sets: dict[int, set[int]]
    ) -> tuple[
        SolutionSet,
        Cost | float,
        ConnectedComponent_BlockKeys,
        BlockResults,
        BlockCosts,
        BlockResults,
        BlockCosts,
    ]:
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances."""
        logger.warning("  solve_as_tree_composite...")

        # Sometimes all that is left are super terminals and there's nothing to be done here...
        if not coverage_sets:
            return set(), float("inf"), {}, {}, {}, {}, {}

        coverage_representatives = set(coverage_sets.keys())

        _root_component_index_map, component_data = nwst._connected_component_mappings(
            G, coverage_representatives
        )
        interactivity_graph, blocked_edges = self._interactivity_graph(G, G, coverage_sets)
        weight_map = {i: G[i][PAYLOAD_WEIGHT_KEY] for i in G.node_indices()}

        # Phase 1 - block preparation
        valid_blocks, num_blocks, num_candidate_blocks = nwst._generate_valid_partition_blocks(
            component_data, interactivity_graph, blocked_edges
        )
        block_terminals = nwst._map_block_terminals(coverage_sets, valid_blocks)

        if blocked_edges:
            root_components = nwst._steiner_connected_root_components(G, coverage_sets)
            valid_blocks = nwst._retain_structurally_admissible_blocks(valid_blocks, root_components)
            new_num_blocks = sum(len(blocks) for blocks in valid_blocks.values())
            num_blocks = new_num_blocks

        # --- Wave 1: singletons. These are the atoms of every decomposition --
        # there's nothing to prune them against, so they always get solved.
        singleton_tasks = [
            (cc_i, block_key, block_terminals[(cc_i, block_key)], 0)
            for cc_i, blocks in valid_blocks.items()
            for block_key in blocks
            if len(block_key) == 1
        ]

        problem_generator = nwst._problem_generator(
            self.instance_id,
            component_data,
            weight_map,
            singleton_tasks,
            do_debug=self.do_debug,
            mip_validation=False,
        )

        # NOTE: Long lived collections throughout the solving phases
        block_results = {}
        block_costs = {}

        # Solve all singletons for usage in composite block filtering (Wave 2)
        logger.warning(
            f"    sequentially solving {len(singleton_tasks)} singleton blocks of {num_candidate_blocks}..."
        )

        self.solved_trees += len(singleton_tasks)
        nwst._solve_treeproblem_tasks(
            problem_generator, len(singleton_tasks), results_dest=block_results, costs_dest=block_costs
        )

        # Phase 2 (Wave 2 gate) - filter by dominance using max_dist/MST bound and primitive cost
        # NOTE: OPT >= W_MST / (2 - 2/k) (KMB). If that lower bound already exceeds the cheapest
        #       decomposition we can prove right now (singleton sum), no joint solve can beat it.
        #       Strict '>' only -- ties fall through to the solver.
        #       (Same as the post-solve dominance check, since a tying composite may still share
        #       more structure with the rest of the forest than any decomposition would.)
        terminal_to_terminal_distances = {}
        all_terminals = sorted(set(coverage_sets.keys()).union(*coverage_sets.values()))
        for i, ti in enumerate(all_terminals):
            for tj in all_terminals[i + 1 :]:
                d = self.shortest_path_length(ti, tj)
                terminal_to_terminal_distances[(ti, tj)] = d
                terminal_to_terminal_distances[(tj, ti)] = d

        # block_costs serves as the primitive block costs...
        surviving_tasks = nwst._retain_dominant_blocks_by_distance(
            valid_blocks, block_terminals, block_costs, terminal_to_terminal_distances
        )

        # Wave 2: Surviving valid block solving...
        self.solved_trees += len(surviving_tasks)
        problem_generator = nwst._problem_generator(
            self.instance_id,
            component_data,
            weight_map,
            surviving_tasks,
            do_debug=self.do_debug,
            mip_validation=False,
        )
        num_problems = len(surviving_tasks)

        logger.warning(
            f"    solving {num_problems} surviving of {num_blocks} unique valid blocks of {num_candidate_blocks}..."
        )

        nwst._solve_treeproblem_tasks(
            problem_generator, num_problems, results_dest=block_results, costs_dest=block_costs
        )

        best_solution, best_cost, dp_block_results, dp_block_costs = nwst._solve_composite_blocks_dp(
            G, coverage_representatives, block_results, block_costs
        )
        if best_solution:
            logger.debug(
                f"    solved for terminals: { {self.get_node_key(r): {self.get_node_key(n) for n in ts} for r, ts in self.terminal_sets.items()} }"
            )
            logger.trace(
                f"    found best solution (cost: {best_cost}): {sorted(self.get_node_key(n) for n in best_solution)}"
            )

        return (
            best_solution,
            best_cost,
            valid_blocks,
            block_results,
            block_costs,
            dp_block_results,
            dp_block_costs,
        )

    def solve_as_tree_composite_super(
        self,
        non_super_G: PyDiGraph,
        non_super_coverage_sets: CoverageSets,
        non_super_valid_blocks: ConnectedComponent_BlockKeys,
        non_super_valid_block_results: BlockResults,
        non_super_valid_block_costs: BlockCosts,
        non_super_dp_block_results: BlockResults,
        non_super_dp_block_costs: BlockCosts,
    ) -> SolutionSet:
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances."""
        logger.warning("  solve_as_tree_composite_super...")

        assert self.super_root_index is not None, "super root index must be set"
        sr_index = self.super_root_index

        # NOTE: any valid block not in block results was dominated prior to solving
        non_super_valid_block_keys = set().union(*non_super_valid_blocks.values())
        non_super_dominated_block_keys = set(non_super_valid_block_keys) - set(
            non_super_dp_block_results.keys()
        )

        super_candidate_sink_sets = self.super_candidate_sink_sets

        node_weight_map = {u: self.graph[u][PAYLOAD_WEIGHT_KEY] for u in self.graph.node_indices()}
        roots = set(self.terminal_sets.keys()) - {sr_index}
        super_terminals = sorted(super_candidate_sink_sets.keys())
        super_terminals_set = set(super_terminals)
        coverage_entities = roots | super_terminals_set

        interactivity_graph, blocked_edges, composite_coverage_sets = self._interactivity_graph_super(
            non_super_G,
            node_weight_map,
            super_candidate_sink_sets,
            non_super_coverage_sets,
        )

        # Component generation
        _root_component_index_map, component_data = nwst._connected_component_mappings(
            self.graph, coverage_entities, sr_index
        )

        # The super root is always its own trivial SCC (pure sink), so `component_data`
        # carries no adjacency entry for it. A block folding all remaining super terminals
        # into a lone-root TreeProblem needs `adj_map[super_root_index]` populated with just
        # the potential roots reachable within that component.
        sr_predecessors = self.reduction_neighbors(sr_index)
        for cc_data in component_data.values():
            local_sr_preds = sr_predecessors & cc_data["component"]
            if local_sr_preds:
                cc_data["adj_map"][sr_index] = sorted(local_sr_preds)

        # Phase 1 - block preparation
        valid_blocks, num_blocks, num_candidate_blocks = nwst._generate_valid_partition_blocks(
            component_data, interactivity_graph, blocked_edges
        )

        for cc_i in valid_blocks:
            valid_blocks[cc_i] -= non_super_dominated_block_keys
        num_surviving_blocks = sum(len(b) for b in valid_blocks.values())
        if num_blocks != num_surviving_blocks:
            logger.warning(
                f"    dominated non-super block filtering lowered valid_blocks from {num_blocks} to {num_surviving_blocks}..."
            )

        # NOTE: Long lived collections throughout the solving phases
        # Pre-populated with non-super block results for filtering and final DP solving
        block_results = dict(non_super_dp_block_results)
        block_costs = dict(non_super_dp_block_costs)

        # Task generation
        singleton_tasks: list[BlockTask] = []
        composite_tasks: list[BlockTask] = []

        skipped_presolved_block_count = 0
        skipped_dominated_block_count = 0

        for cc_i, blocks in valid_blocks.items():
            for block_key in blocks:
                terminals = set()

                # A super terminal behaves as a terminal in the presence of another root,
                # yet it behaves as a root with a single terminal (the super root) when
                # there are no other roots. This enforces potential root transit when the
                # super terminals are considered in isolation.
                block_sr_index = 0  # disabled

                # Adding the super root to the terminals of a block that contains only super terminals
                # allows the arbor root to be the super root.
                if all(i in super_terminals for i in block_key):
                    terminals.add(self.super_root_index)
                    block_sr_index = sr_index

                # We purposefully omit the collision envelope's covered entities (candidate sinks) from the block
                # since the super terminal is being considered as a normal terminal within the composite block.
                for k in block_key:
                    terminals.add(k)
                    if k not in super_terminals:
                        terminals.update(self.terminal_sets[k])

                # Any block without any super terminals has already been pre-solved or dominated.
                if not terminals & super_terminals_set:
                    skipped_presolved_block_count += 1
                    continue

                # When a previously solved non-super block covers the super terminals of the
                # augmented block, we can use its solution as the solution for the augmented block.
                non_super_part = tuple(sorted(k for k in block_key if k not in super_terminals))
                non_super_mask = non_super_valid_block_results.get(non_super_part) if non_super_part else None
                if non_super_mask is not None:
                    non_super_witness_nodes = nwst._unmask_solution(self.graph, non_super_mask)
                    if (set(block_key) & super_terminals_set) <= non_super_witness_nodes:
                        block_results[block_key] = non_super_mask
                        block_costs[block_key] = non_super_valid_block_costs[non_super_part]
                        skipped_dominated_block_count += 1
                        continue

                terminals = sorted(terminals)
                task = (cc_i, block_key, terminals, block_sr_index)
                if len(block_key) == 1 or len(block_key) == 2 and block_sr_index == block_key[1]:
                    singleton_tasks.append(task)
                else:
                    composite_tasks.append(task)

        if skipped_presolved_block_count:
            logger.warning(f"    skipped re-solving {skipped_presolved_block_count} non-super blocks...")
        if skipped_dominated_block_count:
            logger.warning(
                f"    reused {skipped_dominated_block_count} non-super witnesses containing super terminals..."
            )

        # --- Wave 1: singletons. These are the atoms of every decomposition --
        # there's nothing to prune them against, so they always get solved.
        problem_generator = nwst._problem_generator(
            self.instance_id,
            component_data,
            node_weight_map,
            singleton_tasks,
            do_debug=self.do_debug,
            mip_validation=False,
        )

        # Solve all singletons for usage in composite block filtering (Wave 2)
        logger.warning(f"    solving {len(singleton_tasks)} singleton blocks of {num_blocks}...")

        self.solved_trees += len(singleton_tasks)
        nwst._solve_treeproblem_tasks(
            problem_generator, len(singleton_tasks), results_dest=block_results, costs_dest=block_costs
        )

        # Phase 2 (Wave 2 gate) - filter by dominance using max_dist/MST bound and primitive cost
        # NOTE: OPT >= W_MST / (2 - 2/k) (KMB). If that lower bound already exceeds the cheapest
        #       decomposition we can prove right now (singleton sum), no joint solve can beat it.
        #       Strict '>' only -- ties fall through to the solver.
        #       (Same as the post-solve dominance check, since a tying composite may still share
        #       more structure with the rest of the forest than any decomposition would.)
        terminal_to_terminal_distances = {}
        all_terminals = sorted(
            set(non_super_coverage_sets.keys()).union(*non_super_coverage_sets.values()) | super_terminals_set
        )
        for i, ti in enumerate(all_terminals):
            for tj in all_terminals[i + 1 :]:
                d = self.shortest_path_length(ti, tj)
                terminal_to_terminal_distances[(ti, tj)] = d
                terminal_to_terminal_distances[(tj, ti)] = d

        # block_costs serves as the primitive and pre-solved block costs...
        composite_tasks = nwst._retain_dominant_composite_tasks_by_distance(
            composite_tasks, block_costs, terminal_to_terminal_distances, self.super_root_index
        )

        if blocked_edges:
            root_components = nwst._steiner_connected_root_components(non_super_G, composite_coverage_sets)
            composite_tasks = nwst._retain_structurally_admissible_tasks(composite_tasks, root_components)

        # Wave 2: Surviving valid block solving...
        problem_generator = nwst._problem_generator(
            self.instance_id,
            component_data,
            node_weight_map,
            composite_tasks,
            do_debug=self.do_debug,
            mip_validation=False,
        )
        num_problems = len(composite_tasks)
        self.solved_trees += num_problems

        logger.warning(
            f"    solving {num_problems} surviving of {num_blocks} unique valid blocks of {num_candidate_blocks}..."
        )

        nwst._solve_treeproblem_tasks(
            problem_generator, num_problems, results_dest=block_results, costs_dest=block_costs
        )

        best_solution, best_cost, _block_results, _block_costs = nwst._solve_composite_blocks_dp(
            self.graph, coverage_entities, block_results, block_costs
        )

        if best_solution:
            logger.debug(
                f"    solved for terminals: { {self.get_node_key(r): {self.get_node_key(n) for n in ts} for r, ts in self.terminal_sets.items()} }"
            )
            logger.debug(
                f"    found best solution (cost: {best_cost}): {sorted(self.get_node_key(n) for n in best_solution)}"
            )

        return best_solution

    def _super_candidate_sink_sets(self, max_drt_factor: int = 2) -> tuple[CoverageSets, set[NodeIndex]]:
        """Collects remaining super terminal roots, potential roots and terminals that can be
        used as demand sinks for each super terminal.

        NOTE: If there are any remaining super terminals we must not remove them from the graph
        _nor_ allow them to become isolated terminals. Therefore, we need to identify the
        demand sinks (roots, potential roots and terminals) in the neighborhood of each
        super terminal and add them to the candidate roots for that super terminal.

        A collision with another super terminal never resolves to a candidate sink -- a
        super terminal id is not a valid promotion target -- but it does prove a second,
        unresolved potential sink route exists (st routes through that other super
        terminal's own eventual path to sr), so any st that collided with another super
        terminal is reported separately as ambiguous, regardless of what else was found.
        A resolved candidate set of size 1 is only a forced sink when st is NOT ambiguous;
        otherwise the untested alternate route may be cheaper once resolved against the
        interactivity graph, and forcing the promotion here would pre-empt that. If every
        collision for st was of this kind (or none occurred), st is still guaranteed a real
        path to sr, as its nearest potential root(s) are used.

        Returns:
            (candidate_sinks, ambiguous_sts)
        """
        if self.super_root_index is None:
            return {}, set()

        sr_index = self.super_root_index
        candidate_sinks: CoverageSets = defaultdict(set)
        ambiguous_sts: set[int] = set()

        real_roots = set(self.terminal_sets.keys()) - {sr_index}
        super_terminals = self.terminal_sets[sr_index]
        terminal_to_root_map = {v: k for k in real_roots for v in self.terminal_sets[k]}

        non_steiners = self.non_steiner_nodes()
        radius_cap = max_drt_factor * self._global_min_max_drt
        weight_map = {i: self.graph[i][PAYLOAD_WEIGHT_KEY] for i in self.graph.node_indices()}

        self.set_edge_weights()

        for st in super_terminals:
            covered = self._collision_envelope(st, non_steiners, weight_map, radius_cap)

            for v in covered:
                if v in real_roots:
                    candidate_sinks[st].add(v)
                    continue

                # A terminal in some root's arbor...
                if v in terminal_to_root_map:
                    candidate_sinks[st].add(terminal_to_root_map[v])
                    continue

                if v in super_terminals:
                    # ambiguous, resolved below if nothing else survives
                    ambiguous_sts.add(st)
                    continue

                if v in self.potential_roots and self.shortest_path_length(st, v) <= self._global_min_max_drt:
                    candidate_sinks[st].add(v)

            if not candidate_sinks[st]:
                paths = rx.all_shortest_paths(
                    self.graph, st, sr_index, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY]
                )
                candidate_sinks[st] |= {p[-2] for p in paths}

        return candidate_sinks, ambiguous_sts

    def _interactivity_graph_super(
        self,
        non_super_G: PyDiGraph,
        node_weight_map: dict[int, int],
        super_candidate_sink_sets: CoverageSets,
        non_super_coverage_sets: CoverageSets,
    ) -> tuple[PyDiGraph, BlockedInteractionEdges, CoverageSets]:
        logger.trace("    _interactivity_graph_super...")
        assert self.super_root_index is not None

        sr_index = self.super_root_index
        super_terminals = set(super_candidate_sink_sets.keys())

        # Make super terminals representative roots with their own cover sets
        super_coverage_sets: CoverageSets = defaultdict(set)
        for t in super_terminals:
            paths = rx.all_shortest_paths(self.graph, t, sr_index, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY])
            trimmed = [p[:-1] for p in paths]  # drop the synthetic sr hop
            super_coverage_sets[t] = {p[-1] for p in trimmed}

        for t in super_terminals:
            super_coverage_sets[t] |= set(super_candidate_sink_sets[t])

        # --- Composite coverage sets ---
        composite_coverage_sets = dict(non_super_coverage_sets)
        composite_coverage_sets.update(super_coverage_sets)

        # Drop all instances of super root from all coverage sets since sr_index isn't
        # a valid root to index into.
        for s in composite_coverage_sets.values():
            s.discard(self.super_root_index)

        # --- Interactivity graph ---
        # NOTE: We have to use self.graph to capture st -> sr paths here...
        arbor_rt_all_shortest_path_unions = {r: set() for r in composite_coverage_sets}
        for r, terminals in composite_coverage_sets.items():
            for t in terminals:
                paths = rx.all_shortest_paths(self.graph, t, r, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY])
                arbor_rt_all_shortest_path_unions[r].update(*paths)

        interactivity_edges = get_interactivity_edges(
            non_super_G,
            composite_coverage_sets,
            node_weight_map,
            arbor_rt_all_shortest_path_unions,
        )

        # NOTE: We can't use self.graph for the interactivity itself or the gaps will collapse
        interactivity_graph, blocked_edges = self._interactivity_graph(
            self.graph,
            non_super_G,
            composite_coverage_sets,
            interactivity_edges=interactivity_edges,
        )

        return interactivity_graph, blocked_edges, composite_coverage_sets

    def _interactivity_graph(
        self,
        paths_G: PyDiGraph,
        interactivity_G: PyDiGraph,
        coverage_sets: CoverageSets,
        interactivity_edges: list[tuple[NodeIndex, NodeIndex, Cost | float]] | None = None,
    ) -> tuple[PyDiGraph, BlockedInteractionEdges]:
        """Creates the interactivity graph for the given reduced state graph over the given coverage sets."""
        interactivity_graph, blocked_edges = nwst._interactivity_graph(
            paths_G,
            interactivity_G,
            coverage_sets,
            self.super_root_index,
            interactivity_edges,
        )

        if self.do_debug:
            tmp_map = {i: (n, self.get_node_key(n)) for i, n in enumerate(interactivity_graph.node_indices())}
            print(f"      interactivity_graph.num_nodes = {interactivity_graph.num_nodes()}")
            print(f"      interactivity_graph.num_edges = {interactivity_graph.num_edges()}")
            print("      interactivity_graph.map:")
            for k, v in tmp_map.items():
                print(f"        {k}: {v}")

            adj_matrix = rx.adjacency_matrix(interactivity_graph)

            interactive_roots: dict[int, list[int]] = {}
            for root_i, row in enumerate(adj_matrix):
                _root = tmp_map[root_i][0]
                root_wp = tmp_map[root_i][1]
                interactive_roots[root_wp] = []
                for col_i, weight in enumerate(row):
                    if weight:
                        col_wp = tmp_map[col_i][1]
                        interactive_roots[root_wp].append(col_wp)

            print("      interactive_roots:")
            for root, cols in interactive_roots.items():
                print(f"        {root}: {cols}")

            print("      blocked_edges:")
            for u, v in blocked_edges:
                print(f"        {u}: {v}")

        return interactivity_graph, blocked_edges

    # MARK: Main Loop

    # NOTE: Reductions that are known and certified but are not implemented here are:
    #
    # - leaf potential root reduction: not implemented because a potential root demotion
    #   to a Steiner node does not directly impact the problem space and holds low impact
    #   for cascading ruductions.
    #
    # - adjacent degree-3 Steiner triangle reduction: not implemented because the reduction
    #   does not directly impact the problem space and holds low impact for cascading
    #   reductions.
    #
    # - 2-cut isolated arbor reduction (super-terminal not present): this reduction is
    #   potentially impactful but expensive to test for if done per pipeline iteration
    #   and rare when terminal coverage percentage gets above ~20%.
    #   (which is a majority of real world cases)

    # NOTE: Potential high impact areas to explore if certification can be shown:
    #
    # - wider exploitation of the drt measure

    def run_pipeline(self) -> tuple[set[int], dict[int, int]]:
        """Runs the main reduction pipeline."""
        logger.trace("run_pipeline...")

        num_edges_start = self.graph.num_edges()
        num_nodes_start = self.graph.num_nodes()
        num_terminal_roots_start = len(self.terminal_sets)
        num_terminals_start = sum(len(ts) for ts in self.terminal_sets.values()) + num_terminal_roots_start

        # In the original problem space the min max drt represents a global bound that
        # superceeds the min max drt of any terminal set in the reduced problem space as
        # trees collide. If clusters a, b and c initially have a min max drt and two of them
        # are merged to form d, then the min max drt of d would be calculated at a greater
        # value than the global yet d could not actually detour further than the global bound
        # because it is constrained by the beneficial detour of the other two clusters.
        self.set_edge_weights()
        min_max_drt = self.min_max_rt_distance
        logger.info(f"  Global maximum RT distance: {min_max_drt}")

        tmp_num_nodes = self.graph.num_nodes() + 1
        iteration = 0

        # NOTE: Any transformed super-terminal that has transformed into a fixed node
        # is no longer a super-terminal, since the 'transform_pairs' is not aware
        # of what nodes are roots versus potential roots we satisfy the demand here.
        if self.super_root_index is not None:
            roots = self.potential_roots
            for sts in self.terminal_sets[self.super_root_index].copy():
                if sts in roots:
                    self.terminal_sets[self.super_root_index].discard(sts)
                    logger.warning(
                        f"  Super terminal {self.get_node_key(sts)} demand satisfied by initial graph transform..."
                    )

        while (num_nodes := self.graph.num_nodes()) != tmp_num_nodes:
            # Update self.do_debug flag when log level is externally modified
            active_severity = logger._core.min_level  # ty:ignore[unresolved-attribute]
            self.do_debug = active_severity <= 10

            tmp_num_nodes = num_nodes
            iteration += 1
            logger.info(f"--- Iteration {iteration} ---")

            # MARK: Simple reductions

            # Isolates in the graph are always safe to directly remove
            if isolates := rx.isolates(self.graph):
                self.graph.remove_nodes_from(isolates)
                logger.info(f"  removed {len(isolates)} isolates...")
                logger.trace(f"    {sorted(self.get_node_key(n) for n in isolates)}")

            simple_trigger_count = len(isolates)

            # Reduce potential roots - this is not a graph mutation it is a potential roots set consolidation only
            simple_trigger_count += self.reduce_potential_roots()

            # Reduce demand roots - this is not a graph mutation it is a terminal set consolidation only
            simple_trigger_count += self.reduce_roots_via_articulation()

            # Reduce demand roots - this is not a graph mutation it is a terminal set consolidation only
            simple_trigger_count += self.reduce_demand_roots()

            # Reduce adjacent terminals - this is a terminal reduction by `consume` with terminal set consolidation
            simple_trigger_count += self.reduce_adjacent_terminals()

            # Reduce degree1 steiner nodes - this is a Steiner node graph reduction by `consume` with terminal set consolidation
            simple_trigger_count += self.reduce_degree1_steiners()

            # Reduce degree1 terminals - this is a Steiner node graph reduction by `remove` only
            simple_trigger_count += self.reduce_degree1_terminals()

            # Reduce degree1 roots - this is a Steiner node graph reduction by `consume` with terminal set consolidation
            simple_trigger_count += self.reduce_degree1_roots()

            if len(self.terminal_sets) == 1 and len(next(iter(self.terminal_sets.values()))) == 1:
                self.solve_as_path()

            if simple_trigger_count != 0:
                logger.info("  repeating simple reductions...")
                tmp_num_nodes = 0
                continue

            # This is a direct backup to the simple_trigger_count check above.
            if num_nodes != self.graph.num_nodes():
                logger.error(
                    "  graph mutation detected during simple reductions not captured by trigger count..."
                )
                logger.info("  repeating simple reductions...")
                continue

            if not self.terminal_sets:
                logger.success("  no terminal sets left. All demands are satisfied!")
                break

            if self.super_root_index is not None:
                self.super_candidate_sink_sets, self.ambiguous_super_terminals = (
                    self._super_candidate_sink_sets()
                )

                # Reduce isolated super terminals - this is a super-root member reduction only
                if self.reduce_isolated_super_terminals():
                    # Force a reduction pass if isolated super terminals were moved
                    tmp_num_nodes = 0
                    continue

            # Solve isolated roots - this is a terminal set and Steiner node reduction by `consumption` only
            if self.solve_isolated_roots_as_trees():
                # Force a reduction pass if isolated roots were solved
                tmp_num_nodes = 0
                continue

            # MARK: Basic reductions

            # Reduce 2-degree articulation points - this is a Steiner node graph reduction with mixed handling
            self.reduce_degree2_steiners_via_articulation()

            # Reduce 2-degree steiner chains - this is a Steiner node graph reduction by `absorb` only
            self.reduce_adjacent_degree2_steiners()

            # Reduce steiner triangle 2-degree legs - this is a Steiner node graph reduction by `absorb` only
            self.reduce_steiner_triangle_degree2_legs()

            # Reduce 2-degree steiner dominance - this is a Steiner node graph reduction by `remove` only
            self.reduce_degree2_steiners_by_dominance()

            # Reduce k-degree steiner dominance - this is a Steiner node graph reduction by `remove` only
            self.reduce_degreek_steiner_dominance()

            # Reduce steiner bridges - this is a Steiner node graph reduction by `consume` only
            self.reduce_steiner_bridges()

            # Reduce Steiner nodes by distance - this is a Steiner node graph reduction by `remove` only
            self.reduce_steiner_nodes_by_distance()
            self.reduce_steiner_nodes_by_distance_recursive()

            # Reduce degreek enclosed steiner - this is a Steiner node graph reduction by `consume` only
            self.reduce_degreek_enclosed_steiner()

            # Reduce blocked roots - this is a terminal set consolidation only
            if num_nodes != self.graph.num_nodes() and self.reduce_blocked_roots() > 0:
                # Force a reduction pass if blocked roots were merged.
                tmp_num_nodes = 0
                continue

            # Reduce enclosed steiner clusters - this is a Steiner node graph reduction by `remove` only
            if num_nodes == self.graph.num_nodes() and self.reduce_enclosed_steiner_clusters():
                # Force a reduction pass if blocked roots were merged.
                tmp_num_nodes = 0
                continue

            # Solve as tree composite - this is a super-root member reduction and a Steiner node reduction by `remove` only
            if not self._solved_as_tree and num_nodes == self.graph.num_nodes() and self.terminal_sets:
                self.dump_reduction_results(
                    "pre: solve_as_tree",
                    num_edges_start,
                    num_nodes_start,
                    num_terminal_roots_start,
                    num_terminals_start,
                )
                self.solve_as_tree()
                self._solved_as_tree = True
                self.dump_reduction_results(
                    "post: solve_as_tree",
                    num_edges_start,
                    num_nodes_start,
                    num_terminal_roots_start,
                    num_terminals_start,
                )

        # Rebuild pairs
        fixed_nodes_wp = {self.get_node_key(n) for n in self.fixed_nodes}
        reduced_root_pairs_wp: dict[int, int] = {}
        for r, comp in self.terminal_sets.items():
            for t in comp:
                reduced_root_pairs_wp[self.get_node_key(t)] = self.get_node_key(r)

        if self.do_debug:
            self.dump_final_report(
                reduced_root_pairs_wp,
                fixed_nodes_wp,
            )

        self.dump_reduction_results(
            "final",
            num_edges_start,
            num_nodes_start,
            num_terminal_roots_start,
            num_terminals_start,
        )

        if reduced_root_pairs_wp:
            logger.warning(f"  Solving remaining {len(reduced_root_pairs_wp)} pairs via MIP...")

        return fixed_nodes_wp, reduced_root_pairs_wp
