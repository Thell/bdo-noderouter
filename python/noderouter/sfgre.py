from __future__ import annotations

import heapq
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from itertools import combinations
from time import time

import networkx as nx
import nwst_dw
import rustworkx as rx
from bidict import bidict
from loguru import logger
from more_itertools import set_partitions
from networkx import PlanarEmbedding
from rustworkx import PyDiGraph

from api_common import PAYLOAD_WEIGHT_KEY
from api_exploration_data import get_exploration_data
from api_nwstp_problem import TreeProblem
from api_nwstp_solver import solve_tree, solve_tree_using_scipstp
from api_rx_pydigraph import subgraph_stable

HYPERNODE_CONTENTS_KEY = "collapsed_nodes"
MAX_ENCLOSED_STEINER_INTERFACES = 6

DW_SOLVE_PAR_THRESHOLD = 16  # This will never trigger when the scipstp setup is done
DW_MAX_TREE_TERMINALS = 6
DW_MAX_TREE_COMPLEXITY = 8_000_000_000

# NOTE: The time sink in solving as trees is the partition block filtering.
# When there is an unreachable root the partitions need to be enumerated and
# this follows the Bell number pattern per root.
SCIPSTP_PAR_ROOTS_THRESHOLD = 8  # This will have 2^{n}-1 partition blocks to solve
SCIPSTP_MAX_PAR_ROOTS = 13  # This will have 2^{n}-1 partition blocks to solve
SCIPSTP_MAX_ROOTS = 6  # Beyond this the problem will be sent to MIP
SCIPSTP_NPERR_THRESHOLD = 15  # Beyond this the problem will be sent to MIP

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
| **𝕡** | Potential Root Node | An individual node member of **𝔹**. |
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

**`|Pattern : Local Guards ⭆ Transform ⇔ Global/Attribute Guards`**

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
    terminal_sets: dict[int, set[int]] = field(default_factory=dict)

    _seen_nonreducible_outer_windows: set[tuple[int, ...]] = field(default_factory=set)
    _seen_nonreducible_steiner_face_clusters: set[tuple[int, ...]] = field(default_factory=set)

    # Global maximum distance between a root and its' furthest terminal
    _maximum_rt_distance: float = float("inf")

    do_debug: bool = False
    call_counts: Counter = field(default_factory=Counter)
    solved_trees = 0

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

        # # NOTE: Debugging - capture removal reduction function
        # # Translate from waypoint_key to graph node index out of band to use here.
        # nodes_set = set(self.graph.node_indices())
        # if any(i not in nodes_set for i in [393, 396, 421]):
        #     logger.error(f"Missing node: {[i for i in [393, 396, 421] if i not in nodes_set]}")

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
                    logger.error(f"Unreachable ts pair: {self.get_node_key(r)} → {self.get_node_key(t)}")
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
            path_lengths = [self.shortest_path_length(r, t) for t in terminals]

            # Compute the maximum distance to terminals
            max_rt = max(path_lengths)
            if max_rt < best_value:
                best_value = max_rt
                best_root = r

        return best_root

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

    def get_maximum_rt_distance(self) -> float:
        """Returns the maximum root to terminal distance."""
        max_rt_distance = 1
        for r, terminals in self.terminal_sets.items():
            for t in terminals:
                if r == self.super_root_index:
                    # Ensure path goes from terminal to root to handle super-root reachability
                    max_rt_distance = max(max_rt_distance, self.shortest_path_length(t, r))
                else:
                    max_rt_distance = max(max_rt_distance, self.shortest_path_length(r, t))

        return max_rt_distance

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

    def reduce_demand_roots(self) -> int:
        """
        Reduces the number of distinct roots by merging fixed nodes (terminals or roots)
        that lie in the same connected component.

        Handles:
        > **`𝑻 ⭆ 𝑻', |𝑻'𐞪| ≤ |𝑻𐞪|, 𝐃 ≡ 𝐃'`**

        A subset of G consisting of the roots and terminals of all active demands with any
        connecting edges amoungst them is taken and augmented with direct edges between each
        terminal and its root. The resulting connected components then define the reduced
        terminal sets.

        Super Terminals are treated as independent of their root (since any super terminal
        can be connected to any potential root).

        This does not reduce the number of terminal-root pairs. It canonicalizes the root
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
            terminal_sets.pop(self.super_root_index)
            self.graph.remove_node(self.super_root_index)
            self.super_root_index = None
            self.potential_roots = set()

        self.terminal_sets = terminal_sets

        merges = num_roots - len(terminal_sets)

        if merges > 0:
            logger.info(f"  merged {merges} roots")
            if self.do_debug:
                self.dump_state("reduce_demand_roots", merges)
                self.validate_reachability()

        return merges

    def reduce_roots_via_articulation_points(self) -> int:
        """Merge roots whose connectivity is separated by cut vertices in the full graph."""
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
                    if rep == 476:
                        print("DEBUG")

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
        > **`⁝Ⓣʲ¹ ⇋ Ⓣʲ²⁝  ⭆  ⁝Ⓣʲ¹⁝, 𝐆 ∖ Ⓣʲ², 𝓢 ⋃ Ⓣʲ²(⦿)}`**
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

    def reduce_degree1_steiner_nodes(self) -> int:
        """Multi-pass degree-1 Steiner node reduction.

        If a Steiner node is a leaf, it cannot satisfy or bridge a demand.

        Handles:
        > **`|Ⓢ ⇋ ○⁝  ⭆  ○⁝, 𝐆 ∖ Ⓢ`**
        """
        logger.trace("reduce_degree1_steiner_nodes...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        removables = []
        while deg1_nodes := {i for i in self.graph.node_indices() if self.reduction_degree(i) == 1}:
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
        """Single-pass degree-1 terminal reduction.

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
        """Single-pass degree-1 root reduction.

        If a root node with unsatisfied demand is a leaf it must use its neighbor to satisfy a demand.

        Handles:
        > **`|Ⓡ ⇋ Ⓢ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ Ⓢ, 𝓢 ⋃ {Ⓢ, Ⓢ(⦿)}`**
        > **`|Ⓡᵏ ⇋ Ⓣᵏ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ Ⓣᵏ, 𝓢 ⋃ Ⓣᵏ(⦿), 𝓐ᵏ ∖ Ⓣᵏ}`**
        > **`|Ⓡᵏ ⇋ Ⓣʲ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ Ⓣʲ, 𝓢 ⋃ Ⓣʲ(⦿), 𝓐ᵏ ⋃ {𝓐ʲ ∖ Ⓣʲ}`**
        > **`|Ⓡᵏ ⇋ 𝕥ʲ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ 𝕥ʲ, 𝓢 ⋃ 𝕥ʲ(⦿), 𝕊 ∖ 𝕥`**
        """
        logger.trace("reduce_degree1_roots...")

        # NOTE: Potential roots are not considered as roots here because a potential
        #       root absorbs rather than consumes its neighbors and can only absorb
        #       a 2-degree neighbor since if it consumes a neighbor with deg() > 2
        #       it can end up in the optimal path of a different demand set which
        #       would incur the cost of the previously absorbed neighbors in its
        #       hyper node contents which may not be part of the optimal solution.

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

    def merge_adjacent_degree2_steiner_chains(self) -> int:
        """Merge degree-2 Steiner chains that are adjacent to each other.

        If a degree-2 Steiner node is adjacent to another degree-2 Steiner node they
        can only satisfy a demand by bridging both Steiner nodes, so they can be
        merged by absorption.

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

    def reduce_degree2_steiner_dominance(self) -> int:
        """
        Removes Steiner nodes of degree 2 that are dominated by alternate paths.

        Handles:
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝ ⭆  ⁝ⓤ ⇋ ⓥ⁝, 𝐆 ∖ Ⓢ  ⇔ 𝔀(ⓤ ↠∣Ⓢ∣↠ ⓥ) <= 𝔀(ⓤ ⇋ Ⓢ ⇋ ⓥ)`**
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
                if graph.out_degree(node) != 2:
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

    def reduce_degree2_articulation(self) -> int:
        """
        Processes degree-2 articulation point / 2-bridge nodes for inclusion/exclusion.

        If an articulation node's removal violates any demands it must be included in the solution, if
        its removal does not violate any demands and no super terminal would prefer it in an optimal
        solution then it is safe to be excluded from the solution.

        Handles:
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝  ⭆  ⁝ⓤ ∣ ⓥ⁝, 𝐆 ∖ Ⓢ  ⇔  Ⓡ (↠∣Ⓢ∣↠) Ⓣ ∀ 𝐃`**
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝  ⭆  ⁝ⓤ ⇋ ⓥ⁝, ⓤ ↣ ○  ⇔  Ⓡ ¬(↠∣Ⓢ∣↠) Ⓣ ∀ 𝐃, dist(𝕥, 𝕡) <= dist(𝕥, Ⓢ)`**
        """
        logger.trace("reduce_degree2_articulation...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        fixes = 0
        removals = 0

        def is_potential_violation(u: int) -> bool:
            from collections import deque

            if self.super_root_index is None:
                return False

            for t in self.terminal_sets[self.super_root_index]:
                queue: deque[int] = deque([t])
                visited = {t}
                while queue:
                    curr = queue.popleft()
                    if curr == u:
                        return True
                    if curr in non_steiners:
                        break

                    for neigh in self.graph.successor_indices(curr):
                        if neigh not in visited:
                            visited.add(neigh)
                            queue.append(neigh)

            return False

        # Get articulation points
        tmp_undir = self.graph.to_undirected()
        if self.super_root_index is not None:
            tmp_undir.remove_node(self.super_root_index)
        articulations = rx.articulation_points(tmp_undir)
        if not articulations:
            return 0

        # Map back to self.graph indices
        tmp_map = {i: u for i, u in enumerate(self.graph.node_indices())}
        articulations = {tmp_map[i] for i in articulations}

        articulations = {u for u in articulations if u not in non_steiners and self.reduction_degree(u) == 2}

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

            violates_demand = any(
                not rx.has_path(tmp, r, t) for r, terminals in self.terminal_sets.items() for t in terminals
            )

            if violates_demand:
                # Must include
                self.consume(u, a if a in self.fixed_nodes else b)
                fixes += 1
            elif not is_potential_violation(u):
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
        """Bridge reduction for Steiner Forest."""
        logger.trace("reduce_steiner_bridges...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        removals = 0
        edge_removals = 0

        # Get bridges
        tmp_undir = self.graph.to_undirected(multigraph=False)
        tmp_map = {i: u for i, u in enumerate(self.graph.node_indices())}

        if self.super_root_index is not None:
            undir_super_root_index = next(iter([i for i, j in tmp_map.items() if j == self.super_root_index]))
            tmp_undir.remove_node(undir_super_root_index)

        bridges = rx.bridges(tmp_undir)
        if not bridges:
            return 0

        bridges = sorted((tmp_map[min(u, v)], tmp_map[max(u, v)]) for u, v in bridges)  # ty:ignore[invalid-assignment]
        logger.trace(f"  {bridges=}")

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

            # Since super root was removed any super terminal test would violate which
            # could potentially cause extra edges to be forced in the solution...
            violates_demand = any(
                not rx.has_path(tmp, r, t)
                for r, terminals in self.terminal_sets.items()
                for t in terminals
                if r != self.super_root_index
            )

            if not violates_demand:
                # In the presence of super root we can't certify exclusion without testing nearby super terminals
                # or the presence of a potential root in the component and since hanging clusters consolidate
                # towards roots/potential roots that would almost always be the case. So, we just skip for
                # now instead of incurring the ovehead of super terminal testing.
                if self.super_root_index is not None:
                    continue

                # NOTE: We can't certify exclusion of either endpoint of a single bridge because it says nothing
                #       about the two endpoints in an optimal Solution. And in a node weighted problem if both
                #       end up being selected then the edge is naturally selected.
                #       But, if the created component from removing the bridge contains no terminals or roots
                #       or only one arborescence then it is safe to remove the bridge edge.
                #       In that case we can remove the edge but not consume either endpoint.
                sccs = rx.strongly_connected_components(tmp)
                for scc in sccs:
                    set_scc = set(scc)

                    # If the component contains no fixed nodes then it is safe to remove
                    if not set_scc & self.fixed_nodes:
                        # The component contains no fixed nodes, thus no-demand and no structure violations
                        self.graph.remove_edge(u, v)
                        self.graph.remove_edge(v, u)
                        edge_removals += 1
                        logger.trace(
                            f"  removed bridge edge: no fixed nodes: ({self.get_node_key(u)}, {self.get_node_key(v)})"
                        )
                        break

                    scc_roots = set_scc & set(self.terminal_sets.keys())
                    if len(scc_roots) == 1:
                        # A single root is present, so if all of its terminals are also present then it is safe to remove
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
            # Consume Steiner into the other side
            if steiner_u:
                self.consume(u, v)
                logger.trace(f"  consumed Steiner bridge node: {self.get_node_key(u)}")
            else:
                self.consume(v, u)
                logger.trace(f"  consumed Steiner bridge node: {self.get_node_key(v)}")
            removals += 1

        if removals > 0:
            logger.info(
                f"  consumed {removals} Steiner bridge nodes and removed {edge_removals} bridge edges"
            )
            if self.do_debug:
                self.dump_state("reduce_bridge_steiner_consumption", removals + edge_removals)
                self.validate_reachability()

        return removals

    def reduce_degreek_steiner_dominance(self, max_degree: int = 4) -> int:
        """
        Removes Steiner nodes of small degree k that are dominated by alternate paths.
        Strict dominance is used in the presence of fixed nodes.

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
        """Absorbs degree-2 Steiner nodes adjacent to degree-3 Steiner triangles.

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
        """Merge roots whose arborescences block another root's demand.

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
        """
        General reduction for enclosed Steiner nodes surrounded by
        degree-2 terminals of the same arborescence.

        If there exists at least one unambiguous outer Steiner endpoint u*
        such that w(e) <= w(u*), then e must be purchased in any optimal
        solution (even if all other outer endpoints are "free" due to
        external sharing).

        This generalizes both the degree-2 and degree-3 enclosed cases.
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

    def reduce_degree2_outer_face_steiners(self) -> int:
        logger.trace("reduce_degree2_outer_face_steiners...")
        # 15 and 8 worked well but took a bit too long (for Python that is expected)
        MAX_OUTER_WINDOW_SIZE = 10
        MAX_INTERFACES = 7
        MAX_FULLY_INTERNAL_ROOTS = 3

        def get_ordered_outer_adjacent_faces(
            embedding: nx.PlanarEmbedding, outer_walk: list[int], outer_face_set: set[int]
        ) -> list[list[int]]:
            last_seen = ()
            ordered = []
            n = len(outer_walk)

            outer_face_key = tuple(sorted(outer_face_set))

            for i in range(n):
                u = outer_walk[i]
                v = outer_walk[(i + 1) % n]

                # traverse the face on the OTHER side of the outer edge
                face = embedding.traverse_face(v, u)
                face_key = tuple(sorted(face))
                if face_key != last_seen and face_key != outer_face_key:
                    ordered.append(face)
                    last_seen = face_key

            return ordered

        non_steiner = self.non_steiner_nodes()

        # --- START: Rustworkx <-> networkx (prototyping-only) ---
        nx_to_rx = {i: j for i, j in enumerate(self.graph.node_indices())}
        rx_to_nx = {v: k for k, v in nx_to_rx.items()}

        Gx = nx.Graph()
        for u, v in self.graph.edge_list():
            if not Gx.has_edge(rx_to_nx[u], rx_to_nx[v]):
                Gx.add_edge(rx_to_nx[u], rx_to_nx[v])

        is_planar, embedding = nx.check_planarity(Gx)
        if not is_planar:
            return 0
        assert isinstance(embedding, PlanarEmbedding)  # For linter

        # Simply here for debugging...
        self.dump_graph("reduce_degree2_outer_face_steiners")

        # --- outer-face extraction ---

        # Iterate through all half-edges to find every face boundary
        visited_edges = set()
        faces = {}
        for edge in embedding.edges():
            u, v = edge
            if edge not in visited_edges:
                face = embedding.traverse_face(u, v, visited_edges)
                faces[edge] = face
        if not faces:
            return 0

        # Select the face with the largest number of nodes as the outer face
        outer_face_entry = max(faces.items(), key=lambda x: len(x[1]))
        outer_face_list = outer_face_entry[1]

        # Filter to all-outer adjacent faces...
        outer_face_set = set(outer_face_list)
        for k in [k for k, f in faces.items() if not any(i in outer_face_set for i in f)]:
            del faces[k]

        # We don't want the outer face to be iterated over during the reduction...
        del faces[outer_face_entry[0]]

        # We need ordering for windowed left, middle, right faces...
        ordered_outer_adjacent_faces = get_ordered_outer_adjacent_faces(
            embedding, outer_face_list, outer_face_set
        )

        # Translate all networkx indices to rustworkx indices
        outer_face_list = [nx_to_rx[i] for i in outer_face_list]
        outer_face_set = set(outer_face_list)
        outer_face_degree2_steiners = {
            i for i in outer_face_list if self.reduction_degree(i) == 2 and i not in non_steiner
        }
        ordered_outer_adjacent_faces = [{nx_to_rx[i] for i in f} for f in ordered_outer_adjacent_faces]

        # We're done with the embedding and face edge traversals.
        # From here on out everything graph related is rustworkx
        # --- END: networkx <-> rustworkx ---

        # --- Dreyfus-Wagner static data ---
        all_inner_and_outer_face_nodes = outer_face_set | set().union(*ordered_outer_adjacent_faces)
        adj_map: dict[int, set[int]] = {n: set() for n in all_inner_and_outer_face_nodes}
        node_weight_map: dict[int, int] = {}
        for u in all_inner_and_outer_face_nodes:
            nbrs = self.reduction_neighbors(u)
            for v in nbrs:
                if v in all_inner_and_outer_face_nodes:
                    adj_map[u].add(v)
            node_weight_map[u] = int(self.graph[u][PAYLOAD_WEIGHT_KEY])

        def steiner_tree_nodes(terminals: tuple[int, ...]) -> set[int]:
            """Invokes the Rust DW solver, returns the OPT_ST solution."""
            terminals_list = list(terminals)
            cost, _d_nodes, solution_nodes = nwst_dw.solve_nwst(
                adj_map, node_weight_map, terminals_list, terminals_list[0]
            )
            if cost == 18446744073709551615:  # Rust u64::MAX representation for INF
                logger.error("  Unsolvable Steiner tree!")
                raise RuntimeError("Steiner tree is empty!")
            return set(solution_nodes)

        # NOTE: This is simply for out of band prototype visualization...
        if self.do_debug:
            logger.warning(
                f"  All inner and outer face nodes: {[self.get_node_key(i) for i in all_inner_and_outer_face_nodes]}"
            )

        # --- Reduction Processing ---
        #
        # NOTE: See the code, doc-strings and commentary of the enclosed DW Steiner and terminal
        #       cluster reduction functions for more details on the DW enclosure logic.
        #
        # NOTE: When all of a root's demands are enclosed within the composite region C,
        #       then any other internal structuring within the composite region creates
        #       ambiguity for that root's arbor and potential route sharing.
        #       We handle this for cases up to 2 small arbors.
        #

        terminal_to_root = {t: r for r, ts in self.terminal_sets.items() for t in ts}
        terminal_to_root.update({r: r for r in self.terminal_sets})

        # Witness preservation
        keep: set[int] = set()
        removables = set()
        n = len(ordered_outer_adjacent_faces)
        if n < 3:
            return 0

        for i in range(n):
            F_left = ordered_outer_adjacent_faces[(i - 1) % n]
            F_middle = ordered_outer_adjacent_faces[i]
            F_right = ordered_outer_adjacent_faces[(i + 1) % n]
            if self.do_debug:
                logger.warning(f"  Outer adjacent face #{i}:")
                logger.warning(f"    F_left: {[self.get_node_key(i) for i in F_left]}")
                logger.warning(f"    F_middle: {[self.get_node_key(i) for i in F_middle]}")
                logger.warning(f"    F_right: {[self.get_node_key(i) for i in F_right]}")

            # NOTE: F_left and F_right must be disjoint to ensure that all ambiguity is handled
            #       within the composite region C, if they are not disjoint then it could still
            #       be a witness preserving case but handling is needed for different root cases
            #       (see below and the doc-strings of the enclosed DW Steiner and terminal cluster
            #       reduction functions for more details).
            if not F_left.isdisjoint(F_right):
                logger.debug("    skipping because F_left and F_right are not disjoint")
                continue

            # NOTE: There must be some degree-2 Steiner on the middle outer face segment for us to potentially remove.
            f_middle_candidates = F_middle & outer_face_degree2_steiners
            if not f_middle_candidates:
                continue

            # Build composite region C
            C_nodes = set(F_left) | set(F_middle) | set(F_right)
            if tuple(sorted(C_nodes)) in self._seen_nonreducible_outer_windows:
                if self.do_debug:
                    logger.warning(f"    skipping seen before: {[self.get_node_key(i) for i in C_nodes]}")
                continue

            # All non-steiner nodes in C must exist for each witnessed composite OPT_ST to exist
            # so even though we can solve the DW for disjoint groups of these nodes, we can't
            # ever omit any of them as they act as anchors.
            anchors = C_nodes & non_steiner
            if self.do_debug:
                logger.warning(f"      anchors: {[self.get_node_key(i) for i in anchors]}")

            # Active roots are those which have a demand represented within the composite region C
            active_roots = {terminal_to_root[t] for t in anchors}

            # Small terminal set clusters may have all demands within the composite region C.
            # Potential roots are never considered as fully internal roots: `.get(r, [])`
            fully_internal_roots = {r for r in active_roots if (set(self.terminal_sets[r]) | {r}) <= C_nodes}
            if len(fully_internal_roots) > MAX_FULLY_INTERNAL_ROOTS:
                if self.do_debug:
                    logger.warning(
                        f"      skipping fully internal: {[self.get_node_key(i) for i in fully_internal_roots]}"
                    )
                continue

            # Independent enclosed-root composites:
            #   𝓐¹, 𝓐², ..., 𝓐ⁿ
            # followed by a single aggregate exterior composite of remaining anchors:
            #   𝓐ⁿ⁺¹
            # The exterior aggregate preserves adversarial coexistence
            # without reintroducing full root activation combinatorics.
            anchor_sets: list[set[int]] = []
            if fully_internal_roots:
                # 𝓐¹, 𝓐², ..., 𝓐ⁿ
                for root in fully_internal_roots:
                    anchor_sets.append({t for t in anchors if terminal_to_root.get(t, t) == root})

                # 𝓐ⁿ⁺¹
                # exterior = anchors - set().union(*anchor_sets)
                # if exterior:
                anchor_sets.append(anchors)

                if self.do_debug:
                    reconstructed = set().union(*anchor_sets)
                    if reconstructed != anchors:
                        logger.error(
                            "      fully internal roots mismatch: "
                            f"{[self.get_node_key(r) for r in fully_internal_roots]}"
                        )
                        continue
            else:
                anchor_sets = [anchors]

            # NOTE: This ensures that a route _must_ make it to the outer middle face from the inside.
            #       In the case of fully_internal_roots only one anchor must be on the outer middle face.
            # TODO: Find a way to determine when this is truly required to eliminate ambiguity because when
            #       we can reduce without an anchor on the outer middle face we can unlock far more cascading
            #       reductions and fully collapse many problems.
            f_middle_has_anchor = F_middle & outer_face_set & anchors
            if not f_middle_has_anchor:
                if self.do_debug:
                    logger.warning(
                        f"      skipping no anchor: {[self.get_node_key(i) for i in f_middle_has_anchor]}"
                    )
                continue

            # Each non-anchor node with a neighbor in the full graph that is not in the composite
            # region C is an interface and each interface must be considered for composite witness
            # preserving cases.
            interfaces = {i for i in C_nodes if any(n not in C_nodes for n in self.reduction_neighbors(i))}
            # We also need to ensure that any fixed node that are no longer 'active' _but_ are still in
            # the composite region C are also considered interfaces.
            interfaces |= self.fixed_nodes & C_nodes
            interfaces -= anchors
            if self.do_debug:
                logger.warning(f"      interfaces: {[self.get_node_key(i) for i in interfaces]}")

            # Performance sanity checks, the DW undergoes a combinatorial explosion for each terminal
            # (anchor + interface) node, and the interface loop also has the combinatorial explosion
            # of all possible interface/anchor set subsets.
            if (
                len(interfaces) + len(anchors) > MAX_OUTER_WINDOW_SIZE
                or len(interfaces) < 2
                or len(interfaces) > MAX_INTERFACES
            ):
                continue

            interface_list = sorted(interfaces)

            # --- DW Witness preserving cases ---
            # We need to enumerate all C(I, 0..=|I|) interface combinations for each of 𝓐ʲ, 𝓐ᵏ, 𝓐ʲᵏ
            for subset_mask in range(1 << len(interface_list)):
                active_interfaces = {
                    interface_list[i] for i in range(len(interface_list)) if subset_mask & (1 << i)
                }

                for witness_anchors in anchor_sets:
                    terminals = tuple(sorted(witness_anchors | active_interfaces))
                    if len(terminals) < 2:
                        continue
                    if self.do_debug:
                        logger.warning(f"      terminals: {[self.get_node_key(i) for i in terminals]}")
                    nodes = steiner_tree_nodes(terminals)
                    keep |= nodes & F_middle

            removable_in_middle = (F_middle - keep) & f_middle_candidates
            if self.do_debug:
                logger.warning(
                    f"    removable_in_middle: {[self.get_node_key(i) for i in removable_in_middle]}"
                )
            for i in removable_in_middle:
                if self.reduction_degree(i) != 2:
                    logger.error(f"ERROR: Node {self.get_node_key(i)} has bad degree!")

            if removable_in_middle:
                removables |= removable_in_middle
                break
            else:
                self._seen_nonreducible_outer_windows.add(tuple(sorted(C_nodes)))

            if removables:
                break

        if removables:
            self.remove_nodes_from(removables)
            logger.warning(f"  => removed {len(removables)} outer face Steiner nodes")
            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(n) for n in removables)}")
                self.dump_state("reduce_outer_face_steiner_nodes", len(removables))
                self.validate_reachability()

        return len(removables)

    def reduce_enclosed_steiner_clusters(self) -> int:
        """
        Removes Steiner nodes provably absent from every optimal Steiner tree
        realization over every interface subset of recursively decomposed
        Steiner components.

        Large interface components are recursively decomposed across bridge edges
        until the interface count becomes tractable for exact Dreyfus-Wagner
        certification.

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
            if cost == 18446744073709551615:  # Matches Rust u64::MAX representation for INF
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

        if len(component) < 6 or len(interfaces) < 2:
            return None

        result = self._find_and_split_bridge_edge_components(component)
        if result is not None:
            # logger.trace("      found single edge bridge split")
            left_component, right_component, bridge_u, bridge_v = result
            return left_component, right_component, {bridge_u}, {bridge_v}

        rxG = subgraph_stable(self.graph, component)
        result = self.find_and_split_2_edge_components(rxG, PAYLOAD_WEIGHT_KEY)
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

    def find_and_split_2_edge_components(
        self, digraph: rx.PyDiGraph, weight_attr: str = "weight"
    ) -> tuple[rx.PyDiGraph, rx.PyDiGraph, set[int], set[int]] | None:
        """
        Finds 2-edge cuts using index-aligned local subgraphs, safely maps endpoints
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
            h = random.getrandbits(64)
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

    def reduce_roots_by_distance(self) -> int:
        """
        Root arbor reduction by distance.

        Following from the reduce_steiner_nodes_by_distance() method where:
        For each Steiner node v: if its weight is greater than the longest root -> terminal path,
        v is dominated and can be removed.
        Also, for each Steiner node v: if the _shortest_ distance to a terminal is greater than the
        longest root -> terminal path, v is dominated and can be removed.

        We can safely state that:
        For each Root node v: if the weight of the gap between any of its terminals (and root) is greater than
        the longest root -> terminal path, then the arbor is isolated and that tree can be solved exactly.

        Returns number of collapsed nodes.
        """
        logger.trace("reduce_steiner_nodes_by_distance...")

        self.set_edge_weights()
        max_rt_distance = min(self._maximum_rt_distance, self.get_maximum_rt_distance())
        if max_rt_distance != self._maximum_rt_distance and max_rt_distance > 0:
            logger.trace(
                f"  => max_rt_distance changed from {self._maximum_rt_distance} to {max_rt_distance}"
            )
            self._maximum_rt_distance = int(max_rt_distance)
        logger.trace(f"  => max_rt_distance: {max_rt_distance}")

        all_consumed = set()

        # Test in reverse order by size of terminal set...
        sr_index = self.super_root_index
        for root in sorted(self.terminal_sets.keys(), key=lambda r: len(self.terminal_sets[r]), reverse=True):
            if root == sr_index:
                continue

            # We now need the minimum gap for this particular cluster set (terminals|{root}) to any other terminal/root
            # not in this cluster...
            cluster = self.terminal_sets[root] | {root}

            gap = float("inf")
            self.set_edge_weights()

            all_others = set().union(*self.terminal_sets.values()) | set(self.terminal_sets.keys())
            all_others -= cluster
            all_others.discard(self.super_root_index)

            # When all_others is empty it means cluster is the sole remaining terminal cluster
            # and gap would be infinite so we can skip calcuting it.
            # NOTE: We can do this more efficiently by using a BFS after prototyping...
            if all_others:
                for u in cluster:
                    for v in all_others:
                        gap = min(gap, self.shortest_path_length(u, v))

            if gap <= max_rt_distance:
                continue

            logger.debug(
                f"  => identified isolated terminal cluster for root {self.get_node_key(root)} with {len(cluster)} terminals (gap: {gap})..."
            )

            consumed = set()

            # Then we'll solve the tree to get the witnessed solution nodes...
            if len(cluster) <= DW_MAX_TREE_TERMINALS:
                witnessed_nodes = self.solve_root_with_dw(root)
            else:
                witnessed_nodes = self.solve_root_with_scipstp(root)

            # Now consume all witnessed nodes, from the root outward; leaving the root node in the graph.
            witnessed_to_remove = set(witnessed_nodes)
            while len(witnessed_to_remove) > 0:
                neighbors = self.reduction_neighbors(root)
                neighbors.discard(self.super_root_index)
                neighbors = neighbors & witnessed_to_remove
                if len(neighbors) == 0:
                    break

                # Places the consumed node and hyper node contents into the fixed solution set
                for n in neighbors:
                    self.consume(n, root)
                    consumed.add(n)
                    witnessed_to_remove.discard(n)

            all_consumed.update(consumed)

            logger.trace(f"    witnessed: {sorted(self.get_node_key(n) for n in witnessed_nodes)}")

            # Handle any settled super-root terminals
            if self.super_root_index is not None:
                settled_super_terminals = {
                    i for i in self.terminal_sets[self.super_root_index] if i in witnessed_nodes
                }
                for t in settled_super_terminals:
                    self.terminal_sets[self.super_root_index].remove(t)
                if len(self.terminal_sets[self.super_root_index]) == 0:
                    self.terminal_sets.pop(self.super_root_index)
                    self.super_root_index = None

            # Lastly remove the terminal set cluster
            self.terminal_sets.pop(root)

        if all_consumed:
            logger.debug(f"  => consumed {len(all_consumed)} witnessed nodes")
            if self.do_debug:
                logger.trace(f"    consumed: {sorted(self.get_node_key(n) for n in all_consumed)}")
                self.dump_state("reduce_roots_by_distance", len(all_consumed))
                self.validate_reachability()

        return len(all_consumed)

    def reduce_steiner_nodes_by_distance(self) -> int:
        """
        Steiner node reduction by distance.

        For each Steiner node v: if its weight is greater than the longest root -> terminal path, v is dominated and can be removed.
        Also, for each Steiner node v: if the _shortest_ distance to a terminal is greater than the longest root -> terminal path, v is dominated and can be removed.

        Returns number of removed/absorbed nodes.
        """
        logger.trace("reduce_steiner_nodes_by_distance...")

        self.set_edge_weights()
        non_steiner_nodes = self.non_steiner_nodes()

        max_rt_distance = min(self._maximum_rt_distance, self.get_maximum_rt_distance())
        if max_rt_distance != self._maximum_rt_distance and max_rt_distance > 0:
            logger.trace(
                f"  => max_rt_distance changed from {self._maximum_rt_distance} to {max_rt_distance}"
            )
            self._maximum_rt_distance = int(max_rt_distance)
        logger.trace(f"  => max_rt_distance: {max_rt_distance}")

        removables = []
        removed_by_weight = 0
        removed_by_distance = 0

        for v in self.graph.node_indices():
            if v in non_steiner_nodes or v == self.super_root_index:
                continue

            if self.graph[v][PAYLOAD_WEIGHT_KEY] > max_rt_distance:
                logger.trace(f"  => removing Steiner node {self.get_node_key(v)} by weight")
                removables.append(v)
                removed_by_weight += 1
                continue

            min_rt_distance = float("inf")
            for r, terminals in self.terminal_sets.items():
                for t in terminals | {r}:
                    min_rt_distance = min(min_rt_distance, self.shortest_path_length(v, t))

            if min_rt_distance > max_rt_distance:
                logger.trace(f"  => removing Steiner node {self.get_node_key(v)} by distance")
                removables.append(v)
                removed_by_distance += 1
                continue

        if removables:
            self.remove_nodes_from(removables)
            logger.info(
                f"  => removed {len(removables)} Steiner nodes [({max_rt_distance}) by weight: {removed_by_weight}, by distance: {removed_by_distance}]"
            )
            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(n) for n in removables)}")
                self.dump_state("reduce_steiner_nodes_by_distance", len(removables))
                self.validate_reachability()

        return len(removables)

    def reduce_local_steiner_dominance_by_distance(self, radius: int = 16, max_demand: int = 16) -> int:
        """
        Node-weighted local dominance reduction over roots ∪ terminals.

        For each Steiner node v:
        - collect nearby demand nodes U_v within node-weighted distance <= radius
        - if |U_v| <= max_demand:
            check whether v ever improves any pair (a, b) in U_v
            if not, v is locally dominated and can be removed/absorbed
        Returns number of removed/absorbed nodes.
        """
        logger.trace("reduce_local_steiner_dominance_by_distance...")
        removals = []

        node_weight_map = {node: self.graph[node][PAYLOAD_WEIGHT_KEY] for node in self.graph.node_indices()}

        def dijkstra_avoiding(start: int, forbidden: int) -> dict[int, float]:
            """Node-weighted Dijkstra that forbids stepping on `forbidden`."""
            import heapq

            if start == forbidden:
                return {}  # unreachable

            dist = {start: node_weight_map[start]}
            heap = [(node_weight_map[start], start)]

            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue

                for v in self.reduction_neighbors(u):
                    if v == forbidden:
                        continue

                    nd = d + node_weight_map[v]
                    if nd < dist.get(v, float("inf")):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))

            return dist

        # Build demand set: roots ∪ terminals
        roots = set(self.terminal_sets.keys())
        terminals = set().union(*self.terminal_sets.values()) if self.terminal_sets else set()
        demand_nodes = roots | terminals
        if not demand_nodes:
            return 0

        # Precompute node weights for quick access
        weight = {u: self.graph[u][PAYLOAD_WEIGHT_KEY] for u in self.graph.node_indices()}

        for v in list(self.graph.node_indices()):
            # Skip non-Steiner nodes
            if v in demand_nodes:
                continue

            # 1) Collect local neighborhood around v up to node-weighted distance <= radius
            U_v = self._collect_local_demand_neighborhood(v, demand_nodes, weight, radius)
            if len(U_v) < 2 or len(U_v) > max_demand:
                continue  # too trivial or too large to bother

            # 2) Precompute distances from v (normal local Dijkstra)
            d_from_v = self._local_dijkstra(v, radius, weight)

            # 3) For each a in U_v, compute distances avoiding v
            d_without_v = {a: dijkstra_avoiding(a, v) for a in U_v}

            # 4) Check if v is ever strictly better for any pair (a, b) in U_v
            v_useful = False
            U_v_list = list(U_v)

            for i in range(len(U_v_list)):
                a = U_v_list[i]

                # If a cannot reach v locally, skip pairs involving a
                if a not in d_from_v:
                    continue

                for j in range(i + 1, len(U_v_list)):
                    b = U_v_list[j]

                    # If b cannot reach v locally, skip
                    if b not in d_from_v:
                        continue

                    # Cost via v
                    cost_v = d_from_v[a] + d_from_v[b] - weight[v]

                    # Cost without v
                    dist_a = d_without_v[a]
                    cost_without = dist_a.get(b, float("inf"))

                    # If v provides a strictly better connection, it's useful
                    if cost_v < cost_without:
                        v_useful = True
                        break

                if v_useful:
                    break

            if v_useful:
                continue  # cannot dominate v

            # 5) v is locally dominated; remove it
            self.remove_node(v)
            removals.append(v)

        if removals:
            # self.dump_graph("pre-remove enclosed steiner cluster nodes", self.graph)
            # self.graph.remove_nodes_from(removables)
            logger.info(f"  => removed {len(removals)} distance-locally dominated steiner nodes")
            # self.dump_graph("post-remove enclosed steiner cluster nodes", self.graph)
            if self.do_debug:
                logger.trace(f"  {sorted(self.get_node_key(n) for n in removals)}")
                self.dump_state("reduce_local_steiner_dominance_by_distance", len(removals))
                self.validate_reachability()

        return len(removals)

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

    def solve_as_tree_super(self):
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances."""
        logger.trace("solve_as_tree...")

        if self.super_root_index is None:
            return 0

        complexity = 0  # 3^t * n + 2^t * n^2 for Dreyfus Wagner

        num_nodes = self.graph.num_nodes()
        num_roots = len(self.terminal_sets)
        num_terminals = len(set().union(*self.terminal_sets.values()))
        num_super_terminals = len(self.terminal_sets[self.super_root_index])

        for choose_count in range(1, num_roots + 1):
            for comb in combinations(self.terminal_sets.keys(), choose_count):
                num_terms = sum(len(self.terminal_sets[c]) for c in comb) + choose_count
                complexity += (3**num_terms * num_nodes) + (2**num_terms) * (num_nodes**2)

        # TODO: Change back to info level after seeing how well scipstp scales...
        logger.warning(
            f"    remaining complexity: {complexity}, |r|: {num_roots}, |t|: {num_terminals}, |st|: {num_super_terminals}, |n|: {num_nodes}"
        )

        presolve_solution, presolve_block_results, presolve_block_costs = self.pre_solve_with_scipstp_super()

        # --- Reassign witnessed super terminals ---
        # If there were any super terminals in the best solution then they can be unambiguously
        # reassigned to the root of the cluster they are witnessed in by removing them from the
        # super_root set and adding them to a root set it is connected to.
        num_reassigned_super_terminals = 0

        roots = set(self.terminal_sets.keys()) - {self.super_root_index}
        witnessed_super_terminals = {
            t for t in self.terminal_sets[self.super_root_index] if t in presolve_solution
        }

        solutionGraph = subgraph_stable(self.graph, presolve_solution)
        components = rx.weakly_connected_components(solutionGraph)

        if witnessed_super_terminals:
            for t in witnessed_super_terminals:
                self.terminal_sets[self.super_root_index].remove(t)
                if not self.terminal_sets[self.super_root_index]:
                    del self.terminal_sets[self.super_root_index]

                # Identify the root set that the super terminal is connected to
                for comp in components:
                    if t in comp:
                        r = set(comp).intersection(roots).pop()
                        logger.trace(
                            f"    super terminal {self.get_node_key(t)} is connected to root {self.get_node_key(r)}"
                        )
                        num_reassigned_super_terminals += 1
                        self.terminal_sets[r].add(t)
                        break

        logger.info(f"      {num_reassigned_super_terminals} super terminals reassigned to roots")

        # --- Collect remaining super terminal candidate roots ---
        # If there are any remaining super terminals we must not remove them from the graph
        # _nor_ allow them to become isolated terminals. Therefore, we need to identify the
        # demand sinks (roots, potential roots and terminals) in the neighborhood of each
        # super terminal and add them to the candidate roots for that super terminal.
        candidate_roots: dict[int, set[int]] = defaultdict(set)

        if self.super_root_index in self.terminal_sets:
            all_terminals = set().union(*self.terminal_sets.values())
            weight_map = {i: self.graph[i][PAYLOAD_WEIGHT_KEY] for i in self.graph.node_indices()}

            for t in self.terminal_sets[self.super_root_index]:
                # We add the super root since there is potential that the super terminal is
                # further from demand sinks than the neighborhood radius and super terminal
                # will ensure we have the nearest potential root.
                # NOTE: While 10 and 5 are arbitrary choices we could make them configurable,
                #       the important thing is the super root is included.

                # Root/Potential Root sinks
                root_sinks = self._collect_local_demand_neighborhood(t, self.potential_roots, weight_map, 10)
                # Ensure that at least one sink is in root sinks by utilizing the super root
                paths = rx.all_shortest_paths(self.graph, t, self.super_root_index)
                for path in paths:
                    for v in path:
                        if v in self.potential_roots:
                            root_sinks.add(v)
                            break
                candidate_roots[t].update(root_sinks)

                # Terminal sinks
                # We include the roots of the neighborhood terminals as well as potential sinks
                # since they could be closer and the root could be out of the neighborhood.
                terminal_sinks = self._collect_local_demand_neighborhood(t, all_terminals, weight_map, 5)
                terminal_roots = set()
                for t in terminal_sinks:
                    for r, ts in self.terminal_sets.items():
                        if t in ts:
                            terminal_roots.add(r)
                            break
                candidate_roots[t].update(terminal_roots)

            logger.warning(
                f"      {len(self.terminal_sets[self.super_root_index])} super terminals not reassigned to roots"
            )

        # Clean up potential roots
        # At this point any potential root that is not represented in the solution can be removed from the
        # class's potential roots list.
        num_prev_potential_roots = len(self.potential_roots)
        self.potential_roots.difference_update(candidate_roots)
        num_potential_roots_removed = num_prev_potential_roots - len(self.potential_roots)
        logger.warning(f"    removed {num_potential_roots_removed} potential roots")

        # If we have reassigned any super terminal or removed any potential roots then there is the
        # potential for the graph to further reduce in the pipeline reductions.
        if num_reassigned_super_terminals > 0 or num_potential_roots_removed > 0:
            return num_potential_roots_removed + num_reassigned_super_terminals

        # --- Solve as tree ---
        # Now that we have a stably reduced graph and have obtained the candidate roots for each
        # super terminal we can solve as a tree.

        start_time = time()
        solution = self.solve_with_scipstp_super(
            candidate_roots, presolve_block_results, presolve_block_costs
        )
        end_time = time()

        removables = set(self.graph.node_indices()) - set(solution)
        if self.super_root_index in removables:
            removables.remove(self.super_root_index)
        removals = len(removables)

        if removals > 0:
            duration = end_time - start_time
            logger.warning(
                f"    solved as tree in {duration:.2f}s with complexity {complexity}, |r|: {num_roots}, |t|: {num_terminals}, |st|: {num_super_terminals}, |n|: {num_nodes}"
            )

            self.graph.remove_nodes_from(removables)

            logger.info(f"    removed {removals} dead tree solution nodes")

            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(i) for i in removables)}")
                self.dump_state("solve_as_tree_super", removals)
                self.validate_reachability()
                self.dump_graph("after solving as tree")

        else:
            logger.error("    failed to solve as tree")

        return removals

    def solve_as_tree(self):
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances."""
        logger.trace("solve_as_tree...")

        if self.super_root_index is not None:
            return 0

        complexity = 0  # 3^t * n + 2^t * n^2 for Dreyfus Wagner

        num_nodes = self.graph.num_nodes()
        num_roots = len(self.terminal_sets)
        num_terminals = len(set().union(*self.terminal_sets.values()))

        for choose_count in range(1, num_roots + 1):
            for comb in combinations(self.terminal_sets.keys(), choose_count):
                num_terms = sum(len(self.terminal_sets[c]) for c in comb) + choose_count
                complexity += (3**num_terms * num_nodes) + (2**num_terms) * (num_nodes**2)

        # TODO: Change back to info level after seeing how well scipstp scales...
        logger.warning(
            f"    remaining complexity: {complexity}, |r|: {num_roots}, |t|: {num_terminals}, |n|: {num_nodes}"
        )

        # Exludes the MIP 'sweet spot'
        use_scip = self.terminal_sets and (num_nodes > 110 or num_roots < 6)
        start_time = time()
        if use_scip:
            logger.warning("      solving as tree composite")
            solution = self.solve_with_scipstp()

        else:
            logger.warning("      solving as MCF MIP using HiGHS solver...")
            return 0
        end_time = time()

        removables = set(self.graph.node_indices()) - set(solution)
        removals = len(removables)

        if removals > 0:
            duration = end_time - start_time
            logger.warning(
                f"    solved as tree in {duration:.2f}s with complexity {complexity}, |r|: {num_roots}, |t|: {num_terminals}, |n|: {num_nodes}"
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
                self.validate_reachability()
                self.dump_graph("after solving as tree")

        else:
            logger.error("    failed to solve as tree")

        return removals

    # MARK: Tree Solvers

    def solve_root_with_dw(self, root: int):
        """Solves a single root from the remaining reduced state graph."""
        import nwst_dw

        logger.trace("solve_root_as_dw...")

        adj_map = {u: set(self.graph.neighbors(u)) for u in self.graph.node_indices()}
        node_weight_map = {u: self.graph[u][PAYLOAD_WEIGHT_KEY] for u in self.graph.node_indices()}

        solver = nwst_dw.Solver(adj_map, node_weight_map)
        terminals_list = list(self.terminal_sets[root] | {root})

        self.solved_trees += 1
        cost, _, solution_nodes = solver.solve(terminals_list, root)
        if cost == 18446744073709551615:
            raise RuntimeError("Unsolvable Steiner tree!")

        if solution_nodes:
            logger.debug(
                f"    solved tree for root {self.get_node_key(root)} containing {len(terminals_list)} terminals with cost {cost}"
            )
            logger.trace(f"    solution: {sorted(self.get_node_key(n) for n in solution_nodes)}")

        return solution_nodes

    def solve_root_with_scipstp(self, root: int):
        """Solves a single root from the remaining reduced state graph."""
        logger.trace("solve_root_with_scipstp...")

        # Even though scipstp NWSTP is fully a node weighted setup the rustworkx graph isn't pickleable
        # so we need to pass in the adjacency map and node weights
        adj_map = {u: sorted(self.reduction_neighbors(u)) for u in self.graph.node_indices()}
        node_weight_map = {u: self.graph[u][PAYLOAD_WEIGHT_KEY] for u in self.graph.node_indices()}

        terminal_keys = [root]
        partitions = set_partitions(terminal_keys)

        valid_blocks = set()
        valid_partitions = []

        # 1. Sequential partition filtering loop
        for partition in partitions:
            is_valid = True
            for block in partition:
                block_key = tuple(sorted(block))
                if len(block) < 2 or block_key in valid_blocks:
                    continue
                block_reachable = True
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        if not rx.has_path(self.graph, block[i], block[j]):
                            block_reachable = False
                            break
                    if not block_reachable:
                        break
                if block_reachable:
                    valid_blocks.add(block_key)
                else:
                    is_valid = False
                    break
            if is_valid:
                valid_partitions.append(partition)

        # 2. Collect all unique blocks that actually exist within valid partitions
        unique_valid_blocks = set()
        for partition in valid_partitions:
            for block in partition:
                unique_valid_blocks.add(tuple(sorted(block)))

        block_results = {}

        # For index masking...
        nodes_list = list(self.graph.node_indices())
        node_index = {u: i for i, u in enumerate(nodes_list)}

        # DIMACS node ids are 1-based, contiguous
        dimacs_id = {u: i + 1 for i, u in enumerate(nodes_list)}
        inv_dimacs_id = {i + 1: u for i, u in enumerate(nodes_list)}

        for block_key in unique_valid_blocks:
            terminals = set()
            for k in block_key:
                terminals.add(k)
                terminals.update(self.terminal_sets[k])
            terminals_list = list(terminals)

            logger.trace(f"    solving block {block_key}...")

            self.solved_trees += 1
            block_key, cost, mask = solve_tree_using_scipstp(
                TreeProblem(
                    instance_id=self.instance_id,
                    block_key=block_key,
                    terminals=terminals_list,
                    adj_map=adj_map,
                    node_weight_map=node_weight_map,
                    node_index_map=node_index,
                    dimacs_id_map=dimacs_id,
                    inv_dimacs_id_map=inv_dimacs_id,
                    enable_super_root_index=0,
                    do_debug=self.do_debug,
                    mip_validation=False,
                )
            )

            logger.trace(
                f"    solved block {block_key} containing {len(terminals_list)} terminals with cost {cost}"
            )
            block_results[block_key] = mask

        # Phase 2: Sequential realization of partitions using memoized blocks
        best_cost = float("inf")
        best_solution_mask = 0

        for partition in valid_partitions:
            solution_mask = 0
            for block in partition:
                block_key = tuple(sorted(block))
                solution_mask |= block_results[block_key]

            # sum weights over bits
            total_cost = 0
            mask = solution_mask
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                u = nodes_list[idx]
                total_cost += node_weight_map[u]
                mask ^= lsb

            if total_cost < best_cost:
                best_cost = total_cost
                best_solution_mask = solution_mask

        best_solution = set()
        mask = best_solution_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            u = nodes_list[idx]
            best_solution.add(u)
            mask ^= lsb

        if best_solution:
            logger.debug(
                f"    solved for terminals: { {self.get_node_key(r): {self.get_node_key(n) for n in ts} for r, ts in self.terminal_sets.items()} }"
            )
            logger.debug(
                f"    found best solution (cost: {best_cost}): {sorted(self.get_node_key(n) for n in best_solution)}"
            )

        return best_solution

    def pre_solve_with_scipstp_super(self):
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances."""
        logger.trace("solve_with_scipstp...")

        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Even though scipstp NWSTP is fully a node weighted setup the rustworkx graph isn't pickleable
        # so we need to pass in the adjacency map and node weights
        adj_map = {u: sorted(self.reduction_neighbors(u)) for u in self.graph.node_indices()}
        node_weight_map = {u: self.graph[u][PAYLOAD_WEIGHT_KEY] for u in self.graph.node_indices()}

        terminal_keys = list(set(self.terminal_sets.keys()) - {self.super_root_index})
        root_bit = {r: 1 << i for i, r in enumerate(terminal_keys)}

        # 1. Partition filtering...
        logger.trace("    generating valid partition blocks...")

        valid_blocks = set()

        # Root reachability for component generation...
        logger.trace("      building reachability matrix...")
        reachable = {r: {r} for r in terminal_keys}
        for i, u in enumerate(terminal_keys):
            for v in terminal_keys[i + 1 :]:
                if rx.has_path(self.graph, u, v):
                    reachable[u].add(v)
                    reachable[v].add(u)

        # Connected root components generation...
        components = []
        remaining_roots = set(terminal_keys)
        while remaining_roots:
            root = next(iter(remaining_roots))
            component = sorted(reachable[root])
            components.append(component)
            remaining_roots -= reachable[root]

        logger.trace(
            f"      identified {len(components)} reachability component(s): {[len(c) for c in components]}"
        )

        # Since the partition DP reconstructs the optimal partitioning, we only need
        # to generate the valid blocks. Any valid block must be wholly contained
        # within a single reachability component.
        for component in components:
            logger.trace(f"      generating blocks for component size={len(component)}...")
            k = len(component)
            for mask in range(1, 1 << k):
                block = tuple(component[i] for i in range(k) if mask & (1 << i))
                valid_blocks.add(block)

        block_results = {}
        block_costs = {}

        num_roots = len(terminal_keys)

        # For index masking...
        nodes_list = list(self.graph.node_indices())
        node_index = {u: i for i, u in enumerate(nodes_list)}

        # DIMACS node ids are 1-based, contiguous
        dimacs_id = {u: i + 1 for i, u in enumerate(nodes_list)}
        inv_dimacs_id = {i + 1: u for i, u in enumerate(nodes_list)}

        # Determine if parallel block solving is warranted
        if num_roots >= SCIPSTP_PAR_ROOTS_THRESHOLD:
            # Map unique blocks to their full, flattened terminal sets for the solver
            logger.warning(f"    concurrently solving {len(valid_blocks)} unique valid blocks...")

            worker_tasks = []
            for block_key in valid_blocks:
                terminals = set()
                for k in block_key:
                    terminals.add(k)
                    terminals.update(self.terminal_sets[k])
                terminals = sorted(terminals)

                worker_tasks.append((block_key, terminals))

            self.solved_trees += len(worker_tasks)

            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        _worker_solve_single_block_scipstp,
                        self.instance_id,
                        task[0],
                        task[1],
                        adj_map,
                        node_weight_map,
                        node_index,
                        dimacs_id,
                        inv_dimacs_id,
                        enable_super_root_index=0,
                        do_debug=self.do_debug,
                    )
                    for task in worker_tasks
                ]

                for f in as_completed(futures):
                    block_key, cost, mask = f.result()
                    block_results[block_key] = mask
                    block_costs[block_key] = cost

        else:
            # Fallback to sequential block solving if below threshold
            # Since we still need to have a temporary file for each unique block and call the executable
            # we'll use the same helper functions as the concurrent version...
            logger.warning(f"    sequentially solving {len(valid_blocks)} unique valid blocks...")

            for block_key in valid_blocks:
                terminals = set()
                for k in block_key:
                    terminals.add(k)
                    terminals.update(self.terminal_sets[k])
                terminals_list = sorted(terminals)

                logger.trace(f"    solving block {block_key}...")

                self.solved_trees += 1

                cost, solution_nodes = _solve_single_block_scipstp_seq(
                    self.instance_id,
                    block_key,
                    terminals_list,
                    adj_map,
                    node_weight_map,
                    node_index,
                    dimacs_id,
                    inv_dimacs_id,
                    enable_super_root_index=0,
                    do_debug=self.do_debug,
                )

                logger.trace(
                    f"    solved block {block_key} containing {len(terminals_list)} terminals with cost {cost}"
                )
                logger.trace(f"    solution: {sorted(self.get_node_key(n) for n in solution_nodes)}")

                mask = 0
                for u in solution_nodes:
                    mask |= 1 << node_index[u]

                block_results[block_key] = mask
                block_costs[block_key] = cost

        # Phase 1: Build block mask tables
        logger.trace("    building block mask table...")

        block_mask_costs = {}
        block_mask_solutions = {}

        for block_key, solution_mask in block_results.items():
            block_mask = 0

            for r in block_key:
                block_mask |= root_bit[r]

            block_mask_costs[block_mask] = block_costs[block_key]
            block_mask_solutions[block_mask] = solution_mask

        # Phase 2: Sequential realization of partitions using memoized blocks
        logger.trace("    solving partition DP...")

        full_mask = (1 << len(terminal_keys)) - 1

        dp = [float("inf")] * (full_mask + 1)
        choice = [None] * (full_mask + 1)

        dp[0] = 0

        for state in range(full_mask + 1):
            if dp[state] == float("inf"):
                continue

            remaining = full_mask ^ state
            sub = remaining

            while sub:
                if sub in block_mask_costs:
                    new_cost = dp[state] + block_mask_costs[sub]

                    next_state = state | sub

                    if new_cost < dp[next_state]:
                        dp[next_state] = new_cost
                        choice[next_state] = (state, sub)

                sub = (sub - 1) & remaining

        best_cost = dp[full_mask]
        best_solution_mask = 0

        if best_cost == float("inf") or choice[full_mask] is None:
            logger.error("    no valid partition solution found")
            return set(), {}, {}

        state = full_mask

        while state:
            prev_choice = choice[state]

            if prev_choice is None:
                logger.error(f"    broken DP reconstruction at state {state}")
                break

            prev_state, block_mask = prev_choice

            best_solution_mask |= block_mask_solutions[block_mask]

            state = prev_state

        # Phase 3: Extract solution
        logger.trace("    extracting solution...")

        best_solution = set()
        mask = best_solution_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            u = nodes_list[idx]
            best_solution.add(u)
            mask ^= lsb

        if best_solution:
            logger.debug(
                f"    solved for terminals: { {self.get_node_key(r): {self.get_node_key(n) for n in ts} for r, ts in self.terminal_sets.items()} }"
            )
            logger.debug(
                f"    found best solution (cost: {best_cost}): {sorted(self.get_node_key(n) for n in best_solution)}"
            )

        return best_solution, block_results, block_costs

    def solve_with_scipstp_super(
        self,
        candidate_roots: dict[int, set[int]],
        presolve_block_results: dict[tuple[int, ...], int],
        presolve_block_costs: dict[tuple[int, ...], int],
    ):
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances."""
        logger.trace("solve_with_scipstp_super...")

        from concurrent.futures import ProcessPoolExecutor, as_completed

        assert self.super_root_index is not None, "super root index must be set"

        # Even though scipstp NWSTP is fully a node weighted setup the rustworkx graph isn't pickleable
        # so we need to pass in the adjacency map and node weights
        adj_map = {u: sorted(self.reduction_neighbors(u)) for u in self.graph.node_indices()}
        node_weight_map = {u: self.graph[u][PAYLOAD_WEIGHT_KEY] for u in self.graph.node_indices()}

        roots = list(set(self.terminal_sets.keys()) - {self.super_root_index})
        surviving_sts = sorted(candidate_roots.keys())
        coverage_entities = roots + surviving_sts
        coverage_bit = {u: 1 << i for i, u in enumerate(coverage_entities)}

        # Partition filtering...
        logger.trace("    generating valid partition blocks...")

        valid_blocks = set()

        # Root reachability for component generation...
        logger.trace("      building reachability matrix...")
        reachable = {r: {r} for r in coverage_entities}
        for i, u in enumerate(coverage_entities):
            for v in coverage_entities[i + 1 :]:
                if rx.has_path(self.graph, u, v):
                    reachable[u].add(v)
                    reachable[v].add(u)

        # Connected root components generation...
        components = []
        remaining_roots = set(coverage_entities)
        while remaining_roots:
            root = next(iter(remaining_roots))
            component = sorted(reachable[root])
            components.append(component)
            remaining_roots -= reachable[root]

        logger.trace(
            f"      identified {len(components)} reachability component(s): {[len(c) for c in components]}"
        )

        # Since the partition DP reconstructs the optimal partitioning, we only need
        # to generate the valid blocks. Any valid block must be wholly contained
        # within a single reachability component.
        for component in components:
            logger.trace(f"      generating blocks for component size={len(component)}...")
            k = len(component)
            for mask in range(1, 1 << k):
                block = tuple(component[i] for i in range(k) if mask & (1 << i))
                valid_blocks.add(block)

        block_results = dict(presolve_block_results)
        block_costs = dict(presolve_block_costs)

        unsolved_blocks = []
        for block_key in valid_blocks:
            if block_key in block_results:
                continue
            unsolved_blocks.append(block_key)

        num_roots = len(coverage_entities)

        # For index masking...
        nodes_list = list(self.graph.node_indices())
        node_index = {u: i for i, u in enumerate(nodes_list)}

        # DIMACS node ids are 1-based, contiguous
        dimacs_id = {u: i + 1 for i, u in enumerate(nodes_list)}
        inv_dimacs_id = {i + 1: u for i, u in enumerate(nodes_list)}

        # Determine if parallel block solving is warranted
        if num_roots >= SCIPSTP_PAR_ROOTS_THRESHOLD:
            # Map unique blocks to their full, flattened terminal sets for the solver
            logger.warning(f"    concurrently solving {len(valid_blocks)} unique valid blocks...")

            worker_tasks = []
            for block_key in unsolved_blocks:
                terminals = set()

                # A super terminal behaves as a terminal in the presence of another root,
                # yet it behaves as a root with a single terminal (the super root) when
                # there are no other roots. This enforces potential root transit when the
                # super terminals are considered in isolation.
                sr_index = 0  # disabled
                if all(i in surviving_sts for i in block_key):
                    terminals.add(self.super_root_index)
                    sr_index = self.super_root_index

                for k in block_key:
                    terminals.add(k)
                    if k not in surviving_sts:
                        terminals.update(self.terminal_sets[k])

                terminals = sorted(terminals)

                worker_tasks.append((block_key, terminals, sr_index))

            self.solved_trees += len(worker_tasks)

            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        _worker_solve_single_block_scipstp,
                        self.instance_id,
                        task[0],
                        task[1],
                        adj_map,
                        node_weight_map,
                        node_index,
                        dimacs_id,
                        inv_dimacs_id,
                        enable_super_root_index=task[2],
                        do_debug=self.do_debug,
                    )
                    for task in worker_tasks
                ]

                for f in as_completed(futures):
                    block_key, cost, mask = f.result()
                    block_results[block_key] = mask
                    block_costs[block_key] = cost

        else:
            # Fallback to sequential block solving if below threshold
            # Since we still need to have a temporary file for each unique block and call the executable
            # we'll use the same helper functions as the concurrent version...
            logger.warning(f"    sequentially solving {len(valid_blocks)} unique valid blocks...")

            num_blocks = len(unsolved_blocks)

            for block_n, block_key in enumerate(unsolved_blocks):
                terminals = set()

                # A super terminal behaves as a terminal in the presence of another root,
                # yet it behaves as a root with a single terminal (the super root) when
                # there are no other roots. The enforces potential root transit when the
                # super terminals are considered in isolation.
                sr_index = 0  # disabled
                if all(i in surviving_sts for i in block_key):
                    terminals.add(self.super_root_index)
                    sr_index = self.super_root_index

                for k in block_key:
                    terminals.add(k)
                    if k not in surviving_sts:
                        terminals.update(self.terminal_sets[k])

                terminals_list = sorted(terminals)

                logger.trace(f"    solving ({block_n} of {num_blocks}) block {block_key}...")

                self.solved_trees += 1

                cost, solution_nodes = _solve_single_block_scipstp_seq(
                    self.instance_id,
                    block_key,
                    terminals_list,
                    adj_map,
                    node_weight_map,
                    node_index,
                    dimacs_id,
                    inv_dimacs_id,
                    enable_super_root_index=sr_index,
                    do_debug=self.do_debug,
                )

                logger.trace(
                    f"    solved ({block_n} of {num_blocks}) block {block_key} containing {len(terminals_list)} terminals with cost {cost}"
                )

                mask = 0
                for u in solution_nodes:
                    mask |= 1 << node_index[u]

                block_results[block_key] = mask
                block_costs[block_key] = cost

        # Build block mask tables
        logger.trace("    building block mask table...")

        block_mask_costs = {}
        block_mask_solutions = {}

        for block_key, solution_mask in block_results.items():
            block_mask = 0

            for r in block_key:
                block_mask |= coverage_bit[r]

            block_mask_costs[block_mask] = block_costs[block_key]
            block_mask_solutions[block_mask] = solution_mask

        # Sequential realization of partitions using memoized blocks
        logger.trace("    solving partition DP...")

        full_mask = (1 << len(coverage_entities)) - 1

        dp = [float("inf")] * (full_mask + 1)
        choice = [None] * (full_mask + 1)

        dp[0] = 0

        for state in range(full_mask + 1):
            if dp[state] == float("inf"):
                continue

            remaining = full_mask ^ state
            sub = remaining

            while sub:
                if sub in block_mask_costs:
                    new_cost = dp[state] + block_mask_costs[sub]

                    next_state = state | sub

                    if new_cost < dp[next_state]:
                        dp[next_state] = new_cost
                        choice[next_state] = (state, sub)

                sub = (sub - 1) & remaining

        best_cost = dp[full_mask]
        best_solution_mask = 0

        if best_cost == float("inf") or choice[full_mask] is None:
            logger.error("    no valid partition solution found")
            return set()

        state = full_mask

        while state:
            prev_choice = choice[state]

            if prev_choice is None:
                logger.error(f"    broken DP reconstruction at state {state}")
                break

            prev_state, block_mask = prev_choice

            best_solution_mask |= block_mask_solutions[block_mask]

            state = prev_state

        # Phase 3: Extract solution
        logger.trace("    solve_with_scipstp_super: extracting solution...")

        best_solution = set()
        mask = best_solution_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            u = nodes_list[idx]
            mask ^= lsb
            best_solution.add(u)

        if best_solution:
            logger.debug(
                f"    solved for terminals: { {self.get_node_key(r): {self.get_node_key(n) for n in ts} for r, ts in self.terminal_sets.items()} }"
            )
            logger.debug(
                f"    found best solution (cost: {best_cost}): {sorted(self.get_node_key(n) for n in best_solution)}"
            )

        return best_solution

    def solve_with_scipstp(self):
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances."""
        logger.trace("solve_with_scipstp...")

        from concurrent.futures import ProcessPoolExecutor

        from sfpgre_cch import get_interactivity_edges

        self.dump_graph("solve_with_scipstp: pre-solve")

        node_weight_map = {u: self.graph[u][PAYLOAD_WEIGHT_KEY] for u in self.graph.node_indices()}

        terminal_keys = list(self.terminal_sets.keys())
        root_bit = {r: 1 << i for i, r in enumerate(terminal_keys)}

        # --- Component Level Data Mapping ---
        # Since scipstp has issues with graphs containing multiple components,
        # we need to partition the graph into connected components and handle
        # the adj mapping and weights for each component separately.
        # NOTE: Since we have no super-root WCC is fine here.
        logger.trace("      building component level data...")

        node_weights = {u: self.graph[u][PAYLOAD_WEIGHT_KEY] for u in self.graph.node_indices()}

        # Union sets of all nodes for all shortest paths for each t -> r for each t in terminal_set r
        self.set_edge_weights()
        arbor_all_shortest_path_unions = {r: set() for r in terminal_keys}
        for r, terminals in self.terminal_sets.items():
            for t in terminals:
                paths = rx.all_shortest_paths(self.graph, t, r, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY])
                arbor_all_shortest_path_unions[r].update(*paths)

        interactivity_edges = get_interactivity_edges(
            self.graph, self.terminal_sets, node_weights, arbor_all_shortest_path_unions
        )

        interactivity_graph = subgraph_stable(self.graph, terminal_keys)
        interactivity_graph.remove_edges_from(interactivity_graph.edges())
        interactivity_graph.add_edges_from(interactivity_edges)

        # print(f"      interactivity_graph.num_nodes = {interactivity_graph.num_nodes()}")
        # print(
        #     f"      map: { {i: (n, self.get_node_key(n)) for i, n in enumerate(interactivity_graph.node_indices())} }"
        # )
        # adj_matrix = rx.adjacency_matrix(interactivity_graph)
        # print(f"      adj_matrix.shape =\n{adj_matrix}")

        component_data = {}
        root_component_map = {}
        reachable = {r: {r} for r in terminal_keys}
        terminal_keys_set = set(terminal_keys)

        wcc = rx.weakly_connected_components(self.graph)
        for cc_i, cc in enumerate(wcc):
            # Root reachability for component generation...
            cc_reachable = cc & terminal_keys_set
            if not cc_reachable:
                continue

            for r in cc_reachable:
                reachable[r].update(cc_reachable)
                root_component_map[r] = cc_i

            # Even though scipstp NWSTP is fully a node weighted setup the rustworkx graph isn't pickleable
            # so we need to pass in the adjacency map and node weights
            adj_map = {}
            for u in cc:
                adj_map[u] = self.reduction_neighbors(u)

            # For index masking...
            nodes_list = list(self.graph.node_indices())
            node_index = {u: i for i, u in enumerate(nodes_list)}

            # DIMACS node ids are 1-based, contiguous
            dimacs_id = {u: i + 1 for i, u in enumerate(nodes_list)}
            inv_dimacs_id = {i + 1: u for i, u in enumerate(nodes_list)}

            component_data[cc_i] = {
                "component": cc,
                "reachable": cc_reachable,
                "adj_map": adj_map,
                "nodes_list": nodes_list,
                "node_index_map": node_index,
                "dimacs_id_map": dimacs_id,
                "inv_dimacs_id_map": inv_dimacs_id,
            }

        logger.warning(
            f"      identified {len(component_data)} reachability component(s): {[len(cc_data.get('reachable')) for cc_data in component_data.values()]}"
        )

        # Partition block filtering...
        logger.trace("    generating valid partition blocks...")

        valid_blocks: dict[int, set[tuple]] = defaultdict(set)
        num_blocks = 0
        num_candidate_blocks = 0

        # Since the partition DP reconstructs the optimal partitioning, we only need
        # to generate the valid blocks. Any valid block must be wholly contained
        # within a single reachability component.
        for cc_i, cc_data in component_data.items():
            reachable = sorted(cc_data.get("reachable"))
            k = len(reachable)
            for mask in range(1, 1 << k):
                block = tuple(reachable[i] for i in range(k) if mask & (1 << i))
                num_candidate_blocks += 1
                subIG = interactivity_graph.subgraph(block)
                if not rx.is_strongly_connected(subIG):
                    logger.trace(
                        f"      skipping invalid block {[self.get_node_key(n) for n in block]} (not strongly connected)"
                    )
                    continue
                valid_blocks[cc_i].add(block)
                num_blocks += 1

        # Map unique blocks to their full, flattened terminal sets for the solver
        worker_tasks = []
        for cc_i, cc_valid_blocks in valid_blocks.items():
            # NOTE: Terminals are translated within the worker process
            for block_key in cc_valid_blocks:
                terminals = set()
                for k in block_key:
                    terminals.add(k)
                    terminals.update(self.terminal_sets[k])

                worker_tasks.append((cc_i, block_key, sorted(terminals)))

        self.solved_trees += len(worker_tasks)

        problem_generator = (
            TreeProblem(
                instance_id=self.instance_id,
                block_key=task[1],
                terminals=task[2],
                adj_map=component_data[task[0]]["adj_map"],
                node_weight_map=node_weight_map,
                node_index_map=component_data[task[0]]["node_index_map"],
                dimacs_id_map=component_data[task[0]]["dimacs_id_map"],
                inv_dimacs_id_map=component_data[task[0]]["inv_dimacs_id_map"],
                enable_super_root_index=0,
                do_debug=self.do_debug,
                mip_validation=self.do_debug,
            )
            for task in worker_tasks
        )

        # Partition block solving...
        block_results = {}
        block_costs = {}

        if num_blocks >= 42:
            logger.warning(
                f"    concurrently solving {num_blocks} unique valid blocks of {num_candidate_blocks} ({num_blocks / num_candidate_blocks:.2%})..."
            )

            try:
                with ProcessPoolExecutor(max_workers=14) as executor:
                    chunksize = max(1, min(256, len(worker_tasks) // 16))
                    logger.warning(f"    chunksize: {chunksize}")

                    results = executor.map(
                        solve_tree,
                        problem_generator,
                        chunksize=chunksize,
                    )

                    for completed_count, (block_key, cost, mask) in enumerate(results, start=1):
                        if completed_count % 1_000 == 0:
                            logger.warning(
                                f"    solved ({completed_count}/{num_blocks}) {completed_count / num_blocks:.2%}..."
                            )

                        block_results[block_key] = mask
                        block_costs[block_key] = cost

            except BrokenProcessPool as e:
                logger.critical(f"Instance failed: Worker process died violently (OOM or Segfault). {e}")
                print(e)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Instance failed: Python exception bubbled up from worker: {e}")
                print(e, file=sys.stderr)

        else:
            logger.warning(
                f"    sequentially solving {num_blocks} unique valid blocks of {num_candidate_blocks} ({num_blocks / num_candidate_blocks:.2%})..."
            )

            for block_n, problem in enumerate(problem_generator, start=1):
                logger.trace(f"    solving ({block_n}/{num_blocks}) block {problem.block_key}...")

                block_key, cost, mask = solve_tree(problem)

                logger.trace(
                    f"    solved ({block_n}/{num_blocks}) block {block_key} containing {len(problem.terminals)} terminals with cost {cost}"
                )

                block_results[block_key] = mask
                block_costs[block_key] = cost

        # Phase 1: Build block mask tables
        logger.trace("    building block mask table...")

        block_mask_costs = {}
        block_mask_solutions = {}

        for block_key, solution_mask in block_results.items():
            block_mask = 0

            for r in block_key:
                block_mask |= root_bit[r]

            block_mask_costs[block_mask] = block_costs[block_key]
            block_mask_solutions[block_mask] = solution_mask

        # Phase 2: Sequential realization of partitions using memoized blocks (Sparse Loop)
        logger.trace("    solving partition DP...")
        start_time = time()

        full_mask = (1 << len(terminal_keys)) - 1

        dp = [float("inf")] * (full_mask + 1)
        choice = [None] * (full_mask + 1)

        dp[0] = 0

        for state in range(full_mask + 1):
            if dp[state] == float("inf"):
                continue

            remaining = full_mask ^ state
            sub = remaining

            while sub:
                if sub in block_mask_costs:
                    new_cost = dp[state] + block_mask_costs[sub]

                    next_state = state | sub

                    if new_cost < dp[next_state]:
                        dp[next_state] = new_cost
                        choice[next_state] = (state, sub)

                sub = (sub - 1) & remaining

        logger.trace(f"    solved in {time() - start_time:.2f}s")

        best_cost = dp[full_mask]
        best_solution_mask = 0

        if best_cost == float("inf") or choice[full_mask] is None:
            logger.error("    no valid partition solution found")
            return set()

        state = full_mask

        while state:
            prev_choice = choice[state]

            if prev_choice is None:
                logger.error(f"    broken DP reconstruction at state {state}")
                break

            prev_state, block_mask = prev_choice

            best_solution_mask |= block_mask_solutions[block_mask]

            state = prev_state

        # Phase 3: Extract solution
        logger.trace("    extracting solution...")

        best_solution = set()
        mask = best_solution_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            u = nodes_list[idx]
            best_solution.add(u)
            mask ^= lsb

        if best_solution:
            logger.debug(
                f"    solved for terminals: { {self.get_node_key(r): {self.get_node_key(n) for n in ts} for r, ts in self.terminal_sets.items()} }"
            )
            logger.debug(
                f"    found best solution (cost: {best_cost}): {sorted(self.get_node_key(n) for n in best_solution)}"
            )

        return best_solution

    # MARK: Main Loop

    def run_pipeline(self) -> tuple[set[int], dict[int, int]]:
        """Runs the main reduction pipeline."""
        logger.trace("run_pipeline...")

        num_edges_start = self.graph.num_edges()
        num_nodes_start = self.graph.num_nodes()
        num_terminal_roots_start = len(self.terminal_sets)
        num_terminals_start = sum(len(ts) for ts in self.terminal_sets.values()) + num_terminal_roots_start

        self.set_edge_weights()
        self._maximum_rt_distance = self.get_maximum_rt_distance()
        logger.info(f"  maximum RT distance: {self._maximum_rt_distance}")

        tmp_num_nodes = self.graph.num_nodes() + 1
        iteration = 0
        while (num_nodes := self.graph.num_nodes()) != tmp_num_nodes:
            # Update self.do_debug flag when log level is externally modified
            active_severity = logger._core.min_level  # ty:ignore[unresolved-attribute]
            self.do_debug = active_severity <= 10

            tmp_num_nodes = num_nodes
            iteration += 1
            logger.info(f"--- Iteration {iteration} ---")

            # Isolates in the graph are always safe to directly remove
            if isolates := rx.isolates(self.graph):
                self.graph.remove_nodes_from(isolates)
                logger.info(f"  removed {len(isolates)} isolates...")
                logger.trace(f"    {sorted(self.get_node_key(n) for n in isolates)}")

            # Reduce demand roots - this is a terminal set consolidation only
            self.reduce_roots_via_articulation_points()
            self.reduce_demand_roots()

            # Reduce adjacent terminals - this is a terminal reduction by consume with terminal set consolidation
            self.reduce_adjacent_terminals()

            # Reduce degree1 steiner nodes - this is a Steiner node graph reduction by `consume` with terminal set consolidation
            self.reduce_degree1_steiner_nodes()

            # Reduce degree1 terminals - this is a Steiner node graph reduction by `remove` only
            self.reduce_degree1_terminals()

            # Reduce degree1 roots - this is a Steiner node graph reduction by `consume` with terminal set consolidation
            self.reduce_degree1_roots()

            if num_nodes != self.graph.num_nodes():
                logger.info("  repeating simple reductions...")
                continue

            if not self.terminal_sets:
                logger.success("  no terminal sets left. All demands are satisfied!")
                break

            # Reduce roots by distance - this is a terminal set consumption and Steiner node removal reduction
            if self.reduce_roots_by_distance():
                continue

            # Reduce 2-degree articulation points - this is a Steiner node graph reduction with mixed handling
            self.reduce_degree2_articulation()

            # Reduce 2-degree steiner chains - this is a Steiner node graph reduction by `absorb` only
            self.merge_adjacent_degree2_steiner_chains()

            # Reduce steiner triangle 2-degree legs - this is a Steiner node graph reduction by `absorb` only
            self.reduce_steiner_triangle_degree2_legs()

            # Reduce 2-degree steiner dominance - this is a Steiner node graph reduction by `remove` only
            self.reduce_degree2_steiner_dominance()

            # Reduce k-degree steiner dominance - this is a Steiner node graph reduction by `remove` only
            self.reduce_degreek_steiner_dominance()

            # Reduce steiner bridges - this is a Steiner node graph reduction by `consume` only
            self.reduce_steiner_bridges()

            # Reduce Steiner nodes by distance - this is a Steiner node graph reduction by `remove` only
            self.reduce_steiner_nodes_by_distance()

            # Reduce degreek enclosed steiner - this is a Steiner node graph reduction by `consume` only
            self.reduce_degreek_enclosed_steiner()

            # Reduce blocked roots - this is a terminal set consolidation only
            if num_nodes != self.graph.num_nodes() and self.reduce_blocked_roots() > 0:
                # Force a reduction pass if blocked roots were merged.
                tmp_num_nodes = 0
                continue

            # Reduce enclosed steiner clusters - this is a Steiner node graph reduction by `remove` only
            if num_nodes == self.graph.num_nodes():
                self.reduce_enclosed_steiner_clusters()

            # # Solve as tree for super root
            # if (
            #     num_nodes == self.graph.num_nodes()
            #     and self.super_root_index is not None
            #     and self.terminal_sets
            # ):
            #     self.dump_reduction_results(
            #         "pre: solve_as_tree_super",
            #         num_edges_start,
            #         num_nodes_start,
            #         num_terminal_roots_start,
            #         num_terminals_start,
            #     )
            #     if self.solve_as_tree_super() > 0:
            #         tmp_num_nodes = 0
            #         continue

            # Solve as tree for non super root
            if num_nodes == self.graph.num_nodes() and self.super_root_index is None and self.terminal_sets:
                self.dump_reduction_results(
                    "pre: solve_as_tree",
                    num_edges_start,
                    num_nodes_start,
                    num_terminal_roots_start,
                    num_terminals_start,
                )
                self.solve_as_tree()

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

        return fixed_nodes_wp, reduced_root_pairs_wp
