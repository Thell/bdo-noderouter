from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import networkx as nx
import nwst_dw
import rustworkx as rx
from bidict import bidict
from loguru import logger
from networkx import PlanarEmbedding
from rustworkx import PyDiGraph

from api_exploration_data import get_exploration_data
from api_rx_pydigraph import subgraph_stable

PAYLOAD_WEIGHT_KEY = "need_exploration_point"
HYPERNODE_CONTENTS_KEY = "collapsed_nodes"
MAX_ENCLOSED_STEINER_INTERFACES = 4

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

    graph: PyDiGraph
    node_to_key: bidict[int, int]
    super_root_index: int | None = None

    fixed_nodes: set[int] = field(default_factory=set)
    potential_roots: set[int] = field(default_factory=set)
    terminal_sets: dict[int, set[int]] = field(default_factory=dict)

    _seen_nonreducible_outer_windows: set[tuple[int, ...]] = field(default_factory=set)

    do_debug: bool = False
    call_counts: Counter = field(default_factory=Counter)

    # MARK: Common Helpers

    # Logging/Debugging helpers
    def dump_graph(self, msg: str, graph: PyDiGraph | None = None):
        if not self.do_debug:
            return
        print(f"=== BEGIN: {msg} ===")
        G = graph if graph is not None else self.graph
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
            f"  Nodes: {self.graph.num_nodes():4}, Edges: {self.graph.num_edges():4}, Roots: {len(sets):4}, Pairs: {sum(len(ts) for ts in sets.values()):4}"
        )
        sets_wp = {self.get_node_key(r): {self.get_node_key(t) for t in ts} for r, ts in sets.items()}
        logger.debug(f"  Demand sets: {sets_wp}")

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
        for u, v in self.graph.edge_list():
            weight = self.graph[v][PAYLOAD_WEIGHT_KEY]
            if v not in self.potential_roots and v in self.fixed_nodes:
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
            rep = next(iter(comp_roots), None)
            if rep is not None:
                terminal_sets[rep] = set(comp) - {rep}
            elif super_root_index is not None:
                terminal_sets[super_root_index].update(comp)
            else:
                raise RuntimeError(
                    f"Component has no roots and super root is missing: {[self.get_node_key(n) for n in comp]}"
                )

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

        removals = 0

        # NOTE: The direct usage of 'out_degree' instead of 'reduction_degree' is intentional for performance.
        #       No node of out_degree == 1 can be a non-leaf node.
        while removables := {i for i in self.graph.node_indices() if self.graph.out_degree(i) == 1}:
            removables.difference_update(non_steiners)
            if not removables:
                break
            self.graph.remove_nodes_from(removables)
            removals += len(removables)

        if removals > 0:
            logger.info(f"  removed {removals} degree-1 steiner nodes")
            if self.do_debug:
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

        removals = 0

        for r_k in list(roots):
            # NOTE: If r_k is the root of a previously consumed terminal set then it is now a terminal.
            if r_k not in terminal_sets:
                continue

            # A root, in the presence of a super_root, can have 2 successors
            # but only ever 1 predecessor when reduction_degree == 1
            if self.reduction_degree(r_k) != 1:
                continue
            neighbor = next(iter(self.reduction_neighbors(r_k)))

            self.consume(neighbor, r_k)
            removals += 1

            r_j = next((r for r, ts in terminal_sets.items() if neighbor in ts), None)
            if r_j is not None:
                # The neighbor was not a Steiner node, fix terminal sets...
                terminal_sets[r_j].remove(neighbor)
                if r_k != r_j and r_j != self.super_root_index:
                    self.consume_terminal_set(terminal_sets, r_j, r_k)

        if removals > 0:
            # Drop empty sets
            self.terminal_sets = {r: ts for r, ts in terminal_sets.items() if ts}
            logger.info(f"  removed {removals} degree-1 root Steiner nodes")
            if self.do_debug:
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
        removals = 0

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
                    made_progress = True
                    removals += 1
                    break

        if removals > 0:
            logger.info(f"  removed {removals} degree-2 Steiner chain nodes")
            if self.do_debug:
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

        self.set_edge_weights()

        graph = self.graph
        non_steiner_nodes = self.non_steiner_nodes()
        removals = 0

        made_progress = True
        while made_progress:
            made_progress = False

            for node in set(graph.node_indices()) - non_steiner_nodes:
                if graph.out_degree(node) != 2:
                    continue

                succ = list(graph.successor_indices(node))
                u, v = succ
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
                made_progress = True

                removals += 1
                break

        if removals > 0:
            logger.info(f"  removed {removals} degree-2 dominated nodes")
            if self.do_debug:
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
        logger.trace("reduce_bridge_steiner_consumption...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        removals = 0

        # Get bridges
        tmp_undir = self.graph.to_undirected()
        if self.super_root_index is not None:
            tmp_undir.remove_node(self.super_root_index)

        bridges = rx.bridges(tmp_undir)

        # Map back to self.graph indices
        tmp_map = {i: u for i, u in enumerate(self.graph.node_indices())}
        bridges = [(tmp_map[u], tmp_map[v]) for u, v in bridges]  # ty:ignore[invalid-assignment]

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
                not rx.has_path(tmp, r, t) for r, terminals in self.terminal_sets.items() for t in terminals
            )
            if not violates_demand:
                # NOTE: We can't certify exclusion of either endpoint of a single bridge because it says nothing
                #       about the two endpoints in an optimal Solution. And in a node weighted problem if both
                #       end up being selected then the edge is naturally selected.
                #       But, if the created component from removing the bridge contains no terminals or roots
                #       or only one arborescence then it is safe to remove the bridge edge.
                #       In that case we can remove the edge but not consume either endpoint.
                sccs = rx.strongly_connected_components(tmp)
                for scc in sccs:
                    set_scc = set(scc)
                    if not set_scc & self.fixed_nodes:
                        # The component contains no fixed nodes, thus no-demand and no structure violations
                        self.graph.remove_edge(u, v)
                        self.graph.remove_edge(v, u)
                        logger.debug("  removed bridge edge without consumption")
                        break
                    scc_roots = set_scc & set(self.terminal_sets.keys())
                    if len(scc_roots) == 1:
                        # A single root is present, so if all of its terminals are also present then it is safe to remove
                        scc_root = next(iter(scc_roots))
                        if set_scc & self.terminal_sets[scc_root]:
                            self.graph.remove_edge(u, v)
                            self.graph.remove_edge(v, u)
                            logger.debug("  removed bridge edge without consumption")
                            break
                continue

            # The edge must be traversed in an optimal Solution...
            # Consume Steiner into the other side
            if steiner_u:
                self.consume(u, v)
            else:
                self.consume(v, u)

            removals += 1

        if removals > 0:
            logger.info(f"  consumed {removals} Steiner bridge nodes")
            if self.do_debug:
                self.dump_state("reduce_bridge_steiner_consumption", removals)
                self.validate_reachability()

        return removals

    def reduce_degreek_steiner_dominance(self, max_degree: int = 4) -> int:
        """
        Removes Steiner nodes of small degree k that are dominated by alternate paths.

        Reduction Rule:
        ```
        Ⓢ : N(Ⓢ) = {○₁, ○₂, …, ○ₖ}, 2 < k ≤ K
        ∧ ∀ {○ᵢ, ○ⱼ} ⊆ N(Ⓢ):
            (○ᵢ ↠∣Ⓢ∣↠ ○ⱼ) exists
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

                        if dist_avoid[v] > dist_full[v]:
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

    def reduce_degree3_steiner_triangle_legs(self):
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
        non_steiners.discard(self.super_root_index)

        steiners = {
            n for n in self.graph.node_indices() if n not in non_steiners and n != self.super_root_index
        }

        if len(steiners) < 3:
            return 0

        g_prime: PyDiGraph = subgraph_stable(self.graph, steiners)

        # Visualizing for clustering...
        self.dump_graph("g_prime for enclosed steiner clusters", g_prime)

        interfaces = set()

        for s in steiners:
            nbrs = self.reduction_neighbors(s)
            if nbrs - steiners:
                interfaces.add(s)

        if len(interfaces) < 2:
            return 0

        removables = set()

        comps = rx.weakly_connected_components(g_prime)
        # for max_interfaces in range(2, MAX_ENCLOSED_STEINER_INTERFACES + 1):
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
            self.graph.remove_nodes_from(removables)
            logger.info(f"  => removed {len(removables)} enclosed Steiner cluster nodes")
            if self.do_debug:
                logger.trace(f"  {sorted(self.get_node_key(n) for n in removables)}")
                self.dump_state("reduce_enclosed_steiner_clusters", len(removables))
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

        Components exceeding the interface limit are recursively split across
        bridge edges. Split endpoints become interfaces in both child
        components.
        """
        if self.do_debug:
            logger.debug(
                f"=> reduce_cluster_dreyfus_wagner({sorted(self.get_node_key(n) for n in component)}, {sorted(self.get_node_key(n) for n in interfaces)}, {interface_limit})"
            )

        k = len(interfaces)

        if k <= 1:
            return set()

        # Recursive decomposition phase
        if k > interface_limit:
            split = self._find_component_bridge_split(component)
            if split is None:
                # No valid decomposition found - refuse certification.
                return set()

            left_nodes, right_nodes, bridge_u, bridge_v = split
            left_interfaces = (interfaces | {bridge_u, bridge_v}) & left_nodes
            right_interfaces = (interfaces | {bridge_u, bridge_v}) & right_nodes

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

        # Preparation phase for Rust Native Solver Input Maps
        local_nodes = list(component)
        adj_map: dict[int, set[int]] = {n: set() for n in local_nodes}
        node_weight_map: dict[int, int] = {}

        for u in local_nodes:
            # Extract standard rustworkx graph topology indices
            nbrs = self.reduction_neighbors(u)
            for v in nbrs:
                if v in component:
                    adj_map[u].add(v)

            # Extract mandatory node weights mapped to u64
            node_weight_map[u] = int(self.graph[u][PAYLOAD_WEIGHT_KEY])

        def steiner_tree_nodes(terminals: tuple[int, ...]) -> set[int]:
            """
            Invokes the high-performance native Rust solver.
            Returns the exact optimal Steiner tree topology nodes.
            """
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

        # Pairwise shortest paths for all interface pairs
        for i in range(len(interface_list)):
            for j in range(i + 1, len(interface_list)):
                s = interface_list[i]
                t = interface_list[j]
                if s not in component or t not in component:
                    continue

                # Fallback to a single pair Steiner evaluation via Rust
                nodes = steiner_tree_nodes((s, t))
                keep.update(nodes)

        # Steiner trees for all interface subsets up to interface_limit
        for subset_mask in range(1, 1 << len(interface_list)):
            subset = tuple(interface_list[i] for i in range(len(interface_list)) if subset_mask & (1 << i))

            # Pairwise shortest paths are already evaluated and populated above
            if 3 <= len(subset) <= interface_limit:
                nodes = steiner_tree_nodes(subset)
                keep.update(nodes)

        # Final component removals: unneeded internal nodes
        return (component - interfaces) - keep

    def reduce_enclosed_terminal_clusters(
        self,
        nearest_terminals: int = 5,
        max_component_interfaces: int = 5,
        max_component_nodes: int = 128,
        max_component_terminals: int = 12,
    ) -> int:
        """
        Identifies terminal-anchored Steiner pockets and removes Steiner nodes
        provably absent from every witnessed optimal realization over all
        feasible interface activations.

        Candidate pockets are constructed from:
            - same-root terminal neighborhoods,
            - induced Steiner routing structure,
            - and iterative dead-leg trimming.

        Each surviving component is certified using exact anchored
        Dreyfus-Wagner witness reconstruction.

        Handles:
        ```
        ∀ ℂ ⊆ G :
        A ≔ enclosed terminals
        I ≔ structural interfaces

        ∀ X ⊆ I :
            Wₓ ∈ OPT_ST(A ⋃ X, ℂ)

        k ≔ ℂ ∖ ( I ⋃ A ⋃ ⋃ Wₓ )

        G ≔ G ∖ k
        ```
        """
        logger.trace("reduce_enclosed_terminal_clusters...")

        self.set_edge_weights()

        #
        # STEP 0:
        # Identify Steiner backbone candidates.
        # We only operate on Steiner-dense regions since terminals
        # are preserved anchors and not reduction targets.
        #
        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        steiners = {
            n for n in self.graph.node_indices() if n not in non_steiners and n != self.super_root_index
        }

        if len(steiners) < 3:
            return 0

        removables: set[int] = set()

        # TODO: Research why this would fail on some incidents when using term and interfaces limits of 6 instead of 5
        # # (sample: [  12/20] 39d3339 random_n_cheapest_capital )
        terminal_clusters = {r: ts | {r} for r, ts in self.terminal_sets.items()}

        # The failure doesn't show up when using the following...
        # terminal_clusters = {r: ts for r, ts in self.terminal_sets.items()}

        #
        # STEP 1:
        # Precompute local terminal neighborhoods.
        # This is used to bias component extraction toward
        # regions that can actually support multi-terminal DP structure.
        #
        terminal_to_nearest: dict[int, list[int]] = {}

        for ts in terminal_clusters.values():
            ts = set(ts)

            if len(ts) < 2:
                continue

            for t in ts:
                dist = rx.dijkstra_shortest_path_lengths(
                    self.graph,
                    t,
                    edge_cost_fn=lambda e: e.get(PAYLOAD_WEIGHT_KEY, 1),
                )

                terminal_to_nearest[t] = sorted(
                    (x for x in ts if x != t and x in dist),
                    key=lambda x: dist[x],
                )[:nearest_terminals]

            #
            # STEP 2:
            # For each terminal cluster, build candidate pockets.
            # The goal is to over-approximate support, not under-approximate it.
            #
            # for max_interfaces in range(2, max_component_interfaces + 1):
            for ts in terminal_clusters.values():
                terminal_set = set(ts)

                if len(terminal_set) < 2:
                    continue

                for anchor in terminal_set:
                    nearest = terminal_to_nearest.get(anchor)
                    if not nearest:
                        continue

                    #
                    # STEP 3:
                    # Seed terminals define the minimal “activation context”
                    # for candidate DP regions.
                    #
                    seed_terminals = {anchor, *nearest}

                    #
                    # STEP 4:
                    # Build Steiner-enriched skeleton.
                    # We intentionally include all Steiner nodes to avoid
                    # prematurely excluding valid DP-supporting structure.
                    #
                    skeleton_nodes = steiners | seed_terminals
                    skeleton = subgraph_stable(self.graph, skeleton_nodes)

                    #
                    # STEP 5:
                    # Extract anchored connected component.
                    # This defines the candidate pocket ℂ.
                    #
                    component: set[int] | None = None
                    for cc in rx.weakly_connected_components(skeleton):
                        if anchor in cc:
                            component = set(cc)
                            break

                    if component is None:
                        continue

                    terminal_in_component = component & terminal_set

                    #
                    # We require at least two terminals to form
                    # a meaningful Steiner interface problem instance.
                    #
                    if len(terminal_in_component) < 2:
                        continue

                    #
                    # STEP 6:
                    # Terminal-supported DFS pruning.
                    #
                    # This removes dead structural “legs” that do not
                    # participate in any terminal-to-terminal reachability
                    # within the component.
                    #
                    # This is critical because DW witness reconstruction
                    # must operate on a structure where every node is
                    # potentially DP-relevant.
                    #
                    component, _ = self._dfs_dead_leg_prune(
                        component=component,
                        terminals=terminal_in_component,
                        interfaces=set(),
                    )

                    terminal_in_component = component & terminal_set

                    if len(terminal_in_component) < 2:
                        continue

                    #
                    # STEP 7:
                    # Structural bounds ensure tractability of DW certification.
                    #
                    if len(component) < 4:
                        continue
                    if len(component) > max_component_nodes:
                        continue
                    if len(terminal_in_component) > max_component_terminals:
                        continue

                    #
                    # STEP 8:
                    # Ensure non-tree structure exists.
                    # If no cycles exist, there is no alternative routing
                    # structure for interface-conditioned OPT_ST.
                    #
                    sub = subgraph_stable(self.graph, component)
                    if sub.num_edges() // 2 <= len(component) - 1:
                        continue

                    #
                    # STEP 9:
                    # Interface extraction.
                    #
                    # Interfaces are boundary nodes of ℂ that connect
                    # to external graph structure and define DP boundary conditions.
                    #
                    interfaces: set[int] = set()

                    for n in component:
                        nbrs = self.reduction_neighbors(n)
                        if nbrs - component:
                            interfaces.add(n)

                    if len(interfaces) < 2:
                        continue

                    if len(interfaces) > max_component_interfaces:
                        continue

                    #
                    # STEP 10:
                    # Exact anchored DW certification.
                    #
                    # This stage determines which nodes participate in at least one
                    # optimal realization over all interface activations.
                    #
                    removable = self._reduce_anchored_interface_dreyfus_wagner(
                        component=component,
                        anchors=terminal_in_component,
                        interfaces=interfaces,
                        interface_limit=max_component_interfaces,
                    )

                    removables.update(removable)

        removables -= non_steiners

        if removables:
            self.graph.remove_nodes_from(removables)
            logger.info(f"  => removed {len(removables)} enclosed terminal cluster nodes")
            if self.do_debug:
                logger.trace(f"  {sorted(self.get_node_key(n) for n in removables)}")
                self.dump_state("enclosed terminal cluster nodes", len(removables))
                self.validate_reachability()

        return len(removables)

    def _dfs_dead_leg_prune(
        self,
        component: set[int],
        terminals: set[int],
        interfaces: set[int],
    ) -> tuple[set[int], set[int]]:
        """
        Terminal-supported DFS closure with dead-leg elimination.

        A node survives iff it is reachable from at least one terminal
        during DFS over the induced component.

        Interfaces are updated by promoting surviving neighbors of
        removed interface nodes along valid adjacency in the pruned graph.
        """

        adj = {n: self.reduction_neighbors(n) & component for n in component}

        visited: set[int] = set()

        #
        # STEP 1:
        # Multi-source DFS over component from all terminals.
        # This computes the terminal-supported closure of the component.
        #

        def dfs(start: int):
            stack = [start]

            while stack:
                u = stack.pop()

                if u in visited:
                    continue

                visited.add(u)

                for v in adj[u]:
                    if v not in visited:
                        stack.append(v)

        for t in terminals:
            if t in component:
                dfs(t)

        #
        # STEP 2:
        # Nodes not reachable from any terminal are dead by definition.
        #

        _dead = component - visited

        #
        # STEP 3:
        # Iteratively remove dead leaves that become isolated
        # after terminal closure.
        #

        changed = True
        while changed:
            changed = False

            new_dead = set()

            for n in visited:
                if n in terminals:
                    continue

                deg = sum(1 for v in adj[n] if v in visited)

                if deg == 0:
                    new_dead.add(n)

            if new_dead:
                visited -= new_dead
                changed = True

        surviving = visited & component

        #
        # STEP 4:
        # Interface promotion based on surviving adjacency only.
        #

        updated_interfaces = set(interfaces)

        for n in interfaces:
            if n not in surviving:
                nbrs = adj[n] & surviving
                if nbrs:
                    updated_interfaces.add(next(iter(nbrs)))

        updated_interfaces &= surviving

        return surviving, updated_interfaces

    def _reduce_anchored_interface_dreyfus_wagner(
        self,
        component: set[int],
        anchors: set[int],
        interfaces: set[int],
        interface_limit: int,
    ) -> set[int]:
        """
        Exact existential witness certification for anchored Steiner pockets.

        Let:
            A ≔ mandatory anchor terminals
            I ≔ optional interface activations

        For every interface activation subset:

            ∀ X ⊆ I :
                Wₓ ∈ OPT_ST(A ⋃ X, ℂ)

        preserve all nodes participating in at least one witnessed optimal
        realization.

        Components exceeding the interface limit are recursively decomposed
        across bridge edges. Split endpoints become interfaces in both child
        components.
        """
        if self.do_debug:
            logger.debug(
                "=> reduce_anchored_interface_dreyfus_wagner("
                f"{sorted(self.get_node_key(n) for n in component)}, "
                f"{sorted(self.get_node_key(n) for n in anchors)}, "
                f"{sorted(self.get_node_key(n) for n in interfaces)}, "
                f"{interface_limit})"
            )

        anchors = anchors & component
        interfaces = interfaces & component

        if len(anchors) < 2:
            return set()

        # Recursive decomposition phase
        if len(interfaces) > interface_limit:
            split = self._find_component_bridge_split(component)

            if split is None:
                # No valid decomposition found. Conservatively refuse certification.
                return set()

            left_nodes, right_nodes, bridge_u, bridge_v = split

            left_interfaces = (interfaces | {bridge_u, bridge_v}) & left_nodes
            right_interfaces = (interfaces | {bridge_u, bridge_v}) & right_nodes

            left_anchors = anchors & left_nodes
            right_anchors = anchors & right_nodes

            removable = set()
            removable |= self._reduce_anchored_interface_dreyfus_wagner(
                component=left_nodes,
                anchors=left_anchors,
                interfaces=left_interfaces,
                interface_limit=interface_limit,
            )
            removable |= self._reduce_anchored_interface_dreyfus_wagner(
                component=right_nodes,
                anchors=right_anchors,
                interfaces=right_interfaces,
                interface_limit=interface_limit,
            )
            return removable

        # Preparation phase for DW Input Maps
        local_nodes = list(component)
        adj_map: dict[int, set[int]] = {n: set() for n in local_nodes}
        node_weight_map: dict[int, int] = {}

        for u in local_nodes:
            nbrs = self.reduction_neighbors(u)
            for v in nbrs:
                if v in component:
                    adj_map[u].add(v)

            node_weight_map[u] = int(self.graph[u][PAYLOAD_WEIGHT_KEY])

        def steiner_tree_nodes(terminals: tuple[int, ...]) -> set[int]:
            """
            Invokes the high-performance native Rust solver.
            Returns the exact optimal Steiner tree topology nodes.
            """
            terminals_list = list(terminals)
            cost, _d_nodes, solution_nodes = nwst_dw.solve_nwst(
                adj_map, node_weight_map, terminals_list, terminals_list[0]
            )

            if cost == 18446744073709551615:  # Matches Rust u64::MAX representation for INF
                return set(terminals)

            return set(solution_nodes)

        # Witness preservation
        keep: set[int] = set()
        interface_list = sorted(interfaces)

        for subset_mask in range(1 << len(interface_list)):
            active_interfaces = {
                interface_list[i] for i in range(len(interface_list)) if subset_mask & (1 << i)
            }

            terminals = tuple(sorted(anchors | active_interfaces))

            if len(terminals) < 2:
                continue

            nodes = steiner_tree_nodes(terminals)
            keep.update(nodes)

        if self.do_debug:
            logger.trace(f"  DW - keep: {sorted(self.get_node_key(n) for n in keep)}")

        return component - anchors - interfaces - keep

    def _find_component_bridge_split(self, component: set[int]) -> tuple[set[int], set[int], int, int] | None:
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

    def reduce_degree2_outer_face_steiners(self) -> int:
        logger.trace("reduce_degree2_outer_face_steiners...")
        # 15 and 8 worked well but took a bit too long (for Python that is expected)
        MAX_OUTER_WINDOW_SIZE = 13
        MAX_INTERFACES = 7

        self.dump_graph("reduce_degree2_outer_face_steiners")

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

        # --- Rustworkx <-> networkx (prototyping-only) ---
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
        if self.do_debug:
            logger.warning(
                f"  Outer face: edge_key: {outer_face_entry[0]} {[self.get_node_key(nx_to_rx[i]) for i in outer_face_list]}"
            )

        # Filter to all-outer adjacent faces...
        outer_face_set = set(outer_face_list)
        for k in [k for k, f in faces.items() if not any(i in outer_face_set for i in f)]:
            del faces[k]
        # pop outer face entry as well
        del faces[outer_face_entry[0]]

        if self.do_debug:
            logger.warning("  Outer-adjacent faces:")
            for k, f in faces.items():
                logger.warning(f"  Face: edge_key: {k} {[self.get_node_key(nx_to_rx[i]) for i in f]}")

        ordered_outer_adjacent_faces = get_ordered_outer_adjacent_faces(
            embedding, outer_face_list, outer_face_set
        )

        if self.do_debug:
            logger.warning("  Ordered outer adjacent faces:")
            for f in ordered_outer_adjacent_faces:
                logger.warning(f"    {[self.get_node_key(nx_to_rx[i]) for i in f]}")

        # Translate all networkx indices to rustworkx indices
        outer_face_list = [nx_to_rx[i] for i in outer_face_list]
        outer_face_set = set(outer_face_list)
        outer_face_degree2_steiners = {
            i for i in outer_face_list if self.reduction_degree(i) == 2 and i not in non_steiner
        }
        ordered_outer_adjacent_faces = [{nx_to_rx[i] for i in f} for f in ordered_outer_adjacent_faces]

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
            """
            Invokes the high-performance native Rust solver.
            Returns the exact optimal Steiner tree topology nodes.
            """
            terminals_list = list(terminals)
            cost, _d_nodes, solution_nodes = nwst_dw.solve_nwst(
                adj_map, node_weight_map, terminals_list, terminals_list[0]
            )

            if cost == 18446744073709551615:  # Matches Rust u64::MAX representation for INF
                return set(terminals)

            return set(solution_nodes)

        # Simply for out of band protype visualization...
        if self.do_debug:
            logger.warning(
                f"  All inner and outer face nodes: {[self.get_node_key(i) for i in all_inner_and_outer_face_nodes]}"
            )

        terminal_to_root = {t: r for r, ts in self.terminal_sets.items() for t in ts}
        terminal_to_root.update({r: r for r in self.terminal_sets})

        # Witness preservation
        keep: set[int] = set()
        removables = set()
        n = len(ordered_outer_adjacent_faces)
        if n < 3:
            return 0

        for i in range(n):
            # NOTE: See the code, doc-strings and commentary of the enclosed DW Steiner and terminal
            #       cluster reduction functions for more details on the DW enclosure logic.
            #
            # NOTE: When all of a root's demands are enclosed within the composite region C,
            #       then any other internal structuring within the composite region creates
            #       ambiguity for that root's arbor and potential route sharing.
            #       This could be handled for cases up to 3 small arbors without too much complexity,
            #       but for prototyping we simply ignore these cases.
            #
            # NOTE: Pretty much all failing cases happen in an eye situation or adjacent left and right
            #       outer faces. So I added a quick check to ensure disjointness.
            #
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

            # There must be some degree-2 Steiner to remove from the middle outerface segment...
            f_middle_candidates = F_middle & outer_face_degree2_steiners
            if not f_middle_candidates:
                continue

            # Build composite region C
            C_nodes = set(F_left) | set(F_middle) | set(F_right)
            if tuple(sorted(C_nodes)) in self._seen_nonreducible_outer_windows:
                if self.do_debug:
                    logger.warning(f"    skipping seen before: {[self.get_node_key(i) for i in C_nodes]}")
                continue

            anchors = {node for node in C_nodes if node in non_steiner}
            if self.do_debug:
                logger.warning(f"      anchors: {[self.get_node_key(i) for i in anchors]}")

            active_roots = {terminal_to_root[t] for t in anchors}

            # NOTE: The case for small fully enclosed roots _might_ be able to be handled, but not yet...
            fully_internal_roots = {r for r in active_roots if (set(self.terminal_sets[r]) | {r}) <= C_nodes}
            if fully_internal_roots:
                if self.do_debug:
                    logger.warning(
                        f"      skipping fully internal: {[self.get_node_key(i) for i in fully_internal_roots]}"
                    )
                continue

            # NOTE: The case for multi-root composites _might_ be able to be handled, but not yet...
            # if active_roots not in [0, 1] or fully_internal_roots:
            #     # logger.warning(f"      skipping multi-root: {[self.get_node_key(i) for i in active_roots]}")
            #     continue
            # logger.warning(f"      checking unfiltered: {[self.get_node_key(i) for i in active_roots]}")

            # NOTE: This ensures that a route _must_ make it to the outer middle face from the inside.
            f_middle_has_anchor = F_middle & outer_face_set & anchors
            if not f_middle_has_anchor:
                if self.do_debug:
                    logger.warning(
                        f"      skipping no anchor: {[self.get_node_key(i) for i in f_middle_has_anchor]}"
                    )
                continue

            interfaces = {i for i in C_nodes if any(n not in C_nodes for n in self.reduction_neighbors(i))}
            interfaces -= anchors
            if self.do_debug:
                logger.warning(f"      interfaces: {[self.get_node_key(i) for i in interfaces]}")
            if (
                len(interfaces) + len(anchors) > MAX_OUTER_WINDOW_SIZE
                or len(interfaces) < 2
                or len(interfaces) > MAX_INTERFACES
            ):
                continue

            interface_list = sorted(interfaces)
            for subset_mask in range(1 << len(interface_list)):
                active_interfaces = {
                    interface_list[i] for i in range(len(interface_list)) if subset_mask & (1 << i)
                }

                terminals = tuple(sorted(anchors | active_interfaces))
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

    # MARK: Main Loop

    def run_pipeline(self) -> tuple[set[int], dict[int, int]]:
        """Runs the main reduction pipeline."""
        logger.trace("run_pipeline...")

        num_edges_start = self.graph.num_edges()
        num_nodes_start = self.graph.num_nodes()
        num_terminal_roots_start = len(self.terminal_sets)
        num_terminals_start = sum(len(ts) for ts in self.terminal_sets.values()) + num_terminal_roots_start

        tmp_num_nodes = self.graph.num_nodes() + 1
        iteration = 0
        while (num_nodes := self.graph.num_nodes()) != tmp_num_nodes:
            tmp_num_nodes = num_nodes
            iteration += 1
            logger.info(f"--- Iteration {iteration} ---")

            # # Isolates in the graph are always safe to directly remove
            # isolates = rx.isolates(self.graph)
            # self.graph.remove_nodes_from(isolates)
            # if self.do_debug:
            #     logger.info(f"  removed {len(isolates)} isolates...")

            # # Reduce degree1 demand roots - this is a terminal set consolidation only
            # self.reduce_demand_roots()

            # # Reduce adjacent terminals - this is a terminal reduction by consume with terminal set consolidation
            # self.reduce_adjacent_terminals()

            # # Reduce degree1 steiner nodes - this is a Steiner node graph reduction by `consume` with terminal set consolidation
            # self.reduce_degree1_steiner_nodes()

            # # Reduce degree1 terminals - this is a Steiner node graph reduction by `remove` only
            # self.reduce_degree1_terminals()

            # # Reduce degree1 roots - this is a Steiner node graph reduction by `consume` with terminal set consolidation
            # self.reduce_degree1_roots()

            # if num_nodes != self.graph.num_nodes():
            #     logger.debug("  repeating simple reductions...")
            #     continue

            # # Reduce 2-degree articulation points - this is a Steiner node graph reduction with mixed handling
            # self.reduce_degree2_articulation()

            # # Reduce 2-degree steiner chains - this is a Steiner node graph reduction by `absorb` only
            # self.merge_adjacent_degree2_steiner_chains()

            # # Reduce steiner bridges - this is a Steiner node graph reduction by `consume` only
            # self.reduce_steiner_bridges()

            # # Reduce 2-degree steiner dominance - this is a Steiner node graph reduction by `remove` only
            # self.reduce_degree2_steiner_dominance()

            # # Reduce k-degree steiner dominance - this is a Steiner node graph reduction by `remove` only
            # self.reduce_degreek_steiner_dominance()

            # # Reduce degree3 steiner triangle legs - this is a Steiner node graph reduction by `absorb` only
            # self.reduce_degree3_steiner_triangle_legs()

            # # Reduce blocked roots - this is a terminal set consolidation only
            # if num_nodes != self.graph.num_nodes():
            #     self.reduce_blocked_roots()

            # # Reduce degreek enclosed steiner - this is a Steiner node graph reduction by `consume` only
            # self.reduce_degreek_enclosed_steiner()

            # # NOTE: Definitely need better clustering setup to make this more powerful...
            # # Reduce enclosed steiner clusters - this is a Steiner node graph reduction by `remove` only
            # if num_nodes == self.graph.num_nodes():
            #     self.reduce_enclosed_steiner_clusters()

            # # NOTE: Definitely need better clustering setup to make this more powerful...
            # # Reduce enclosed terminal clusters - this is a Steiner node graph reduction by `remove` only
            # if num_nodes == self.graph.num_nodes():
            #     self.reduce_enclosed_terminal_clusters()

            # # Reduce degree2 outer face steiners - this is a Steiner node graph reduction by `remove` only
            # if num_nodes == self.graph.num_nodes():
            #     self.reduce_degree2_outer_face_steiners()

        # Rebuild pairs
        fixed_nodes_wp = {self.get_node_key(n) for n in self.fixed_nodes}
        reduced_root_pairs_wp: dict[int, int] = {}
        for r, comp in self.terminal_sets.items():
            for t in comp:
                reduced_root_pairs_wp[self.get_node_key(t)] = self.get_node_key(r)

        if self.do_debug:
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

        # Reduction percentages
        num_edges_end = self.graph.num_edges()
        num_nodes_end = self.graph.num_nodes()
        num_terminal_roots_end = len(self.terminal_sets)
        num_terminals_end = sum(len(ts) for ts in self.terminal_sets.values()) + num_terminal_roots_end
        per_edges = (num_edges_end - num_edges_start) / num_edges_start
        per_nodes = (num_nodes_end - num_nodes_start) / num_nodes_start
        per_terminals = (num_terminals_end - num_terminals_start) / num_terminals_start
        per_roots = (num_terminal_roots_end - num_terminal_roots_start) / num_terminal_roots_start
        print(
            f"  Reduction Percentages: Edges: {per_edges * 100:.2f}%, Nodes: {per_nodes * 100:.2f}%, Terminals: {per_terminals * 100:.2f}%, Roots: {per_roots * 100:.2f}%"
        )

        return fixed_nodes_wp, reduced_root_pairs_wp

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
