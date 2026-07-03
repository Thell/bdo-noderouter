# optimizer_mip.py

"""
This module formulates our integer baseline of the Steiner Forest problem solution
from a multi-commodity flow problem with cumulative flow from the terminals to the roots.

** All reductions and transformations guarantee that the remaining problem space contains
an optimal solution. **

Given:
- a directed graph of the exploration data
- a 'terminals' dict in the form of {terminal: root, ...}:
- a config dict
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from typing import cast

import fast_paths as fp
import rustworkx as rx
from bidict import bidict
from loguru import logger
from rustworkx import PyDiGraph

import api_data_store as ds
from api_common import PAYLOAD_WEIGHT_KEY, set_logger
from api_exploration_data import SUPER_ROOT, get_exploration_data
from api_highs_solver import create_model, extract_solution, get_highs
from api_rx_pydigraph import set_graph_terminal_sets_attribute, subgraph_stable
from orchestrator import Solution
from orchestrator_terminal_pairs import PairingStrategy
from solver_highspy import SolverController, solve

type LeafMap = dict[int, tuple[int, int, list[int]]]

PAYLOAD_WEIGHT_KEY: str = "need_exploration_point"
DO_DEBUG: bool = ds.get_config("config")["logger"]["level"] in ["DEBUG", "TRACE"]

_fp_cache: dict[bool, fp.PyFastGraph] = {}


def get_fp_graph(has_super: bool) -> fp.PyFastGraph:
    if has_super in _fp_cache:
        return _fp_cache[has_super]

    exploration_data = get_exploration_data()
    base_graph = exploration_data.super_graph if has_super else exploration_data.graph
    input_graph = fp.PyInputGraph()
    e_map = base_graph.edge_index_map()
    SUPER_ROOT_INDEX = base_graph.attrs["node_key_by_index"].inv.get(SUPER_ROOT, None)

    for i in base_graph.edge_indices():
        u, v, edata = e_map[i]
        if u == SUPER_ROOT_INDEX or v == SUPER_ROOT_INDEX:
            continue
        # NOTE: fastpaths does not accept zero weight edges, so we scale by 10_000
        #       such that zero-weight edges have a nominal weight of 1
        input_graph.add_edge(u, v, edata["weight"] * 10_000 + 1)

    input_graph.freeze()
    _fp_cache[has_super] = fp.prepare(input_graph)
    return _fp_cache[has_super]


def surviving_ancestor(leaf_map: LeafMap, idx: int) -> int:
    """Returns the surviving ancestor of the given node index.
    If the node was never removed, returns the node itself.
    """
    entry = leaf_map.get(idx)
    if entry is None:
        return idx
    return entry[1]  # (parent, ancestor, chain)


def pendant_chain(leaf_map: LeafMap, idx: int) -> list[int]:
    """Returns the pendant chain for the given node, excluding the ancestor.
    Ordered root-ward → leaf-ward.
    Returns empty list if node was not contracted.
    """
    entry = leaf_map.get(idx)
    if entry is None:
        return []
    return entry[2]  # (parent, ancestor, chain)


def transform_pairs_to_reduced_pairs(
    leaf_map: LeafMap,
    pairs: list[tuple[int, int]],
    fixed_nodes: set[int],
    fp_graph: fp.PyFastGraph,
    super_root_index: int | None = None,
) -> list[tuple[int, int]]:
    """Transforms original graph index pairs into reduced-graph-compatible pairs.

    Modifies `fixed_nodes` in-place by adding pendant nodes.

    Returns:
      list of working pairs (both endpoints present in reduced graph).
    """
    logger.trace("transform_pairs_to_reduced_pairs...")

    # NOTE: This can not be a set because transformed sources may have different targets
    transformed_pairs: set[tuple[int, int]] = set()  # (s, t) -> (anc_s, anc_t)

    for orig_s, orig_t in pairs:
        if orig_s == orig_t:
            raise RuntimeError(f"Self-loop pair: {orig_s} → {orig_t}")

        fixed_nodes.add(orig_s)
        fixed_nodes.add(orig_t)

        # Lookup surviving ancestors + pendant chains
        entry_s = leaf_map.get(orig_s)
        if entry_s is None:
            anc_s = orig_s
            chain_s = []
        else:
            _, anc_s, chain_s = entry_s

        entry_t = leaf_map.get(orig_t)
        if entry_t is None:
            anc_t = orig_t
            chain_t = []
        else:
            _, anc_t, chain_t = entry_t

        # Same surviving ancestor → same pendant subtree use original graph's path
        if anc_s == anc_t:
            path = fp_graph.calc_path(orig_s, orig_t)
            if path is None:
                raise RuntimeError(
                    f"Disconnected pair in same subtree: {orig_s} → {orig_t} (ancestor {anc_s})"
                )
            fixed_nodes.update(path.get_nodes())
            continue

        # Append pendant chains only when contracted
        if chain_s:
            fixed_nodes.update(set(chain_s))
        if chain_t:
            fixed_nodes.update(set(chain_t))

        # Only keep lifted pair if endpoints remain distinct
        if anc_s != anc_t:
            fixed_nodes.add(anc_s)
            fixed_nodes.add(anc_t)
            transformed_pairs.add((anc_s, anc_t))

    # NOTE: SUPER_ROOT is never a fixed node, so don't leak it!
    if super_root_index is not None:
        fixed_nodes.remove(super_root_index)

    return list(transformed_pairs)


# def validate_reachability(graph: PyDiGraph, terminal_sets: dict[int, set[int]], pairs: list[tuple[int, int]]):
#     logger.trace("validate_reachability...")
#     node_key_by_index = graph.attrs["node_key_by_index"]
#     all_ts_reachable = True
#     for r, terminals in terminal_sets.items():
#         for t in terminals:
#             if not rx.has_path(graph, r, t):
#                 logger.error(f"Unreachable ts pair: {node_key_by_index[r]} → {node_key_by_index[t]}")
#                 all_ts_reachable = False
#     all_pairs_reachable = True
#     for s, t in pairs:
#         if not rx.has_path(graph, s, t):
#             logger.error(f"Unreachable pair: {node_key_by_index[s]} → {node_key_by_index[t]}")
#             all_pairs_reachable = False
#     if not all_ts_reachable and not all_pairs_reachable:
#         raise RuntimeError("Unreachable pairs")


# def get_node_key(graph: PyDiGraph, node_index: int) -> int:
#     key: int | None = graph.attrs.get("node_key_by_index", {}).get(node_index, None)
#     if key is None:
#         raise RuntimeError(f"Node {node_index} not found in index <-> key map.")
#     return key


# def reduction_degree(graph: PyDiGraph, node: int) -> int:
#     """Returns the degree of the given node pertaining to reduction semantics."""
#     # NOTE: In the node weighted setting all edges are treated as unweighted and undirected
#     #       reresentated by bi-directional anti-parallel arcs with the exception of a super root.
#     #       If present, the super root has incoming edges only from each potential root
#     #       that can satisfy an associated super terminal demand.
#     #       As such, for the purposes of graph reduction the SUPER_ROOT degree is zero,
#     #       potential roots use in_degree and out_degree is the determinant.
#     in_deg = graph.in_degree(node)
#     out_deg = graph.out_degree(node)
#     return min(in_deg, out_deg)


# def non_steiner_nodes(
#     graph: PyDiGraph, terminal_sets: dict[int, set[int]], fixed_nodes: set[int], super_root_index: int | None
# ) -> set[int]:
#     """Returns the set of nodes that are not steiner nodes."""
#     roots = set(terminal_sets.keys())

#     potential_roots = set()
#     if super_root_index is not None and super_root_index in roots:
#         node_key_by_index = graph.attrs["node_key_by_index"]
#         potential_roots = {node_key_by_index.inv[t] for t in get_exploration_data().towns}
#         all_active = set(terminal_sets.keys()) | {v for s in terminal_sets.values() for v in s}
#         non_steiner_nodes = potential_roots | all_active
#     else:
#         all_active = set(terminal_sets.keys()) | {v for s in terminal_sets.values() for v in s}
#         non_steiner_nodes = all_active

#     return non_steiner_nodes


# # MARK: Primitive Graph Actions


# def remove_node(graph: PyDiGraph, u: int):
#     """Removes the given node from the graph.
#     `G ≔ G ∖ ⓤ`
#     """
#     assert graph.has_node(u), f"node {get_node_key(graph, u)} not in graph"
#     graph.remove_node(u)


# def remove_nodes_from(graph: PyDiGraph, nodes: Iterable[int]):
#     """Removes the given node from the graph.
#     `∀ x : x ∈ nodes ⇒ G ≔ G ∖ ⓤ`
#     """
#     for x in nodes:
#         assert graph.has_node(x), f"node {get_node_key(graph, x)} not in graph"
#     graph.remove_nodes_from(nodes)


# def move_edges(graph: PyDiGraph, u: int, v: int):
#     """Move all incident edges of ⓤ to ⓥ, avoiding self-loops and parallel edges, isolates ⓤ.
#     ```
#     ∀ x : (ⓤ ⇋ x ∈ G ∧ x ≠ ⓥ) ⇒ (ⓥ ⇋ x ∈ G), ∣ⓤ∣
#     ```
#     """
#     assert graph.has_node(u), f"node {get_node_key(graph, u)} not in graph"
#     assert graph.has_node(v), f"node {get_node_key(graph, v)} not in graph"

#     preds = list(graph.predecessor_indices(u))
#     succs = list(graph.successor_indices(u))

#     for w in preds:
#         if w == v:
#             continue
#         if graph.has_edge(w, u):
#             graph.remove_edge(w, u)
#         if not graph.has_edge(w, v):
#             graph.add_edge(w, v, {})

#     for w in succs:
#         if w == v:
#             continue
#         if graph.has_edge(u, w):
#             graph.remove_edge(u, w)
#         if not graph.has_edge(v, w):
#             graph.add_edge(v, w, {})


# def absorb(graph: PyDiGraph, u: int, v: int):
#     """ⓥ absorbs ⓤ, ⓥ survives combining weights and embedded collapsed nodes.
#     **ⓤ ⇴ ⓥ**
#     ```
#     move_edges(ⓤ, ⓥ)
#     𝔀(ⓥ) ≔ 𝔀(ⓥ) + 𝔀(ⓤ)
#     ⓥ(⦿) ≔ ⓥ(⦿) ⋃ ⓤ(⦿) ⋃ {ⓤ}
#     remove(ⓤ)
#     ```
#     """
#     move_edges(graph, u, v)
#     graph[v][PAYLOAD_WEIGHT_KEY] += graph[u][PAYLOAD_WEIGHT_KEY]
#     graph[v]["collapsed_nodes"].update(graph[u]["collapsed_nodes"])
#     graph[v]["collapsed_nodes"].add(u)
#     remove_node(graph, u)


# def consume(graph: PyDiGraph, u: int, v: int):
#     """ⓥ consumes ⓤ, ⓥ survives.
#     **ⓤ ↣ ⓥ**
#     ```
#     move_edges(ⓤ, ⓥ)
#     remove(ⓤ)
#     ```
#     """
#     move_edges(graph, u, v)
#     remove_node(graph, u)


# # MARK: Reductions


# def reduce_demand_roots(
#     graph: PyDiGraph,
#     terminal_root_pairs: list[tuple[int, int]],
#     super_root_index: int | None = None,
# ) -> tuple[dict[int, int], dict[int, set[int]]]:
#     """
#     Reduces the number of distinct roots by merging fixed nodes (terminals or roots)
#     that lie in the same connected component.

#     Super Terminals are treated as independent of their root (since any super terminal
#     can be connected to any potential root).

#     This does NOT reduce the number of terminal-root pairs. It only canonicalizes
#     the root side of each pair and adds root->root pairs for collapsed roots.

#     Returns:
#         - list of reduced root pairs
#         - map of root: terminal set clusters
#     """
#     logger.trace("reduce_roots...")

#     node_key_by_index: bidict[int, int] = cast(bidict[int, int], graph.attrs["node_key_by_index"])

#     # NOTE: SUPER_ROOT directly connects to any base-town and any super terminal can
#     #       be connected to any base-town, so it is excluded from the fixed set, and
#     #       treated separately as isolated nodes during mapping.
#     logger.debug(f"reducing roots for {len(terminal_root_pairs)} pairs")
#     logger.trace(f"{ {node_key_by_index[n] for n, _ in terminal_root_pairs} }")

#     roots = {r for _, r in terminal_root_pairs}
#     roots.discard(super_root_index)
#     fixed_nodes = {t for t, _ in terminal_root_pairs} | roots

#     # This acts as a union-find data structure for the terminal set clusters.
#     sub = subgraph_stable(graph, fixed_nodes)
#     for u, v in terminal_root_pairs:
#         if v == super_root_index:
#             continue
#         _ = sub.add_edge(u, v, {})
#         _ = sub.add_edge(v, u, {})
#     components = rx.strongly_connected_components(sub)

#     terminal_sets: dict[int, set[int]] = {}
#     if super_root_index is not None:
#         terminal_sets[super_root_index] = set()

#     for comp in components:
#         comp_roots = set(comp) & roots
#         rep = next(iter(comp_roots), None)
#         if rep is not None:
#             terminal_sets[rep] = set(comp) - {rep}
#         elif super_root_index is not None:
#             terminal_sets[super_root_index].update(comp)
#         else:
#             raise RuntimeError(
#                 f"Component has no roots and super root is missing: {[node_key_by_index[n] for n in comp]}"
#             )

#     collapsed_pairs: dict[int, int] = {}
#     for r, comp in terminal_sets.items():
#         for t in comp:
#             collapsed_pairs[t] = r

#     return collapsed_pairs, terminal_sets


# def reduce_adjacent_terminals(
#     graph: PyDiGraph,
#     fixed_nodes: set[int],
#     terminal_sets: dict[int, set[int]],
#     super_root_index: int | None,
# ) -> tuple[dict[int, int], dict[int, set[int]]]:
#     """Reduces the number of distinct terminals by collapsing adjacent terminals per root.

#     NOTE: Run after reduce_roots (terminal_sets must be disjoint by root).
#     """
#     logger.trace("reduce_adjacent_terminals...")

#     def consume_adjacent_terminals(survivor: int, terminals: set[int]) -> bool:
#         # NOTE: rustworkx adjacency queries of node indices not present in the graph return empty sets.
#         neighbors = set(graph.predecessor_indices(survivor)) | set(graph.successor_indices(survivor))
#         consumables = neighbors & terminals
#         for c in list(consumables):
#             # absorb collapsed nodes into solution, ignoring the weight - since we are buying it.
#             fixed_nodes.update(graph[c]["collapsed_nodes"])
#             terminals.discard(c)
#             consume(graph, c, survivor)
#         return bool(consumables)

#     reduced_terminal_sets: dict[int, set[int]] = {r: set(ts) for r, ts in terminal_sets.items()}

#     for r, terminals in reduced_terminal_sets.items():
#         # Collapse terminal adjacent terminals
#         made_progress = True
#         while made_progress:
#             made_progress = False
#             for t in list(terminals):
#                 made_progress |= consume_adjacent_terminals(t, terminals)

#         # Collapse root adjacent terminals
#         if r != super_root_index:
#             consume_adjacent_terminals(r, terminals)

#     # Drop empty sets
#     for r, terminals in reduced_terminal_sets.copy().items():
#         if not terminals:
#             del reduced_terminal_sets[r]

#     # Rebuild pairs
#     reduced_pairs: dict[int, int] = {}
#     for r, comp in reduced_terminal_sets.items():
#         for t in comp:
#             reduced_pairs[t] = r

#     return reduced_pairs, reduced_terminal_sets


# def reduce_degree1_steiner_nodes(
#     graph: PyDiGraph,
#     terminal_sets: dict[int, set[int]],
#     fixed_nodes: set[int],
#     super_root_index: int | None,
# ):
#     """Multi-pass degree-1 steiner node reduction.

#     For any steiner node s with out_degree(s) == 1 remove s since it can not bridge or satisfy any demands.
#     """
#     logger.trace("reduce_degree1_steiner_nodes...")

#     non_steiners = non_steiner_nodes(graph, terminal_sets, fixed_nodes, super_root_index)
#     if super_root_index is not None:
#         non_steiners.discard(super_root_index)

#     removals = 0

#     while removables := {i for i in graph.node_indices() if graph.out_degree(i) == 1}:
#         removables.difference_update(non_steiners)
#         if not removables:
#             break
#         graph.remove_nodes_from(removables)
#         removals += len(removables)

#     logger.debug(f"  removed {removals} degree-1 steiner nodes")


# def reduce_degree1_terminals(
#     graph: PyDiGraph,
#     terminal_sets: dict[int, set[int]],
#     fixed_nodes: set[int],
#     super_root_index: int | None,
# ):
#     """Single-pass degree-1 terminal reduction.

#     For any terminal t with out_degree(t) == 1
#     collapse s into t by moving all edges of s to t and removing s.
#     Roots (keys of terminal_sets) and SUPER_ROOT are never collapsed.
#     Removed Steiner nodes are added to fixed_nodes.
#     """
#     logger.trace("reduce_degree1_terminals...")

#     roots = set(terminal_sets.keys())
#     removals = 0

#     for r, terminals in terminal_sets.items():
#         if r == super_root_index:
#             continue

#         # iterate over a snapshot; graph structure may change
#         for t in list(terminals):
#             if graph.out_degree(t) != 1:
#                 continue

#             succs = list(graph.successor_indices(t))
#             s = succs[0]

#             # Don't consume fixed nodes or SUPER_ROOT
#             # (adjacent terminals are handled in reduce_adjacent_terminals)
#             if s in fixed_nodes or s == super_root_index:
#                 logger.debug(
#                     f"  ...skipping degree-1 reduction: Root: {s in roots} Super Root: {s == super_root_index} Adjacent Terminal: {s in terminals} Fixed: {s in fixed_nodes}"
#                 )
#                 continue

#             # absorb collapsed nodes into solution, ignoring the weight - since we are buying it.
#             fixed_nodes.update(graph[s]["collapsed_nodes"])
#             fixed_nodes.add(s)
#             consume(graph, s, t)

#             removals += 1

#     logger.debug(f"  removed {removals} degree-1 terminals")


# def reduce_degree1_roots(
#     graph: PyDiGraph,
#     terminal_sets: dict[int, set[int]],
#     fixed_nodes: set[int],
#     super_root_index: int | None,
# ):
#     """Single-pass degree-1 root reduction.

#     For any root r with out_degree(r) == 1 with a non-empty terminal set and neighbor s
#     collapse s into r by moving all edges of s to r and removing s into fixed_nodes.

#     Adjacent fixed nodes (roots, terminals, super-root) are ignored here;
#     those reductions occur in other transforms.
#     """
#     logger.trace("reduce_degree1_roots...")

#     node_key_by_index = graph.attrs["node_key_by_index"]

#     # Only active roots are considered
#     roots = set(terminal_sets.keys())
#     roots.discard(super_root_index)
#     roots = roots - {r for r, ts in terminal_sets.items() if not ts}

#     removals = 0

#     # iterate over a snapshot; graph structure may change
#     for r in list(roots):
#         # A root, in the presence of a super_root, can have 2 successors
#         # but only ever 1 predecessor when reduction_degree == 1
#         if reduction_degree(graph, r) != 1:
#             continue
#         preds = list(graph.predecessor_indices(r))
#         s = preds[0]

#         # absorb collapsed nodes into solution, ignoring the weight - since we are buying it.
#         fixed_nodes.update(graph[s]["collapsed_nodes"])
#         fixed_nodes.add(s)
#         consume(graph, s, r)

#         # Fix terminal sets...
#         s_root = [r for r, ts in terminal_sets.items() if s in ts]
#         if s_root:
#             s_root = s_root[0]

#             if s_root == super_root_index:
#                 # A super-terminal was consumed
#                 logger.debug(f"  consumed super-terminal: {node_key_by_index[s]}")
#                 terminal_sets[s_root].remove(s)
#                 continue

#             if r == s_root:
#                 # A terminal was consumed of the same root
#                 logger.debug(f"  consumed terminal: {node_key_by_index[s]}")
#                 terminal_sets[r].remove(s)
#                 continue

#             terminal_sets[s_root].remove(s)
#             if r != s_root:
#                 # A terminal was consumed of a different root, union the sets
#                 logger.debug(
#                     f"  consumed terminal of different root: s: {node_key_by_index[s]}, s_root: {node_key_by_index[s_root]}"
#                 )
#                 terminal_sets[r].update(terminal_sets[s_root])
#                 terminal_sets[r].add(s_root)
#                 del terminal_sets[s_root]

#         removals += 1

#     if removals > 0:
#         # Drop empty sets
#         for r, terminals in terminal_sets.copy().items():
#             if not terminals:
#                 del terminal_sets[r]

#         if DO_DEBUG:
#             # Rebuild pairs
#             reduced_pairs: dict[int, int] = {}
#             for r, comp in terminal_sets.items():
#                 for t in comp:
#                     reduced_pairs[t] = r
#             validate_reachability(graph, terminal_sets, list(reduced_pairs.items()))

#             logger.debug(f"  removed {removals} degree-1 root Steiner nodes")


# def merge_adjacent_degree2_steiner_chains(
#     graph: PyDiGraph,
#     terminal_sets: dict[int, set[int]],
#     fixed_nodes: set[int],
#     super_root_index: int | None,
# ):
#     """
#     Merge degree-2 Steiner chains that are adjacent to each other.
#     """
#     logger.trace("merge_adjacent_degree2_steiner_chains...")

#     non_steiners = non_steiner_nodes(graph, terminal_sets, fixed_nodes, super_root_index)
#     removals = 0

#     made_progress = True
#     while made_progress:
#         made_progress = False

#         for node in set(graph.node_indices()) - non_steiners:
#             # A stale entry will have reduction_degree == 0
#             if reduction_degree(graph, node) != 2:
#                 continue

#             neighbor = next(
#                 (
#                     n
#                     for n in graph.predecessor_indices(node)
#                     if reduction_degree(graph, n) == 2 and n not in non_steiners
#                 ),
#                 None,
#             )
#             if neighbor:
#                 absorb(graph, neighbor, node)
#                 made_progress = True
#                 removals += 1
#                 break

#     logger.debug(f"  removed {removals} degree-2 nodes")


# def reduce_degree2_steiner_dominance(
#     graph: PyDiGraph,
#     terminal_sets: dict[int, set[int]],
#     fixed_nodes: set[int],
#     super_root_index: int | None,
# ):
#     """
#     Removes degree-2 Steiner nodes that are dominated by alternate shortest paths.
#     """
#     logger.trace("reduce_degree2_steiner_dominance...")

#     for u, v in graph.edge_list():
#         graph.update_edge(u, v, {PAYLOAD_WEIGHT_KEY: graph[v][PAYLOAD_WEIGHT_KEY]})

#     non_steiners = non_steiner_nodes(graph, terminal_sets, fixed_nodes, super_root_index)
#     removals = 0

#     made_progress = True
#     while made_progress:
#         made_progress = False

#         for s in set(graph.node_indices()) - non_steiners:
#             if graph.out_degree(s) != 2:
#                 continue

#             succ = list(graph.successor_indices(s))
#             u, v = succ
#             if u == v:
#                 continue

#             try:
#                 paths = rx.all_shortest_paths(graph, u, v, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY])
#             except rx.NoPathFound:
#                 continue

#             dominated = any(s not in P for P in paths)
#             if not dominated:
#                 continue

#             graph.remove_node(s)
#             made_progress = True

#             removals += 1
#             break

#     logger.debug(f"  removed {removals} degree-2 dominated nodes")


# def reduce_degreek_steiner_dominance(
#     graph: PyDiGraph,
#     terminal_sets: dict[int, set[int]],
#     fixed_nodes: set[int],
#     super_root_index: int | None,
#     max_degree: int = 4,
# ):
#     """
#     Removes Steiner nodes of small degree k ≥ 3 that are dominated by alternate paths.

#     Reduction Rule:
#     ```
#     Ⓢ : N(Ⓢ) = {○₁, ○₂, …, ○ₖ}, 2 < k ≤ K
#     ∧ ∀ {○ᵢ, ○ⱼ} ⊆ N(Ⓢ):
#         (○ᵢ ↠∣Ⓢ∣↠ ○ⱼ) exists
#         ∧ 𝔀(○ᵢ ↠∣Ⓢ∣↠ ○ⱼ) ≤ 𝔀(○ᵢ 🡒 Ⓢ 🡒 ○ⱼ)

#     ⭆  𝐆 ≔ 𝐆 ∖ Ⓢ
#     ```
#     """

#     logger.trace("reduce_degreek_steiner_dominance...")

#     # Project node weights onto edges (same as your degree-2 reduction)
#     for u, v in graph.edge_list():
#         graph.update_edge(u, v, {PAYLOAD_WEIGHT_KEY: graph[v][PAYLOAD_WEIGHT_KEY]})

#     non_steiners = non_steiner_nodes(graph, terminal_sets, fixed_nodes, super_root_index)
#     removals = 0

#     def dijkstra_avoiding(start: int, forbidden: int) -> dict[int, float]:
#         """Dijkstra that skips a single forbidden node."""
#         import heapq

#         dist: dict[int, float] = {start: 0.0}
#         heap = [(0.0, start)]

#         while heap:
#             d, u = heapq.heappop(heap)
#             if d > dist[u]:
#                 continue

#             for v in graph.successor_indices(u):
#                 if v == forbidden:
#                     continue

#                 w = graph.get_edge_data(u, v)[PAYLOAD_WEIGHT_KEY]
#                 nd = d + w

#                 if v not in dist or nd < dist[v]:
#                     dist[v] = nd
#                     heapq.heappush(heap, (nd, v))

#         return dist

#     made_progress = True
#     while made_progress:
#         made_progress = False

#         for s in set(graph.node_indices()) - non_steiners:
#             deg = graph.out_degree(s)

#             if deg <= 2 or deg > max_degree:
#                 continue

#             neighbors = list(graph.successor_indices(s))
#             if len(neighbors) != deg:
#                 continue

#             dominated = True

#             for i in range(len(neighbors)):
#                 if not dominated:
#                     break

#                 u = neighbors[i]

#                 # Full distances
#                 try:
#                     dist_full = rx.dijkstra_shortest_path_lengths(graph, u, lambda e: e[PAYLOAD_WEIGHT_KEY])
#                 except Exception:  # noqa: BLE001
#                     dominated = False
#                     break

#                 # Distances avoiding s
#                 dist_avoid = dijkstra_avoiding(u, s)

#                 for j in range(i + 1, len(neighbors)):
#                     v = neighbors[j]

#                     if v not in dist_full or v not in dist_avoid:
#                         dominated = False
#                         break

#                     if dist_avoid[v] > dist_full[v]:
#                         dominated = False
#                         break

#             if not dominated:
#                 continue

#             graph.remove_node(s)
#             removals += 1
#             made_progress = True
#             break

#     logger.debug(f"  removed {removals} degree-k dominated nodes")


# def reduce_blocked_roots(
#     graph: PyDiGraph,
#     terminal_sets: dict[int, set[int]],
#     super_root_index: int | None,
# ) -> tuple[dict[int, int], dict[int, set[int]]]:
#     logger.trace("reduce_blocked_roots...")

#     merges = 0
#     made_progress = True

#     while made_progress:
#         made_progress = False

#         # Build node → root lookup
#         node_to_root: dict[int, int] = {}
#         for r, ts in terminal_sets.items():
#             for t in ts:
#                 node_to_root[t] = r
#             node_to_root[r] = r

#         # Precompute cluster node sets
#         cluster_nodes: dict[int, set[int]] = {r: set(ts) | {r} for r, ts in terminal_sets.items()}

#         def is_blocking_cluster(t: int, r: int, blocked: set[int]) -> bool:
#             visited = set()
#             stack = [t]

#             while stack:
#                 u = stack.pop()

#                 if u == r:
#                     return False

#                 if u in visited or u in blocked:
#                     continue

#                 visited.add(u)

#                 stack.extend(graph.successor_indices(u))
#                 stack.extend(graph.predecessor_indices(u))

#             return True

#         root_dependencies: dict[int, set[int]] = {r: set() for r in terminal_sets}

#         for r_i, terminals in terminal_sets.items():
#             if r_i == super_root_index:
#                 continue

#             candidates: set[int] = set()

#             for t in terminals:
#                 visited = set()
#                 stack = [t]

#                 while stack:
#                     u = stack.pop()
#                     if u in visited:
#                         continue
#                     visited.add(u)

#                     if u in node_to_root:
#                         r_j = node_to_root[u]
#                         if r_j != r_i:
#                             candidates.add(r_j)
#                             continue  # stop expanding past other clusters

#                     stack.extend(graph.successor_indices(u))
#                     stack.extend(graph.predecessor_indices(u))

#             for r_j in candidates:
#                 Cj = cluster_nodes[r_j]

#                 if any(is_blocking_cluster(t, r_i, Cj) for t in terminals):
#                     root_dependencies[r_i].add(r_j)

#         for r_i, deps in list(root_dependencies.items()):
#             if r_i not in terminal_sets:
#                 continue

#             if len(deps) != 1:
#                 continue

#             r_j = next(iter(deps))

#             if r_j not in terminal_sets:
#                 continue

#             if r_j == super_root_index or r_i == r_j:
#                 continue

#             terminal_sets[r_i].update(terminal_sets[r_j])
#             terminal_sets[r_i].add(r_j)

#             del terminal_sets[r_j]

#             merges += 1
#             made_progress = True
#             break  # restart with fresh state

#     # Drop empty sets
#     for r, ts in list(terminal_sets.items()):
#         if not ts:
#             del terminal_sets[r]

#     # Rebuild pairs
#     reduced_pairs: dict[int, int] = {}
#     for r, comp in terminal_sets.items():
#         for t in comp:
#             reduced_pairs[t] = r

#     logger.debug(f"  merged {merges} blocked roots")

#     return reduced_pairs, terminal_sets


def transform_terminal_pairs(
    graph: PyDiGraph, fp_graph: fp.PyFastGraph, mappings: dict, terminals: dict
) -> tuple[set[int], dict[int, int]]:
    """Transforms the terminals dict using the reduced graph.

    Returns:
        set[int]: pre-added fixed terminals for solution
        dict[int, int]: terminal, root mapping
    """
    logger.trace("transform_terminal_pairs...")

    # NOTE: Incoming terminals dict contains waypoint pairs so we need to translate to indices
    #       before the transform and reduce and then translate back to waypoints after.
    node_key_by_index = graph.attrs.get("node_key_by_index", {})
    super_root_index = node_key_by_index.inv.get(SUPER_ROOT, None)
    terminal_idx_pairs = {node_key_by_index.inv[t]: node_key_by_index.inv[r] for t, r in terminals.items()}

    # Transform terminal pairs to reduced graph pairs
    fixed_nodes: set[int] = set()
    for node in graph.nodes():
        node["collapsed_nodes"] = set()
    transformed_pairs = transform_pairs_to_reduced_pairs(
        mappings,
        list(terminal_idx_pairs.items()),
        fixed_nodes,
        fp_graph,
        super_root_index,
    )

    if DO_DEBUG:
        fixed_nodes_wp = {node_key_by_index[n] for n in fixed_nodes}
        transformed_pairs_wp = [(node_key_by_index[t], node_key_by_index[r]) for t, r in transformed_pairs]
        logger.debug(f"  transformed pairs to {len(transformed_pairs)} working pairs")
        logger.trace(f"  {transformed_pairs_wp=}")
        logger.trace(f"  {fixed_nodes_wp=}")

    from sfgre import SFGraphReductionEngine

    potential_roots = {node_key_by_index.inv[t] for t in get_exploration_data().towns}
    terminal_sets = {}
    for t, r in transformed_pairs:
        if r not in terminal_sets:
            terminal_sets[r] = set()
        terminal_sets[r].add(t)

    reduction_engine = SFGraphReductionEngine(
        graph,
        graph.attrs["node_key_by_index"],
        super_root_index,
        fixed_nodes,
        potential_roots,
        terminal_sets,
        do_debug=DO_DEBUG,
    )
    fixed_nodes_wp, reduced_root_pairs_wp = reduction_engine.run_pipeline()

    return fixed_nodes_wp, reduced_root_pairs_wp


# def transform_terminal_pairs(
#     graph: PyDiGraph, fp_graph: fp.PyFastGraph, mappings: dict, terminals: dict
# ) -> tuple[set[int], dict[int, int]]:
#     """Transforms the terminals dict using the reduced graph.

#     Returns:
#         set[int]: pre-added fixed terminals for solution
#         dict[int, int]: terminal, root mapping
#     """
#     logger.trace("transform_terminal_pairs...")

#     # NOTE: Incoming terminals dict contains waypoint pairs so we need to translate to indices
#     #       before the transform and reduce and then translate back to waypoints after.
#     if DO_DEBUG:

#         def dump_pairs(msg: str, graph: PyDiGraph, pairs, sets):
#             logger.debug(msg)
#             logger.debug(f"  {graph.num_nodes()} nodes, {graph.num_edges()} edges")
#             pairs_wp = {node_key_by_index[t]: node_key_by_index[r] for t, r in pairs.items()}
#             sets_wp = {node_key_by_index[r]: {node_key_by_index[t] for t in ts} for r, ts in sets.items()}
#             logger.debug(f"  {len(pairs)} pairs")
#             logger.trace(f"  {pairs_wp=}")
#             logger.debug(f"  {sets_wp=}")
#             terminals_ok = True
#             for r, ts in sets.items():
#                 r_in_graph = graph.has_node(r)
#                 logger.debug(f"  {node_key_by_index[r]} in graph = {r_in_graph}")
#                 for t in ts:
#                     t_in_graph = graph.has_node(t)
#                     logger.debug(f"  {node_key_by_index[t]} in graph = {t_in_graph}")
#                     if not rx.has_path(graph, r, t):
#                         terminals_ok = False
#                         logger.error(f" Unreachable ts pair: {node_key_by_index[r]} → {node_key_by_index[t]}")

#             if not terminals_ok:
#                 raise RuntimeError("Unreachable demand pairs!")

#     node_key_by_index = graph.attrs.get("node_key_by_index", {})
#     super_root_index = node_key_by_index.inv.get(SUPER_ROOT, None)
#     terminal_idx_pairs = {node_key_by_index.inv[t]: node_key_by_index.inv[r] for t, r in terminals.items()}

#     # Transform terminal pairs to reduced graph pairs
#     fixed_nodes: set[int] = set()
#     for node in graph.nodes():
#         node["collapsed_nodes"] = set()
#     transformed_pairs = transform_pairs_to_reduced_pairs(
#         mappings,
#         list(terminal_idx_pairs.items()),
#         fixed_nodes,
#         fp_graph,
#         super_root_index,
#     )

#     if DO_DEBUG:
#         fixed_nodes_wp = {node_key_by_index[n] for n in fixed_nodes}
#         transformed_pairs_wp = [(node_key_by_index[t], node_key_by_index[r]) for t, r in transformed_pairs]
#         logger.debug(f"  transformed pairs to {len(transformed_pairs)} working pairs")
#         logger.trace(f"  {transformed_pairs_wp=}")
#         logger.trace(f"  {fixed_nodes_wp=}")

#     tmp_num_nodes = graph.num_nodes() + 1
#     while (num_nodes := graph.num_nodes()) != tmp_num_nodes:
#         tmp_num_nodes = num_nodes

#         # Isolates in the graph are always safe to remove
#         graph.remove_nodes_from(rx.isolates(graph))

#         reduced_root_pairs, rooted_terminal_sets = reduce_demand_roots(
#             graph, transformed_pairs, super_root_index
#         )
#         if DO_DEBUG:
#             dump_pairs("Post reduce_demand_roots:", graph, reduced_root_pairs, rooted_terminal_sets)

#         reduced_root_pairs, rooted_terminal_sets = reduce_adjacent_terminals(
#             graph, fixed_nodes, rooted_terminal_sets, super_root_index
#         )
#         if DO_DEBUG:
#             dump_pairs("Post reduce_adjacent_terminals:", graph, reduced_root_pairs, rooted_terminal_sets)

#         reduce_degree1_steiner_nodes(graph, rooted_terminal_sets, fixed_nodes, super_root_index)
#         if DO_DEBUG:
#             dump_pairs("Post reduce_degree1_steiner_nodes:", graph, reduced_root_pairs, rooted_terminal_sets)

#         # Reduce degree1 terminals - this is a Steiner node graph reduction only
#         reduce_degree1_terminals(graph, rooted_terminal_sets, fixed_nodes, super_root_index)
#         if DO_DEBUG:
#             dump_pairs("Post reduce_degree1_terminals:", graph, reduced_root_pairs, rooted_terminal_sets)

#         # Reduce degree1 roots - this is a Steiner node graph reduction only
#         reduce_degree1_roots(graph, rooted_terminal_sets, fixed_nodes, super_root_index)
#         if DO_DEBUG:
#             dump_pairs("Post reduce_degree1_roots:", graph, reduced_root_pairs, rooted_terminal_sets)

#         if DO_DEBUG:
#             logger.debug(f"Removed {num_nodes - graph.num_nodes()} nodes...")

#         if num_nodes != graph.num_nodes():
#             # Repeat the simple reductions if the graph has changed
#             logger.trace("Graph has changed, repeating simple reductions...")

#             # Rebuild pairs
#             reduced_root_pairs: dict[int, int] = {}
#             for r, comp in rooted_terminal_sets.items():
#                 for t in comp:
#                     reduced_root_pairs[t] = r
#             transformed_pairs = [(t, r) for t, r in reduced_root_pairs.items() if t != r]

#             continue

#         # Reduce 2-degree steiner chains - this is a Steiner node graph reduction by absorbtion only
#         merge_adjacent_degree2_steiner_chains(graph, rooted_terminal_sets, fixed_nodes, super_root_index)
#         if DO_DEBUG:
#             dump_pairs(
#                 "Post merge_adjacent_degree2_steiner_chains:", graph, reduced_root_pairs, rooted_terminal_sets
#             )

#         # Reduce 2-degree steiner dominance - this is a Steiner node graph reduction by dominance only
#         reduce_degree2_steiner_dominance(graph, rooted_terminal_sets, fixed_nodes, super_root_index)
#         if DO_DEBUG:
#             dump_pairs(
#                 "Post reduce_degree2_steiner_dominance:", graph, reduced_root_pairs, rooted_terminal_sets
#             )

#         # Reduce k-degree steiner dominance - this is a Steiner node graph reduction by dominance only
#         reduce_degreek_steiner_dominance(graph, rooted_terminal_sets, fixed_nodes, super_root_index)
#         if DO_DEBUG:
#             dump_pairs(
#                 "Post reduce_degreek_steiner_dominance:", graph, reduced_root_pairs, rooted_terminal_sets
#             )

#         # Reduce blocked roots - this is purely a terminal cluster set reduction
#         reduced_root_pairs, rooted_terminal_sets = reduce_blocked_roots(
#             graph, rooted_terminal_sets, super_root_index
#         )
#         if DO_DEBUG:
#             dump_pairs("Post reduce_blocked_roots:", graph, reduced_root_pairs, rooted_terminal_sets)

#         # Rebuild pairs
#         reduced_root_pairs: dict[int, int] = {}
#         for r, comp in rooted_terminal_sets.items():
#             for t in comp:
#                 reduced_root_pairs[t] = r

#         transformed_pairs = [(t, r) for t, r in reduced_root_pairs.items() if t != r]

#     fixed_nodes_wp = {node_key_by_index[n] for n in fixed_nodes}
#     reduced_root_pairs_wp = {
#         node_key_by_index[t]: node_key_by_index[r] for t, r in reduced_root_pairs.items()
#     }

#     if DO_DEBUG:
#         print("=== Post final reduction ===")
#         logger.debug(f"  {graph.num_nodes()} nodes, {graph.num_edges()} edges")
#         logger.debug(f"  {len(reduced_root_pairs)} pairs")
#         logger.trace(f"  {reduced_root_pairs_wp=}")
#         logger.trace(f"  {fixed_nodes_wp=}")
#         logger.trace(f"  num components: {rx.number_strongly_connected_components(graph)}")
#         node_degrees = [graph.out_degree(n) for n in graph.node_indices()]
#         nodes_by_degree = Counter(node_degrees)
#         nodes_by_degree = sorted(nodes_by_degree.items())
#         print(f"  nodes by degree: {nodes_by_degree}")

#         # Breakdown by degree into fixed, basetown, and steiner
#         towns = get_exploration_data().towns
#         for d, c in nodes_by_degree:
#             sub_counter = Counter()
#             for i in graph.node_indices():
#                 is_town = node_key_by_index[i] in towns
#                 if graph.out_degree(i) == d:
#                     if i in fixed_nodes:
#                         sub_counter["fixed"] += 1
#                     if is_town:
#                         sub_counter["basetown"] += 1
#                     if i not in fixed_nodes and not is_town:
#                         sub_counter["steiner"] += 1
#             print(f"    {d}: {sub_counter}")

#         logger.trace("=== PATHS ===")
#         for u, v in reduced_root_pairs.items():
#             has_u = graph.has_node(u)
#             has_v = graph.has_node(v)
#             has_path = rx.has_path(graph, u, v) if has_u and has_v else False
#             logger.trace(
#                 f"  {node_key_by_index[u]} ({has_u}) -> {node_key_by_index[v]} ({has_v}) => {has_path}"
#             )
#         logger.trace("===  ===")

#     return fixed_nodes_wp, reduced_root_pairs_wp


def validate_solution(graph: PyDiGraph, terminals: dict):
    """Confirms that there exists a path between each terminal and root."""
    exploration_data = get_exploration_data()
    node_key_by_index = graph.attrs["node_key_by_index"]

    has_all_paths = True
    towns = exploration_data.towns
    for t, r in terminals.items():
        t_idx = node_key_by_index.inv[t]
        has_path = False
        if r != SUPER_ROOT:
            r_idx = node_key_by_index.inv[r]
            has_path = rx.has_path(graph, t_idx, r_idx)
        else:
            # Super terminals can be reached from any of the towns
            for bt in towns:
                bt_idx = node_key_by_index.inv[bt]
                if has_path := rx.has_path(graph, t_idx, bt_idx):
                    break
        if has_path:
            continue
        has_all_paths = False
        logger.error(f"Failed to find a path from {t} to {r} in solution graph")
    assert has_all_paths


def optimize_with_terminals(terminals: dict, config: dict) -> Solution:
    """Optimization entry point using the HiGHS MIP solver."""
    # NOTE: `terminals` is a dict of {terminal waypoint: root waypoint}
    logger.debug(f"Optimizing with terminals: {terminals}")

    # NOTE: The MIP problem will have many extra variables if there
    #       is a SUPER ROOT present when no SUPER TERMINAL is present.
    exploration_data = get_exploration_data()
    has_super = SUPER_ROOT in terminals.values()

    # TODO: Utilize reduced exploration graph and reduced terminals

    # SAFETY: Deepcopy is required to avoid modifying the original graph upon attribute modification.
    #         The main graph is not deep copied because it is not modified and is used only for the subgraphing.
    if has_super:
        tmp_graph, mappings = exploration_data.reduced_super_graph
        exploration_graph_reduced = deepcopy(tmp_graph.copy())
        exploration_graph = deepcopy(exploration_data.super_graph.copy())
    else:
        tmp_graph, mappings = exploration_data.reduced_graph
        exploration_graph_reduced = deepcopy(tmp_graph.copy())
        exploration_graph = deepcopy(exploration_data.graph.copy())

    fp_graph = get_fp_graph(has_super)

    start_time = time.perf_counter()

    # NOTE: exploration_graph_reduced gets modified in-place during reduce_terminals
    fixed_nodes, reduced_terminals = transform_terminal_pairs(
        exploration_graph_reduced, fp_graph, mappings, terminals
    )

    if config["logger"]["level"] == "TRACE":
        tmp_G = exploration_graph_reduced
        print(f"{reduced_terminals=}")
        print("reduced graph nodes (as waypoints):")
        node_key_by_index = tmp_G.attrs["node_key_by_index"]
        print(f"{[node_key_by_index[n] for n in tmp_G.node_indices()]}")
        print("reduced graph undirected edges (waypoint endpoints):")
        print(f"{[(node_key_by_index[s], node_key_by_index[t]) for s, t in tmp_G.edge_list()]}")
        print("reduced graph node weights:")
        print(f"{ {node_key_by_index[n]: tmp_G[n][PAYLOAD_WEIGHT_KEY] for n in tmp_G.node_indices()} }")

    if reduced_terminals:
        set_graph_terminal_sets_attribute(exploration_graph_reduced, reduced_terminals)
        model = get_highs(config)
        model, vars = create_model(model, graph=exploration_graph_reduced)

        controller = SolverController()
        model = solve(model, config, controller)
        mip_solution_graph = extract_solution(model, vars, exploration_graph_reduced, config)

        # MIP Validation
        calculated_cost = sum(n[PAYLOAD_WEIGHT_KEY] for n in mip_solution_graph.nodes())
        objective_value = model.getObjectiveValue()
        objective_value = round(objective_value) if objective_value else 0
        assert calculated_cost == objective_value, (
            "Extraction error: Objective value does not match calculated cost!"
        )
    else:
        mip_solution_graph = PyDiGraph()

    duration = time.perf_counter() - start_time

    node_key_by_index = exploration_graph_reduced.attrs["node_key_by_index"]

    # Ensure all fixed nodes are included
    solution_nodes: set[int] = set(mip_solution_graph.node_indices())

    # Expand 2-degree collapsed chains
    # NOTE: fixed nodes are waypoints, collapsed nodes are indices
    hyper_nodes = {i for i in solution_nodes if exploration_graph_reduced[i]["collapsed_nodes"]}
    hyper_nodes = hyper_nodes & solution_nodes
    for i in hyper_nodes:
        logger.trace(
            f"  expanding {node_key_by_index[i]} containing {sorted([node_key_by_index[n] for n in exploration_graph_reduced[i]['collapsed_nodes']])}"
        )
        fixed_nodes.update([node_key_by_index[n] for n in exploration_graph_reduced[i]["collapsed_nodes"]])

    solution_nodes.update({node_key_by_index.inv[s] for s in fixed_nodes})
    solution_graph = subgraph_stable(exploration_graph, solution_nodes)
    set_graph_terminal_sets_attribute(solution_graph, terminals)

    objective_value = sum(n[PAYLOAD_WEIGHT_KEY] for n in solution_graph.nodes())
    objective_value = round(objective_value) if objective_value else 0

    num_components = len(rx.strongly_connected_components(solution_graph))
    solution = [solution_graph[i]["waypoint_key"] for i in solution_graph.node_indices()]
    solution = sorted(solution)

    if ds.get_config("config")["logger"]["level"] in ["DEBUG", "TRACE"]:
        logger.debug(f"Solution: {solution}")
        validate_solution(solution_graph, terminals)

    return Solution(
        duration=duration,
        cost=objective_value,
        num_nodes=solution_graph.num_nodes(),
        num_edges=solution_graph.num_edges(),
        num_components=num_components,
        waypoints=solution,
    )


if __name__ == "__main__":
    from orchestrator import Plan, execute_plan
    from orchestrator_terminal_pairs import PairingStrategy
    from test_baselines import baselines

    config = ds.get_config("config")
    config["name"] = "node_router"
    set_logger(config)

    strat_optimized = PairingStrategy.optimized
    strat_random_town = PairingStrategy.random_town

    def make_plan(
        budget: int,
        include_danger: bool,
        pairing_type: PairingStrategy,
        percent: int,
    ) -> Plan:
        return Plan(
            optimize_with_terminals,
            budget,
            percent,
            0,
            include_danger,
            pairing_type,
            False,
        )

    if config.get("actions", {}).get("baseline_tests", False):
        success = baselines(optimize_with_terminals, config)
        if not success:
            logger.error("Baseline tests failed!")
        else:
            logger.success("Baseline tests passed!")

    if config.get("actions", {}).get("scaling_tests", False):
        total_time_start = time.perf_counter()
        for budget in range(550, 555, 5):
            if budget >= 400:
                config["solver"]["log_via_callback"] = True
            print(f"Test: optimal terminals budget: {budget}")
            _ = execute_plan(make_plan(budget, False, strat_optimized, 0), config)
            # _ = execute_plan(make_plan(budget, True, strat_optimized, 0), config)

        # for percent in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
        for percent in [20]:
            if percent >= 5:
                config["solver"]["log_via_callback"] = True
            print(f"Test: random terminals coverage percent: {percent}")
            _ = execute_plan(make_plan(0, False, strat_random_town, percent), config)
            # _ = execute_plan(make_plan(0, True, strat_random_town, percent), config)
        # print(f"Cumulative testing runtime: {time.perf_counter() - total_time_start:.2f}s")

    # # fmt:off
    # terminals = {61:1, 301:1, 302:1, 601:1, 602:1, 604:1, 608:1, 1002:1, 1101:1, 1141:1, 1301:1, 1314:1, 1319:1, 1343:1, 1380:1, 1604:1, 1623:1, 1649:1, 1691:1, 1750:1, 1781:1, 1785:1, 1795:1, 1834:1, 1843:1, 1853:1, 1857:1, 1858:1, 2001:1}
    # # fmt:on
    # result = optimize_with_terminals(terminals, config)
    # print(result.waypoints)
    # print(result.cost)
