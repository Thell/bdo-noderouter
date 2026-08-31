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

import atexit
import threading
import time
from copy import deepcopy

import fast_paths as fp
import rustworkx as rx
from loguru import logger
from rustworkx import PyDiGraph

import api_data_store as ds
from api_common import PAYLOAD_WEIGHT_KEY, set_logger
from api_exploration_data import SUPER_ROOT, get_exploration_data
from api_highs_solver import SolverController, create_model, extract_solution, get_highs, solve
from api_rx_pydigraph import set_graph_terminal_sets_attribute, subgraph_stable
from orchestrator import Solution
from orchestrator_terminal_pairs import PairingStrategy


# Protect against stuck threads:
#    Fatal Python error: gilstate_tss_set: failed to set current tstate (TSS)
# caused by any of the numerous threads created by the HiGHS solver and rustworkx
def shutdown_barrier():
    main = threading.current_thread()
    for t in threading.enumerate():
        if t is main or t.daemon:
            continue
        # Best effort: don't hang forever if something is stuck
        t.join(timeout=5.0)


atexit.register(shutdown_barrier)

type LeafMap = dict[int, tuple[int, int, list[int]]]

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


def dump_reduction_results(
    msg: str,
    graph: PyDiGraph,
    terminal_sets: dict[int, set[int]],
    super_root_index: int | None,
    num_edges_start: int,
    num_nodes_start: int,
    num_terminal_roots_start: int,
    num_terminals_start: int,
):
    # Reduction percentages
    num_edges_end = graph.num_edges()
    num_nodes_end = graph.num_nodes()
    num_terminal_roots_end = len(terminal_sets)
    num_terminals_end = sum(len(ts) for ts in terminal_sets.values()) + num_terminal_roots_end
    per_edges = (num_edges_end - num_edges_start) / num_edges_start
    per_nodes = (num_nodes_end - num_nodes_start) / num_nodes_start
    per_terminals = (num_terminals_end - num_terminals_start) / num_terminals_start
    per_roots = (num_terminal_roots_end - num_terminal_roots_start) / num_terminal_roots_start
    num_super_terminals = 0 if super_root_index is None else len(terminal_sets.get(super_root_index, []))
    print(
        f"  {msg}: Reduction Percentages: Edges ({num_edges_start} -> {num_edges_end}): {per_edges * 100:.2f}%, Nodes ({num_nodes_start} -> {num_nodes_end}): {per_nodes * 100:.2f}%, Terminals ({num_terminals_start} -> {num_terminals_end}): {per_terminals * 100:.2f}%, Roots ({num_terminal_roots_start} -> {num_terminal_roots_end}): {per_roots * 100:.2f}% ({num_super_terminals} super terminals) [Trees solved: 0]"
    )


def transform_terminal_pairs(
    seed: str, graph: PyDiGraph, fp_graph: fp.PyFastGraph, mappings: dict, terminals: dict
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

    exploration_data = get_exploration_data()
    if super_root_index is not None:
        num_edges_start = exploration_data.super_graph.num_edges()
        num_nodes_start = exploration_data.super_graph.num_nodes()
    else:
        num_edges_start = exploration_data.graph.num_edges()
        num_nodes_start = exploration_data.graph.num_nodes()

    num_terminal_roots_start = len(set(terminals.values()))
    num_terminals_start = len(terminals) + num_terminal_roots_start

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

    # Reduce graph
    from sfgre import SFGraphReductionEngine

    potential_roots = {node_key_by_index.inv[t] for t in get_exploration_data().towns}
    terminal_sets = {}
    for t, r in transformed_pairs:
        if r not in terminal_sets:
            terminal_sets[r] = set()
        terminal_sets[r].add(t)

    for node in fixed_nodes:
        if graph.has_node(node):
            graph[node][PAYLOAD_WEIGHT_KEY] = 0

    dump_reduction_results(
        "Initial Transformation",
        graph,
        terminal_sets,
        super_root_index,
        num_edges_start,
        num_nodes_start,
        num_terminal_roots_start,
        num_terminals_start,
    )

    reduction_engine = SFGraphReductionEngine(
        seed,
        graph,
        graph.attrs["node_key_by_index"],
        super_root_index,
        fixed_nodes,
        potential_roots,
        terminal_sets,
        do_debug=DO_DEBUG,
    )
    fixed_nodes_wp, reduced_root_pairs_wp = reduction_engine.run_pipeline()

    # # A simple test bypassing reductions
    # fixed_nodes_wp = {node_key_by_index[n] for n in fixed_nodes}
    # reduced_root_pairs_wp: dict[int, int] = {
    #     node_key_by_index[t]: node_key_by_index[r] for t, r in transformed_pairs
    # }

    return fixed_nodes_wp, reduced_root_pairs_wp


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


def optimize_with_terminals(seed: str, terminals: dict, config: dict) -> Solution:
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
        seed, exploration_graph_reduced, fp_graph, mappings, terminals
    )

    if config["logger"]["level"] == "TRACE":
        print("=== REDUCED GRAPH ===")
        tmp_G = exploration_graph_reduced
        node_key_by_index = tmp_G.attrs["node_key_by_index"]
        print(f"    terminals = {reduced_terminals}")
        print(f"    actual = { {node_key_by_index[n] for n in tmp_G.node_indices()} }")
        print(
            f"    actual_edges = {[(node_key_by_index[s], node_key_by_index[t]) for s, t in tmp_G.edge_list()]}"
        )
        print(
            f"    actual_weights = { {node_key_by_index[n]: tmp_G[n][PAYLOAD_WEIGHT_KEY] for n in tmp_G.node_indices()} }"
        )
        print("=== END REDUCED GRAPH ===")

    if reduced_terminals:
        print(f"MIP: num_nodes: {exploration_graph_reduced.num_nodes()}, num_fixed_nodes: {len(fixed_nodes)}")
        set_graph_terminal_sets_attribute(exploration_graph_reduced, reduced_terminals)
        model = get_highs(config)
        if exploration_graph_reduced.num_nodes() / 2 < len(fixed_nodes):
            model, vars = create_model(model, graph=exploration_graph_reduced)
        else:
            model, vars = create_model(model, graph=exploration_graph_reduced, fixed_nodes=fixed_nodes)

        controller = SolverController()
        model = solve(model, config, controller)
        mip_solution_graph = extract_solution(model, vars, exploration_graph_reduced, config)

        # MIP Validation
        calculated_cost = sum(n[PAYLOAD_WEIGHT_KEY] for n in mip_solution_graph.nodes())
        objective_value = model.getObjectiveValue()
        objective_value = round(objective_value) if objective_value else 0
        # assert calculated_cost == objective_value, (
        if calculated_cost != objective_value:
            logger.error("Extraction error: Objective value does not match calculated cost!")
    else:
        mip_solution_graph = PyDiGraph()

    duration = time.perf_counter() - start_time

    node_key_by_index = exploration_graph_reduced.attrs["node_key_by_index"]

    # Ensure all fixed nodes are included
    solution_nodes: set[int] = set(mip_solution_graph.node_indices())

    if config["logger"]["level"] == "TRACE":
        print("=== MIP SOLUTION OF REDUCED GRAPH ===")
        if mip_solution_graph.num_nodes() == 0:
            logger.warning("    no nodes in solution graph!")
        else:
            tmp_G = mip_solution_graph
            node_key_by_index = tmp_G.attrs["node_key_by_index"]
            print(f"    terminals = {reduced_terminals}")
            print(f"    actual = { {node_key_by_index[n] for n in tmp_G.node_indices()} }")
            print(
                f"    actual_edges = {[(node_key_by_index[s], node_key_by_index[t]) for s, t in tmp_G.edge_list()]}"
            )
            print(
                f"    actual_weights = { {node_key_by_index[n]: tmp_G[n][PAYLOAD_WEIGHT_KEY] for n in tmp_G.node_indices()} }"
            )
            print("=== END OF MIP SOLUTION OF REDUCED GRAPH ===")

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

    if config["logger"]["level"] == "TRACE":
        print("=== MIP EXPANDED FINAL SOLUTION OF REDUCED GRAPH ===")
        tmp_G = solution_graph
        node_key_by_index = tmp_G.attrs["node_key_by_index"]
        print(f"    terminals = {terminals}")
        print(f"    actual = { {node_key_by_index[n] for n in tmp_G.node_indices()} }")
        print(
            f"    actual_edges = {[(node_key_by_index[s], node_key_by_index[t]) for s, t in tmp_G.edge_list()]}"
        )
        print(
            f"    actual_weights = { {node_key_by_index[n]: tmp_G[n][PAYLOAD_WEIGHT_KEY] for n in tmp_G.node_indices()} }"
        )
        print("=== END OF EXPANDED FINAL MIP SOLUTION OF REDUCED GRAPH ===")

    # if True:
    #     print("=== FINAL MIP SOLUTION SUPER TERMINAL ROOTS ===")
    #     tmpG = solution_graph = subgraph_stable(exploration_graph, solution_nodes | {node_key_by_index.inv[99999]})
    #     node_key_by_index = solution_graph.attrs["node_key_by_index"]
    #     print(f"terminal wps: {terminals}")
    #     print(f"terminal indices: { {node_key_by_index.inv[t]: node_key_by_index.inv[r] for t, r in terminals.items()} }")
    #     all_terminals = set(terminals.keys())
    #     for t in (t for t, r in terminals.items() if r == 99999):
    #         paths = rx.all_shortest_paths(tmpG, node_key_by_index.inv[t], node_key_by_index.inv[99999])
    #         for p in paths:
    #             weight = sum(tmpG[n][PAYLOAD_WEIGHT_KEY] for n in p)
    #             print(f"path (weight: {weight}): {p} (terminal: { t }, root: { node_key_by_index[p[-2]] })")
    #         # Find first terminal in path after the super terminal
    #         for p in paths:
    #             for u in p[1:]:
    #                 if node_key_by_index[u] in fixed_nodes_copy:
    #                     length_to_terminal = sum(tmpG[n][PAYLOAD_WEIGHT_KEY] for n in p[:p.index(u)])
    #                     print(f"first terminal: {u} {node_key_by_index[u]} (length: {length_to_terminal})")
    #                     break
    #             else:
    #                 print("no 'attachment' terminal found in path")
    #             break

    if ds.get_config("config")["logger"]["level"] in ["DEBUG", "TRACE"]:
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
    config["logger"]["level"] = "TRACE"
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
        for budget in range(5, 555, 50):
            # if budget >= 400:
            #     config["solver"]["log_via_callback"] = True
            print(f"Test: optimal terminals budget: {budget}")
            _ = execute_plan(make_plan(budget, False, strat_optimized, 0), config)
            # _ = execute_plan(make_plan(budget, True, strat_optimized, 0), config)

        # # for percent in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
        # for percent in [20]:
        #     if percent >= 5:
        #         config["solver"]["log_via_callback"] = True
        #     print(f"Test: random terminals coverage percent: {percent}")
        #     _ = execute_plan(make_plan(0, False, strat_random_town, percent), config)
        #     # _ = execute_plan(make_plan(0, True, strat_random_town, percent), config)

        print(f"Cumulative testing runtime: {time.perf_counter() - total_time_start:.2f}s")

    # # Example: Run one-off for
    # # f3bd7a9 random_n_cheapest_town_in_territory  Budget: 20
    # MAX_BUDGET = 550
    # budget = 20
    # percent = round(budget / MAX_BUDGET * 100)
    # seed = "f3bd7a9"
    # mip_plan = Plan(
    #     optimize_with_terminals,
    #     budget,
    #     percent,
    #     seed,
    #     False,
    #     PairingStrategy.random_n_cheapest_town_in_territory,
    #     False,
    # )
    # execute_plan(mip_plan, config)

    # # fmt:off
    # terminals = {61:1, 301:1, 302:1, 601:1, 602:1, 604:1, 608:1, 1002:1, 1101:1, 1141:1, 1301:1, 1314:1, 1319:1, 1343:1, 1380:1, 1604:1, 1623:1, 1649:1, 1691:1, 1750:1, 1781:1, 1785:1, 1795:1, 1834:1, 1843:1, 1853:1, 1857:1, 1858:1, 2001:1}
    # # fmt:on
    # result = optimize_with_terminals(terminals, config)
    # print(result.waypoints)
    # print(result.cost)

    from api_common import MAX_BUDGET

    # # Example: Run one-off for
    # # 95faa43 nearest_captial  Budget: 150
    budget = 145
    percent = round(budget / MAX_BUDGET * 100)
    seed = "86e63d9"
    mip_plan = Plan(
        optimize_with_terminals,
        budget,
        percent,
        seed,
        True,
        PairingStrategy.cheapest_town,
        False,
    )
    print(mip_plan)
    execute_plan(mip_plan, config)

    # # Example: Run one-off for
    # # 55e1b9d random_n_cheapest_town_in_territory  Budget: 215
    # # CONFIRMED SUBOPTIMAL SCIPSTP result using reduciton 2
    # budget = 215
    # percent = round(budget / MAX_BUDGET * 100)
    # seed = "55e1b9d"
    # mip_plan = Plan(
    #     optimize_with_terminals,
    #     budget,
    #     percent,
    #     seed,
    #     False,
    #     PairingStrategy.random_n_cheapest_town_in_territory,
    #     False,
    # )
    # print(mip_plan)
    # execute_plan(mip_plan, config)
