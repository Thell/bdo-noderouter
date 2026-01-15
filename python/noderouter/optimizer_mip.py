# optimizer_mip.py

"""
This module formulates our integer baseline of the Steiner Forest problem solution
from a multi-commodity flow problem with cumulative flow from the terminals to the roots.

Given:
- a directed graph of the exploration data
- a 'terminals' dict in the form of {terminal: root, ...}:
- a config dict
"""

from __future__ import annotations
from copy import deepcopy
import time

import rustworkx as rx
from loguru import logger

import api_data_store as ds
from api_common import set_logger
from api_rx_pydigraph import set_graph_terminal_sets_attribute
from api_highs_solver import get_highs, create_model, solve, extract_solution
from api_exploration_data import get_exploration_data, SUPER_ROOT
from orchestrator_terminal_pairs import PairingStrategy
from orchestrator import Solution


def optimize_with_terminals(terminals: dict, config: dict) -> Solution:
    """Optimization entry point using the HiGHS MIP solver."""
    logger.debug(f"Optimizing with terminals: {terminals}")

    # NOTE: The MIP problem will have many extra variables if there
    #       is a SUPER ROOT present when no SUPER TERMINAL is present.
    exploration_data = get_exploration_data()
    has_super = SUPER_ROOT in terminals.values()

    # SAFETY: Deepcopy is required to avoid modifying the original graph upon attribute modification
    if has_super:
        exploration_graph = deepcopy(exploration_data.super_graph.copy())
    else:
        exploration_graph = deepcopy(exploration_data.graph.copy())
    set_graph_terminal_sets_attribute(exploration_graph, terminals)

    model = get_highs(config)
    model, vars = create_model(model, graph=exploration_graph)

    start_time = time.perf_counter()
    model = solve(model, config)
    duration = time.perf_counter() - start_time

    solution_graph = extract_solution(model, vars, exploration_graph, config)
    calculated_cost = sum(n["need_exploration_point"] for n in solution_graph.nodes())

    objective_value = model.getObjectiveValue()
    objective_value = round(objective_value) if objective_value else 0
    assert calculated_cost == objective_value, (
        "Extraction error: Objective value does not match calculated cost!"
    )

    num_components = len(rx.strongly_connected_components(solution_graph))
    solution = [solution_graph[i]["waypoint_key"] for i in solution_graph.node_indices()]
    solution = sorted(solution)
    return Solution(
        duration=duration,
        cost=objective_value,
        num_nodes=solution_graph.num_nodes(),
        num_edges=solution_graph.num_edges(),
        num_components=num_components,
        waypoints=solution,
    )


if __name__ == "__main__":
    from orchestrator import execute_plan, Plan
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
        return Plan(optimize_with_terminals, config, budget, percent, 0, include_danger, pairing_type, False)

    if config.get("actions", {}).get("baseline_tests", False):
        success = baselines(optimize_with_terminals, config)
        if not success:
            logger.error("Baseline tests failed!")
        else:
            logger.success("Baseline tests passed!")

    if config.get("actions", {}).get("scaling_tests", False):
        total_time_start = time.perf_counter()
        for budget in range(5, 555, 5):
            print(f"Test: optimal terminals budget: {budget}")
            _ = execute_plan(make_plan(budget, False, strat_optimized, 0))
            _ = execute_plan(make_plan(budget, True, strat_optimized, 0))
        for percent in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
            print(f"Test: random terminals coverage percent: {percent}")
            _ = execute_plan(make_plan(0, False, strat_random_town, percent))
            _ = execute_plan(make_plan(0, True, strat_random_town, percent))
        print(f"Cumulative testing runtime: {time.perf_counter() - total_time_start:.2f}s")

    # # fmt:off
    # terminals = {61:1, 301:1, 302:1, 601:1, 602:1, 604:1, 608:1, 1002:1, 1101:1, 1141:1, 1301:1, 1314:1, 1319:1, 1343:1, 1380:1, 1604:1, 1623:1, 1649:1, 1691:1, 1750:1, 1781:1, 1785:1, 1795:1, 1834:1, 1843:1, 1853:1, 1857:1, 1858:1, 2001:1}
    # # fmt:on
    # result = optimize_with_terminals(terminals, config)
    # print(result.waypoints)
    # print(result.cost)
