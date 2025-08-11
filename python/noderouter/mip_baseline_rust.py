# mip_baseline.py
"""
This module formulates our integer baseline of the Steiner Forest problem solution
from a multi-commodity flow problem with cumulative flow from the terminals to the roots.

Given:
- a directed graph of the exploration data
- a 'terminals' dict in the form of {terminal: root, ...}:
- a config dict
"""

from __future__ import annotations
from typing import TypedDict

from loguru import logger
import rustworkx as rx

import data_store as ds
from api_common import set_logger, SUPER_ROOT
from api_rx_pydigraph import set_graph_terminal_sets_attribute, inject_super_root
from api_highs_model import get_highs, create_model, solve
from api_solution_handling import extract_solution_from_x_vars


class ResultDict(TypedDict):
    solution_graph: rx.PyDiGraph
    objective_value: int | None


def optimize_with_terminals(exploration_graph: rx.PyDiGraph, terminals: dict, config: dict) -> ResultDict:
    G = exploration_graph

    if SUPER_ROOT in terminals.values():
        inject_super_root(config, G)
    set_graph_terminal_sets_attribute(G, terminals)

    model = get_highs(config)
    model, vars = create_model(model, graph=G)
    model = solve(model, config)

    solution_graph = extract_solution_from_x_vars(model, vars, G, config)

    objective_value = model.getObjectiveValue()
    objective_value = round(objective_value) if objective_value else 0
    calculated_cost = sum(n["need_exploration_point"] for n in solution_graph.nodes())
    if calculated_cost != objective_value:
        logger.warning("Objective value does not match calculated cost!")

    return {
        "solution_graph": solution_graph,
        "objective_value": objective_value,
    }


if __name__ == "__main__":
    import time
    import testing as test

    config = ds.get_config("config")
    config["name"] = "mip_baseline"
    set_logger(config)

    if config.get("actions", {}).get("baseline_tests", False):
        success = test.baselines(optimize_with_terminals, config)
        if not success:
            raise ValueError("Baseline tests failed!")
        logger.success("Baseline tests passed!")

    if config.get("actions", {}).get("scaling_tests", False):
        total_time_start = time.perf_counter()
        for budget in range(5, 555, 5):
            print(f"Test: optimal terminals budget: {budget}")
            test.workerman_terminals(optimize_with_terminals, config, budget, False)
            test.workerman_terminals(optimize_with_terminals, config, budget, True)
        for percent in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
            print(f"Test: random terminals coverage percent: {percent}")
            test.random_terminals(optimize_with_terminals, config, percent, False, max_danger=5)
            test.random_terminals(optimize_with_terminals, config, percent, True, max_danger=5)
        print(f"Cumulative testing runtime: {time.perf_counter() - total_time_start:.2f}s")
