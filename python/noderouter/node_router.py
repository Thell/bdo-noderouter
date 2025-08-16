# node_router.py

"""
Primal-dual Node-Weighted Steiner Forest approximation solver with bridge heuristics.
"""

import json
import time

import rustworkx as rx
from loguru import logger

from api_exploration_graph import get_exploration_graph
import data_store as ds
from api_common import set_logger, ResultDict, get_clean_exploration_data
from noderouter import NodeRouter

NR = None
WAYPOINT_TO_INDEX: dict[int, int] = {}


def optimize_with_terminals(
    exploration_graph: rx.PyDiGraph, terminals: dict[int, int], config: dict
) -> ResultDict:
    """Public-facing function to optimize graph with terminals."""
    global NR, WAYPOINT_TO_INDEX

    if NR is None:
        index_to_waypoint: dict[int, int] = dict({
            i: exploration_graph[i]["waypoint_key"] for i in exploration_graph.node_indices()
        })
        WAYPOINT_TO_INDEX = {v: k for k, v in index_to_waypoint.items()}
        exploration_data = get_clean_exploration_data(config)
        exploration_json_dumps = json.dumps(exploration_data)
        NR = NodeRouter(exploration_json_dumps)
    logger.debug(f"Optimizing graph with {len(terminals)} terminals...")
    start_time = time.perf_counter()

    solution_waypoints = NR.solve_for_terminal_pairs(list(terminals.items()))

    logger.info(f"solution time (ms): {(time.perf_counter() - start_time) * 1000:.2f}")
    solution_indices = [WAYPOINT_TO_INDEX[w] for w in solution_waypoints]
    solution_graph = exploration_graph.subgraph(solution_indices)
    objective_value = sum(v["need_exploration_point"] for v in solution_graph.nodes())
    return ResultDict({"solution_graph": solution_graph, "objective_value": objective_value})


if __name__ == "__main__":
    import testing as test

    config = ds.get_config("config")
    config["name"] = "node_router"
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

    # Direct Super-Terminal tests
    test_sets = [
        ({1132: 99999}, 5),
        ({1152: 99999}, 7),
        ({1154: 99999}, 6),
        ({1160: 99999}, 6),
        ({1162: 99999}, 5),
        ({155: 1, 1132: 99999}, 9),
        ({155: 1, 1151: 99999}, 10),
        ({155: 1, 1152: 99999}, 11),
        ({155: 1, 1154: 99999}, 10),
        ({155: 1, 1160: 99999}, 10),
        ({155: 1, 1162: 99999}, 9),
        ({160: 1, 1132: 99999}, 8),
        ({155: 1, 1132: 99999, 1152: 99999}, 11),
        ({155: 1, 1132: 99999, 1136: 99999, 1152: 99999}, 14),
    ]
    graph = get_exploration_graph(config)
    assert isinstance(graph, rx.PyDiGraph)
    for terminals, optimal_cost in test_sets:
        print(f"\nOptimizing graph with {terminals=}...")
        result = optimize_with_terminals(graph, terminals, config)
        solution_graph = result["solution_graph"]
        objective_value = result["objective_value"]
        solution_waypoints = []
        for node in solution_graph.nodes():
            solution_waypoints.append(node["waypoint_key"])
        logger.info(solution_waypoints)
        if objective_value == optimal_cost:
            logger.success(f"PASS: Solution cost: {objective_value}")
        else:
            logger.error(f"FAIL: Solution cost: {objective_value} (expected: {optimal_cost})")
