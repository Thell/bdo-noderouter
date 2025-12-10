# node_router.py

"""
Primal-dual Node-Weighted Steiner Forest approximation solver with bridge heuristics.
"""

import json
import time

import rustworkx as rx
from loguru import logger

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
    solution_waypoints, cost = NR.solve_for_terminal_pairs(list(terminals.items()))
    duration = time.perf_counter() - start_time

    logger.info(f"solution time (ms): {duration * 1000:.2f}")
    solution_indices = [WAYPOINT_TO_INDEX[w] for w in solution_waypoints]
    solution_graph = exploration_graph.subgraph(solution_indices)
    return ResultDict({"solution_graph": solution_graph, "objective_value": cost, "duration": duration})


if __name__ == "__main__":
    import testing as test
    from testing_baselines import baselines

    config = ds.get_config("config")
    config["name"] = "node_router"
    set_logger(config)

    if config.get("actions", {}).get("baseline_tests", False):
        success = baselines(optimize_with_terminals, config)
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
            test.generate_terminals(optimize_with_terminals, config, percent, False)
            test.generate_terminals(optimize_with_terminals, config, percent, True)
        print(f"Cumulative testing runtime: {time.perf_counter() - total_time_start:.2f}s")

    from api_exploration_graph import get_exploration_graph

    # fmt:off
    terminals = {61:1, 301:1, 302:1, 601:1, 602:1, 604:1, 608:1, 1002:1, 1101:1, 1141:1, 1301:1, 1314:1, 1319:1, 1343:1, 1380:1, 1604:1, 1623:1, 1649:1, 1691:1, 1750:1, 1781:1, 1785:1, 1795:1, 1834:1, 1843:1, 1853:1, 1857:1, 1858:1, 2001:1}
    # fmt:on
    exploration_graph = get_exploration_graph(config)
    assert isinstance(exploration_graph, rx.PyDiGraph)
    result = optimize_with_terminals(exploration_graph, terminals, config)
    print([n["waypoint_key"] for n in result["solution_graph"].nodes()])
    print(result["objective_value"])
