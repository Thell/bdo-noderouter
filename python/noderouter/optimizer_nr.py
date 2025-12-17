# optimizer_nr.py

"""
Primal-dual Node-Weighted Steiner Forest approximation solver with bridge heuristics.
"""

import json
import time

import rustworkx as rx
from loguru import logger

import api_data_store as ds
from api_common import set_logger, memory
from api_exploration_data import get_exploration_data
from noderouter import NodeRouter
from orchestrator_terminal_pairs import PairingStrategy
from orchestrator import Solution

NR = None
WAYPOINT_TO_INDEX: dict[int, int] = {}


def optimize_with_terminals(terminals: dict[int, int], _config: dict) -> Solution:
    """Public-facing function to optimize graph with terminal pairs."""
    # NOTE: NodeRouter's graph doesn't really care if there is a SUPER ROOT
    # present when no SUPER TERMINAL is present. So we just use super_graph.
    global NR, WAYPOINT_TO_INDEX

    exploration_data = get_exploration_data()
    exploration_graph = exploration_data.super_graph

    if NR is None:
        index_to_waypoint: dict[int, int] = dict({
            i: exploration_graph[i]["waypoint_key"] for i in exploration_graph.node_indices()
        })
        WAYPOINT_TO_INDEX = {v: k for k, v in index_to_waypoint.items()}
        exploration_json_dumps = json.dumps(exploration_data.data)
        NR = NodeRouter(exploration_json_dumps)
    logger.debug(f"Optimizing graph with {len(terminals)} terminals...")

    start_time = time.perf_counter()
    solution_waypoints, cost = NR.solve_for_terminal_pairs(list(terminals.items()))
    duration = time.perf_counter() - start_time
    NR = None  # Resetting releases the NodeRouter instance.

    solution_indices = [WAYPOINT_TO_INDEX[w] for w in solution_waypoints]
    solution_graph = exploration_graph.subgraph(solution_indices)

    logger.info(f"solution time (ms): {duration * 1000:.2f}")

    return Solution(
        duration=duration,
        cost=cost,
        num_nodes=solution_graph.num_nodes(),
        num_edges=solution_graph.num_edges(),
        num_components=len(rx.strongly_connected_components(solution_graph)),
        waypoints=solution_waypoints,
    )


def optimize_with_terminals_single(terminals: dict[int, int], _config: dict) -> Solution:
    """Visualizer function to optimize graph with terminal pairs."""
    # NOTE: NodeRouter's graph doesn't really care if there is a SUPER ROOT
    # present when no SUPER TERMINAL is present. So we just use super_graph.
    exploration_data = get_exploration_data()
    exploration_graph = exploration_data.super_graph

    index_to_waypoint: dict[int, int] = dict({
        i: exploration_graph[i]["waypoint_key"] for i in exploration_graph.node_indices()
    })
    waypoint_to_index = {v: k for k, v in index_to_waypoint.items()}

    exploration_json_dumps = json.dumps(exploration_data.data)
    nr = NodeRouter(exploration_json_dumps)

    logger.debug(f"Optimizing graph with {len(terminals)} terminals...")
    start_time = time.perf_counter()
    solution_waypoints, cost = nr.solve_for_terminal_pairs(list(terminals.items()))
    duration = time.perf_counter() - start_time

    solution_indices = [waypoint_to_index[w] for w in solution_waypoints]
    solution_graph = exploration_graph.subgraph(solution_indices)

    logger.info(f"solution time (ms): {duration * 1000:.2f}")

    return Solution(
        duration=duration,
        cost=cost,
        num_nodes=solution_graph.num_nodes(),
        num_edges=solution_graph.num_edges(),
        num_components=len(rx.strongly_connected_components(solution_graph)),
        waypoints=solution_waypoints,
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
        for budget in range(5, 25, 5):
            print(f"Test: optimal terminals budget: {budget}")
            _ = execute_plan(make_plan(budget, False, strat_optimized, 0))
            _ = execute_plan(make_plan(budget, True, strat_optimized, 0))
        for percent in [1, 2, 3, 4, 5]:
            print(f"Test: random terminals coverage percent: {percent}")
            _ = execute_plan(make_plan(0, False, strat_random_town, percent))
            _ = execute_plan(make_plan(0, True, strat_random_town, percent))
        print(f"Cumulative testing runtime: {time.perf_counter() - total_time_start:.2f}s")

    # fmt:off
    terminals = {61:1, 301:1, 302:1, 601:1, 602:1, 604:1, 608:1, 1002:1, 1101:1, 1141:1, 1301:1, 1314:1, 1319:1, 1343:1, 1380:1, 1604:1, 1623:1, 1649:1, 1691:1, 1750:1, 1781:1, 1785:1, 1795:1, 1834:1, 1843:1, 1853:1, 1857:1, 1858:1, 2001:1}
    # fmt:on
    result = optimize_with_terminals(terminals, config)
    print(result.waypoints)
    print(result.cost)
