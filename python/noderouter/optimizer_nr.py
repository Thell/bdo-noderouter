# optimizer_nr.py

"""
Primal-dual Node-Weighted Steiner Forest approximation solver with bridge heuristics.
"""

import json
import time

import rustworkx as rx
from loguru import logger

from api_exploration_data import get_exploration_data
from noderouter import NodeRouter
from orchestrator import Solution

_NODEROUTER: NodeRouter | None = None


def _optimize_with_terminals(nr: NodeRouter, terminals: dict[int, int]) -> Solution:
    logger.debug(f"Optimizing graph with {len(terminals)} terminals...")

    start_time = time.perf_counter()
    solution_waypoints, cost = nr.solve_for_terminal_pairs(list(terminals.items()))
    duration = time.perf_counter() - start_time
    logger.info(f"solution time (ms): {duration * 1000:.2f}")

    exploration_graph = get_exploration_data().super_graph
    node_key_by_index = exploration_graph.attrs["node_key_by_index"]

    solution_indices = [node_key_by_index.inv[w] for w in solution_waypoints]
    solution_graph = exploration_graph.subgraph(solution_indices)

    return Solution(
        duration=duration,
        cost=cost,
        num_nodes=solution_graph.num_nodes(),
        num_edges=solution_graph.num_edges(),
        num_components=len(rx.strongly_connected_components(solution_graph)),
        waypoints=solution_waypoints,
    )


def optimize_with_terminals(terminals: dict[int, int], _config: dict) -> Solution:
    """Primary optimization entry point using a long-lived NodeRouter instance.

    The NodeRouter instance is constructed once and reused across calls to avoid
    repeated initialization overhead.

    SAFETY: Not thread-safe. The underlying Rust implementation is not Send;
    moving the instance across thread boundaries will panic.
    """
    global _NODEROUTER
    if _NODEROUTER is None:
        exploration_data = get_exploration_data()
        exploration_json_dumps = json.dumps(exploration_data.data)
        _NODEROUTER = NodeRouter(exploration_json_dumps)

    return _optimize_with_terminals(_NODEROUTER, terminals)


def optimize_with_terminals_single(terminals: dict[int, int], _config: dict) -> Solution:
    """One-off optimization creating a fresh NodeRouter instance each call.

    Constructs and immediately destroys the NodeRouter within the same thread,
    avoiding any cross-thread lifetime issues.

    SAFETY: Thread-safe.
    """
    exploration_data = get_exploration_data()
    exploration_json_dumps = json.dumps(exploration_data.data)
    nr = NodeRouter(exploration_json_dumps)
    return _optimize_with_terminals(nr, terminals)


if __name__ == "__main__":
    import api_data_store as ds
    from api_common import set_logger
    from orchestrator import execute_plan, Plan
    from orchestrator_terminal_pairs import PairingStrategy
    from test_baselines import baselines

    config = ds.get_config("config")
    config["name"] = "node_router"
    set_logger({"logger": {"level": "ERROR", "format": "<level>{message}</level>"}})

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
