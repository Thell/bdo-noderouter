# testing.py

"""
Testing module for Steiner Forest Problem solver, providing:
- Baseline tests for simple cases.
- Random terminal tests for stress testing edge cases.
- Workerman terminal tests for benchmarking and validity (Empire Optimizer solutions).
"""

from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, TypedDict

from pathlib import Path
import json
import random
import time

import rustworkx as rx
from loguru import logger

import data_store as ds
from api_common import get_clean_exploration_data, save_graph, set_logger, SUPER_ROOT
from api_exploration_graph import (
    get_exploration_graph,
    generate_territory_root_sets,
    get_neighboring_territories,
)


class TestCaseType(Enum):
    RANDOM = "random"
    TERRITORIAL = "territorial"
    WORKERMAN = "workerman"


class TestResult(TypedDict):
    test_type: TestCaseType
    param: int
    duration: float
    roots: int
    workers: int
    dangers: int
    cost: int


class TestCaseConfig:
    """Configuration for test case generation."""

    def __init__(self, config: dict | None = None):
        self.config = config or get_config("config")
        self.exploration_data = None
        self.territory_root_sets = None

    def get_exploration_data(self) -> dict:
        """Cache and return exploration data."""
        if self.exploration_data is None:
            self.exploration_data = get_clean_exploration_data(self.config)
        return self.exploration_data

    def get_territory_root_sets(self) -> dict:
        """Cache and return territory root sets."""
        if self.territory_root_sets is None:
            self.territory_root_sets = generate_territory_root_sets(self.get_exploration_data())
        return self.territory_root_sets


def get_config(config_name: str) -> dict:
    """Retrieve configuration from data store."""
    return ds.get_config(config_name)


def _validate_worker_percent(worker_percent: int) -> None:
    """Validate worker percentage."""
    if not 0 <= worker_percent <= 100:
        raise ValueError("worker_percent must be between 0 and 100")


def _validate_cost(cost: int) -> None:
    """Validate cost parameter."""
    if cost % 5 != 0:
        raise ValueError("Cost must be a multiple of 5")
    if not 5 <= cost <= 550:
        raise ValueError("Cost must be between 5 and 550")


def _select_nodes(nodes: list[int], count: int, seed: int) -> list[int]:
    """Select a random sample of nodes."""
    if not nodes:
        logger.warning("No nodes available for selection")
        return []
    random.seed(seed)
    return random.sample(nodes, min(count, len(nodes)))


def _assign_worker_roots(
    workers: list[int], exploration_data: dict, territory_root_sets: dict, seed: int
) -> dict[int, int]:
    """Assign random roots to worker terminals."""
    random.seed(seed)
    src_dst = {}
    for worker in workers:
        worker_data = exploration_data[worker]
        territory_key = worker_data["territory_key"]
        roots = territory_root_sets[territory_key]
        src_dst[worker] = random.choice(roots)
    return src_dst


def _assign_danger_nodes(dangers: list[int]) -> dict[int, int]:
    """Assign danger nodes to SUPER_ROOT."""
    return {danger: SUPER_ROOT for danger in dangers}


def _count_metrics(src_dst: dict[int, int]) -> tuple[int, int, int]:
    """Count roots, workers, and dangers in terminal mappings."""
    root_count = len(set(src_dst.values()) - {SUPER_ROOT})
    worker_count = sum(1 for v in src_dst.values() if v != SUPER_ROOT)
    danger_count = len(src_dst) - worker_count
    return root_count, worker_count, danger_count


def _generate_random_terminals(
    worker_percent: int,
    include_danger: bool = False,
    max_danger: int | None = None,
    seed: int = 42,
    config: TestCaseConfig | None = None,
) -> tuple[dict[int, int], int, int, int]:
    """
    Generate random terminal-to-root mappings for testing.

    Args:
        worker_percent: Percentage of worker terminals (0-100).
        include_danger: Include danger terminals if True.
        max_danger: Maximum number of danger terminals.
        seed: Random seed for reproducibility.
        config: Test case configuration.

    Returns:
        Tuple of terminal-to-root mappings, root count, worker count, danger count.
    """
    _validate_worker_percent(worker_percent)
    config = config or TestCaseConfig()
    exploration_data = config.get_exploration_data()

    worker_nodes = [k for k, v in exploration_data.items() if v["is_workerman_plantzone"]]
    danger_nodes = [k for k, v in exploration_data.items() if v["node_type"] == 9]

    num_workers = int(len(worker_nodes) * worker_percent / 100)
    selected_workers = _select_nodes(worker_nodes, num_workers, seed)

    selected_dangers = []
    if include_danger:
        danger_count = (
            min(round(1 + num_workers / 12.5), len(danger_nodes))
            if max_danger is None
            else random.randint(1, min(max_danger, len(danger_nodes)))
        )
        selected_dangers = _select_nodes(danger_nodes, danger_count, seed)

    src_dst: dict[int, int] = {
        **_assign_worker_roots(selected_workers, exploration_data, config.get_territory_root_sets(), seed),
        **_assign_danger_nodes(selected_dangers),
    }
    root_count, worker_count, danger_count = _count_metrics(src_dst)

    return src_dst, root_count, worker_count, danger_count


def _generate_territorial_terminals(
    worker_percent: int,
    include_danger: bool = False,
    max_danger: int | None = None,
    seed: int = 42,
    config: TestCaseConfig | None = None,
) -> tuple[dict[int, int], int, int, int]:
    """
    Generate territorial terminals using the same worker_percent scaling as random_terminals.

    This ensures perfect comparability: same budget → same % → same expected worker count
    across random and territorial generators.

    Args:
        worker_percent: Percentage of total plantzones to use as workers (0–100).
        include_danger: Include danger terminals if True.
        max_danger: Maximum number of danger terminals.
        seed: Random seed.
        config: Test case configuration.

    Returns:
        terminals, root_count, worker_count, danger_count
    """
    _validate_worker_percent(worker_percent)
    random.seed(seed)
    config = config or TestCaseConfig()
    exploration_data = config.get_exploration_data()
    territory_root_sets = config.get_territory_root_sets()
    territory_neighbors = get_neighboring_territories(exploration_data)

    plantzone_nodes = [k for k, v in exploration_data.items() if v["is_workerman_plantzone"]]

    # Pick a random starting plantzone and grow the connected empire
    start_node = random.choice(plantzone_nodes)
    start_territory = exploration_data[start_node]["territory_key"]

    empire: set[int] = {start_territory}
    frontier: set[int] = {start_territory}

    while frontier:
        current = frontier.pop()
        for neighbor in territory_neighbors[current]:
            if neighbor not in empire:
                empire.add(neighbor)
                frontier.add(neighbor)

    # Candidates = all plantzones in the empire
    candidates = [k for k in plantzone_nodes if exploration_data[k]["territory_key"] in empire]

    # Apply the same percentage logic as random_terminals
    num_workers = int(len(candidates) * worker_percent / 100)
    selected_workers = random.sample(candidates, min(num_workers, len(candidates)))

    src_dst: dict[int, int] = {}
    for w in selected_workers:
        terr = exploration_data[w]["territory_key"]
        src_dst[w] = random.choice(territory_root_sets[terr])

    # Danger nodes — same as workerman/random
    selected_dangers = []
    if include_danger:
        danger_nodes = [k for k, v in exploration_data.items() if v["node_type"] == 9]
        danger_cnt = (
            max(round(len(src_dst) / 25), 1)
            if max_danger is None
            else random.randint(1, min(max_danger, len(danger_nodes)))
        )
        selected_dangers = _select_nodes(danger_nodes, danger_cnt, seed)
        src_dst.update(_assign_danger_nodes(selected_dangers))

    root_count, worker_count_out, danger_count = _count_metrics(src_dst)
    return src_dst, root_count, worker_count_out, danger_count


def _generate_workerman_terminals(
    cost: int = 5,
    include_danger: bool = False,
    max_danger: int | None = None,
    seed: int = 42,
    config: TestCaseConfig | None = None,
) -> tuple[dict[int, int], int, int, int]:
    """
    Generate terminal-to-root mappings from Empire Optimizer workerman output.

    Args:
        cost: Cost value (multiple of 5, 5-550).
        include_danger: Include danger terminals if True.
        max_danger: Maximum number of danger terminals.
        seed: Random seed for reproducibility.
        config: Test case configuration.

    Returns:
        Tuple of terminal-to-root mappings, root count, worker count, danger count.
    """
    _validate_cost(cost)
    random.seed(seed)
    config = config or TestCaseConfig()
    incident_path = Path(ds.path()) / "workerman"

    for file in incident_path.glob(f"{cost}_*"):
        break
    else:
        raise ValueError(f"No incident file found for cost {cost} => {incident_path}")

    logger.info(f"Using workerman test file: {file}")
    with open(file, "r") as f:
        incident = json.load(f)

    src_dst = {int(worker["job"]["pzk"]): int(worker["job"]["storage"]) for worker in incident["userWorkers"]}

    if include_danger:
        exploration_data = config.get_exploration_data()
        danger_nodes = [k for k, v in exploration_data.items() if v["node_type"] == 9]
        danger_count = (
            max(round(len(src_dst) / 25), 1)
            if max_danger is None
            else random.randint(1, min(max_danger, len(danger_nodes)))
        )
        selected_dangers = _select_nodes(danger_nodes, danger_count, seed)
        src_dst.update(_assign_danger_nodes(selected_dangers))

    root_count, worker_count, danger_count = _count_metrics(src_dst)

    return src_dst, root_count, worker_count, danger_count


def _get_solution_metrics(
    solution_graph: rx.PyDiGraph, objective_value: int, metrics: dict[str, str | int]
) -> int:
    """Calculate and log solution metrics."""
    metrics["nodes"] = solution_graph.num_nodes()
    metrics["edges"] = solution_graph.num_edges()
    num_components = len(rx.strongly_connected_components(solution_graph))
    metrics["components"] = num_components
    metrics["cost"] = objective_value
    logger.success(f"Test Case: {', '.join(f'{k}: {v}' for k, v in metrics.items())}")
    return num_components


def _run_test(
    optimization_fn: Callable[[rx.PyDiGraph, dict, dict], Mapping[str, Any]],
    config: dict,
    terminals: dict[int, int],
    metrics: dict[str, str | int],
) -> tuple[float, rx.PyDiGraph, int]:
    """Run a test case and return solve time, solution graph, and objective value."""
    exploration_graph = get_exploration_graph(config)
    assert isinstance(exploration_graph, rx.PyDiGraph)

    start_time = time.perf_counter()
    result = optimization_fn(exploration_graph, terminals, config)
    solve_time = time.perf_counter() - start_time

    solution_graph = result["solution_graph"]
    objective_value = result["objective_value"]
    assert isinstance(solution_graph, rx.PyDiGraph)
    assert isinstance(objective_value, int)

    metrics["time"] = f"{solve_time:.6f}s"
    num_components = _get_solution_metrics(solution_graph, objective_value, metrics)

    logger.info(f"Solution nodes: {[n['waypoint_key'] for n in solution_graph.nodes()]}")

    logger.info(f"Solve time: {solve_time:.6f}s, Components: {num_components}, Cost: {objective_value}")
    return solve_time, solution_graph, objective_value


def random_terminals(
    optimization_fn: Callable[[rx.PyDiGraph, dict, dict], Mapping[str, Any]],
    config: dict,
    worker_percent: int,
    include_danger: bool = False,
    max_danger: int | None = None,
    seed: int = 42,
) -> TestResult:
    """
    Run random terminals test.

    Args:
        optimization_fn: Function to optimize the graph.
        config: Configuration dictionary.
        worker_percent: Percentage of worker terminals.
        include_danger: Include danger terminals if True.
        max_danger: Maximum number of danger terminals.
        seed: Random seed for reproducibility.

    Returns:
        objective_value, solve_time, terminals
    """
    logger.info("\n==> Random terminals test...")
    terminals, root_count, worker_count, danger_count = _generate_random_terminals(
        worker_percent, include_danger, max_danger, seed
    )
    logger.info(f"{worker_percent=} {root_count=} {worker_count=} {danger_count=}")
    logger.info(f"{terminals=}\n")

    solve_time, solution_graph, objective_value = _run_test(
        optimization_fn,
        config,
        terminals,
        {"percent": worker_percent, "roots": root_count, "workers": worker_count, "dangers": danger_count},
    )

    result: TestResult = {
        "test_type": TestCaseType.RANDOM,
        "param": worker_percent,
        "duration": solve_time,
        "roots": root_count,
        "workers": worker_count,
        "dangers": danger_count,
        "cost": objective_value,
    }

    return result


def territorial_terminals(
    optimization_fn: Callable[[rx.PyDiGraph, dict, dict], Mapping[str, Any]],
    config: dict,
    worker_percent: int,
    include_danger: bool = False,
    max_danger: int | None = None,
    seed: int = 42,
) -> TestResult:
    """
    Run territorial terminals test — clustered, realistic, using same % scaling as random.

    Args:
        optimization_fn: Solver function.
        config: Configuration.
        worker_percent: Percentage of plantzones in the empire to use as workers.
        include_danger: Include danger nodes if True.
        max_danger: Max danger nodes.
        seed: Random seed.

    Returns:
        TestResult with cost, duration, metrics
    """
    logger.info("\n==> Territorial terminals test...")
    terminals, roots, workers, dangers = _generate_territorial_terminals(
        worker_percent, include_danger, max_danger, seed
    )
    logger.info(f"{worker_percent=} {roots=} {workers=} {dangers=}")
    logger.info(f"{terminals=}\n")

    duration, _, cost = _run_test(
        optimization_fn, config, terminals, {"workers": workers, "roots": roots, "dangers": dangers}
    )

    return {
        "test_type": TestCaseType.TERRITORIAL,
        "param": worker_percent,
        "duration": duration,
        "roots": roots,
        "workers": workers,
        "dangers": dangers,
        "cost": cost,
    }


def workerman_terminals(
    optimization_fn: Callable[[rx.PyDiGraph, dict, dict], Mapping[str, Any]],
    config: dict,
    budget: int,
    include_danger: bool = False,
    max_danger: int | None = None,
    seed: int = 42,
) -> TestResult:
    """
    Run workerman terminals test.

    Args:
        optimization_fn: Function to optimize the graph.
        config: Configuration dictionary.
        budget: Budget value for workerman test.
        include_danger: Include danger terminals if True.
        max_danger: Maximum number of danger terminals.
        seed: Random seed for reproducibility.

    Returns:
        Objective value.
        Solve time in seconds.
    """
    logger.info("\n==> Workerman test set...")
    terminals, root_count, worker_count, danger_count = _generate_workerman_terminals(
        budget, include_danger, max_danger, seed
    )
    logger.info(f"{budget=} {root_count=} {worker_count=} {danger_count=}")
    logger.info(f"{terminals=}\n")

    solve_time, solution_graph, objective_value = _run_test(
        optimization_fn,
        config,
        terminals,
        {"budget": budget, "roots": root_count, "workers": worker_count, "dangers": danger_count},
    )

    if config.get("solution", {}).get("save", False):
        solution_filename = (
            f"{config.get('type', 'unknown')}_budget_{budget}"
            f"_include_danger_{include_danger}_max_danger_{max_danger}_seed_{seed}.json"
        )
        save_graph(solution_graph, terminals, solution_filename)

    result: TestResult = {
        "test_type": TestCaseType.RANDOM,
        "param": budget,
        "duration": solve_time,
        "roots": root_count,
        "workers": worker_count,
        "dangers": danger_count,
        "cost": objective_value,
    }

    return result


def baselines(
    optimization_fn: Callable[[rx.PyDiGraph, dict, dict], Mapping[str, Any]],
    config: dict,
) -> bool:
    """
    Run baseline tests for simple cases.

    Args:
        optimization_fn: Function to optimize the graph.
        config: Configuration dictionary.

    Returns:
        True if all tests pass, False otherwise.
    """
    set_logger(config)
    test_cases = [
        ("1) 1 root to 1 terminal (short)", {480: 301}, 6),
        ("2) 1 root to 1 terminal (long - through other base-towns)", {1556: 1623}, 54),
        ("3) nearest root to 1 terminal (super root - test)", {480: SUPER_ROOT}, 4),
        ("4) 1 root to 2 terminals", {488: 301, 480: 301}, 8),
        ("5) 1 root to 2 terminals (leaf and parent terminals)", {488: 302, 347: 302}, 4),
        (
            "6) 1 root to 3 terminals (leaf, parent and neighbor to same root)",
            {488: 302, 480: 302, 455: 302},
            11,
        ),
        ("7) 2 roots to 2 terminals (no-overlapping route)", {488: 301, 480: 302}, 6),
        ("8) 2 roots to 2 terminals (overlapping route)", {488: 302, 480: 301}, 8),
        (
            "9) 3 roots to 3 terminals (small overlapping and non-overlapping long)",
            {488: 302, 480: 301, 1556: 1623},
            60,
        ),
        (
            "10) 2 roots to 3 terminals (small overlapping and nearest root) (super root - test)",
            {488: 302, 480: 301, 1135: SUPER_ROOT},
            11,
        ),
        (
            "11) 3 roots to 4 terminals (small overlapping, nearest root and long) (super root - test)",
            {488: 302, 480: 301, 1135: SUPER_ROOT, 1556: 1623},
            63,
        ),
        ("Mediah area super terminals", {1132: 99999}, 5),
        ("Mediah area super terminals", {1152: 99999}, 7),
        ("Mediah area super terminals", {1154: 99999}, 6),
        ("Mediah area super terminals", {1160: 99999}, 6),
        ("Mediah area super terminals", {1162: 99999}, 5),
        ("Mediah area super terminals", {155: 1, 1132: 99999}, 9),
        ("Mediah area super terminals", {155: 1, 1151: 99999}, 10),
        ("Mediah area super terminals", {155: 1, 1152: 99999}, 11),
        ("Mediah area super terminals", {155: 1, 1154: 99999}, 10),
        ("Mediah area super terminals", {155: 1, 1160: 99999}, 10),
        ("Mediah area super terminals", {155: 1, 1162: 99999}, 9),
        ("Mediah area super terminals", {160: 1, 1132: 99999}, 8),
        ("Mediah area super terminals", {155: 1, 1132: 99999, 1152: 99999}, 11),
        ("Mediah area super terminals", {155: 1, 1132: 99999, 1136: 99999, 1152: 99999}, 14),
        # passes if max removal attempts is >= 4_045 with singletons emitted for 4 frontier rings
        # which puts the timing results back into the 2.3s range for the full test suite meaning
        # larger workerman instances would be in the 10-20ms range native.
        ("bridge test (expect fail if removal attempts < 4_045)", {910: 601, 1683: 302}, 30),
        # passes if max removal attempts is 9_000 with singletons emitted for 4 frontier rings
        # which puts the timing results back into the 5s range for the full test suite with a
        # single ascending pass
        ("bridge test (expect fail if removal attempts < 8_500)", {910: 601, 1683: 302, 480: 302}, 32),
        # Requires a new type of bridge to be emitted that would have F1, F2, F1, F2, F2, F1, F1
        # note the disjoint F2 outer ring singleton and mutli
        ("bridge test (expect fail until new bridge type is emitted)", {1903: 301}, 14),
    ]

    logger.info(f"*** Solving {config.get('name', 'unknown')} ***")

    exploration_graph = get_exploration_graph(config)
    assert isinstance(exploration_graph, rx.PyDiGraph)

    start_time = time.perf_counter()
    all_pass = True
    for test_name, terminals, expected_value in test_cases:
        result = optimization_fn(exploration_graph, terminals, config)
        all_pass &= _validate_baselines(test_name, result, expected_value, config)

    logger.info(f"Total testing time: {(time.perf_counter() - start_time):.6f}s")
    return all_pass


def _validate_baselines(
    name: str,
    result: Mapping[str, Any],
    expected_objective_value: int,
    config: dict,
) -> bool:
    """
    Validate baseline test results.

    Args:
        name: Test case name.
        result: Optimization result.
        expected_objective_value: Expected objective value.
        config: Configuration dictionary.

    Returns:
        True if test passes, False otherwise.
    """
    solution_graph = result["solution_graph"]
    assert isinstance(solution_graph, rx.PyDiGraph)

    cost = sum(n["need_exploration_point"] for n in solution_graph.nodes())
    logger.info(f"Expected Cost: {expected_objective_value}, Actual: {cost}")
    logger.info(f"Connected components: {len(rx.strongly_connected_components(solution_graph))}")

    success = True
    if result["objective_value"] != expected_objective_value:
        logger.error(f"❌ Test: {name}: fail optimization")
        success = False
    elif cost != expected_objective_value:
        logger.warning(f"🟠 Test: {name}: fail extraction")
        success = False
    else:
        logger.success(f"✅ Test: {name}: success")

    logger.info("=" * 100)
    return success


if __name__ == "__main__":
    config = TestCaseConfig()

    # Workerman terminals
    for cost in range(5, 555, 5):
        for include_danger in [False, True]:
            src_dst, root_count, worker_count, danger_count = _generate_workerman_terminals(
                cost, include_danger, config=config
            )
            logger.info(
                f"Workerman Terminals - Cost: {cost}, Danger: {include_danger} - "
                f"Roots: {root_count}, Workers: {worker_count}, Dangers: {danger_count}, Terminals: {src_dst}"
            )

    # Random terminals (percent-based)
    for percent in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
        for include_danger in [False, True]:
            src_dst, root_count, worker_count, danger_count = _generate_random_terminals(
                percent, include_danger, config=config
            )
            logger.info(
                f"Random Terminals - Percent: {percent}, Danger: {include_danger} - "
                f"Roots: {root_count}, Workers: {worker_count}, Dangers: {danger_count}, Terminals: {src_dst}"
            )

    # Territorial terminals (worker count based)
    for worker_count in [1, 3, 5, 10, 20, 30, 50, 75, 100]:
        for include_danger in [False, True]:
            src_dst, root_count, wc_out, danger_count = _generate_territorial_terminals(
                worker_count, include_danger, config=config
            )
            logger.info(
                f"Territorial Terminals - Workers: {worker_count}, Danger: {include_danger} - "
                f"Roots: {root_count}, Workers: {wc_out}, Dangers: {danger_count}, Terminals: {src_dst}"
            )
            include_danger = True
            src_dst, root_count, wc_out, danger_count = _generate_territorial_terminals(
                worker_count, include_danger, config=config
            )
            logger.info(
                f"Territorial Terminals - Workers: {worker_count}, Danger: {include_danger} - "
                f"Roots: {root_count}, Workers: {wc_out}, Dangers: {danger_count}, Terminals: {src_dst}"
            )
