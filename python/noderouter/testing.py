# # testing.py

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import json
import random
import time

from joblib import Memory

import rustworkx as rx
from rustworkx import PyDiGraph
from loguru import logger

import data_store as ds
from api_common import NodeType, get_clean_exploration_data, ResultDict, SUPER_ROOT
from api_exploration_graph import (
    RootPairingType,
    get_exploration_graph,
)

memory = Memory(location=".cache", verbose=0)


class TestCaseType(StrEnum):
    STRICTNESS_LEVEL = "strictness_level"
    WORKERMAN = "workerman"


@dataclass
class TestResult:
    duration: float
    cost: int
    num_nodes: int
    num_edges: int
    num_components: int
    solution: list[int]


@dataclass
class TestInstance:
    test_type: TestCaseType
    optimization_fn: Callable[[PyDiGraph, dict[int, int], dict], ResultDict]
    config: dict

    budget: int | None
    percent: int
    strictness: RootPairingType | None

    roots: int
    workers: int
    dangers: int
    terminals: dict[int, int]

    result: TestResult | None = None

    def run(self) -> None:
        exploration_graph = get_exploration_graph(self.config)
        assert isinstance(exploration_graph, PyDiGraph)

        result: ResultDict = self.optimization_fn(exploration_graph, self.terminals, self.config)
        solution_graph = result["solution_graph"]
        objective = result["objective_value"]
        duration = result["duration"]

        solution = [n["waypoint_key"] for n in solution_graph.nodes()]
        num_components = len(rx.strongly_connected_components(solution_graph))

        self.result = TestResult(
            duration=duration,
            cost=objective,
            num_nodes=solution_graph.num_nodes(),
            num_edges=solution_graph.num_edges(),
            num_components=num_components,
            solution=solution,
        )

        logger.info(f"Terminals: {self.terminals}")
        logger.info(f"Solution: {solution}")

        if self.test_type == TestCaseType.WORKERMAN:
            param_string = f"budget: {self.budget}"
        else:
            assert self.strictness
            param_string = f"strictness: {self.strictness.value}, percent: {self.percent}"

        logger.success(
            f"{self.test_type.value}, {param_string}, params: terminals={len(self.terminals)}, "
            f"roots={self.roots}, workers={self.workers}, dangers={self.dangers}, "
            f"result: nodes={solution_graph.num_nodes()}, edges={solution_graph.num_edges()}, "
            f"components={num_components}, cost={objective}, duration={duration:.6f}s"
        )


@dataclass
class TestCase:
    config: dict | None = None

    def __post_init__(self):
        if self.config is None:
            self.config = ds.get_config("config")
        self._exploration_data: dict[int, dict] | None = None
        self._territory_root_sets: dict[int, list[int]] | None = None

    @property
    def exploration_data(self) -> dict[int, dict]:
        if self._exploration_data is None:
            assert self.config
            self._exploration_data = get_clean_exploration_data(self.config)
        return self._exploration_data

    # @property
    # def territory_root_sets(self) -> dict[int, list[int]]:
    #     if self._territory_root_sets is None:
    #         self._territory_root_sets = generate_territory_root_sets(self.exploration_data)
    #     return self._territory_root_sets


class TestGenerator:
    def __init__(self, test_case: TestCase, seed: int = 42):
        self.test_case = test_case
        self.random = random.Random(seed)

    def _count_metrics(self, src_dst: dict[int, int]) -> tuple[int, int, int]:
        root_count = len(set(src_dst.values()) - {SUPER_ROOT})
        worker_count = sum(1 for v in src_dst.values() if v != SUPER_ROOT)
        danger_count = len(src_dst) - worker_count
        return root_count, worker_count, danger_count

    def max_plantzone_count(self) -> int:
        return sum(1 for v in self.test_case.exploration_data.values() if v["is_workerman_plantzone"])

    # --- Terminal generation internal methods ---
    def _generate_random_terminals(
        self,
        worker_percent: int,
        include_danger: bool,
        max_danger: int | None,
        pairing_type: RootPairingType = RootPairingType.random_town,
    ) -> tuple[dict[int, int], tuple[int, int, int]]:
        ed = self.test_case.exploration_data

        # collect eligible worker terminals
        plantzones = [k for k, v in ed.items() if v.get("is_workerman_plantzone")]
        if not plantzones:
            raise ValueError("No plantzone nodes available")

        num_workers = int(len(plantzones) * worker_percent / 100)
        selected_workers = self.random.sample(plantzones, min(num_workers, len(plantzones)))

        # terminal‑centric assignment
        src_dst: dict[int, int] = {}
        for w in selected_workers:
            terminal = ed[w]
            candidates = pairing_type.candidates(terminal)
            if not candidates:
                raise ValueError(f"No candidate roots found for terminal {w} under {pairing_type}")
            root = self.random.choice(candidates)
            src_dst[w] = root

        # danger terminals
        if include_danger:
            dangers = [k for k, v in ed.items() if v.get("node_type") == NodeType.dangerous]
            if not dangers:
                raise ValueError("No danger nodes available")
            danger_count = (
                min(round(1 + num_workers / 12.5), len(dangers))
                if max_danger is None
                else self.random.randint(1, min(max_danger, len(dangers)))
            )
            src_dst.update({d: SUPER_ROOT for d in self.random.sample(dangers, danger_count)})

        return src_dst, self._count_metrics(src_dst)

    def _generate_territorial_terminals(
        self,
        worker_percent: int,
        include_danger: bool,
        max_danger: int | None,
        pairing_type: RootPairingType = RootPairingType.cheapest_town_in_territory,
    ) -> tuple[dict[int, int], tuple[int, int, int]]:
        """Generate terminal-to-root mappings based on a pairing strategy.

        Args:
            worker_percent: Percentage of worker terminals (0-100).
            include_danger: Include danger terminals if True.
            max_danger: Maximum number of danger terminals.
            pairing_type: Strategy for computing candidate roots.

        Returns:
            Tuple of terminal-to-root mapping and metrics.
        """
        ed = self.test_case.exploration_data

        # collect eligible worker terminals
        plantzones = [k for k, v in ed.items() if v.get("is_workerman_plantzone")]
        if not plantzones:
            raise ValueError("No plantzone nodes available")

        num_workers = int(len(plantzones) * worker_percent / 100)
        selected_workers = self.random.sample(plantzones, min(num_workers, len(plantzones)))

        # terminal‑centric assignment
        src_dst: dict[int, int] = {}
        for p in selected_workers:
            terminal = ed[p]
            candidates = pairing_type.candidates(terminal)
            if not candidates:
                raise ValueError(f"No candidate roots found for terminal {p} under {pairing_type}")
            root = self.random.choice(candidates)
            src_dst[p] = root

        # danger terminals
        if include_danger:
            dangers = [k for k, v in ed.items() if v.get("node_type") == NodeType.dangerous]
            if not dangers:
                raise ValueError("No danger nodes available")
            danger_count = (
                max(round(len(src_dst) / 25), 1)
                if max_danger is None
                else self.random.randint(1, min(max_danger, len(dangers)))
            )
            src_dst.update({d: SUPER_ROOT for d in self.random.sample(dangers, danger_count)})

        return src_dst, self._count_metrics(src_dst)

    def _generate_workerman_terminals(
        self, cost: int, include_danger: bool, max_danger: int | None
    ) -> tuple[dict[int, int], tuple[int, int, int]]:
        if cost % 5 != 0 or not 5 <= cost <= 550:
            raise ValueError("Invalid workerman cost")
        path = Path(ds.path()) / "workerman"
        files = list(path.glob(f"{cost}_*"))
        if not files:
            raise ValueError(f"No workerman file found for cost {cost}")
        with open(files[0]) as f:
            incident = json.load(f)
        src_dst = {int(w["job"]["pzk"]): int(w["job"]["storage"]) for w in incident["userWorkers"]}
        if include_danger:
            dangers = [k for k, v in self.test_case.exploration_data.items() if v["node_type"] == 9]
            danger_count = (
                max(round(len(src_dst) / 25), 1)
                if max_danger is None
                else self.random.randint(1, min(max_danger, len(dangers)))
            )
            src_dst.update({d: SUPER_ROOT for d in self.random.sample(dangers, danger_count)})
        return src_dst, self._count_metrics(src_dst)


# --- Public module functions ---

_default_generator: TestGenerator | None = None


def _get_default_generator(config: dict) -> TestGenerator:
    global _default_generator
    if _default_generator is None:
        _default_generator = TestGenerator(TestCase(config))
    return _default_generator


def generate_terminals_mip(
    optimization_fn,
    config: dict,
    worker_percent: int,
    include_danger: bool = False,
    max_danger: int | None = None,
    seed: int = 42,
    strictness: RootPairingType = RootPairingType.nearest_town,
) -> TestInstance:
    start_time = time.perf_counter()
    incident = generate_terminals_cached(
        optimization_fn, config, worker_percent, include_danger, max_danger, seed, strictness
    )
    duration = time.perf_counter() - start_time

    assert incident.result is not None
    if incident.result.duration > duration:
        # Cache hit - output log success msg
        if incident.test_type == TestCaseType.WORKERMAN:
            param_string = f"budget: {incident.budget}"
        else:
            assert incident.strictness
            param_string = f"strictness: {incident.strictness.value}, percent: {incident.percent}"

        logger.success(
            f"{incident.test_type.value}, {param_string}, params: terminals={len(incident.terminals)}, "
            f"roots={incident.roots}, workers={incident.workers}, dangers={incident.dangers}, "
            f"result: nodes={incident.result.num_nodes}, edges={incident.result.num_edges}, "
            f"components={incident.result.num_components}, cost={incident.result.cost}, duration={incident.result.duration:.6f}s ♻️"
        )

    return incident


@memory.cache
def generate_terminals_cached(
    optimization_fn,
    config: dict,
    worker_percent: int,
    include_danger: bool = False,
    max_danger: int | None = None,
    seed: int = 42,
    strictness: RootPairingType = RootPairingType.nearest_town,
) -> TestInstance:
    return territorial_terminals(
        optimization_fn, config, worker_percent, include_danger, max_danger, seed, strictness
    )


def generate_terminals(
    optimization_fn,
    config: dict,
    worker_percent: int,
    include_danger: bool = False,
    max_danger: int | None = None,
    seed: int = 42,
    strictness: RootPairingType = RootPairingType.nearest_town,
) -> TestInstance:
    """
    Generate a test instance with terminal-to-root mappings.

    Args:
        optimization_fn: Optimization function to run.
        config: Configuration dictionary.
        worker_percent: Percentage of worker terminals (0–100).
        include_danger: Whether to include danger terminals.
        max_danger: Maximum number of danger terminals.
        seed: Random seed for reproducibility.
        strictness: Passed through to `generate_territory_root_sets` to control
            locality assumptions. See `api_exploration_graph.py` for valid
            values and their definitions.

    Returns:
        A TestInstance with results populated.
    """
    gen = _get_default_generator(config)
    gen.random = random.Random(seed)

    terminals, (roots, workers, dangers) = gen._generate_territorial_terminals(
        worker_percent, include_danger, max_danger, strictness
    )

    instance = TestInstance(
        test_type=TestCaseType.STRICTNESS_LEVEL,
        optimization_fn=optimization_fn,
        config=config,
        budget=None,
        percent=worker_percent,
        strictness=strictness,
        roots=roots,
        workers=workers,
        dangers=dangers,
        terminals=terminals,
    )
    instance.run()
    return instance


def random_terminals(
    optimization_fn, config: dict, worker_percent: int, include_danger=False, max_danger=None, seed=42
) -> TestInstance:
    return generate_terminals(
        optimization_fn,
        config,
        worker_percent,
        include_danger,
        max_danger,
        seed,
        strictness=RootPairingType.random_any_root,
    )


def territorial_terminals(
    optimization_fn,
    config: dict,
    worker_percent: int,
    include_danger=False,
    max_danger=None,
    seed=42,
    strictness=RootPairingType.nearest_town,
) -> TestInstance:
    return generate_terminals(
        optimization_fn, config, worker_percent, include_danger, max_danger, seed, strictness
    )


def workerman_terminals_mip(
    optimization_fn,
    config: dict,
    budget: int,
    include_danger: bool = False,
    max_danger: int | None = None,
    seed: int = 42,
) -> TestInstance:
    start_time = time.perf_counter()
    incident = workerman_terminals_cached(optimization_fn, config, budget, include_danger, max_danger, seed)
    duration = time.perf_counter() - start_time

    assert incident.result is not None
    if incident.result.duration > duration:
        # Cache hit - output log success msg
        if incident.test_type == TestCaseType.WORKERMAN:
            param_string = f"budget: {incident.budget}"
        else:
            assert incident.strictness
            param_string = f"strictness: {incident.strictness.value}, percent: {incident.percent}"

        logger.success(
            f"{incident.test_type.value}, {param_string}, params: terminals={len(incident.terminals)}, "
            f"roots={incident.roots}, workers={incident.workers}, dangers={incident.dangers}, "
            f"result: nodes={incident.result.num_nodes}, edges={incident.result.num_edges}, "
            f"components={incident.result.num_components}, cost={incident.result.cost}, duration={incident.result.duration:.6f}s ♻️"
        )

    return incident


@memory.cache
def workerman_terminals_cached(
    optimization_fn,
    config: dict,
    budget: int,
    include_danger: bool = False,
    max_danger: int | None = None,
    seed: int = 42,
) -> TestInstance:
    return workerman_terminals(optimization_fn, config, budget, include_danger, max_danger, seed)


def workerman_terminals(
    optimization_fn, config: dict, budget: int, include_danger=False, max_danger=None, seed=42
) -> TestInstance:
    gen = _get_default_generator(config)
    gen.random = random.Random(seed)
    terminals, (roots, workers, dangers) = gen._generate_workerman_terminals(
        budget, include_danger, max_danger
    )
    instance = TestInstance(
        test_type=TestCaseType.WORKERMAN,
        optimization_fn=optimization_fn,
        config=config,
        budget=budget,
        percent=round(workers / gen.max_plantzone_count() * 100),
        strictness=None,
        roots=roots,
        workers=workers,
        dangers=dangers,
        terminals=terminals,
    )
    instance.run()
    return instance


if __name__ == "__main__":
    config = TestCase()
    generator = TestGenerator(config)

    # Workerman terminals
    # for budget in range(5, 555, 5):
    for budget in [5]:
        for include_danger in [False, True]:
            terminals, (roots, workers, dangers) = generator._generate_workerman_terminals(
                budget, include_danger, max_danger=None
            )
            logger.info(
                f"Workerman Terminals - Budget: {budget}, Danger: {include_danger} - "
                f"Roots: {roots}, Workers: {workers}, Dangers: {dangers}, Terminals: {terminals}"
            )

    # Strictness-level terminals (percent-based)
    for pairing_type in RootPairingType:
        # for percent in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
        for percent in [5]:
            for include_danger in [False, True]:
                terminals, (roots, workers, dangers) = generator._generate_territorial_terminals(
                    percent, include_danger, max_danger=None, pairing_type=pairing_type
                )
                logger.info(
                    f"Strictness: {pairing_type.value} - Percent: {percent}, Danger: {include_danger} - "
                    f"Roots: {roots}, Workers: {workers}, Dangers: {dangers}, Terminals: {terminals}"
                )
