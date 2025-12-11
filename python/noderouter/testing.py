# testing.py

"""
Testing utilities.

NOTE: Running this file directly outputs the generated test paramaters.
"""

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

import api_data_store as ds
from api_common import NodeType, get_clean_exploration_data, ResultDict, SUPER_ROOT
from api_exploration_graph import (
    get_exploration_graph,
)
from testing_root_pairing import PairingStrategy

# joblib caching for mip results during fuzz testing.
memory = Memory(location=".cache", verbose=0)


# MARK: Test Instance Generation
class TestCaseType(StrEnum):
    STRATEGY = "strategy"
    WORKERMAN = "workerman"


@dataclass
class TerminalSpecs:
    terminals: dict[int, int]
    roots: int = 0
    workers: int = 0
    dangers: int = 0

    def __post_init__(self):
        # compute metrics from src_dst mapping
        values = list(self.terminals.values())
        self.roots = len(set(values) - {SUPER_ROOT})
        self.workers = sum(1 for v in values if v != SUPER_ROOT)
        self.dangers = len(self.terminals) - self.workers


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
    strategy: PairingStrategy | None
    specs: TerminalSpecs

    result: TestResult | None = None

    def run(self) -> None:
        exploration_graph = get_exploration_graph(self.config)
        assert isinstance(exploration_graph, PyDiGraph)

        result: ResultDict = self.optimization_fn(exploration_graph, self.specs.terminals, self.config)

        solution_graph = result["solution_graph"]
        solution = result["solution"]
        objective = result["objective"]
        duration = result["duration"]

        num_components = len(rx.strongly_connected_components(solution_graph))

        self.result = TestResult(
            duration=duration,
            cost=objective,
            num_nodes=solution_graph.num_nodes(),
            num_edges=solution_graph.num_edges(),
            num_components=num_components,
            solution=solution,
        )

        logger.info(f"Terminals: {self.specs.terminals}")
        logger.info(f"Solution: {solution}")

        if self.test_type == TestCaseType.WORKERMAN:
            param_string = f"budget: {self.budget}"
        else:
            assert self.strategy
            param_string = f"{self.strategy.value}, percent: {self.percent}"

        logger.success(
            f"{self.test_type.value}, {param_string}, params: terminals={len(self.specs.terminals)}, "
            f"roots={self.specs.roots}, workers={self.specs.workers}, dangers={self.specs.dangers}, "
            f"result: nodes={solution_graph.num_nodes()}, edges={solution_graph.num_edges()}, "
            f"components={num_components}, cost={objective}, duration={duration:.6f}s"
        )


@dataclass
class TestContext:
    config: dict | None = None

    def __post_init__(self):
        if self.config is None:
            self.config = ds.get_config("config")

        # populate once at instantiation
        self.exploration_data: dict[int, dict] = get_clean_exploration_data(self.config)
        self.plantzones: list[int] = [
            k for k, v in self.exploration_data.items() if v.get("is_workerman_plantzone")
        ]
        self.dangers: list[int] = [
            k for k, v in self.exploration_data.items() if v.get("node_type") == NodeType.dangerous
        ]
        self.max_plantzone_count: int = len(self.plantzones)
        self.max_danger_count: int = len(self.dangers)

    def select_terminals(self, worker_percent: int, random: random.Random) -> list[int]:
        num_workers = int(self.max_plantzone_count * worker_percent / 100)
        return random.sample(self.plantzones, min(num_workers, self.max_plantzone_count))

    def select_dangers(self, selected_terminal_count: int, random: random.Random) -> list[int]:
        danger_count = max(round(selected_terminal_count / 25), 1)
        return random.sample(self.dangers, danger_count)


class TestGenerator:
    def __init__(self, test_context: TestContext, seed: int):
        self.context = test_context
        self.random = random.Random(seed)

    def _generate_terminal_pairs(
        self,
        worker_percent: int,
        include_danger: bool,
        strategy: PairingStrategy = PairingStrategy.cheapest_town_in_territory,
    ) -> TerminalSpecs:
        """Generate terminal-to-root mappings based on a pairing strategy."""
        exploration_data = self.context.exploration_data

        # terminal assignment
        selected_terminals = self.context.select_terminals(worker_percent, self.random)
        src_dst: dict[int, int] = {}
        for t in selected_terminals:
            terminal = exploration_data[t]
            candidates = strategy.candidates(terminal)
            src_dst[t] = self.random.choice(candidates)

        # danger terminals
        if include_danger:
            dangers = self.context.select_dangers(len(src_dst), self.random)
            src_dst.update({d: SUPER_ROOT for d in dangers})

        return TerminalSpecs(src_dst)

    def _generate_workerman_terminals(self, cost: int, include_danger: bool) -> TerminalSpecs:
        """Generate terminal-to-root mappings based on workerman data."""
        # Optimal solutions only exist for certain budgets
        if cost % 5 != 0 or not 5 <= cost <= 550:
            raise ValueError("Invalid workerman budget!")

        # load workerman optimal terminals
        path = Path(ds.path()) / "workerman"
        files = list(path.glob(f"{cost}_*"))
        if not files:
            raise ValueError(f"No workerman file found for budget {budget}")
        with open(files[0]) as f:
            incident = json.load(f)

        # danger terminals
        src_dst = {int(w["job"]["pzk"]): int(w["job"]["storage"]) for w in incident["userWorkers"]}
        if include_danger:
            dangers = self.context.select_dangers(len(src_dst), self.random)
            src_dst.update({d: SUPER_ROOT for d in dangers})

        return TerminalSpecs(src_dst)


@memory.cache
def _get_test_context(config: dict) -> TestContext:
    return TestContext(config)


_default_generator: TestGenerator | None = None


def _get_default_generator(config: dict, seed: int) -> TestGenerator:
    global _default_generator
    context = _get_test_context(config)
    if _default_generator is None or _default_generator.context.config != config:
        _default_generator = TestGenerator(context, seed)
    return _default_generator


# --- Public module functions ---


def generate_terminals_mip(
    optimization_fn,
    config: dict,
    seed: int,
    worker_percent: int,
    include_danger: bool = False,
    strategy: PairingStrategy = PairingStrategy.nearest_town,
) -> TestInstance:
    """Launcher for fuzzer mip incidents via springboard for caching."""
    start_time = time.perf_counter()
    incident = _generate_terminals_cached(
        optimization_fn, config, seed, worker_percent, include_danger, strategy
    )
    duration = time.perf_counter() - start_time

    assert incident.result is not None
    if incident.result.duration > duration:
        # Cache hit - output log success msg
        assert incident.strategy
        param_string = f"{incident.strategy.value}, percent: {incident.percent}"
        logger.success(
            f"{incident.test_type.value}, {param_string}, params: terminals={len(incident.specs.terminals)}, "
            f"roots={incident.specs.roots}, workers={incident.specs.workers}, dangers={incident.specs.dangers}, "
            f"result: nodes={incident.result.num_nodes}, edges={incident.result.num_edges}, "
            f"components={incident.result.num_components}, cost={incident.result.cost}, duration={incident.result.duration:.6f}s ♻️"
        )

    return incident


@memory.cache
def _generate_terminals_cached(
    optimization_fn,
    config: dict,
    seed: int,
    worker_percent: int,
    include_danger: bool = False,
    strategy: PairingStrategy = PairingStrategy.nearest_town,
) -> TestInstance:
    """Springboard to the test case via cached mip solver."""
    return generate_terminals(optimization_fn, config, seed, worker_percent, include_danger, strategy)


def generate_terminals(
    optimization_fn,
    config: dict,
    worker_percent: int,
    seed: int,
    include_danger: bool = False,
    strategy: PairingStrategy = PairingStrategy.random_town,
) -> TestInstance:
    """
    Generate and run a test instance with terminal-to-root mappings based on a pairing strategy.

    Args:
        optimization_fn: Optimization function to run.
        config: Configuration dictionary.
        seed: Random seed for reproducibility.
        worker_percent: Percentage of worker terminals (0–100).
        include_danger: Whether to include danger terminals.
        strategy: Passed through to `_generate_terminal_pairs`.

    Returns:
        A TestInstance with results populated.
    """
    gen = _get_default_generator(config, seed)
    gen.random = random.Random(seed)

    specs = gen._generate_terminal_pairs(worker_percent, include_danger, strategy)

    instance = TestInstance(
        test_type=TestCaseType.STRATEGY,
        optimization_fn=optimization_fn,
        config=config,
        budget=None,
        percent=worker_percent,
        strategy=strategy,
        specs=specs,
        result=None,
    )
    instance.run()
    return instance


def workerman_terminals_mip(
    optimization_fn,
    config: dict,
    budget: int,
    seed: int,
    include_danger: bool = False,
) -> TestInstance:
    """Launcher for fuzzer mip incidents via springboard for caching."""
    start_time = time.perf_counter()
    incident = _workerman_terminals_cached(optimization_fn, config, budget, seed, include_danger)
    duration = time.perf_counter() - start_time

    assert incident.result is not None
    if incident.result.duration > duration:
        # Cache hit - output log success msg based on original results
        assert incident.test_type == TestCaseType.WORKERMAN
        param_string = f"budget: {incident.budget}"
        logger.success(
            f"{incident.test_type.value}, {param_string}, params: terminals={len(incident.specs.terminals)}, "
            f"roots={incident.specs.roots}, workers={incident.specs.workers}, dangers={incident.specs.dangers}, "
            f"result: nodes={incident.result.num_nodes}, edges={incident.result.num_edges}, "
            f"components={incident.result.num_components}, cost={incident.result.cost}, duration={incident.result.duration:.6f}s ♻️"
        )

    return incident


@memory.cache
def _workerman_terminals_cached(
    optimization_fn,
    config: dict,
    budget: int,
    seed: int,
    include_danger: bool = False,
) -> TestInstance:
    """Springboard to the workerman terminals test case via cached mip solver."""
    return workerman_terminals(optimization_fn, config, budget, seed, include_danger)


def workerman_terminals(
    optimization_fn, config: dict, budget: int, seed: int, include_danger=False
) -> TestInstance:
    """Generate and run a test instance with terminal-to-root mappings based on pre-optimized budget solution."""
    gen = _get_default_generator(config, seed)
    gen.random = random.Random(seed)
    specs = gen._generate_workerman_terminals(budget, include_danger)
    instance = TestInstance(
        test_type=TestCaseType.WORKERMAN,
        optimization_fn=optimization_fn,
        config=config,
        budget=budget,
        percent=round(specs.workers / gen.context.max_plantzone_count * 100),
        strategy=None,
        specs=specs,
        result=None,
    )
    instance.run()
    return instance


if __name__ == "__main__":
    from api_common import set_logger

    config = ds.get_config("config")
    config["name"] = "test_parms"
    set_logger(config)

    config = TestContext()
    generator = TestGenerator(config, seed=42)

    # Workerman terminals
    for budget in range(5, 555, 5):
        for include_danger in [False, True]:
            specs = generator._generate_workerman_terminals(budget, include_danger)
            logger.info(
                f"Workerman Terminals - Budget: {budget}, Danger: {include_danger} - "
                f"Roots: {specs.roots}, Workers: {specs.workers}, Dangers: {specs.dangers}, Terminals: {specs.terminals}"
            )

    # strategy-level terminals (percent-based)
    for pairing_type in PairingStrategy:
        for percent in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
            for include_danger in [False, True]:
                specs = generator._generate_terminal_pairs(percent, include_danger, strategy=pairing_type)
                logger.info(
                    f"strategy: {pairing_type.value} - Percent: {percent}, Danger: {include_danger} - "
                    f"Roots: {specs.roots}, Workers: {specs.workers}, Dangers: {specs.dangers}, Terminals: {specs.terminals}"
                )
