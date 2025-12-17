# orchestrator_types.py

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from api_exploration_data import SUPER_ROOT
from orchestrator_pairing_strategy import PairingStrategy


@dataclass
class Solution:
    duration: float
    cost: int
    num_nodes: int
    num_edges: int
    num_components: int
    waypoints: list[int]


OptimizationFn = Callable[[dict[int, int], dict], Solution]


@dataclass
class Terminals:
    terminals: dict[int, int]
    roots: int = 0
    workers: int = 0
    dangers: int = 0

    def __post_init__(self):
        values = list(self.terminals.values())
        self.roots = len(set(values) - {SUPER_ROOT})
        self.workers = sum(1 for v in values if v != SUPER_ROOT)
        self.dangers = len(self.terminals) - self.workers


@dataclass(frozen=True)
class Plan:
    optimization_fn: OptimizationFn
    config: dict
    budget: int  # used for optimized
    worker_percent: int  # required for candidate strategies
    seed: int
    include_danger: bool
    strategy: PairingStrategy
    allow_cache: bool


@dataclass
class Instance:
    plan: Plan
    terminals: Terminals
    solution: Solution | None = None

    def __post_init__(self):
        result = self.plan.optimization_fn(self.terminals.terminals, self.plan.config)
        self.solution = result
        logger.info(f"Terminals: {self.terminals.terminals}")
        logger.info(f"Solution: {result.waypoints}")

    @property
    def log_msg(self) -> str:
        assert self.solution
        if self.plan.strategy == PairingStrategy.optimized:
            param_str = f"budget: {self.plan.budget}, "
        else:
            param_str = f"percent: {self.plan.worker_percent}, "

        return (
            f"{self.plan.strategy}, "
            f"{param_str}"
            f"terminals={len(self.terminals.terminals)}, "
            f"roots={self.terminals.roots}, "
            f"workers={self.terminals.workers}, "
            f"dangers={self.terminals.dangers}, "
            f"result: nodes={self.solution.num_nodes}, "
            f"edges={self.solution.num_edges}, "
            f"components={self.solution.num_components}, "
            f"cost={self.solution.cost}, "
            f"duration={self.solution.duration:.6f}s"
        )
