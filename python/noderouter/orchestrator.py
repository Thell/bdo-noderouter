# orchestrator.py

"""
Testing utilities.

NOTE: Running this file directly outputs the generated test paramaters.
"""

from __future__ import annotations

import time

from loguru import logger

import api_data_store as ds
from api_common import memory
from orchestrator_pairing_strategy import PairingStrategy
from orchestrator_types import Plan, Instance, Solution
from orchestrator_terminal_pairs import generate_terminal_pairs


def execute_plan(plan: Plan) -> Instance:
    """Generate and solve terminal-to-root mappings based on a pairing strategy."""
    if plan.allow_cache:
        start_time = time.perf_counter()
        instance = _execute_plan_cached(plan)
        duration = time.perf_counter() - start_time
    else:
        instance = _execute_plan_uncached(plan)
        duration = None

    log_msg = instance.log_msg
    assert instance.solution
    if duration is not None and duration < instance.solution.duration:
        log_msg = log_msg + " ♻️"
    logger.success(log_msg)

    return instance


@memory.cache
def _execute_plan_cached(plan: Plan) -> Instance:
    # Springboard for cached instances
    return _execute_plan_uncached(plan)


def _execute_plan_uncached(plan: Plan) -> Instance:
    return Instance(plan, generate_terminal_pairs(plan))


if __name__ == "__main__":
    from api_common import set_logger
    from orchestrator_terminal_pairs import generate_terminal_pairs

    config = ds.get_config("config")
    config["name"] = "test_parms"
    set_logger(config)

    def make_plan(
        budget: int,
        include_danger: bool,
        pairing_type: PairingStrategy,
        percent: int,
    ) -> Plan:
        return Plan(
            lambda _a, _b: Solution(0.0, 0, 0, 0, 0, []),
            config,
            budget,
            percent,
            0,
            include_danger,
            pairing_type,
            False,
        )

    def dummy_fn(_a: dict[int, int], _b: dict) -> Solution:
        return Solution(0.0, 0, 0, 0, 0, [])

    for budget in range(5, 555, 5):
        for include_danger in [False, True]:
            plan = make_plan(budget, include_danger, PairingStrategy.optimized, 0)
            specs = generate_terminal_pairs(plan)
            logger.info(
                f"Workerman Terminals - Budget: {budget}, Danger: {include_danger} - "
                f"Roots: {specs.roots}, Workers: {specs.workers}, Dangers: {specs.dangers}, Terminals: {specs.terminals}"
            )

    for pairing_type in PairingStrategy:
        if pairing_type == PairingStrategy.optimized:
            continue
        for percent in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
            for include_danger in [False, True]:
                plan = make_plan(0, include_danger, pairing_type, percent)
                specs = generate_terminal_pairs(plan)
                logger.info(
                    f"strategy: {pairing_type.value} - Percent: {percent}, Danger: {include_danger} - "
                    f"Roots: {specs.roots}, Workers: {specs.workers}, Dangers: {specs.dangers}, Terminals: {specs.terminals}"
                )
