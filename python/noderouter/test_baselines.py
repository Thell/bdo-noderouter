# test_baselines.py

"""Baseline unit tests for simple cases."""

import time

from loguru import logger
import rustworkx as rx

from api_common import set_logger
from api_exploration_data import get_exploration_data, SUPER_ROOT
from orchestrator_types import Solution, OptimizationFn


def _validate_baselines(
    name: str,
    solution: Solution,
    expected_objective_value: int,
    _config: dict,
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
    objective = solution.cost

    logger.info(f"Expected Cost: {expected_objective_value}, Actual: {objective}")
    logger.info(f"Connected components: {solution.num_components}")

    success = True
    if objective != expected_objective_value:
        logger.error(f"❌ Test: {name}: fail optimization")
        success = False
    elif objective != expected_objective_value:
        logger.warning(f"🟠 Test: {name}: fail extraction")
        success = False
    else:
        logger.success(f"✅ Test: {name}: success")

    logger.info("=" * 100)
    return success


def baselines(
    optimization_fn: OptimizationFn,
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

    exploration_data = get_exploration_data()
    exploration_graph = exploration_data.graph
    assert isinstance(exploration_graph, rx.PyDiGraph)

    start_time = time.perf_counter()
    all_pass = True
    for test_name, terminals, expected_value in test_cases:
        result = optimization_fn(terminals, config)
        all_pass &= _validate_baselines(test_name, result, expected_value, config)

    logger.info(f"Total testing time: {(time.perf_counter() - start_time):.6f}s")
    return all_pass
