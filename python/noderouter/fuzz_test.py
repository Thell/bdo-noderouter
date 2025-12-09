# fuzz_test.py

from __future__ import annotations
import time
from collections import defaultdict
from functools import partial
from statistics import mean, stdev
import signal
import sys
from dataclasses import dataclass

from loguru import logger
from tabulate import tabulate

import data_store as ds
import testing as test
from api_common import set_logger
from mip_baseline_rust import optimize_with_terminals as mip_optimize
from node_router import optimize_with_terminals as nr_optimize

_shutdown_requested = False


@dataclass
class FuzzInstanceMetrics:
    seed: int
    test_type: str
    budget: int
    percent: int
    strictness: test.RootPairingType | None
    include_danger: bool
    roots: int
    workers: int
    dangers: int
    mip_cost: float
    mip_duration: float
    nr_cost: float
    nr_duration: float

    @property
    def ratio(self) -> float:
        return self.nr_cost / self.mip_cost

    @property
    def speedup(self) -> float:
        return self.mip_duration / self.nr_duration if self.nr_duration > 0 else float("inf")


def _signal_handler(signum, frame, containers):
    global _shutdown_requested
    if _shutdown_requested:
        print("\nHard exit — terminating immediately.", file=sys.stderr)
        sys.exit(1)
    _shutdown_requested = True
    print("\nShutdown requested — finishing current test and reporting results...", file=sys.stderr)


def install_graceful_shutdown(containers):
    handler = partial(_signal_handler, containers=containers)
    signal.signal(signal.SIGINT, handler)


def run_single_config(
    samples: int, budget: int, test_type: str, include_danger: bool
) -> list[FuzzInstanceMetrics]:
    """
    Run a fuzz test for a single configuration.

    Args:
        samples: Number of instances to generate.
        budget: Workerman budget or basis for percent coverage.
        test_type: Either "workerman" or "strictness_level".
        include_danger: Whether to include danger nodes.

    Returns:
        List of FuzzInstanceMetrics for all instances in this config.
    """
    config = ds.get_config("config")
    config["name"] = "fuzz_test"
    set_logger(config)

    base_seed = 10_000
    all_metrics: list[FuzzInstanceMetrics] = []

    # Build the job list: one entry for workerman, or one per strictness level
    if test_type == "workerman":
        jobs: list[tuple[test.RootPairingType | None, int, test.TestCaseType]] = [
            (None, budget, test.TestCaseType.WORKERMAN)
        ]
    elif test_type == "strictness_level":
        percent = round(budget / 550 * 100)
        jobs = [(s, percent, test.TestCaseType.STRICTNESS_LEVEL) for s in test.RootPairingType]
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    for strictness, value, case_type in jobs:
        if _shutdown_requested:
            break

        if case_type == test.TestCaseType.WORKERMAN:
            desc = f"workerman budget={budget:3} | danger={'yes' if include_danger else 'no'}"
        else:
            assert strictness is not None
            desc = f"strictness={strictness} {value:3}% workers (budget {budget:3}) "
            desc += f"| danger={'yes' if include_danger else 'no'}"

        print(f"\nStarting: {desc} | samples={samples}")

        case_metrics: list[FuzzInstanceMetrics] = []

        for i in range(samples):
            if _shutdown_requested:
                break
            seed = base_seed + i

            if case_type == test.TestCaseType.WORKERMAN:
                mip_instance = test.workerman_terminals_mip(
                    mip_optimize, config, budget, include_danger, None, seed
                )
                nr_instance = test.workerman_terminals(
                    nr_optimize, config, budget, include_danger, None, seed
                )
            else:
                assert strictness is not None
                mip_instance = test.generate_terminals_mip(
                    mip_optimize, config, value, include_danger, None, seed, strictness
                )
                nr_instance = test.generate_terminals(
                    nr_optimize, config, value, include_danger, None, seed, strictness
                )

            assert mip_instance.result and nr_instance.result

            metric = FuzzInstanceMetrics(
                seed=seed,
                test_type=case_type,
                budget=budget,
                percent=nr_instance.percent,
                strictness=strictness if case_type == test.TestCaseType.STRICTNESS_LEVEL else None,
                include_danger=include_danger,
                roots=nr_instance.roots,
                workers=nr_instance.workers,
                dangers=nr_instance.dangers,
                mip_cost=mip_instance.result.cost,
                mip_duration=mip_instance.result.duration,
                nr_cost=nr_instance.result.cost,
                nr_duration=nr_instance.result.duration,
            )
            case_metrics.append(metric)
            all_metrics.append(metric)

            if metric.nr_cost > metric.mip_cost:
                gap = metric.nr_cost - metric.mip_cost
                logger.warning(f"SUBOPTIMAL → {desc} | seed={seed} | +{gap}")

            print(
                f"[{i + 1:>4}/{samples}] seed={seed:5}  {desc:65}  "
                f"|T|={metric.roots + metric.workers + metric.dangers:>3} "
                f"w:{metric.workers:>3} d:{metric.dangers:>2}  "
                f"MIP {metric.mip_cost:>4} ({metric.mip_duration:6.3f}s) → "
                f"NR {metric.nr_cost:>4} ({metric.nr_duration:6.3f}s)  "
                f"ratio={metric.ratio:5.3f}  {metric.speedup:7.1f}x"
            )

        # --- per‑case summary (scoped to this single job) ---
        if case_metrics:
            ratios = [m.ratio for m in case_metrics]
            optimal = sum(1 for m in case_metrics if m.nr_cost == m.mip_cost)
            suboptimal = len(case_metrics) - optimal

            print("\n" + "=" * 110)
            print(f"SUMMARY — {desc}")
            print(f"Instances tested       : {len(case_metrics)}")
            print(f"NodeRouter optimal     : {optimal} ({optimal / len(case_metrics):.1%})")
            print(f"NodeRouter suboptimal  : {suboptimal} ({suboptimal / len(case_metrics):.1%})")
            print(f"Avg approx ratio       : {mean(ratios):.4f} ± {stdev(ratios):.4f}")
            print(f"Best / Worst ratio     : {min(ratios):.3f} / {max(ratios):.3f}")
            print(f"Avg speedup            : {mean([m.speedup for m in case_metrics]):.0f}x")
            print(
                f"Avg roots / workers / dangers : "
                f"{mean([m.roots for m in case_metrics]):.1f} / "
                f"{mean([m.workers for m in case_metrics]):.1f} / "
                f"{mean([m.dangers for m in case_metrics]):.1f}"
            )
            logger.success(f"Completed: {desc}")
            print("=" * 110)

    return all_metrics


def _print_global_summary(all_metrics: list[FuzzInstanceMetrics]) -> None:
    if not all_metrics:
        logger.warning("No results to summarize.")
        return

    start_time = time.perf_counter()

    grouped = defaultdict(list)
    for m in all_metrics:
        key = (m.test_type, m.budget, m.strictness, m.include_danger)
        grouped[key].append(m)

    def sort_key(item):
        case_type, budget, strictness, inc_danger = item[0]
        strictness_order = list(test.RootPairingType).index(strictness) if strictness else -1
        return (case_type.value, budget or 0, strictness_order, inc_danger)

    max_inst = max(len(metrics) for metrics in grouped.values())
    have_incomplete = False

    table_data = []
    all_ratios, all_speedups = [], []
    total_instances, total_optimal = 0, 0

    for (case_type, budget, strictness, inc_danger), metrics in sorted(grouped.items(), key=sort_key):
        instances = len(metrics)
        optimal = sum(1 for m in metrics if m.nr_cost == m.mip_cost)
        ratios = [m.ratio for m in metrics]
        speedups = [m.speedup for m in metrics]
        terminals = [m.roots + m.workers + m.dangers for m in metrics]

        total_instances += instances
        total_optimal += optimal
        all_ratios.extend(ratios)
        all_speedups.extend(speedups)

        # include percent (average across metrics in this group)
        avg_percent = (
            mean([m.percent for m in metrics if m.percent is not None])
            if any(m.percent is not None for m in metrics)
            else ""
        )

        case_type_string = "workerman" if case_type == test.TestCaseType.WORKERMAN else strictness

        if instances < max_inst:
            have_incomplete = True
            case_type_string += "*"

        table_data.append([
            case_type_string,
            budget,
            avg_percent,
            f"{mean([m.roots for m in metrics]):.1f}",
            f"{mean([m.workers for m in metrics]):.0f}",
            f"{mean([m.dangers for m in metrics]):.0f}",
            f"{mean(terminals):.1f}",
            instances,
            optimal,
            instances - optimal,
            f"{optimal / instances:.1%}",
            f"{mean(ratios):.4f}",
            f"{max(ratios):.3f}",
            f"{mean(speedups):.0f}x",
        ])

    # TOTAL row
    table_data.append([
        "TOTAL",
        "",
        "",
        "",
        "",
        "",
        "",
        total_instances,
        total_optimal,
        total_instances - total_optimal,
        f"{total_optimal / total_instances:.1%}",
        f"{mean(all_ratios):.4f}",
        f"{max(all_ratios):.3f}",
        f"{mean(all_speedups):.0f}x",
    ])

    headers = [
        "Type",
        "Budget",
        "Percent",
        "Roots",
        "Workers",
        "Dangers",
        "Terminals",
        "Inst",
        "Opt",
        "Sub",
        "% Opt",
        "Avg Ratio",
        "Worst",
        "Speedup",
    ]

    print("\n" + "#" * 160)
    logger.success("GLOBAL SUMMARY — BY TEST TYPE, BUDGET, STRICTNESS, AND PERCENT")
    print(tabulate(table_data, headers=headers, tablefmt="simple", stralign="right", numalign="right"))
    if have_incomplete:
        logger.warning("* incomplete test type.")
    print("#" * 160)

    # Worst suboptimal instances
    subopts = [m for m in all_metrics if m.nr_cost > m.mip_cost]
    if subopts:
        worst_sorted = sorted(subopts, key=lambda m: m.ratio, reverse=True)[:20]

        print("\nWORST SUBOPTIMAL INSTANCES (top 20)")
        print("-" * 100)

        for m in worst_sorted:
            gap = m.nr_cost - m.mip_cost
            print(
                f"seed={m.seed:<5d} | {m.test_type:<15} | "
                f"budget={m.budget if m.budget else '':<3} | "
                f"strictness={m.strictness.value if m.strictness else '':<8} | "
                f"percent={m.percent if m.percent is not None else '':<3} | "
                f"roots={m.roots:2d} | workers={m.workers:3d} | dngr={m.dangers:1d} | "
                f"|T|={m.roots + m.workers + m.dangers:3d} | "
                f"MIP={m.mip_cost:4d} → NR={m.nr_cost:4d} (+{gap:2d}) | "
                f"ratio={m.ratio:.3f}"
            )

    print("#" * 160)
    logger.success("Fuzz test suite completed")
    print(f"\nSummary Completed in {time.perf_counter() - start_time:.3f}s")


def run_fuzz_comparison(samples: int = 100, budgets: list[int] | None = None) -> None:
    if budgets is None:
        budgets = list(range(5, 551, 5))

    all_metrics: list[FuzzInstanceMetrics] = []
    install_graceful_shutdown((all_metrics,))

    try:
        for budget in budgets:
            for include_danger in (False, True):
                for test_type in ("workerman", "strictness_level"):
                    if _shutdown_requested:
                        raise KeyboardInterrupt
                    metrics = run_single_config(samples, budget, test_type, include_danger)
                    all_metrics.extend(metrics)

        _print_global_summary(all_metrics)

    except KeyboardInterrupt:
        print("\nShutdown complete — generating summary from accumulated data...")
        _print_global_summary(all_metrics)
        sys.exit(0)


if __name__ == "__main__":
    config = ds.get_config("config")
    if config.get("actions", {}).get("fuzz_test", True):
        cfg = config.get("fuzz_test_config", {})
        run_fuzz_comparison(
            samples=cfg.get("samples", 10),
            budgets=cfg.get("budgets", range(5, 555, 5)),
        )
        logger.success("Fuzz test suite finished")
    else:
        logger.info("fuzz_test not enabled — set actions.fuzz_test: true in config")
