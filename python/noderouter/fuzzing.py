# fuzzing.py

from __future__ import annotations
import time
from functools import partial
import signal
import sys

from loguru import logger
import polars as pl

import api_data_store as ds
import testing as test
from api_common import set_logger
from mip_baseline import optimize_with_terminals as mip_optimize
from node_router import optimize_with_terminals as nr_optimize
from testing_root_pairing import PairingStrategy

_shutdown_requested = False


class FuzzInstanceMetrics(dict):
    """Factory for producing Polars rows from test incidents."""

    def __init__(
        self,
        seed: int,
        case_type: test.TestCaseType,
        budget: int,
        include_danger: bool,
        strategy: PairingStrategy | None,
        nr_instance: test.TestInstance,
        mip_instance: test.TestInstance,
    ):
        # If any of these fail, something is wrong with either the MIP/NodeRouter input or
        # the solution extraction/result assignment. This should never happen!
        assert mip_instance.result, f"MIP handling error!: {mip_instance.result=}"
        assert mip_instance.result.cost > 0, f"MIP handling error!: {mip_instance.result.cost=}"
        assert mip_instance.result.duration > 0, f"MIP handling error!: {mip_instance.result.duration=}"

        assert nr_instance.result, f"NodeRouter handling error!: {nr_instance.result=}"
        assert nr_instance.result.cost > 0, f"NodeRouter handling error!: {nr_instance.result.cost=}"
        assert nr_instance.result.duration > 0, f"NodeRouter handling error!: {nr_instance.result.duration=}"

        ratio = nr_instance.result.cost / mip_instance.result.cost
        assert ratio >= 1.0, "NodeRouter should never have lower cost than MIP!"
        speedup = mip_instance.result.duration / nr_instance.result.duration
        assert speedup > 0, "NodeRouter should always be faster than MIP!"

        super().__init__({
            "seed": seed,
            "test_type": case_type.value,
            "budget": budget,
            "strategy": strategy.value if strategy else "",
            "include_danger": include_danger,
            "percent": nr_instance.percent,
            "roots": nr_instance.specs.roots,
            "workers": nr_instance.specs.workers,
            "dangers": nr_instance.specs.dangers,
            "mip_cost": mip_instance.result.cost,
            "mip_duration": mip_instance.result.duration,
            "nr_cost": nr_instance.result.cost,
            "nr_duration": nr_instance.result.duration,
            "ratio": ratio,
            "speedup": speedup,
        })


def _signal_handler(signum, frame, containers):
    global _shutdown_requested
    if _shutdown_requested:
        print("\nHard exit — terminating immediately.", file=sys.stderr)
        sys.exit(1)
    _shutdown_requested = True
    print("\nShutdown requested — finishing current test and reporting results...", file=sys.stderr)


def install_shutdown_handler(containers):
    handler = partial(_signal_handler, containers=containers)
    signal.signal(signal.SIGINT, handler)


def run_single_config(samples: int, budget: int, test_type: str, include_danger: bool) -> pl.DataFrame:
    """
    Run a fuzz test for a single configuration.

    Args:
        samples: Number of instances to generate.
        budget: Workerman budget or basis for percent coverage.
        test_type: Either "workerman" or "strategy".
        include_danger: Whether to include danger nodes.

    Returns:
        List of FuzzInstanceMetrics for all instances in this config.
    """
    config = ds.get_config("config")

    base_seed = 10_000
    all_cases_df = pl.DataFrame()

    # Build the job list: one entry for workerman, or one per strategy
    if test_type == "workerman":
        jobs: list[tuple[PairingStrategy | None, int, test.TestCaseType]] = [
            (None, budget, test.TestCaseType.WORKERMAN)
        ]
    elif test_type == "strategy":
        percent = round(budget / 550 * 100)
        jobs = [(s, percent, test.TestCaseType.STRATEGY) for s in PairingStrategy]
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    for strategy, value, case_type in jobs:
        if _shutdown_requested:
            break

        if case_type == test.TestCaseType.WORKERMAN:
            desc = f"workerman budget={budget:3} | danger={'yes' if include_danger else 'no'}"
        else:
            assert strategy is not None
            desc = f"strategy={strategy.value} {value:3}% workers (budget {budget:3}) "
            desc += f"| danger={'yes' if include_danger else 'no'}"

        print(f"\nStarting: {desc} | samples={samples}")

        case_rows: list[FuzzInstanceMetrics] = []

        for i in range(samples):
            if _shutdown_requested:
                break
            seed = base_seed + i

            if case_type == test.TestCaseType.WORKERMAN:
                mip_instance = test.workerman_terminals_mip(
                    mip_optimize, config, budget, seed, include_danger
                )
                nr_instance = test.workerman_terminals(nr_optimize, config, budget, seed, include_danger)
            else:
                assert strategy is not None
                mip_instance = test.generate_terminals_mip(
                    mip_optimize, config, value, seed, include_danger, strategy
                )
                nr_instance = test.generate_terminals(
                    nr_optimize, config, value, seed, include_danger, strategy
                )

            row = FuzzInstanceMetrics(
                seed,
                case_type,
                budget,
                include_danger,
                strategy,
                nr_instance,
                mip_instance,
            )
            case_rows.append(row)

            assert mip_instance.result and nr_instance.result

            row_str = (
                f"[{i + 1:>4}/{samples}] seed={seed:5}  {desc:65}  "
                f"|T|={row['roots'] + row['workers'] + row['dangers']:>3} "
                f"w:{row['workers']:>3} d:{row['dangers']:>2}  "
                f"MIP {row['mip_cost']:>4} ({row['mip_duration']:6.3f}s) → "
                f"NR {row['nr_cost']:>4} ({row['nr_duration']:6.3f}s)  "
                f"ratio={row['ratio']:5.3f}  {row['speedup']:7.1f}x"
            )
            if (gap := row["nr_cost"] - row["mip_cost"]) > 0:
                logger.warning(f"SUBOPTIMAL → {row_str} (gap: +{gap})")
            else:
                # use level greater on par with success but without success coloring
                logger.log(25, f"   OPTIMAL → {row_str}")

        case_df = pl.DataFrame(case_rows)
        all_cases_df = all_cases_df.vstack(case_df)

        case_summary = case_df.group_by(["test_type", "budget", "strategy", "include_danger"]).agg([
            pl.len().alias("instances"),
            (pl.col("nr_cost") == pl.col("mip_cost")).sum().alias("optimal"),
            (pl.col("nr_cost") != pl.col("mip_cost")).sum().alias("suboptimal"),
            pl.mean("ratio").alias("avg_ratio"),
            pl.max("ratio").alias("worst_ratio"),
            pl.mean("speedup").alias("avg_speedup"),
        ])
        with pl.Config(tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True, set_tbl_cols=-1):
            print(case_summary)

    return all_cases_df


def _print_global_summary(all_cases_df: pl.DataFrame) -> None:
    if all_cases_df.is_empty():
        logger.warning("No results to summarize.")
        return

    start_time = time.perf_counter()

    # --- Workerman summary ---
    workerman_df = all_cases_df.filter(pl.col("test_type") == "workerman")
    if not workerman_df.is_empty():
        workerman_summary = (
            workerman_df.group_by(["test_type", "budget", "include_danger"])
            .agg([
                pl.len().alias("instances"),
                (pl.col("nr_cost") == pl.col("mip_cost")).sum().alias("optimal"),
                (pl.col("nr_cost") != pl.col("mip_cost")).sum().alias("suboptimal"),
                (pl.col("nr_cost") == pl.col("mip_cost")).mean().alias("optimal_percent"),
                pl.mean("ratio").alias("avg_ratio"),
                pl.max("ratio").alias("worst_ratio"),
                pl.mean("speedup").alias("avg_speedup"),
                pl.mean("roots").alias("avg_roots"),
                pl.mean("workers").alias("avg_workers"),
                pl.mean("dangers").alias("avg_dangers"),
                (pl.col("roots") + pl.col("workers") + pl.col("dangers")).mean().alias("avg_terminals"),
            ])
            .sort(by=["budget", "include_danger"])
            .drop(["include_danger", "test_type"])
        )
        print("\n### WORKERMAN SUMMARY ###")
        with pl.Config(
            tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True, set_tbl_cols=-1, tbl_rows=-1
        ):
            print(workerman_summary)

    # --- Strategy summary ---
    strategy_df = all_cases_df.filter(pl.col("test_type") == "strategy")
    if not strategy_df.is_empty():
        strategy_summary = (
            strategy_df.group_by(["strategy", "budget", "include_danger"])
            .agg([
                pl.mean("roots").alias("avg_roots"),
                pl.mean("workers").alias("avg_workers"),
                pl.mean("dangers").alias("avg_dangers"),
                pl.len().alias("instances"),
                (pl.col("nr_cost") == pl.col("mip_cost")).sum().alias("optimal"),
                (pl.col("nr_cost") != pl.col("mip_cost")).sum().alias("suboptimal"),
                (pl.col("nr_cost") == pl.col("mip_cost")).mean().alias("optimal_percent"),
                pl.mean("ratio").alias("avg_ratio"),
                pl.max("ratio").alias("worst_ratio"),
                pl.mean("speedup").alias("avg_speedup"),
                (pl.col("roots") + pl.col("workers") + pl.col("dangers")).mean().alias("avg_terminals"),
            ])
            .sort(["strategy", "budget", "include_danger"])
            .drop("include_danger")
        )
        print("\n### STRATEGY SUMMARY ###")
        with pl.Config(
            tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True, set_tbl_cols=-1, tbl_rows=-1
        ):
            print(strategy_summary)

    # --- Suboptimal breakdown diagnostics ---
    sub_df = all_cases_df.filter(pl.col("nr_cost") > pl.col("mip_cost"))
    if not sub_df.is_empty():
        breakdown = (
            sub_df.group_by(["strategy", "include_danger"])
            .agg([
                pl.min("budget").alias("first_budget"),
                pl.max("budget").alias("last_budget"),
                pl.len().alias("instances"),
                pl.mean("ratio").alias("avg_ratio"),
                pl.max("ratio").alias("worst_ratio"),
                pl.mean("speedup").alias("avg_speedup"),
                pl.col("ratio")
                .filter(pl.col("budget") == pl.min("budget"))
                .mean()
                .alias("ratio_at_first_budget"),
                pl.col("ratio")
                .filter(pl.col("budget") == pl.max("budget"))
                .mean()
                .alias("ratio_at_last_budget"),
            ])
            .sort(["avg_ratio"], descending=True)
        )
        print("\n### SUBOPTIMAL BREAKDOWN ###")
        with pl.Config(
            tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True, set_tbl_cols=-1, tbl_rows=-1
        ):
            print(breakdown)

    print(f"\nSummary Completed in {time.perf_counter() - start_time:.3f}s")


def run_fuzz_comparison(samples: int, budgets: list[int]) -> None:
    all_metrics: pl.DataFrame = pl.DataFrame()
    install_shutdown_handler((all_metrics,))

    try:
        for budget in budgets:
            for include_danger in (False, True):
                for test_type in ("workerman", "strategy"):
                    if _shutdown_requested:
                        raise KeyboardInterrupt
                    metrics = run_single_config(samples, budget, test_type, include_danger)
                    all_metrics = all_metrics.vstack(metrics)

        _print_global_summary(all_metrics)

    except KeyboardInterrupt:
        print("\nShutdown complete — generating summary from accumulated data...")
        _print_global_summary(all_metrics)
        sys.exit(0)


if __name__ == "__main__":
    config = ds.get_config("config")
    set_logger(config)

    run_fuzz_comparison(samples=5, budgets=list(range(5, 11, 5)))
    logger.success("Fuzz test suite finished")
