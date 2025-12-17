# fuzz.py

from __future__ import annotations

import time
import signal
import sys
from functools import partial

import polars as pl
from loguru import logger

import api_data_store as ds
from api_common import MAX_BUDGET, set_logger
from orchestrator import execute_plan
from orchestrator_types import Plan, Instance
from orchestrator_pairing_strategy import PairingStrategy

from optimizer_mip import optimize_with_terminals as mip_optimize
from optimizer_nr import optimize_with_terminals as nr_optimize

FLOAT_DECIMALS = 3
WORST_SUBOPTIMAL_REPORTING_COUNT = 20
_shutdown_requested = False


class FuzzInstanceMetrics(dict):
    """Factory for producing Polars rows from test incidents."""

    def __init__(
        self,
        seed: int,
        budget: int,
        percent: int,
        include_danger: bool,
        strategy: PairingStrategy,
        nr_instance: Instance,
        mip_instance: Instance,
    ):
        # Plan should be the same except for allow_cache.
        assert seed == nr_instance.plan.seed == mip_instance.plan.seed
        assert budget == nr_instance.plan.budget == mip_instance.plan.budget
        assert percent == nr_instance.plan.worker_percent == mip_instance.plan.worker_percent
        assert include_danger == nr_instance.plan.include_danger == mip_instance.plan.include_danger
        assert strategy == nr_instance.plan.strategy == mip_instance.plan.strategy

        # Terminals should always be the same.
        assert mip_instance.terminals == nr_instance.terminals

        # Solution sanity checks
        assert mip_instance.solution and nr_instance.solution
        assert mip_instance.solution.cost > 0 and mip_instance.solution.duration > 0
        assert nr_instance.solution.cost > 0 and nr_instance.solution.duration > 0

        # Solver sanity checks
        ratio = nr_instance.solution.cost / mip_instance.solution.cost
        speedup = mip_instance.solution.duration / nr_instance.solution.duration
        assert ratio >= 1.0, "NodeRouter should never have lower cost than MIP!"
        assert speedup > 0, "NodeRouter should always be faster than MIP!"

        super().__init__({
            "seed": seed,
            "budget": budget,
            "strategy": strategy.value,
            "include_danger": include_danger,
            "percent": nr_instance.plan.worker_percent,
            "terminals": len(nr_instance.terminals.terminals),
            "roots": nr_instance.terminals.roots,
            "workers": nr_instance.terminals.workers,
            "dangers": nr_instance.terminals.dangers,
            "mip_cost": mip_instance.solution.cost,
            "mip_duration": mip_instance.solution.duration,
            "nr_cost": nr_instance.solution.cost,
            "nr_duration": nr_instance.solution.duration,
            "ratio": ratio,
            "speedup": speedup,
        })


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print("\nHard exit — terminating immediately.", file=sys.stderr)
        sys.exit(1)
    _shutdown_requested = True
    print("\nShutdown requested — finishing current test and reporting results...", file=sys.stderr)


def install_shutdown_handler():
    handler = partial(_signal_handler)
    signal.signal(signal.SIGINT, handler)


def run_single_config(samples: int, budget: int, include_danger: bool) -> pl.DataFrame:
    """
    Run fuzz tests for a given budget across all pairing strategies.
    NOTE: Percent is ignored upon _input_ for PairingStrategy.optimized and populated upon output.
    """
    config = ds.get_config("config")

    all_cases_df = pl.DataFrame()
    base_seed = 10_000

    assert budget <= MAX_BUDGET
    percent = round(budget / MAX_BUDGET * 100)

    for strategy in PairingStrategy:
        if _shutdown_requested:
            break

        if strategy == PairingStrategy.custom:
            continue

        desc = f"strategy={strategy.value}, budget={budget:3}, danger={'yes' if include_danger else 'no'}"
        print(f"\nStarting: {desc}, samples={samples}")

        case_rows: list[FuzzInstanceMetrics] = []

        for i in range(samples):
            if _shutdown_requested:
                break
            seed = base_seed + i

            mip_plan = Plan(mip_optimize, config, budget, percent, seed, include_danger, strategy, True)
            nr_plan = Plan(nr_optimize, config, budget, percent, seed, include_danger, strategy, False)

            mip_instance = execute_plan(mip_plan)
            nr_instance = execute_plan(nr_plan)

            row = FuzzInstanceMetrics(
                seed,
                budget,
                percent,
                include_danger,
                strategy,
                nr_instance,
                mip_instance,
            )
            case_rows.append(row)

            assert mip_instance.solution and nr_instance.solution

            row_str = (
                f"[{i + 1:>4}/{samples}] seed={seed:5}, {desc}, "
                f"percent={percent:3}% "
                f"|T|={len(nr_instance.terminals.terminals)} "
                f"w:{row['workers']} d:{row['dangers']} "
                f"MIP {row['mip_cost']} ({row['mip_duration']:6.3f}s) → "
                f"NR {row['nr_cost']} ({row['nr_duration']:6.3f}s) "
                f"ratio={row['ratio']:5.3f} {row['speedup']:7.1f}x"
            )
            if (gap := row["nr_cost"] - row["mip_cost"]) > 0:
                logger.warning(f"SUBOPTIMAL → {row_str} (gap: +{gap})")
            else:
                # Output at same level as SUCCESS but without color
                logger.log(25, f"   OPTIMAL → {row_str}")

        case_df = pl.DataFrame(case_rows)
        all_cases_df = all_cases_df.vstack(case_df)

        case_summary = case_df.group_by(["budget", "percent", "strategy", "include_danger"]).agg([
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


def generate_summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["strategy", "budget", "include_danger"])
        .agg([
            pl.len().alias("instances"),
            (pl.col("nr_cost") == pl.col("mip_cost")).sum().alias("optimal"),
            (pl.col("nr_cost") != pl.col("mip_cost")).sum().alias("suboptimal"),
            (pl.col("nr_cost") == pl.col("mip_cost")).mean().alias("optimal_percent"),
            pl.mean("terminals").alias("avg_terminals"),
            pl.mean("roots").alias("avg_roots"),
            pl.mean("workers").alias("avg_workers"),
            pl.mean("dangers").alias("avg_dangers"),
            pl.mean("ratio").alias("avg_ratio"),
            pl.max("ratio").alias("worst_ratio"),
            pl.mean("speedup").alias("avg_speedup"),
        ])
        .sort(["strategy", "budget", "include_danger"])
    )


def generate_summary_total(df: pl.DataFrame) -> pl.DataFrame:
    """Generate a 1 row summary for all cases in df."""
    longest_strategy = len(max(df["strategy"], key=len))
    col_widths = [len(col) for col in df.columns]
    col_widths[0] = max(longest_strategy, col_widths[0])

    total_df = df.select([
        pl.lit("TOTAL").alias("strategy"),
        pl.lit("-").alias("budget"),
        pl.col("instances").sum(),
        pl.col("optimal").sum(),
        pl.col("suboptimal").sum(),
        pl.col("optimal_percent").mean(),
        pl.lit("-").alias("avg_terminals"),
        pl.lit("-").alias("avg_roots"),
        pl.lit("-").alias("avg_workers"),
        pl.lit("-").alias("avg_dangers"),
        pl.mean("avg_ratio"),
        pl.max("worst_ratio"),
        pl.mean("avg_speedup"),
    ])

    # Round and cast to string
    total_df = total_df.with_columns([
        pl.col(c).cast(pl.Float64).round(FLOAT_DECIMALS).cast(pl.String).alias(c)
        for c in total_df.select(pl.selectors.numeric()).columns
    ]).with_columns(pl.all().cast(pl.String))

    # Pad strings
    padded_strings = []
    for str_value in total_df.row(0):
        padding_needed = col_widths.pop(0) - len(str_value)
        if padding_needed > 0:
            padded_str_value = str_value + " " * padding_needed
        else:
            padded_str_value = str_value
        padded_strings.append(padded_str_value)
    return pl.DataFrame([padded_strings], schema=total_df.schema, orient="row")


def generate_strategy_aggregate_summary(summary_df: pl.DataFrame) -> pl.DataFrame:
    return (
        summary_df.group_by("strategy")
        .agg([
            pl.lit("-").alias("budget"),
            pl.col("instances").sum(),
            pl.col("optimal").sum(),
            pl.col("suboptimal").sum(),
            pl.col("optimal_percent").mean(),
            pl.mean("avg_terminals").alias("avg_terminals"),
            pl.mean("avg_roots").alias("avg_roots"),
            pl.mean("avg_workers").alias("avg_workers"),
            pl.mean("avg_dangers").alias("avg_dangers"),
            pl.col("avg_ratio").mean(),
            pl.col("worst_ratio").max(),
            pl.col("avg_speedup").mean(),
        ])
        .sort("strategy")
    )


def generate_suboptimal_breakdown(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["strategy", "include_danger"])
        .agg([
            pl.len().alias("instances"),
            pl.mean("ratio").alias("avg_ratio"),
            pl.max("ratio").alias("worst_ratio"),
            pl.mean("speedup").alias("avg_speedup"),
            pl.min("budget").alias("first_budget"),
            pl.col("ratio")
            .filter(pl.col("budget") == pl.min("budget"))
            .mean()
            .alias("ratio_at_first_budget"),
            pl.max("budget").alias("last_budget"),
            pl.col("ratio").filter(pl.col("budget") == pl.max("budget")).mean().alias("ratio_at_last_budget"),
        ])
        .sort(["avg_ratio"], descending=True)
    )


def generate_suboptimal_by_danger(suboptimal_df: pl.DataFrame) -> pl.DataFrame:
    return (
        suboptimal_df.group_by("include_danger")
        .agg([
            pl.col("instances").sum(),
            pl.col("avg_ratio").mean(),
            pl.col("worst_ratio").max(),
            pl.col("avg_speedup").mean(),
        ])
        .sort("avg_ratio", descending=True)
    )


def generate_suboptimal_by_danger_total(suboptimal_by_danger_df: pl.DataFrame) -> pl.DataFrame:
    longest = max(len(str(v)) for v in suboptimal_by_danger_df["include_danger"].to_list())
    col_widths = [len(c) for c in suboptimal_by_danger_df.columns]
    col_widths[0] = max(longest, col_widths[0])

    total = suboptimal_by_danger_df.select([
        pl.lit("TOTAL").alias("include_danger"),
        pl.col("instances").sum(),
        pl.mean("avg_ratio"),
        pl.max("worst_ratio"),
        pl.mean("avg_speedup"),
    ])

    total = total.with_columns([
        pl.col(c).cast(pl.Float64).round(FLOAT_DECIMALS).cast(pl.String).alias(c)
        for c in total.select(pl.selectors.numeric()).columns
    ]).with_columns(pl.all().cast(pl.String))

    padded = []
    for val, w in zip(total.row(0), col_widths):
        pad = w - len(val)
        padded.append(val + (" " * pad if pad > 0 else ""))
    return pl.DataFrame([padded], schema=total.schema, orient="row")


def generate_worst_suboptimal_summary(suboptimal_df: pl.DataFrame) -> pl.DataFrame:
    return (
        suboptimal_df.with_columns(
            (pl.col("nr_cost") - pl.col("mip_cost")).alias("gap"),
        )
        .sort("ratio", descending=True)
        .head(WORST_SUBOPTIMAL_REPORTING_COUNT)
        .select([
            "seed",
            "strategy",
            "budget",
            "percent",
            "terminals",
            "roots",
            "workers",
            "dangers",
            "mip_cost",
            "nr_cost",
            "gap",
            "ratio",
        ])
    )


def print_summary(df: pl.DataFrame) -> None:
    with pl.Config(
        set_float_precision=FLOAT_DECIMALS,
        set_fmt_str_lengths=100,
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        set_tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=-1,
    ):
        print(df)


def print_total(df: pl.DataFrame) -> None:
    with pl.Config(
        set_float_precision=FLOAT_DECIMALS,
        set_fmt_str_lengths=100,
        set_tbl_hide_column_names=True,
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        set_tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=-1,
    ):
        print(df)


def generate_summaries(all_cases_df: pl.DataFrame) -> None:
    if all_cases_df.is_empty():
        logger.warning("No results to summarize.")
        return

    start_time = time.perf_counter()

    # --- Optimized (workerman) summary ---
    optimized_df = all_cases_df.filter(pl.col("strategy") == PairingStrategy.optimized.value)
    if not optimized_df.is_empty():
        optimized_df_summary = generate_summary(optimized_df).drop(["include_danger"])
        optimized_df_total = generate_summary_total(optimized_df_summary)

        print("\n### OPTIMIZED (WORKERMAN) SUMMARY ###")
        print_summary(optimized_df_summary)
        print_total(optimized_df_total)

    # --- Strategy summary (all non-optimized) ---
    strategy_df = all_cases_df.filter(pl.col("strategy") != PairingStrategy.optimized.value)
    if not strategy_df.is_empty():
        strategy_df_summary = generate_summary(strategy_df).drop(["include_danger"])
        strategy_df_aggregate_summary = generate_strategy_aggregate_summary(strategy_df_summary)
        strategy_df_total = generate_summary_total(strategy_df_summary)
        out_path = ds.path() / "strategy_summary.csv"
        strategy_df_summary.with_columns(pl.selectors.float().round(3)).write_csv(out_path)

        print("\n### STRATEGY SUMMARY ###")
        print_summary(strategy_df_summary)
        print_summary(strategy_df_aggregate_summary)
        print_total(strategy_df_total)

    # --- Suboptimal breakdown diagnostics ---
    suboptimal_df = all_cases_df.filter(pl.col("nr_cost") > pl.col("mip_cost"))
    if not suboptimal_df.is_empty():
        suboptimal_breakdown_df = generate_suboptimal_breakdown(suboptimal_df)
        print("\n### SUBOPTIMAL BREAKDOWN ###")
        print_summary(suboptimal_breakdown_df)

        # --- Suboptimal by danger ---
        suboptimal_by_danger_df = generate_suboptimal_by_danger(suboptimal_breakdown_df)
        suboptimal_by_danger_total = generate_suboptimal_by_danger_total(suboptimal_by_danger_df)
        print("\n### SUBOPTIMAL BY DANGER ###")
        print_summary(suboptimal_by_danger_df)
        print_total(suboptimal_by_danger_total)

        # --- Worst suboptimal instances ---
        worst_suboptimal_df = generate_worst_suboptimal_summary(suboptimal_df)
        out_path = ds.path() / "worst_suboptimal_instances.json"
        worst_suboptimal_df.with_columns(pl.selectors.float().round(3)).write_json(out_path)

        print(f"\n### WORST SUBOPTIMAL INSTANCES (top {WORST_SUBOPTIMAL_REPORTING_COUNT}) ###")
        print_summary(worst_suboptimal_df)

    print("#" * 160)
    print(f"\nSummary Completed in {time.perf_counter() - start_time:.3f}s")


def run_fuzz_comparison(samples: int, budgets: list[int]) -> None:
    all_metrics: pl.DataFrame = pl.DataFrame()
    install_shutdown_handler()

    try:
        for budget in budgets:
            for include_danger in (False, True):
                if _shutdown_requested:
                    raise KeyboardInterrupt
                metrics = run_single_config(samples, budget, include_danger)
                all_metrics = all_metrics.vstack(metrics)

        generate_summaries(all_metrics)

    except KeyboardInterrupt:
        print("\nShutdown complete — generating summary from accumulated data...")
        generate_summaries(all_metrics)
        sys.exit(0)


if __name__ == "__main__":
    config = ds.get_config("config")
    set_logger(config)

    run_fuzz_comparison(samples=20, budgets=list(range(5, 201, 5)))
    logger.success("Fuzz test suite finished")
