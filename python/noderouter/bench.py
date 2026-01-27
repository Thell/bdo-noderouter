# bench.py

from __future__ import annotations

import hashlib
import time
import signal
import sys
from functools import partial

import polars as pl
from loguru import logger

import api_data_store as ds
from api_common import MAX_BUDGET, set_logger
from orchestrator import execute_plan
from orchestrator_types import Plan, Instance, SeedType
from orchestrator_pairing_strategy import PairingStrategy, MAX_LEN_PAIRING_STRATEGY

from optimizer_nr import optimize_with_terminals as nr_optimize

SUMMARY_FLOAT_PRECISION = 3

_shutdown_requested = False


class _BenchInstanceMetrics(dict):
    """Factory for producing Polars rows from test incidents."""

    def __init__(
        self,
        seed: SeedType,
        budget: int,
        percent: int,
        include_danger: bool,
        strategy: PairingStrategy,
        nr_instance: Instance,
    ):
        # Plan should be the same except for allow_cache.
        assert seed == nr_instance.plan.seed
        assert budget == nr_instance.plan.budget
        assert percent == nr_instance.plan.worker_percent
        assert include_danger == nr_instance.plan.include_danger
        assert strategy == nr_instance.plan.strategy

        # Solution sanity checks
        assert nr_instance.solution
        assert nr_instance.solution.cost > 0 and nr_instance.solution.duration > 0

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
            "nr_cost": nr_instance.solution.cost,
            "nr_duration": nr_instance.solution.duration,
        })

    def generate_log_string(self, i: int, samples: int) -> str:
        """Generates the single-line log string for this instance."""
        return (
            f"[{i + 1:>4}/{samples}] "
            f"{self['seed']:7} "
            f"{self['strategy']:<{MAX_LEN_PAIRING_STRATEGY}} "
            f"Budget: {self['budget']:<3} "
            f"Pct: {self['percent']:2}% "
            f"|T|: {self['terminals']:<3} "
            f"|R|: {self['roots']:<2} "
            f"|W|: {self['workers']:<3} "
            f"|D|: {self['dangers']:<2} "
            f"NR {self['nr_cost']:<3} ({self['nr_duration']:6.3f}s)"
        )


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print("\nHard exit — terminating immediately.", file=sys.stderr)
        sys.exit(1)
    _shutdown_requested = True
    print("\nShutdown requested — finishing current test and reporting results...", file=sys.stderr)


def _install_shutdown_handler():
    handler = partial(_signal_handler)
    signal.signal(signal.SIGINT, handler)


# MARK: Summary Reporting
def _generate_all_cases_summaries(all_cases_df: pl.DataFrame) -> None:
    if all_cases_df.is_empty():
        logger.warning("No results to summarize.")
        return

    start_time = time.perf_counter()

    # --- Strategy summary ---
    summary_df = _generate_summary(all_cases_df).drop(["include_danger"])

    if not summary_df.is_empty():
        summary_total = _generate_summary_total(summary_df)
        strategy_df_aggregate_summary = _generate_strategy_aggregate_summary(summary_df)
        strategy_df_budget_summary = _generate_budget_aggregate_summary(summary_df)

        print("\n### NODE ROUTER SUMMARY ###")
        _print_summary(summary_df)
        print("\n--- BY STRATEGY ---")
        _print_summary(strategy_df_aggregate_summary)
        print("\n--- BY BUDGET ---")
        _print_summary(strategy_df_budget_summary)
        _print_total(summary_total)

    print("#" * 160)
    print(f"\nSummary Completed in {time.perf_counter() - start_time:.3f}s")


def _generate_single_case_summary(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(["budget", "percent", "strategy", "include_danger"]).agg([
        pl.len().alias("instances"),
        pl.mean("terminals").alias("avg_terminals"),
        pl.mean("roots").alias("avg_roots"),
        pl.mean("workers").alias("avg_workers"),
        pl.mean("dangers").alias("avg_dangers"),
        pl.mean("nr_cost").alias("avg_cost"),
        pl.mean("nr_duration").alias("avg_duration"),
    ])


def _generate_summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .group_by(["budget", "percent", "strategy", "include_danger"])
        .agg([
            pl.len().alias("instances"),
            pl.mean("terminals").alias("avg_terminals"),
            pl.mean("roots").alias("avg_roots"),
            pl.mean("workers").alias("avg_workers"),
            pl.mean("dangers").alias("avg_dangers"),
            pl.mean("nr_cost").alias("avg_cost"),
            pl.mean("nr_duration").alias("avg_duration"),
        ])
        .sort(["strategy", "budget", "include_danger"])
    )


def _generate_summary_total(df: pl.DataFrame) -> pl.DataFrame:
    """Generate a 1 row summary for all cases in df."""
    longest_strategy = len(max(df["strategy"], key=len))
    col_widths = [len(col) for col in df.columns]
    col_widths[0] = max(longest_strategy, col_widths[0])

    total_df = df.select([
        pl.lit("TOTAL").alias("strategy"),
        pl.lit("-").alias("budget"),
        pl.col("instances").sum(),
        pl.lit("-").alias("avg_terminals"),
        pl.lit("-").alias("avg_roots"),
        pl.lit("-").alias("avg_workers"),
        pl.lit("-").alias("avg_dangers"),
        pl.mean("avg_cost"),
        pl.mean("avg_duration"),
    ])

    # Round and cast to string
    total_df = total_df.with_columns([
        pl.col(c).cast(pl.Float64).round(SUMMARY_FLOAT_PRECISION).cast(pl.String).alias(c)
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


def _generate_strategy_aggregate_summary(summary_df: pl.DataFrame) -> pl.DataFrame:
    return (
        summary_df
        .group_by("strategy")
        .agg([
            pl.lit("-").alias("budget"),
            pl.col("instances").sum(),
            pl.mean("avg_terminals").alias("avg_terminals"),
            pl.mean("avg_roots").alias("avg_roots"),
            pl.mean("avg_workers").alias("avg_workers"),
            pl.mean("avg_dangers").alias("avg_dangers"),
            pl.mean("avg_cost").alias("avg_cost"),
            pl.mean("avg_duration").alias("avg_duration"),
        ])
        .sort("strategy")
    )


def _generate_budget_aggregate_summary(summary_df: pl.DataFrame) -> pl.DataFrame:
    longest = max(len(str(v)) for v in summary_df["strategy"].to_list())
    col_widths = [len(c) for c in summary_df.columns]
    col_widths[0] = max(longest, col_widths[0])

    tmp_df = (
        summary_df
        .group_by(["budget"])
        .agg([
            pl.lit("-" + " " * (col_widths[0] - len("-"))).alias("strategy"),
            pl.col("instances").sum(),
            pl.mean("avg_terminals").alias("avg_terminals"),
            pl.mean("avg_roots").alias("avg_roots"),
            pl.mean("avg_workers").alias("avg_workers"),
            pl.mean("avg_dangers").alias("avg_dangers"),
            pl.mean("avg_cost").alias("avg_cost"),
            pl.mean("avg_duration").alias("avg_duration"),
        ])
        .sort("budget")
    )

    return tmp_df.select("strategy", "budget", pl.all().exclude(["strategy", "budget"]))


def _print_summary(df: pl.DataFrame) -> None:
    with pl.Config(
        thousands_separator=True,
        set_float_precision=SUMMARY_FLOAT_PRECISION,
        set_fmt_str_lengths=100,
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        set_tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=-1,
    ):
        print(df)


def _print_total(df: pl.DataFrame) -> None:
    with pl.Config(
        thousands_separator=True,
        set_float_precision=SUMMARY_FLOAT_PRECISION,
        set_fmt_str_lengths=100,
        set_tbl_hide_column_names=True,
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        set_tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=-1,
    ):
        print(df)


# MARK: Main Benchmark
def _make_seed(budget: int, strategy: PairingStrategy, i: int) -> SeedType:
    """Produce a deterministic seed for a given sample."""
    # NOTE: For reproducibility purposes we use deterministic seeds.
    # This ensures that each sample's random terminals are not a 'core' of future budgets.
    # This is not tied to the danger inclusion for a pairing strategy.
    # The solver's methodology for handling dangers can potentially 'break' an otherwise
    # optimally solved problem and by keeping them fixed for danger inclusive and exclusive
    # samples we ensure that such cases are identifiable.
    return hashlib.sha256(f"{budget}:{strategy}:{i}".encode("utf-8")).hexdigest()[:7]


def _run_single_config(
    strategies: list[PairingStrategy], samples: int, budget: int, include_danger: bool
) -> pl.DataFrame:
    """
    Run NodeRouter benchmarks for a given budget across all pairing strategies.
    """
    config = ds.get_config("config")

    all_cases_df = pl.DataFrame()

    assert budget <= MAX_BUDGET
    percent = round(budget / MAX_BUDGET * 100)

    for strategy in strategies:
        if _shutdown_requested:
            break

        if strategy == PairingStrategy.custom:
            continue

        desc = f"strategy={strategy.value}, budget={budget:3}, danger={'yes' if include_danger else 'no'}"
        logger.info(f"\nStarting: {desc}, samples={samples}")

        case_rows: list[_BenchInstanceMetrics] = []

        for i in range(samples):
            if _shutdown_requested:
                break

            seed = _make_seed(budget, strategy, i)
            nr_plan = Plan(nr_optimize, config, budget, percent, seed, include_danger, strategy, False)

            nr_instance = execute_plan(nr_plan)

            row = _BenchInstanceMetrics(
                seed,
                budget,
                percent,
                include_danger,
                strategy,
                nr_instance,
            )
            case_rows.append(row)

            log_str = row.generate_log_string(i, samples)
            logger.success(f"✅ {log_str}")

            # The MIP optimized strategy should only be executed once since
            # the pairs will always be the same for a given budget.
            if strategy == PairingStrategy.optimized:
                break

        case_df = pl.DataFrame(case_rows)
        all_cases_df = all_cases_df.vstack(case_df)

        case_summary = _generate_single_case_summary(case_df)
        _print_summary(case_summary)

    return all_cases_df


def bench_main(
    strategies: list[PairingStrategy], samples: int, budgets: list[int] | range, quiet: bool
) -> None:
    # NOTE: When doing a long run writing per test outputs can become a bottleneck.
    if quiet:
        set_logger({"logger": {"level": "ERROR", "format": "<level>{message}</level>"}})
    else:
        set_logger(ds.get_config("config"))

    all_metrics: pl.DataFrame = pl.DataFrame()
    _install_shutdown_handler()

    start_time = time.time()
    try:
        for budget in budgets:
            for include_danger in (False, True):
                if _shutdown_requested:
                    raise KeyboardInterrupt
                metrics = _run_single_config(strategies, samples, budget, include_danger)
                all_metrics = all_metrics.vstack(metrics)
        _generate_all_cases_summaries(all_metrics)
    except KeyboardInterrupt:
        print("\nShutdown complete — generating summary from accumulated data...")
        _generate_all_cases_summaries(all_metrics)
        sys.exit(0)

    # Save results to csv
    all_metrics.write_csv("bench_results_extended_bridge.csv")

    logger.success("NodeRouter stress benchmark finished")
    print(f"Total runtime: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    # NOTE: ****
    #       Ensure that NodeRouter is built with csv results writing capabilities
    #       before running this script if the intent is to run this to generate
    #       algorithmic approach ranking data for analysis using 'rank_nr.py'
    #
    #       You may also want to ensure that the NodeRouter 'algo_results.csv' file
    #       is empty before running this script because NodeRouter will write
    #       to this file in append mode.
    #
    #       ****

    # NOTE: For full benchmarking we would want to include all possible strategies
    # strategies = [s for s in PairingStrategy]

    # NOTE: For testing purposes we can use a subset of strategies
    # strategies = [PairingStrategy.optimized, PairingStrategy.random_town]

    # NOTE: For normal benchmarking we use a subset of budgets since the bench
    #       is run for each strategy within each budget times the number of samples.
    #       The logical limit would be 5..=600, stepping by less than 5 does not
    #       make sense.
    # budgets = range(5, 46, 5)

    # NOTE: For normal benchmarking or testing purposes the sample count
    #       can be adjusted as desired. The default is 20 to allow reasonable
    #       statistics without excessive runtime per configuration.
    # samples = 20

    # # Settings for a very long run to collect extensive router data
    strategies = [s for s in PairingStrategy if s != PairingStrategy.custom]
    budgets = range(5, MAX_BUDGET + 1, 5)
    samples = 50

    # NOTE: When doing a long run writing per test outputs can become a bottleneck.
    quiet = True

    bench_main(strategies, samples, budgets, quiet)
