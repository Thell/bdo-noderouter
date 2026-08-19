import polars as pl
from loguru import logger

import api_data_store as ds
from orchestrator_pairing_strategy import PairingStrategy

SUMMARY_FLOAT_PRECISION = 3
WORST_SUBOPTIMAL_REPORTING_COUNT = 50


# MARK: Summary Reporting
def _generate_all_cases_summaries(all_cases_df: pl.DataFrame) -> None:
    if all_cases_df.is_empty():
        logger.warning("No results to summarize.")
        return

    # --- Optimized (workerman) summary ---
    optimized_df = all_cases_df.filter(pl.col("strategy") == PairingStrategy.optimized.value)
    if not optimized_df.is_empty():
        optimized_df_summary = _generate_summary(optimized_df).drop(["include_danger"])
        optimized_df_total = _generate_summary_total(optimized_df_summary)

        print("\n### OPTIMIZED (WORKERMAN) SUMMARY ###")
        _print_summary(optimized_df_summary)
        _print_total(optimized_df_total)

    # --- Strategy summary (all non-optimized) ---
    strategy_df = all_cases_df.filter(pl.col("strategy") != PairingStrategy.optimized.value)

    if not strategy_df.is_empty():
        strategy_df_summary = _generate_summary(strategy_df).drop(["include_danger"])
        strategy_df_aggregate_summary = _generate_strategy_aggregate_summary(strategy_df_summary)
        strategy_df_budget_summary = _generate_budget_aggregate_summary(strategy_df_summary)
        strategy_df_danger_summary = _generate_danger_aggregate_summary(strategy_df_summary)

        strategy_df_total = _generate_summary_total(strategy_df_summary)
        out_path = ds.path() / "strategy_summary.csv"
        strategy_df_summary.with_columns(pl.selectors.float().round(3)).write_csv(out_path)

        print("\n### STRATEGY SUMMARY ###")
        _print_summary(strategy_df_summary)
        print("\n--- BY STRATEGY ---")
        _print_summary(strategy_df_aggregate_summary)
        print("\n--- BY BUDGET ---")
        _print_summary(strategy_df_budget_summary)
        print("\n--- BY DANGER ---")
        _print_summary(strategy_df_danger_summary)
        _print_total(strategy_df_total)

    # --- Suboptimal breakdown diagnostics ---
    suboptimal_df = all_cases_df.filter(pl.col("nr_cost") > pl.col("mip_cost"))

    subset = ["strategy", "seed", "roots", "workers", "dangers"]
    suboptimal_df = suboptimal_df.sort(["ratio"], descending=True).unique(subset=subset, keep="first")

    if not suboptimal_df.is_empty():
        suboptimal_breakdown_df = _generate_suboptimal_breakdown(suboptimal_df)
        print("\n### SUBOPTIMAL BREAKDOWN ###")
        _print_summary(suboptimal_breakdown_df)

        # --- Suboptimal by danger ---
        suboptimal_by_danger_df = _generate_suboptimal_by_danger(suboptimal_breakdown_df)
        suboptimal_by_danger_total = _generate_suboptimal_by_danger_total(suboptimal_by_danger_df)
        print("\n### SUBOPTIMAL BY DANGER ###")
        _print_summary(suboptimal_by_danger_df)
        _print_total(suboptimal_by_danger_total)

        # --- Worst suboptimal instances ---
        worst_suboptimal_df = _generate_worst_suboptimal_summary(suboptimal_df)
        out_path = ds.path() / "worst_suboptimal_instances.json"
        worst_suboptimal_df.with_columns(pl.selectors.float().round(3)).write_json(out_path)

        print(f"\n### WORST SUBOPTIMAL INSTANCES (top {WORST_SUBOPTIMAL_REPORTING_COUNT}) ###")
        _print_summary(worst_suboptimal_df)

    print("#" * 160)


def _generate_single_case_summary(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(["budget", "percent", "strategy", "include_danger"]).agg([
        pl.len().alias("instances"),
        (pl.col("nr_cost") == pl.col("mip_cost")).sum().alias("optimal"),
        (pl.col("nr_cost") != pl.col("mip_cost")).sum().alias("suboptimal"),
        pl.mean("ratio").alias("avg_ratio"),
        pl.max("ratio").alias("worst_ratio"),
        pl.mean("speedup").alias("avg_speedup"),
    ])


def _generate_summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .group_by(["strategy", "budget", "include_danger"])
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


def _generate_summary_total(df: pl.DataFrame) -> pl.DataFrame:
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
        .sort("budget")
    )

    return tmp_df.select("strategy", "budget", pl.all().exclude(["strategy", "budget"]))


def _generate_danger_aggregate_summary(summary_df: pl.DataFrame) -> pl.DataFrame:
    longest = max(len(str(v)) for v in summary_df["strategy"].to_list())
    col_widths = [len(c) for c in summary_df.columns]
    col_widths[0] = max(longest, col_widths[0])

    return (
        summary_df
        .group_by(["avg_dangers"])
        .agg([
            pl.lit("-" + " " * (col_widths[0] - len("-"))).alias("strategy"),
            pl.lit("-").alias("budget"),
            pl.col("instances").sum(),
            pl.col("optimal").sum(),
            pl.col("suboptimal").sum(),
            pl.col("optimal_percent").mean(),
            pl.mean("avg_terminals").alias("avg_terminals"),
            pl.mean("avg_roots").alias("avg_roots"),
            pl.mean("avg_workers").alias("avg_workers"),
            # pl.mean("avg_dangers").alias("avg_dangers"),
            pl.col("avg_ratio").mean(),
            pl.col("worst_ratio").max(),
            pl.col("avg_speedup").mean(),
        ])
        .sort(["strategy", "avg_dangers"], descending=False)
        .select(
            "strategy",
            "budget",
            "instances",
            "optimal",
            "suboptimal",
            "optimal_percent",
            "avg_terminals",
            "avg_roots",
            "avg_workers",
            "avg_dangers",
            "avg_ratio",
            "worst_ratio",
            "avg_speedup",
        )
    )


def _generate_suboptimal_breakdown(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .group_by(["strategy", "include_danger"])
        .agg([
            pl.len().alias("instances"),
            pl.mean("ratio").alias("avg_ratio"),
            pl.max("ratio").alias("worst_ratio"),
            pl.mean("speedup").alias("avg_speedup"),
            pl.min("budget").alias("first_budget"),
            pl
            .col("ratio")
            .filter(pl.col("budget") == pl.min("budget"))
            .mean()
            .alias("ratio_at_first_budget"),
            pl.max("budget").alias("last_budget"),
            pl.col("ratio").filter(pl.col("budget") == pl.max("budget")).mean().alias("ratio_at_last_budget"),
        ])
        .sort(["avg_ratio"], descending=True)
    )


def _generate_suboptimal_by_danger(suboptimal_df: pl.DataFrame) -> pl.DataFrame:
    return (
        suboptimal_df
        .group_by("include_danger")
        .agg([
            pl.col("instances").sum(),
            pl.col("avg_ratio").mean(),
            pl.col("worst_ratio").max(),
            pl.col("avg_speedup").mean(),
        ])
        .sort("avg_ratio", descending=True)
    )


def _generate_suboptimal_by_danger_total(suboptimal_by_danger_df: pl.DataFrame) -> pl.DataFrame:
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
        pl.col(c).cast(pl.Float64).round(SUMMARY_FLOAT_PRECISION).cast(pl.String).alias(c)
        for c in total.select(pl.selectors.numeric()).columns
    ]).with_columns(pl.all().cast(pl.String))

    padded = []
    for val, w in zip(total.row(0), col_widths):
        pad = w - len(val)
        padded.append(val + (" " * pad if pad > 0 else ""))
    return pl.DataFrame([padded], schema=total.schema, orient="row")


def _generate_worst_suboptimal_summary(suboptimal_df: pl.DataFrame) -> pl.DataFrame:
    return (
        suboptimal_df
        .with_columns(
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


def _print_summary(df: pl.DataFrame) -> None:
    with pl.Config(
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
