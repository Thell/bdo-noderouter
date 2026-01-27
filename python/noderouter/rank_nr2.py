# rank_nr.py

#################################################################################
# This is a research script that is only to be run after bench.py scripts have been run.
# The bench.py scripts generate the raw data that is used for ranking the algorithms.
#
# See the notes in bench.py __main__ for details.
#################################################################################

# There are N Algorithms with 4 recorded solution costs for each test.
# - pre_XXX_fwd -> is the pure pre-treatment approximation solution cost
# - XXX_fwd -> is the PBS forward treated approximation solution cost
# - pre_XXX_rev -> is the pure pre-treatment approximation solution cost
# - XXX_rev -> is the PBS reverse treated approximation solution cost
#
# * pre_XXX_fwd and pre_XXX_rev are duplicate values
#
# For example the following: pre_{alg}_fwd, ALG_fwd,pre_{alg}_rev, {alg}_rev are for one algorithm during one test
# The 3 values per algorithm for reporting are: pre_{alg}_fwd, {alg}_fwd, {alg}_rev
#
# The reported columns names should be transformed to: {alg}_raw, {alg}_fwd, {alg}_rev

# CVS Rank columns are:
# pre_{alg}_fwd, {alg}_fwd, pre_{alg}_rev, {alg}_rev, ... for each algorithm

# Each Bench row breaks down the parameters for a single test run.
# All tests run 1 time without danger and 1 time with danger.
# The seed, budget, percent, terminals, roots, workers, dangers are the same for both runs.
# The percent, terminals, roots, workers and dangers are calculated directly from the budget.
# The percent is a test run's coverage percentage of the graph.
# - Terminals are selected to the coverage percentage of the graph. Terminals are then paired with
#   randomly selected rooots, thus the root count can vary within the same coverage percentage.

# CVS Bench columns are:
# seed,budget,strategy,include_danger,percent,terminals,roots,workers,dangers,nr_cost,nr_duration

# We want to report test results in the aggregate across budget (coverage) bands.
# The original budgets used for testing are 5..=550 step by 5, so aggregating by 50 is appropriate.
# That will yield 5..50, 55..100, 105..150, 155..200, 205..250, 255..300, 305..350, 355..400, 405..450, 455..550
# which is 10 coverage bands each with a single band representing 10 test run parameter sets.

# Each test run has 50 samples for each budget for each strategy except 'optimized' which has
# 2 samples (one with danger and one without danger).

# This means that there are (2 + 50 * sample_count) * | budget| test runs for a total of 165,220 test runs.
from matplotlib.pylab import ylim


import polars as pl
import api_data_store as ds
import seaborn as sns
import matplotlib.pyplot as plt

from time import time

start_time = time()

SUMMARY_FLOAT_PRECISION = 1

file_path = ds.path().joinpath("rank_results_extended_bridge.csv")
raw_rank_df = pl.read_csv(file_path)

# Drop duplicate pre‑treatment columns
raw_rank_df = raw_rank_df.drop([
    "pre_gssp_rev",
    "pre_pd_rev",
])

file_path = ds.path().joinpath("bench_results_extended_bridge.csv")
raw_bench_df = pl.read_csv(file_path)

# Number of tests must match
total_rank_rows = raw_rank_df.height
total_bench_rows = raw_bench_df.height
assert total_rank_rows == total_bench_rows, "Number of rank rows does not match number of benchmark rows"


# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
#  MARK: PRINT HELPERS
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
def print_summary(title: str | None, df: pl.DataFrame) -> None:
    with pl.Config(
        set_tbl_formatting="MARKDOWN",
        thousands_separator=True,
        set_float_precision=SUMMARY_FLOAT_PRECISION,
        tbl_cell_numeric_alignment="RIGHT",
        set_fmt_str_lengths=100,
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=-1,
    ):
        if title is not None:
            print(f"\n=== {title} ===")
        print(df)


def print_improvement_by_rank(title: str, df: pl.DataFrame) -> None:
    target_col = df.columns[1]
    display_df = df.with_columns(pl.col(target_col).cast(pl.String))

    with pl.Config(
        thousands_separator=True,
        set_float_precision=SUMMARY_FLOAT_PRECISION,
        tbl_cell_alignment="RIGHT",
        tbl_cell_numeric_alignment="RIGHT",
        set_fmt_str_lengths=100,
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=-1,
    ):
        print(f"\n=== {title} ===")
        print(display_df)


# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# MARK: Core Dataframe (alg_df)
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

# ============================================================
#  UNIFIED RANK + BENCH FRAME
# ============================================================

# ---- Step 1: Join raw_rank_df and raw_bench_df horizontally
unified_raw = pl.concat([raw_rank_df, raw_bench_df], how="horizontal")

# ---- Step 2: Compute coverage_band (50‑wide buckets)
unified_raw = unified_raw.with_columns([(((pl.col("budget") - 5) // 50) * 50 + 5).alias("coverage_band")])

# ============================================================
#  NORMALIZATION — ONE ROW PER ALGORITHM
# ============================================================

# Identify cost columns to unpivot
cost_cols = [
    "pre_gssp_fwd",
    "gssp_fwd",
    "gssp_rev",
    "pre_pd_fwd",
    "pd_fwd",
    "pd_rev",
]

# Unpivot
alg_long = unified_raw.unpivot(
    index=[
        "seed",
        "budget",
        "strategy",
        "include_danger",
        "percent",
        "terminals",
        "roots",
        "workers",
        "dangers",
        "nr_cost",
        "nr_duration",
        "coverage_band",
    ],
    on=cost_cols,
    variable_name="metric",
    value_name="value",
)

# Extract algorithm name
alg_long = alg_long.with_columns([pl.col("metric").str.extract(r"(gssp|pd)").alias("algorithm")])

# Normalize metric type
alg_long = alg_long.with_columns([
    pl
    .when(pl.col("metric").str.starts_with("pre_"))
    .then(pl.lit("raw"))
    .otherwise(pl.col("metric").str.extract(r"(fwd|rev)"))
    .alias("metric_type")
])

# Pivot back to wide
alg_df = alg_long.pivot(
    values="value",
    index=[
        "seed",
        "budget",
        "strategy",
        "include_danger",
        "percent",
        "terminals",
        "roots",
        "workers",
        "dangers",
        "nr_cost",
        "nr_duration",
        "coverage_band",
        "algorithm",
    ],
    on="metric_type",
).rename({"raw": "cost_raw", "fwd": "cost_fwd", "rev": "cost_rev"})

# ============================================================
#  MARK: RANK CALCULATIONS
# ============================================================

GROUP_KEYS = [
    "seed",
    "budget",
    "strategy",
    "include_danger",
    "percent",
    "terminals",
    "roots",
    "workers",
    "dangers",
    "nr_cost",
    "nr_duration",
    "coverage_band",
]

ranked_temp = alg_df.with_columns([
    pl.col("cost_raw").rank("dense", descending=False).over(GROUP_KEYS).alias("pre_rank"),
    pl.col("cost_fwd").rank("dense", descending=False).over(GROUP_KEYS).alias("fwd_rank"),
    pl.col("cost_rev").rank("dense", descending=False).over(GROUP_KEYS).alias("rev_rank"),
])

ranked = ranked_temp.with_columns([
    pl.min_horizontal("fwd_rank", "rev_rank").alias("final_rank"),
    pl.min_horizontal("cost_fwd", "cost_rev").alias("final_cost"),
])

# ============================================================
#  MARK: PER‑BAND PARAMETER AGGREGATION
# ============================================================

band_params = (
    ranked
    .group_by("coverage_band")
    .agg([
        pl.mean("percent").alias("mean_percent"),
        pl.mean("terminals").alias("mean_terminals"),
        pl.mean("roots").alias("mean_roots"),
        pl.mean("workers").alias("mean_workers"),
        pl.mean("dangers").alias("mean_dangers"),
    ])
    .sort("coverage_band")
)

# ============================================================
#  MARK: Dataframe Transformations
# ============================================================


def placement_table_band(band_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute placement counts and percentages for a single coverage band.
    Uses a seed frame to ensure all algorithm-rank combinations are included, with zeros for missing.
    """
    total_rows = band_df.height
    band = band_df["coverage_band"][0]

    pre = (
        band_df
        .group_by(["algorithm", "pre_rank"])
        .agg(pl.len().cast(pl.Int64).alias("pre_count"))
        .rename({"pre_rank": "rank"})
    )

    fwd = (
        band_df
        .group_by(["algorithm", "fwd_rank"])
        .agg(pl.len().cast(pl.Int64).alias("fwd_count"))
        .rename({"fwd_rank": "rank"})
    )

    rev = (
        band_df
        .group_by(["algorithm", "rev_rank"])
        .agg(pl.len().cast(pl.Int64).alias("rev_count"))
        .rename({"rev_rank": "rank"})
    )

    # Seed to guarantee complete table
    seed = pl.DataFrame({
        "algorithm": ["gssp", "pd"] * 3,
        "rank": [1, 1, 1, 2, 2, 2],
    })

    combined = seed.join(pre, on=["algorithm", "rank"], how="left")
    combined = combined.join(fwd, on=["algorithm", "rank"], how="left")
    combined = combined.join(rev, on=["algorithm", "rank"], how="left")

    combined = combined.with_columns([
        pl.col("pre_count").fill_null(0),
        pl.col("fwd_count").fill_null(0),
        pl.col("rev_count").fill_null(0),
    ])

    combined = combined.with_columns([
        (pl.col("pre_count") / total_rows * 100).alias("pre_percent"),
        (pl.col("fwd_count") / total_rows * 100).alias("fwd_percent"),
        (pl.col("rev_count") / total_rows * 100).alias("rev_percent"),
    ])

    combined = combined.with_columns([
        (pl.col("fwd_count") - pl.col("pre_count")).alias("Δfwd_count"),
        (pl.col("fwd_percent") - pl.col("pre_percent")).alias("Δfwd_percent"),
        (pl.col("rev_count") - pl.col("pre_count")).alias("Δrev_count"),
        (pl.col("rev_percent") - pl.col("pre_percent")).alias("Δrev_percent"),
        pl.lit(band).alias("coverage_band"),
    ])

    return combined.sort(["coverage_band", "algorithm", "rank"])


def rank_composition_band(band_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute rank composition for pre/fwd/rev/final within a single band.
    Loops per rank to avoid alias conflicts and ensure scalar mask values.
    """
    band = band_df["coverage_band"][0]
    total_tests = band_df.select(GROUP_KEYS).n_unique()

    MASK_LABELS = {
        1: "GSSP",
        2: "PD",
        3: "GSSP+PD",
        0: "NONE",
    }

    rows = []

    for r in [1, 2, 3]:
        masks = (
            band_df
            .with_columns([
                pl
                .when((pl.col("algorithm") == "gssp") & (pl.col("pre_rank") == r))
                .then(1)
                .when((pl.col("algorithm") == "pd") & (pl.col("pre_rank") == r))
                .then(2)
                .otherwise(0)
                .cast(pl.Int32)
                .alias("pre_bit"),
                pl
                .when((pl.col("algorithm") == "gssp") & (pl.col("fwd_rank") == r))
                .then(1)
                .when((pl.col("algorithm") == "pd") & (pl.col("fwd_rank") == r))
                .then(2)
                .otherwise(0)
                .cast(pl.Int32)
                .alias("fwd_bit"),
                pl
                .when((pl.col("algorithm") == "gssp") & (pl.col("rev_rank") == r))
                .then(1)
                .when((pl.col("algorithm") == "pd") & (pl.col("rev_rank") == r))
                .then(2)
                .otherwise(0)
                .cast(pl.Int32)
                .alias("rev_bit"),
                pl
                .when((pl.col("algorithm") == "gssp") & (pl.col("final_rank") == r))
                .then(1)
                .when((pl.col("algorithm") == "pd") & (pl.col("final_rank") == r))
                .then(2)
                .otherwise(0)
                .cast(pl.Int32)
                .alias("final_bit"),
            ])
            .group_by(GROUP_KEYS)
            .agg([
                pl.col("pre_bit").sum().alias("pre_mask"),
                pl.col("fwd_bit").sum().alias("fwd_mask"),
                pl.col("rev_bit").sum().alias("rev_mask"),
                pl.col("final_bit").sum().alias("final_mask"),
            ])
        )

        for mask_value in MASK_LABELS:
            if mask_value == 0:  # omit NONE
                continue
            label = MASK_LABELS[mask_value]
            pre_count = (masks["pre_mask"] == mask_value).sum()
            fwd_count = (masks["fwd_mask"] == mask_value).sum()
            rev_count = (masks["rev_mask"] == mask_value).sum()
            final_count = (masks["final_mask"] == mask_value).sum()

            rows.append((
                band,
                r,
                label,
                int(pre_count),
                pre_count / total_tests * 100.0,
                int(fwd_count),
                fwd_count / total_tests * 100.0,
                int(rev_count),
                rev_count / total_tests * 100.0,
                int(final_count),
                final_count / total_tests * 100.0,
            ))

    df = pl.DataFrame(
        rows,
        schema=[
            "coverage_band",
            "rank",
            "composition",
            "pre_count",
            "pre_percent",
            "fwd_count",
            "fwd_percent",
            "rev_count",
            "rev_percent",
            "final_count",
            "final_percent",
        ],
        orient="row",
    )

    return df.sort(["coverage_band", "rank", "final_count"], descending=[False, False, True])


def movement_table_band(band_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute movement (UP / DOWN / UNCHANGED) for each algorithm within a single coverage band.
    """
    band = band_df["coverage_band"][0]

    rows = []

    for alg in ["gssp", "pd"]:
        df_alg = band_df.filter(pl.col("algorithm") == alg)

        up = df_alg.filter(pl.col("final_rank") < pl.col("pre_rank")).height
        down = df_alg.filter(pl.col("final_rank") > pl.col("pre_rank")).height
        same = df_alg.filter(pl.col("final_rank") == pl.col("pre_rank")).height

        total = up + down + same

        rows.append((band, alg.upper(), "UP", up, up / total * 100 if total else 0.0))
        rows.append((band, alg.upper(), "DOWN", down, down / total * 100 if total else 0.0))
        rows.append((band, alg.upper(), "UNCHANGED", same, same / total * 100 if total else 0.0))

    return pl.DataFrame(
        rows,
        schema=[
            "coverage_band",
            "algorithm",
            "movement",
            "count",
            "percent",
        ],
        orient="row",
    ).sort(["coverage_band", "algorithm", "movement"])


def placement_table_strategy(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute placement for a single strategy and band, with seed for completeness.
    """
    total_rows = df.height
    band = df["coverage_band"][0]
    strat = df["strategy"][0]

    seed = pl.DataFrame(
        [(alg, r) for alg in ["gssp", "pd"] for r in [1, 2]],
        schema=["algorithm", "rank"],
        orient="row",
    )

    pre = (
        df
        .group_by(["algorithm", "pre_rank"])
        .agg(pl.len().cast(pl.Int64).alias("pre_count"))
        .rename({"pre_rank": "rank"})
    )

    final = (
        df
        .group_by(["algorithm", "final_rank"])
        .agg(pl.len().cast(pl.Int64).alias("final_count"))
        .rename({"final_rank": "rank"})
    )

    combined = seed.join(pre, on=["algorithm", "rank"], how="left")
    combined = combined.join(final, on=["algorithm", "rank"], how="left")

    combined = combined.with_columns([
        pl.col("pre_count").fill_null(0),
        pl.col("final_count").fill_null(0),
    ])

    combined = combined.with_columns([
        (pl.col("pre_count") / total_rows * 100).alias("pre_percent"),
        (pl.col("final_count") / total_rows * 100).alias("final_percent"),
    ])

    combined = combined.with_columns([
        (pl.col("final_count") - pl.col("pre_count")).alias("Δcount"),
        (pl.col("final_percent") - pl.col("pre_percent")).alias("Δpercent"),
        pl.lit(band).alias("coverage_band"),
        pl.lit(strat).alias("strategy"),
    ])

    return combined.sort(["coverage_band", "strategy", "algorithm", "rank"])


def placement_table_strategy_total(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate placement across all bands for a single strategy, with seed.
    """
    total_rows = df.height
    strat = df["strategy"][0]

    seed = pl.DataFrame(
        [(alg, r) for alg in ["gssp", "pd"] for r in [1, 2]],
        schema=["algorithm", "rank"],
        orient="row",
    )

    pre = (
        df
        .group_by(["algorithm", "pre_rank"])
        .agg(pl.len().cast(pl.Int64).alias("pre_count"))
        .rename({"pre_rank": "rank"})
    )

    final = (
        df
        .group_by(["algorithm", "final_rank"])
        .agg(pl.len().cast(pl.Int64).alias("final_count"))
        .rename({"final_rank": "rank"})
    )

    combined = seed.join(pre, on=["algorithm", "rank"], how="left")
    combined = combined.join(final, on=["algorithm", "rank"], how="left")

    combined = combined.with_columns([
        pl.col("pre_count").fill_null(0),
        pl.col("final_count").fill_null(0),
    ])

    combined = combined.with_columns([
        (pl.col("pre_count") / total_rows * 100).alias("pre_percent"),
        (pl.col("final_count") / total_rows * 100).alias("final_percent"),
    ])

    combined = combined.with_columns([
        (pl.col("final_count") - pl.col("pre_count")).alias("Δcount"),
        (pl.col("final_percent") - pl.col("pre_percent")).alias("Δpercent"),
        pl.lit(strat).alias("strategy"),
    ])

    return combined.sort(["strategy", "algorithm", "rank"])


def strategy_rank1_dominance(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rank-1 dominance for a single strategy and band.
    """
    band = df["coverage_band"][0]
    strat = df["strategy"][0]

    seed = pl.DataFrame(
        [(alg, 1) for alg in ["gssp", "pd"]],
        schema=["algorithm", "rank"],
        orient="row",
    )

    final = (
        df
        .group_by(["algorithm", "final_rank"])
        .agg(pl.len().cast(pl.Int64).alias("final_rank1_count"))
        .rename({"final_rank": "rank"})
    )

    combined = seed.join(final, on=["algorithm", "rank"], how="left").with_columns(
        pl.col("final_rank1_count").fill_null(0)
    )

    combined = combined.with_columns([
        pl.lit(band).alias("coverage_band"),
        pl.lit(strat).alias("strategy"),
    ])

    return combined.select("strategy", "coverage_band", "algorithm", "final_rank1_count").sort([
        "strategy",
        "coverage_band",
        "algorithm",
    ])


def strategy_rank1_dominance_per_budget(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rank-1 dominance for a single strategy and budget.
    """
    budget = df["budget"][0]
    strat = df["strategy"][0]

    seed = pl.DataFrame(
        [(alg, 1) for alg in ["gssp", "pd"]],
        schema=["algorithm", "rank"],
        orient="row",
    )

    final = (
        df
        .group_by(["algorithm", "final_rank"])
        .agg(pl.len().cast(pl.Int64).alias("final_rank1_count"))
        .rename({"final_rank": "rank"})
    )

    combined = seed.join(final, on=["algorithm", "rank"], how="left").with_columns(
        pl.col("final_rank1_count").fill_null(0)
    )

    combined = combined.with_columns([
        pl.lit(budget).alias("budget"),
        pl.lit(strat).alias("strategy"),
    ])

    return combined.select("strategy", "budget", "algorithm", "final_rank1_count")


def supplanted_winner_counts(ranked: pl.DataFrame, all_improved: int) -> pl.DataFrame:
    """
    Count cases where a non-rank1 algorithm becomes the sole rank1 winner after PBS.
    Percentages relative to all_improved.
    """
    winners = ranked.group_by(GROUP_KEYS).agg([
        pl.col("algorithm").filter(pl.col("pre_rank") == 1).alias("pre_winners"),
        pl.col("algorithm").filter(pl.col("final_rank") == 1).alias("final_winners"),
    ])

    winners = winners.with_columns((pl.col("final_winners").list.len() == 1).alias("final_solo"))

    winners = winners.with_columns(pl.col("final_winners").list.first().alias("new_winner"))

    winners = winners.with_columns(
        (~pl.col("pre_winners").list.contains(pl.col("new_winner"))).alias("was_not_rank1_pre")
    )

    winners = winners.with_columns((pl.col("final_solo") & pl.col("was_not_rank1_pre")).alias("supplanted"))

    sup_counts = (
        winners
        .filter(pl.col("supplanted"))
        .group_by("new_winner")
        .agg(pl.len().alias("count"))
        .rename({"new_winner": "algorithm"})
        .sort("algorithm")
    )

    sup_counts = sup_counts.with_columns((pl.col("count") / all_improved * 100).alias("percent"))

    total_row = sup_counts.select(
        pl.lit("TOTAL").alias("algorithm"),
        pl.col("count").sum().alias("count"),
        (pl.col("count").sum() / all_improved * 100).alias("percent"),
    )

    return pl.concat([sup_counts, total_row], how="vertical")


# ============================================================
#  MARK: Dataframe Filters
# ============================================================


def select_placements(all_placements: pl.DataFrame, bands: list[int]) -> pl.DataFrame:
    return all_placements.filter(pl.col("coverage_band").is_in(bands)).sort([
        "coverage_band",
        "algorithm",
        "rank",
    ])


def select_compositions(all_compositions: pl.DataFrame, bands: list[int]) -> pl.DataFrame:
    return all_compositions.filter(pl.col("coverage_band").is_in(bands)).sort(
        ["coverage_band", "rank", "final_count"],
        descending=[False, False, True],
    )


def select_movements(all_movements: pl.DataFrame, bands: list[int]) -> pl.DataFrame:
    return all_movements.filter(pl.col("coverage_band").is_in(bands)).sort([
        "coverage_band",
        "algorithm",
        "movement",
    ])


def select_strategy_placements(
    all_strategy_placements: pl.DataFrame, bands: list[int], strategies: list[str]
) -> pl.DataFrame:
    return all_strategy_placements.filter(
        pl.col("coverage_band").is_in(bands) & pl.col("strategy").is_in(strategies)
    ).sort(["coverage_band", "strategy", "algorithm", "rank"])


# ============================================================
#  MARK: UNIFIED FRAMES
# ============================================================

coverage_bands = ranked.select("coverage_band").unique().sort("coverage_band")["coverage_band"]

all_placements = (
    ranked
    .group_by("coverage_band", maintain_order=True)
    .map_groups(placement_table_band)
    .sort(["coverage_band", "algorithm", "rank"])
)

all_compositions = (
    ranked
    .group_by("coverage_band", maintain_order=True)
    .map_groups(rank_composition_band)
    .sort(["coverage_band", "rank", "final_count"], descending=[False, False, True])
)

all_movements = (
    ranked
    .group_by("coverage_band", maintain_order=True)
    .map_groups(movement_table_band)
    .sort(["coverage_band", "algorithm", "movement"])
)

strategies = ranked["strategy"].unique().sort()

all_strategy_placements = pl.concat(
    [
        placement_table_strategy(
            ranked.filter((pl.col("coverage_band") == band) & (pl.col("strategy") == strat))
        )
        for band in coverage_bands
        for strat in strategies
    ],
    how="vertical",
).sort(["coverage_band", "strategy", "algorithm", "rank"])


all_strategy_totals = (
    ranked
    .group_by("strategy", maintain_order=True)
    .map_groups(placement_table_strategy_total)
    .sort(["strategy", "algorithm", "rank"])
)

strategy_rank1_all = pl.concat(
    [
        strategy_rank1_dominance(
            ranked.filter((pl.col("coverage_band") == band) & (pl.col("strategy") == strat))
        )
        for strat in strategies
        for band in coverage_bands
    ],
    how="vertical",
).sort(["strategy", "coverage_band", "algorithm"])

strategy_band_totals = (
    ranked
    .select(GROUP_KEYS)
    .unique()
    .group_by(["strategy", "coverage_band"])
    .agg(pl.len().alias("total_tests"))
)

strategy_rank1_all = (
    strategy_rank1_all
    .join(strategy_band_totals, on=["strategy", "coverage_band"], how="left")
    .with_columns([(pl.col("final_rank1_count") / pl.col("total_tests") * 100).alias("final_rank1_percent")])
    .sort(["strategy", "coverage_band", "algorithm"])
)

strategy_rank1_per_budget = pl.concat(
    [
        strategy_rank1_dominance_per_budget(
            ranked.filter((pl.col("budget") == b) & (pl.col("strategy") == strat))
        )
        for strat in strategies
        for b in ranked["budget"].unique().sort()
    ],
    how="vertical",
).sort(["strategy", "budget", "algorithm"])

budget_totals = (
    ranked.select(GROUP_KEYS).unique().group_by(["strategy", "budget"]).agg(pl.len().alias("total_tests"))
)

strategy_rank1_per_budget = strategy_rank1_per_budget.join(
    budget_totals, on=["strategy", "budget"], how="left"
).with_columns((pl.col("final_rank1_count") / pl.col("total_tests") * 100).alias("final_rank1_percent"))


# ============================================================
#  MARK: FINAL RANK‑1 COUNTS
# ============================================================

final_rank1 = ranked.group_by(GROUP_KEYS).agg(
    pl.col("algorithm").filter(pl.col("final_rank") == 1).alias("winners")
)


final_rank1_exploded = final_rank1.explode("winners")

total_tests = final_rank1.height

final_rank1_totals = (
    final_rank1_exploded
    .group_by("winners")
    .agg(pl.len().alias("count"))
    .rename({"winners": "algorithm"})
    .with_columns([(pl.col("count") / total_tests * 100).alias("percent")])
    .sort("count", descending=True)
)


# ============================================================
#  MARK: MOVEMENT COUNTS
# ============================================================
def total_movements(all_movements: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate movement counts across all coverage bands.
    Produces a long-form totals table:
        algorithm | movement | total_count | total_percent
    """

    # Step 1: Sum counts per algorithm × movement
    totals = all_movements.group_by(["algorithm", "movement"]).agg(pl.col("count").sum().alias("total_count"))

    # Step 2: Compute percentages per algorithm
    totals = (
        totals
        .with_columns([
            pl.col("total_count").sum().over("algorithm").alias("algo_total"),
        ])
        .with_columns([(pl.col("total_count") / pl.col("algo_total") * 100).alias("total_percent")])
        .drop("algo_total")
    )

    # Step 3: Explicit movement ordering (UP → DOWN → UNCHANGED)
    totals = totals.with_columns(
        pl
        .when(pl.col("movement") == "UP")
        .then(0)
        .when(pl.col("movement") == "DOWN")
        .then(1)
        .otherwise(2)
        .alias("movement_order")
    )

    # Step 4: Sort by algorithm, then movement_order
    totals = totals.sort(["algorithm", "movement_order"]).drop("movement_order")

    return totals


# ============================================================
#  MARK: PBS IMPROVEMENT COUNTS
# ============================================================

improved = ranked.with_columns([
    (pl.col("final_cost") < pl.col("cost_raw")).alias("improved_flag"),
])

improved = improved.with_columns([
    ((pl.col("algorithm") == "gssp") & pl.col("improved_flag")).alias("gssp_imp"),
    ((pl.col("algorithm") == "pd") & pl.col("improved_flag")).alias("pd_imp"),
])

improved = improved.with_columns([
    (pl.col("cost_raw") - pl.col("final_cost")).clip(lower_bound=0).alias("delta"),
])

# Improvement by pre-rank
imp_by_rank_rows = []

for alg in ["gssp", "pd"]:
    alg_df = improved.filter(pl.col("algorithm") == alg)
    for r in [1, 2, 3]:
        subset = alg_df.filter(pl.col("pre_rank") == r)
        cases = subset.height
        imp_count = subset.filter(pl.col(f"{alg}_imp")).height
        pct = (imp_count / cases * 100) if cases > 0 else 0.0
        imp_by_rank_rows.append((alg.upper(), r, cases, imp_count, pct))

imp_by_rank = pl.DataFrame(
    imp_by_rank_rows,
    schema=["algorithm", "pre_rank", "cases", "improved_count", "improved_pct"],
    orient="row",
)

imp_by_rank = imp_by_rank.with_columns(pl.col("pre_rank").cast(pl.String))

max_len = imp_by_rank["pre_rank"].str.len_chars().max()
assert isinstance(max_len, int)
imp_by_rank = imp_by_rank.with_columns(pl.col("pre_rank").str.pad_start(max_len, " "))

# Total
all_improved = improved.filter(pl.col("improved_flag")).height

total_alg_instances = ranked.height
all_pct = all_improved / total_alg_instances * 100

total_row = pl.DataFrame([
    {
        "algorithm": "TOTAL",
        "pre_rank": " " * max_len,
        "cases": total_alg_instances,
        "improved_count": all_improved,
        "improved_pct": all_pct,
    }
])

combined_imp = imp_by_rank.vstack(total_row)


# Magnitudes
magn_rows = []

for alg in ["gssp", "pd"]:
    subset = improved.filter((pl.col("algorithm") == alg) & pl.col(f"{alg}_imp"))

    if subset.is_empty():
        magn_rows.append((alg.upper(), 0, 0, 0, 0, 0, 0))
        continue

    deltas = subset["delta"]
    pre_costs = subset["cost_raw"]

    rel = (deltas / pre_costs * 100).drop_nulls()

    magn_rows.append((
        alg.upper(),
        deltas.len(),
        deltas.quantile(0.25),
        deltas.median(),
        deltas.quantile(0.75),
        deltas.max(),
        rel.quantile(0.25),
        rel.median(),
        rel.quantile(0.75),
        rel.max(),
    ))

all_subset = improved.filter(pl.col("improved_flag"))
all_deltas = all_subset["delta"]
all_pre = all_subset["cost_raw"]
all_rel = (all_deltas / all_pre * 100).drop_nulls()

magn_rows.append((
    "ALL",
    all_deltas.len(),
    all_deltas.quantile(0.25),
    all_deltas.median(),
    all_deltas.quantile(0.75),
    all_deltas.max(),
    all_rel.quantile(0.25),
    all_rel.median(),
    all_rel.quantile(0.75),
    all_rel.max(),
))

magnitudes = pl.DataFrame(
    magn_rows,
    schema=[
        "algorithm",
        "improved_count",
        "abs_p25",
        "abs_p50",
        "abs_p75",
        "abs_max",
        "rel_p25_pct",
        "rel_p50_pct",
        "rel_p75_pct",
        "rel_max_pct",
    ],
    orient="row",
)


# ============================================================
#  MARK: REPORTING
# ============================================================

reporting_bands = [105, 205, 305, 405, 505]

print_summary("COVERAGE BAND PARAMETER SUMMARY", band_params)

# Used in both placement and composition reporting
rank1_composition_table = all_compositions.filter(pl.col("rank") == 1).sort(
    ["coverage_band", "final_count"], descending=[False, True]
)

# ---- Placement Reporting
rank1_algo_totals = (
    rank1_composition_table
    .group_by("composition")
    .agg(pl.col("final_count").sum().alias("total_final_count"))
    .sort("total_final_count", descending=True)
)
filtered_placements = select_placements(all_placements, reporting_bands)
rank1_placement_table = filtered_placements.filter(pl.col("rank") == 1)

print_summary("RANK-1 FINAL TOTALS", final_rank1_totals)
print_summary("RANK-1 FINAL TOTALS PER ALGORITHM", rank1_algo_totals)
print_summary("RANK-1 DOMINANCE PLACEMENT", rank1_placement_table)
print_summary("ALL RANK PRE/POST FWD/REV PBS PLACEMENT", filtered_placements)

# ---- Composition Reporting
filtered_rank1_compositions = select_compositions(rank1_composition_table, reporting_bands)
rank_composition_table = select_compositions(all_compositions, reporting_bands)
rank_composition_table = rank_composition_table.sort(
    ["coverage_band", "rank", "final_count"], descending=[False, False, True]
)
print_summary("RANK-1 DOMINANCE COMPOSITION (TOTALS)", filtered_rank1_compositions)
print_summary("RANK COMPOSITION (FULL)", rank_composition_table)

# ---- Movement Reporting
filtered_movements = select_movements(all_movements, reporting_bands)
movement_totals = total_movements(all_movements)
supplanted = supplanted_winner_counts(ranked, all_improved)

print_summary("MOVEMENT TABLE (PRE → FINAL)", filtered_movements)
print_summary("SUCCESSFUL UNDERDOGS", supplanted)
print_summary("TOTAL MOVEMENT (PRE → FINAL)", movement_totals)

print_improvement_by_rank("PBS IMPROVEMENT RATES BY PRE-RANK", combined_imp)
print_summary("IMPROVEMENT MAGNITUDES (IMPROVED CASES ONLY)", magnitudes)

print_summary("RANK-BY-STRATEGY TOTALS (ALL COVERAGE BANDS)", all_strategy_totals)

print_summary(
    "STRATEGY RANK‑1 DOMINANCE",
    strategy_rank1_all.filter(pl.col("algorithm") == "gssp").sort(
        ["coverage_band", "final_rank1_percent"], descending=[False, True]
    ),
)
print_summary(
    "STRATEGY RANK‑1 DOMINANCE",
    strategy_rank1_all.filter(pl.col("algorithm") == "pd").sort(
        ["coverage_band", "final_rank1_percent"], descending=[False, True]
    ),
)

print("Runtime: {:.2f} seconds".format(time() - start_time))

# ============================================================
#  MARK: PLOTTING
# ============================================================

combined_budget_rank1 = strategy_rank1_per_budget.pivot(
    on="algorithm",
    index=["strategy", "budget"],
    values="final_rank1_percent",
)

df_long = combined_budget_rank1.unpivot(
    index=["strategy", "budget"],
    on=["gssp", "pd"],
    variable_name="algorithm",
    value_name="percent",
)

df_long = df_long.with_columns([
    pl.col("percent").rolling_mean(window_size=5).over(["strategy", "algorithm"]).alias("percent")
])

sns.set_theme(style="whitegrid")

g = sns.relplot(
    data=df_long,
    x="budget",
    y="percent",
    hue="algorithm",
    col="strategy",
    col_wrap=4,
    kind="line",
    facet_kws={"sharey": True, "sharex": True},
    height=3,
    aspect=1.2,
    legend="full",
)
g.set(ylim=(30, 100))
g.set_axis_labels("Budget", "Rank 1 Dominance (%)")
g.set_titles("{col_name}")
g.fig.suptitle("Rank 1 Dominance Across All Budgets", y=1.02)

plt.savefig("strategy_transitions_extended_bridge.png", bbox_inches="tight")
