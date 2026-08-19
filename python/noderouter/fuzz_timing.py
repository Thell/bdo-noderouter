# fuzz_timing.py — original MIP/NR solve-time accounting + metrics persistence

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import polars as pl

import api_data_store as ds
from orchestrator_pairing_strategy import MAX_LEN_PAIRING_STRATEGY, PairingStrategy


class ReportLevel(Enum):
    STRATEGY = auto()
    DANGER = auto()
    BUDGET = auto()
    PROCESS = auto()


@dataclass
class TimingAccumulator:
    """Accumulate original MIP/NR solve times (not wall-clock / cache-hit times)."""

    # transient — reset on strategy change
    strategy_mip: float = 0.0
    strategy_nr: float = 0.0
    strategy_n: int = 0

    # cumulative by danger mode (survive across budgets)
    danger_mip: float = 0.0
    danger_nr: float = 0.0
    danger_n: int = 0
    no_danger_mip: float = 0.0
    no_danger_nr: float = 0.0
    no_danger_n: int = 0

    # cumulative for current budget — reset on budget change
    budget_mip: float = 0.0
    budget_nr: float = 0.0
    budget_n: int = 0

    # process lifetime — never reset
    process_mip: float = 0.0
    process_nr: float = 0.0
    process_n: int = 0

    prev_strategy: PairingStrategy | None = None
    prev_danger: bool | None = None
    prev_budget: int | None = None

    def update(self, row: Any) -> None:
        """Ingest one completed sample; print + reset any levels that just ended."""
        strategy = PairingStrategy(row["strategy"]) if isinstance(row["strategy"], str) else row["strategy"]
        danger = bool(row["include_danger"])
        budget = int(row["budget"])
        mip = float(row["mip_duration"])
        nr = float(row["nr_duration"])

        # --- boundary detection (print the level that just finished) ---
        if self.prev_strategy is not None and strategy != self.prev_strategy:
            if self.strategy_n > 0:
                self.print_summary(ReportLevel.STRATEGY)
            self._reset_strategy()

        if self.prev_budget is not None and budget != self.prev_budget:
            if self.strategy_n > 0:
                self.print_summary(ReportLevel.STRATEGY)
                self._reset_strategy()
            if self.budget_n > 0:
                self.print_summary(ReportLevel.BUDGET)
            self._reset_budget()

        # --- accumulate ---
        self.strategy_mip += mip
        self.strategy_nr += nr
        self.strategy_n += 1

        if danger:
            self.danger_mip += mip
            self.danger_nr += nr
            self.danger_n += 1
        else:
            self.no_danger_mip += mip
            self.no_danger_nr += nr
            self.no_danger_n += 1

        self.budget_mip += mip
        self.budget_nr += nr
        self.budget_n += 1

        self.process_mip += mip
        self.process_nr += nr
        self.process_n += 1

        self.prev_strategy = strategy
        self.prev_danger = danger
        self.prev_budget = budget

    def flush(self) -> None:
        """Print any non-empty open levels (call on graceful exit / end of run)."""
        if self.strategy_n > 0:
            self.print_summary(ReportLevel.STRATEGY)
            self._reset_strategy()
        if self.budget_n > 0:
            self.print_summary(ReportLevel.BUDGET)
            self._reset_budget()
        if self.process_n > 0:
            self.print_summary(ReportLevel.PROCESS)

    def print_summary(self, level: ReportLevel) -> None:
        if level == ReportLevel.STRATEGY:
            strat = self.prev_strategy.value if self.prev_strategy else "?"
            danger_s = "yes" if self.prev_danger else "no"
            budget = self.prev_budget if self.prev_budget is not None else "?"
            print(
                f"[TIMING] strategy={strat:<{MAX_LEN_PAIRING_STRATEGY}} "
                f"budget={budget} danger={danger_s} n={self.strategy_n}"
                f"         MIP {self.strategy_mip:10.1f}s   NR {self.strategy_nr:8.1f}s"
            )
        elif level == ReportLevel.DANGER:
            if self.danger_n > 0:
                print(
                    f"[TIMING] danger=yes   n={self.danger_n}"
                    f"         MIP {self.danger_mip:10.1f}s   NR {self.danger_nr:8.1f}s"
                )
            if self.no_danger_n > 0:
                print(
                    f"[TIMING] danger=no    n={self.no_danger_n}"
                    f"         MIP {self.no_danger_mip:10.1f}s   NR {self.no_danger_nr:8.1f}s"
                )
        elif level == ReportLevel.BUDGET:
            budget = self.prev_budget if self.prev_budget is not None else "?"
            print(
                f"[TIMING] budget={budget}  n={self.budget_n}"
                f"         MIP {self.budget_mip:10.1f}s   NR {self.budget_nr:8.1f}s"
            )
        elif level == ReportLevel.PROCESS:
            print(
                f"[TIMING] PROCESS  n={self.process_n}"
                f"         MIP {self.process_mip:10.1f}s   NR {self.process_nr:8.1f}s"
            )
            if self.danger_n > 0 or self.no_danger_n > 0:
                self.print_summary(ReportLevel.DANGER)

    def prime(self, df: pl.DataFrame) -> None:
        """
        Seed cumulative danger/process totals from prior metrics (budgets strictly
        before the current campaign). Does not touch strategy/budget transient slots —
        those are filled by the live/fast-forward path for the current window.
        """
        if df.is_empty():
            return

        for danger_val, mip_attr, nr_attr, n_attr in (
            (True, "danger_mip", "danger_nr", "danger_n"),
            (False, "no_danger_mip", "no_danger_nr", "no_danger_n"),
        ):
            subset = df.filter(pl.col("include_danger") == danger_val)
            if subset.is_empty():
                continue
            setattr(self, mip_attr, getattr(self, mip_attr) + float(subset["mip_duration"].sum()))
            setattr(self, nr_attr, getattr(self, nr_attr) + float(subset["nr_duration"].sum()))
            setattr(self, n_attr, getattr(self, n_attr) + subset.shape[0])

        self.process_mip += float(df["mip_duration"].sum())
        self.process_nr += float(df["nr_duration"].sum())
        self.process_n += df.shape[0]

        print(
            f"[TIMING] PRIMED from prior metrics  n={df.shape[0]}"
            f"         MIP {float(df['mip_duration'].sum()):10.1f}s   "
            f"NR {float(df['nr_duration'].sum()):8.1f}s"
        )
        if self.danger_n > 0 or self.no_danger_n > 0:
            self.print_summary(ReportLevel.DANGER)

    def _reset_strategy(self) -> None:
        self.strategy_mip = 0.0
        self.strategy_nr = 0.0
        self.strategy_n = 0

    def _reset_budget(self) -> None:
        self.budget_mip = 0.0
        self.budget_nr = 0.0
        self.budget_n = 0


FUZZ_TIMER = TimingAccumulator()

# Persisted metrics used to prime the timer across process restarts
_METRICS_PATH_NAME = "all_metrics.parquet"
_METRICS_DEDUP_SUBSET = ["seed", "strategy", "budget", "include_danger"]


def _metrics_path():
    return ds.path() / _METRICS_PATH_NAME


def load_historical_metrics() -> pl.DataFrame:
    path = _metrics_path()
    if not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        print(f"[TIMING] WARNING: failed to load {path}: {e}", file=sys.stderr)
        return pl.DataFrame()


def persist_merged_metrics(historical: pl.DataFrame, this_run: pl.DataFrame) -> None:
    """Merge this run into the historical frame and write to disk."""
    if this_run.is_empty() and historical.is_empty():
        return
    if historical.is_empty():
        merged = this_run
    elif this_run.is_empty():
        merged = historical
    else:
        shared = [c for c in historical.columns if c in this_run.columns]
        merged = pl.concat([historical.select(shared), this_run.select(shared)], how="vertical_relaxed")
    _save_historical_metrics(merged)


def _save_historical_metrics(df: pl.DataFrame) -> None:
    if df.is_empty():
        return
    path = _metrics_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        (
            df
            .unique(subset=_METRICS_DEDUP_SUBSET, keep="last")
            .sort(["budget", "strategy", "include_danger", "seed"])
            .write_parquet(path)
        )
        print(f"[TIMING] Persisted {df.shape[0]} metrics rows → {path}")
    except Exception as e:  # noqa: BLE001
        print(f"[TIMING] WARNING: failed to save {path}: {e}", file=sys.stderr)


def build_primer_df(
    historical: pl.DataFrame,
    strategies: list[PairingStrategy],
    budgets: list[int] | range,
    danger_states: list[bool],
) -> pl.DataFrame:
    """Rows from prior runs that sit strictly before the current campaign start."""
    if historical.is_empty():
        return historical

    budget_list = list(budgets)
    if not budget_list:
        return historical.clear()

    start_budget = min(budget_list)
    strategy_values = [s.value for s in strategies if s != PairingStrategy.custom]

    return historical.filter(
        (pl.col("budget") < start_budget)
        & (pl.col("strategy").is_in(strategy_values))
        & (pl.col("include_danger").is_in(danger_states))
    )
