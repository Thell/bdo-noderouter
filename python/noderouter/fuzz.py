# fuzz.py

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from enum import Enum, IntEnum, auto

import polars as pl
from loguru import logger

import api_data_store as ds
import api_polars_summaries as ps
from api_common import MAX_BUDGET, set_logger
from optimizer_mip import optimize_with_terminals as mip_optimize
from optimizer_nr import optimize_with_terminals as nr_optimize
from orchestrator import execute_plan
from orchestrator_pairing_strategy import MAX_LEN_PAIRING_STRATEGY, PairingStrategy
from orchestrator_types import Instance, Plan, SeedType

SUMMARY_FLOAT_PRECISION = 3
WORST_SUBOPTIMAL_REPORTING_COUNT = 50

SKIPPED_SEEDS_HEX = {}


class ShutdownLevel(IntEnum):
    NONE = 0
    BUDGET = 1  # Stop after current budget block (coverage band)
    STRATEGY = 2  # Stop after current strategy
    SAMPLE = 3  # Stop after current test
    IMMEDIATE = 4  # Exit immediately


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

    def update(self, row: _FuzzInstanceMetrics) -> None:
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
            # budget change implies we should also flush strategy if not already
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
            # dual report for both danger modes
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

    def _reset_strategy(self) -> None:
        self.strategy_mip = 0.0
        self.strategy_nr = 0.0
        self.strategy_n = 0

    def _reset_budget(self) -> None:
        self.budget_mip = 0.0
        self.budget_nr = 0.0
        self.budget_n = 0


FUZZ_TIMER = TimingAccumulator()


LOG_LEVELS = ["SUCCESS", "INFO", "DEBUG", "TRACE"]
current_log_level_index = 1

_shutdown_level = ShutdownLevel.NONE
_shutdown_state_index = 0

# Pause-after-test control (set by UI, observed by sample loop)
_pause_after_test = False
_paused = False
_pause_condition = threading.Condition()

SHUTDOWN_STATES = [
    (ShutdownLevel.BUDGET, "Graceful: Budget", "orange"),
    (ShutdownLevel.STRATEGY, "Graceful: Strategy", "darkorange"),
    (ShutdownLevel.SAMPLE, "Graceful: Test", "orangered"),
    (ShutdownLevel.IMMEDIATE, "Immediate: Exit", "red"),
]


def _create_ui():
    root = tk.Tk()
    root.title("Test Control")
    root.geometry("250x180")
    root.attributes("-topmost", True)

    # --- 1. SHUTDOWN BUTTON LOGIC ---
    def on_shutdown_click():
        global _shutdown_level, _shutdown_state_index

        _shutdown_state_index += 1

        if _shutdown_state_index > len(SHUTDOWN_STATES):
            print("\nHard exit — terminating immediately.", file=sys.stderr)
            os._exit(1)

        level, _text, _bg = SHUTDOWN_STATES[_shutdown_state_index - 1]
        _shutdown_level = level
        print(f"\nShutdown requested — finishing current {level.name.lower()}...", file=sys.stderr)

        if _shutdown_state_index < len(SHUTDOWN_STATES):
            _, next_text, next_bg = SHUTDOWN_STATES[_shutdown_state_index]
            shutdown_btn.config(text=next_text, bg=next_bg)
        else:
            shutdown_btn.config(text="FORCE TERMINATE", bg="red")

    _, initial_text, initial_bg = SHUTDOWN_STATES[0]
    shutdown_btn = tk.Button(root, text=initial_text, command=on_shutdown_click, bg=initial_bg)
    shutdown_btn.pack(expand=True, fill="both", padx=10, pady=(10, 5))

    # --- 2. PAUSE AFTER TEST BUTTON LOGIC ---
    def on_pause_click():
        global _pause_after_test, _paused

        with _pause_condition:
            if _paused:
                # Currently paused → resume
                _paused = False
                _pause_after_test = False
                pause_btn.config(text="Pause After Test", bg="lightblue")
                print("\n[CONTROL] Resumed — continuing tests...", file=sys.stderr)
                _pause_condition.notify_all()
            else:
                # Not paused → request pause after current test
                _pause_after_test = True
                pause_btn.config(text="Pause Requested...", bg="gold")
                print(
                    "\n[CONTROL] Pause after current test requested — will pause when test finishes...",
                    file=sys.stderr,
                )

    pause_btn = tk.Button(root, text="Pause After Test", command=on_pause_click, bg="lightblue")
    pause_btn.pack(expand=True, fill="both", padx=10, pady=5)

    # Keep a reference so the sample loop can update the button when it actually pauses
    root.pause_btn = pause_btn  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]

    # --- 3. LOG LEVEL TOGGLE LOGIC ---
    def on_log_click():
        global current_log_level_index
        current_log_level_index = (current_log_level_index + 1) % len(LOG_LEVELS)
        new_level = LOG_LEVELS[current_log_level_index]

        log_btn.config(text=f"Log Level: {new_level}")

        # Keep format configuration synchronised by pulling the original if available
        try:
            cfg = ds.get_config("config")
            log_format = cfg.get("logger", {}).get("format", "<level>{message}</level>")
        except Exception:  # noqa: BLE001
            log_format = "<level>{message}</level>"

        set_logger({"logger": {"level": new_level, "format": log_format}})
        print(f"[CONTROL] Loguru runtime level updated to: {new_level}", file=sys.stdout)

    # Use the globally resolved index to display the true initialization level
    initial_text = f"Log Level: {LOG_LEVELS[current_log_level_index]}"
    log_btn = tk.Button(root, text=initial_text, command=on_log_click, bg="lightgray")
    log_btn.pack(expand=True, fill="both", padx=10, pady=(5, 10))

    # Expose root so the main thread can safely update the pause button when pausing
    global _ui_root
    _ui_root = root

    root.mainloop()


_ui_root: tk.Tk | None = None


def _set_paused_ui_state():
    """Update the pause button to the 'Paused / click to Resume' state (must run on UI thread)."""
    if _ui_root is None:
        return
    try:
        btn = getattr(_ui_root, "pause_btn", None)
        if btn is not None:
            btn.config(text="▶ RESUME", bg="limegreen")
    except Exception:  # noqa: BLE001  # ruff: ignore[try-except-pass]
        pass


def _wait_if_pause_requested() -> None:
    """
    If a pause-after-test was requested, enter a paused state and block until the user
    clicks Resume. Safe to call from the main (fuzzer) thread.
    """
    global _pause_after_test, _paused

    with _pause_condition:
        if not _pause_after_test:
            return

        _paused = True
        _pause_after_test = False
        print("\n[CONTROL] Paused after test — click RESUME to continue...", file=sys.stderr)

        # Schedule UI update on the Tk thread
        if _ui_root is not None:
            try:
                _ui_root.after(0, _set_paused_ui_state)
            except Exception:  # noqa: BLE001  # ruff: ignore[try-except-pass]
                pass

        while _paused:
            # Wait with a short timeout so we can still react to Immediate shutdown
            _pause_condition.wait(timeout=0.5)
            if _shutdown_level >= ShutdownLevel.IMMEDIATE:
                _paused = False
                break


def _install_shutdown_handler():
    global current_log_level_index

    # 1. Inspect the live configuration dict before launching the UI thread
    try:
        startup_config = ds.get_config("config")
        startup_level = startup_config.get("logger", {}).get("level", "INFO").upper()

        # 2. Match the string (e.g. "DEBUG") to our array index. Fallback safely if unknown string.
        if startup_level in LOG_LEVELS:
            current_log_level_index = LOG_LEVELS.index(startup_level)

    except Exception as e:  # noqa: BLE001
        print(
            f"[CONTROL WARNING] Failed to read startup config level: {e}. Defaulting index to INFO.",
            file=sys.stderr,
        )

    # 3. Safely start the UI thread with the correct initial state loaded
    t = threading.Thread(target=_create_ui, daemon=True)
    t.start()


class _FuzzInstanceMetrics(dict):
    """Factory for producing Polars rows from test incidents."""

    def __init__(
        self,
        seed: SeedType,
        budget: int,
        percent: int,
        include_danger: bool,
        strategy: PairingStrategy,
        nr_instance: Instance,
        mip_instance: Instance,
        i: int,
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
        if ratio < 1.0:
            logger.error(
                f"NodeRouter should never have lower cost than MIP! {seed} => {budget}:{strategy}:{i}"
            )
            raise ValueError(
                f"NodeRouter should never have lower cost than MIP! {seed} => {budget}:{strategy}:{i}"
            )

        if speedup < 1.0:
            logger.warning(f"NodeRouter should always be faster than MIP! {seed} -> {budget}:{strategy}:{i}")

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
            f"MIP {self['mip_cost']:<3} ({self['mip_duration']:6.3f}s) "
            f"NR {self['nr_cost']:<3} ({self['nr_duration']:6.3f}s) "
            f"ratio: {self['ratio']:5.3f} ({self['speedup']:7.1f}x)"
        )


# MARK: Main Fuzzer
def _make_seed(budget: int, strategy: PairingStrategy, i: int) -> SeedType:
    """Produce a deterministic seed for a given sample."""
    # NOTE: For reproducibility purposes we use deterministic seeds.
    # This ensures that each sample's random terminals are not a 'core' of future budgets.
    # This is not tied to the danger inclusion for a pairing strategy.
    # The solver's methodology for handling dangers can potentially 'break' an otherwise
    # optimally solved problem and by keeping them fixed for danger inclusive and exclusive
    # samples we ensure that such cases are identifiable.
    return hashlib.sha256(f"{budget}:{strategy}:{i}".encode()).hexdigest()[:7]


def find_and_remove_cache_by_seed(seed):
    import shutil
    from pathlib import Path

    cache_dir = Path(".cache/joblib/orchestrator/_execute_plan_cached/")
    folders_to_delete = []

    for metadata_path in cache_dir.rglob("metadata.json"):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                # Read as raw text instead of json...
                content = f.read()
                if seed in content:
                    hash_folder = metadata_path.parent
                    folders_to_delete.append(hash_folder)
                    print(f" Found target seed in: {metadata_path}")
                    print(f"Hash Folder: {hash_folder.name}")

        except Exception as e:  # noqa: BLE001
            print(f"Could not read {metadata_path}: {e}")

    if folders_to_delete:
        for hash_folder in folders_to_delete:
            shutil.rmtree(hash_folder)
            print(f" Successfully removed: {hash_folder}\n")
    else:
        print(f"No cache entries found matching seed '{seed}'.")


def _run_single_config(
    strategies: list[PairingStrategy], samples: int, budget: int, include_danger: bool
) -> pl.DataFrame:
    """
    Run fuzz tests for a given budget across all pairing strategies.
    NOTE: Percent is ignored upon _input_ for PairingStrategy.optimized and populated upon output.
    """
    config = ds.get_config("config")

    all_cases_df = pl.DataFrame()

    assert budget <= MAX_BUDGET
    percent = round(budget / MAX_BUDGET * 100)

    for strategy in strategies:
        if _shutdown_level >= ShutdownLevel.STRATEGY:
            break

        if strategy == PairingStrategy.custom:
            continue

        desc = f"strategy={strategy.value}, budget={budget:3}, danger={'yes' if include_danger else 'no'}"
        logger.info(f"\nStarting: {desc}, samples={samples}")

        case_rows: list[_FuzzInstanceMetrics] = []

        for i in range(samples):
            if _shutdown_level >= ShutdownLevel.SAMPLE:
                break

            seed = _make_seed(budget, strategy, i)
            if seed in SKIPPED_SEEDS_HEX:
                logger.error(f"\nSCIPSTP Error Avoidance: skipping seed {seed}\n")
                continue

            # # Debugging - one-offs
            # target_seed = "7f6ae3f"
            # if seed != target_seed:
            #     continue
            # else:
            #     find_and_remove_cache_by_seed(seed)

            # # fmt: off
            # # target_seeds = ["0181f0b", "7bc6454", "3f9f8ff", "f43d040", "eec0177", "708bad0", "fd367b5", "8ecc41a", "182c9c7", "18ac274", "e936ec6", "7cce804", "446a70a", "19662a1", "067dab4", "d10c799", "ea57e16", "693caf0", "e6b2db3", "9c69237"]
            # target_seeds = ["5ec7724", "cafacf9", "c68ef41"]
            # # fmt: on
            # if seed not in target_seeds:
            #     continue
            # else:
            #     find_and_remove_cache_by_seed(seed)

            logger.success(f"Processing seed: {seed}")

            mip_plan = Plan(mip_optimize, budget, percent, seed, include_danger, strategy, True)
            nr_plan = Plan(nr_optimize, budget, percent, seed, include_danger, strategy, False)

            try:
                mip_instance = execute_plan(mip_plan, config)
                nr_instance = execute_plan(nr_plan, config)

                row = _FuzzInstanceMetrics(
                    seed,
                    budget,
                    percent,
                    include_danger,
                    strategy,
                    nr_instance,
                    mip_instance,
                    i,
                )
                case_rows.append(row)
                FUZZ_TIMER.update(row)

            except Exception as e:  # noqa: BLE001
                logger.error(f"[ERROR]: skipping seed {seed} due to internal error...")
                print(e)
                continue

            log_str = row.generate_log_string(i, samples)
            if (gap := row["nr_cost"] - row["mip_cost"]) > 0:
                logger.opt(colors=True).warning(f"⚠️  <n>{log_str} (gap: +{gap})</>")
            else:
                logger.success(f"✅ {log_str}")

            # Honour "Pause After Test" request (blocks until Resume or Immediate shutdown)
            _wait_if_pause_requested()

            # The MIP optimized strategy should only be executed once since
            # the pairs will always be the same for a given budget.
            if strategy == PairingStrategy.optimized:
                break

        case_df = pl.DataFrame(case_rows)

        if case_df.shape[0] == 0:
            continue

        all_cases_df = all_cases_df.vstack(case_df)

        case_summary = ps._generate_single_case_summary(case_df)
        ps._print_summary(case_summary)

        # Explicit strategy boundary report (covers final strategy / single-strategy budgets)
        if FUZZ_TIMER.strategy_n > 0:
            FUZZ_TIMER.print_summary(ReportLevel.STRATEGY)
            FUZZ_TIMER._reset_strategy()

    return all_cases_df


def fuzzer_main(
    strategies: list[PairingStrategy], samples: int, budgets: list[int] | range, danger_states: list[bool]
) -> None:
    set_logger(ds.get_config("config"))
    # set_logger({"logger": {"level": "ERROR", "format": "<level>{message}</level>"}})
    all_metrics: pl.DataFrame = pl.DataFrame()
    _install_shutdown_handler()

    start_time = time.time()
    try:
        for budget in budgets:
            for danger_state in danger_states:
                if _shutdown_level == ShutdownLevel.IMMEDIATE:
                    raise KeyboardInterrupt

                metrics = _run_single_config(strategies, samples, budget, danger_state)
                if metrics.shape[0] == 0:
                    continue
                all_metrics = all_metrics.vstack(metrics)

                current_elapsed = time.time() - start_time
                total_test_cases = all_metrics.shape[0]
                print(f"=> {total_test_cases} test cases completed in {current_elapsed:.2f}s")

            # Budget boundary (after all danger states for this budget)
            if FUZZ_TIMER.budget_n > 0:
                FUZZ_TIMER.print_summary(ReportLevel.BUDGET)
                FUZZ_TIMER._reset_budget()

            if _shutdown_level == ShutdownLevel.BUDGET:
                break

        FUZZ_TIMER.flush()
        ps._generate_all_cases_summaries(all_metrics)

    except KeyboardInterrupt:
        print("\nShutdown complete — generating summary from accumulated data...")
        FUZZ_TIMER.flush()
        ps._generate_all_cases_summaries(all_metrics)
        sys.exit(0)

    logger.success("Fuzz test suite finished")
    print(f"Total runtime: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    # NOTE: For full fuzzing we would want to include all possible strategies
    strategies = [s for s in PairingStrategy]

    # NOTE: For testing purposes we can use a subset of strategies
    # strategies = [PairingStrategy.optimized, PairingStrategy.random_town]
    # strategies = [PairingStrategy.nearest_town]

    # NOTE: For full fuzzing we should use a subset of budgets since the MIP
    # solver takes a long time and is executed for each strategy within each budget
    # times the number of samples.
    budgets = range(275, 555, 5)

    # NOTE: For testing purposes or limited subsets the range can be increased
    # to include all possible budgets.
    # NOTE: MIP optimal solutions (from EmpireOptimizer) are available for (5, 555, 5).
    # budgets = range(5, 555, 5)

    # NOTE: For normal fuzzing or testing purposes the sample count can be adjusted
    # as desired. The default is 20 to allow for a diverse random selection of pairs.
    samples = 20

    # # For normal fuzzing use [False, True]
    # include_danger = [False, True]
    danger_states = [True]

    fuzzer_main(strategies, samples, budgets, danger_states)
