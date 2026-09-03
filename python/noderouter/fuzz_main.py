# fuzz.py

from __future__ import annotations

import hashlib
import sys
import time
from collections import UserDict

import polars as pl
from loguru import logger

import api_data_store as ds
import api_polars_summaries as ps
from api_common import MAX_BUDGET, set_logger
from fuzz_timing import (
    FUZZ_TIMER,
    ReportLevel,
    build_primer_df,
    load_historical_metrics,
    persist_merged_metrics,
)
from fuzz_ui import (
    ShutdownLevel,
    get_shutdown_level,
    install_control_ui,
    wait_if_pause_requested,
)
from optimizer_mip import optimize_with_terminals as mip_optimize
from optimizer_nr import optimize_with_terminals as nr_optimize
from orchestrator import execute_plan
from orchestrator_pairing_strategy import MAX_LEN_PAIRING_STRATEGY, PairingStrategy
from orchestrator_types import Instance, Plan, SeedType

SKIPPED_SEEDS_HEX = {}


class _FuzzInstanceMetrics(UserDict):
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
        if get_shutdown_level() >= ShutdownLevel.STRATEGY:
            break

        if strategy == PairingStrategy.custom:
            continue

        desc = f"strategy={strategy.value}, budget={budget:3}, danger={'yes' if include_danger else 'no'}"
        logger.info(f"\nStarting: {desc}, samples={samples}")

        case_rows: list[_FuzzInstanceMetrics] = []

        for i in range(samples):
            if get_shutdown_level() >= ShutdownLevel.SAMPLE:
                break

            seed = _make_seed(budget, strategy, i)
            if seed in SKIPPED_SEEDS_HEX:
                logger.error(f"\nSCIPSTP Error Avoidance: skipping seed {seed}\n")
                continue

            # # Debugging - one-offs
            # target_seed = "e4c4d0b"
            # if seed != target_seed:
            #     continue
            # else:
            #     find_and_remove_cache_by_seed(seed)

            # # fmt: off
            # # target_seeds = ["0181f0b", "7bc6454", "3f9f8ff", "f43d040", "eec0177", "708bad0", "fd367b5", "8ecc41a", "182c9c7", "18ac274", "e936ec6", "7cce804", "446a70a", "19662a1", "067dab4", "d10c799", "ea57e16", "693caf0", "e6b2db3", "9c69237"]
            # target_seeds = ["f9f8ae1"]
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

            wait_if_pause_requested()

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
            FUZZ_TIMER.print_summary(ReportLevel.DANGER)
            FUZZ_TIMER._reset_strategy()

    return all_cases_df


def fuzzer_main(
    strategies: list[PairingStrategy], samples: int, budgets: list[int] | range, danger_states: list[bool]
) -> None:
    set_logger(ds.get_config("config"))
    # set_logger({"logger": {"level": "ERROR", "format": "<level>{message}</level>"}})

    # --- load prior metrics & prime timer (budgets strictly before this campaign) ---
    historical = load_historical_metrics()
    primer_df = build_primer_df(historical, strategies, budgets, danger_states)
    if not primer_df.is_empty():
        FUZZ_TIMER.prime(primer_df)
    else:
        print("[TIMING] No prior metrics to prime (fresh start or nothing before current range)")

    # Metrics collected by *this* process only (used for end-of-run summaries)
    all_metrics: pl.DataFrame = pl.DataFrame()
    install_control_ui()

    start_time = time.time()
    try:
        for budget in budgets:
            for danger_state in danger_states:
                if get_shutdown_level() == ShutdownLevel.IMMEDIATE:
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
                FUZZ_TIMER.print_summary(ReportLevel.DANGER)
                FUZZ_TIMER.print_summary(ReportLevel.PROCESS)
                FUZZ_TIMER._reset_budget()

            if get_shutdown_level() == ShutdownLevel.BUDGET:
                break

        FUZZ_TIMER.flush()
        ps._generate_all_cases_summaries(all_metrics)

    except KeyboardInterrupt:
        print("\nShutdown complete — generating summary from accumulated data...")
        FUZZ_TIMER.flush()
        ps._generate_all_cases_summaries(all_metrics)
        persist_merged_metrics(historical, all_metrics)
        sys.exit(0)

    persist_merged_metrics(historical, all_metrics)
    logger.success("Fuzz test suite finished")
    print(f"Total runtime: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    # NOTE: MIP optimal solutions (from EmpireOptimizer), PairingStrategy.optimized,
    #       are only available for budgets (5, 555, 5).

    # NOTE: For full fuzzing we would want to include all possible strategies
    # NOTE: For testing purposes we can use a list of a subset of strategies
    # strategies = [PairingStrategy.optimized, PairingStrategy.random_town]
    strategies = [s for s in PairingStrategy]

    # NOTE: For full fuzzing we should use a subset of budgets since the MIP
    # solver takes a long time and is executed for each strategy within each budget
    # times the number of samples.
    budgets = range(5, 555, 5)

    # NOTE: For normal fuzzing or testing purposes the sample count can be adjusted
    # as desired. The default is 20 to allow for a diverse random selection of pairs.
    samples = 20

    # For normal fuzzing use [False, True]
    danger_states = [True]

    fuzzer_main(strategies, samples, budgets, danger_states)
