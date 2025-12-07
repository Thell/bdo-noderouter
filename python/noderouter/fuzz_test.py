# fuzz_test.py

from __future__ import annotations

from statistics import mean, stdev
from typing import Any
from collections import defaultdict
import signal
import sys
from functools import partial

from loguru import logger
from tabulate import tabulate

import data_store as ds
import testing as test
from api_common import set_logger
from mip_baseline_rust import optimize_with_terminals as mip_optimize
from node_router import optimize_with_terminals as nr_optimize


# Global shutdown flag
_shutdown_requested = False


def _signal_handler(signum, frame, containers):
    global _shutdown_requested
    if _shutdown_requested:
        print("\nHard exit — terminating immediately.", file=sys.stderr)
        sys.exit(1)
    _shutdown_requested = True
    print("\nGraceful shutdown requested — finishing current test and reporting results...", file=sys.stderr)


def install_graceful_shutdown(containers):
    handler = partial(_signal_handler, containers=containers)
    signal.signal(signal.SIGINT, handler)


def run_single_config(
    samples: int,
    budget: int,
    test_type: str,
    include_danger: bool,
) -> dict[str, Any]:
    config = ds.get_config("config")
    config["name"] = "fuzz_test"
    set_logger(config)

    results: list[dict[str, Any]] = []
    suboptimal: list[dict[str, Any]] = []
    base_seed = 10_000

    if test_type == "workerman":
        desc = f"workerman budget={budget:3} | danger={'yes' if include_danger else 'no'}"
        value = budget
    elif test_type == "random":
        coverage = round(budget / 550 * 100)
        desc = f"random {coverage:3}% cov (budget {budget:3}) | danger={'yes' if include_danger else 'no'}"
        value = coverage
    else:  # territorial
        coverage = round(budget / 550 * 100)
        desc = f"territorial {coverage:3}% workers (budget {budget:3}) | danger={'yes' if include_danger else 'no'}"
        value = coverage

    logger.info(f"Starting: {desc} | samples={samples}")

    for i in range(samples):
        if _shutdown_requested:
            break

        seed = base_seed + i

        try:
            mip_res = (
                test.workerman_terminals(mip_optimize, config, value, include_danger, None, seed)
                if test_type == "workerman"
                else test.random_terminals(mip_optimize, config, value, include_danger, None, seed)
                if test_type == "random"
                else test.territorial_terminals(mip_optimize, config, value, include_danger, None, seed)
            )

            nr_res = (
                test.workerman_terminals(nr_optimize, config, value, include_danger, None, seed)
                if test_type == "workerman"
                else test.random_terminals(nr_optimize, config, value, include_danger, None, seed)
                if test_type == "random"
                else test.territorial_terminals(nr_optimize, config, value, include_danger, None, seed)
            )

            if mip_res["cost"] is None or nr_res["cost"] is None:
                logger.error(f"seed {seed} → solver failed")
                continue

            ratio = nr_res["cost"] / mip_res["cost"]
            speedup = mip_res["duration"] / nr_res["duration"] if nr_res["duration"] > 0 else float("inf")

            result = {
                "seed": seed,
                "budget": budget,
                "test_type": test_type,
                "include_danger": include_danger,
                "param": value,
                "roots": mip_res["roots"],
                "workers": mip_res["workers"],
                "dangers": mip_res["dangers"],
                "n_term": mip_res["workers"] + mip_res["dangers"],
                "mip_cost": mip_res["cost"],
                "mip_time": mip_res["duration"],
                "nr_cost": nr_res["cost"],
                "nr_time": nr_res["duration"],
                "ratio": ratio,
                "speedup": speedup,
            }
            results.append(result)

            print(
                f"[{i + 1:>4}/{samples}] seed={seed:5}  {desc:50}  "
                f"|T|={result['n_term']:>3} w:{result['workers']:>3} d:{result['dangers']:>2}  "
                f"MIP {mip_res['cost']:>4} ({mip_res['duration']:6.3f}s) → NR {nr_res['cost']:>4} ({nr_res['duration']:6.3f}s)  "
                f"ratio={ratio:5.3f}  {speedup:7.1f}x"
            )

            if nr_res["cost"] > mip_res["cost"]:
                gap = nr_res["cost"] - mip_res["cost"]
                logger.warning(f"SUBOPTIMAL → {desc} | seed={seed} | +{gap}")
                suboptimal.append(result)

        except Exception as e:
            logger.error(f"seed {seed} → {e}")
            continue

    # Per-config summary
    if results:
        ratios = [r["ratio"] for r in results]
        optimal = sum(1 for r in results if r["nr_cost"] == r["mip_cost"])

        logger.success(f"Completed: {desc}")
        print("\n" + "=" * 110)
        print(f"SUMMARY — {desc}")
        print(f"Instances tested       : {len(results)}")
        print(f"NodeRouter optimal     : {optimal} ({optimal / len(results):.1%})")
        print(f"NodeRouter suboptimal  : {len(suboptimal)} ({len(suboptimal) / len(results):.1%})")
        print(f"Avg approx ratio       : {mean(ratios):.4f} ± {stdev(ratios):.4f}")
        print(f"Best / Worst ratio     : {min(ratios):.3f} / {max(ratios):.3f}")
        print(
            f"Avg speedup            : {mean([r['speedup'] for r in results if r['speedup'] != float('inf')]):.0f}x"
        )
        print(
            f"Avg roots / workers / dangers : {mean([r['roots'] for r in results]):.1f} / {mean([r['workers'] for r in results]):.1f} / {mean([r['dangers'] for r in results]):.1f}"
        )

        if suboptimal:
            print("\nSUBOPTIMAL CASES")
            print("-" * 100)
            for case in sorted(suboptimal, key=lambda x: x["ratio"], reverse=True):
                print(
                    f"seed={case['seed']:5} | {desc:50} | "
                    f"r:{case['roots']:>2} w:{case['workers']:>3} d:{case['dangers']:>2} | "
                    f"MIP={case['mip_cost']:>4} → NR={case['nr_cost']:>4} (+{case['nr_cost'] - case['mip_cost']:>2}) | "
                    f"ratio={case['ratio']:.3f}"
                )
        print("=" * 110 + "\n")

    return {"results": results, "suboptimal": suboptimal}


def _print_global_summary(all_results: list[dict[str, Any]], all_suboptimal: list[dict[str, Any]]) -> None:
    if not all_results:
        logger.warning("No results to summarize.")
        return

    grouped = defaultdict(list)
    for r in all_results:
        key = (r["test_type"], r["budget"], r["include_danger"])
        grouped[key].append(r)

    table_data = []
    total_instances = total_optimal = 0
    all_ratios = []

    for (ttype, budget, inc_danger), results in sorted(grouped.items()):
        instances = len(results)
        optimal = sum(1 for r in results if r["nr_cost"] == r["mip_cost"])
        ratios = [r["ratio"] for r in results]
        speedups = [r["speedup"] for r in results if r["speedup"] != float("inf")]
        roots = mean(r["roots"] for r in results)
        workers = mean(r["workers"] for r in results)
        dangers = mean(r["dangers"] for r in results)
        terminals = mean(r["n_term"] for r in results)

        total_instances += instances
        total_optimal += optimal
        all_ratios.extend(ratios)

        table_data.append([
            ttype,
            budget,
            f"{roots:.1f}",
            f"{workers:.0f}",
            f"{dangers:.0f}",
            f"{terminals:.1f}",
            instances,
            optimal,
            instances - optimal,
            f"{optimal / instances:.1%}",
            f"{mean(ratios):.4f}",
            f"{max(ratios):.3f}",
            f"{mean(speedups):.0f}x",
        ])

    table_data.append([
        "TOTAL",
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
        f"{mean([r['speedup'] for r in all_results if r['speedup'] != float('inf')]):.0f}x",
    ])

    headers = [
        "Type",
        "Budget",
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
    logger.success("GLOBAL SUMMARY — BY TEST TYPE AND BUDGET")
    print(tabulate(table_data, headers=headers, tablefmt="simple", stralign="right", numalign="right"))
    print("#" * 160)

    if all_suboptimal:
        print("\nWORST SUBOPTIMAL INSTANCES (top 20)")
        print("-" * 100)
        for case in sorted(all_suboptimal, key=lambda x: x["ratio"], reverse=True)[:20]:
            print(
                f"seed={case['seed']:5} | {case['test_type']:<11} | budget={case['budget']:>3} | "
                f"roots={case['roots']:>2} | workers={case['workers']:>3} | dngr={case['dangers']:>2} | "
                f"|T|={case['n_term']:>3} | MIP={case['mip_cost']:>4} → NR={case['nr_cost']:>4} (+{case['nr_cost'] - case['mip_cost']:>2}) | "
                f"ratio={case['ratio']:.3f}"
            )
    print("#" * 160)
    logger.success("Fuzz test suite completed")


def run_fuzz_comparison(
    samples: int = 100,
    budgets: list[int] | None = None,
) -> None:
    if budgets is None:
        budgets = list(range(5, 551, 5))

    all_results: list[dict[str, Any]] = []
    all_suboptimal: list[dict[str, Any]] = []

    install_graceful_shutdown((all_results, all_suboptimal))

    try:
        for budget in budgets:
            for include_danger in (False, True):
                for test_type in ("workerman", "random", "territorial"):
                    if _shutdown_requested:
                        raise KeyboardInterrupt

                    res = run_single_config(samples, budget, test_type, include_danger)
                    all_results.extend(res["results"])
                    all_suboptimal.extend(res["suboptimal"])

        _print_global_summary(all_results, all_suboptimal)

    except KeyboardInterrupt:
        print("\nShutdown complete — generating summary from accumulated data...")
        _print_global_summary(all_results, all_suboptimal)
        sys.exit(0)


if __name__ == "__main__":
    config = ds.get_config("config")

    if config.get("actions", {}).get("fuzz_test", True):
        cfg = config.get("fuzz_test_config", {})
        run_fuzz_comparison(
            samples=cfg.get("samples", 100),
            budgets=cfg.get("budgets", range(5, 551, 5)),
        )
        logger.success("Fuzz test suite finished")
    else:
        logger.info("fuzz_test not enabled — set actions.fuzz_test: true in config")
