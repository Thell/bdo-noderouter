# orchestrator_terminal_pairs.py

"""
terminal, root pair assignments by strategy.
"""

import random

from api_exploration_data import get_exploration_data, SUPER_ROOT
from orchestrator_types import PairingStrategy, Plan, Terminals


def _load_optimized_terminal_pairs(plan: Plan) -> dict[int, int]:
    if plan.budget is None or plan.budget % 5 != 0 or not 5 <= plan.budget <= 550:
        raise ValueError("Optimized strategy requires valid budget (5–550, step 5).")

    from pathlib import Path
    import json
    import api_data_store as ds

    path = Path(ds.path()) / "workerman"
    files = list(path.glob(f"{plan.budget}_*"))
    if not files:
        raise ValueError(f"No optimized workerman file found for budget {plan.budget}")
    with open(files[0]) as f:
        incident = json.load(f)

    return {int(w["job"]["pzk"]): int(w["job"]["storage"]) for w in incident["userWorkers"]}


def _load_custom_terminal_pairs() -> dict[int, int]:
    import api_data_store as ds

    return ds.read_json("custom_strategy_terminal_pairs.json")


def generate_terminal_pairs(plan: Plan) -> Terminals:
    """
    Generate terminal→root mappings for any pairing strategy.

    - For candidate-based strategies: select terminals, then assign roots from candidates().
    - For optimized strategy: load pre-solved pairs from file.
    - For custom strategy: load pairs from file.
    """
    exploration_data = get_exploration_data()
    src_dst: dict[int, int] = {}
    rng = random.Random(plan.seed)

    if plan.strategy == PairingStrategy.optimized:
        src_dst = _load_optimized_terminal_pairs(plan)
    elif plan.strategy == PairingStrategy.custom:
        src_dst = _load_custom_terminal_pairs()
    else:
        # Strategy based terminal selection
        selected_terminals = exploration_data.select_terminals(plan.worker_percent, rng)
        for t in selected_terminals:
            terminal = exploration_data.data[t]
            candidates = plan.strategy.candidates(terminal)
            src_dst[t] = rng.choice(candidates)

    if plan.include_danger:
        dangers = exploration_data.select_dangers(len(src_dst), rng)
        src_dst.update({d: SUPER_ROOT for d in dangers})

    return Terminals(src_dst)
