# orchestrator_pairing_strategy.py
from __future__ import annotations

from enum import StrEnum

from api_exploration_data import get_exploration_data

# number of nearest/cheapest roots to consider for pairing
PAIRING_TOP_N = 3

# NOTE: MAX_LEN_PAIRING_STRATEGY is assigned after the class definition
MAX_LEN_PAIRING_STRATEGY = 0


class PairingStrategy(StrEnum):
    """NOTES:
    - nearest denotes as-the-bird=flies cartesian distance
    - cheapest denotes shortest path cost
    - terminals can only root to base_towns
    - town selections are inclusive of the capital
    """

    # Optimized (via EmpireOptimizer solution extraction)
    optimized = "optimized"

    # Deterministic
    nearest_capital = "nearest_capital"
    cheapest_capital = "cheapest_capital"
    territory_capital = "territory_capital"

    nearest_town = "nearest_town"
    cheapest_town = "cheapest_town"
    nearest_town_in_territory = "nearest_town_in_territory"
    cheapest_town_in_territory = "cheapest_town_in_territory"

    # Randomized (top-N constrained)
    random_n_nearest_capital = "random_n_nearest_capital"
    random_n_nearest_town = "random_n_nearest_town"
    random_n_cheapest_capital = "random_n_cheapest_capital"
    random_n_cheapest_town = "random_n_cheapest_town"
    random_n_nearest_town_in_territory = "random_n_nearest_town_in_territory"
    random_n_cheapest_town_in_territory = "random_n_cheapest_town_in_territory"

    # Randomized (unconstrained)
    random_capital = "random_capital"
    random_town = "random_town"

    # Custom
    custom = "custom"

    def candidates(self, terminal: dict) -> list[int]:
        """
        Return candidate root IDs for the given terminal under this pairing type.
        """
        data = get_exploration_data()
        match self:
            # Optimized (via EmpireOptimizer solution extraction)
            case PairingStrategy.optimized:
                from loguru import logger

                # Pairs are already optimized and loaded from file.
                logger.warning("candidates() should not be called for optimized pairs.")
                return []

            # Capital-based
            case PairingStrategy.nearest_capital:
                nearest = min(
                    data.capitals.values(),
                    key=lambda c: data.cartesian_distance(terminal["waypoint_key"], c),
                )
                return [nearest]

            case PairingStrategy.cheapest_capital:
                cheapest = min(
                    data.capitals.values(),
                    key=lambda c: data.path_length(terminal["waypoint_key"], c),
                )
                return [cheapest]

            case PairingStrategy.territory_capital:
                territory_id = terminal["territory_key"]
                return [data.capitals[territory_id]]

            # Town-based
            case PairingStrategy.nearest_town:
                nearest = min(
                    data.towns,
                    key=lambda t: data.cartesian_distance(terminal["waypoint_key"], t),
                )
                return [nearest]

            case PairingStrategy.cheapest_town:
                cheapest = min(
                    data.towns,
                    key=lambda t: data.path_length(terminal["waypoint_key"], t),
                )
                return [cheapest]

            case PairingStrategy.nearest_town_in_territory:
                towns = data.towns_in_territory(terminal["territory_key"])
                nearest = min(
                    towns,
                    key=lambda t: data.cartesian_distance(terminal["waypoint_key"], t),
                )
                return [nearest]

            case PairingStrategy.cheapest_town_in_territory:
                towns = data.towns_in_territory(terminal["territory_key"])
                cheapest = min(
                    towns,
                    key=lambda t: data.path_length(terminal["waypoint_key"], t),
                )
                return [cheapest]

            # Randomized (top-N constrained)
            case PairingStrategy.random_n_nearest_capital:
                nearest_sorted = sorted(
                    data.capitals.values(),
                    key=lambda t: data.cartesian_distance(terminal["waypoint_key"], t),
                )
                return nearest_sorted[:PAIRING_TOP_N]

            case PairingStrategy.random_n_nearest_town:
                nearest_sorted = sorted(
                    data.towns,
                    key=lambda t: data.cartesian_distance(terminal["waypoint_key"], t),
                )
                return nearest_sorted[:PAIRING_TOP_N]

            case PairingStrategy.random_n_cheapest_capital:
                cheapest_sorted = sorted(
                    data.capitals.values(),
                    key=lambda t: data.path_length(terminal["waypoint_key"], t),
                )
                return cheapest_sorted[:PAIRING_TOP_N]

            case PairingStrategy.random_n_cheapest_town:
                cheapest_sorted = sorted(
                    data.towns,
                    key=lambda t: data.path_length(terminal["waypoint_key"], t),
                )
                return cheapest_sorted[:PAIRING_TOP_N]

            case PairingStrategy.random_n_nearest_town_in_territory:
                towns = data.towns_in_territory(terminal["territory_key"])
                nearest_sorted = sorted(
                    towns,
                    key=lambda t: data.cartesian_distance(terminal["waypoint_key"], t),
                )
                return nearest_sorted[:PAIRING_TOP_N]

            case PairingStrategy.random_n_cheapest_town_in_territory:
                towns = data.towns_in_territory(terminal["territory_key"])
                cheapest_sorted = sorted(
                    towns,
                    key=lambda t: data.path_length(terminal["waypoint_key"], t),
                )
                return cheapest_sorted[:PAIRING_TOP_N]

            # Randomized (unconstrained)
            case PairingStrategy.random_capital:
                return list(data.capitals.values())

            case PairingStrategy.random_town:
                return data.towns

            # Custom
            case PairingStrategy.custom:
                return []

            case _:
                raise ValueError(f"Unknown pairing type: {self}")

    def max_name_length(self) -> int:
        return max(len(s.name) for s in PairingStrategy)


MAX_LEN_PAIRING_STRATEGY: int = PairingStrategy("optimized").max_name_length() + 1
