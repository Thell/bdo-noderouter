# testing_root_pairing.py

"""
Pairing strategies used for terminal, root pair assignments during fuzz testing.
Each strategy should return a list of roots that are valid for a given terminal.

- running this file directly unit tests the pairing strategies
"""

import unittest
from dataclasses import dataclass
from enum import StrEnum

import api_data_store as ds
from api_common import NodeType
from api_exploration_graph import (
    get_all_pairs_path_lengths,
    get_all_pairs_cartesian_distances,
    get_clean_exploration_data,
    get_exploration_graph,
)

# number of nearest/cheapest roots to consider for pairing
PAIRING_TOP_N = 3


# Module level pairing data container used by the pairing strategies.
# (instantiated after definition)
@dataclass
class PairingData:
    def __init__(self):
        config = ds.get_config("config")
        exploration_data = get_clean_exploration_data(config)
        graph = get_exploration_graph(exploration_data)

        self.exploration_data = exploration_data
        self.graph = graph

        tmp = get_all_pairs_path_lengths(graph)
        self.path_lengths = {
            (graph[s]["waypoint_key"], graph[d]["waypoint_key"]): v for (s, d), v in tmp.items()
        }

        self.cartesian_distances = get_all_pairs_cartesian_distances(exploration_data)

        self.towns = [n for n, d in exploration_data.items() if d.get("is_base_town", False)]

        self.territories = set(n["territory_key"] for n in exploration_data.values())
        self.capitals = {
            exploration_data[t]["territory_key"]: t
            for t in self.towns
            if exploration_data[t]["node_type"] == NodeType.city
        }

        tmp = {territory: [] for territory in self.territories}
        for t in self.towns:
            tmp[exploration_data[t]["territory_key"]].append(t)
        self.territory_towns = tmp

    def path_length(self, u: int, v: int) -> int:
        return self.path_lengths[(u, v)]

    def cartesian_distance(self, u: int, v: int) -> float:
        return self.cartesian_distances[(u, v)]

    def towns_in_territory(self, territory: int) -> list[int]:
        return self.territory_towns[territory]


PAIRING_DATA = PairingData()


class PairingStrategy(StrEnum):
    # nearest denotes as-the-bird=flies cartesian distance
    # cheapest denotes shortest path cost
    # town selections are inclusive of the capital
    # NOTE: terminal, root pairs typically only go to worker npc towns
    # but this is not a requirement and doesn't add any helpful fuzzer coverage

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

    # Disabled for normal fuzzing
    # random_terminal_to_terminal = "random_terminal_to_terminal"  # pathological stress test

    def candidates(self, terminal: dict) -> list[int]:
        """
        Return candidate root IDs for the given terminal under this pairing type.
        """
        match self:
            # Capital-based
            case PairingStrategy.nearest_capital:
                nearest = min(
                    PAIRING_DATA.capitals.values(),
                    key=lambda c: PAIRING_DATA.cartesian_distance(terminal["waypoint_key"], c),
                )
                return [nearest]

            case PairingStrategy.cheapest_capital:
                cheapest = min(
                    PAIRING_DATA.capitals.values(),
                    key=lambda c: PAIRING_DATA.path_length(terminal["waypoint_key"], c),
                )
                return [cheapest]

            case PairingStrategy.territory_capital:
                territory_id = terminal["territory_key"]
                return [PAIRING_DATA.capitals[territory_id]]

            # Town-based
            case PairingStrategy.nearest_town:
                nearest = min(
                    PAIRING_DATA.towns,
                    key=lambda t: PAIRING_DATA.cartesian_distance(terminal["waypoint_key"], t),
                )
                return [nearest]

            case PairingStrategy.cheapest_town:
                cheapest = min(
                    PAIRING_DATA.towns,
                    key=lambda t: PAIRING_DATA.path_length(terminal["waypoint_key"], t),
                )
                return [cheapest]

            case PairingStrategy.nearest_town_in_territory:
                towns = PAIRING_DATA.towns_in_territory(terminal["territory_key"])
                nearest = min(
                    towns,
                    key=lambda t: PAIRING_DATA.cartesian_distance(terminal["waypoint_key"], t),
                )
                return [nearest]

            case PairingStrategy.cheapest_town_in_territory:
                towns = PAIRING_DATA.towns_in_territory(terminal["territory_key"])
                cheapest = min(
                    towns,
                    key=lambda t: PAIRING_DATA.path_length(terminal["waypoint_key"], t),
                )
                return [cheapest]

            # Randomized (top-N constrained)
            case PairingStrategy.random_n_nearest_capital:
                nearest_sorted = sorted(
                    PAIRING_DATA.capitals.values(),
                    key=lambda t: PAIRING_DATA.cartesian_distance(terminal["waypoint_key"], t),
                )
                return nearest_sorted[:PAIRING_TOP_N]

            case PairingStrategy.random_n_nearest_town:
                nearest_sorted = sorted(
                    PAIRING_DATA.towns,
                    key=lambda t: PAIRING_DATA.cartesian_distance(terminal["waypoint_key"], t),
                )
                return nearest_sorted[:PAIRING_TOP_N]

            case PairingStrategy.random_n_cheapest_capital:
                cheapest_sorted = sorted(
                    PAIRING_DATA.capitals.values(),
                    key=lambda t: PAIRING_DATA.path_length(terminal["waypoint_key"], t),
                )
                return cheapest_sorted[:PAIRING_TOP_N]

            case PairingStrategy.random_n_cheapest_town:
                cheapest_sorted = sorted(
                    PAIRING_DATA.towns,
                    key=lambda t: PAIRING_DATA.path_length(terminal["waypoint_key"], t),
                )
                return cheapest_sorted[:PAIRING_TOP_N]

            case PairingStrategy.random_n_nearest_town_in_territory:
                towns = PAIRING_DATA.towns_in_territory(terminal["territory_key"])
                nearest_sorted = sorted(
                    towns,
                    key=lambda t: PAIRING_DATA.cartesian_distance(terminal["waypoint_key"], t),
                )
                return nearest_sorted[:PAIRING_TOP_N]

            case PairingStrategy.random_n_cheapest_town_in_territory:
                towns = PAIRING_DATA.towns_in_territory(terminal["territory_key"])
                cheapest_sorted = sorted(
                    towns,
                    key=lambda t: PAIRING_DATA.path_length(terminal["waypoint_key"], t),
                )
                return cheapest_sorted[:PAIRING_TOP_N]

            # Randomized (unconstrained)
            case PairingStrategy.random_capital:
                return list(PAIRING_DATA.capitals.values())

            case PairingStrategy.random_town:
                return PAIRING_DATA.towns

            # case PairingStrategy.random_terminal_to_terminal:
            #     # stress-test: allow any node to be treated as a root
            #     return list(PAIRING_DATA.exploration_data.keys())

            case _:
                raise ValueError(f"Unknown pairing type: {self}")


class TestPairingStrategies(unittest.TestCase):
    def setUp(self):
        # Pick an arbitrary terminal from the exploration data
        # (just grab the first one for testing purposes)
        self.terminal_id, self.terminal = next(iter(PAIRING_DATA.exploration_data.items()))

        # terminal keys for testing
        self.kamasylve_key = 1136
        self.dorman_lumber_key = 1665

        # capital/town keys for testing
        self.velia_key = 1
        self.heidel_key = 301
        self.glish_key = 302
        self.keplan_key = 602
        self.altinova_key = 1101
        self.tarif_key = 1141
        self.duvencrune_key = 1649
        self.eilton_key = 1750
        self.asparken_key = 1834

    def test_nearest_capital_returns_single(self):
        roots = PairingStrategy.nearest_capital.candidates(self.terminal)
        self.assertEqual(len(roots), 1)
        self.assertIn(roots[0], PAIRING_DATA.capitals.values())

    def test_cheapest_capital_returns_single(self):
        roots = PairingStrategy.cheapest_capital.candidates(self.terminal)
        self.assertEqual(len(roots), 1)
        self.assertIn(roots[0], PAIRING_DATA.capitals.values())

    def test_random_n_nearest_town_top_n(self):
        roots = PairingStrategy.random_n_nearest_town.candidates(self.terminal)
        self.assertLessEqual(len(roots), PAIRING_TOP_N)
        for r in roots:
            self.assertIn(r, PAIRING_DATA.towns)

    def test_territory_capital_matches_terminal_territory(self):
        roots = PairingStrategy.territory_capital.candidates(self.terminal)
        self.assertEqual(len(roots), 1)
        expected_capital = PAIRING_DATA.capitals[self.terminal["territory_key"]]
        self.assertEqual(roots[0], expected_capital)

    def test_dorman_nearest_capital(self):
        terminal = PAIRING_DATA.exploration_data[self.dorman_lumber_key]
        roots = PairingStrategy.nearest_capital.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_nearest_capital(self):
        terminal = PAIRING_DATA.exploration_data[self.kamasylve_key]
        roots = PairingStrategy.nearest_capital.candidates(terminal)
        self.assertEqual(roots[0], self.heidel_key)

    def test_dorman_cheapest_capital(self):
        terminal = PAIRING_DATA.exploration_data[self.dorman_lumber_key]
        roots = PairingStrategy.cheapest_capital.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_cheapest_capital(self):
        terminal = PAIRING_DATA.exploration_data[self.kamasylve_key]
        roots = PairingStrategy.cheapest_capital.candidates(terminal)
        self.assertEqual(roots[0], self.heidel_key)

    def test_dorman_territory_capital(self):
        terminal = PAIRING_DATA.exploration_data[self.dorman_lumber_key]
        roots = PairingStrategy.territory_capital.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_territory_capital(self):
        terminal = PAIRING_DATA.exploration_data[self.kamasylve_key]
        roots = PairingStrategy.territory_capital.candidates(terminal)
        self.assertEqual(roots[0], self.altinova_key)

    def test_dorman_nearest_town(self):
        terminal = PAIRING_DATA.exploration_data[self.dorman_lumber_key]
        roots = PairingStrategy.nearest_town.candidates(terminal)
        self.assertEqual(roots[0], self.glish_key)

    def test_kamasylve_nearest_town(self):
        terminal = PAIRING_DATA.exploration_data[self.kamasylve_key]
        roots = PairingStrategy.nearest_town.candidates(terminal)
        self.assertEqual(roots[0], self.tarif_key)

    def test_dorman_cheapest_town(self):
        terminal = PAIRING_DATA.exploration_data[self.dorman_lumber_key]
        roots = PairingStrategy.cheapest_town.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_cheapest_town(self):
        terminal = PAIRING_DATA.exploration_data[self.kamasylve_key]
        roots = PairingStrategy.cheapest_town.candidates(terminal)
        self.assertEqual(roots[0], self.heidel_key)

    def test_dorman_nearest_town_in_territory(self):
        terminal = PAIRING_DATA.exploration_data[self.dorman_lumber_key]
        roots = PairingStrategy.nearest_town_in_territory.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_nearest_town_in_territory(self):
        terminal = PAIRING_DATA.exploration_data[self.kamasylve_key]
        roots = PairingStrategy.nearest_town_in_territory.candidates(terminal)
        self.assertEqual(roots[0], self.tarif_key)

    def test_dorman_cheapest_town_in_territory(self):
        terminal = PAIRING_DATA.exploration_data[self.dorman_lumber_key]
        roots = PairingStrategy.cheapest_town_in_territory.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_cheapest_town_in_territory(self):
        terminal = PAIRING_DATA.exploration_data[self.kamasylve_key]
        roots = PairingStrategy.cheapest_town_in_territory.candidates(terminal)
        self.assertEqual(roots[0], self.tarif_key)

    def test_dorman_random_n_nearest_capital(self):
        terminal = PAIRING_DATA.exploration_data[self.dorman_lumber_key]
        roots = PairingStrategy.random_n_nearest_capital.candidates(terminal)
        self.assertLessEqual(len(roots), PAIRING_TOP_N)
        for c in [self.duvencrune_key, self.heidel_key, self.eilton_key]:
            self.assertIn(c, roots)

    def test_kamasylve_random_n_nearest_capital(self):
        terminal = PAIRING_DATA.exploration_data[self.kamasylve_key]
        roots = PairingStrategy.random_n_nearest_capital.candidates(terminal)
        self.assertLessEqual(len(roots), PAIRING_TOP_N)
        for c in [self.heidel_key, self.velia_key, self.asparken_key]:
            self.assertIn(c, roots)

    def test_dorman_random_n_nearest_town(self):
        terminal = PAIRING_DATA.exploration_data[self.dorman_lumber_key]
        roots = PairingStrategy.random_n_nearest_town.candidates(terminal)
        self.assertLessEqual(len(roots), PAIRING_TOP_N)
        for t in [self.glish_key, self.duvencrune_key, self.keplan_key]:
            self.assertIn(t, roots)

    def test_kamasylve_random_n_nearest_town(self):
        terminal = PAIRING_DATA.exploration_data[self.kamasylve_key]
        roots = PairingStrategy.random_n_nearest_town.candidates(terminal)
        self.assertLessEqual(len(roots), PAIRING_TOP_N)
        for t in [self.tarif_key, self.heidel_key, self.velia_key]:
            self.assertIn(t, roots)


if __name__ == "__main__":
    print("Pairing data:")
    print(f"{len(PAIRING_DATA.capitals)} territories and {len(PAIRING_DATA.towns)} towns.")
    print(f"Territory capitals: {PAIRING_DATA.capitals}")
    print(f"Towns: {PAIRING_DATA.towns}")

    unittest.main()

    print("Done.")
