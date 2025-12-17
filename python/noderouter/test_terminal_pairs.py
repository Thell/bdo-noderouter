# test_terminal_pairs.py

"""Unit tests for terminal pairing strategies."""

import unittest
from api_exploration_data import get_exploration_data
from orchestrator_pairing_strategy import PairingStrategy, PAIRING_TOP_N


class TestPairingStrategies(unittest.TestCase):
    def setUp(self):
        self.data = get_exploration_data()

        # Pick an arbitrary terminal from the exploration data
        # (just grab the first one for testing purposes)
        self.terminal_id, self.terminal = next(iter(self.data.data.items()))

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
        self.assertIn(roots[0], self.data.capitals.values())

    def test_cheapest_capital_returns_single(self):
        roots = PairingStrategy.cheapest_capital.candidates(self.terminal)
        self.assertEqual(len(roots), 1)
        self.assertIn(roots[0], self.data.capitals.values())

    def test_random_n_nearest_town_top_n(self):
        roots = PairingStrategy.random_n_nearest_town.candidates(self.terminal)
        self.assertLessEqual(len(roots), PAIRING_TOP_N)
        for r in roots:
            self.assertIn(r, self.data.towns)

    def test_territory_capital_matches_terminal_territory(self):
        roots = PairingStrategy.territory_capital.candidates(self.terminal)
        self.assertEqual(len(roots), 1)
        expected_capital = self.data.capitals[self.terminal["territory_key"]]
        self.assertEqual(roots[0], expected_capital)

    def test_dorman_nearest_capital(self):
        terminal = self.data.data[self.dorman_lumber_key]
        roots = PairingStrategy.nearest_capital.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_nearest_capital(self):
        terminal = self.data.data[self.kamasylve_key]
        roots = PairingStrategy.nearest_capital.candidates(terminal)
        self.assertEqual(roots[0], self.heidel_key)

    def test_dorman_cheapest_capital(self):
        terminal = self.data.data[self.dorman_lumber_key]
        roots = PairingStrategy.cheapest_capital.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_cheapest_capital(self):
        terminal = self.data.data[self.kamasylve_key]
        roots = PairingStrategy.cheapest_capital.candidates(terminal)
        self.assertEqual(roots[0], self.heidel_key)

    def test_dorman_territory_capital(self):
        terminal = self.data.data[self.dorman_lumber_key]
        roots = PairingStrategy.territory_capital.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_territory_capital(self):
        terminal = self.data.data[self.kamasylve_key]
        roots = PairingStrategy.territory_capital.candidates(terminal)
        self.assertEqual(roots[0], self.altinova_key)

    def test_dorman_nearest_town(self):
        terminal = self.data.data[self.dorman_lumber_key]
        roots = PairingStrategy.nearest_town.candidates(terminal)
        self.assertEqual(roots[0], self.glish_key)

    def test_kamasylve_nearest_town(self):
        terminal = self.data.data[self.kamasylve_key]
        roots = PairingStrategy.nearest_town.candidates(terminal)
        self.assertEqual(roots[0], self.tarif_key)

    def test_dorman_cheapest_town(self):
        terminal = self.data.data[self.dorman_lumber_key]
        roots = PairingStrategy.cheapest_town.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_cheapest_town(self):
        terminal = self.data.data[self.kamasylve_key]
        roots = PairingStrategy.cheapest_town.candidates(terminal)
        self.assertEqual(roots[0], self.heidel_key)

    def test_dorman_nearest_town_in_territory(self):
        terminal = self.data.data[self.dorman_lumber_key]
        roots = PairingStrategy.nearest_town_in_territory.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_nearest_town_in_territory(self):
        terminal = self.data.data[self.kamasylve_key]
        roots = PairingStrategy.nearest_town_in_territory.candidates(terminal)
        self.assertEqual(roots[0], self.tarif_key)

    def test_dorman_cheapest_town_in_territory(self):
        terminal = self.data.data[self.dorman_lumber_key]
        roots = PairingStrategy.cheapest_town_in_territory.candidates(terminal)
        self.assertEqual(roots[0], self.duvencrune_key)

    def test_kamasylve_cheapest_town_in_territory(self):
        terminal = self.data.data[self.kamasylve_key]
        roots = PairingStrategy.cheapest_town_in_territory.candidates(terminal)
        self.assertEqual(roots[0], self.tarif_key)

    def test_dorman_random_n_nearest_capital(self):
        terminal = self.data.data[self.dorman_lumber_key]
        roots = PairingStrategy.random_n_nearest_capital.candidates(terminal)
        self.assertLessEqual(len(roots), PAIRING_TOP_N)
        for c in [self.duvencrune_key, self.heidel_key, self.eilton_key]:
            self.assertIn(c, roots)

    def test_kamasylve_random_n_nearest_capital(self):
        terminal = self.data.data[self.kamasylve_key]
        roots = PairingStrategy.random_n_nearest_capital.candidates(terminal)
        self.assertLessEqual(len(roots), PAIRING_TOP_N)
        for c in [self.heidel_key, self.velia_key, self.asparken_key]:
            self.assertIn(c, roots)

    def test_dorman_random_n_nearest_town(self):
        terminal = self.data.data[self.dorman_lumber_key]
        roots = PairingStrategy.random_n_nearest_town.candidates(terminal)
        self.assertLessEqual(len(roots), PAIRING_TOP_N)
        for t in [self.glish_key, self.duvencrune_key, self.keplan_key]:
            self.assertIn(t, roots)

    def test_kamasylve_random_n_nearest_town(self):
        terminal = self.data.data[self.kamasylve_key]
        roots = PairingStrategy.random_n_nearest_town.candidates(terminal)
        self.assertLessEqual(len(roots), PAIRING_TOP_N)
        for t in [self.tarif_key, self.heidel_key, self.velia_key]:
            self.assertIn(t, roots)


if __name__ == "__main__":
    import time
    import api_data_store as ds
    from api_common import set_logger

    set_logger(ds.get_config("config"))

    start_time = time.time()
    data = get_exploration_data()
    print(f"Loading exploration data took {time.time() - start_time:.2f}s")

    print("Pairing data:")
    print(f"{len(data.capitals)} territories and {len(data.towns)} towns.")
    print(f"Territory capitals: {data.capitals}")
    print(f"Towns: {data.towns}")

    print("Pairing tests:")
    unittest.main()

    print("Done.")
