"""Regression checks for sustainable yields and the extinction-boundary map."""

import unittest

from plot_generators.numeric_yield_zone_map import (
    EXTINCTION,
    PROTECT_A_HARVEST_B,
    PROTECT_B_HARVEST_A,
    local_neighborhood_effort_search,
    numeric_zone,
    simulate_equilibrium_yield,
)
from plot_generators.common import simulate_linked_equilibrium_and_yield


class SustainableYieldTests(unittest.TestCase):
    def test_disconnected_patches_match_scalar_beverton_holt_yields(self):
        effort_a, effort_b = 0.1, 0.2
        r_a, r_b = 2.0, 1.8
        k_a, k_b = 2.0, 1.5
        expected = (
            k_a * effort_a * (r_a - 1 / (1 - effort_a))
            + k_b * effort_b * (r_b - 1 / (1 - effort_b))
        )
        actual = simulate_equilibrium_yield(
            effort_a, effort_b, r_a=r_a, r_b=r_b, m=0,
            k_a=k_a, k_b=k_b,
        )
        self.assertAlmostEqual(actual, expected, delta=2e-9)

    def test_harvested_sink_matches_exact_source_yield(self):
        m, k_b = 0.45, 1.7
        q = 1 - m
        # With r_A=0 and e_A=1, the source obeys a scalar BH map.
        for r_b in (55 / 30, 2.2, 4.0):
            with self.subTest(r_b=r_b):
                expected = m * k_b * (q * r_b - 1) / q
                actual = simulate_equilibrium_yield(
                    1, 0, r_a=0, r_b=r_b, m=m, k_a=2, k_b=k_b,
                )
                self.assertAlmostEqual(actual, expected, delta=2e-9)

    def test_critical_source_has_zero_sustainable_yield(self):
        m = 0.45
        actual = simulate_equilibrium_yield(
            1, 0, r_a=0, r_b=1 / (1 - m), m=m, k_a=2, k_b=1,
        )
        self.assertLessEqual(abs(actual), 2e-12)

    def test_near_critical_source_resolves_positive_yield(self):
        m, k_b = 0.45, 1.0
        q = 1 - m
        r_b = 1 / q + 1e-7
        expected = m * k_b * (q * r_b - 1) / q
        actual = simulate_equilibrium_yield(
            1, 0, r_a=0, r_b=r_b, m=m, k_a=2, k_b=k_b,
        )
        self.assertGreater(actual, 0)
        self.assertAlmostEqual(actual, expected, delta=2e-12)

    def test_transient_harvest_does_not_count_as_sustainable_yield(self):
        # The old 300-step evaluation returned about 1.21e-4 here,
        # although these fishing efforts drive both patches to extinction.
        actual = simulate_equilibrium_yield(
            0, 3 / 39, r_a=31 / 30, r_b=31 / 30,
            m=0.45, k_a=2, k_b=1,
        )
        self.assertLessEqual(abs(actual), 2e-12)

    def test_yield_is_invariant_under_swapping_patches(self):
        original = simulate_equilibrium_yield(
            0.14, 0.27, r_a=1.8, r_b=2.4, m=0.45, k_a=2, k_b=0.7,
        )
        swapped = simulate_equilibrium_yield(
            0.27, 0.14, r_a=2.4, r_b=1.8, m=0.45, k_a=0.7, k_b=2,
        )
        self.assertAlmostEqual(original, swapped, delta=2e-9)

    def test_insufficient_iteration_budget_reports_nonconvergence(self):
        with self.assertRaises(RuntimeError):
            simulate_equilibrium_yield(
                0.2, 0.08, r_a=1.6, r_b=1.1, m=0.45, k_a=2, k_b=1,
                yield_atol=1e-15, yield_rtol=1e-13, max_iterations=1,
            )

    def test_nearly_complete_harvest_matches_long_direct_simulation(self):
        for r_a, r_b, effort_a, effort_b in (
            (0.1, 10 / 3, 0.99999990218724966, 0.004156650641025641),
            (1.0, 139 / 30, 0.99999999999999978, 0.1553735977564103),
        ):
            with self.subTest(r_a=r_a, r_b=r_b):
                parameters = dict(r_a=r_a, r_b=r_b, m=0.45, k_a=2, k_b=1)
                _, _, expected = simulate_linked_equilibrium_and_yield(
                    e_a=effort_a, e_b=effort_b, steps=4000, **parameters,
                )
                actual = simulate_equilibrium_yield(effort_a, effort_b, **parameters)
                self.assertAlmostEqual(actual, expected, delta=1e-9)

    def test_nearly_empty_sink_keeps_a_tight_catch_bound(self):
        m, r_b = 0.45, 55 / 30
        effort_a = 0.99999847412109377
        effort_b = 9.1552734375000014e-6
        source_retention = (1 - effort_b) * (1 - m)
        source_population = source_retention * r_b - 1
        expected = (
            (effort_a * m + effort_b * (1 - m))
            * source_population / source_retention
        )
        actual = simulate_equilibrium_yield(
            effort_a, effort_b, r_a=0, r_b=r_b, m=m, k_a=2, k_b=1,
        )
        self.assertAlmostEqual(actual, expected, delta=1e-11)


class ZoneClassificationTests(unittest.TestCase):
    def test_refinement_recenters_before_cutting_off_boundary_optimum(self):
        results = []
        for coarse_size in (21, 40, 81):
            with self.subTest(coarse_size=coarse_size):
                effort_a, effort_b, yield_value = local_neighborhood_effort_search(
                    1.005, 1.005, m=0.45, k_a=2, k_b=1,
                    effort_grid_size=coarse_size,
                )
                self.assertLessEqual(effort_a, 1e-4)
                self.assertGreater(effort_b, 0)
                self.assertLess(effort_b, 1)
                results.append(yield_value)
        self.assertLess(max(results) - min(results), 2e-10)

    def test_insufficient_refinement_budget_reports_nonconvergence(self):
        with self.assertRaisesRegex(RuntimeError, "effort search did not converge"):
            local_neighborhood_effort_search(
                1.8, 1.2, m=0.45, k_a=2, k_b=1, max_search_iterations=1,
            )

    def test_refinement_retains_a_small_sustainable_peak(self):
        parameters = dict(r_a=31 / 30, r_b=31 / 30, m=0.45, k_a=2, k_b=1)
        candidate_yield = simulate_equilibrium_yield(0, 1 / 39, **parameters)
        self.assertGreater(candidate_yield, 6e-4)
        effort_a, effort_b, best_yield = local_neighborhood_effort_search(
            **parameters,
        )
        self.assertGreaterEqual(best_yield, candidate_yield - 2e-12)
        self.assertAlmostEqual(
            best_yield,
            simulate_equilibrium_yield(effort_a, effort_b, **parameters),
            delta=2e-12,
        )

    def test_old_red_spur_is_extinction(self):
        # These are actual coordinates from the project's 151x151 map.
        for r_b in (52 / 30, 53 / 30, 54 / 30):
            with self.subTest(r_b=r_b):
                self.assertEqual(
                    numeric_zone(0, r_b, m=0.45, k_a=2, k_b=1),
                    EXTINCTION,
                )

    def test_source_endpoint_immediately_above_spur_remains(self):
        self.assertEqual(
            numeric_zone(0, 55 / 30, m=0.45, k_a=2, k_b=1),
            PROTECT_B_HARVEST_A,
        )

    def test_zone_swaps_with_patch_names_and_population_scales(self):
        self.assertEqual(
            numeric_zone(0.3, 2.4, m=0.45, k_a=2, k_b=1),
            PROTECT_B_HARVEST_A,
        )
        self.assertEqual(
            numeric_zone(2.4, 0.3, m=0.45, k_a=1, k_b=2),
            PROTECT_A_HARVEST_B,
        )


if __name__ == "__main__":
    unittest.main()
