import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from wmsr import Config, retained_indices, simulate, wmsr_step
from wmsr.simulation import radius_graph


class FilteringTests(unittest.TestCase):
    def test_up_to_f_on_each_side_own_and_equal_values_survive(self):
        values = [-100, 0, 0, 1, 2, 3]
        assert_array_equal(retained_indices(0, values, 2), [1, 2, 3])
        assert_array_equal(retained_indices(0, [-1, 1], 5), [])

    def test_f_zero_keeps_every_neighbor(self):
        assert_array_equal(retained_indices(0, [3, -1, 0], 0), [0, 1, 2])

    def test_stable_tie_removal(self):
        assert_array_equal(retained_indices(0, [-1, -1, 1, 1], 1), [1, 2])

    def test_axes_use_different_retained_senders(self):
        points = [[0, 0], [-2, 1], [1, 3], [3, -2]]
        # x retains sender 2; y retains sender 1. Own is in BOTH means.
        assert_allclose(wmsr_step(points, f=1, radius=10)[0], [0.5, 0.5])

    def test_synchronous_update(self):
        points = [[0, 0], [2, 0], [4, 0]]
        assert_allclose(wmsr_step(points, f=0, radius=2), [[1, 0], [2, 0], [3, 0]])

    def test_isolated_and_radius_boundary(self):
        points = np.array([[0., 0.], [3., 4.], [50., 50.]])
        graph = radius_graph(points, 5)
        self.assertTrue(graph[0, 1])
        self.assertFalse(graph[0, 0])
        assert_allclose(wmsr_step(points, 0, 5), [[1.5, 2], [1.5, 2], [50, 50]])

    def test_no_common_gathering_despite_connected_visibility(self):
        # Knowing an unnecessarily large f can freeze even fault-free robots.
        points = [[0, 0], [1, 1]]
        result = simulate(Config(n=2, f=1, radius=5, steps=10), points)
        assert_allclose(result.positions[-1], points)
        self.assertEqual(result.components.max(), 1)
        self.assertFalse(result.summary()["within_tolerance_at_final_step"])


class ByzantineTests(unittest.TestCase):
    def test_sensing_uses_physical_position_not_report(self):
        points = np.array([[0., 0.], [1., 0.], [100., 0.]])
        reports = np.broadcast_to(points, (3, 3, 2)).copy()
        reports[:, 2] = [0.5, 0]  # far-away sender pretends to be nearby
        reports[0, 0] = [999, 999]  # own value must never be taken from reports
        updated = wmsr_step(points, 0, 2, reports)
        assert_allclose(updated[0], [0.5, 0])

    def test_nearby_sender_can_lie_far_outside_radius(self):
        points = np.array([[0., 0.], [1., 0.], [2., 0.]])
        reports = np.broadcast_to(points, (3, 3, 2)).copy()
        reports[0, 1] = [100, 100]
        reports[2, 1] = [-100, -100]
        updated = wmsr_step(points, 0, 5, reports)
        assert_allclose(updated[0], [34, 100 / 3])
        assert_allclose(updated[2], [-98 / 3, -100 / 3])

    def test_extreme_byzantine_removed_without_knowing_fault_identity(self):
        points = np.array([[0., 0.], [1., 2.], [2., 4.], [3., 1.], [4., 3.]])
        reports = np.broadcast_to(points, (5, 5, 2)).copy()
        reports[1::2, 0] = [100, 100]
        reports[::2, 0] = [-100, -100]
        updated = wmsr_step(points, 1, 10, reports)
        lower, upper = points[1:].min(axis=0), points[1:].max(axis=0)
        self.assertTrue(np.all(updated[1:] >= lower))
        self.assertTrue(np.all(updated[1:] <= upper))

    def test_byzantine_receiver_specific_reports_and_physical_freeze(self):
        result = simulate(Config(n=8, f=1, radius=15, steps=10,
                                 fault_count=1, fault_mode="byzantine"))
        self.assertFalse(np.allclose(result.fault_reports[0, 1], result.fault_reports[0, 2]))
        assert_allclose(result.positions[:, 0], np.tile(result.positions[0, 0], (11, 1)))
        self.assertLessEqual(result.box_violation.max(), 1e-12)

    def test_all_attack_strategies_preserve_normal_initial_box(self):
        for strategy in ("split", "random", "inside"):
            with self.subTest(strategy=strategy):
                result = simulate(Config(n=14, f=2, fault_count=2, steps=40,
                                         radius=6, attack_strategy=strategy))
                self.assertLessEqual(result.box_violation.max(), 1e-12)


class SimulationTests(unittest.TestCase):
    def test_f_zero_complete_graph_reaches_initial_centroid_in_one_step(self):
        initial = np.array([[0., 0.], [2., 0.], [1., 3.]])
        result = simulate(Config(n=3, f=0, radius=10, steps=3), initial)
        assert_allclose(result.positions[1], np.tile(initial.mean(axis=0), (3, 1)))
        self.assertEqual(result.summary()["within_tolerance_through_remaining_recorded_steps_from"], 1)

    def test_random_motion_length_and_seed_reproducibility(self):
        config = Config(n=8, f=1, steps=10, fault_count=1,
                        fault_mode="random", fault_step=0.7)
        first, second = simulate(config), simulate(config)
        assert_array_equal(first.positions, second.positions)
        assert_allclose(np.linalg.norm(np.diff(first.positions[:, 0], axis=0), axis=1), 0.7)
        assert_allclose(first.fault_reports[:, 1, 0], first.positions[:-1, 0])

    def test_initial_state_is_shared_across_scenarios(self):
        clean = simulate(Config(n=8, steps=2))
        faulty = simulate(Config(n=8, steps=2, fault_count=1, fault_mode="random"))
        assert_array_equal(clean.positions[0], faulty.positions[0])

    def test_one_agent_and_zero_steps(self):
        result = simulate(Config(n=1, f=0, steps=0), [[1, 2]])
        self.assertEqual(result.diameter[0], 0)
        self.assertEqual(result.fault_reports.shape, (0, 1, 0, 2))
        self.assertEqual(result.min_normal_degree[0], 0)
        self.assertTrue(result.summary()["within_tolerance_at_final_step"])

    def test_zero_radius_coincident_neighbors(self):
        points = [[1, 1], [1, 1], [2, 2]]
        self.assertTrue(radius_graph(points, 0)[0, 1])
        assert_allclose(wmsr_step(points, 0, 0), points)

    def test_explicit_fault_ids_and_stuck_behavior(self):
        result = simulate(Config(n=6, f=2, fault_count=2, faulty_ids=(2, 5),
                                 fault_mode="stuck", steps=4))
        assert_array_equal(np.flatnonzero(result.faulty), [2, 5])
        for agent in [2, 5]:
            assert_allclose(result.positions[:, agent], np.tile(result.positions[0, agent], (5, 1)))

    def test_validation(self):
        cases = [dict(n=0), dict(f=-1), dict(n=2, f=2), dict(radius=-1),
                 dict(radius=float("nan")), dict(fault_count=2, f=1), dict(n=True),
                 dict(steps=1.5), dict(fault_count=1, fault_mode="none"),
                 dict(fault_count=2, f=2, faulty_ids=(0, 0)),
                 dict(fault_count=1, faulty_ids=(100,)), dict(attack_strategy="unknown")]
        for params in cases:
            with self.subTest(params=params), self.assertRaises(ValueError):
                Config(**params)
        with self.assertRaises(ValueError):
            simulate(Config(n=3), [[0, 0]])
        with self.assertRaises(ValueError):
            simulate(Config(n=3), [[0, 0], [1, float("nan")], [2, 2]])


if __name__ == "__main__":
    unittest.main()
