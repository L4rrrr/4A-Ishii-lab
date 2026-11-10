import unittest

import numpy as np
from numpy.testing import assert_allclose

from wmsr import Config, simulate
from wmsr.hull import convex_hull, distance_to_hull, guarded_hull


class HullUpdateTests(unittest.TestCase):
    def test_targets_use_centers_of_visible_neighbors_only(self):
        points = [[0, 0], [2, 0], [4, 0], [10, 0]]
        result = simulate(Config(n=4, f=0, radius=2, steps=1, algorithm="mec-hull"), points)
        assert_allclose(result.circle_centers[0], [[1, 0], [2, 0], [3, 0], [10, 0]])
        assert_allclose(result.positions[1], [[1.5, 0], [2, 0], [2.5, 0], [10, 0]])

    def test_empty_intersection_holds_own_position(self):
        points = [[0, 0], [2, 0]]
        result = simulate(Config(n=2, f=1, radius=2, steps=1, algorithm="mec-hull-safe"), points)
        assert_allclose(result.positions[1], points)
        self.assertEqual(result.summary()["normal_empty_guard_updates"], 2)

    def test_duplicate_reports_still_count_as_distinct_senders(self):
        reports = np.array([[0., 0.], [1., 0.], [1., 0.], [4., 3.]])
        region = guarded_hull(convex_hull(reports), reports, 1)
        assert_allclose(region[:, 1], 0, atol=1e-12)
        assert_allclose([region[:, 0].min(), region[:, 0].max()], [0, 1], atol=1e-12)

    def test_guard_stays_in_every_retained_hull(self):
        positions = np.array([[0., 0.], [4., 0.], [0., 4.], [40., 40.]])
        region = guarded_hull(convex_hull(positions), positions, 1)
        self.assertGreater(len(region), 0)
        for omitted in (1, 2, 3):
            retained = convex_hull(np.delete(positions, omitted, axis=0))
            self.assertLess(distance_to_hull(region, retained).max(), 1e-10)
        self.assertLess(distance_to_hull(np.array([[1., 1.]]), region).max(), 1e-10)

    def test_byzantine_reception_in_both_rounds_and_safe_gathering(self):
        config = Config(radius=15, fault_count=1, steps=20, algorithm="mec-hull-safe")
        result = simulate(config)
        self.assertFalse(np.allclose(result.fault_reports[0, 1], result.fault_reports[0, 2]))
        self.assertFalse(np.allclose(result.center_fault_reports[0, 1], result.center_fault_reports[0, 2]))
        self.assertLess(result.hull_violation.max(), 1e-10)
        self.assertLessEqual(result.diameter[-1], config.tolerance)

    def test_plain_center_hull_can_leave_initial_normal_hull(self):
        result = simulate(Config(radius=6, fault_count=1, steps=20, algorithm="mec-hull"))
        self.assertGreater(result.hull_violation.max(), 1.0)
