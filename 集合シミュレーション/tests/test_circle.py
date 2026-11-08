import unittest

import numpy as np
from numpy.testing import assert_allclose

from wmsr import Config, mec_targets, minimum_enclosing_circle, simulate


class MinimumCircleTests(unittest.TestCase):
    def test_duplicate_point_has_zero_radius(self):
        center, radius = minimum_enclosing_circle([[3, -4], [3, -4]])
        assert_allclose(center, [3, -4])
        self.assertEqual(radius, 0)

    def test_collinear_and_obtuse_sets_use_farthest_pair(self):
        for points in ([[0, 0], [1, 0], [4, 0]], [[0, 0], [1, 1], [4, 0]]):
            with self.subTest(points=points):
                center, radius = minimum_enclosing_circle(points)
                assert_allclose(center, [2, 0])
                self.assertEqual(radius, 2)

    def test_acute_triangle_uses_three_points(self):
        center, radius = minimum_enclosing_circle([[0, 0], [4, 0], [2, 3], [2, 1]])
        assert_allclose(center, [2, 5 / 6])
        self.assertAlmostEqual(radius, 13 / 6)

    def test_local_visibility_and_synchronous_partial_step(self):
        positions = [[0, 0], [2, 0], [4, 0], [20, 0]]
        result = simulate(Config(n=4, f=0, radius=2, steps=1, algorithm="mec", move_fraction=0.5),
                          positions)
        assert_allclose(result.circle_centers[0], [[1, 0], [2, 0], [3, 0], [20, 0]])
        assert_allclose(result.circle_radii[0], [1, 2, 1, 0])
        assert_allclose(result.positions[1], [[0.5, 0], [2, 0], [3.5, 0], [20, 0]])

    def test_mec_uses_visible_report_and_trusted_own_position(self):
        positions = np.array([[0, 0], [1, 0], [10, 0]])
        reports = np.broadcast_to(positions, (3, 3, 2)).copy()
        reports[0] = [[100, 0], [8, 0], [-1, 0]]
        centers, radii = mec_targets(positions, 2, reports)
        assert_allclose(centers[0], [4, 0])
        self.assertEqual(radii[0], 4)
        assert_allclose(centers[2], positions[2])

    def test_byzantine_comparison_uses_same_initial_state_but_can_leave_normal_box(self):
        positions = [[0, 0], [1, 0], [2, 0], [3, 0]]
        settings = dict(n=4, f=1, radius=5, steps=1, fault_count=1)
        wmsr = simulate(Config(**settings), positions)
        mec = simulate(Config(**settings, algorithm="mec"), positions)
        assert_allclose(mec.positions[0], wmsr.positions[0])
        assert_allclose(mec.fault_reports, wmsr.fault_reports)
        self.assertEqual(wmsr.box_violation[-1], 0)
        self.assertGreater(mec.box_violation[-1], 0)
