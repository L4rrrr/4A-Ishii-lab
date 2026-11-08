"""Deterministic minimum enclosing circle for a finite planar point set."""

from itertools import combinations

import numpy as np


def minimum_enclosing_circle(points):
    """Return (center, radius) for a nonempty array of planar points.

    A feasible diameter circle of a farthest pair attains the lower bound
    given by that pair. Otherwise the optimum has three support points,
    so test the circumcircles of noncollinear triples.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("有限な二次元座標を1点以上指定してください")
    points = np.unique(points, axis=0)
    if len(points) == 1:
        return points[0], 0.0

    # Shift and scale before calculating squared distances and determinants.
    origin = points[0]
    shifted = points - origin
    scale = np.max(np.abs(shifted))
    if not np.isfinite(scale):
        raise ValueError("座標の差が大きすぎます。座標を縮尺してください")
    unit = shifted / scale
    differences = unit[:, None] - unit[None, :]
    distances2 = np.sum(differences**2, axis=2)
    i, j = np.unravel_index(np.argmax(distances2), distances2.shape)
    center = (unit[i] + unit[j]) / 2
    radius2 = distances2[i, j] / 4
    tolerance = 64 * np.finfo(float).eps

    def encloses(candidate, r2):
        return np.max(np.sum((unit - candidate)**2, axis=1)) <= r2 * (1 + tolerance)

    if not encloses(center, radius2):
        radius2 = np.inf
        for a, b, c in combinations(unit, 3):
            u, v = b - a, c - a
            determinant = 2 * (u[0] * v[1] - u[1] * v[0])
            if determinant == 0:
                continue
            u2, v2 = u @ u, v @ v
            # The optimal three-point circle has its center inside the triangle.
            # An obtuse triangle's longest edge already encloses its third point.
            if u @ v < 0 or u @ (u - v) < 0 or v @ (v - u) < 0:
                continue
            offset = np.array([u2 * v[1] - v2 * u[1], v2 * u[0] - u2 * v[0]]) / determinant
            candidate_r2 = offset @ offset
            candidate = a + offset
            if candidate_r2 < radius2 and encloses(candidate, candidate_r2):
                center, radius2 = candidate, candidate_r2
        if not np.isfinite(radius2):
            raise ValueError("点を含む最小円を数値的に求められませんでした")

    center = origin + center * scale
    # Account for roundoff when translating the center back to world coordinates.
    offsets = points - center
    radius = np.max(np.hypot(offsets[:, 0], offsets[:, 1]))
    return center, float(radius)
