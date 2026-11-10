"""Convex hulls and the two-dimensional f-safe region."""

import numpy as np


def convex_hull(points):
    """Counterclockwise vertices, or the endpoints of a point/segment hull."""
    unique = sorted(set(map(tuple, points)))
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float).reshape(-1, 2)

    def chain(ordered):
        vertices = []
        for c in ordered:
            while len(vertices) >= 2:
                a, b = vertices[-2:]
                if (b[0] - a[0]) * (c[1] - a[1]) > (b[1] - a[1]) * (c[0] - a[0]):
                    break
                vertices.pop()
            vertices.append(c)
        return vertices

    return np.array(chain(unique)[:-1] + chain(reversed(unique))[:-1])


def _clip(vertices, direction, bound):
    """Intersect a convex polygon, segment, or point with x·direction <= bound."""
    signed = vertices @ direction - bound
    inside = signed <= 0
    if inside.all():
        return vertices
    if not inside.any():
        return np.empty((0, 2))
    clipped = []
    for i, point in enumerate(vertices):
        j = i - 1
        if inside[i] != inside[j]:
            fraction = signed[j] / (signed[j] - signed[i])
            intersection = vertices[j] + fraction * (point - vertices[j])
            clipped.append(intersection)
        if inside[i]:
            clipped.append(point)
    return convex_hull(clipped)


def guarded_hull(center_hull, positions, f):
    """Intersect position hulls after f neighbor deletions; positions[0] is trusted."""
    origin = positions[0]
    points = positions - origin
    vertices = center_hull - origin
    a, b = np.triu_indices(len(points), k=1)
    differences = points[b] - points[a]
    lengths = np.hypot(differences[:, 0], differences[:, 1])
    edges = differences[lengths > 0] / lengths[lengths > 0, None]
    perpendicular = np.column_stack((-edges[:, 1], edges[:, 0]))
    # Every polygon edge is parallel to a pair of inputs. Pair directions also
    # bound segments at their endpoints, and the coordinate axes bound points.
    directions = np.vstack((np.eye(2), edges, perpendicular))
    directions = np.unique(np.vstack((directions, -directions)), axis=0)
    projections = directions @ points.T
    # If fewer than f neighbors are visible, the most conservative subset
    # retains only the agent's trusted own position.
    index = len(points) - f - 2
    bounds = projections[:, 0]
    if index >= 0:
        neighbor_bound = np.partition(projections[:, 1:], index, axis=1)[:, index]
        bounds = np.maximum(bounds, neighbor_bound)
    tolerance = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(points))))
    for direction, bound in zip(directions, bounds):
        vertices = _clip(vertices, direction, bound + tolerance)
        if not len(vertices):
            break
    return vertices + origin


def distance_to_hull(points, hull):
    """Euclidean distance to a nonempty point, segment, or polygon hull."""
    if len(hull) == 1:
        return np.linalg.norm(points - hull[0], axis=1)
    edges = np.roll(hull, -1, axis=0) - hull
    offsets = points[:, None, :] - hull
    t = np.sum(offsets * edges, axis=2) / np.sum(edges**2, axis=1)
    residual = offsets - np.clip(t, 0, 1)[..., None] * edges
    distance = np.linalg.norm(residual, axis=2).min(axis=1)
    if len(hull) > 2:
        cross = edges[:, 0] * offsets[..., 1] - edges[:, 1] * offsets[..., 0]
        distance[np.all(cross >= 0, axis=1)] = 0
    return distance
