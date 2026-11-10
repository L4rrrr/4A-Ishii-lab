"""Synchronous, point-agent simulation. Lengths are in metres; time is a step index."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .circle import minimum_enclosing_circle
from .hull import convex_hull, distance_to_hull, guarded_hull

ALGORITHM_NAMES = {"wmsr": "W-MSR", "mec": "最小包含円",
                   "mec-hull": "MEC中心の凸包", "mec-hull-safe": "安全領域つき凸包"}
FAULT_NAMES = {"none": "故障なし", "byzantine": "ビザンチン故障", "stuck": "停止故障", "random": "ランダム移動故障"}


@dataclass(frozen=True)
class Config:
    n: int = 20
    f: int = 1
    radius: float = 6.0
    steps: int = 100
    seed: int = 7
    box_size: float = 10.0
    fault_count: int = 0
    faulty_ids: Optional[Tuple[int, ...]] = None
    fault_mode: str = "byzantine"
    fault_step: float = 1.0
    attack_strategy: str = "split"
    attack_amplitude: float = 20.0
    tolerance: float = 0.001
    algorithm: str = "wmsr"
    move_fraction: float = 1.0

    def __post_init__(self):
        for name in ("n", "f", "steps", "seed", "fault_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.n < 1 or not 0 <= self.f < self.n:
            raise ValueError("require n >= 1 and 0 <= f < n")
        if self.steps < 0 or self.seed < 0:
            raise ValueError("steps and seed must be nonnegative")
        for name in ("radius", "box_size", "fault_step", "tolerance", "attack_amplitude", "move_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.box_size == 0 or self.tolerance == 0:
            raise ValueError("box_size and tolerance must be positive")
        if self.algorithm not in ALGORITHM_NAMES:
            raise ValueError(f"algorithm は {', '.join(ALGORITHM_NAMES)} から指定してください")
        if not 0 < self.move_fraction <= 1:
            raise ValueError("move_fraction は 0 より大きく 1 以下にしてください")
        if self.fault_mode not in ("none", "stuck", "random", "byzantine"):
            raise ValueError("fault_mode must be none, stuck, random, or byzantine")
        if self.attack_strategy not in ("split", "random", "inside"):
            raise ValueError("attack_strategy must be split, random, or inside")
        if not 0 <= self.fault_count <= self.f:
            raise ValueError("require 0 <= fault_count <= f")
        if self.faulty_ids is not None:
            try:
                ids = tuple(self.faulty_ids)
            except TypeError as exc:
                raise ValueError("faulty_ids must be an array of integer IDs") from exc
            if any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
                raise ValueError("faulty_ids must contain integers")
            if len(set(ids)) != len(ids) or any(i < 0 or i >= self.n for i in ids):
                raise ValueError("faulty_ids must be unique IDs in [0, n)")
            if len(ids) != self.fault_count:
                raise ValueError("fault_count must match the length of faulty_ids")
            object.__setattr__(self, "faulty_ids", ids)
        if self.fault_mode == "none" and self.fault_count:
            raise ValueError("fault_mode=none requires fault_count=0")

    def fault_mask(self):
        mask = np.zeros(self.n, dtype=bool)
        ids = self.faulty_ids if self.faulty_ids is not None else range(self.fault_count)
        mask[list(ids)] = True
        return mask


def _positions(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise ValueError("positions must have shape (n, 2), with n >= 1")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must be finite")
    return array


def radius_graph(positions, radius):
    """True for visible distinct agents, including the boundary distance r."""
    positions = _positions(positions)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    delta = positions[:, None, :] - positions[None, :, :]
    graph = np.hypot(delta[..., 0], delta[..., 1]) <= radius
    np.fill_diagonal(graph, False)
    return graph


def retained_indices(own, neighbor_values, f):
    """Indices into neighbor_values after standard W-MSR filtering.

    Remove up to f strictly smaller values and up to f strictly larger
    values relative to own. Equal values stay; own is handled separately.
    Sorting is stable so ties are reproducible.
    """
    values = np.asarray(neighbor_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(own) or not np.all(np.isfinite(values)):
        raise ValueError("own and one-dimensional neighbor_values must be finite")
    if isinstance(f, bool) or not isinstance(f, int) or f < 0:
        raise ValueError("f must be a nonnegative integer")
    keep = np.ones(len(values), dtype=bool)
    if f:
        order = np.argsort(values, kind="stable")
        below = order[values[order] < own]
        above = order[values[order] > own]
        keep[below[:f]] = False
        keep[above[-f:]] = False
    return np.flatnonzero(keep)


def _validated_reports(positions, received_positions):
    if received_positions is None:
        return None
    reports = np.asarray(received_positions, dtype=float)
    if reports.shape != (len(positions), len(positions), 2) or not np.all(np.isfinite(reports)):
        raise ValueError("received_positions must be finite with shape (n, n, 2)")
    return reports


def wmsr_step(positions, f, radius, received_positions=None):
    """Compute all candidates from the SAME old state; retain own on both axes.

    received_positions[i, j] is the position reported by sender j to receiver i.
    The physical positions ALWAYS determine visibility. Own is trusted and is
    read from positions, ignoring any report in received_positions[i, i].
    This function never uses fault identities. The simulator overrides the
    faulty candidates afterwards to model their physical motion.
    """
    positions = _positions(positions)
    if isinstance(f, bool) or not isinstance(f, int) or not 0 <= f < len(positions):
        raise ValueError("require integer 0 <= f < n")
    graph = radius_graph(positions, radius)
    reports = _validated_reports(positions, received_positions)
    updated = positions.copy()
    for i in range(len(positions)):
        neighbors = np.flatnonzero(graph[i])
        for axis in range(2):
            values = (positions[neighbors, axis] if reports is None
                      else reports[i, neighbors, axis])
            retained = retained_indices(positions[i, axis], values, f)
            # Uniform positive weights, including the agent's own value.
            selected = np.concatenate(([positions[i, axis]], values[retained]))
            updated[i, axis] = np.mean(selected)
    return updated


def mec_targets(positions, radius, received_positions=None):
    """Compute each agent's circle from its true own position and all visible reports."""
    positions = _positions(positions)
    graph = radius_graph(positions, radius)
    reports = _validated_reports(positions, received_positions)
    centers = np.empty_like(positions)
    radii = np.empty(len(positions))
    for i in range(len(positions)):
        neighbors = np.flatnonzero(graph[i])
        visible = positions[neighbors] if reports is None else reports[i, neighbors]
        points = np.vstack((positions[i], visible))
        centers[i], radii[i] = minimum_enclosing_circle(points)
    return centers, radii


def _byzantine_reports(values, config, faulty, rng):
    """One message per sender and recipient, preserving the existing attack modes."""
    reports = np.broadcast_to(values, (config.n, config.n, 2)).copy()
    if config.fault_mode != "byzantine" or not faulty.any():
        return reports
    lower, upper = values[~faulty].min(axis=0), values[~faulty].max(axis=0)
    for receiver in range(config.n):
        if config.attack_strategy == "random":
            reports[receiver, faulty] = values[faulty] + rng.uniform(
                -config.attack_amplitude, config.attack_amplitude, (config.fault_count, 2))
        elif config.attack_strategy == "inside":
            fraction = 0.25 if receiver % 2 == 0 else 0.75
            reports[receiver, faulty] = lower + fraction * (upper - lower)
        else:
            reports[receiver, faulty] = (lower - config.attack_amplitude if receiver % 2 == 0
                                         else upper + config.attack_amplitude)
    return reports


def _hull_targets(positions, centers, graph, position_reports, center_reports, f, guarded):
    targets = np.empty_like(positions)
    held = np.zeros(len(positions), dtype=bool)
    for i, visible in enumerate(graph):
        values = np.vstack((centers[i], center_reports[i, visible]))
        if not guarded:
            targets[i] = values.mean(axis=0)
            continue
        local_positions = np.vstack((positions[i], position_reports[i, visible]))
        region = guarded_hull(convex_hull(values), local_positions, f)
        held[i] = not len(region)
        targets[i] = positions[i] if held[i] else region.mean(axis=0)
    return targets, held


def component_count(graph):
    """Connected components of the undirected visibility graph."""
    unseen = set(range(len(graph)))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            found = unseen.intersection(np.flatnonzero(graph[node]).tolist())
            unseen.difference_update(found)
            stack.extend(found)
    return count


@dataclass
class Result:
    config: Config
    positions: np.ndarray  # (steps + 1, n, 2), including the initial state
    faulty: np.ndarray
    diameter: np.ndarray  # maximum pairwise distance among NORMAL agents
    axis_span: np.ndarray
    normal_centroid: np.ndarray
    min_normal_degree: np.ndarray  # visible neighbors, including faulty ones
    components: np.ndarray  # whole visibility graph
    box_violation: np.ndarray  # normal coordinates outside initial normal box
    fault_reports: np.ndarray  # (steps, receivers, faulty senders, 2); ID order ascending
    circle_centers: Optional[np.ndarray] = None  # (steps, n, 2), MEC only
    circle_radii: Optional[np.ndarray] = None  # (steps, n), MEC only
    hull_violation: Optional[np.ndarray] = None  # distance outside initial normal convex hull
    update_targets: Optional[np.ndarray] = None  # (steps, n, 2), center-hull methods
    center_fault_reports: Optional[np.ndarray] = None  # same layout as fault_reports
    held: Optional[np.ndarray] = None  # empty guarded region at this update

    def summary(self):
        below = self.diameter <= self.config.tolerance
        # Require the criterion to hold for every remaining recorded state.
        # This describes only the finite simulation horizon, not a proof.
        suffix_below = np.logical_and.accumulate(below[::-1])[::-1]
        hits = np.flatnonzero(suffix_below)
        summary = {
            "normal_count": int((~self.faulty).sum()),
            "faulty_ids": np.flatnonzero(self.faulty).tolist(),
            "initial_normal_diameter_m": float(self.diameter[0]),
            "final_normal_diameter_m": float(self.diameter[-1]),
            "tolerance_m": self.config.tolerance,
            "within_tolerance_at_final_step": bool(below[-1]),
            "within_tolerance_through_remaining_recorded_steps_from":
                int(hits[0]) if len(hits) else None,
            "final_normal_centroid_m": self.normal_centroid[-1].tolist(),
            "max_initial_normal_box_violation_m": float(self.box_violation.max()),
            "max_visibility_components": int(self.components.max()),
            "note": "Finite-horizon observation; graph robustness is not certified.",
        }
        if self.hull_violation is not None:
            summary["max_initial_normal_hull_violation_m"] = float(self.hull_violation.max())
        if self.held is not None:
            summary["normal_empty_guard_updates"] = int(self.held[:, ~self.faulty].sum())
        return summary


def simulate(config=Config(), initial_positions=None):
    """Run a fixed horizon; faulty robots remain physically visible.

    stuck: remain at their initial positions, still visible to normal robots.
    random: fixed-length random walk, independent direction per step/robot.
    byzantine: physical positions stay fixed; reported positions depend on the
    receiver and need not match the physical position. This models messages,
    not perfect direct observation of true physical coordinates.
    No collisions or physical dynamics are modeled.
    """
    initial_seed, fault_seed, center_seed = np.random.SeedSequence(config.seed).spawn(3)
    initial_rng = np.random.default_rng(initial_seed)
    fault_rng = np.random.default_rng(fault_seed)
    center_rng = np.random.default_rng(center_seed)
    if initial_positions is None:
        initial = initial_rng.uniform(0, config.box_size, (config.n, 2))
    else:
        initial = _positions(initial_positions).copy()
        if initial.shape != (config.n, 2):
            raise ValueError("initial_positions must match config.n")
    faulty = config.fault_mask()
    normal = ~faulty
    history = np.empty((config.steps + 1, config.n, 2))
    fault_reports = np.empty((config.steps, config.n, config.fault_count, 2))
    circle_centers = circle_radii = None
    if config.algorithm != "wmsr":
        circle_centers = np.empty((config.steps, config.n, 2))
        circle_radii = np.empty((config.steps, config.n))
    targets_history = center_fault_history = held_history = None
    if config.algorithm in ("mec-hull", "mec-hull-safe"):
        targets_history = np.empty_like(circle_centers)
        center_fault_history = np.empty_like(fault_reports)
    if config.algorithm == "mec-hull-safe":
        held_history = np.zeros((config.steps, config.n), dtype=bool)
    history[0] = initial
    for k in range(config.steps):
        old = history[k]
        reports = _byzantine_reports(old, config, faulty, fault_rng)
        fault_reports[k] = reports[:, faulty, :]
        if circle_centers is None:
            new = wmsr_step(old, config.f, config.radius, reports)
        else:
            centers, radii = mec_targets(old, config.radius, reports)
            circle_centers[k], circle_radii[k] = centers, radii
            targets = centers
            if targets_history is not None:
                center_reports = _byzantine_reports(centers, config, faulty, center_rng)
                if not np.all(np.isfinite(center_reports)):
                    raise ValueError("MEC中心の報告に有限でない値があります")
                targets, held = _hull_targets(old, centers, radius_graph(old, config.radius),
                                              reports, center_reports, config.f,
                                              held_history is not None)
                targets_history[k] = targets
                center_fault_history[k] = center_reports[:, faulty, :]
                if held_history is not None:
                    held_history[k] = held
            fraction = config.move_fraction
            new = (1 - fraction) * old + fraction * targets
        if config.fault_mode == "random":
            angles = fault_rng.uniform(0, 2 * np.pi, config.fault_count)
            displacement = config.fault_step * np.column_stack((np.cos(angles), np.sin(angles)))
            new[faulty] = old[faulty] + displacement
        else:
            new[faulty] = old[faulty]
        if not np.all(np.isfinite(new)):
            raise ValueError("non-finite state: reduce coordinate magnitudes or fault_step")
        history[k + 1] = new

    length = config.steps + 1
    diameter = np.empty(length)
    spans = np.empty((length, 2))
    centroids = np.empty((length, 2))
    degrees = np.empty(length, dtype=int)
    components = np.empty(length, dtype=int)
    violations = np.empty(length)
    hull_violations = np.empty(length)
    initial_hull = convex_hull(initial[normal])
    lower, upper = initial[normal].min(axis=0), initial[normal].max(axis=0)
    for k, positions in enumerate(history):
        good = positions[normal]
        delta = good[:, None, :] - good[None, :, :]
        diameter[k] = np.hypot(delta[..., 0], delta[..., 1]).max()
        spans[k] = np.ptp(good, axis=0)
        centroids[k] = good.mean(axis=0)
        graph = radius_graph(positions, config.radius)
        degrees[k] = graph.sum(axis=1)[normal].min()
        components[k] = component_count(graph)
        violations[k] = max(0.0, float(np.max(lower - good)), float(np.max(good - upper)))
        hull_violations[k] = distance_to_hull(good, initial_hull).max()
    return Result(config, history, faulty, diameter, spans, centroids, degrees,
                  components, violations, fault_reports, circle_centers, circle_radii,
                  hull_violations, targets_history, center_fault_history, held_history)
