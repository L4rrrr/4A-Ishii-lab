"""Headless publication of plots and a Pillow-based GIF; ffmpeg is unnecessary."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from .simulation import radius_graph


def _limits(result):
    points = result.positions.reshape(-1, 2)
    low, high = points.min(axis=0), points.max(axis=0)
    middle = (low + high) / 2
    half = max(float(np.max(high - low)) * 0.57, 1.0)
    return middle - half, middle + half


def plot_overview(result, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    history, faulty = result.positions, result.faulty
    steps = np.arange(len(history))
    xy, xplot, yplot, error = axes.flat
    for i in range(result.config.n):
        color = "#dc4b40" if faulty[i] else "#2367a3"
        style = "--" if faulty[i] else "-"
        xy.plot(history[:, i, 0], history[:, i, 1], style, c=color, alpha=0.55, lw=1)
        xplot.plot(steps, history[:, i, 0], style, c=color, alpha=0.55, lw=1)
        yplot.plot(steps, history[:, i, 1], style, c=color, alpha=0.55, lw=1)
    xy.scatter(*history[0, ~faulty].T, marker="o", facecolors="none", edgecolors="#2367a3",
               s=36, label="Normal: start")
    xy.scatter(*history[-1, ~faulty].T, c="#2367a3", s=28, label="Normal: end", zorder=4)
    if faulty.any():
        xy.scatter(*history[0, faulty].T, marker="s", facecolors="none",
                   edgecolors="#dc4b40", s=60, label="Faulty: start")
        xy.scatter(*history[-1, faulty].T, marker="x", c="#dc4b40", s=60,
                   label="Faulty: end", zorder=5)
    low, high = _limits(result)
    xy.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="x [m]", ylabel="y [m]",
           title="Trajectories (red dashed: faulty)")
    xy.set_aspect("equal")
    xy.legend(fontsize=8, loc="best")
    xplot.set(title="x coordinates", xlabel="Step k", ylabel="x [m]")
    yplot.set(title="y coordinates", xlabel="Step k", ylabel="y [m]")
    # A log axis cannot display exact zero; the floor affects the figure only.
    error.semilogy(steps, np.maximum(result.diameter, 1e-15), color="#2367a3",
                   label="Normal pairwise diameter")
    error.axhline(result.config.tolerance, c="#dc4b40", ls="--", label="Tolerance")
    error.set(title="Gathering error (normal agents only)", xlabel="Step k", ylabel="Diameter [m]")
    error.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    c = result.config
    fig.suptitle(f"Coordinate-wise W-MSR | n={c.n}, f={c.f}, r={c.radius:g} m | "
                 f"faults={c.fault_count} ({c.fault_mode}), seed={c.seed}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_animation(result, path, stride=2, fps=10):
    if stride < 1 or fps < 1:
        raise ValueError("stride and fps must be positive")
    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    history, faulty = result.positions, result.faulty
    low, high = _limits(result)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    edges, = ax.plot([], [], c="#b4bdc6", lw=0.6, alpha=0.45, zorder=1)
    normal_points = ax.scatter([], [], c="#2367a3", s=40, label="Normal", zorder=4)
    fault_points = ax.scatter([], [], c="#dc4b40", s=65, marker="x", label="Faulty", zorder=4)
    reference = int(np.flatnonzero(~faulty)[0])
    circle = Circle(history[0, reference], result.config.radius, fill=False, ls="--",
                    color="#2367a3", alpha=0.45, label=f"r around normal #{reference}")
    ax.add_patch(circle)
    labels = [ax.text(0, 0, str(i), fontsize=8, zorder=5) for i in range(result.config.n)]
    cluster_label = ax.text(0, 0, "", fontsize=9, color="#2367a3", zorder=6)
    ax.legend(loc="upper right", fontsize=8)
    title = ax.set_title("")
    frames = list(range(0, len(history), stride))
    if frames[-1] != len(history) - 1:
        frames.append(len(history) - 1)

    def update(k):
        positions = history[k]
        normal_points.set_offsets(positions[~faulty])
        fault_points.set_offsets(positions[faulty].reshape(-1, 2))
        graph = radius_graph(positions, result.config.radius)
        segments = []
        for i, j in zip(*np.where(np.triu(graph, k=1))):
            segments.extend([positions[i], positions[j], [np.nan, np.nan]])
        if segments:
            segments = np.asarray(segments)
            edges.set_data(segments[:, 0], segments[:, 1])
        else:
            edges.set_data([], [])
        circle.center = positions[reference]
        clustered = result.diameter[k] < 0.03 * (high[0] - low[0])
        for i, (label, position) in enumerate(zip(labels, positions)):
            label.set_position(position + 0.06)
            label.set_visible(bool(faulty[i] or not clustered))
        cluster_label.set_visible(clustered)
        cluster_label.set_position(positions[~faulty].mean(axis=0) + 0.15)
        cluster_label.set_text(f"{int((~faulty).sum())} normal agents")
        c = result.config
        title.set_text(f"{c.fault_mode}, faults={c.fault_count} | k={k}\n"
                       f"Normal diameter = {result.diameter[k]:.5g} m | r={c.radius:g} m")
        return normal_points, fault_points, edges, circle, title, cluster_label, *labels

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def plot_messages(result, path):
    """Compare Byzantine messages delivered to two normal receivers."""
    if result.config.fault_mode != "byzantine" or not result.faulty.any() or result.config.steps == 0:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
    steps = np.arange(result.config.steps)
    senders = np.flatnonzero(result.faulty)
    graphs = np.array([radius_graph(p, result.config.radius) for p in result.positions[:-1]])
    candidates = np.flatnonzero(~result.faulty)
    # Prefer one receiver from each split group that actually sees the sender.
    receivers = []
    for parity in (0, 1):
        group = [int(i) for i in candidates if i % 2 == parity]
        if group:
            receivers.append(max(group, key=lambda i: int(graphs[:, i, senders].sum())))
    if len(receivers) == 1 and len(candidates) > 1:
        receivers.append(next(int(i) for i in candidates if i not in receivers))
    active = np.any(graphs[:, receivers][:, :, senders], axis=(1, 2))
    active_steps = np.flatnonzero(active)
    last_visible = int(active_steps[-1]) if len(active_steps) else 0
    right = max(1, min(last_visible + 2, result.config.steps - 1))
    normal_history = result.positions[:-1, ~result.faulty]
    for axis, ax in enumerate(axes):
        ax.fill_between(steps, normal_history[:, :, axis].min(axis=1),
                        normal_history[:, :, axis].max(axis=1), color="#2367a3", alpha=0.15,
                        label="Current normal coordinate range")
        for receiver in receivers:
            for j, sender in enumerate(senders):
                values = result.fault_reports[:, receiver, j, axis].copy()
                values[~graphs[:, receiver, sender]] = np.nan
                ax.plot(steps, values, ".-", markersize=4, lw=1.2,
                        label=f"Reported: sender {sender} to receiver {receiver}")
        for sender in senders:
            ax.plot(steps, result.positions[:-1, sender, axis], "--", color="#555555",
                    alpha=0.7, label=f"Physical: sender {sender}")
        ax.set_ylabel(f"{'xy'[axis]} [m]")
        ax.set_xlim(0, right)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Step k (message used for k to k+1)")
    fig.suptitle(f"Byzantine reports ({result.config.attack_strategy}); only visible messages shown")
    fig.savefig(path, dpi=160)
    plt.close(fig)
