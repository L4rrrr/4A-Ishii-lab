"""Readable MP4 comparisons with synchronized views and discrete simulation steps."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .hull import convex_hull
from .plotting import fault_name
from .simulation import ALGORITHM_NAMES, msr_center_mask, radius_graph


COLORS = {"wmsr": "#4a61a7", "mec": "#aa491f", "mec-hull": "#196a9c",
          "mec-hull-safe": "#137b5a", "mec-hull-msr": "#6f50a1"}
NORMAL = "#25729e"
FAULT = "#c84a3c"


def _msr_panel(ax, result):
    """Show the center reports and selected target of one normal receiver."""
    reference = int(np.flatnonzero(~result.faulty)[0])
    initial_graph = radius_graph(result.positions[0], result.config.radius)
    sees_fault = np.flatnonzero(~result.faulty & initial_graph[:, result.faulty].any(axis=1))
    if len(sees_fault):
        reference = int(sees_fault[0])
    reports = result.circle_centers.copy()
    reports[:, result.faulty] = result.center_fault_reports[:, reference]
    points = reports.reshape(-1, 2) if len(reports) else result.positions[0]
    low, high = points.min(axis=0), points.max(axis=0)
    midpoint = (low + high) / 2
    margin = max(float(np.max(high - low)) * .58, .5)
    ax.set(xlim=(midpoint[0] - margin, midpoint[0] + margin),
           ylim=(midpoint[1] - margin, midpoint[1] + margin), xlabel="中心のx座標 [m]",
           ylabel="中心のy座標 [m]", title=f"正常機 {reference} が受け取った全中心")
    ax.set_aspect("equal")
    ax.grid(color="#e0e6ed", lw=.7)
    ax.set_axisbelow(True)
    removed = ax.scatter([], [], s=75, color="#9b6c3e", marker="x", label="MSRで除いた中心", zorder=4)
    kept = ax.scatter([], [], s=64, facecolors="none", edgecolors="#6f50a1", label="残った中心", zorder=5)
    hull, = ax.plot([], [], color="#6f50a1", lw=1.5, alpha=.65, zorder=3)
    target = ax.scatter([], [], s=125, marker="*", color="#137b5a", label="移動先", zorder=6)
    caption = ax.text(.03, .97, "", transform=ax.transAxes, va="top", fontsize=10, zorder=7,
                      bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    ax.legend(loc="lower right", fontsize=8)

    def draw(k):
        if not len(reports):
            caption.set_text("更新なし")
            return
        last = k == result.config.steps
        step = min(k, result.config.steps - 1)
        values = reports[step]
        keep = msr_center_mask(values, result.config.f)
        removed.set_offsets(values[~keep])
        kept.set_offsets(values[keep])
        vertices = convex_hull(values[keep])
        if len(vertices):
            hull.set_data(*np.vstack((vertices, vertices[0])).T)
            target.set_offsets(result.update_targets[step, reference][None])
        else:
            hull.set_data([], [])
            target.set_offsets(np.empty((0, 2)))
        prefix = "直前の更新\n" if last else ""
        caption.set_text(f"{prefix}{len(values)}件中 {int(keep.sum())}件を残す" if keep.any()
                         else f"{prefix}中心が残らず、その場にとどまる")

    return draw


def plot_hull_comparison(results, path):
    """Show the first moves separately from the 100-step outcome."""
    c = results[0].config
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    first, all_steps, escape, connectivity = axes.flat
    floor = c.tolerance / 10
    for result in results:
        steps = np.arange(len(result.diameter))
        name = ALGORITHM_NAMES[result.config.algorithm]
        color = COLORS[result.config.algorithm]
        diameter = np.maximum(result.diameter, floor)
        for ax in (first, all_steps):
            ax.semilogy(steps, diameter, color=color, lw=2, label=name)
        escape.plot(steps, result.hull_violation, color=color, lw=2)
        connectivity.step(steps, result.components, where="post", color=color, lw=2)
    for ax in (first, all_steps):
        ax.axhline(c.tolerance, color="#70757e", ls="--", lw=1, label="集合のしきい値")
        ax.set_ylabel("正常機間の最大距離 [m]")
    first.set_title("序盤の集合")
    first.set_xlim(0, max(1, min(c.steps, 12)))
    first.xaxis.set_major_locator(MaxNLocator(integer=True))
    all_steps.set_title("全期間の集合")
    all_steps.legend(fontsize=9, loc="best")
    escape.set(title="正常機の初期凸包からの最大逸脱", ylabel="凸包の外への距離 [m]")
    connectivity.set(title="全エージェントの視認グラフ", ylabel="連結成分数")
    connectivity.yaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes.flat:
        ax.set_xlabel("更新ステップ")
        ax.grid(color="#dce1e7", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(f"MECと凸包による集合の比較　{fault_name(c)}　視認半径 {c.radius:g} m", fontsize=15)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_video(results, path):
    """Write a comparison MP4, showing early steps slowly and holding the final frame."""
    if FFMpegWriter.isAvailable():
        backend = "ffmpeg"
    else:
        try:
            import cv2
        except ImportError as exc:
            raise ValueError("MP4生成にはFFmpegまたは opencv-python が必要です") from exc
        backend = "opencv"

    path = Path(path)
    c = results[0].config
    fps, dpi = 12, 100
    fig = plt.figure(figsize=(16, 9), dpi=dpi, facecolor="#f4f7fa")
    msr_detail = len(results) == 1 and c.algorithm == "mec-hull-msr"
    grid = fig.add_gridspec(2, len(results) + int(msr_detail), height_ratios=[1.65, 1], left=.065,
                           right=.98, top=.83, bottom=.09, hspace=.36, wspace=.24)
    all_positions = np.concatenate([r.positions.reshape(-1, 2) for r in results])
    low, high = all_positions.min(axis=0), all_positions.max(axis=0)
    midpoint = (low + high) / 2
    half_width = max(0.58 * float(np.max(high - low)), 0.5)
    left, right = midpoint - half_width, midpoint + half_width
    header = fig.text(.065, .955, "", color="#17283b", fontsize=20, weight="bold")
    fig.text(.065, .912, f"視認半径 {c.radius:g} m　　全 {c.n} 体　　故障 {c.fault_count} 体　　上限 f={c.f}",
             color="#435465", fontsize=12)
    handles = [
        Line2D([], [], marker="o", ls="", color=NORMAL, label="正常機"),
        Line2D([], [], marker="x", ls="", color=FAULT, label="故障機"),
        Line2D([], [], ls="--", color="#758394", label="正常機の初期凸包"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.985, .941),
               ncols=3, frameon=False, fontsize=11)
    floor = c.tolerance / 10
    panels = []
    for i, result in enumerate(results):
        color = COLORS[result.config.algorithm]
        ax = fig.add_subplot(grid[0, i])
        ax.set_title(ALGORITHM_NAMES[result.config.algorithm], color=color, fontsize=15, pad=8)
        ax.set(xlim=(left[0], right[0]), ylim=(left[1], right[1]), xlabel="x座標 [m]")
        if i == 0:
            ax.set_ylabel("y座標 [m]")
        ax.set_aspect("equal")
        ax.set_axisbelow(True)
        ax.grid(color="#e0e6ed", lw=.7)
        hull = convex_hull(result.positions[0, ~result.faulty])
        hull = np.vstack((hull, hull[0]))
        ax.plot(*hull.T, ls="--", color="#758394", lw=1, alpha=.65)
        trail = LineCollection([], colors=NORMAL, lw=1.15, alpha=.30, zorder=2)
        ax.add_collection(trail)
        good = ax.scatter([], [], s=58, color=NORMAL, edgecolor="white", lw=.65, zorder=4)
        bad = ax.scatter([], [], s=85, color=FAULT, marker="x", lw=2.1, zorder=5)
        readout = ax.text(.04, .95, "", transform=ax.transAxes, va="top", fontsize=10, zorder=6,
                          bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
        panels.append((result, trail, good, bad, readout))
    show_msr = _msr_panel(fig.add_subplot(grid[0, 1]), results[0]) if msr_detail else None

    error = fig.add_subplot(grid[1, :])
    # Compact a long steady tail while keeping each of the early updates legible.
    split = min(12, c.steps)
    steps = np.arange(c.steps + 1)
    compressed = steps.astype(float)
    if c.steps > split:
        compressed[split:] = split + 8 * (steps[split:] - split) / (c.steps - split)
    lines, marks = [], []
    for result in results:
        color = COLORS[result.config.algorithm]
        series = np.maximum(result.diameter, floor)
        error.semilogy(compressed, series, color=color, alpha=.19, lw=2)
        line, = error.semilogy([], [], color=color, lw=2.4,
                               label=ALGORITHM_NAMES[result.config.algorithm])
        mark, = error.semilogy([], [], "o", color=color, markersize=6)
        lines.append((line, series))
        marks.append(mark)
    error.axhline(c.tolerance, color="#687584", ls="--", lw=1, label="集合のしきい値")
    ticks = sorted(set([0, min(3, c.steps), min(6, c.steps), split, c.steps]))
    error.set_xticks(compressed[ticks], labels=ticks)
    error.set_xlim(0, max(1, compressed[-1]))
    error.set(xlabel="更新ステップ（13以降は横幅を短縮）" if c.steps > split else "更新ステップ",
              ylabel="正常機間の最大距離 [m]", title="正常機がどのくらい近づいたか")
    error.grid(color="#e0e6ed", lw=.7)
    error.set_axisbelow(True)
    error.legend(ncols=2, fontsize=10, loc="upper right")

    states = list(range(split + 1))
    if c.steps > split:
        states.extend(np.unique(np.linspace(split + 1, c.steps, min(21, c.steps - split), dtype=int)))

    def draw(k):
        header.set_text(f"集合シミュレーション　{fault_name(c)}　　ステップ {k:>3} / {c.steps}")
        for result, trail, good, bad, label in panels:
            good.set_offsets(result.positions[k, ~result.faulty])
            bad.set_offsets(result.positions[k, result.faulty])
            paths = result.positions[max(0, k - 4):k + 1, ~result.faulty].transpose(1, 0, 2)
            trail.set_segments(paths)
            label.set_text(f"最大距離 {result.diameter[k]:.3g} m\n"
                           f"凸包の外へ {result.hull_violation[k]:.3g} m")
        for (line, series), mark in zip(lines, marks):
            line.set_data(compressed[:k + 1], series[:k + 1])
            mark.set_data([compressed[k]], [series[k]])
        if show_msr is not None:
            show_msr(k)

    try:
        with TemporaryDirectory(dir=path.parent) as tmp:
            output = Path(tmp) / "集合の動画.mp4"
            if backend == "ffmpeg":
                writer = FFMpegWriter(fps=fps, codec="h264",
                                      extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
                with writer.saving(fig, str(output), dpi):
                    for k in states:
                        draw(k)
                        for _ in range(18 if k == c.steps else 12 if k == 0 else 7 if k <= split else 3):
                            writer.grab_frame()
            else:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"avc1"),
                                         fps, fig.canvas.get_width_height())
                if not writer.isOpened():
                    writer.release()
                    raise ValueError("H.264動画の作成にはFFmpegをインストールしてください")
                try:
                    for k in states:
                        draw(k)
                        fig.canvas.draw()
                        frame = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
                        for _ in range(18 if k == c.steps else 12 if k == 0 else 7 if k <= split else 3):
                            writer.write(frame)
                finally:
                    writer.release()
            if not output.is_file() or output.stat().st_size == 0:
                raise ValueError("MP4動画を保存できませんでした")
            output.replace(path)
    finally:
        plt.close(fig)
