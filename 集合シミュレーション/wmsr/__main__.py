import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
from importlib.metadata import version

import numpy as np

from .simulation import ALGORITHM_NAMES, FAULT_NAMES, Config, radius_graph, simulate


def save_result(result, directory, gif=False, stride=2):
    from .plotting import plot_messages, plot_overview, save_animation

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(result.config), "summary": result.summary(),
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "matplotlib": version("matplotlib"), "pillow": version("Pillow")}}
    (directory / "run.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    history = {"positions": result.positions, "faulty": result.faulty,
               "fault_reports": result.fault_reports}
    if result.circle_centers is not None:
        history.update(circle_centers=result.circle_centers, circle_radii=result.circle_radii)
        with (directory / "最小包含円.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["更新ステップ", "正常機ID", "中心x座標 [m]", "中心y座標 [m]", "半径 [m]"])
            for k, centers in enumerate(result.circle_centers):
                for i in np.flatnonzero(~result.faulty):
                    writer.writerow([k, i, *centers[i], result.circle_radii[k, i]])
    np.savez_compressed(directory / "history.npz", **history)
    with (directory / "fault_messages.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "receiver_id", "sender_id", "receiver_faulty", "visible",
                         "reported_x_m", "reported_y_m"])
        senders = np.flatnonzero(result.faulty)
        for k, messages in enumerate(result.fault_reports):
            graph = radius_graph(result.positions[k], result.config.radius)
            for receiver in range(result.config.n):
                for j, sender in enumerate(senders):
                    writer.writerow([k, receiver, sender, int(result.faulty[receiver]),
                                     int(graph[receiver, sender]), *messages[receiver, j]])
    with (directory / "positions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "agent_id", "faulty", "x_m", "y_m"])
        for k, positions in enumerate(result.positions):
            for i, (x, y) in enumerate(positions):
                writer.writerow([k, i, int(result.faulty[i]), x, y])
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "normal_diameter_m", "normal_x_span_m", "normal_y_span_m",
                         "normal_centroid_x_m", "normal_centroid_y_m", "min_normal_degree",
                         "visibility_components", "normal_box_violation_m"])
        for k in range(len(result.positions)):
            writer.writerow([k, result.diameter[k], *result.axis_span[k],
                             *result.normal_centroid[k], result.min_normal_degree[k],
                             result.components[k], result.box_violation[k]])
    plot_overview(result, directory / "overview.png")
    plot_messages(result, directory / "attack_messages.png")
    if gif:
        save_animation(result, directory / "animation.gif", stride=stride)
    step = payload["summary"]["within_tolerance_through_remaining_recorded_steps_from"]
    outcome = f"ステップ {step} からしきい値以下" if step is not None else "最終時刻でしきい値未達"
    print(f"保存先：{directory.resolve()}")
    print(f"正常機間の最終最大距離：{result.diameter[-1]:.6g} m（{outcome}）")


def main():
    parser = argparse.ArgumentParser(description="W-MSRと最小包含円による2次元集合シミュレーション")
    parser.add_argument("--config", type=Path, help="JSON設定ファイル。コマンドの指定が優先")
    parser.add_argument("--algorithm", choices=ALGORITHM_NAMES, help="wmsr（既定）または mec（最小包含円）")
    parser.add_argument("--move-fraction", type=float, help="最小包含円の中心まで進む割合（既定1）")
    parser.add_argument("--compare", action="store_true", help="同じ初期配置で両方のアルゴリズムを比較")
    parser.add_argument("--n", type=int)
    parser.add_argument("--f", type=int, help="known fault upper bound, independent of actual count")
    parser.add_argument("--r", dest="radius", type=float, help="visibility radius [m]")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--box-size", type=float, help="initial square side [m]")
    parser.add_argument("--fault-count", type=int, help="actual fault count (must be <= f)")
    parser.add_argument("--faulty-ids", help="comma-separated, zero-based IDs; sets fault-count if omitted")
    parser.add_argument("--fault-mode", choices=["none", "stuck", "random", "byzantine"])
    parser.add_argument("--fault-step", type=float, help="random fault movement [m/step]")
    parser.add_argument("--attack-strategy", choices=["split", "random", "inside"])
    parser.add_argument("--attack-amplitude", type=float, help="false report offset [m], split/random only")
    parser.add_argument("--tolerance", type=float, help="normal diameter threshold [m]")
    parser.add_argument("--initial", type=Path, help="headerless x,y CSV, one agent per row (must match n)")
    parser.add_argument("--out", type=Path, help="出力先（既定は実験結果の下）")
    parser.add_argument("--gif", action="store_true", help="アニメーションも保存")
    parser.add_argument("--gif-stride", type=int, default=2, help="GIFのステップ間隔")
    parser.add_argument("--demo", action="store_true", help="故障なし、ビザンチン、停止、ランダム移動を比較")
    args = parser.parse_args()
    try:
        data = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(data, dict):
            raise ValueError("config must be a JSON object")
        fields = Config.__dataclass_fields__
        unknown = set(data) - set(fields)
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        for name in fields:
            value = getattr(args, name, None)
            if value is not None:
                data[name] = value
        if args.faulty_ids is not None:
            data["faulty_ids"] = tuple(int(i.strip()) for i in args.faulty_ids.split(","))
            if args.fault_count is None:
                data["fault_count"] = len(data["faulty_ids"])
        elif "faulty_ids" in data and "fault_count" not in data:
            data["fault_count"] = len(data["faulty_ids"])
        config = Config(**data)
        if args.gif_stride < 1:
            raise ValueError("gif-stride must be positive")
        initial = np.loadtxt(args.initial, delimiter=",", ndmin=2) if args.initial else None
        if args.demo:
            if config.f < 1:
                raise ValueError("demo needs f >= 1; use a single run for f=0")
            count = config.fault_count or 1
            ids = config.faulty_ids or tuple(range(count))
            scenarios = [
                (FAULT_NAMES["none"], replace(config, fault_count=0, faulty_ids=(), fault_mode="none")),
                (FAULT_NAMES["byzantine"], replace(config, fault_count=count, faulty_ids=ids, fault_mode="byzantine")),
                (FAULT_NAMES["stuck"], replace(config, fault_count=count, faulty_ids=ids, fault_mode="stuck")),
                (FAULT_NAMES["random"], replace(config, fault_count=count, faulty_ids=ids, fault_mode="random")),
            ]
        else:
            scenarios = [(None, config)]
        # Compute and validate all scenarios before writing any result files.
        algorithms = tuple(ALGORITHM_NAMES) if args.compare else (config.algorithm,)
        groups = [(name, [simulate(replace(c, algorithm=algorithm), initial) for algorithm in algorithms])
                  for name, c in scenarios]
        from .plotting import configure_japanese_font, plot_comparison

        configure_japanese_font()
        base_name = "比較" if args.compare else ALGORITHM_NAMES[config.algorithm]
        out = args.out if args.out else Path("実験結果") / base_name
        for name, results in groups:
            directory = out / name if name else out
            for result in results:
                destination = directory / ALGORITHM_NAMES[result.config.algorithm] if args.compare else directory
                save_result(result, destination, args.gif, args.gif_stride)
            if args.compare:
                plot_comparison(results, directory / "比較グラフ.png")
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
