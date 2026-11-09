"""Copy selected simulation figures and settings into a Git-tracked folder."""

import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import quote


FIGURE_NAMES = {
    "比較グラフ.png": "比較グラフ",
    "overview.png": "軌跡と集合の進み具合",
    "attack_messages.png": "ビザンチン故障機の送信値",
    "animation.gif": "集合アニメーション",
}


def share_results(source, experiment_name):
    source = Path(source).resolve()
    root = Path(__file__).resolve().parents[1] / "共有結果"
    if not source.is_dir():
        raise ValueError(f"実験結果のフォルダが見つかりません: {source}")
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("共有結果とは別の実験結果フォルダを指定してください")
    name = experiment_name.strip()
    if not name or name.startswith(".") or any(c in name for c in '/\\:*?"<>|') or not name.isprintable():
        raise ValueError("実験名にはフォルダ名に使える文字を指定してください")
    destination = root / name
    if destination.exists():
        raise ValueError(f"同じ実験名があります。別の名前を指定してください: {name}")

    files = sorted(p for p in source.rglob("*")
                   if p.name in (*FIGURE_NAMES, "run.json") and p.is_file() and not p.is_symlink())
    figures = [p for p in files if p.name in FIGURE_NAMES]
    if not figures:
        raise ValueError(f"共有できるグラフやGIFがありません: {source}")

    def link(path, label):
        return f"[{label}]({quote(path.as_posix())})"

    lines = [f"# {name}", "", "画像を選ぶと、グラフやアニメーションを表示できます。", ""]
    for figure in figures:
        relative = figure.relative_to(source)
        folder = " / ".join(relative.parent.parts)
        folder = folder.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        title = FIGURE_NAMES[figure.name]
        label = f"{folder} / {title}" if folder else title
        item = f"- {link(relative, label)}"
        settings = figure.parent / "run.json"
        if settings in files and figure.name == "overview.png":
            item += f" · {link(settings.relative_to(source), '実行条件と数値結果')}"
        lines.append(item)

    root.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=root, prefix=".共有準備-") as temporary:
        staging = Path(temporary)
        for path in files:
            target = staging / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        (staging / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        staging.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description="生成したグラフとGIFをGitHub共有用にコピー")
    parser.add_argument("source", type=Path, help="実験結果のフォルダ")
    parser.add_argument("experiment_name", help="共有結果に付ける実験名")
    args = parser.parse_args()
    try:
        destination = share_results(args.source, args.experiment_name)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"共有用に保存しました：{destination}")
    print("GitHub Desktopでこの変更をコミットし、Push origin を押してください。")


if __name__ == "__main__":
    main()
