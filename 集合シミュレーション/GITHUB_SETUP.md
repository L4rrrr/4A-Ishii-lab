# GitHubでの管理

対象リポジトリは [L4rrrr/4A-Ishii-lab](https://github.com/L4rrrr/4A-Ishii-lab) です。
シミュレーションの作業フォルダは `集合シミュレーション/` です。

## GitHubから取得

ターミナルで作業したい場所に移動し、次を実行します。

```sh
git clone https://github.com/L4rrrr/4A-Ishii-lab.git
cd 4A-Ishii-lab/集合シミュレーション
```

Pythonのセットアップと実行コマンドは [README.md](README.md) を参照してください。

## 変更を保存

cloneしたフォルダでコードを編集してから、次を実行します。
pushには対象リポジトリへの書き込み権限とGitHubの認証が必要です。

```sh
git status
git add wmsr README.md config.example.json tests
git commit -m "Update W-MSR simulation"
git push
```

仮想環境と生成直後の `実験結果/` は `.gitignore` で除外しています。
配布ZIPのコードはそのまま実行できます。GitHubへ変更を反映する作業には、上の手順でcloneしたフォルダを使ってください。

## 実験結果の画像を追加

画像を生成したら、このREADMEのあるフォルダで次を実行します。最後の引数は、条件などが分かる新しい実験名に置き換えてください。

```sh
python -m wmsr.share "実験結果/比較" "視認半径6mでの比較"
```

例えば故障なしの画像だけを共有するなら、最初の引数を `実験結果/比較/故障なし` にします。
画像（PNG）、アニメーション（GIF）、実行条件と数値結果（`run.json`）を `共有結果/視認半径6mでの比較/` にコピーし、画像一覧を作ります。
座標の生データ（CSV・NPZ）は元の実験結果フォルダに保存します。
同名の共有フォルダがある場合は、別の実験名を指定してください。

GitHub Desktopの **Changes**（変更）で追加された画像を確認し、**Commit to main**（コミット）、**Push origin**（GitHubへ送信）の順に押します。
送信後は、GitHubの **集合シミュレーション → 共有結果 → 実験名** から画像を開けます。

画像を数枚だけ共有するときは、Finderで `共有結果/` に実験名のフォルダを作り、画像をコピーして同じ手順でコミットする方法もあります。
