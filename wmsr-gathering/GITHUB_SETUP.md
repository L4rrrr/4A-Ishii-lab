# GitHubでの管理

対象リポジトリは [L4rrrr/4A-Ishii-lab](https://github.com/L4rrrr/4A-Ishii-lab) です。
シミュレーションの作業フォルダは `wmsr-gathering/` です。

## GitHubから取得

ターミナルで作業したい場所に移動し、次を実行します。

```sh
git clone https://github.com/L4rrrr/4A-Ishii-lab.git
cd 4A-Ishii-lab/wmsr-gathering
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

仮想環境と実行結果は `.gitignore` で除外しています。図やCSVを共有する場合は、必要なファイルと実行条件を選んで追加してください。
配布ZIPのコードはそのまま実行できます。GitHubへ変更を反映する作業には、上の手順でcloneしたフォルダを使ってください。
