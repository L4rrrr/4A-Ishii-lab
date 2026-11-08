# 実行例（2026-11-08）

共通条件はn=20、f=1、初期正方形10m×10m、seed=7、更新100回、集合しきい値1mmです。
故障あり条件はID=0の1体です。正常側の最大距離はD(k)で評価します。

| 条件 | r [m] | 正常機数 | 最後までD≤1mmとなる最初のステップ | 最終D [m] |
|---|---:|---:|---:|---:|
| 故障なし（フィルタはf=1） | 6 | 20 | 5 | 約8.88×10⁻¹⁶ |
| Byzantine / split | 6 | 19 | 5 | 0（浮動小数点上） |
| 停止 | 6 | 19 | 5 | 0（浮動小数点上） |
| ランダム移動 | 6 | 19 | 5 | 約8.88×10⁻¹⁶ |
| Byzantine / inside | 15 | 19 | 4 | 0（浮動小数点上） |
| Byzantine / split、半径不足 | 1 | 19 | 未達 | 約12.3951 |

これは指定したseed・条件での数値結果であり、任意の配置・攻撃に対する保証ではありません。
r=6mのByzantine条件では途中で故障機が正常集団の視認範囲から外れ、全体グラフは2成分になります。
正常側だけの集合と、故障機を含む全体の連結性は区別してください。
r=15mのinside条件は全体が連結のままで、故障機からの送信も継続します。
r=1mでは視認グラフが11成分に分かれ、正常側の最大距離は縮まりませんでした。

## 再現コマンド

```sh
python -m wmsr --n 20 --f 1 --r 6 --steps 100 --demo --gif --out results/demo
python -m wmsr --n 20 --f 1 --r 15 --steps 100 --fault-count 1 --fault-mode byzantine --attack-strategy inside --out results/byzantine_inside
python -m wmsr --n 20 --f 1 --r 1 --steps 100 --fault-count 1 --fault-mode byzantine --out results/disconnected
```

実行環境：Python 3.9.6、NumPy 2.0.2、Matplotlib 3.9.4、Pillow 11.2.1。
19件の単体テストを通過し、CLIからJSON設定の読み込み、図・GIF・CSV・NPZの生成も実行しました。
生成結果はローカルの `results/` にあります。コードのGit管理からは除外しています。
