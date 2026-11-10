# 2次元ロボットの集合シミュレーション

半径 `r [m]` 以内の近傍を使い、W-MSR、最小包含円、MEC中心の凸包による集合を試すPythonシミュレーションです。
故障なし、Byzantine（相手ごとに異なる虚偽座標を送信）、停止、ランダム移動を比較できます。

## すぐに実行

Python 3.9以上。macOS / Linuxでは、ターミナルでこのREADMEのあるフォルダに移動して実行します。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# n=20、故障上限f=1、実際のByzantine故障1体、視認半径6m
python -m wmsr --n 20 --f 1 --r 6 --fault-count 1 --fault-mode byzantine --gif

# 故障なしで、最小包含円の中心へ移動
python -m wmsr --algorithm mec --n 20 --r 6 --gif --gif-stride 1

# 同じ初期配置・同じfで、両手法を4種類の故障条件で比較
python -m wmsr --compare --demo --n 20 --f 1 --r 6 --steps 100 --gif
```

Windowsでは `python3` を `python`、仮想環境の有効化を `.venv\Scripts\activate` に置き換えてください。
GIFはPillowで生成するため、ffmpegは不要です。グラフはファイルに保存し、画面ウィンドウは開きません。
新しい比較は `--video` でMP4を保存できます。[MEC凸包の更新方法・動画・実験結果](MEC凸包の検証.md) に手順があります。
日本語表示にはシステムの日本語フォントを使います。macOSのヒラギノ角ゴ、Windowsの游ゴシックに対応します。
Ubuntuで日本語フォントが見つからない場合は `sudo apt install fonts-noto-cjk` で追加できます。

## 最小包含円による更新

正常機iは、更新前の自分の実位置と、距離r以内の全近傍から受け取った座標を含む最小の円を求めます。その中心を `c_i(k)` とすると、次の位置は `p_i(k+1) = (1-α)p_i(k) + αc_i(k)` です。
`--move-fraction` がαで、既定の1では中心まで移動します。0.5なら中心との中点に移動します。全機が同じ時刻の位置を使って同時に更新します。

- すべての点が同じなら、円の半径は0で、その位置が中心です。近傍なしもこの場合に当たります。
- 最も遠い2点を直径とする円に全点が入れば、その円を使います。直線上の点や鈍角三角形もこの場合です。
- その円に入りきらなければ、一直線上にない3点の外接円を調べ、全点を含む最小の円を選びます。

最小包含円ではfによる除去を行わないので、虚偽座標も円の計算に入り、中心が正常機の範囲から離れることがあります。
視認距離の判定は実位置に基づきます。中心へ進む際の連結性を維持する制約は設けていません。
本実装の候補の全探索は近傍点数をmとして最悪 O(m⁴) です。計算手順を確認しやすい小規模な実験用です。

`--compare` は配置、故障機ID、乱数シードを両手法で揃え、正常機間の最大距離と、全体の視認グラフの連結成分数を比較します。
ビザンチンの `split` と `inside` は各手法の現在の正常座標を参照するため、位置が分かれた後の虚偽値も手法ごとに変化します。
結果の数値は [実行例](RESULTS.md) にあります。

## モデルと更新式

エージェント数は `n`、真の位置は `p_i(k)=(x_i(k), y_i(k))`、故障数の既知の上限は `f` です。
実際の故障数 `fault_count` は `0 <= fault_count <= f < n` を満たします。
ここでは**全体で高々f体の故障（f-total）**を扱います。

近傍は各ステップの更新前の**実位置**から決めます。

```math
\mathcal N_i(k)=\{j\ne i:\|p_j(k)-p_i(k)\|_2\le r\}.
```

境界の距離がちょうどrの相手も含みます。自分以外の近傍がいない場合はその場に残ります。
正常エージェントは現在の真の位置を送信し、Byzantineエージェントは受信相手ごとに異なる有限の座標値を送れます。
正常エージェントiがjから受け取る値を `z_ij^a(k)`、座標軸を `a in {x,y}` とします。

各正常エージェントは、xとyのそれぞれについて次の処理を行います。

1. 自分の座標より**厳密に小さい**受信値から、小さい順に最大f個を除去する。
2. 自分の座標より**厳密に大きい**受信値から、大きい順に最大f個を除去する。
3. 残った近傍値と自分の座標の単純平均へ更新する。自分は必ず残す。同値の近傍も残す。

残った近傍の集合を `S_i^a(k)` とすると、等重みのW-MSRは次の式です。

```math
p_i^a(k+1)=\frac{p_i^a(k)+\sum_{j\in S_i^a(k)}z_{ij}^a(k)}{1+|S_i^a(k)|}.
```

全正常エージェントは同じ時刻kの値から**同期・同時更新**します。実装は [wmsr/simulation.py](wmsr/simulation.py) の `retained_indices`、`wmsr_step`、`simulate` にあります。
正常側の更新関数は故障IDを参照しません。IDは故障動作の生成と正常側だけの評価に用います。

**「上下f体ずつ」について：** 標準W-MSRは、常に無条件で上下f個を切り捨てる方法とは異なります。
例えば自分が最小値なら、自分より下の削除数は0です。上側がf個未満なら上側をすべて削除します。
`f=0` のときは、自分と視認できる全近傍の通常の平均です。

**「重心」について：** xとyでは除去される相手が異なる場合があります。
したがって、更新点は同じロボット集合の2次元の重心とは限りません。全エージェントの初期重心への収束も一般には保証しません。

## Byzantineと視認の区別

Byzantineモデルでは「実位置から距離を判定できるが、位置更新に使う座標は通信等で受け取る」という仮定を置きます。
虚偽座標を送っても、物理的にrより遠いエージェントは近傍に入れません。
逆に、r以内の相手が送る座標は、実位置から離れた虚偽値でも受信します。
送信者IDを偽装して数を増やす攻撃、遅延、欠損、NaN/無限値の送信は対象外です。

もし「視認」が、すべての相手の真の座標を直接・正確に観測できるという意味なら、虚偽座標の送信はその観測には影響しません。
その場合の故障動作は `stuck` / `random` が対応します。

| `fault_mode` | 実際の移動 | 正常側が受け取る値 |
|---|---|---|
| `none` | 全員がW-MSR | 真の位置。故障数は0 |
| `byzantine` | 故障機は初期位置に固定 | 故障機は相手ごとに虚偽値を送信 |
| `stuck` | 故障機は初期位置に固定 | 真の位置。停止しても視認・通信は続く |
| `random` | 故障機は毎ステップ、独立な一様乱数方向に一定距離だけ移動 | 真の位置 |

`byzantine` では `--attack-strategy` を選べます。

| 戦略 | 虚偽値の生成方法 |
|---|---|
| `split`（既定） | 受信者IDが偶数なら現在の正常座標の最小値−A、奇数なら最大値＋A。x,yそれぞれで計算 |
| `random` | 故障機の実位置の各座標に `Uniform(-A,A)` を加える。時刻・送信者・受信者・軸ごとに生成 |
| `inside` | 現在の正常座標範囲の25%点／75%点を受信者IDの偶奇で送り分ける |

`A = --attack-amplitude`（既定20m）です。`inside` ではAを使いません。
`split` と `inside` の攻撃生成器だけは現在の全正常座標を知る強い攻撃者を想定します。
正常側は自分と近傍の情報だけで更新します。これらは代表的な攻撃のサンプルであり、すべてのByzantine動作を尽くすものではありません。

```sh
# 故障IDを明示する例（0始まり、故障数はIDリストから設定）
python -m wmsr --n 20 --f 2 --r 6 --faulty-ids 0,3 --fault-mode byzantine --attack-strategy inside --out 実験結果/正常範囲内の攻撃

# 実際の故障なし。ただしf=1としてフィルタは続ける
python -m wmsr --f 1 --fault-count 0 --fault-mode none --out 実験結果/故障なし

# 故障を想定しない通常の平均との比較
python -m wmsr --f 0 --fault-count 0 --fault-mode none --out 実験結果/通常の平均

# JSONの設定を使う。コマンドライン指定がJSONより優先
python -m wmsr --config config.example.json --steps 200 --gif --out 実験結果/設定ファイル実験
```

`--demo` の故障あり条件の故障数は、指定があればその数、0なら1です。`--demo` にはf>=1が必要です。
各条件は同じseedから独立に開始します。初期配置の乱数と故障動作の乱数を分け、条件間で初期位置が変わらないようにしています。
既定の故障IDは `0,1,...,fault_count-1` で、実行中の故障発生・復旧は扱いません。

## パラメータ

| オプション | 意味 | 既定値 |
|---|---|---|
| `--n` | 全体数 | 20 |
| `--f` | 既知の故障数上限 | 1 |
| `--r` | 視認半径 [m] | 6 |
| `--steps` | 更新回数 | 100 |
| `--seed` | 再現用乱数seed | 7 |
| `--box-size` | 初期配置 `[0,L] × [0,L]` のL [m] | 10 |
| `--fault-count` | 実際の故障数 | 0 |
| `--fault-mode` | 故障動作 | byzantine |
| `--faulty-ids` | 故障IDのカンマ区切り | 先頭からfault_count体 |
| `--fault-step` | random故障の移動距離 [m/step] | 1 |
| `--attack-strategy` | Byzantine送信戦略 | split |
| `--attack-amplitude` | 虚偽値の振幅A [m] | 20 |
| `--tolerance` | 正常側の最大距離による集合判定 [m] | 0.001 |
| `--initial` | 初期位置のヘッダーなしx,y CSV | seedで生成 |
| `--algorithm` | `wmsr`、`mec`、`mec-hull`、`mec-hull-safe` | wmsr |
| `--move-fraction` | MEC系の目標点まで進む割合 | 1 |
| `--compare` | 同じ初期配置でW-MSRと最小包含円を実行 | 指定時のみ |
| `--compare-hulls` | 最小包含円、MEC中心の凸包、安全領域つき凸包を比較 | 指定時のみ |
| `--video` | 日本語のMP4を保存（FFmpegまたはOpenCVが必要） | 指定時のみ |
| `--out` | 出力フォルダ | 実験結果/手法名 または 実験結果/比較 |
| `--gif` | アニメーションも保存 | 指定時のみ |
| `--gif-stride` | GIFに使うステップ間隔 | 2 |

`--initial` はn行2列で、行番号がIDに対応します。`--n` も行数に合わせて指定します。
初期座標を指定した場合は `box_size` は使いません。任意の向きの配置を指定できますが、軸ごとのフィルタなので回転に対して同じ挙動になるとは限りません。

## 出力と集合の判定

| ファイル | 内容 |
|---|---|
| `overview.png` | 2次元軌跡、x/yの時系列、正常側の最大距離 |
| `比較グラフ.png` | 比較時のみ。2手法の最大距離と全体の連結成分数 |
| `最小包含円.csv` | 最小包含円を使った場合のみ。正常機ごとの円の中心・半径 |
| `animation.gif` | 実位置と距離rに基づく視認グラフ。青：正常、赤×：故障。破線円：正常機1体の視認半径 |
| `attack_messages.png` | Byzantine時のみ。可視回数の多い正常受信者を偶奇から1体ずつ選んだ虚偽値と、正常側の座標範囲・故障機の実位置。最後の可視メッセージ付近までを表示 |
| `positions.csv` | 全時刻の実位置。step, agent_id, faulty, x_m, y_m |
| `fault_messages.csv` | 故障機から各相手への送信候補値と可視フラグ。visible=1の相手だけが受信し、正常機だけが更新に使用 |
| `metrics.csv` | 正常側の最大距離、軸ごとの幅、重心、最小近傍数、全体の連結成分数、座標範囲の逸脱 |
| `history.npz` | positions `(steps+1,n,2)`、faulty `(n,)`、fault_reports `(steps,n,fault_count,2)` |
| `run.json` | 実行条件と最終結果 |

`fault_reports` の送信者方向は故障IDの昇順で、`faulty` から対応するIDを取得できます。
送信候補値は不可視の相手や故障機自身宛てにも記録しますが、自己ループはなく、実際の正常更新に使うのは可視の故障送信者から正常受信者への値だけです。
GIFの灰色の線は通信が可能な近傍を示します。最小包含円では、代表する正常機1体の円と中心を緑色で示します。
表示範囲は実位置に合わせているため、大きな虚偽座標による円の一部は画面外になることがあります。
比較結果の例は `実験結果/比較/故障なし/最小包含円/` と `実験結果/比較/故障なし/比較グラフ.png` です。
同じ `--out` を再利用すると同名ファイルを上書きするため、条件ごとに出力先を分けてください。

正常機集合をHとし、各時刻で次を計算します。

```math
D(k)=\max_{i,j\in H}\|p_i(k)-p_j(k)\|_2.
```

最終時刻の `D <= tolerance` を集合の数値判定とします。正常機が1体ならD=0です。
`within_tolerance_through_remaining_recorded_steps_from` は、そこから最終記録時刻までしきい値内にある最初のステップです。
しきい値到達で打ち切らず、指定した全ステップを計算します。漸近収束の証明や将来の保持を意味しません。
誤差図だけは対数表示のため、距離0を1e-15に丸めます。CSVや判定値は丸めません。

## 集合できるとは限らない条件

- 半径が小さいと、近傍が分断されたまま複数の集団にとどまることがあります。
- 故障が0体でもfが大きすぎると有用な値まで除去します。例：正常機2体、f=1では相手を除去して双方が停止します。
- 連結であることや、各機の近傍数だけではW-MSRの収束保証になりません。故障モデルに対応するグラフのロバスト性が必要です。
- 本モデルは位置更新によって視認グラフも変化します。x/yで別の相手を採用するため、更新点は近傍点の2次元凸包内とは限らず、視認関係の維持も保証しません。
- 各軸で正常初期値の最小値・最大値からなる矩形内にとどまる性質と、正常初期位置の2次元凸包内にとどまる性質は別です。
- 故障機も含む全員の集合は評価しません。任意に振る舞う故障機を強制的に集合させることはできません。

グラフのロバスト性の自動証明は実装していません。`components` / `min_normal_degree` は診断値です。
点ロボットの離散時間モデルであり、速度・加速度上限、衝突回避、障害物、センサー誤差は含みません。
位置の単位はm、時間は更新ステップkです。物理的な秒への換算は置いていません。
近傍判定と最大距離評価は各時刻O(n²)、保存は位置履歴O(Tn)と故障送信履歴O(Tn×fault_count)です。
大規模化する際は空間探索と履歴保存方法の見直しが必要です。

## Pythonから利用

```python
import numpy as np
from wmsr import Config, simulate

config = Config(n=20, f=1, radius=6, steps=100,
                fault_count=1, fault_mode="byzantine", attack_strategy="inside")
result = simulate(config)
print(result.summary())
normal_trajectories = result.positions[:, ~result.faulty, :]

# 指定配置も可能
# result = simulate(config, initial_positions=np.array([...]))
```

独自の攻撃値を作る場合は `wmsr_step(positions, f, radius, received_positions)` を呼べます。
`received_positions[i,j,:]` は「受信者iが送信者jから受け取った座標」です。
正常送信者の値は真の位置にし、故障送信者の列だけ変更してください。
自分自身の値には `positions[i]` を使います。故障機の物理的な次位置は呼び出し側で置き換えます。

## テスト

```sh
python -m unittest discover -s tests -v
```

除去数の不足、同値、自分自身の保持、x/y独立処理、同期更新、距離rの境界、孤立、f=0、
Byzantineの相手別送信、物理位置と受信値の区別、正常初期矩形内の保持、乱数再現性等を検証します。
GitHub Actionsも同じテストと短いGIF生成を実行します。

## GitHubでの管理

GitHubでは [L4rrrr/4A-Ishii-lab](https://github.com/L4rrrr/4A-Ishii-lab) の `集合シミュレーション/` にコードを配置します。
GitHubのコードを取得して編集する手順は [GITHUB_SETUP.md](GITHUB_SETUP.md) を参照してください。
掲載した画像とGIFは [共有結果](共有結果/README.md) から見られます。
新しい実験の画像を追加する方法は [実験結果の画像を追加](GITHUB_SETUP.md#実験結果の画像を追加) を参照してください。

## 参考文献

H. J. LeBlanc, H. Zhang, X. Koutsoukos, S. Sundaram,
“Resilient Asymptotic Consensus in Robust Networks,” IEEE JSAC, 31(4), 766–781, 2013.
[原著論文PDF](https://cs.vuse.vanderbilt.edu/koutsoxd/www/Publications/resilientConsensusJSAC2013.pdf)
（特にSection IIIのW-MSR定義）。本実装はそのスカラー更新を各軸に適用したシミュレーションであり、
この移動ロボット系の収束定理を主張するものではありません。

最小包含円の支持点については [Welzl, Smallest Enclosing Disks (Balls and Ellipsoids)](https://doi.org/10.1007/BFB0038202)、
近傍関係を保つ制約付きの集合アルゴリズムについては [Bullo・Cortés・Martínez, Distributed Control of Robotic Networks, 第4章](https://motion.me.ucsb.edu/book-dcrn/pdfs/DCRN-BulloCortesMartinez-chapter4.pdf) を参照してください。
