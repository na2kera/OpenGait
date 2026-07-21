# 関連論文の提案 — 「データそのものの良さ」を測る研究

本ディレクトリの実験（`dgv2_analysis/README.md`）は、**歩容認識の精度そのもの**が目的ではなく、
「ある学習データ（サブセット）がどれだけ良いか＝最終精度をどれだけ引き出せるか」を
**深層特徴空間で計算したデータ特性指標から予測・優劣判定する**枠組みである。

使っている指標を整理すると:

| 指標 | 中身 | 「データの良さ」の観点 |
|---|---|---|
| `*_MSD` | 特徴空間での平均的な散らばり | 多様性 / 集中度 |
| `*_1NN`, `*_kNN` | 近傍ベースの2標本識別性 | train と test（あるいは real/fake）が特徴空間で見分くか |
| `*_FID` | Fréchet 距離（分布間距離） | 分布としての品質・多様性 |
| `FID_train_test_*` | train↔test の分布距離 | **分布シフト**（これが最終精度の最有力説明変数だった） |
| `Number of people` | データ量 | 規模 |
| → 回帰 (Ridge/LODO) / 優劣判定 (Logistic) | 上記から `acc_all` を予測 | データ品質 → 性能の写像 |

したがって関連研究として探すべきは **歩容の論文ではなく、「データセット/分布の良さ」そのものを
特徴空間の指標で定量化し、それが下流性能とどう対応するかを論じた論文**である。
以下、この実験のパイプラインの各要素に対応づけて提案する。すべて主要な国際会議/学会採択論文を優先した。

---

## A. 分布間距離でデータ/生成物の質を測る（`*_FID`, `MSD` の系譜）

### 1. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
- **著者 / 発表**: Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter — **NeurIPS 2017**
- **なぜ関係するか**: 本実験の中核指標 **FID（Fréchet Inception Distance）の初出**。
  Inception 特徴空間で分布を多変量ガウスと見なし Fréchet(=Wasserstein-2) 距離で
  「サンプル集合の質＋多様性」を測る、という発想そのものがこの実験の `*_FID` /
  `FID_train_test_*` の土台になっている。特徴抽出器を DeepGaitV2 に差し替えている点は
  本論文の枠組みの自然な拡張として位置づけられる。
- リンク: https://arxiv.org/abs/1706.08500

### 2. Reliable Fidelity and Diversity Metrics for Generative Models
- **著者 / 発表**: Naeem, Oh, Uh, Choi, Yoo — **ICML 2020**
- **なぜ関係するか**: FID が「質」と「多様性」を分離できない問題に対し、**Density / Coverage**
  という近傍ベース指標を提案。本実験の `1NN`/`kNN`/`MSD` はまさに近傍・散らばり系の指標であり、
  「単一のスカラーではなく質と多様性を分けて測るべき」という主張は、
  データの良さを多面的に評価しようとする本実験の方向性と直結する。
- 併読推奨: Kynkäänniemi et al., *Improved Precision and Recall Metric for Assessing
  Generative Models*, **NeurIPS 2019**（Precision/Recall を特徴空間の多様体被覆で定義した先行研究）。
- リンク: https://proceedings.mlr.press/v119/naeem20a.html

### 3. Revisiting Classifier Two-Sample Tests (C2ST)
- **著者 / 発表**: Lopez-Paz, Oquab — **ICLR 2017**
- **なぜ関係するか**: 「2つの標本が同じ分布か」を分類器の識別精度で測る枠組み。
  本実験の `1NN`/`kNN` 指標（近傍分類による2標本識別）の理論的裏付けであり、
  `FID_train_test`（train と test が見分くか）を「分類可能性」として解釈する視点を与える。
  1-NN 版の2標本検定はここで定式化されている。
- リンク: https://arxiv.org/abs/1610.06545

---

## B. データセット間距離が下流性能（転移）を予測する（`FID_train_test` → `acc` の核心）

### 4. Geometric Dataset Distances via Optimal Transport (OTDD)
- **著者 / 発表**: Alvarez-Melis, Fusi — **NeurIPS 2020**
- **なぜ関係するか**: **本実験の主張に最も近い外部研究のひとつ。** 特徴とラベルの同時分布を
  最適輸送で比較する「データセット間距離」を定義し、**その距離が転移学習の成否と強く（負に）相関する**
  ことを実験的に示した。本実験の「`FID_train_test`（分布距離）が最終精度を最もよく説明する」という
  発見と同じ構図であり、FID の代替/補強指標として OTDD を導入する提案が自然に立つ。
- リンク: https://proceedings.neurips.cc/paper/2020/hash/f52a7b2610fb4d3f74b4106fb80b233d-Abstract.html

### 5. Predicting with Confidence on Unseen Distributions（AutoEval 系）
- **著者 / 発表**: Guillory, Shankar, Ebrahimi, Darrell, Schmidt — **ICCV 2021**
- **なぜ関係するか**: **手法の形が本実験とほぼ同型。** ラベル無しのテスト集合に対して、
  **特徴分布間の Fréchet 距離などを説明変数に、回帰でモデル精度を予測**する。
  本実験の「`FID_train_test` などの指標 → Ridge 回帰で `acc_all` を予測」と発想が一致する。
  同時に「素の Fréchet 距離だけでは不十分で、Difference of Confidence (DoC) が有効」と論じており、
  **本実験の指標セットを拡張する具体的な次の一手**（confidence 系特徴の追加）を示唆する。
- リンク: https://arxiv.org/abs/2107.03315

### 6. Accuracy on the Line: On the Strong Correlation Between OOD and ID Generalization
- **著者 / 発表**: Miller, Taori, Raghunathan, Sagawa, Koh, Shankar, Liang, Carmon, Schmidt — **ICML 2021**
- **なぜ関係するか**: 分布シフト下の性能が in-distribution 性能と強い線形相関を持つことを大規模に実証。
  本実験が「データ特性（分布シフト量）と最終精度の間に予測可能な関係がある」ことを回帰で示そうと
  しているのに対し、**そうした相関がどこまで頑健に成立するか/破れるか**の理論的・実証的な参照点を与える。
  Spearman 相関で評価している本実験の結果解釈の枠組みとして有用。
- リンク: https://proceedings.mlr.press/v139/miller21b.html

---

## C. データ/表現の価値を無学習で見積もる（回帰・優劣判定の代替枠組み）

### 7. LogME: Practical Assessment of Pre-trained Models for Transfer Learning
- **著者 / 発表**: You, Liu, Wang, Long — **ICML 2021**
- **なぜ関係するか**: 抽出済み特徴に対する**ラベルの証拠量（evidence）を計算し、fine-tune せずに
  転移性能を予測**する転移可能性推定。本実験が「特徴＋精度」を回帰で結ぶのと同じく、
  「特徴の良さを最終性能の代理指標で測る」問題設定。
  データ側でなくモデル/表現側から見た双対的アプローチとして、指標比較のベースラインに使える。
- 併読推奨: Nguyen et al., *LEEP*, **ICML 2020**（ラベル分布ベースの転移可能性推定）。
- リンク: https://arxiv.org/abs/2102.11005

### 8. Data Shapley: Equitable Valuation of Data for Machine Learning
- **著者 / 発表**: Ghorbani, Zou — **ICML 2019**
- **なぜ関係するか**: 個々の学習データ点の「価値」を、最終性能への限界寄与（Shapley 値）として定量化。
  本実験が**サブセット単位**で良し悪しを測るのに対し、**データ点単位**での価値付けを与える相補的視点。
  「良いデータとは何か」を性能ベースで定義するという problem framing が共通し、
  優劣判定（pairwise logreg）を Shapley 的な価値順位づけへ発展させる方向を示す。
- リンク: https://proceedings.mlr.press/v97/ghorbani19c.html

---

## まとめ（提案の使い分け）

- **枠組み全体の先行研究として引く**なら → 4 (OTDD) と 5 (AutoEval)。手法・主張が本実験に最も近い。
- **各指標の出典/正当化**なら → 1 (FID), 2 (Density/Coverage), 3 (C2ST=1NN)。
- **結果の解釈・相関の頑健性**なら → 6 (Accuracy on the Line)。
- **相補的な問題設定（モデル側・データ点側）**なら → 7 (LogME), 8 (Data Shapley)。

いずれも歩容固有ではなく「データ/分布そのものの良さ」を対象にした論文であり、
本実験を歩容ドメインに閉じない一般的なデータ有効性評価の文脈へ接続できる。
（4・5 は特に、次の実験で指標追加・ベースライン比較として直接取り込める候補。）
