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

> **読み方**: セクション A〜C が指標・枠組みの土台となる基礎/中核研究、**セクション D が近年（2023–2024）の
> 近しい研究**。各項目に短いサマリーを付けた。

---

## A. 分布間距離でデータ/生成物の質を測る（`*_FID`, `MSD` の系譜）

### 1. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
- **著者 / 発表**: Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter — **NeurIPS 2017**
- **サマリー**: GAN の学習収束を論じた論文だが、実務上の最大の貢献は評価指標 **FID（Fréchet Inception
  Distance）の提案**。Inception 特徴を抽出して実データ集合と生成データ集合をそれぞれ多変量ガウスで
  近似し、両者の Fréchet(=Wasserstein-2) 距離を「質＋多様性」の一括指標とする。人間の主観評価と整合し、
  ノイズやモード崩壊に敏感で Inception Score より頑健、と示した。
- **なぜ関係するか**: 本実験の中核指標 `*_FID` / `FID_train_test_*` の直接の出典。特徴抽出器を
  Inception から DeepGaitV2 に差し替える本実験は、この枠組みの自然な拡張として位置づけられる。
- リンク: https://arxiv.org/abs/1706.08500

### 2. Reliable Fidelity and Diversity Metrics for Generative Models
- **著者 / 発表**: Naeem, Oh, Uh, Choi, Yoo — **ICML 2020**
- **サマリー**: FID が「忠実度(質)」と「多様性」を1つのスカラーに混ぜてしまう問題を指摘し、
  特徴空間の近傍多様体に基づく **Density / Coverage** を提案。従来の Precision/Recall が
  同一分布を検出できない・外れ値に弱いという欠点を、より頑健な推定量で改善した。
- **なぜ関係するか**: 本実験の `1NN`/`kNN`/`MSD` は近傍・散らばり系の指標そのもの。「単一スカラーでなく
  質と多様性を分けて測る」という主張は、データの良さを多面的に評価する本実験の方向性と直結する。
- **併読推奨**: Kynkäänniemi et al., *Improved Precision and Recall Metric for Assessing Generative
  Models*, **NeurIPS 2019**（特徴空間の多様体被覆で Precision/Recall を定義した先行研究）。
- リンク: https://proceedings.mlr.press/v119/naeem20a.html

### 3. Revisiting Classifier Two-Sample Tests (C2ST)
- **著者 / 発表**: Lopez-Paz, Oquab — **ICLR 2017**
- **サマリー**: 「2つの標本が同じ分布から来たか」を、両者を分類する二値分類器の識別精度で検定する
  枠組み（C2ST）。同分布なら識別精度はチャンスレベルに留まる、という原理に基づく。分類器として
  1-NN や浅い MLP を使え、GAN 評価や因果探索への応用も示した。
- **なぜ関係するか**: 本実験の `1NN`/`kNN`（近傍分類による2標本識別）の理論的裏付け。
  `FID_train_test`（train と test が見分くか）を「識別可能性」として解釈する視点を与える。
- リンク: https://arxiv.org/abs/1610.06545

---

## B. データセット間距離が下流性能（転移）を予測する（`FID_train_test` → `acc` の核心）

### 4. Geometric Dataset Distances via Optimal Transport (OTDD)
- **著者 / 発表**: Alvarez-Melis, Fusi — **NeurIPS 2020**
- **サマリー**: 特徴とラベルの**同時分布**を最適輸送で比較する「データセット間距離 (OTDD)」を提案。
  モデル非依存・学習不要で、ラベル集合が異なるデータセット同士でも距離を定義できる。実験で
  **OTDD が転移学習の難しさと強い（負の）相関**を示し、距離が近いほど転移が成功しやすいことを確認した。
- **なぜ関係するか**: 本実験の「`FID_train_test`（分布距離）が最終精度を最もよく説明する」という発見と
  同じ構図。FID の代替/補強指標として OTDD を導入する提案が自然に立つ。
- リンク: https://proceedings.neurips.cc/paper/2020/hash/f52a7b2610fb4d3f74b4106fb80b233d-Abstract.html

### 5. Predicting with Confidence on Unseen Distributions（AutoEval 系）
- **著者 / 発表**: Guillory, Shankar, Ebrahimi, Darrell, Schmidt — **ICCV 2021**
- **サマリー**: ラベルの無いテスト集合に対して**モデル精度をラベルなしで推定**する問題を扱う。
  特徴分布間の Fréchet 距離や、モデル出力信頼度の差 **Difference of Confidence (DoC)** を
  データセットの表現とし、回帰でシフト下の精度を予測。素の分布距離だけでは不十分で、
  DoC を組み合わせると多様なシフトで安定して効くことを示した。
- **なぜ関係するか**: **手法の形が本実験とほぼ同型**。「`FID_train_test` などの指標 → 回帰で `acc_all`
  を予測」という本実験の骨格そのもの。指標セット拡張（confidence 系特徴の追加）という次の一手も示す。
- リンク: https://arxiv.org/abs/2107.03315

### 6. Accuracy on the Line: On the Strong Correlation Between OOD and ID Generalization
- **著者 / 発表**: Miller, Taori, Raghunathan, Sagawa, Koh, Shankar, Liang, Carmon, Schmidt — **ICML 2021**
- **サマリー**: CIFAR/ImageNet の各種変種や WILDS 等、多数のモデル・分布シフトにわたって、
  分布外(OOD)性能が分布内(ID)性能と**強い線形相関**を持つことを大規模に実証。相関はアーキテクチャ・
  ハイパラ・学習データ量・学習時間を通じて成立するが、破れる場合もあり得ることに触れる。
- **なぜ関係するか**: 本実験が「データ特性（分布シフト量）と最終精度の間に予測可能な関係がある」ことを
  回帰・Spearman 相関で示そうとする際の、相関の頑健性についての実証的な参照点になる。
- リンク: https://proceedings.mlr.press/v139/miller21b.html

---

## C. データ/表現の価値を無学習で見積もる（回帰・優劣判定の代替枠組み）

### 7. LogME: Practical Assessment of Pre-trained Models for Transfer Learning
- **著者 / 発表**: You, Liu, Wang, Long — **ICML 2021**
- **サマリー**: 事前学習モデルが抽出した特徴に対し、ラベルの**証拠量 (log maximum evidence)** を計算して
  fine-tune せず転移性能を予測する転移可能性推定。過学習に強く、教師/自己教師あり、分類/回帰、
  視覚/言語に汎用で、総当り fine-tune 比で最大 3000 倍高速と報告。
- **なぜ関係するか**: 本実験が「特徴＋精度」を回帰で結ぶのと同じく、「特徴の良さを最終性能の代理指標で
  測る」問題設定。データ側でなく**モデル/表現側**から見た双対的アプローチで、指標比較のベースラインに使える。
- **併読推奨**: Nguyen et al., *LEEP*, **ICML 2020**（ラベル分布ベースの転移可能性推定）。
- リンク: https://arxiv.org/abs/2102.11005

### 8. Data Shapley: Equitable Valuation of Data for Machine Learning
- **著者 / 発表**: Ghorbani, Zou — **ICML 2019**
- **サマリー**: 個々の学習データ点の「価値」を、性能への限界寄与を協力ゲームの **Shapley 値**で
  公平に配分して定量化。等価な価値付けが満たすべき性質を一意に満たすことを示し、価値の低い（有害な）
  データの検出などへの応用を示した。
- **なぜ関係するか**: 本実験が**サブセット単位**で良し悪しを測るのに対し、**データ点単位**の価値付けを
  与える相補的視点。「良いデータとは何か」を性能ベースで定義する枠組みが共通し、優劣判定（pairwise
  logreg）を Shapley 的な価値順位づけへ発展させる方向を示す。
- リンク: https://proceedings.mlr.press/v97/ghorbani19c.html

---

## D. 近年（2023–2024）の近しい研究

本実験に最も枠組みが近い最近の論文。特に 9 (LAVA) と 10 (Data Distribution Valuation) は
「分布間距離 → データの価値/優劣」という本実験の核心とほぼ同じ問題を、より新しい理論で扱っている。

### 9. LAVA: Data Valuation without Pre-Specified Learning Algorithms
- **著者 / 発表**: Just, Kang, Wang, Zeng, Ko, Jin, Jia — **ICLR 2023 (Oral, top-5%)**
- **サマリー**: 学習アルゴリズムに依存しない**モデル非依存のデータ価値評価**。train と validation の間の
  **クラス毎 Wasserstein 距離（最適輸送）を検証性能の代理（上界）**として用い、実際の学習を一切行わずに
  データの価値を測る。さらに最適輸送解の感度解析から、個々のデータ点の価値を追加コストなしで得る手法を提案。
- **なぜ関係するか**: **本実験と最も近い近年研究のひとつ**。「train↔（validation/test）の分布距離が
  最終性能を説明する」という本実験の `FID_train_test` の発想を、最適輸送で理論的に基礎づけた形。
  FID の代わりにクラス毎 Wasserstein を指標に据える拡張、および学習不要でのデータ選別への発展を直接示唆する。
- リンク: https://openreview.net/forum?id=JJuP86nBl4q ／ コード: https://github.com/reds-lab/LAVA

### 10. Data Distribution Valuation
- **著者 / 発表**: Xu, Wu, Sim, et al. — **NeurIPS 2024**
- **サマリー**: 個々の「データ集合」ではなく、その背後の**データ分布**の価値を評価する問題を定式化。
  各ベンダーの小さなプレビュー標本だけを見て、どの分布が最も有用かを比較する設定で、
  **最大平均差 (MMD) に基づく価値付け**を提案し、標本から分布同士を比較する理論保証付きの方針を導いた。
- **なぜ関係するか**: 本実験の**サブセット優劣判定 (pairwise logistic)** ＝「どのデータ分布がより良いか」を
  少数標本で比較する問題そのもの。FID/1NN に加え **MMD** を指標候補として取り込み、優劣判定に
  理論的裏付けを与える方向を示す、直近の対応研究。
- リンク: https://proceedings.neurips.cc/paper_files/paper/2024/hash/04b98fd38bd42810d0764cb6c46d10d8-Abstract-Conference.html ／ arXiv: https://arxiv.org/abs/2410.04386

### 11. DataComp: In Search of the Next Generation of Multimodal Datasets
- **著者 / 発表**: Gadre, Ilharco, Fang, et al. — **NeurIPS 2023 (Datasets & Benchmarks)**
- **サマリー**: モデルを固定し**データ側**を競う data-centric ベンチマーク。128 億の画像-テキスト対から
  各参加者がフィルタリング/キュレーションで部分集合を作り、共通の CLIP 学習コード・計算予算で
  38 の下流タスクを評価する。結果、**より小さく厳選したデータの方が大きいデータより汎化する**ことを示し、
  DataComp-1B は同手順の OpenAI CLIP を上回った。
- **なぜ関係するか**: 「モデルではなくデータの良さが性能を決める」「良いサブセットの選別が鍵」という
  本実験の問題意識そのものを、大規模かつ実証的に裏づける最近の代表的研究。個別指標というより、
  本実験を data-centric AI の潮流に位置づける文脈・動機付けとして引用できる。
- リンク: https://proceedings.neurips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html

---

## まとめ（提案の使い分け）

- **枠組み全体の先行研究として引く**なら → 4 (OTDD), 5 (AutoEval)、そして近年なら **9 (LAVA)**。
  手法・主張が本実験に最も近い。
- **各指標の出典/正当化**なら → 1 (FID), 2 (Density/Coverage), 3 (C2ST=1NN)。
- **優劣判定の理論化・新指標(MMD)の導入**なら → **10 (Data Distribution Valuation)**。
- **結果の解釈・相関の頑健性**なら → 6 (Accuracy on the Line)。
- **相補的な問題設定（モデル側・データ点側）**なら → 7 (LogME), 8 (Data Shapley)。
- **研究の位置づけ・動機（data-centric AI）**なら → **11 (DataComp)**。

いずれも歩容固有ではなく「データ/分布そのものの良さ」を対象にした論文であり、本実験を歩容ドメインに
閉じない一般的なデータ有効性評価の文脈へ接続できる。次の実験で指標追加・ベースライン比較として
直接取り込める候補は特に **4 (OTDD)・5 (AutoEval)・9 (LAVA)・10 (Data Distribution Valuation)**。
