# プルリクエスト

このリポジトリでは `origin` は `git@github.com:na2kera/OpenGait.git`、`upstream` は `https://github.com/ShiqiYu/OpenGait.git` である。pushやPRの前に実際のremote URLを再確認する。

PRを作成するときは、ユーザーから別の指示がない限り、ベースリポジトリを `https://github.com/na2kera/OpenGait`、ベースブランチを `master` にする。`gh` では `--repo na2kera/OpenGait --base master --head <head-branch>` を明示し、タイトルと本文は日本語で書く。作成後にPR URL、base、head、公開先を確認してからレビューを依頼する。

本家の `upstream` へのpush、force push、PRのcloseやmergeは、ユーザーの明示的な依頼なしに行わない。

# 研究状況の確認

- 研究進捗を尋ねられたら、最新の `daily_reports/` だけでなく、関連コード、config、結果ファイル、checkpoint、Git状態、必要なら実行中プロセスも突き合わせる。
- 実験値、モデル、データ分割、checkpointの出所を混同しない。観測した事実と解釈を区別し、日付と根拠ファイルを示せる状態で説明する。
- 中村さん側の `/home/ryu` とkera側の `/home/kera` のコード、結果、モデルを必ず区別する。

# 日報

- 「daily-report」「日報」は、特に指定がなければ `/home/kera/OpenGait/daily_reports/` を指す。
- 日報とNotion由来の研究メモは非公開情報として扱い、明示依頼なしにGitへ追加、commit、pushしない。

# 学習とテスト

- 学習・テストの開始前に、対象config、データ分割、GPU、Docker/tmux、出力先、既存checkpointとの衝突を検算する。
- 「状態確認」だけでは学習を開始しない。「学習を回して」などの明示依頼があれば、事前検算後に起動し、初期ログとGPU使用を確認する。
- 既存checkpoint、実験結果、データセットを黙って上書き・削除・移動しない。

# CASIA-B partition の罠（2系統のプロトコル、混ぜ厳禁）

`datasets/CASIA-B/` の partition JSON は分割プロトコルが2系統同居しており、差は被験者 **075番ただ1人**（独自側では学習プール、標準側ではテストセット）。混ぜてもエラーは一切出ず、学習も評価も正常に完走したうえで、他の実験と比較不能な精度が正常な顔をして混入する。

- **この研究の全実験は test 49人（076-124）系のみ使う**: `CASIA-B.json`（train 75人）/ `CASIA-B-20.json`（15人）/ `CASIA-B-33.json`（25人）/ `CASIA-B-66.json`（50人）。この4つの TEST_SET はリスト完全一致（2026-07-20 検証済み）。
- **使用禁止（OpenGait標準プロトコル、test 50人 = 075-124）**: `CASIA-B-20-2.json` / `-40` / `-60` / `-80`。`-20-2` は `-20` の「バージョン2」ではなく別プロトコルで、学習者も別人（001-015 vs 016-030）。stock `DeepGaitV2_casiab.yaml`（class_num 74）はこちら側なので、config の派生元にしない。
- 検算方法: config の `dataset_partition` が指す JSON の TEST_SET が `CASIA-B.json` の TEST_SET とリスト完全一致すること。fine-tuning 用 config は `dgv2_finetune/generate_configs.py` の `validate_source` がこれを機械検証する（`--check` でも毎回実行される）。
