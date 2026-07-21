# プルリクエスト

このリポジトリで PR を作成するときは、ユーザーから別の指示がない限り、ベースリポジトリを `https://github.com/na2kera/OpenGait`、ベースブランチを `master` にする。`gh` では `--repo na2kera/OpenGait --base master --head <head-branch>` を明示し、作成後に PR URL とベースブランチを確認してからレビューを依頼する。

# CASIA-B partition の罠（2系統のプロトコル、混ぜ厳禁）

`datasets/CASIA-B/` の partition JSON は分割プロトコルが2系統同居しており、差は被験者 **075番ただ1人**（独自側では学習プール、標準側ではテストセット）。混ぜてもエラーは一切出ず、学習も評価も正常に完走したうえで、他の実験と比較不能な精度が正常な顔をして混入する。

- **この研究の全実験は test 49人（076-124）系のみ使う**: `CASIA-B.json`（train 75人）/ `CASIA-B-20.json`（15人）/ `CASIA-B-33.json`（25人）/ `CASIA-B-66.json`（50人）。この4つの TEST_SET はリスト完全一致（2026-07-20 検証済み）。
- **使用禁止（OpenGait標準プロトコル、test 50人 = 075-124）**: `CASIA-B-20-2.json` / `-40` / `-60` / `-80`。`-20-2` は `-20` の「バージョン2」ではなく別プロトコルで、学習者も別人（001-015 vs 016-030）。stock `DeepGaitV2_casiab.yaml`（class_num 74）はこちら側なので、config の派生元にしない。
- 検算方法: config の `dataset_partition` が指す JSON の TEST_SET が `CASIA-B.json` の TEST_SET とリスト完全一致すること。fine-tuning 用 config は `dgv2_finetune/generate_configs.py` の `validate_source` がこれを機械検証する（`--check` でも毎回実行される）。
