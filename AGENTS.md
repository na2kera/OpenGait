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

# 共有コンテキスト（skill 索引）

この研究のドメイン知識・手順・確定値は `/home/kera/.claude/skills/<name>/SKILL.md` に **1本ずつ実体として** 置いてある。
Claude Code は skill として自動で読む。**Codex / Cursor CLI は、下表の条件に当てはまったら自分で該当ファイルを read すること**（記憶や推測で答えない）。
Cursor CLI からは `~/.cursor/skills/<name>`、Codex CLI からは `~/.agents/skills/<name>` に**ディレクトリ単位の**シンボリックリンクを張ってあるので、どちらでも個人スキル（Codex は `$<name>`）として見える（Codex はファイル単位の symlink を辿らないのでディレクトリ単位にしてある）。実体は上記の1か所だけなので、更新は実体側を直す。

| skill | いつ読むか | 実体パス |
| --- | --- | --- |
| `paper-manuscript` | 原稿・予稿・技報・執筆・章立て・9/14締切の話。**「論文の主張は」「何を言いたいか」「仮説」「2×2」「汎用 vs 歩容特化」「CL に強い理由」と聞かれたら §3 を読んでそのまま答える**（9/10 確定） | `/home/kera/.claude/skills/paper-manuscript/SKILL.md`（参照論文PDFとtxtが `refs/`、主張は §3） |
| `research-glossary` | ρ・MAE・LODO・Δ・zero-shot・優劣判定など用語と確定値 | `/home/kera/.claude/skills/research-glossary/SKILL.md` |
| `training-status` | 学習・テスト・キューの進捗確認、GPU の空き | `/home/kera/.claude/skills/training-status/SKILL.md` |
| `train-deepgaitv2` | DeepGaitV2 の学習・テストを回す手順 | `/home/kera/.claude/skills/train-deepgaitv2/SKILL.md` |
| `nakamura-research` | 中村さん側（`/home/ryu`）の研究・コード・結果 | `/home/kera/.claude/skills/nakamura-research/SKILL.md` |
| `daily-report-sync` | 日報の作成・更新と Notion への反映 | `/home/kera/.claude/skills/daily-report-sync/SKILL.md` |

補足:
- skill 内の数値はスナップショット。**出典ファイルと食い違ったらファイル側が正**で、skill を更新する。
- Claude Code の自動メモリ `/home/kera/.claude/projects/-home-kera/memory/` にも進捗の要約がある（`MEMORY.md` が索引）。読むのは自由だが、更新は Claude Code 側に任せる。
