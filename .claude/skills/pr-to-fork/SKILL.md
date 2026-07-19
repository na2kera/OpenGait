---
name: pr-to-fork
description: OpenGaitリポジトリのPR操作（作成・一覧・レビュー）を、本家 ShiqiYu/OpenGait ではなく自分のfork na2kera/OpenGait に必ず向ける手順。ghがデフォルトで本家に解決される罠への対策。「PRを作って」「プルリクを出して」「PRを確認して」「PRをレビューして」の時に必ず使う。
---

# OpenGait: PRは必ず na2kera/OpenGait に向ける

## 前提

| remote | リポジトリ | 役割 |
|---|---|---|
| `origin` | `na2kera/OpenGait` | 自分のfork。**PRのbaseはここ。デフォルトブランチは `master`** |
| `upstream` | `ShiqiYu/OpenGait` | 本家。**pushもPRも絶対にしない** |

`na2kera/OpenGait` はGitHub上で本家のforkとして登録されているため、`gh` はデフォルトのbaseリポジトリを**本家に解決する**。実測:

```console
$ gh repo view --json nameWithOwner
{"nameWithOwner":"ShiqiYu/OpenGait"}     # ← 本家に解決されてしまう
```

これが引き起こす事故は2種類:

- **誤PR**: オプション無しの `gh pr create` は本家（公開リポジトリ）へのPRになる
- **サイレント失敗**: `-R` 無しの `gh pr list` は本家の一覧を返すため、forkにOPENのPRがあっても「0件」に見える。この経路で得た「PRは無い」という結論は信用しない

## 恒久対策（設定済み、壊れていたら再設定）

確認:

```bash
gh repo view --json nameWithOwner
```

`na2kera/OpenGait` なら正常。`ShiqiYu/OpenGait` が返ったら設定が消えているので再設定する:

```bash
gh repo set-default na2kera/OpenGait
```

`set-default` コマンドが存在しないと言われたら（古いgh。v2.4.0で確認済み）、同じ効果のあるこちらを使う:

```bash
git config --local remote.origin.gh-resolved base
```

どちらも `.git/config` に入るので全worktreeで共有され、cloneし直すと消える。

## PR作成手順

恒久対策が入っていても **`-R` と `--base` は毎回明示する**（設定が消えていても事故らないための二重化）。

チェックリストをコピーして進捗を管理する:

```
- [ ] 1. baseリポジトリ確認（gh repo view → na2kera/OpenGait）
- [ ] 2. ブランチをoriginへpush
- [ ] 3. -R と --base を明示してPR作成
- [ ] 4. 作成後にPRのurlとbaseRefNameを検証
```

**1. baseリポジトリの確認**（省略禁止）

```bash
gh repo view --json nameWithOwner
```

`na2kera/OpenGait` 以外なら止まって上の恒久対策を再設定。

**2. ブランチをoriginへpush**（`upstream` へは絶対にpushしない）

```bash
git push -u origin <branch>
```

**3. PR作成 — `-R` と `--base` を必ず付ける**

```bash
gh pr create -R na2kera/OpenGait --base master --head <branch> \
  --title "<title>" --body "<body>"
```

- `--base master` : forkのデフォルトブランチは `master`（`main` ではない）
- `-R` を省略すると本家に飛ぶ。`--base` だけでは防げない
- `--fill` で対話プロンプトに任せない（非対話環境では本家を選ぶ）

**4. 作成後の検証**（省略禁止）

```bash
gh pr view <PR番号> -R na2kera/OpenGait --json url,baseRefName,headRefName,state
```

合格条件:

- `url` が `https://github.com/na2kera/OpenGait/pull/...` で始まる
- `baseRefName` が `master`

`url` に `ShiqiYu` が含まれていたら**即座にユーザーに報告**する。本家に対する操作になるため、勝手にcloseせず指示を仰ぐこと。

## PRを見る・レビューする時

読み取り系コマンドも `-R na2kera/OpenGait` を付ける。付け忘れると本家の情報を自分のforkのものだと誤認する。PR番号も `-R` 無しで解決しない（本家の同番号PRを掴む）。

```bash
gh pr list   -R na2kera/OpenGait
gh pr view   -R na2kera/OpenGait <PR番号>
gh pr status -R na2kera/OpenGait
```
