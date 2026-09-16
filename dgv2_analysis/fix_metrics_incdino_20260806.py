#!/usr/bin/env python3
"""
フェーズ3（0805進捗 設計v4.1）§8 手順1。

metrics_ryu.json の incdino 説明変数のうち、計算源と不一致の3セルを
計算源JSONから読んだ生値で置き換えた metrics_incdino_fixed_20260806.json を作る。

方針（設計より）:
  - 原本 metrics_ryu.json は変更しない
  - リテラルの手転記をしない（計算源JSONから読んだ値をそのまま書き込む）
  - 修正セル・修正値・根拠パス・検証結果を _provenance としてJSON内に埋め込む
  - ホストのpython3で実行（/home/ryu の原典を読むため。依存は標準ライブラリのみ）
"""
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))
HERE = os.path.dirname(os.path.abspath(__file__))

SRC_METRICS = os.path.join(HERE, "metrics_ryu.json")
SRC_METRICS_MD5 = "f7d9ced09650bd28e622985558375c46"          # 設計§3で固定
SRC_MSD = "/home/ryu/results/MSD_results_2025-11-15_17-11-15.json"
SRC_FIDTT = "/home/ryu/fid_train_test_results.json"
SNAPSHOT_FIDTT = os.path.join(HERE, "fid_train_test_results_ryu_snapshot.json")
OUT = os.path.join(HERE, "metrics_incdino_fixed_20260806.json")

# 設計§8のキー対応（PNGディレクトリ名 → 集計表id）
ID2DIR = {
    "1": "20-data-png", "2": "nm1-2-png", "3": "bg1-2-png", "4": "cl1-2-png",
    "5": "nm1-bg1-png", "6": "nm1-cl1-png", "7": "bg1-cl1-png",
    "8": "000-180-png", "9": "000-090-png", "10": "090-180-png",
    "11": "nm1-bg1-cl1-png", "12": "nm2-bg2-cl2-png", "13": "nm6-png",
}
RYU_PREFIX = "./OpenGait/CASIA-B-png/"

# 集計表の列 → MSD計算源JSONの列。採用条件は "all"（13サブセット全体で確認済み）
MSD_COLS = {"MSD_raw": "MSD_image_space",
            "Inception_MSD": "MSD_inception",
            "DINO_MSD": "MSD_dinov2"}
MSD_COND = "all"

# 設計§3で修正対象と決まった3セル。旧値は「一致してはならない値」としてのみ使う
FIX_CELLS = [
    ("3", "Inception_MSD", 0.5035662, "#1 計算源の10倍"),
    ("4", "Inception_MSD", 0.500962, "#1 計算源の10倍"),
    ("11", "MSD_raw", 2066.2546, "#2 id13の値が混入"),
]

FIDTT_COLS = {"FID_train_test_Inception": "Inception",
              "FID_train_test_DINO": "DINOv2"}


def fail(msg):
    print(f"  ✗ {msg}")
    sys.exit(1)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    prov = {
        "generated_at": datetime.now(JST).isoformat(),
        "generator": os.path.relpath(os.path.abspath(__file__), os.path.dirname(HERE)),
        "design": "Notion 0805進捗 (フェーズ3 設計v4.1) §3・§8手順1",
        "sources": {},
        "checks": {},
        "fixed_cells": [],
        "notes": [],
    }

    # --- 1. 原本のmd5検証 -------------------------------------------------
    print("[1] 原本の同一性")
    got = md5(SRC_METRICS)
    if got != SRC_METRICS_MD5:
        fail(f"metrics_ryu.json のmd5が設計で固定した値と異なる: {got} != {SRC_METRICS_MD5}")
    print(f"  ✓ metrics_ryu.json md5 = {got}")
    prov["sources"]["metrics_ryu.json"] = {"path": SRC_METRICS, "md5": got}

    for p in (SRC_MSD, SRC_FIDTT):
        if not os.path.exists(p):
            fail(f"計算源が見つからない: {p}")
        prov["sources"][os.path.basename(p)] = {"path": p, "md5": md5(p)}
        print(f"  ✓ {p}")

    metrics = json.load(open(SRC_METRICS, encoding="utf-8"))
    msd_src = json.load(open(SRC_MSD, encoding="utf-8"))
    fidtt_src = json.load(open(SRC_FIDTT, encoding="utf-8"))

    # --- 2. MSD系の全セル突合（3セルの例外を含む） -----------------------
    print("[2] MSD系 13サブセット×3列の突合（採用条件 = all）")
    known = {(i, c) for i, c, _, _ in FIX_CELLS}
    mismatches, checked = [], 0
    for i, d in ID2DIR.items():
        s = msd_src[RYU_PREFIX + d]
        if MSD_COND not in s:
            fail(f"id{i} ({d}) に条件 '{MSD_COND}' が無い")
        for tcol, scol in MSD_COLS.items():
            checked += 1
            tv, sv = metrics[i][tcol], s[MSD_COND][scol]
            if round(tv, 4) != round(sv, 4):
                mismatches.append((i, tcol, tv, sv))
    unexpected = [m for m in mismatches if (m[0], m[1]) not in known]
    missing = [k for k in known if k not in {(m[0], m[1]) for m in mismatches}]
    if unexpected:
        for i, c, tv, sv in unexpected:
            print(f"  ✗ 想定外の不一致 id{i} {c}: 集計表={tv} 計算源={sv}")
        fail(f"想定外の不一致が {len(unexpected)} 件。設計§3の想定と違うので停止する")
    if missing:
        fail(f"修正対象のはずのセルが一致してしまっている: {missing}（原本が想定と違う）")
    print(f"  ✓ {checked} セル中、不一致は設計§3の想定3セルのみ")
    prov["checks"]["msd_all_cells"] = {
        "condition": MSD_COND, "checked": checked, "tolerance": "round(x,4)",
        "mismatches": [{"id": i, "column": c, "table": tv, "source": sv}
                       for i, c, tv, sv in mismatches],
    }

    # --- 3. FID_tt の突合とスナップショット ------------------------------
    print("[3] FID_train_test 13×2 の突合")
    n_ok = 0
    for tcol, model in FIDTT_COLS.items():
        for i, d in ID2DIR.items():
            key = RYU_PREFIX + d
            if key not in fidtt_src[model]:
                fail(f"{model} に {key} が無い")
            tv, sv = metrics[i][tcol], fidtt_src[model][key]
            if round(sv, 4) != tv:
                fail(f"id{i} {tcol}: 集計表={tv} 計算源の4桁丸め={round(sv, 4)}")
            n_ok += 1
    print(f"  ✓ {n_ok}/26 が4桁丸めで一致")
    shutil.copyfile(SRC_FIDTT, SNAPSHOT_FIDTT)
    print(f"  ✓ スナップショット: {os.path.basename(SNAPSHOT_FIDTT)}")
    prov["checks"]["fid_train_test"] = {
        "matched": n_ok, "total": 26, "tolerance": "round(source,4) == table",
        "snapshot": os.path.basename(SNAPSHOT_FIDTT),
        "definition_source": "/home/ryu/caldataset-CASIA-B-FID-train-test.py",
    }

    # --- 4. 3セルの修正 ---------------------------------------------------
    print("[4] 修正3セルの書き込み（計算源の生値をそのまま）")
    fixed = json.loads(json.dumps(metrics))          # deep copy
    for i, col, old_expected, reason in FIX_CELLS:
        cur = fixed[i][col]
        if cur != old_expected:
            fail(f"id{i} {col} の現在値 {cur} が想定の旧値 {old_expected} と違う")
        new = msd_src[RYU_PREFIX + ID2DIR[i]][MSD_COND][MSD_COLS[col]]
        fixed[i][col] = new
        print(f"  ✓ id{i:>2} {col:<14} {old_expected} → {new!r}  ({reason})")
        prov["fixed_cells"].append({
            "id": i, "column": col, "old": old_expected, "new": new, "reason": reason,
            "source": SRC_MSD, "source_key": RYU_PREFIX + ID2DIR[i],
            "source_field": f"{MSD_COND}.{MSD_COLS[col]}",
        })

    prov["notes"].append(
        "集計表は列ごとに精度が混在する（id1-4・id13の一部は約4桁丸め、id5-12は16桁生値）。"
        "本スクリプトは設計§3の3セルのみを計算源の生値に置き換えるため、修正セルだけ精度が上がる。"
        "標準化後の回帰には影響しない水準（差 <1e-7）。"
    )
    prov["notes"].append(
        "MSDの採用条件が 'all' であることは、13サブセット×3列の突合で確認した"
        "（不一致は設計§3の3セルのみ）。"
    )
    prov["notes"].append(
        "設計§3-#3（id11のDINO_1NN/kNN）は計算源が残っていないため修正しない。S_9var感度で扱う。"
    )
    prov["notes"].append(
        "設計§3-#4（71人問題）は本スクリプトの対象外。原因は未特定で、後続工程のゲートとして管理する。"
    )

    fixed["_provenance"] = prov
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)
    print(f"[5] 出力: {os.path.basename(OUT)}  (md5 {md5(OUT)})")
    print(f"  ✓ 原本 metrics_ryu.json は未変更（md5 {md5(SRC_METRICS)}）")


if __name__ == "__main__":
    main()
