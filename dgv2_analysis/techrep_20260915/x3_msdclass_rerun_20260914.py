#!/usr/bin/env python3
"""X3: 汎用画像特徴の MSD（Inception_MSD / DINO_MSD）を式(1)（人物ごと）版に差し替えて、原稿に使う数値を再計算する（2026-09-14）。

入力
  - X2 指標: /home/kera/worktrees/incdino-ft-phase3/dgv2_analysis/metrics_incdino_kera_X2.json（MSD 2 列以外はそのまま）
  - s5 結果: /home/kera/OpenGait/incdino_recalc/results/MSDclass_results_<subset>_<ts>.json（13 本。gate.all_pass=True かつ
             対応ログ logs/s5_msdclass_<subset>.log の最終区間に JOB_DONE rc=0 があるものだけ採用。複数あれば最新）
出力（同ディレクトリ）
  - metrics_incdino_kera_X3.json（_provenance 付き。X2 の md5、各 s5 JSON の md5、差し替えセル）
  - result_x3_msdclass_20260914.json / x3_vs_x2_20260914.md（2×2 の ρ/MAE/p、条件別、指標↔精度の相関、MSD 変動幅を X2 と並べる）
分析は X2 本体（analysis_incdino_ft_20260806.py）の lodo_ridge / FEATURE_SETS / load_df をそのまま import する。
X2 本体は G4 で Inception_MSD の値域（0.050〜0.051）を検査するため式(1)版には使えない（ここでは触らない）。

実行: docker run --rm -v /home/kera/OpenGait:/app/OpenGait -v /home/kera/worktrees:/home/kera/worktrees \
        -w /app/OpenGait opengait:latest python dgv2_analysis/techrep_20260915/x3_msdclass_rerun_20260914.py
"""
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WT = "/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis"
sys.path.insert(0, WT)
import analysis_incdino_ft_20260806 as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RECALC = "/home/kera/OpenGait/incdino_recalc" if os.path.exists("/home/kera/OpenGait/incdino_recalc") else "/app/OpenGait/incdino_recalc"
X2_METRICS = os.path.join(WT, A.ARM_DEFAULTS["X2"]["incdino"])
X2_RESULT = os.path.join(WT, A.ARM_DEFAULTS["X2"]["out"])
ID2NAME = {"1": "20-data", "2": "nm1-2", "3": "bg1-2", "4": "cl1-2", "5": "nm1-bg1", "6": "nm1-cl1", "7": "bg1-cl1",
           "8": "000-180", "9": "000-090", "10": "090-180", "11": "nm1-bg1-cl1", "12": "nm2-bg2-cl2", "13": "nm6"}
N_PERM = 1000


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def latest_valid_s5(subset):
    """gate 通過・JOB_DONE rc=0 の最新 s5 JSON を返す（無ければ None）。"""
    log = os.path.join(RECALC, "logs", f"s5_msdclass_{subset}.log")
    if not os.path.exists(log):
        return None, "log missing"
    txt = open(log, errors="replace").read()
    seg = txt.split("=== RUN_CONTEXT begin")[-1]
    if not re.search(r"JOB_DONE rc=0\b", seg):
        return None, "last run not JOB_DONE rc=0"
    m = re.findall(r"saved (results/MSDclass_results_%s_[0-9_-]+\.json)" % re.escape(subset), seg)
    if len(m) != 1:
        return None, f"saved line count {len(m)} != 1"
    p = os.path.join(RECALC, m[0])
    if not os.path.exists(p):
        return None, "json missing"
    d = json.load(open(p))
    if not d.get("gate", {}).get("all_pass"):
        return None, "gate not passed"
    if d["subset"] != subset:
        return None, "subset mismatch"
    # JSON が指す特徴ファイルが現物と一致すること（JSON と保存特徴の対応の証明）
    for k, v in d["feature_files"].items():
        host = v["path"].replace("/feat/", "/home/kera/incdino_data/msdclass_features/", 1)
        if not os.path.exists(host):
            return None, f"feature file missing: {host}"
        if md5(host) != v["md5"]:
            return None, f"feature md5 mismatch: {host}"
    if not (0.049 < d["gate"]["inception"]["adopted"] < 0.053):
        return None, "adopted target out of expected range"
    return (p, d), "ok"


def main():
    x2 = json.load(open(X2_METRICS))
    x3 = {k: (dict(v) if not k.startswith("_") else v) for k, v in x2.items()}
    prov = {"arm": "X3", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "generator": os.path.relpath(__file__, "/"),
            "base_X2": {"path": X2_METRICS, "md5": md5(X2_METRICS)}, "replaced_columns": ["Inception_MSD", "DINO_MSD"],
            "definition": "eq.(1): per-subject mean of ||f_i - mu_s||^2, averaged over subjects (float64); previously MSDall.py global variance",
            "sources": {}, "replaced_cells": [],
            "s5_template_md5": md5(os.path.join(RECALC, "scripts_kera", "s5_msdclass_TEMPLATE.py")),
            "s5_targets_md5": md5(os.path.join(RECALC, "scripts_kera", "s5_msd_targets.json"))}
    missing = []
    for sid, name in ID2NAME.items():
        r, why = latest_valid_s5(name)
        if r is None:
            missing.append((name, why)); continue
        p, d = r
        prov["sources"][name] = {"path": p, "md5": md5(p), "n_images": d["n_images"], "n_subjects": d["n_subjects"],
                                 "gate": d["gate"], "feature_md5": {k: v["md5"] for k, v in d["feature_files"].items()}}
        for col, space in (("Inception_MSD", "inception"), ("DINO_MSD", "dinov2")):
            old, new = x3[sid][col], d["class_msd"][space]["all"]
            x3[sid][col] = new
            prov["replaced_cells"].append({"id": sid, "subset": name, "column": col, "old_global": old, "new_class": new,
                                           "class_by_cond": {c: d["class_msd"][space].get(c) for c in ("nm", "bg", "cl")}})
    if missing:
        print("未完了:", missing); sys.exit(2)
    x3["_provenance"] = prov
    out_metrics = os.path.join(HERE, "metrics_incdino_kera_X3.json")
    json.dump(x3, open(out_metrics, "w"), ensure_ascii=False, indent=1)
    print("wrote", out_metrics)

    # ---- 再計算（X2 と X3 を同じ関数で）
    res = {"note": __doc__, "n_perm": N_PERM, "arms": {}}
    dfs = {"X2": A.load_df(X2_METRICS)[0], "X3": A.load_df(out_metrics)[0]}
    for arm, df in dfs.items():
        R = {"grid": {}, "per_condition": {}, "corr": {}, "msd_width": {}, "permutation": {}}
        for fs in ("S_inc6", "S_dino6", "S_incdino11", "S_sustech6"):
            R["grid"][fs] = {}
            for tgt in ("scratch_all", "ft_all", "delta_pt"):
                r = A.lodo_ridge(df, tgt, A.FEATURE_SETS[fs], keep_pred=True)
                R["grid"][fs][tgt] = {"rho": r["spearman"], "mae": r["mae"], "coef": r["coef"], "per_subset": r["per_subset"]}
            R["per_condition"][fs] = {}
            for cond in A.CONDS:
                for side in ("scratch", "ft"):
                    r = A.lodo_ridge(df, f"{side}_{cond}", A.FEATURE_SETS[fs])
                    R["per_condition"][fs][f"{side}_{cond}"] = {"rho": r["spearman"], "mae": r["mae"]}
        for col in ("Inception_MSD", "DINO_MSD", "DeepGaitV2_MSD", "FID_train_test_Inception", "FID_train_test_DINO", "FID_train_test_DeepGaitV2"):
            x = df[col].values
            R["corr"][col] = {t: float(spearmanr(x, df[t])[0]) for t in ("scratch_all", "ft_all", "scratch_nm", "ft_nm", "scratch_cl", "ft_cl")}
            R["msd_width"][col] = float((x.max() - x.min()) / x.mean())
        # 並べ替え検定（主 endpoint 相当: S_incdino11 / S_inc6 / S_dino6 × ft_all, scratch_all）。seed は X2/事後追加と同じ体系
        for k, (fs, tgt, seed) in enumerate([("S_incdino11", "ft_all", 1000), ("S_inc6", "ft_all", 1002), ("S_dino6", "ft_all", 1004),
                                             ("S_incdino11", "scratch_all", 1010), ("S_inc6", "scratch_all", 1011), ("S_dino6", "scratch_all", 1012)]):
            if arm == "X2":
                continue  # X2 の p は既存結果（result_incdino_ft_20260806_X2.json / perm_scratch_20260910.json）を使う
            rng = np.random.RandomState(seed)
            obs = R["grid"][fs][tgt]["rho"]; y = df[tgt].values.copy()
            cnt = sum(A.lodo_ridge(df, tgt, A.FEATURE_SETS[fs], y_override=rng.permutation(y))["spearman"] >= obs for _ in range(N_PERM))
            R["permutation"][f"{fs}_{tgt}"] = {"observed_spearman": obs, "p_perm": (cnt + 1) / (N_PERM + 1), "seed": seed, "count_ge": int(cnt)}
            print(f"  perm {arm} {fs}×{tgt}: rho={obs:.3f} p={(cnt+1)/(N_PERM+1):.4f}", flush=True)
        res["arms"][arm] = R
    res["inputs_md5"] = {"X2_metrics": md5(X2_METRICS), "X3_metrics": md5(out_metrics), "analysis_code": md5(A.__file__)}
    json.dump(res, open(os.path.join(HERE, "result_x3_msdclass_20260914.json"), "w"), ensure_ascii=False, indent=1)

    # ---- 比較 md
    L = ["# X3（汎用 MSD を式(1)版に差し替え）vs X2（全体分散）", "", f"生成: x3_msdclass_rerun_20260914.py。LODO Ridge は X2 本体と同一関数。並べ替え検定 N={N_PERM}（X3 のみ再計算）。", "",
         "## 2×2（ρ / MAE pt）", "", "| 特徴セット | 対象 | X2 ρ | X3 ρ | Δρ | X2 MAE | X3 MAE |", "|---|---|---:|---:|---:|---:|---:|"]
    for fs in ("S_inc6", "S_dino6", "S_incdino11", "S_sustech6"):
        for tgt in ("scratch_all", "ft_all"):
            a, b = res["arms"]["X2"]["grid"][fs][tgt], res["arms"]["X3"]["grid"][fs][tgt]
            L.append(f"| {fs} | {tgt} | {a['rho']:.3f} | {b['rho']:.3f} | {b['rho']-a['rho']:+.3f} | {a['mae']*100:.2f} | {b['mae']*100:.2f} |")
    L += ["", "## X3 の並べ替え p", "", "| 検定 | ρ | p | 超過 |", "|---|---:|---:|---:|"]
    for k, v in res["arms"]["X3"]["permutation"].items():
        L.append(f"| {k} | {v['observed_spearman']:.3f} | {v['p_perm']:.4f} | {v['count_ge']}/{N_PERM} |")
    L += ["", "## 条件別 ρ（S_incdino11 / S_sustech6）", "", "| 対象 | X2 汎用 | X3 汎用 | 歩容特化（不変） |", "|---|---:|---:|---:|"]
    for key in ("scratch_nm", "scratch_bg", "scratch_cl", "ft_nm", "ft_bg", "ft_cl"):
        L.append(f"| {key} | {res['arms']['X2']['per_condition']['S_incdino11'][key]['rho']:.3f} | {res['arms']['X3']['per_condition']['S_incdino11'][key]['rho']:.3f} | {res['arms']['X3']['per_condition']['S_sustech6'][key]['rho']:.3f} |")
    L += ["", "## MSD 生値と精度の Spearman・変動幅", "", "| 指標 | arm | scratch_cl | ft_cl | scratch_nm | ft_nm | 相対幅 |", "|---|---|---:|---:|---:|---:|---:|"]
    for col in ("Inception_MSD", "DINO_MSD", "DeepGaitV2_MSD"):
        for arm in ("X2", "X3"):
            c = res["arms"][arm]["corr"][col]; w = res["arms"][arm]["msd_width"][col]
            L.append(f"| {col} | {arm} | {c['scratch_cl']:+.2f} | {c['ft_cl']:+.2f} | {c['scratch_nm']:+.2f} | {c['ft_nm']:+.2f} | {w:.1%} |")
    L += ["", "## 差し替えセル（全体分散 → 式(1)）", "", "| subset | Inception 旧→新 | DINOv2 旧→新 |", "|---|---|---|"]
    by = {}
    for c in prov["replaced_cells"]:
        by.setdefault(c["subset"], {})[c["column"]] = c
    for name in ID2NAME.values():
        i, d = by[name]["Inception_MSD"], by[name]["DINO_MSD"]
        L.append(f"| {name} | {i['old_global']:.5g} → {i['new_class']:.5g} | {d['old_global']:.5g} → {d['new_class']:.5g} |")
    open(os.path.join(HERE, "x3_vs_x2_20260914.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
