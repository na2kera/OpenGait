#!/usr/bin/env python3
"""目的変数の定義（acc_all=系列数重み付き平均 vs mean3=NM/BG/CL 単純平均）で LODO Ridge の結果がどう動くかの試算。

決定B-4（0907.md）の判断材料。本分析ではない（事前登録外・探索的）。
中村論文 表1・表3 の目的変数は NM/BG/CL Rank-1 の単純平均（metrics_ryu.json から 13/13 で検算済み）。
本研究の acc_all はプローブ系列数（NM 3222 / BG 1078 / CL 1076）で重み付けした平均。

手順:
  1. acc_all を目的変数にして X2 の E_grid の ρ/MAE を再現できることを確認（パイプライン同一性のガード）
  2. 同じ特徴行列で目的変数を mean3 に替えて ρ/MAE を出す
  3. scratch / ft / Δ × 4 特徴セットを表にする

実行: docker run --rm -v /home/kera/OpenGait:/app/OpenGait -v /home/kera/worktrees:/home/kera/worktrees \
        -w /app/OpenGait opengait:latest python dgv2_analysis/techrep_20260915/check_target_definition_20260907.py
出力: techrep_20260915/target_definition_check_20260907.{json,md}
"""
import json
import math
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WT = "/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis"
X2_METRICS = os.path.join(WT, "metrics_incdino_kera_X2.json")
X2_RESULT = os.path.join(WT, "result_incdino_ft_20260806_X2.json")
SUSTECH_JSON = os.path.join(REPO, "dgv2_analysis", "metrics_dgv2_sustech.json")
FT_JSON = os.path.join(REPO, "dgv2_analysis", "metrics_ft_sustech_full_20260726.json")

ID2SUBSET = {"1": "default", "2": "nm2", "3": "bg2", "4": "cl2", "5": "nm1-bg1", "6": "nm1-cl1", "7": "bg1-cl1",
             "8": "000-180", "9": "000-090", "10": "090-180", "11": "nm1-bg1-cl1", "12": "nm2-bg2-cl2", "13": "nm6"}
INC5 = ["Inception_MSD", "Inception_1NN", "Inception_kNN", "Inception_FID", "FID_train_test_Inception"]
DINO5 = ["DINO_MSD", "DINO_1NN", "DINO_kNN", "DINO_FID", "FID_train_test_DINO"]
SUS5 = ["DeepGaitV2_MSD", "DeepGaitV2_1NN", "DeepGaitV2_kNN", "DeepGaitV2_FID", "FID_train_test_DeepGaitV2"]
FEATURE_SETS = {"S_inc6": ["log_people"] + INC5, "S_dino6": ["log_people"] + DINO5,
                "S_incdino11": ["log_people"] + INC5 + DINO5, "S_sustech6": ["log_people"] + SUS5}


def lodo(X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        sc = StandardScaler().fit(X[tr])
        p[i] = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[i:i + 1]))[0]
    return p


def main():
    inc = json.load(open(X2_METRICS, encoding="utf-8"))
    sus = json.load(open(SUSTECH_JSON, encoding="utf-8"))
    ft = json.load(open(FT_JSON, encoding="utf-8"))
    x2 = json.load(open(X2_RESULT, encoding="utf-8"))
    assert x2["arm"] == "X2"

    rows = []
    for sid, s in ID2SUBSET.items():
        u, f_, m = sus[sid], ft[s], inc[sid]
        r = {"subset": s, "log_people": math.log(u["Number of people"])}
        for c in INC5 + DINO5:
            r[c] = m[c]
        for c in SUS5:
            r[c] = u[c]
        r["scratch_all"] = u["acc_all"]
        r["scratch_mean3"] = (u["acc_nm"] + u["acc_bg"] + u["acc_cl"]) / 3
        r["ft_all"] = f_["acc_all"]
        r["ft_mean3"] = (f_["acc_nm"] + f_["acc_bg"] + f_["acc_cl"]) / 3
        r["delta_all_pt"] = (r["ft_all"] - r["scratch_all"]) * 100
        r["delta_mean3_pt"] = (r["ft_mean3"] - r["scratch_mean3"]) * 100
        rows.append(r)
    names = [r["subset"] for r in rows]

    targets = {
        "scratch": ("scratch_all", "scratch_mean3", 100.0),
        "ft": ("ft_all", "ft_mean3", 100.0),
        "delta": ("delta_all_pt", "delta_mean3_pt", 1.0),
    }
    egrid_key = {"scratch": "scratch_all", "ft": "ft_all", "delta": "delta_pt"}
    out = {"rank_agreement": {}, "results": {}}

    # 目的変数の定義間の順位一致
    for tgt, (ka, km, _) in targets.items():
        ya = np.array([r[ka] for r in rows]); ym = np.array([r[km] for r in rows])
        out["rank_agreement"][tgt] = {"spearman_all_vs_mean3": float(spearmanr(ya, ym).correlation),
                                      "range_all_pt": float((ya.max() - ya.min()) * (100 if tgt != "delta" else 1)),
                                      "range_mean3_pt": float((ym.max() - ym.min()) * (100 if tgt != "delta" else 1))}

    print("--- ガード: acc_all で X2 E_grid を再現 ---")
    for fs, cols in FEATURE_SETS.items():
        X = np.array([[r[c] for c in cols] for r in rows])
        out["results"][fs] = {}
        for tgt, (ka, km, scale) in targets.items():
            ya = np.array([r[ka] for r in rows]); ym = np.array([r[km] for r in rows])
            pa, pm = lodo(X, ya), lodo(X, ym)
            rho_a, rho_m = spearmanr(pa, ya).correlation, spearmanr(pm, ym).correlation
            mae_a, mae_m = float(np.abs(pa - ya).mean() * scale), float(np.abs(pm - ym).mean() * scale)
            ref = x2["E_grid"][fs][egrid_key[tgt]]
            ref_mae = ref["mae"] * (100 if tgt != "delta" else 1)
            ok = abs(rho_a - ref["rho"]) < 1e-9 and abs(mae_a - ref_mae) < 1e-6
            print(f"  {'ok ' if ok else 'NG '} {fs:12s} {tgt:8s} all: rho={rho_a:.4f} (E_grid {ref['rho']:.4f}) MAE={mae_a:.2f} (E_grid {ref_mae:.2f})"
                  f"  |  mean3: rho={rho_m:.4f} MAE={mae_m:.2f}")
            if not ok:
                raise SystemExit(f"パイプライン再現に失敗: {fs}×{tgt}")
            out["results"][fs][tgt] = {"all": {"rho": float(rho_a), "mae_pt": mae_a},
                                       "mean3": {"rho": float(rho_m), "mae_pt": mae_m},
                                       "per_subset_mean3": {n: {"true": float(t * scale), "pred": float(p * scale)}
                                                            for n, t, p in zip(names, ym, pm)}}

    json.dump(out, open(os.path.join(HERE, "target_definition_check_20260907.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    md = ["# 目的変数の定義による LODO Ridge 結果の違い（試算・事前登録外）", "",
          "all = 系列数重み付き平均（本研究の acc_all、E_grid と一致を確認済み）、mean3 = NM/BG/CL 単純平均（中村論文の定義）。MAE は pt。", "",
          "| 特徴セット | 対象 | ρ (all) | ρ (mean3) | Δρ | MAE (all) | MAE (mean3) |", "|---|---|---:|---:|---:|---:|---:|"]
    for fs in FEATURE_SETS:
        for tgt in targets:
            a, m = out["results"][fs][tgt]["all"], out["results"][fs][tgt]["mean3"]
            md.append(f"| {fs} | {tgt} | {a['rho']:.3f} | {m['rho']:.3f} | {m['rho'] - a['rho']:+.3f} | {a['mae_pt']:.2f} | {m['mae_pt']:.2f} |")
    md += ["", "定義間の順位一致（13サブセット）:", ""]
    for tgt, v in out["rank_agreement"].items():
        md.append(f"- {tgt}: Spearman(all, mean3) = {v['spearman_all_vs_mean3']:.3f}, レンジ all {v['range_all_pt']:.1f}pt / mean3 {v['range_mean3_pt']:.1f}pt")
    md += ["", "注: mean3 は CL（精度が低く分散が大きい条件）の重みが 1/3 に増えるため、all より値が低くレンジが広い。優劣判定（logreg）は再計算していない。"]
    open(os.path.join(HERE, "target_definition_check_20260907.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
