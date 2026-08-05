#!/usr/bin/env python3
"""フェーズ2 追加分析 (2026-07-26): ②条件別(NM/BG/CL)回帰 と ④統計的補強。

②: LODO Ridge — 目的変数を ft acc_nm/bg/cl と Δ_nm/bg/cl(pt) に拡張
   (事前登録の「NM/BG/CL別」比較の定量版。フェーズ1のscratch側per-condition結果
    result_ridge_sustech.json と FID_tt 係数を突き合わせる)
④: 統計的補強 (事前登録外のオプション):
   - LODO Ridge Spearman の並べ替え検定 (目的変数を13サブセット間でシャッフルし
     LODO全体を再実行、1000回)
   - 優劣判定 pairwise logistic の並べ替え検定 (サブセットレベルで目的変数を
     シャッフルしてからペアを再構成、200回 — ペア単位のシャッフルは依存構造を
     壊すため不可)
   - 指標間 Spearman 相関行列 (FID_tt と scratch の情報重複の可視化)

実行: docker exec opengait_container python /app/OpenGait/dgv2_analysis/analysis_extra_20260726.py
"""
import itertools
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))

ID2SUBSET = {"1": "default", "2": "nm2", "3": "bg2", "4": "cl2", "5": "nm1-bg1",
             "6": "nm1-cl1", "7": "bg1-cl1", "8": "000-180", "9": "000-090",
             "10": "090-180", "11": "nm1-bg1-cl1", "12": "nm2-bg2-cl2", "13": "nm6"}

FEATURE_COLS = [
    "log_people",
    "DeepGaitV2_MSD",
    "DeepGaitV2_1NN",
    "DeepGaitV2_kNN",
    "DeepGaitV2_FID",
    "FID_train_test_DeepGaitV2",
]
TIE, A_BETTER, B_BETTER = 0, 1, 2
RNG = np.random.RandomState(0)
N_PERM_RIDGE = 1000
N_PERM_LOGREG = 200


def load_df():
    with open(os.path.join(HERE, "metrics_dgv2_sustech.json"), encoding="utf-8") as f:
        scratch = json.load(f)
    with open(os.path.join(HERE, "metrics_ft_sustech_full_20260726.json"), encoding="utf-8") as f:
        ft = json.load(f)
    rows = []
    for sid, sub in ID2SUBSET.items():
        s, f_ = scratch[sid], ft[sub]
        row = {"subset": sub, "log_people": math.log(s["Number of people"])}
        for c in FEATURE_COLS[1:]:
            row[c] = s[c]
        for cond in ("nm", "bg", "cl", "all"):
            row[f"scratch_{cond}"] = s[f"acc_{cond}"]
            row[f"ft_{cond}"] = f_[f"acc_{cond}"]
            row[f"delta_{cond}_pt"] = (f_[f"acc_{cond}"] - s[f"acc_{cond}"]) * 100
        rows.append(row)
    df = pd.DataFrame(rows)
    assert len(df) == 13
    return df


def lodo_ridge(df, target_col, feature_cols=FEATURE_COLS, y_override=None):
    y_all = df[target_col].values if y_override is None else y_override
    y_true, y_pred, coefs = [], [], []
    X_all = df[feature_cols].values
    for i in range(len(df)):
        mask = np.arange(len(df)) != i
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(X_all[mask], y_all[mask])
        y_pred.append(model.predict(X_all[~mask])[0])
        y_true.append(y_all[~mask][0])
        coefs.append(model.named_steps["ridge"].coef_)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(rho),
        "coef": {f: float(c) for f, c in zip(feature_cols, np.mean(coefs, axis=0))},
    }


def pairwise_logreg_acc(df, y_map, tau):
    """leave-one-pair-out の Accuracy のみ返す軽量版。y_map: subset->target値。"""
    ids = df["subset"].tolist()
    fmap = {r["subset"]: r[FEATURE_COLS].values.astype(float) for _, r in df.iterrows()}
    feats, labels, keys = [], [], []
    for a, b in itertools.permutations(ids, 2):
        feats.append(fmap[a] - fmap[b])
        d = y_map[a] - y_map[b]
        labels.append(TIE if abs(d) <= tau else (A_BETTER if d > 0 else B_BETTER))
        keys.append((a, b))
    X, y = np.array(feats), np.array(labels)
    y_true_all, y_pred_all = [], []
    for i, (a, b) in enumerate(keys):
        train_mask = np.array([set(k) != {a, b} for k in keys])
        if len(np.unique(y[train_mask])) < 2:
            continue
        scaler = StandardScaler().fit(X[train_mask])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                 multi_class="multinomial")
        clf.fit(scaler.transform(X[train_mask]), y[train_mask])
        y_pred_all.append(clf.predict(scaler.transform(X[i:i + 1]))[0])
        y_true_all.append(y[i])
    return accuracy_score(y_true_all, y_pred_all)


def main():
    df = load_df()
    out = {}

    # ---------- ② 条件別 (NM/BG/CL) LODO Ridge ----------
    print("=" * 70)
    print("② 条件別 LODO Ridge (説明変数: log_people + 5指標)")
    with open(os.path.join(HERE, "result_ridge_sustech.json")) as f:
        phase1 = json.load(f)
    out["per_condition_ridge"] = {}
    print(f"\n{'target':16s} {'MAE':>8s} {'Spearman':>9s} {'FID_tt係数':>10s}  (参考: scratch側FID_tt係数)")
    for cond in ("nm", "bg", "cl"):
        r_ft = lodo_ridge(df, f"ft_{cond}")
        r_dl = lodo_ridge(df, f"delta_{cond}_pt")
        out["per_condition_ridge"][f"ft_{cond}"] = r_ft
        out["per_condition_ridge"][f"delta_{cond}_pt"] = r_dl
        p1 = phase1[f"acc_{cond}"]["coef"]["FID_train_test_DeepGaitV2"]
        print(f"ft_{cond:12s} {r_ft['mae']:8.4f} {r_ft['spearman']:9.3f} "
              f"{r_ft['coef']['FID_train_test_DeepGaitV2']:+10.4f}  (scratch: {p1:+.4f}, "
              f"ρ={phase1[f'acc_{cond}']['spearman']:.3f})")
        print(f"Δ_{cond}(pt){'':6s} {r_dl['mae']:8.4f} {r_dl['spearman']:9.3f} "
              f"{r_dl['coef']['FID_train_test_DeepGaitV2']:+10.4f}")

    # ---------- ④-1 並べ替え検定: LODO Ridge ----------
    print("\n" + "=" * 70)
    print(f"④-1 並べ替え検定 (LODO Ridge Spearman, {N_PERM_RIDGE}回)")
    out["permutation_ridge"] = {}
    for tgt in ("delta_all_pt", "ft_all"):
        obs = lodo_ridge(df, tgt)["spearman"]
        y = df[tgt].values.copy()
        count = 0
        for _ in range(N_PERM_RIDGE):
            perm = RNG.permutation(y)
            rho = lodo_ridge(df, tgt, y_override=perm)["spearman"]
            if rho >= obs:
                count += 1
        p = (count + 1) / (N_PERM_RIDGE + 1)
        out["permutation_ridge"][tgt] = {"observed_spearman": obs, "p_perm": p}
        print(f"  {tgt:14s} 実測ρ={obs:.3f}  並べ替えp={p:.4f}")

    # ---------- ④-2 並べ替え検定: pairwise logistic ----------
    print(f"\n④-2 並べ替え検定 (優劣判定Accuracy, サブセットレベルシャッフル, {N_PERM_LOGREG}回)")
    out["permutation_logreg"] = {}
    for tgt, tau, name in [("delta_all_pt", 5.0, "delta_tau5pt"),
                           ("ft_all", 0.05, "ft_tau0.05")]:
        y_map = dict(zip(df["subset"], df[tgt]))
        obs = pairwise_logreg_acc(df, y_map, tau)
        vals = df[tgt].values.copy()
        count = 0
        for _ in range(N_PERM_LOGREG):
            perm = RNG.permutation(vals)
            pm = dict(zip(df["subset"], perm))
            acc = pairwise_logreg_acc(df, pm, tau)
            if acc >= obs:
                count += 1
        p = (count + 1) / (N_PERM_LOGREG + 1)
        out["permutation_logreg"][name] = {"observed_accuracy": obs, "p_perm": p}
        print(f"  {name:14s} 実測Acc={obs:.3f}  並べ替えp={p:.4f}")

    # ---------- ④-3 指標間相関行列 ----------
    print("\n④-3 Spearman相関行列 (指標 + scratch/ft/Δ)")
    cols = FEATURE_COLS + ["scratch_all", "ft_all", "delta_all_pt"]
    mat = df[cols].corr(method="spearman")
    short = {"log_people": "logP", "DeepGaitV2_MSD": "MSD", "DeepGaitV2_1NN": "1NN",
             "DeepGaitV2_kNN": "kNN", "DeepGaitV2_FID": "FID",
             "FID_train_test_DeepGaitV2": "FID_tt", "scratch_all": "scratch",
             "ft_all": "ft", "delta_all_pt": "Δ"}
    mat_s = mat.rename(index=short, columns=short).round(2)
    print(mat_s.to_string())
    out["spearman_matrix"] = {r: {c: float(mat.iloc[i, j]) for j, c in enumerate(cols)}
                              for i, r in enumerate(cols)}

    out_path = os.path.join(HERE, "result_extra_analysis_20260726.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
