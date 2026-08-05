#!/usr/bin/env python3
"""フェーズ2 事前登録済み本分析 (2026-07-26)。

入力:
- metrics_dgv2_sustech.json  : 5指標 + Number of people + scratch精度 (キー "1".."13")
- metrics_ft_sustech_full_20260726.json : ft精度とΔ (キー subset名, 13/13)

分析(事前登録: Notion「0718再学習実験方針決め」+「0725進捗」残タスク):
A. LODO Ridge — 目的変数: Δ(pt) と ft acc_all。説明変数: log_people + 5指標
   (フェーズ1 lodo_ridge_sustech.py と同一パイプライン: StandardScaler + Ridge(alpha=1))
B. 優劣判定 pairwise logistic (leave-one-pair-out, tau=0.05, フェーズ1と同一) —
   目的変数: ft acc_all と Δ。Δはノイズ閾値 tau=0.0118 (1.18pt) の感度版も併記
C. 核心の問い「scratch精度を統制してもFID_ttがΔを予測するか」:
   偏Spearman (rank化→偏相関) と、LODO Ridge の特徴量アブレーション
D. scratch vs ft 比較: Spearman、rank flips (シードノイズ1.18pt閾値)

実行: docker exec opengait_container python /app/OpenGait/dgv2_analysis/analysis_ft_20260726.py
"""
import itertools
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, t as t_dist
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
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
NOISE_PT = 1.18  # 偵察2seedで実測したシードノイズ (Notion 0723)

TIE, A_BETTER, B_BETTER = 0, 1, 2


def load_df():
    with open(os.path.join(HERE, "metrics_dgv2_sustech.json"), encoding="utf-8") as f:
        scratch = json.load(f)
    with open(os.path.join(HERE, "metrics_ft_sustech_full_20260726.json"), encoding="utf-8") as f:
        ft = json.load(f)
    rows = []
    for sid, sub in ID2SUBSET.items():
        s, f_ = scratch[sid], ft[sub]
        assert abs(s["acc_all"] - f_["scratch_acc_all"]) < 1e-9, sub
        rows.append({
            "subset": sub,
            "log_people": math.log(s["Number of people"]),
            "DeepGaitV2_MSD": s["DeepGaitV2_MSD"],
            "DeepGaitV2_1NN": s["DeepGaitV2_1NN"],
            "DeepGaitV2_kNN": s["DeepGaitV2_kNN"],
            "DeepGaitV2_FID": s["DeepGaitV2_FID"],
            "FID_train_test_DeepGaitV2": s["FID_train_test_DeepGaitV2"],
            "scratch_all": s["acc_all"],          # fraction
            "ft_all": f_["acc_all"],              # fraction
            "delta_pt": f_["delta_acc_all_pt"],   # pt
        })
    df = pd.DataFrame(rows)
    assert len(df) == 13
    return df


def lodo_ridge(df, target_col, feature_cols):
    y_true, y_pred, coefs = [], [], []
    for i in range(len(df)):
        tr = df.drop(index=i)
        te = df.iloc[[i]]
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(tr[feature_cols].values, tr[target_col].values)
        y_pred.append(model.predict(te[feature_cols].values)[0])
        y_true.append(te[target_col].values[0])
        coefs.append(model.named_steps["ridge"].coef_)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(rho),
        "coef": {f: float(c) for f, c in zip(feature_cols, np.mean(coefs, axis=0))},
    }


def pairwise_logreg(df, target_col, tau):
    """leave-one-pair-out (cv=pair, フェーズ1既定) の優劣判定。"""
    ids = df["subset"].tolist()
    fmap = {r["subset"]: r[FEATURE_COLS].values.astype(float) for _, r in df.iterrows()}
    amap = {r["subset"]: float(r[target_col]) for _, r in df.iterrows()}
    feats, labels, keys = [], [], []
    for a, b in itertools.permutations(ids, 2):
        feats.append(fmap[a] - fmap[b])
        d = amap[a] - amap[b]
        labels.append(TIE if abs(d) <= tau else (A_BETTER if d > 0 else B_BETTER))
        keys.append((a, b))
    X, y = np.array(feats), np.array(labels)
    y_true_all, y_pred_all, coefs = [], [], []
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
        if clf.coef_.shape[0] == 3:
            coefs.append(clf.coef_)
    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
    res = {
        "target": target_col, "tau": tau, "n_pairs": int(len(y_true_all)),
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro")),
        "label_dist": {int(c): int((y_true_all == c).sum()) for c in np.unique(y_true_all)},
    }
    if coefs:
        coef_mean = np.mean(coefs, axis=0)
        importance = np.mean(np.abs(coef_mean), axis=0)
        res["importance"] = {f: float(v) for f, v in
                             sorted(zip(FEATURE_COLS, importance), key=lambda kv: -kv[1])}
    return res


def partial_spearman(x, y, z):
    """Spearman偏相関 rho(x, y | z) と両側p値 (t近似, df=n-3)。"""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    r = (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    n = len(x)
    df = n - 3
    t_stat = r * math.sqrt(df / (1 - r ** 2))
    p = 2 * t_dist.sf(abs(t_stat), df)
    return float(r), float(p)


def main():
    df = load_df()
    out = {}

    # ---------- A. LODO Ridge ----------
    print("=" * 70)
    print("A. LODO Ridge (StandardScaler + Ridge alpha=1.0, フェーズ1と同一)")
    out["ridge"] = {}
    for tgt in ["delta_pt", "ft_all"]:
        r = lodo_ridge(df, tgt, FEATURE_COLS)
        out["ridge"][tgt] = r
        print(f"\n----- target = {tgt} -----")
        print(f"MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  Spearman={r['spearman']:.4f}")
        for f, c in sorted(r["coef"].items(), key=lambda kv: -abs(kv[1])):
            print(f"  {f:28s} {c:+.4f}")

    # ---------- B. pairwise logistic ----------
    print("\n" + "=" * 70)
    print("B. 優劣判定 pairwise logistic (leave-one-pair-out)")
    out["logreg"] = {}
    for tgt, tau, name in [("ft_all", 0.05, "ft_all_tau0.05"),
                           ("delta_pt", 5.0, "delta_tau5pt"),
                           ("delta_pt", NOISE_PT, "delta_tau1.18pt")]:
        r = pairwise_logreg(df, tgt, tau)
        out["logreg"][name] = r
        print(f"\n----- target={tgt}, tau={tau} -----")
        print(f"pairs={r['n_pairs']}  label_dist(0=Tie,1=A,2=B)={r['label_dist']}")
        print(f"Accuracy={r['accuracy']:.4f}  Macro-F1={r['macro_f1']:.4f}")
        if "importance" in r:
            for f, v in r["importance"].items():
                print(f"  {f:28s} {v:.3f}")

    # ---------- C. 核心の問い ----------
    print("\n" + "=" * 70)
    print("C. scratch精度を統制してもFID_ttがΔを予測するか")
    fid = df["FID_train_test_DeepGaitV2"].values
    scr = df["scratch_all"].values
    dlt = df["delta_pt"].values
    rho_raw, p_raw = spearmanr(dlt, fid)
    r_p, p_p = partial_spearman(dlt, fid, scr)
    r_p2, p_p2 = partial_spearman(dlt, scr, fid)
    out["core_question"] = {
        "spearman_delta_fidtt_raw": {"rho": float(rho_raw), "p": float(p_raw)},
        "partial_spearman_delta_fidtt_given_scratch": {"rho": r_p, "p": p_p},
        "partial_spearman_delta_scratch_given_fidtt": {"rho": r_p2, "p": p_p2},
    }
    print(f"素のSpearman(Δ, FID_tt)            = {rho_raw:+.3f} (p={p_raw:.3f})")
    print(f"偏Spearman(Δ, FID_tt | scratch)    = {r_p:+.3f} (p={p_p:.3f})")
    print(f"偏Spearman(Δ, scratch | FID_tt)    = {r_p2:+.3f} (p={p_p2:.3f})")

    print("\nLODO Ridge アブレーション (target=Δpt):")
    out["ablation_delta"] = {}
    for name, cols in [
        ("scratchのみ", ["scratch_all"]),
        ("scratch+FID_tt", ["scratch_all", "FID_train_test_DeepGaitV2"]),
        ("FID_ttのみ", ["FID_train_test_DeepGaitV2"]),
        ("6指標(scratchなし)", FEATURE_COLS),
        ("scratch+6指標", ["scratch_all"] + FEATURE_COLS),
    ]:
        r = lodo_ridge(df, "delta_pt", cols)
        out["ablation_delta"][name] = {k: r[k] for k in ("mae", "rmse", "spearman")}
        print(f"  {name:20s} MAE={r['mae']:6.3f}pt  Spearman={r['spearman']:+.3f}")

    # ---------- D. scratch vs ft ----------
    print("\n" + "=" * 70)
    print("D. scratch vs ft 比較")
    rho_sf, _ = spearmanr(df["scratch_all"], df["ft_all"])
    print(f"Spearman(scratch, ft) = {rho_sf:.4f}")
    rs = df.sort_values("scratch_all", ascending=False)["subset"].tolist()
    rf = df.sort_values("ft_all", ascending=False)["subset"].tolist()
    flips = []
    amap_s = dict(zip(df["subset"], df["scratch_all"] * 100))
    amap_f = dict(zip(df["subset"], df["ft_all"] * 100))
    for a, b in itertools.combinations(df["subset"], 2):
        s_sign = np.sign(amap_s[a] - amap_s[b])
        f_sign = np.sign(amap_f[a] - amap_f[b])
        if s_sign != f_sign:
            flips.append({
                "pair": f"{a} vs {b}",
                "scratch_gap_pt": round(abs(amap_s[a] - amap_s[b]), 2),
                "ft_gap_pt": round(abs(amap_f[a] - amap_f[b]), 2),
                "scratch_gap_over_noise": bool(abs(amap_s[a] - amap_s[b]) > NOISE_PT),
                "ft_gap_over_noise": bool(abs(amap_f[a] - amap_f[b]) > NOISE_PT),
            })
    out["scratch_vs_ft"] = {
        "spearman": float(rho_sf),
        "rank_scratch": rs, "rank_ft": rf,
        "flips": flips,
        "range_scratch_pt": float((df["scratch_all"].max() - df["scratch_all"].min()) * 100),
        "range_ft_pt": float((df["ft_all"].max() - df["ft_all"].min()) * 100),
    }
    print(f"flip: {len(flips)}組")
    for fl in flips:
        sig = "★意味あり" if (not fl["scratch_gap_over_noise"] or fl["ft_gap_over_noise"]) and fl["ft_gap_over_noise"] else ("⚠scratch側が有意" if fl["scratch_gap_over_noise"] else "ノイズ内")
        print(f"  {fl['pair']:28s} scratch差{fl['scratch_gap_pt']:5.2f}pt ft差{fl['ft_gap_pt']:5.2f}pt {sig}")

    out_path = os.path.join(HERE, "result_ft_analysis_20260726.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
