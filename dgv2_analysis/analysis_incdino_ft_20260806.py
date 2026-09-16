#!/usr/bin/env python3
"""フェーズ3 事前登録済み本分析（設計: Notion「0805進捗」v4.1 §5-§8）。

Inception/DINOv2 汎用特徴空間の指標で fine-tuning 精度・Δ を予測できるかを検証する。
新規学習ゼロ・新規特徴抽出ゼロ。CPUのみ・決定的（乱数は並べ替え検定のみ、シード固定）。

実行:
  docker run --rm -v /home/kera/OpenGait:/app/OpenGait -w /app/OpenGait opengait:latest \
      python dgv2_analysis/analysis_incdino_ft_20260806.py
"""
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, t as t_dist
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

ID2SUBSET = {"1": "default", "2": "nm2", "3": "bg2", "4": "cl2", "5": "nm1-bg1",
             "6": "nm1-cl1", "7": "bg1-cl1", "8": "000-180", "9": "000-090",
             "10": "090-180", "11": "nm1-bg1-cl1", "12": "nm2-bg2-cl2", "13": "nm6"}
# 設計§8のPNGディレクトリ対応（G3のスナップショット突合に使う）
ID2DIR = {"1": "20-data-png", "2": "nm1-2-png", "3": "bg1-2-png", "4": "cl1-2-png",
          "5": "nm1-bg1-png", "6": "nm1-cl1-png", "7": "bg1-cl1-png",
          "8": "000-180-png", "9": "000-090-png", "10": "090-180-png",
          "11": "nm1-bg1-cl1-png", "12": "nm2-bg2-cl2-png", "13": "nm6-png"}
RYU_PREFIX = "./OpenGait/CASIA-B-png/"

INC5 = ["Inception_MSD", "Inception_1NN", "Inception_kNN", "Inception_FID",
        "FID_train_test_Inception"]
DINO5 = ["DINO_MSD", "DINO_1NN", "DINO_kNN", "DINO_FID", "FID_train_test_DINO"]
SUS5 = ["DeepGaitV2_MSD", "DeepGaitV2_1NN", "DeepGaitV2_kNN", "DeepGaitV2_FID",
        "FID_train_test_DeepGaitV2"]

FEATURE_SETS = {
    "S_incdino11": ["log_people"] + INC5 + DINO5,
    "S_inc6": ["log_people"] + INC5,
    "S_dino6": ["log_people"] + DINO5,
    "S_9var": ["log_people"] + INC5 + [c for c in DINO5 if c not in ("DINO_1NN", "DINO_kNN")],
    "S_sustech6": ["log_people"] + SUS5,          # 参照のみ（A''と劣化時診断・方式2）
}
TARGETS = ["ft_all", "delta_pt", "scratch_all"]

NOISE_PT = 1.18          # 偵察2seedのシードノイズ（Notion 0723）
PRACTICAL_MAE_PT = 2.36  # = NOISE_PT × 2（設計§7 実務判定）
N_PERM_RIDGE = 1000
N_PERM_LOGREG = 200
N_PERM_FL = 10000        # Freedman-Lane（C系）
TIE, A_BETTER, B_BETTER = 0, 1, 2

# 設計§3で修正されたセル（G4の旧値チェック用。修正値の正本は _provenance）
OLD_VALUES = {("3", "Inception_MSD"): 0.5035662,
              ("4", "Inception_MSD"): 0.500962,
              ("11", "MSD_raw"): 2066.2546}
G5_CONFIGS = ["configs/deepgaitv2/DeepGaitV2_casiab-nAG-000-180.yaml",
              "configs/deepgaitv2/DeepGaitV2_casiab-nAG-090-180.yaml",
              "configs/deepgaitv2/DeepGaitV2_casiab-nAG-ryu-000-090.yaml"]

gate_log = []


def gate(name, ok, detail=""):
    gate_log.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"  {'✓' if ok else '✗'} {name}: {detail}")
    if not ok:
        sys.exit(f"ゲート {name} が失敗したため停止する（設計§10: 黙って進めない）")


# --------------------------------------------------------------------------- #
# データ読み込み
# --------------------------------------------------------------------------- #
def load_df(incdino_file):
    """設計§8のカラム単位供給元表どおりに組み立てる。"""
    with open(os.path.join(HERE, incdino_file), encoding="utf-8") as f:
        inc = json.load(f)
    with open(os.path.join(HERE, "metrics_dgv2_sustech.json"), encoding="utf-8") as f:
        sus = json.load(f)
    with open(os.path.join(HERE, "metrics_ft_sustech_full_20260726.json"), encoding="utf-8") as f:
        ft = json.load(f)
    prov = inc.get("_provenance")

    rows = []
    for sid, sub in ID2SUBSET.items():
        s, u, f_ = inc[sid], sus[sid], ft[sub]
        row = {"id": sid, "subset": sub,
               "log_people": math.log(u["Number of people"]),
               "n_people": u["Number of people"],
               "MSD_raw": s["MSD_raw"],
               "scratch_all": u["acc_all"], "ft_all": f_["acc_all"],
               "delta_pt": f_["delta_acc_all_pt"]}
        for c in INC5 + DINO5:                      # incdino は修正版のみから
            row[c] = s[c]
        for c in SUS5:                              # sustech は metrics_dgv2_sustech から
            row[c] = u[c]
        for cond in ("nm", "bg", "cl"):
            row[f"scratch_{cond}"] = u[f"acc_{cond}"]
            row[f"ft_{cond}"] = f_[f"acc_{cond}"]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, prov


def run_gates(df, prov, fixed):
    """G1〜G6。fixed=False（A'用の原本X）ではG4を逆向き（旧値のまま）に適用する。"""
    tag = "修正X" if fixed else "原本X"
    print(f"\n--- ゲート ({tag}) ---")
    gate(f"G1 id対応 [{tag}]", len(df) == 13 and df.notna().all().all(),
         f"13/13・欠損ゼロ (rows={len(df)})")

    ok2 = all(abs(r.scratch_all - r.ft_all + r.delta_pt / 100) < 1e-6 for r in df.itertuples())
    with open(os.path.join(HERE, "metrics_ft_sustech_full_20260726.json"), encoding="utf-8") as f:
        ftj = json.load(f)
    ok2a = all(abs(r.scratch_all - ftj[r.subset]["scratch_acc_all"]) < 1e-9 for r in df.itertuples())
    gate(f"G2 acc・Δ整合 [{tag}]", ok2 and ok2a,
         "scratch一致<1e-9 かつ |Δ-(ft-scratch)*100|<1e-6 を13/13")

    snap = os.path.join(HERE, "fid_train_test_results_ryu_snapshot.json")
    ok3 = os.path.exists(snap)
    if ok3:
        with open(snap, encoding="utf-8") as f:
            sj = json.load(f)
        n = 0
        for col, model in (("FID_train_test_Inception", "Inception"),
                           ("FID_train_test_DINO", "DINOv2")):
            for sid, d in ID2DIR.items():
                v = df.loc[df["id"] == sid, col].iloc[0]
                if round(sj[model][RYU_PREFIX + d], 4) != v:
                    ok3 = False
                n += 1
        detail = f"スナップショットと4桁丸めで {n}/26 突合"
    else:
        detail = "スナップショットが無い（fixスクリプト未実行）"
    if fixed:
        ok3 = ok3 and prov is not None and prov["checks"]["fid_train_test"]["matched"] == 26
    gate(f"G3 FID_tt出典突合 [{tag}]", ok3, detail)

    if fixed:
        ok4 = prov is not None and len(prov["fixed_cells"]) == 3
        if ok4:
            for c in prov["fixed_cells"]:
                v = df.loc[df["id"] == c["id"], c["column"]].iloc[0]
                if v != c["new"] or v == c["old"]:
                    ok4 = False
        ok4 = ok4 and all(0.050 < df.loc[df["id"] == i, "Inception_MSD"].iloc[0] < 0.051
                          for i in ("3", "4"))
        ok4 = ok4 and 2168 < df.loc[df["id"] == "11", "MSD_raw"].iloc[0] < 2169
        gate("G4 修正値の適用 [修正X]", ok4, "_provenance記録値と厳密一致・旧値と不一致・範囲チェック")
    else:
        ok4 = all(df.loc[df["id"] == i, c].iloc[0] == v for (i, c), v in OLD_VALUES.items())
        gate("G4' 旧値の保持 [原本X]", ok4, "3セルが旧値のままであること（取り違え防止）")

    part = os.path.join(REPO, "datasets/CASIA-B/CASIA-B.json")
    ok5 = os.path.exists(part)
    if ok5:
        with open(part, encoding="utf-8") as f:
            ok5 = len(json.load(f)["TRAIN_SET"]) == 75
        for cfg in G5_CONFIGS:
            p = os.path.join(REPO, cfg)
            if not os.path.exists(p) or "./datasets/CASIA-B/CASIA-B.json" not in open(p, encoding="utf-8").read():
                ok5 = False
    gate(f"G5 学習側パーティション [{tag}]", ok5, "config3本が同一partitionを指し TRAIN_SET==75")

    gate(f"G6 ゴミ列 [{tag}]", "列1" not in df.columns,
         "「列1」はFEATURE_COLS明示指定で構造的に排除")


# --------------------------------------------------------------------------- #
# モデル
# --------------------------------------------------------------------------- #
def lodo_ridge(df, target_col, feature_cols, y_override=None, keep_pred=False):
    y_all = df[target_col].values if y_override is None else y_override
    X_all = df[feature_cols].values
    y_true, y_pred, coefs = [], [], []
    for i in range(len(df)):
        mask = np.arange(len(df)) != i
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(X_all[mask], y_all[mask])
        y_pred.append(model.predict(X_all[~mask])[0])
        y_true.append(y_all[~mask][0])
        coefs.append(model.named_steps["ridge"].coef_)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    res = {"mae": float(mean_absolute_error(y_true, y_pred)),
           "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
           "spearman": float(rho),
           "coef": {f: float(c) for f, c in zip(feature_cols, np.mean(coefs, axis=0))}}
    if keep_pred:
        res["per_subset"] = {s: {"true": float(t), "pred": float(p)}
                             for s, t, p in zip(df["subset"], y_true, y_pred)}
    return res


def _pairs(df, y_map, tau, feature_cols):
    ids = df["subset"].tolist()
    fmap = {r["subset"]: r[feature_cols].values.astype(float) for _, r in df.iterrows()}
    feats, labels, keys = [], [], []
    for a, b in itertools.permutations(ids, 2):
        feats.append(fmap[a] - fmap[b])
        d = y_map[a] - y_map[b]
        labels.append(TIE if abs(d) <= tau else (A_BETTER if d > 0 else B_BETTER))
        keys.append((a, b))
    return np.array(feats), np.array(labels), keys


def pairwise_logreg(df, y_map, tau, feature_cols, full=True):
    X, y, keys = _pairs(df, y_map, tau, feature_cols)
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
        if full and clf.coef_.shape[0] == 3:
            coefs.append(clf.coef_)
    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
    acc = float(accuracy_score(y_true_all, y_pred_all))
    if not full:
        return acc
    res = {"tau": tau, "n_pairs": int(len(y_true_all)), "accuracy": acc,
           "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro")),
           "label_dist": {int(c): int((y_true_all == c).sum()) for c in np.unique(y_true_all)},
           "n_tie_over_total": f"{int((y_true_all == TIE).sum())}/{len(keys)}"}
    if coefs:
        imp = np.mean(np.abs(np.mean(coefs, axis=0)), axis=0)
        res["importance"] = {f: float(v) for f, v in
                             sorted(zip(feature_cols, imp), key=lambda kv: -kv[1])}
    return res


# --------------------------------------------------------------------------- #
# 偏相関（ランク上の残差化。|Z|=1 → df=n-3、|Z|=2 → df=n-4）
# --------------------------------------------------------------------------- #
def _resid(y, Z):
    Z1 = np.column_stack([np.ones(len(y))] + [z for z in Z])
    beta, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    return y - Z1 @ beta, Z1 @ beta


def partial_spearman(x, y, Z=()):
    rx, ry = rankdata(x), rankdata(y)
    rZ = [rankdata(z) for z in Z]
    if rZ:
        ex, _ = _resid(rx, rZ)
        ey, _ = _resid(ry, rZ)
    else:
        ex, ey = rx - rx.mean(), ry - ry.mean()
    r = float(np.corrcoef(ex, ey)[0, 1])
    dof = len(x) - 2 - len(rZ)
    t_stat = r * math.sqrt(dof / max(1 - r ** 2, 1e-300))
    return r, float(2 * t_dist.sf(abs(t_stat), dof)), dof


def freedman_lane_p(x, y, Z, c_index, n_perm=N_PERM_FL):
    """Freedman-Lane: yをZに回帰した残差を並べ替えて帰無分布を作る。両側。"""
    rng = np.random.RandomState(2000 + c_index)
    r_obs, _, _ = partial_spearman(x, y, Z)
    ry = rankdata(y)
    rZ = [rankdata(z) for z in Z]
    if rZ:
        e, fit = _resid(ry, rZ)
    else:
        e, fit = ry - ry.mean(), np.full(len(ry), ry.mean())
    count = 0
    for _ in range(n_perm):
        r_star, _, _ = partial_spearman(x, fit + rng.permutation(e), Z)
        if abs(r_star) >= abs(r_obs):
            count += 1
    return (count + 1) / (n_perm + 1)


# --------------------------------------------------------------------------- #
def main():
    out = {"design": "Notion 0805進捗 フェーズ3 設計v4.1",
           "seeds": {"ridge_perm": "RandomState(1000+k)", "logreg_perm": "RandomState(1000+k)",
                     "freedman_lane": "RandomState(2000+c)"},
           "gates": None, "seed_table": {}}

    print("=" * 74)
    print("フェーズ3 本分析（incdino空間でft精度・Δを予測できるか）")
    print("=" * 74)

    df, prov = load_df("metrics_incdino_fixed_20260806.json")
    run_gates(df, prov, fixed=True)
    df_unfixed, _ = load_df("metrics_ryu.json")
    run_gates(df_unfixed, None, fixed=False)
    out["gates"] = gate_log
    out["provenance_of_X"] = {"fixed_cells": prov["fixed_cells"],
                              "sources": prov["sources"]} if prov else None

    # ---------- 内部順序1: scratch側再計算 → FID_tt主変数の確定 ----------
    print("\n" + "=" * 74)
    print("§4 FID_tt主変数の選択（修正Xでのscratch側優劣重要度が高い方）")
    scr_map = dict(zip(df["subset"], df["scratch_all"]))
    lr_scratch = pairwise_logreg(df, scr_map, 0.05, FEATURE_SETS["S_incdino11"])
    imp = lr_scratch.get("importance", {})
    inc_imp = imp.get("FID_train_test_Inception", float("nan"))
    dino_imp = imp.get("FID_train_test_DINO", float("nan"))
    primary = "FID_train_test_Inception" if inc_imp >= dino_imp else "FID_train_test_DINO"
    secondary = "FID_train_test_DINO" if primary == "FID_train_test_Inception" else "FID_train_test_Inception"
    print(f"  重要度 Inception={inc_imp:.4f} / DINO={dino_imp:.4f} → 主変数 = {primary}")
    out["fid_tt_primary"] = {"primary": primary, "secondary": secondary,
                             "importance_inception": inc_imp, "importance_dino": dino_imp,
                             "rule": "設計§4: 修正XのS_incdino11×scratch(τ=0.05)優劣重要度が高い方",
                             "scratch_logreg": lr_scratch}

    # ---------- A. LODO Ridge（4セット×3ターゲット） ----------
    print("\n" + "=" * 74)
    print("A. LODO Ridge（StandardScaler + Ridge α=1.0、13-fold）")
    out["grid"] = {}
    for sname in ("S_incdino11", "S_inc6", "S_dino6", "S_9var"):
        out["grid"][sname] = {}
        for tgt in TARGETS:
            r = lodo_ridge(df, tgt, FEATURE_SETS[sname], keep_pred=True)
            out["grid"][sname][tgt] = r
            print(f"  {sname:12s} × {tgt:11s}  ρ={r['spearman']:+.4f}  MAE={r['mae']:.4f}")

    # ---------- A'. 未修正X感度 ----------
    print("\nA'. 未修正X（原本 metrics_ryu.json）での感度")
    out["grid_unfixed"] = {}
    for tgt in TARGETS:
        r = lodo_ridge(df_unfixed, tgt, FEATURE_SETS["S_incdino11"], keep_pred=True)
        out["grid_unfixed"][tgt] = r
        print(f"  S_incdino11 × {tgt:11s}  ρ={r['spearman']:+.4f}  MAE={r['mae']:.4f}")

    # ---------- A''. S_sustech6×ft_all の再現（劣化時診断の入力） ----------
    print("\nA''. S_sustech6 × ft_all の決定的再現（既報JSONと絶対差<1e-9）")
    a2 = lodo_ridge(df, "ft_all", FEATURE_SETS["S_sustech6"], keep_pred=True)
    with open(os.path.join(HERE, "result_ft_analysis_20260726.json"), encoding="utf-8") as f:
        ref = json.load(f)["ridge"]["ft_all"]
    gate("A'' 再現一致", abs(a2["spearman"] - ref["spearman"]) < 1e-9 and abs(a2["mae"] - ref["mae"]) < 1e-9,
         f"ρ差={abs(a2['spearman']-ref['spearman']):.2e} MAE差={abs(a2['mae']-ref['mae']):.2e}")
    out["A_sustech6_ft_reproduced"] = a2

    # ---------- B. 優劣判定（全グリッド） ----------
    print("\n" + "=" * 74)
    print("B. 優劣判定 pairwise logistic（leave-one-pair-out）")
    out["logreg_grid"] = {}
    logreg_cells = [("scratch_all", 0.05, "scratch_tau0.05"),
                    ("ft_all", 0.05, "ft_tau0.05"),
                    ("delta_pt", 5.0, "delta_tau5pt"),
                    ("delta_pt", NOISE_PT, "delta_tau1.18pt")]
    for sname in ("S_incdino11", "S_inc6", "S_dino6", "S_9var"):
        out["logreg_grid"][sname] = {}
        for tgt, tau, cname in logreg_cells:
            ymap = dict(zip(df["subset"], df[tgt]))
            r = pairwise_logreg(df, ymap, tau, FEATURE_SETS[sname])
            out["logreg_grid"][sname][cname] = r
            print(f"  {sname:12s} × {cname:16s} Acc={r['accuracy']:.4f} F1={r['macro_f1']:.4f} tie={r['n_tie_over_total']}")

    # ---------- C. 核心の問い ----------
    print("\n" + "=" * 74)
    print("C. 核心の問い（偏Spearman、Freedman-Lane併記）")
    dlt = df["delta_pt"].values
    scr = df["scratch_all"].values
    prim = df[primary].values
    seco = df[secondary].values
    sus = df["FID_train_test_DeepGaitV2"].values
    C_SPECS = [
        (0, "素(Δ, 主)", dlt, prim, ()),
        (1, "偏(Δ, 主 | scratch)", dlt, prim, (scr,)),
        (2, "偏(Δ, scratch | 主)", dlt, scr, (prim,)),
        (3, "素(Δ, 副)", dlt, seco, ()),
        (4, "偏(Δ, 副 | scratch)", dlt, seco, (scr,)),
        (5, "偏(Δ, scratch | 副)", dlt, scr, (seco,)),
        (6, "偏(Δ, sus | 主)", dlt, sus, (prim,)),
        (7, "偏(Δ, 主 | sus)", dlt, prim, (sus,)),
        (8, "偏(Δ, sus | 主, scratch) ★C2'", dlt, sus, (prim, scr)),
        (9, "偏(Δ, 主 | sus, scratch) ★C2'", dlt, prim, (sus, scr)),
    ]
    out["C"] = {}
    for c, label, x, y, Z in C_SPECS:
        r, p_t, dof = partial_spearman(x, y, Z)
        p_fl = freedman_lane_p(x, y, Z, c)
        out["C"][f"c{c}"] = {"label": label, "rho": r, "p_t": p_t, "df": dof,
                             "p_freedman_lane": p_fl, "seed": 2000 + c}
        out["seed_table"][f"c{c}"] = 2000 + c
        print(f"  c{c} {label:30s} ρ={r:+.4f} p_t={p_t:.4f}(df={dof}) p_FL={p_fl:.4f}")

    cols = [primary, secondary, "FID_train_test_DeepGaitV2", "scratch_all", "delta_pt"]
    out["C_corr_matrix"] = {a: {b: float(spearmanr(df[a], df[b])[0]) for b in cols} for a in cols}

    print("\n  LODOアブレーション（target=Δpt、全変数セット=S_incdino11）")
    out["C_ablation_delta"] = {}
    for name, cc in [("scratchのみ", ["scratch_all"]),
                     ("scratch+FID_tt主", ["scratch_all", primary]),
                     ("FID_tt主のみ", [primary]),
                     ("11変数(scratchなし)", FEATURE_SETS["S_incdino11"]),
                     ("scratch+11変数", ["scratch_all"] + FEATURE_SETS["S_incdino11"])]:
        r = lodo_ridge(df, "delta_pt", cc)
        out["C_ablation_delta"][name] = {k: r[k] for k in ("mae", "rmse", "spearman")}
        print(f"    {name:22s} MAE={r['mae']:6.3f}pt ρ={r['spearman']:+.3f}")

    # ---------- D. 条件別 ----------
    print("\n" + "=" * 74)
    print("D. 条件別（NM/BG/CL）LODO Ridge・S_incdino11")
    out["D_per_condition"] = {}
    for side in ("ft", "scratch"):
        out["D_per_condition"][side] = {}
        for cond in ("nm", "bg", "cl"):
            r = lodo_ridge(df, f"{side}_{cond}", FEATURE_SETS["S_incdino11"], keep_pred=True)
            out["D_per_condition"][side][cond] = r
            print(f"  {side:7s} {cond}  ρ={r['spearman']:+.4f} MAE={r['mae']:.4f}")

    # ---------- F. exploratory ----------
    print("\nF. exploratory: S_sustech6 + FID_tt主（7変数・参考値）")
    out["F_exploratory"] = {}
    for tgt in ("ft_all", "delta_pt"):
        r = lodo_ridge(df, tgt, FEATURE_SETS["S_sustech6"] + [primary])
        out["F_exploratory"][tgt] = r
        print(f"  {tgt:11s} ρ={r['spearman']:+.4f} MAE={r['mae']:.4f}  ※本文の主張には使わない")

    # ---------- 並べ替え検定（事前列挙8本） ----------
    print("\n" + "=" * 74)
    print(f"並べ替え検定（Ridge N={N_PERM_RIDGE} ×6 / LogReg N={N_PERM_LOGREG} ×2）")
    out["permutation"] = {}
    k = 0
    for sname in ("S_incdino11", "S_inc6", "S_dino6"):
        for tgt in ("ft_all", "delta_pt"):
            rng = np.random.RandomState(1000 + k)
            obs = out["grid"][sname][tgt]["spearman"]
            y = df[tgt].values.copy()
            cnt = sum(lodo_ridge(df, tgt, FEATURE_SETS[sname],
                                 y_override=rng.permutation(y))["spearman"] >= obs
                      for _ in range(N_PERM_RIDGE))
            p = (cnt + 1) / (N_PERM_RIDGE + 1)
            key = f"k{k}_{sname}_{tgt}"
            out["permutation"][key] = {"observed_spearman": obs, "p_perm": p, "seed": 1000 + k,
                                       "main_endpoint": k == 0}
            out["seed_table"][f"k{k}"] = 1000 + k
            print(f"  k={k} {sname:12s}×{tgt:9s} ρ={obs:+.4f} p={p:.4f}{'  ★主判定' if k == 0 else ''}")
            k += 1
    for tgt, tau, cname in [("ft_all", 0.05, "ft_tau0.05"), ("delta_pt", 5.0, "delta_tau5pt")]:
        rng = np.random.RandomState(1000 + k)
        obs = out["logreg_grid"]["S_incdino11"][cname]["accuracy"]
        vals = df[tgt].values.copy()
        cnt = 0
        for _ in range(N_PERM_LOGREG):
            pm = dict(zip(df["subset"], rng.permutation(vals)))
            if pairwise_logreg(df, pm, tau, FEATURE_SETS["S_incdino11"], full=False) >= obs:
                cnt += 1
        p = (cnt + 1) / (N_PERM_LOGREG + 1)
        out["permutation"][f"k{k}_S_incdino11_{cname}"] = {
            "observed_accuracy": obs, "p_perm": p, "seed": 1000 + k}
        out["seed_table"][f"k{k}"] = 1000 + k
        print(f"  k={k} S_incdino11 ×{cname:14s} Acc={obs:.4f} p={p:.4f}")
        k += 1

    # ---------- E. 2×2完成グリッド表 ----------
    print("\n" + "=" * 74)
    print("E. 2×2完成グリッド")
    ref_files = {"ridge_sustech": "result_ridge_sustech.json",
                 "logreg_sustech": "result_logreg_sustech.json",
                 "ft_analysis": "result_ft_analysis_20260726.json"}
    refs = {k2: json.load(open(os.path.join(HERE, v), encoding="utf-8"))
            for k2, v in ref_files.items()}
    E = {}
    for sname in ("S_incdino11", "S_inc6", "S_dino6", "S_9var"):
        E[sname] = {t: {"rho": out["grid"][sname][t]["spearman"],
                        "mae": out["grid"][sname][t]["mae"],
                        "logreg": {c: {"acc": out["logreg_grid"][sname][c]["accuracy"],
                                       "f1": out["logreg_grid"][sname][c]["macro_f1"],
                                       "label_dist": out["logreg_grid"][sname][c]["label_dist"],
                                       "tie": out["logreg_grid"][sname][c]["n_tie_over_total"]}
                                   for c in (cn for _, _, cn in logreg_cells)}}
                     for t in TARGETS}
    E["S_sustech6"] = {"_source": ref_files, "raw_refs": refs}
    E["_footnote"] = ("優劣Acc/F1の比較は同一ターゲット内の行間（特徴セット間）のみ有効。"
                      "scratch列とft列の比較はレンジ圧縮でtie率が異なるため行わない（フェーズ2知見）。")
    out["E_grid"] = E
    for sname in ("S_incdino11", "S_inc6", "S_dino6", "S_9var"):
        row = " ".join(f"{t}:ρ={E[sname][t]['rho']:+.3f}/MAE={E[sname][t]['mae']:.3f}" for t in TARGETS)
        print(f"  {sname:12s} {row}")

    # ---------- §7 判定 ----------
    print("\n" + "=" * 74)
    print("§7 判定")
    p_k0 = out["permutation"]["k0_S_incdino11_ft_all"]["p_perm"]
    p_k2 = out["permutation"]["k2_S_inc6_ft_all"]["p_perm"]
    rho_inc6 = out["grid"]["S_inc6"]["ft_all"]["spearman"]
    rho_sus6 = a2["spearman"]
    rho_diff = rho_inc6 - rho_sus6            # 汎用が上回ると正
    D = rho_sus6 - rho_inc6                   # 劣化時診断の量
    delta_mae_inc6 = out["grid"]["S_inc6"]["delta_pt"]["mae"]

    coef11 = out["grid"]["S_incdino11"]["ft_all"]["coef"]
    coef11d = out["grid"]["S_incdino11"]["delta_pt"]["coef"]
    coef6 = out["grid"]["S_inc6"]["ft_all"]["coef"]
    judge_var = "FID_train_test_Inception"    # 設計§7※: 主変数がDINOでも係数構造はInception版で判定
    rank6 = sorted(coef6, key=lambda c: -abs(coef6[c])).index(judge_var) + 1
    coef_ok = coef11[judge_var] < 0 and coef11d[judge_var] > 0 and rank6 <= 2

    if p_k0 < 0.05 and rho_diff >= 0.10:
        row = 1
    elif p_k0 < 0.05 and abs(rho_diff) < 0.10 and coef_ok:
        row = 2
    elif p_k0 < 0.05 and rho_diff <= -0.10 and coef_ok:
        row = 3
    elif p_k0 < 0.05:
        row = 4
    elif p_k2 >= 0.05:
        row = 5
    else:
        row = 6
    out["verdict"] = {
        "row": row, "p_k0": p_k0, "p_k2": p_k2,
        "rho_inc6_ft": rho_inc6, "rho_sustech6_ft": rho_sus6,
        "rho_diff_controlled": rho_diff, "D_for_diagnosis": D,
        "coef_structure_match": bool(coef_ok),
        "coef_detail": {"ft_coef": coef11[judge_var], "delta_coef": coef11d[judge_var],
                        "rank_in_S_inc6": rank6, "judged_on": judge_var},
        "delta_mae_S_inc6_pt": delta_mae_inc6,
        "practical_criterion_met": bool(delta_mae_inc6 <= PRACTICAL_MAE_PT),
        "degradation_diagnosis_required": row in (3, 5),
    }
    print(f"  主判定 p(k=0)={p_k0:.4f} / 統制比較 ρ_inc6={rho_inc6:+.4f} vs ρ_sustech6={rho_sus6:+.4f}"
          f" (差={rho_diff:+.4f}, D={D:+.4f})")
    print(f"  係数構造一致={coef_ok} / ΔMAE(S_inc6)={delta_mae_inc6:.3f}pt "
          f"({'実務基準達成' if delta_mae_inc6 <= PRACTICAL_MAE_PT else '★実務基準未達'})")
    print(f"  → 解釈マトリクス 行{row}" + ("（劣化時診断が必須）" if row in (3, 5) else ""))

    out_path = os.path.join(HERE, "result_incdino_ft_20260806.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
