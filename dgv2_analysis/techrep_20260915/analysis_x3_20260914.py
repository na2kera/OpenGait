#!/usr/bin/env python3
"""X3 本分析（2026-09-14）: 汎用側 MSD を式(1)版に差し替えた指標 metrics_incdino_kera_X3.json で、原稿が使う全数値を再計算する。

出力は X2 本体（analysis_incdino_ft_20260806.py）の結果 JSON と同じキー構造にし、make_tables_20260915.py / make_figure_*.py が
ARM=X3 でそのまま読めるようにする。関数（lodo_ridge / pairwise_logreg / FEATURE_SETS / load_df）は X2 本体から import して同一性を担保する。
  - result_incdino_ft_X3.json : arm, inputs, E_grid, grid, logreg_grid, permutation(k0..k7), D_per_condition, A_sustech6_ft_reproduced（X2 から複製）,
                                verdict（coef_detail など make_tables が読む最小限）, C_corr_matrix, C_ablation_delta, extras（000-180 除外 ρ、目的変数 mean3 感度、指標相関、MSD 幅）
  - perm_scratch_X3.json      : scratch_all × 4 セットの並べ替え p（perm_scratch_20260910.json と同形式。seed 1010..1013）
ゲート: S_sustech6 の ρ/MAE が X2 と 1e-12 で一致（歩容側は無変更のはず）。X3 指標の _provenance.arm == "X3"、差し替え 26 セルが全て記録されていること。

実行: docker run --rm -v /home/kera/OpenGait:/app/OpenGait -v /home/kera/worktrees:/home/kera/worktrees -w /app/OpenGait opengait:latest \
        python dgv2_analysis/techrep_20260915/analysis_x3_20260914.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

WT = "/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis"
sys.path.insert(0, WT)
import analysis_incdino_ft_20260806 as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X3_METRICS = os.path.join(HERE, "metrics_incdino_kera_X3.json")
X2_RESULT = os.path.join(WT, A.ARM_DEFAULTS["X2"]["out"])
OUT = os.path.join(HERE, "result_incdino_ft_X3.json")
OUT_PERM = os.path.join(HERE, "perm_scratch_X3.json")
SETS = ("S_incdino11", "S_inc6", "S_dino6", "S_9var", "S_sustech6")
TARGETS = ("scratch_all", "ft_all", "delta_pt")
LR_NAME = {"scratch_all": "scratch_tau0.05", "ft_all": "ft_tau0.05", "delta_pt": "delta_tau5pt"}
LR_TAU = {"scratch_all": 0.05, "ft_all": 0.05, "delta_pt": 5.0}


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def sp(a, b):
    return float(spearmanr(a, b)[0])


def main():
    t0 = time.time()
    x3m = json.load(open(X3_METRICS))
    prov = x3m["_provenance"]
    assert prov["arm"] == "X3" and len(prov["replaced_cells"]) == 26, "X3 指標の来歴が不正"
    x2 = json.load(open(X2_RESULT))
    df, _, keys = A.load_df(X3_METRICS)
    assert len(df) == 13 and df.notna().all().all()
    out = {"design": "X3 = X2 の Inception_MSD / DINO_MSD を式(1)（人物ごと）で再計測した値に差し替え（2026-09-14）",
           "arm": "X3", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "inputs": {"incdino": os.path.relpath(X3_METRICS, HERE), "sustech": "metrics_dgv2_sustech.json", "ft": "metrics_ft_sustech_full_20260726.json",
                      "x2_result": X2_RESULT, "x3_metrics_provenance": {"base_X2_md5": prov["base_X2"]["md5"], "s5_template_md5": prov.get("s5_template_md5")}},
           "inputs_md5": {"x3_metrics": md5(X3_METRICS), "x2_result": md5(X2_RESULT), "analysis_code": md5(A.__file__)},
           "seeds": {"ridge_perm": "RandomState(1000+k)", "logreg_perm": "RandomState(1000+k)", "scratch_perm": "RandomState(1010+i)"},
           "E_grid": {}, "grid": {}, "logreg_grid": {}, "permutation": {}, "D_per_condition": {"ft": {}, "delta_pt": {}, "scratch": {}}}

    # ---- grid / E_grid / logreg（S_sustech6 も同じ関数で再計算し X2 と一致を確認）
    for fs in SETS:
        out["grid"][fs] = {}; out["E_grid"][fs] = {}; out["logreg_grid"][fs] = {}
        for tgt in TARGETS:
            r = A.lodo_ridge(df, tgt, A.FEATURE_SETS[fs], keep_pred=True)
            out["grid"][fs][tgt] = r
            ymap = dict(zip(df["subset"], df[tgt]))
            lr = A.pairwise_logreg(df, ymap, LR_TAU[tgt], A.FEATURE_SETS[fs])
            out["logreg_grid"][fs][LR_NAME[tgt]] = lr
            out["E_grid"][fs][tgt] = {"rho": r["spearman"], "mae": r["mae"],
                                      "logreg": {LR_NAME[tgt]: {"acc": lr["accuracy"], "f1": lr["macro_f1"], "label_dist": lr["label_dist"], "tie": lr["n_tie_over_total"]}}}
            print(f"  {fs:12s} {tgt:11s} rho={r['spearman']:+.4f} mae={r['mae']:.4f} acc={lr['accuracy']:.4f} f1={lr['macro_f1']:.4f}", flush=True)
    for tgt in TARGETS:  # 歩容側は無変更 → X2 と一致（ゲート）
        a, b = out["E_grid"]["S_sustech6"][tgt], x2["E_grid"]["S_sustech6"][tgt]
        assert abs(a["rho"] - b["rho"]) < 1e-12 and abs(a["mae"] - b["mae"]) < 1e-12, f"S_sustech6 {tgt} が X2 と不一致"
    out["A_sustech6_ft_reproduced"] = x2["A_sustech6_ft_reproduced"]

    # ---- 条件別（S_incdino11、X2 D と同型）
    for cond in A.CONDS:
        for side, col in (("ft", f"ft_{cond}"), ("delta_pt", f"delta_{cond}_pt"), ("scratch", f"scratch_{cond}")):
            out["D_per_condition"][side][cond] = A.lodo_ridge(df, col, A.FEATURE_SETS["S_incdino11"], keep_pred=True)

    # ---- 並べ替え検定: X2 と同じ 8 本（k0..k7）＋ scratch 4 本（seed 1010..1013）
    k = 0
    for sname in ("S_incdino11", "S_inc6", "S_dino6"):
        for tgt in ("ft_all", "delta_pt"):
            rng = np.random.RandomState(1000 + k); obs = out["grid"][sname][tgt]["spearman"]; y = df[tgt].values.copy()
            cnt = sum(A.lodo_ridge(df, tgt, A.FEATURE_SETS[sname], y_override=rng.permutation(y))["spearman"] >= obs for _ in range(A.N_PERM_RIDGE))
            out["permutation"][f"k{k}_{sname}_{tgt}"] = {"observed_spearman": obs, "p_perm": (cnt + 1) / (A.N_PERM_RIDGE + 1), "seed": 1000 + k, "main_endpoint": k == 0, "count_ge": int(cnt)}
            print(f"  perm k{k} {sname}×{tgt} rho={obs:.3f} p={(cnt+1)/(A.N_PERM_RIDGE+1):.4f}", flush=True); k += 1
    for tgt, tau, cname in [("ft_all", 0.05, "ft_tau0.05"), ("delta_pt", 5.0, "delta_tau5pt")]:
        rng = np.random.RandomState(1000 + k); obs = out["logreg_grid"]["S_incdino11"][cname]["accuracy"]; vals = df[tgt].values.copy(); cnt = 0
        for _ in range(A.N_PERM_LOGREG):
            pm = dict(zip(df["subset"], rng.permutation(vals)))
            if A.pairwise_logreg(df, pm, tau, A.FEATURE_SETS["S_incdino11"], full=False) >= obs:
                cnt += 1
        out["permutation"][f"k{k}_S_incdino11_{cname}"] = {"observed_accuracy": obs, "p_perm": (cnt + 1) / (A.N_PERM_LOGREG + 1), "seed": 1000 + k, "count_ge": int(cnt)}
        print(f"  perm k{k} S_incdino11×{cname} acc={obs:.4f} p={(cnt+1)/(A.N_PERM_LOGREG+1):.4f}", flush=True); k += 1
    perm_scratch = {"note": "X3。X2 本体と同一の LODO Ridge、N_PERM=1000、片側 (perm ρ >= 観測 ρ)。seed 1010..1013", "n_perm": A.N_PERM_RIDGE, "arm": "X3", "target": "scratch_all", "results": {}}
    for i, sname in enumerate(("S_incdino11", "S_inc6", "S_dino6", "S_sustech6")):
        rng = np.random.RandomState(1010 + i); obs = out["grid"][sname]["scratch_all"]["spearman"]; y = df["scratch_all"].values.copy()
        cnt = sum(A.lodo_ridge(df, "scratch_all", A.FEATURE_SETS[sname], y_override=rng.permutation(y))["spearman"] >= obs for _ in range(A.N_PERM_RIDGE))
        perm_scratch["results"][sname] = {"observed_spearman": float(obs), "p_perm": (cnt + 1) / (A.N_PERM_RIDGE + 1), "seed": 1010 + i, "count_ge": int(cnt)}
        print(f"  perm scratch {sname} rho={obs:.3f} p={(cnt+1)/(A.N_PERM_RIDGE+1):.4f}", flush=True)

    # ---- verdict（make_tables が読む最小限）
    coef_inc6_ft = out["grid"]["S_inc6"]["ft_all"]["coef"]
    rank = sorted(coef_inc6_ft, key=lambda c: -abs(coef_inc6_ft[c])).index("FID_train_test_Inception") + 1
    out["verdict"] = {"row": None, "p_k0": out["permutation"]["k0_S_incdino11_ft_all"]["p_perm"], "p_k2": out["permutation"]["k2_S_inc6_ft_all"]["p_perm"],
                      "rho_inc6_ft": out["grid"]["S_inc6"]["ft_all"]["spearman"], "rho_sustech6_ft": out["grid"]["S_sustech6"]["ft_all"]["spearman"],
                      "rho_incdino11_ft": out["grid"]["S_incdino11"]["ft_all"]["spearman"],
                      "rho_diff_controlled": out["grid"]["S_inc6"]["ft_all"]["spearman"] - out["grid"]["S_sustech6"]["ft_all"]["spearman"],
                      "coef_structure_match": coef_inc6_ft["FID_train_test_Inception"] < 0,
                      "delta_mae_S_inc6_pt": out["grid"]["S_inc6"]["delta_pt"]["mae"], "practical_criterion_met": None,
                      "coef_detail": {"ft_coef": coef_inc6_ft["FID_train_test_Inception"], "delta_coef": out["grid"]["S_inc6"]["delta_pt"]["coef"]["FID_train_test_Inception"],
                                      "rank_in_S_inc6": rank, "judged_on": "FID_train_test_Inception"},
                      "note": "X3 は解釈マトリクス判定を行わない（row=None）。make_tables が参照するキーのみ"}
    # ---- C 系（相関行列・Δ アブレーション）: X2 と同型
    primary = "FID_train_test_Inception"
    cols = [primary, "FID_train_test_DINO", "FID_train_test_DeepGaitV2", "scratch_all", "delta_pt"]
    out["C_corr_matrix"] = {a: {b: sp(df[a], df[b]) for b in cols} for a in cols}
    out["C_ablation_delta"] = {}
    for name, cc in [("scratchのみ", ["scratch_all"]), ("scratch+FID_tt主", ["scratch_all", primary]), ("FID_tt主のみ", [primary]),
                     ("11変数(scratchなし)", A.FEATURE_SETS["S_incdino11"]), ("scratch+11変数", ["scratch_all"] + A.FEATURE_SETS["S_incdino11"])]:
        r = A.lodo_ridge(df, "delta_pt", cc); out["C_ablation_delta"][name] = {k_: r[k_] for k_ in ("mae", "rmse", "spearman")}

    # ---- extras: 000-180 除外 ρ / 目的変数 mean3 感度 / 指標相関 / MSD 幅
    ex = {"wo_000-180": {}, "mean3": {}, "corr": {}, "msd_width": {}}
    for fs in ("S_incdino11", "S_inc6", "S_dino6", "S_sustech6"):
        for tgt in ("scratch_all", "ft_all"):
            ps = out["grid"][fs][tgt]["per_subset"]; subs = list(ps)
            t = np.array([ps[s]["true"] for s in subs]) * 100; p = np.array([ps[s]["pred"] for s in subs]) * 100; keep = np.array([s != "000-180" for s in subs])
            err = np.abs(t - p)
            ex["wo_000-180"][f"{fs}/{tgt}"] = {"rho_13": sp(t, p), "mae_13": float(err.mean()), "rho_12": sp(t[keep], p[keep]), "mae_12": float(err[keep].mean()),
                                               "err_share_000-180": float(err[~keep][0] / err.sum()), "pred_000-180_pct": float(p[~keep][0]), "true_000-180_pct": float(t[~keep][0])}
    df3 = df.copy()
    for side in ("scratch", "ft"):
        df3[f"{side}_mean3"] = (df3[f"{side}_nm"] + df3[f"{side}_bg"] + df3[f"{side}_cl"]) / 3
    for fs in ("S_incdino11", "S_inc6", "S_dino6", "S_sustech6"):
        for side in ("scratch", "ft"):
            r_all = out["grid"][fs][f"{side}_all"]; r_m3 = A.lodo_ridge(df3, f"{side}_mean3", A.FEATURE_SETS[fs])
            ex["mean3"][f"{fs}/{side}"] = {"rho_all": r_all["spearman"], "rho_mean3": r_m3["spearman"], "mae_all": r_all["mae"], "mae_mean3": r_m3["mae"]}
    for col in ("Inception_MSD", "DINO_MSD", "DeepGaitV2_MSD", "FID_train_test_Inception", "FID_train_test_DINO", "FID_train_test_DeepGaitV2", "Inception_1NN", "Inception_kNN"):
        x = df[col].values
        ex["corr"][col] = {t: sp(x, df[t]) for t in ("scratch_all", "ft_all", "scratch_nm", "ft_nm", "scratch_cl", "ft_cl")}
        ex["msd_width"][col] = float((x.max() - x.min()) / x.mean())
    ex["scratch_vs_ft_rho"] = sp(df["scratch_all"], df["ft_all"])
    ex["majority_baseline"] = {LR_NAME[t]: max(out["logreg_grid"]["S_incdino11"][LR_NAME[t]]["label_dist"].values()) / 156 for t in TARGETS}
    ex["msd_class_values"] = {name: {c["column"]: c["new_class"] for c in prov["replaced_cells"] if c["subset"] == name} for name in sorted({c["subset"] for c in prov["replaced_cells"]})}
    ex["msd_class_by_cond"] = {c["subset"] + "/" + c["column"]: c["class_by_cond"] for c in prov["replaced_cells"]}
    out["extras"] = ex
    out["wall_s"] = time.time() - t0
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    perm_scratch["inputs_md5"] = {"sustech": {"path": os.path.join(WT, "metrics_dgv2_sustech.json"), "md5": md5(os.path.join(WT, "metrics_dgv2_sustech.json"))},
                                  "ft": {"path": os.path.join(WT, "metrics_ft_sustech_full_20260726.json"), "md5": md5(os.path.join(WT, "metrics_ft_sustech_full_20260726.json"))},
                                  "incdino": {"path": X3_METRICS, "md5": md5(X3_METRICS)}, "analysis_code": {"path": A.__file__, "md5": md5(A.__file__)},
                                  "x2_result": {"path": OUT, "md5": md5(OUT)}}
    json.dump(perm_scratch, open(OUT_PERM, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {OUT} and {OUT_PERM} ({out['wall_s']:.0f}s)")
    print("FID_tt rank in S_inc6 ft coef:", rank)


if __name__ == "__main__":
    main()
