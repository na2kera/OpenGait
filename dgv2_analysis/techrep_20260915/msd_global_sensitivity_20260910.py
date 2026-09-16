#!/usr/bin/env python3
"""感度分析（2026-09-10）: DeepGaitV2 の MSD を汎用側（中村実装 MSDall.py）と同じ集計＝「人物を区別しない全画像
（ここでは全系列）の特徴平均からの二乗偏差を要素平均」に置き換えたとき、S_sustech6 の LODO 予測と CL との相関がどう動くか。

- 採用版の MSD（式(1)）: 人物ごとに平均を引き、次元方向に和、人物間で平均（compute_dgv2_metrics_sustech.py compute_msd）
- 全体版の MSD: np.mean(np.square(feats - feats.mean(0)))（incdino_recalc/scripts_kera/s4_msdall_*.py compute_feature_msd と同形）
主結果は変えない。原稿 6.2 節の留保を「集計差では説明できるか」で書き分けるための材料。

実行: docker run --rm -v /home/kera/OpenGait:/app/OpenGait -v /home/kera/worktrees:/home/kera/worktrees \
        -w /app/OpenGait opengait:latest python dgv2_analysis/techrep_20260915/msd_global_sensitivity_20260910.py
"""
import hashlib
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

WT = "/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis"
sys.path.insert(0, WT)
import analysis_incdino_ft_20260806 as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FEAT_DIR = os.path.join(REPO, "dgv2_features_sustech")
SUS_JSON = os.path.join(REPO, "dgv2_analysis", "metrics_dgv2_sustech.json")


def msd_class(emb, labels):           # 採用版（式(1)）: 検算用
    tot = 0.0
    for s_ in np.unique(labels):
        x = emb[labels == s_]
        tot += np.mean(np.sum((x - x.mean(0)) ** 2, axis=1))
    return float(tot / len(np.unique(labels)))


def msd_global(emb):                  # 汎用側と同じ集計
    return float(np.mean(np.square(emb - emb.mean(0))))


df, prov, keys = A.load_df(os.path.join(WT, A.ARM_DEFAULTS["X2"]["incdino"]))
sus = json.load(open(SUS_JSON))
g = {}
for sid, row in sus.items():
    z = np.load(os.path.join(FEAT_DIR, row["subset_name"], "train.npz"), allow_pickle=True)
    emb = z["embeddings"].astype(np.float64).reshape(z["embeddings"].shape[0], -1)
    c = msd_class(emb, z["labels"])
    assert abs(c - row["DeepGaitV2_MSD"]) < 1e-6 * max(1, abs(c)), f"{row['subset_name']}: 式(1) の再計算 {c} が採用値 {row['DeepGaitV2_MSD']} と不一致"
    g[row["subset_name"]] = {"msd_class": c, "msd_global": msd_global(emb), "n_seq": int(emb.shape[0])}
print("式(1) の再計算は 13/13 で採用値と一致")

df2 = df.copy()
df2["DeepGaitV2_MSD"] = [g[s_]["msd_global"] for s_ in df2["subset"]]
feats = A.FEATURE_SETS["S_sustech6"]
out = {"note": __doc__, "per_subset": g, "ridge": {}, "corr": {}, "width": {}}
for tag, d in (("class(採用)", df), ("global(感度)", df2)):
    out["ridge"][tag] = {}
    for tgt in ("scratch_all", "ft_all"):
        r = A.lodo_ridge(d, tgt, feats)
        out["ridge"][tag][tgt] = {"rho": r["spearman"], "mae_pt": r["mae"] * 100, "coef_MSD_pt": r["coef"]["DeepGaitV2_MSD"] * 100}
    for cond in A.CONDS:
        for side, col in (("scratch", f"scratch_{cond}"), ("ft", f"ft_{cond}")):
            r = A.lodo_ridge(d, col, feats)
            out["ridge"][tag][f"{side}_{cond}"] = {"rho": r["spearman"], "mae_pt": r["mae"] * 100}
    x = d["DeepGaitV2_MSD"].values
    out["corr"][tag] = {c: float(spearmanr(x, d[c])[0]) for c in ("scratch_cl", "ft_cl", "scratch_nm", "ft_nm", "scratch_all", "ft_all")}
    out["width"][tag] = float((x.max() - x.min()) / x.mean())
    out["ridge"][tag]["_corr_with_class_msd"] = float(spearmanr(x, df["DeepGaitV2_MSD"])[0])

out["inputs_md5"] = {"sustech_json": hashlib.md5(open(SUS_JSON, "rb").read()).hexdigest(),
                     "analysis_code": hashlib.md5(open(A.__file__, "rb").read()).hexdigest()}
json.dump(out, open(os.path.join(HERE, "msd_global_sensitivity_20260910.json"), "w"), ensure_ascii=False, indent=1)

md = ["# 感度分析: DeepGaitV2 の MSD を汎用側と同じ「全体分散」に置き換えた場合（S_sustech6, LODO Ridge）", "",
      "採用版 = 式(1)（人物ごと・系列単位）。全体版 = 人物を区別しない全系列の特徴平均からの二乗偏差の要素平均（中村実装 MSDall.py と同形）。他 5 変数は同じ。", "",
      "| 指標 | 採用版 (class) | 全体版 (global) |", "|---|---:|---:|"]
for k in ("scratch_all", "ft_all", "scratch_nm", "scratch_bg", "scratch_cl", "ft_nm", "ft_bg", "ft_cl"):
    a, b = out["ridge"]["class(採用)"][k], out["ridge"]["global(感度)"][k]
    md.append(f"| {k} ρ / MAE | {a['rho']:.3f} / {a['mae_pt']:.2f} | {b['rho']:.3f} / {b['mae_pt']:.2f} |")
md += ["", "MSD 生値と精度の Spearman:", "", "| 対象 | 採用版 | 全体版 |", "|---|---:|---:|"]
for c in ("scratch_cl", "ft_cl", "scratch_nm", "ft_nm"):
    md.append(f"| {c} | {out['corr']['class(採用)'][c]:+.2f} | {out['corr']['global(感度)'][c]:+.2f} |")
md += ["", f"MSD の相対幅 (max−min)/mean: 採用版 {out['width']['class(採用)']:.1%} / 全体版 {out['width']['global(感度)']:.1%}。",
       f"採用版と全体版の MSD の順位相関: {out['ridge']['global(感度)']['_corr_with_class_msd']:+.3f}。", "",
       "| subset | n_seq | MSD class | MSD global |", "|---|---:|---:|---:|"]
for s_, v in g.items():
    md.append(f"| {s_} | {v['n_seq']} | {v['msd_class']:.2f} | {v['msd_global']:.5f} |")
open(os.path.join(HERE, "msd_global_sensitivity_20260910.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
