#!/usr/bin/env python3
"""予稿(中間発表)用の図と表数値を生成する (2026-07-27)。

- LODO(13-fold hold-one-out) Ridge の予測値を scratch / ft / delta について算出
  (フェーズ1の既存結果と同一レシピ: StandardScaler + Ridge(alpha=1.0))
- 図1: 実測精度 vs LODO予測精度 の散布図 (フェーズ1=scratch, フェーズ2=ft)
- 表1用の数値 (rho, MAE, 並べ替え検定p, FID_tt標準化係数[pt/1SD]) を JSON に保存

実行: docker exec opengait_container python /app/OpenGait/dgv2_analysis/make_figures_20260727.py
出力: fig_pred_vs_true_20260727.{pdf,png}, prep_table1_20260727.json
"""
import json
import math
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ID2SUBSET = {1: "default", 2: "nm2", 3: "bg2", 4: "cl2", 5: "nm1-bg1", 6: "nm1-cl1",
             7: "bg1-cl1", 8: "000-180", 9: "000-090", 10: "090-180",
             11: "nm1-bg1-cl1", 12: "nm2-bg2-cl2", 13: "nm6"}
FEATS = ["log_people", "DeepGaitV2_MSD", "DeepGaitV2_1NN", "DeepGaitV2_kNN",
         "DeepGaitV2_FID", "FID_train_test_DeepGaitV2"]
N_PERM = 1000
SEED = 0

# dataviz スキルの検証済みパレット (categorical slot 1 / 2, light mode)
C_PHASE1 = "#2a78d6"
C_PHASE2 = "#eb6834"
C_GRID = "#c9c8c3"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"


def load():
    m = json.load(open(os.path.join(HERE, "metrics_dgv2_sustech.json")))
    ft = json.load(open(os.path.join(HERE, "metrics_ft_sustech_full_20260726.json")))
    names, X, ys, yf = [], [], [], []
    for i in range(1, 14):
        s = ID2SUBSET[i]
        d = dict(m[str(i)])
        d["log_people"] = math.log(d["Number of people"])
        names.append(s)
        X.append([d[k] for k in FEATS])
        ys.append(d["acc_all"] * 100)
        yf.append(ft[s]["acc_all"] * 100)
    return names, np.array(X), np.array(ys), np.array(yf)


def lodo(X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        sc = StandardScaler().fit(X[tr])
        p[i] = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[i:i + 1]))[0]
    return p


def fold_mean_coef(X, y):
    """既存結果JSONと同じ定義: 13フォールドの標準化係数の平均 (full-fit ではない)。"""
    cs = []
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        sc = StandardScaler().fit(X[tr])
        cs.append(Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr]).coef_)
    return dict(zip(FEATS, np.mean(cs, axis=0)))


def perm_p(X, y, observed, n=N_PERM, seed=SEED):
    """サブセットラベルを並べ替えて LODO の Spearman rho の帰無分布を作る。"""
    rng = np.random.RandomState(seed)
    cnt = 1  # observed 自身を含める (p の下限を 1/(n+1) にする慣例)
    for _ in range(n):
        yp = rng.permutation(y)
        if spearmanr(lodo(X, yp), yp).correlation >= observed:
            cnt += 1
    return cnt / (n + 1)


def main():
    names, X, ys, yf = load()
    yd = yf - ys
    ps, pf, pd_ = lodo(X, ys), lodo(X, yf), lodo(X, yd)

    out = {"_note": "予稿の表1・図1の生成元。LODO=13-fold hold-one-out, Ridge(alpha=1.0)+StandardScaler。"}
    rows = []
    for tag, y, p in [("scratch", ys, ps), ("ft", yf, pf), ("delta", yd, pd_)]:
        rho = spearmanr(p, y).correlation
        rows.append({
            "target": tag,
            "spearman": rho,
            "mae_pt": float(np.abs(p - y).mean()),
            "p_perm": perm_p(X, y, rho),
            "coef_fidtt_pt_per_sd": fold_mean_coef(X, y)["FID_train_test_DeepGaitV2"],
            "range_pt": float(y.max() - y.min()),
        })
        print(f"{tag:8s} rho={rho:.3f} MAE={np.abs(p-y).mean():.2f}pt "
              f"p_perm={rows[-1]['p_perm']:.4f} FID_tt={rows[-1]['coef_fidtt_pt_per_sd']:+.2f}pt/SD "
              f"range={rows[-1]['range_pt']:.1f}pt")
    out["table1"] = rows
    out["identity_check"] = {
        "max_abs_diff_delta_vs_ft_minus_scratch": float(np.abs(pd_ - (pf - ps)).max()),
        "corr_of_errors_scratch_ft": float(np.corrcoef(ps - ys, pf - yf)[0, 1]),
    }
    out["per_subset"] = {
        n: {"scratch_true": float(a), "scratch_pred": float(b),
            "ft_true": float(c), "ft_pred": float(d)}
        for n, a, b, c, d in zip(names, ys, ps, yf, pf)}

    with open(os.path.join(HERE, "prep_table1_20260727.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- 図1: 実測 vs LODO予測 ----
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    lo, hi = 35, 95
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=C_GRID, zorder=1)
    ax.scatter(ys, ps, s=34, marker="o", facecolor=C_PHASE1, edgecolor="white",
               linewidth=0.8, zorder=3, label="Phase 1: from scratch")
    ax.scatter(yf, pf, s=38, marker="^", facecolor=C_PHASE2, edgecolor="white",
               linewidth=0.8, zorder=3, label="Phase 2: fine-tuned")
    i8 = names.index("000-180")
    ax.annotate("000-180 (ft)", (yf[i8], pf[i8]), textcoords="offset points",
                xytext=(7, -3), fontsize=7, color=C_INK2)
    ax.annotate("000-180 (scratch)", (ys[i8], ps[i8]), textcoords="offset points",
                xytext=(7, -3), fontsize=7, color=C_INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured accuracy [%]", fontsize=8, color=C_INK)
    ax.set_ylabel("LODO-predicted accuracy [%]", fontsize=8, color=C_INK)
    ax.tick_params(labelsize=7, colors=C_INK2, length=3)
    ax.grid(True, lw=0.4, color=C_GRID, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C_GRID)
    leg = ax.legend(fontsize=7, loc="upper left", frameon=False, handletextpad=0.3,
                    borderpad=0.2, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_color(C_INK)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_pred_vs_true_20260727.{ext}"), dpi=300)
    print("\nwrote fig_pred_vs_true_20260727.{pdf,png}, prep_table1_20260727.json")


if __name__ == "__main__":
    main()
