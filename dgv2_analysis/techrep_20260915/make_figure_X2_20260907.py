#!/usr/bin/env python3
"""信学技報用 図1（実測精度 vs LODO 予測精度）を X2 結果 JSON から描く。再計算なし。

入力: result_incdino_ft_20260806_X2.json の grid[*][target].per_subset（true/pred）と
      A_sustech6_ft_reproduced.per_subset（DeepGaitV2 特徴・ft・既報と ρ 差 0 を確認済みの再現値）
出力（techrep_20260915/）:
  fig_pred_vs_true_X2_20260907.{pdf,png}       (a) ft のみ。Inception(6変数) vs DeepGaitV2(6変数)  ← 案A+B の主図候補
  fig_pred_vs_true_X2_2panel_20260907.{pdf,png} (b) 左 scratch / 右 ft。各パネルに Inception と DeepGaitV2
  fig_X2_20260907.json                          描画に使った値（true/pred, ρ, MAE）と PDF の MediaBox（\\includegraphics の bb 用）

実行: docker run --rm -v /home/kera/OpenGait:/app/OpenGait -v /home/kera/worktrees:/home/kera/worktrees \
        -w /app/OpenGait opengait:latest python dgv2_analysis/techrep_20260915/make_figure_X2_20260907.py

決定B-11（図をどれにするか）の判断材料として (a)(b) の両方を出す。
"""
import json
import os
import re

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.environ.get("ARM", "X2")
GEN_SET = os.environ.get("GEN_SET", "S_inc6")          # 汎用側の系列: S_inc6（既定、6 変数）/ S_incdino11（11 変数、2026-09-14 追加）
assert GEN_SET in ("S_inc6", "S_incdino11")
GEN_LABEL = {"S_inc6": "Inception", "S_incdino11": "Inception+DINOv2"}[GEN_SET]
GEN_SUFFIX = "" if GEN_SET == "S_inc6" else "_incdino11"
X2_RESULT = ("/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis/result_incdino_ft_20260806_X2.json" if ARM == "X2"
             else os.path.join(HERE, "result_incdino_ft_X3.json"))
SUBSETS = ["default", "nm2", "bg2", "cl2", "nm1-bg1", "nm1-cl1", "bg1-cl1", "000-180", "000-090",
           "090-180", "nm1-bg1-cl1", "nm2-bg2-cl2", "nm6"]

# 7/27 の図と同じパレット
C_GENERIC = "#2a78d6"   # Inception（汎用）
C_GAIT = "#eb6834"      # DeepGaitV2（歩容特化）
C_GRID = "#c9c8c3"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"


def series(per_subset):
    t = np.array([per_subset[s]["true"] for s in SUBSETS]) * 100
    p = np.array([per_subset[s]["pred"] for s in SUBSETS]) * 100
    return t, p


def style(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=C_GRID, zorder=1)
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


def legend(ax):
    leg = ax.legend(fontsize=7, loc="upper left", frameon=False, handletextpad=0.3,
                    borderpad=0.2, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_color(C_INK)


def mediabox(pdf_path):
    with open(pdf_path, "rb") as f:
        m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", f.read())
    return [float(x) for x in m.groups()] if m else None


def main():
    x2 = json.load(open(X2_RESULT, encoding="utf-8"))
    assert x2["arm"] == ARM
    data = {
        "inc6_ft": series(x2["grid"][GEN_SET]["ft_all"]["per_subset"]),
        "inc6_scratch": series(x2["grid"][GEN_SET]["scratch_all"]["per_subset"]),
        "sus6_ft": series(x2["A_sustech6_ft_reproduced"]["per_subset"]),
    }
    # DeepGaitV2 × scratch の per-subset は X2 JSON に無い（フェーズ1 既報）。presentation_tables_20260726.json から転記
    pt = json.load(open(os.path.join(os.path.dirname(HERE), "presentation_tables_20260726.json"), encoding="utf-8"))
    ps = pt["lodo_predictions"]["DGV2(6変数)"]["per_subset"]
    data["sus6_scratch"] = (np.array([ps[s]["true_pct"] for s in SUBSETS]), np.array([ps[s]["pred_pct"] for s in SUBSETS]))
    if ARM == "X3":  # X3 結果には S_sustech6 の grid があるので、既報 JSON との一致を確認する（歩容側は無変更のはず）
        g = x2["grid"]["S_sustech6"]["scratch_all"]["per_subset"]
        assert all(abs(g[s]["true"] * 100 - ps[s]["true_pct"]) < 1e-6 and abs(g[s]["pred"] * 100 - ps[s]["pred_pct"]) < 1e-6 for s in SUBSETS), "S_sustech6 scratch の per_subset が X3 結果と既報 JSON で不一致"

    # 整合性: JSON の ρ/MAE と per_subset から出る値が一致
    summary = {}
    for k, (t, p) in data.items():
        rho = spearmanr(p, t).correlation
        mae = float(np.abs(p - t).mean())
        summary[k] = {"rho": float(rho), "mae_pt": mae}
    assert abs(summary["inc6_ft"]["rho"] - x2["grid"][GEN_SET]["ft_all"]["spearman"]) < 1e-9
    assert abs(summary["inc6_ft"]["mae_pt"] - x2["grid"][GEN_SET]["ft_all"]["mae"] * 100) < 1e-6
    assert abs(summary["sus6_ft"]["rho"] - 0.7417582417582418) < 1e-9
    assert abs(summary["sus6_scratch"]["rho"] - 0.8076923076923077) < 1e-9
    assert abs(summary["inc6_scratch"]["rho"] - x2["grid"][GEN_SET]["scratch_all"]["spearman"]) < 1e-9
    for k, v in summary.items():
        print(f"{k:14s} rho={v['rho']:.3f} MAE={v['mae_pt']:.2f}pt")

    i8 = SUBSETS.index("000-180")
    out = {"source": X2_RESULT, "summary": summary,
           "per_subset": {k: {s: {"true": float(t[i]), "pred": float(p[i])} for i, s in enumerate(SUBSETS)}
                          for k, (t, p) in data.items()}}

    # ---- (a) ft のみ、特徴空間2種 ----
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    lo, hi = 30, 95
    style(ax, lo, hi)
    t, p = data["inc6_ft"]
    ax.scatter(t, p, s=34, marker="o", facecolor=C_GENERIC, edgecolor="white", linewidth=0.8, zorder=3,
               label=f"{GEN_LABEL} ($\\rho$={summary['inc6_ft']['rho']:.3f})")
    ax.annotate("000-180", (t[i8], p[i8]), textcoords="offset points", xytext=(6, -3), fontsize=7, color=C_INK2)
    t, p = data["sus6_ft"]
    ax.scatter(t, p, s=38, marker="^", facecolor=C_GAIT, edgecolor="white", linewidth=0.8, zorder=3,
               label=f"DeepGaitV2 ($\\rho$={summary['sus6_ft']['rho']:.3f})")
    ax.annotate("000-180", (t[i8], p[i8]), textcoords="offset points", xytext=(6, -3), fontsize=7, color=C_INK2)
    legend(ax)
    fig.tight_layout(pad=0.3)
    base_a = os.path.join(HERE, "fig_pred_vs_true_X2_20260907" if ARM == "X2" else "fig_pred_vs_true_X3" + GEN_SUFFIX + "_20260914")
    for ext in ("pdf", "png"):
        fig.savefig(f"{base_a}.{ext}", dpi=300)
    plt.close(fig)

    # ---- (b) 2パネル: scratch / ft ----
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    for ax, tgt, title in zip(axes, ("scratch", "ft"), ("(a) From scratch", "(b) Fine-tuned")):
        style(ax, lo, hi)
        t, p = data[f"inc6_{tgt}"]
        ax.scatter(t, p, s=30, marker="o", facecolor=C_GENERIC, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"{GEN_LABEL} ($\\rho$={summary[f'inc6_{tgt}']['rho']:.3f})")
        ax.annotate("000-180", (t[i8], p[i8]), textcoords="offset points", xytext=(6, -3), fontsize=7, color=C_INK2)
        t, p = data[f"sus6_{tgt}"]
        ax.scatter(t, p, s=34, marker="^", facecolor=C_GAIT, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"DeepGaitV2 ($\\rho$={summary[f'sus6_{tgt}']['rho']:.3f})")
        ax.annotate("000-180", (t[i8], p[i8]), textcoords="offset points", xytext=(6, -3), fontsize=7, color=C_INK2)
        ax.set_title(title, fontsize=8, color=C_INK, loc="left")
        legend(ax)
    # 000-180 の予測が軸の下限（lo）を下回る場合は、そのパネルだけ y 軸を下に広げる（外挿破綻を見せる）。
    # scratch の汎用側は負になり、11 変数では ft も 30% を下回る（2026-09-14 に両パネル対応に変更）
    for ax, tgt in zip(axes, ("scratch", "ft")):
        lo_t = min(float(data[f"inc6_{tgt}"][1].min()), float(data[f"sus6_{tgt}"][1].min()), lo)
        if lo_t < lo:
            ax.set_ylim(lo_t - 5, hi)
            ax.set_xlim(lo, hi)
            ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=C_GRID, zorder=1)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    base_b = os.path.join(HERE, "fig_pred_vs_true_X2_2panel_20260907" if ARM == "X2" else "fig_pred_vs_true_X3_2panel" + GEN_SUFFIX + "_20260914")
    for ext in ("pdf", "png"):
        fig.savefig(f"{base_b}.{ext}", dpi=300)
    plt.close(fig)

    out["general_set"] = GEN_SET
    out["mediabox_bp"] = {os.path.basename(base_a) + ".pdf": mediabox(base_a + ".pdf"),
                          os.path.basename(base_b) + ".pdf": mediabox(base_b + ".pdf")}
    with open(os.path.join(HERE, "fig_X2_20260907.json" if ARM == "X2" else "fig_X3" + GEN_SUFFIX + "_20260914.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("MediaBox (bp):", out["mediabox_bp"])
    print("wrote", os.path.basename(base_a), os.path.basename(base_b), "fig_X2_20260907.json")


if __name__ == "__main__":
    main()
