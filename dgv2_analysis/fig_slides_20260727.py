#!/usr/bin/env python3
"""0727進捗報告スライド用の図を生成する (SVG/PNG)。

- 図A: scratch → ft のダンベルプロット (ft精度順、zero-shot基準線つき)
- 図B: 実測 vs LODO予測 の散布図 (make_figures_20260727.py の図1をスライド向けに再レンダリング。
       予測値は prep_table1_20260727.json の per_subset を再利用し再計算しない)

実行: docker exec opengait_container python /app/OpenGait/dgv2_analysis/fig_slides_20260727.py
出力: fig_slide_dumbbell_20260727.{svg,png}, fig_slide_pred_vs_true_20260727.{svg,png}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"  # SVGを軽くする(テキストをパス化しない)
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ID2SUBSET = {1: "default", 2: "nm2", 3: "bg2", 4: "cl2", 5: "nm1-bg1", 6: "nm1-cl1",
             7: "bg1-cl1", 8: "000-180", 9: "000-090", 10: "090-180",
             11: "nm1-bg1-cl1", 12: "nm2-bg2-cl2", 13: "nm6"}
ZS = 70.83  # zero-shot acc_all (ft_sustech_sensitivity_grid_20260726.json)

# dataviz スキルの検証済みパレット (categorical slot 1 / 2, light mode)
C_SCRATCH = "#2a78d6"
C_FT = "#eb6834"
C_GRID = "#c9c8c3"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"


def load():
    m = json.load(open(os.path.join(HERE, "metrics_dgv2_sustech.json")))
    ft = json.load(open(os.path.join(HERE, "metrics_ft_sustech_full_20260726.json")))
    rows = []
    for i in range(1, 14):
        s = ID2SUBSET[i]
        rows.append((s, m[str(i)]["acc_all"] * 100, ft[s]["acc_all"] * 100))
    return rows


def fig_dumbbell(rows):
    rows = sorted(rows, key=lambda r: r[2])  # ft昇順 → 上が最良
    names = [r[0] for r in rows]
    ys = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axvline(ZS, ls="--", lw=1.2, color=C_INK2, zorder=2)
    ax.text(ZS, len(rows) - 0.05, f"zero-shot {ZS:.1f}%", fontsize=9, color=C_INK2,
            ha="center", va="bottom")
    for y, (s, sc, ft) in zip(ys, rows):
        ax.plot([sc, ft], [y, y], lw=1.6, color=C_GRID, zorder=3)
        ax.annotate(f"+{ft - sc:.1f}", (max(sc, ft), y), textcoords="offset points",
                    xytext=(8, -3), fontsize=8.5, color=C_INK2)
    ax.scatter([r[1] for r in rows], ys, s=64, marker="o", facecolor=C_SCRATCH,
               edgecolor="white", linewidth=1.0, zorder=4, label="scratch (phase 1)")
    ax.scatter([r[2] for r in rows], ys, s=72, marker="^", facecolor=C_FT,
               edgecolor="white", linewidth=1.0, zorder=4, label="fine-tuned (phase 2)")
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=9.5, color=C_INK)
    ax.set_xlim(35, 100)
    ax.set_ylim(-0.7, len(rows) - 0.3 + 0.8)
    ax.set_xlabel("Rank-1 accuracy acc_all [%]", fontsize=10, color=C_INK)
    ax.tick_params(axis="x", labelsize=9, colors=C_INK2, length=3)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", lw=0.4, color=C_GRID, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(C_GRID)
    leg = ax.legend(fontsize=9, loc="lower right", frameon=False,
                    handletextpad=0.3, borderpad=0.2, labelspacing=0.4)
    for t in leg.get_texts():
        t.set_color(C_INK)
    fig.tight_layout(pad=0.4)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(HERE, f"fig_slide_dumbbell_20260727.{ext}"), dpi=200)
    plt.close(fig)


def fig_scatter():
    prep = json.load(open(os.path.join(HERE, "prep_table1_20260727.json")))
    ps = prep["per_subset"]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    lo, hi = 35, 95
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=C_GRID, zorder=1)
    ax.scatter([v["scratch_true"] for v in ps.values()],
               [v["scratch_pred"] for v in ps.values()],
               s=56, marker="o", facecolor=C_SCRATCH, edgecolor="white",
               linewidth=1.0, zorder=3, label="Phase 1: from scratch")
    ax.scatter([v["ft_true"] for v in ps.values()],
               [v["ft_pred"] for v in ps.values()],
               s=64, marker="^", facecolor=C_FT, edgecolor="white",
               linewidth=1.0, zorder=3, label="Phase 2: fine-tuned")
    v8 = ps["000-180"]
    ax.annotate("000-180 (ft)", (v8["ft_true"], v8["ft_pred"]),
                textcoords="offset points", xytext=(9, -4), fontsize=9, color=C_INK2)
    ax.annotate("000-180 (scratch)", (v8["scratch_true"], v8["scratch_pred"]),
                textcoords="offset points", xytext=(9, -4), fontsize=9, color=C_INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured accuracy [%]", fontsize=10, color=C_INK)
    ax.set_ylabel("LODO-predicted accuracy [%]", fontsize=10, color=C_INK)
    ax.tick_params(labelsize=9, colors=C_INK2, length=3)
    ax.grid(True, lw=0.4, color=C_GRID, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C_GRID)
    leg = ax.legend(fontsize=9, loc="upper left", frameon=False,
                    handletextpad=0.3, borderpad=0.2, labelspacing=0.4)
    for t in leg.get_texts():
        t.set_color(C_INK)
    fig.tight_layout(pad=0.4)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(HERE, f"fig_slide_pred_vs_true_20260727.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    rows = load()
    fig_dumbbell(rows)
    fig_scatter()
    print("wrote fig_slide_dumbbell_20260727.{svg,png}, fig_slide_pred_vs_true_20260727.{svg,png}")
