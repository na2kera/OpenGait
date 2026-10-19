#!/usr/bin/env python3
"""スクラッチ学習精度 vs ファインチューニング後精度の散布図（13 サブセット、実測値どうし）。

本文 5.1 節・7 章の「スクラッチ学習精度とファインチューニング後精度の順位相関は 0.973 と高く」を図にしたもの。
数値は ../metrics_ft_sustech_full_20260726.json から読む（acc_all = NM/BG/CL の固定重み 0.6/0.2/0.2 平均）。再計算なし。

実行 (ホストに matplotlib が無いので docker):
  docker run --rm -v /home/kera/OpenGait:/app/OpenGait \
    -v /usr/share/fonts/opentype/noto:/fonts:ro opengait-opengait:latest \
    python /app/OpenGait/dgv2_analysis/techrep_20260915/fig_scratch_vs_ft_20261007.py
出力: fig_scratch_vs_ft_20261007.{svg,png,pdf}
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in ("/fonts/NotoSansCJK-Regular.ttc", "/fonts/NotoSansCJK-Bold.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["svg.fonttype"] = "path"  # スライドに貼ってもフォントが化けないようにする
matplotlib.rcParams["pdf.fonttype"] = 42

SUBSETS = ["default", "nm2", "bg2", "cl2", "nm1-bg1", "nm1-cl1", "bg1-cl1", "000-180", "000-090",
           "090-180", "nm1-bg1-cl1", "nm2-bg2-cl2", "nm6"]

# 汎用（青）/ 歩容特化（橙）は特徴空間の色なので使わない。1 系列なので濃いグレー
C_MARK = "#3b3a38"
C_GRID = "#c9c8c3"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"

# ラベルの置き場所（offset points, ha）。既定は右。重なる点だけ個別に指定
OFFSET = {
    "nm1-cl1": (-8, 5, "right"),
    "bg1-cl1": (8, -6, "left"),
    "nm1-bg1-cl1": (-8, 2, "right"),
    "bg2": (-8, 3, "right"),
    "cl2": (-8, 4, "right"),          # cl2 と nm2-bg2-cl2 はほぼ同じ位置（73.9, 83.1）/（73.7, 83.1）
    "nm2-bg2-cl2": (0, -13, "center"),
    "default": (0, -13, "center"),
}
DEFAULT_OFFSET = (7, -4, "left")


def main():
    d = json.load(open(os.path.join(os.path.dirname(HERE), "metrics_ft_sustech_full_20260726.json")))
    assert sorted(d) == sorted(SUBSETS)
    x = np.array([d[s]["scratch_acc_all"] for s in SUBSETS]) * 100
    y = np.array([d[s]["acc_all"] for s in SUBSETS]) * 100
    rho = spearmanr(x, y).correlation
    assert abs(rho - 0.973) < 5e-4, rho  # 本文の値と一致すること
    for s, a, b in sorted(zip(SUBSETS, x, y), key=lambda r: r[1]):
        print(f"{s:12s} scratch={a:6.2f} ft={b:6.2f} delta={b - a:+6.2f}")
    print(f"rho={rho:.4f}")

    lo, hi = 35, 95
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(x, y, s=70, marker="o", facecolor=C_MARK, edgecolor="white", linewidth=1.2, zorder=3)
    for s, a, b in zip(SUBSETS, x, y):
        dx, dy, ha = OFFSET.get(s, DEFAULT_OFFSET)
        ax.annotate(s, (a, b), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center",
                    fontsize=10, color=C_INK2)
    ax.text(0.04, 0.96, f"順位相関 ρ = {rho:.3f}", transform=ax.transAxes, ha="left", va="top",
            fontsize=16, fontweight="bold", color=C_INK)
    ax.text(0.04, 0.895, "13 サブセット，実測値", transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color=C_INK2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xticks(range(40, 91, 10))
    ax.set_yticks(range(40, 91, 10))
    ax.set_xlabel("スクラッチ学習精度 [%]", fontsize=14, color=C_INK, labelpad=8)
    ax.set_ylabel("ファインチューニング後精度 [%]", fontsize=14, color=C_INK, labelpad=8)
    ax.tick_params(labelsize=12, colors=C_INK2, length=0)
    ax.grid(True, lw=0.6, color=C_GRID, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C_GRID)

    fig.tight_layout(pad=0.6)
    base = os.path.join(HERE, "fig_scratch_vs_ft_20261007")
    for ext in ("svg", "png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=200)
    print("wrote", os.path.basename(base) + ".{svg,png,pdf}")


if __name__ == "__main__":
    main()


def main_rank():
    """順位どうしの散布図（ρ は順位の相関なので、こちらが ρ=0.973 を直接表す）。出力: fig_scratch_vs_ft_rank_20261007.*"""
    from scipy.stats import pearsonr, rankdata
    d = json.load(open(os.path.join(os.path.dirname(HERE), "metrics_ft_sustech_full_20260726.json")))
    x = np.array([d[s]["scratch_acc_all"] for s in SUBSETS])
    y = np.array([d[s]["acc_all"] for s in SUBSETS])
    rx, ry = rankdata(-x), rankdata(-y)  # 1 位 = 最高精度
    rho = spearmanr(x, y).correlation
    assert abs(pearsonr(rx, ry)[0] - rho) < 1e-12
    keep = np.array([s != "000-180" for s in SUBSETS])
    print(f"pearson r={pearsonr(x, y)[0]:.3f}  rho(000-180 を除く 12 点)={spearmanr(x[keep], y[keep]).correlation:.3f}")
    print("順位が動いたサブセット:", [(s, int(a), int(b)) for s, a, b in zip(SUBSETS, rx, ry) if a != b])

    n = len(SUBSETS)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0.3, n + 0.7], [0.3, n + 0.7], ls="--", lw=1.2, color=C_GRID, zorder=1)
    ax.scatter(rx, ry, s=70, marker="o", facecolor=C_MARK, edgecolor="white", linewidth=1.2, zorder=3)
    for s, a, b in zip(SUBSETS, rx, ry):
        left = a <= 3  # 画面右端の 3 点はラベルを左に置く
        ax.annotate(s, (a, b), textcoords="offset points", xytext=(-8 if left else 8, 0),
                    ha="right" if left else "left", va="center", fontsize=10, color=C_INK2)
    ax.text(0.04, 0.96, f"順位相関 ρ = {rho:.3f}", transform=ax.transAxes, ha="left", va="top",
            fontsize=16, fontweight="bold", color=C_INK)
    ax.text(0.04, 0.895, "13 サブセット，実測精度の順位\n破線：順位が一致", transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color=C_INK2)
    ax.set_xlim(n + 0.7, 0.3)  # 右上が 1 位
    ax.set_ylim(n + 0.7, 0.3)
    ax.set_aspect("equal")
    ax.set_xticks(range(1, n + 1, 2))
    ax.set_yticks(range(1, n + 1, 2))
    ax.set_xlabel("スクラッチ学習精度の順位", fontsize=14, color=C_INK, labelpad=8)
    ax.set_ylabel("ファインチューニング後精度の順位", fontsize=14, color=C_INK, labelpad=8)
    ax.tick_params(labelsize=12, colors=C_INK2, length=0)
    ax.grid(True, lw=0.6, color=C_GRID, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C_GRID)
    fig.tight_layout(pad=0.6)
    base = os.path.join(HERE, "fig_scratch_vs_ft_rank_20261007")
    for ext in ("svg", "png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=200)
    print("wrote", os.path.basename(base) + ".{svg,png,pdf}")


if __name__ == "__main__":
    main_rank()
