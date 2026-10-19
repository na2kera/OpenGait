#!/usr/bin/env python3
"""発表スライド用の図を生成する (SVG/PNG)。数値は tables/tables_20260915.json (X3) から読む。

- 結果①: 表2 (main_grid) を簡略化。スクラッチ / ファインチューニングの順位相関 ρ を
          汎用 (Inception+DINOv2 11 変数) と歩容特化 (DeepGaitV2 6 変数) で並べる
- 結果②: 表3 (condition) の NM / BG / CL 別の順位相関 ρ。CL を強調

実行 (ホストに matplotlib が無いので docker):
  docker run --rm -v /home/kera/OpenGait:/app/OpenGait \
    -v /usr/share/fonts/opentype/noto:/fonts:ro opengait-opengait:latest \
    python /app/OpenGait/dgv2_analysis/techrep_20260915/fig_slides_20261001.py
出力: fig_slide_result1_rho_20261001.{svg,png}, fig_slide_result2_condition_20261001.{svg,png},
      fig_slide_result1_full_20261001.{svg,png}（予備スライド用。表2 の 4 特徴空間すべて）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in ("/fonts/NotoSansCJK-Regular.ttc", "/fonts/NotoSansCJK-Bold.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["svg.fonttype"] = "path"  # スライドに貼ってもフォントが化けないようにする

# dataviz スキルの検証済みパレット (categorical slot 1 / 2, light mode)。fig_slides_20260727.py と同じ
C_GEN = "#2a78d6"   # 汎用
C_GAIT = "#eb6834"  # 歩容特化
C_GRID = "#c9c8c3"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_BAND = "#f3f1ea"

GEN, GAIT = "S_incdino11", "S_sustech6"
L_GEN = "汎用特徴（InceptionV3+DINOv2）"
L_GAIT = "歩容特化特徴（DeepGaitV2）"
BW = 0.34
GAP = 0.03


def load():
    d = json.load(open(os.path.join(HERE, "tables", "tables_20260915.json")))
    assert d["provenance"]["arm"] == "X3"
    main = {r["key"]: r for r in d["main_grid"]}
    cond = {(r["target"], r["cond"]): r for r in d["condition"]}
    return main, cond


def style(ax):
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=13, colors=C_INK2, length=0)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.grid(True, axis="y", lw=0.6, color=C_GRID, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(C_GRID)


def bars(ax, x, gen, gait, bold=False, alpha=1.0):
    """1 グループ分 (汎用・歩容特化) の棒と値ラベルを描く。"""
    out = []
    for dx, v, c in ((-(BW + GAP) / 2, gen, C_GEN), ((BW + GAP) / 2, gait, C_GAIT)):
        b = FancyBboxPatch((x + dx - BW / 2, 0), BW, v, linewidth=0, facecolor=c, zorder=3,
                           alpha=alpha, boxstyle="round,pad=0,rounding_size=0.02", mutation_aspect=1.0)
        ax.add_patch(b)
        if alpha == 1.0:
            ax.add_patch(plt.Rectangle((x + dx - BW / 2, 0), BW, min(v, 0.05), linewidth=0,
                                       facecolor=c, zorder=3))  # 下端は角を丸めない
        ax.text(x + dx, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                color=C_INK if alpha == 1.0 else C_INK2,
                fontsize=17 if bold else 15, fontweight="bold" if bold else "normal")
        out.append(b)
    return out


def legend(fig):
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=C_GEN), plt.Rectangle((0, 0), 1, 1, facecolor=C_GAIT)]
    leg = fig.legend(h, [L_GEN, L_GAIT], loc="upper center", ncol=2, frameon=False, fontsize=14,
                     handlelength=1.0, handleheight=1.0, columnspacing=2.0,
                     bbox_to_anchor=(0.5, 1.0))
    for t in leg.get_texts():
        t.set_color(C_INK)


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200, facecolor="white")
    plt.close(fig)


NOTE_RHO = "ρ は予測順位と実測順位の一致度（認証精度ではない）"
NOTE_CL = "今回の 13 サブセットで観測された傾向．特徴空間間の差の検定は行っていない"


def fig_result1(main, note=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.9 if note else 4.6))
    for x, (key, label) in enumerate((("scratch_all", "スクラッチ"), ("ft_all", "ファインチューニング"))):
        bars(ax, x, main[GEN][key]["rho"], main[GAIT][key]["rho"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["スクラッチ", "ファインチューニング"], fontsize=16, color=C_INK)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("順位相関 ρ（Spearman）", fontsize=14, color=C_INK)
    style(ax)
    legend(fig)
    fig.tight_layout(pad=0.6, rect=(0, 0.07 if note else 0, 1, 0.91))
    if note:
        fig.text(0.99, 0.015, NOTE_RHO, ha="right", va="bottom", fontsize=11, color=C_INK2)
    save(fig, "fig_slide_result1_rho_note_20261001" if note else "fig_slide_result1_rho_20261001")


def fig_result1_full(main):
    """予備スライド用: 表2 の 4 特徴空間すべて。左がスクラッチ、右がファインチューニング。"""
    rows = (("S_inc6", "InceptionV3", C_GEN), ("S_dino6", "DINOv2", C_GEN),
            ("S_incdino11", "InceptionV3\n+DINOv2", C_GEN), ("S_sustech6", "DeepGaitV2", C_GAIT))
    w = 0.62
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True)
    for ax, (key, title) in zip(axes, (("scratch_all", "スクラッチ"), ("ft_all", "ファインチューニング"))):
        for x, (k, _, c) in enumerate(rows):
            v = main[k][key]["rho"]
            ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, v, linewidth=0, facecolor=c, zorder=3,
                                        boxstyle="round,pad=0,rounding_size=0.02"))
            ax.add_patch(plt.Rectangle((x - w / 2, 0), w, 0.05, linewidth=0, facecolor=c, zorder=3))
            ax.text(x, v + 0.015, f"{v:.3f}", ha="center", va="bottom", color=C_INK, fontsize=15)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[1] for r in rows], fontsize=12.5, color=C_INK, linespacing=1.25)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_title(title, fontsize=16, color=C_INK, pad=10)
        style(ax)
    axes[0].set_ylabel("順位相関 ρ（Spearman）", fontsize=14, color=C_INK)
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=C_GEN), plt.Rectangle((0, 0), 1, 1, facecolor=C_GAIT)]
    leg = fig.legend(h, ["汎用特徴", "歩容特化特徴"], loc="upper center", ncol=2, frameon=False,
                     fontsize=14, handlelength=1.0, handleheight=1.0, columnspacing=2.0,
                     bbox_to_anchor=(0.5, 1.0))
    for t in leg.get_texts():
        t.set_color(C_INK)
    fig.tight_layout(pad=0.6, w_pad=2.0, rect=(0, 0, 1, 0.92))
    save(fig, "fig_slide_result1_full_20261001")


def fig_result2(cond, variant="band"):
    """variant: band=CL に薄い背景 / box=CL を枠で囲む / dim=NM・BG を薄くする / note=band+脚注。"""
    note = variant == "note"
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2 if note else 4.8), sharey=True)
    for ax, (tgt, title) in zip(axes, (("scratch", "スクラッチ"), ("ft", "ファインチューニング"))):
        if variant in ("band", "note"):
            ax.axvspan(1.5, 2.5, color=C_BAND, zorder=0)  # CL を強調
        elif variant == "box":
            ax.add_patch(FancyBboxPatch((1.56, 0.0), 0.88, 1.05, fill=False, edgecolor=C_INK,
                                        linewidth=1.6, zorder=5, clip_on=False,
                                        boxstyle="round,pad=0,rounding_size=0.04"))
        for x, c in enumerate(("NM", "BG", "CL")):
            r = cond[(tgt, c)]
            bars(ax, x, r[GEN]["rho"], r[GAIT]["rho"], bold=(c == "CL"),
                 alpha=0.35 if (variant == "dim" and c != "CL") else 1.0)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["NM\n通常歩行", "BG\n鞄所持", "CL\nコート着用"], fontsize=14, color=C_INK,
                           linespacing=1.25)
        ax.get_xticklabels()[2].set_fontweight("bold")
        ax.set_xlim(-0.55, 2.5)
        ax.set_title(title, fontsize=16, color=C_INK, pad=10)
        style(ax)
    axes[0].set_ylabel("順位相関 ρ（Spearman）", fontsize=14, color=C_INK)
    legend(fig)
    fig.tight_layout(pad=0.6, w_pad=2.0, rect=(0, 0.07 if note else 0, 1, 0.92))
    if note:
        fig.text(0.99, 0.015, NOTE_CL, ha="right", va="bottom", fontsize=11, color=C_INK2)
    suffix = "" if variant == "band" else f"_{variant}"
    save(fig, f"fig_slide_result2_condition{suffix}_20261001")


if __name__ == "__main__":
    main, cond = load()
    fig_result1(main)
    fig_result1_full(main)
    fig_result1(main, note=True)
    for v in ("band", "box", "dim", "note"):
        fig_result2(cond, v)
    for k in ("scratch_all", "ft_all"):
        print(k, round(main[GEN][k]["rho"], 3), round(main[GAIT][k]["rho"], 3))
    for (t, c), r in cond.items():
        print(t, c, round(r[GEN]["rho"], 3), round(r[GAIT]["rho"], 3))
