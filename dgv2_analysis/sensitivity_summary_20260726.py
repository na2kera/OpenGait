#!/usr/bin/env python3
"""③ checkpoint感度分析の集計 (2026-07-26)。

入力ログ:
- output/ft_sustech_sensitivity_zeroshot_20260726.log (zeroshot + 11本×5iter + 偵察2本×15k/25k)
- output/ft_sustech_ckpt_sensitivity_20260723.log     (偵察s0 2本 × 5k/10k/20k)
- metrics_ft_sustech_full_20260726.json               (全13本の30k)

出力:
- 13サブセット × iter {5k,10k,15k,20k,25k,30k} の acc_all グリッド + zeroshot
- 感度サマリ: 各iterでのΔ全正性 / Spearman(Δ_iter, Δ_30k) / LODO Ridge・
  優劣判定の主要量のiter安定性
- ft_sustech_sensitivity_grid_20260726.json

実行: docker exec opengait_container python /app/OpenGait/dgv2_analysis/sensitivity_summary_20260726.py
"""
import itertools
import json
import math
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "output")

ID2SUBSET = {"1": "default", "2": "nm2", "3": "bg2", "4": "cl2", "5": "nm1-bg1",
             "6": "nm1-cl1", "7": "bg1-cl1", "8": "000-180", "9": "000-090",
             "10": "090-180", "11": "nm1-bg1-cl1", "12": "nm2-bg2-cl2", "13": "nm6"}
SUBSETS = list(ID2SUBSET.values())
ITERS = [5000, 10000, 15000, 20000, 25000, 30000]

FEATURE_COLS = [
    "log_people",
    "DeepGaitV2_MSD",
    "DeepGaitV2_1NN",
    "DeepGaitV2_kNN",
    "DeepGaitV2_FID",
    "FID_train_test_DeepGaitV2",
]

MARKER = re.compile(r"=== TEST (\S+?)(?:-s0)? (?:iter=(\d+) )?START")
SUMMARY = re.compile(r"NM@R1: ([\d.]+)%\s+BG@R1: ([\d.]+)%\s+CL@R1: ([\d.]+)%")


def acc_all(nm, bg, cl):
    return 0.6 * nm + 0.2 * bg + 0.2 * cl


def parse_log(path):
    """(subset, iter) -> {nm, bg, cl, all} (percent)。iterなしマーカーはiter=None。"""
    results = {}
    current = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = MARKER.search(line)
            if m:
                current = (m.group(1), int(m.group(2)) if m.group(2) else None)
                continue
            s = SUMMARY.search(line)
            if s and current is not None:
                nm, bg, cl = map(float, s.groups())
                results[current] = {"nm": nm, "bg": bg, "cl": cl,
                                    "all": acc_all(nm, bg, cl)}
                current = None
    return results


def main():
    grid = {}  # subset -> iter -> acc_all(%)
    r_new = parse_log(os.path.join(OUT_DIR, "ft_sustech_sensitivity_zeroshot_20260726.log"))
    r_old = parse_log(os.path.join(OUT_DIR, "ft_sustech_ckpt_sensitivity_20260723.log"))

    zeroshot = r_new.get(("zeroshot", None))
    print("=== zero-shot (fine-tuningなしのベースモデル、CASIA-B 49人テスト) ===")
    if zeroshot:
        print(f"NM@R1={zeroshot['nm']:.2f}  BG@R1={zeroshot['bg']:.2f}  "
              f"CL@R1={zeroshot['cl']:.2f}  acc_all={zeroshot['all']:.2f}")
    else:
        print("!! zeroshot結果がログに見つからない")

    with open(os.path.join(HERE, "metrics_ft_sustech_full_20260726.json")) as f:
        ft30k = json.load(f)
    with open(os.path.join(HERE, "metrics_dgv2_sustech.json")) as f:
        scratch = json.load(f)
    scratch_all = {ID2SUBSET[k]: v["acc_all"] * 100 for k, v in scratch.items()}

    merged = {}
    merged.update({k: v for k, v in r_old.items() if k[1] is not None})
    merged.update({k: v for k, v in r_new.items() if k[1] is not None})
    for sub in SUBSETS:
        grid[sub] = {}
        for it in ITERS:
            if it == 30000:
                grid[sub][it] = ft30k[sub]["acc_all"] * 100
            elif (sub, it) in merged:
                grid[sub][it] = merged[(sub, it)]["all"]
            else:
                grid[sub][it] = None

    missing = [(s, i) for s in SUBSETS for i in ITERS if grid[s][i] is None]
    if missing:
        print(f"\n!! 欠損セル {len(missing)}: {missing}")

    print("\n=== acc_all グリッド (13サブセット × 6 iter, %) ===")
    gdf = pd.DataFrame(grid).T[ITERS]
    print(gdf.round(2).to_string())

    # --- 感度サマリ ---
    print("\n=== 感度サマリ ===")
    d30 = {s: grid[s][30000] - scratch_all[s] for s in SUBSETS}
    feats = {}
    for k, sub in ID2SUBSET.items():
        s = scratch[k]
        feats[sub] = [math.log(s["Number of people"])] + [s[c] for c in FEATURE_COLS[1:]]
    X = np.array([feats[s] for s in SUBSETS])

    summary = {}
    print(f"{'iter':>6s} {'Δ全正?':>7s} {'Δmin':>7s} {'ρ(Δ,Δ30k)':>10s} "
          f"{'ρ(ft,ft30k)':>11s} {'LODOρ(Δ)':>9s} {'FID_tt係数':>10s}")
    for it in ITERS:
        if any(grid[s][it] is None for s in SUBSETS):
            continue
        d_it = {s: grid[s][it] - scratch_all[s] for s in SUBSETS}
        all_pos = all(v > 0 for v in d_it.values())
        rho_d, _ = spearmanr([d_it[s] for s in SUBSETS], [d30[s] for s in SUBSETS])
        rho_f, _ = spearmanr([grid[s][it] for s in SUBSETS], [grid[s][30000] for s in SUBSETS])
        # LODO ridge (target=Δ_iter)
        y = np.array([d_it[s] for s in SUBSETS])
        yt, yp, coefs = [], [], []
        for i in range(len(SUBSETS)):
            mask = np.arange(len(SUBSETS)) != i
            mdl = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=1.0))])
            mdl.fit(X[mask], y[mask])
            yp.append(mdl.predict(X[~mask])[0])
            yt.append(y[~mask][0])
            coefs.append(mdl.named_steps["r"].coef_)
        lodo_rho, _ = spearmanr(yt, yp)
        fid_coef = float(np.mean(coefs, axis=0)[FEATURE_COLS.index("FID_train_test_DeepGaitV2")])
        summary[it] = {"delta_all_positive": bool(all_pos),
                       "delta_min_pt": float(min(d_it.values())),
                       "spearman_delta_vs_30k": float(rho_d),
                       "spearman_ft_vs_30k": float(rho_f),
                       "lodo_spearman_delta": float(lodo_rho),
                       "fid_tt_coef": fid_coef}
        print(f"{it:>6d} {str(all_pos):>7s} {min(d_it.values()):7.2f} {rho_d:10.3f} "
              f"{rho_f:11.3f} {lodo_rho:9.3f} {fid_coef:+10.3f}")

    out = {
        "zeroshot": zeroshot,
        "grid_acc_all_pct": {s: {str(i): grid[s][i] for i in ITERS} for s in SUBSETS},
        "grid_conditions_pct": {f"{s}@{i}": merged[(s, i)] for (s, i) in merged},
        "scratch_acc_all_pct": scratch_all,
        "sensitivity_summary": {str(k): v for k, v in summary.items()},
    }
    out_path = os.path.join(HERE, "ft_sustech_sensitivity_grid_20260726.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
