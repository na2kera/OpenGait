#!/usr/bin/env python3
"""特徴セット（S_incdino11 / S_inc6 / S_dino6 / S_sustech6）× scratch_all の並べ替え検定（事後追加, 2026-09-10）。

X2 本分析（analysis_incdino_ft_20260806.py）は事前登録の 8 本（ft_all / delta_pt × 3 セット＋LogReg 2 本）しか
並べ替え検定を回していない。主表を 2×2（scratch / ft）にするにあたり scratch 列にも p を付けるための追加計算。
手法・N・LODO Ridge は本分析と同一（lodo_ridge / FEATURE_SETS / load_df をそのまま import）。seed は既存の
k0..k7 (1000..1007) と重ならない 1010..1013 を使う。事前登録外なので原稿では「事後追加」と明記する。

実行（docker, worktree をマウント）:
  docker run --rm -v /home/kera/OpenGait:/app/OpenGait -v /home/kera/worktrees:/home/kera/worktrees \
      -w /app/OpenGait opengait:latest python dgv2_analysis/techrep_20260915/perm_scratch_20260910.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

WT = "/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis"
sys.path.insert(0, WT)
import analysis_incdino_ft_20260806 as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "perm_scratch_20260910.json")
X2 = json.load(open(os.path.join(WT, A.ARM_DEFAULTS["X2"]["out"])))

df, prov, keys = A.load_df(os.path.join(WT, A.ARM_DEFAULTS["X2"]["incdino"]))
def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()

inputs = {"incdino": os.path.join(WT, A.ARM_DEFAULTS["X2"]["incdino"]),
          "sustech": os.path.join(WT, "metrics_dgv2_sustech.json"),
          "ft": os.path.join(WT, "metrics_ft_sustech_full_20260726.json"),
          "analysis_code": A.__file__,
          "x2_result": os.path.join(WT, A.ARM_DEFAULTS["X2"]["out"])}
out = {"inputs_md5": {k: {"path": v, "md5": md5(v)} for k, v in inputs.items()},
       "note": "事後追加（事前登録外）。X2 と同一の LODO Ridge、N_PERM=1000、片側 (perm ρ >= 観測 ρ)。seed 1010..1013（S_sustech6 は 7/27 予稿の転記値 0.001 の再計算）",
       "n_perm": A.N_PERM_RIDGE, "arm": "X2", "target": "scratch_all", "results": {}}
for k, sname in enumerate(("S_incdino11", "S_inc6", "S_dino6", "S_sustech6")):
    seed = 1010 + k
    rng = np.random.RandomState(seed)
    obs = A.lodo_ridge(df, "scratch_all", A.FEATURE_SETS[sname])["spearman"]
    ref = X2["E_grid"][sname]["scratch_all"]["rho"]
    assert abs(obs - ref) < 1e-9, f"{sname}: 観測 ρ {obs} が X2 E_grid {ref} と一致しない"
    y = df["scratch_all"].values.copy()
    t0 = time.time()
    cnt = sum(A.lodo_ridge(df, "scratch_all", A.FEATURE_SETS[sname],
                           y_override=rng.permutation(y))["spearman"] >= obs
              for _ in range(A.N_PERM_RIDGE))
    p = (cnt + 1) / (A.N_PERM_RIDGE + 1)
    out["results"][sname] = {"observed_spearman": float(obs), "p_perm": float(p), "seed": seed,
                             "count_ge": int(cnt)}
    print(f"{sname:12s} scratch_all ρ={obs:+.4f} p={p:.4f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
print("wrote", OUT)
