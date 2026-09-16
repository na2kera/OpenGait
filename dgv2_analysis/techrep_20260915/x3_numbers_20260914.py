#!/usr/bin/env python3
"""原稿本文に載せる数値を X3 結果 JSON（result_incdino_ft_X3.json, perm_scratch_X3.json, tables_20260915.json）から一覧表示する（転記用）。
旧値（X2）も並べて表示するので、本文の置換対応表として使う。ホストの python3 で動く（numpy 不要）。"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "result_incdino_ft_X3.json")))
x2 = json.load(open("/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis/result_incdino_ft_20260806_X2.json"))
ps3 = json.load(open(os.path.join(HERE, "perm_scratch_X3.json")))["results"]
ps2 = json.load(open(os.path.join(HERE, "perm_scratch_20260910.json")))["results"]
rob2 = json.load(open(os.path.join(HERE, "robustness_000180_20260910.json")))
ind2 = open(os.path.join(HERE, "indicator_vs_condition_20260910.md")).read()
ex = r["extras"]
def f3(x): return f"{x:.3f}"
def p_(v): return "<0.001" if v < 0.001 else f"{v:.3f}"
rows = []
def add(name, old, new): rows.append((name, old, new))
for fs, lab in (("S_incdino11", "11変数"), ("S_inc6", "Inc6"), ("S_dino6", "DINO6"), ("S_sustech6", "DGV2")):
    for tgt in ("scratch_all", "ft_all"):
        add(f"{lab} {tgt} ρ", f3(x2["E_grid"][fs][tgt]["rho"]), f3(r["E_grid"][fs][tgt]["rho"]))
        add(f"{lab} {tgt} MAE pt", f"{x2['E_grid'][fs][tgt]['mae']*100:.2f}", f"{r['E_grid'][fs][tgt]['mae']*100:.2f}")
    add(f"{lab} scratch p", p_(ps2[fs]["p_perm"]) + f" ({ps2[fs]['count_ge']})", p_(ps3[fs]["p_perm"]) + f" ({ps3[fs]['count_ge']})")
for k in ("k0_S_incdino11_ft_all", "k2_S_inc6_ft_all", "k4_S_dino6_ft_all", "k6_S_incdino11_ft_tau0.05"):
    add(f"perm {k}", p_(x2["permutation"][k]["p_perm"]), p_(r["permutation"][k]["p_perm"]))
for fs, lab in (("S_incdino11", "11変数"), ("S_inc6", "Inc6"), ("S_dino6", "DINO6"), ("S_sustech6", "DGV2")):
    for tgt, key in (("scratch_all", "scratch_tau0.05"), ("ft_all", "ft_tau0.05")):
        a, b = x2["E_grid"][fs][tgt]["logreg"][key], r["E_grid"][fs][tgt]["logreg"][key]
        add(f"{lab} {tgt} Acc/F1", f"{a['acc']:.3f}/{a['f1']:.3f} tie {a['tie']}", f"{b['acc']:.3f}/{b['f1']:.3f} tie {b['tie']}")
add("majority baseline", "0.385 / 0.397", " / ".join(f"{ex['majority_baseline'][k]:.3f}" for k in ("scratch_tau0.05", "ft_tau0.05")))
for side in ("scratch", "ft"):
    for c in ("nm", "bg", "cl"):
        add(f"cond {side} {c} 汎用11 ρ/MAE", f"{x2['D_per_condition'][side][c]['spearman']:.3f}/{x2['D_per_condition'][side][c]['mae']*100:.2f}",
            f"{r['D_per_condition'][side][c]['spearman']:.3f}/{r['D_per_condition'][side][c]['mae']*100:.2f}")
for fs, tgt in (("S_inc6", "ft_all"), ("S_inc6", "scratch_all"), ("S_incdino11", "ft_all")):
    co2 = {k: v * 100 for k, v in x2["grid"][fs][tgt]["coef"].items()}; co3 = {k: v * 100 for k, v in r["grid"][fs][tgt]["coef"].items()}
    add(f"coef {fs} {tgt} FID_tt", f"{co2['FID_train_test_Inception']:+.2f}", f"{co3['FID_train_test_Inception']:+.2f}")
    if fs == "S_incdino11":
        add("coef 11var ft top5", " ".join(f"{k.replace('Inception_','Inc_').replace('FID_train_test_','FIDtt_')}{v:+.2f}" for k, v in sorted(co2.items(), key=lambda kv: -abs(kv[1]))[:5]),
            " ".join(f"{k.replace('Inception_','Inc_').replace('FID_train_test_','FIDtt_')}{v:+.2f}" for k, v in sorted(co3.items(), key=lambda kv: -abs(kv[1]))[:5]))
    if fs == "S_inc6" and tgt == "ft_all":
        rank = sorted(co3, key=lambda c: -abs(co3[c])).index("FID_train_test_Inception") + 1
        add("FID_tt rank in Inc6 ft", "1", str(rank))
for col in ("Inception_MSD", "DINO_MSD", "DeepGaitV2_MSD", "FID_train_test_Inception", "FID_train_test_DeepGaitV2"):
    c = ex["corr"][col]
    add(f"corr {col} (sCL,fCL,sNM,fNM)", "(see indicator_vs_condition_20260910.md)", f"{c['scratch_cl']:+.2f},{c['ft_cl']:+.2f},{c['scratch_nm']:+.2f},{c['ft_nm']:+.2f}")
for col in ("Inception_MSD", "DINO_MSD", "DeepGaitV2_MSD"):
    add(f"msd width {col}", {"Inception_MSD": "4.0%", "DINO_MSD": "26.4%", "DeepGaitV2_MSD": "53.2%"}[col], f"{ex['msd_width'][col]:.1%}")
for key in ("S_incdino11/scratch_all", "S_incdino11/ft_all", "S_inc6/scratch_all", "S_inc6/ft_all", "S_sustech6/scratch_all", "S_sustech6/ft_all"):
    a, b = rob2[key], ex["wo_000-180"][key]
    add(f"000-180 {key} pred%", f"{a.get('pred_000-180_pct', float('nan')):.1f}" if 'pred_000-180_pct' in a else "?", f"{b['pred_000-180_pct']:.1f} (true {b['true_000-180_pct']:.1f})")
    add(f"000-180 {key} share/MAE12/rho12", f"{a['err_share_000-180']:.1%}/{a['mae_12_wo_000-180']:.2f}/{a['rho_12_wo_000-180']:.3f}", f"{b['err_share_000-180']:.1%}/{b['mae_12']:.2f}/{b['rho_12']:.3f}")
for key in ("S_inc6/ft", "S_sustech6/ft", "S_incdino11/ft", "S_incdino11/scratch"):
    m = ex["mean3"][key]; add(f"mean3 {key} ρ all→mean3", "(0.885→0.786 / 0.742→0.791 / 0.890→0.879)", f"{m['rho_all']:.3f}→{m['rho_mean3']:.3f}")
add("scratch vs ft ρ", "0.973", f3(ex["scratch_vs_ft_rho"]))
w = max(len(n) for n, _, _ in rows)
for n, o, nw in rows:
    flag = "" if o == nw else "  <-- 変更"
    print(f"{n:<{w}}  旧 {o:<40} 新 {nw}{flag}")
