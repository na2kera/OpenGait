#!/usr/bin/env python3
"""X3 結果の確定処理（analysis_x3_20260914.py の後に実行。Codex レビュー 2026-09-14 の指摘を反映）。
 1. X3 指標のゲート強化: 差し替え 26 セルが一意・列は MSD 2 列のみ・MSD 以外の全列が X2 と完全一致・base_X2 md5 が現物と一致
 2. verdict を X2 本体と同じ関数で確定: coef_structure（11変数 ft 負・11変数 Δ 正・S_inc6 で絶対値順位 2 位以内）、matrix_row（解釈マトリクス行）
 3. FID_tt 主変数の再判定（X2 §4: scratch 優劣判定の重要度が高い方）を記録し、Inception でなければ警告
 4. verdict.finalized=True を書き、perm_scratch_X3.json の x2_result md5 を更新後の結果 JSON に合わせる
実行: docker run --rm -v /home/kera/OpenGait:/app/OpenGait -v /home/kera/worktrees:/home/kera/worktrees -w /app/OpenGait opengait:latest python dgv2_analysis/techrep_20260915/finalize_x3_20260914.py
"""
import hashlib, json, os, sys
WT = "/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis"; sys.path.insert(0, WT)
import analysis_incdino_ft_20260806 as A  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
X3M = os.path.join(HERE, "metrics_incdino_kera_X3.json"); X2M = os.path.join(WT, A.ARM_DEFAULTS["X2"]["incdino"])
RES = os.path.join(HERE, "result_incdino_ft_X3.json"); PERM = os.path.join(HERE, "perm_scratch_X3.json")
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()

x3, x2 = json.load(open(X3M)), json.load(open(X2M))
prov = x3["_provenance"]
cells = {(c["id"], c["column"]) for c in prov["replaced_cells"]}
assert len(cells) == 26 and {c for _, c in cells} == {"Inception_MSD", "DINO_MSD"}, "差し替えセルが 26 一意・2 列ではない"
assert prov["base_X2"]["md5"] == md5(X2M), "base_X2 md5 が現物の X2 指標と不一致"
for sid in map(str, range(1, 14)):
    for k, v in x2[sid].items():
        if k in ("Inception_MSD", "DINO_MSD"):
            assert x3[sid][k] != v, f"id{sid} {k} が差し替わっていない"
        else:
            assert x3[sid][k] == v, f"id{sid} {k} が X2 と異なる（MSD 以外は同一のはず）"
print("gate: X3 指標は MSD 2 列以外 X2 と完全一致、26 セル差し替え済み")

r = json.load(open(RES)); assert r["arm"] == "X3"
g = r["grid"]
ok, detail = A.coef_structure(g["S_incdino11"]["ft_all"]["coef"], g["S_incdino11"]["delta_pt"]["coef"], g["S_inc6"]["ft_all"]["coef"])
p_k0, p_k2 = r["permutation"]["k0_S_incdino11_ft_all"]["p_perm"], r["permutation"]["k2_S_inc6_ft_all"]["p_perm"]
rho_diff = g["S_inc6"]["ft_all"]["spearman"] - g["S_sustech6"]["ft_all"]["spearman"]
row = A.matrix_row(p_k0, p_k2, rho_diff, ok)
imp = r["logreg_grid"]["S_incdino11"]["scratch_tau0.05"].get("importance", {})
primary = "FID_train_test_Inception" if imp.get("FID_train_test_Inception", 0) >= imp.get("FID_train_test_DINO", 0) else "FID_train_test_DINO"
r["verdict"].update({"row": row, "coef_structure_match": bool(ok), "coef_detail": detail, "rho_diff_controlled": rho_diff,
                     "fid_tt_primary": primary, "primary_importance": {k: imp.get(k) for k in ("FID_train_test_Inception", "FID_train_test_DINO")},
                     "finalized": True, "note": "verdict は X2 本体の coef_structure / matrix_row で確定（finalize_x3_20260914.py）。C.c0-c9 / degradation_diagnosis / F_exploratory は原稿で使わないため未計算"})
if primary != "FID_train_test_Inception":
    print("WARNING: FID_tt 主変数が Inception ではない:", primary)
json.dump(r, open(RES, "w"), ensure_ascii=False, indent=1)
ps = json.load(open(PERM)); ps["inputs_md5"]["x2_result"] = {"path": RES, "md5": md5(RES)}; json.dump(ps, open(PERM, "w"), ensure_ascii=False, indent=1)
print(f"verdict: row={row} coef_structure_match={ok} detail={detail} rho_diff={rho_diff:+.3f} primary={primary}")
print("finalized; perm_scratch_X3 の x2_result md5 を更新")
