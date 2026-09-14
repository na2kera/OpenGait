#!/usr/bin/env python3
"""信学技報（2026-09-14 締切）用の表を JSON から機械転記する。手打ち禁止。

入力（すべて読み取りのみ。数値は一切再計算しない。系列数だけ npz から数える）:
  - dgv2_features_sustech/{subset}/train.npz            系列数・人数（採用版 = SUSTech1K 共通抽出器で抽出した学習データ）
  - dgv2_analysis/metrics_dgv2_sustech.json             scratch 精度（acc_nm/bg/cl/all）, Number of people
  - dgv2_analysis/metrics_ft_sustech_full_20260726.json  ft 精度, scratch_acc_all, delta_acc_all_pt
  - X2_RESULT (worktree)                                 フェーズ3 X2: E_grid / grid(coef) / permutation / logreg_grid / verdict / D_per_condition
  - dgv2_analysis/presentation_tables_20260726.json      S_sustech6 の条件別（phase1_per_condition / phase2_extra.per_condition_ridge）
  - techrep_20260915/perm_scratch_20260910.json          4特徴セット × scratch の並べ替え p（事後追加。無ければ停止）
  - dgv2_analysis/result_extra_analysis_20260726.json    S_sustech6 ft/Δ の並べ替え p

出力（techrep_20260915/tables/）:
  - table1_subsets.{tex,md}     表1: 構成（NM/BG/CL/計/人数）＋ scratch / ft / Δ
  - table_main_grid.{tex,md}    主表: 特徴空間 × 予測対象（scratch / ft）の ρ・MAE・並べ替え p（2×2。Δ は md のみ＝副次）
  - table_condition.{tex,md}    条件別（NM/BG/CL）: 汎用（S_incdino11）と歩容特化（S_sustech6）の ρ・MAE（scratch / ft）
  - table_pairwise.{tex,md}     優劣判定: 特徴空間 × 予測対象（scratch / ft）の Acc / Macro-F1（Δ は md のみ）
  - table_coef_ft.{tex,md}      ft 予測の標準化 Ridge 係数（pt / 1SD）
  - tables_20260915.json        全数値＋出典 md5

実行（host に numpy が無いので docker）:
  docker run --rm -v /home/kera/OpenGait:/app/OpenGait -v /home/kera/worktrees:/home/kera/worktrees \
      -w /app/OpenGait opengait:latest python dgv2_analysis/techrep_20260915/make_tables_20260915.py

設計上の注意:
  - 精度は % 表記（小数2桁）。MAE は pt（小数値×100、小数2桁）。ρ は小数3桁。
  - 主軸は 2×2（汎用 / 歩容特化 × scratch / ft）。Δ は主表・優劣表の tex から外し md に残す（paper-manuscript skill §3、2026-09-10）。
  - md には判断材料として追加列（単純平均精度、S_9var 行、Δ、Tie 数）を出す。
  - 入力間の整合性を assert で確認し、食い違いがあれば止まる（黙って進めない）。
"""
import collections
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "tables")

FEAT_DIR = os.path.join(REPO, "dgv2_features_sustech")
SUSTECH_JSON = os.path.join(REPO, "dgv2_analysis", "metrics_dgv2_sustech.json")
FT_JSON = os.path.join(REPO, "dgv2_analysis", "metrics_ft_sustech_full_20260726.json")
ARM = os.environ.get("ARM", "X2")   # X2（既定）/ X3（汎用側 MSD を式(1)版に差し替え、2026-09-14）
assert ARM in ("X2", "X3"), f"ARM は X2 か X3: {ARM}"
X2_RESULT = ("/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis/result_incdino_ft_20260806_X2.json" if ARM == "X2"
             else os.path.join(HERE, "result_incdino_ft_X3.json"))
PRES_JSON = os.path.join(REPO, "dgv2_analysis", "presentation_tables_20260726.json")   # S_sustech6 条件別
EXTRA_JSON = os.path.join(REPO, "dgv2_analysis", "result_extra_analysis_20260726.json")  # S_sustech6 ft/Δ の並べ替え p
PERM_SCRATCH = os.path.join(HERE, "perm_scratch_20260910.json" if ARM == "X2" else "perm_scratch_X3.json")   # 事後追加の並べ替え p
CONDS = [("nm", "NM"), ("bg", "BG"), ("cl", "CL")]

ID2SUBSET = {"1": "default", "2": "nm2", "3": "bg2", "4": "cl2", "5": "nm1-bg1",
             "6": "nm1-cl1", "7": "bg1-cl1", "8": "000-180", "9": "000-090",
             "10": "090-180", "11": "nm1-bg1-cl1", "12": "nm2-bg2-cl2", "13": "nm6"}
SUBSETS = list(ID2SUBSET.values())

# 主表の行（表示名, X2 E_grid のキー）。S_9var は md のみ
FEATURE_ROWS = [
    ("Inception（6 変数）", "S_inc6"),
    ("DINOv2（6 変数）", "S_dino6"),
    ("Inception+DINOv2（11 変数）", "S_incdino11"),
    ("DeepGaitV2（6 変数）", "S_sustech6"),
]
FEATURE_ROWS_MD_EXTRA = [("Inception+DINOv2 冗長除去（9変数, 感度）", "S_9var")]

# 係数表の変数順（S_inc6 / S_sustech6 の6変数を並置）
COEF_ROWS = [
    ("$\\log P$（人数）", "log_people", "log_people"),
    ("MSD（平均クラス内分散）", "Inception_MSD", "DeepGaitV2_MSD"),
    ("1NN", "Inception_1NN", "DeepGaitV2_1NN"),
    ("kNN ($k=5$)", "Inception_kNN", "DeepGaitV2_kNN"),
    ("pairFID（平均クラス間距離）", "Inception_FID", "DeepGaitV2_FID"),
    ("FID$_{\\mathrm{tt}}$（Train-Test 分布距離）", "FID_train_test_Inception", "FID_train_test_DeepGaitV2"),
]


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check(cond, msg):
    if not cond:
        sys.exit(f"[整合性エラー] {msg}")
    print(f"  ok: {msg}")


def seq_counts():
    """採用版の学習データ埋め込み npz から NM/BG/CL 系列数と人数を数える。"""
    out = {}
    for s in SUBSETS:
        z = np.load(os.path.join(FEAT_DIR, s, "train.npz"), allow_pickle=True)
        types = [str(x)[:2] for x in z["types"]]
        c = collections.Counter(types)
        check(set(c) <= {"nm", "bg", "cl"}, f"{s}: types は nm/bg/cl のみ ({dict(c)})")
        out[s] = {"people": int(len(np.unique(z["labels"]))),
                  "nm": c.get("nm", 0), "bg": c.get("bg", 0), "cl": c.get("cl", 0),
                  "total": int(len(types))}
    return out


def fmt_pct(x, nd=2):
    return f"{x * 100:.{nd}f}"


def fmt_signed(x, nd=2):
    return f"{x:+.{nd}f}"


def fmt_rho(x):
    return f"{x:.3f}"


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  wrote {os.path.relpath(path, REPO)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("--- 入力の読み込みと整合性確認 ---")
    sus = load_json(SUSTECH_JSON)
    ft = load_json(FT_JSON)
    x2 = load_json(X2_RESULT)
    counts = seq_counts()

    check(x2.get("arm") == ARM, f"結果 JSON の arm は {ARM} (実際: {x2.get('arm')})")
    print(f"  ARM={ARM}  result={X2_RESULT}")
    if ARM == "X3":
        x3m = os.path.join(HERE, "metrics_incdino_kera_X3.json")
        check(x2["inputs_md5"]["x3_metrics"] == md5(x3m), "X3 結果が記録した X3 指標の md5 が現物と一致")
        check(x2["inputs_md5"]["analysis_code"] == md5("/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis/analysis_incdino_ft_20260806.py"), "X3 結果が記録した分析コードの md5 が現物と一致")
        check(x2["inputs_md5"]["x2_result"] == md5("/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis/result_incdino_ft_20260806_X2.json"), "X3 結果が記録した X2 結果の md5 が現物と一致")
        check(set(x2["permutation"]) == {f"k{i}_" + n for i, n in enumerate(["S_incdino11_ft_all", "S_incdino11_delta_pt", "S_inc6_ft_all", "S_inc6_delta_pt", "S_dino6_ft_all", "S_dino6_delta_pt", "S_incdino11_ft_tau0.05", "S_incdino11_delta_tau5pt"])}, "X3 の並べ替え検定キー k0..k7 が X2 と同じ集合")
        check(all(x2["permutation"][k]["seed"] == 1000 + i for i, k in enumerate(sorted(x2["permutation"], key=lambda k: int(k[1:k.index("_")])))), "X3 の並べ替え seed が 1000+k")
        check(x2.get("verdict", {}).get("finalized") is True, "X3 の verdict が finalize_x3_20260914.py で確定済み")
    check(set(ft) == set(SUBSETS), "ft JSON は13サブセットちょうど")
    check(set(sus) == set(ID2SUBSET), "sustech JSON は id 1..13 ちょうど")

    # ---- 表1 ------------------------------------------------------------
    table1 = []
    for sid, s in ID2SUBSET.items():
        u, f_, c = sus[sid], ft[s], counts[s]
        check(u["subset_name"] == s, f"id{sid} の subset_name == {s}")
        check(u["Number of people"] == c["people"], f"{s}: 人数 json={u['Number of people']} npz={c['people']}")
        check(abs(u["acc_all"] - f_["scratch_acc_all"]) < 1e-9, f"{s}: scratch acc_all が sustech/ft JSON 間で一致")
        check(abs((f_["acc_all"] - f_["scratch_acc_all"]) * 100 - f_["delta_acc_all_pt"]) < 1e-6,
              f"{s}: Δ == (ft − scratch)×100")
        check(f_["iter"] == 30000 and f_["seed"] == 0, f"{s}: ft は iter 30000 / seed 0")
        table1.append({
            "id": int(sid), "subset": s, "people": c["people"],
            "nm": c["nm"], "bg": c["bg"], "cl": c["cl"], "total": c["total"],
            "scratch_nm": u["acc_nm"], "scratch_bg": u["acc_bg"], "scratch_cl": u["acc_cl"],
            "scratch_all": u["acc_all"],
            "scratch_mean3": (u["acc_nm"] + u["acc_bg"] + u["acc_cl"]) / 3,   # 中村論文 表1 の定義（単純平均）
            "ft_nm": f_["acc_nm"], "ft_bg": f_["acc_bg"], "ft_cl": f_["acc_cl"],
            "ft_all": f_["acc_all"],
            "ft_mean3": (f_["acc_nm"] + f_["acc_bg"] + f_["acc_cl"]) / 3,
            "delta_pt": f_["delta_acc_all_pt"],
        })
    # 中村論文 表1（refs/*.txt）と同じ系列数か: 視野角3本以外は一致するはず（71人→75人で変わるのは 8/9/10 のみ）
    nakamura_total = {"default": 1640, "nm2": 1642, "bg2": 1642, "cl2": 1642, "nm1-bg1": 1641,
                      "nm1-cl1": 1643, "bg1-cl1": 1644, "000-180": 1408, "000-090": 1410,
                      "090-180": 1418, "nm1-bg1-cl1": 1643, "nm2-bg2-cl2": 1644, "nm6": 1644}
    for r in table1:
        if r["subset"] in ("000-180", "000-090", "090-180"):
            check(r["total"] > nakamura_total[r["subset"]], f"{r['subset']}: 75人版の系列数 {r['total']} > 中村表1(71人版) {nakamura_total[r['subset']]}")
        else:
            check(r["total"] == nakamura_total[r["subset"]], f"{r['subset']}: 系列数 {r['total']} が中村表1と一致")

    # ---- 主表・優劣・係数（X2 E_grid / grid / logreg_grid / permutation） ----
    eg, grid, perm = x2["E_grid"], x2["grid"], x2["permutation"]
    # E_grid の ρ/MAE は grid と同値であることを確認（S_sustech6 は既報 JSON 由来なので grid には無い）
    for _, key in FEATURE_ROWS + FEATURE_ROWS_MD_EXTRA:
        for tgt in ("scratch_all", "ft_all", "delta_pt"):
            if key in grid and tgt in grid[key]:
                check(abs(eg[key][tgt]["rho"] - grid[key][tgt]["spearman"]) < 1e-12 and
                      abs(eg[key][tgt]["mae"] - grid[key][tgt]["mae"]) < 1e-12,
                      f"E_grid と grid の一致: {key}×{tgt}")
    # 既報（7/27 予稿）の S_sustech6 値と一致
    check(abs(eg["S_sustech6"]["ft_all"]["rho"] - 0.7417582417582418) < 1e-12 and
          abs(eg["S_sustech6"]["scratch_all"]["rho"] - 0.8076923076923077) < 1e-12 and
          abs(eg["S_sustech6"]["delta_pt"]["rho"] - 0.8571428571428571) < 1e-12,
          "S_sustech6 の ρ (0.808 / 0.742 / 0.857) が 7/27 予稿と一致")
    perm_map = {  # (feature, target) -> permutation key
        ("S_incdino11", "ft_all"): "k0_S_incdino11_ft_all", ("S_incdino11", "delta_pt"): "k1_S_incdino11_delta_pt",
        ("S_inc6", "ft_all"): "k2_S_inc6_ft_all", ("S_inc6", "delta_pt"): "k3_S_inc6_delta_pt",
        ("S_dino6", "ft_all"): "k4_S_dino6_ft_all", ("S_dino6", "delta_pt"): "k5_S_dino6_delta_pt",
    }
    # 既報の S_sustech6 並べ替え p は出典 JSON から読む（scratch は 7/27 予稿 表1 の転記 0.001 で生ファイル無し → perm_scratch を使う）
    extra = load_json(EXTRA_JSON)
    perm_sustech = {"ft_all": extra["permutation_ridge"]["ft_all"]["p_perm"],
                    "delta_pt": extra["permutation_ridge"]["delta_all_pt"]["p_perm"]}
    check(abs(perm_sustech["ft_all"] - 0.003996003996003996) < 1e-12 and abs(perm_sustech["delta_pt"] - 0.000999000999000999) < 1e-12,
          "S_sustech6 ft/Δ の並べ替え p（0.004 / <0.001）が 7/27 予稿と一致")
    for (fk, tk), pk in perm_map.items():
        check(abs(perm[pk]["observed_spearman"] - eg[fk][tk]["rho"]) < 1e-12, f"並べ替え検定 {pk} の観測 ρ が E_grid と一致")
    # scratch の並べ替え p（事後追加）。4 特徴セット揃っていなければ止まる（S_sustech6 が欠けると旧転記値に黙って戻るのを防ぐ）
    check(os.path.exists(PERM_SCRATCH), "perm_scratch_20260910.json がある（無ければ perm_scratch_20260910.py を docker で実行）")
    ps = load_json(PERM_SCRATCH)
    check(ps["arm"] == ARM and ps["target"] == "scratch_all" and ps["n_perm"] == 1000, f"perm_scratch は {ARM} × scratch_all, N=1000")
    check(set(ps["results"]) == {"S_incdino11", "S_inc6", "S_dino6", "S_sustech6"}, "perm_scratch は 4 特徴セット揃っている")
    perm_scratch = {}
    for i, fk in enumerate(("S_incdino11", "S_inc6", "S_dino6", "S_sustech6")):
        r = ps["results"][fk]
        check(r["seed"] == 1010 + i, f"perm_scratch {fk} の seed が {1010 + i}")
        check(0 <= r["count_ge"] <= 1000 and abs(r["p_perm"] - (r["count_ge"] + 1) / 1001) < 1e-12,
              f"perm_scratch {fk}: p = (count_ge+1)/(N+1) = ({r['count_ge']}+1)/1001")
        check(abs(r["observed_spearman"] - eg[fk]["scratch_all"]["rho"]) < 1e-9, f"perm_scratch {fk} の観測 ρ が E_grid と一致")
        perm_scratch[fk] = r["p_perm"]
    perm_scratch_meta = {fk: {"seed": ps["results"][fk]["seed"], "count_ge": ps["results"][fk]["count_ge"]} for fk in perm_scratch}
    # 検定が読んだ入力（worktree 側のコピー）と本スクリプトが読む入力・X2 結果が同一内容であることを md5 で照合
    pim = ps["inputs_md5"]
    check(pim["sustech"]["md5"] == md5(SUSTECH_JSON) and pim["ft"]["md5"] == md5(FT_JSON),
          "perm_scratch の入力（sustech / ft）が本スクリプトの入力と同一内容")
    check(pim["x2_result"]["md5"] == md5(X2_RESULT), "perm_scratch が参照した X2 結果が本スクリプトの X2_RESULT と同一内容")

    main_grid = []
    for label, key in FEATURE_ROWS + FEATURE_ROWS_MD_EXTRA:
        row = {"label": label, "key": key}
        for tgt in ("scratch_all", "ft_all", "delta_pt"):
            e = eg[key][tgt]
            row[tgt] = {"rho": e["rho"],
                        "mae_pt": e["mae"] if tgt == "delta_pt" else e["mae"] * 100}
            if (key, tgt) in perm_map:
                row[tgt]["p_perm"] = perm[perm_map[(key, tgt)]]["p_perm"]
            elif tgt == "scratch_all" and key in perm_scratch:
                # scratch は 9/10 の再計算（生 JSON あり）を優先。S_sustech6 の 7/27 転記値 0.001 は生ファイルが無い
                row[tgt]["p_perm"] = perm_scratch[key]
                row[tgt]["p_perm_posthoc"] = True
            elif key == "S_sustech6" and tgt in perm_sustech:
                row[tgt]["p_perm"] = perm_sustech[tgt]
            lr = e.get("logreg", {})
            if tgt == "scratch_all":
                row[tgt]["logreg"] = lr.get("scratch_tau0.05")
            elif tgt == "ft_all":
                row[tgt]["logreg"] = lr.get("ft_tau0.05")
            else:
                row[tgt]["logreg"] = lr.get("delta_tau5pt")
        main_grid.append(row)

    coef = {
        "S_inc6": {k: v * 100 for k, v in grid["S_inc6"]["ft_all"]["coef"].items()},
        "S_sustech6": {k: v * 100 for k, v in x2["A_sustech6_ft_reproduced"]["coef"].items()}
        if "coef" in x2.get("A_sustech6_ft_reproduced", {}) else None,
        "S_incdino11": {k: v * 100 for k, v in grid["S_incdino11"]["ft_all"]["coef"].items()},
    }
    if coef["S_sustech6"] is None:
        ftres = load_json(os.path.join(REPO, "dgv2_analysis", "result_ft_analysis_20260726.json"))
        coef["S_sustech6"] = {k: v * 100 for k, v in ftres["ridge"]["ft_all"]["coef"].items()}
    # 7/27 予稿: FID_tt 係数 ft −5.17
    check(abs(coef["S_sustech6"]["FID_train_test_DeepGaitV2"] - (-5.168715277015569)) < 1e-9,
          "S_sustech6 ft の FID_tt 係数 −5.17pt が 7/27 予稿と一致")
    if ARM == "X2":
        check(x2["verdict"]["coef_detail"]["rank_in_S_inc6"] == 1, "S_inc6 で FID_tt^inc が最大絶対値（verdict）")
    else:
        print(f"  note: S_inc6 ft の FID_tt^inc 係数の絶対値順位 = {x2['verdict']['coef_detail']['rank_in_S_inc6']}（X3）")

    # ---- 出力: 表1 -------------------------------------------------------
    tex = [
        "%% 自動生成: make_tables_20260915.py（手で編集しない）",
        "\\begin{table*}[tb]",
        "\\caption{CASIA-B サブセットの構成（学習系列数・人数）と精度（\\%，4.\\,1 節の定義）．$\\Delta$ はファインチューニング後 $-$ スクラッチ学習（pt）．}",
        "\\label{tab:subsets}",
        "\\begin{center}",
        "\\begin{tabular}{l|rrrr|r|rrr}",
        "\\Hline",
        "サブセット & NM & BG & CL & 計 & 人数 & scratch & ft & $\\Delta$ \\\\",
        "\\hline",
    ]
    for r in table1:
        tex.append(f"{r['subset']} & {r['nm']} & {r['bg']} & {r['cl']} & {r['total']} & {r['people']} & "
                   f"{fmt_pct(r['scratch_all'])} & {fmt_pct(r['ft_all'])} & ${fmt_signed(r['delta_pt'])}$ \\\\")
    tex += ["\\Hline", "\\end{tabular}", "\\end{center}", "\\end{table*}", ""]
    write(os.path.join(OUT, "table1_subsets.tex"), "\n".join(tex))

    md = ["# 表1: サブセット構成と精度（自動生成）", "",
          "系列数・人数は `dgv2_features_sustech/{subset}/train.npz`（採用版・75人版）から数えた実測値。",
          "精度の `all` は固定重み 0.6/0.2/0.2 の平均（本研究の acc_all。系列数比 3222/1078/1076 の丸め）、`mean3` は NM/BG/CL 単純平均（中村論文 表1 の定義）。決定B-4 の判断材料。", "",
          "| id | サブセット | NM | BG | CL | 計 | 人数 | scratch all | scratch mean3 | ft all | ft mean3 | Δ(pt, all) | Δ(pt, mean3) |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table1:
        md.append(f"| {r['id']} | {r['subset']} | {r['nm']} | {r['bg']} | {r['cl']} | {r['total']} | {r['people']} | "
                  f"{fmt_pct(r['scratch_all'])} | {fmt_pct(r['scratch_mean3'])} | {fmt_pct(r['ft_all'])} | {fmt_pct(r['ft_mean3'])} | "
                  f"{fmt_signed(r['delta_pt'])} | {fmt_signed((r['ft_mean3'] - r['scratch_mean3']) * 100)} |")
    md += ["", "条件別（%）:", "",
           "| サブセット | scratch NM | BG | CL | ft NM | BG | CL |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in table1:
        md.append(f"| {r['subset']} | {fmt_pct(r['scratch_nm'])} | {fmt_pct(r['scratch_bg'])} | {fmt_pct(r['scratch_cl'])} | "
                  f"{fmt_pct(r['ft_nm'])} | {fmt_pct(r['ft_bg'])} | {fmt_pct(r['ft_cl'])} |")
    write(os.path.join(OUT, "table1_subsets.md"), "\n".join(md) + "\n")

    # ---- 出力: 主表（2×2: 特徴空間 × scratch/ft。Δ は tex に載せない） ----
    tex = [
        "%% 自動生成: make_tables_20260915.py（手で編集しない）",
        "\\begin{table*}[tb]",
        "\\caption{学習前予測の性能（13 サブセットの LODO）．$\\rho$ は Spearman 順位相関，MAE の単位は pt，$p$ は並べ替え検定（1,000 回，片側，$p=(r+1)/(N+1)$．$<$0.001 は超過 0 回）．$\\dagger$ はスクラッチ学習について事後に追加した検定．}",
        "\\label{tab:main}",
        "\\begin{center}",
        "\\begin{tabular}{l|ccc|ccc}",
        "\\Hline",
        " & \\multicolumn{3}{c|}{スクラッチ} & \\multicolumn{3}{c}{ファインチューニング} \\\\",
        "特徴空間 & $\\rho$ & MAE & $p$ & $\\rho$ & MAE & $p$ \\\\",
        "\\hline",
    ]
    for row in main_grid:
        if row["key"] == "S_9var":
            continue
        cells = [row["label"]]
        for tgt in ("scratch_all", "ft_all"):
            e = row[tgt]
            pc = fmt_p(e["p_perm"]) if "p_perm" in e else "---"
            if e.get("p_perm_posthoc"):
                pc += "$^{\\dagger}$"
            cells += [fmt_rho(e["rho"]), f"{e['mae_pt']:.2f}", pc]
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\Hline", "\\end{tabular}", "\\end{center}", "\\end{table*}", ""]
    write(os.path.join(OUT, "table_main_grid.tex"), "\n".join(tex))

    md = ["# 主表: 特徴空間 × 予測対象（自動生成）", "",
          f"出典: `{os.path.basename(X2_RESULT)}` の E_grid（S_sustech6 は既報 JSON からの転記）。MAE は pt。p は並べ替え検定（1000回、片側）。",
          "tex は scratch / ft の 2×2 のみ。Δ は副次として md にだけ残す（paper-manuscript skill §3）。", "",
          "| 特徴セット | scratch ρ | scratch MAE | p | ft ρ | ft MAE | p | Δ ρ（副次） | Δ MAE | p |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in main_grid:
        cells = [row["label"] + f" `{row['key']}`"]
        for tgt in ("scratch_all", "ft_all", "delta_pt"):
            e = row[tgt]
            pc = fmt_p(e["p_perm"]) if "p_perm" in e else "—"
            if e.get("p_perm_posthoc"):
                pc += "†"
            cells += [fmt_rho(e["rho"]), f"{e['mae_pt']:.2f}", pc]
        md.append("| " + " | ".join(cells) + " |")
    md += ["", "† 事後追加の並べ替え検定（`perm_scratch_20260910.py`、seed 1010–1013、事前登録外）。"
           + " ".join(f"{k}: seed {v['seed']}, 超過 {v['count_ge']}/1000, p=({v['count_ge']}+1)/1001" for k, v in perm_scratch_meta.items()) + "。",
           "中村論文（Inception/DINO 11変数＋MSD_raw, scratch, 目的変数は NM/BG/CL 単純平均）: ρ 0.8352, MAE 0.1146, RMSE 0.2436。目的変数の定義が違うので同じ表には並べない。",
           "S_sustech6 の ft/Δ の p は result_extra_analysis_20260726.json から読む。scratch の p は本稿では事後再計算値 0.003（seed 1013、超過 2/1000、補正後 p=3/1001）を採用。既報（7/27 予稿 表1）は 0.001 だが旧検定の seed・超過回数は確認できないので、差の原因は断定しない。S_9var は並べ替え検定を回していないので —。"]
    write(os.path.join(OUT, "table_main_grid.md"), "\n".join(md) + "\n")

    # ---- 出力: 条件別（NM/BG/CL）: 汎用 S_incdino11（X2 D_per_condition）vs 歩容特化 S_sustech6（7/27 JSON） ----
    pres = load_json(PRES_JSON)
    dpc = x2["D_per_condition"]
    p1 = pres["phase1_per_condition"]                 # scratch, S_sustech6
    p2 = pres["phase2_extra"]["per_condition_ridge"]  # ft_{cond}, S_sustech6
    check(abs(p1["all"]["spearman"] - eg["S_sustech6"]["scratch_all"]["rho"]) < 1e-12,
          "phase1_per_condition の all が S_sustech6 scratch ρ と一致（同じ特徴セット・同じ LODO）")
    check(abs(pres["phase2_main"]["ridge"]["ft_all"]["spearman"] - eg["S_sustech6"]["ft_all"]["rho"]) < 1e-12,
          "phase2_main の ft_all が S_sustech6 ft ρ と一致")
    condition = []
    for tgt_key, tgt_label in (("scratch", "スクラッチ"), ("ft", "ファインチューニング")):
        for ck, cl in CONDS:
            g = dpc[tgt_key][ck]
            su = p1[ck] if tgt_key == "scratch" else p2[f"ft_{ck}"]
            condition.append({"target": tgt_key, "target_label": tgt_label, "cond": cl,
                              "S_incdino11": {"rho": g["spearman"], "mae_pt": g["mae"] * 100},
                              "S_sustech6": {"rho": su["spearman"], "mae_pt": su["mae"] * 100}})
    tex = [
        "%% 自動生成: make_tables_20260915.py（手で編集しない）",
        "\\begin{table}[tb]",
        "\\caption{条件別（NM/BG/CL Rank-1）の学習前予測性能．汎用は Inception+DINOv2（11 変数），歩容特化は DeepGaitV2（6 変数）．MAE の単位は pt．13 サブセットの記述統計で検定は行っていない．}",
        "\\label{tab:condition}",
        "\\begin{center}",
        "\\begin{tabular}{ll|cc|cc}",
        "\\Hline",
        " & & \\multicolumn{2}{c|}{汎用} & \\multicolumn{2}{c}{歩容特化} \\\\",
        "予測対象 & 条件 & $\\rho$ & MAE & $\\rho$ & MAE \\\\",
        "\\hline",
    ]
    for i, r in enumerate(condition):
        lead = r["target_label"] if r["cond"] == "NM" else ""
        if i == 3:
            tex.append("\\hline")
        tex.append(f"{lead} & {r['cond']} & {fmt_rho(r['S_incdino11']['rho'])} & {r['S_incdino11']['mae_pt']:.2f} & "
                   f"{fmt_rho(r['S_sustech6']['rho'])} & {r['S_sustech6']['mae_pt']:.2f} \\\\")
    tex += ["\\Hline", "\\end{tabular}", "\\end{center}", "\\end{table}", ""]
    write(os.path.join(OUT, "table_condition.tex"), "\n".join(tex))

    md = ["# 条件別（NM/BG/CL）の予測性能（自動生成）", "",
          f"汎用 = S_incdino11（`{os.path.basename(X2_RESULT)}` D_per_condition）、歩容特化 = S_sustech6（`presentation_tables_20260726.json` phase1_per_condition / phase2_extra）。",
          "MAE は pt。n=13 の記述統計で並べ替え検定は無し。ρ の大小は「傾向」として書く。", "",
          "| 予測対象 | 条件 | 汎用 ρ | 汎用 MAE | 歩容特化 ρ | 歩容特化 MAE |", "|---|---|---:|---:|---:|---:|"]
    for r in condition:
        md.append(f"| {r['target']} | {r['cond']} | {fmt_rho(r['S_incdino11']['rho'])} | {r['S_incdino11']['mae_pt']:.2f} | "
                  f"{fmt_rho(r['S_sustech6']['rho'])} | {r['S_sustech6']['mae_pt']:.2f} |")
    md += ["", "注意: scratch の NM は歩容特化が 0.17 高い。「NM・BG は同じ」ではなく「NM・BG では大差がなく CL で差が開く」と書く（skill §3.3）。"]
    write(os.path.join(OUT, "table_condition.md"), "\n".join(md) + "\n")

    # ---- 出力: 優劣判定 ---------------------------------------------------
    tex = [
        "%% 自動生成: make_tables_20260915.py（手で編集しない）",
        "\\begin{table}[tb]",
        "\\caption{データセット優劣判定の性能（156 順序対，leave-one-pair-out，$\\tau=0.05$）．M-F1 は Macro-F1．}",
        "\\label{tab:pairwise}",
        "\\begin{center}",
        "\\begin{tabular}{l|cc|cc}",
        "\\Hline",
        " & \\multicolumn{2}{c|}{スクラッチ} & \\multicolumn{2}{c}{ファインチューニング} \\\\",
        "特徴空間 & Acc & M-F1 & Acc & M-F1 \\\\",
        "\\hline",
    ]
    for row in main_grid:
        if row["key"] == "S_9var":
            continue
        cells = [row["label"]]
        for tgt in ("scratch_all", "ft_all"):
            lr = row[tgt]["logreg"]
            cells += [f"{lr['acc']:.3f}", f"{lr['f1']:.3f}"]
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\Hline", "\\end{tabular}", "\\end{center}", "\\end{table}", ""]
    write(os.path.join(OUT, "table_pairwise.tex"), "\n".join(tex))

    md = ["# 優劣判定: 特徴空間 × 予測対象（自動生成）", "",
          "Tie 数は τ で決まりターゲット内で共通（scratch 36/156, ft 62/156, Δ 72/156）。列間（scratch vs ft）の Acc 比較はレンジ圧縮で Tie 率が違うため行わない。", "",
          "| 特徴セット | scratch Acc | F1 | tie | ft Acc | F1 | tie | Δ(τ5pt) Acc | F1 | tie |",
          "|---|---:|---:|---|---:|---:|---|---:|---:|---|"]
    for row in main_grid:
        cells = [row["label"] + f" `{row['key']}`"]
        for tgt in ("scratch_all", "ft_all", "delta_pt"):
            lr = row[tgt]["logreg"]
            cells += [f"{lr['acc']:.4f}", f"{lr['f1']:.4f}", lr["tie"]]
        md.append("| " + " | ".join(cells) + " |")
    k6, k7 = perm["k6_S_incdino11_ft_tau0.05"], perm["k7_S_incdino11_delta_tau5pt"]
    md += ["", f"並べ替え検定（200回）: S_incdino11×ft Acc {k6['observed_accuracy']:.4f} p={k6['p_perm']:.4f}, "
                f"S_incdino11×Δ Acc {k7['observed_accuracy']:.4f} p={k7['p_perm']:.4f}。",
           "中村論文（scratch, τ=0.05）: Acc 0.7949, Macro-F1 0.7296。"]
    write(os.path.join(OUT, "table_pairwise.md"), "\n".join(md) + "\n")

    # ---- 出力: 係数表 -----------------------------------------------------
    tex = [
        "%% 自動生成: make_tables_20260915.py（手で編集しない）",
        "\\begin{table}[tb]",
        "\\caption{ファインチューニング後精度に対する標準化リッジ係数（LODO 13 回の平均，pt/SD）．}",
        "\\label{tab:coef}",
        "\\begin{center}",
        "\\begin{tabular}{l|rr}",
        "\\Hline",
        "指標 & Inception & DeepGaitV2 \\\\",
        "\\hline",
    ]
    for label, k_inc, k_sus in COEF_ROWS:
        tex.append(f"{label} & ${fmt_signed(coef['S_inc6'][k_inc])}$ & ${fmt_signed(coef['S_sustech6'][k_sus])}$ \\\\")
    tex += ["\\Hline", "\\end{tabular}", "\\end{center}", "\\end{table}", ""]
    write(os.path.join(OUT, "table_coef_ft.tex"), "\n".join(tex))

    md = ["# ft 予測の標準化 Ridge 係数（pt / 1SD, 13 fold 平均, 自動生成）", "",
          "| 指標 | S_inc6 | S_sustech6 |", "|---|---:|---:|"]
    for label, k_inc, k_sus in COEF_ROWS:
        md.append(f"| {label} | {fmt_signed(coef['S_inc6'][k_inc])} | {fmt_signed(coef['S_sustech6'][k_sus])} |")
    md += ["", "S_incdino11（11変数）の ft 係数（絶対値降順）:", "", "| 変数 | 係数 |", "|---|---:|"]
    for k, v in sorted(coef["S_incdino11"].items(), key=lambda kv: -abs(kv[1])):
        md.append(f"| {k} | {fmt_signed(v)} |")
    md += ["", "注意（0907.md セルフレビュー）: FID_tt が最大絶対値なのは S_inc6 と S_sustech6。S_incdino11 では Inception_kNN / Inception_MSD の方が大きい。"]
    write(os.path.join(OUT, "table_coef_ft.md"), "\n".join(md) + "\n")

    # ---- JSON（全数値＋出典） -----------------------------------------------
    prov = {
        "generated_by": os.path.relpath(__file__, REPO),
        "arm": ARM,
        "inputs": {
            "metrics_dgv2_sustech.json": md5(SUSTECH_JSON),
            "metrics_ft_sustech_full_20260726.json": md5(FT_JSON),
            os.path.basename(X2_RESULT): md5(X2_RESULT),
            "presentation_tables_20260726.json": md5(PRES_JSON),
            "result_extra_analysis_20260726.json": md5(EXTRA_JSON),
            os.path.basename(PERM_SCRATCH): md5(PERM_SCRATCH),
            "RESULT_path": X2_RESULT,
            "seq_counts_from": "dgv2_features_sustech/{subset}/train.npz",
        },
        "result_inputs": x2.get("inputs"),      # 結果 JSON（X2 または X3）が参照した入力ファイル名（md5 ではない）
        "perm_scratch_inputs_md5": ps.get("inputs_md5"),
        "verdict": {k: x2["verdict"][k] for k in ("row", "p_k0", "p_k2", "rho_inc6_ft", "rho_sustech6_ft",
                                                  "rho_incdino11_ft", "rho_diff_controlled", "coef_structure_match",
                                                  "delta_mae_S_inc6_pt", "practical_criterion_met")},
    }
    write(os.path.join(OUT, "tables_20260915.json"),
          json.dumps({"provenance": prov, "table1": table1, "main_grid": main_grid, "condition": condition,
                      "perm_scratch_posthoc": perm_scratch, "perm_scratch_meta": perm_scratch_meta, "coef_ft_pt": coef},
                     ensure_ascii=False, indent=1))
    print("--- 完了 ---")


if __name__ == "__main__":
    main()
