#!/usr/bin/env python3
"""発表用ページ（Notion「発表で使用するデータ」）の全表を1か所で再生成する (2026-07-26)。

目的: Notionページに載っている数値が、すべてこのスクリプトの出力から辿れる状態にする。
      これまで一部の値（条件別Δ分解、ft−zero-shot、LODO予測値、サブセット構成）は
      ターミナルで都度計算してページに直接書いており、保存物がなかった。

方針:
  - 既存の結果JSONがあるものは**再計算せず読み込む**（公開済みの値との不一致を防ぐ）
  - 保存物がなかったものだけここで計算する
  - 出典が文献値のものは literature としてハードコードし出典を明記する

出力:
  - presentation_tables_20260726.json : 機械可読（全表）
  - presentation_tables_20260726.txt  : 人が読む用（整形済み。Notionと突き合わせる）

実行: docker exec opengait_container python /app/OpenGait/dgv2_analysis/export_presentation_tables_20260726.py
"""
import io
import json
import math
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(os.path.dirname(HERE), "dgv2_features_sustech")

ID2SUBSET = {"1": "default", "2": "nm2", "3": "bg2", "4": "cl2", "5": "nm1-bg1",
             "6": "nm1-cl1", "7": "bg1-cl1", "8": "000-180", "9": "000-090",
             "10": "090-180", "11": "nm1-bg1-cl1", "12": "nm2-bg2-cl2", "13": "nm6"}
SUBSETS = [ID2SUBSET[str(i)] for i in range(1, 14)]
S2ID = {v: k for k, v in ID2SUBSET.items()}

F_DGV2 = ["log_people", "DeepGaitV2_MSD", "DeepGaitV2_1NN", "DeepGaitV2_kNN",
          "DeepGaitV2_FID", "FID_train_test_DeepGaitV2"]
F_INCDINO = ["log_people", "Inception_MSD", "Inception_1NN", "Inception_kNN", "Inception_FID",
             "DINO_MSD", "DINO_1NN", "DINO_kNN", "DINO_FID",
             "FID_train_test_Inception", "FID_train_test_DINO"]

# 文献値（中村, IEICE技報の報告値）。出典: dgv2_analysis/README.md:69
PAPER = {"ridge_spearman": 0.8352, "logreg_accuracy": 0.7949, "logreg_macro_f1": 0.7296}

OUT = io.StringIO()


def emit(s=""):
    print(s)
    OUT.write(s + "\n")


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def table(rows, headers, align=None):
    """整形テキスト表。rows: list of list(str)"""
    cols = len(headers)
    w = [max(len(str(headers[i])), max((len(str(r[i])) for r in rows), default=0)) for i in range(cols)]
    align = align or ["<"] + [">"] * (cols - 1)
    line = "  ".join(f"{str(headers[i]):{align[i]}{w[i]}}" for i in range(cols))
    emit(line)
    emit("  ".join("-" * w[i] for i in range(cols)))
    for r in rows:
        emit("  ".join(f"{str(r[i]):{align[i]}{w[i]}}" for i in range(cols)))


def lodo_predictions(X, y):
    pred = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        mdl = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1.0))])
        mdl.fit(X[m], y[m])
        pred.append(float(mdl.predict(X[~m])[0]))
    return np.array(pred)


def main():
    out = {"_note": "Notion『発表で使用するデータ』の全表の生成元。"
                    "既存結果JSONは読み込み、保存物のなかった値のみ再計算している。"}

    scratch = load("metrics_dgv2_sustech.json")
    ft = load("metrics_ft_sustech_full_20260726.json")
    ryu = load("metrics_ryu.json")
    feat_meta = json.load(open(os.path.join(FEAT_DIR, "dgv2_metrics_sustech.json"), encoding="utf-8"))
    r_ridge_s = load("result_ridge_sustech.json")
    r_logreg_s = load("result_logreg_sustech.json")
    r_ridge_i = load("result_ridge_incdino.json")
    r_logreg_i = load("result_logreg_incdino.json")
    r_ft = load("result_ft_analysis_20260726.json")
    r_extra = load("result_extra_analysis_20260726.json")
    r_sens = load("ft_sustech_sensitivity_grid_20260726.json")
    r_comp = load("result_comparability_20260726.json")

    emit("=" * 100)
    emit("発表用データ 全表エクスポート (2026-07-26)")
    emit("=" * 100)

    # ---------- 表1: サブセット定義（埋め込みnpzのtypes/viewsから復元＝指標計算に入った実体） ----------
    emit("\n\n【表1】13サブセットの定義と規模  ※ 学習埋め込み npz の実体から復元")
    comp, rows = {}, []
    for sub in SUBSETS:
        d = np.load(os.path.join(FEAT_DIR, sub, "train.npz"))
        types = sorted(set(str(t) for t in d["types"]))
        views = sorted(set(str(v) for v in d["views"]))
        labels = d["labels"]
        n, p = len(labels), len(set(labels.tolist()))
        comp[sub] = {"types": types, "views": views, "n_people": p,
                     "n_train_seqs": int(n), "seqs_per_person": round(n / p, 1)}
        rows.append([sub, ",".join(t.replace("-0", "") for t in types),
                     f"{len(views)}視角" if len(views) > 2 else ",".join(views),
                     p, n, f"{n/p:.0f}"])
    out["subset_composition"] = comp
    table(rows, ["subset", "学習系列(条件)", "視野角", "人数", "学習系列数", "系列/人"])
    emit(f"  → 総サンプル数は設計上ほぼ一定（{min(c['n_train_seqs'] for c in comp.values())}"
         f"〜{max(c['n_train_seqs'] for c in comp.values())}）、系列/人は"
         f"{min(c['seqs_per_person'] for c in comp.values()):.0f}〜"
         f"{max(c['seqs_per_person'] for c in comp.values()):.0f}と5倍の開き")

    # ---------- 表2: 説明変数 ----------
    emit("\n\n【表2】説明変数の値（出典: metrics_dgv2_sustech.json）")
    rows, feats = [], {}
    for sub in SUBSETS:
        s = scratch[S2ID[sub]]
        feats[sub] = {"log_people": math.log(s["Number of people"]),
                      **{c: s[c] for c in F_DGV2[1:]}}
        rows.append([sub, s["Number of people"], f"{s['DeepGaitV2_MSD']:.2f}",
                     f"{s['DeepGaitV2_1NN']:.3f}", f"{s['DeepGaitV2_kNN']:.3f}",
                     f"{s['DeepGaitV2_FID']:.2f}", f"{s['FID_train_test_DeepGaitV2']:.2f}"])
    out["explanatory_variables"] = feats
    table(rows, ["subset", "人数", "MSD", "1NN", "kNN", "FID", "FID_tt"])
    v = [scratch[S2ID[s]]["FID_train_test_DeepGaitV2"] for s in SUBSETS]
    sd = float(np.std(v, ddof=1))
    out["fid_tt_stats"] = {"mean": float(np.mean(v)), "sd_sample": sd,
                           "min": min(v), "max": max(v), "range_in_sd": (max(v) - min(v)) / sd}
    emit(f"  → FID_tt: 平均 {np.mean(v):.2f} / 標準偏差 {sd:.2f} / 範囲 {min(v):.2f}〜{max(v):.2f}"
         f"（{(max(v)-min(v))/sd:.1f}SD分）")
    emit(f"     標準化係数 −0.097 の解釈: 1SD(={sd:.2f})あたり −9.7pt、"
         f"実測レンジ全体で約 {-0.097*100*(max(v)-min(v))/sd:.0f}pt")

    # ---------- 表3: scratch精度 ----------
    emit("\n\n【表3】scratch精度（出典: metrics_dgv2_sustech.json）")
    rows = []
    for sub in SUBSETS:
        s = scratch[S2ID[sub]]
        rows.append([sub] + [f"{s[f'acc_{c}']*100:.2f}" for c in ("nm", "bg", "cl", "all")])
    table(rows, ["subset", "NM", "BG", "CL", "acc_all"])
    out["scratch_accuracy_pct"] = {s: {c: scratch[S2ID[s]][f"acc_{c}"] * 100
                                       for c in ("nm", "bg", "cl", "all")} for s in SUBSETS}

    # ---------- 表4: 手法比較 ----------
    emit("\n\n【表4】手法比較（フェーズ1、目的変数 acc_all）")
    methods = [
        ["中村論文(報告値)", PAPER["ridge_spearman"], None, PAPER["logreg_accuracy"],
         PAPER["logreg_macro_f1"], 11, "README.md:69 (文献値)"],
        ["Inception/DINO(本環境)", r_ridge_i["acc_all"]["spearman"], r_ridge_i["acc_all"]["mae"],
         r_logreg_i["accuracy"], r_logreg_i["macro_f1"], len(r_ridge_i["acc_all"]["coef"]),
         "result_ridge_incdino.json / result_logreg_incdino.json"],
        ["DGV2・共通抽出(採用)", r_ridge_s["acc_all"]["spearman"], r_ridge_s["acc_all"]["mae"],
         r_logreg_s["accuracy"], r_logreg_s["macro_f1"], len(r_ridge_s["acc_all"]["coef"]),
         "result_ridge_sustech.json / result_logreg_sustech.json"],
    ]
    table([[m[0], f"{m[1]:.4f}", "—" if m[2] is None else f"{m[2]:.4f}",
            f"{m[3]:.4f}", f"{m[4]:.4f}", m[5]] for m in methods],
          ["手法", "Ridge ρ", "Ridge MAE", "優劣Acc", "Macro-F1", "変数数"])
    out["method_comparison"] = [{"method": m[0], "ridge_spearman": m[1], "ridge_mae": m[2],
                                 "logreg_accuracy": m[3], "logreg_macro_f1": m[4],
                                 "n_features": m[5], "source": m[6]} for m in methods]
    for m in methods:
        emit(f"    {m[0]:24s} ← {m[6]}")

    # ---------- 表5: LODO予測値（ρとMAEの乖離の根拠。これまで保存物なし） ----------
    emit("\n\n【表5】LODO予測値の内訳（ρとMAEが逆転する理由の根拠。★新規保存）")
    preds = {}
    for tag, src, fl in [("DGV2(6変数)", scratch, F_DGV2), ("Inception/DINO(11変数)", ryu, F_INCDINO)]:
        X = np.array([[math.log(src[S2ID[s]]["Number of people"]) if f == "log_people"
                       else src[S2ID[s]][f] for f in fl] for s in SUBSETS], float)
        y = np.array([src[S2ID[s]]["acc_all"] for s in SUBSETS], float)
        p = lodo_predictions(X, y)
        rho = float(spearmanr(y, p)[0])
        mae = float(np.mean(np.abs(y - p)))
        rt = len(y) - np.argsort(np.argsort(y))
        rp = len(y) - np.argsort(np.argsort(p))
        preds[tag] = {"spearman": rho, "mae": mae,
                      "per_subset": {SUBSETS[i]: {"true_pct": float(y[i] * 100),
                                                  "pred_pct": float(p[i] * 100),
                                                  "error_pt": float((p[i] - y[i]) * 100),
                                                  "rank_true": int(rt[i]), "rank_pred": int(rp[i])}
                                     for i in range(len(y))}}
        emit(f"\n  --- {tag}:  Spearman={rho:.4f}  MAE={mae:.4f} ---")
        order = sorted(range(len(y)), key=lambda i: rt[i])
        table([[SUBSETS[i], f"{y[i]*100:.2f}", f"{p[i]*100:.2f}", f"{(p[i]-y[i])*100:+.2f}",
                rt[i], rp[i], f"{rp[i]-rt[i]:+d}"] for i in order],
              ["subset", "実測", "予測", "誤差pt", "実順位", "予順位", "ズレ"])
    out["lodo_predictions"] = preds
    inc = preds["Inception/DINO(11変数)"]["per_subset"]["000-180"]
    emit(f"\n  → Inception/DINOは000-180を {inc['pred_pct']:.1f}%（負の精度）と予測。"
         f"順位は{inc['rank_pred']}位で正解するためρは高いが、誤差{inc['error_pt']:+.1f}ptでMAEが悪化。"
         f"\n     11変数に対しLODOの学習点は12しかなく、学習分布外の000-180で外挿が破綻している。")

    # ---------- 表6: 条件別ρ・FID_tt係数 ----------
    emit("\n\n【表6】フェーズ1 条件別 LODO Ridge（出典: result_ridge_sustech.json）")
    table([[c.upper(), f"{r_ridge_s['acc_'+c]['spearman']:.3f}", f"{r_ridge_s['acc_'+c]['mae']:.4f}",
            f"{r_ridge_s['acc_'+c]['coef']['FID_train_test_DeepGaitV2']:+.4f}"]
           for c in ("nm", "bg", "cl", "all")],
          ["条件", "Spearman", "MAE", "FID_tt係数"])
    emit("\n  優劣判定の重要度（出典: result_logreg_sustech.json）")
    table([[k, f"{v:.3f}"] for k, v in sorted(r_logreg_s["importance"].items(),
                                              key=lambda kv: -kv[1])], ["変数", "重要度"])
    out["phase1_per_condition"] = {c: r_ridge_s[f"acc_{c}"] for c in ("nm", "bg", "cl", "all")}
    out["phase1_logreg_importance"] = r_logreg_s["importance"]

    # ---------- 表7: zero-shot / ft / Δ / ft−zeroshot（ft−zs はこれまで保存物なし） ----------
    zs = r_sens["zeroshot"]
    emit(f"\n\n【表7】zero-shot・ft精度・Δ・ft−zero-shot（★ft−zsは新規保存）")
    emit(f"  zero-shot基準値（出典: ft_sustech_sensitivity_grid_20260726.json）: "
         f"NM {zs['nm']:.2f} / BG {zs['bg']:.2f} / CL {zs['cl']:.2f} → acc_all {zs['all']:.2f}%")
    rows, full = [], {}
    for sub in SUBSETS:
        f_, s = ft[sub], scratch[S2ID[sub]]
        ftall = f_["acc_all"] * 100
        rec = {"scratch_all_pct": s["acc_all"] * 100, "ft_all_pct": ftall,
               "delta_vs_scratch_pt": f_["delta_acc_all_pt"],
               "ft_minus_zeroshot_pt": round(ftall - zs["all"], 2),
               **{f"ft_{c}_pct": f_[f"acc_{c}"] * 100 for c in ("nm", "bg", "cl")},
               **{f"delta_{c}_pt": round((f_[f"acc_{c}"] - s[f"acc_{c}"]) * 100, 2)
                  for c in ("nm", "bg", "cl")},
               **{f"delta_zs_{c}_pt": round(f_[f"acc_{c}"] * 100 - zs[c], 2)
                  for c in ("nm", "bg", "cl")}}
        full[sub] = rec
        rows.append([sub, f"{rec['scratch_all_pct']:.2f}", f"{rec['ft_nm_pct']:.2f}",
                     f"{rec['ft_bg_pct']:.2f}", f"{rec['ft_cl_pct']:.2f}", f"{ftall:.2f}",
                     f"{rec['delta_vs_scratch_pt']:+.2f}", f"{rec['ft_minus_zeroshot_pt']:+.2f}"])
    out["zeroshot"] = zs
    out["phase2_full"] = full
    table(rows, ["subset", "scratch", "ftNM", "ftBG", "ftCL", "ft_all", "Δ(vs scratch)", "ft−zs"])
    neg = [s for s in SUBSETS if full[s]["ft_minus_zeroshot_pt"] < 0]
    emit(f"  → ft がzero-shotを下回るのは {neg}（{full[neg[0]]['ft_minus_zeroshot_pt']:+.2f}pt）")

    emit("\n  条件別Δ分解（★新規保存。scratch比とzero-shot比を併記）")
    table([[s, f"{full[s]['delta_nm_pt']:+.2f}", f"{full[s]['delta_bg_pt']:+.2f}",
            f"{full[s]['delta_cl_pt']:+.2f}",
            f"{full[s]['delta_zs_nm_pt']:+.2f}", f"{full[s]['delta_zs_bg_pt']:+.2f}",
            f"{full[s]['delta_zs_cl_pt']:+.2f}", f"{full[s]['ft_cl_pct']:.2f}"]
           for s in sorted(SUBSETS, key=lambda s: -full[s]["delta_vs_scratch_pt"])],
          ["subset", "ΔNM(scr)", "ΔBG(scr)", "ΔCL(scr)",
           "ΔNM(zs)", "ΔBG(zs)", "ΔCL(zs)", "ft CL@R1"])
    emit("  → Δzs(acc_all) = ft − 定数(70.83) なので、acc_all の順位・回帰は ft精度の分析と数学的に同一。"
         "\n     条件別のΔzsは『fine-tuningが元モデルに何を足し、何を壊したか』の診断レンズとして使う。")

    # ---------- 表8〜: 既存の結果JSONをそのまま同梱 ----------
    emit("\n\n【表8】フェーズ2 本分析 / 追加分析 / 感度 / 比較可能性")
    emit("  以下は既存JSONをそのまま同梱（再計算していない）:")
    for k, v, src in [("phase2_main", r_ft, "result_ft_analysis_20260726.json"),
                      ("phase2_extra", r_extra, "result_extra_analysis_20260726.json"),
                      ("sensitivity", r_sens, "ft_sustech_sensitivity_grid_20260726.json"),
                      ("comparability", r_comp, "result_comparability_20260726.json")]:
        out[k] = v
        emit(f"    {k:16s} ← {src}")
    emit(f"\n  主要値の確認: Ridge Δ ρ={r_ft['ridge']['delta_pt']['spearman']:.3f} / "
         f"優劣Δ Acc={r_ft['logreg']['delta_tau5pt']['accuracy']:.3f} / "
         f"偏ρ(Δ,FID_tt|scratch)={r_ft['core_question']['partial_spearman_delta_fidtt_given_scratch']['rho']:+.3f}"
         f" (p={r_ft['core_question']['partial_spearman_delta_fidtt_given_scratch']['p']:.3f})")

    jp = os.path.join(HERE, "presentation_tables_20260726.json")
    tp = os.path.join(HERE, "presentation_tables_20260726.txt")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(tp, "w", encoding="utf-8") as f:
        f.write(OUT.getvalue())
    print(f"\nwrote {jp}\nwrote {tp}")


if __name__ == "__main__":
    main()
