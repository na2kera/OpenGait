#!/usr/bin/env python3
"""指標のサブセット間比較可能性の検証 (2026-07-26)。

問い: MSD / 1NN / kNN / FID / FID_tt は、構成（人数・1人あたり系列数・条件・視野角）
      が異なるサブセット間で比較しても意味のある値になっているか。
      つまり観測された差は「データの中身の差」か「サンプリング構造の差による人工物」か。

方法: 全サブセット共通のテストセット埋め込み (49人 × 約110系列 = 5375, 全条件・全視野角)
      をサンドボックスに使い、"中身を固定して構造だけ変える" / "構造を固定して中身だけ変える"
      の2方向で指標を再計算し、効果量を実サブセット間のレンジと比較する。

実験A (構造の影響): 中身固定 (全条件・全視野角) で 人数P × 1人あたり系列数S を変え、
                    総数nをほぼ一定に保つ。1NN/kNNの log_people 相関 (実測 -0.82) が
                    人工物かどうかを判定する。
実験B (中身の影響): 構造固定 (同じP・同じn) で 視野角制限 / 条件制限 をかける。
                    000-180 が全指標で極端な値を示すのが中身由来かを判定する。
実験C (FID_tt の総数依存): 実サブセットの n の幅 (1488〜1644) が FID_tt をどれだけ動かすか。

実行: docker exec opengait_container python /app/OpenGait/dgv2_analysis/metric_comparability_20260726.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(os.path.dirname(HERE), "dgv2_features_sustech")
KNN_K = 5
RNG = np.random.RandomState(0)


def load_flat(path):
    d = np.load(path)
    emb = d["embeddings"].astype(np.float64)
    emb = emb.reshape(emb.shape[0], -1)
    return emb, d["labels"], d["types"], d["views"]


def compute_msd(emb, labels):
    total = 0.0
    subjects = np.unique(labels)
    for s in subjects:
        x = emb[labels == s]
        total += np.mean(np.sum((x - x.mean(axis=0)) ** 2, axis=1))
    return float(total / len(subjects))


def compute_knn(emb, labels, k=KNN_K):
    n = emb.shape[0]
    sq = np.sum(emb ** 2, axis=1)
    one_nn, k_nn = [], []
    block = 512
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        d2 = sq[i0:i1, None] - 2.0 * emb[i0:i1] @ emb.T + sq[None, :]
        np.clip(d2, 0, None, out=d2)
        dist = np.sqrt(d2)
        for r in range(i1 - i0):
            valid = dist[r][labels != labels[i0 + r]]
            one_nn.append(valid.min())
            k_nn.append(np.mean(np.partition(valid, k)[:k]))
    return float(np.mean(one_nn)), float(np.mean(k_nn))


def _center_scale(x):
    n = x.shape[0]
    return (x - x.mean(axis=0)) / np.sqrt(max(n - 1, 1))


def fid_between(x1, x2):
    mu1, mu2 = x1.mean(axis=0), x2.mean(axis=0)
    a1, a2 = _center_scale(x1), _center_scale(x2)
    tr1, tr2 = float(np.sum(a1 ** 2)), float(np.sum(a2 ** 2))
    sv = np.linalg.svd(a1 @ a2.T, compute_uv=False)
    return float(np.sum((mu1 - mu2) ** 2)) + tr1 + tr2 - 2.0 * float(np.sum(sv))


def compute_pair_fid(emb, labels):
    subjects = np.unique(labels)
    groups = {s: emb[labels == s] for s in subjects}
    fids = []
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            fids.append(fid_between(groups[subjects[i]], groups[subjects[j]]))
    return float(np.mean(fids))


def subsample(emb, labels, types, views, n_people=None, n_seq=None,
              keep_views=None, keep_types=None, seed=0):
    """条件でフィルタしてから 人数 / 1人あたり系列数 を絞る。"""
    rng = np.random.RandomState(seed)
    mask = np.ones(len(labels), dtype=bool)
    if keep_views is not None:
        mask &= np.isin(views, keep_views)
    if keep_types is not None:
        mask &= np.array([any(t.startswith(p) for p in keep_types) for t in types])
    e, l = emb[mask], labels[mask]
    subjects = np.unique(l)
    if n_people is not None and n_people < len(subjects):
        subjects = rng.choice(subjects, n_people, replace=False)
    idx = []
    for s in subjects:
        pos = np.where(l == s)[0]
        if n_seq is not None and n_seq < len(pos):
            pos = rng.choice(pos, n_seq, replace=False)
        idx.extend(pos.tolist())
    idx = np.array(sorted(idx))
    return e[idx], l[idx]


def metrics_of(emb, labels, label=""):
    t0 = time.time()
    msd = compute_msd(emb, labels)
    one_nn, k_nn = compute_knn(emb, labels)
    pair_fid = compute_pair_fid(emb, labels)
    print(f"  {label:34s} n={emb.shape[0]:5d} P={len(np.unique(labels)):3d} "
          f"MSD={msd:7.2f} 1NN={one_nn:6.3f} kNN={k_nn:6.3f} FID={pair_fid:7.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {"n": int(emb.shape[0]), "P": int(len(np.unique(labels))),
            "MSD": msd, "1NN": one_nn, "kNN": k_nn, "FID": pair_fid}


def main():
    emb, labels, types, views = load_flat(os.path.join(FEAT_DIR, "default", "test.npz"))
    print(f"共通テストセット: n={emb.shape[0]}, 人数={len(np.unique(labels))}, "
          f"条件={sorted(set(types))}, 視野角数={len(set(views))}")
    seq_per = emb.shape[0] / len(np.unique(labels))
    print(f"1人あたり系列数 ≈ {seq_per:.0f}\n")

    out = {}

    # ---- 実験A: 中身固定・構造だけ変える (総数nをほぼ一定に) ----
    print("=" * 100)
    print("実験A: 中身固定（全条件・全視野角）、人数P×1人あたり系列数Sを変えて総数nを一定に保つ")
    print("       → 実サブセットの 15人/25人/50人/75人 に対応する構造差だけを再現")
    out["A_structure"] = {}
    for P, S, tag in [(15, 109, "P=15, S=109 (defaultの構造)"),
                      (25, 65, "P=25, S=65 (nm6/nm2-bg2-cl2の構造)"),
                      (49, 33, "P=49, S=33 (nm1-bg1-cl1の構造)"),
                      (49, 22, "P=49, S=22 (75人系の構造の代理)")]:
        e, l = subsample(emb, labels, types, views, n_people=P, n_seq=S)
        out["A_structure"][tag] = metrics_of(e, l, tag)

    # ---- 実験B: 構造固定・中身だけ変える ----
    print("\n" + "=" * 100)
    print("実験B: 構造固定（P=49, S=20前後で総数を揃える）、視野角/条件の中身だけ変える")
    print("       → 000-180 の極端な指標値が中身由来かを判定")
    out["B_content"] = {}
    for kv, kt, S, tag in [
        (None, None, 20, "全条件・全視野角 (S=20)"),
        (["000", "180"], None, 20, "視野角 {000,180} のみ ← 000-180相当"),
        (["000", "090"], None, 20, "視野角 {000,090} のみ ← 000-090相当"),
        (["090", "180"], None, 20, "視野角 {090,180} のみ ← 090-180相当"),
        (None, ["nm"], 20, "条件 NM のみ・全視野角 ← nm2/nm6相当"),
        (None, ["cl"], 20, "条件 CL のみ・全視野角 ← cl2相当"),
    ]:
        e, l = subsample(emb, labels, types, views, n_seq=S, keep_views=kv, keep_types=kt)
        out["B_content"][tag] = metrics_of(e, l, tag)

    # ---- 実験C: FID_tt の総数依存 ----
    print("\n" + "=" * 100)
    print("実験C: FID_tt の総サンプル数依存（実サブセットの n は 1488〜1644 の範囲）")
    subjects = np.unique(labels)
    ref_subj = subjects[:24]
    src_subj = subjects[24:]
    ref_mask = np.isin(labels, ref_subj)
    ref_emb = emb[ref_mask]
    src_emb, src_lab = emb[~ref_mask], labels[~ref_mask]
    print(f"  参照側（24人）n={ref_emb.shape[0]}、抽出元（25人）n={src_emb.shape[0]}")
    out["C_fidtt_n"] = {}
    for n in [1000, 1488, 1644, 2000]:
        if n > src_emb.shape[0]:
            continue
        t0 = time.time()
        idx = RNG.choice(src_emb.shape[0], n, replace=False)
        v = fid_between(src_emb[idx], ref_emb)
        out["C_fidtt_n"][str(n)] = v
        print(f"  n={n:5d} → FID_tt={v:7.3f}  ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(HERE, "result_comparability_20260726.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote result_comparability_20260726.json")


if __name__ == "__main__":
    main()
