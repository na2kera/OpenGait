# -*- coding: utf-8 -*-
"""s5: 汎用画像特徴（InceptionV3 / DINOv2）の平均クラス内分散を、論文の式(1)＝人物ごとの定義で計測する。

背景（2026-09-14）: 採用中の Inception_MSD / DINO_MSD は中村実装 MSDall.py（= s4_msdall_*.py）の
「人物を区別しない全画像の特徴平均からの二乗偏差の要素平均」（全体分散）であり、DeepGaitV2 側の
式(1)（人物ごと・次元和・人物平均）と集計が違う。汎用側を式(1)で計測し直して揃える。

設計:
  1. 画像を 1 回だけ読み、InceptionV3（fc=Identity, 2048 次元）と DINOv2 ViT-S/14（384 次元）の特徴を
     バッチ推論で抽出して float32 で保存する（/feat/<subset>/{inception,dinov2}.npy と labels/cond/path）。
     前処理は s4_msdall_*.py と同一（L 変換 → Resize → ToTensor → 3ch 複製 → Normalize(0.5,0.5)）。
  2. ゲート: 保存した特徴から s4 と同じ式で全体分散（cond=all）を再計算し、採用値
     （metrics_incdino_kera_X2.json の Inception_MSD / DINO_MSD）と相対誤差 RTOL 以内で一致することを確認する。
     一致しなければ rc=3 で失敗させる（特徴抽出が既存経路と同一であることの証明）。
  3. 式(1) の MSD を人物ごとに計算する（float64）。cond=all と nm/bg/cl 別、人物別の内訳も残す。
  4. JSON を results/MSDclass_results_<subset>_<ts>.json に保存する。

差分は SUBSET / ROOT の 2 行のみ（gen_s5_msdclass.py が生成）。
"""
import glob
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3

SUBSET = "nm1-bg1"            # 例: nm6
ROOT = "/data/CASIA-B-png/nm1-bg1-png"                # 例: /data/CASIA-B-png/nm6-png
FEAT_DIR = f"/feat/{SUBSET}"     # ホスト /home/kera/incdino_data/msdclass_features/<subset>
TARGETS_JSON = "/work/scripts_kera/s5_msd_targets.json"   # 採用値（全体分散）: ゲートの比較対象
RTOL = 1e-3                      # ゲート許容相対誤差（バッチ推論による下位ビット差を許す。cudnn 非決定性の影響も含む）
BATCH = 128
WORKERS = int(os.environ.get("S5_WORKERS", "6"))


# ---------------------------------------------------------------- 画像収集（s4 と同じ走査、人物ラベル付き）
def collect(root):
    paths, labels, conds = [], [], []
    for sid in sorted(os.listdir(root)):
        sub_dir = os.path.join(root, sid)
        if not os.path.isdir(sub_dir):
            continue
        for cond in sorted(os.listdir(sub_dir)):
            cond_dir = os.path.join(sub_dir, cond)
            if not os.path.isdir(cond_dir):
                continue
            key = cond.split("-")[0]
            assert key in ("nm", "bg", "cl"), (sid, cond)
            for view in sorted(os.listdir(cond_dir)):
                view_dir = os.path.join(cond_dir, view)
                if not os.path.isdir(view_dir):
                    continue
                for p in sorted(glob.glob(os.path.join(view_dir, "*.png"))):
                    paths.append(p); labels.append(sid); conds.append(key)
    return paths, np.array(labels), np.array(conds)


class ImgDS(Dataset):
    def __init__(self, paths, t_inc, t_dino):
        self.paths, self.t_inc, self.t_dino = paths, t_inc, t_dino

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")          # s4: Image.open(p).convert("L")
        return self.t_inc(img), self.t_dino(img), i


# ---------------------------------------------------------------- 抽出器（s4 と同一定義）
class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        inception.fc = nn.Identity()
        inception.eval()
        for p in inception.parameters():
            p.requires_grad_(False)
        self.features = inception

    def forward(self, x):
        return self.features(x)


class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- MSD の 2 定義
def msd_global(feats):
    """s4 compute_feature_msd と同一: 全画像平均からの二乗偏差の要素平均（float32）。"""
    avg = np.mean(feats, axis=0)
    return float(np.mean(np.square(feats - avg)))


def msd_class(feats, labels):
    """式(1): 人物ごとに平均を引き、次元方向に和、画像で平均、人物で平均（float64）。"""
    x = feats.astype(np.float64)
    per = {}
    for s in np.unique(labels):
        xs = x[labels == s]
        per[str(s)] = float(np.mean(np.sum((xs - xs.mean(0)) ** 2, axis=1)))
    return float(np.mean(list(per.values()))), per


def main():
    t0 = time.time()
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA が見えない（docker --gpus を確認）"
    os.makedirs(FEAT_DIR, exist_ok=True)
    print(f"=== s5 msdclass {SUBSET} root={ROOT} workers={WORKERS} batch={BATCH}", flush=True)

    paths, labels, conds = collect(ROOT)
    n = len(paths)
    print(f"images={n} subjects={len(np.unique(labels))} conds={dict(zip(*np.unique(conds, return_counts=True)))}", flush=True)
    assert n > 0

    t_inc = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    t_dino = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                 transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                 transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

    f_inc = os.path.join(FEAT_DIR, "inception.npy"); f_dino = os.path.join(FEAT_DIR, "dinov2.npy")
    if os.path.exists(f_inc) and os.path.exists(f_dino) and os.path.exists(os.path.join(FEAT_DIR, "meta.json")):
        print("features already exist -> reuse", flush=True)
        feats_inc = np.load(f_inc); feats_dino = np.load(f_dino)
        meta = json.load(open(os.path.join(FEAT_DIR, "meta.json")))
        assert meta["paths"] == paths, "保存済み特徴と画像一覧が一致しない"
    else:
        inc = InceptionFeatureExtractor().to(device)
        dino = DINOv2FeatureExtractor(device=device)
        dl = DataLoader(ImgDS(paths, t_inc, t_dino), batch_size=BATCH, shuffle=False,
                        num_workers=WORKERS, pin_memory=True)
        feats_inc = np.zeros((n, 2048), dtype=np.float32)
        feats_dino = np.zeros((n, 384), dtype=np.float32)
        done = 0
        with torch.no_grad():
            for xi, xd, idx in dl:
                fi = inc(xi.to(device, non_blocking=True)).cpu().numpy()
                fd = dino(xd.to(device, non_blocking=True)).cpu().numpy()
                idx = idx.numpy()
                feats_inc[idx] = fi; feats_dino[idx] = fd
                done += len(idx)
                if done % (BATCH * 50) < BATCH:
                    print(f"  extracted {done}/{n} ({time.time()-t0:.0f}s)", flush=True)
        assert feats_inc.shape[1] == 2048 and feats_dino.shape[1] == 384
        np.save(f_inc, feats_inc); np.save(f_dino, feats_dino)
        json.dump({"subset": SUBSET, "root": ROOT, "paths": paths, "labels": labels.tolist(),
                   "conds": conds.tolist(), "batch": BATCH, "torch": torch.__version__},
                  open(os.path.join(FEAT_DIR, "meta.json"), "w"))
        print(f"features saved to {FEAT_DIR} ({time.time()-t0:.0f}s)", flush=True)

    # ---- ゲート: 全体分散（s4 定義）が採用値と一致するか
    targets = json.load(open(TARGETS_JSON))[SUBSET]
    out = {"subset": SUBSET, "root": ROOT, "n_images": n, "n_subjects": int(len(np.unique(labels))),
           "cond_counts": {k: int(v) for k, v in zip(*np.unique(conds, return_counts=True))},
           "global_msd": {}, "gate": {}, "class_msd": {}, "class_msd_per_subject": {},
           "feature_files": {"inception": {"path": f_inc, "md5": md5(f_inc)}, "dinov2": {"path": f_dino, "md5": md5(f_dino)}},
           "definition": {"global": "mean over images and dims of (f - mean_all)^2, float32 (= s4/MSDall.py)",
                          "class": "eq.(1): mean_s mean_i ||f_i^s - mu_s||^2, float64"}}
    ok = True
    for space, feats, tkey in (("inception", feats_inc, "Inception_MSD"), ("dinov2", feats_dino, "DINO_MSD")):
        g = {c: msd_global(feats[conds == c]) for c in ("nm", "bg", "cl") if (conds == c).any()}
        g["all"] = msd_global(feats)
        out["global_msd"][space] = g
        tgt = targets[tkey]
        rel = abs(g["all"] - tgt) / abs(tgt)
        out["gate"][space] = {"recomputed_all": g["all"], "adopted": tgt, "rel_err": rel, "pass": bool(rel <= RTOL)}
        ok &= rel <= RTOL
        print(f"GATE {space}: recomputed={g['all']:.9g} adopted={tgt:.9g} rel_err={rel:.2e} -> {'PASS' if rel <= RTOL else 'FAIL'}", flush=True)
        cm, per = msd_class(feats, labels)
        out["class_msd"][space] = {"all": cm}
        for c in ("nm", "bg", "cl"):
            m = conds == c
            if m.any():
                out["class_msd"][space][c] = msd_class(feats[m], labels[m])[0]
        out["class_msd_per_subject"][space] = per
        print(f"CLASS_MSD {space}: all={cm:.6g} " + " ".join(f"{c}={out['class_msd'][space][c]:.6g}" for c in ("nm", "bg", "cl") if c in out["class_msd"][space]), flush=True)

    out["gate"]["all_pass"] = bool(ok); out["rtol"] = RTOL
    out["wall_s"] = time.time() - t0
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    op = f"results/MSDclass_results_{SUBSET}_{ts}.json"
    json.dump(out, open(op, "w"), indent=2, ensure_ascii=False)
    print(f"saved {op} wall={out['wall_s']:.0f}s gate_all_pass={ok}", flush=True)
    if not ok:
        print("GATE FAILED: 全体分散が採用値と一致しない。特徴抽出経路の差異を疑う（rc=3）", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
