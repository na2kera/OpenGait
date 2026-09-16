# =====================================================
# Multiple FID Evaluation Script for CASIA-B (per subject + ALL)
# =====================================================
import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm
from scipy import linalg
# =====================================================
# 1. CASIA-B構造に基づく画像パス収集（人物×条件）
# =====================================================
def collect_image_paths_by_subject_and_condition(root):
    data = {'nm': {}, 'bg': {}, 'cl': {}}
    if not os.path.exists(root):
        print(f"❌ Path not found: {root}")
        return data

    subjects = sorted(os.listdir(root))
    for sid in subjects:
        sub_dir = os.path.join(root, sid)
        if not os.path.isdir(sub_dir):
            continue
        for cond in os.listdir(sub_dir):
            cond_type = cond[:2]
            if cond_type not in data:
                continue
            cond_dir = os.path.join(sub_dir, cond)
            all_imgs = []
            for view in os.listdir(cond_dir):
                view_dir = os.path.join(cond_dir, view)
                if not os.path.isdir(view_dir):
                    continue
                imgs = glob.glob(os.path.join(view_dir, "*.png"))
                all_imgs.extend(imgs)
            if all_imgs:
                data[cond_type].setdefault(sid, []).extend(sorted(all_imgs))
    return data
# =====================================================
# 2. Inception 特徴抽出器
# =====================================================
class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,aux_logits=True)
        model.fc = nn.Identity()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        
    def forward(self, x):
        return self.model(x)

# =====================================================
# 3. FID計算関数
# =====================================================
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    mu1 = np.nan_to_num(mu1)
    mu2 = np.nan_to_num(mu2)
    sigma1 = np.nan_to_num(sigma1)
    sigma2 = np.nan_to_num(sigma2)
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        covmean = np.nan_to_num(covmean)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(np.abs(fid))

# =====================================================
# 4. 特徴抽出 + 分布計算（人物単位）
# =====================================================
def compute_subject_features(paths, extractor, device):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    feats = []
    for p in tqdm(paths, desc="Extracting features", leave=False):
        img = Image.open(p).convert("L")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = extractor(x).detach().cpu().numpy()
        feats.append(feat)
    feats = np.concatenate(feats, axis=0)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma
# =====================================================
# 5. 条件ごとのFID計算
# =====================================================
def compute_condition_fid(data_dict, extractor, device, cond_name):
    print(f"\n🧩 Condition = {cond_name}")
    if not data_dict:
        print(f"⚠️ No subjects found for condition {cond_name}.")
        return None

    subject_stats = {}
    for sid, paths in data_dict.items():
        if len(paths) == 0:
            print(f" ⚠️ Subject {sid} has no images, skipping.")
            continue
        print(f" 🧍 Subject {sid} ({len(paths)} images)")
        mu, sigma = compute_subject_features(paths, extractor, device)
        subject_stats[sid] = (mu, sigma)
    subjects = list(subject_stats.keys())
    if len(subjects) < 2:
            print(f"⚠️ Not enough subjects to compute FID for {cond_name}.")
            return None
    fids = []
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            mu1, sigma1 = subject_stats[subjects[i]]
            mu2, sigma2 = subject_stats[subjects[j]]
            fid = calculate_fid(mu1, sigma1, mu2, sigma2)
            fids.append(fid)
    avg_fid = np.mean(fids)
    print(f" ✅ Average FID ({cond_name}) = {avg_fid:.4f}")
    return avg_fid

# =====================================================
# 6. 新規追加: 全条件統合FID（ALL）
# =====================================================
def compute_all_fid(data_dict, extractor, device):
    """ nm, bg, cl 全ての条件を統合し、人物単位で全画像から特徴を抽出 → FID算出 """
    print("\n🧩 Condition = ALL (nm+bg+cl combined)")
    # 条件を統合して {sid: 全画像リスト} を作成
    all_data = {}
    for cond in ["nm", "bg", "cl"]:
        for sid, paths in data_dict.get(cond, {}).items():
            all_data.setdefault(sid, []).extend(paths)

    if len(all_data) < 2:
        print("⚠️ Not enough subjects for ALL condition.")
        return None
    subject_stats = {}
    for sid, paths in all_data.items():
        if len(paths) == 0:
            continue
        print(f" 🧍 Subject {sid} ({len(paths)} images total)")
        mu, sigma = compute_subject_features(paths, extractor, device)
        subject_stats[sid] = (mu, sigma)

    subjects = list(subject_stats.keys())
    fids = []
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            mu1, sigma1 = subject_stats[subjects[i]]
            mu2, sigma2 = subject_stats[subjects[j]]
            fid = calculate_fid(mu1, sigma1, mu2, sigma2)
            fids.append(fid)

    avg_fid = np.mean(fids)
    print(f" ✅ Average FID (ALL) = {avg_fid:.4f}")
    return avg_fid

# =====================================================
# 7. メイン実行：複数フォルダを順に処理
# =====================================================
if __name__ == "__main__":
    dataset_roots = [ "/data/CASIA-B-png/nm6-png" ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = InceptionFeatureExtractor().to(device)

    overall_results = {}
    for root in dataset_roots:
        print("\n" + "="*60)
        print(f"📁 Processing dataset: {root}")
        print("="*60)

        if not os.path.exists(root):
            print(f"❌ Path not found, skipping: {root}")
            overall_results[root] = {"nm": None, "bg": None, "cl": None, "ALL": None}
            continue

        data = collect_image_paths_by_subject_and_condition(root)
        fid_results = {}

        for cond in ["nm", "bg", "cl"]:
            if len(data[cond]) == 0:
                print(f"⚠️ No data found for condition: {cond}")
                fid_results[cond] = None
                continue

            fid_results[cond] = compute_condition_fid(data[cond], extractor, device, cond)

        # 🔹 新規追加: 全体 (ALL)
        fid_results["ALL"] = compute_all_fid(data, extractor, device)
        overall_results[root] = fid_results
# =====================================================
# 8. 全体結果表示
# =====================================================
    print("\n📊 ======= Final Summary =======")
    for root, conds in overall_results.items():
        print(f"\n📂 Dataset: {root}")
        for cond, fid in conds.items():
            if fid is None:
                print(f" {cond.upper()} FID: null")
            else: print(f" {cond.upper()} FID: {fid:.4f}")
