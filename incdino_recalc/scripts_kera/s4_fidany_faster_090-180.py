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
# 1. 画像パス収集
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
# 2. Inception Feature Extractor (pool3=2048)
# =====================================================
class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                             transform_input=False,
                             aux_logits=True)
        model.fc = nn.Identity()
        model.eval()

        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x).detach()
        return features


# =====================================================
# 3. バッチ版特徴抽出（結果不変）
# =====================================================
def extract_features_batch(paths, extractor, device, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    all_features = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch_imgs = []

        for p in batch_paths:
            img = Image.open(p).convert("L")
            x = transform(img)
            batch_imgs.append(x)

        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            feats = extractor(batch_tensor).cpu().numpy()

        all_features.append(feats)

    all_features = np.concatenate(all_features, axis=0)
    return all_features


# =====================================================
# 4. FID 計算（安定版: eps I 正則化）
# =====================================================
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    # === 安定化: eps * I を追加 ===
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        covmean = np.nan_to_num(covmean)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(np.abs(fid))


# =====================================================
# 5. 特徴分布計算
# =====================================================
def compute_subject_stats(paths, extractor, device, batch_size=64):
    feats = extract_features_batch(paths, extractor, device, batch_size)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


# =====================================================
# 6. 条件ごとのFID
# =====================================================
def compute_condition_fid(data_dict, extractor, device, cond_name):
    print(f"\n🧩 Condition = {cond_name}")

    subject_stats = {}
    for sid, paths in data_dict.items():
        if len(paths) == 0:
            continue
        print(f" 🧍 Subject {sid} ({len(paths)} images)")
        mu, sigma = compute_subject_stats(paths, extractor, device)
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
# 7. ALL 条件
# =====================================================
def compute_all_fid(data_dict, extractor, device):
    print("\n🧩 Condition = ALL (nm + bg + cl)")

    all_data = {}
    for cond in ["nm", "bg", "cl"]:
        for sid, paths in data_dict.get(cond, {}).items():
            all_data.setdefault(sid, []).extend(paths)

    subject_stats = {}
    for sid, paths in all_data.items():
        if len(paths) == 0:
            continue
        print(f" 🧍 Subject {sid} ({len(paths)} images total)")
        mu, sigma = compute_subject_stats(paths, extractor, device)
        subject_stats[sid] = (mu, sigma)

    subjects = list(subject_stats.keys())
    if len(subjects) < 2:
        print("⚠️ Not enough subjects for ALL.")
        return None

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
# 8. メイン
# =====================================================
if __name__ == "__main__":

    dataset_roots = [ "/data/CASIA-B-png-75/090-180-png" ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = InceptionFeatureExtractor().to(device)

    overall_results = {}

    for root in dataset_roots:
        print("\n" + "=" * 60)
        print(f"📁 Processing dataset: {root}")
        print("=" * 60)

        if not os.path.exists(root):
            print(f"❌ Path not found, skipping.")
            continue

        data = collect_image_paths_by_subject_and_condition(root)

        fid_results = {}
        for cond in ["nm", "bg", "cl"]:
            if len(data[cond]) == 0:
                fid_results[cond] = None
                continue
            fid_results[cond] = compute_condition_fid(
                data[cond], extractor, device, cond
            )

        fid_results["ALL"] = compute_all_fid(data, extractor, device)
        overall_results[root] = fid_results

    print("\n📊 ======= Final Summary =======")
    for root, conds in overall_results.items():
        print(f"\n📂 Dataset: {root}")
        for cond, fid in conds.items():
            if fid is None:
                print(f" {cond.upper()} FID: null")
            else:
                print(f" {cond.upper()} FID: {fid:.4f}")
