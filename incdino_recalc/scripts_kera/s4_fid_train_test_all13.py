# ===========================================================
# Train vs Test  FID evaluation (Inception & DINOv2)
# 全画像を対象とした特徴抽出 → 分布比較
# 複数Trainフォルダを順に処理 & JSON保存
# ===========================================================

import os
import glob
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm
from scipy import linalg


# ===========================================================
# 0. 全画像パス収集
# ===========================================================
def collect_all_image_paths(root):
    paths = glob.glob(os.path.join(root, "**/*.png"), recursive=True)
    paths += glob.glob(os.path.join(root, "**/*.jpg"), recursive=True)
    paths += glob.glob(os.path.join(root, "**/*.jpeg"), recursive=True)
    return sorted(paths)


# ===========================================================
# 1. InceptionV3 Feature Extractor
# ===========================================================
class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        model.fc = nn.Identity()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

    def forward(self, x):
        return self.model(x)


# ===========================================================
# 2. DINOv2 Feature Extractor
# ===========================================================
class DINOv2FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            out = self.model(x)
        if isinstance(out, dict):
            out = out["x_norm_patchtokens"].mean(dim=1)
        return out


# ===========================================================
# 3. FID Calculation
# ===========================================================
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

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


# ===========================================================
# 4. Extract Feature Distribution for a dataset
# ===========================================================
def compute_dataset_features(paths, extractor, transform, device):
    feats = []
    for p in tqdm(paths, desc="Extracting features", leave=False):
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = extractor(x).cpu().numpy()
        feats.append(feat)

    feats = np.concatenate(feats, axis=0)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


# ===========================================================
# 5. Main Process
# ===========================================================
def compute_fid_train_test(train_roots, test_root, output_json="fid_results.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------ Load Feature Extractors ------------------
    inception = InceptionFeatureExtractor().to(device)
    dinov2 = DINOv2FeatureExtractor().to(device)

    # ------------------ Test features ------------------
    print("\n=== Loading Test Dataset ===")
    test_paths = collect_all_image_paths(test_root)
    print(f"Test images: {len(test_paths)}")

    transform_inception = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    transform_dino = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    print("\nExtracting Test Features (Inception)")
    test_mu_inc, test_sigma_inc = compute_dataset_features(test_paths, inception, transform_inception, device)

    print("\nExtracting Test Features (DINOv2)")
    test_mu_dino, test_sigma_dino = compute_dataset_features(test_paths, dinov2, transform_dino, device)

    # ------------------ Train datasets processing ------------------
    results = {"Inception": {}, "DINOv2": {}}

    for train_root in train_roots:
        print("\n===============================================")
        print(f"Processing Train Dataset → {train_root}")
        print("===============================================")

        train_paths = collect_all_image_paths(train_root)
        print(f"Train images: {len(train_paths)}")

        # ---- Inception ----
        print("\nExtracting Train Features (Inception)")
        mu_inc, sigma_inc = compute_dataset_features(train_paths, inception, transform_inception, device)
        fid_inc = calculate_fid(mu_inc, sigma_inc, test_mu_inc, test_sigma_inc)
        results["Inception"][train_root] = fid_inc

        # ---- DINOv2 ----
        print("\nExtracting Train Features (DINOv2)")
        mu_dino, sigma_dino = compute_dataset_features(train_paths, dinov2, transform_dino, device)
        fid_dino = calculate_fid(mu_dino, sigma_dino, test_mu_dino, test_sigma_dino)
        results["DINOv2"][train_root] = fid_dino

        print(f"\n➡ FID (Inception): {fid_inc:.4f}")
        print(f"➡ FID (DINOv2):   {fid_dino:.4f}")

    # ------------------ Save JSON ------------------
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n\n=====================")
    print("FID計算完了 → 出力:", output_json)
    print("=====================")

    return results


# ===========================================================
# Usage Example
# ===========================================================
if __name__ == "__main__":
    train_roots = [   "/data/CASIA-B-png/20-data-png",
                      "/data/CASIA-B-png/nm1-2-png",
                      "/data/CASIA-B-png/bg1-2-png",
                      "/data/CASIA-B-png/cl1-2-png",
                      "/data/CASIA-B-png/nm1-bg1-png",
                      "/data/CASIA-B-png/nm1-cl1-png",
                      "/data/CASIA-B-png/bg1-cl1-png",
                      "/data/CASIA-B-png-75/000-180-png",
                      "/data/CASIA-B-png-75/000-090-png",
                      "/data/CASIA-B-png-75/090-180-png",
                      "/data/CASIA-B-png/nm1-bg1-cl1-png",
                      "/data/CASIA-B-png/nm2-bg2-cl2-png",
                      "/data/CASIA-B-png/nm6-png",
    ]
    test_root = "/data/CASIA-B-png/GaitDatasetB-silh-test-png"

    compute_fid_train_test(train_roots, test_root, output_json="/work/results/fid_train_test_results_kera_all13.json")
