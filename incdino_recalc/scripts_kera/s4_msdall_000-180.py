import os
import glob
import json
import numpy as np
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights


# ============================
# 1. 画像収集
# ============================
def collect_image_paths_by_condition(root):
    data = {"nm": [], "bg": [], "cl": []}
    subjects = sorted(os.listdir(root))
    for sid in subjects:
        sub_dir = os.path.join(root, sid)
        if not os.path.isdir(sub_dir):
            continue
        for cond in os.listdir(sub_dir):
            cond_dir = os.path.join(sub_dir, cond)
            if not os.path.isdir(cond_dir):
                continue
            key = cond.split("-")[0]
            for view in os.listdir(cond_dir):
                view_dir = os.path.join(cond_dir, view)
                if not os.path.isdir(view_dir):
                    continue
                imgs = glob.glob(os.path.join(view_dir, "*.png"))
                data[key].extend(imgs)
    data["all"] = data["nm"] + data["bg"] + data["cl"]
    return data


# ============================
# 2. 平均画像と MSD（輝度空間）
# ============================
def compute_average_image(paths, resize=(240, 320)):
    sum_img, count = None, 0
    for p in tqdm(paths, desc="Computing average image"):
        img = Image.open(p).convert("L").resize(resize)
        arr = np.array(img, dtype=np.float32)
        if sum_img is None:
            sum_img = np.zeros_like(arr, dtype=np.float64)
        sum_img += arr
        count += 1
    return (sum_img / count).astype(np.float32)


def compute_msd(paths, avg_img, resize=(240, 320)):
    diffs = []
    for p in tqdm(paths, desc="Computing MSD (image space)"):
        img = Image.open(p).convert("L").resize(resize)
        arr = np.array(img, dtype=np.float32)
        diffs.append(np.mean((arr - avg_img) ** 2))
    return np.mean(diffs)


# ============================
# 3. InceptionV3 特徴抽出器
# ============================
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


# ============================
# 4. DINOv2 特徴抽出器
# ============================
class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        print("🔹 Loading DINOv2 model (ViT-S/14)...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            feats = self.model(x)
        return feats


# ============================
# 5. 特徴空間MSD計算
# ============================
def compute_feature_msd(paths, extractor, transform, device="cuda"):
    feats = []
    for p in tqdm(paths, desc="Extracting features for MSD"):
        img = Image.open(p).convert("L")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = extractor(x).cpu().numpy().flatten()
        feats.append(feat)
    feats = np.array(feats)
    avg_feat = np.mean(feats, axis=0)
    msd_feat = np.mean(np.square(feats - avg_feat))
    return msd_feat


# ============================
# 6. メイン処理
# ============================
def process_dataset(root, device="cuda"):
    print(f"\n=== 📁 Processing dataset: {root} ===")
    if not os.path.exists(root):
        print(f"❌ Path not found: {root}")
        return {}

    data = collect_image_paths_by_condition(root)
    results = {}

    # 特徴抽出モデル準備
    inception_extractor = InceptionFeatureExtractor().to(device)
    dino_extractor = DINOv2FeatureExtractor(device=device)

    transform_inception = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    transform_dino = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    for cond, paths in data.items():
        if len(paths) == 0:
            print(f"⚠️ No images for {cond}, skipping.")
            continue

        print(f"\n=== 🧩 Condition: {cond.upper()} ({len(paths)} images) ===")
        cond_result = {}

        try:
            avg_img = compute_average_image(paths)
            msd_img = compute_msd(paths, avg_img)
            cond_result["MSD_image_space"] = float(msd_img)
        except Exception as e:
            cond_result["MSD_image_space"] = None
            print(f"⚠️ Image MSD failed: {e}")

        try:
            msd_inception = compute_feature_msd(paths, inception_extractor, transform_inception, device)
            cond_result["MSD_inception"] = float(msd_inception)
        except Exception as e:
            cond_result["MSD_inception"] = None
            print(f"⚠️ Inception MSD failed: {e}")

        try:
            msd_dino = compute_feature_msd(paths, dino_extractor, transform_dino, device)
            cond_result["MSD_dinov2"] = float(msd_dino)
        except Exception as e:
            cond_result["MSD_dinov2"] = None
            print(f"⚠️ DINOv2 MSD failed: {e}")

        results[cond] = cond_result

    return results


# ============================
# 7. JSON出力
# ============================
def save_results_json(results, output_dir="./results", prefix="MSD_results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(output_dir, f"{prefix}_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Results saved to: {out_path}")


# ============================
# 8. 実行部
# ============================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_roots = [ "/data/CASIA-B-png-75/000-180-png" ]

    all_results = {}
    for root in dataset_roots:
        result = process_dataset(root, device=device)
        all_results[root] = result

    save_results_json(all_results, "./results", "MSD_results_000-180")
