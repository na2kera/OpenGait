import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import pairwise_distances


# ============================
# 1. 画像収集（nm/bg/cl）
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
# 2. 平均画像と MSD
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
    for p in tqdm(paths, desc="Computing MSD"):
        img = Image.open(p).convert("L").resize(resize)
        arr = np.array(img, dtype=np.float32)
        diffs.append(np.mean((arr - avg_img) ** 2))
    return np.mean(diffs)


# ============================
# 3. DINOv2 特徴抽出器
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
# 4. k-NN 指標
# ============================
def extract_features_for_knn(paths, device="cuda"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # DINOv2 input size
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Gray→RGB
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    extractor = DINOv2FeatureExtractor(device=device)

    feats, labels = [], []
    for p in tqdm(paths, desc="Extracting DINOv2 features"):
        img = Image.open(p).convert("L")
        x = transform(img).unsqueeze(0).to(device)
        feat = extractor(x).cpu().numpy().flatten()
        feats.append(feat)
        label = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))
        labels.append(label)
    return np.array(feats), np.array(labels)


def compute_knn_metrics(feats, labels, k=5):
    print(f"🔍 Computing k-NN (k={k}) excluding same-subject pairs...")
    dists = pairwise_distances(feats, feats, metric="euclidean")
    mean_dists = []
    for i in range(len(feats)):
        mask = labels != labels[i]
        dist_i = dists[i][mask]
        if len(dist_i) == 0:
            continue
        nearest_k = np.sort(dist_i)[:k]
        mean_dists.append(np.mean(nearest_k))
    one_nn_mean = np.mean([np.min(dists[i][labels != labels[i]]) for i in range(len(feats))])
    k_nn_mean = np.mean(mean_dists)
    return one_nn_mean, k_nn_mean


# ============================
# 5. 特徴空間MSD（pilot_msdall_nm6.py と同じ定義。
#    再抽出はせず extract_features_for_knn で得た特徴を再利用する）
# ============================
def compute_feature_msd(feats):
    avg_feat = np.mean(feats, axis=0)
    return np.mean(np.square(feats - avg_feat))


# ============================
# 6. メイン処理
# ============================
def process_dataset(root, device="cuda"):
    print(f"\n=== 📁 Processing dataset: {root} ===")
    if not os.path.exists(root):
        print(f"❌ Path not found: {root}")
        return

    data = collect_image_paths_by_condition(root)
    failed_conditions = []

    for cond, paths in data.items():
        if len(paths) == 0:
            print(f"⚠️ No images for {cond}, skipping.")
            print(f"MSD = null | 1-NN = null | 5-NN = null")
            continue

        print(f"\n=== 🧩 Condition: {cond.upper()} ({len(paths)} images) ===")

        try:
            avg_img = compute_average_image(paths)
            msd = compute_msd(paths, avg_img)
        except ZeroDivisionError:
            print("⚠️ No valid images for average/MSD, skipping.")
            msd = None

        if msd is not None:
            print(f"MSD = {msd:.4f}")
        else:
            print(f"MSD = null")

        # 特徴量抽出と近傍法
        feats = None
        try:
            feats, labels = extract_features_for_knn(paths, device=device)
            if len(feats) == 0:
                raise ValueError("No features extracted")
            one_nn, k5_nn = compute_knn_metrics(feats, labels, k=5)
            print(f"1-NN mean distance = {one_nn:.4f}")
            print(f"5-NN mean distance = {k5_nn:.4f}")
        except Exception as e:
            print(f"⚠️ Skipping k-NN due to error: {e}")
            print(f"1-NN = null | 5-NN = null")


        try:
            if feats is None or len(feats) == 0:
                raise ValueError("no features available (extraction failed)")
            feat_msd = compute_feature_msd(feats)
            print(f"Feature-space MSD = {feat_msd:.4f}")
        except Exception as e:
            print(f"⚠️ Skipping feature MSD due to error: {e}")
            failed_conditions.append(cond)

    # 例外に飲まれて欠落したまま rc=0 で done 扱いになるのを防ぐ
    if failed_conditions:
        raise RuntimeError(f"Feature MSD failed for: {', '.join(failed_conditions)}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_roots = [ "/data/CASIA-B-png/nm1-bg1-png" ]

    for root in dataset_roots:
        process_dataset(root, device=device)
