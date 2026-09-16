# 追補: s4_knn_dinov2_000-090 実行時に compute_feature_msd 未定義バグで
# feature MSD が全条件スキップされたため、その分だけを再計算する。
# 特徴抽出は s4_knn_dinov2_000-090.py と完全に同一
# （DINOv2 ViT-S/14, Resize(224,224), Gray→RGB repeat, Normalize(0.5)）。
# nm/bg/cl を各1回だけ抽出し、all は3条件の特徴を連結して計算する（抽出は1周のみ）。
# 出力行の書式（Condition ヘッダ / Feature-space MSD = ...）は
# s4_knn_dinov2_*.py と揃えてあり、ログのパースを共通化できる。
import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm


# ============================
# 1. 画像収集（nm/bg/cl）: s4_knn_dinov2_000-090.py と同一
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
# 2. DINOv2 特徴抽出器: s4_knn_dinov2_000-090.py と同一
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
# 3. 特徴抽出: s4_knn_dinov2_000-090.py の extract_features_for_knn と同一
#    （labels は MSD では使わないが、同一性維持のためそのまま）
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


# ============================
# 4. 特徴空間MSD（pilot_msdall_nm6.py と同じ定義）
# ============================
def compute_feature_msd(feats):
    avg_feat = np.mean(feats, axis=0)
    return np.mean(np.square(feats - avg_feat))


# ============================
# 5. メイン処理
# ============================
def process_dataset(root, device="cuda"):
    print(f"\n=== 📁 Processing dataset: {root} ===")
    if not os.path.exists(root):
        print(f"❌ Path not found: {root}")
        return

    data = collect_image_paths_by_condition(root)
    cached = {}
    failed_conditions = []

    for cond in ["nm", "bg", "cl", "all"]:
        paths = data[cond]
        if len(paths) == 0:
            print(f"⚠️ No images for {cond}, skipping.")
            continue

        print(f"\n=== 🧩 Condition: {cond.upper()} ({len(paths)} images) ===")

        try:
            if cond == "all" and all(k in cached for k in ["nm", "bg", "cl"]):
                # data["all"] は nm+bg+cl の連結なので、特徴も同順で連結すれば再抽出と同一
                feats = np.concatenate([cached["nm"], cached["bg"], cached["cl"]], axis=0)
            else:
                feats, _ = extract_features_for_knn(paths, device=device)
                cached[cond] = feats
            if len(feats) == 0:
                raise ValueError("No features extracted")
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

    dataset_roots = [ "/data/CASIA-B-png-75/000-090-png" ]

    for root in dataset_roots:
        process_dataset(root, device=device)
