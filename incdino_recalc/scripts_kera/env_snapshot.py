"""環境固定スナップショット (計画v2 手順1): ライブラリ版・重みSHA256を記録し、重みをキャッシュに固定する。"""
import hashlib
import json
import os
import platform

import numpy
import PIL
import scipy
import sklearn
import torch
import torchvision
from torchvision.models import Inception_V3_Weights, inception_v3

info = {
    "date": "2026-08-06",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "sklearn": sklearn.__version__,
    "pillow": PIL.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
}

print("downloading/loading Inception weights ...", flush=True)
inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
print("downloading/loading DINOv2 via torch.hub ...", flush=True)
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)

weights = {}
for root, _dirs, files in os.walk("/root/.cache/torch"):
    for f in files:
        if f.endswith((".pth", ".pt")):
            p = os.path.join(root, f)
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            weights[f] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
info["weights"] = weights

hub_dirs = []
hub_root = "/root/.cache/torch/hub"
if os.path.isdir(hub_root):
    hub_dirs = sorted(d for d in os.listdir(hub_root) if os.path.isdir(os.path.join(hub_root, d)))
info["hub_repo_dirs"] = hub_dirs

out = "/work/results/env_snapshot_20260806.json"
with open(out, "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print("saved", out, flush=True)
print(json.dumps(info, indent=2)[:1500])
