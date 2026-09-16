# 診断用（完全踏襲の対象外・値の生成には使わない）
# faster版の実装を固定したまま transform_input だけを False/True で振り、
# Inception_FID の差が本当にこの1フラグに帰属するかを対照実験で確かめる。
import os, glob, numpy as np, torch, torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg

ROOT = "/data/CASIA-B-png/nm6-png"
N_SUBJECTS = 4          # 4人 = 6ペア
BATCH = 64

def collect(root):
    data = {}
    for sid in sorted(os.listdir(root)):
        sub = os.path.join(root, sid)
        if not os.path.isdir(sub): continue
        imgs = []
        for cond in os.listdir(sub):
            if cond[:2] != 'nm': continue
            cd = os.path.join(sub, cond)
            for view in os.listdir(cd):
                vd = os.path.join(cd, view)
                if os.path.isdir(vd): imgs.extend(glob.glob(os.path.join(vd, "*.png")))
        if imgs: data[sid] = sorted(imgs)
    return data

class Ext(nn.Module):
    def __init__(self, ti):
        super().__init__()
        m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=ti, aux_logits=True)
        m.fc = nn.Identity(); m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        self.model = m
    def forward(self, x):
        with torch.no_grad(): return self.model(x).detach()

tf = transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor(),
                         transforms.Lambda(lambda x: x.repeat(3,1,1)),
                         transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])

def feats(paths, ext, dev):
    out = []
    for i in range(0, len(paths), BATCH):
        b = torch.stack([tf(Image.open(p).convert("L")) for p in paths[i:i+BATCH]]).to(dev)
        out.append(ext(b).cpu().numpy())
    return np.concatenate(out, axis=0)

def fid(mu1, s1, mu2, s2, eps=1e-6):
    s1 = s1 + np.eye(s1.shape[0])*eps; s2 = s2 + np.eye(s2.shape[0])*eps
    d = mu1 - mu2
    cm, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(cm).all(): cm = np.nan_to_num(cm)
    if np.iscomplexobj(cm): cm = cm.real
    return float(np.abs(d @ d + np.trace(s1 + s2 - 2*cm)))

dev = "cuda"
data = collect(ROOT)
sids = list(data.keys())[:N_SUBJECTS]
print(f"対象: {sids}  枚数={[len(data[s]) for s in sids]}", flush=True)

res = {}
for ti in (False, True):
    ext = Ext(ti).to(dev)
    stats = {s: (lambda f: (f.mean(0), np.cov(f, rowvar=False)))(feats(data[s], ext, dev)) for s in sids}
    fids = [fid(*stats[sids[i]], *stats[sids[j]]) for i in range(len(sids)) for j in range(i+1, len(sids))]
    res[ti] = (float(np.mean(fids)), fids)
    print(f"transform_input={ti!s:5s}  平均FID={np.mean(fids):.4f}  各ペア={[round(x,4) for x in fids]}", flush=True)
    del ext; torch.cuda.empty_cache()

a, b = res[False][0], res[True][0]
print(f"\n差: False={a:.4f} / True={b:.4f} / 差={a-b:+.4f} ({(a-b)/b*100:+.2f}%)")
print(f"参考: 全25人でのfaster(False)=31.8721 / v1・v2(True)=31.1590 / 差=+0.7131 (+2.29%)")
