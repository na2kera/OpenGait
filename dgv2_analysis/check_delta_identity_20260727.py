#!/usr/bin/env python3
"""Delta(=ft-scratch)のLODO予測が「ft予測 - scratch予測」と厳密に一致するかの検算 (2026-07-27)。
一致するなら Δ回帰は2つの精度回帰の線形帰結であり、独立した証拠にはならない。"""
import json, math, os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ID2SUBSET = {1:"default",2:"nm2",3:"bg2",4:"cl2",5:"nm1-bg1",6:"nm1-cl1",7:"bg1-cl1",
             8:"000-180",9:"000-090",10:"090-180",11:"nm1-bg1-cl1",12:"nm2-bg2-cl2",13:"nm6"}
FEATS = ["log_people","DeepGaitV2_MSD","DeepGaitV2_1NN","DeepGaitV2_kNN",
         "DeepGaitV2_FID","FID_train_test_DeepGaitV2"]

m = json.load(open(os.path.join(HERE, "metrics_dgv2_sustech.json")))
ft = json.load(open(os.path.join(HERE, "metrics_ft_sustech_full_20260726.json")))

X, ys, yf = [], [], []
for i in range(1, 14):
    s = ID2SUBSET[i]
    d = dict(m[str(i)])
    d["log_people"] = math.log(d["Number of people"])
    X.append([d[k] for k in FEATS])
    ys.append(d["acc_all"] * 100)
    yf.append(ft[s]["acc_all"] * 100)
X, ys, yf = np.array(X), np.array(ys), np.array(yf)
yd = yf - ys

def lodo(y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        sc = StandardScaler().fit(X[tr])
        p[i] = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[i:i+1]))[0]
    return p

ps, pf, pd_ = lodo(ys), lodo(yf), lodo(yd)
print(f"MAE  scratch={np.abs(ps-ys).mean():.3f}pt  ft={np.abs(pf-yf).mean():.3f}pt  delta={np.abs(pd_-yd).mean():.3f}pt")
print(f"max|delta_pred - (ft_pred - scratch_pred)| = {np.abs(pd_-(pf-ps)).max():.3e}")
print(f"厳密一致か: {np.allclose(pd_, pf-ps, atol=1e-8)}")
print(f"誤差の相関 corr(scratch誤差, ft誤差) = {np.corrcoef(ps-ys, pf-yf)[0,1]:.3f}")
