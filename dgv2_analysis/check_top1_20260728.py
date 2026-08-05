import json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
raw = json.load(open(os.path.join(HERE,'metrics_dgv2_sustech.json')))
ft  = json.load(open(os.path.join(HERE,'metrics_ft_sustech_full_20260726.json')))
FEATS=['log_people','DeepGaitV2_MSD','DeepGaitV2_1NN','DeepGaitV2_kNN','DeepGaitV2_FID','FID_train_test_DeepGaitV2']
rows=[]
for sid,s in raw.items():
    sub=s['subset_name']
    r=dict(subset=sub, log_people=math.log(s['Number of people']),
           scratch=s['acc_all'], ft=ft[sub]['acc_all'])
    for k in FEATS[1:]: r[k]=s[k]
    rows.append(r)
X=np.array([[r[f] for f in FEATS] for r in rows])
def lodo(y):
    preds=[]
    for i in range(len(y)):
        idx=[j for j in range(len(y)) if j!=i]
        Xtr,ytr=X[idx],y[idx]
        mu,sd=Xtr.mean(0),Xtr.std(0)
        Zt=(Xtr-mu)/sd; zi=(X[i]-mu)/sd
        w=np.linalg.solve(Zt.T@Zt+np.eye(len(FEATS)), Zt.T@(ytr-ytr.mean()))
        preds.append(float(zi@w+ytr.mean()))
    return np.array(preds)
for tgt in ['scratch','ft']:
    y=np.array([r[tgt] for r in rows]); p=lodo(y)
    ot=np.argsort(-y); op=np.argsort(-p)
    print(f'=== {tgt} ===  MAE={np.mean(np.abs(p-y))*100:.2f}pt')
    print('  真の上位3 :', [rows[i]['subset'] for i in ot[:3]])
    print('  予測上位3 :', [rows[i]['subset'] for i in op[:3]])
    print('  真の下位2 :', [rows[i]['subset'] for i in ot[-2:]])
    print('  予測下位2 :', [rows[i]['subset'] for i in op[-2:]])
    rt={rows[i]['subset']:k+1 for k,i in enumerate(ot)}
    rp={rows[i]['subset']:k+1 for k,i in enumerate(op)}
    print('  順位ずれ:', {s:(rt[s],rp[s]) for s in rt if rt[s]!=rp[s]})
