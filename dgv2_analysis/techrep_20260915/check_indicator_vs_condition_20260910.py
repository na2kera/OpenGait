"""各指標（生値）と条件別精度（scratch/ft の NM, CL）・CL含有フラグの Spearman ρ（n=13, 記述統計・検定なし）。
汎用: metrics_incdino_kera_X2.json、歩容特化: metrics_dgv2_sustech.json、ft: metrics_ft_sustech_full_20260726.json"""
import json
def rank(x):
    s=sorted(range(len(x)),key=lambda i:x[i]); r=[0]*len(x); i=0
    while i<len(s):
        j=i
        while j+1<len(s) and x[s[j+1]]==x[s[i]]: j+=1
        for k in range(i,j+1): r[s[k]]=(i+j)/2+1
        i=j+1
    return r
def sp(a,b):
    ra,rb=rank(a),rank(b); n=len(a); ma=sum(ra)/n; mb=sum(rb)/n
    num=sum((p-ma)*(q-mb) for p,q in zip(ra,rb)); den=(sum((p-ma)**2 for p in ra)*sum((q-mb)**2 for q in rb))**.5
    return num/den
mi=json.load(open('/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis/metrics_incdino_kera_X2.json'))
ms=json.load(open('../metrics_dgv2_sustech.json'))
mf=json.load(open('../metrics_ft_sustech_full_20260726.json'))
ks=list(ms.keys()); name={k:ms[k]['subset_name'] for k in ks}
cl_in={'default':1,'nm2':0,'bg2':0,'cl2':1,'nm1-bg1':0,'nm1-cl1':1,'bg1-cl1':1,'000-180':1,'000-090':1,'090-180':1,'nm1-bg1-cl1':1,'nm2-bg2-cl2':1,'nm6':0}
tgt={'scratch_CL':[ms[k]['acc_cl'] for k in ks],'ft_CL':[mf[name[k]]['acc_cl'] for k in ks],
     'scratch_NM':[ms[k]['acc_nm'] for k in ks],'ft_NM':[mf[name[k]]['acc_nm'] for k in ks],
     'CL_in_train':[cl_in[name[k]] for k in ks]}
gen=['Inception_MSD','Inception_1NN','Inception_kNN','Inception_FID','FID_train_test_Inception','DINO_MSD','DINO_1NN','DINO_kNN','DINO_FID','FID_train_test_DINO']
gait=['DeepGaitV2_MSD','DeepGaitV2_1NN','DeepGaitV2_kNN','DeepGaitV2_FID','FID_train_test_DeepGaitV2']
out=['# 指標（生値）と条件別精度の Spearman ρ（n=13、記述統計）\n','生成: check_indicator_vs_condition_20260910.py（2026-09-10）。ridge 係数ではなく生の順位相関。CL_in_train は学習データに CL 系列を含むか (0/1)。\n',
     '| 指標 | '+' | '.join(tgt)+' |','|---|'+'---:|'*len(tgt)]
for v in gen+gait:
    src=mi if v in gen else ms; x=[src[k][v] for k in ks]
    out.append(f'| {v} | '+' | '.join(f'{sp(x,tgt[t]):+.2f}' for t in tgt)+' |')
out+=['\n## サブセット別の生値\n','| subset | CL含 | Inception_MSD | DINO_MSD | DeepGaitV2_MSD | FIDtt_inc | FIDtt_dgv2 | scratch CL | ft CL | scratch NM |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for k in ks:
    n=name[k]; out.append(f"| {n} | {cl_in[n]} | {mi[k]['Inception_MSD']:.4f} | {mi[k]['DINO_MSD']:.3f} | {ms[k]['DeepGaitV2_MSD']:.2f} | {mi[k]['FID_train_test_Inception']:.2f} | {ms[k]['FID_train_test_DeepGaitV2']:.2f} | {ms[k]['acc_cl']:.3f} | {mf[n]['acc_cl']:.3f} | {ms[k]['acc_nm']:.3f} |")
def rng(v,src): xs=[src[k][v] for k in ks]; return min(xs),max(xs),(max(xs)-min(xs))/(sum(xs)/len(xs))
out+=['\n## クラス内分散 (MSD) のサブセット間変動幅（(max−min)/mean）\n']
for v,src in [('Inception_MSD',mi),('DINO_MSD',mi),('DeepGaitV2_MSD',ms)]:
    lo,hi,r=rng(v,src); out.append(f'- {v}: {lo:.4g} 〜 {hi:.4g}（相対幅 {r:.1%}）')
open('indicator_vs_condition_20260910.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out[-4:]))
