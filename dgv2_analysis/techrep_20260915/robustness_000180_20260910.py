#!/usr/bin/env python3
"""000-180 を除いた 12 サブセットでの順位相関・MAE（LODO 予測値をそのまま使った再計算。再学習はしない）。
統計レビュー（9/10）M5 の材料。出典: X2 grid の per_subset（S_incdino11 / S_inc6）、X2 A_sustech6_ft_reproduced（S_sustech6 ft）、
presentation_tables_20260726.json lodo_predictions（S_sustech6 scratch）。"""
import json, os
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
x2 = json.load(open('/home/kera/worktrees/incdino-ft-phase3/dgv2_analysis/result_incdino_ft_20260806_X2.json'))
pres = json.load(open('/app/OpenGait/dgv2_analysis/presentation_tables_20260726.json' if os.path.exists('/app/OpenGait') else '/home/kera/OpenGait/dgv2_analysis/presentation_tables_20260726.json'))
series = {}
for fs in ('S_incdino11', 'S_inc6'):
    for tgt in ('scratch_all', 'ft_all'):
        ps = x2['grid'][fs][tgt]['per_subset']; series[(fs, tgt)] = {k: (v['true'] * 100, v['pred'] * 100) for k, v in ps.items()}
ps = pres['lodo_predictions']['DGV2(6変数)']['per_subset']; series[('S_sustech6', 'scratch_all')] = {k: (v['true_pct'], v['pred_pct']) for k, v in ps.items()}
rep = x2['A_sustech6_ft_reproduced']
ps = rep.get('per_subset'); assert ps, list(rep.keys())
series[('S_sustech6', 'ft_all')] = {k: (v['true'] * 100, v['pred'] * 100) for k, v in ps.items()}
out = {}
for key, d in series.items():
    subs = list(d); t = np.array([d[s][0] for s in subs]); p = np.array([d[s][1] for s in subs])
    keep = np.array([s != '000-180' for s in subs])
    full_rho, full_mae = spearmanr(t, p)[0], np.mean(np.abs(t - p))
    r12, m12 = spearmanr(t[keep], p[keep])[0], np.mean(np.abs(t[keep] - p[keep]))
    share = np.abs(t - p)[~keep][0] / np.sum(np.abs(t - p))
    out['%s/%s' % key] = {'rho_13': float(full_rho), 'mae_13': float(full_mae), 'rho_12_wo_000-180': float(r12), 'mae_12_wo_000-180': float(m12), 'err_share_000-180': float(share), 'sd_target_13': float(np.std(t, ddof=1))}
    print('%-24s rho %.3f -> %.3f  MAE %.2f -> %.2f  share %.1f%%  SD %.2f' % ('%s/%s' % key, full_rho, r12, full_mae, m12, share * 100, np.std(t, ddof=1)))
json.dump(out, open(os.path.join(HERE, 'robustness_000180_20260910.json'), 'w'), ensure_ascii=False, indent=1)
