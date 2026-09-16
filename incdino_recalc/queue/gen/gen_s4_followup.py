#!/usr/bin/env python3
"""手順4 後続分（残り10サブセット×6指標）のスクリプトとキューファイルを生成する。

G-a（無改変性）: 各 s4_*.py は先行分テンプレート scripts_kera/s4_{type}_000-090.py から
dataset_roots の1行のみを置換して作る（knn_dinov2 は L163 修正済みの s4 版がテンプレート）。
キュー .sh は queue/done/gpu1-130_s4_knn_dinov2_000-090.sh と同じ形で、
ジョブ名・ログ名・device だけを差し替える。

既に完走済みで値が確定しているものは生成対象から外す:
  nm6(id13): 4種とも G-b 反復で3回確定
  nm2-bg2-cl2(id12) / nm1-2(id2): fidany_faster は手順1で完走済み
使い方: python3 gen_s4_followup.py [--write]   （--write なしは一覧表示のみ）
"""
import os, re, sys, stat

BASE = "/home/kera/OpenGait/incdino_recalc"
SK = os.path.join(BASE, "scripts_kera")
Q = os.path.join(BASE, "queue")
WRITE = "--write" in sys.argv

# id順（s4_fid_train_test_all13.py の train_roots と同じ並び）。視野角3本は先行分で実施済み
SUBSETS = ["20-data", "nm1-2", "bg1-2", "cl1-2", "nm1-bg1", "nm1-cl1", "bg1-cl1",
           "nm1-bg1-cl1", "nm2-bg2-cl2", "nm6"]
ROOT = "/data/CASIA-B-png/{}-png"
SKIP = {("nm6", t) for t in ("knn_inception", "knn_dinov2", "fidany_faster", "fidany_dinov2")}
SKIP |= {("nm2-bg2-cl2", "fidany_faster"), ("nm1-2", "fidany_faster")}

# レーン設計（REVIEW §8: CPU律速。sqrtm 系 FID は1レーンに直列、kNN は別レーン）
#   gpu1: kNN Inception ×9 → kNN DINOv2 ×9
#   gpu2: faster（穴G の id1/id3/id4 を先頭）→ dinov2 ×9   ※150_msdall_090-180 の後ろ
#   gpu0: 追加なし（030_msdall_000-090 が最後。CPU を空ける）
FASTER_ORDER = ["20-data", "bg1-2", "cl1-2", "nm1-bg1", "nm1-cl1", "bg1-cl1", "nm1-bg1-cl1"]
LANES = [
    (1, "knn_inception", SUBSETS), (1, "knn_dinov2", SUBSETS),
    (2, "fidany_faster", FASTER_ORDER), (2, "fidany_dinov2", SUBSETS),
]

SH = """#!/bin/bash
# 手順2 未完了なら実行しない（wait ジョブ失敗時に worker が後続を止めないため、各ジョブでも guard する）
grep -q STEP2_DONE /home/kera/OpenGait/incdino_recalc/logs/step2_build.log || {{ echo "ABORT: step2 not done"; exit 2; }}
L=/home/kera/OpenGait/incdino_recalc/logs/{job}.log
{{ echo "=== RUN_CONTEXT begin $(date '+%F %T %Z') ==="; echo "loadavg: $(cat /proc/loadavg)"; echo "concurrent containers:"; docker ps --format '  {{{{.Names}}}}\\t{{{{.Status}}}}'; echo "=== RUN_CONTEXT end ==="; }} >> "$L"
docker run --rm --gpus "device={gpu}" --name "incdino_q_{job}" \\
 -v /home/ryu/OpenGait/CASIA-B-png:/data/CASIA-B-png:ro \\
 -v /home/kera/incdino_data/CASIA-B-png-75:/data/CASIA-B-png-75:ro \\
 -v /home/kera/incdino_data/torch_cache:/root/.cache/torch \\
 -v /home/kera/OpenGait/incdino_recalc:/work -w /work casia-b-fid-3 \\
 bash -c 'S=$(date +%s); python -u scripts_kera/{job}.py >> logs/{job}.log 2>&1; RC=$?; E=$(date +%s); echo "JOB_DONE rc=$RC wall_s=$((E-S))" >> logs/{job}.log; chown -R 1012:1012 /work/results /work/logs /root/.cache/torch; exit $RC'
RC=$?
{{ echo "=== RUN_CONTEXT finish $(date '+%F %T %Z') ==="; echo "loadavg: $(cat /proc/loadavg)"; }} >> "$L"
exit $RC
"""

ROOTS_RE = re.compile(r'^(\s*dataset_roots = \[ )"/data/CASIA-B-png-75/000-090-png"( \]\s*)$', re.M)

def put(path, text, mode=0o775):
    if WRITE:
        with open(path, "w") as f: f.write(text)
        os.chmod(path, mode)

seq = {0: 200, 1: 200, 2: 200}
n_py = n_sh = 0
for gpu, typ, order in LANES:
    tmpl_path = os.path.join(SK, f"s4_{typ}_000-090.py")
    tmpl = open(tmpl_path).read()
    assert len(ROOTS_RE.findall(tmpl)) == 1, tmpl_path
    for sub in order:
        if (sub, typ) in SKIP:
            print(f"  skip  gpu{gpu} {typ:14s} {sub}  (完走済み)"); continue
        job = f"s4_{typ}_{sub}"
        py = ROOTS_RE.sub(lambda m: f'{m.group(1)}"{ROOT.format(sub)}"{m.group(2)}', tmpl)
        assert py != tmpl and py.count(ROOT.format(sub)) == 1
        py_path = os.path.join(SK, f"{job}.py")
        sh_name = f"{seq[gpu]:03d}_{job}.sh"; seq[gpu] += 10
        sh_path = os.path.join(Q, f"gpu{gpu}", sh_name)
        for p in (py_path, sh_path):
            if os.path.exists(p): sys.exit(f"exists: {p}")
        put(py_path, py, 0o664); put(sh_path, SH.format(job=job, gpu=gpu)); n_py += 1; n_sh += 1
        print(f"  {'WRITE' if WRITE else 'plan '} gpu{gpu}/{sh_name}  ->  scripts_kera/{job}.py")
print(f"{n_py} scripts, {n_sh} queue files ({'written' if WRITE else 'dry run'})")
