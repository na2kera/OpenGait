#!/usr/bin/env python3
"""s5（汎用特徴の式(1) MSD 再計測）のスクリプト 13 本とキュー .sh を生成する（2026-09-14）。

- scripts_kera/s5_msdclass_<subset>.py = TEMPLATE の SUBSET / ROOT の 2 行置換
- レーン: gpu0 = nm6（パイロット）→ 20-data, nm1-2, bg1-2, cl1-2
          gpu1 = wait(パイロット完了) → nm1-bg1, nm1-cl1, bg1-cl1, nm1-bg1-cl1
          gpu2 = wait(パイロット完了) → nm2-bg2-cl2, 000-180, 000-090, 090-180（75 人版ルート）
- パイロット nm6 のゲート（全体分散が採用値と一致）が通らなければ、他 12 本は各ジョブ冒頭の guard で abort する
- 視野角 3 本は /data/CASIA-B-png-75、他は /data/CASIA-B-png（gen_s4_followup.py と同じ）
使い方: python3 gen_s5_msdclass.py [--write]（--write なしは一覧表示のみ）

既知の制約（Codex レビュー 2026-09-14）:
  - パイロット判定は追記式ログ全体の grep なので、**再実行時**は旧ログの S5_PILOT_PASS / JOB_DONE で通ってしまう。
    再実行する場合は logs/s5_msdclass_*.log と results/MSDclass_results_* を退避（*.old-<日付>）してから生成すること。
  - 同名ジョブが queue/{gpu*,running,hold,done,failed} のどこかにあれば --write は失敗する（二重投入防止。初回 9/14 04:29 投入済み）。
"""
import os, sys, stat

BASE = "/home/kera/OpenGait/incdino_recalc"
SK = os.path.join(BASE, "scripts_kera")
Q = os.path.join(BASE, "queue")
WRITE = "--write" in sys.argv
TEMPLATE = os.path.join(SK, "s5_msdclass_TEMPLATE.py")

ROOT75 = {"000-180", "000-090", "090-180"}
def root(s):
    return f"/data/CASIA-B-png-75/{s}-png" if s in ROOT75 else f"/data/CASIA-B-png/{s}-png"

PILOT = "nm6"
LANES = {0: ["20-data", "nm1-2", "bg1-2", "cl1-2"],
         1: ["nm1-bg1", "nm1-cl1", "bg1-cl1", "nm1-bg1-cl1"],
         2: ["nm2-bg2-cl2", "000-180", "000-090", "090-180"]}

GUARD_STEP2 = 'grep -q STEP2_DONE /home/kera/OpenGait/incdino_recalc/logs/step2_build.log || { echo "ABORT: step2 not done"; exit 2; }'
GUARD_PILOT = 'grep -q "S5_PILOT_PASS" /home/kera/OpenGait/incdino_recalc/logs/s5_msdclass_nm6.log || { echo "ABORT: s5 pilot (nm6) gate not passed"; exit 2; }'

SH = """#!/bin/bash
{guards}
L=/home/kera/OpenGait/incdino_recalc/logs/{job}.log
{{ echo "=== RUN_CONTEXT begin $(date '+%F %T %Z') ==="; echo "loadavg: $(cat /proc/loadavg)"; echo "concurrent containers:"; docker ps --format '  {{{{.Names}}}}\\t{{{{.Status}}}}'; echo "=== RUN_CONTEXT end ==="; }} >> "$L"
docker run --rm --gpus "device={gpu}" --shm-size=4g --name "incdino_q_{job}" \\
 -e S5_WORKERS=5 \\
 -v /home/ryu/OpenGait/CASIA-B-png:/data/CASIA-B-png:ro \\
 -v /home/kera/incdino_data/CASIA-B-png-75:/data/CASIA-B-png-75:ro \\
 -v /home/kera/incdino_data/torch_cache:/root/.cache/torch \\
 -v /home/kera/incdino_data/msdclass_features:/feat \\
 -v /home/kera/OpenGait/incdino_recalc:/work -w /work casia-b-fid-3 \\
 bash -c 'S=$(date +%s); python -u scripts_kera/{job}.py >> logs/{job}.log 2>&1; RC=$?; E=$(date +%s); echo "JOB_DONE rc=$RC wall_s=$((E-S))" >> logs/{job}.log; {pilot_mark}chown -R 1012:1012 /work/results /work/logs /root/.cache/torch /feat; exit $RC'
RC=$?
{{ echo "=== RUN_CONTEXT finish $(date '+%F %T %Z') ==="; echo "loadavg: $(cat /proc/loadavg)"; }} >> "$L"
exit $RC
"""
PILOT_MARK = '[ $RC -eq 0 ] && grep -q "gate_all_pass=True" logs/{job}.log && echo "S5_PILOT_PASS $(date +%F_%T)" >> logs/{job}.log; '

WAIT = """#!/bin/bash
# パイロット（gpu0 の s5_msdclass_nm6）の完了を待つ。失敗していても exit 0 で抜け、後続ジョブ側の guard で abort させる
L=/home/kera/OpenGait/incdino_recalc/logs/s5_msdclass_nm6.log
for i in $(seq 720); do
  grep -q "JOB_DONE" "$L" 2>/dev/null && break
  sleep 60
done
grep -q "S5_PILOT_PASS" "$L" 2>/dev/null && echo "pilot passed" || echo "pilot not passed (or timeout 12h); followers will abort"
exit 0
"""

def existing_jobs():
    found = set()
    for d in ("gpu0", "gpu1", "gpu2", "running", "hold", "done", "failed"):
        for r, _, files in os.walk(os.path.join(Q, d)):
            for f in files:
                if "s5_msdclass" in f or "wait_s5_pilot" in f:
                    found.add(os.path.join(os.path.relpath(r, Q), f))
    return sorted(found)

if WRITE and existing_jobs():
    sys.exit("ABORT: s5 ジョブが既にキュー内に存在する（二重投入防止）:\n  " + "\n  ".join(existing_jobs()))

def put(path, text, mode=0o775):
    if WRITE:
        with open(path, "w") as f: f.write(text)
        os.chmod(path, mode)
    print(("WRITE " if WRITE else "plan  ") + os.path.relpath(path, BASE))

tpl = open(TEMPLATE).read()
assert '"__SUBSET__"' in tpl and '"__ROOT__"' in tpl

def gen_py(subset):
    txt = tpl.replace('"__SUBSET__"', f'"{subset}"', 1).replace('"__ROOT__"', f'"{root(subset)}"', 1)
    assert txt.count(f'"{subset}"') >= 1 and root(subset) in txt
    put(os.path.join(SK, f"s5_msdclass_{subset}.py"), txt, 0o664)

def gen_sh(gpu, order, subset, pilot=False):
    job = f"s5_msdclass_{subset}"
    guards = GUARD_STEP2 if pilot else GUARD_STEP2 + "\n" + GUARD_PILOT
    put(os.path.join(Q, f"gpu{gpu}", f"{order:03d}_{job}.sh"),
        SH.format(job=job, gpu=gpu, guards=guards, pilot_mark=PILOT_MARK.format(job=job) if pilot else ""))

# gpu0: パイロット + 4 本
gen_py(PILOT); gen_sh(0, 400, PILOT, pilot=True)
for i, s in enumerate(LANES[0]):
    gen_py(s); gen_sh(0, 410 + 10 * i, s)
for g in (1, 2):
    put(os.path.join(Q, f"gpu{g}", f"400_wait_s5_pilot.sh"), WAIT)
    for i, s in enumerate(LANES[g]):
        gen_py(s); gen_sh(g, 410 + 10 * i, s)
print("total subsets:", 1 + sum(len(v) for v in LANES.values()))
