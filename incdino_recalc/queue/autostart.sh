#!/bin/bash
# サーバー再起動時にジョブキューを自動復旧する（crontab @reboot から起動）
# やること: docker/GPU準備待ち → 中断ジョブ(running/)の再キュー＋書きかけログ退避 → tmuxワーカー3本起動
Q=/home/kera/OpenGait/incdino_recalc/queue
exec >> "$Q/autostart.log" 2>&1
echo "=== $(date '+%F %T') autostart invoked ==="

# 既にセッションが生きていれば何もしない（手動起動と共存可）
if tmux has-session -t incdino_queue 2>/dev/null; then
  echo "session already running; nothing to do"
  exit 0
fi

# dockerとGPUドライバが応答するまで待つ（最大10分）
for i in $(seq 60); do
  docker info >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && break
  sleep 10
done
if ! docker info >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "$(date '+%F %T') docker/GPU not ready after 10min; abort"
  exit 1
fi

# 前回の残骸コンテナを掃除（--name固定のため残っていると再実行が失敗する）
docker ps -aq --filter name=incdino_q_ | xargs -r docker rm -f

# 中断ジョブをキューへ戻す。追記式ログは退避して再実行結果との混在を防ぐ
ts=$(date +%Y%m%d-%H%M)
for f in "$Q"/running/gpu*-*.sh; do
  [ -e "$f" ] || continue
  b=$(basename "$f"); g=${b%%-*}; job=${b#*-}
  for lg in $(grep -o 'logs/[A-Za-z0-9._-]*\.log' "$f" | sort -u); do
    [ -f "$Q/../$lg" ] && mv "$Q/../$lg" "$Q/../$lg.interrupted-$ts"
  done
  mv "$f" "$Q/$g/$job"
  echo "$(date '+%F %T') NOTE autostart requeued $g/$job after reboot (partial logs -> *.interrupted-$ts)" >> "$Q/queue.log"
done

tmux new-session -d -s incdino_queue -n gpu0 "bash $Q/worker.sh 0"
tmux new-window  -t incdino_queue -n gpu1 "bash $Q/worker.sh 1"
tmux new-window  -t incdino_queue -n gpu2 "bash $Q/worker.sh 2"
echo "$(date '+%F %T') workers started (tmux session incdino_queue)"
