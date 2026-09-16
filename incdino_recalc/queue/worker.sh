#!/bin/bash
# ジョブキューworker: queue/gpu$1 のジョブをファイル名順に直列実行し続ける（常駐）
Q=/home/kera/OpenGait/incdino_recalc/queue
G=$1
while true; do
  job=$(ls "$Q/gpu$G" 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then sleep 60; continue; fi
  mv "$Q/gpu$G/$job" "$Q/running/gpu$G-$job" || continue
  echo "$(date '+%F %T') START gpu$G $job" >> "$Q/queue.log"
  bash "$Q/running/gpu$G-$job" >> "$Q/queue.log" 2>&1
  rc=$?
  dest=done; [ $rc -ne 0 ] && dest=failed
  mv "$Q/running/gpu$G-$job" "$Q/$dest/gpu$G-$job"
  echo "$(date '+%F %T') END gpu$G $job rc=$rc -> $dest" >> "$Q/queue.log"
done
