#!/bin/bash
# JEV scalar head, same query-text router: re-score the SAME rival texts with THIS scorer.
# GPU policy: another user holds 1/3/5/7 -> use ONLY 0/2/4/6.
set -uo pipefail
cd "$HOME/SVD/agentic_probe"
export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
PY="$HOME/.miniconda3/envs/grpo_ayb/bin/python"
GPUS=(0 2 4 6)
for s in 0 1 2 3; do
  g=${GPUS[$s]}
  nohup $PY collect_jevhead.py --gpu $g \
    --rivals probe_textroute_all.jsonl \
    --out trprobe_jev_shard$s.jsonl \
    --shard $s --nshards 4 > tr_jev_$s.log 2>&1 &
done
wait
cat trprobe_jev_shard*.jsonl > trprobe_jev_all.jsonl
echo "JEV_TEXTROUTE_DONE $(date -u +%FT%TZ) lines=$(wc -l < trprobe_jev_all.jsonl)"
