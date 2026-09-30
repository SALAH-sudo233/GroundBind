#!/bin/bash
# Label-independent (query-text) probe collection, OMNI verifier.
# GPU policy: another user runs scripts/stage7 on GPU 1/3/5/7 -> we use ONLY 0/2/4/6.
set -uo pipefail
cd "$HOME/SVD/agentic_probe"
export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
PY="$HOME/.miniconda3/envs/grpo_ayb/bin/python"
GPUS=(0 2 4 6)
for s in 0 1 2 3; do
  g=${GPUS[$s]}
  CUDA_VISIBLE_DEVICES=$g nohup $PY collect_probe_textroute.py \
    --out trprobe_omni_shard$s.jsonl \
    --have probe_upstream.jsonl probe_11rep.jsonl \
    --shard $s --nshard 4 > tr_omni_$s.log 2>&1 &
done
wait
cat trprobe_omni_shard*.jsonl > trprobe_omni_all.jsonl
echo "OMNI_TEXTROUTE_DONE $(date -u +%FT%TZ) lines=$(wc -l < trprobe_omni_all.jsonl)"
