#!/bin/bash
cd ~/SVD/agentic_probe
export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH
P=$HOME/.miniconda3/envs/grpo_ayb/bin/python
for i in 0 1 2; do
  GPU=$((i+5))
  nohup $P collect_aswap.py --out swap_shard$i.jsonl --gpu $GPU \
      --shard $i --nshards 3 > swap_$i.log 2>&1 &
done
echo launched
