#!/bin/bash
cd ~/SVD/agentic_probe
export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH
P=$HOME/.miniconda3/envs/grpo_ayb/bin/python
for i in 0 1 2 3; do
  GPU=$((i+1))
  nohup $P collect_2b_select.py --out sel2b_shard$i.jsonl --gpu $GPU \
      --shard $i --nshards 4 > sel2b_$i.log 2>&1 &
done
echo launched
