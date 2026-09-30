#!/bin/bash
# Third CBR line: jev_verifier_v2 scalar head (calibrated, T=1.7766) over the FULL
# 2500-row grid per model, paper-aligned canonical runs. 3 shards on the GPUs left
# free by the r1probe / jevz11 jobs. Resumable by (model,sid).
cd ~/SVD/agentic_probe
export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH
P=$HOME/.miniconda3/envs/grpo_ayb/bin/python
for i in 0 1 2; do
  GPU=$((i+1))
  [ $i -eq 2 ] && GPU=4
  nohup $P collect_jevhead.py --out jevhead_shard$i.jsonl --gpu $GPU \
      --shard $i --nshards 3 > jevhead_$i.log 2>&1 &
done
echo launched
