#!/bin/bash
# Stage 2 of the jevhead line: re-score the cached competitive rivals with the SAME
# scalar head, so gap = z(rival) - z0 is a within-scorer difference. Waits for the
# z0 shards to finish so the two stages never contend for a GPU.
cd ~/SVD/agentic_probe
export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH
P=$HOME/.miniconda3/envs/grpo_ayb/bin/python
while pgrep -f "[c]ollect_jevhead.py --out jevhead_shard" > /dev/null; do sleep 60; done
echo "z0 shards done $(date -u +%FT%TZ)"
cat jevhead_shard*.jsonl > jevhead_all.jsonl
wc -l jevhead_all.jsonl
# main grid (13mod/paper canon) rivals, 3 shards
for i in 0 1 2; do
  GPU=$((i+1)); [ $i -eq 2 ] && GPU=4
  nohup $P collect_jevhead.py --rivals probe_upstream.jsonl \
      --out jevheadprobe_shard$i.jsonl --gpu $GPU --shard $i --nshards 3 \
      > jhr_$i.log 2>&1 &
done
# the re-provenanced model gets its own rival file from the 11rep cache
nohup $P collect_jevhead.py --rivals probe_11rep.jsonl --models qwen2.5-vl-7b \
    --out jevheadprobe_11rep.jsonl --gpu 5 > jhr_11rep.log 2>&1 &
wait
cat jevheadprobe_shard*.jsonl > jevheadprobe_all.jsonl
wc -l jevheadprobe_all.jsonl jevheadprobe_11rep.jsonl
echo "RIVALS_DONE $(date -u +%FT%TZ)"
