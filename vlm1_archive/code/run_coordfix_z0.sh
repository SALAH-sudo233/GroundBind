#!/bin/bash
# z0 for the coordinate-corrected run: OMNI (GPU 0,2) + JEV head (GPU 4,6).
# Distinct tags/filenames; the original arm files are never touched.
set -uo pipefail
cd "$HOME/SVD/agentic_probe"
export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
PY="$HOME/.miniconda3/envs/grpo_ayb/bin/python"
CF="$HOME/benchmark/coordfix_500"

cd "$HOME/SVD/grpo_verifier"
CUDA_VISIBLE_DEVICES=0 nohup $PY s5_omni_filter.py --models UniVG-R1 \
  --tag omnicf --gpu 0 --root "$CF" > $HOME/SVD/agentic_probe/cf_omni_univg.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup $PY s5_omni_filter.py --models visual-rft \
  --tag omnicf --gpu 2 --root "$CF" > $HOME/SVD/agentic_probe/cf_omni_vrft.log 2>&1 &

cd "$HOME/SVD/agentic_probe"
nohup $PY collect_jevhead.py --gpu 4 --canon canon_roots_coordfix.json \
  --models UniVG-R1 --out jevheadcf_UniVG-R1.jsonl > cf_jev_univg.log 2>&1 &
nohup $PY collect_jevhead.py --gpu 6 --canon canon_roots_coordfix.json \
  --models visual-rft --out jevheadcf_visual-rft.jsonl > cf_jev_vrft.log 2>&1 &
wait
echo "COORDFIX_Z0_DONE $(date -u +%FT%TZ)"
