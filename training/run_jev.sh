#!/bin/bash
set -uo pipefail
cd "$HOME/SVD/grpo_verifier"
export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
OUT="$HOME/SVD/grpo_verifier/runs/jev_verifier_v2"
if [ -f "$OUT/head_final.pt" ]; then echo "SKIP (already done)"; exit 0; fi
echo "JEV_START $(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES=1 $HOME/.miniconda3/envs/grpo_ayb/bin/python jev_verifier.py \
  --steps 1000 --bs 4 --accum 4 --lr 5e-5 --head_lr 1e-4 --brier_w 0.1 \
  --n_cal 512 --out "$OUT" > $HOME/jev_verifier_v2.log 2>&1
echo "JEV_DONE rc=$? $(date -u +%FT%TZ)"
