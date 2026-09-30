#!/bin/bash
# Re-score qwen2.5-vl-7b against the 11rep canonical run so all 13 models share
# the v0.48 Table 4 provenance (that model is the ONLY one whose canon disagreed:
# 13mod n_c=247 vs paper/11rep n_c=242, and its existing z0 was measured on 13mod
# boxes -- 500/500 IoU match vs 138/500 for 11rep -- so the root cannot simply be
# repointed without re-scoring, that would mix runs).
# Distinct tags (omni11 / jevz11) so the 13mod files are never appended to.
set -uo pipefail
cd "$HOME/SVD/grpo_verifier"
export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:${LD_LIBRARY_PATH:-}"
P="$HOME/.miniconda3/envs/grpo_ayb/bin/python"
R11="/home/u2025141034/benchmark/refcocog_eval_11models_500_repaired/run_500_semantic_strict"
Q2="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
CK="$HOME/SVD/grpo_verifier/runs/largebatch_q35_2b_R1/v0-20260916-093618/checkpoint-400"

nohup "$P" s5_omni_filter.py --models qwen2.5-vl-7b --tag omni11 --root "$R11" --gpu 0 \
  > ~/SVD/agentic_probe/q25_omni11.log 2>&1 &
nohup "$P" s5_jev_logits.py --models qwen2.5-vl-7b --verifier_model "$Q2" --lora "$CK" \
  --tag q35_2b --out_tag jevz11 --root "$R11" --gpu 3 \
  > ~/SVD/agentic_probe/q25_jevz11.log 2>&1 &
cd ~/SVD/agentic_probe
nohup "$P" deploy_upstream.py --models qwen2.5-vl-7b --canon canon_roots_paper.json \
  --out probe_11rep.jsonl --gpu 4 > q25_probe11.log 2>&1 &
echo launched
