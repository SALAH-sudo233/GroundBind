#!/bin/bash
# Text-routed probes for the TWO coordinate-corrected models only.
# canon_roots_cfonly.json contains just UniVG-R1 + visual-rft, so we do not
# re-probe the other 11 models (their scores already exist).
# Other users hold most GPUs -> strictly serial, one card at a time.
set -uo pipefail
cd "$HOME/SVD/agentic_probe"
export LD_LIBRARY_PATH="$HOME/.miniconda3/envs/grpo_ayb/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$HOME/.miniconda3/envs/grpo_ayb/bin/python"
G=$(python3 -c "
import subprocess
o=subprocess.run([\"nvidia-smi\",\"--query-gpu=index,memory.used\",\"--format=csv,noheader,nounits\"],capture_output=True,text=True).stdout
r=sorted((int(x.split(\",\")[1]),int(x.split(\",\")[0])) for x in o.strip().split(chr(10)))
print(r[0][1])
")
echo "using GPU $G"
CUDA_VISIBLE_DEVICES=$G $PY collect_probe_textroute.py \
  --canon canon_roots_cfonly.json --out probecf_omni.jsonl > cf_probe_omni.log 2>&1
echo "OMNI_CF_PROBE_DONE lines=$(wc -l < probecf_omni.jsonl 2>/dev/null || echo 0)"
$PY collect_jevhead.py --gpu $G --canon canon_roots_cfonly.json \
  --models UniVG-R1,visual-rft --rivals probecf_omni.jsonl \
  --out probecf_jev.jsonl > cf_probe_jev.log 2>&1
echo "JEV_CF_PROBE_DONE lines=$(wc -l < probecf_jev.jsonl 2>/dev/null || echo 0)"
