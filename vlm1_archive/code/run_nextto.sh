export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH
cd $HOME/SVD/agentic_probe
$HOME/.miniconda3/envs/grpo_ayb/bin/python probe_nextto.py \
  --out nextto_rivals.jsonl --device cuda:1 > nextto.log 2>&1
