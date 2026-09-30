export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH
cd $HOME/SVD/agentic_probe
$HOME/.miniconda3/envs/grpo_ayb/bin/python collect_actions.py \
  --out actions_relation.jsonl --device cuda:1 > collect.log 2>&1
