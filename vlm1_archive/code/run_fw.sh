export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH
cd $HOME/SVD/agentic_probe
$HOME/.miniconda3/envs/grpo_ayb/bin/python run_framework.py \
  --out fw_all4.jsonl --device cuda:1 \
  --htypes object,co_occurrence,attribute,relation > fw.log 2>&1
