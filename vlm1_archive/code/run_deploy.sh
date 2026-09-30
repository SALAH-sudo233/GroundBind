export LD_LIBRARY_PATH=$HOME/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH
cd $HOME/SVD/agentic_probe
$HOME/.miniconda3/envs/grpo_ayb/bin/python deploy_upstream.py \
  --models InternVL3.5-8B,LENS,Orsta-7B,Qwen3-VL-8B,Seg-R1,Seg-zero,TreeVGR,UniVG-R1,Vision-R1,VisionReasoner,llava-ov-7b,qwen2.5-vl-7b,visual-rft \
  --out probe_upstream.jsonl --gpu 1 > deploy.log 2>&1
