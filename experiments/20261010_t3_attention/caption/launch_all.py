#!/usr/bin/env python3
"""Launch only GPUs 4--7 for the authorized T3 supplemental experiment."""
import json, os, pathlib, subprocess, time
out=pathlib.Path('/home/u2025141034/benchmark/rrwr_20261010_t3_attention/t3')
python='/home/u2025141034/.miniconda3/envs/mllm_ayb/bin/python'
held=json.loads((out/'heldout_summary.json').read_text())
assert held['n']==64 and held['n_valid']==64
assert held['accuracy_all_controls'] >= .95, 'Independent semantic control accuracy below 95%; do not launch full evaluation.'
ledger=[]
for shard,gpu in enumerate([4,5,6,7]):
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'LD_LIBRARY_PATH':'/home/u2025141034/.miniconda3/envs/mllm_ayb/lib','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
    cmd=[python,'-u',str(out/'t3_semantic_eval.py'),'run','--shard',str(shard),'--nshards','4','--batch','8']
    log=out/f'run_shard_{shard}.log'
    f=open(log,'w')
    proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    f.close()
    ledger.append({'shard':shard,'physical_gpu':gpu,'pid':proc.pid,'log':str(log),'started_epoch':time.time(),'argv':cmd})
(out/'launch_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps(ledger))
