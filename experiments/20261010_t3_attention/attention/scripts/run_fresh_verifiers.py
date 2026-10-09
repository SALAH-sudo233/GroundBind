"""Score freshly generated candidates with the original frozen verifier protocols."""
from pathlib import Path
import json,sys,argparse,hashlib,time,re,traceback
import torch,numpy as np
from PIL import Image
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor,AutoModelForImageTextToText
ROOT=Path('/home/u2025141034/benchmark/rrwr_20261010_t3_attention/attention')
SRC=Path('/home/u2025141034/SVD/agentic_probe');sys.path[:0]=[str(SRC),'/home/u2025141034/SVD/grpo_verifier']
import s5_omni_filter as O
import collect_jevhead as J
import simple_relations as S
import eval_attribution as T
import eval_paper_tables as P
ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['omni','jev'],required=True);a=ap.parse_args()
out=ROOT/'verifiers'/a.arm;out.mkdir(parents=True,exist_ok=True)
start=time.monotonic()
if a.arm=='omni':
    model_path='/home/u2025141034/models/OmniVerifier-7B'
    proc=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    net=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,dtype=torch.bfloat16,attn_implementation='eager',device_map='cuda:0',local_files_only=True).eval()
    ids_true=[];ids_false=[]
    for s in ('true',' true','True',' True'):ids_true+=proc.tokenizer.encode(s,add_special_tokens=False)[:1]
    for s in ('false',' false','False',' False'):ids_false+=proc.tokenizer.encode(s,add_special_tokens=False)[:1]
    ids_true=sorted(set(ids_true));ids_false=sorted(set(ids_false))
else:
    from peft import PeftModel
    model_path=J.BASE
    proc=AutoProcessor.from_pretrained(model_path,local_files_only=True,trust_remote_code=True,min_pixels=256*28*28,max_pixels=768*28*28)
    base=AutoModelForImageTextToText.from_pretrained(model_path,dtype=torch.bfloat16,device_map='cuda:0',trust_remote_code=True,local_files_only=True,attn_implementation='eager')
    net=PeftModel.from_pretrained(base,str(Path(J.JEV)/'adapter_final')).eval()
    hidden=net.config.get_text_config().hidden_size
    head=torch.nn.Linear(hidden,1,dtype=torch.float32).to('cuda:0')
    ck=torch.load(str(Path(J.JEV)/'head_final.pt'),map_location='cuda:0',weights_only=False);head.load_state_dict(ck['head']);head.eval()
    temperature=float(ck.get('temperature') or 0) or json.loads((Path(J.JEV)/'temperature.json').read_text())['temperature']
print(json.dumps({'event':'loaded','arm':a.arm,'seconds':time.monotonic()-start,'model_path':model_path,'allocated_gb':torch.cuda.memory_allocated()/1e9}),flush=True)
def get_inputs(ip,bbox,query):
    if a.arm=='omni':
        image=O.draw_box(str(ip),bbox)
        prompt=O.Omni.question(None,query)
        msgs=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        rendered=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)+'{"answer":'
    else:
        image=Image.open(ip).convert('RGB');b=[int(round(v)) for v in bbox]
        prompt=J.PROMPT.format(x0=b[0],y0=b[1],x1=b[2],y1=b[3],q=query)
        msgs=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        rendered=proc.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False,enable_thinking=False)
    inp=proc(text=[rendered],images=[image],return_tensors='pt')
    return {k:v.to('cuda:0') if hasattr(v,'to') else v for k,v in inp.items()},prompt,rendered,image.size
def score(ip,bbox,query,save_attention=False,stem=None):
    inp,prompt,rendered,processed_size=get_inputs(ip,bbox,query)
    with torch.inference_mode():
        result=net(**inp,use_cache=False,output_hidden_states=a.arm=='jev',output_attentions=save_attention,return_dict=True)
    if a.arm=='omni':
        logits=result.logits[0,-1].float();z=float(torch.logsumexp(logits[ids_true],0)-torch.logsumexp(logits[ids_false],0))
    else:z=float(head(result.hidden_states[-1][:,-1,:].float()).squeeze(-1)[0])
    meta={'z':z,'prompt':prompt,'rendered_prompt':rendered,'prompt_token_count':int(inp['input_ids'].shape[1]),'model_path':model_path,'image_input_size_wh':list(processed_size),'coordinate_protocol':'draw actual red rectangle, original Omni protocol' if a.arm=='omni' else 'original JEV textual pixel coordinates, integer-rounded identically to collection script'}
    if a.arm=='jev':meta.update({'temperature':temperature,'p_head':float(torch.sigmoid(torch.tensor(z/temperature)))})
    if save_attention:
        ats=getattr(result,'attentions',None);available=[i for i,x in enumerate(ats or []) if x is not None]
        if available:
            layers=available[-4:];ids=inp['input_ids'][0].detach().cpu().numpy();itid=int(net.config.image_token_id);positions=np.flatnonzero(ids==itid);grid=inp['image_grid_thw'][0].detach().cpu().numpy().astype(int);merge=int(net.config.vision_config.spatial_merge_size)
            raw=np.stack([ats[i][0,:,-1,:].index_select(-1,torch.as_tensor(positions,device=ats[i].device)).to('cpu',dtype=torch.float16).numpy() for i in layers])
            np.savez_compressed(out/(stem+'.npz'),decision_producing_attention_raw_heads=raw,input_ids=ids,image_token_positions=positions,image_grid_thw=grid,layer_indices=np.array(layers))
            meta.update({'attention_available':True,'attention_source':'Actual output_attentions=True weights at final prompt token. This row produces original verifier support logits or JEV head evidence, not an emitted true/false token.','layer_indices':layers,'n_heads':raw.shape[1],'image_token_id':itid,'image_grid_thw':grid.tolist(),'spatial_merge_size':merge,'merged_grid_thw':[int(grid[0]),int(grid[1]//merge),int(grid[2]//merge)],'patch_size':int(net.config.vision_config.patch_size),'input_token_ids':ids.tolist(),'input_token_pieces':[proc.tokenizer.decode([int(i)],skip_special_tokens=False) for i in ids],'npz_file':stem+'.npz'})
        else:meta.update({'attention_available':False,'attention_source':'Model did not return conventional attention tensors; no attention image fabricated.'})
    del result,inp
    torch.cuda.empty_cache()
    return meta
files=sorted((ROOT/'results').glob('*/*_grounding.json'))
for i,p in enumerate(files):
    key=p.parent.name+'__'+p.stem;dest=out/(key+'.json')
    current_hash=hashlib.sha256(p.read_bytes()).hexdigest()
    if dest.exists():
        old=json.loads(dest.read_text())
        if old['upstream_record_sha256']==current_hash:continue
        import shutil
        archive=ROOT/'budget_archive_verifiers'/a.arm;archive.mkdir(parents=True,exist_ok=True)
        for ext in ['.json','.npz']:
            source=out/(key+ext);target=archive/(key+'__'+old['upstream_record_sha256'][:12]+ext)
            if source.exists() and not target.exists():shutil.copy2(source,target)
    meta=json.loads(p.read_text());bbox=meta.get('parsed_bbox_xyxy');now=time.monotonic();torch.cuda.reset_peak_memory_stats()
    row={'arm':a.arm,'model':meta['model'],'example_index':meta['example_index'],'sample_id':meta['sample_id'],'hallucination_type':meta['hallucination_type'],'query':meta['query'],'candidate_bbox_xyxy':bbox,'upstream_record_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_record':str(p),'image_sha256':meta['image_sha256'],'source_model_path':model_path,'torch':torch.__version__,'transformers':transformers.__version__}
    try:
        if bbox is None:
            row.update({'drew':False,'original':None,'rivals':[],'reason':'original_no_box','attention_available':False})
        else:
            row['drew']=True;ip=ROOT/'images'/meta['image_filename'];query=meta['query']
            row['original']=score(ip,bbox,query,save_attention=a.arm=='omni',stem=key)
            predicate=T.predicate_of(query);probes,probe_metadata=S.build(query,max_candidates=2)
            legal=[q for q in probes if q['probe_type']=='competitive'] if predicate in P.OPPOSED else []
            row['predicate']=predicate;row['routed_predicate']=predicate in P.OPPOSED;row['probe_metadata']=probe_metadata;row['rivals']=[]
            for rival in legal:row['rivals'].append({**rival,**score(ip,bbox,rival['query'])})
        row['seconds']=time.monotonic()-now;row['peak_allocated_gb']=torch.cuda.max_memory_allocated()/1e9
        dest.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'event':'complete','arm':a.arm,'key':key,'z':(row['original'] or {}).get('z'),'n_rivals':len(row['rivals']),'seconds':row['seconds'],'peak_allocated_gb':row['peak_allocated_gb']}),flush=True)
    except Exception as exc:
        row.update({'original':None,'rivals':[],'error':repr(exc),'traceback':traceback.format_exc(),'reason':'verifier_input_unscorable','seconds':time.monotonic()-now,'attention_available':False})
        dest.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'event':'error','key':key,'error':repr(exc),'continued':True}),flush=True)
        torch.cuda.empty_cache()
