"""Greedy VQA/grounding generation and faithful causal replay attention extraction.

Creates only files below the explicitly selected experiment root. No training.
"""
from pathlib import Path
import os,json,time,argparse,hashlib,re,traceback
import numpy as np
import torch
from PIL import Image
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor

ap=argparse.ArgumentParser()
ap.add_argument('--root',required=True);ap.add_argument('--model',choices=['qwen25','visionr1'],required=True)
ap.add_argument('--indices',default='');ap.add_argument('--types',default='positive,object,co_occurrence,attribute,relation')
ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshard',type=int,default=1)
ap.add_argument('--max-new-tokens',type=int,default=160)
ap.add_argument('--only-truncated',action='store_true')
a=ap.parse_args(); root=Path(a.root).resolve()
assert str(root)=='/home/u2025141034/benchmark/rrwr_20261010_t3_attention/attention'
out=root/'results'/a.model;out.mkdir(parents=True,exist_ok=True)
selection=json.loads((root/'selection_15.json').read_text());chosen={int(x) for x in a.indices.split(',') if x};types=set(a.types.split(','))
model_path='/home/u2025141034/models/Qwen2.5-VL-7B-Instruct-Vision-R1' if a.model=='visionr1' else '/home/u2025141034/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct'
if not Path(model_path,'config.json').exists():
    candidates=list(Path(model_path).glob('snapshots/*/config.json'))
    assert len(candidates)==1,candidates
    model_path=str(candidates[0].parent)
print(json.dumps({'event':'loading','model':a.model,'model_path':model_path,'torch':torch.__version__,'transformers':transformers.__version__}),flush=True)
tload=time.monotonic()
processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,min_pixels=256*28*28,max_pixels=512*28*28)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='eager',local_files_only=True).eval()
print(json.dumps({'event':'loaded','seconds':time.monotonic()-tload,'allocated_gb':torch.cuda.memory_allocated()/1e9}),flush=True)
tasks=[]
for case in selection['cases']:
    if chosen and case['index'] not in chosen:continue
    queries=[('positive',case['base_sample_id'],case['positive_text'])]+[(p['difficulty_tier'],p['sample_id'],p['negative_text']) for p in case['pairs']]
    for ht,sid,query in queries:
        if ht not in types:continue
        for task in ('vqa','grounding'):
            tasks.append((case,ht,sid,query,task))
if a.only_truncated:
    tasks=[t for t in tasks if (out/(f'{t[0]["index"]:02d}_{t[1]}_{t[4]}.json')).exists() and json.loads((out/(f'{t[0]["index"]:02d}_{t[1]}_{t[4]}.json')).read_text())['generation_truncated']]
tasks=[t for i,t in enumerate(tasks) if i%a.nshard==a.shard]
logfile=out/f'events_shard{a.shard}.jsonl'
print(json.dumps({'event':'planned','n_calls':len(tasks),'shard':a.shard,'nshard':a.nshard}),flush=True)
def event(row):
    with logfile.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)
for case,ht,sid,query,task in tasks:
    stem=f'{case["index"]:02d}_{ht}_{task}'
    if (out/(stem+'.json')).exists() and (out/(stem+'.npz')).exists():
        if not a.only_truncated:continue
        old_meta=json.loads((out/(stem+'.json')).read_text());archive=root/f'budget_archive_max{old_meta["max_new_tokens"]}'/a.model;archive.mkdir(parents=True,exist_ok=True)
        import shutil
        for ext in ['.json','.npz']:
            if not (archive/(stem+ext)).exists():shutil.copy2(out/(stem+ext),archive/(stem+ext))
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        ip=root/'images'/case['image_filename'];image=Image.open(ip).convert('RGB');w,h=image.size
        if task=='vqa':
            prompt=f'Does this image support the expression "{query}"? Briefly explain the relevant visual evidence. Then finish with <answer>yes</answer> or <answer>no</answer>.'
        else:
            prompt=f'Locate the object described by "{query}" in this image. The original image is {w} pixels wide and {h} pixels high. Briefly explain the relevant visual evidence. If a matching object exists, finish with <answer>[x0, y0, x1, y1]</answer>, using original-image pixel coordinates. If the expression is not supported, finish with <answer>none</answer>.'
        msgs=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        rendered=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=processor(text=[rendered],images=[image],return_tensors='pt')
        inp={k:v.to('cuda:0') if hasattr(v,'to') else v for k,v in inp.items()}
        plen=inp['input_ids'].shape[1]
        with torch.inference_mode():
            seq=model.generate(**inp,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True)
        generated=seq[0,plen:];response=processor.tokenizer.decode(generated,skip_special_tokens=True)
        replay=dict(inp);replay['input_ids']=seq;replay['attention_mask']=torch.ones_like(seq)
        if 'token_type_ids' in replay:replay.pop('token_type_ids')
        if 'mm_token_type_ids' in replay:
            replay['mm_token_type_ids']=torch.cat([inp['mm_token_type_ids'],torch.zeros((1,len(generated)),dtype=inp['mm_token_type_ids'].dtype,device=seq.device)],dim=1)
        with torch.inference_mode():
            result=model(**replay,use_cache=False,output_attentions=True,return_dict=True)
        attentions=result.attentions
        assert attentions and all(x is not None for x in attentions[-4:]),'Model did not return real attention weights'
        ids=seq[0].detach().cpu().numpy(); image_token_id=int(model.config.image_token_id)
        image_positions=np.flatnonzero(ids==image_token_id);g=inp['image_grid_thw'][0].detach().cpu().numpy().astype(int)
        merge=int(model.config.vision_config.spatial_merge_size);gh,gw=int(g[1]//merge),int(g[2]//merge)
        assert len(image_positions)==int(g[0])*gh*gw,(len(image_positions),g.tolist(),merge)
        token_ids=generated.detach().cpu().numpy(); token_strings=processor.tokenizer.convert_ids_to_tokens(token_ids.tolist()); token_pieces=[processor.tokenizer.decode([int(x)],skip_special_tokens=False) for x in token_ids]
        # Preserve all output-token maps, plus unsummarized heads for decision/coordinate tokens.
        token_text=''.join(token_pieces)
        answer_span=re.search(r'<answer>(.*?)</answer>',token_text,re.S)
        piece_starts=np.cumsum([0]+[len(s) for s in token_pieces]).tolist()
        eligible_indices=[i for i in range(len(token_pieces)) if not answer_span or (piece_starts[i+1]>answer_span.start(1) and piece_starts[i]<answer_span.end(1))]
        decision_indices=[i for i in eligible_indices if re.search(r'\b(yes|no|none)\b',token_pieces[i],re.I)]
        answer_match=re.search(r'<answer>(.*?)</answer>',response,re.S)
        parsed_bbox=None
        if task=='grounding':
            target=(answer_match.group(1) if answer_match else response).strip()
            bm=re.search(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',target)
            if bm:parsed_bbox=[float(x) for x in bm.groups()]
            raw_indices=[i for i in eligible_indices if re.search(r'\d',token_pieces[i])]
            if not raw_indices:raw_indices=decision_indices
        else:raw_indices=decision_indices
        if not raw_indices:raw_indices=[max(0,len(token_ids)-1)]
        layer_indices=list(range(len(attentions)-4,len(attentions)))
        all_output_maps=[];raw_heads=[]
        for layer in layer_indices:
            aa=attentions[layer][0,:,plen:,:]
            patch_rows=aa.index_select(-1,torch.as_tensor(image_positions,device=aa.device))
            all_output_maps.append(patch_rows.float().mean(0).to('cpu',dtype=torch.float16).numpy())
            raw_heads.append(patch_rows[:,raw_indices,:].to('cpu',dtype=torch.float16).numpy())
        final_prompt=[]
        for layer in layer_indices:
            final_prompt.append(attentions[layer][0,:,plen-1,image_positions].to('cpu',dtype=torch.float16).numpy())
        np.savez_compressed(out/(stem+'.npz'),output_token_attention_head_mean=np.stack(all_output_maps),selected_token_attention_raw_heads=np.stack(raw_heads),last_prompt_token_attention_raw_heads=np.stack(final_prompt),input_ids=ids,generated_token_ids=token_ids,image_token_positions=image_positions,selected_output_token_indices=np.array(raw_indices),layer_indices=np.array(layer_indices),image_grid_thw=g)
        meta={'model':a.model,'model_path':model_path,'model_config_sha256':hashlib.sha256(Path(model_path,'config.json').read_bytes()).hexdigest(),'task':task,'example_index':case['index'],'hallucination_type':ht,'sample_id':sid,'query':query,'image_filename':case['image_filename'],'image_sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),'original_size_wh':[w,h],'prompt':prompt,'rendered_prompt':rendered,'response':response,'answer_text':answer_match.group(1).strip() if answer_match else None,'parsed_bbox_xyxy':parsed_bbox,'prompt_token_count':plen,'generated_token_count':len(token_ids),'generated_token_ids':token_ids.tolist(),'generated_token_strings':token_strings,'generated_token_pieces':token_pieces,'decision_token_indices':decision_indices,'raw_head_selected_output_token_indices':raw_indices,'attention_source':'Actual decoder self-attention returned by output_attentions=True during teacher-forced causal replay of the exact greedy generated token sequence. Row indices refer to consumed emitted output tokens; last-prompt row produces the first generated token.','attention_implementation':'eager','aggregation':'Preserve all 28 heads for selected output tokens and mean of heads for every output token, in each of final four decoder layers. No bbox-derived or saliency-derived heatmap.','stored_dtype':'float16','layer_indices':layer_indices,'n_total_layers':len(attentions),'n_heads':int(attentions[-1].shape[1]),'image_token_id':image_token_id,'image_token_count':len(image_positions),'image_token_positions':image_positions.tolist(),'image_grid_thw':g.tolist(),'spatial_merge_size':merge,'merged_grid_thw':[int(g[0]),gh,gw],'patch_size':int(model.config.vision_config.patch_size),'resized_image_size_wh':[int(g[2])*int(model.config.vision_config.patch_size),int(g[1])*int(model.config.vision_config.patch_size)],'grid_order':'Qwen2.5-VL image tokens after inverse window reorder, row-major merged spatial grid. One still-image temporal group.','processor_min_pixels':256*28*28,'processor_max_pixels':512*28*28,'generation_do_sample':False,'max_new_tokens':a.max_new_tokens,'generation_truncated':len(token_ids)>=a.max_new_tokens,'torch':torch.__version__,'transformers':transformers.__version__,'seconds':time.monotonic()-start,'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'npz_file':stem+'.npz','asv_semantics':'ASV filters the same upstream candidate; this attention does not change because of post-hoc filtering. Independent verifier attention must be labeled separately.','not_aggregate_paper_evaluation':'Illustrative reasoning-and-answer prompt for attention inspection; does not update benchmark metrics.'}
        (out/(stem+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        event({'event':'complete','stem':stem,'response':response,'seconds':meta['seconds'],'peak_allocated_gb':meta['peak_allocated_gb'],'grid':meta['merged_grid_thw'],'n_generated':len(token_ids),'truncated':meta['generation_truncated']})
        del result,attentions,seq,generated,inp,replay,aa,patch_rows
        torch.cuda.empty_cache()
    except Exception as exc:
        event({'event':'error','stem':stem,'error':repr(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-start})
        raise
