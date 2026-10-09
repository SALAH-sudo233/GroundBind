"""Audit real tensors, apply restored ASV policy to fresh scores, and plot overlays."""
from pathlib import Path
import json,re,hashlib,math,collections,textwrap
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for font in (Path(__file__).resolve().parent/'fonts').glob('*.ttf'):font_manager.fontManager.addfont(str(font))
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'visuals';OUT.mkdir(exist_ok=True)
policy=json.loads((ROOT/'restored_asv_policy.json').read_text())
selection=json.loads((ROOT/'selection_15.json').read_text())
MODELS={'qwen25':'Qwen2.5-VL-7B (General)','visionr1':'Vision-R1 (RL)'}
POLICY_MODELS={'qwen25':'qwen2.5-vl-7b','visionr1':'Vision-R1'}
HT=['positive','object','co_occurrence','attribute','relation']
HT_LABEL={'positive':'Positive','object':'Object negative','co_occurrence':'Association negative','attribute':'Attribute negative','relation':'Relation negative'}
FIGURE_FONT='Times New Roman' if any(f.name=='Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
plt.rcParams.update({'font.family':FIGURE_FONT,'font.size':8,'axes.titleweight':'normal','pdf.fonttype':42,'figure.facecolor':'white'})
records={};asv={};tensor_audit=[];plot_metadata=[]
def load_record(model,index,ht,task):return records[(model,index,ht,task)]
def decision(meta):
    if meta['generation_truncated']:return None
    matches=re.findall(r'answer>\s*(yes|no)\b',meta['response'],re.I)
    if matches:return matches[-1].lower()
    matches=re.findall(r'(?:answer is|answer:)\s*(yes|no)\b',meta['response'],re.I)
    if matches:return matches[-1].lower()
    z=re.search(r'\b(yes|no)[.\s]*$',meta['response'],re.I)
    return z.group(1).lower() if z else None
def chosen_tokens(meta):
    pieces=meta['generated_token_pieces'];txt=''.join(pieces);starts=np.cumsum([0]+[len(s) for s in pieces])
    if meta['task']=='grounding' and meta.get('parsed_bbox_xyxy') is not None:
        matches=list(re.finditer(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',txt))
        matches=[m for m in matches if [float(z) for z in m.groups()]==meta['parsed_bbox_xyxy']]
        match=matches[-1] if matches else None
        if match:
            ii=[i for i,s in enumerate(pieces) if re.search(r'\d',s) and starts[i+1]>match.start() and starts[i]<match.end()]
            if ii:return ii,'coordinate tokens (capped prefix)' if meta['generation_truncated'] else 'coordinate tokens'
    desired=decision(meta) if meta['task']=='vqa' else 'none'
    if desired:
        ii=[i for i,s in enumerate(pieces) if re.search(r'\b'+desired+r'\b',s,re.I)]
        if ii:return [ii[-1]],'decision token' if meta['task']=='vqa' else 'none token'
    return list(range(len(pieces))),'output prefix tokens (no parsed decision)'
def bbox_state(meta):
    b=meta['parsed_bbox_xyxy']
    if b is None:return 'capped prefix / no parsed bbox' if meta['generation_truncated'] else 'no parsed bbox'
    if b[2]<=b[0] or b[3]<=b[1]:label='invalid / zero-area coordinates'
    elif b[0]<0 or b[1]<0 or b[2]>meta['original_size_wh'][0] or b[3]>meta['original_size_wh'][1]:label='out-of-image coordinates'
    else:label='coordinates'
    return ('capped prefix: '+label) if meta['generation_truncated'] else label
def sigmoid(x):
    return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
def readout(head,features):
    z=head['b']+sum(w*(x-m)/s for w,x,m,s in zip(head['w'],features,head['mu'],head['sd']))
    return sigmoid(z)
def fold_of(sid):
    base=str(sid).split('__')[0];i=base.find('COCO_');image=(base[i:] if i>=0 else base)+'.jpg'
    return int(hashlib.md5(image.encode()).hexdigest(),16)%2
for p in sorted((ROOT/'results').glob('*/*.json')):
    r=json.loads(p.read_text());npz=p.with_suffix('.npz');arr=np.load(npz);weights=arr['output_token_attention_head_mean']
    assert np.isfinite(weights).all() and np.all(weights>=0)
    assert weights.shape[0]==4 and weights.shape[1]==r['generated_token_count']
    assert weights.shape[2]==r['image_token_count']
    assert np.array_equal(arr['generated_token_ids'],r['generated_token_ids'])
    assert weights.sum(-1).max()<1.01
    r['local_json_path']=str(p);r['local_npz_path']=str(npz)
    records[(r['model'],r['example_index'],r['hallucination_type'],r['task'])]=r
    tensor_audit.append({'record':str(p.relative_to(ROOT)),'npz_sha256':hashlib.sha256(npz.read_bytes()).hexdigest(),'shape':list(weights.shape),'all_finite_nonnegative':True,'max_image_attention_mass':float(weights.sum(-1).max()),'real_attention_source':r['attention_source']})
assert len(records)==300
for model in MODELS:
    for case in selection['cases']:
        for ht in HT:
            for task in ['vqa','grounding']:assert (model,case['index'],ht,task) in records
            r=load_record(model,case['index'],ht,'grounding');stem=Path(r['local_json_path']).stem;key=model+'__'+stem
            oo=json.loads((ROOT/'verifiers/omni'/(key+'.json')).read_text());jj=json.loads((ROOT/'verifiers/jev'/(key+'.json')).read_text())
            current_hash=hashlib.sha256(Path(r['local_json_path']).read_bytes()).hexdigest()
            assert oo['upstream_record_sha256']==jj['upstream_record_sha256']==current_hash
            assert oo['candidate_bbox_xyxy']==jj['candidate_bbox_xyxy']==r['parsed_bbox_xyxy']
            assert oo['image_sha256']==jj['image_sha256']==r['image_sha256']
            f=fold_of(r['sample_id']);t=policy['models'][POLICY_MODELS[model]]['thresholds'][str(f)]
            zo=(oo.get('original') or {}).get('z');zj=(jj.get('original') or {}).get('z')
            ro=oo.get('rivals',[]);rj=jj.get('rivals',[])
            if ro and rj:assert [x['query'] for x in ro]==[x['query'] for x in rj]
            zmo=max((x['z'] for x in ro),default=None);zmj=max((x['z'] for x in rj),default=None)
            drew=r['parsed_bbox_xyxy'] is not None;support_usable=drew and zo is not None and zj is not None
            support=readout(t['support_head'],[zo,zj]) if support_usable else None
            support_keep=False if not drew else True if not support_usable else support>=t['support_threshold']
            routed=support_usable and zmo is not None and zmj is not None and oo.get('routed_predicate') and t['full_head'] is not None
            features=[zo,zj,(zmo-zo if zmo is not None and zo is not None else 0),(zmj-zj if zmj is not None and zj is not None else 0)]
            full=readout(t['full_head'],features) if routed else None
            keep=support_keep and full>=t['full_threshold'] if routed else support_keep
            reason='no_parsed_box' if not drew else 'verifier_unusable_fallback_retain' if not support_usable else 'support_reject' if not support_keep else 'structural_reject' if not keep else 'kept'
            row={'model':model,'example_index':case['index'],'hallucination_type':ht,'sample_id':r['sample_id'],'query':r['query'],'candidate_bbox_xyxy':r['parsed_bbox_xyxy'],'asv_bbox_xyxy':r['parsed_bbox_xyxy'] if keep else None,'keep':bool(keep),'reason':reason,'fold':f,'z_omni':zo,'z_jev':zj,'zmax_omni':zmo,'zmax_jev':zmj,'gap_omni':features[2],'gap_jev':features[3],'support_score':support,'support_threshold':t['support_threshold'],'structural_routed':bool(routed),'full_score':full,'full_threshold':t['full_threshold'] if routed else None,'source_record_sha256':current_hash,'source_record':str(Path(r['local_json_path']).relative_to(ROOT)),'original_attention_unchanged_by_asv':True,'scope':'New illustrative candidate, original source-trained cross-fit readout and threshold. No use of selected-case labels for fit or threshold.'}
            asv[(model,case['index'],ht)]=row
(ROOT/'fresh_asv_decisions.json').write_text(json.dumps(list(asv.values()),ensure_ascii=False,indent=2)+'\n')

def map_for(meta):
    z=np.load(meta['local_npz_path']);ii,label=chosen_tokens(meta);raw=z['output_token_attention_head_mean'][:,ii,:].astype(np.float32).mean((0,1))
    grid=meta['merged_grid_thw'];assert grid[0]==1
    mass=float(raw.sum());rel=raw/(float(raw.mean())+1e-12)
    return rel.reshape(grid[1],grid[2]),mass,ii,label
def verifier_map(model,index,ht):
    stem=f'{index:02d}_{ht}_grounding';key=model+'__'+stem;meta=json.loads((ROOT/'verifiers/omni'/(key+'.json')).read_text());original=meta.get('original')
    if not original or not original.get('attention_available'):return None,None,None
    z=np.load(ROOT/'verifiers/omni'/original['npz_file']);heads=z['decision_producing_attention_raw_heads'];g=original['merged_grid_thw'];assert g[0]==1
    assert np.isfinite(heads).all() and (heads>=0).all() and heads.shape[-1]==g[1]*g[2]
    assert len(z['image_token_positions'])==g[1]*g[2]
    raw=heads.astype(np.float32).mean((0,1));assert raw.sum()<1.01
    return (raw/(raw.mean()+1e-12)).reshape(g[1],g[2]),float(raw.sum()),original
def overlay(ax,image,relative,vmax,bbox=None,bbox_color='#F18D36'):
    w,h=image.size;ax.imshow(image)
    if relative is not None:
        norm=Normalize(0,vmax);rgba=plt.colormaps['Blues'](norm(relative));rgba[...,3]=.67*np.clip(norm(relative),0,1)**.7
        ax.imshow(rgba,extent=[0,w,h,0],interpolation='bilinear')
    if bbox is not None and bbox[2]>bbox[0] and bbox[3]>bbox[1]:
        ax.add_patch(Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],fill=False,edgecolor=bbox_color,linewidth=1.6))
    ax.set(xlim=(0,w),ylim=(h,0));ax.axis('off')
def render_sheet(model,case):
    image=Image.open(ROOT/'images'/case['image_filename']).convert('RGB');maps={};all_values=[]
    for ht in HT:
        for task in ['vqa','grounding']:
            meta=load_record(model,case['index'],ht,task);maps[(ht,task)]=map_for(meta);all_values.append(maps[(ht,task)][0].ravel())
        maps[(ht,'verifier')]=verifier_map(model,case['index'],ht)
        if maps[(ht,'verifier')][0] is not None:all_values.append(maps[(ht,'verifier')][0].ravel())
    vmax=max(1,float(np.quantile(np.concatenate(all_values),.95)))
    query_texts=[]
    fig,axes=plt.subplots(5,4,figsize=(13.4,13.8),gridspec_kw={'hspace':.60,'wspace':.12})
    fig.suptitle(f'Case {case["index"]:02d} | {MODELS[model]}\n{case["positive_text"]}',fontsize=12,y=.986)
    for row,ht in enumerate(HT):
        v=load_record(model,case['index'],ht,'vqa');g=load_record(model,case['index'],ht,'grounding');a=asv[(model,case['index'],ht)]
        vm,vmass,vi,vlabel=maps[(ht,'vqa')];gm,gmass,gi,glabel=maps[(ht,'grounding')];om,omass,ometa=maps[(ht,'verifier')]
        query='\n'.join(textwrap.wrap(v['query'],95));query_texts.append(fig.text(.04,.945-row*.176,f'{HT_LABEL[ht]}: {query}',fontsize=8.4,va='top'))
        overlay(axes[row,0],image,vm,vmax);vd='CAPPED PREFIX' if v['generation_truncated'] else decision(v) or 'unparsed';axes[row,0].set_title(f'VQA: {vd} | {vlabel}\nimage mass={vmass:.3f}',fontsize=8)
        overlay(axes[row,1],image,gm,vmax,g['parsed_bbox_xyxy']);bboxstate=bbox_state(g);axes[row,1].set_title(f'Grounding before: {bboxstate}\n{glabel}; image mass={gmass:.3f}',fontsize=8)
        overlay(axes[row,2],image,gm,vmax,a['asv_bbox_xyxy']);ast='KEEP' if a['keep'] else 'REJ' if a['candidate_bbox_xyxy'] else 'NO PARSED BOX'
        if g['generation_truncated']:ast+=' (prefix candidate)'
        if 'fallback' in a['reason']:ast='FALLBACK RETAIN (verifier failed)'
        axes[row,2].set_title(f'After ASV: {ast}\nsame upstream attention',fontsize=8,color='#B74561' if not a['keep'] else '#35405A')
        overlay(axes[row,3],image,om,vmax,g['parsed_bbox_xyxy'] if om is not None else None,bbox_color='#CA3B3B')
        if om is not None:axes[row,3].set_title(f'Independent OmniVerifier attention\nz={a["z_omni"]:.2f}; image mass={omass:.3f}',fontsize=8)
        else:axes[row,3].set_title('Independent OmniVerifier\nno attention: '+('no parsed candidate' if g['parsed_bbox_xyxy'] is None else 'unscorable candidate'),fontsize=8)
        qfig,qaxes=plt.subplots(1,4,figsize=(13.4,4.0),gridspec_kw={'wspace':.12})
        for ax,arr,box,title in zip(qaxes,[vm,gm,gm,om],[None,g['parsed_bbox_xyxy'],a['asv_bbox_xyxy'],g['parsed_bbox_xyxy'] if om is not None else None],[f'VQA: {vd} | {vlabel}',f'Grounding before: {bboxstate}',f'After ASV: {ast}\nSame upstream attention','Independent OmniVerifier attention' if om is not None else 'Omni: no attention / no scorable candidate']):
            overlay(ax,image,arr,vmax,box,bbox_color='#CA3B3B' if ax is qaxes[3] else '#F18D36');ax.set_title(title,fontsize=9)
        qfig.suptitle(f'Case {case["index"]:02d} | {MODELS[model]} | {HT_LABEL[ht]}\n'+v['query'],fontsize=11,y=.975)
        qfig.subplots_adjust(left=.025,right=.98,top=(.79 if image.height>image.width else .84),bottom=.30)
        qcax=qfig.add_axes([.31,.19,.4,.022]);qfig.colorbar(ScalarMappable(norm=Normalize(0,vmax),cmap='Blues'),cax=qcax,orientation='horizontal',extend='max').set_label('Relative attention (mean image patch = 1)',fontsize=8)
        qfig.text(.03,.026,'Actual replay rows consuming selected output tokens; last 4 layers / mean heads. Low weights transparent. Orange = model candidate; red = verifier input candidate. Before/after share the same upstream map.\nPer-map normalization; vmax = pooled 95th percentile (higher values clipped). Colors do not compare absolute attention across models. Independent Omni map uses its final prompt token.',fontsize=7.5)
        qfig.savefig(OUT/f'query_{case["index"]:02d}_{model}_{ht}.png',dpi=155);plt.close(qfig)
        plot_metadata.append({'model':model,'case':case['index'],'query_type':ht,'vqa_tokens':vi,'vqa_token_pieces':[v['generated_token_pieces'][i] for i in vi],'grounding_tokens':gi,'grounding_token_pieces':[g['generated_token_pieces'][i] for i in gi],'vqa_attention_image_mass':vmass,'grounding_attention_image_mass':gmass,'asv':a,'normalization':'Each selected-token attention map divided by its mean weight across image patches. Mean image patch=1; this removes total image attention mass. Shared 0..vmax color mapping within this sheet, vmax pooled 95th percentile and larger values clipped; do not compare colors as absolute attention across models.','colorbar_vmax':vmax,'colorbar_percentile_clip':95,'colorbar_clip_scope':'All available VQA/DG/Omni maps within one model-case sheet (representative figures: both model rows)','cmap':'Blues','overlay_alpha':'0.67*clip(relative/vmax,0,1)**0.7; zero attention transparent','interpolation':'bilinear display interpolation only; no Gaussian, bbox-derived attention or saliency','displayed_attention_row_semantics':'Teacher-forced causal replay rows consuming selected emitted output tokens, not the preceding row that produces the token. Omni uses its final prompt row producing support logits.','before_upstream_map_sha256':hashlib.sha256(gm.astype(np.float32).tobytes()).hexdigest(),'after_upstream_map_sha256':hashlib.sha256(gm.astype(np.float32).tobytes()).hexdigest(),'before_after_heatmap_identity':'exact same upstream array used on both sides; only kept/rejected candidate bbox changes'})
    fig.subplots_adjust(left=.04,right=.98,top=.895,bottom=.105)
    for row,qtext in enumerate(query_texts):qtext.set_position((.04,axes[row,0].get_position().y1+.054))
    cax=fig.add_axes([.28,.058,.44,.012]);fig.colorbar(ScalarMappable(norm=Normalize(0,vmax),cmap='Blues'),cax=cax,orientation='horizontal',extend='max').set_label('Relative attention (mean image patch = 1)',fontsize=8)
    fig.text(.04,.009,'Actual decoder attention, final 4 layers / mean heads. Low weights transparent. Orange = model candidate; red = verifier input candidate. ASV filters this same candidate.\nPer-map normalization; pooled 95th percentile color clipping. Color intensity is not an absolute cross-model attention comparison. Details and all token maps are saved in NPZ/JSON.',fontsize=7.5)
    stem=f'case_{case["index"]:02d}_{model}_all_queries';fig.savefig(OUT/(stem+'.png'),dpi=170);fig.savefig(OUT/(stem+'.pdf'));plt.close(fig)
    return str((OUT/(stem+'.png')).relative_to(ROOT))
budget_rows=[]
for old in sorted(ROOT.glob('budget_archive_max160/*/*.json')):
    first=json.loads(old.read_text());key=(first['model'],first['example_index'],first['hallucination_type'],first['task']);current=records[key]
    stages=[]
    for budget in [160,512]:
        jp=ROOT/f'budget_archive_max{budget}'/first['model']/old.name
        if jp.exists():
            r=json.loads(jp.read_text());stages.append({'max_new_tokens':r['max_new_tokens'],'generated_token_count':r['generated_token_count'],'truncated':r['generation_truncated'],'seconds':r['seconds'],'peak_allocated_gb':r['peak_allocated_gb'],'json':str(jp.relative_to(ROOT)),'npz':str(jp.with_suffix('.npz').relative_to(ROOT))})
    stages.append({'max_new_tokens':current['max_new_tokens'],'generated_token_count':current['generated_token_count'],'truncated':current['generation_truncated'],'seconds':current['seconds'],'peak_allocated_gb':current['peak_allocated_gb'],'json':str(Path(current['local_json_path']).relative_to(ROOT)),'npz':str(Path(current['local_npz_path']).relative_to(ROOT))})
    budget_rows.append({'model':key[0],'case':key[1],'query_type':key[2],'task':key[3],'original_prompt_and_greedy_unchanged':all(json.loads((ROOT/x['json']).read_text())['prompt']==current['prompt'] for x in stages),'stages':stages,'still_capped':current['generation_truncated']})
assert len(budget_rows)==12 and all(x['original_prompt_and_greedy_unchanged'] for x in budget_rows)
budget_manifest={'original_budget':160,'initial_capped_record_count':12,'untouched_record_count':288,'budget_only_rerun_count':12,'final_still_capped_count':sum(x['still_capped'] for x in budget_rows),'records':budget_rows,'capped_semantics':'An incomplete output prefix is not a complete model decision. A parseable emitted coordinate quadruple may be scored as a prefix candidate; it remains explicitly capped.'}
(ROOT/'generation_budget_manifest.json').write_text(json.dumps(budget_manifest,ensure_ascii=False,indent=2)+'\n')
import sys
if '--preview' in sys.argv:
    render_sheet('qwen25',next(c for c in selection['cases'] if c['index']==8));print('preview_complete',flush=True);raise SystemExit(0)
if '--representatives-only' in sys.argv:
    plot_metadata=json.loads((ROOT/'visualization_metadata.json').read_text())
    sheet_paths=[str((OUT/f'case_{c["index"]:02d}_{m}_all_queries.png').relative_to(ROOT)) for c in selection['cases'] for m in MODELS]
elif '--render-indices' in sys.argv:
    subset={int(v) for v in sys.argv[sys.argv.index('--render-indices')+1].split(',')}
    plot_metadata=json.loads((ROOT/'visualization_metadata.json').read_text());plot_metadata=[x for x in plot_metadata if x['case'] not in subset]
    for c in selection['cases']:
        if c['index'] in subset:
            for m in MODELS:render_sheet(m,c)
    sheet_paths=[str((OUT/f'case_{c["index"]:02d}_{m}_all_queries.png').relative_to(ROOT)) for c in selection['cases'] for m in MODELS]
else:
    sheet_paths=[render_sheet(m,c) for c in selection['cases'] for m in MODELS]

(ROOT/'visualization_metadata.json').write_text(json.dumps(plot_metadata,ensure_ascii=False,indent=2)+'\n')
def attention_spatial_metrics(case,meta):
    rel,mass,ii,label=map_for(meta);gh,gw=rel.shape;w,h=meta['original_size_wh'];box=case['positive_bbox_xyxy'];x0,y0,x1,y1=box
    xs=np.linspace(0,w,gw+1);ys=np.linspace(0,h,gh+1)
    xx=np.maximum(0,np.minimum(xs[1:],x1)-np.maximum(xs[:-1],x0))/(w/gw)
    yy=np.maximum(0,np.minimum(ys[1:],y1)-np.maximum(ys[:-1],y0))/(h/gh)
    coverage=yy[:,None]*xx[None,:];fraction=float((rel*coverage).sum()/rel.sum());area=float(coverage.mean())
    positive=load_record(meta['model'],case['index'],'positive',meta['task']);positive_map=map_for(positive);pr=positive_map[0];pb=positive.get('parsed_bbox_xyxy');positive_coordinate_valid=not positive['generation_truncated'] and positive_map[3].startswith('coordinate tokens') and pb is not None and pb[2]>pb[0] and pb[3]>pb[1]
    assert pr.shape==rel.shape
    cosine=float(np.vdot(rel,pr)/(np.linalg.norm(rel)*np.linalg.norm(pr)+1e-12))
    return {'model':meta['model'],'case':case['index'],'query_type':meta['hallucination_type'],'task':meta['task'],'source':str(Path(meta['local_json_path']).relative_to(ROOT)),'token_aggregation':label,'selected_output_tokens':ii,'image_attention_mass':mass,'supported_positive_anchor_bbox':box,'anchor_role':'Source positive-reference anchor only, never a GT box for a negative query. Original annotations need human review.','anchor_image_area_fraction':area,'fraction_of_image_attention_inside_supported_positive_anchor':fraction,'anchor_attention_density_enrichment_over_uniform':fraction/(area+1e-12),'current_coordinate_geometry_positive_area':bool(meta.get('parsed_bbox_xyxy') and meta['parsed_bbox_xyxy'][2]>meta['parsed_bbox_xyxy'][0] and meta['parsed_bbox_xyxy'][3]>meta['parsed_bbox_xyxy'][1]),'positive_reference_source':str(Path(positive['local_json_path']).relative_to(ROOT)),'positive_reference_truncated':positive['generation_truncated'],'positive_reference_token_aggregation':positive_map[3],'positive_reference_complete_valid_coordinate_token_map':bool(positive_coordinate_valid),'cosine_to_same_task_positive_query_map':cosine,'truncated':meta['generation_truncated'],'measurement':'Patch weight distributed uniformly within its projected cell; fractional cell intersection with source positive anchor. Descriptive aggregation, not causal attribution or negative GT localization.'}
spatial_metrics=[attention_spatial_metrics(c,load_record(m,c['index'],h,t)) for c in selection['cases'] for m in MODELS for h in HT for t in ['vqa','grounding']]
(ROOT/'attention_spatial_metrics.json').write_text(json.dumps(spatial_metrics,ensure_ascii=False,indent=2)+'\n')

REP=ROOT/'representative_candidates';REP.mkdir(exist_ok=True)
representatives=[]
for index,ht in [(8,'attribute'),(12,'relation'),(2,'co_occurrence'),(14,'object')]:
    case=next(x for x in selection['cases'] if x['index']==index);image=Image.open(ROOT/'images'/case['image_filename']).convert('RGB');data={};vals=[]
    for m in MODELS:
        vm=map_for(load_record(m,index,ht,'vqa'));gm=map_for(load_record(m,index,ht,'grounding'));om=verifier_map(m,index,ht);data[m]=(vm,gm,om)
        vals.extend([vm[0].ravel(),gm[0].ravel()])
        if om[0] is not None:vals.append(om[0].ravel())
    vmax=max(1,float(np.quantile(np.concatenate(vals),.95)));fig,axes=plt.subplots(2,4,figsize=(7.6,5.0),gridspec_kw={'wspace':.08,'hspace':.42})
    query=load_record('qwen25',index,ht,'vqa')['query'];fig.suptitle(f'Case {index:02d} | {HT_LABEL[ht]}\n'+query,fontsize=10,y=.985)
    meta_rows=[]
    for row,m in enumerate(MODELS):
        v=load_record(m,index,ht,'vqa');g=load_record(m,index,ht,'grounding');a=asv[(m,index,ht)];vm,gm,om=data[m]
        ast='KEEP' if a['keep'] else 'REJ' if a['candidate_bbox_xyxy'] is not None else 'NO PARSED BOX'
        if 'fallback' in a['reason']:ast='FALLBACK RETAIN'
        titles=[('Qwen2.5-VL' if m=='qwen25' else 'Vision-R1')+' | VQA: '+(decision(v) or 'unparsed'),'DG: '+('coordinates' if g['parsed_bbox_xyxy'] is not None else 'no parsed bbox'),'DG w/ ASV: '+ast+'\nSame upstream attention','Independent OmniVerifier']
        for col,(mp,bx,title) in enumerate(zip([vm[0],gm[0],gm[0],om[0]],[None,g['parsed_bbox_xyxy'],a['asv_bbox_xyxy'],g['parsed_bbox_xyxy'] if om[0] is not None else None],titles)):
            overlay(axes[row,col],image,mp,vmax,bx,bbox_color='#CA3B3B' if col==3 else '#F18D36');axes[row,col].set_title(title,fontsize=7.4)
            if mp is None:axes[row,col].text(.5,.04,'No scorable candidate',transform=axes[row,col].transAxes,ha='center',fontsize=7,color='white',bbox={'facecolor':'#263348','alpha':.8,'pad':2})
        meta_rows.append({'model':m,'vqa_source':str(Path(v['local_json_path']).relative_to(ROOT)),'grounding_source':str(Path(g['local_json_path']).relative_to(ROOT)),'vqa_selected_output_tokens':vm[2],'grounding_selected_output_tokens':gm[2],'vqa_map_label':vm[3],'grounding_map_label':gm[3],'asv':a})
    fig.subplots_adjust(left=.02,right=.985,top=.83,bottom=.24);cax=fig.add_axes([.28,.16,.45,.023]);fig.colorbar(ScalarMappable(norm=Normalize(0,vmax),cmap='Blues'),cax=cax,orientation='horizontal',extend='max').set_label('Relative attention (mean image patch = 1)',fontsize=7.5)
    fig.text(.025,.018,'Real replay rows consuming selected VQA decision / DG coordinate tokens; last 4 layers, mean heads.\nOrange: model candidate; red: verifier input candidate. ASV filters the same candidate and attention. Omni: independent final-prompt attention.\nPer-map mean normalization; pooled 95th percentile color clipping. Colors do not compare absolute attention across models.',fontsize=7)
    stem=f'case_{index:02d}_{ht}_two_models';fig.savefig(REP/(stem+'.png'),dpi=260);fig.savefig(REP/(stem+'.pdf'));plt.close(fig)
    representatives.append({'case':index,'query_type':ht,'query':query,'png':str((REP/(stem+'.png')).relative_to(ROOT)),'pdf':str((REP/(stem+'.pdf')).relative_to(ROOT)),'colorbar_vmax':vmax,'colorbar_percentile_clip':95,'colorbar_clip_scope':'All available VQA/DG/Omni maps within one model-case sheet (representative figures: both model rows)','cmap':'Blues','models':meta_rows,'paper_edited':False})
(ROOT/'representative_candidates/manifest.json').write_text(json.dumps(representatives,ensure_ascii=False,indent=2)+'\n')
(REP/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>代表图候选</title><style>body{font:16px system-ui;margin:25px;max-width:1200px}img{width:100%}</style><h1>代表图候选</h1><p>供独立讨论；尚未写入论文。每图保留真实上游注意力与独立Omni证据关注。</p>'+''.join(f'<h2>#{r["case"]:02d} {r["query"]}</h2><p><a href="{Path(r["pdf"]).name}">PDF</a> · <a href="{Path(r["png"]).name}">PNG</a></p><img src="{Path(r["png"]).name}">' for r in representatives))

summary={}
observations=[]
for model in MODELS:
    stats={'total_images':15,'vqa_queries':75,'grounding_queries':75,'vqa_explicit_yes':0,'vqa_explicit_no':0,'vqa_unparsed':0,'grounding_parseable_coordinates':0,'grounding_no_parsed_bbox':0,'grounding_invalid_order_or_zero_area':0,'generation_truncated_records':0,'negative_vqa_no_with_coordinates':0,'negative_vqa_yes_with_coordinates':0,'asv_rejected_emitted_candidates':0,'asv_retained_emitted_candidates':0,'asv_missing_verifier_fallback_retain':0}
    per=[]
    for case in selection['cases']:
        for ht in HT:
            v=load_record(model,case['index'],ht,'vqa');g=load_record(model,case['index'],ht,'grounding');a=asv[(model,case['index'],ht)];d=decision(v);bbox=g['parsed_bbox_xyxy'];stats['vqa_explicit_'+d if d else 'vqa_unparsed']+=1;stats['grounding_parseable_coordinates' if bbox else 'grounding_no_parsed_bbox']+=1
            stats['generation_truncated_records']+=int(v['generation_truncated'])+int(g['generation_truncated'])
            if bbox and (bbox[2]<=bbox[0] or bbox[3]<=bbox[1]):stats['grounding_invalid_order_or_zero_area']+=1
            if ht!='positive' and bbox and d=='no':stats['negative_vqa_no_with_coordinates']+=1
            if ht!='positive' and bbox and d=='yes':stats['negative_vqa_yes_with_coordinates']+=1
            if bbox:stats['asv_retained_emitted_candidates' if a['keep'] else 'asv_rejected_emitted_candidates']+=1
            if 'fallback' in a['reason']:stats['asv_missing_verifier_fallback_retain']+=1
            row={'model':model,'case':case['index'],'type':ht,'query':v['query'],'vqa_decision':d,'grounding_bbox':bbox,'asv':a['reason'],'vqa_truncated':v['generation_truncated'],'grounding_truncated':g['generation_truncated'],'vqa_source':str(Path(v['local_json_path']).relative_to(ROOT)),'grounding_source':str(Path(g['local_json_path']).relative_to(ROOT))};per.append(row);observations.append(row)
    maps=[r for r in spatial_metrics if r['model']==model and r['query_type']!='positive' and r['task']=='grounding' and r['token_aggregation'].startswith('coordinate tokens') and not r['truncated'] and r['current_coordinate_geometry_positive_area']]
    refs=[r for r in maps if r['positive_reference_complete_valid_coordinate_token_map']]
    stats['negative_complete_coordinate_maps']=len(maps);stats['anchor_density_above_uniform_maps']=sum(r['anchor_attention_density_enrichment_over_uniform']>1 for r in maps)
    stats['cosine_eligible_complete_positive_reference_maps']=len(refs);stats['cosine_excluded_incomplete_or_invalid_positive_reference_maps']=len(maps)-len(refs);stats['cosine_ge_08_with_complete_positive_reference_maps']=sum(r['cosine_to_same_task_positive_query_map']>=.8 for r in refs)
    summary[model]={'counts':stats,'case_observations':per}

(ROOT/'observed_case_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
audit={'complete_case_count':15,'models':list(MODELS),'expected_upstream_records':300,'actual_upstream_records':len(records),'representative_figures':[{'png':r['png'],'pdf':r['pdf']} for r in representatives],'expected_fresh_candidate_verifier_records':300,'actual_fresh_candidate_verifier_records':sum(len(list((ROOT/'verifiers'/arm).glob('*.json'))) for arm in ['omni','jev']),'all_fresh_candidate_sha_bbox_image_crosschecks_passed':True,'frozen_original_policy_decisions_verified_count':5000,'selected_case_labels_used_for_fit_or_threshold':False,'all_tensor_finite_nonnegative_and_token_grid_alignment_checks_passed':True,'independent_omni_attention_maps_checked':sum(verifier_map(m,c['index'],h)[0] is not None for c in selection['cases'] for m in MODELS for h in HT),'attention_tensors':tensor_audit,'all_case_sheets':sheet_paths,'normalization':'Per-map mean image patch=1; shared colors within each sheet; color strength does not compare absolute attention across models.','grounding_before_after_uses_identical_attention_array':True,'paper_or_main_table_modified':False,'vision_spatial_projection_source_check':'vision_token_projection_qa.json','generation_budget_history':'generation_budget_manifest.json','visual_qa':'pending'}
(ROOT/'attention_experiment_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
lines=['# 15图 × General/RL 真实注意力案例：独立讨论报告','','这是一组按展示清晰度选择的定性案例，使用新的解释加答案提示，不能用本报告的计数替代论文基线/ASV主表，也不能从注意力直接推出错误的因果机制。原案例语义标记仍为 needs_human_review。','','使用 Qwen2.5-VL-7B（General）和 Vision-R1（RL），15图各1正4负，共300条VQA/grounding记录。完整响应、真实权重、词元ID、patch/grid与处理尺寸都保留；初次12条达到160-token上限，仅这12条提高至512，仍截断者再提高至1024，完整保存各预算的原记录。最终仍截断的响应明确作为未完成前缀，不作完整VQA判断；已发出的四坐标仅作为prefix候选接受独立ASV评分。解析缺失或无效坐标单独标记。','','ASV对本次新生成的同一候选重新执行原Omni红框协议、JEV文字坐标协议和适用的opposed rivals。两模型原固定cross-fit头/阈值从原source复原，与原5000条判断逐行一致；15例标签未用于拟合或调阈值。ASV过滤前后用同一上游注意力数组，区别仅在候选框是否保留；Omni是独立验证器注意力。','','热图由真实输出词元到图像patch的decoder attention绘制。解释是greedy生成后的teacher-forced因果重放，每图保留最后4层28头的选定decision/coordinate-token权重和全部输出token的head均值。相对热图按每map的图像patch均值归一到1，低权重透明，使用Blues蓝色与原照片对比。每case/model全部map的95%分位设置共用颜色上限，超出值截顶并标箭头；颜色不能比较跨模型绝对注意力强度。绝对图像注意力mass另保留在元数据。','','## 本次观测计数（每模型75查询，含15正/60负；四坐标可含显式标记的截断prefix）','','|模型|显式VQA yes / no / 未解析|DG四坐标 / 无解析框|VQA no且DG输出框的负例|ASV拒绝 / 保留已输出候选|无效框顺序或零面积|截断记录|','|---|---|---|---|---|---|---|']
for m,d in summary.items():
    c=d['counts'];lines.append(f'|{MODELS[m]}|{c["vqa_explicit_yes"]} / {c["vqa_explicit_no"]} / {c["vqa_unparsed"]}|{c["grounding_parseable_coordinates"]} / {c["grounding_no_parsed_bbox"]}|{c["negative_vqa_no_with_coordinates"]}|{c["asv_rejected_emitted_candidates"]} / {c["asv_retained_emitted_candidates"]}|{c["grounding_invalid_order_or_zero_area"]}|{c["generation_truncated_records"]}|')
lines+=['','可讨论的观测是：同一图像和表达式，VQA的明确否定与grounding坐标输出可以并存；ASV作用于候选，不重新改变生成模型的注意力。下面给出逐例索引。注意力图可描述关注分布，不能作为该现象发生原因的证明。','','## VQA明确no、DG仍输出四坐标的逐例来源','']
for m,d in summary.items():
    rr=[r for r in d['case_observations'] if r['type']!='positive' and r['vqa_decision']=='no' and r['grounding_bbox']]
    lines.append(f'### {MODELS[m]}：{len(rr)}例\n')
    lines.extend([f'- #{r["case"]:02d} / {r["type"]}: `{r["query"]}`；bbox `{r["grounding_bbox"]}`；ASV `{r["asv"]}`；来源 `{Path(r["grounding_source"]).name}`。' for r in rr])
lines+=['','## 解释限制','','- 新提示要求简短解释，区别于论文标准yes/no或直接框输出协议；这里的计数仅描述本次推理，未更新论文任何基线或ASV值。','- 没有解析框不能一律解释为模型主动拒绝；可能是none、格式错误或达到输出上限。完整响应已保存。','- 两条原Omni输入因坐标顺序无效而不能画红框；固定策略在验证器缺失时保留原候选，已明确标为fallback retain。另有零面积框，保留真实坐标而不修正。','- 没有negative GT框。图中橙框是本次模型候选，独立Omni列红框显示其真实红框输入协议；代表图若附绿虚线，只表示supported-positive anchor。','- Qwen2.5/Vision-R1 decoder attention来自真实权重，但只是所选层/头/token聚合；不同聚合选择可能改变可视化。','- ASV的判定与独立Omni热图不是上游重推理，更不代表attention变化本身使模型纠错。','','## 文件','','- `visuals/`：30份完整case/model热图总表（PNG/PDF），覆盖全部15图和两模型、每图1正4负。','- `results/`：300份真实推理JSON与NPZ；`verifiers/`：300份新候选原Omni/JEV评分及适用rivals。','- `fresh_asv_decisions.json`：150个本次候选按原固定头/阈值计算的保留/拒绝。','- `visualization_metadata.json`：热图具体token、图像attention mass、归一化、颜色范围和ASV来源。','- `observed_case_summary.json`：逐case/task观测与计数，可复核本报告。']
lines+=['','## 注意力分布的描述性观测','','这里把支持正查询的源框仅作为 supported-positive anchor；它不是负查询的GT框。先按真实图像patch权重计算落在该锚点中的图像attention比例，再除以锚点面积比例；大于1表示该锚点中的密度超过全图均匀分布。正负查询map的余弦相似度≥0.8仅是预设描述分组，不是ASV阈值，不能推出注意力导致错误。规律计数仅纳入完整、正面积的负查询四坐标；无效坐标map仍保留原始记录，但不进入规律汇总。余弦比较还要求正参照完整且为正面积四坐标coordinate-token map，截断或无解析坐标的参照单独列出。patch内按均匀权重处理部分相交，全部值和来源见 `attention_spatial_metrics.json`。','']
for m in MODELS:
    eligible=[r for r in spatial_metrics if r['model']==m and r['query_type']!='positive' and r['task']=='grounding' and r['token_aggregation'].startswith('coordinate tokens') and not r['truncated'] and r['current_coordinate_geometry_positive_area']]
    dense=[r for r in eligible if r['anchor_attention_density_enrichment_over_uniform']>1]
    cosine_eligible=[r for r in eligible if r['positive_reference_complete_valid_coordinate_token_map']]
    excluded_reference=[r for r in eligible if not r['positive_reference_complete_valid_coordinate_token_map']]
    similar=[r for r in cosine_eligible if r['cosine_to_same_task_positive_query_map']>=.8]
    lines.append(f'### {MODELS[m]}：完整、正面积负查询坐标-token map 共{len(eligible)}条')
    lines.append(f'其中{len(dense)}条在supported-positive anchor内的attention密度高于全图均匀密度，覆盖{len(set(r["case"] for r in dense))}张照片；另有{len(cosine_eligible)}条具有完整、正面积四坐标的正查询coordinate-token参照，其中{len(similar)}条与同任务正查询的聚合map余弦相似度≥0.8，覆盖{len(set(r["case"] for r in similar))}张照片。查询共享同一照片，条数不代表独立样本。')
    lines.append('密度>1来源：'+('; '.join(f'#{r["case"]:02d}/{r["query_type"]} (ratio={r["anchor_attention_density_enrichment_over_uniform"]:.2f}, [{Path(r["source"]).name}]({r["source"]}))' for r in dense) or '无')+'。')
    lines.append('不纳入完整正负map相似度汇总的参照：'+('; '.join(f'#{r["case"]:02d}/{r["query_type"]} (positive truncated={r["positive_reference_truncated"]}, token aggregation={r["positive_reference_token_aggregation"]})' for r in excluded_reference) or '无')+'。这些负map仍可独立用于anchor density度量。')
    lines.append('相似度≥0.8来源：'+('; '.join(f'#{r["case"]:02d}/{r["query_type"]} (cos={r["cosine_to_same_task_positive_query_map"]:.3f}, [{Path(r["source"]).name}]({r["source"]}))' for r in similar) or '无')+'。')
lines+=['','## 代表图选择','','选择#08属性负例和#02关联负例展示VQA否定与DG候选并存、ASV过滤；选择#12关系负例检查错误关系回答与结构验证；#14物体负例展示两个模型的坐标输出/无解析框差异。图为候选，未改论文。具体输入、输出、token与决定可在 `representative_candidates/manifest.json` 追溯。','','## 生成预算与资源','','初次12条截断逐条用原prompt、greedy补到512，仍截断者补到1024，其余288条不重跑。最终'+str(budget_manifest['final_still_capped_count'])+'条在1024仍未完整结束；不把未完成的prefix当完整VQA判断。已发出坐标的prefix有单独候选评分和显式标签。每轮用时、峰值显存与原记录见 `generation_budget_manifest.json`。','']
(ROOT/'讨论报告_真实注意力案例.md').write_text('\n'.join(lines)+'\n')
import html as htmllib
def report_inline(line):
    line=htmllib.escape(line)
    line=re.sub(r'\[([^]]+)\]\(([^)]+)\)',lambda m:'<a href="'+m[2]+'">'+m[1]+'</a>',line)
    line=re.sub(r'`([^`]+)`',r'<code>\1</code>',line)
    return line
report_blocks=[];in_table=False
for line in lines:
    if line.startswith('|'):
        if not in_table:report_blocks.append('<table>');in_table=True
        cells=line.strip('|').split('|')
        if all(re.fullmatch(r'[-: ]+',c) for c in cells):continue
        report_blocks.append('<tr>'+''.join('<td>'+report_inline(c.strip())+'</td>' for c in cells)+'</tr>');continue
    if in_table:report_blocks.append('</table>');in_table=False
    if line.startswith('#'):
        level=len(line)-len(line.lstrip('#'));report_blocks.append(f'<h{level}>'+report_inline(line[level:].strip())+f'</h{level}>')
    elif line.strip():report_blocks.append('<p>'+report_inline(line.lstrip('- '))+'</p>')
if in_table:report_blocks.append('</table>')
(ROOT/'discussion_report.html').write_text('<!doctype html><meta charset="utf-8"><title>真实注意力案例：独立讨论报告</title><style>body{font:16px/1.75 system-ui;color:#263348;margin:30px auto;padding:0 25px;max-width:1200px}table{border-collapse:collapse;width:100%;font-size:14px}td{border:1px solid #ced6e3;padding:8px}tr:first-child{font-weight:600;background:#eef2f8}code{background:#f1f4f8;padding:2px 4px;border-radius:3px;font-size:.9em}a{color:#385faf}p{overflow-wrap:anywhere}</style><p><a href="index.html">返回案例图库</a> · <a href="讨论报告_真实注意力案例.md">Markdown原报告</a></p>'+''.join(report_blocks))

gallery={'models':MODELS,'types':HT_LABEL,'cases':[{'index':c['index'],'positive':c['positive_text']} for c in selection['cases']],'rows':{f'{r["model"]}|{r["case"]}|{r["query_type"]}':r for r in plot_metadata},'records':{f'{m}|{i}|{h}|{t}':{'response':r['response'],'truncated':r['generation_truncated'],'max_new_tokens':r['max_new_tokens'],'npz':f'results/{m}/{i:02d}_{h}_{t}.npz','json':f'results/{m}/{i:02d}_{h}_{t}.json','grid':r['merged_grid_thw'],'prompt':r['prompt']} for (m,i,h,t),r in records.items()},'summary':{m:d['counts'] for m,d in summary.items()}}
payload=json.dumps(gallery,ensure_ascii=False).replace('<','\\u003c')
html=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RRWR 真实注意力案例</title><style>body{font-family:system-ui,-apple-system,sans-serif;color:#263348;background:#f7f9fc;margin:0}main{max-width:1500px;margin:auto;padding:26px}h1{font-size:27px;margin:0 0 12px}p{line-height:1.6}section{background:white;border:1px solid #dce3ed;border-radius:12px;padding:18px;margin:15px 0}.controls{display:flex;gap:20px;flex-wrap:wrap}label{display:flex;flex-direction:column;gap:6px;font-weight:600}select{font:inherit;padding:9px;border:1px solid #bbc6d6;border-radius:7px;max-width:470px}img{width:100%;display:block}pre{max-height:360px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:14px;line-height:1.5;color:#22334d}a{color:#385faf}small{color:#596880}.tag{display:inline-block;padding:5px 9px;background:#eaf0f9;border-radius:5px;margin:0 6px 8px 0}.warn{color:#a63655}details{margin-top:12px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:850px){.cols{grid-template-columns:1fr}}</style><main><h1>15 张案例 · 两模型 · 真实注意力</h1><p>照片叠加来自真实 decoder attention。ASV 对本次生成的同一个候选重新打分；过滤前后使用完全相同的上游注意力图，仅候选框保留状态改变。Omni 是独立验证器。</p><div class="controls"><label>案例<select id="cas"></select></label><label>模型<select id="mod"></select></label><label>查询类型<select id="typ"></select></label></div><section><h2 id="query"></h2><div id="status"></div><a id="imageLink" target="_blank"><img id="image" alt="真实注意力：VQA、grounding before、ASV后同attention和独立Omni"></a><small>每张map除以其图像patch均值，得到 relative attention；低权重透明。本行共享色条，上限为同case/model全部map的95%分位，更高值截顶（箭头）。颜色强弱不能比较跨模型的绝对注意力强度。注意力分布不能证明错误发生的因果机制。</small></section><section><h2>原始响应与可追溯数据</h2><div class="cols"><div><h3>VQA</h3><div id="vlinks"></div><pre id="vresp"></pre></div><div><h3>Grounding</h3><div id="glinks"></div><pre id="gresp"></pre></div></div><details><summary>词元选择、网格与本次 ASV 判定</summary><pre id="meta"></pre></details></section><section><h2>独立讨论报告</h2><p>这是新解释加答案提示下的非随机定性子集，不替代论文基线或 ASV 主表。原案例标注状态为 needs_human_review；未用这 15 例标签拟合或调阈值。</p><p><a href="discussion_report.html">中文逐例观测报告</a> · <a href="observed_case_summary.json">完整观测计数与来源</a> · <a href="fresh_asv_decisions.json">150 个新候选的 ASV 判定</a> · <a href="attention_experiment_audit.json">完整审计</a> · <a href="restored_asv_policy.json">原固定头与阈值</a> · <a href="vision_token_projection_qa.json">视觉词元空间映射核查</a> · <a href="generation_budget_manifest.json">12条生成预算补跑记录</a></p><pre id="summary"></pre></section><section><h2>全案例总表与代表图</h2><p id="sheet"></p><p><a href="representative_candidates/index.html">代表图候选文件夹</a>（候选图供讨论，尚未写入论文）</p></section></main><script>const DATA=__PAYLOAD__;const $=s=>document.getElementById(s);for(const c of DATA.cases){$('cas').add(new Option('#'+String(c.index).padStart(2,'0')+' '+c.positive,c.index))}for(const [k,v] of Object.entries(DATA.models))$('mod').add(new Option(v,k));for(const [k,v] of Object.entries(DATA.types))$('typ').add(new Option(v,k));function update(){const m=$('mod').value,i=$('cas').value,h=$('typ').value,r=DATA.rows[m+'|'+i+'|'+h],a=r.asv;const v=DATA.records[m+'|'+i+'|'+h+'|vqa'],g=DATA.records[m+'|'+i+'|'+h+'|grounding'];$('query').textContent=a.query;const path='visuals/query_'+i.padStart(2,'0')+'_'+m+'_'+h+'.png';$('image').src=path;$('imageLink').href=path;$('status').replaceChildren();for(const s of ['ASV: '+a.reason,'Support: '+(a.support_score===null?'unavailable':a.support_score.toFixed(4)),'Fold: '+a.fold,'VQA grid '+v.grid.join('×'),'DG grid '+g.grid.join('×')]){const e=document.createElement('span');e.className='tag';e.textContent=s;$('status').append(e)}for(const [z,d] of [['v',v],['g',g]]){$(z+'resp').textContent=(d.truncated?'[达到'+d.max_new_tokens+'-token上限：未完成的输出前缀]\n':'')+d.response;$(z+'links').innerHTML='<a href="'+d.json+'">JSON原记录</a> · <a href="'+d.npz+'">NPZ真实权重</a>'}$('meta').textContent=JSON.stringify({vqa_tokens:r.vqa_tokens,vqa_token_pieces:r.vqa_token_pieces,grounding_tokens:r.grounding_tokens,grounding_token_pieces:r.grounding_token_pieces,vqa_image_attention_mass:r.vqa_attention_image_mass,grounding_image_attention_mass:r.grounding_attention_image_mass,normalization:r.normalization,colorbar_vmax:r.colorbar_vmax,asv:a,prompts:{vqa:v.prompt,grounding:g.prompt}},null,2);$('summary').textContent=JSON.stringify(DATA.summary,null,2);$('sheet').innerHTML='<a href="visuals/case_'+i.padStart(2,'0')+'_'+m+'_all_queries.png">当前模型五查询总表 PNG</a> · <a href="visuals/case_'+i.padStart(2,'0')+'_'+m+'_all_queries.pdf">PDF</a>'}for(const id of ['cas','mod','typ'])$(id).addEventListener('change',update);$('cas').value='8';$('typ').value='attribute';update();</script></html>'''.replace('__PAYLOAD__',payload)
(ROOT/'index.html').write_text(html)
print(json.dumps({'sheets':len(sheet_paths),'records':len(records),'asv_decisions':len(asv),'summary':{m:d['counts'] for m,d in summary.items()}},ensure_ascii=False))
