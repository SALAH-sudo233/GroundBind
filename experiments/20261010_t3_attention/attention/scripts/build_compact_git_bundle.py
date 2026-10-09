"""Create a compact audit bundle; preserve full raw tensors only in desktop package.

No Git write or upload occurs here. Fonts, photographs and full raw tensors are excluded.
"""
from pathlib import Path
import ast,json,hashlib,re,shutil,argparse
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root');ap.add_argument('--verify',action='store_true');ap.add_argument('--render-representatives',action='store_true');ap.add_argument('--bundle-root');ap.add_argument('--image-root');ap.add_argument('--font-file');args=ap.parse_args()
R=Path(args.root).resolve() if args.root else Path(__file__).resolve().parent;B=R/'git_sync_bundle'
if not args.verify:B.mkdir(exist_ok=True)
if args.render_representatives:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from PIL import Image
    B=Path(args.bundle_root).resolve() if args.bundle_root else (R/'git_sync_bundle')
    image_root=Path(args.image_root).resolve() if args.image_root else B/'images'
    if args.font_file:font_manager.fontManager.addfont(args.font_file)
    family='Times New Roman' if any(f.name=='Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
    plt.rcParams.update({'font.family':family,'font.size':8,'pdf.fonttype':42})
    dest=B/'reproduced_representatives';dest.mkdir(exist_ok=True)
    decision_tree=ast.parse((B/'scripts/build_attention_visuals.py').read_text());decision_ns={'re':re};exec(compile(ast.Module(body=[n for n in decision_tree.body if isinstance(n,ast.FunctionDef) and n.name=='decision'],type_ignores=[]),'<decision_parser>','exec'),decision_ns)
    decisions={(r['model'],r['example_index'],r['hallucination_type']):r for r in json.load(open(B/'asv_and_run_metadata/fresh_asv_decisions.json'))}
    for index,ht in [(8,'attribute'),(12,'relation'),(2,'co_occurrence'),(14,'object')]:
        rows=[];pool=[]
        for m in ['qwen25','visionr1']:
            v=json.load(open(B/f'metadata/results/{m}/{index:02d}_{ht}_vqa.json'));g=json.load(open(B/f'metadata/results/{m}/{index:02d}_{ht}_grounding.json'))
            vm=np.load(B/f'compact_attention/{m}/{index:02d}_{ht}_vqa.npz')['relative_image_attention'];gm=np.load(B/f'compact_attention/{m}/{index:02d}_{ht}_grounding.npz')['relative_image_attention'];o=B/f'compact_attention/omni/{m}__{index:02d}_{ht}_grounding.npz';om=np.load(o)['relative_image_attention'] if o.exists() else None
            pool.extend([vm.ravel(),gm.ravel()]);pool.extend([om.ravel()] if om is not None else []);rows.append((m,v,g,vm,gm,om,decisions[(m,index,ht)]))
        vmax=max(1,float(np.quantile(np.concatenate(pool),.95)));fig,axes=plt.subplots(2,4,figsize=(7.6,5),gridspec_kw={'wspace':.08,'hspace':.42});fig.suptitle(f'Case {index:02d} | {ht}\n'+rows[0][1]['query'],fontsize=10,y=.985)
        for row,(m,v,g,vm,gm,om,a) in enumerate(rows):
            image=Image.open(image_root/g['image_filename']).convert('RGB');w,h=image.size
            assert hashlib.sha256((image_root/g['image_filename']).read_bytes()).hexdigest()==g['image_sha256']
            assert hashlib.sha256((B/f'metadata/results/{m}/{index:02d}_{ht}_grounding.json').read_bytes()).hexdigest()==a['source_record_sha256']
            answer=decision_ns['decision'](v) or 'unparsed'
            for col,(att,bbox,title) in enumerate(zip([vm,gm,gm,om],[None,g['parsed_bbox_xyxy'],a['asv_bbox_xyxy'],g['parsed_bbox_xyxy'] if om is not None else None],[('Qwen2.5-VL' if m=='qwen25' else 'Vision-R1')+' | VQA: '+answer,'DG: '+('coordinates' if g['parsed_bbox_xyxy'] is not None else 'no parsed bbox'),'DG w/ ASV: '+('KEEP' if a['keep'] else 'REJ' if a['candidate_bbox_xyxy'] else 'NO PARSED BOX')+'\nSame upstream attention','Independent OmniVerifier'])):
                ax=axes[row,col];ax.imshow(image)
                if att is not None:
                    rgba=plt.colormaps['Blues'](Normalize(0,vmax)(att));rgba[...,3]=.67*np.clip(att/vmax,0,1)**.7;ax.imshow(rgba,extent=[0,w,h,0],interpolation='bilinear')
                if bbox is not None and bbox[2]>bbox[0] and bbox[3]>bbox[1]:ax.add_patch(Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],fill=False,edgecolor='#CA3B3B' if col==3 else '#F18D36',linewidth=1.6))
                ax.set(xlim=(0,w),ylim=(h,0));ax.axis('off');ax.set_title(title,fontsize=7.4)
        fig.subplots_adjust(left=.02,right=.985,top=.83,bottom=.24);cax=fig.add_axes([.28,.16,.45,.023]);fig.colorbar(ScalarMappable(norm=Normalize(0,vmax),cmap='Blues'),cax=cax,orientation='horizontal',extend='max').set_label('Relative attention (mean image patch = 1)',fontsize=7.5)
        fig.text(.025,.018,'Real replay rows consuming selected VQA decision / DG coordinate tokens; last 4 layers, mean heads.\nOrange: model candidate; red: verifier input candidate. ASV filters the same candidate and attention.\nPer-map mean normalization; pooled 95th percentile color clipping. Colors do not compare absolute attention across models.',fontsize=7)
        stem=f'case_{index:02d}_{ht}_two_models';fig.savefig(dest/(stem+'.png'),dpi=260);fig.savefig(dest/(stem+'.pdf'));plt.close(fig)
    print('Reproduced four representative overlays from compact real mean grids and 15 original photographs. No full raw tensors or font installation needed.');raise SystemExit(0)

source=ast.parse((R/'build_attention_visuals.py').read_text());funcs=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ['decision','chosen_tokens']]
ns={'np':np,'re':re};exec(compile(ast.Module(body=funcs,type_ignores=[]),'<faithful_token_selection>','exec'),ns)
if args.verify:
    proof=json.load(open(B/'manifest.json'));n=0
    for row in proof['compact_upstream_maps']:
        p=R/row['source'];m=json.load(open(p));z=np.load(p.with_suffix('.npz'));ii,label=ns['chosen_tokens'](m);g=m['merged_grid_thw'];raw=z['output_token_attention_head_mean'][:,ii,:].astype(np.float32).mean((0,1)).reshape(g[1],g[2]);c=np.load(B/row['file'])
        assert row['source_record_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
        assert row['source_raw_npz_sha256']==hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()
        assert np.array_equal(c['selected_output_token_indices'],ii) and np.array_equal(c['raw_image_attention_mean'],raw)
        assert str(c['source_record_sha256'])==row['source_record_sha256'];assert np.allclose(c['relative_image_attention'],raw/(raw.mean()+1e-12),atol=0,rtol=0);n+=1
    for row in proof['compact_omni_maps']:
        p=R/row['source'];m=json.load(open(p));z=np.load(p.with_suffix('.npz'));g=m['original']['merged_grid_thw'];raw=z['decision_producing_attention_raw_heads'].astype(np.float32).mean((0,1)).reshape(g[1],g[2]);c=np.load(B/row['file'])
        assert row['source_record_sha256']==hashlib.sha256(p.read_bytes()).hexdigest() and row['source_raw_npz_sha256']==hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()
        assert np.array_equal(c['raw_image_attention_mean'],raw);n+=1
    assert n==405;print('Verified 405 compact maps exactly against full raw tensors, source JSON SHA, raw NPZ SHA and selected token indices.');raise SystemExit(0)
manifest={'scope':'Compact Git mirror of the same completed illustrative run. Full raw output/head NPZ and budget archives remain in desktop package; not included here.','excluded':['fonts','original_photographs','full_raw_output_and_head_attention_tensors','full_case_sheets','budget_raw_tensor_archives'],'upstream_metadata':[],'compact_upstream_maps':[],'new_verifier_metadata':[],'compact_omni_maps':[],'representative_pngs':[],'scripts':[]}
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);return str(dst.relative_to(B))
for p in sorted(R.glob('results/*/*.json')):
    m=json.load(open(p));z=np.load(p.with_suffix('.npz'));ii,label=ns['chosen_tokens'](m);raw=z['output_token_attention_head_mean'][:,ii,:].astype(np.float32).mean((0,1));g=m['merged_grid_thw'];raw=raw.reshape(g[1],g[2]);dst=B/'compact_attention'/m['model']/(p.stem+'.npz');dst.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dst,raw_image_attention_mean=raw,relative_image_attention=raw/(raw.mean()+1e-12),selected_output_token_indices=np.array(ii,dtype=np.int32),selected_output_token_ids=np.array(m['generated_token_ids'],dtype=np.int64)[ii],merged_grid_thw=np.array(g),layer_indices=z['layer_indices'],source_record_sha256=np.array(hashlib.sha256(p.read_bytes()).hexdigest()),original_raw_npz_sha256=np.array(hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()),attention_row_semantics=np.array('Consumed selected emitted output-token rows in exact teacher-forced causal replay.'))
    manifest['upstream_metadata'].append(copy(p,B/'metadata/results'/m['model']/p.name));manifest['compact_upstream_maps'].append({'file':str(dst.relative_to(B)),'source':str(p.relative_to(R)),'token_aggregation':label,'source_record_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_raw_npz_sha256':hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest(),'compact_npz_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'image_attention_mass':float(raw.sum())})
for arm in ['omni','jev']:
    for p in sorted((R/'verifiers'/arm).glob('*.json')):
        m=json.load(open(p));manifest['new_verifier_metadata'].append(copy(p,B/'metadata/verifiers'/arm/p.name));o=m.get('original') or {}
        if arm!='omni' or not o.get('attention_available'):continue
        z=np.load(p.with_suffix('.npz'));g=o['merged_grid_thw'];raw=z['decision_producing_attention_raw_heads'].astype(np.float32).mean((0,1)).reshape(g[1],g[2]);dst=B/'compact_attention/omni'/p.with_suffix('.npz').name;dst.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dst,raw_image_attention_mean=raw,relative_image_attention=raw/(raw.mean()+1e-12),merged_grid_thw=np.array(g),layer_indices=z['layer_indices'],source_record_sha256=np.array(hashlib.sha256(p.read_bytes()).hexdigest()),original_raw_npz_sha256=np.array(hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()),attention_row_semantics=np.array('Original independent verifier final-prompt row producing support logits.'))
        manifest['compact_omni_maps'].append({'file':str(dst.relative_to(B)),'source':str(p.relative_to(R)),'source_record_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_raw_npz_sha256':hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest(),'compact_npz_sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
for name in ['selection_15.json','restored_asv_policy.json','fresh_asv_decisions.json','generation_budget_manifest.json','vision_token_projection_qa.json','attention_experiment_audit.json']:
    copy(R/name,B/'asv_and_run_metadata'/name)
for p in sorted((R/'representative_candidates').glob('*.png')):manifest['representative_pngs'].append(copy(p,B/'representative_candidates'/p.name))
for name in ['run_real_attention.py','run_fresh_verifiers.py','restore_asv_policy.py','build_attention_visuals.py','build_compact_git_bundle.py']:
    manifest['scripts'].append(copy(R/name,B/'scripts'/name))
assert len(manifest['upstream_metadata'])==300 and len(manifest['compact_upstream_maps'])==300 and len(manifest['new_verifier_metadata'])==300 and len(manifest['compact_omni_maps'])==105 and len(manifest['representative_pngs'])==4
assert not list(B.rglob('*.ttf'))
manifest['counts']={k:len(v) for k,v in manifest.items() if isinstance(v,list) and k!='excluded'};manifest['total_bytes']=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());(B/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(B/'README.txt').write_text('This compact mirror contains the same 15-photo, two-model illustrative run.\n300 upstream JSON records, 300 mean-layer/head/token image-attention grids, 300 new Omni/JEV verifier records and 105 actual independent Omni mean-head grids, frozen ASV policy and four representative PNGs.\nFull raw token/head NPZ, original photographs, fonts and budget raw NPZ archives are preserved in the separate desktop browser package; none are included here.\nAll mean grids derive directly from saved real attention tensors, with source raw SHA256 and selected output-token indices. No bbox-derived heatmaps. The compact export includes source-record SHA256, source full-raw NPZ SHA256, actual selected emitted-token indices and IDs. Use --verify to recompute all 405 compact means exactly from the complete desktop raw tensors.\nBlue visualization maps are relative to each grid mean, with pooled 95th-percentile color clipping; colors do not compare absolute attention across models.\nGeneration metadata is not an update to the paper main evaluation. ASV filters the same fresh candidate, with no upstream attention change.\nThe scripts expect the complete desktop experiment root for rerendering; photographs and full raw tensors intentionally are not in Git. Times New Roman is optional from a user-provided local fonts directory, never installed or committed here.\n')
print(json.dumps(manifest['counts'],indent=2));print('bundle_bytes',manifest['total_bytes'])
