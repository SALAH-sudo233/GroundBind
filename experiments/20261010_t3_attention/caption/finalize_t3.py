#!/usr/bin/env python3
"""Finalize the completed T3 experiment without changing locked scoring code."""
import collections, csv, gzip, hashlib, json, pathlib, shutil, subprocess, time
out=pathlib.Path('/home/u2025141034/benchmark/rrwr_20261010_t3_attention/t3')
script=out/'t3_semantic_eval.py'
lock=json.loads((out/'protocol_lock.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(script)==lock['locked_script_sha256']
assert sha(out/'heldout_controls_independent.jsonl')==lock['heldout_file_sha256_predeclared']
subprocess.run(['/home/u2025141034/.miniconda3/envs/mllm_ayb/bin/python',str(script),'summarize'],check=True)
shutil.copyfile(out/'t3_summary.json',out/'t3_summary_core.json')
s=json.loads((out/'t3_summary_core.json').read_text())
reports={name:json.loads((out/file).read_text()) for name,file in [('development_v1','development_v1/control_summary.json'),('development_v2','control_summary.json'),('development_final','control_final_summary.json'),('heldout_independent','heldout_summary.json')]}
s['control_validation']={name:{'n':d['n'],'n_valid':d['n_valid'],'correct':d['correct'],'accuracy':d['accuracy_all_controls'],'by_category':d['by_category'],'n_failed':len(d['failed_controls'])} for name,d in reports.items()}
s['control_accuracy']=reports['heldout_independent']['accuracy_all_controls']
s['control_n']=reports['heldout_independent']['n']
s['control_accuracy_scope']='Independent authored held-out textual diagnostic set; not human-review benchmark accuracy.'
s['protocol_lock']=lock
s['locked_scoring_script_sha256']=sha(script)
s['independent_control_source_sha256']=sha(out/'heldout_controls_independent.jsonl')
s['source_annotation_scope_note']='Object proposition scope is resolved from existing annotated inventories; no judgment is assigned by keyword checks.'
s['evaluator_limitations']='The independent judge made one held-out quoted-claim error out of 64. All benchmark labels here are model-assisted semantic judgments; developmental and held-out failures remain available.'
(out/'t3_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
order=['Qwen3-VL-8B','qwen2.5-vl-7b','InternVL3.5-8B','llava-ov-7b','LENS','Orsta-7B','Seg-R1','Seg-zero','TreeVGR','UniVG-R1','Vision-R1','VisionReasoner','visual-rft']
with open(out/'t3_main_table_rows.csv','w') as f:
 writer=csv.writer(f);writer.writerow(['model','family','caption_n','object_count','association_count','attribute_count','relation_count','EOH_count','EOH_n','EOH_FAR_percent','ROH_count','ROH_n','ROH_FAR_percent'])
 for m in order:
  r=s['per_model'][m];writer.writerow([m,r['family'],500,*[r['by_type'][h]['assertion_count'] for h in ['object','co_occurrence','attribute','relation']],r['EOH']['assertion_count'],r['EOH']['n_units'],r['EOH']['FAR_percent'],r['ROH']['assertion_count'],r['ROH']['n_units'],r['ROH']['FAR_percent']])
for name in ['judgments_all.jsonl','cached_captions.jsonl','source_annotations.jsonl','judgment_inputs.jsonl']:
 with open(out/name,'rb') as src,gzip.open(out/(name+'.gz'),'wb') as dst:shutil.copyfileobj(src,dst)
manifest={'audit_epoch':time.time(),'completed_judgments':26000,'source_files_unchanged':s['source_files_unchanged'],'no_original_evaluation_or_model_files_written':True,'locked_judge_prompt_sha256':lock['locked_system_sha256'],'judge_script_sha256':sha(script),'heldout_sha256':sha(out/'heldout_controls_independent.jsonl'),'all_judge_outputs_valid':s['all_judgments_parse_valid'],'files':[]}
for p in sorted(out.rglob('*')):
 if not p.is_file() or p.name=='experiment_sha_audit.json':continue
 manifest['files'].append({'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
(out/'experiment_sha_audit.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'complete':True,'per_model':s['per_model'],'family_equal_model_means':s['family_equal_model_means'],'control_validation':s['control_validation']},ensure_ascii=False))
