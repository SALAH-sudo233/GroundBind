"""Restore the frozen cross-fit ASV readout/thresholds, using existing score files."""
import sys,json,pathlib,inspect,hashlib
ROOT=pathlib.Path('/home/u2025141034/benchmark/rrwr_20261010_t3_attention/attention')
SRC=pathlib.Path('/home/u2025141034/SVD/agentic_probe')
sys.path.insert(0,str(SRC))
import eval_upstream as E
import eval_cbr_paper_aligned as A
import eval_unified as U
import eval_attribution as T
import eval_paper_tables as P
U.PROBE=str(SRC);U.S5='/home/u2025141034/SVD/grpo_verifier/s5grpo'
canon=json.loads((SRC/'canon_roots_paper.json').read_text())
po=U.load_probe('probe_textroute_all.jsonl','z');pj=U.load_probe('trprobe_jev_all.jsonl','z_head')
def fit(rows,mode):
    rr=[r for r in rows if U.usable(r,mode)]
    if len(rr)<30 or len(set(r['label'] for r in rr))<2:return None,None
    f,_=E.logreg([U.feats(mode,r) for r in rr],[r['label'] for r in rr])
    params={k:v for k,v in inspect.getclosurevars(f).nonlocals.items() if k in ('w','b','mu','sd','k')}
    params['n_fit']=len(rr)
    return lambda r:f(U.feats(mode,r)),params
out={'source_path':str(SRC),'policy':'paper_tables.full / final_policy.A_reject_opp','target':.95,'models':{},'files_sha256':{}}
for name in ('qwen2.5-vl-7b','Vision-R1'):
    pos,neg=A.load_boxes(name,canon[name]);elig={q['sid'] for q in pos.values() if A.valid_box(q['pred']) and A.iou(q['pred'],q['gt'])>=.5}
    qmap=T.query_map(canon[name],name);rows=U.merge(name,po,pj,False);thresholds={};keep={}
    gate=lambda r:(T.predicate_of(qmap.get(r['sid'],'')) or '') in P.OPPOSED
    for tf in (0,1):
        fitting=[r for r in rows if r['fold']!=tf];test=[r for r in rows if r['fold']==tf]
        sup,sp=fit(fitting,'support');assert sup is not None
        ps=sorted(sup(r) for r in fitting if r['is_pos'] and r['sid'] in elig and U.usable(r,'support'))
        k=int(round((1.0-.95)*len(ps)));ts=ps[max(0,min(k,len(ps)-1))] if ps else float('-inf')
        def ks(r):return False if not r['drew'] else True if not U.usable(r,'support') else sup(r)>=ts
        adm=[r for r in fitting if U.usable(r,'full') and gate(r)]
        full,fp=fit(adm,'full') if len(adm)>=30 and len(set(r['label'] for r in adm))>=2 else (None,None)
        ep=[r for r in fitting if r['is_pos'] and r['sid'] in elig and r['drew']]
        cov=[r for r in ep if U.usable(r,'full') and gate(r)];unc=[r for r in ep if not(U.usable(r,'full') and gate(r))]
        fixed=sum(ks(r) for r in unc);need=max(0,min(int(round(.95*len(ep)))-fixed,len(cov)))
        sc=sorted((full(r) for r in cov),reverse=True) if full else []
        full_threshold=sc[need-1] if need>0 and full else float('inf')
        thresholds[str(tf)]={'test_fold':tf,'fit_fold':1-tf,'support_head':sp,'support_threshold':ts,'full_head':fp,'full_threshold':full_threshold,'n_fit_eligible':len(ep),'n_routed_eligible':len(cov),'fixed_uncov_keep':fixed,'need':need}
        for r in test:
            routed=U.usable(r,'full') and gate(r) and full is not None
            keep[r['sid']]=ks(r) and full(r)>=full_threshold if routed else ks(r)
    frozen=P.build(rows,elig,.95,'full',gate=gate,monotone=True)
    assert keep==frozen,'Restored ASV readout differs from frozen evaluator'
    out['models'][name]={'n_rows':len(rows),'n_c':len(elig),'thresholds':thresholds,'all_original_decisions_match_frozen_evaluator':True,'fold_definition':'md5(base COCO image identity) modulo 2, from eval_upstream.fold'}
    print(name,'policy restore verified',len(rows),len(elig),flush=True)
for fn in ('eval_upstream.py','eval_cbr_paper_aligned.py','eval_unified.py','eval_attribution.py','eval_paper_tables.py','simple_relations.py','collect_jevhead.py'):
    p=SRC/fn;out['files_sha256'][fn]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'restored_asv_policy.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('ASV policy restoration complete',flush=True)
