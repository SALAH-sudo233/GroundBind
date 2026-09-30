#!/usr/bin/env python3
"""Collect verifier scores on JOINT grounding (t4) boxes, for t4 mitigation.

WHY A NEW COLLECTION IS UNAVOIDABLE. Every existing score file was collected on
t2 boxes: collect_jevhead.py and collect_probe_textroute.py both hard-filter
`task == 't2_vqa_grounding'`. The verifier scores a SPECIFIC box, so a t2 score
says nothing about the t4 box for the same query -- and the boxes genuinely differ
(t4 adds a caption, which moves the box). Reusing t2 scores here would be the same
class of error as mixing two scorers in one gap feature.

WHAT IS COLLECTED, per eligible t4 row:
  z0        Omni original-claim score on the t4 box (the support-verification arm)
  z_rival   Omni score for each competitive probe (<=2), same box, edited predicate
  z_jev     jev_verifier_v2 scalar head on the t4 box, temperature-calibrated

COORDINATE CORRECTION IS APPLIED BEFORE SCORING for UniVG-R1 and visual-rft,
reusing export_cf_joint.rescale(). Their stored t4 boxes are the 0-1000-as-pixels
defect (write_rescaled_run.py only repaired t2); scoring the broken box would
measure the defect, not the method.

HARD RULE OBSERVED HERE: z0 and z_rival must come from the SAME scorer, because
the gap feature is their difference. Both are Omni. z_jev is a separate arm and is
never subtracted from an Omni score.

Sharding is by plan index so 8 GPUs can run disjoint slices; output is append-only
and resume-safe by (model, sid, variant).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, '/home/u2025141034/SVD/grpo_verifier')
sys.path.insert(0, '/home/u2025141034/SVD/agentic_probe')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import simple_relations as sr
from deploy_upstream import parse_bbox, img_path
import export_cf_joint as CF

MAX_RIVALS = 2
COORDFIX_T4 = {'UniVG-R1', 'visual-rft'}
TASK = 't4_caption_grounding'


def build_plan(canon, models_filter=None):
    """One entry per eligible t4 row: (model, sid, box, query, htype, role, iou, probes)."""
    plan = []
    stats = {}
    for m, root in sorted(canon.items()):
        if models_filter and m not in models_filter:
            continue
        p = os.path.join(root, m, 'records.jsonl')
        if not os.path.exists(p):
            print('[skip] %s: no records.jsonl' % m, flush=True)
            continue
        cache = {}
        nrow = ncall = nfix = 0
        for line in open(p, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if str(r.get('task', '')) != TASK:
                continue
            # the box actually scored: corrected where the run left t4 broken
            if m in COORDFIX_T4:
                bb = CF.rescale(r, cache)
                if bb is not None:
                    nfix += 1
            else:
                bb = parse_bbox(r.get('pred_bbox_xyxy'))
            if bb is None or (bb[2] - bb[0]) <= 0 or (bb[3] - bb[1]) <= 0:
                continue
            q = r.get('query') or r.get('referring_expression') or ''
            probes, _ = sr.build(q, max_candidates=MAX_RIVALS)  # TEXT ONLY, no labels
            comp = [x for x in probes if x['probe_type'] == 'competitive']
            sid = r.get('sample_id') or r.get('base_sample_id')
            ip = img_path(r)
            if not ip:
                continue
            nrow += 1
            ncall += 1 + len(comp)          # z0 + rivals
            plan.append(dict(model=m, sid=sid, box=bb, query=q, img=ip,
                             htype=r.get('hallucination_type'),
                             role=r.get('query_role'),
                             label_exists=r.get('label_exists'),
                             iou=r.get('iou'), probes=comp,
                             coordfixed=m in COORDFIX_T4))
        stats[m] = dict(rows=nrow, omni_calls=ncall, coordfixed=nfix)
        print('%-18s rows=%5d omni_calls=%5d coordfixed=%5d'
              % (m, nrow, ncall, nfix), flush=True)
    return plan, stats


def load_jev():
    """Load jev_verifier_v2 exactly as collect_jevhead.main() does.

    The scalar head reads the hidden state at the LAST PROMPT POSITION, so the
    prompt must be byte-identical to the training template. collect_jevhead.PROMPT
    is that template -- it is imported, never retyped. A hand-written short template
    shifts z by about -2 while leaving AUROC almost unchanged (a common-mode offset
    does not reorder), so an ablation passing is NOT evidence the threshold is usable.
    """
    import math
    import torch
    import torch.nn as nn
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import collect_jevhead as JH

    proc = AutoProcessor.from_pretrained(
        JH.BASE, trust_remote_code=True,
        min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    mdl = AutoModelForImageTextToText.from_pretrained(
        JH.BASE, dtype=torch.bfloat16, device_map='cuda:0', trust_remote_code=True)
    net = PeftModel.from_pretrained(mdl, os.path.join(JH.JEV, 'adapter_final')).eval()
    hidden = net.config.get_text_config().hidden_size
    head = nn.Linear(hidden, 1, dtype=torch.float32).to('cuda:0')
    ck = torch.load(os.path.join(JH.JEV, 'head_final.pt'),
                    map_location='cuda:0', weights_only=False)
    head.load_state_dict(ck['head'])
    head.eval()
    T = float(ck.get('temperature') or 0) or \
        json.load(open(os.path.join(JH.JEV, 'temperature.json')))['temperature']
    print('jev head loaded, T=%.4f  (prompt template imported from collect_jevhead)'
          % T, flush=True)

    @torch.no_grad()
    def score(img_path_, box, query):
        b = [int(round(v)) for v in box]
        text = JH.PROMPT.format(x0=b[0], y0=b[1], x1=b[2], y1=b[3], q=query)
        msg = {'role': 'user',
               'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}
        pt = proc.apply_chat_template([msg], add_generation_prompt=True,
                                      tokenize=False, enable_thinking=False)
        inp = proc(text=[pt], images=[Image.open(img_path_).convert('RGB')],
                   return_tensors='pt')
        inp = {k: (v.to('cuda:0') if hasattr(v, 'to') else v)
               for k, v in inp.items()}
        o = net(**inp, output_hidden_states=True, use_cache=False)
        z = head(o.hidden_states[-1][:, -1, :].float()).squeeze(-1).item()
        return z, 1.0 / (1.0 + math.exp(-z / T)), T

    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canon', default='canon_roots_paper.json')
    ap.add_argument('--omni', default=os.path.expanduser('~/models/OmniVerifier-7B'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--models', default='')
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--arms', default='omni,jev')
    ap.add_argument('--plan-only', action='store_true')
    a = ap.parse_args()

    canon = json.load(open(os.path.join('/home/u2025141034/SVD/agentic_probe', a.canon)))
    mf = set(x for x in a.models.split(',') if x) or None
    plan, stats = build_plan(canon, mf)

    plan = [x for i, x in enumerate(plan) if i % a.nshard == a.shard]
    if a.smoke:
        plan = plan[:a.smoke]
    n_omni = sum(1 + len(x['probes']) for x in plan)
    print('shard %d/%d rows=%d omni_calls=%d est_omni=%.1fmin est_jev=%.1fmin'
          % (a.shard, a.nshard, len(plan), n_omni, n_omni * 0.13 / 60,
             len(plan) * 0.67 / 60), flush=True)
    if a.plan_only:
        json.dump(stats, open(a.out + '.plan.json', 'w'), indent=2)
        return

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding='utf-8'):
            try:
                d = json.loads(line)
                done.add((d['model'], d['sid'], d['variant']))
            except Exception:
                pass
        print('resume: %d rows already present' % len(done), flush=True)

    arms = set(a.arms.split(','))
    omni = None
    jev = None
    if 'omni' in arms:
        from s5_omni_filter import Omni
        omni = Omni(a.omni, 'cuda:0')
        print('omni loaded', flush=True)
    if 'jev' in arms:
        jev = load_jev()

    fh = open(a.out, 'a', encoding='utf-8')
    t0, n = time.time(), 0
    for row in plan:
        base = dict(model=row['model'], sid=row['sid'], htype=row['htype'],
                    query_role=row['role'], label_exists=row['label_exists'],
                    iou=row['iou'], coordfixed=row['coordfixed'], task=TASK)
        # z0: the original claim on this box
        if 'omni' in arms and (row['model'], row['sid'], '__z0__') not in done:
            try:
                z = omni.score(row['img'], row['box'], row['query'])
                err = None
            except Exception as e:
                z, err = None, str(e)[:200]
            fh.write(json.dumps(dict(base, variant='__z0__', kind='z0',
                                     query=row['query'], z=z, error=err),
                                ensure_ascii=False) + '\n')
            n += 1
        # competitive probes, same scorer, same box
        if 'omni' in arms:
            for pr in row['probes']:
                if (row['model'], row['sid'], pr['edited_surface']) in done:
                    continue
                try:
                    z = omni.score(row['img'], row['box'], pr['query'])
                    err = None
                except Exception as e:
                    z, err = None, str(e)[:200]
                fh.write(json.dumps(dict(base, variant=pr['edited_surface'],
                                         kind='rival', rule_id=pr['rule_id'],
                                         routed_by='query_text',
                                         query=pr['query'], z=z, error=err),
                                    ensure_ascii=False) + '\n')
                n += 1
        # jev scalar head on the same corrected box (separate arm, never subtracted
        # from an Omni score)
        if 'jev' in arms and (row['model'], row['sid'], '__jev__') not in done:
            try:
                zj, pj, T = jev(row['img'], row['box'], row['query'])
                err = None
            except Exception as e:
                zj, pj, T, err = None, None, None, str(e)[:200]
            fh.write(json.dumps(dict(base, variant='__jev__', kind='jev',
                                     query=row['query'], z_head=zj, p_head=pj,
                                     T=T, error=err), ensure_ascii=False) + '\n')
            n += 1
        if n and n % 200 == 0:
            fh.flush()
            el = (time.time() - t0) / 60
            print('  %d scored  %.1fmin  %.2f/s'
                  % (n, el, n / max(1e-9, el * 60)), flush=True)
    fh.flush()
    fh.close()
    print('ALLDONE n=%d %.1fmin' % (n, (time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
