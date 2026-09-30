#!/usr/bin/env python3
"""H2 test: does the 2B+JEV verifier work as a PROBE SELECTOR?

Plan v0.2 section 5.2. The existing probe cache (`probe_upstream.jsonl`) scored
EVERY competitive rival with OmniVerifier, which is the plan's A4 control
("Omni scores, selects and executes the same probes"), NOT A2. A2 spends one 2B
call per candidate to pick ONE rival, then one Omni call on the pick. The two
arms are only comparable on rows where a choice actually exists, i.e. rows with
TWO legal rivals (6768 of 6794 in the cache).

What this script does (GPU, resumable):
  for each (model, sid) row with >=2 cached rivals:
      for each rival query q_c:  z_Q(q_c) = JEV scalar head reading
      also score the ORIGINAL query z_Q(q_0) so u_Q = z_Q(q_c) - z_Q(q_0)
                                   is available per plan section 5.2

It writes ONLY 2B readings. No Omni call is made here: every rival already has
an Omni score in probe_upstream.jsonl, so A2 is assembled OFFLINE by replaying
"Omni復核 the 2B-selected rival only" against that cache. That is exactly the
plan's requirement to report both the full exploration cost and the deployment
cost (section 5.5 / section 10) -- deployment cost is 1 Omni + n_cand 2B calls,
while the offline replay costs nothing extra.

Scoring primitive is copied byte-for-byte from stage3_jev_score.py: the JEV head
reads the hidden state at the LAST PROMPT POSITION, and the prompt template and
temperature were fitted on that exact string. A "cleaned up" prompt shifted z by
-2.08 in an earlier experiment and pushed every probability below 0.5, so the
template below must not be reformatted.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

BASE = os.path.expanduser(
    '~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/'
    '15852e8c16360a2fea060d615a32b45270f8a8fc')
JEV = os.path.expanduser('~/SVD/grpo_verifier/runs/jev_verifier_v2')
PROBES = os.path.expanduser('~/SVD/agentic_probe/probe_upstream.jsonl')
CANON = os.path.expanduser('~/SVD/agentic_probe/canon_roots.json')
IMG_ROOT = os.path.expanduser('~/models/LENS/data/refcoco/train2014')

# DO NOT REFORMAT: byte-identical to the JEV training prompt (see module docstring).
PROMPT = ('Inspect the image region [{x0},{y0},{x1},{y1}]. Determine whether the phrase '
          '"{q}" is fully true for the object in that exact region. Return exactly one '
          'JSON object with exactly these two fields: {{"binding":"MATCH" or "MISMATCH" '
          'or "NA", "decision":"KEEP" or "REJECT"}}. Do not explain. Do not use Markdown.')


def parse_bbox(v):
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        b = [float(x) for x in v]
    except Exception:
        return None
    if any(x != x for x in b):
        return None
    return b


def img_path(sid):
    base = str(sid).split('__')[0]
    i = base.find('COCO_')
    if i < 0:
        return None
    p = os.path.join(IMG_ROOT, base[i:] + '.jpg')
    return p if os.path.exists(p) else None


def load_plan():
    """Rows that have >=2 cached Omni-scored rivals -> a real selection exists.

    The original query text and the upstream box come from the canonical run's
    records.jsonl (the probe cache stores only the EDITED query).
    """
    rivals = {}
    for line in open(PROBES, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('error') or r.get('z') is None:
            continue
        rivals.setdefault((r['model'], r['sid']), []).append(
            dict(variant=r['variant'], rule_id=r['rule_id'],
                 query=r['query'], z_omni=r['z']))

    canon = json.load(open(CANON))
    plan = []
    for m in sorted(canon):
        rp = os.path.join(canon[m], m, 'records.jsonl')
        if not os.path.exists(rp):
            print('[skip] %s: no records' % m, flush=True)
            continue
        for line in open(rp, encoding='utf-8'):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if str(rec.get('task', '')).lower() not in ('t2', 't2_vqa_grounding'):
                continue
            sid = rec.get('sample_id') or rec.get('base_sample_id')
            rv = rivals.get((m, sid))
            if not rv or len(rv) < 2:
                continue        # no choice to make -> selection_performed=False
            bb = parse_bbox(rec.get('pred_bbox_xyxy'))
            if bb is None:
                continue
            q0 = rec.get('query') or rec.get('referring_expression') or ''
            ip = img_path(sid)
            if not ip:
                continue
            plan.append(dict(
                model=m, sid=sid, img=ip, bbox=bb, q0=q0,
                htype=rec.get('hallucination_type') or rec.get('query_role'),
                query_role=rec.get('query_role'),
                label_exists=bool(rec.get('label_exists')),
                iou=float(rec.get('iou') or 0.0),
                rivals=rv))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args()

    plan = load_plan()
    if a.nshards > 1:
        plan = [r for r in plan
                if int(hashlib.md5((r['model'] + '|' + r['sid']).encode()).hexdigest(), 16)
                % a.nshards == a.shard]
    if a.smoke:
        plan = plan[:a.smoke]
    # one reading for the original query + one per rival
    calls = sum(1 + len(r['rivals']) for r in plan)
    print('rows=%d  2B readings=%d  est=%.1fmin'
          % (len(plan), calls, calls * 0.18 / 60), flush=True)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding='utf-8'):
            try:
                d = json.loads(line)
                done.add((d['model'], d['sid']))
            except Exception:
                pass
        print('resume: %d rows already done' % len(done), flush=True)
    plan = [r for r in plan if (r['model'], r['sid']) not in done]
    if not plan:
        print('ALLDONE nothing to do', flush=True)
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    import torch
    import torch.nn as nn
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(
        BASE, trust_remote_code=True,
        min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    mdl = AutoModelForImageTextToText.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map='cuda:0', trust_remote_code=True)
    net = PeftModel.from_pretrained(mdl, os.path.join(JEV, 'adapter_final')).eval()
    hidden = net.config.get_text_config().hidden_size
    head = nn.Linear(hidden, 1, dtype=torch.float32).to('cuda:0')
    ck = torch.load(os.path.join(JEV, 'head_final.pt'),
                    map_location='cuda:0', weights_only=False)
    head.load_state_dict(ck['head'])
    head.eval()
    T = float(ck.get('temperature') or 1.0)
    if T == 1.0:
        T = json.load(open(os.path.join(JEV, 'temperature.json')))['temperature']
    print('jev loaded T=%.4f' % T, flush=True)

    def score(img, bb, q):
        b = [int(v) for v in bb]
        text = PROMPT.format(x0=b[0], y0=b[1], x1=b[2], y1=b[3], q=q)
        msg = {'role': 'user',
               'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}
        pt = proc.apply_chat_template([msg], add_generation_prompt=True,
                                      tokenize=False, enable_thinking=False)
        inp = proc(text=[pt], images=[img], return_tensors='pt')
        inp = {k: (v.to('cuda:0') if hasattr(v, 'to') else v) for k, v in inp.items()}
        o = net(**inp, output_hidden_states=True, use_cache=False)
        return head(o.hidden_states[-1][:, -1, :].float()).squeeze(-1).item()

    fh = open(a.out, 'a', encoding='utf-8')
    t0, n = time.time(), 0
    with torch.no_grad():
        for r in plan:
            try:
                img = Image.open(r['img']).convert('RGB')
                z_q0 = score(img, r['bbox'], r['q0'])
                rv = []
                for c in r['rivals']:
                    rv.append(dict(variant=c['variant'], rule_id=c['rule_id'],
                                   query=c['query'], z_omni=c['z_omni'],
                                   z_jev=score(img, r['bbox'], c['query'])))
                err = None
            except Exception as exc:
                z_q0, rv, err = None, [], '%s: %s' % (type(exc).__name__, exc)
            fh.write(json.dumps(dict(
                model=r['model'], sid=r['sid'], htype=r['htype'],
                query_role=r['query_role'], label_exists=r['label_exists'],
                iou=r['iou'], q0=r['q0'], z_jev_q0=z_q0, T=T,
                rivals=rv, error=err), ensure_ascii=False) + '\n')
            fh.flush()
            n += 1
            if n % 200 == 0:
                el = time.time() - t0
                print('  %d/%d  %.3fs/row  %.1fmin'
                      % (n, len(plan), el / n, el / 60), flush=True)
    fh.close()
    print('ALLDONE rows=%d elapsed=%.1fmin' % (n, (time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
