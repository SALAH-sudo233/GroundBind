#!/usr/bin/env python3
"""Score the cached competitive rivals with the R1 verifier (2B + GRPO ckpt-400).

WHY. The JEV line needs `gap = z(rival) - z(original)` where BOTH terms come from
ONE scorer. The first attempt mixed two:
    z0     from s5jevz_q35_2b_<m>.jsonl  = R1 GRPO ckpt-400, forced-decision logit margin
    rival  from sel2b_all.jsonl          = jev_verifier_v2 adapter + scalar head
Measured on LENS (589 common rows): mean +4.301 (sd 3.764) vs -3.405 (sd 4.369),
Pearson r = 0.3224. Different model AND different score definition, so their
difference is meaningless and the resulting "JEV A2-B1 = +0.01pp" was an artefact
of my own bug, not a property of the verifier.

Fix: re-score every cached rival with the SAME primitive that produced z0.
`s5jevz` covers all 2500 rows/model, which CBR needs (it must decide on all four
htypes' negatives, not only the 6764 probe-covered rows), so R1 ckpt-400 is the
scorer the JEV line is built on.

The reading procedure below is copied from s5_jev_logits.py step for step:
  1. greedy generate (max_new_tokens=64, do_sample=False) -> parse the model's OWN
     binding. Conditioning the decision logit on the model's own binding is what
     the original run did; writing a fixed binding would change the condition.
  2. forced tail '{"binding":"<pb>","decision":"' then read
     logsumexp over KEEP first-token ids minus logsumexp over REJECT ids.
Prompt template, candidate id sets and response-prefix handling all come from
binding_eval_core, not from a re-typed string.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser('~/SVD/grpo_verifier'))

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES = os.path.join(HERE, 'probe_upstream.jsonl')
CANON = os.path.join(HERE, 'canon_roots.json')
IMG_ROOT = os.path.expanduser('~/models/LENS/data/refcoco/train2014')
BASE = os.path.expanduser(
    '~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/'
    '15852e8c16360a2fea060d615a32b45270f8a8fc')
CKPT = os.path.expanduser(
    '~/SVD/grpo_verifier/runs/largebatch_q35_2b_R1/v0-20260916-093618/checkpoint-400')


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


def load_plan(canon_path=CANON, probes_path=PROBES):
    """Every cached rival, joined to its upstream box from the canonical run."""
    rivals = {}
    for line in open(probes_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('error') or r.get('z') is None:
            continue
        rivals.setdefault((r['model'], r['sid']), []).append(
            dict(variant=r['variant'], query=r['query'], z_omni=r['z']))

    canon = json.load(open(canon_path))
    plan = []
    for m in sorted(canon):
        rp = os.path.join(canon[m], m, 'records.jsonl')
        if not os.path.exists(rp):
            print('[skip] %s: no records.jsonl' % m, flush=True)
            continue
        for line in open(rp, encoding='utf-8'):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if str(rec.get('task', '')) != 't2_vqa_grounding':
                continue
            sid = rec.get('sample_id')
            rv = rivals.get((m, sid))
            if not rv:
                continue
            box = parse_bbox(rec.get('pred_bbox_xyxy'))
            if box is None:
                continue
            ip = img_path(sid)
            if ip is None:
                continue
            plan.append(dict(model=m, sid=sid, img=ip, box=box,
                             htype=rec.get('hallucination_type'),
                             rivals=rv))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--probes', default=PROBES,
                    help='rival source; must have been collected against the '
                         'SAME canonical run as --canon')
    ap.add_argument('--canon', default=CANON,
                    help='canonical-run map; pass canon_roots_paper.json to match '
                         'the v0.48 Table 4 provenance (qwen2.5-vl-7b -> 11rep)')
    a = ap.parse_args()

    plan = load_plan(a.canon, a.probes)
    if a.nshards > 1:
        plan = [r for r in plan
                if int(hashlib.md5(('%s|%s' % (r['model'], r['sid'])).encode())
                       .hexdigest(), 16) % a.nshards == a.shard]
    if a.smoke:
        plan = plan[:a.smoke]
    calls = sum(len(r['rivals']) for r in plan)
    print('rows=%d rival-scorings=%d est=%.1fmin'
          % (len(plan), calls, calls * 0.85 / 60), flush=True)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding='utf-8'):
            try:
                d = json.loads(line)
                done.add((d['model'], d['sid'], d['variant']))
            except Exception:
                pass
        print('resume: %d already scored' % len(done), flush=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import binding_eval_core as C

    scheme, resp_prefix = C.model_scheme('q35_2b')
    tmpl = C.PROMPTS[scheme]

    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True,
                                         min_pixels=256 * 28 * 28,
                                         max_pixels=768 * 28 * 28)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map='cuda:0', trust_remote_code=True)
    model = PeftModel.from_pretrained(model, CKPT)
    model.eval()

    tok = proc.tokenizer
    ids_keep, ids_rej = [], []
    for s in ('KEEP', ' KEEP', '"KEEP', 'Keep'):
        ids_keep += tok.encode(s, add_special_tokens=False)[:1]
    for s in ('REJECT', ' REJECT', '"REJECT', 'Reject'):
        ids_rej += tok.encode(s, add_special_tokens=False)[:1]
    ids_keep, ids_rej = sorted(set(ids_keep)), sorted(set(ids_rej))
    print('[cfg] scheme=%s keep_ids=%s rej_ids=%s' % (scheme, ids_keep, ids_rej),
          flush=True)

    def build_inputs(image, q, box, forced_tail):
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        text = tmpl.format(x0=x0, y0=y0, x1=x1, y1=y1, q=q)
        msg = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                            {'type': 'text', 'text': text}]}]
        pt = proc.apply_chat_template([msg], add_generation_prompt=True,
                                      tokenize=False)
        if isinstance(pt, list):
            pt = pt[0]
        if resp_prefix:
            pt = C.apply_response_prefix(pt, resp_prefix)
        pt = pt + forced_tail
        return proc(text=[pt], images=[Image.open(image).convert('RGB')],
                    return_tensors='pt').to('cuda:0')

    @torch.no_grad()
    def verify(image, q, box):
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        text = tmpl.format(x0=x0, y0=y0, x1=x1, y1=y1, q=q)
        msg = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                            {'type': 'text', 'text': text}]}]
        if not resp_prefix:
            inp = proc.apply_chat_template([msg], add_generation_prompt=True,
                                           tokenize=True, return_dict=True,
                                           return_tensors='pt').to('cuda:0')
        else:
            pt = proc.apply_chat_template([msg], add_generation_prompt=True,
                                          tokenize=False)
            if isinstance(pt, list):
                pt = pt[0]
            pt = C.apply_response_prefix(pt, resp_prefix)
            inp = proc(text=[pt], images=[Image.open(image).convert('RGB')],
                       return_tensors='pt').to('cuda:0')
        out = model.generate(**inp, max_new_tokens=64, do_sample=False)
        comp = proc.decode(out[0][inp['input_ids'].shape[1]:],
                           skip_special_tokens=True)
        pb, pd = C.parse(C.first_json_object(
            C.normalize(comp, response_prefix=resp_prefix)))
        b = pb if pb in ('MATCH', 'MISMATCH', 'NA') else 'MISMATCH'
        tail = '{"binding":"%s","decision":"' % b
        if resp_prefix == '{':
            tail = tail[1:]
        inp2 = build_inputs(image, q, box, tail)
        logits = model(**inp2).logits[0, -1].float()
        zk = torch.logsumexp(logits[ids_keep], 0).item()
        zr = torch.logsumexp(logits[ids_rej], 0).item()
        return pb, pd, zk - zr

    fh = open(a.out, 'a', encoding='utf-8')
    t0, n = time.time(), 0
    for row in plan:
        for c in row['rivals']:
            key = (row['model'], row['sid'], c['variant'])
            if key in done:
                continue
            try:
                pb, pd, z = verify(row['img'], c['query'], row['box'])
                err = None
            except Exception as exc:
                pb, pd, z, err = None, None, None, \
                    '%s: %s' % (type(exc).__name__, exc)
            fh.write(json.dumps(dict(
                model=row['model'], sid=row['sid'], htype=row['htype'],
                variant=c['variant'], query=c['query'],
                z_omni=c['z_omni'], pred_binding=pb, decision=pd,
                z_r1=z, error=err), ensure_ascii=False) + '\n')
            fh.flush()
            n += 1
            if n % 200 == 0:
                el = time.time() - t0
                print('  %d/%d %.2fs/it %.1fmin' % (n, calls, el / n, el / 60),
                      flush=True)
    fh.close()
    print('ALLDONE scored=%d elapsed=%.1fmin' % (n, (time.time() - t0) / 60),
          flush=True)


if __name__ == '__main__':
    main()
