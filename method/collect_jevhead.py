#!/usr/bin/env python3
"""Third line: jev_verifier_v2 SCALAR HEAD over all 13 upstreams' T2 boxes.

WHY A THIRD LINE. CBR needs a decision on every negative (2000/model), so the two
existing lines use verifiers that cover all 2500 rows:
    OMNI line  OmniVerifier-7B, zero-training, red-rectangle prompt
    R1   line  Qwen3.5-2B + GRPO ckpt-400, forced-decision logit margin
The jev_verifier_v2 head (full-layer LoRA + (1,2048) fp32 scalar head + NLL +
0.1*Brier + post-hoc temperature 1.7766) is the only arm in this project with a
CALIBRATED, genuinely continuous score -- 39.4% of its probabilities land in
(0.05, 0.95), where R1's confidence is saturated at {0,1}. That is what makes a
threshold sweep and a matched-pos_keep comparison meaningful. It had never been
run outside the probe-covered rows, so this collects it on the full grid.

Two score fields are written per row:
    z_head  raw scalar-head reading at the LAST PROMPT POSITION
    p_head  1/(1+exp(-z/T)) with T from temperature.json (calibrated on 512
            calibration rows disjoint from training)
Both are kept: p_head is the calibrated quantity to threshold, z_head is stored so
any later re-calibration does not require another GPU pass.

Prompt is byte-identical to the training/eval template (stage3_jev_score.py). A
"cleaned up" shorter template shifted z by -2.08 and pushed every probability below
0.5 while leaving AUROC almost unchanged, so passing an ablation does NOT license
reformatting this string.

Covers positives AND all four negative types, per canonical run (paper-aligned
map), resumable by (model, sid).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.expanduser(
    '~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/'
    '15852e8c16360a2fea060d615a32b45270f8a8fc')
JEV = os.path.expanduser('~/SVD/grpo_verifier/runs/jev_verifier_v2')
IMG_ROOT = os.path.expanduser('~/models/LENS/data/refcoco/train2014')

# DO NOT REFORMAT (see module docstring).
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


def load_rival_plan(canon_path, probes_path, models_filter=None):
    """Score the cached competitive rivals with THIS scorer.

    The gap feature must be a within-scorer difference, so the jevhead line cannot
    reuse the Omni or R1 rival scores. Rival TEXTS come from the existing probe
    cache; the box and image come from the canonical run.
    """
    rivals = {}
    for line in open(probes_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('error'):
            continue
        rivals.setdefault((r['model'], r['sid']), []).append(
            dict(variant=r['variant'], query=r['query']))

    canon = json.load(open(canon_path))
    plan = []
    for m in sorted(canon):
        if models_filter and m not in models_filter:
            continue
        rp = os.path.join(canon[m], m, 'records.jsonl')
        if not os.path.exists(rp):
            continue
        for line in open(rp, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if str(r.get('task', '')) != 't2_vqa_grounding':
                continue
            sid = r.get('sample_id')
            rv = rivals.get((m, sid))
            if not rv:
                continue
            box = parse_bbox(r.get('pred_bbox_xyxy'))
            ip = img_path(sid)
            if box is None or ip is None:
                continue
            for c in rv:
                plan.append(dict(model=m, sid=sid, htype=r.get('hallucination_type'),
                                 query_role=r.get('query_role'),
                                 label_exists=str(r.get('label_exists')).lower() == 'true',
                                 iou=float(r.get('iou') or 0.0), drew=True,
                                 box=box, img=ip, query=c['query'],
                                 variant=c['variant']))
    return plan


def load_plan(canon_path, models_filter=None):
    canon = json.load(open(canon_path))
    plan = []
    for m in sorted(canon):
        if models_filter and m not in models_filter:
            continue
        rp = os.path.join(canon[m], m, 'records.jsonl')
        if not os.path.exists(rp):
            print('[skip] %s: no records.jsonl' % m, flush=True)
            continue
        n = 0
        for line in open(rp, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if str(r.get('task', '')) != 't2_vqa_grounding':
                continue
            sid = r.get('sample_id')
            box = parse_bbox(r.get('pred_bbox_xyxy'))
            drew = str(r.get('pred_found')).lower() == 'true' and box is not None
            ip = img_path(sid) if drew else None
            plan.append(dict(
                model=m, sid=sid, htype=r.get('hallucination_type'),
                query_role=r.get('query_role'),
                label_exists=str(r.get('label_exists')).lower() == 'true',
                iou=float(r.get('iou') or 0.0), drew=drew, box=box,
                img=ip, query=r.get('query') or ''))
            n += 1
        print('%-18s rows=%d' % (m, n), flush=True)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--canon', default=os.path.join(HERE, 'canon_roots_paper.json'))
    ap.add_argument('--models', default='')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--rivals', default='',
                    help='probe cache to re-score as rivals (rival mode); when set, '
                         'output rows carry a `variant` field')
    a = ap.parse_args()

    mf = set(x.strip() for x in a.models.split(',') if x.strip()) or None
    plan = (load_rival_plan(a.canon, a.rivals, mf) if a.rivals
            else load_plan(a.canon, mf))
    if a.nshards > 1:
        plan = [r for r in plan
                if int(hashlib.md5(('%s|%s' % (r['model'], r['sid'])).encode())
                       .hexdigest(), 16) % a.nshards == a.shard]
    if a.smoke:
        plan = plan[:a.smoke]
    n_score = sum(1 for r in plan if r['drew'] and r['img'])
    print('\nshard rows=%d  GPU scorings=%d  est=%.1fmin'
          % (len(plan), n_score, n_score * 0.45 / 60), flush=True)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding='utf-8'):
            try:
                d = json.loads(line)
                done.add((d['model'], d['sid'], d.get('variant')))
            except Exception:
                pass
        print('resume: %d rows done' % len(done), flush=True)
    plan = [r for r in plan
            if (r['model'], r['sid'], r.get('variant')) not in done]
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
    T = float(ck.get('temperature') or 0) or \
        json.load(open(os.path.join(JEV, 'temperature.json')))['temperature']
    print('jev head loaded, T=%.4f' % T, flush=True)

    fh = open(a.out, 'a', encoding='utf-8')
    t0, n = time.time(), 0
    with torch.no_grad():
        for r in plan:
            rec = dict(model=r['model'], sid=r['sid'], htype=r['htype'],
                       query_role=r['query_role'], label_exists=r['label_exists'],
                       iou=r['iou'], drew=r['drew'], T=T,
                       variant=r.get('variant'),
                       z_head=None, p_head=None, error=None)
            if r['drew'] and r['img']:
                try:
                    b = [int(round(v)) for v in r['box']]
                    text = PROMPT.format(x0=b[0], y0=b[1], x1=b[2], y1=b[3],
                                         q=r['query'])
                    msg = {'role': 'user',
                           'content': [{'type': 'image'},
                                       {'type': 'text', 'text': text}]}
                    pt = proc.apply_chat_template(
                        [msg], add_generation_prompt=True, tokenize=False,
                        enable_thinking=False)
                    inp = proc(text=[pt],
                               images=[Image.open(r['img']).convert('RGB')],
                               return_tensors='pt')
                    inp = {k: (v.to('cuda:0') if hasattr(v, 'to') else v)
                           for k, v in inp.items()}
                    o = net(**inp, output_hidden_states=True, use_cache=False)
                    z = head(o.hidden_states[-1][:, -1, :].float()).squeeze(-1).item()
                    rec['z_head'] = z
                    rec['p_head'] = 1.0 / (1.0 + math.exp(-z / T))
                except Exception as exc:
                    rec['error'] = '%s: %s' % (type(exc).__name__, exc)
            elif r['drew'] and not r['img']:
                rec['error'] = 'no_image'
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fh.flush()
            n += 1
            if n % 300 == 0:
                el = time.time() - t0
                print('  %d/%d %.3fs/row %.1fmin'
                      % (n, len(plan), el / n, el / 60), flush=True)
    fh.close()
    print('ALLDONE rows=%d elapsed=%.1fmin' % (n, (time.time() - t0) / 60),
          flush=True)


if __name__ == '__main__':
    main()
