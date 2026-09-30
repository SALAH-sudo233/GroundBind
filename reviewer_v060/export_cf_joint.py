#!/usr/bin/env python3
"""Coordinate-corrected JOINT grounding (t4) for UniVG-R1 and visual-rft.

WHY THIS EXISTS. write_rescaled_run.py corrected only t2: its filter is
T2 = ('t2', 't2_vqa_grounding'), so every other task was copied through verbatim.
Verified: t4 boxes in ~/benchmark/coordfix_500 are byte-identical to the uncorrected
run (2500/2500 identical for both models), while t2 boxes differ on 2496/2500 and
974/2500. So the published joint-grounding numbers for these two models carry the
same 0-1000-read-as-pixels defect the reviewer flagged for t2, and the repo's t4
script simply read the stored `iou`, inheriting it.

Evidence the defect is real on t4, not assumed: raw [54,700,236,999] was stored as
[54,428,236,428] -- a zero-height box after clipping to a 428px-tall image.
Parseable 0-1000 responses: UniVG-R1 2246/2500, visual-rft 1798/2500.

THE CORRECTION IS THE SAME ONE, REUSED, NOT REINVENTED: parse up to four numbers from
the answer span, divide by 1000, multiply by true image width/height, clamp to the
image, recompute IoU against untouched GT. Rows with no parseable box stay
pred_found=False with zero IoU -- a genuine no-output, not a coordinate problem.

Emits, for each model: n_correct (positives with IoU >= 0.5), full-positive mIoU
(denominator 500, refusals score zero), and the four CBR columns under the paper's
contract (theta=0.5, rho=0.8 against the positive PREDICTED box, denominator the
shared eligible count, refusals counted zero but kept in the denominator).

ACCEPTANCE GATE. The same code path, restricted to t2, must reproduce the published
t2 CBR for both models (38.6/37.0/50.8/55.1 and 3.8/14.7/18.5/37.9). If that fails
the t4 output is not trustworthy either and the script says so.
"""
import json
import os
import re
import sys

PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
import eval_upstream as E
import eval_cbr_paper_aligned as A
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = '/home/u2025141034/models/LENS/data/refcoco/train2014'
NUM = re.compile(r'-?\d+\.?\d*')
SCALE = 1000.0
MODELS = ['UniVG-R1', 'visual-rft']
HT4 = A.HT4
TAU, RHO = 0.5, 0.8


def iou(a, b):
    if not a or not b:
        return 0.0
    if (a[2] - a[0]) <= 0 or (a[3] - a[1]) <= 0 or \
       (b[2] - b[0]) <= 0 or (b[3] - b[1]) <= 0:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def raw_box(rec):
    raw = rec.get('raw_output_text') or ''
    ans = raw.split('<answer>')[-1] if '<answer>' in raw else raw
    nums = [float(x) for x in NUM.findall(ans)]
    return nums[:4] if len(nums) >= 4 else None


def rescale(rec, cache):
    """Return the corrected pixel box, or None when nothing parses."""
    v = raw_box(rec)
    if v is None:
        return None
    sid = rec['sample_id']
    if sid not in cache:
        fp = os.path.join(IMG, E.img_of(sid))
        cache[sid] = Image.open(fp).size if os.path.exists(fp) else None
    wh = cache[sid]
    if not wh:
        return None
    w, h = wh
    nb = [v[0] / SCALE * w, v[1] / SCALE * h, v[2] / SCALE * w, v[3] / SCALE * h]
    return [max(0.0, min(nb[0], w)), max(0.0, min(nb[1], h)),
            max(0.0, min(nb[2], w)), max(0.0, min(nb[3], h))]


def load(root, model, task):
    p = os.path.join(root, model, 'records.jsonl')
    out = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get('task') == task:
                out.append(d)
    return out


def evaluate(rows, cache):
    """Apply the correction, then compute n_correct / mIoU / four CBR columns."""
    pos, neg = {}, {}
    for r in rows:
        box = rescale(r, cache)
        rec = dict(box=box, gt=r.get('gt_bbox_xyxy'),
                   iou=iou(box, r.get('gt_bbox_xyxy')) if box else 0.0)
        bid = r.get('base_sample_id')
        if r.get('query_role') == 'positive':
            pos[bid] = rec
        else:
            neg.setdefault(bid, {})[r.get('hallucination_type')] = rec

    # eligibility frozen on the corrected positives, exactly as the t2 contract
    elig = [b for b, q in pos.items()
            if q['box'] and (q['box'][2] - q['box'][0]) > 0
            and (q['box'][3] - q['box'][1]) > 0 and q['iou'] >= TAU]
    n_c = len(elig)
    cbr = {}
    for h in HT4:
        hit = 0
        for b in elig:
            nb = neg.get(b, {}).get(h)
            if nb and nb['box'] and iou(nb['box'], pos[b]['box']) >= RHO:
                hit += 1
        cbr[h] = hit / n_c if n_c else None
    n_pos = len(pos)
    miou = sum(q['iou'] for q in pos.values()) / n_pos if n_pos else 0.0
    parsed = sum(1 for q in pos.values() if q['box'])
    return dict(n_positives=n_pos, n_correct=n_c, pos_success=n_c / n_pos if n_pos else None,
                pos_miou=miou, positives_parsed=parsed, cbr=cbr)


def main():
    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    out, gate_ok = {}, True
    ref = {'UniVG-R1': [38.6, 37.0, 50.8, 55.1],
           'visual-rft': [3.8, 14.7, 18.5, 37.9]}

    for m in MODELS:
        cache = {}
        t2 = evaluate(load(canon[m], m, 't2_vqa_grounding'), cache)
        t4 = evaluate(load(canon[m], m, 't4_caption_grounding'), cache)
        out[m] = dict(t2_gatecheck=t2, t4_joint=t4)
        got = [t2['cbr'][h] * 100 for h in HT4]
        ok = all(abs(g - r) <= 0.06 for g, r in zip(got, ref[m]))
        gate_ok &= ok
        print('[%s] t2 gate %s  got %s expect %s'
              % (m, 'OK' if ok else 'MISMATCH',
                 ' '.join('%.1f' % g for g in got),
                 ' '.join('%.1f' % r for r in ref[m])), flush=True)

    print('\n' + '=' * 96)
    print('GATE: t2 reproduced via this code path -> %s'
          % ('PASS' if gate_ok else 'FAIL, t4 numbers not usable'))
    print('=' * 96)
    if not gate_ok:
        json.dump(out, open(os.path.join(HERE, 'cf_joint_grounding.json'), 'w'),
                  indent=2, ensure_ascii=False)
        sys.exit(1)

    print('\nCOORDINATE-CORRECTED JOINT GROUNDING (t4)')
    print('%-12s %9s %9s %9s | %8s %8s %8s %8s'
          % ('model', 'n_correct', 'posSucc', 'mIoU',
             'CBR obj', 'CBR cooc', 'CBR attr', 'CBR rel'))
    for m in MODELS:
        t = out[m]['t4_joint']
        print('%-12s %9d %8.1f%% %9.4f | %7.2f%% %7.2f%% %7.2f%% %7.2f%%'
              % (m, t['n_correct'], t['pos_success'] * 100, t['pos_miou'],
                 *[t['cbr'][h] * 100 for h in HT4]))

    print('\nfor comparison, the same models under direct grounding (t2):')
    for m in MODELS:
        t = out[m]['t2_gatecheck']
        print('%-12s %9d %8.1f%% %9.4f | %7.2f%% %7.2f%% %7.2f%% %7.2f%%'
              % (m, t['n_correct'], t['pos_success'] * 100, t['pos_miou'],
                 *[t['cbr'][h] * 100 for h in HT4]))

    json.dump(dict(models=out, gate_passed=True, scale=SCALE,
                   note='t4 boxes corrected here; ~/benchmark/coordfix_500 '
                        'left t4 uncorrected because write_rescaled_run.py '
                        'filters on t2 only'),
              open(os.path.join(HERE, 'cf_joint_grounding.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('\nwrote cf_joint_grounding.json')


if __name__ == '__main__':
    main()
