#!/usr/bin/env python3
"""Reviewer W2: is the low positive-localization rate a capability failure or a
coordinate-convention bug?

For every model, re-parse the ORIGINAL model response, then score the positive
queries three ways against the same GT:

  stored  the box the evaluation pipeline actually used (what the paper reports)
  px      raw <answer> numbers read as pixels
  /1000   raw numbers read as a 0-1000 normalized scale, mapped to the real image

A model whose raw coordinates top out at exactly 1000 and whose boxes mostly fall
outside the image is emitting normalized coordinates. If its `stored` score matches
`px` rather than `/1000`, the pipeline failed to de-normalize it, and the reported
failure is ours, not the model's.

Also counts degenerate (zero-area) boxes, which is what border clipping produces
after an out-of-range coordinate is clamped.

Writes rescale_fix.json: de-normalized boxes + recomputed IoU for the affected
models, so the downstream tables can be recomputed without re-running inference.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
import eval_upstream as E

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = '/home/u2025141034/models/LENS/data/refcoco/train2014'
NUM = re.compile(r'-?\d+\.?\d*')
T2 = ('t2', 't2_vqa_grounding')


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def raw_box(rec):
    """Last <answer> block's first four numbers."""
    raw = rec.get('raw_output_text') or ''
    ans = raw.split('<answer>')[-1] if '<answer>' in raw else raw
    nums = [float(x) for x in NUM.findall(ans)]
    return nums[:4] if len(nums) >= 4 else None


def main():
    canon = json.load(open(os.path.join(HERE, 'canon_roots_paper.json')))
    print('%-16s %5s %7s %6s %7s %8s %8s %8s' %
          ('model', 'n', 'maxRaw', 'oob%', 'degen', 'stored', 'px', '/1000'))
    flagged = []
    for m, root in sorted(canon.items()):
        p = os.path.join(root, m, 'records.jsonl')
        if not os.path.exists(p):
            print('[skip] %s' % m)
            continue
        rows = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
        pos = [r for r in rows
               if str(r.get('task', '')).lower() in T2
               and r.get('query_role') == 'positive']
        n = oob = degen = 0
        maxv = 0.0
        stored = px = norm = 0
        for r in pos:
            bb = r.get('pred_bbox_xyxy')
            gt = r.get('gt_bbox_xyxy')
            if bb and (bb[2] - bb[0] <= 0 or bb[3] - bb[1] <= 0):
                degen += 1
            if bb and gt and iou(bb, gt) >= 0.5:
                stored += 1
            v = raw_box(r)
            fp = os.path.join(IMG, E.img_of(r['sample_id']))
            if v is None or not gt or not os.path.exists(fp):
                continue
            w, h = Image.open(fp).size
            n += 1
            maxv = max(maxv, max(v))
            if v[2] > w + 1 or v[3] > h + 1:
                oob += 1
            if iou(v, gt) >= 0.5:
                px += 1
            sc = [v[0] / 1000.0 * w, v[1] / 1000.0 * h,
                  v[2] / 1000.0 * w, v[3] / 1000.0 * h]
            if iou(sc, gt) >= 0.5:
                norm += 1
        print('%-16s %5d %7.0f %5.0f%% %7d %7.1f%% %7.1f%% %7.1f%%'
              % (m, n, maxv, 100.0 * oob / max(1, n), degen,
                 100.0 * stored / 500, 100.0 * px / 500, 100.0 * norm / 500))
        # normalized emitter whose stored score tracks px, not /1000 -> unscaled
        if maxv > 990 and norm - stored > 50:
            flagged.append(m)

    print('\nmodels emitting 0-1000 coords that the pipeline did NOT de-normalize:')
    print('   ', ', '.join(flagged) if flagged else '(none)')

    fix = {}
    for m in flagged:
        p = os.path.join(canon[m], m, 'records.jsonl')
        rec = {}
        for line in open(p, encoding='utf-8'):
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r.get('task', '')).lower() not in T2:
                continue
            v = raw_box(r)
            fp = os.path.join(IMG, E.img_of(r['sample_id']))
            if v is None or not os.path.exists(fp):
                continue
            w, h = Image.open(fp).size
            sc = [v[0] / 1000.0 * w, v[1] / 1000.0 * h,
                  v[2] / 1000.0 * w, v[3] / 1000.0 * h]
            gt = r.get('gt_bbox_xyxy')
            rec[r['sample_id']] = dict(
                rescaled=sc, iou=(iou(sc, gt) if gt else None),
                role=r.get('query_role'), htype=r.get('hallucination_type'),
                drew=True)
        fix[m] = rec
        pos = [v for v in rec.values() if v['role'] == 'positive']
        print('%-12s rescaled rows=%d positives=%d acc50=%.1f%%'
              % (m, len(rec), len(pos),
                 100.0 * sum(1 for v in pos if v['iou'] and v['iou'] >= 0.5) / 500))
    json.dump(fix, open(os.path.join(HERE, 'rescale_fix.json'), 'w'))
    print('wrote rescale_fix.json')


if __name__ == '__main__':
    main()
