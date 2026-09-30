#!/usr/bin/env python3
"""Materialise a coordinate-corrected run for the models that emit 0-1000 boxes.

Reviewer W2: UniVG-R1 and visual-rft output normalized 0-1000 coordinates that the
pipeline read as pixels, then clipped to the image border, collapsing many boxes to
zero area. Everything downstream (CBR eligibility, FGR, verifier red-box rendering)
reads `pred_bbox_xyxy` out of records.jsonl, so the correction belongs there.

This writes a NEW run directory with the same layout:

    <outroot>/<model>/records.jsonl

with `pred_bbox_xyxy` de-normalized to pixels and `iou` recomputed against the
untouched GT. Every other field is copied verbatim. The original run is never
modified, and the corrected run carries `coord_fix` provenance on each row.

Rows whose raw response has no parseable box keep pred_found=False and a null box
(a genuine no-output, not a coordinate problem).
"""
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
import eval_upstream as E

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = '/home/u2025141034/models/LENS/data/refcoco/train2014'
NUM = re.compile(r'-?\d+\.?\d*')
T2 = ('t2', 't2_vqa_grounding')
SCALE = 1000.0


def iou(a, b):
    if not a or not b:
        return 0.0
    if (a[2] - a[0]) <= 0 or (a[3] - a[1]) <= 0:
        return 0.0
    if (b[2] - b[0]) <= 0 or (b[3] - b[1]) <= 0:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def raw_box(rec):
    raw = rec.get('raw_output_text') or ''
    ans = raw.split('<answer>')[-1] if '<answer>' in raw else raw
    nums = [float(x) for x in NUM.findall(ans)]
    return nums[:4] if len(nums) >= 4 else None


def main():
    canon = json.load(open(os.path.join(HERE, 'canon_roots_paper.json')))
    models = sys.argv[1].split(',') if len(sys.argv) > 1 else \
        ['UniVG-R1', 'visual-rft']
    outroot = os.path.expanduser('~/benchmark/coordfix_500')
    os.makedirs(outroot, exist_ok=True)

    summary = {}
    for m in models:
        src = os.path.join(canon[m], m, 'records.jsonl')
        dst_dir = os.path.join(outroot, m)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, 'records.jsonl')
        n = fixed = nobox = 0
        pos_ok = pos_n = 0
        deg_before = deg_after = 0
        with open(dst, 'w', encoding='utf-8') as out:
            for line in open(src, encoding='utf-8'):
                if not line.strip():
                    continue
                r = json.loads(line)
                if str(r.get('task', '')).lower() in T2:
                    n += 1
                    old = r.get('pred_bbox_xyxy')
                    if old and ((old[2] - old[0]) <= 0 or (old[3] - old[1]) <= 0):
                        deg_before += 1
                    v = raw_box(r)
                    fp = os.path.join(IMG, E.img_of(r['sample_id']))
                    if v is not None and os.path.exists(fp):
                        w, h = Image.open(fp).size
                        nb = [v[0] / SCALE * w, v[1] / SCALE * h,
                              v[2] / SCALE * w, v[3] / SCALE * h]
                        # clamp to image, as any consumer would
                        nb = [max(0.0, min(nb[0], w)), max(0.0, min(nb[1], h)),
                              max(0.0, min(nb[2], w)), max(0.0, min(nb[3], h))]
                        r['pred_bbox_xyxy'] = nb
                        r['pred_found'] = True
                        r['coord_fix'] = 'div%d_to_pixels' % int(SCALE)
                        r['iou'] = iou(nb, r.get('gt_bbox_xyxy'))
                        fixed += 1
                        if (nb[2] - nb[0]) <= 0 or (nb[3] - nb[1]) <= 0:
                            deg_after += 1
                    else:
                        r['pred_bbox_xyxy'] = None
                        r['pred_found'] = False
                        r['coord_fix'] = 'no_parseable_box'
                        r['iou'] = 0.0
                        nobox += 1
                    if r.get('query_role') == 'positive':
                        pos_n += 1
                        if (r.get('iou') or 0) >= 0.5:
                            pos_ok += 1
                out.write(json.dumps(r, ensure_ascii=False) + '\n')
        summary[m] = dict(t2_rows=n, rescaled=fixed, no_box=nobox,
                          degenerate_before=deg_before,
                          degenerate_after=deg_after,
                          positives=pos_n, pos_acc50=pos_ok / 500.0)
        print('%-12s t2=%d rescaled=%d nobox=%d degen %d->%d  acc50=%.1f%%'
              % (m, n, fixed, nobox, deg_before, deg_after,
                 100.0 * pos_ok / 500))

    # a canon map pointing the two models at the corrected run
    cf = dict(canon)
    for m in models:
        cf[m] = outroot
    json.dump(cf, open(os.path.join(HERE, 'canon_roots_coordfix.json'), 'w'),
              indent=2)
    json.dump(summary, open(os.path.join(HERE, 'coordfix_summary.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('\nwrote canon_roots_coordfix.json and coordfix_summary.json')
    print('corrected run root: %s' % outroot)


if __name__ == '__main__':
    main()
