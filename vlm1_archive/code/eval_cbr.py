#!/usr/bin/env python3
"""CBR (conditional box-reuse rate) before/after the competitive-probe mitigation.

Contract (locked by the project's CBR spec, see vsight-experiments skill):
  c_i   = 1[ positive PREDICTED box valid AND IoU(pred_pos, gt) >= 0.5 ]
  r_i,t = 1[ negative box valid AND IoU(neg_box, pred_pos_box) >= 0.8 ]
          NOTE: similarity is measured against the model's OWN positive
          PREDICTION, not the GT box -- the semantics is "reusing the box it
          just drew".
  CBR_t = sum_i c_i * r_i,t / sum_i c_i

Hard rules:
  - eligibility c_i and the positive reference box are FROZEN pre-mitigation.
  - rejected / low-overlap / invalid negatives count as ZERO reuse but STAY in
    the denominator.
  - c_i = 0 removes the whole group from the denominator.
  - no eligible positive -> CBR = undefined, never 0.
  - all four htypes share ONE positive eligibility set, so qualified_pos must be
    identical across the four columns for a given model (best self-check).
  - true htype is used ONLY to stratify in the evaluator, never as a feature.
"""
import argparse, hashlib, json, os, sys
from collections import defaultdict
import statistics as st

sys.path.insert(0, '/home/u2025141034/SVD/agentic_probe')
import eval_upstream as E

HT4 = ['object', 'co_occurrence', 'attribute', 'relation']


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


def valid_box(b):
    """Exclude all-zero and degenerate (zero width/height) boxes."""
    if b is None:
        return False
    if all(abs(x) < 1e-9 for x in b):
        return False
    return (b[2] - b[0]) > 0 and (b[3] - b[1]) > 0


def iou(a, b):
    if not valid_box(a) or not valid_box(b):
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_boxes(model, root):
    """Return per-base-sid: positive pred box + gt, and each htype's neg box."""
    rp = os.path.join(root, model, 'records.jsonl')
    pos, neg = {}, defaultdict(dict)
    for line in open(rp, encoding='utf-8'):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if str(r.get('task', '')).lower() not in ('t2', 't2_vqa_grounding'):
            continue
        sid = r.get('sample_id') or ''
        base = str(sid).split('__')[0]
        bb = parse_bbox(r.get('pred_bbox_xyxy'))
        if r.get('query_role') == 'positive':
            pos[base] = dict(pred=bb, gt=parse_bbox(r.get('gt_bbox_xyxy')),
                             iou=float(r.get('iou') or 0.0), sid=sid)
        else:
            ht = r.get('hallucination_type')
            if ht in HT4:
                neg[base][ht] = dict(pred=bb, sid=sid)
    return pos, neg


def _selftest():
    assert abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    # degenerate box must never count as reuse
    assert iou([0, 0, 10, 10], [5, 5, 5, 9]) == 0.0
    assert not valid_box([0, 0, 0, 0])
    assert not valid_box([3, 4, 3, 9])
    assert valid_box([1, 1, 2, 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos-keep', type=float, default=0.95)
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()
    _selftest()
    print('CBR geometry self-test PASSED (degenerate boxes -> 0 reuse)\n')

    canon = json.load(open(E.CANON))
    models = sorted(canon)
    data, _ = E.load_all(models)

    def fit(A, use_gap):
        Ad = [r for r in A if r['drew'] and r['z0'] is not None]
        y = [r['label'] for r in Ad]
        vec = (lambda r: [r['z0'], E.gap(r)]) if use_gap else (lambda r: [r['z0']])
        f, w = E.logreg([vec(r) for r in Ad], y)
        return (lambda r: f(vec(r))), w

    out = {}
    print('=' * 104)
    print('CBR：缓释前 / Omni 单臂 / +竞争探针   （分母=合格正例组，四类共享同一合格集）')
    print('阈值在 fold A 选 (pos_keep %.2f)，fold B 读一次；'
          '未探测行的 gap=0 不改变其 z0 决策' % a.pos_keep)
    print('=' * 104)
    hdr = (f"{'model':17s}{'n_c':>5}  " +
           ''.join(f'{h[:9]:>11}' for h in HT4))
    print(hdr)

    agg = {arm: defaultdict(list) for arm in ('B0', 'B1', 'A2')}
    for m in models:
        root = canon[m]
        pos, neg = load_boxes(m, root)
        rows = data[m]
        A = [r for r in rows if r['fold'] == 0]
        B = [r for r in rows if r['fold'] == 1]
        s1, _ = fit(A, False)
        s2, _ = fit(A, True)
        t1 = E.pick_tau(A, s1, a.pos_keep)
        t2 = E.pick_tau(A, s2, a.pos_keep)
        keep1, keep2 = {}, {}
        for r in rows:
            if not (r['drew'] and r['z0'] is not None):
                keep1[r['sid']] = keep2[r['sid']] = bool(r['drew'])
                continue
            keep1[r['sid']] = s1(r) >= t1
            keep2[r['sid']] = s2(r) >= t2

        foldB = {r['sid'] for r in B}
        # eligibility frozen on the ORIGINAL positive prediction
        elig = [b for b, p in pos.items()
                if p['sid'] in foldB and valid_box(p['pred']) and
                iou(p['pred'], p['gt']) >= 0.5]
        n_c = len(elig)
        res = {}
        for arm, keep in (('B0', None), ('B1', keep1), ('A2', keep2)):
            vals = []
            for ht in HT4:
                num = 0
                for b in elig:
                    nb = neg.get(b, {}).get(ht)
                    if not nb:
                        continue
                    if keep is not None and not keep.get(nb['sid'], True):
                        continue          # rejected -> zero reuse, stays in denom
                    if iou(nb['pred'], pos[b]['pred']) >= 0.8:
                        num += 1
                vals.append(num / n_c if n_c else float('nan'))
                if n_c:
                    agg[arm][ht].append(vals[-1])
            res[arm] = vals
        out[m] = dict(n_c=n_c, **{k: v for k, v in res.items()})
        for arm in ('B0', 'B1', 'A2'):
            tag = m if arm == 'B0' else ''
            print(f'{tag:17s}{n_c if arm=="B0" else "":>5}  ' +
                  ''.join(f'{v*100:>10.1f}%' for v in res[arm]) +
                  f'   {arm}')

    print('\n' + '=' * 104)
    print('池化（13 模型等权）')
    print('=' * 104)
    print(f"{'arm':10s}" + ''.join(f'{h[:9]:>12}' for h in HT4))
    for arm in ('B0', 'B1', 'A2'):
        print(f'{arm:10s}' +
              ''.join(f'{st.mean(agg[arm][h])*100:>11.1f}%' for h in HT4))

    print('\n相对缓释前的绝对下降 (pp)')
    print(f"{'arm':10s}" + ''.join(f'{h[:9]:>12}' for h in HT4))
    for arm in ('B1', 'A2'):
        print(f'{arm:10s}' +
              ''.join(f'{(st.mean(agg[arm][h])-st.mean(agg["B0"][h]))*100:>11.2f}'
                      for h in HT4))

    print('\n' + '=' * 104)
    print('模型级配对检验 (n=13 等权) — A2 vs B1')
    print('=' * 104)
    for ht in HT4:
        d = [out[m]['A2'][HT4.index(ht)] - out[m]['B1'][HT4.index(ht)]
             for m in models if out[m]['n_c']]
        mm, lo, hi = E.model_level_paired(d)
        tag = '显著' if (lo > 0 or hi < 0) else '不显著'
        print(f'  {ht:16s} {mm*100:+7.2f}pp CI[{lo*100:+6.2f},{hi*100:+6.2f}] '
              f'{tag:6s} 下降 {sum(1 for x in d if x<0)}/{len(d)}')

    if a.json_out:
        json.dump(out, open(a.json_out, 'w'), indent=2, ensure_ascii=False)
        print(f'\nwrote {a.json_out}')


if __name__ == '__main__':
    main()
