#!/usr/bin/env python3
"""Coordinate-corrected models under the PAPER'S framework (JEV + OMNI combined).

The earlier coordfix run reported each verifier separately. The paper's method is the
two-verifier combination (Figure 3: JEV-2B + OmniVerifier-7B), so the mitigation
side must be evaluated as that combination, not as single lines.

Feature set per row, exactly as in the 13-model combo evaluation:

    support-only : [z_omni, z_jev]                      (no probes)
    full         : [z_omni, z_jev, gap_omni, gap_jev]   (text-routed rivals)

Each `gap` is a within-scorer difference (that verifier's rival max minus its own
z0); the fusion consumes four features and lets the fitted head weight them, so no
cross-scorer subtraction occurs.

Protocol is unchanged from the reviewer-facing evaluation: paper denominators
(500 positives, 500 per negative type), image-level two-fold cross-fitting
(md5(image) % 2, fold A decided by fold-B parameters and vice versa), the
label-independent query-text router, and thresholds chosen on the fitting fold to
hit a CorrectRetain target.

All inputs are the FRESH scores collected on the corrected boxes:
    s5omni_omnicf_<m>.jsonl  jevheadcf_<m>.jsonl  probecf_omni.jsonl  probecf_jev.jsonl
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_upstream as E
import eval_cbr_paper_aligned as A

HT4 = A.HT4
S5 = os.path.expanduser('~/SVD/grpo_verifier/s5grpo')
MODELS = ('UniVG-R1', 'visual-rft')


def load_probe(path, key):
    out = {}
    for line in open(os.path.join(HERE, path), encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get('error') or d.get(key) is None:
            continue
        out.setdefault((d['model'], d['sid']), {})[d['variant']] = d[key]
    return out


def z0_map(model, line):
    if line == 'omni':
        p = os.path.join(S5, 's5omni_omnicf_%s.jsonl' % model)
        zk = 'z_omni'
    else:
        p = os.path.join(HERE, 'jevheadcf_%s.jsonl' % model)
        zk = 'z_head'
    out = {}
    for l in open(p, encoding='utf-8'):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if r.get('model') and r['model'] != model:
            continue
        out[r['sid']] = r
    return out, zk


def merge(model, po, pj):
    """One row per sid carrying BOTH verifiers' z0 and gap."""
    mo, ko = z0_map(model, 'omni')
    mj, kj = z0_map(model, 'jev')
    rows = []
    for sid, r in mo.items():
        q = mj.get(sid)
        if q is None:
            continue
        ht = r.get('htype')
        rvo = [v for v in po.get((model, sid), {}).values() if v is not None]
        rvj = [v for v in pj.get((model, sid), {}).values() if v is not None]
        zo, zj = r.get(ko), q.get(kj)
        rows.append(dict(
            model=model, sid=sid, htype=ht, is_pos=(ht == 'positive'),
            label=1 if ht == 'positive' else 0,
            iou=float(r.get('iou') or 0.0),
            drew=bool(r.get('drew')) and bool(q.get('drew')),
            z_o=zo, z_j=zj,
            zmax_o=(max(rvo) if rvo else None),
            zmax_j=(max(rvj) if rvj else None),
            fold=E.fold_of(sid)))
    for r in rows:
        r['g_o'] = (r['zmax_o'] - r['z_o']) \
            if (r['zmax_o'] is not None and r['z_o'] is not None) else 0.0
        r['g_j'] = (r['zmax_j'] - r['z_j']) \
            if (r['zmax_j'] is not None and r['z_j'] is not None) else 0.0
    return rows


def covered(r):
    return r['zmax_o'] is not None and r['zmax_j'] is not None


def feats(mode, r):
    if mode == 'support':
        return [r['z_o'], r['z_j']]
    return [r['z_o'], r['z_j'], r['g_o'], r['g_j']]


def usable(r, mode):
    return (r['drew'] and r['z_o'] is not None and r['z_j'] is not None
            and (mode == 'support' or covered(r)))


def fit(rows, mode):
    R = [r for r in rows if usable(r, mode)]
    if len(R) < 30 or len(set(r['label'] for r in R)) < 2:
        return None
    f, _ = E.logreg([feats(mode, r) for r in R], [r['label'] for r in R])
    return lambda r: f(feats(mode, r))


def thr(rows, score, elig, target, mode):
    v = sorted(score(r) for r in rows
               if r['is_pos'] and r['sid'] in elig and usable(r, mode))
    if not v:
        return float('-inf')
    k = int(round((1.0 - target) * len(v)))
    return v[max(0, min(k, len(v) - 1))]


def keepmap(rows, elig, target, mode):
    keep = {}
    for tf in (0, 1):
        f_ = [r for r in rows if r['fold'] != tf]
        t_ = [r for r in rows if r['fold'] == tf]
        s_sup = fit(f_, 'support')
        if s_sup is None:
            for r in t_:
                keep[r['sid']] = bool(r['drew'])
            continue
        t_sup = thr(f_, s_sup, elig, target, 'support')
        s_full = fit(f_, 'full') if mode == 'full' else None
        t_full = thr([r for r in f_ if covered(r)], s_full, elig, target, 'full') \
            if s_full else None
        for r in t_:
            if not r['drew']:
                keep[r['sid']] = False
            elif not usable(r, 'support'):
                keep[r['sid']] = True
            elif s_full is not None and usable(r, 'full'):
                keep[r['sid']] = s_full(r) >= t_full
            else:
                keep[r['sid']] = s_sup(r) >= t_sup
    return keep


def measure(rows, keep, elig, pos, neg):
    p = [r for r in rows if r['is_pos']]
    n = [r for r in rows if not r['is_pos']]
    by = {ht: sum(1 for r in n if r['htype'] == ht and keep.get(r['sid']))
          / 500.0 for ht in HT4}
    ce = [r for r in p if r['sid'] in elig]
    km = {s: (v, v) for s, v in keep.items()}
    return dict(
        fgr_all=sum(1 for r in n if keep.get(r['sid'])) / 2000.0,
        by_ht=by,
        boh=(by['object'] + by['co_occurrence']) / 2.0,
        roh=(by['attribute'] + by['relation']) / 2.0,
        pos_keep=sum(1 for r in p if keep.get(r['sid'])) / 500.0,
        pos_miou=sum(r['iou'] for r in p if keep.get(r['sid'])) / 500.0,
        correct_retain=(sum(1 for r in ce if keep.get(r['sid'])) / len(ce))
        if ce else float('nan'),
        cbr=A.cbr(list(elig), pos, neg, km, 0),
        n_covered=sum(1 for r in rows if covered(r)),
        n_rows=len(rows))


def main():
    targets = [float(x) for x in
               (sys.argv[1] if len(sys.argv) > 1 else '0.95,0.90,0.85').split(',')]
    cf = json.load(open(os.path.join(HERE, 'canon_roots_coordfix.json')))
    po = load_probe('probecf_omni.jsonl', 'z')
    pj = load_probe('probecf_jev.jsonl', 'z_head')

    res = {}
    for m in MODELS:
        pos, neg = A.load_boxes(m, cf[m])
        elig = {q['sid'] for b, q in pos.items()
                if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
        rows = merge(m, po, pj)
        res[m] = dict(n_c=len(elig), arms={})
        for tgt in targets:
            for mode in ('support', 'full'):
                k = keepmap(rows, elig, tgt, mode)
                res[m]['arms']['%.2f|%s' % (tgt, mode)] = \
                    measure(rows, k, elig, pos, neg)

    print('=' * 104)
    print('两个坐标修正模型 × 论文最优框架（JEV-2B + OmniVerifier-7B 双读数融合）')
    print('论文分母 / 图像级交叉拟合 / query 文本路由')
    print('=' * 104)
    print('%-12s%-7s%-14s%11s%9s%9s%9s%9s%9s' %
          ('model', 'target', 'arm', 'CorrRetain', 'posmIoU', 'FGR', 'relFGR',
           'relCBR', 'attrCBR'))
    for m in MODELS:
        first = True
        for tgt in targets:
            for mode in ('support', 'full'):
                d = res[m]['arms']['%.2f|%s' % (tgt, mode)]
                print('%-12s%-7s%-14s%10.3f%10.4f%8.2f%%%8.2f%%%8.1f%%%8.1f%%' % (
                    m if first else '', ('%.2f' % tgt) if mode == 'support' else '',
                    'support(2z0)' if mode == 'support' else 'full(2z0+2gap)',
                    d['correct_retain'], d['pos_miou'], d['fgr_all'] * 100,
                    d['by_ht']['relation'] * 100, d['cbr'][3] * 100,
                    d['cbr'][2] * 100))
                first = False

    print('\n探针净效应（full − support，同 target）：')
    for m in MODELS:
        for tgt in targets:
            s = res[m]['arms']['%.2f|support' % tgt]
            f = res[m]['arms']['%.2f|full' % tgt]
            print('  %-12s t=%.2f  relCBR %+6.2fpp  relFGR %+6.2fpp  '
                  'ALLFGR %+6.2fpp  CorrRetain %+.3f'
                  % (m, tgt,
                     (f['cbr'][3] - s['cbr'][3]) * 100,
                     (f['by_ht']['relation'] - s['by_ht']['relation']) * 100,
                     (f['fgr_all'] - s['fgr_all']) * 100,
                     f['correct_retain'] - s['correct_retain']))

    print('\n覆盖：')
    for m in MODELS:
        d = res[m]['arms']['%.2f|full' % targets[0]]
        print('  %-12s rows=%d  probe-covered=%d  n_c=%d'
              % (m, d['n_rows'], d['n_covered'], res[m]['n_c']))

    # ---------- frontier: read both arms at EQUAL ACHIEVED CorrectRetain.
    # full consistently lands at a more permissive achieved retention than
    # support (+0.008..+0.043 here), so a fixed-target comparison partly measures
    # the operating point rather than discrimination.
    print('\n' + '=' * 104)
    print('前沿比较：同【实际达到】CorrectRetain（逐模型内插，不可达不外推）')
    print('=' * 104)
    METS = [(lambda d: d['fgr_all'] * 100, 'ALL FGR'),
            (lambda d: d['by_ht']['relation'] * 100, 'rel FGR'),
            (lambda d: d['cbr'][3] * 100, 'rel CBR'),
            (lambda d: d['cbr'][2] * 100, 'attr CBR')]

    def curve(m, mode, f):
        pts = []
        for tgt in targets:
            d = res[m]['arms']['%.2f|%s' % (tgt, mode)]
            cr = d['correct_retain']
            if cr == cr:
                pts.append((cr, f(d)))
        pts.sort()
        return pts

    def at(pts, x):
        if len(pts) < 2 or x < pts[0][0] or x > pts[-1][0]:
            return None
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            if x0 <= x <= x1:
                return y0 if x1 == x0 else y0 + (x - x0) / (x1 - x0) * (y1 - y0)
        return None

    front = {}
    for m in MODELS:
        lo = max(curve(m, 'support', METS[0][0])[0][0],
                 curve(m, 'full', METS[0][0])[0][0])
        hi = min(curve(m, 'support', METS[0][0])[-1][0],
                 curve(m, 'full', METS[0][0])[-1][0])
        print('  %-12s 两臂共同可达区间 CorrectRetain = [%.3f, %.3f]' % (m, lo, hi))
        for lvl in (0.95, 0.90, 0.88, 0.86):
            if not (lo <= lvl <= hi):
                print('     %.2f  不在共同可达区间，跳过（不外推）' % lvl)
                continue
            out = []
            for f, nm in METS:
                vs = at(curve(m, 'support', f), lvl)
                vf = at(curve(m, 'full', f), lvl)
                if vs is None or vf is None:
                    out.append('%s=n/a' % nm)
                    continue
                out.append('%s %+.2fpp' % (nm, vf - vs))
                front['%s|%.2f|%s' % (m, lvl, nm)] = dict(
                    support=vs, full=vf, delta=vf - vs)
            print('     %.2f  %s' % (lvl, '   '.join(out)))

    json.dump(dict(targets=targets, models=list(MODELS), result=res,
                   frontier=front),
              open(os.path.join(HERE, 'coordfix_combo.json'), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote coordfix_combo.json')


if __name__ == '__main__':
    main()
