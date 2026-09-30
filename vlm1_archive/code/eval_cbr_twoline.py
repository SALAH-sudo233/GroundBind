#!/usr/bin/env python3
"""Two-line CBR comparison: the JEV/2B verifier line vs the Omni line.

Motivation (user's call): H2 killed the "2B selects the probe" role, but that only
ruled out ONE job for the 2B verifier. It can still be the DECIDING verifier on
its own line. So we keep the Omni mitigation result as-is and evaluate a parallel
JEV line, each line using its own verifier for both the original expression and
the competitive probe.

Lines (each is self-contained; no cross-line fusion here):
  OMNI line   z0 = s5omni_omni_<m>.jsonl     probe z = probe_upstream.jsonl
  JEV  line   z0 = s5jevz_q35_2b_<m>.jsonl   probe z = sel2b_all.jsonl (z_jev)

Arms within each line:
  B0    upstream unfiltered (identical for both lines; sanity anchor)
  B1    that line's verifier alone, threshold on z0
  A2    B1 + competitive-probe gap, GATED to rows the probe actually covers
        (gate is mandatory: adding a feature that is 0 on uncovered rows moves
         the global threshold and silently changes those rows' decisions -- it
         made co_occurrence/attribute significantly WORSE in an earlier run)

CBR contract is unchanged and re-asserted here:
  c_i   = 1[positive PREDICTED box valid AND IoU(pred_pos, gt) >= 0.5]
  r_i,t = 1[negative box valid AND IoU(neg_box, pred_pos_box) >= 0.8]
  eligibility and the positive reference box are FROZEN pre-mitigation; rejected /
  low-overlap / invalid negatives are ZERO reuse but STAY in the denominator;
  c_i = 0 drops the group; no eligible positive -> undefined, never 0.
  All four htypes share ONE eligibility set -> qualified_pos must be identical
  across the four columns per model (asserted).

Thresholds are picked on fold A at a MATCHED positive-keep target so the two
lines are compared at equal aggressiveness, not at whatever operating point each
verifier happens to sit on. Per the project's rule, a verifier that rejects
almost everything can post a great neg-reject rate while being useless.
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_upstream as E          # logreg / pick_tau / model_level_paired / auroc

S5 = os.path.expanduser('~/SVD/grpo_verifier/s5grpo')
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


def _selftest():
    assert abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 5, 5, 9]) == 0.0     # degenerate -> no reuse
    assert not valid_box([0, 0, 0, 0])
    assert abs(E.auroc([1, 1, 1, 1], [1, 1, 0, 0]) - 0.5) < 1e-12


def load_boxes(model, root):
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
            pos[base] = dict(pred=bb, gt=parse_bbox(r.get('gt_bbox_xyxy')), sid=sid)
        else:
            ht = r.get('hallucination_type')
            if ht in HT4:
                neg[base][ht] = dict(pred=bb, sid=sid)
    return pos, neg


def load_line(line, models):
    """Rows per model with that line's own z0 and its own probe gap."""
    # probe scores: OMNI from probe_upstream.jsonl, JEV from sel2b_all.jsonl
    probe = defaultdict(dict)
    if line == 'omni':
        for l in open(os.path.join(HERE, 'probe_upstream.jsonl'), encoding='utf-8'):
            d = json.loads(l)
            if d.get('error') or d.get('z') is None:
                continue
            probe[(d['model'], d['sid'])][d['variant']] = d['z']
    else:
        for l in open(os.path.join(HERE, 'sel2b_all.jsonl'), encoding='utf-8'):
            d = json.loads(l)
            if d.get('error') or d.get('z_jev_q0') is None:
                continue
            for c in d['rivals']:
                if c.get('z_jev') is not None:
                    probe[(d['model'], d['sid'])][c['variant']] = c['z_jev']

    data = {}
    for m in models:
        f = ('s5omni_omni_%s.jsonl' if line == 'omni'
             else 's5jevz_q35_2b_%s.jsonl') % m
        p = os.path.join(S5, f)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print('[skip] %s line=%s: missing/empty %s' % (m, line, f))
            continue
        rows = []
        for l in open(p, encoding='utf-8'):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            sid = r['sid']
            ht = r.get('htype')
            z0 = r.get('z_omni') if line == 'omni' else r.get('z_jev')
            rv = probe.get((m, sid), {})
            zs = [v for v in rv.values() if v is not None]
            rows.append(dict(
                model=m, sid=sid, htype=ht, is_pos=(ht == 'positive'),
                label=1 if ht == 'positive' else 0,
                iou=float(r.get('iou') or 0.0), drew=bool(r.get('drew')),
                z0=z0, zmax=(max(zs) if zs else None), n_rivals=len(zs),
                fold=E.fold_of(sid), img=E.img_of(sid)))
        data[m] = rows
    return data


def gap(r):
    if r['zmax'] is None or r['z0'] is None:
        return 0.0
    return r['zmax'] - r['z0']


def covered(r):
    return r['zmax'] is not None and r['z0'] is not None


def build(A, pos_keep):
    """Fit B1 and the gated A2 on fold A. A2 is fitted on COVERED rows only, so
    its threshold is not contaminated by rows where the gap is structurally 0."""
    Ad = [r for r in A if r['drew'] and r['z0'] is not None]
    y = [r['label'] for r in Ad]
    f1, w1 = E.logreg([[r['z0']] for r in Ad], y)
    s1 = lambda r: f1([r['z0']])
    t1 = E.pick_tau(A, s1, pos_keep)

    Ac = [r for r in Ad if covered(r)]
    if len(Ac) >= 30 and len(set(r['label'] for r in Ac)) > 1:
        f2, w2 = E.logreg([[r['z0'], gap(r)] for r in Ac], [r['label'] for r in Ac])
        s2 = lambda r: f2([r['z0'], gap(r)])
        posc = sorted(s2(r) for r in Ac if r['is_pos'])
        if posc:
            k = int(round((1.0 - pos_keep) * len(posc)))
            t2 = posc[max(0, min(k, len(posc) - 1))]
        else:
            t2 = float('-inf')
    else:
        f2, w2, s2, t2 = None, [], None, float('-inf')

    def keep_b1(r):
        if not r['drew']:
            return False
        if r['z0'] is None:
            return True
        return s1(r) >= t1

    def keep_a2(r):
        # GATE: only covered rows switch to the probe head; everything else keeps
        # the single-arm decision byte-for-byte.
        if not covered(r) or s2 is None:
            return keep_b1(r)
        return s2(r) >= t2

    return keep_b1, keep_a2, w1, w2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos-keep', type=float, default=0.95)
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()
    _selftest()
    print('geometry + AUROC self-tests PASSED\n')

    canon = json.load(open(os.path.join(HERE, 'canon_roots.json')))
    models = sorted(canon)
    lines = {}
    for ln in ('omni', 'jev'):
        lines[ln] = load_line(ln, models)

    common = [m for m in models
              if lines['omni'].get(m) and lines['jev'].get(m)]
    print('models present in BOTH lines: %d / %d' % (len(common), len(models)))
    if len(common) < len(models):
        print('  missing: %s' % ', '.join(m for m in models if m not in common))
    print('注：两条线必须在同一模型集合上比较；缺失模型一并列出，不做静默丢弃。\n')

    out = {}
    agg = {ln: {arm: defaultdict(list) for arm in ('B0', 'B1', 'A2')}
           for ln in ('omni', 'jev')}
    per = {ln: {} for ln in ('omni', 'jev')}

    for ln in ('omni', 'jev'):
        print('=' * 104)
        print('%s 线 CBR   阈值 fold A 选 (pos_keep=%.2f)，fold B 读一次；'
              'A2 只改探针覆盖行' % (ln.upper(), a.pos_keep))
        print('=' * 104)
        print('%-17s%5s  %s' % ('model', 'n_c',
                                ''.join('%11s' % h[:9] for h in HT4)))
        for m in common:
            rows = lines[ln][m]
            A = [r for r in rows if r['fold'] == 0]
            B = [r for r in rows if r['fold'] == 1]
            k1, k2, w1, w2 = build(A, a.pos_keep)
            pos, neg = load_boxes(m, canon[m])
            foldB = {r['sid'] for r in B}
            elig = [b for b, p in pos.items()
                    if p['sid'] in foldB and valid_box(p['pred'])
                    and iou(p['pred'], p['gt']) >= 0.5]
            n_c = len(elig)
            keepmap = {}
            for r in rows:
                keepmap[r['sid']] = (k1(r), k2(r))
            res, npair = {}, {}
            for arm, idx in (('B0', None), ('B1', 0), ('A2', 1)):
                vals = []
                for ht in HT4:
                    num = den = 0
                    for b in elig:
                        nb = neg.get(b, {}).get(ht)
                        if not nb:
                            continue
                        den += 1
                        if idx is not None:
                            kv = keepmap.get(nb['sid'])
                            if kv is not None and not kv[idx]:
                                continue
                        if iou(nb['pred'], pos[b]['pred']) >= 0.8:
                            num += 1
                    vals.append(num / n_c if n_c else float('nan'))
                    npair[ht] = den
                    if n_c:
                        agg[ln][arm][ht].append(vals[-1])
                res[arm] = vals
            # self-check: the four htypes MUST share one eligibility set
            assert len(set(npair.values())) == 1, \
                ('%s %s: qualified_pos differs across htypes %s'
                 % (ln, m, npair))
            per[ln][m] = dict(n_c=n_c, **res)
            for arm in ('B0', 'B1', 'A2'):
                tag = m if arm == 'B0' else ''
                print('%-17s%5s  %s   %s'
                      % (tag, (n_c if arm == 'B0' else ''),
                         ''.join('%10.1f%%' % (v * 100) for v in res[arm]), arm))
        print()

    print('=' * 104)
    print('两条线池化对比（%d 模型等权）' % len(common))
    print('=' * 104)
    print('%-14s%-6s%s' % ('line', 'arm', ''.join('%12s' % h[:9] for h in HT4)))
    for ln in ('omni', 'jev'):
        for arm in ('B0', 'B1', 'A2'):
            print('%-14s%-6s%s' % (ln.upper() if arm == 'B0' else '', arm,
                                   ''.join('%11.1f%%' % (st.mean(agg[ln][arm][h]) * 100)
                                           for h in HT4)))

    print('\n模型级配对检验 (n=%d 等权)' % len(common))
    stat = {}
    for ln in ('omni', 'jev'):
        print('  %s 线   A2 − B1' % ln.upper())
        for ht in HT4:
            i = HT4.index(ht)
            d = [per[ln][m]['A2'][i] - per[ln][m]['B1'][i]
                 for m in common if per[ln][m]['n_c']]
            mm, lo, hi = E.model_level_paired(d)
            print('    %-16s %+7.2fpp CI[%+6.2f,%+6.2f] %-6s 下降 %d/%d'
                  % (ht, mm * 100, lo * 100, hi * 100,
                     '显著' if (lo > 0 or hi < 0) else '不显著',
                     sum(1 for x in d if x < 0), len(d)))
            stat['%s_A2-B1_%s' % (ln, ht)] = dict(mean=mm, ci=[lo, hi])

    print('\n  线间对比：JEV 线 − OMNI 线（同 pos_keep 目标）')
    for arm in ('B1', 'A2'):
        print('    %s' % arm)
        for ht in HT4:
            i = HT4.index(ht)
            d = [per['jev'][m][arm][i] - per['omni'][m][arm][i]
                 for m in common if per['jev'][m]['n_c'] and per['omni'][m]['n_c']]
            mm, lo, hi = E.model_level_paired(d)
            print('      %-16s %+7.2fpp CI[%+6.2f,%+6.2f] %s'
                  % (ht, mm * 100, lo * 100, hi * 100,
                     '显著' if (lo > 0 or hi < 0) else '不显著'))
            stat['jev-omni_%s_%s' % (arm, ht)] = dict(mean=mm, ci=[lo, hi])

    if a.json_out:
        json.dump(dict(per_line=per, comparisons=stat, models=common,
                       pos_keep_target=a.pos_keep),
                  open(a.json_out, 'w'), indent=2, ensure_ascii=False)
        print('\nwrote %s' % a.json_out)


if __name__ == '__main__':
    main()
