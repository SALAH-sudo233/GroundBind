#!/usr/bin/env python3
"""Baseline + mitigation evaluation for the two coordinate-corrected models.

Reviewer W2 established that UniVG-R1 and visual-rft emit 0-1000 normalized boxes
that the pipeline read as pixels. `write_rescaled_run.py` materialised a corrected
run; this script reports, for those two models only:

  B0            corrected upstream, no verifier
  support-only  z0 from both verifiers, no probes
  full          z0 + text-routed contrastive gap from both verifiers

under the paper's denominators (500 positives, 500 per negative type), image-level
two-fold cross-fitting, and the label-independent query-text router.

All verifier scores are FRESH: the original z0/probe files scored a red box drawn on
the CLIPPED coordinates, so they cannot be reused for the corrected boxes. Inputs:
  s5omni_omnicf_<model>.jsonl   OMNI z0 on corrected boxes
  jevheadcf_<model>.jsonl       JEV  z0 on corrected boxes
  probecf_omni.jsonl            OMNI rivals, text-routed
  probecf_jev.jsonl             JEV  rivals, same rival texts, same scorer rule

These two models are reported SEPARATELY from the 13-model pooled tables: their
boxes come from a different (corrected) run, so pooling them with the 11 untouched
models would mix provenance.
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_upstream as E
import eval_cbr_paper_aligned as A

HT4 = A.HT4
S5 = os.path.expanduser('~/SVD/grpo_verifier/s5grpo')
MODELS = ('UniVG-R1', 'visual-rft')


def load_probe(path, key):
    out = defaultdict(dict)
    if not os.path.exists(path):
        return out
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get('error') or d.get(key) is None:
            continue
        out[(d['model'], d['sid'])][d['variant']] = d[key]
    return out


def load_rows(model, line, probe):
    if line == 'omni':
        p = os.path.join(S5, 's5omni_omnicf_%s.jsonl' % model)
        zk = 'z_omni'
    else:
        p = os.path.join(HERE, 'jevheadcf_%s.jsonl' % model)
        zk = 'z_head'
    if not os.path.exists(p):
        sys.exit('missing %s' % p)
    rows = []
    for line_ in open(p, encoding='utf-8'):
        line_ = line_.strip()
        if not line_:
            continue
        r = json.loads(line_)
        if r.get('model') and r['model'] != model:
            continue
        sid = r['sid']
        rv = probe.get((model, sid), {})
        zs = [v for v in rv.values() if v is not None]
        ht = r.get('htype')
        rows.append(dict(
            model=model, sid=sid, htype=ht, is_pos=(ht == 'positive'),
            label=1 if ht == 'positive' else 0,
            iou=float(r.get('iou') or 0.0), drew=bool(r.get('drew')),
            z0=r.get(zk), zmax=(max(zs) if zs else None),
            fold=E.fold_of(sid)))
    return rows


def gap(r):
    if r['zmax'] is None or r['z0'] is None:
        return 0.0
    return r['zmax'] - r['z0']


def covered(r):
    return r['zmax'] is not None and r['z0'] is not None


def fit(rows, use_gap):
    R = [r for r in rows if r['drew'] and r['z0'] is not None
         and (not use_gap or covered(r))]
    if len(R) < 30 or len(set(r['label'] for r in R)) < 2:
        return None
    X = [([r['z0'], gap(r)] if use_gap else [r['z0']]) for r in R]
    f, _ = E.logreg(X, [r['label'] for r in R])
    return (lambda r: f([r['z0'], gap(r)])) if use_gap else (lambda r: f([r['z0']]))


def thr(rows, score, elig, target):
    v = sorted(score(r) for r in rows
               if r['is_pos'] and r['sid'] in elig and r['drew']
               and r['z0'] is not None)
    if not v:
        return float('-inf')
    k = int(round((1.0 - target) * len(v)))
    return v[max(0, min(k, len(v) - 1))]


def keepmap(rows, elig, target, mode):
    keep = {}
    for tf in (0, 1):
        f_ = [r for r in rows if r['fold'] != tf]
        t_ = [r for r in rows if r['fold'] == tf]
        s0 = fit(f_, False)
        if s0 is None:
            for r in t_:
                keep[r['sid']] = bool(r['drew'])
            continue
        t0 = thr(f_, s0, elig, target)
        s1 = fit(f_, True) if mode == 'full' else None
        t1 = thr([r for r in f_ if covered(r)], s1, elig, target) if s1 else None
        for r in t_:
            if not r['drew']:
                keep[r['sid']] = False
            elif r['z0'] is None:
                keep[r['sid']] = True
            elif s1 is not None and covered(r):
                keep[r['sid']] = s1(r) >= t1
            else:
                keep[r['sid']] = s0(r) >= t0
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
        n_covered=sum(1 for r in rows if covered(r)))


def main():
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90
    cf = json.load(open(os.path.join(HERE, 'canon_roots_coordfix.json')))
    paper = json.load(open(os.path.join(HERE, 'canon_roots_paper.json')))
    po = load_probe(os.path.join(HERE, 'probecf_omni.jsonl'), 'z')
    pj = load_probe(os.path.join(HERE, 'probecf_jev.jsonl'), 'z_head')
    print('probe keys: omni=%d jev=%d   target CorrectRetain=%.2f\n'
          % (len(po), len(pj), target))

    res = {}
    for m in MODELS:
        pos, neg = A.load_boxes(m, cf[m])
        elig = {q['sid'] for b, q in pos.items()
                if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
        elig_base = {b for b, q in pos.items()
                     if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
        # --- before/after baseline
        opos, oneg = A.load_boxes(m, paper[m])
        o_nc = sum(1 for b, q in opos.items()
                   if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5)
        o_fgr = {ht: sum(1 for b in oneg if oneg[b].get(ht)
                         and A.valid_box(oneg[b][ht]['pred'])) / 500.0
                 for ht in HT4}
        o_miou = sum(A.iou(q['pred'], q['gt']) for q in opos.values()) / 500.0
        o_cbr = A.cbr(list(elig_base if False else
                          {b for b, q in opos.items()
                           if A.valid_box(q['pred'])
                           and A.iou(q['pred'], q['gt']) >= 0.5}),
                      opos, oneg, {}, None)
        n_nc = len(elig_base)
        n_fgr = {ht: sum(1 for b in neg if neg[b].get(ht)
                         and A.valid_box(neg[b][ht]['pred'])) / 500.0
                 for ht in HT4}
        n_miou = sum(A.iou(q['pred'], q['gt']) for q in pos.values()) / 500.0
        n_cbr = A.cbr(list(elig_base), pos, neg, {}, None)

        arms = {}
        for lname in ('omni', 'jev'):
            probe = po if lname == 'omni' else pj
            rows = load_rows(m, lname, probe)
            for mode in ('support', 'full'):
                k = keepmap(rows, elig, target, mode)
                arms[(lname, mode)] = measure(rows, k, elig, pos, neg)
            arms[(lname, 'rows')] = len(rows)
        res[m] = dict(before=dict(n_c=o_nc, fgr=o_fgr, miou=o_miou, cbr=o_cbr),
                      after=dict(n_c=n_nc, fgr=n_fgr, miou=n_miou, cbr=n_cbr),
                      arms=arms)

    # ---------- baseline before/after
    print('=' * 104)
    print('坐标修正前后的 500-dev 基线（论文分母；CBR 在各自的合格集上）')
    print('=' * 104)
    print('%-12s%-9s%6s%8s%9s%9s%9s%9s%9s' %
          ('model', 'run', 'n_c', 'posmIoU', 'objFGR', 'cooFGR', 'attFGR',
           'relFGR', 'relCBR'))
    for m in MODELS:
        for tag in ('before', 'after'):
            d = res[m][tag]
            print('%-12s%-9s%6d%8.4f%8.1f%%%8.1f%%%8.1f%%%8.1f%%%8.1f%%' % (
                m if tag == 'before' else '', tag, d['n_c'], d['miou'],
                d['fgr']['object'] * 100, d['fgr']['co_occurrence'] * 100,
                d['fgr']['attribute'] * 100, d['fgr']['relation'] * 100,
                d['cbr'][3] * 100))

    # ---------- mitigation
    print('\n' + '=' * 104)
    print('缓释评测（修正坐标 + 文本路由 + 交叉拟合，target CorrectRetain=%.2f）' % target)
    print('=' * 104)
    print('%-12s%-6s%-9s%11s%9s%9s%9s%9s%9s' %
          ('model', 'line', 'arm', 'CorrRetain', 'posmIoU', 'FGR', 'relFGR',
           'relCBR', 'attrCBR'))
    for m in MODELS:
        for lname in ('omni', 'jev'):
            for mode in ('support', 'full'):
                d = res[m]['arms'][(lname, mode)]
                print('%-12s%-6s%-9s%10.3f%10.4f%8.2f%%%8.2f%%%8.1f%%%8.1f%%' % (
                    m if (lname, mode) == ('omni', 'support') else '',
                    lname if mode == 'support' else '', mode,
                    d['correct_retain'], d['pos_miou'], d['fgr_all'] * 100,
                    d['by_ht']['relation'] * 100, d['cbr'][3] * 100,
                    d['cbr'][2] * 100))

    print('\n探针覆盖行数（文本路由）：')
    for m in MODELS:
        print('  %-12s omni=%d  jev=%d  (arm rows=%d)'
              % (m, res[m]['arms'][('omni', 'full')]['n_covered'],
                 res[m]['arms'][('jev', 'full')]['n_covered'],
                 res[m]['arms'][('omni', 'rows')]))

    ser = {m: dict(before=res[m]['before'], after=res[m]['after'],
                   arms={'%s|%s' % (k[0], k[1]): v
                         for k, v in res[m]['arms'].items() if k[1] != 'rows'})
           for m in MODELS}
    json.dump(dict(target=target, models=list(MODELS), result=ser),
              open(os.path.join(HERE, 'coordfix_eval.json'), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote coordfix_eval.json')


if __name__ == '__main__':
    main()
