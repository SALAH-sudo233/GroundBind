#!/usr/bin/env python3
"""Mitigation on JOINT grounding (t4): support verification and the full policy.

WHY SEPARATE FROM eval_paper_tables.py. That script hardcodes the t2 structure --
500 positives and 2000 negatives per model, so its rates divide by 500/2000. On t4
a model does not draw a box on every row, and the eligible set differs, so those
denominators are simply wrong here. This evaluator counts its own denominators from
the t4 rows it actually has and prints them, so a reader can check them.

SCORES COME FROM collect_t4_mitigation.py, collected on t4 boxes. t2 scores are NOT
reused: the verifier scores one specific box, and the t4 box differs from the t2 box
for the same query (adding a caption moves the box). Reusing them would be the same
error class as mixing two scorers inside one gap feature.

POLICY IS UNCHANGED FROM THE MAIN TABLE, deliberately: monotone gate (the probe head
may only turn keep -> reject, never overturn a support rejection) restricted to
predicates with a genuine opposing configuration. Rows the gate blocks inherit the
support decision bit-for-bit. That is what makes object / co_occurrence / attribute
strictly additive -- they are mitigated by a single Omni posterior and never enter
predicate-table competitive scoring, so a regression there would be an implementation
bug, not a property of the method.

FOLDS: image-level md5 % 2, the project-wide convention. Threshold is fitted on one
fold and read on the other, never on the rows it scores.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import statistics as st
import sys

PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
import eval_upstream as E
import eval_cbr_paper_aligned as A
import export_cf_joint as CF

HERE = os.path.dirname(os.path.abspath(__file__))
HT4 = ['object', 'co_occurrence', 'attribute', 'relation']
OPPOSED = ('behind', 'in front of', 'under', 'on top of', 'above', 'below')
COORDFIX_T4 = {'UniVG-R1', 'visual-rft'}
GENERAL = {'InternVL3.5-8B', 'Qwen3-VL-8B', 'llava-ov-7b', 'qwen2.5-vl-7b'}
TASK = 't4_caption_grounding'


def fold_of(sid):
    base = str(sid).split('__')[0]
    return int(hashlib.md5(base.encode()).hexdigest(), 16) % 2


def load_scores(paths):
    """(model, sid) -> {z0, rivals[], jev, ...} from the sharded collection."""
    by = {}
    nerr = 0
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p, encoding='utf-8'):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('error'):
                nerr += 1
            k = (d['model'], d['sid'])
            e = by.setdefault(k, dict(z0=None, rivals=[], jev=None, meta=d))
            kind = d.get('kind')
            if kind == 'z0':
                e['z0'] = d.get('z')
            elif kind == 'rival':
                if d.get('z') is not None:
                    e['rivals'].append((d.get('variant'), d['z']))
            elif kind == 'jev':
                e['jev'] = d.get('z_head')
    return by, nerr


def build_rows(canon, scores):
    """One row per t4 record that drew a usable box, with its scores attached."""
    rows = collections.defaultdict(list)
    for m, root in sorted(canon.items()):
        p = os.path.join(root, m, 'records.jsonl')
        if not os.path.exists(p):
            continue
        cache = {}
        for line in open(p, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if str(r.get('task', '')) != TASK:
                continue
            sid = r.get('sample_id') or r.get('base_sample_id')
            if m in COORDFIX_T4:
                box = CF.rescale(r, cache)
                iou = CF.iou(box, r.get('gt_bbox_xyxy')) if box else 0.0
            else:
                box = r.get('pred_bbox_xyxy')
                iou = r.get('iou') or 0.0
            drew = bool(box) and (box[2] - box[0]) > 0 and (box[3] - box[1]) > 0
            sc = scores.get((m, sid), {})
            is_pos = r.get('query_role') == 'positive'
            rows[m].append(dict(
                sid=sid, base=str(sid).split('__')[0], model=m,
                is_pos=is_pos, htype=r.get('hallucination_type'),
                drew=drew, box=box, iou=iou if is_pos else None,
                query=r.get('query') or '',
                z0=sc.get('z0'), rivals=sc.get('rivals', []),
                jev=sc.get('jev'),
                label=1 if (is_pos and iou >= 0.5) else 0,
                fold=fold_of(sid)))
    return rows


def gap_of(r):
    if r['z0'] is None or not r['rivals']:
        return None
    return max(z for _, z in r['rivals']) - r['z0']


def opposed_gate(r):
    """True only when a rival is a genuine opposing configuration."""
    return any(str(v or '').strip().lower() in OPPOSED for v, _ in r['rivals'])


def usable_support(r):
    return r['z0'] is not None or r['jev'] is not None


def usable_full(r):
    return gap_of(r) is not None and usable_support(r)


def feats_support(r):
    f = []
    f.append(r['z0'] if r['z0'] is not None else 0.0)
    f.append(r['jev'] if r['jev'] is not None else 0.0)
    return f


def feats_full(r):
    return feats_support(r) + [gap_of(r)]


def fit_head(rows, mode):
    sub = [r for r in rows
           if (usable_full(r) if mode == 'full' else usable_support(r))]
    if len(sub) < 30 or len(set(r['label'] for r in sub)) < 2:
        return None
    fx = feats_full if mode == 'full' else feats_support
    f, _ = E.logreg([fx(r) for r in sub], [r['label'] for r in sub])
    return lambda r: f(fx(r))


def build_keep(rows, elig, target, mode):
    """Cross-fitted decisions. mode='unfiltered'|'support'|'full'."""
    keep = {}
    if mode == 'unfiltered':
        return {r['sid']: bool(r['drew']) for r in rows}
    for tf in (0, 1):
        fit = [r for r in rows if r['fold'] != tf]
        tst = [r for r in rows if r['fold'] == tf]
        s_sup = fit_head(fit, 'support')
        if s_sup is None:
            for r in tst:
                keep[r['sid']] = bool(r['drew'])
            continue
        ps = sorted(s_sup(r) for r in fit
                    if r['is_pos'] and r['sid'] in elig and usable_support(r))
        k = int(round((1.0 - target) * len(ps)))
        t_sup = ps[max(0, min(k, len(ps) - 1))] if ps else float('-inf')

        def keep_sup(r):
            if not r['drew']:
                return False
            if not usable_support(r):
                return True
            return s_sup(r) >= t_sup

        if mode == 'support':
            for r in tst:
                keep[r['sid']] = keep_sup(r)
            continue

        adm = [r for r in fit if usable_full(r) and opposed_gate(r)]
        if len(adm) < 30 or len(set(r['label'] for r in adm)) < 2:
            for r in tst:
                keep[r['sid']] = keep_sup(r)
            continue
        s_full = fit_head(adm, 'full')
        ep = [r for r in fit if r['is_pos'] and r['sid'] in elig and r['drew']]
        cov = [r for r in ep if usable_full(r) and opposed_gate(r)]
        unc = [r for r in ep if not (usable_full(r) and opposed_gate(r))]
        fixed = sum(1 for r in unc if keep_sup(r))
        need = max(0, min(int(round(target * len(ep))) - fixed, len(cov)))
        sc = sorted((s_full(r) for r in cov), reverse=True)
        t_full = sc[need - 1] if need > 0 else float('inf')
        for r in tst:
            if not r['drew']:
                keep[r['sid']] = False
                continue
            if not (usable_full(r) and opposed_gate(r)):
                keep[r['sid']] = keep_sup(r)
                continue
            # monotone: the probe head may only turn keep -> reject
            keep[r['sid']] = keep_sup(r) and (s_full(r) >= t_full)
    return keep


def measure(rows, keep, elig):
    """Denominators counted from the t4 rows present, not assumed to be 500/2000."""
    pos = [r for r in rows if r['is_pos']]
    neg = [r for r in rows if not r['is_pos']]
    n_pos, n_neg = len(pos), len(neg)
    by, by_den = {}, {}
    for h in HT4:
        sub = [r for r in neg if r['htype'] == h]
        by_den[h] = len(sub)
        by[h] = (sum(1 for r in sub if keep.get(r['sid'])) / len(sub)
                 if sub else None)
    ce = [r for r in pos if r['sid'] in elig]
    posmap = {r['base']: r for r in pos}
    negmap = collections.defaultdict(dict)
    for r in neg:
        negmap[r['base']][r['htype']] = r
    cbr = {}
    for h in HT4:
        hit = tot = 0
        for b in elig:
            pr = posmap.get(b)
            nr = negmap.get(b, {}).get(h)
            if pr is None or nr is None:
                continue
            tot += 1
            if keep.get(nr['sid']) and nr['box'] and pr['box'] and \
                    CF.iou(nr['box'], pr['box']) >= 0.8:
                hit += 1
        cbr[h] = hit / tot if tot else None
    return dict(
        n_pos=n_pos, n_neg=n_neg, n_neg_by_ht=by_den,
        fgr=sum(1 for r in neg if keep.get(r['sid'])) / n_neg if n_neg else None,
        by_ht=by,
        boh=st.mean([by[h] for h in HT4[:2]]) if all(by[h] is not None for h in HT4[:2]) else None,
        roh=st.mean([by[h] for h in HT4[2:]]) if all(by[h] is not None for h in HT4[2:]) else None,
        pos_keep=sum(1 for r in pos if keep.get(r['sid'])) / n_pos if n_pos else None,
        pos_miou=sum((r['iou'] or 0) for r in pos if keep.get(r['sid'])) / n_pos
        if n_pos else None,
        correct_retain=(sum(1 for r in ce if keep.get(r['sid'])) / len(ce))
        if ce else None,
        n_eligible=len(ce), cbr=cbr)


def boot_ci(diffs, n=2000, seed=17):
    """Paired bootstrap over MODELS (equal weight), the project convention."""
    import random
    if not diffs:
        return (None, None)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        out.append(st.mean(s))
    out.sort()
    return (out[int(0.025 * n)], out[int(0.975 * n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=float, default=0.95)
    ap.add_argument('--scores', default='t4mitig')
    ap.add_argument('--json-out', default='t4_mitigation.json')
    a = ap.parse_args()

    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    paths = []
    d = os.path.join(PROBE, a.scores) if not os.path.isabs(a.scores) else a.scores
    if os.path.isdir(d):
        paths = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.endswith('.jsonl')]
    scores, nerr = load_scores(paths)
    print('loaded scores for %d (model,sid) keys from %d files, %d errored rows'
          % (len(scores), len(paths), nerr), flush=True)

    rows = build_rows(canon, scores)
    per = {}
    for m in sorted(rows):
        rs = rows[m]
        cov = sum(1 for r in rs if r['z0'] is not None)
        if cov == 0:
            print('[skip] %s: no t4 scores collected' % m, flush=True)
            continue
        pos = [r for r in rs if r['is_pos']]
        elig = set(r['base'] for r in pos
                   if r['drew'] and (r['iou'] or 0) >= 0.5)
        arms = {}
        for mode in ('unfiltered', 'support', 'full'):
            keep = build_keep(rs, elig, a.target, mode)
            arms[mode] = measure(rs, keep, elig)
        per[m] = arms
        u, s, f = arms['unfiltered'], arms['support'], arms['full']
        print('%-16s n_pos=%3d n_neg=%4d elig=%3d cov=%4d | FGR %6.2f -> %6.2f -> %6.2f'
              % (m, u['n_pos'], u['n_neg'], u['n_eligible'], cov,
                 u['fgr'] * 100, s['fgr'] * 100, f['fgr'] * 100), flush=True)

    if not per:
        print('no models evaluated -- is the collection finished?')
        return

    print()
    print('=' * 118)
    print('t4 JOINT GROUNDING MITIGATION  (%d models, target %.2f)'
          % (len(per), a.target))
    print('=' * 118)
    hdr = '%-26s %8s %8s %8s %9s %9s' % ('arm', 'FGR', 'BOH', 'ROH', 'Rcorrect', 'mIoU')
    for h in HT4:
        hdr += ' %9s' % ('CBR ' + h[:4])
    print(hdr)
    print('-' * 118)
    labels = [('unfiltered', 'unfiltered'), ('support', 'support verification only'),
              ('full', 'adaptive structured ver.')]
    pooled = {}
    for key, lab in labels:
        vals = lambda f: [f(per[m][key]) for m in per if f(per[m][key]) is not None]
        row = dict(
            fgr=st.mean(vals(lambda x: x['fgr'])),
            boh=st.mean(vals(lambda x: x['boh'])),
            roh=st.mean(vals(lambda x: x['roh'])),
            correct_retain=st.mean(vals(lambda x: x['correct_retain'])),
            pos_miou=st.mean(vals(lambda x: x['pos_miou'])),
            cbr={h: st.mean([per[m][key]['cbr'][h] for m in per
                             if per[m][key]['cbr'][h] is not None]) for h in HT4})
        pooled[key] = row
        line = '%-26s %7.2f%% %7.2f%% %7.2f%% %8.2f%% %9.4f' % (
            lab, row['fgr'] * 100, row['boh'] * 100, row['roh'] * 100,
            row['correct_retain'] * 100, row['pos_miou'])
        for h in HT4:
            line += ' %8.2f%%' % (row['cbr'][h] * 100)
        print(line)
    print('-' * 118)
    print('per-type FGR:')
    for h in HT4:
        print('  %-16s %6.2f%% -> %6.2f%% -> %6.2f%%'
              % (h, *[st.mean([per[m][k]['by_ht'][h] for m in per
                               if per[m][k]['by_ht'][h] is not None]) * 100
                      for k in ('unfiltered', 'support', 'full')]))

    print()
    print('=' * 118)
    print('STRICT-ADDITIVITY CHECK: full must not be worse than support on any type')
    print('=' * 118)
    ok = True
    for h in HT4:
        d = [per[m]['full']['by_ht'][h] - per[m]['support']['by_ht'][h]
             for m in per if per[m]['full']['by_ht'][h] is not None]
        lo, hi = boot_ci(d)
        worse = sum(1 for x in d if x > 1e-12)
        flag = 'OK' if st.mean(d) <= 1e-12 else 'REGRESSION'
        ok &= (st.mean(d) <= 1e-12)
        print('  FGR %-16s delta %+7.3fpp CI[%+.3f,%+.3f] worse_models=%d/%d  %s'
              % (h, st.mean(d) * 100, (lo or 0) * 100, (hi or 0) * 100,
                 worse, len(d), flag))
    dall = [per[m]['full']['fgr'] - per[m]['support']['fgr'] for m in per]
    lo, hi = boot_ci(dall)
    print('  FGR %-16s delta %+7.3fpp CI[%+.3f,%+.3f] improved=%d/%d'
          % ('ALL', st.mean(dall) * 100, (lo or 0) * 100, (hi or 0) * 100,
             sum(1 for x in dall if x < 0), len(dall)))
    dr = [per[m]['full']['correct_retain'] - per[m]['support']['correct_retain']
          for m in per]
    print('  correct retention delta %+.4f  (cost of the full arm)' % st.mean(dr))
    print('\n  %s' % ('no type regressed -- the full arm is strictly additive on t4'
                      if ok else 'A TYPE REGRESSED: check the gate, not the method'))

    json.dump(dict(per=per, pooled=pooled, target=a.target,
                   n_models=len(per), strict_additive=bool(ok),
                   coordfix_t4=sorted(COORDFIX_T4)),
              open(os.path.join(HERE, a.json_out), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote %s' % a.json_out)


if __name__ == '__main__':
    main()
