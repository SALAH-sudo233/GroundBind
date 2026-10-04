#!/usr/bin/env python3
"""Matched-retention comparison for the CURRENT policy (A_reject_opp).

WHY NEW. matched_retention.json was generated before final_policy.json and ran the
old A_all (pooled head, since withdrawn). That result showed ALL FGR +2.05pp
CI[+0.61,+3.66], 9/11 models worse — exactly the numbers CORRECTION.md withdrew.

This script recomputes the matched comparison using eval_paper_tables.build(),
which implements the current policy: monotone gate (probe head may only turn
keep->reject) restricted to predicates with a genuine opposing configuration.

PROCEDURE:
  [1] For each model, sweep retention target in {0.99,0.98,...,0.8} for both arms,
      recording ACHIEVED CorrectRetain against relation/ALL FGR, CBR, mIoU.
  [2] At the base target (0.95), take the support arm's achieved retention r.
  [3] Linearly interpolate the full arm's curve at that same r.
  [4] Report paired deltas (full - support) at matched retention, not at matched target.

Curves use the test-fold portion only (evaluation, not calibration).
"""
import argparse
import collections
import json
import os
import random
import statistics as st
import sys

PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
import eval_cbr_paper_aligned as A

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_unified import COORDFIX_MODELS, load_probe, merge, measure
from eval_paper_tables import build, OPPOSED
from eval_attribution import predicate_of, query_map

HT4 = A.HT4
TARGETS = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.9, 0.88, 0.85, 0.8]


def interp(pts, x):
    """Linear interpolation. pts: [(x_i, y_i)] sorted by x. Return y at x, or None."""
    pts = sorted(pts)
    if not pts:
        return None
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def boot_ci(diffs, n=10000, seed=17):
    """Paired bootstrap over MODELS (equal weight). diffs: per-model deltas."""
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
    ap.add_argument('--base-target', type=float, default=0.95)
    ap.add_argument('--json-out', default='matched_retention_current.json')
    a = ap.parse_args()

    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    cfroots = json.load(open(os.path.join(PROBE, 'canon_roots_coordfix.json')))
    models = sorted(canon)

    po_main = load_probe('probe_textroute_all.jsonl', 'z')
    pj_main = load_probe('trprobe_jev_all.jsonl', 'z_head')
    po_cf = load_probe('probecf_omni.jsonl', 'z', COORDFIX_MODELS)
    pj_cf = load_probe('probecf_jev.jsonl', 'z_head', COORDFIX_MODELS)

    curves = {}
    for m in models:
        cf = m in COORDFIX_MODELS
        root = cfroots[m] if cf else canon[m]
        qmap = query_map(root, m)
        pos, neg = A.load_boxes(m, root)
        elig = {q['sid'] for b, q in pos.items()
                if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
        rows = merge(m, po_cf if cf else po_main, pj_cf if cf else pj_main, cf)
        if not rows:
            continue
        pr_of = {r['sid']: (predicate_of(qmap.get(r['sid'], '')) or '') for r in rows}
        gate = lambda r: pr_of.get(r['sid'], '') in OPPOSED

        curves[m] = {'support': [], 'full': []}
        for mode in ('support', 'full'):
            for t in TARGETS:
                k = build(rows, elig, t, mode, gate=gate, monotone=True)
                d = measure(rows, k, elig, pos, neg)
                curves[m][mode].append(dict(
                    target=t,
                    achieved=d['correct_retain'],
                    rel_fgr=d['by_ht']['relation'] * 100,
                    all_fgr=d['fgr_all'] * 100,
                    rel_cbr=d['cbr'][3] * 100,
                    obj_cbr=d['cbr'][0] * 100,
                    cooc_cbr=d['cbr'][1] * 100,
                    attr_cbr=d['cbr'][2] * 100,
                    pos_miou=d['pos_miou']))
        print('[ok] %-16s %d curve points' % (m, len(TARGETS)), flush=True)

    common = sorted(curves)
    print('\n' + '=' * 110)
    print('FGR–Rcorrect curves (pooled, %d models equal weight, CURRENT POLICY)' % len(common))
    print('=' * 110)
    print('%-9s | %-35s | %-35s' % ('target', 'support: achieved / relFGR / relCBR',
                                      'full: achieved / relFGR / relCBR'))
    for i, t in enumerate(TARGETS):
        row = []
        for mode in ('support', 'full'):
            ach = st.mean([curves[m][mode][i]['achieved'] for m in common])
            rf = st.mean([curves[m][mode][i]['rel_fgr'] for m in common])
            rc = st.mean([curves[m][mode][i]['rel_cbr'] for m in common])
            row.append('%.4f / %6.2f%% / %6.2f%%' % (ach, rf, rc))
        print('%-9.2f | %-35s | %-35s' % (t, row[0], row[1]))

    # ---------- matched comparison ----------
    print('\n' + '=' * 110)
    print('MATCHED achieved retention: support@%.2f vs full interpolated at the SAME retention' % a.base_target)
    print('=' * 110)
    print('%-16s%10s%12s%12s%10s%12s%12s%10s%10s' %
          ('model', 'R_match', 'supRelFGR', 'fullRelFGR', 'dFGR',
           'supRelCBR', 'fullRelCBR', 'dCBR', 'dALL'))

    matched = {}
    for m in common:
        sup = [c for c in curves[m]['support'] if c['target'] == a.base_target][0]
        r = sup['achieved']
        f_rel_fgr = interp([(c['achieved'], c['rel_fgr']) for c in curves[m]['full']], r)
        f_rel_cbr = interp([(c['achieved'], c['rel_cbr']) for c in curves[m]['full']], r)
        f_all_fgr = interp([(c['achieved'], c['all_fgr']) for c in curves[m]['full']], r)
        f_miou = interp([(c['achieved'], c['pos_miou']) for c in curves[m]['full']], r)
        f_obj = interp([(c['achieved'], c['obj_cbr']) for c in curves[m]['full']], r)
        f_cooc = interp([(c['achieved'], c['cooc_cbr']) for c in curves[m]['full']], r)
        f_attr = interp([(c['achieved'], c['attr_cbr']) for c in curves[m]['full']], r)

        matched[m] = dict(
            r=r,
            sup_rel_fgr=sup['rel_fgr'], full_rel_fgr=f_rel_fgr,
            sup_rel_cbr=sup['rel_cbr'], full_rel_cbr=f_rel_cbr,
            sup_all_fgr=sup['all_fgr'], full_all_fgr=f_all_fgr,
            sup_miou=sup['pos_miou'], full_miou=f_miou,
            sup_obj_cbr=sup['obj_cbr'], full_obj_cbr=f_obj,
            sup_cooc_cbr=sup['cooc_cbr'], full_cooc_cbr=f_cooc,
            sup_attr_cbr=sup['attr_cbr'], full_attr_cbr=f_attr,
            in_range=(f_rel_fgr is not None))

        if f_rel_fgr is not None:
            d_rel_fgr = f_rel_fgr - sup['rel_fgr']
            d_rel_cbr = f_rel_cbr - sup['rel_cbr']
            d_all_fgr = f_all_fgr - sup['all_fgr']
            print('%-16s%10.4f%11.2f%%%11.2f%%%+9.2f%10.2f%%%11.2f%%%+9.2f%+9.2f' %
                  (m, r, sup['rel_fgr'], f_rel_fgr, d_rel_fgr,
                   sup['rel_cbr'], f_rel_cbr, d_rel_cbr, d_all_fgr))
        else:
            print('%-16s%10.4f%11.2f%%      OUT_OF_RANGE' % (m, r, sup['rel_fgr']))

    inr = {k: v for k, v in matched.items() if v['in_range']}
    print('\n' + '=' * 110)
    print('POOLED paired deltas (full − support at matched retention), %d/%d models in range' %
          (len(inr), len(matched)))
    print('=' * 110)

    deltas = {}
    for k in ('all_fgr', 'rel_fgr', 'rel_cbr', 'obj_cbr', 'cooc_cbr', 'attr_cbr', 'miou'):
        ds = [inr[m]['full_' + k] - inr[m]['sup_' + k] for m in inr]
        lo, hi = boot_ci(ds)
        worse = sum(1 for x in ds if x > 0)
        unit = '' if k == 'miou' else 'pp'
        deltas[k] = dict(mean=st.mean(ds), ci_lo=lo, ci_hi=hi, worse_models=worse, n=len(ds))
        print('%-14s %+8.3f%-2s  CI[%+.3f,%+.3f]  worse_models=%d/%d' %
              (k.upper(), st.mean(ds), unit, lo or 0, hi or 0, worse, len(ds)))

    print('\nPer-model ALL FGR delta (full − support):')
    for m in sorted(inr, key=lambda x: inr[x]['full_all_fgr'] - inr[x]['sup_all_fgr']):
        v = inr[m]
        print('  %-16s %+7.2fpp  (r=%.4f)' % (m, v['full_all_fgr'] - v['sup_all_fgr'], v['r']))

    json.dump(dict(
        base_target=a.base_target,
        targets=TARGETS,
        models=common,
        curves=curves,
        matched=matched,
        pooled_deltas=deltas,
        n_in_range=len(inr),
        policy='A_reject_opp',
        note='monotone gate + scope gate, current policy from eval_paper_tables.py'
    ), open(os.path.join(HERE, a.json_out), 'w'), indent=2, ensure_ascii=False, default=str)
    print('\nwrote %s' % a.json_out)


if __name__ == '__main__':
    main()
