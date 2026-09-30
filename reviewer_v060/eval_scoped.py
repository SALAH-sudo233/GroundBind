#!/usr/bin/env python3
"""Where does relation contrast actually help? Scope the claim to its support.

The pooled decisive test (blexO+gap vs blexO) came out null. The gap-vs-prior
diagnostic rules out "the gap is redundant with the predicate prior" (predicate
identity explains only R^2=0.08 of gap variance) and rules out "the gap is
uninformative" (within-predicate AUROC: behind .807, under .890, on top of .984).

Remaining explanation: the pooled metric is dominated by `next to`, which has no
logical opposite -- its rivals are merely OTHER configurations that can hold
simultaneously, so its gap AUROC is .550 (near chance) on 5,815 rows.

This script re-runs the decisive comparison on the subset where the query's
predicate HAS a genuine opposing configuration (depth/vertical family), and on the
`next to` subset separately, so the paper claims only what the evidence supports.
"""
import json
import os
import statistics as st
import sys

PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
import eval_upstream as E

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_unified import COORDFIX_MODELS, load_probe, merge
from eval_attribution import predicate_of, query_map, crossfit_auroc

# Predicates with a genuine opposing configuration (my earlier measurement:
# `next to` is a CONFIGURATION statement whose opposite is vertical/depth, not
# distance -- `far from` scored 0.4369, an anti-signal).
OPPOSED = ('behind', 'in front of', 'under', 'on top of', 'above', 'below')

ARMS = {
    'support_z0': lambda r, b: [r['z_o'], r['z_j']],
    'full_gap':   lambda r, b: [r['z_o'], r['z_j'], r['g_o'], r['g_j']],
    'blex_only':  lambda r, b: [b],
    'blexO':      lambda r, b: [r['z_o'], r['z_j'], b],
    'blexO_gap':  lambda r, b: [r['z_o'], r['z_j'], r['g_o'], r['g_j'], b],
}


def main():
    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    cfroots = json.load(open(os.path.join(PROBE, 'canon_roots_coordfix.json')))
    models = sorted(canon)

    po_main = load_probe('probe_textroute_all.jsonl', 'z')
    pj_main = load_probe('trprobe_jev_all.jsonl', 'z_head')
    po_cf = load_probe('probecf_omni.jsonl', 'z', COORDFIX_MODELS)
    pj_cf = load_probe('probecf_jev.jsonl', 'z_head', COORDFIX_MODELS)

    scopes = {'opposed': lambda pr: pr in OPPOSED,
              'next_to': lambda pr: pr == 'next to'}
    res = {s: {} for s in scopes}

    for m in models:
        cf = m in COORDFIX_MODELS
        qmap = query_map(cfroots[m] if cf else canon[m], m)
        rows = merge(m, po_cf if cf else po_main, pj_cf if cf else pj_main, cf)
        rows = [r for r in rows if r['drew'] and r['z_o'] is not None
                and r['z_j'] is not None and r['zmax_o'] is not None
                and r['zmax_j'] is not None]
        for sname, sel in scopes.items():
            sub = [r for r in rows if sel(predicate_of(qmap.get(r['sid'], '')) or '')]
            if len(sub) < 80 or len(set(r['label'] for r in sub)) < 2:
                continue
            d = {}
            for nm, fn in ARMS.items():
                auc, n = crossfit_auroc(sub, qmap, fn)
                d[nm] = dict(auroc=auc, n=n)
            d['_n'] = len(sub)
            res[sname][m] = d
        print('[ok] %-16s opposed=%s next_to=%s'
              % (m,
                 res['opposed'].get(m, {}).get('_n', '-'),
                 res['next_to'].get(m, {}).get('_n', '-')), flush=True)

    out = {}
    for sname in ('opposed', 'next_to'):
        R = res[sname]
        common = sorted(R)
        if not common:
            continue
        print('\n' + '=' * 96)
        print('SCOPE = %s   (%d models, %d rows pooled)'
              % (sname, len(common), sum(R[m]['_n'] for m in common)))
        print('=' * 96)
        for nm in ('blex_only', 'support_z0', 'blexO', 'full_gap', 'blexO_gap'):
            vals = [R[m][nm]['auroc'] for m in common
                    if R[m][nm]['auroc'] == R[m][nm]['auroc']]
            if vals:
                print('  %-14s %.4f' % (nm, st.mean(vals)))
        st_ = {}
        for x, y, lab in (('full_gap', 'support_z0', 'full - support'),
                          ('blexO_gap', 'blexO', 'gap ON TOP of prior (decisive)')):
            dd = [R[m][x]['auroc'] - R[m][y]['auroc'] for m in common]
            mu, lo, hi = E.model_level_paired(dd)
            sig = (lo > 0 or hi < 0)
            print('  %-34s %+8.4f CI[%+7.4f,%+7.4f] %-9s up %d/%d'
                  % (lab, mu, lo, hi, 'SIG' if sig else 'ns',
                     sum(1 for v in dd if v > 0), len(dd)))
            st_[lab] = dict(mean=mu, ci=[lo, hi], significant=sig,
                            n_up=sum(1 for v in dd if v > 0), n=len(dd))
        out[sname] = dict(per=R, stats=st_, models=common,
                          n_rows=sum(R[m]['_n'] for m in common))

    json.dump(out, open(os.path.join(HERE, 'scoped.json'), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote scoped.json')


if __name__ == '__main__':
    main()
