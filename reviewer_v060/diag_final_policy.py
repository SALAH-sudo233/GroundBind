#!/usr/bin/env python3
"""FINAL: A_reject on OPPOSED scope = method + attribution requirement intersection.

User's objection (CORRECT): object/co_occurrence/attribute use a single Omni-7B
posterior — they NEVER enter structured predicate-table competitive scoring.
The full arm should be STRICTLY ADDITIVE on them and they must not move.

What I reported (A_all): probe head decides EVERY covered row. With the legal text
router "covered" now spans all four types, so those rows were handed to a head
fitted POOLED across types — gap's fitted weight has opposite signs per type
(object +0.418 / cooc +0.334 / attr +0.283 / relation -0.673). Pooling cancels
the relation signal and perturbs the other three.

A_reject (monotone: probe may only turn keep->reject, never rescue) already shows
the three types DO NOT GET WORSE (object -0.092pp ns, cooc -0.215pp SIG, attr -0.108pp SIG,
attr CBR -0.030pp ns), while relation收益 survives.

THIS RUN: A_reject_opposed = monotone gate + OPPOSED-predicate scope.
This is the method (composite accept) + attribution requirement (scoped to predicates
that have a genuine opposing rival) intersection, and the headline for P1-6.

Comparison set:
  support         base
  A_reject        monotone gate, all covered
  A_reject_opp    monotone gate, OPPOSED predicates only
  A_all           what I reported (for reference, shows the pooling damage)
"""
import json, os, sys, statistics as st
from collections import defaultdict

PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
import eval_upstream as E
import eval_cbr_paper_aligned as A

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_unified import (COORDFIX_MODELS, load_probe, merge, measure,
                          usable, covered, head, feats)
from eval_attribution import predicate_of, query_map

HT4 = A.HT4
OPPOSED = ('behind', 'in front of', 'under', 'on top of', 'above', 'below')

def build_gated(rows, elig, target, gate, monotone=False):
    keep = {}
    for tf in (0, 1):
        fit = [r for r in rows if r['fold'] != tf]
        tst = [r for r in rows if r['fold'] == tf]
        s_sup = head(fit, 'support')
        if s_sup is None:
            for r in tst: keep[r['sid']] = bool(r['drew'])
            continue
        ps = sorted(s_sup(r) for r in fit
                    if r['is_pos'] and r['sid'] in elig and usable(r, 'support'))
        k = int(round((1.0 - target) * len(ps)))
        t_sup = ps[max(0, min(k, len(ps) - 1))] if ps else float('-inf')

        def keep_sup(r):
            if not r['drew']: return False
            if not usable(r, 'support'): return True
            return s_sup(r) >= t_sup

        adm = [r for r in fit if usable(r, 'full') and gate(r)]
        if len(adm) < 30 or len(set(r['label'] for r in adm)) < 2:
            for r in tst: keep[r['sid']] = keep_sup(r)
            continue
        f, _ = E.logreg([feats('full', r) for r in adm], [r['label'] for r in adm])
        s_full = lambda r: f(feats('full', r))

        ep = [r for r in fit if r['is_pos'] and r['sid'] in elig and r['drew']]
        cov = [r for r in ep if usable(r, 'full') and gate(r)]
        unc = [r for r in ep if not (usable(r, 'full') and gate(r))]
        fixed = sum(1 for r in unc if keep_sup(r))
        need = max(0, min(int(round(target * len(ep))) - fixed, len(cov)))
        sc = sorted((s_full(r) for r in cov), reverse=True)
        t_full = sc[need - 1] if need > 0 else float('inf')

        for r in tst:
            if not r['drew']:
                keep[r['sid']] = False
                continue
            if not (usable(r, 'full') and gate(r)):
                keep[r['sid']] = keep_sup(r)
                continue
            d = s_full(r) >= t_full
            keep[r['sid']] = (keep_sup(r) and d) if monotone else d
    return keep

def main():
    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    cfroots = json.load(open(os.path.join(PROBE, 'canon_roots_coordfix.json')))
    models = sorted(canon)
    target = 0.95

    po_main = load_probe('probe_textroute_all.jsonl', 'z')
    pj_main = load_probe('trprobe_jev_all.jsonl', 'z_head')
    po_cf = load_probe('probecf_omni.jsonl', 'z', COORDFIX_MODELS)
    pj_cf = load_probe('probecf_jev.jsonl', 'z_head', COORDFIX_MODELS)

    POL = ('support', 'A_all', 'A_reject', 'A_reject_opp')
    per = {p: {} for p in POL}

    for m in models:
        cf = m in COORDFIX_MODELS
        root = cfroots[m] if cf else canon[m]
        qmap = query_map(root, m)
        pos, neg = A.load_boxes(m, root)
        elig = {q['sid'] for b, q in pos.items()
                if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
        rows = merge(m, po_cf if cf else po_main, pj_cf if cf else pj_main, cf)
        if not rows: continue
        pr_of = {r['sid']: (predicate_of(qmap.get(r['sid'], '')) or '') for r in rows}

        gates = {
            'A_all': lambda r: True,
            'A_reject': lambda r: True,
            'A_reject_opp': lambda r: pr_of.get(r['sid'], '') in OPPOSED,
        }
        ksup = build_gated(rows, elig, target, lambda r: False)
        per['support'][m] = measure(rows, ksup, elig, pos, neg)
        per['support'][m]['n_c'] = len(elig)
        for pol in ('A_all', 'A_reject', 'A_reject_opp'):
            mono = pol.startswith('A_reject')
            k = build_gated(rows, elig, target, gates[pol], monotone=mono)
            per[pol][m] = measure(rows, k, elig, pos, neg)
            per[pol][m]['n_c'] = len(elig)
        print('[ok] %s' % m, flush=True)

    common = sorted(per['support'])

    print('\n' + '=' * 100)
    print('Per-type FGR (pooled %d models, target %.2f)' % (len(common), target))
    print('=' * 100)
    print('%-16s' % 'policy' + ''.join('%12s' % h for h in HT4) + '%10s%12s' % ('ALL', 'CorrectRet'))
    for pol in POL:
        row = ''.join('%11.2f%%' % (st.mean([per[pol][m]['by_ht'][h] for m in common]) * 100)
                      for h in HT4)
        print('%-16s%s%9.2f%%%12.4f'
              % (pol, row,
                 st.mean([per[pol][m]['fgr_all'] for m in common]) * 100,
                 st.mean([per[pol][m]['correct_retain'] for m in common])))

    print('\n' + '=' * 100)
    print('vs support: paired model-level bootstrap (n=%d)' % len(common))
    print('=' * 100)
    out = {}
    for pol in ('A_all', 'A_reject', 'A_reject_opp'):
        print('\n[%s]' % pol)
        out[pol] = {}
        for h in HT4:
            dd = [(per[pol][m]['by_ht'][h] - per['support'][m]['by_ht'][h]) * 100
                  for m in common]
            mu, lo, hi = E.model_level_paired(dd)
            sig = (lo > 0 or hi < 0)
            print('   %-16s FGR %+7.3fpp CI[%+7.3f,%+7.3f] %-9s down %d/%d'
                  % (h, mu, lo, hi, 'SIG' if sig else 'ns',
                     sum(1 for v in dd if v < 0), len(dd)))
            out[pol]['fgr_' + h] = dict(mean=mu, ci=[lo, hi], significant=sig)
        for lab, g in (('ALL FGR', lambda d: d['fgr_all'] * 100),
                       ('relation CBR', lambda d: d['cbr'][3] * 100),
                       ('attribute CBR', lambda d: d['cbr'][2] * 100),
                       ('CorrectRetain', lambda d: d['correct_retain'])):
            dd = [g(per[pol][m]) - g(per['support'][m]) for m in common]
            mu, lo, hi = E.model_level_paired(dd)
            sig = (lo > 0 or hi < 0)
            print('   %-16s     %+7.3f   CI[%+7.3f,%+7.3f] %-9s down %d/%d'
                  % (lab, mu, lo, hi, 'SIG' if sig else 'ns',
                     sum(1 for v in dd if v < 0), len(dd)))
            out[pol][lab] = dict(mean=mu, ci=[lo, hi], significant=sig)

    json.dump(dict(target=target, models=common, per=per, stats=out),
              open(os.path.join(HERE, 'final_policy.json'), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote final_policy.json')

if __name__ == '__main__':
    main()
