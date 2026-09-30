#!/usr/bin/env python3
"""Why do the three non-relation types get worse? Decompose the gate.

USER'S OBJECTION (correct): object / co_occurrence / attribute are mitigated by a
single Omni-7B posterior -- they never go through structured predicate-table
competitive scoring. So the full arm should be STRICTLY ADDITIVE on them and they
should not move at all.

WHAT I ACTUALLY RAN. eval_unified.py lets the probe head decide EVERY covered row.
With the legal text router, "covered" now spans all four types (the router fires on
"the red chair next to the person"), so those rows got handed to a head fitted
POOLED across types -- and the gap's fitted weight has opposite signs per type
(measured earlier: object +0.418 / cooc +0.334 / attr +0.283 / relation -0.673).
Pooling cancels the relation signal and perturbs the other three.

THIS SCRIPT separates policy from method:
  A_all      probe head decides every covered row            (what I reported)
  A_opposed  probe head decides only rows whose predicate has a genuine opposing
             configuration (behind / in front of / under / on top of / above /
             below) -- still legally routable from query text + frozen table
  A_reject   probe head may only turn keep -> reject, never rescue (monotone)

For each policy it reports per-type FGR against support-only, plus how many rows
changed and in which direction, so "worse" can be attributed to a concrete
mechanism instead of to relation contrast.
"""
import json
import os
import statistics as st
import sys
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
    """Cross-fitted composite. `gate(r)` decides whether the probe head MAY act.

    Rows the gate blocks keep the support decision bit-for-bit, so any movement on
    them would be a bug, not a policy effect.
    """
    keep = {}
    for tf in (0, 1):
        fit = [r for r in rows if r['fold'] != tf]
        tst = [r for r in rows if r['fold'] == tf]
        s_sup = head(fit, 'support')
        if s_sup is None:
            for r in tst:
                keep[r['sid']] = bool(r['drew'])
            continue
        ps = sorted(s_sup(r) for r in fit
                    if r['is_pos'] and r['sid'] in elig and usable(r, 'support'))
        k = int(round((1.0 - target) * len(ps)))
        t_sup = ps[max(0, min(k, len(ps) - 1))] if ps else float('-inf')

        def keep_sup(r):
            if not r['drew']:
                return False
            if not usable(r, 'support'):
                return True
            return s_sup(r) >= t_sup

        # probe head fitted ONLY on rows the gate admits
        adm = [r for r in fit if usable(r, 'full') and gate(r)]
        if len(adm) < 30 or len(set(r['label'] for r in adm)) < 2:
            for r in tst:
                keep[r['sid']] = keep_sup(r)
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

    POL = ('support', 'A_all', 'A_opposed', 'A_reject')
    per = {p: {} for p in POL}
    changed = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # pol->ht->[n,k2r,r2k]
    gap_auc = defaultdict(list)
    covrate = defaultdict(lambda: defaultdict(list))

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

        # per-type: how often does the router fire, and is the gap informative?
        for ht in HT4:
            sub = [r for r in rows if r['htype'] == ht]
            if sub:
                covrate[m][ht] = [sum(1 for r in sub if covered(r)), len(sub)]
        for ht in HT4:
            sub = [r for r in rows if r['htype'] == ht and covered(r)]
            pp = [r for r in rows if r['is_pos'] and covered(r)]
            ys = [0] * len(sub) + [1] * len(pp)
            ss = [-r['g_o'] for r in sub] + [-r['g_o'] for r in pp]
            if len(set(ys)) > 1 and len(sub) >= 20:
                gap_auc[ht].append(E.auroc(ss, ys))

        gates = {
            'A_all': lambda r: True,
            'A_opposed': lambda r: pr_of.get(r['sid'], '') in OPPOSED,
            'A_reject': lambda r: True,
        }
        ksup = build_gated(rows, elig, target, lambda r: False)
        per['support'][m] = measure(rows, ksup, elig, pos, neg)
        per['support'][m]['n_c'] = len(elig)
        for pol in ('A_all', 'A_opposed', 'A_reject'):
            k = build_gated(rows, elig, target, gates[pol],
                            monotone=(pol == 'A_reject'))
            per[pol][m] = measure(rows, k, elig, pos, neg)
            per[pol][m]['n_c'] = len(elig)
            for r in rows:
                a, b = ksup.get(r['sid']), k.get(r['sid'])
                if a != b:
                    ht = r['htype'] if r['htype'] in HT4 else 'positive'
                    changed[pol][ht][0] += 1
                    changed[pol][ht][1 if (a and not b) else 2] += 1
        print('[ok] %s' % m, flush=True)

    common = sorted(per['support'])

    print('\n' + '=' * 100)
    print('Router firing rate per type (text router, all four types)')
    print('=' * 100)
    for ht in HT4:
        num = sum(covrate[m][ht][0] for m in common if ht in covrate[m])
        den = sum(covrate[m][ht][1] for m in common if ht in covrate[m])
        print('  %-16s covered %6d / %6d  = %.3f' % (ht, num, den, num / den if den else 0))

    print('\ngap AUROC per type (negatives of that type vs positives, covered rows)')
    for ht in HT4:
        if gap_auc[ht]:
            print('  %-16s %.4f  (%d models)' % (ht, st.mean(gap_auc[ht]), len(gap_auc[ht])))

    print('\n' + '=' * 100)
    print('Per-type FGR by policy (pooled %d models, target %.2f)' % (len(common), target))
    print('=' * 100)
    print('%-12s' % 'policy' + ''.join('%12s' % h for h in HT4) + '%10s%12s' % ('ALL', 'CorrRetain'))
    for pol in POL:
        row = ''.join('%11.2f%%' % (st.mean([per[pol][m]['by_ht'][h] for m in common]) * 100)
                      for h in HT4)
        print('%-12s%s%9.2f%%%12.4f'
              % (pol, row,
                 st.mean([per[pol][m]['fgr_all'] for m in common]) * 100,
                 st.mean([per[pol][m]['correct_retain'] for m in common])))

    print('\n' + '=' * 100)
    print('vs support-only: paired model-level bootstrap (n=%d)' % len(common))
    print('=' * 100)
    out = {}
    for pol in ('A_all', 'A_opposed', 'A_reject'):
        print('\n[%s]  rows changed vs support: %s'
              % (pol, {h: changed[pol][h][0] for h in list(HT4) + ['positive']}))
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
        for lab, g in (('relation CBR', lambda d: d['cbr'][3] * 100),
                       ('attribute CBR', lambda d: d['cbr'][2] * 100),
                       ('CorrectRetain', lambda d: d['correct_retain']),
                       ('pos_mIoU', lambda d: d['pos_miou'])):
            dd = [g(per[pol][m]) - g(per['support'][m]) for m in common]
            mu, lo, hi = E.model_level_paired(dd)
            sig = (lo > 0 or hi < 0)
            print('   %-16s     %+7.3f   CI[%+7.3f,%+7.3f] %-9s down %d/%d'
                  % (lab, mu, lo, hi, 'SIG' if sig else 'ns',
                     sum(1 for v in dd if v < 0), len(dd)))
            out[pol][lab] = dict(mean=mu, ci=[lo, hi], significant=sig)

    json.dump(dict(target=target, models=common, per=per, stats=out,
                   changed={p: {h: changed[p][h] for h in changed[p]} for p in changed}),
              open(os.path.join(HERE, 'type_gate.json'), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote type_gate.json')


if __name__ == '__main__':
    main()
