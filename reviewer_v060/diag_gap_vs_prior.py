#!/usr/bin/env python3
"""Why does the relation gap add nothing once a predicate prior is present?

Two competing explanations for blexO+gap == blexO:
  (H1) the gap is redundant WITH the predicate prior -- its information is
       recoverable from predicate identity, so a prior that already encodes
       "behind is usually false" absorbs it;
  (H2) the gap is uninformative in general (contradicted by full>support 13/13
       and by the wrong-image control).

Discriminating measurement: regress the gap on predicate identity alone and see
how much of its variance predicate identity explains. If R^2 is high AND the
gap's residual (gap minus its predicate mean) has little standalone AUROC, H1
holds; if the residual keeps AUROC, the redundancy story fails.

Also reports gap AUROC WITHIN each predicate, where predicate identity is
constant and therefore cannot carry any signal.
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
from eval_unified import COORDFIX_MODELS, load_probe, merge
from eval_attribution import predicate_of, query_map


def main():
    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    cfroots = json.load(open(os.path.join(PROBE, 'canon_roots_coordfix.json')))
    models = sorted(canon)

    po_main = load_probe('probe_textroute_all.jsonl', 'z')
    pj_main = load_probe('trprobe_jev_all.jsonl', 'z_head')
    po_cf = load_probe('probecf_omni.jsonl', 'z', COORDFIX_MODELS)
    pj_cf = load_probe('probecf_jev.jsonl', 'z_head', COORDFIX_MODELS)

    pooled = defaultdict(lambda: defaultdict(list))   # predicate -> label -> gaps
    per_model = {}
    for m in models:
        cf = m in COORDFIX_MODELS
        qmap = query_map(cfroots[m] if cf else canon[m], m)
        rows = merge(m, po_cf if cf else po_main, pj_cf if cf else pj_main, cf)
        rows = [r for r in rows if r['drew'] and r['zmax_o'] is not None
                and r['zmax_j'] is not None]
        if len(rows) < 60:
            continue
        # use the omni gap (same conclusion holds for jev; reported separately)
        recs = []
        for r in rows:
            pr = predicate_of(qmap.get(r['sid'], ''))
            if pr is None:
                continue
            recs.append((pr, r['label'], r['g_o'], r['g_j']))
            pooled[pr][r['label']].append(r['g_o'])
        if not recs:
            continue

        # variance of gap explained by predicate identity (one-way ANOVA R^2)
        g = [x[2] for x in recs]
        gm = st.mean(g)
        sst = sum((v - gm) ** 2 for v in g)
        bygrp = defaultdict(list)
        for pr, lab, go, gj in recs:
            bygrp[pr].append(go)
        ssb = sum(len(v) * (st.mean(v) - gm) ** 2 for v in bygrp.values())
        r2 = ssb / sst if sst > 0 else float('nan')

        # residual gap = gap - predicate mean  (predicate info removed)
        pm = {pr: st.mean(v) for pr, v in bygrp.items()}
        resid = [(x[2] - pm[x[0]]) for x in recs]
        y = [x[1] for x in recs]
        auc_raw = E.auroc([-v for v in g], y) if len(set(y)) > 1 else float('nan')
        auc_res = E.auroc([-v for v in resid], y) if len(set(y)) > 1 else float('nan')

        # within-predicate AUROC (predicate identity constant by construction)
        wa = []
        for pr, vals in bygrp.items():
            sub = [(x[2], x[1]) for x in recs if x[0] == pr]
            ys = [b for _, b in sub]
            if len(set(ys)) < 2 or len(sub) < 30:
                continue
            wa.append((pr, E.auroc([-a for a, _ in sub], ys), len(sub)))
        per_model[m] = dict(n=len(recs), r2_predicate=r2, auroc_gap=auc_raw,
                            auroc_residual=auc_res,
                            within=[(p, round(a, 4), n) for p, a, n in wa])
        print('[ok] %-16s n=%4d  R2(pred)=%.3f  AUROC gap=%.4f  residual=%.4f'
              % (m, len(recs), r2, auc_raw, auc_res), flush=True)

    common = sorted(per_model)
    print('\n' + '=' * 96)
    print('Is the relation gap redundant with predicate identity?')
    print('=' * 96)
    print('  pooled R^2 of gap on predicate identity   %.4f'
          % st.mean([per_model[m]['r2_predicate'] for m in common]))
    print('  pooled AUROC, raw gap                     %.4f'
          % st.mean([per_model[m]['auroc_gap'] for m in common]))
    print('  pooled AUROC, gap MINUS predicate mean    %.4f'
          % st.mean([per_model[m]['auroc_residual'] for m in common]))
    dd = [per_model[m]['auroc_residual'] - per_model[m]['auroc_gap'] for m in common]
    mu, lo, hi = E.model_level_paired(dd)
    print('  residual - raw   %+.4f CI[%+.4f,%+.4f] %s'
          % (mu, lo, hi, 'SIG' if (lo > 0 or hi < 0) else 'ns'))

    print('\nwithin-predicate gap AUROC (predicate identity constant):')
    agg = defaultdict(list)
    for m in common:
        for pr, a, n in per_model[m]['within']:
            agg[pr].append(a)
    for pr in sorted(agg, key=lambda p: -len(agg[p])):
        print('  %-14s %.4f   (%d models)' % (pr, st.mean(agg[pr]), len(agg[pr])))

    print('\npooled gap by predicate and label (mean omni gap; higher = rival wins):')
    print('  %-14s%12s%12s%10s%10s' % ('predicate', 'neg mean', 'pos mean', 'n_neg', 'n_pos'))
    for pr in sorted(pooled, key=lambda p: -(len(pooled[p][0]) + len(pooled[p][1]))):
        n0, n1 = pooled[pr][0], pooled[pr][1]
        if not n0 and not n1:
            continue
        print('  %-14s%12s%12s%10d%10d'
              % (pr,
                 '%.3f' % st.mean(n0) if n0 else '-',
                 '%.3f' % st.mean(n1) if n1 else '-',
                 len(n0), len(n1)))

    json.dump(dict(per_model=per_model), open(os.path.join(HERE, 'gap_vs_prior.json'), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote gap_vs_prior.json')


if __name__ == '__main__':
    main()
