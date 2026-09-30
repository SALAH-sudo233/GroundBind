#!/usr/bin/env python3
"""P1: is the gain visual-candidate binding, or a text prior / extra capacity?

Three controls, all on the unified corrected candidates, same folds, same protocol.

 [A] TEXT PRIOR (Blex / BlexO). Predicate log-likelihood ratio learnt on the fitting
     fold WITHOUT looking at the image or calling any model: score(q) = LLR of its
     predicate. If a free text prior matches the full arm, the "visual relation
     evidence" story is unsupported. BlexO adds the support scores to the prior, which
     is the fair strong-text baseline (my notes: Blex alone beat the Omni single arm
     on GT boxes, so omitting it would be self-serving).
     Predicate-balanced subset zeroes the prior by construction: equal-count
     down-sampling of pos/neg within each predicate, where Blex must land at ~0.5000.

 [B] EQUAL-BUDGET PARAPHRASE. The full arm calls the scorer more often than support.
     To separate "relation contrast" from "more forward passes + bigger head", compare
     the real rival gap against a same-shape feature built from an EQUIVALENT
     (semantics-preserving) rival, which costs the same number of calls and the same
     head capacity. Reuses already-collected equivalent probes.

 [C] WRONG-IMAGE (Aswap). Fixes the query, candidate box, routing and control flow and
     only swaps the probe's IMAGE input. If the score does not move, the verifier is
     not reading this candidate's image.

AUROC is rank-sum (ties averaged) -- the naive sort implementation inflates binary
signals to ~1.0.
"""
import argparse
import json
import os
import statistics as st
import sys
import random
from collections import defaultdict, Counter

PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
import eval_upstream as E
import eval_cbr_paper_aligned as A
import simple_relations as sr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_unified import COORDFIX_MODELS, load_probe, merge

PREDS = ('next to', 'behind', 'in front of', 'on top of', 'under', 'above',
         'below', 'beside', 'near', 'on')


def predicate_of(q):
    """Longest-match predicate from the frozen table; None if absent."""
    ql = (q or '').lower()
    best = None
    for p in PREDS:
        if p in ql and (best is None or len(p) > len(best)):
            best = p
    return best


def query_map(root, model):
    """sid -> query text, from the canonical DG records."""
    p = os.path.join(root, model, 'records.jsonl')
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if str(r.get('task', '')).lower() not in ('t2', 't2_vqa_grounding'):
            continue
        sid = r.get('sample_id') or r.get('base_sample_id')
        out[sid] = r.get('query') or ''
    return out


def blex_scores(fit_rows, tst_rows, qmap):
    """Predicate LLR learnt on fit rows; applied to tst. No image, no model."""
    cp, cn = Counter(), Counter()
    for r in fit_rows:
        pr = predicate_of(qmap.get(r['sid'], ''))
        if pr is None:
            continue
        (cp if r['label'] == 1 else cn)[pr] += 1
    import math

    def sc(r):
        pr = predicate_of(qmap.get(r['sid'], ''))
        if pr is None:
            return 0.0
        return math.log((cp[pr] + 1) / (cn[pr] + 1))
    return {r['sid']: sc(r) for r in tst_rows}


def crossfit_auroc(rows, qmap, featfn, need=None):
    """Cross-fitted logreg AUROC over relation-probe-covered rows."""
    ys, ss = [], []
    for tf in (0, 1):
        fit = [r for r in rows if r['fold'] != tf]
        tst = [r for r in rows if r['fold'] == tf]
        if need:
            fit = [r for r in fit if need(r)]
            tst = [r for r in tst if need(r)]
        if len(fit) < 30 or len(set(r['label'] for r in fit)) < 2 or not tst:
            continue
        bf = blex_scores(fit, fit, qmap)
        bt = blex_scores(fit, tst, qmap)
        X = [featfn(r, bf.get(r['sid'], 0.0)) for r in fit]
        y = [r['label'] for r in fit]
        f, _ = E.logreg(X, y)
        for r in tst:
            ss.append(f(featfn(r, bt.get(r['sid'], 0.0))))
            ys.append(r['label'])
    if not ys or len(set(ys)) < 2:
        return float('nan'), 0
    return E.auroc(ss, ys), len(ys)


def balanced_subset(rows, qmap, seed=0):
    """Equal-count pos/neg within each predicate -> text prior is exactly 0.5."""
    rnd = random.Random(seed)
    by = defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        pr = predicate_of(qmap.get(r['sid'], ''))
        if pr is None:
            continue
        by[pr][r['label']].append(r)
    out = []
    for pr, d in by.items():
        k = min(len(d[0]), len(d[1]))
        if k == 0:
            continue
        out += rnd.sample(d[0], k) + rnd.sample(d[1], k)
    return out


def load_equivalent(fn):
    """[B] equal-budget control: GT-box probe library keyed by sid (NO model field).

    Schema: {sid, probe_type, z, ...}. `original` is the reference reading and
    `equivalent_candidate` is a semantics-preserving rewrite, so
    |z_equiv - z_orig| costs the same extra call and the same head width as the
    competitive gap while carrying no relation contrast.
    """
    out = {}
    p = os.path.join(PROBE, fn)
    if not os.path.exists(p):
        return out
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get('error') or d.get('z') is None:
            continue
        out.setdefault(d['sid'], {}).setdefault(d.get('probe_type'), []).append(d['z'])
    return out


def load_swap(fn):
    """[C] wrong-image control: rows carry z_real and z_swap side by side."""
    out = {}
    p = os.path.join(PROBE, fn)
    if not os.path.exists(p):
        return out
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get('error') or d.get('z_real') is None or d.get('z_swap') is None:
            continue
        out.setdefault((d['model'], d['sid']), []).append(
            (float(d['z_real']), float(d['z_swap'])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-out', default='attribution.json')
    a = ap.parse_args()

    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    cfroots = json.load(open(os.path.join(PROBE, 'canon_roots_coordfix.json')))
    models = sorted(canon)

    po_main = load_probe('probe_textroute_all.jsonl', 'z')
    pj_main = load_probe('trprobe_jev_all.jsonl', 'z_head')
    po_cf = load_probe('probecf_omni.jsonl', 'z', COORDFIX_MODELS)
    pj_cf = load_probe('probecf_jev.jsonl', 'z_head', COORDFIX_MODELS)
    # [B]/[C] use their OWN schemas, not load_probe's (model,sid)->variant->z form.
    equiv = load_equivalent('actions_relation.jsonl')
    swap = load_swap('swap_all.jsonl')
    print('equivalent-probe sids=%d   wrong-image rows=%d' % (len(equiv), len(swap)))

    res = {}
    for m in models:
        cf = m in COORDFIX_MODELS
        root = cfroots[m] if cf else canon[m]
        qmap = query_map(root, m)
        rows = merge(m, po_cf if cf else po_main, pj_cf if cf else pj_main, cf)
        rows = [r for r in rows if r['drew'] and r['z_o'] is not None
                and r['z_j'] is not None]
        cov = [r for r in rows
               if r['zmax_o'] is not None and r['zmax_j'] is not None]
        if len(cov) < 60:
            print('[skip] %s: covered=%d' % (m, len(cov)))
            continue

        arms = {
            'support_z0':  lambda r, b: [r['z_o'], r['z_j']],
            'full_gap':    lambda r, b: [r['z_o'], r['z_j'], r['g_o'], r['g_j']],
            'blex_only':   lambda r, b: [b],
            'blexO':       lambda r, b: [r['z_o'], r['z_j'], b],
            'blexO_gap':   lambda r, b: [r['z_o'], r['z_j'], r['g_o'], r['g_j'], b],
        }
        d = {}
        for nm, fn in arms.items():
            auc, n = crossfit_auroc(cov, qmap, fn)
            d[nm] = dict(auroc=auc, n=n)

        bal = balanced_subset(cov, qmap)
        d['_balanced_n'] = len(bal)
        if len(bal) >= 60:
            for nm in ('support_z0', 'full_gap', 'blex_only'):
                auc, n = crossfit_auroc(bal, qmap, arms[nm])
                d['bal_' + nm] = dict(auroc=auc, n=n)
        res[m] = d
        print('[ok] %-16s cov=%4d bal=%4d  sup=%.4f full=%.4f blex=%.4f '
              'blexO=%.4f blexO+gap=%.4f'
              % (m, len(cov), len(bal), d['support_z0']['auroc'],
                 d['full_gap']['auroc'], d['blex_only']['auroc'],
                 d['blexO']['auroc'], d['blexO_gap']['auroc']), flush=True)

    common = sorted(res)
    print('\n' + '=' * 100)
    print('[A] TEXT-PRIOR CONTROL (cross-fitted AUROC on probe-covered rows)')
    print('=' * 100)
    print('%-26s%10s%10s' % ('arm', 'pooled', 'n_models'))
    for nm in ('blex_only', 'support_z0', 'blexO', 'full_gap', 'blexO_gap'):
        vals = [res[m][nm]['auroc'] for m in common
                if nm in res[m] and res[m][nm]['auroc'] == res[m][nm]['auroc']]
        print('%-26s%10.4f%10d' % (nm, st.mean(vals), len(vals)))

    print('\npaired tests (model-level bootstrap, n=%d)' % len(common))
    stats = {}
    for x, y, lab in (('full_gap', 'support_z0', 'full - support'),
                      ('full_gap', 'blex_only', 'full - text prior'),
                      ('blexO_gap', 'blexO', 'gap ON TOP of text prior (decisive)')):
        dd = [res[m][x]['auroc'] - res[m][y]['auroc'] for m in common]
        mu, lo, hi = E.model_level_paired(dd)
        sig = (lo > 0 or hi < 0)
        print('  %-38s %+8.4f CI[%+7.4f,%+7.4f] %-9s up %d/%d'
              % (lab, mu, lo, hi, 'SIG' if sig else 'ns',
                 sum(1 for v in dd if v > 0), len(dd)))
        stats[lab] = dict(mean=mu, ci=[lo, hi], significant=sig,
                          n_up=sum(1 for v in dd if v > 0), n=len(dd))

    bk = [m for m in common if 'bal_blex_only' in res[m]]
    if bk:
        print('\npredicate-BALANCED subset (text prior must be ~0.5000)')
        for nm in ('bal_blex_only', 'bal_support_z0', 'bal_full_gap'):
            vals = [res[m][nm]['auroc'] for m in bk
                    if res[m][nm]['auroc'] == res[m][nm]['auroc']]
            print('  %-26s %.4f  (n_models=%d)' % (nm, st.mean(vals), len(vals)))
        dd = [res[m]['bal_full_gap']['auroc'] - res[m]['bal_support_z0']['auroc']
              for m in bk]
        mu, lo, hi = E.model_level_paired(dd)
        print('  balanced full - support        %+.4f CI[%+.4f,%+.4f] %s up %d/%d'
              % (mu, lo, hi, 'SIG' if (lo > 0 or hi < 0) else 'ns',
                 sum(1 for v in dd if v > 0), len(dd)))
        stats['balanced_full_minus_support'] = dict(
            mean=mu, ci=[lo, hi], significant=(lo > 0 or hi < 0),
            n_up=sum(1 for v in dd if v > 0), n=len(dd))

    # ---------- [B] equal-budget paraphrase control ----------
    print('\n' + '=' * 100)
    print('[B] EQUAL-BUDGET CONTROL: competitive gap vs semantics-preserving rewrite')
    print('=' * 100)
    if equiv:
        ys, comp, eqv = [], [], []
        for sid, d in equiv.items():
            if 'original' not in d:
                continue
            z0 = st.mean(d['original'])
            lab = 1 if sid.endswith('__rel') is False else None
            # polarity is encoded per row; recover from the file directly
            ys.append(sid)
            comp.append(max(d.get('competitive', [z0])) - z0)
            eqv.append(max(abs(v - z0) for v in d.get('equivalent_candidate', [z0])))
        # labels come from the file's own `label` field
        labs = {}
        for line in open(os.path.join(PROBE, 'actions_relation.jsonl'), encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            dd = json.loads(line)
            if dd.get('label') is not None:
                labs[dd['sid']] = int(dd['label'])
        keep = [i for i, sid in enumerate(ys) if sid in labs]
        y = [labs[ys[i]] for i in keep]
        if len(set(y)) > 1:
            ac = E.auroc([-comp[i] for i in keep], y)
            ae = E.auroc([-eqv[i] for i in keep], y)
            print('  n=%d  competitive gap AUROC=%.4f   equivalent-rewrite AUROC=%.4f'
                  % (len(keep), ac, ae))
            print('  -> %s' % ('relation contrast carries the signal, not the extra call'
                               if ac > ae + 0.02 else
                               'NOT separable on this subset; do not claim contrast-specific gain'))
            stats['equal_budget'] = dict(n=len(keep), competitive_auroc=ac,
                                         equivalent_auroc=ae)
    else:
        print('  actions_relation.jsonl absent -> control not run')

    # ---------- [C] wrong-image control ----------
    print('\n' + '=' * 100)
    print('[C] WRONG-IMAGE CONTROL: same query/box/routing, only the probe image swapped')
    print('=' * 100)
    if swap:
        per_model_delta = {}
        for (m, sid), pairs in swap.items():
            for zr, zs in pairs:
                per_model_delta.setdefault(m, []).append(zr - zs)
        allr = [zr for v in swap.values() for zr, _ in v]
        alls = [zs for v in swap.values() for _, zs in v]
        print('  rows=%d  mean z(real image)=%+.3f  mean z(wrong image)=%+.3f  diff=%+.3f'
              % (len(allr), st.mean(allr), st.mean(alls),
                 st.mean(allr) - st.mean(alls)))
        dd = [st.mean(v) for v in per_model_delta.values()]
        mu, lo, hi = E.model_level_paired(dd)
        print('  per-model mean(real - wrong) %+.3f CI[%+.3f,%+.3f] %s  (n=%d models)'
              % (mu, lo, hi, 'SIG' if (lo > 0 or hi < 0) else 'ns', len(dd)))
        stats['wrong_image'] = dict(n_rows=len(allr), mean_real=st.mean(allr),
                                    mean_swap=st.mean(alls), delta=mu, ci=[lo, hi],
                                    significant=(lo > 0 or hi < 0))
    else:
        print('  swap_all.jsonl absent -> control not run')

    json.dump(dict(models=common, per=res, stats=stats),
              open(os.path.join(HERE, a.json_out), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote %s' % a.json_out)


if __name__ == '__main__':
    main()
