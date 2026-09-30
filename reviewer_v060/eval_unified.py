#!/usr/bin/env python3
"""P0-1: ONE 13-model table on the coordinate-CORRECTED candidates.

THE DEFECT. `eval_combo_fixedbudget.py --coordfix` is a GLOBAL boolean, but the
corrected score files (`s5omni_omnicf_*`, `jevheadcf_*`, `probecf_*`) exist only
for UniVG-R1 and visual-rft. So the flag could produce either a 13-model table on
OLD coordinates (`main_fixedbudget_095.json`, coordfix=False) or a 2-model table on
corrected ones (`cf_fixedbudget_095.json`) -- never one consistent 13-model table.
That is exactly the reviewer's P0: Table 2 says UniVG-R1 localizes 254 positives,
Table 4 uses the nc=30 old-coordinate candidates.

THE FIX. Resolve the candidate source PER MODEL: corrected files for the two
models that have them, the paper's canonical run for the other eleven. Every model
is then read on its own canonical candidates exactly once.

Protocol is inherited unchanged from eval_combo_fixedbudget.py: paper denominators
(500 positives / 500 per negative type), image-level two-fold cross-fitting
(md5(image)%2, each fold decided by the other fold's head AND threshold), frozen
eligibility from the unfiltered positive prediction, and -- critically -- BOTH arms
calibrated to the same realised retention over ALL eligible positives, so a CBR/FGR
difference is discrimination and not strictness.

Routing is the label-free text router (`probe_textroute_all.jsonl`).

Outputs absolute values, not deltas: per model and arm we emit CorrectRetain,
pos_mIoU, FGR overall and per type, all four CBR values, and n_c.
"""
import argparse
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
HT4 = A.HT4
S5 = os.path.expanduser('~/SVD/grpo_verifier/s5grpo')

# Only these two have corrected-coordinate candidates + freshly collected scores.
COORDFIX_MODELS = ('UniVG-R1', 'visual-rft')


def load_probe(path, key, models=None):
    out = defaultdict(dict)
    p = path if os.path.isabs(path) else os.path.join(PROBE, path)
    if not os.path.exists(p):
        return out
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get('error') or d.get(key) is None:
            continue
        if models and d['model'] not in models:
            continue
        out[(d['model'], d['sid'])][d['variant']] = d[key]
    return out


def arm_rows(model, line, cf):
    """z0 rows for one verifier on this model's canonical candidates."""
    if line == 'omni':
        if cf:
            f = 's5omni_omnicf_%s.jsonl' % model
        else:
            spec = A.REPROV.get(model)
            f = (spec['omni_arm'] % model) if spec else ('s5omni_omni_%s.jsonl' % model)
        p = os.path.join(S5, f)
        zk = 'z_omni'
        src = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    else:
        p = os.path.join(PROBE, ('jevheadcf_%s.jsonl' % model) if cf else 'jevhead_all.jsonl')
        zk = 'z_head'
        src = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
        src = [r for r in src if (r.get('model') or model) == model]
    return {r['sid']: (r, zk) for r in src}


def merge(model, po, pj, cf):
    mo, mj = arm_rows(model, 'omni', cf), arm_rows(model, 'jev', cf)
    rows = []
    for sid, (r, ko) in mo.items():
        q = mj.get(sid)
        if q is None:
            continue
        qr, kj = q
        ht = r.get('htype')
        rvo = [v for v in po.get((model, sid), {}).values() if v is not None]
        rvj = [v for v in pj.get((model, sid), {}).values() if v is not None]
        zo, zj = r.get(ko), qr.get(kj)
        d = dict(model=model, sid=sid, htype=ht, is_pos=(ht == 'positive'),
                 label=1 if ht == 'positive' else 0,
                 iou=float(r.get('iou') or 0.0),
                 drew=bool(r.get('drew')) and bool(qr.get('drew')),
                 z_o=zo, z_j=zj,
                 zmax_o=(max(rvo) if rvo else None),
                 zmax_j=(max(rvj) if rvj else None),
                 fold=E.fold_of(sid))
        d['g_o'] = (d['zmax_o'] - zo) if (d['zmax_o'] is not None and zo is not None) else 0.0
        d['g_j'] = (d['zmax_j'] - zj) if (d['zmax_j'] is not None and zj is not None) else 0.0
        rows.append(d)
    return rows


def covered(r):
    return r['zmax_o'] is not None and r['zmax_j'] is not None


def feats(mode, r):
    return [r['z_o'], r['z_j']] if mode == 'support' \
        else [r['z_o'], r['z_j'], r['g_o'], r['g_j']]


def usable(r, mode):
    return (r['drew'] and r['z_o'] is not None and r['z_j'] is not None
            and (mode == 'support' or covered(r)))


def head(rows, mode):
    R = [r for r in rows if usable(r, mode)]
    if len(R) < 30 or len(set(r['label'] for r in R)) < 2:
        return None
    f, _ = E.logreg([feats(mode, r) for r in R], [r['label'] for r in R])
    return lambda r: f(feats(mode, r))


def build(rows, elig, target, mode):
    """Cross-fitted keep map; realised retention targets `target` on ALL
    eligible positives for BOTH arms (identical budget definition)."""
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

        if mode == 'support':
            for r in tst:
                keep[r['sid']] = keep_sup(r)
            continue

        s_full = head(fit, 'full')
        if s_full is None:
            for r in tst:
                keep[r['sid']] = keep_sup(r)
            continue

        ep = [r for r in fit if r['is_pos'] and r['sid'] in elig and r['drew']]
        cov = [r for r in ep if usable(r, 'full')]
        unc = [r for r in ep if not usable(r, 'full')]
        fixed_keep = sum(1 for r in unc if keep_sup(r))
        need = max(0, min(int(round(target * len(ep))) - fixed_keep, len(cov)))
        sc = sorted((s_full(r) for r in cov), reverse=True)
        t_full = sc[need - 1] if need > 0 else float('inf')

        def keep_full(r):
            if not r['drew']:
                return False
            if not usable(r, 'full'):
                return keep_sup(r)
            return s_full(r) >= t_full

        for r in tst:
            keep[r['sid']] = keep_full(r)
    return keep


def measure(rows, keep, elig, pos, neg):
    p = [r for r in rows if r['is_pos']]
    n = [r for r in rows if not r['is_pos']]
    by = {ht: sum(1 for r in n if r['htype'] == ht and keep.get(r['sid'])) / 500.0
          for ht in HT4}
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
        n_eligible=len(ce),
        cbr=A.cbr(list(elig), pos, neg, km, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=float, default=0.95)
    ap.add_argument('--json-out', default='unified_13models.json')
    a = ap.parse_args()

    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    cfroots = json.load(open(os.path.join(PROBE, 'canon_roots_coordfix.json')))
    models = sorted(canon)

    # probes: corrected files for the two cf models, text-routed file for the rest
    po_main = load_probe('probe_textroute_all.jsonl', 'z')
    pj_main = load_probe('trprobe_jev_all.jsonl', 'z_head')
    po_cf = load_probe('probecf_omni.jsonl', 'z', COORDFIX_MODELS)
    pj_cf = load_probe('probecf_jev.jsonl', 'z_head', COORDFIX_MODELS)

    per, src = {}, {}
    for m in models:
        cf = m in COORDFIX_MODELS
        root = cfroots[m] if cf else canon[m]
        pos, neg = A.load_boxes(m, root)
        elig = {q['sid'] for b, q in pos.items()
                if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
        rows = merge(m, po_cf if cf else po_main, pj_cf if cf else pj_main, cf)
        if not rows:
            print('[skip] %s: no rows' % m)
            continue
        per[m] = {}
        for mode in ('support', 'full'):
            per[m][mode] = measure(rows, build(rows, elig, a.target, mode),
                                   elig, pos, neg)
        per[m]['n_c'] = len(elig)
        src[m] = dict(coordfix=cf, root=root, n_rows=len(rows))
        print('[ok] %-16s coordfix=%-5s n_c=%3d rows=%d'
              % (m, cf, len(elig), len(rows)), flush=True)

    common = sorted(per)
    print('\n' + '=' * 108)
    print('UNIFIED 13-model table, corrected candidates, target CorrectRetain=%.2f' % a.target)
    print('=' * 108)
    print('%-16s%5s %-8s%11s%9s%8s%9s%9s%9s%9s' %
          ('model', 'n_c', 'arm', 'CorrRetain', 'posmIoU', 'FGR', 'relFGR',
           'objCBR', 'attrCBR', 'relCBR'))
    for m in common:
        for mode in ('support', 'full'):
            d = per[m][mode]
            print('%-16s%5s %-8s%10.3f%10.4f%7.2f%%%8.2f%%%8.1f%%%8.1f%%%8.1f%%' % (
                m if mode == 'support' else '',
                per[m]['n_c'] if mode == 'support' else '', mode,
                d['correct_retain'], d['pos_miou'], d['fgr_all'] * 100,
                d['by_ht']['relation'] * 100, d['cbr'][0] * 100,
                d['cbr'][2] * 100, d['cbr'][3] * 100))

    print('\npooled (%d models, equal weight)' % len(common))
    for mode in ('support', 'full'):
        f = lambda g: st.mean([g(per[m][mode]) for m in common])
        print('  %-8s CorrRetain=%.4f mIoU=%.4f FGR=%.2f%% relFGR=%.2f%% '
              'relCBR=%.2f%% attrCBR=%.2f%%'
              % (mode, f(lambda d: d['correct_retain']), f(lambda d: d['pos_miou']),
                 f(lambda d: d['fgr_all']) * 100,
                 f(lambda d: d['by_ht']['relation']) * 100,
                 f(lambda d: d['cbr'][3]) * 100, f(lambda d: d['cbr'][2]) * 100))

    print('\nfull - support (model-level paired bootstrap, n=%d)' % len(common))
    stats = {}
    for g, nm in ((lambda d: d['correct_retain'], 'CorrectRetain'),
                  (lambda d: d['pos_miou'], 'pos_mIoU'),
                  (lambda d: d['fgr_all'] * 100, 'ALL FGR (pp)'),
                  (lambda d: d['by_ht']['relation'] * 100, 'rel FGR (pp)'),
                  (lambda d: d['by_ht']['attribute'] * 100, 'attr FGR (pp)'),
                  (lambda d: d['by_ht']['object'] * 100, 'obj FGR (pp)'),
                  (lambda d: d['by_ht']['co_occurrence'] * 100, 'cooc FGR (pp)'),
                  (lambda d: d['cbr'][3] * 100, 'rel CBR (pp)'),
                  (lambda d: d['cbr'][2] * 100, 'attr CBR (pp)')):
        dd = [g(per[m]['full']) - g(per[m]['support']) for m in common]
        mu, lo, hi = E.model_level_paired(dd)
        sig = (lo > 0 or hi < 0)
        print('  %-16s %+8.3f CI[%+7.3f,%+7.3f] %-9s down %d/%d'
              % (nm, mu, lo, hi, 'SIG' if sig else 'ns',
                 sum(1 for v in dd if v < 0), len(dd)))
        stats[nm] = dict(mean=mu, ci=[lo, hi], significant=sig,
                         n_down=sum(1 for v in dd if v < 0), n=len(dd))

    json.dump(dict(target=a.target, models=common, sources=src,
                   per=per, stats=stats),
              open(os.path.join(HERE, a.json_out), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote %s' % a.json_out)


if __name__ == '__main__':
    main()
