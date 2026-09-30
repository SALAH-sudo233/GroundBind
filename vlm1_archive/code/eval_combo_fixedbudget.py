#!/usr/bin/env python3
"""Budget-correct combo evaluation: calibrate BOTH arms on the SAME positives.

THE DEFECT THIS FIXES. In eval_combo.py / eval_coordfix_combo.py the A2 ("full")
threshold was the (1-target) quantile of the probe head's scores over the positives
that head could see -- i.e. only probe-COVERED positives (55-59% of the eligible
ones). The B1 ("support") threshold used ALL drawn positives. Two different
calibration bases mean the two arms do not sit at the same positive budget: the
full arm systematically ended up more permissive (achieved CorrectRetain +0.008 to
+0.043 higher), which by itself lowers CBR and raises FGR. Frontier interpolation
patched this after the fact; this script removes the cause.

THE FIX. An A2 policy is a composite: covered rows are decided by the probe head,
uncovered rows keep the support decision. So calibrate the composite, not the probe
head alone. For a target retention t we need one scalar knob; we shift the probe
head's threshold until the COMPOSITE policy retains t of ALL eligible positives:

    keep_A2(r) = probe_head(r) >= tau      if covered(r)
                 keep_B1(r)                otherwise

    choose tau s.t.  #{eligible positives kept by keep_A2} / #eligible = t

Because uncovered positives contribute a fixed count (whatever B1 keeps), tau is
found by taking the appropriate quantile among COVERED eligible positives after
subtracting that fixed count. Both arms are then read at the same realised budget,
so a difference in CBR/FGR is a difference in discrimination, not in strictness.

Everything else is unchanged: paper denominators, image-level two-fold
cross-fitting, query-text routing, frozen eligibility.
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


def load_probe(path, key):
    out = defaultdict(dict)
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        return out
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get('error') or d.get(key) is None:
            continue
        out[(d['model'], d['sid'])][d['variant']] = d[key]
    return out


def arm_rows(model, line, coordfix):
    """z0 rows for one verifier."""
    if line == 'omni':
        f = ('s5omni_omnicf_%s.jsonl' if coordfix else 's5omni_omni_%s.jsonl') % model
        spec = A.REPROV.get(model)
        if spec and not coordfix:
            f = spec['omni_arm'] % model
        p = os.path.join(S5, f)
        zk = 'z_omni'
        src = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    else:
        p = os.path.join(HERE, ('jevheadcf_%s.jsonl' % model) if coordfix
                         else 'jevhead_all.jsonl')
        zk = 'z_head'
        src = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
        src = [r for r in src if (r.get('model') or model) == model]
    return {r['sid']: (r, zk) for r in src}


def merge(model, po, pj, coordfix):
    mo = arm_rows(model, 'omni', coordfix)
    mj = arm_rows(model, 'jev', coordfix)
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
    """Cross-fitted keep map whose realised retention targets `target` on ALL
    eligible positives -- identical budget definition for both arms."""
    keep = {}
    for tf in (0, 1):
        fit = [r for r in rows if r['fold'] != tf]
        tst = [r for r in rows if r['fold'] == tf]
        s_sup = head(fit, 'support')
        if s_sup is None:
            for r in tst:
                keep[r['sid']] = bool(r['drew'])
            continue

        # support threshold: quantile over ALL eligible positives it can score
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

        # composite calibration on the FITTING fold:
        # eligible positives split into covered / uncovered; uncovered keep the
        # support decision (a fixed count), so the probe threshold only has to
        # supply the remainder of the budget.
        ep = [r for r in fit if r['is_pos'] and r['sid'] in elig and r['drew']]
        cov = [r for r in ep if usable(r, 'full')]
        unc = [r for r in ep if not usable(r, 'full')]
        fixed_keep = sum(1 for r in unc if keep_sup(r))
        want = int(round(target * len(ep)))          # total positives to retain
        need = want - fixed_keep                     # must come from covered ones
        need = max(0, min(need, len(cov)))
        sc = sorted((s_full(r) for r in cov), reverse=True)
        # keep the `need` highest-scoring covered positives -> threshold = need-th
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
        cbr=A.cbr(list(elig), pos, neg, km, 0))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--canon', default='canon_roots_paper.json')
    ap.add_argument('--coordfix', action='store_true')
    ap.add_argument('--omni-probes', default='probe_textroute_all.jsonl')
    ap.add_argument('--jev-probes', default='trprobe_jev_all.jsonl')
    ap.add_argument('--target', type=float, default=0.90)
    ap.add_argument('--json-out', default='combo_fixedbudget.json')
    a = ap.parse_args()

    canon = json.load(open(os.path.join(HERE, a.canon)))
    models = sorted(canon)
    po = load_probe(a.omni_probes, 'z')
    pj = load_probe(a.jev_probes, 'z_head')

    per = {}
    for m in models:
        pos, neg = A.load_boxes(m, canon[m])
        elig = {q['sid'] for b, q in pos.items()
                if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
        rows = merge(m, po, pj, a.coordfix)
        if not rows:
            print('[skip] %s: no rows' % m)
            continue
        per[m] = {}
        for mode in ('support', 'full'):
            k = build(rows, elig, a.target, mode)
            per[m][mode] = measure(rows, k, elig, pos, neg)
        per[m]['n_c'] = len(elig)
    common = sorted(per)

    print('=' * 100)
    print('预算对齐的组合框架（两臂阈值都按【全部合格正例】校准，target=%.2f）' % a.target)
    print('=' * 100)
    print('%-16s%6s%-10s%11s%9s%9s%9s%9s%9s' %
          ('model', 'n_c', 'arm', 'CorrRetain', 'posmIoU', 'FGR', 'relFGR',
           'relCBR', 'attrCBR'))
    for m in common:
        for mode in ('support', 'full'):
            d = per[m][mode]
            print('%-16s%6s%-10s%10.3f%10.4f%8.2f%%%8.2f%%%8.1f%%%8.1f%%' % (
                m if mode == 'support' else '',
                per[m]['n_c'] if mode == 'support' else '',
                mode, d['correct_retain'], d['pos_miou'], d['fgr_all'] * 100,
                d['by_ht']['relation'] * 100, d['cbr'][3] * 100, d['cbr'][2] * 100))

    if len(common) > 2:
        print('\n池化（%d 模型等权）' % len(common))
        for mode in ('support', 'full'):
            print('  %-8s CorrRetain=%.3f mIoU=%.4f FGR=%.2f%% relFGR=%.2f%% '
                  'relCBR=%.1f%% attrCBR=%.1f%%'
                  % (mode,
                     st.mean([per[m][mode]['correct_retain'] for m in common]),
                     st.mean([per[m][mode]['pos_miou'] for m in common]),
                     st.mean([per[m][mode]['fgr_all'] for m in common]) * 100,
                     st.mean([per[m][mode]['by_ht']['relation'] for m in common]) * 100,
                     st.mean([per[m][mode]['cbr'][3] for m in common]) * 100,
                     st.mean([per[m][mode]['cbr'][2] for m in common]) * 100))
        print('\n配对检验 full − support（模型级 bootstrap n=%d）' % len(common))
        for f, nm in ((lambda d: d['correct_retain'], 'CorrectRetain'),
                      (lambda d: d['pos_miou'], 'pos_mIoU'),
                      (lambda d: d['fgr_all'] * 100, 'ALL FGR (pp)'),
                      (lambda d: d['by_ht']['relation'] * 100, 'rel FGR (pp)'),
                      (lambda d: d['cbr'][3] * 100, 'rel CBR (pp)'),
                      (lambda d: d['cbr'][2] * 100, 'attr CBR (pp)')):
            dd = [f(per[m]['full']) - f(per[m]['support']) for m in common]
            mu, lo, hi = E.model_level_paired(dd)
            print('  %-16s %+8.3f CI[%+7.3f,%+7.3f] %-6s 降 %d/%d'
                  % (nm, mu, lo, hi, '显著' if (lo > 0 or hi < 0) else '不显著',
                     sum(1 for v in dd if v < 0), len(dd)))
    else:
        print('\n逐模型 full − support（n=%d，不做显著性判断）' % len(common))
        for m in common:
            s_, f_ = per[m]['support'], per[m]['full']
            print('  %-14s relCBR %+6.2fpp  relFGR %+6.2fpp  ALLFGR %+6.2fpp  '
                  'CorrRetain %+.3f  mIoU %+.4f'
                  % (m, (f_['cbr'][3] - s_['cbr'][3]) * 100,
                     (f_['by_ht']['relation'] - s_['by_ht']['relation']) * 100,
                     (f_['fgr_all'] - s_['fgr_all']) * 100,
                     f_['correct_retain'] - s_['correct_retain'],
                     f_['pos_miou'] - s_['pos_miou']))

    json.dump(dict(target=a.target, coordfix=a.coordfix, models=common, per=per),
              open(os.path.join(HERE, a.json_out), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote %s' % a.json_out)


if __name__ == '__main__':
    main()
