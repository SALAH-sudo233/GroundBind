#!/usr/bin/env python3
"""Fairness + full line-to-line comparison for the three-line CBR result.

Two gaps in eval_cbr_paper_aligned.py that must be closed before any line is
called better:

1. It only printed R1 - OMNI. JEVHEAD - OMNI is the comparison that matters, since
   JEVHEAD posts the lowest CBR.
2. It reported no positive-side cost. A line that simply rejects more will show a
   lower CBR on every type, which is not an improvement. The project rule is
   explicit: match the positive-keep rate before comparing rejection quality,
   otherwise a near-always-reject arm wins by construction.

So this recomputes, under the SAME paper denominator and cross-fitting as block
[2] of that script (fold A decided by the fold-B head and vice versa, thresholds
never fitted on the rows they score):

  pos_keep   kept positives / 500          (all positives, refusals count as lost)
  pos_mIoU   sum(IoU * keep) / 500         (rejected/no-box = 0, never dropped)
  FGR        negatives still carrying a box / 2000
  CBR        as defined in the paper (theta=0.5, tau=0.8)

All four are printed per line so the CBR numbers can be read against their cost.
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


def main():
    canon = json.load(open(os.path.join(HERE, 'canon_roots_paper.json')))
    models = sorted(canon)
    lines = {ln: A.load_line(ln, models) for ln in A.LINES}
    common = [m for m in models if all(lines[ln].get(m) for ln in A.LINES)]
    boxes = {m: A.load_boxes(m, canon[m]) for m in models}
    print('models: %d   lines: %s\n' % (len(common), ', '.join(A.LINES)))

    per = {ln: {} for ln in A.LINES}
    for ln in A.LINES:
        for m in common:
            rows = lines[ln][m]
            fA = [r for r in rows if r['fold'] == 0]
            fB = [r for r in rows if r['fold'] == 1]
            kA1, kA2 = A.fit_on(fB, 0.95)     # decide fold A with the fold-B head
            kB1, kB2 = A.fit_on(fA, 0.95)     # decide fold B with the fold-A head
            keep = {}
            for r in rows:
                k1, k2 = (kA1, kA2) if r['fold'] == 0 else (kB1, kB2)
                keep[r['sid']] = (k1(r), k2(r))

            pos = [r for r in rows if r['is_pos']]
            neg = [r for r in rows if not r['is_pos']]
            res = {}
            for arm, idx in (('B0', None), ('B1', 0), ('A2', 1)):
                def kept(r):
                    if not r['drew']:
                        return False
                    if idx is None:
                        return True
                    kv = keep.get(r['sid'])
                    return True if kv is None else kv[idx]
                res[arm] = dict(
                    pos_keep=sum(1 for r in pos if kept(r)) / 500.0,
                    pos_miou=sum(r['iou'] for r in pos if kept(r)) / 500.0,
                    fgr=sum(1 for r in neg if kept(r)) / 2000.0)
            # CBR on the frozen eligibility set
            p, n = boxes[m]
            elig = [b for b, q in p.items()
                    if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5]
            for arm, idx in (('B0', None), ('B1', 0), ('A2', 1)):
                res[arm]['cbr'] = A.cbr(elig, p, n, keep, idx)
            res['n_c'] = len(elig)
            per[ln][m] = res

    # ---------- cost table
    print('=' * 100)
    print('公平性：各线实际达到的正例代价（论文分母，交叉拟合，pos_keep 目标 0.95）')
    print('=' * 100)
    print('%-10s%-5s%11s%11s%10s%11s' %
          ('line', 'arm', 'pos_keep', 'pos_mIoU', 'FGR', 'rel_CBR'))
    for ln in A.LINES:
        for arm in ('B0', 'B1', 'A2'):
            print('%-10s%-5s%10.3f%11.4f%9.2f%%%10.1f%%' % (
                ln.upper() if arm == 'B0' else '', arm,
                st.mean([per[ln][m][arm]['pos_keep'] for m in common]),
                st.mean([per[ln][m][arm]['pos_miou'] for m in common]),
                st.mean([per[ln][m][arm]['fgr'] for m in common]) * 100,
                st.mean([per[ln][m][arm]['cbr'][3] for m in common]) * 100))
    print('\nB0 的 pos_keep<1 是上游本身未出框，不是缓释造成的。')

    # ---------- every line pair, on CBR and on the positive cost
    print('\n' + '=' * 100)
    print('线间配对检验（模型级 n=13 等权）：CBR 差 + 同时核验正例代价差')
    print('=' * 100)
    pairs = [('jevhead', 'omni'), ('jevhead', 'r1'), ('r1', 'omni')]
    out = {}
    for x, y in pairs:
        print('  %s − %s' % (x.upper(), y.upper()))
        for arm in ('B1', 'A2'):
            for i, ht in enumerate(HT4):
                d = [per[x][m][arm]['cbr'][i] - per[y][m][arm]['cbr'][i]
                     for m in common]
                mu, lo, hi = E.model_level_paired(d)
                print('    %-3s %-15s CBR %+7.2fpp CI[%+6.2f,%+6.2f] %s'
                      % (arm, ht, mu * 100, lo * 100, hi * 100,
                         '显著' if (lo > 0 or hi < 0) else '不显著'))
                out['%s-%s_%s_cbr_%s' % (x, y, arm, ht)] = \
                    dict(mean=mu, ci=[lo, hi])
            for key, lab in (('pos_keep', 'posKeep'), ('pos_miou', 'pos_mIoU')):
                d = [per[x][m][arm][key] - per[y][m][arm][key] for m in common]
                mu, lo, hi = E.model_level_paired(d)
                print('    %-3s %-15s     %+7.4f CI[%+7.4f,%+7.4f] %s'
                      % (arm, lab, mu, lo, hi,
                         '显著' if (lo > 0 or hi < 0) else '不显著'))
                out['%s-%s_%s_%s' % (x, y, arm, key)] = dict(mean=mu, ci=[lo, hi])

    # ---------- within-line probe effect, with its positive cost
    print('\n' + '=' * 100)
    print('各线内部 A2 − B1（探针净效应）+ 其正例代价')
    print('=' * 100)
    for ln in A.LINES:
        d = [per[ln][m]['A2']['cbr'][3] - per[ln][m]['B1']['cbr'][3]
             for m in common]
        mu, lo, hi = E.model_level_paired(d)
        dk = [per[ln][m]['A2']['pos_keep'] - per[ln][m]['B1']['pos_keep']
              for m in common]
        mk, lok, hik = E.model_level_paired(dk)
        dm = [per[ln][m]['A2']['pos_miou'] - per[ln][m]['B1']['pos_miou']
              for m in common]
        mm, lom, him = E.model_level_paired(dm)
        print('  %-8s relation CBR %+7.2fpp CI[%+6.2f,%+6.2f] %-6s 下降 %d/13'
              % (ln.upper(), mu * 100, lo * 100, hi * 100,
                 '显著' if (lo > 0 or hi < 0) else '不显著',
                 sum(1 for v in d if v < 0)))
        print('           posKeep %+.4f CI[%+.4f,%+.4f] %s | pos_mIoU %+.4f CI[%+.4f,%+.4f] %s'
              % (mk, lok, hik, '显著' if (lok > 0 or hik < 0) else 'n.s.',
                 mm, lom, him, '显著' if (lom > 0 or him < 0) else 'n.s.'))
        out['%s_A2-B1_relcbr' % ln] = dict(mean=mu, ci=[lo, hi])
        out['%s_A2-B1_poskeep' % ln] = dict(mean=mk, ci=[lok, hik])

    json.dump(dict(per_line_per_model=per, comparisons=out, models=common),
              open(os.path.join(HERE, 'cbr_fairness.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('\nwrote cbr_fairness.json')


if __name__ == '__main__':
    main()
