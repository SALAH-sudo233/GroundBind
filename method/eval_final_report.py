#!/usr/bin/env python3
"""Integrated FGR + CBR evaluation for the final report (two lines, paper口径).

Scope decision: the GRPO-2B (R1) line is DROPPED. It was significantly worse than
the zero-training Omni line on all four CBR types at matched positive-keep, and its
probe effect was not significant (-0.32pp, 3/13). What remains needs at most one
lightly-trained component.

Lines kept:
  ZT   zero-training     OmniVerifier-7B (Apache-2.0), no training at all
  LT   light-training    jev_verifier_v2 = all-layer LoRA r=8 + a (1,2048) scalar
                         head + one temperature scalar; 1000 steps on the 2000-image
                         training pool (disjoint from the 500-image eval set)

Both metrics are computed under ONE protocol so they can appear in the same table:
  * paper denominators: FGR per type over ALL 500 negatives of that type, ALL-FGR
    over all 2000 negatives, pos_mIoU and pos_keep over ALL 500 positives
    (rejected / no-box count as zero and are never removed)
  * eligibility c_i and the positive reference box FROZEN pre-mitigation
  * cross-fitted thresholds: fold A rows decided by the fold-B-fitted head and vice
    versa, so no row is scored by a threshold fitted on itself
  * probes gated to the rows they actually cover; all other rows keep the
    single-arm decision byte-for-byte

It also re-derives the FGR numbers the earlier drafts reported, to confirm the
suspected halving: those tables evaluated FOLD B rows while dividing by the
FULL-set denominators (500 / 2000), which understates every FGR by ~2x. The
corrected values are the ones this script prints.
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
BOH = ('object', 'co_occurrence')
ROH = ('attribute', 'relation')
LINES = (('omni', 'ZT'), ('jevhead', 'LT'))


def metrics(rows, keep, idx):
    """FGR per type / BOH / ROH / ALL + positive-side cost, paper denominators."""
    pos = [r for r in rows if r['is_pos']]
    neg = [r for r in rows if not r['is_pos']]

    def kept(r):
        if not r['drew']:
            return False
        if idx is None:
            return True
        kv = keep.get(r['sid'])
        return True if kv is None else kv[idx]

    by = {}
    for ht in HT4:
        sub = [r for r in neg if r['htype'] == ht]
        by[ht] = sum(1 for r in sub if kept(r)) / 500.0
    return dict(
        fgr_all=sum(1 for r in neg if kept(r)) / 2000.0,
        boh=sum(by[h] for h in BOH) / 2.0,
        roh=sum(by[h] for h in ROH) / 2.0,
        by_ht=by,
        pos_keep=sum(1 for r in pos if kept(r)) / 500.0,
        pos_miou=sum(r['iou'] for r in pos if kept(r)) / 500.0,
        n_neg_rows=len(neg), n_pos_rows=len(pos))


def main():
    canon = json.load(open(os.path.join(HERE, 'canon_roots_paper.json')))
    models = sorted(canon)
    lines = {ln: A.load_line(ln, models) for ln, _ in LINES}
    common = [m for m in models if all(lines[ln].get(m) for ln, _ in LINES)]
    boxes = {m: A.load_boxes(m, canon[m]) for m in models}
    print('models=%d  lines=%s\n' % (len(common),
                                     ', '.join('%s(%s)' % (t, ln)
                                               for ln, t in LINES)))

    per = {ln: {} for ln, _ in LINES}
    halving = {}
    for ln, _ in LINES:
        for m in common:
            rows = lines[ln][m]
            fA = [r for r in rows if r['fold'] == 0]
            fB = [r for r in rows if r['fold'] == 1]
            kA1, kA2 = A.fit_on(fB, 0.95)
            kB1, kB2 = A.fit_on(fA, 0.95)
            keep = {}
            for r in rows:
                k1, k2 = (kA1, kA2) if r['fold'] == 0 else (kB1, kB2)
                keep[r['sid']] = (k1(r), k2(r))
            res = {}
            for arm, idx in (('B0', None), ('B1', 0), ('A2', 1)):
                res[arm] = metrics(rows, keep, idx)
            p, n = boxes[m]
            elig = [b for b, q in p.items()
                    if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5]
            for arm, idx in (('B0', None), ('B1', 0), ('A2', 1)):
                res[arm]['cbr'] = A.cbr(elig, p, n, keep, idx)
            res['n_c'] = len(elig)
            per[ln][m] = res
            # reproduce the old (halved) reading: fold B rows, full denominators
            if ln == 'omni':
                negB = [r for r in fB if not r['is_pos']]
                halving[m] = dict(
                    old_B0=sum(1 for r in negB if r['drew']) / 2000.0,
                    correct_B0=res['B0']['fgr_all'],
                    foldB_negrows=len(negB))

    # ---------- halving check
    print('=' * 100)
    print('校验：早先 FGR 表在 fold B 行上算、分母却按全集 2000 → 系统性减半')
    print('=' * 100)
    ob = st.mean([halving[m]['old_B0'] for m in common]) * 100
    cb = st.mean([halving[m]['correct_B0'] for m in common]) * 100
    nr = st.mean([halving[m]['foldB_negrows'] for m in common])
    print('  B0 ALL-FGR   旧读数(fold B 行 / 2000) = %.2f%%' % ob)
    print('               正确读数(全部 2000 负例)  = %.2f%%' % cb)
    print('  fold B 平均负例行数 = %.0f（约为 2000 的一半）→ 比值 %.2f'
          % (nr, cb / ob if ob else float('nan')))
    print('  故旧报告的 FGR 数值一律作废，本报告用正确口径。')

    # ---------- main integrated table
    print('\n' + '=' * 100)
    print('主表：FGR + CBR 同口径（论文分母，交叉拟合，pos_keep 目标 0.95，13 模型等权）')
    print('=' * 100)
    print('%-6s%-5s %8s%8s%8s%9s  %8s%9s  %9s%8s' %
          ('line', 'arm', 'FGR', 'BOH', 'ROH', 'relFGR',
           'posKeep', 'pos_mIoU', 'relCBR', 'attrCBR'))

    def mn(ln, arm, f):
        return st.mean([f(per[ln][m][arm]) for m in common])

    # B0 is identical for both lines (upstream, no verifier) -> print once
    print('%-6s%-5s %7.2f%%%7.2f%%%7.2f%%%8.2f%%  %8.3f%9.4f  %8.1f%%%7.1f%%' % (
        'B0', '--', mn('omni', 'B0', lambda r: r['fgr_all']) * 100,
        mn('omni', 'B0', lambda r: r['boh']) * 100,
        mn('omni', 'B0', lambda r: r['roh']) * 100,
        mn('omni', 'B0', lambda r: r['by_ht']['relation']) * 100,
        mn('omni', 'B0', lambda r: r['pos_keep']),
        mn('omni', 'B0', lambda r: r['pos_miou']),
        mn('omni', 'B0', lambda r: r['cbr'][3]) * 100,
        mn('omni', 'B0', lambda r: r['cbr'][2]) * 100))
    for ln, tag in LINES:
        for arm in ('B1', 'A2'):
            print('%-6s%-5s %7.2f%%%7.2f%%%7.2f%%%8.2f%%  %8.3f%9.4f  %8.1f%%%7.1f%%' % (
                tag if arm == 'B1' else '', arm,
                mn(ln, arm, lambda r: r['fgr_all']) * 100,
                mn(ln, arm, lambda r: r['boh']) * 100,
                mn(ln, arm, lambda r: r['roh']) * 100,
                mn(ln, arm, lambda r: r['by_ht']['relation']) * 100,
                mn(ln, arm, lambda r: r['pos_keep']),
                mn(ln, arm, lambda r: r['pos_miou']),
                mn(ln, arm, lambda r: r['cbr'][3]) * 100,
                mn(ln, arm, lambda r: r['cbr'][2]) * 100))

    # ---------- per-htype FGR detail
    print('\n逐类 FGR（分母=该类 500 个负例）')
    print('%-6s%-5s%s' % ('line', 'arm', ''.join('%13s' % h[:11] for h in HT4)))
    print('%-6s%-5s%s' % ('B0', '--', ''.join(
        '%12.2f%%' % (mn('omni', 'B0', lambda r, h=h: r['by_ht'][h]) * 100)
        for h in HT4)))
    for ln, tag in LINES:
        for arm in ('B1', 'A2'):
            print('%-6s%-5s%s' % (tag if arm == 'B1' else '', arm, ''.join(
                '%12.2f%%' % (mn(ln, arm, lambda r, h=h: r['by_ht'][h]) * 100)
                for h in HT4)))

    # ---------- statistics
    print('\n' + '=' * 100)
    print('统计（模型级配对 bootstrap，n=13 等权）')
    print('=' * 100)
    out = {}

    def paired(ln, arm_a, arm_b, getter, label, indent='  '):
        d = [getter(per[ln][m][arm_a]) - getter(per[ln][m][arm_b])
             for m in common]
        mu, lo, hi = E.model_level_paired(d)
        sig = '显著' if (lo > 0 or hi < 0) else '不显著'
        print('%s%-26s %+8.3f  CI[%+7.3f,%+7.3f] %-6s  下降 %d/13'
              % (indent, label, mu, lo, hi, sig, sum(1 for v in d if v < 0)))
        return dict(mean=mu, ci=[lo, hi], n_down=sum(1 for v in d if v < 0))

    for ln, tag in LINES:
        print('%s 线  A2 − B1（探针净效应）' % tag)
        out['%s_A2-B1_relFGR' % tag] = paired(
            ln, 'A2', 'B1', lambda r: r['by_ht']['relation'] * 100, 'relation FGR (pp)')
        out['%s_A2-B1_FGR' % tag] = paired(
            ln, 'A2', 'B1', lambda r: r['fgr_all'] * 100, 'ALL FGR (pp)')
        out['%s_A2-B1_relCBR' % tag] = paired(
            ln, 'A2', 'B1', lambda r: r['cbr'][3] * 100, 'relation CBR (pp)')
        out['%s_A2-B1_posKeep' % tag] = paired(
            ln, 'A2', 'B1', lambda r: r['pos_keep'], 'posKeep')
        out['%s_A2-B1_mIoU' % tag] = paired(
            ln, 'A2', 'B1', lambda r: r['pos_miou'], 'pos_mIoU')

    print('LT − ZT（同 pos_keep 目标，A2 臂）')
    for key, lab in ((lambda r: r['by_ht']['relation'] * 100, 'relation FGR (pp)'),
                     (lambda r: r['fgr_all'] * 100, 'ALL FGR (pp)'),
                     (lambda r: r['cbr'][3] * 100, 'relation CBR (pp)'),
                     (lambda r: r['cbr'][2] * 100, 'attribute CBR (pp)'),
                     (lambda r: r['pos_keep'], 'posKeep'),
                     (lambda r: r['pos_miou'], 'pos_mIoU')):
        d = [key(per['jevhead'][m]['A2']) - key(per['omni'][m]['A2'])
             for m in common]
        mu, lo, hi = E.model_level_paired(d)
        print('  %-26s %+8.3f  CI[%+7.3f,%+7.3f] %s'
              % (lab, mu, lo, hi, '显著' if (lo > 0 or hi < 0) else '不显著'))
        out['LT-ZT_A2_%s' % lab.split()[0]] = dict(mean=mu, ci=[lo, hi])

    # ---------- per-model table for the two headline metrics
    print('\n' + '=' * 100)
    print('逐模型（A2 臂）：relation FGR 与 relation CBR，B1 → A2')
    print('=' * 100)
    print('%-16s%5s | %-26s | %-26s' % ('model', 'n_c',
                                        'ZT  relFGR / relCBR',
                                        'LT  relFGR / relCBR'))
    for m in sorted(common, key=lambda x: -per['omni'][x]['n_c']):
        z, l = per['omni'][m], per['jevhead'][m]
        print('%-16s%5d | %5.1f→%5.1f  %5.1f→%5.1f | %5.1f→%5.1f  %5.1f→%5.1f' % (
            m, z['n_c'],
            z['B1']['by_ht']['relation'] * 100, z['A2']['by_ht']['relation'] * 100,
            z['B1']['cbr'][3] * 100, z['A2']['cbr'][3] * 100,
            l['B1']['by_ht']['relation'] * 100, l['A2']['by_ht']['relation'] * 100,
            l['B1']['cbr'][3] * 100, l['A2']['cbr'][3] * 100))

    json.dump(dict(per_line=per, stats=out, models=common,
                   halving_check=halving),
              open(os.path.join(HERE, 'final_report.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('\nwrote final_report.json')


if __name__ == '__main__':
    main()
