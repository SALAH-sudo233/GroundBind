#!/usr/bin/env python3
"""Per-model mitigation table, columns aligned with the paper's evaluation tables.

Metric set and column order follow the manuscript so rows can be dropped next to
the existing tables without re-deriving anything:

  from Table 2 (Direct grounding):  mIoU, FGR
  from Table 4 (CBR):               object / co-occurrence / attribute / relation

Three arms per model:
  B0        upstream prediction, no verifier   (this is what Table 2/4 report)
  support   [z_omni, z_jev]                    (no probes)
  full      [z_omni, z_jev, gap_omni, gap_jev] (the paper's framework)

Both mitigation arms are calibrated on the SAME positives (the budget-aligned fix),
so support and full sit at one positive budget and a difference in FGR/CBR is a
difference in discrimination.

Denominators are the paper's: mIoU and CorrectRetain over 500 positives, FGR per
type over that type's 500 negatives, ALL-FGR over 2000, CBR over the frozen
eligible set. Rejected / no-box rows count as zero and are never dropped.
"""
import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_cbr_paper_aligned as A
import eval_combo_fixedbudget as F

HT4 = A.HT4
# Backbone scale column as printed in the manuscript (Table 2).
BACKBONE = {
    'Qwen3-VL-8B': '8', 'InternVL3.5-8B': '8',
    'Qwen2.5-VL-7B': '7', 'qwen2.5-vl-7b': '7',
    'LLaVA-OV-7B': '7', 'llava-ov-7b': '7',
    'Orsta-7B': '7', 'TreeVGR': '7+', 'Vision-R1': '7+',
    'VisionReasoner': '7+', 'Seg-R1': '3/7+', 'Seg-zero': '2/3/7+',
    'LENS': '2/3+', 'UniVG-R1': '2/7+', 'visual-rft': '2/7+',
}
DISPLAY = {'Seg-zero': 'Seg-Zero', 'qwen2.5-vl-7b': 'Qwen2.5-VL-7B',
           'llava-ov-7b': 'LLaVA-OV-7B', 'visual-rft': 'Visual-RFT'}


def b0_metrics(pos, neg, elig):
    """Upstream, no verifier: exactly the quantities Table 2 / Table 4 report.

    FGR uses the paper's rule -- a negative counts as falsely grounded when the
    model RETURNED a box (pred_found), regardless of that box's geometry. Counting
    only geometrically valid boxes would silently drop the 823 / 174 degenerate
    boxes of the two coordinate-broken models and understate their FGR (58.65% vs
    the paper's 99.80% for UniVG-R1). On the other 11 models the two rules agree.
    """
    miou = sum(A.iou(q['pred'], q['gt']) for q in pos.values()) / 500.0
    fgr = {ht: sum(1 for b in neg if neg[b].get(ht)
                   and neg[b][ht]['pred'] is not None) / 500.0 for ht in HT4}
    return dict(
        pos_miou=miou, by_ht=fgr,
        fgr_all=sum(fgr[h] for h in HT4) / 4.0,
        cbr=A.cbr(list(elig), pos, neg, {}, None),
        correct_retain=1.0,
        pos_keep=sum(1 for q in pos.values() if A.valid_box(q['pred'])) / 500.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canon', default='canon_roots_paper.json')
    ap.add_argument('--coordfix', action='store_true')
    ap.add_argument('--omni-probes', default='probe_textroute_all.jsonl')
    ap.add_argument('--jev-probes', default='trprobe_jev_all.jsonl')
    ap.add_argument('--target', type=float, default=0.95)
    ap.add_argument('--json-out', default='permodel_095.json')
    a = ap.parse_args()

    canon = json.load(open(os.path.join(HERE, a.canon)))
    models = sorted(canon)
    po = F.load_probe(a.omni_probes, 'z')
    pj = F.load_probe(a.jev_probes, 'z_head')

    per = {}
    for m in models:
        pos, neg = A.load_boxes(m, canon[m])
        elig = {q['sid'] for b, q in pos.items()
                if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
        elig_b = {b for b, q in pos.items()
                  if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
        rows = F.merge(m, po, pj, a.coordfix)
        if not rows:
            print('[skip] %s' % m)
            continue
        npos = sum(1 for r in rows if r['is_pos'])
        nneg = len(rows) - npos
        d = dict(n_c=len(elig_b), n_pos=npos, n_neg=nneg,
                 B0=b0_metrics(pos, neg, elig_b))
        for mode in ('support', 'full'):
            k = F.build(rows, elig, a.target, mode)
            d[mode] = F.measure(rows, k, elig_b, pos, neg)
        per[m] = d
    common = sorted(per)

    # ---------- Table 2 aligned: mIoU + FGR
    print('=' * 112)
    print('逐模型 · 对齐论文 Table 2（Direct grounding: mIoU / FGR），target CorrectRetain=%.2f'
          % a.target)
    print('=' * 112)
    print('%-16s%-8s%6s  %-24s  %-24s' %
          ('Model', 'Bb(B)', 'n_c', 'mIoU  B0 / sup / full',
           'FGR%  B0 / sup / full'))
    for m in common:
        d = per[m]
        print('%-16s%-8s%6d  %6.4f %6.4f %6.4f   %6.2f %6.2f %6.2f' % (
            DISPLAY.get(m, m), BACKBONE.get(m, '?'), d['n_c'],
            d['B0']['pos_miou'], d['support']['pos_miou'], d['full']['pos_miou'],
            d['B0']['fgr_all'] * 100, d['support']['fgr_all'] * 100,
            d['full']['fgr_all'] * 100))

    # ---------- Table 4 aligned: CBR per htype
    print('\n' + '=' * 112)
    print('逐模型 · 对齐论文 Table 4（CBR%，四类），B0 → support-only → 完整框架')
    print('=' * 112)
    print('%-16s%6s  %s' % ('Model', 'n_c', ''.join(
        '%-22s' % h[:12] for h in HT4)))
    for m in common:
        d = per[m]
        cells = ''
        for i in range(4):
            cells += '%5.1f %5.1f %5.1f    ' % (
                d['B0']['cbr'][i] * 100, d['support']['cbr'][i] * 100,
                d['full']['cbr'][i] * 100)
        print('%-16s%6d  %s' % (DISPLAY.get(m, m), d['n_c'], cells))

    # ---------- per-type FGR
    print('\n' + '=' * 112)
    print('逐模型 · 逐类 FGR%（分母=该类 500 负例），B0 → support-only → 完整框架')
    print('=' * 112)
    print('%-16s  %s' % ('Model', ''.join('%-22s' % h[:12] for h in HT4)))
    for m in common:
        d = per[m]
        cells = ''
        for h in HT4:
            cells += '%5.1f %5.1f %5.1f    ' % (
                d['B0']['by_ht'][h] * 100, d['support']['by_ht'][h] * 100,
                d['full']['by_ht'][h] * 100)
        print('%-16s  %s' % (DISPLAY.get(m, m), cells))

    # ---------- positive-side cost
    print('\n' + '=' * 112)
    print('逐模型 · 正例侧代价（CorrectRetain，分母=该模型正确定位的正例）')
    print('=' * 112)
    print('%-16s%6s%12s%12s%12s' %
          ('Model', 'n_c', 'support', 'full', 'delta'))
    for m in common:
        d = per[m]
        print('%-16s%6d%12.3f%12.3f%+12.3f' % (
            DISPLAY.get(m, m), d['n_c'], d['support']['correct_retain'],
            d['full']['correct_retain'],
            d['full']['correct_retain'] - d['support']['correct_retain']))

    if len(common) > 2:
        print('\n池化（%d 模型等权）' % len(common))
        for arm in ('B0', 'support', 'full'):
            print('  %-9s mIoU=%.4f FGR=%.2f%% relFGR=%.2f%% relCBR=%.1f%% '
                  'attrCBR=%.1f%%'
                  % (arm,
                     st.mean([per[m][arm]['pos_miou'] for m in common]),
                     st.mean([per[m][arm]['fgr_all'] for m in common]) * 100,
                     st.mean([per[m][arm]['by_ht']['relation'] for m in common]) * 100,
                     st.mean([per[m][arm]['cbr'][3] for m in common]) * 100,
                     st.mean([per[m][arm]['cbr'][2] for m in common]) * 100))

    # denominator receipt
    bad = [(m, per[m]['n_pos'], per[m]['n_neg']) for m in common
           if per[m]['n_pos'] != 500 or per[m]['n_neg'] != 2000]
    print('\n分母核验：%s' % ('全部 13 模型均为 500 正 / 2000 负'
                          if not bad else 'OFF-SPEC %s' % bad))

    json.dump(dict(target=a.target, coordfix=a.coordfix, models=common,
                   backbone={m: BACKBONE.get(m, '?') for m in common}, per=per),
              open(os.path.join(HERE, a.json_out), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('wrote %s' % a.json_out)


if __name__ == '__main__':
    main()
