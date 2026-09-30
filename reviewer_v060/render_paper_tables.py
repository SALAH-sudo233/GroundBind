#!/usr/bin/env python3
"""Render the paper's Table 2 / Table 3 in the paper's own metric names.

WHY THIS EXISTS. paper_tables.json stores per-model arms, but the tables the paper
actually prints use BOH / ROH / positive-success / family aggregates, and its pooled
block does not carry BOH or ROH at all. Those tables were previously assembled by
hand, which is exactly the kind of step nobody can reproduce later. This script is
the reproducible path: it reads paper_tables.json and prints both tables.

It recomputes rather than trusts: every pooled figure is re-derived from the
per-model entries, BOH/ROH are re-derived from by_ht, and the run aborts if the
stored pooled block disagrees or if the family aggregates miss the values printed in
the paper's own section 4.1. Runs on the repo checkout -- no server, no GPU.

  python3 render_paper_tables.py [--json paper_tables.json] [--panel main13]

Definitions, from section 3 of the paper:
  BOH  mean(object, co_occurrence) -- required object absent
  ROH  mean(attribute, relation)   -- constraint fails on a visible instance
  positive success  share of the 500 positives with IoU >= 0.5, i.e. n_c / 500
  FGR  denominator 2000;  full-positive mIoU  denominator 500, refusals score zero
  CBR  theta=0.5, rho=0.8 against the positive PREDICTED box, denominator n_c
  Rcorrect  retained share of originally-correct positives; eligibility frozen
Family split follows appendix B: four general VLMs, nine RL-adapted grounding models.
"""
import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HT4 = ['object', 'co_occurrence', 'attribute', 'relation']
GENERAL = {'InternVL3.5-8B', 'Qwen3-VL-8B', 'llava-ov-7b', 'qwen2.5-vl-7b'}
ARMS = (('unfiltered', 'unfiltered'),
        ('support', 'shared support verification'),
        ('full', 'adaptive structured verification'))
# printed in section 4.1 of the paper; used as an acceptance gate
PAPER_41 = {'general': (0.4331, 45.0, 37.64), 'RL': (0.4240, 42.9, 68.83)}


def mean(per, sub, arm, path):
    vals = []
    for m in sub:
        v = per[m][arm]
        for k in path:
            v = v[k] if not isinstance(k, int) else v[k]
        vals.append(v)
    return st.mean(vals)


def check_boh_roh(per, models):
    """BOH/ROH must be the mean of their two edit types, per model and per arm."""
    bad = []
    for m in models:
        for arm, _ in ARMS:
            u = per[m][arm]
            b = (u['by_ht']['object'] + u['by_ht']['co_occurrence']) / 2.0
            r = (u['by_ht']['attribute'] + u['by_ht']['relation']) / 2.0
            if abs(u['boh'] - b) > 1e-12 or abs(u['roh'] - r) > 1e-12:
                bad.append((m, arm))
    return bad


def check_pooled(per, pooled, models):
    """Stored pooled block must equal an equal-weight mean over per-model entries."""
    bad = []
    for arm, _ in ARMS:
        s = pooled[arm]
        got = [('FGR', mean(per, models, arm, ['fgr_all']) * 100, s['FGR']),
               ('CorrectRetain', mean(per, models, arm, ['correct_retain']) * 100,
                s['CorrectRetain']),
               ('mIoU', mean(per, models, arm, ['pos_miou']), s['mIoU'])]
        got += [('CBR ' + h, mean(per, models, arm, ['cbr', i]) * 100, s['CBR'][h])
                for i, h in enumerate(HT4)]
        got += [('FGR ' + h, mean(per, models, arm, ['by_ht', h]) * 100,
                 s['FGR_by_type'][h]) for h in HT4]
        for nm, a, b in got:
            if abs(a - b) > 1e-6:
                bad.append((arm, nm, a, b))
    return bad


def table2(per, models):
    print('=' * 122)
    print('TABLE 2  benchmark evaluation, unfiltered (%d models)' % len(models))
    print('=' * 122)
    print('%-16s %5s %8s %8s | %8s %8s %8s | %8s %8s %8s %8s'
          % ('model', 'family', 'pos.succ', 'mIoU', 'FGR', 'BOH', 'ROH',
             'CBR obj', 'CBR cooc', 'CBR attr', 'CBR rel'))
    print('-' * 122)
    for m in sorted(models, key=lambda x: -per[x]['unfiltered']['fgr_all']):
        u = per[m]['unfiltered']
        print('%-16s %6s %7.1f%% %8.4f | %7.2f%% %7.2f%% %7.2f%% | %7.2f%% %7.2f%% %7.2f%% %7.2f%%'
              % (m, 'general' if m in GENERAL else 'RL', per[m]['n_c'] / 5.0,
                 u['pos_miou'], u['fgr_all'] * 100, u['boh'] * 100, u['roh'] * 100,
                 *[u['cbr'][i] * 100 for i in range(4)]))
    print('-' * 122)
    groups = [('general (%d)' % len([m for m in models if m in GENERAL]),
               [m for m in models if m in GENERAL]),
              ('RL-adapted (%d)' % len([m for m in models if m not in GENERAL]),
               [m for m in models if m not in GENERAL]),
              ('all %d' % len(models), sorted(models))]
    for nm, sub in groups:
        if not sub:
            continue
        print('%-16s %6s %7.1f%% %8.4f | %7.2f%% %7.2f%% %7.2f%% | %7.2f%% %7.2f%% %7.2f%% %7.2f%%'
              % (nm, '', st.mean([per[m]['n_c'] for m in sub]) / 5.0,
                 mean(per, sub, 'unfiltered', ['pos_miou']),
                 mean(per, sub, 'unfiltered', ['fgr_all']) * 100,
                 mean(per, sub, 'unfiltered', ['boh']) * 100,
                 mean(per, sub, 'unfiltered', ['roh']) * 100,
                 *[mean(per, sub, 'unfiltered', ['cbr', i]) * 100 for i in range(4)]))


def table3(per, models):
    print()
    print('=' * 122)
    print('TABLE 3  mitigation (%d models, equal weight)' % len(models))
    print('=' * 122)
    print('%-34s %8s %8s %8s | %8s %8s %8s %8s | %9s %8s'
          % ('configuration', 'FGR', 'BOH', 'ROH', 'CBR obj', 'CBR cooc',
             'CBR attr', 'CBR rel', 'Rcorrect', 'mIoU'))
    print('-' * 122)
    for arm, name in ARMS:
        print('%-34s %7.2f%% %7.2f%% %7.2f%% | %7.2f%% %7.2f%% %7.2f%% %7.2f%% | %8.2f%% %8.4f'
              % (name, mean(per, models, arm, ['fgr_all']) * 100,
                 mean(per, models, arm, ['boh']) * 100,
                 mean(per, models, arm, ['roh']) * 100,
                 *[mean(per, models, arm, ['cbr', i]) * 100 for i in range(4)],
                 mean(per, models, arm, ['correct_retain']) * 100,
                 mean(per, models, arm, ['pos_miou'])))
    print('-' * 122)
    print('per-type FGR')
    for arm, name in ARMS:
        print('  %-32s %s' % (name, '  '.join(
            '%s %6.2f%%' % (h[:4], mean(per, models, arm, ['by_ht', h]) * 100)
            for h in HT4)))
    mono = all(per[m]['full']['fgr_all'] <= per[m]['support']['fgr_all']
               for m in models)
    print('\nfull arm no worse than support on every model: %s (%d/%d)'
          % (mono, sum(per[m]['full']['fgr_all'] <= per[m]['support']['fgr_all']
                       for m in models), len(models)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=os.path.join(HERE, 'paper_tables.json'))
    ap.add_argument('--panel', default='main13', choices=['main13', 'table3', 'appendixC'])
    a = ap.parse_args()

    d = json.load(open(a.json))
    if not d.get('unfiltered_cbr_gate_passed'):
        sys.exit('ABORT: %s was produced with a FAILED unfiltered-CBR gate' % a.json)
    panel = d[a.panel]
    per, models = panel['per'], sorted(panel['per'])

    bad = check_boh_roh(per, models)
    if bad:
        sys.exit('ABORT: BOH/ROH do not match by_ht for %s' % bad[:5])
    bad = check_pooled(per, panel['pooled'], models)
    if bad:
        sys.exit('ABORT: stored pooled block disagrees with per-model mean: %s' % bad[:5])

    table2(per, models)
    table3(per, models)

    print()
    print('=' * 122)
    print('ACCEPTANCE GATES')
    print('=' * 122)
    print('  BOH/ROH match by_ht on all %d models x 3 arms         PASS' % len(models))
    print('  stored pooled block == per-model equal-weight mean    PASS')
    allok = True
    if a.panel == 'main13' and len(models) == 13:
        for nm, sub in (('general', [m for m in models if m in GENERAL]),
                        ('RL', [m for m in models if m not in GENERAL])):
            mi = mean(per, sub, 'unfiltered', ['pos_miou'])
            ps = st.mean([per[m]['n_c'] for m in sub]) / 5.0
            fg = mean(per, sub, 'unfiltered', ['fgr_all']) * 100
            p = PAPER_41[nm]
            ok = abs(mi - p[0]) < 5e-5 and abs(ps - p[1]) < 0.05 and abs(fg - p[2]) < 0.005
            allok &= ok
            print('  section 4.1 %-7s mIoU %.4f/%.4f  succ %.1f/%.1f  FGR %.2f/%.2f  %s'
                  % (nm, mi, p[0], ps, p[1], fg, p[2], 'PASS' if ok else 'FAIL'))
        print('\n  %s' % ('all gates PASS -- tables reproduce the paper' if allok
                          else 'GATE FAILED -- do not cite these tables'))
        if not allok:
            sys.exit(1)


if __name__ == '__main__':
    main()
