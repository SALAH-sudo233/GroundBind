#!/usr/bin/env python3
"""JEV + OMNI combined verifier arm, on the same 500-dev protocol.

The two lines were only ever evaluated separately. Each contributes a different
reading of the SAME upstream box:

  OMNI  : OmniVerifier-7B, zero training, red-box overlay + true/false logit gap
  JEV   : Qwen3.5-2B + LoRA r=8 + scalar head + temperature, last-prompt-position
          readout, trained 1000 steps on the disjoint 2000-image pool

A combined arm is legitimate here because each line's rival/probe score comes from
its OWN scorer, so no cross-scorer subtraction happens: the fusion consumes the two
z0 values and the two gaps as four separate features and lets the fitted head decide
the weights.

Arms (all at the same pos_keep target, paper denominators, cross-fitted):
  B0        upstream, no verifier
  B1_omni   single-arm, omni z0 only
  B1_jev    single-arm, jev z0 only
  B1_combo  both z0 (no probes)
  A2_omni   omni z0 + omni gap
  A2_jev    jev z0 + jev gap
  A2_combo  both z0 + both gaps        <- the full combination
Gate is unchanged: only probe-covered relation rows may switch heads.
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


def merge(models):
    """Per model: rows keyed by sid carrying BOTH lines' z0 and gap."""
    o = A.load_line('omni', models)
    j = A.load_line('jevhead', models)
    out = {}
    for m in models:
        if not o.get(m) or not j.get(m):
            continue
        jm = {r['sid']: r for r in j[m]}
        rows = []
        for r in o[m]:
            q = jm.get(r['sid'])
            if q is None:
                continue
            rows.append(dict(
                model=m, sid=r['sid'], htype=r['htype'], is_pos=r['is_pos'],
                label=r['label'], iou=r['iou'], drew=r['drew'], fold=r['fold'],
                z_o=r['z0'], g_o=A.gap(r), zmax_o=r['zmax'],
                z_j=q['z0'], g_j=A.gap(q), zmax_j=q['zmax']))
        out[m] = rows
    return out


def feats(kind, r):
    """Feature vector per arm; None entries are dropped by the caller."""
    if kind == 'B1_omni':
        return [r['z_o']]
    if kind == 'B1_jev':
        return [r['z_j']]
    if kind == 'B1_combo':
        return [r['z_o'], r['z_j']]
    if kind == 'A2_omni':
        return [r['z_o'], r['g_o']]
    if kind == 'A2_jev':
        return [r['z_j'], r['g_j']]
    if kind == 'A2_combo':
        return [r['z_o'], r['z_j'], r['g_o'], r['g_j']]
    raise ValueError(kind)


def covered(r):
    """Row carrying a rival score from BOTH lines.

    Coverage is defined by rival presence, not by htype: positives are probed too
    and must stay in the probe head's fitting set, otherwise it sees only label-0
    rows and collapses. Probes exist only for relation negatives and positives, so
    the other three types keep the single-arm decision automatically.
    """
    return r['zmax_o'] is not None and r['zmax_j'] is not None


def ok(v):
    return v is not None


def fit(rows, kind, pos_keep):
    """Fit one arm's head on THESE rows; return a keep() decision function."""
    base = 'B1_' + ('combo' if kind.endswith('combo') else kind.split('_')[1])
    is_a2 = kind.startswith('A2')

    def usable(r, k):
        return r['drew'] and all(ok(x) for x in feats(k, r))

    # single-arm head over all drawn rows
    Rb = [r for r in rows if usable(r, base)]
    if len(Rb) < 30 or len(set(r['label'] for r in Rb)) < 2:
        return lambda r: bool(r['drew'])
    fb, _ = E.logreg([feats(base, r) for r in Rb], [r['label'] for r in Rb])
    pb = sorted(fb(feats(base, r)) for r in Rb if r['is_pos'])
    tb = pb[max(0, min(int(round((1 - pos_keep) * len(pb))), len(pb) - 1))] \
        if pb else float('-inf')

    def keep_base(r):
        if not r['drew']:
            return False
        if not usable(r, base):
            return True
        return fb(feats(base, r)) >= tb

    if not is_a2:
        return keep_base

    # probe head, fitted only on covered rows
    Rc = [r for r in rows if covered(r) and usable(r, kind)]
    if len(Rc) < 30 or len(set(r['label'] for r in Rc)) < 2:
        return keep_base
    fc, _ = E.logreg([feats(kind, r) for r in Rc], [r['label'] for r in Rc])
    pc = sorted(fc(feats(kind, r)) for r in Rc if r['is_pos'])
    tc = pc[max(0, min(int(round((1 - pos_keep) * len(pc))), len(pc) - 1))] \
        if pc else float('-inf')

    def keep_a2(r):
        if not covered(r) or not usable(r, kind):
            return keep_base(r)
        return fc(feats(kind, r)) >= tc

    return keep_a2


def metrics(rows, keep, elig, pos, neg, keepmap):
    p = [r for r in rows if r['is_pos']]
    n = [r for r in rows if not r['is_pos']]
    by = {}
    for ht in HT4:
        sub = [r for r in n if r['htype'] == ht]
        by[ht] = sum(1 for r in sub if keep(r)) / 500.0
    return dict(
        fgr_all=sum(1 for r in n if keep(r)) / 2000.0,
        boh=sum(by[h] for h in BOH) / 2.0,
        roh=sum(by[h] for h in ROH) / 2.0,
        by_ht=by,
        pos_keep=sum(1 for r in p if keep(r)) / 500.0,
        pos_miou=sum(r['iou'] for r in p if keep(r)) / 500.0,
        cbr=A.cbr(elig, pos, neg, keepmap, 0))


ARMS = ('B1_omni', 'B1_jev', 'B1_combo', 'A2_omni', 'A2_jev', 'A2_combo')


def main():
    canon = json.load(open(os.path.join(HERE, 'canon_roots_paper.json')))
    models = sorted(canon)
    data = merge(models)
    common = sorted(data)
    boxes = {m: A.load_boxes(m, canon[m]) for m in common}
    print('models=%d  (rows carry BOTH lines\' z0 and gap)\n' % len(common))

    per = defaultdict(dict)
    for m in common:
        rows = data[m]
        fA = [r for r in rows if r['fold'] == 0]
        fB = [r for r in rows if r['fold'] == 1]
        p, n = boxes[m]
        elig = [b for b, q in p.items()
                if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5]

        # B0
        km0 = {r['sid']: (True, True) for r in rows if r['drew']}
        per['B0'][m] = metrics(rows, lambda r: bool(r['drew']),
                               elig, p, n, km0)
        per['B0'][m]['n_c'] = len(elig)

        for kind in ARMS:
            kA = fit(fB, kind, 0.95)      # decide fold A with fold-B head
            kB = fit(fA, kind, 0.95)      # decide fold B with fold-A head
            dec = {}
            for r in rows:
                dec[r['sid']] = (kA(r) if r['fold'] == 0 else kB(r))
            keep = lambda r: dec[r['sid']]
            km = {s: (v, v) for s, v in dec.items()}
            per[kind][m] = metrics(rows, keep, elig, p, n, km)
            per[kind][m]['n_c'] = len(elig)

    def mn(arm, f):
        return st.mean([f(per[arm][m]) for m in common])

    print('=' * 104)
    print('500-dev，论文分母，交叉拟合，pos_keep 目标 0.95，%d 模型等权' % len(common))
    print('=' * 104)
    print('%-10s %8s%8s%8s%9s  %8s%9s  %9s%9s' %
          ('arm', 'FGR', 'BOH', 'ROH', 'relFGR', 'posKeep', 'pos_mIoU',
           'relCBR', 'attrCBR'))
    for arm in ('B0',) + ARMS:
        print('%-10s %7.2f%%%7.2f%%%7.2f%%%8.2f%%  %8.3f%9.4f  %8.1f%%%8.1f%%' % (
            arm,
            mn(arm, lambda r: r['fgr_all']) * 100,
            mn(arm, lambda r: r['boh']) * 100,
            mn(arm, lambda r: r['roh']) * 100,
            mn(arm, lambda r: r['by_ht']['relation']) * 100,
            mn(arm, lambda r: r['pos_keep']),
            mn(arm, lambda r: r['pos_miou']),
            mn(arm, lambda r: r['cbr'][3]) * 100,
            mn(arm, lambda r: r['cbr'][2]) * 100))

    print('\n' + '=' * 104)
    print('配对检验（模型级 bootstrap，n=%d）：组合臂 vs 各单线' % len(common))
    print('=' * 104)
    out = {}
    tests = [('A2_combo', 'A2_jev', '组合 − JEV单线'),
             ('A2_combo', 'A2_omni', '组合 − OMNI单线'),
             ('A2_combo', 'B1_combo', '组合探针净效应 (A2−B1, 双z0)'),
             ('B1_combo', 'B1_jev', '双z0 − JEV单z0 (无探针)')]
    for a, b, lab in tests:
        print(lab)
        for f, nm, scale in (
                (lambda r: r['by_ht']['relation'] * 100, 'relation FGR (pp)', 1),
                (lambda r: r['fgr_all'] * 100, 'ALL FGR (pp)', 1),
                (lambda r: r['cbr'][3] * 100, 'relation CBR (pp)', 1),
                (lambda r: r['cbr'][2] * 100, 'attribute CBR (pp)', 1),
                (lambda r: r['pos_keep'], 'posKeep', 1),
                (lambda r: r['pos_miou'], 'pos_mIoU', 1)):
            d = [f(per[a][m]) - f(per[b][m]) for m in common]
            mu, lo, hi = E.model_level_paired(d)
            print('   %-20s %+8.3f  CI[%+7.3f,%+7.3f] %-6s 降 %d/%d'
                  % (nm, mu, lo, hi,
                     '显著' if (lo > 0 or hi < 0) else '不显著',
                     sum(1 for v in d if v < 0), len(common)))
            out['%s_vs_%s_%s' % (a, b, nm.split()[0])] = dict(mean=mu, ci=[lo, hi])

    print('\n逐模型 relation CBR / relation FGR（A2 臂）')
    print('%-16s%5s | %-16s %-16s %-16s' %
          ('model', 'n_c', 'OMNI', 'JEV', 'COMBO'))
    for m in sorted(common, key=lambda x: -per['B0'][x]['n_c']):
        print('%-16s%5d | %5.1f%% %5.1f%%  %5.1f%% %5.1f%%  %5.1f%% %5.1f%%' % (
            m, per['B0'][m]['n_c'],
            per['A2_omni'][m]['cbr'][3] * 100,
            per['A2_omni'][m]['by_ht']['relation'] * 100,
            per['A2_jev'][m]['cbr'][3] * 100,
            per['A2_jev'][m]['by_ht']['relation'] * 100,
            per['A2_combo'][m]['cbr'][3] * 100,
            per['A2_combo'][m]['by_ht']['relation'] * 100))

    json.dump(dict(per_arm=per, stats=out, models=common),
              open(os.path.join(HERE, 'combo.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('\nwrote combo.json')


if __name__ == '__main__':
    main()
