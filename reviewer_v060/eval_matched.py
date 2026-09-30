#!/usr/bin/env python3
"""P0-3: matched-retention curves + Eq.(12) feasibility + image-clustered CIs.

WHY. At a shared calibration TARGET the two arms land at different ACHIEVED
retention (support 0.943 vs full 0.922 on the unified table), so their CBR/FGR gap
mixes two things: better discrimination, and simply rejecting more. The reviewer's
decisive question is whether relation contrast still helps at EQUAL achieved
correct-localization retention.

WHAT THIS DOES.
 [1] Sweeps the retention target per arm, recording ACHIEVED CorrectRetain against
     relation FGR / relation CBR / ALL FGR / pos_mIoU. Gives the FGR-Rcorrect curve.
 [2] Matched comparison: for each model, take the support arm at target 0.95 and its
     achieved retention r; linearly interpolate the FULL arm's curve at that same r.
     The difference is then at equal positive cost. This is a post-hoc curve
     comparison, NOT a deployable threshold rule -- thresholds are still fitted on
     the fitting fold only, but choosing which curve point to read uses the held
     fold, so it diagnoses discrimination rather than prescribing an operating point.
 [3] Eq.(12) feasibility: with the bypass threshold fixed and only the relation-branch
     threshold free, achievable retention is bounded by [U/Nc, (U+V)/Nc] where U =
     correct positives kept by stage 1 and NOT routed, V = correct positives routed.
     Reports the band per model and whether the 0.95 target is inside it.
 [4] Image-clustered bootstrap for the pooled relation-FGR difference: resamples the
     500 SOURCE IMAGE ids, carrying every model's rows for that image. Avoids
     treating model x image pairs as independent.
"""
import argparse
import json
import os
import statistics as st
import sys
import random
from collections import defaultdict

PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
import eval_upstream as E
import eval_cbr_paper_aligned as A

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_unified import (COORDFIX_MODELS, load_probe, merge, build, measure,
                          usable, covered, head)

TARGETS = (0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90, 0.88, 0.85, 0.80)


def interp(curve, r):
    """Linear interpolation of metric at achieved retention r.
    curve = sorted list of (achieved_retention, metric). Returns None if r is
    outside the observed range (no extrapolation)."""
    pts = sorted(curve)
    if not pts or r < pts[0][0] or r > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= r <= x1:
            if x1 == x0:
                return y0
            w = (r - x0) / (x1 - x0)
            return y0 + w * (y1 - y0)
    return pts[-1][1]


def feasible_band(rows, elig, target):
    """Eq.(12): retention range reachable by moving only the relation threshold."""
    out = {}
    for tf in (0, 1):
        fit = [r for r in rows if r['fold'] != tf]
        s_sup = head(fit, 'support')
        if s_sup is None:
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

        ep = [r for r in fit if r['is_pos'] and r['sid'] in elig and r['drew']]
        if not ep:
            continue
        cov = [r for r in ep if usable(r, 'full')]
        unc = [r for r in ep if not usable(r, 'full')]
        U = sum(1 for r in unc if keep_sup(r))
        V = len(cov)
        out['fold%d' % tf] = dict(Nc=len(ep), U=U, V=V,
                                  lo=U / len(ep), hi=(U + V) / len(ep),
                                  target_feasible=(U / len(ep) <= target <= (U + V) / len(ep)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-target', type=float, default=0.95)
    ap.add_argument('--json-out', default='matched_retention.json')
    a = ap.parse_args()

    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    cfroots = json.load(open(os.path.join(PROBE, 'canon_roots_coordfix.json')))
    models = sorted(canon)

    po_main = load_probe('probe_textroute_all.jsonl', 'z')
    pj_main = load_probe('trprobe_jev_all.jsonl', 'z_head')
    po_cf = load_probe('probecf_omni.jsonl', 'z', COORDFIX_MODELS)
    pj_cf = load_probe('probecf_jev.jsonl', 'z_head', COORDFIX_MODELS)

    curves, band, keepmaps, rowstore = {}, {}, {}, {}
    for m in models:
        cf = m in COORDFIX_MODELS
        pos, neg = A.load_boxes(m, cfroots[m] if cf else canon[m])
        elig = {q['sid'] for b, q in pos.items()
                if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
        rows = merge(m, po_cf if cf else po_main, pj_cf if cf else pj_main, cf)
        if not rows:
            continue
        rowstore[m] = rows
        curves[m] = {'support': [], 'full': []}
        keepmaps[m] = {}
        for mode in ('support', 'full'):
            for t in TARGETS:
                k = build(rows, elig, t, mode)
                d = measure(rows, k, elig, pos, neg)
                curves[m][mode].append(dict(
                    target=t, achieved=d['correct_retain'],
                    rel_fgr=d['by_ht']['relation'] * 100,
                    all_fgr=d['fgr_all'] * 100,
                    rel_cbr=d['cbr'][3] * 100,
                    attr_cbr=d['cbr'][2] * 100,
                    pos_miou=d['pos_miou']))
                if t == a.base_target:
                    keepmaps[m][mode] = k
        band[m] = feasible_band(rows, elig, a.base_target)
        print('[ok] %-16s curve points=%d' % (m, len(TARGETS)), flush=True)

    common = sorted(curves)

    # ---------- [1] curves ----------
    print('\n' + '=' * 104)
    print('FGR-Rcorrect curve (pooled, %d models equal weight)' % len(common))
    print('=' * 104)
    print('%-9s | %-30s | %-30s' % ('target', 'support: achieved / relFGR / relCBR',
                                    'full: achieved / relFGR / relCBR'))
    for i, t in enumerate(TARGETS):
        row = []
        for mode in ('support', 'full'):
            ach = st.mean([curves[m][mode][i]['achieved'] for m in common])
            rf = st.mean([curves[m][mode][i]['rel_fgr'] for m in common])
            rc = st.mean([curves[m][mode][i]['rel_cbr'] for m in common])
            row.append('%.4f / %6.2f%% / %6.2f%%' % (ach, rf, rc))
        print('%-9.2f | %-30s | %-30s' % (t, row[0], row[1]))

    # ---------- [2] matched comparison ----------
    print('\n' + '=' * 104)
    print('MATCHED achieved retention: support@%.2f vs full interpolated at the '
          'SAME achieved retention' % a.base_target)
    print('=' * 104)
    print('%-16s%10s%12s%12s%10s%12s%12s%10s' %
          ('model', 'R_match', 'supRelFGR', 'fullRelFGR', 'dFGR',
           'supRelCBR', 'fullRelCBR', 'dCBR'))
    matched = {}
    for m in common:
        sup = [c for c in curves[m]['support'] if c['target'] == a.base_target][0]
        r = sup['achieved']
        f_fgr = interp([(c['achieved'], c['rel_fgr']) for c in curves[m]['full']], r)
        f_cbr = interp([(c['achieved'], c['rel_cbr']) for c in curves[m]['full']], r)
        f_all = interp([(c['achieved'], c['all_fgr']) for c in curves[m]['full']], r)
        f_mio = interp([(c['achieved'], c['pos_miou']) for c in curves[m]['full']], r)
        matched[m] = dict(r=r, sup_rel_fgr=sup['rel_fgr'], full_rel_fgr=f_fgr,
                          sup_rel_cbr=sup['rel_cbr'], full_rel_cbr=f_cbr,
                          sup_all_fgr=sup['all_fgr'], full_all_fgr=f_all,
                          sup_miou=sup['pos_miou'], full_miou=f_mio,
                          in_range=(f_fgr is not None))
        print('%-16s%10.4f%11.2f%%%11.2f%%%+9.2f%10.2f%%%11.2f%%%+9.2f' % (
            m, r, sup['rel_fgr'],
            f_fgr if f_fgr is not None else float('nan'),
            (f_fgr - sup['rel_fgr']) if f_fgr is not None else float('nan'),
            sup['rel_cbr'],
            f_cbr if f_cbr is not None else float('nan'),
            (f_cbr - sup['rel_cbr']) if f_cbr is not None else float('nan')))

    ok = [m for m in common if matched[m]['in_range']]
    print('\nmodels with the matched point inside the observed curve: %d/%d'
          % (len(ok), len(common)))
    stats = {}
    for key, nm in (('rel_fgr', 'relation FGR (pp)'), ('rel_cbr', 'relation CBR (pp)'),
                    ('all_fgr', 'ALL FGR (pp)'), ('miou', 'pos_mIoU')):
        dd = [matched[m]['full_' + key] - matched[m]['sup_' + key] for m in ok]
        mu, lo, hi = E.model_level_paired(dd)
        sig = (lo > 0 or hi < 0)
        print('  %-22s %+8.3f CI[%+7.3f,%+7.3f] %-9s down %d/%d'
              % (nm, mu, lo, hi, 'SIG' if sig else 'ns',
                 sum(1 for v in dd if v < 0), len(dd)))
        stats['matched_' + key] = dict(mean=mu, ci=[lo, hi], significant=sig,
                                       n_down=sum(1 for v in dd if v < 0), n=len(dd))

    # ---------- [3] Eq.(12) feasibility ----------
    print('\n' + '=' * 104)
    print('Eq.(12) feasibility band at target %.2f  [U/Nc, (U+V)/Nc]' % a.base_target)
    print('=' * 104)
    print('%-16s%7s%7s%7s%10s%10s%12s' % ('model', 'Nc', 'U', 'V', 'lo', 'hi', 'feasible'))
    for m in common:
        for fk in sorted(band[m]):
            b = band[m][fk]
            print('%-16s%7d%7d%7d%10.4f%10.4f%12s'
                  % ('%s[%s]' % (m, fk[-1]), b['Nc'], b['U'], b['V'],
                     b['lo'], b['hi'], 'yes' if b['target_feasible'] else 'NO'))
    nfeas = sum(1 for m in common for fk in band[m] if band[m][fk]['target_feasible'])
    ntot = sum(len(band[m]) for m in common)
    print('\nfeasible folds: %d/%d' % (nfeas, ntot))

    # ---------- [4] image-clustered bootstrap ----------
    print('\n' + '=' * 104)
    print('Image-clustered bootstrap of pooled relation-FGR difference (full - support)')
    print('at the shared TARGET %.2f; resamples the 500 source images, all models carried'
          % a.base_target)
    print('=' * 104)
    per_img = defaultdict(lambda: defaultdict(dict))
    for m in common:
        for r in rowstore[m]:
            if r['htype'] != 'relation':
                continue
            img = E.img_of(r['sid'])
            for mode in ('support', 'full'):
                per_img[img][m][mode] = bool(keepmaps[m][mode].get(r['sid']))
    imgs = sorted(per_img)
    print('images with relation negatives: %d' % len(imgs))

    def pooled_diff(sample):
        vals = []
        for m in common:
            num_s = num_f = den = 0
            for img in sample:
                d = per_img[img].get(m)
                if not d or 'support' not in d or 'full' not in d:
                    continue
                den += 1
                num_s += d['support']
                num_f += d['full']
            if den:
                vals.append((num_f - num_s) / den * 100)
        return st.mean(vals) if vals else float('nan')

    point = pooled_diff(imgs)
    rnd = random.Random(0)
    boot = []
    for _ in range(2000):
        s = [imgs[rnd.randrange(len(imgs))] for _ in range(len(imgs))]
        boot.append(pooled_diff(s))
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    print('pooled relation FGR diff = %+.3f pp   image-clustered 95%% CI [%+.3f, %+.3f]  %s'
          % (point, lo, hi, 'SIG' if (lo > 0 or hi < 0) else 'ns'))
    stats['image_clustered_rel_fgr'] = dict(mean=point, ci=[lo, hi],
                                            significant=(lo > 0 or hi < 0),
                                            n_images=len(imgs))

    json.dump(dict(base_target=a.base_target, targets=list(TARGETS),
                   models=common, curves=curves, matched=matched,
                   feasibility=band, stats=stats),
              open(os.path.join(HERE, a.json_out), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote %s' % a.json_out)


if __name__ == '__main__':
    main()
