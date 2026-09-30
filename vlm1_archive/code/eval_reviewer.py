#!/usr/bin/env python3
"""Reviewer-facing evaluation: W1 label-independent routing, W3 matched-budget.

W1 (P0). The earlier collection routed on `hallucination_type == 'relation'`, which
is evaluation metadata and cannot be read at inference time. This script consumes
probes collected by `collect_probe_textroute.py`, whose router is

    route(query, fixed_config) -> competitive rivals exist for this query

reading ONLY the query string and a frozen predicate table. That router also fires
on object / co_occurrence / attribute rows whose queries contain a spatial
predicate, so all four types can now change -- the three-type "exactly 0.00pp" of
the metadata-gated version was an artifact of the illegitimate gate, not a property
of the method.

W1 (fitting). Every learned quantity is cross-fitted at the IMAGE level: fold A
rows are decided by parameters fitted on fold B and vice versa, and folds are
md5(image) % 2, so no row is scored by a head or threshold that saw its own image's
labels. Reported per fold and pooled.

W3 (P1). The headline comparison is at MATCHED positive-localization retention.
Thresholds for support-only and for the full method are chosen on the FITTING fold
to hit a common CorrectRetain target, then the held fold reports what was actually
achieved. A one-sided drop is structurally guaranteed for any stricter gate, so the
only meaningful question is error reduction at equal positive cost.

CorrectRetain (reviewer's requested quantity, NOT mIoU ratio):

    CorrectRetain_m = sum_i c_{m,i} * k+_{m,i} / sum_i c_{m,i}

i.e. of the positives this model originally localized correctly, what fraction
survives the filter. This is distinct from 0.3549/0.3738 = 94.9%.
"""
import argparse
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
REPROV = A.REPROV


def load_probes(path, key):
    """(model, sid) -> {variant: z} from a text-routed probe file."""
    out = defaultdict(dict)
    for line in open(os.path.join(HERE, path), encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get('error') or d.get(key) is None:
            continue
        out[(d['model'], d['sid'])][d['variant']] = d[key]
    return out


def load_arm(models, line, probes):
    """Upstream rows + this line's own z0 + its own text-routed rival max."""
    data = {}
    for m in models:
        spec = REPROV.get(m)
        if line == 'omni':
            f = (spec['omni_arm'] % m) if spec else ('s5omni_omni_%s.jsonl' % m)
            zk = 'z_omni'
        else:
            f = 'jevhead_all.jsonl'
            zk = 'z_head'
        rows = []
        if line == 'jev':
            src = [json.loads(l) for l in open(os.path.join(HERE, f), encoding='utf-8')
                   if l.strip()]
            src = [r for r in src if r.get('model') == m]
        else:
            p = os.path.join(S5, f)
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                print('[skip] %s %s' % (m, f))
                continue
            src = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
        for r in src:
            sid = r['sid']
            rv = probes.get((m, sid), {})
            zs = [v for v in rv.values() if v is not None]
            ht = r.get('htype')
            rows.append(dict(
                model=m, sid=sid, htype=ht, is_pos=(ht == 'positive'),
                label=1 if ht == 'positive' else 0,
                iou=float(r.get('iou') or 0.0), drew=bool(r.get('drew')),
                z0=r.get(zk), zmax=(max(zs) if zs else None),
                n_rivals=len(zs), fold=E.fold_of(sid)))
        if rows:
            data[m] = rows
    return data


def gap(r):
    if r['zmax'] is None or r['z0'] is None:
        return 0.0
    return r['zmax'] - r['z0']


def covered(r):
    return r['zmax'] is not None and r['z0'] is not None


def fit_head(rows, use_gap):
    """Logistic head on [z0] or [z0, gap]; returns score fn over usable rows."""
    R = [r for r in rows if r['drew'] and r['z0'] is not None
         and (not use_gap or covered(r))]
    if len(R) < 30 or len(set(r['label'] for r in R)) < 2:
        return None
    X = [([r['z0'], gap(r)] if use_gap else [r['z0']]) for r in R]
    f, _ = E.logreg(X, [r['label'] for r in R])
    return (lambda r: f([r['z0'], gap(r)])) if use_gap else (lambda r: f([r['z0']]))


def thr_for_retain(rows, score, elig_sids, target):
    """Threshold hitting CorrectRetain >= target on correctly-localized positives."""
    v = sorted(score(r) for r in rows
               if r['is_pos'] and r['sid'] in elig_sids and r['drew']
               and r['z0'] is not None)
    if not v:
        return float('-inf')
    k = int(round((1.0 - target) * len(v)))
    return v[max(0, min(k, len(v) - 1))]


def build_keep(rows, elig_sids, target, mode):
    """Cross-fitted keep map. mode: 'support' = z0 only, 'full' = z0 + gap gate."""
    keep = {}
    for tf in (0, 1):
        fit = [r for r in rows if r['fold'] != tf]
        tst = [r for r in rows if r['fold'] == tf]
        s0 = fit_head(fit, False)
        if s0 is None:
            for r in tst:
                keep[r['sid']] = bool(r['drew'])
            continue
        t0 = thr_for_retain(fit, s0, elig_sids, target)
        s1 = fit_head(fit, True) if mode == 'full' else None
        t1 = thr_for_retain([r for r in fit if covered(r)], s1, elig_sids, target) \
            if s1 else None
        for r in tst:
            if not r['drew']:
                keep[r['sid']] = False
                continue
            if r['z0'] is None:
                keep[r['sid']] = True
                continue
            if s1 is not None and covered(r):
                keep[r['sid']] = s1(r) >= t1
            else:
                keep[r['sid']] = s0(r) >= t0
    return keep


def measure(rows, keep, elig_sids, pos, neg, boxes_keep):
    p = [r for r in rows if r['is_pos']]
    n = [r for r in rows if not r['is_pos']]
    by = {ht: sum(1 for r in n if r['htype'] == ht and keep.get(r['sid']))
          / 500.0 for ht in HT4}
    ce = [r for r in p if r['sid'] in elig_sids]
    return dict(
        fgr_all=sum(1 for r in n if keep.get(r['sid'])) / 2000.0,
        by_ht=by,
        boh=(by['object'] + by['co_occurrence']) / 2.0,
        roh=(by['attribute'] + by['relation']) / 2.0,
        pos_keep=sum(1 for r in p if keep.get(r['sid'])) / 500.0,
        pos_miou=sum(r['iou'] for r in p if keep.get(r['sid'])) / 500.0,
        correct_retain=(sum(1 for r in ce if keep.get(r['sid'])) / len(ce))
        if ce else float('nan'),
        cbr=A.cbr(list(elig_sids), pos, neg, boxes_keep, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--omni-probes', default='probe_textroute_all.jsonl')
    ap.add_argument('--jev-probes', default='trprobe_jev_all.jsonl')
    ap.add_argument('--targets',
                    default='0.995,0.99,0.98,0.97,0.96,0.95,0.93,0.90,0.85,0.80')
    ap.add_argument('--match-at', default='0.95,0.90,0.85',
                    help='achieved-CorrectRetain levels for the frontier comparison')
    ap.add_argument('--json-out', default='reviewer.json')
    a = ap.parse_args()

    canon = json.load(open(os.path.join(HERE, 'canon_roots_paper.json')))
    models = sorted(canon)
    po = load_probes(a.omni_probes, 'z')
    pj = load_probes(a.jev_probes, 'z_head') if os.path.exists(
        os.path.join(HERE, a.jev_probes)) else {}
    print('omni probe rows=%d   jev probe rows=%d' % (len(po), len(pj)))

    ao = load_arm(models, 'omni', po)
    aj = load_arm(models, 'jev', pj) if pj else {}
    common = sorted(set(ao) & (set(aj) if aj else set(ao)))
    boxes = {m: A.load_boxes(m, canon[m]) for m in common}
    print('models=%d\n' % len(common))

    # routing coverage, by htype -- the W1 evidence
    print('=' * 100)
    print('W1 路由覆盖（route 只读 query 文本）：各类型被探针覆盖的负例行数')
    print('=' * 100)
    cov = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(lambda: defaultdict(int))
    for m in common:
        for r in ao[m]:
            ht = r['htype'] or 'positive'
            tot[ht]['n'] += 1
            if covered(r):
                cov[ht]['n'] += 1
    print('%-16s%10s%10s%9s' % ('htype', 'covered', 'rows', 'share'))
    for ht in ['positive'] + list(HT4):
        c, t = cov[ht]['n'], tot[ht]['n']
        print('%-16s%10d%10d%8.1f%%' % (ht, c, t, 100.0 * c / max(1, t)))

    targets = [float(x) for x in a.targets.split(',')]
    out = {}
    per = defaultdict(lambda: defaultdict(dict))
    for tgt in targets:
        for m in common:
            p, n = boxes[m]
            elig = {b for b, q in p.items()
                    if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
            for lname, arm in (('omni', ao), ('jev', aj)):
                if not arm:
                    continue
                rows = arm[m]
                for mode in ('support', 'full'):
                    keep = build_keep(rows, elig, tgt, mode)
                    km = {s: (v, v) for s, v in keep.items()}
                    per[(tgt, lname, mode)][m] = measure(
                        rows, keep, elig, p, n, km)

    def mn(k, f):
        return st.mean([f(per[k][m]) for m in common])

    print('\n' + '=' * 100)
    print('W3 同 CorrectRetain 预算下的错误率（交叉拟合，论文分母）')
    print('=' * 100)
    print('%-7s%-6s%-9s%11s%9s%9s%9s%9s%9s' %
          ('target', 'line', 'arm', 'CorrRetain', 'posmIoU', 'FGR',
           'relFGR', 'relCBR', 'attrCBR'))
    for tgt in targets:
        for lname in ('omni', 'jev'):
            if not aj and lname == 'jev':
                continue
            for mode in ('support', 'full'):
                k = (tgt, lname, mode)
                print('%-7.3f%-6s%-9s%10.3f%10.4f%8.2f%%%8.2f%%%8.1f%%%8.1f%%' % (
                    tgt, lname if mode == 'support' else '', mode,
                    mn(k, lambda r: r['correct_retain']),
                    mn(k, lambda r: r['pos_miou']),
                    mn(k, lambda r: r['fgr_all']) * 100,
                    mn(k, lambda r: r['by_ht']['relation']) * 100,
                    mn(k, lambda r: r['cbr'][3]) * 100,
                    mn(k, lambda r: r['cbr'][2]) * 100))

    print('\n' + '=' * 100)
    print('W3 配对检验：full − support（同 target，模型级 bootstrap n=%d）' % len(common))
    print('=' * 100)
    for tgt in targets:
        for lname in ('omni', 'jev'):
            if not aj and lname == 'jev':
                continue
            ks, kf = (tgt, lname, 'support'), (tgt, lname, 'full')
            print('target=%.2f  line=%s' % (tgt, lname))
            for f, nm in ((lambda r: r['correct_retain'], 'CorrectRetain'),
                          (lambda r: r['pos_miou'], 'pos_mIoU'),
                          (lambda r: r['fgr_all'] * 100, 'ALL FGR (pp)'),
                          (lambda r: r['by_ht']['relation'] * 100, 'rel FGR (pp)'),
                          (lambda r: r['cbr'][3] * 100, 'rel CBR (pp)'),
                          (lambda r: r['cbr'][2] * 100, 'attr CBR (pp)')):
                d = [f(per[kf][m]) - f(per[ks][m]) for m in common]
                mu, lo, hi = E.model_level_paired(d)
                print('   %-16s %+8.3f CI[%+7.3f,%+7.3f] %-6s 降 %d/%d'
                      % (nm, mu, lo, hi,
                         '显著' if (lo > 0 or hi < 0) else '不显著',
                         sum(1 for v in d if v < 0), len(common)))
                out['t%.3f_%s_full-support_%s'
                    % (tgt, lname, nm.replace(' (pp)', '').replace(' ', '_'))] = \
                    dict(mean=mu, ci=[lo, hi])

    # ---------- frontier: compare at EQUAL ACHIEVED retention, per model
    # Matching the TARGET is not enough: the full arm often lands at a more
    # permissive achieved retention than support, which by itself raises FGR.
    # So build each arm's (achieved_retain -> metric) curve per model and read
    # both arms at the same achieved level by linear interpolation.
    print('\n' + '=' * 100)
    print('W3 前沿比较：在相同【实际达到】的 CorrectRetain 上读数（逐模型内插后配对）')
    print('=' * 100)

    def curve(lname, mode, m, f):
        pts = []
        for tgt in targets:
            r = per[(tgt, lname, mode)].get(m)
            if not r:
                continue
            cr = r['correct_retain']
            if cr == cr:      # not NaN
                pts.append((cr, f(r)))
        pts.sort()
        return pts

    def read_at(pts, x):
        """Linear interpolation; None if x outside the achieved range."""
        if len(pts) < 2 or x < pts[0][0] or x > pts[-1][0]:
            return None
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            if x0 <= x <= x1:
                if x1 == x0:
                    return y0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return None

    METRICS = [(lambda r: r['fgr_all'] * 100, 'ALL FGR (pp)'),
               (lambda r: r['by_ht']['relation'] * 100, 'rel FGR (pp)'),
               (lambda r: r['cbr'][3] * 100, 'rel CBR (pp)'),
               (lambda r: r['cbr'][2] * 100, 'attr CBR (pp)')]
    for lvl in [float(x) for x in a.match_at.split(',')]:
        for lname in ('omni', 'jev'):
            if not aj and lname == 'jev':
                continue
            print('achieved CorrectRetain = %.2f   line=%s' % (lvl, lname))
            for f, nm in METRICS:
                d, usable = [], 0
                for m in common:
                    vs = read_at(curve(lname, 'support', m, f), lvl)
                    vf = read_at(curve(lname, 'full', m, f), lvl)
                    if vs is None or vf is None:
                        continue
                    usable += 1
                    d.append(vf - vs)
                if len(d) < 3:
                    print('   %-16s 可用模型仅 %d 个，该操作点不可达，不报告'
                          % (nm, len(d)))
                    continue
                mu, lo, hi = E.model_level_paired(d)
                print('   %-16s %+8.3f CI[%+7.3f,%+7.3f] %-6s n=%d 降 %d'
                      % (nm, mu, lo, hi,
                         '显著' if (lo > 0 or hi < 0) else '不显著',
                         usable, sum(1 for v in d if v < 0)))
                slug = nm.replace(' (pp)', '').replace(' ', '_')
                out['frontier%.2f_%s_%s' % (lvl, lname, slug)] = \
                    dict(mean=mu, ci=[lo, hi], n=usable)

    json.dump(dict(per=({'%s|%s|%s' % (k[0], k[1], k[2]): v
                         for k, v in per.items()}),
                   stats=out, models=common),
              open(os.path.join(HERE, a.json_out), 'w'),
              indent=2, ensure_ascii=False)
    print('\nwrote %s' % a.json_out)


if __name__ == '__main__':
    main()
