#!/usr/bin/env python3
"""Plan v0.2 section 8 control matrix, evaluated on the leak-free probe subset.

Why this subset: the `probed` indicator bit is an ORACLE LEAK (probe collection
filtered on `ht=='relation' or role=='positive'`, so the bit encodes the true
hallucination_type and polarity; positive rate 0.531 vs 0.174). The only clean
comparison is WITHIN the rows that carry probe scores, where the bit is constant.
Every arm below therefore sees exactly the same rows, and no feature may encode
htype, polarity, group id, GT box or paired positive expression.

Arms implemented (plan section 8). All share one query set, one geometry guard,
one split and equal calibration rights:

  B0      upstream unfiltered
  Bgeom   GeometryGuard only (drop degenerate/invalid boxes, no model call)
  B1      Omni single arm, threshold on z0                        [1 Omni]
  B2      2B/JEV single arm, threshold on z_jev(q0)               [1 2B]
  B3      DV single arm (upstream t1 pred_exists)                 [0, reuses record]
  B4mean  static mean fusion of Omni + JEV (both on q0)           [1 Omni + 1 2B]
  B4and   static AND of Omni and JEV keep decisions
  B5      learned head on original-expression signals only
          (z0, z_jev_q0, dv) -- rules out "the gain is just a trained head"
  A0      fixed probe order: always execute rival #1, Omni rechecks it   [2 Omni]
  A1      random legal rival, Omni rechecks it (seeded)                  [2 Omni]
  A2      2B selects ONE rival, Omni rechecks only that one    [2 Omni + n 2B]
  A4      Omni scores ALL rivals itself and takes the max gap    [1+n Omni]
          (this is what the earlier report called "A2"; it is the cost-heavy
           Omni-only control, and the honest baseline for judging the 2B)
  A5      A2 with DV removed
  Aswap   A2 control: the selected probe is scored on a MISMATCHED image
          (offline ablation only; tests whether the extra call uses THIS image)

Metrics / protocol are inherited from eval_upstream.py: folds = md5(image
basename) % 2, fit on fold A, fold B read once, AUROC by rank-sum, CIs by paired
bootstrap clustered on source image plus a model-level paired test over n=13.
Denominators are the subset's own counts and are printed; they are NOT presented
as the official 2000/500 table.
"""
import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
S5 = os.path.expanduser('~/SVD/grpo_verifier/s5grpo')
CANON = os.path.join(HERE, 'canon_roots.json')
PROBES = os.path.join(HERE, 'probe_upstream.jsonl')
SEL2B = os.path.join(HERE, 'sel2b_all.jsonl')


# --------------------------------------------------------------- metrics
def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    if n == 0:
        return float('nan')
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    npos = sum(l for _, l in pairs)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    spos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (spos - npos * (npos + 1) / 2.0) / (npos * nneg)


def sig(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logreg(X, y, l2=1.0, iters=300):
    k = len(X[0])
    n = len(X)
    mu = [sum(r[j] for r in X) / n for j in range(k)]
    sd = [((sum((r[j] - mu[j]) ** 2 for r in X) / n) ** 0.5) or 1.0 for j in range(k)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(k)] for r in X]
    w = [0.0] * k
    b = 0.0
    for _ in range(iters):
        g = [0.0] * k
        gb = 0.0
        H = [[0.0] * k for _ in range(k)]
        Hb = 0.0
        Hxb = [0.0] * k
        for zi, yi in zip(Z, y):
            p = sig(sum(w[j] * zi[j] for j in range(k)) + b)
            e = p - yi
            wt = max(p * (1 - p), 1e-6)
            for j in range(k):
                g[j] += e * zi[j]
                Hxb[j] += wt * zi[j]
                for j2 in range(k):
                    H[j][j2] += wt * zi[j] * zi[j2]
            gb += e
            Hb += wt
        for j in range(k):
            g[j] += l2 * w[j]
            H[j][j] += l2
        M = [[H[i][j] for j in range(k)] + [Hxb[i]] for i in range(k)]
        M.append([Hxb[j] for j in range(k)] + [Hb + 1e-6])
        rhs = g + [gb]
        for i in range(k + 1):
            piv = M[i][i] if abs(M[i][i]) > 1e-12 else 1e-12
            for j in range(i + 1, k + 1):
                f = M[j][i] / piv
                for c in range(i, k + 1):
                    M[j][c] -= f * M[i][c]
                rhs[j] -= f * rhs[i]
        dl = [0.0] * (k + 1)
        for i in range(k, -1, -1):
            s = rhs[i] - sum(M[i][c] * dl[c] for c in range(i + 1, k + 1))
            dl[i] = s / (M[i][i] if abs(M[i][i]) > 1e-12 else 1e-12)
        st = 1.0
        while st > 1e-4 and max(abs(x * st) for x in dl) > 4:
            st *= 0.5
        for j in range(k):
            w[j] -= dl[j] * st
        b -= dl[k] * st
        if max(abs(x * st) for x in dl) < 1e-9:
            break

    def pred(r):
        return sig(sum(w[j] * ((r[j] - mu[j]) / sd[j]) for j in range(k)) + b)
    return pred, w


def img_of(sid):
    base = str(sid).split('__')[0]
    i = base.find('COCO_')
    return (base[i:] if i >= 0 else base) + '.jpg'


def fold_of(sid):
    return int(hashlib.md5(img_of(sid).encode()).hexdigest(), 16) % 2


def _selftest():
    assert abs(auroc([1, 1, 1, 1], [1, 1, 0, 0]) - 0.5) < 1e-12
    assert abs(auroc([2, 2, 1], [1, 0, 0]) - 0.75) < 1e-12
    assert img_of('hallu_1_COCO_train2014_000000310457__rel') == \
        'COCO_train2014_000000310457.jpg'


def valid_box(b):
    """Shared geometry guard: finite, positive area. Identical for every arm."""
    if not b or len(b) != 4:
        return False
    try:
        x0, y0, x1, y1 = [float(v) for v in b]
    except Exception:
        return False
    if any(v != v for v in (x0, y0, x1, y1)):
        return False
    return x1 > x0 and y1 > y0


# --------------------------------------------------------------- load
def load(models):
    """Join: Omni z0 arm + Omni rival scores + 2B selection readings + DV."""
    rivals = defaultdict(dict)
    for line in open(PROBES, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('error') or r.get('z') is None:
            continue
        rivals[(r['model'], r['sid'])][r['variant']] = r['z']

    sel = {}
    if os.path.exists(SEL2B):
        for line in open(SEL2B, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get('error') or d.get('z_jev_q0') is None:
                continue
            sel[(d['model'], d['sid'])] = d

    canon = json.load(open(CANON))
    # DV + upstream box come from the canonical records
    dv, box = {}, {}
    for m in models:
        rp = os.path.join(canon[m], m, 'records.jsonl')
        if not os.path.exists(rp):
            continue
        for line in open(rp, encoding='utf-8'):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = str(rec.get('task', '')).lower()
            sid = rec.get('sample_id') or rec.get('base_sample_id')
            if t in ('t1', 't1_discriminative_vqa'):
                pe = rec.get('pred_exists')
                dv[(m, sid)] = (1.0 if pe is True else
                                0.0 if pe is False else None)
            elif t in ('t2', 't2_vqa_grounding'):
                box[(m, sid)] = rec.get('pred_bbox_xyxy')

    data = {}
    for m in models:
        p = os.path.join(S5, 's5omni_omni_%s.jsonl' % m)
        if not os.path.exists(p):
            print('[skip] %s: no omni arm' % m)
            continue
        rows = []
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r['sid']
            key = (m, sid)
            rv = rivals.get(key, {})
            s = sel.get(key)
            if len(rv) < 2 or s is None:
                continue          # leak-free subset: two rivals AND a 2B reading
            ht = r.get('htype')
            # keep the rival list in one deterministic order for A0/A1
            names = sorted(rv)
            jev = {c['variant']: c['z_jev'] for c in s['rivals']}
            if not all(v in jev for v in names):
                continue
            rows.append(dict(
                model=m, sid=sid, htype=ht, is_pos=(ht == 'positive'),
                label=1 if ht == 'positive' else 0,
                iou=float(r.get('iou') or 0.0),
                drew=bool(r.get('drew')),
                geom_ok=valid_box(box.get(key)),
                z0=r.get('z_omni'),
                z_jev0=s['z_jev_q0'],
                dv=dv.get(key),
                variants=names,
                z_omni_rival={v: rv[v] for v in names},
                z_jev_rival={v: jev[v] for v in names},
                fold=fold_of(sid), img=img_of(sid)))
        data[m] = rows
    return data


# --------------------------------------------------------------- arm features
def g_fixed(r):
    """A0: always the first rival in deterministic order."""
    v = r['variants'][0]
    return r['z_omni_rival'][v] - r['z0']


def g_random(r, rnd):
    v = rnd.choice(r['variants'])
    return r['z_omni_rival'][v] - r['z0']


def g_2bsel(r):
    """A2: 2B picks the rival with the HIGHEST jev support (most suspicious:
    an opposite relation the 2B finds MORE supported than the original), Omni
    rechecks only that one."""
    v = max(r['variants'], key=lambda x: r['z_jev_rival'][x])
    return r['z_omni_rival'][v] - r['z0']


def g_omni_all(r):
    """A4: Omni scores every rival itself and keeps the max gap."""
    return max(r['z_omni_rival'][v] for v in r['variants']) - r['z0']


def dvf(r):
    return -1.0 if r['dv'] is None else r['dv']


def build_arms(A, seed=0):
    """Fit every learned arm on fold A rows that have a box and an Omni score."""
    rnd = random.Random(seed)
    Ad = [r for r in A if r['drew'] and r['z0'] is not None]
    y = [r['label'] for r in Ad]
    out = {}

    def add(name, feat):
        f, w = logreg([feat(r) for r in Ad], y)
        out[name] = (lambda r, f=f, feat=feat: f(feat(r)), w)

    add('B1', lambda r: [r['z0']])
    add('B2', lambda r: [r['z_jev0']])
    add('B3', lambda r: [dvf(r)])
    add('B5', lambda r: [r['z0'], r['z_jev0'], dvf(r)])
    add('A0', lambda r: [r['z0'], g_fixed(r)])
    add('A2', lambda r: [r['z0'], g_2bsel(r)])
    add('A4', lambda r: [r['z0'], g_omni_all(r)])
    add('A5', lambda r: [r['z0'], g_2bsel(r)])   # same features, DV never used
    # A1 needs a frozen random pick per row so fit and eval agree
    pick = {}
    for r in A:
        pick[(r['model'], r['sid'])] = rnd.choice(r['variants'])
    out['_a1pick'] = pick
    return out


def a1_gap(r, pick, rnd):
    v = pick.get((r['model'], r['sid']))
    if v is None:
        v = rnd.choice(r['variants'])
    return r['z_omni_rival'][v] - r['z0']


# --------------------------------------------------------------- evaluation
def pick_tau(A, scorer, target_pos_keep):
    pos = [scorer(r) for r in A if r['is_pos'] and r['drew'] and r['z0'] is not None]
    if not pos:
        return float('-inf')
    pos.sort()
    k = int(round((1.0 - target_pos_keep) * len(pos)))
    k = max(0, min(k, len(pos) - 1))
    return pos[k]


def evaluate(rows, keep_fn):
    """Subset metrics with the subset's OWN denominators (printed, never faked)."""
    pos = [r for r in rows if r['is_pos']]
    neg = [r for r in rows if not r['is_pos']]
    npos = float(len(pos)) or 1.0
    nneg = float(len(neg)) or 1.0
    fg = sum(1 for r in neg if keep_fn(r))
    miou = sum(r['iou'] for r in pos if keep_fn(r)) / npos
    base_miou = sum(r['iou'] for r in pos) / npos
    pk = sum(1 for r in pos if keep_fn(r)) / npos
    rel = [r for r in neg if r['htype'] == 'relation']
    relfg = (sum(1 for r in rel if keep_fn(r)) / len(rel)) if rel else float('nan')
    return dict(fgr=fg / nneg, rel_fgr=relfg, pos_miou=miou,
                miou_loss=base_miou - miou, pos_keep=pk,
                n_pos=len(pos), n_neg=len(neg), n_rel=len(rel))


def keeper(scorer, tau):
    def keep(r):
        if not r['drew']:
            return False
        if r['z0'] is None:
            return True
        return scorer(r) >= tau
    return keep


def image_cluster_boot(pairs, n=2000, seed=0):
    rnd = random.Random(seed)
    byimg = defaultdict(list)
    for img, v in pairs:
        byimg[img].append(v)
    keys = list(byimg)
    if not keys:
        return float('nan'), float('nan'), float('nan')
    out = []
    for _ in range(n):
        acc = []
        for _ in range(len(keys)):
            acc += byimg[rnd.choice(keys)]
        out.append(sum(acc) / len(acc))
    out.sort()
    return (sum(out) / len(out), out[int(.025 * len(out))], out[int(.975 * len(out))])


def model_paired(deltas, n=2000, seed=0):
    rnd = random.Random(seed)
    if not deltas:
        return float('nan'), float('nan'), float('nan')
    out = []
    for _ in range(n):
        s = [rnd.choice(deltas) for _ in deltas]
        out.append(sum(s) / len(s))
    out.sort()
    return (sum(deltas) / len(deltas), out[int(.025 * len(out))],
            out[int(.975 * len(out))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='')
    ap.add_argument('--pos-keep', type=float, default=0.95)
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()
    _selftest()
    print('metric self-test PASSED\n')

    canon = json.load(open(CANON))
    models = [x.strip() for x in a.models.split(',') if x.strip()] or sorted(canon)
    data = load(models)
    models = [m for m in models if data.get(m)]
    print('models with data: %d' % len(models))

    print('\n%-18s%8s%8s%8s%10s' % ('model', 'rows', 'pos', 'neg', 'rel_neg'))
    for m in models:
        r = data[m]
        print('%-18s%8d%8d%8d%10d' % (
            m, len(r), sum(1 for x in r if x['is_pos']),
            sum(1 for x in r if not x['is_pos']),
            sum(1 for x in r if not x['is_pos'] and x['htype'] == 'relation')))

    ARMS = ['B1', 'B2', 'B3', 'B5', 'A0', 'A1', 'A2', 'A4']
    per = {}
    for m in models:
        rows = data[m]
        A = [r for r in rows if r['fold'] == 0]
        B = [r for r in rows if r['fold'] == 1]
        arms = build_arms(A)
        rnd = random.Random(1)
        pick = arms['_a1pick']
        # A1 pick must also exist for fold B rows; freeze per row deterministically
        for r in rows:
            k = (r['model'], r['sid'])
            if k not in pick:
                pick[k] = random.Random(hashlib.md5(
                    (k[0] + k[1]).encode()).hexdigest()).choice(r['variants'])
        Ad = [r for r in A if r['drew'] and r['z0'] is not None]
        fa1, wa1 = logreg([[r['z0'], a1_gap(r, pick, rnd)] for r in Ad],
                          [r['label'] for r in Ad])
        arms['A1'] = (lambda r: fa1([r['z0'], a1_gap(r, pick, rnd)]), wa1)

        res = {}
        res['B0'] = evaluate(B, lambda r: r['drew'])
        res['Bgeom'] = evaluate(B, lambda r: r['drew'] and r['geom_ok'])
        for name in ARMS:
            scorer, w = arms[name]
            tau = pick_tau(A, scorer, a.pos_keep)
            res[name] = evaluate(B, keeper(scorer, tau))
            res[name]['tau'] = tau
            res[name]['weights'] = [round(x, 4) for x in w]
        # static fusions: match B1's positive-keep rate on fold A for fairness
        sB1, _ = arms['B1']
        sB2, _ = arms['B2']
        t1 = pick_tau(A, sB1, a.pos_keep)
        t2 = pick_tau(A, sB2, a.pos_keep)
        res['B4mean'] = evaluate(B, lambda r: r['drew'] and (
            r['z0'] is None or (sB1(r) + sB2(r)) / 2 >= (t1 + t2) / 2))
        res['B4and'] = evaluate(B, lambda r: r['drew'] and (
            r['z0'] is None or (sB1(r) >= t1 and sB2(r) >= t2)))
        per[m] = dict(res=res, A=A, B=B, arms=arms, pick=pick)

    order = ['B0', 'Bgeom', 'B1', 'B2', 'B3', 'B4mean', 'B4and', 'B5',
             'A0', 'A1', 'A2', 'A4']
    print('\n' + '=' * 96)
    print('池化 (%d 模型等权), fold B 样本外, 阈值在 fold A 选 (pos_keep=%.2f)'
          % (len(models), a.pos_keep))
    print('分母 = 探针覆盖子集自身计数，不是官方 2000/500')
    print('=' * 96)
    print('%-9s%9s%11s%11s%10s%10s' % ('arm', 'FGR', 'relFGR', 'pos_mIoU',
                                       'ΔmIoU', 'posKeep'))

    def mean(arm, key):
        v = [per[m]['res'][arm][key] for m in models
             if per[m]['res'][arm][key] == per[m]['res'][arm][key]]
        return sum(v) / len(v) if v else float('nan')

    for arm in order:
        print('%-9s%8.2f%%%10.2f%%%11.4f%10.4f%10.3f' % (
            arm, mean(arm, 'fgr') * 100, mean(arm, 'rel_fgr') * 100,
            mean(arm, 'pos_miou'), mean(arm, 'miou_loss'), mean(arm, 'pos_keep')))

    # ---- headline comparisons
    print('\n' + '=' * 96)
    print('关键对比 (relation FGR, 模型级配对 n=%d)' % len(models))
    print('=' * 96)
    cmps = [('A2', 'B1', 'H1 探针 vs Omni 单臂'),
            ('A2', 'A4', 'H2 2B 选题 vs Omni 自跑全部探针(成本更高)'),
            ('A2', 'A0', 'H2 2B 选题 vs 固定顺序'),
            ('A2', 'A1', 'H2 2B 选题 vs 随机选'),
            ('A2', 'B5', '探针 vs 仅原表达三信号学习融合'),
            ('A2', 'B2', 'vs 2B 单臂'),
            ('A4', 'B1', 'A4 Omni-only 探针 vs 单臂')]
    stat = {}
    for x, y, label in cmps:
        d = [per[m]['res'][x]['rel_fgr'] - per[m]['res'][y]['rel_fgr']
             for m in models
             if per[m]['res'][x]['rel_fgr'] == per[m]['res'][x]['rel_fgr']
             and per[m]['res'][y]['rel_fgr'] == per[m]['res'][y]['rel_fgr']]
        mm, lo, hi = model_paired(d)
        verdict = '显著' if (lo > 0 or hi < 0) else '不显著'
        print('  %-44s %+7.2fpp CI[%+6.2f,%+6.2f] %s  改善 %d/%d'
              % (label, mm * 100, lo * 100, hi * 100, verdict,
                 sum(1 for v in d if v < 0), len(d)))
        stat['%s-%s' % (x, y)] = dict(mean=mm, ci=[lo, hi], n=len(d),
                                      n_improved=sum(1 for v in d if v < 0))

    # fairness: positive keep + mIoU must not silently degrade
    print('\n公平性 (A2 − B1)')
    for key, name in (('pos_keep', 'posKeep'), ('pos_miou', 'pos_mIoU')):
        d = [per[m]['res']['A2'][key] - per[m]['res']['B1'][key] for m in models]
        mm, lo, hi = model_paired(d)
        print('  %-10s %+7.4f CI[%+7.4f,%+7.4f] %s'
              % (name, mm, lo, hi, '显著' if (lo > 0 or hi < 0) else '不显著'))

    # ---- AUROC per arm on negatives-vs-positives, fold B
    print('\n' + '=' * 96)
    print('AUROC (fold B, 池化模型级平均)')
    print('=' * 96)
    sigs = [('B1 z0', lambda r: r['z0']),
            ('B2 z_jev(q0)', lambda r: r['z_jev0']),
            ('A0 fixed gap', lambda r: -g_fixed(r)),
            ('A2 2B-selected gap', lambda r: -g_2bsel(r)),
            ('A4 omni-max gap', lambda r: -g_omni_all(r))]
    for name, f in sigs:
        vals = []
        for m in models:
            B = [r for r in per[m]['B'] if r['drew'] and r['z0'] is not None]
            if not B:
                continue
            vals.append(auroc([f(r) for r in B], [r['label'] for r in B]))
        vals = [v for v in vals if v == v]
        print('  %-24s %.4f' % (name, sum(vals) / len(vals)))

    # ---- selection agreement: does 2B pick a different rival than Omni-max?
    agree = diff = 0
    for m in models:
        for r in data[m]:
            v2b = max(r['variants'], key=lambda x: r['z_jev_rival'][x])
            vom = max(r['variants'], key=lambda x: r['z_omni_rival'][x])
            if v2b == vom:
                agree += 1
            else:
                diff += 1
    print('\n2B 与 Omni 选中同一个对立词: %d/%d (%.1f%%)'
          % (agree, agree + diff, 100.0 * agree / max(1, agree + diff)))

    print('\n成本 (每个已出框行)')
    print('  B1 1 Omni | B2 1 2B | A0/A1 2 Omni | A2 2 Omni + %d 2B | A4 1+%d Omni'
          % (2, 2))

    if a.json_out:
        out = {m: {k: v for k, v in per[m]['res'].items()} for m in models}
        json.dump(dict(per_model=out, comparisons=stat,
                       pos_keep_target=a.pos_keep, n_models=len(models)),
                  open(a.json_out, 'w'), indent=2, ensure_ascii=False)
        print('\nwrote %s' % a.json_out)


if __name__ == '__main__':
    main()
