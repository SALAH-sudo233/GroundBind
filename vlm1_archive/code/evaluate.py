#!/usr/bin/env python3
"""Evaluate the agentic competitive-probe framework. Fit fold A, read fold B once.

Arms (plan v0.2 sec 8). All share the same rows, folds and fitting budget:
  B1      Omni original expression only              -- main reference
  Blex    predicate log-likelihood ratio, NO image   -- shortcut control
  BlexO   Omni + predicate prior                     -- shortcut+vision control
  A0      first rival only + Omni                    -- does a probe help at all
  A2      all legal rivals, max aggregation          -- full framework
  A2lex   A2 + predicate prior                       -- deployment-realistic

Reported on: full set, supported subset, per htype, and the predicate-balanced
subset (where the lexical prior is zeroed by construction).
CIs: paired bootstrap clustered on source image, 2000 reps, fixed seed.
"""
import argparse, json, math, random, sys
from collections import defaultdict, Counter

HT4 = ['object', 'co_occurrence', 'attribute', 'relation']


# ---------------------------------------------------------------- metrics
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


def ece(ps, ys, bins=10):
    if not ps:
        return float('nan')
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in zip(ps, ys)
               if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not sel:
            continue
        conf = sum(p for p, _ in sel) / len(sel)
        acc = sum(y for _, y in sel) / len(sel)
        tot += len(sel) / len(ps) * abs(conf - acc)
    return tot


def logreg(X, y, l2=1.0, iters=300):
    """Newton with L2, standardised inputs. Returns predict(list)->p."""
    k = len(X[0])
    n = len(X)
    mu = [sum(r[j] for r in X) / n for j in range(k)]
    sd = [((sum((r[j] - mu[j]) ** 2 for r in X) / n) ** 0.5) or 1.0
          for j in range(k)]
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


def cluster_boot(rows, s1, s2, n=2000, seed=0):
    """Paired AUROC difference, resampling source images."""
    rnd = random.Random(seed)
    byimg = defaultdict(list)
    for i, r in enumerate(rows):
        byimg[r['img']].append(i)
    keys = list(byimg)
    out = []
    for _ in range(n):
        idx = []
        for _ in range(len(keys)):
            idx += byimg[rnd.choice(keys)]
        y = [rows[i]['y'] for i in idx]
        if not (0 < sum(y) < len(y)):
            continue
        out.append(auroc([s1[i] for i in idx], y) -
                   auroc([s2[i] for i in idx], y))
    out.sort()
    if not out:
        return float('nan'), float('nan'), float('nan')
    return (sum(out) / len(out), out[int(.025 * len(out))],
            out[int(.975 * len(out))])


def _selftest():
    assert abs(auroc([1, 1, 1, 1], [1, 1, 0, 0]) - 0.5) < 1e-12
    assert abs(auroc([2, 1], [1, 0]) - 1.0) < 1e-12
    assert abs(auroc([2, 2, 1], [1, 0, 0]) - 0.75) < 1e-12


# ---------------------------------------------------------------- load
def load(path):
    """Group the per-query scores back into one record per (sid, polarity)."""
    by = defaultdict(dict)
    meta = {}
    nerr = 0
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('error'):
            nerr += 1
            continue
        k = (r['sid'], r['polarity'])
        by[k][r['variant']] = r['z']
        meta[k] = dict(y=r['label'], img=r['image_filename'],
                       ht=r['htype'], pred=r['predicate'] or '<none>',
                       fam=r['family'], supported=r['supported'])
    rows = []
    for k, zs in by.items():
        if zs.get('original') is None:
            continue
        m = meta[k]
        rivals = [v for name, v in zs.items()
                  if name != 'original' and v is not None]
        rows.append(dict(
            sid=k[0], pol=k[1], y=m['y'], img=m['img'], ht=m['ht'],
            pred=m['pred'], fam=m['fam'],
            z0=zs['original'],
            rivals=rivals,
            zmax=max(rivals) if rivals else None,
            zfirst=rivals[0] if rivals else None,
            supported=bool(rivals),
        ))
    rows.sort(key=lambda r: (r['sid'], r['pol']))
    return rows, nerr


def fold_of(img):
    import hashlib
    return int(hashlib.md5(img.encode()).hexdigest(), 16) % 2


# ---------------------------------------------------------------- arms
def build_arms(A, B):
    """Fit every arm on A, return {name: scores_on_B} plus fitted weights."""
    cp, cn = Counter(), Counter()
    for r in A:
        (cp if r['y'] == 1 else cn)[r['pred']] += 1

    def blex(p, al=1.0):
        return math.log((cp[p] + al) / (cn[p] + al))

    yA = [r['y'] for r in A]
    arms, info = {}, {}

    arms['B1 Omni 原表达'] = [r['z0'] for r in B]
    arms['Blex 谓词先验(不看图)'] = [blex(r['pred']) for r in B]

    f, w = logreg([[r['z0'], blex(r['pred'])] for r in A], yA)
    arms['BlexO Omni+谓词先验'] = [f([r['z0'], blex(r['pred'])]) for r in B]
    info['BlexO'] = w

    def gap_first(r):
        return (r['zfirst'] - r['z0']) if r['supported'] else 0.0

    def gap_max(r):
        return (r['zmax'] - r['z0']) if r['supported'] else 0.0

    f, w = logreg([[r['z0'], gap_first(r), 1.0 if r['supported'] else 0.0]
                   for r in A], yA)
    arms['A0 首个对立词'] = [f([r['z0'], gap_first(r),
                          1.0 if r['supported'] else 0.0]) for r in B]
    info['A0'] = w

    f, w = logreg([[r['z0'], gap_max(r), 1.0 if r['supported'] else 0.0]
                   for r in A], yA)
    arms['A2 全部对立词(max)'] = [f([r['z0'], gap_max(r),
                              1.0 if r['supported'] else 0.0]) for r in B]
    info['A2'] = w

    f, w = logreg([[r['z0'], gap_max(r), 1.0 if r['supported'] else 0.0,
                    blex(r['pred'])] for r in A], yA)
    arms['A2+谓词先验'] = [f([r['z0'], gap_max(r),
                        1.0 if r['supported'] else 0.0,
                        blex(r['pred'])]) for r in B]
    info['A2lex'] = w
    return arms, info, blex


def report(rows_B, arms, title, order=None):
    y = [r['y'] for r in rows_B]
    print(f"\n=== {title} (n={len(rows_B)}, pos {sum(y)}/neg {len(y)-sum(y)}) ===")
    names = order or list(arms)
    for k in names:
        v = arms[k]
        pv = v if (min(v) >= 0 and max(v) <= 1) else None
        e = f"  ECE {ece(pv, y):.4f}" if pv else ""
        print(f"  {k:26s} AUROC {auroc(v, y):.4f}{e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()
    _selftest()
    print("metric self-test PASSED (ties -> 0.5)")

    rows, nerr = load(a.scores)
    print(f"rows={len(rows)}  dropped_errors={nerr}")
    print("htype:", dict(Counter(r['ht'] for r in rows)))
    ns = sum(1 for r in rows if r['supported'])
    print(f"supported (有合法对立词): {ns}/{len(rows)} = {ns/len(rows):.1%}")
    print("rival 数分布:", dict(Counter(len(r['rivals']) for r in rows)))

    A = [r for r in rows if fold_of(r['img']) == 0]
    B = [r for r in rows if fold_of(r['img']) == 1]
    print(f"fold A {len(A)} / fold B {len(B)}")

    arms, info, blex = build_arms(A, B)
    ORDER = ['Blex 谓词先验(不看图)', 'B1 Omni 原表达', 'BlexO Omni+谓词先验',
             'A0 首个对立词', 'A2 全部对立词(max)', 'A2+谓词先验']
    print("\n拟合权重 (fold A):")
    for k, w in info.items():
        print(f"  {k:8s} {[round(x,3) for x in w]}")

    report(B, arms, "全 4 类 fold B", ORDER)

    # supported subset -- where the framework actually acts
    idx = [i for i, r in enumerate(B) if r['supported']]
    if idx:
        sub = [B[i] for i in idx]
        sarms = {k: [v[i] for i in idx] for k, v in arms.items()}
        report(sub, sarms, "supported 子集(有对立词)", ORDER)

    # per htype
    for ht in HT4:
        idx = [i for i, r in enumerate(B) if r['ht'] == ht]
        if len(idx) < 30:
            continue
        sub = [B[i] for i in idx]
        sarms = {k: [v[i] for i in idx] for k, v in arms.items()}
        report(sub, sarms, f"htype={ht}", ORDER)

    # predicate-balanced subset: lexical prior zeroed by construction
    idx_sup = [i for i, r in enumerate(B) if r['supported']]
    byp = defaultdict(lambda: {1: [], 0: []})
    for i in idx_sup:
        byp[B[i]['pred']][B[i]['y']].append(i)
    rnd = random.Random(0)
    bal = []
    for p, g in byp.items():
        k = min(len(g[1]), len(g[0]))
        if k:
            bal += rnd.sample(g[1], k) + rnd.sample(g[0], k)
    if len(bal) >= 40:
        sub = [B[i] for i in bal]
        sarms = {k: [v[i] for i in bal] for k, v in arms.items()}
        report(sub, sarms, "谓词平衡子集(词先验归零)", ORDER)

    # paired bootstrap on the supported subset
    idx = idx_sup
    sub = [B[i] for i in idx]
    sarms = {k: [v[i] for i in idx] for k, v in arms.items()}
    print(f"\n=== 图像聚类配对 bootstrap, supported 子集 n={len(sub)} (2000 次) ===")
    for x, base in [('A2 全部对立词(max)', 'B1 Omni 原表达'),
                    ('A2 全部对立词(max)', 'A0 首个对立词'),
                    ('A2 全部对立词(max)', 'Blex 谓词先验(不看图)'),
                    ('A2+谓词先验', 'BlexO Omni+谓词先验'),
                    ('B1 Omni 原表达', 'Blex 谓词先验(不看图)')]:
        m, lo, hi = cluster_boot(sub, sarms[x], sarms[base])
        tag = '显著' if (lo > 0 or hi < 0) else '不显著'
        print(f"  {x[:20]:20s} − {base[:20]:20s} {m:+.4f} "
              f"CI[{lo:+.4f},{hi:+.4f}] {tag}")

    if a.json_out:
        out = {'n_rows': len(rows), 'fold_A': len(A), 'fold_B': len(B),
               'supported_rate': ns / len(rows),
               'weights': {k: w for k, w in info.items()},
               'foldB_all': {k: auroc(v, [r['y'] for r in B])
                             for k, v in arms.items()}}
        json.dump(out, open(a.json_out, 'w'), indent=2, ensure_ascii=False)
        print(f"\nwrote {a.json_out}")


if __name__ == '__main__':
    main()
