#!/usr/bin/env python3
"""H3: does DV (the upstream's own t1 yes/no) help allocate verification budget?

This closes a gap in eval_arms.py. There, arm "A5" was declared as "A2 minus DV"
but was given the SAME feature vector as A2, because DV never entered the A2 head
in the first place. Two arms with identical features produce identical numbers,
which is not an ablation -- it is a no-op, and per the project's own rule ("if an
arm's implementation difference does not change the decision set, that arm tested
nothing") it must not be reported as evidence about DV.

So H3 is tested the only way the cached data supports: as a feature increment on
the decision head, measured WITHIN a candidate pair (where predicate identity is
constant, so the strong language prior cannot masquerade as signal -- see
diag_a0_leak.py for why cross-pair numbers are unusable here).

  head0 = [z0]                      Omni alone
  headP = [z0, gap_2b_selected]      + probe
  headD = [z0, gap, dv]              + probe + DV
DV is encoded as 1/0 for true/false and a separate missing flag; `missing` is
never folded into `false` (plan v0.2 section 2.3).
"""
import collections
import hashlib
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
S5 = os.path.expanduser('~/SVD/grpo_verifier/s5grpo')
CANON = os.path.join(HERE, 'canon_roots.json')


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


def logreg(X, y, l2=1.0, iters=200):
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
    b = str(sid).split('__')[0]
    i = b.find('COCO_')
    return (b[i:] if i >= 0 else b) + '.jpg'


def fold_of(sid):
    return int(hashlib.md5(img_of(sid).encode()).hexdigest(), 16) % 2


def boot_diff(rows, fa, fb, n=2000, seed=0):
    rnd = random.Random(seed)
    byimg = collections.defaultdict(list)
    for r in rows:
        byimg[r['img']].append(r)
    keys = list(byimg)
    y = [r['label'] for r in rows]
    base = auroc([fa(r) for r in rows], y) - auroc([fb(r) for r in rows], y)
    out = []
    for _ in range(n):
        s = []
        for _ in range(len(keys)):
            s += byimg[rnd.choice(keys)]
        yy = [r['label'] for r in s]
        if len(set(yy)) < 2:
            continue
        out.append(auroc([fa(r) for r in s], yy) - auroc([fb(r) for r in s], yy))
    out.sort()
    if not out:
        return base, float('nan'), float('nan')
    return base, out[int(.025 * len(out))], out[int(.975 * len(out))]


def main():
    riv = collections.defaultdict(dict)
    for line in open(os.path.join(HERE, 'probe_upstream.jsonl'), encoding='utf-8'):
        d = json.loads(line)
        if d.get('error') or d.get('z') is None:
            continue
        riv[(d['model'], d['sid'])][d['variant']] = d['z']

    sel = {}
    for line in open(os.path.join(HERE, 'sel2b_all.jsonl'), encoding='utf-8'):
        d = json.loads(line)
        if d.get('error') or d.get('z_jev_q0') is None:
            continue
        sel[(d['model'], d['sid'])] = d

    canon = json.load(open(CANON))
    dv = {}
    for m in sorted(canon):
        rp = os.path.join(canon[m], m, 'records.jsonl')
        if not os.path.exists(rp):
            continue
        for line in open(rp, encoding='utf-8'):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if str(rec.get('task', '')).lower() not in ('t1', 't1_discriminative_vqa'):
                continue
            sid = rec.get('sample_id') or rec.get('base_sample_id')
            pe = rec.get('pred_exists')
            dv[(m, sid)] = (1.0 if pe is True else 0.0 if pe is False else None)

    rows = []
    for f in sorted(os.listdir(S5)):
        if not f.startswith('s5omni_omni_'):
            continue
        m = f[len('s5omni_omni_'):-len('.jsonl')]
        for line in open(os.path.join(S5, f), encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (m, r['sid'])
            rv = riv.get(key, {})
            s = sel.get(key)
            if len(rv) < 2 or s is None or not r.get('drew') or r.get('z_omni') is None:
                continue
            names = sorted(rv)
            jev = {c['variant']: c['z_jev'] for c in s['rivals']}
            if not all(v in jev for v in names):
                continue
            v = max(names, key=lambda x: jev[x])
            d = dv.get(key)
            rows.append(dict(
                model=m, sid=r['sid'], img=img_of(r['sid']),
                label=1 if r.get('htype') == 'positive' else 0,
                z0=r['z_omni'], gap=rv[v] - r['z_omni'],
                dv=(0.0 if d is None else d),
                dv_missing=(1.0 if d is None else 0.0),
                pair=tuple(names), fold=fold_of(r['sid'])))

    n_miss = sum(1 for r in rows if r['dv_missing'])
    print('rows=%d  DV missing=%d (%.2f%%)'
          % (len(rows), n_miss, 100.0 * n_miss / max(1, len(rows))))
    print('DV=true rate: %.3f  (label=positive rate: %.3f)'
          % (sum(r['dv'] for r in rows) / len(rows),
             sum(r['label'] for r in rows) / len(rows)))

    A = [r for r in rows if r['fold'] == 0]
    B = [r for r in rows if r['fold'] == 1]

    print('\n' + '=' * 80)
    print('H3：DV 作为决策头的特征增量（同候选对内部，谓词身份被控制）')
    print('=' * 80)
    bypair = collections.defaultdict(list)
    for r in B:
        bypair[r['pair']].append(r)
    byA = collections.defaultdict(list)
    for r in A:
        byA[r['pair']].append(r)

    print('%-24s%6s%9s%9s%9s   %s' %
          ('pair', 'n', 'z0', '+probe', '+DV', 'DV 增量 CI'))
    for pair, sub in sorted(bypair.items(), key=lambda kv: -len(kv[1])):
        if len(sub) < 30:
            continue
        Ap = byA.get(pair, [])
        yy = [r['label'] for r in sub]
        if len(set(yy)) < 2 or len(Ap) < 30 or len(set(r['label'] for r in Ap)) < 2:
            continue
        ya = [r['label'] for r in Ap]
        f0, _ = logreg([[r['z0']] for r in Ap], ya)
        fp, _ = logreg([[r['z0'], r['gap']] for r in Ap], ya)
        fd, wd = logreg([[r['z0'], r['gap'], r['dv'], r['dv_missing']]
                         for r in Ap], ya)

        def s0(r):
            return f0([r['z0']])

        def sp(r):
            return fp([r['z0'], r['gap']])

        def sd(r):
            return fd([r['z0'], r['gap'], r['dv'], r['dv_missing']])

        d, lo, hi = boot_diff(sub, sd, sp)
        print('%-24s%6d%9.4f%9.4f%9.4f   %+.4f CI[%+.4f,%+.4f] %s'
              % ('+'.join(pair), len(sub),
                 auroc([s0(r) for r in sub], yy),
                 auroc([sp(r) for r in sub], yy),
                 auroc([sd(r) for r in sub], yy),
                 d, lo, hi, '显著' if (lo > 0 or hi < 0) else '不显著'))
        print('%28s DV 权重=%+.3f  missing 权重=%+.3f' % ('', wd[2], wd[3]))


if __name__ == '__main__':
    main()
