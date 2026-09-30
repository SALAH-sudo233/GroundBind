#!/usr/bin/env python3
"""Corrected H2 test + the plan's language-prior and visual controls.

WHY THIS FILE EXISTS. The first cross-pair run said A0 ("fixed probe order")
beat the 2B selector by 8.34pp. `diag_a0_leak.py` showed that result is an
artefact: sorting rivals alphabetically picks `behind` on 3921/6764 rows, and
the A0 pick identity alone -- no image, no verifier call -- scores AUROC 0.8651
because `behind` sits on the negative side of this benchmark's predicate
distribution (pick=behind -> 83.1% positive, `in front of` -> 13.6%,
`next to` -> 7.8%). A "control" that carries the label that strongly is not a
control, so cross-pair A0 numbers must not be quoted.

The clean comparison is WITHIN a candidate pair. Inside one pair the A0 pick is
constant by construction, so predicate identity cannot vary and any remaining
A0-vs-A2 difference is a genuine selection effect.

Arms compared here (all on the same rows, same folds, same calibration rights):
  z0        Omni on the original expression (B1 signal)
  A0        gap from the sorted-first rival
  A1        gap from a seeded random legal rival
  A2        gap from the rival the 2B/JEV selector picks
  A4        gap from max over ALL rivals (Omni does the work itself)
  Blex      predicate-surface log-likelihood ratio, fit on fold A, no image
  BlexO     Blex + z0, the plan's section 18.4 "is it just prior + Omni" check
  Aswap     A2's selected rival re-scored on a MISMATCHED image (section 18.6);
            tests whether the extra call needs THIS image

Protocol: folds = md5(image basename) % 2, priors/weights fit on fold A, fold B
read once, AUROC by rank-sum with tied scores averaged, CIs by bootstrap
clustered on SOURCE IMAGE (never on model x sample). Paired differences are
computed on the identical row set for both arms.
"""
import collections
import hashlib
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
S5 = os.path.expanduser('~/SVD/grpo_verifier/s5grpo')


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


def img_of(sid):
    b = str(sid).split('__')[0]
    i = b.find('COCO_')
    return (b[i:] if i >= 0 else b) + '.jpg'


def fold_of(sid):
    return int(hashlib.md5(img_of(sid).encode()).hexdigest(), 16) % 2


def boot_auroc_diff(rows, fa, fb, n=2000, seed=0):
    """Paired AUROC difference fa-fb, bootstrap clustered on source image."""
    rnd = random.Random(seed)
    byimg = collections.defaultdict(list)
    for r in rows:
        byimg[r['img']].append(r)
    keys = list(byimg)
    base = auroc([fa(r) for r in rows], [r['label'] for r in rows]) - \
        auroc([fb(r) for r in rows], [r['label'] for r in rows])
    out = []
    for _ in range(n):
        samp = []
        for _ in range(len(keys)):
            samp += byimg[rnd.choice(keys)]
        y = [r['label'] for r in samp]
        if len(set(y)) < 2:
            continue
        out.append(auroc([fa(r) for r in samp], y) -
                   auroc([fb(r) for r in samp], y))
    out.sort()
    if not out:
        return base, float('nan'), float('nan')
    return base, out[int(.025 * len(out))], out[int(.975 * len(out))]


def load():
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

    swap = collections.defaultdict(dict)
    p = os.path.join(HERE, 'swap_all.jsonl')
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            d = json.loads(line)
            if d.get('error') or d.get('z_swap') is None:
                continue
            swap[(d['model'], d['sid'])][d['variant']] = d['z_swap']

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
            if len(rv) < 2 or s is None:
                continue
            if not r.get('drew') or r.get('z_omni') is None:
                continue
            names = sorted(rv)
            jev = {c['variant']: c['z_jev'] for c in s['rivals']}
            if not all(v in jev for v in names):
                continue
            q0 = s.get('q0') or ''
            rows.append(dict(
                model=m, sid=r['sid'], img=img_of(r['sid']),
                label=1 if r.get('htype') == 'positive' else 0,
                z0=r['z_omni'], zo=rv, jev=jev, names=tuple(names),
                pair=tuple(names), q0=q0,
                zswap=swap.get(key, {}),
                fold=fold_of(r['sid'])))
    return rows


def orig_predicate(q0, names):
    """The predicate in the ORIGINAL query: the audited rival table maps it to
    these variants, so the original surface is the one NOT among the rivals."""
    ql = (q0 or '').lower()
    for p in ('in front of', 'on top of', 'next to', 'behind', 'below',
              'above', 'under', 'beside', 'near', 'on'):
        if p in ql:
            return p
    return '?'


def main():
    rows = load()
    A = [r for r in rows if r['fold'] == 0]
    B = [r for r in rows if r['fold'] == 1]
    print('rows=%d  foldA=%d  foldB=%d' % (len(rows), len(A), len(B)))
    n_swap = sum(1 for r in rows if r['zswap'])
    print('rows with Aswap scores: %d' % n_swap)

    # ---------- arm score functions (higher = more positive-looking)
    def f_z0(r):
        return r['z0']

    def f_a0(r):
        return -(r['zo'][r['names'][0]] - r['z0'])

    def f_a2(r):
        v = max(r['names'], key=lambda x: r['jev'][x])
        return -(r['zo'][v] - r['z0'])

    def f_a4(r):
        return -(max(r['zo'][v] for v in r['names']) - r['z0'])

    rnd_pick = {}
    for r in rows:
        rnd_pick[(r['model'], r['sid'])] = random.Random(
            hashlib.md5((r['model'] + r['sid']).encode()).hexdigest()
        ).choice(list(r['names']))

    def f_a1(r):
        v = rnd_pick[(r['model'], r['sid'])]
        return -(r['zo'][v] - r['z0'])

    def f_swap(r):
        v = max(r['names'], key=lambda x: r['jev'][x])
        z = r['zswap'].get(v)
        if z is None:
            return 0.0
        return -(z - r['z0'])

    # Blex: predicate-surface LLR fitted on fold A only
    cnt = collections.Counter()
    pos = collections.Counter()
    for r in A:
        p = orig_predicate(r['q0'], r['names'])
        cnt[p] += 1
        pos[p] += r['label']
    llr = {p: math.log((pos[p] + 1.0) / (cnt[p] - pos[p] + 1.0)) for p in cnt}
    dflt = math.log((sum(pos.values()) + 1.0) /
                    (sum(cnt.values()) - sum(pos.values()) + 1.0))

    def f_blex(r):
        return llr.get(orig_predicate(r['q0'], r['names']), dflt)

    # ---------- cross-pair table, with the A0 caveat attached
    print('\n' + '=' * 84)
    print('fold B 全子集 AUROC（跨候选对；A0 因选中词泄漏不可作对照，见下）')
    print('=' * 84)
    arms = [('z0  (B1 Omni 单臂)', f_z0), ('Blex 原谓词先验(不看图)', f_blex),
            ('A0  排序首个(泄漏)', f_a0), ('A1  随机选', f_a1),
            ('A2  2B 选题', f_a2), ('A4  Omni 自跑全部', f_a4)]
    y = [r['label'] for r in B]
    for name, f in arms:
        print('  %-26s %.4f' % (name, auroc([f(r) for r in B], y)))
    Bs = [r for r in B if r['zswap']]
    if Bs:
        print('  %-26s %.4f  (n=%d)'
              % ('Aswap 错图复核', auroc([f_swap(r) for r in Bs],
                                     [r['label'] for r in Bs]), len(Bs)))

    # ---------- the corrected H2: within candidate pair
    print('\n' + '=' * 84)
    print('修正后的 H2：同候选对内部（A0 选中词恒定 → 谓词身份被控制）')
    print('=' * 84)
    bypair = collections.defaultdict(list)
    for r in B:
        bypair[r['pair']].append(r)
    print('%-26s%6s%6s%8s%8s%8s%8s%8s' %
          ('pair', 'n', 'pos', 'z0', 'A0', 'A1', 'A2', 'A4'))
    keep = []
    for pair, sub in sorted(bypair.items(), key=lambda kv: -len(kv[1])):
        yy = [r['label'] for r in sub]
        if len(set(yy)) < 2 or len(sub) < 30:
            continue
        keep.append((pair, sub))
        print('%-26s%6d%6d%8.4f%8.4f%8.4f%8.4f%8.4f'
              % ('+'.join(pair), len(sub), sum(yy),
                 auroc([f_z0(r) for r in sub], yy),
                 auroc([f_a0(r) for r in sub], yy),
                 auroc([f_a1(r) for r in sub], yy),
                 auroc([f_a2(r) for r in sub], yy),
                 auroc([f_a4(r) for r in sub], yy)))

    print('\n同对内配对差（图像聚类 bootstrap 2000 次）')
    for label, fa, fb in (('A2 − A0  2B 选题 vs 固定顺序', f_a2, f_a0),
                          ('A2 − A1  2B 选题 vs 随机', f_a2, f_a1),
                          ('A2 − A4  2B 选题 vs Omni 自跑(更贵)', f_a2, f_a4),
                          ('A2 − z0  探针 vs Omni 单臂', f_a2, f_z0),
                          ('A4 − z0  Omni 探针 vs 单臂', f_a4, f_z0)):
        print('  %s' % label)
        for pair, sub in keep:
            d, lo, hi = boot_auroc_diff(sub, fa, fb)
            v = '显著' if (lo > 0 or hi < 0) else '不显著'
            print('      %-24s %+.4f CI[%+.4f,%+.4f] %s'
                  % ('+'.join(pair), d, lo, hi, v))

    # ---------- language-prior controls (plan 18.4)
    print('\n' + '=' * 84)
    print('词先验对照（plan §18.4）：探针收益能否被「Omni + 谓词先验」解释')
    print('=' * 84)
    for pair, sub in keep:
        yy = [r['label'] for r in sub]
        print('  %-24s n=%4d  Blex=%.4f  z0=%.4f  A2gap=%.4f'
              % ('+'.join(pair), len(sub),
                 auroc([f_blex(r) for r in sub], yy),
                 auroc([f_z0(r) for r in sub], yy),
                 auroc([f_a2(r) for r in sub], yy)))
    print('  注：同对内所有行的原谓词几乎相同 → Blex 接近 0.5 即证明该对内无词先验可用。')

    # ---------- Aswap visual control (plan 18.6)
    if Bs:
        print('\n' + '=' * 84)
        print('Aswap 视觉对照（plan §18.6）：追加核验是否需要「当前这张图」')
        print('=' * 84)
        d, lo, hi = boot_auroc_diff(Bs, f_a2, f_swap)
        print('  A2(真图) − Aswap(错图)  %+.4f CI[%+.4f,%+.4f]  %s  n=%d'
              % (d, lo, hi, '显著' if (lo > 0 or hi < 0) else '不显著', len(Bs)))
        zr = [r['zo'][max(r['names'], key=lambda x: r['jev'][x])] for r in Bs]
        zs = [r['zswap'][max(r['names'], key=lambda x: r['jev'][x])]
              for r in Bs if r['zswap'].get(
                  max(r['names'], key=lambda x: r['jev'][x])) is not None]
        print('  选中探针分数均值：真图 %+.3f | 错图 %+.3f'
              % (sum(zr) / len(zr), sum(zs) / len(zs)))

    # ---------- selection agreement
    ag = sum(1 for r in rows
             if max(r['names'], key=lambda x: r['jev'][x]) ==
             max(r['names'], key=lambda x: r['zo'][x]))
    print('\n2B 与 Omni 选中同一个对立词: %d/%d (%.1f%%)'
          % (ag, len(rows), 100.0 * ag / len(rows)))


if __name__ == '__main__':
    main()
