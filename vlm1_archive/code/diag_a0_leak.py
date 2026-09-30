#!/usr/bin/env python3
"""Is A0's "fixed probe order" a neutral control, or does it leak the predicate?

A0 in eval_arms.py picks the first rival in SORTED order. The pick distribution is
`behind` 3923 / `in front of` 1985 / `next to` 645, and `behind` is the dominant
NEGATIVE-side predicate in this benchmark (168 neg vs 23 pos over 500 relation
groups). If "which rival got chosen" correlates with the label, then A0's gap
feature carries predicate identity, i.e. the language prior the plan's Blex /
BlexO controls exist to isolate (plan v0.2 section 18.4). A control arm that
smuggles in the prior is not a control.

Three checks, all on the same leak-free subset:
  1. label rate per A0-chosen rival, and per 2B-chosen rival
  2. Blex-style arm built ONLY from the chosen rival's surface form (no image, no
     verifier score). If this alone separates, the choice identity is predictive.
  3. within-pair analysis: restrict to ONE candidate pair at a time, where the
     A0 pick is CONSTANT by construction, so predicate identity cannot vary.
     Any surviving A0-vs-A2 difference there is a real selection effect.
Fold discipline is unchanged: fit on fold A, read fold B once.
"""
import collections
import hashlib
import json
import math
import os

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
            rows.append(dict(
                model=m, sid=r['sid'], label=1 if r.get('htype') == 'positive' else 0,
                z0=r['z_omni'], names=names,
                a0=names[0], a2=max(names, key=lambda x: jev[x]),
                zo=rv, fold=fold_of(r['sid'])))

    print('rows=%d' % len(rows))

    # ---- 1. label rate conditioned on the chosen rival
    for tag in ('a0', 'a2'):
        cnt = collections.Counter()
        pos = collections.Counter()
        for r in rows:
            cnt[r[tag]] += 1
            pos[r[tag]] += r['label']
        print('\n%s pick -> positive rate' % tag.upper())
        for v, n in cnt.most_common():
            print('  %-14s n=%5d  pos_rate=%.3f' % (v, n, pos[v] / n))

    # ---- 2. choice identity alone as a predictor (no image, no verifier)
    print('\n' + '=' * 70)
    print('选中词身份单独作为预测器 (不看图、不用验证器分数)')
    print('=' * 70)
    A = [r for r in rows if r['fold'] == 0]
    B = [r for r in rows if r['fold'] == 1]
    for tag in ('a0', 'a2'):
        cnt = collections.Counter()
        pos = collections.Counter()
        for r in A:
            cnt[r[tag]] += 1
            pos[r[tag]] += r['label']
        llr = {v: math.log((pos[v] + 1.0) / (cnt[v] - pos[v] + 1.0)) for v in cnt}
        default = math.log((sum(pos.values()) + 1.0) /
                           (sum(cnt.values()) - sum(pos.values()) + 1.0))
        sc = [llr.get(r[tag], default) for r in B]
        print('  %s 选中词 LLR  AUROC(fold B) = %.4f'
              % (tag.upper(), auroc(sc, [r['label'] for r in B])))

    # ---- 3. within-pair: A0 pick is constant, so identity cannot vary
    print('\n' + '=' * 70)
    print('同候选对内部 (A0 选中词恒定 -> 谓词身份被控制住)')
    print('=' * 70)
    bypair = collections.defaultdict(list)
    for r in B:
        bypair[tuple(r['names'])].append(r)
    print('%-28s%7s%7s%11s%11s%11s' % ('pair', 'n', 'pos', 'A0 gap', 'A2 gap', 'z0'))
    tot = collections.Counter()
    for pair, sub in sorted(bypair.items(), key=lambda kv: -len(kv[1])):
        y = [r['label'] for r in sub]
        if len(set(y)) < 2:
            continue
        a0 = auroc([-(r['zo'][r['a0']] - r['z0']) for r in sub], y)
        a2 = auroc([-(r['zo'][r['a2']] - r['z0']) for r in sub], y)
        z0 = auroc([r['z0'] for r in sub], y)
        print('%-28s%7d%7d%11.4f%11.4f%11.4f'
              % ('+'.join(pair), len(sub), sum(y), a0, a2, z0))
        tot['n'] += len(sub)
    print('\n注：同对内 A0 与 A2 若接近，说明跨对时 A0 的优势来自选中词分布，'
          '而不是选择策略本身。')


if __name__ == '__main__':
    main()
