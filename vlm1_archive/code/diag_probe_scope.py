#!/usr/bin/env python3
"""Should the contrastive probe cover htypes other than relation?

Three CPU-only diagnostics, no GPU, no new collection.

[1] WHICH CONSTITUENT IS ALTERED per hallucination type. Diff each negative
    expression against its paired positive and classify what actually changed:
    head noun, adjective, predicate, or an added/removed clause. A predicate-swap
    probe can only be informative for rows whose altered constituent IS the
    predicate.

[2] DISCRIMINATIVE POWER of the predicate gap, per htype. AUROC of
    max(z_rival) - z0 for separating positives from negatives of that type, on
    probe-covered rows only. AUROC ~ 0.5 means the feature is noise for that type,
    so feeding it to the fused head can only add variance.

[3] HEADROOM after support-only verification, per htype: how much CBR / FGR is
    left for a probe to remove at all.

Together these say whether other types need their own rival tables, or whether one
support-verification pass already exhausts them.
"""
import json
import os
import re
import sys
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_upstream as E
import eval_combo_fixedbudget as F
import simple_relations as sr

BENCH = '/home/u2025141034/benchmark/repaired/refcocog_500_dev.semantic_strict.json'
HT4 = ('object', 'co_occurrence', 'attribute', 'relation')


def toks(s):
    return re.findall(r"[a-z]+", (s or '').lower())


def classify_edit(pos_text, neg_text):
    """What changed between the paired positive and this negative?"""
    p, n = toks(pos_text), toks(neg_text)
    sp, sn = set(p), set(n)
    added, removed = sn - sp, sp - sn

    pred_p, _, _ = sr.find_predicate(pos_text or '')
    pred_n, _, _ = sr.find_predicate(neg_text or '')
    pred_changed = (pred_p or '') != (pred_n or '')

    # predicate word sets, to tell predicate edits from content edits
    pw = set()
    for s in sr.PREDICATES:
        pw.update(s.split())

    content_added = {w for w in added if w not in pw}
    content_removed = {w for w in removed if w not in pw}

    if pred_changed and not content_added and not content_removed:
        return 'predicate_only'
    if pred_changed and (content_added or content_removed):
        return 'predicate+content'
    if not pred_changed and (content_added or content_removed):
        return 'content_only'
    return 'other'


def main():
    recs = json.load(open(BENCH, encoding='utf-8'))
    # group by image to recover the paired positive text
    by_pair = defaultdict(dict)
    for r in recs:
        by_pair[r['image_filename']][r.get('hallucination_type')] = r
    pos_text = {}
    for r in recs:
        # 'chosen' carries the supported expression for the group
        pos_text[r['image_filename']] = r.get('chosen')

    print('=' * 100)
    print('[1] 每类幻觉真正被改动的成分（负例 vs 其配对正例，500 组）')
    print('=' * 100)
    tab = defaultdict(Counter)
    for r in recs:
        ht = r.get('hallucination_type')
        if ht not in HT4:
            continue
        tab[ht][classify_edit(pos_text.get(r['image_filename']),
                              r.get('negative_text'))] += 1
    kinds = ['predicate_only', 'predicate+content', 'content_only', 'other']
    print('%-16s%18s%18s%14s%8s' % ('htype', *kinds))
    for ht in HT4:
        c = tab[ht]
        tot = sum(c.values()) or 1
        print('%-16s%18s%18s%14s%8s' % (
            ht, *['%d (%.0f%%)' % (c[k], 100.0 * c[k] / tot) for k in kinds]))
    print('\n解读：只有 predicate_only 这一列的行，"换谓词"才是对症的探针。')

    # ---------- [2] AUROC of the predicate gap per htype
    print('\n' + '=' * 100)
    print('[2] 谓词 gap 特征的判别力（AUROC，按类型，仅探针覆盖行）')
    print('=' * 100)
    canon = json.load(open(os.path.join(HERE, 'canon_roots_paper.json')))
    po = F.load_probe('probe_textroute_all.jsonl', 'z')
    pj = F.load_probe('trprobe_jev_all.jsonl', 'z_head')
    per_ht = defaultdict(lambda: ([], []))   # ht -> (scores, labels)
    for m in sorted(canon):
        rows = F.merge(m, po, pj, False)
        pos_cov = [r for r in rows if r['is_pos'] and F.covered(r)]
        for ht in HT4:
            neg_cov = [r for r in rows
                       if not r['is_pos'] and r['htype'] == ht and F.covered(r)]
            if not neg_cov or not pos_cov:
                continue
            s, y = per_ht[ht]
            for r in pos_cov:
                s.append(max(r['g_o'], r['g_j'])); y.append(0)
            for r in neg_cov:
                s.append(max(r['g_o'], r['g_j'])); y.append(1)
    print('%-16s%10s%10s%12s' % ('htype', 'n_pos', 'n_neg', 'AUROC(neg>pos)'))
    for ht in HT4:
        s, y = per_ht[ht]
        if not s:
            continue
        print('%-16s%10d%10d%12.4f'
              % (ht, y.count(0), y.count(1), E.auroc(s, y)))
    print('\n解读：0.5 = 该类型上 gap 不携带信息，送入融合头只会增加方差。')

    # ---------- [3] headroom after support-only
    print('\n' + '=' * 100)
    print('[3] support-only 之后各类型的剩余错误（池化 13 模型，target 0.95）')
    print('=' * 100)
    pm = os.path.join(HERE, 'permodel_095.json')
    if not os.path.exists(pm):
        print('missing permodel_095.json'); return
    d = json.load(open(pm, encoding='utf-8'))
    per, ms = d['per'], d['models']
    import statistics as st

    def pool(arm, f):
        return st.mean([f(per[m][arm]) for m in ms])
    print('%-16s%12s%12s%12s   %12s%12s%12s' %
          ('htype', 'CBR B0', 'CBR sup', 'CBR full', 'FGR B0', 'FGR sup', 'FGR full'))
    for i, ht in enumerate(HT4):
        print('%-16s%11.1f%%%11.1f%%%11.1f%%   %11.1f%%%11.1f%%%11.1f%%' % (
            ht,
            pool('B0', lambda r, i=i: float(r['cbr'][i])) * 100,
            pool('support', lambda r, i=i: float(r['cbr'][i])) * 100,
            pool('full', lambda r, i=i: float(r['cbr'][i])) * 100,
            pool('B0', lambda r, h=ht: r['by_ht'][h]) * 100,
            pool('support', lambda r, h=ht: r['by_ht'][h]) * 100,
            pool('full', lambda r, h=ht: r['by_ht'][h]) * 100))
    print('\n解读：support 列已很低的类型，词表扩展的可得空间本来就小。')


if __name__ == '__main__':
    main()
