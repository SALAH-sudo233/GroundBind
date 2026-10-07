#!/usr/bin/env python3
"""Reviewer #4 (text-only control): 去掉真实图像(喂固定灰图)后，
OmniVerifier 是否仍能分 positive/negative。

z_text = 无视觉信号下的 true-vs-false logit 差。
label: htype==positive -> 1，其余 -> 0（口径同 eval_nobox）。

三条件对比（同一候选集）：
  text-only (z_text,  无图像信号)
  no-box    (z_nobox, 有图像无框)
  boxed     (z_box,   有图像有红框)

AUROC(text-only) 判读：
  ~0.5  -> 文本本身无法分 pos/neg，benchmark 无显著 lexical artifact（期望）
  显著>0.5 -> 文本泄露，negative 可被纯语言模式识别（lexical artifact 风险）
对照：AUROC(image) - AUROC(text-only) = 视觉信号贡献的判别力增量。
"""
import collections, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.expanduser('~/SVD/agentic_probe')
PROBES = os.path.join(PROBE, 'probe_textroute_all.jsonl')


def auroc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            rank[k] = r
        i = j
    sum_pos = sum(rank[k] for k in range(len(allv)) if allv[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def boot_auroc_ci(pos, neg, n_boot=2000, seed=0):
    import random
    rnd = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        p = [pos[rnd.randrange(len(pos))] for _ in pos]
        n = [neg[rnd.randrange(len(neg))] for _ in neg]
        vals.append(auroc(p, n))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def main():
    ht = {}
    for line in open(PROBES, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        key = (r['model'], r['sid'])
        if key not in ht:
            ht[key] = r.get('htype')

    def label_of(d):
        h = ht.get((d['model'], d['sid']))
        if h is None:
            return 1 if '__' not in d['sid'] else 0
        return 1 if h == 'positive' else 0

    # text-only
    tpath = os.path.join(PROBE, 'textonly_all.jsonl')
    if not os.path.exists(tpath):
        tpath = 'textonly_all.jsonl'
    trows = [json.loads(l) for l in open(tpath, encoding='utf-8')]
    trows = [d for d in trows if not d.get('error')]

    # nobox (image) 对照
    nbpath = os.path.join(PROBE, 'nobox_all.jsonl')
    nb = {}
    if os.path.exists(nbpath):
        for l in open(nbpath, encoding='utf-8'):
            d = json.loads(l)
            if d.get('error'):
                continue
            nb[(d['model'], d['sid'])] = d

    pos_t, neg_t = [], []
    pos_nb, neg_nb, pos_b, neg_b = [], [], [], []
    per_model = collections.defaultdict(lambda: dict(pt=[], nt=[]))
    for d in trows:
        lab = label_of(d)
        zt = d['z_text']
        pm = per_model[d['model']]
        if lab == 1:
            pos_t.append(zt); pm['pt'].append(zt)
        else:
            neg_t.append(zt); pm['nt'].append(zt)
        # 对照 image 分数
        img = nb.get((d['model'], d['sid']))
        if img:
            if lab == 1:
                pos_nb.append(img['z_nobox']); pos_b.append(img['z_box'])
            else:
                neg_nb.append(img['z_nobox']); neg_b.append(img['z_box'])

    au_t = auroc(pos_t, neg_t)
    ci_t = boot_auroc_ci(pos_t, neg_t)
    au_nb = auroc(pos_nb, neg_nb) if pos_nb else None
    au_b = auroc(pos_b, neg_b) if pos_b else None

    print('=== Text-only control (reviewer #4) ===')
    print(f'有效候选: {len(trows)}  (positive={len(pos_t)}, other={len(neg_t)})')
    print(f'AUROC(text-only, 无视觉信号): {au_t:.4f}  95%CI[{ci_t[0]:.4f},{ci_t[1]:.4f}]')
    if au_nb is not None:
        print(f'AUROC(no-box, 有图像):       {au_nb:.4f}  (对照子集 n_pos={len(pos_nb)})')
    if au_b is not None:
        print(f'AUROC(boxed, 有图像有框):     {au_b:.4f}')
    if au_nb is not None:
        print(f'视觉信号增量 AUROC(image)-AUROC(text): {au_nb - au_t:+.4f}')
    verdict = ('接近0.5：无显著 lexical artifact' if ci_t[0] <= 0.55
               else '显著>0.5：存在文本泄露风险')
    print(f'判读: {verdict}')
    print()

    pm_stats = {}
    aus = []
    print('逐模型 AUROC(text-only):')
    for m in sorted(per_model):
        pm = per_model[m]
        a = auroc(pm['pt'], pm['nt'])
        pm_stats[m] = dict(auroc_text=a, n_pos=len(pm['pt']), n_neg=len(pm['nt']))
        if a is not None:
            aus.append(a)
            print(f'  {m:18s} {a:.4f}  (pos={len(pm["pt"])}, neg={len(pm["nt"])})')
    print(f'\n模型级 AUROC(text-only) 均值: {st.mean(aus):.4f}  '
          f'范围[{min(aus):.4f},{max(aus):.4f}]')

    out = dict(
        n=len(trows), n_pos=len(pos_t), n_neg=len(neg_t),
        auroc_text=au_t, auroc_text_ci=list(ci_t),
        auroc_nobox_image=au_nb, auroc_boxed_image=au_b,
        visual_gain=(au_nb - au_t) if au_nb is not None else None,
        per_model=pm_stats,
        model_level_text_auroc_mean=st.mean(aus),
        verdict=verdict,
        note='z_text = Omni true-vs-false logit diff with a fixed gray blank image '
             '(no visual signal). label: htype==positive->1. AUROC~0.5 means the '
             'claim text alone cannot separate pos/neg -> no lexical artifact.')
    json.dump(out, open(os.path.join(HERE, 'textonly.json'), 'w'), indent=1)
    # 远程也存一份
    try:
        json.dump(out, open(os.path.join(PROBE, 'textonly.json'), 'w'), indent=1)
    except Exception:
        pass
    print('\nwrote textonly.json')


if __name__ == '__main__':
    main()
