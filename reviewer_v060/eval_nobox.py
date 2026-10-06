#!/usr/bin/env python3
"""Q5 去框消融分析：合并 nobox_shard*.jsonl，统计 box vs no-box 的判别力落差。

z = OmniVerifier 的 true-vs-false logit 差 (越高越"接受")。
对每个候选我们有 z_box(有红框) 与 z_nobox(无红框)。

关键指标：
  Δz 均值/中位数：去框后分数整体漂移
  AUROC(有框) vs AUROC(无框)：用 z 分离 positive(label=1) / 非 positive 的能力
    - htype=='positive' 的候选 label=1，其余 label=0
  若去框后 AUROC 向 0.5 塌，说明验证器真用候选区域；
  若 AUROC 几乎不变，说明它靠图像+文本先验，没用框。

label 从 sid 后缀/htype 推断不可靠，这里用 probe 的 htype。
"""
import collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.expanduser('~/SVD/agentic_probe')
PROBES = os.path.join(PROBE, 'probe_textroute_all.jsonl')


def auroc(pos, neg):
    """Mann-Whitney AUROC: P(score_pos > score_neg)."""
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


def main():
    # htype per (model, sid)
    ht = {}
    for line in open(PROBES, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        key = (r['model'], r['sid'])
        if key not in ht:
            ht[key] = r.get('htype')

    rows = []
    for s in range(8):
        p = os.path.join(PROBE, f'nobox_shard{s}.jsonl')
        if not os.path.exists(p):
            p = f'nobox_shard{s}.jsonl'
        if not os.path.exists(p):
            continue
        for line in open(p, encoding='utf-8'):
            d = json.loads(line)
            if d.get('error'):
                continue
            rows.append(d)

    # 收集 box/nobox 分数 + label(positive=1)
    def label_of(d):
        h = ht.get((d['model'], d['sid']))
        if h is None:
            # 从 sid 判断：无后缀=positive
            sid = d['sid']
            return 1 if '__' not in sid else 0
        return 1 if h == 'positive' else 0

    pos_box, neg_box, pos_nb, neg_nb = [], [], [], []
    dz = []
    per_model = collections.defaultdict(lambda: dict(
        pb=[], nb_=[], pnb=[], nnb=[]))
    for d in rows:
        lab = label_of(d)
        zb, zn = d['z_box'], d['z_nobox']
        dz.append(zn - zb)
        pm = per_model[d['model']]
        if lab == 1:
            pos_box.append(zb); pos_nb.append(zn)
            pm['pb'].append(zb); pm['pnb'].append(zn)
        else:
            neg_box.append(zb); neg_nb.append(zn)
            pm['nb_'].append(zb); pm['nnb'].append(zn)

    au_box = auroc(pos_box, neg_box)
    au_nb = auroc(pos_nb, neg_nb)
    import statistics as st

    print('=== Q5 去框消融 (box vs no-box) ===')
    print(f'有效候选: {len(rows)}  (positive={len(pos_box)}, other={len(neg_box)})')
    print(f'Δz (nobox-box) 均值: {st.mean(dz):+.4f}  中位数: {st.median(dz):+.4f}')
    print(f'AUROC 有框: {au_box:.4f}')
    print(f'AUROC 去框: {au_nb:.4f}')
    print(f'AUROC 落差: {au_box - au_nb:+.4f}  '
          f'(向0.5塌陷量: {(au_box-0.5)-(au_nb-0.5):+.4f})')
    print()

    # 逐模型 AUROC 落差
    pm_stats = {}
    print('逐模型 AUROC (有框 -> 去框):')
    drops = []
    for m in sorted(per_model):
        pm = per_model[m]
        ab = auroc(pm['pb'], pm['nb_'])
        an = auroc(pm['pnb'], pm['nnb'])
        pm_stats[m] = dict(auroc_box=ab, auroc_nobox=an,
                           drop=(ab - an) if (ab and an) else None,
                           n_pos=len(pm['pb']), n_neg=len(pm['nb_']))
        if ab and an:
            drops.append(ab - an)
            print(f'  {m:18s} {ab:.4f} -> {an:.4f}  (Δ {ab-an:+.4f})')

    print()
    print(f'模型级 AUROC 落差: 均值 {st.mean(drops):+.4f}, '
          f'{sum(1 for d in drops if d>0)}/{len(drops)} 模型去框后判别力下降')

    out = dict(
        n=len(rows), n_pos=len(pos_box), n_neg=len(neg_box),
        dz_mean=st.mean(dz), dz_median=st.median(dz),
        auroc_box=au_box, auroc_nobox=au_nb,
        auroc_drop=au_box - au_nb,
        per_model=pm_stats,
        model_level_drop_mean=st.mean(drops),
        models_degraded=sum(1 for d in drops if d > 0),
        n_models=len(drops),
        note='z = Omni true-vs-false logit diff; label: htype==positive -> 1. '
             'box draws red rect at candidate bbox; nobox removes it and asks '
             'about "the image". Lower no-box AUROC => verifier uses the region.')
    json.dump(out, open(os.path.join(HERE, 'nobox.json'), 'w'), indent=1)
    print('\nwrote nobox.json')


if __name__ == '__main__':
    main()
