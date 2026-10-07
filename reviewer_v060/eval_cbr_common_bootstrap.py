#!/usr/bin/env python3
"""Reviewer #9 + #10: CBR common-success subset + image-level bootstrap CI.

#9: 13-model mean CBR 的分母是各模型自己的 positive-correct image 子集，
    不同模型在不同样本集上，不是 common-difficulty leaderboard。
    这里取若干代表模型【共同 positive-correct】的 image 子集重算 CBR，
    看 ROH>BOH / relation-hardest 的趋势是否保持。

#10: 对 CBR 做 image-level paired bootstrap 95% CI（source image ID 为重采样单元）。

CONTRACT 逐字复用 eval_cbr_paper_aligned.py，不新增任何条件：
  elig  = {base | valid_box(pos.pred) and iou(pos.pred,pos.gt) >= 0.5}   theta=0.5
  r_i,t = 1[iou(neg.pred, POSITIVE PREDICTED box) >= 0.8]                 rho=0.8
  CBR_t = mean over elig of r_i,t
"""
import json, os, sys, random, argparse

PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
import eval_cbr_paper_aligned as A

TYPES = ('object', 'co_occurrence', 'attribute', 'relation')


def cbr_on_subset(subset, pos, neg, model):
    """在给定 image 子集上算四类 CBR，口径 = A.cbr(elig_subset, ...)。"""
    elig = list(subset)
    vals = A.cbr(elig, pos, neg, {}, None)   # idx=None -> unfiltered
    return {TYPES[i]: vals[i] * 100 for i in range(4)}


def bootstrap_cbr(elig, pos, neg, n_boot=2000, seed=0):
    """image-level paired bootstrap：对 elig image 有放回重采样，
    每次重算四类 CBR，返回每类 (mean, lo, hi)。"""
    rnd = random.Random(seed)
    elig = list(elig)
    N = len(elig)
    boots = {t: [] for t in TYPES}
    for _ in range(n_boot):
        samp = [elig[rnd.randrange(N)] for _ in range(N)]
        vals = A.cbr(samp, pos, neg, {}, None)
        for i, t in enumerate(TYPES):
            boots[t].append(vals[i] * 100)
    res = {}
    for t in TYPES:
        v = sorted(boots[t])
        res[t] = dict(mean=sum(v) / len(v),
                      lo=v[int(0.025 * n_boot)],
                      hi=v[int(0.975 * n_boot)])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--models', default='')  # 逗号分隔；空=全部 canon
    a = ap.parse_args()

    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    models = a.models.split(',') if a.models else sorted(canon)

    # 1) 每模型 eligible image 集合 + per-model CBR + bootstrap CI
    elig_sets = {}
    posneg = {}
    per_model = {}
    for m in models:
        if m not in canon:
            print(f'skip {m}: not in canon', file=sys.stderr); continue
        pos, neg = A.load_boxes(m, canon[m])
        elig = {b for b, p in pos.items()
                if A.valid_box(p['pred']) and A.iou(p['pred'], p['gt']) >= 0.5}
        elig_sets[m] = elig
        posneg[m] = (pos, neg)
        own_cbr = cbr_on_subset(elig, pos, neg, m)
        ci = bootstrap_cbr(elig, pos, neg, a.n_boot)
        per_model[m] = dict(n_elig=len(elig), cbr=own_cbr, ci=ci)
        print(f'{m:<16} n_elig={len(elig):>4}  '
              + '  '.join(f'{t[:3]}={own_cbr[t]:5.1f}[{ci[t]["lo"]:.1f},{ci[t]["hi"]:.1f}]'
                          for t in TYPES))

    # 2) common-success subset：所有选定模型都 positive-correct 的 image
    common = None
    for m in models:
        if m not in elig_sets:
            continue
        common = elig_sets[m] if common is None else (common & elig_sets[m])
    common = common or set()
    print(f'\n=== COMMON-SUCCESS subset: {len(common)} images '
          f'(positive-correct across all {len(elig_sets)} models) ===')

    common_cbr = {}
    for m in models:
        if m not in posneg:
            continue
        pos, neg = posneg[m]
        c = cbr_on_subset(common, pos, neg, m)
        ci = bootstrap_cbr(common, pos, neg, a.n_boot)
        common_cbr[m] = dict(cbr=c, ci=ci)
        print(f'{m:<16} '
              + '  '.join(f'{t[:3]}={c[t]:5.1f}' for t in TYPES))

    # 3) 趋势检查：relation 是否仍最难（CBR 最高）；ROH(attr+rel) vs BOH(obj+cooc)
    def roh_boh(cbr):
        roh = (cbr['attribute'] + cbr['relation']) / 2
        boh = (cbr['object'] + cbr['co_occurrence']) / 2
        return roh, boh
    print('\n=== trend check on common subset ===')
    rel_hardest = 0
    roh_gt_boh = 0
    nm = 0
    for m, d in common_cbr.items():
        c = d['cbr']
        nm += 1
        if c['relation'] == max(c.values()):
            rel_hardest += 1
        roh, boh = roh_boh(c)
        if roh > boh:
            roh_gt_boh += 1
    print(f'relation is hardest (max CBR): {rel_hardest}/{nm} models')
    print(f'ROH > BOH: {roh_gt_boh}/{nm} models')

    out = dict(
        contract='theta=0.5, rho=0.8, denom=len(elig), unfiltered; '
                 'image-level paired bootstrap',
        n_boot=a.n_boot,
        models=models,
        per_model=per_model,
        common_subset_size=len(common),
        common_cbr=common_cbr,
        trend=dict(relation_hardest=f'{rel_hardest}/{nm}',
                   roh_gt_boh=f'{roh_gt_boh}/{nm}'),
    )
    p = os.path.join(PROBE, 'cbr_common_bootstrap.json')
    json.dump(out, open(p, 'w'), indent=2, ensure_ascii=False)
    print('\nwrote', p)


if __name__ == '__main__':
    main()
