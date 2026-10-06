#!/usr/bin/env python3
"""Q7: 决策头跨模型迁移 —— leave-one-model-out & 统一头.

当前主表每个模型在**自身**数据的一折上拟合支持头和结构头(per-model)。
审稿人Q7问：需要多少目标域标签？统一头或LOMO是否仍有效？

feats()只用4个模型无关标量 [z_o, z_j, g_o, g_j]，所以可以把一个模型的头
用到另一个模型上。本脚本比较三种拟合方式，均在同一target下评测、同一elig：

  per_model : 现状。模型m的头在m自身fit折上拟合(交叉拟合两折)。
  unified   : 一个头在**全部13模型**的fit数据上拟合，用到每个模型。
  lomo      : 模型m的头在**其余12个模型全部数据**上拟合(m完全不参与)，用到m。
              这是最严格的"零目标域标签"设置。

阈值(t_sup/t_full)仍按各评测模型自身的保留率目标选(target定召回预算)，
因为阈值是部署时按每个上游的工作点设的；变的只是**判别头的权重来源**。

用法 (vlm1 ~/SVD/agentic_probe):
    python3 eval_lomo.py --target 0.95 --json-out lomo.json
"""
import argparse, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
sys.path.insert(0, HERE)

import eval_cbr_paper_aligned as A
import eval_upstream as E
from eval_unified import (COORDFIX_MODELS, load_probe, merge, measure,
                          usable, feats)
from eval_attribution import predicate_of, query_map
from eval_paper_tables import OPPOSED, PROBE as PT_PROBE


def fit_head(train_rows, mode):
    R = [r for r in train_rows if usable(r, mode)]
    if len(R) < 30 or len(set(r['label'] for r in R)) < 2:
        return None
    f, _ = E.logreg([feats(mode, r) for r in R], [r['label'] for r in R])
    return lambda r: f(feats(mode, r))


def apply_policy(rows_test, s_sup, s_full, target, elig, gate):
    """用给定的两个头，在rows_test上按target选阈值并产生keep判决。

    阈值在rows_test自身的fit折(非test折)上选 —— 但头来自外部。
    为简洁：阈值用test集正例的保留率目标在同一集合上选分位（部署工作点）。
    """
    keep = {}
    # support阈值：test集合格正例分数的(1-target)分位
    ps = sorted(s_sup(r) for r in rows_test
                if r['is_pos'] and r['sid'] in elig and usable(r, 'support'))
    k = int(round((1.0 - target) * len(ps)))
    t_sup = ps[max(0, min(k, len(ps) - 1))] if ps else float('-inf')

    def keep_sup(r):
        if not r['drew']:
            return False
        if not usable(r, 'support'):
            return True
        return s_sup(r) >= t_sup

    if s_full is None:
        for r in rows_test:
            keep[r['sid']] = keep_sup(r)
        return keep

    ep = [r for r in rows_test if r['is_pos'] and r['sid'] in elig and r['drew']]
    cov = [r for r in ep if usable(r, 'full') and gate(r)]
    unc = [r for r in ep if not (usable(r, 'full') and gate(r))]
    fixed = sum(1 for r in unc if keep_sup(r))
    need = int(round(target * len(ep))) - fixed
    sc = sorted((s_full(r) for r in cov), reverse=True)
    t_full = sc[need - 1] if 0 < need <= len(sc) else (
        float('-inf') if need > len(sc) else float('inf'))

    for r in rows_test:
        if not r['drew']:
            keep[r['sid']] = False
        elif not (usable(r, 'full') and gate(r)):
            keep[r['sid']] = keep_sup(r)
        else:
            keep[r['sid']] = keep_sup(r) and (s_full(r) >= t_full)
    return keep


def boot_ci(diffs, n=10000, seed=19):
    if not diffs:
        return (None, None)
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        s = [diffs[rnd.randrange(len(diffs))] for _ in diffs]
        out.append(st.mean(s))
    out.sort()
    return (out[int(0.025 * n)], out[int(0.975 * n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=float, default=0.95)
    ap.add_argument('--json-out', default='lomo.json')
    a = ap.parse_args()

    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    cfroots = json.load(open(os.path.join(PROBE, 'canon_roots_coordfix.json')))
    models = sorted(canon)
    po_main = load_probe('probe_textroute_all.jsonl', 'z')
    pj_main = load_probe('trprobe_jev_all.jsonl', 'z_head')
    po_cf = load_probe('probecf_omni.jsonl', 'z', COORDFIX_MODELS)
    pj_cf = load_probe('probecf_jev.jsonl', 'z_head', COORDFIX_MODELS)

    # 预加载每个模型的 rows / elig / gate / pos,neg
    M = {}
    all_rows = []
    for m in models:
        cf = m in COORDFIX_MODELS
        root = cfroots[m] if cf else canon[m]
        qmap = query_map(root, m)
        pos, neg = A.load_boxes(m, root)
        elig = {q['sid'] for b, q in pos.items()
                if A.valid_box(q['pred']) and A.iou(q['pred'], q['gt']) >= 0.5}
        rows = merge(m, po_cf if cf else po_main, pj_cf if cf else pj_main, cf)
        if not rows:
            continue
        pr = {r['sid']: (predicate_of(qmap.get(r['sid'], '')) or '') for r in rows}
        gate = (lambda pr: (lambda r: pr.get(r['sid'], '') in OPPOSED))(pr)
        M[m] = dict(rows=rows, elig=elig, gate=gate, pos=pos, neg=neg)
        all_rows += rows

    def eval_mode(mode_name):
        per = {}
        for m in M:
            d = M[m]
            rows = d['rows']
            if mode_name == 'per_model':
                # 交叉拟合两折
                keep = {}
                for tf in (0, 1):
                    fit = [r for r in rows if r['fold'] != tf]
                    tst = [r for r in rows if r['fold'] == tf]
                    ss = fit_head(fit, 'support')
                    sf = fit_head([r for r in fit
                                   if usable(r, 'full') and d['gate'](r)], 'full')
                    kk = apply_policy(tst, ss, sf, a.target, d['elig'], d['gate'])
                    keep.update(kk)
            else:
                if mode_name == 'unified':
                    train = all_rows
                else:  # lomo
                    train = [r for mm in M if mm != m for r in M[mm]['rows']]
                ss = fit_head(train, 'support')
                sf = fit_head([r for r in train
                               if usable(r, 'full') and
                               M[r['model']]['gate'](r)], 'full')
                keep = apply_policy(rows, ss, sf, a.target, d['elig'], d['gate'])
            per[m] = measure(rows, keep, d['elig'], d['pos'], d['neg'])
        return per

    modes = {}
    for name in ('per_model', 'unified', 'lomo'):
        modes[name] = eval_mode(name)
        print(f'[done] {name}')

    fgr = lambda o: o['fgr_all'] * 100
    relc = lambda o: o['cbr'][3] * 100
    relf = lambda o: (o['by_ht']['relation'] * 100
                      if o['by_ht'].get('relation') is not None else None)
    rcor = lambda o: o['correct_retain'] * 100

    def agg(per, f):
        v = [f(per[m]) for m in per if f(per[m]) is not None]
        return st.mean(v) if v else None

    def delta(f, m1, m2):
        ds = [f(modes[m1][m]) - f(modes[m2][m]) for m in modes[m1]
              if f(modes[m1][m]) is not None and f(modes[m2][m]) is not None]
        lo, hi = boot_ci(ds)
        return dict(mean=st.mean(ds) if ds else None, ci_lo=lo, ci_hi=hi,
                    worse=sum(1 for d in ds if d > 0), n=len(ds))

    pooled = {name: {k: agg(modes[name], f)
                     for k, f in (('all_fgr', fgr), ('rel_fgr', relf),
                                  ('rel_cbr', relc), ('correct_retain', rcor))}
              for name in modes}
    stats = {
        'lomo_minus_permodel_all_fgr': delta(fgr, 'lomo', 'per_model'),
        'lomo_minus_permodel_rel_fgr': delta(relf, 'lomo', 'per_model'),
        'lomo_minus_permodel_rel_cbr': delta(relc, 'lomo', 'per_model'),
        'lomo_minus_permodel_retain': delta(rcor, 'lomo', 'per_model'),
        'unified_minus_permodel_all_fgr': delta(fgr, 'unified', 'per_model'),
        'unified_minus_permodel_rel_cbr': delta(relc, 'unified', 'per_model'),
    }

    print('\n=== pooled (model-equal) ===')
    for name in modes:
        p = pooled[name]
        print(' %-10s FGR %6.3f  relFGR %6.3f  relCBR %6.3f  Rcorr %6.3f' % (
            name, p['all_fgr'], p['rel_fgr'] or -1, p['rel_cbr'],
            p['correct_retain']))
    print('\n=== deltas vs per_model ===')
    for k, v in stats.items():
        print(' %-32s %+7.3f  CI[%+7.3f,%+7.3f]  worse %d/%d' % (
            k, v['mean'], v['ci_lo'], v['ci_hi'], v['worse'], v['n']))

    out = dict(target=a.target, models=list(M.keys()),
               per_mode={name: {m: modes[name][m] for m in modes[name]}
                         for name in modes},
               pooled=pooled, stats=stats,
               note='per_model=cross-fit self; unified=one head on all 13; '
                    'lomo=head on other 12, zero target-model labels. '
                    'thresholds set per eval-model working point (target).')
    json.dump(out, open(os.path.join(HERE, a.json_out), 'w'), indent=1)
    print('\nwrote', a.json_out)


if __name__ == '__main__':
    main()
