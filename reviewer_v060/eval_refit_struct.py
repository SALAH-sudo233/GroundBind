#!/usr/bin/env python3
"""Q4: 结构阈值的校准顺序 —— 并行校准 vs 串行校准.

当前 eval_paper_tables.build() 的 'full' 模式按 **并行校准** 选结构阈值:

    ep   = 所有合格正例 (fold fit)
    cov  = ep 中落在结构门作用域内且可用的
    unc  = ep 中其余的
    fixed = sum(keep_sup(r) for r in unc)          # 作用域外沿用支持判决
    need  = round(target * len(ep)) - fixed         # <-- 关键
    t_full = 第 need 高的 s_full 分数

问题(审稿人 Q4): `need` 的目标数按 **全部** ep 算, 而 cov 里有一部分正例
**本来就会被支持门拒绝**。推理时判决是 `keep_sup(r) and d`(单调门), 所以那些
正例无论 d 如何都留不下来。把它们计入目标数, 等效于让结构门去凑一个它无法
达成的保留率 -> need 偏大 -> t_full 偏松 -> 结构门比名义 target 更保守不足。

串行校准(本脚本 'serial' 模式)改为:

    cov_s = [r for r in cov if keep_sup(r)]        # 只在支持门已放行的正例上
    need_s = round(target * len(ep)) - fixed - (支持门已放行但在作用域外的已计入 fixed)
           = max(0, round(target*len(ep)) - fixed)  并 clip 到 len(cov_s)
    t_full = 第 need_s 高的 s_full 分数, 仅在 cov_s 上取分位

即: 结构阈值只在"支持门已经放行的作用域内正例"上选, 分母不含必然被拒的正例。

两种设计都在 fit 折选阈值、test 折评测(交叉拟合), 共用同一个 `build()` 的
支持臂, 所以差异只来自结构阈值的选取分母。

用法(vlm1 ~/SVD/agentic_probe):
    python3 eval_refit_struct.py --target 0.95 --json-out refit_struct.json
"""
import argparse, collections, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_cbr_paper_aligned as A
import eval_upstream as E
from eval_unified import (COORDFIX_MODELS, load_probe, merge, measure,
                          usable, head, feats)
from eval_attribution import predicate_of, query_map
from eval_paper_tables import OPPOSED, build, PROBE

HT4 = A.HT4  # ['object', 'co_occurrence', 'attribute', 'relation']


def build_serial(rows, elig, target, gate=None):
    """串行校准: 结构阈值只在支持门已放行的作用域内正例上选。

    与 build(mode='full', monotone=True) 的唯一差别是 t_full 的选取分母。
    推理判决保持一致: keep_sup(r) and d (单调门), 作用域外沿用 keep_sup。
    """
    keep = {}
    diag = []
    for tf in (0, 1):
        fit = [r for r in rows if r['fold'] != tf]
        tst = [r for r in rows if r['fold'] == tf]
        s_sup = head(fit, 'support')
        if s_sup is None:
            for r in tst:
                keep[r['sid']] = bool(r['drew'])
            continue
        ps = sorted(s_sup(r) for r in fit
                    if r['is_pos'] and r['sid'] in elig and usable(r, 'support'))
        k = int(round((1.0 - target) * len(ps)))
        t_sup = ps[max(0, min(k, len(ps) - 1))] if ps else float('-inf')

        def keep_sup(r):
            if not r['drew']:
                return False
            if not usable(r, 'support'):
                return True
            return s_sup(r) >= t_sup

        g = gate or (lambda r: True)
        adm = [r for r in fit if usable(r, 'full') and g(r)]
        if len(adm) < 30 or len(set(r['label'] for r in adm)) < 2:
            for r in tst:
                keep[r['sid']] = keep_sup(r)
            continue
        f, _ = E.logreg([feats('full', r) for r in adm], [r['label'] for r in adm])
        s_full = lambda r: f(feats('full', r))

        ep = [r for r in fit if r['is_pos'] and r['sid'] in elig and r['drew']]
        cov = [r for r in ep if usable(r, 'full') and g(r)]
        unc = [r for r in ep if not (usable(r, 'full') and g(r))]
        fixed = sum(1 for r in unc if keep_sup(r))

        # ---- 串行校准的核心: 只在支持门已放行的 cov 上选阈值 ----
        cov_s = [r for r in cov if keep_sup(r)]
        n_dropped = len(cov) - len(cov_s)
        need = max(0, min(int(round(target * len(ep))) - fixed, len(cov_s)))
        sc = sorted((s_full(r) for r in cov_s), reverse=True)
        t_full = sc[need - 1] if need > 0 else float('inf')

        diag.append(dict(test_fold=tf, n_ep=len(ep), n_cov=len(cov),
                         n_cov_support_kept=len(cov_s),
                         n_cov_support_rejected=n_dropped,
                         fixed=fixed, need=need,
                         t_full=None if math.isinf(t_full) else t_full))

        for r in tst:
            if not r['drew']:
                keep[r['sid']] = False
                continue
            if not (usable(r, 'full') and g(r)):
                keep[r['sid']] = keep_sup(r)
                continue
            keep[r['sid']] = keep_sup(r) and (s_full(r) >= t_full)
    return keep, diag


def boot_ci(diffs, n=10000, seed=17):
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
    ap.add_argument('--json-out', default='refit_struct.json')
    a = ap.parse_args()

    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    cfroots = json.load(open(os.path.join(PROBE, 'canon_roots_coordfix.json')))
    models = sorted(canon)
    probes = (load_probe('probe_textroute_all.jsonl', 'z'),
              load_probe('trprobe_jev_all.jsonl', 'z_head'),
              load_probe('probecf_omni.jsonl', 'z', COORDFIX_MODELS),
              load_probe('probecf_jev.jsonl', 'z_head', COORDFIX_MODELS))
    po_main, pj_main, po_cf, pj_cf = probes

    per, diags = {}, {}
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
        pr_of = {r['sid']: (predicate_of(qmap.get(r['sid'], '')) or '') for r in rows}
        gate = lambda r: pr_of.get(r['sid'], '') in OPPOSED

        k_sup = build(rows, elig, a.target, 'support')
        k_par = build(rows, elig, a.target, 'full', gate=gate, monotone=True)
        k_ser, dg = build_serial(rows, elig, a.target, gate=gate)

        per[m] = dict(
            support=measure(rows, k_sup, elig, pos, neg),
            parallel=measure(rows, k_par, elig, pos, neg),
            serial=measure(rows, k_ser, elig, pos, neg),
            n_c=len(elig))
        diags[m] = dg
        s, p, q = per[m]['support'], per[m]['parallel'], per[m]['serial']
        print('%-16s FGR sup %6.2f | par %6.2f | ser %6.2f   '
              'relCBR sup %6.2f | par %6.2f | ser %6.2f   '
              'Rcorr sup %6.2f | par %6.2f | ser %6.2f'
              % (m, s['fgr_all'] * 100, p['fgr_all'] * 100, q['fgr_all'] * 100,
                 s['cbr'][3] * 100, p['cbr'][3] * 100, q['cbr'][3] * 100,
                 s['correct_retain'] * 100, p['correct_retain'] * 100,
                 q['correct_retain'] * 100))

    def agg(key, f):
        d = [f(per[m][key]) for m in per if f(per[m][key]) is not None]
        return st.mean(d) if d else None

    def delta(f, k1, k2):
        ds = [f(per[m][k1]) - f(per[m][k2]) for m in per
              if f(per[m][k1]) is not None and f(per[m][k2]) is not None]
        lo, hi = boot_ci(ds)
        return dict(mean=st.mean(ds) if ds else None, ci_lo=lo, ci_hi=hi,
                    worse_models=sum(1 for d in ds if d > 0), n=len(ds))

    fgr = lambda o: o['fgr_all'] * 100
    relc = lambda o: o['cbr'][3] * 100
    relf = lambda o: o['by_ht']['relation'] * 100 if o['by_ht'].get('relation') is not None else None
    rcor = lambda o: o['correct_retain'] * 100

    stats = {
        'serial_minus_support_all_fgr': delta(fgr, 'serial', 'support'),
        'serial_minus_support_rel_fgr': delta(relf, 'serial', 'support'),
        'serial_minus_support_rel_cbr': delta(relc, 'serial', 'support'),
        'serial_minus_parallel_all_fgr': delta(fgr, 'serial', 'parallel'),
        'serial_minus_parallel_rel_cbr': delta(relc, 'serial', 'parallel'),
        'serial_minus_parallel_retain': delta(rcor, 'serial', 'parallel'),
    }
    pooled = {k: dict(support=agg('support', f), parallel=agg('parallel', f),
                      serial=agg('serial', f))
              for k, f in (('all_fgr', fgr), ('rel_fgr', relf),
                           ('rel_cbr', relc), ('correct_retain', rcor))}

    print('\n=== pooled (model-equal) ===')
    for k, v in pooled.items():
        print(' %-16s sup %s  par %s  ser %s' % (
            k, *['%7.3f' % x if x is not None else '   None'
                 for x in (v['support'], v['parallel'], v['serial'])]))
    print('\n=== paired deltas ===')
    for k, v in stats.items():
        print(' %-34s %+7.3f  CI[%+7.3f,%+7.3f]  worse %d/%d' % (
            k, v['mean'], v['ci_lo'], v['ci_hi'], v['worse_models'], v['n']))

    out = dict(target=a.target, models=list(per.keys()), per=per,
               pooled=pooled, stats=stats, fold_diag=diags,
               note='parallel = current build(full,monotone); '
                    'serial = structural threshold chosen only on '
                    'support-kept covered positives')
    with open(a.json_out, 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote', a.json_out)


if __name__ == '__main__':
    main()
