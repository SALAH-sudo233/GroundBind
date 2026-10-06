#!/usr/bin/env python3
"""Q9: 路由覆盖率与结构探针触发统计（成本的调用侧，纯CPU）.

审稿人Q9问 ASV 的实际成本：路由覆盖、fallback比例、调用数。
延迟/显存是部署指标(需单独profile)，这里给**调用侧**可从打分文件直接算的部分：

  每个候选都过一次 OmniVerifier(z0) + 一次 JEV(jev)      —— 支持核验，100%覆盖
  只有 gate(r)=True（谓词在OPPOSED且有合法替代）才加结构探针：
      ≤2个竞争探针(rivals) 的 OmniVerifier + JEV 增强分

统计：
  - 结构门触发率 = gate(r)=True 的候选占比(全体/各htype)
  - 平均每触发候选的竞争探针数(rivals长度) -> 额外OmniVerifier调用数
  - fallback比例 = 路由判定需结构证据但无合法替代(OPPOSED外或无rival) -> 退回支持判决

用法 (vlm1 ~/SVD/agentic_probe):
    python3 eval_routing_cost.py --json-out routing_cost.json
"""
import argparse, collections, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
sys.path.insert(0, HERE)

import eval_cbr_paper_aligned as A
from eval_unified import COORDFIX_MODELS, load_probe, merge, usable
from eval_attribution import predicate_of, query_map
from eval_paper_tables import OPPOSED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-out', default='routing_cost.json')
    a = ap.parse_args()

    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    cfroots = json.load(open(os.path.join(PROBE, 'canon_roots_coordfix.json')))
    models = sorted(canon)
    po_main = load_probe('probe_textroute_all.jsonl', 'z')
    pj_main = load_probe('trprobe_jev_all.jsonl', 'z_head')
    po_cf = load_probe('probecf_omni.jsonl', 'z', COORDFIX_MODELS)
    pj_cf = load_probe('probecf_jev.jsonl', 'z_head', COORDFIX_MODELS)

    per = {}
    tot = collections.Counter()
    rival_counts = []

    for m in models:
        cf = m in COORDFIX_MODELS
        root = cfroots[m] if cf else canon[m]
        qmap = query_map(root, m)
        rows = merge(m, po_cf if cf else po_main, pj_cf if cf else pj_main, cf)
        if not rows:
            continue
        pm = collections.Counter()
        by_ht_fire = collections.defaultdict(lambda: [0, 0])  # [total, fired]
        po_m = po_cf if cf else po_main
        for r in rows:
            pm['rows'] += 1
            pred = predicate_of(qmap.get(r['sid'], '')) or ''
            in_opp = pred in OPPOSED
            has_full = usable(r, 'full')
            # 竞争探针数 = 该候选的 rival OmniVerifier 打分条目数
            rvo = po_m.get((m, r['sid']), {})
            nr = sum(1 for v in rvo.values() if v is not None) if isinstance(rvo, dict) else 0
            # 结构门：谓词在OPPOSED且有可用增强分(zmax非空=确有竞争探针)
            fired = in_opp and has_full and r.get('zmax_o') is not None
            ht = r.get('htype') or ('positive' if r['is_pos'] else 'unknown')
            by_ht_fire[ht][0] += 1
            if fired:
                pm['struct_fired'] += 1
                by_ht_fire[ht][1] += 1
                pm['rival_omni_calls'] += nr
                rival_counts.append(nr)
            elif in_opp and not (has_full and r.get('zmax_o') is not None):
                # 路由想要结构证据但增强分不可用 -> fallback到支持判决
                pm['fallback_no_evidence'] += 1
            # 支持核验：每候选1次Omni + 1次JEV
            pm['support_omni_calls'] += 1
            pm['support_jev_calls'] += 1
        per[m] = dict(pm)
        per[m]['by_htype_fire'] = {k: dict(total=v[0], fired=v[1],
                                           rate=v[1] / v[0] if v[0] else 0.0)
                                   for k, v in by_ht_fire.items()}
        for k, v in pm.items():
            tot[k] += v

    # 池化率
    R = tot['rows']
    pooled = dict(
        total_rows=R,
        struct_fire_rate=tot['struct_fired'] / R if R else 0.0,
        fallback_rate=tot['fallback_no_evidence'] / R if R else 0.0,
        support_omni_calls=tot['support_omni_calls'],
        support_jev_calls=tot['support_jev_calls'],
        extra_omni_calls_from_rivals=tot['rival_omni_calls'],
        extra_call_overhead_pct=(tot['rival_omni_calls'] /
                                 tot['support_omni_calls'] * 100
                                 if tot['support_omni_calls'] else 0.0),
        mean_rivals_per_fired=(st.mean(rival_counts) if rival_counts else 0.0),
    )

    # 按htype聚合触发率
    ht_fire = collections.defaultdict(lambda: [0, 0])
    for m in per:
        for ht, d in per[m]['by_htype_fire'].items():
            ht_fire[ht][0] += d['total']
            ht_fire[ht][1] += d['fired']
    pooled['by_htype_fire_rate'] = {
        ht: dict(total=v[0], fired=v[1], rate=v[1] / v[0] if v[0] else 0.0)
        for ht, v in sorted(ht_fire.items())}

    print('=== ASV 路由与调用成本 (Q9) ===')
    print(f'总候选: {R}')
    print(f'结构门触发率: {pooled["struct_fire_rate"]*100:.2f}%')
    print(f'fallback(想要结构证据但不可用): {pooled["fallback_rate"]*100:.2f}%')
    print(f'支持核验 Omni 调用: {pooled["support_omni_calls"]} (每候选1次)')
    print(f'支持核验 JEV 调用:  {pooled["support_jev_calls"]}')
    print(f'结构探针额外 Omni 调用: {pooled["extra_omni_calls_from_rivals"]}')
    print(f'额外调用开销: +{pooled["extra_call_overhead_pct"]:.2f}% (相对支持核验Omni)')
    print(f'平均每触发候选竞争探针数: {pooled["mean_rivals_per_fired"]:.3f}')
    print()
    print('按htype结构门触发率:')
    for ht, d in pooled['by_htype_fire_rate'].items():
        print(f'  {ht:14s} {d["fired"]:5d}/{d["total"]:5d} = {d["rate"]*100:5.2f}%')

    out = dict(pooled=pooled, per_model=per,
               note='support verification = 1 Omni + 1 JEV per candidate (100%); '
                    'structural probe fires only when predicate in OPPOSED and '
                    'enhancement scores usable; rivals<=2 extra Omni each.')
    json.dump(out, open(os.path.join(HERE, a.json_out), 'w'), indent=1)
    print(f'\nwrote {a.json_out}')


if __name__ == '__main__':
    main()
