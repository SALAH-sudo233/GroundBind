#!/usr/bin/env python3
"""
Q8 补充：量化 parse 失败默认 NO 对 FA/TPR/BA 的影响。
对比两种口径：
  1）当前（含 parse-fail → pred_exists 非 True → 算作 NO）
  2）仅 parse_valid 子集（剔除解析失败行）
重点：UniVG-R1（37% 解析失败）、TreeVGR（25%），是否「语言判断 vs 定位行为差异」依赖解析失败的虚高/虚低。
"""
import json, os, sys
import statistics as st

canon = json.load(open('canon_roots_paper.json'))
HT4 = ['object', 'co_occurrence', 'attribute', 'relation']

def load_t1(model):
    """加载一个模型的 t1 记录（task=t1_discriminative_vqa）"""
    root = canon[model]
    p = os.path.join(root, model, 'records.jsonl')
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            try:
                d = json.loads(line)
            except:
                continue
            if d.get('task') == 't1_discriminative_vqa':
                out.append(d)
    return out

def metrics(rows, name):
    """当前口径：pred_exists is True 算 YES，其余（含 parse-fail）算 NO"""
    pos = [r for r in rows if r.get('label_exists') is True]
    neg = [r for r in rows if r.get('label_exists') is False]
    if not pos or not neg:
        return None
    fa = sum(1 for r in neg if r.get('pred_exists') is True) / len(neg)
    tpr = sum(1 for r in pos if r.get('pred_exists') is True) / len(pos)
    bal = (tpr + (1.0 - fa)) / 2.0
    bad = sum(1 for r in rows if not r.get('parse_valid', True))
    return {
        'scope': name,
        'n_pos': len(pos),
        'n_neg': len(neg),
        'n_total': len(rows),
        'parse_fail': bad,
        'parse_fail_rate': bad / len(rows) if rows else 0,
        'FA': fa,
        'TPR': tpr,
        'BA': bal
    }

def audit_model(model):
    """对比「当前口径」vs「仅 parse_valid 子集」"""
    rows = load_t1(model)
    if not rows:
        return None
    # 当前口径（含 parse-fail）
    curr = metrics(rows, 'current_all')
    # 仅 parse_valid 子集
    clean = [r for r in rows if r.get('parse_valid', True)]
    valid_only = metrics(clean, 'parse_valid_only')
    d_fa = d_tpr = d_ba = None
    if valid_only and curr:
        d_fa = valid_only['FA'] - curr['FA']
        d_tpr = valid_only['TPR'] - curr['TPR']
        d_ba = valid_only['BA'] - curr['BA']
    return {
        'model': model,
        'current': curr,
        'valid_only': valid_only,
        'delta_FA': d_fa,
        'delta_TPR': d_tpr,
        'delta_BA': d_ba
    }

if __name__ == '__main__':
    focus = ['UniVG-R1', 'TreeVGR']  # 高解析失败率模型
    others = [m for m in sorted(canon) if m not in focus]
    
    out = {}
    print("=== 高解析失败率模型（UniVG-R1 37%, TreeVGR 25%）===")
    for m in focus:
        res = audit_model(m)
        if res:
            out[m] = res
            c = res['current']
            v = res['valid_only']
            print(f"\n{m}:")
            print(f"  当前口径：n={c['n_total']} parse_fail={c['parse_fail']}({c['parse_fail_rate']*100:.1f}%) FA={c['FA']*100:.2f}% TPR={c['TPR']*100:.2f}% BA={c['BA']*100:.2f}%")
            print(f"  仅parse_valid：n={v['n_total']} FA={v['FA']*100:.2f}% TPR={v['TPR']*100:.2f}% BA={v['BA']*100:.2f}%")
            if res['delta_BA'] is not None:
                print(f"  Δ(valid−curr)：FA={res['delta_FA']*100:+.2f}pp TPR={res['delta_TPR']*100:+.2f}pp BA={res['delta_BA']*100:+.2f}pp")
    
    print("\n=== 其余 11 模型（解析率 ≥99%）===")
    for m in others[:3]:  # 抽样 3 个
        res = audit_model(m)
        if res:
            out[m] = res
            c = res['current']
            v = res['valid_only']
            print(f"{m}: 当前 FA={c['FA']*100:.2f}% BA={c['BA']*100:.2f}% parse_fail={c['parse_fail']} Δ_BA={res['delta_BA']*100:+.2f}pp")
    
    with open('audit_parse_impact.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    print("\n写入 audit_parse_impact.json")
