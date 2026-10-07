#!/usr/bin/env python3
"""
Q8 实质修正：parse 失败不再默认 NO，而是单列为「弃权 abstention」。

背景：当前 eval_tasks_t1_t3_t4.py 把 parse 失败行的 pred_exists 硬写成 False，
使解析失败等价于「模型判否」，直接进 FA/TPR/BA。已证实（audit_parse_impact）：
  - 解析失败强非随机：UniVG-R1 pos 46.2% vs neg 35.0%；TreeVGR pos 0.8% vs neg 30.9%
  - 偏差方向因模型而异：UniVG-R1 BA 被低估 12.66pp，TreeVGR BA 被高估 9.82pp
因此默认 NO 不能靠附录披露抵消。

三套口径（并报）：
  CURRENT    当前论文口径：parse-fail → NO（pred_exists is True 才算 YES）
  ABSTAIN    弃权口径：FA/TPR/BA 只在「给出可解析判断」的样本上定义，
             解析失败单列为 abstention_rate（响应完整性），不塞进 YES/NO
  VALID_ONLY 干净子集：等价于 ABSTAIN 的 FA/TPR/BA（因为弃权样本本就被排除），
             但显式报告子集分母与 pos/neg 分母变化，暴露选择效应

核心判定：「语言判断 vs 定位行为差异」是否依赖解析失败伪信号。
在 ABSTAIN 口径下重算全 13 模型，看跨模型排序/差异是否幸存。
"""
import json, os
import statistics as st

canon = json.load(open('canon_roots_paper.json'))
HT4 = ['object', 'co_occurrence', 'attribute', 'relation']

def load_t1(model):
    root = canon[model]
    p = os.path.join(root, model, 'records.jsonl')
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        try:
            d = json.loads(line)
        except:
            continue
        if d.get('task') == 't1_discriminative_vqa':
            out.append(d)
    return out

def is_yes(r):
    return r.get('pred_exists') is True

def answered(r):
    """给出可解析判断（弃权口径下的有效样本）"""
    return bool(r.get('parse_valid', True))

def eval_current(rows):
    """当前口径：parse-fail 已被硬写 pred_exists=False，直接算 NO"""
    pos = [r for r in rows if r.get('label_exists') is True]
    neg = [r for r in rows if r.get('label_exists') is False]
    if not pos or not neg:
        return None
    fa = sum(1 for r in neg if is_yes(r)) / len(neg)
    tpr = sum(1 for r in pos if is_yes(r)) / len(pos)
    return {'FA': fa, 'TPR': tpr, 'BA': (tpr + (1 - fa)) / 2,
            'n_pos': len(pos), 'n_neg': len(neg)}

def eval_abstain(rows):
    """弃权口径：只在 answered 样本上算 FA/TPR/BA；弃权率单列"""
    pos = [r for r in rows if r.get('label_exists') is True]
    neg = [r for r in rows if r.get('label_exists') is False]
    if not pos or not neg:
        return None
    pos_a = [r for r in pos if answered(r)]
    neg_a = [r for r in neg if answered(r)]
    if not pos_a or not neg_a:
        return None
    fa = sum(1 for r in neg_a if is_yes(r)) / len(neg_a)
    tpr = sum(1 for r in pos_a if is_yes(r)) / len(pos_a)
    # 弃权率（响应完整性），分 pos/neg 报告暴露不对称
    abst_pos = 1 - len(pos_a) / len(pos)
    abst_neg = 1 - len(neg_a) / len(neg)
    abst_all = 1 - (len(pos_a) + len(neg_a)) / (len(pos) + len(neg))
    # by-type FA（neg）在 answered 上
    by = {}
    for h in HT4:
        sub = [r for r in neg_a if r.get('hallucination_type') == h]
        by[h] = (sum(1 for r in sub if is_yes(r)) / len(sub)) if sub else None
    return {'FA': fa, 'TPR': tpr, 'BA': (tpr + (1 - fa)) / 2,
            'n_pos_ans': len(pos_a), 'n_neg_ans': len(neg_a),
            'abstain_pos': abst_pos, 'abstain_neg': abst_neg,
            'abstain_all': abst_all, 'FA_by_type': by,
            'ROH': st.mean([by[h] for h in HT4[2:]]) if all(by[h] is not None for h in HT4[2:]) else None,
            'BOH': st.mean([by[h] for h in HT4[:2]]) if all(by[h] is not None for h in HT4[:2]) else None}

if __name__ == '__main__':
    out = {}
    rows_summary = []
    for m in sorted(canon):
        rows = load_t1(m)
        if not rows:
            continue
        cur = eval_current(rows)
        ab = eval_abstain(rows)
        if not cur or not ab:
            continue
        out[m] = {'current': cur, 'abstain': ab,
                  'delta_BA_abstain_minus_current': ab['BA'] - cur['BA'],
                  'delta_FA': ab['FA'] - cur['FA'],
                  'delta_TPR': ab['TPR'] - cur['TPR']}
        rows_summary.append((m, cur['BA'], ab['BA'], ab['abstain_all'],
                             ab['abstain_pos'], ab['abstain_neg']))

    # 打印对比表
    print("=== 当前口径(parse-fail→NO) vs 弃权口径(abstention) ===")
    print(f"{'model':<18}{'BA_cur':>8}{'BA_abs':>8}{'ΔBA':>8}{'abst%':>7}{'abst_pos%':>10}{'abst_neg%':>10}")
    for m, bac, baa, aba, abp, abn in sorted(rows_summary, key=lambda x: -abs(x[2]-x[1])):
        print(f"{m:<18}{bac*100:>7.2f}{baa*100:>8.2f}{(baa-bac)*100:>+8.2f}{aba*100:>6.1f}{abp*100:>9.1f}{abn*100:>10.1f}")

    # 跨模型排序一致性：当前口径 BA 排序 vs 弃权口径 BA 排序
    cur_rank = [m for m, _, _, _, _, _ in sorted(rows_summary, key=lambda x: -x[1])]
    abs_rank = [m for m, _, _, _, _, _ in sorted(rows_summary, key=lambda x: -x[2])]
    print(f"\n当前口径 BA 降序: {cur_rank}")
    print(f"弃权口径 BA 降序: {abs_rank}")
    # 排序变化
    moved = [m for m in cur_rank if cur_rank.index(m) != abs_rank.index(m)]
    print(f"排序改变的模型: {moved if moved else '无'}")

    with open('abstention.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n写入 abstention.json")
