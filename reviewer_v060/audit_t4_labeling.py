#!/usr/bin/env python3
"""Q2审计：t2与t4的标签定义是否真的不同？drew(合格性)影响多大？

审稿人Q2前提："joint扩展改用IoU条件标签"。核查代码后前提不成立：

  t2 (eval_paper_tables): label = 1 if (is_pos and iou >= 0.5) else 0
  t4 (eval_t4_mitigation L112): label = 1 if (is_pos and iou >= 0.5) else 0

两者**完全一致**，都是纯IoU≥0.5。没有任何"valid"进入label。

真实差异在**合格集(eligibility)**，不在label：
  t2 elig: valid_box(pred) and iou >= 0.5   —— 画了合法框的正例
  t4 elig: drew and iou >= 0.5               —— drew = box存在且w>0且h>0

两者都是"模型画了可用框的正例"。t4的drew等价于t2的valid_box。
本脚本量化：t4正例里 IoU≥0.5 但 not drew 的比例（被合格性排除的正例数）。
若接近0，说明两任务合格集定义等价，无口径差异。

用法 (vlm1 ~/SVD/agentic_probe):
    python3 audit_t4_labeling.py
"""
import collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
sys.path.insert(0, HERE)

TASK = 't4_caption_grounding'


def main():
    import eval_t4_mitigation as T4
    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    COORDFIX = getattr(T4, 'COORDFIX_T4', set())
    CF = getattr(T4, 'CF', None)

    pos_total = 0
    pos_iou_ge_half = 0
    pos_iou_ge_half_drew = 0
    pos_iou_ge_half_notdrew = 0
    per_model = collections.defaultdict(lambda: [0, 0, 0])  # ge_half, drew, notdrew

    for m, root in sorted(canon.items()):
        p = os.path.join(root, m, 'records.jsonl')
        if not os.path.exists(p):
            print(f'SKIP {m}: no records.jsonl', file=sys.stderr)
            continue
        cache = {}
        for line in open(p, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if str(r.get('task', '')) != TASK:
                continue
            if r.get('query_role') != 'positive':
                continue
            pos_total += 1
            if m in COORDFIX and CF is not None:
                box = CF.rescale(r, cache)
                iou = CF.iou(box, r.get('gt_bbox_xyxy')) if box else 0.0
            else:
                box = r.get('pred_bbox_xyxy')
                iou = r.get('iou') or 0.0
            if iou < 0.5:
                continue
            pos_iou_ge_half += 1
            drew = bool(box) and (box[2] - box[0]) > 0 and (box[3] - box[1]) > 0
            per_model[m][0] += 1
            if drew:
                pos_iou_ge_half_drew += 1
                per_model[m][1] += 1
            else:
                pos_iou_ge_half_notdrew += 1
                per_model[m][2] += 1

    frac = (pos_iou_ge_half_notdrew / pos_iou_ge_half
            if pos_iou_ge_half > 0 else 0.0)

    print('=== t2/t4 标签与合格集审计 (Q2) ===')
    print(f'正例总数: {pos_total}')
    print(f'IoU≥0.5 正例: {pos_iou_ge_half}')
    print(f'  其中 drew(合格): {pos_iou_ge_half_drew}')
    print(f'  其中 not drew(被合格性排除): {pos_iou_ge_half_notdrew}')
    print(f'被排除占比: {frac*100:.4f}%')
    print()
    print('逐模型 (IoU≥0.5 | drew | not_drew):')
    for m in sorted(per_model):
        g, d, nd = per_model[m]
        print(f'  {m:18s} {g:5d} | {d:5d} | {nd:3d}')
    print()
    if frac < 0.01:
        print('✓ 结论：IoU≥0.5的正例几乎全部drew。t4的drew等价于t2的valid_box，')
        print('  两任务合格集定义一致，label定义完全相同（纯IoU≥0.5）。')
        print('  审稿人Q2的"t4改用IoU条件标签"前提不成立。')
    else:
        print(f'⚠ {frac*100:.2f}% 的IoU≥0.5正例未drew，需在回复中说明合格集差异。')

    out = os.path.join(HERE, 'audit_t4_labeling.json')
    json.dump(dict(
        pos_total=pos_total, pos_iou_ge_half=pos_iou_ge_half,
        drew=pos_iou_ge_half_drew, not_drew=pos_iou_ge_half_notdrew,
        excluded_fraction=frac,
        per_model={m: dict(iou_ge_half=v[0], drew=v[1], not_drew=v[2])
                   for m, v in per_model.items()},
        note='t2 and t4 use identical label = (is_pos and iou>=0.5); '
             'drew(t4) == valid_box(t2) eligibility, not a label change'),
        open(out, 'w'), indent=1)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
