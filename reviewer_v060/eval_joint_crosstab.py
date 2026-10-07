#!/usr/bin/env python3
"""Reviewer #5: 语言判断 × 定位行为的 exact query-level cross-tab。

reviewer 指出论文只报 marginal rates（Figure 4），没有报最直接的
per-query 联合统计。本脚本按 (model, sample_id) 精确配对 t1(语言判断) 与
t2(定位行为)，在 negative query 上交叉制表：

                      grounding=empty      grounding=box
  language=unsupported  correct refusal    LANGUAGE-GROUNDING CONTRADICTION
  language=supported    semantic FA only   false acceptance + grounding

join 键 = (model, sample_id)，sample_id 已含幻觉类型后缀，保证同一
(image, expression, model, run) 的语言与定位来自同一条样本。

语言判断口径（abstention-aware，与 Q8 修正一致）：
  pred_exists is True  → supported(says yes)
  pred_exists is False → unsupported(says no)
  pred_exists is None / parse_valid False → ABSTAIN(单列，不塞进 yes/no)

定位口径：pred_found True 或 pred_bbox 可解析为 4 元素 → emit box；否则 empty。
"""
import json, os, sys

PROBE = os.path.expanduser('~/SVD/agentic_probe')
CANON = os.path.join(PROBE, 'canon_roots_paper.json')


def parse_bbox(v):
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        return v
    return None


def emits_box(rec):
    if rec.get('pred_found') is True:
        return True
    return parse_bbox(rec.get('pred_bbox_xyxy')) is not None


def main():
    canon = json.load(open(CANON))
    models = sorted(canon)

    # 全局 cross-tab（负例）+ 逐模型 + 逐 htype
    def new_cell():
        return dict(unsup_empty=0, unsup_box=0, sup_empty=0, sup_box=0,
                    abstain_empty=0, abstain_box=0, n_neg=0)

    glob = new_cell()
    per_model = {}
    per_htype = {}

    for m in models:
        rp = os.path.join(canon[m], m, 'records.jsonl')
        if not os.path.exists(rp):
            print(f'MISSING {rp}', file=sys.stderr)
            continue
        # 收集 t1(语言) 与 t2(定位)，按 sample_id 配对
        t1 = {}
        t2 = {}
        for line in open(rp, encoding='utf-8'):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = str(rec.get('task', '')).lower()
            sid = rec.get('sample_id')
            if sid is None:
                continue
            if t == 't1_discriminative_vqa':
                t1[sid] = rec
            elif t == 't2_vqa_grounding':
                t2[sid] = rec

        pm = per_model.setdefault(m, new_cell())
        for sid, r1 in t1.items():
            # 只统计 negative query（label_exists False）
            if r1.get('label_exists') is not False:
                continue
            r2 = t2.get(sid)
            if r2 is None:
                continue  # 无配对定位记录
            box = emits_box(r2)
            pe = r1.get('pred_exists')
            pv = r1.get('parse_valid')
            ht = r1.get('hallucination_type', 'unknown')
            ph = per_htype.setdefault(ht, new_cell())

            for cell in (glob, pm, ph):
                cell['n_neg'] += 1

            if pv is False or pe is None:
                key = 'abstain_box' if box else 'abstain_empty'
            elif pe is True:  # says supported
                key = 'sup_box' if box else 'sup_empty'
            else:  # says unsupported
                key = 'unsup_box' if box else 'unsup_empty'
            for cell in (glob, pm, ph):
                cell[key] += 1

    def rates(c):
        n = c['n_neg'] or 1
        return {k: dict(count=c[k], rate=round(100.0 * c[k] / n, 2))
                for k in ('unsup_empty', 'unsup_box', 'sup_empty', 'sup_box',
                          'abstain_empty', 'abstain_box')}

    out = dict(
        description='negative-query language(t1) x grounding(t2) cross-tab, '
                    'joined by (model, sample_id)',
        n_models=len(per_model),
        global_counts=glob,
        global_rates=rates(glob),
        per_model={m: dict(counts=c, rates=rates(c))
                   for m, c in per_model.items()},
        per_htype={h: dict(counts=c, rates=rates(c))
                   for h, c in per_htype.items()},
    )
    with open(os.path.join(PROBE, 'joint_crosstab.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # 打印摘要
    g = glob
    n = g['n_neg']
    print(f'=== Negative-query language x grounding cross-tab (n={n}) ===')
    print(f'{"":22}{"empty box":>14}{"emits box":>14}')
    print(f'{"says UNSUPPORTED":22}{g["unsup_empty"]:>8}'
          f'({100*g["unsup_empty"]/n:.1f}%){g["unsup_box"]:>6}'
          f'({100*g["unsup_box"]/n:.1f}%)  <- CONTRADICTION')
    print(f'{"says SUPPORTED":22}{g["sup_empty"]:>8}'
          f'({100*g["sup_empty"]/n:.1f}%){g["sup_box"]:>6}'
          f'({100*g["sup_box"]/n:.1f}%)')
    print(f'{"ABSTAIN(parse-fail)":22}{g["abstain_empty"]:>8}'
          f'({100*g["abstain_empty"]/n:.1f}%){g["abstain_box"]:>6}'
          f'({100*g["abstain_box"]/n:.1f}%)')
    contra = 100 * g['unsup_box'] / n
    print(f'\nLANGUAGE-GROUNDING CONTRADICTION rate = {contra:.2f}% '
          f'(model says unsupported yet emits a box)')
    print('\nper-htype contradiction rate:')
    for h, c in sorted(per_htype.items()):
        nn = c['n_neg'] or 1
        print(f'  {h:14} {100*c["unsup_box"]/nn:6.2f}%  (n={c["n_neg"]})')
    print('\nwrote joint_crosstab.json')


if __name__ == '__main__':
    main()
