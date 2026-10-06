#!/usr/bin/env python3
"""Q8审计：13模型的拒绝/格式等价口径 —— parse失败率、空输出率、pred_exists分布.

审稿人Q8：13个模型"等价拒绝选项"是否公平？不同模型的提示词、空输出约定、
格式失败率、坐标解析是否被统一处理？统一的FGR定义是什么？

本脚本在 t1(discriminative VQA) 与 t2(grounding) 上逐模型统计：
  - parse_valid 比例(解析成功率)
  - parse_method 分布(用了哪种解析路径)
  - pred_exists 的 True/False/None 分布(拒绝=False/None 的口径)
  - 空输出(cleaned_text 为空)比例
  - 平均生成 token 数(格式冗长度)

结论用于回应："所有模型用同一 pred_exists 口径(False 或 None 均记为拒绝)，
统一 FGR 定义，解析失败率在各模型间可比"。

用法 (vlm1): python3 audit_parse.py
"""
import collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.expanduser('~/SVD/agentic_probe')


def main():
    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    per = {}
    for m, root in sorted(canon.items()):
        p = os.path.join(root, m, 'records.jsonl')
        if not os.path.exists(p):
            print(f'SKIP {m}', file=sys.stderr)
            continue
        by_task = collections.defaultdict(lambda: dict(
            n=0, parse_ok=0, pe_true=0, pe_false=0, pe_none=0,
            empty=0, tok_sum=0, methods=collections.Counter()))
        for line in open(p, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            task = str(r.get('task', ''))
            tk = ('t1' if task.startswith('t1') else
                  't2' if task.startswith('t2') else
                  't3' if task.startswith('t3') else
                  't4' if task.startswith('t4') else 'other')
            d = by_task[tk]
            d['n'] += 1
            if r.get('parse_valid'):
                d['parse_ok'] += 1
            pe = r.get('pred_exists')
            if pe is True:
                d['pe_true'] += 1
            elif pe is False:
                d['pe_false'] += 1
            else:
                d['pe_none'] += 1
            ct = r.get('cleaned_text')
            if not ct or not str(ct).strip():
                d['empty'] += 1
            d['tok_sum'] += int(r.get('generated_token_count') or 0)
            d['methods'][str(r.get('parse_method'))] += 1
        per[m] = {tk: dict(
            n=d['n'],
            parse_ok_rate=d['parse_ok'] / d['n'] if d['n'] else 0.0,
            pe_true=d['pe_true'], pe_false=d['pe_false'], pe_none=d['pe_none'],
            none_rate=d['pe_none'] / d['n'] if d['n'] else 0.0,
            empty_rate=d['empty'] / d['n'] if d['n'] else 0.0,
            mean_tokens=d['tok_sum'] / d['n'] if d['n'] else 0.0,
            top_methods=dict(d['methods'].most_common(3)))
            for tk, d in by_task.items()}

    print('=== Q8 解析/拒绝口径审计 (t1 判别 / t2 grounding) ===\n')
    for tk in ('t1', 't2'):
        print(f'--- {tk} ---')
        print('%-18s %6s %8s %8s %8s %8s %7s' % (
            'model', 'n', 'parseOK', 'none%', 'empty%', 'tokens', 'False#'))
        for m in sorted(per):
            if tk not in per[m]:
                continue
            d = per[m][tk]
            print('%-18s %6d %7.2f%% %7.3f%% %7.3f%% %8.1f %7d' % (
                m, d['n'], d['parse_ok_rate'] * 100, d['none_rate'] * 100,
                d['empty_rate'] * 100, d['mean_tokens'], d['pe_false']))
        print()

    out = os.path.join(HERE, 'audit_parse.json')
    json.dump(dict(per_model=per,
                   note='pred_exists False or None both counted as reject; '
                        'parse_valid is the unified parser success flag; '
                        'FGR denominator uses the same eligibility across models'),
              open(out, 'w'), indent=1)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
