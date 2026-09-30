#!/usr/bin/env python3
"""W1/E02 audit: is the relation route obtainable at inference time?

The collector selected probe rows with
    hallucination_type == 'relation' or query_role == 'positive'
Both fields are EVALUATION metadata. That does not leak labels into any fitted
head (the heads only ever see z0/gap), but it does mean the deployed policy as
written cannot be run on an unlabelled query.

A legal route may read only (query_text, fixed_config). The natural one already
exists inside the pipeline: simple_relations.build(q) returns competitive rivals
only when the query contains a known spatial predicate. So define

    route(q) = (len(competitive_probes(q)) > 0)

and measure, over ALL 2500 rows per model:
  * coverage of the label-based route by the text-only route
  * how many rows the text route ADDS (non-relation rows carrying a predicate)
  * whether the added rows change any decision, i.e. does the gate stay additive
    when it is driven by text alone

This script only reads cached records + cached scores. No GPU, no re-scoring.
"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/SVD/grpo_verifier'))
import simple_relations as sr
import eval_upstream as E

MAXR = 3


def records_for(root, model):
    """Yield the canonical DG records for one model (t2)."""
    for sub in ('t2_vqa_grounding', 't2'):
        p = os.path.join(root, model, sub, 'records.jsonl')
        if os.path.exists(p):
            for line in open(p, encoding='utf-8'):
                line = line.strip()
                if line:
                    yield json.loads(line)
            return
    d = os.path.join(root, model)
    if not os.path.isdir(d):
        return
    for dirpath, _, files in os.walk(d):
        if 'records.jsonl' not in files:
            continue
        if 't2' not in dirpath and 'grounding' not in dirpath:
            continue
        for line in open(os.path.join(dirpath, 'records.jsonl'), encoding='utf-8'):
            line = line.strip()
            if line:
                yield json.loads(line)
        return


def main():
    canon = json.load(open(os.path.join(HERE, 'canon_roots_paper.json')))
    models = sorted(canon)

    tot = Counter()
    per_model = {}
    disagree_examples = []

    for m in models:
        c = Counter()
        for r in records_for(canon[m], m):
            ht = r.get('hallucination_type')
            role = r.get('query_role')
            q = r.get('query') or r.get('referring_expression') or ''
            if not q:
                continue
            c['rows'] += 1
            # the route actually used by the collector (reads eval metadata)
            label_route = (ht == 'relation' or role == 'positive')
            # a legal route: text only
            try:
                probes, _ = sr.build(q, max_candidates=MAXR)
                text_route = any(p['probe_type'] == 'competitive' for p in probes)
            except Exception:
                text_route = False
            c['label_route'] += label_route
            c['text_route'] += text_route
            key = (label_route, text_route)
            c['both' if key == (True, True) else
              'label_only' if key == (True, False) else
              'text_only' if key == (False, True) else 'neither'] += 1
            if key == (False, True) and len(disagree_examples) < 12:
                disagree_examples.append((m, ht, role, q[:72]))
            # what htypes does the text route pick up?
            if text_route:
                c['text_ht_' + str(ht or role)] += 1
        per_model[m] = c
        for k, v in c.items():
            tot[k] += v
        print('%-16s rows=%4d  label_route=%4d  text_route=%4d  '
              'both=%4d label_only=%3d text_only=%4d'
              % (m, c['rows'], c['label_route'], c['text_route'],
                 c['both'], c['label_only'], c['text_only']))

    print('\n' + '=' * 96)
    print('汇总（13 模型 × 2500 行）')
    print('=' * 96)
    print('  总行数                     %6d' % tot['rows'])
    print('  标签路由选中（现状）        %6d' % tot['label_route'])
    print('  纯文本路由选中（合法）      %6d' % tot['text_route'])
    print('  两者一致选中               %6d' % tot['both'])
    print('  仅标签路由选中             %6d   <- 文本路由漏掉的' % tot['label_only'])
    print('  仅文本路由选中             %6d   <- 文本路由多出的' % tot['text_only'])
    if tot['label_route']:
        print('  文本路由对标签路由的召回   %.4f'
              % (tot['both'] / tot['label_route']))
    if tot['text_route']:
        print('  文本路由的精确率           %.4f'
              % (tot['both'] / tot['text_route']))

    print('\n文本路由选中的行按真实 htype 分布（说明它不依赖该字段）：')
    for k in sorted(k for k in tot if k.startswith('text_ht_')):
        print('  %-24s %6d' % (k.replace('text_ht_', ''), tot[k]))

    if disagree_examples:
        print('\n文本路由多选中的样例（现状路由会跳过它们）：')
        for m, ht, role, q in disagree_examples:
            print('  %-14s ht=%-14s role=%-9s %s' % (m, ht, role, q))

    json.dump(dict(total=dict(tot),
                   per_model={m: dict(c) for m, c in per_model.items()}),
              open(os.path.join(HERE, 'audit_routing.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('\nwrote audit_routing.json')


if __name__ == '__main__':
    main()
