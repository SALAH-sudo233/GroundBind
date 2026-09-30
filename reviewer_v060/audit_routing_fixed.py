#!/usr/bin/env python3
"""W1 audit, with the record-reader bug fixed.

The previous `audit_routing.py` looked for `<root>/<model>/<task>/records.jsonl`.
The real layout is `<root>/<model>/records.jsonl`, one file holding all four
tasks with a `task` field. Every counter therefore stayed at zero and
`audit_routing.json` was written as empty dicts -- the routing-legality claim had
no receipt behind it.

Question: can the relation route be computed WITHOUT evaluation metadata?
  label route (illegal at inference): hallucination_type=='relation' or role=='positive'
  text route  (legal): simple_relations.build(query) yields a competitive rival
Reads cached records only. No GPU.
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
import simple_relations as sr

MAXR = 2   # match collect_probe_textroute.py's MAX_RIVALS


def t2_records(root, model):
    p = os.path.join(root, model, 'records.jsonl')
    if not os.path.exists(p):
        return
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if str(r.get('task', '')).lower() in ('t2', 't2_vqa_grounding'):
            yield r


def main():
    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))
    tot = Counter()
    per_model = {}
    examples = []

    for m in sorted(canon):
        c = Counter()
        for r in t2_records(canon[m], m):
            q = r.get('query') or r.get('referring_expression') or ''
            if not q:
                continue
            ht, role = r.get('hallucination_type'), r.get('query_role')
            c['rows'] += 1
            label_route = (ht == 'relation' or role == 'positive')
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
            if text_route:
                c['ht_' + str(ht or role)] += 1
            if key == (False, True) and len(examples) < 10:
                examples.append((m, ht, q[:70]))
        per_model[m] = dict(c)
        for k, v in c.items():
            tot[k] += v
        print('%-16s rows=%4d label=%4d text=%4d both=%4d label_only=%4d text_only=%4d'
              % (m, c['rows'], c['label_route'], c['text_route'],
                 c['both'], c['label_only'], c['text_only']), flush=True)

    print('\n' + '=' * 92)
    print('13 models x 2500 DG rows')
    print('=' * 92)
    for k, lab in (('rows', 'total rows'), ('label_route', 'label route (illegal)'),
                   ('text_route', 'text route (legal)'), ('both', 'agree'),
                   ('label_only', 'label only = text route MISSES'),
                   ('text_only', 'text only  = text route ADDS')):
        print('  %-34s %6d' % (lab, tot[k]))
    if tot['label_route']:
        print('  recall of label route          %.4f' % (tot['both'] / tot['label_route']))
    if tot['text_route']:
        print('  precision vs label route       %.4f' % (tot['both'] / tot['text_route']))

    print('\ntext-route rows by TRUE htype (it does not read this field):')
    for k in sorted(k for k in tot if k.startswith('ht_')):
        print('  %-20s %6d' % (k[3:], tot[k]))
    if examples:
        print('\nrows the text route adds (label route skipped them):')
        for m, ht, q in examples:
            print('  %-15s ht=%-14s %s' % (m, ht, q))

    json.dump(dict(total=dict(tot), per_model=per_model),
              open(os.path.join(HERE, 'audit_routing_fixed.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('\nwrote audit_routing_fixed.json')


if __name__ == '__main__':
    main()
