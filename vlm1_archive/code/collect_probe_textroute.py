#!/usr/bin/env python3
"""Label-independent probe collection (reviewer W1).

The earlier collection gated on `hallucination_type == 'relation' or
query_role == 'positive'`. Both fields are evaluation metadata, so the resulting
policy could not run at inference time. This script replaces that gate with

    route(query, fixed_config) = simple_relations.build(query) yields a
                                 competitive probe

which reads ONLY the query text and a frozen predicate table. It therefore also
probes object / co_occurrence / attribute rows whose queries happen to contain a
spatial predicate ("the red chair next to the person" is an attribute negative),
which the metadata gate silently skipped.

Nothing else changes: same scorer, same red-box rendering, same rival table,
same resume-by-key append. Output goes to a NEW file; the old one is untouched.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simple_relations as sr
from deploy_upstream import parse_bbox, img_path

MAX_RIVALS = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canon', default='canon_roots_paper.json')
    ap.add_argument('--omni', default=os.path.expanduser('~/models/OmniVerifier-7B'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--have', nargs='*', default=[])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    canon = json.load(open(os.path.join(HERE, a.canon)))

    have = set()
    for fn in a.have:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            for l in open(p, encoding='utf-8'):
                try:
                    d = json.loads(l)
                    have.add((d['model'], d['sid'], d['variant']))
                except Exception:
                    pass
    print('reusing %d existing probe scores' % len(have), flush=True)

    plan = []
    for m, root in sorted(canon.items()):
        p = os.path.join(root, m, 'records.jsonl')
        if not os.path.exists(p):
            print('[skip] %s: no records.jsonl' % m, flush=True)
            continue
        nr = nc = 0
        for line in open(p, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if str(r.get('task', '')).lower() not in ('t2', 't2_vqa_grounding'):
                continue
            q = r.get('query') or r.get('referring_expression') or ''
            probes, _ = sr.build(q, max_candidates=MAX_RIVALS)   # TEXT ONLY
            comp = [x for x in probes if x['probe_type'] == 'competitive']
            if not comp:
                continue
            bb = parse_bbox(r.get('pred_bbox_xyxy'))
            if bb is None:
                continue
            sid = r.get('sample_id') or r.get('base_sample_id')
            todo = [x for x in comp if (m, sid, x['edited_surface']) not in have]
            if not todo:
                continue
            nr += 1; nc += len(todo)
            plan.append((m, r, bb, q, todo))
        print('%-18s rows=%5d calls=%5d' % (m, nr, nc), flush=True)

    plan = [x for i, x in enumerate(plan) if i % a.nshard == a.shard]
    if a.smoke:
        plan = plan[:a.smoke]
    total = sum(len(c) for *_, c in plan)
    print('shard %d/%d rows=%d calls=%d est=%.1fmin'
          % (a.shard, a.nshard, len(plan), total, total * 0.13 / 60), flush=True)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding='utf-8'):
            try:
                d = json.loads(line)
                done.add((d['model'], d['sid'], d['variant']))
            except Exception:
                pass
        print('resume: %d already in out' % len(done), flush=True)

    from s5_omni_filter import Omni
    model = Omni(a.omni, 'cuda:0')
    print('omni loaded', flush=True)

    fh = open(a.out, 'a', encoding='utf-8')
    t0 = n = 0
    t0 = time.time()
    for m, r, bb, q, comp in plan:
        sid = r.get('sample_id') or r.get('base_sample_id')
        ip = img_path(r)
        if not ip:
            continue
        for pr in comp:
            key = (m, sid, pr['edited_surface'])
            if key in done:
                continue
            try:
                z = model.score(ip, bb, pr['query'])
                err = None
            except Exception as e:
                z, err = None, str(e)[:200]
            fh.write(json.dumps(dict(
                model=m, sid=sid, htype=r.get('hallucination_type'),
                query_role=r.get('query_role'), query=q,
                variant=pr['edited_surface'], rule_id=pr['rule_id'],
                routed_by='query_text', z=z, error=err,
                iou=r.get('iou'), label_exists=r.get('label_exists')),
                ensure_ascii=False) + '\n')
            n += 1
            if n % 200 == 0:
                fh.flush()
                el = (time.time() - t0) / 60
                print('  %d done  %.1fmin  %.2f/s' % (n, el, n / max(1e-9, el * 60)),
                      flush=True)
    fh.flush(); fh.close()
    print('ALLDONE n=%d %.1fmin' % (n, (time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
