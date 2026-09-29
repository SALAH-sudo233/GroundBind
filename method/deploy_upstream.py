#!/usr/bin/env python3
"""Deploy the competitive-probe framework onto REAL upstream predicted boxes.

Scope decision (from the 2026-09-27 per-htype evaluation): the probe only carries
signal on `relation`. On object / co_occurrence / attribute the fitted gap weight
has the OPPOSITE sign and A2-B1 is not significant, so probes are NOT called
there -- those rows keep the Omni single-arm decision. This saves ~1.2 forwards
per non-relation row and is the honest deployment form.

What this script adds: Omni scores for the competitive rivals of each relation
query, evaluated on the UPSTREAM PREDICTED box (drawn as a red rectangle).

What it reuses (not recomputed):
  z0 = s5grpo/s5omni_omni_<model>.jsonl   (original expression, same box/prompt)
  canonical run per model = canon_roots.json (13/13 resolved)

Denominator contract (skill: aligned metrics):
  FGR       = #{negatives still carrying a box} / ALL 2000 negatives  (official
              pred_found semantics; degenerate boxes are NOT excluded)
  pos_mIoU  = sum(IoU * keep) / 500          (rejected/no-box count as zero)
"""
import argparse, json, os, sys, time
from collections import Counter, defaultdict

sys.path.insert(0, '/home/u2025141034/SVD/grpo_verifier')
sys.path.insert(0, '/home/u2025141034/SVD/agentic_probe')

import simple_relations as sr

IMG_ROOT = '/home/u2025141034/models/LENS/data/refcoco/train2014'
CANON = '/home/u2025141034/SVD/agentic_probe/canon_roots.json'
MAX_RIVALS = 2


def parse_bbox(v):
    """Same contract as s5_omni_filter.parse_bbox: 4 finite numbers or None."""
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        b = [float(x) for x in v]
    except Exception:
        return None
    if any(x != x for x in b):
        return None
    return b


def img_path(rec):
    """Image name lives in base_sample_id, not in a dedicated field."""
    base = rec.get('base_sample_id') or rec.get('sample_id') or ''
    base = str(base).split('__')[0]
    i = base.find('COCO_')
    if i < 0:
        return None
    name = base[i:] + '.jpg'
    p = os.path.join(IMG_ROOT, name)
    return p if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--omni', default=os.path.expanduser('~/models/OmniVerifier-7B'))
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--canon', default=CANON,
                    help='canonical-run map; pass canon_roots_paper.json to match '
                         'the v0.48 Table 4 provenance (qwen2.5-vl-7b -> 11rep)')
    a = ap.parse_args()

    canon = json.load(open(a.canon))
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu

    # ---- plan the work first, so cost is known before any GPU load
    plan = []
    for m in [x.strip() for x in a.models.split(',') if x.strip()]:
        root = canon.get(m)
        if not root:
            print(f'[skip] {m}: no canonical run', flush=True)
            continue
        rp = os.path.join(root, m, 'records.jsonl')
        if not os.path.exists(rp):
            print(f'[skip] {m}: {rp} missing', flush=True)
            continue
        n_rel = n_probe = 0
        for line in open(rp, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if str(r.get('task', '')).lower() not in ('t2', 't2_vqa_grounding'):
                continue
            ht = r.get('hallucination_type')
            role = r.get('query_role')
            # relation negatives + ALL positives (positives are the shared
            # denominator for every htype)
            if not (ht == 'relation' or role == 'positive'):
                continue
            bb = parse_bbox(r.get('pred_bbox_xyxy'))
            if bb is None:
                continue          # upstream drew nothing -> nothing to probe
            q = r.get('query') or r.get('referring_expression') or ''
            probes, meta = sr.build(q, max_candidates=MAX_RIVALS)
            comp = [p for p in probes if p['probe_type'] == 'competitive']
            if not comp:
                continue
            n_rel += 1
            n_probe += len(comp)
            plan.append((m, r, bb, q, comp))
        print(f'{m:18s} probe-rows={n_rel:5d} probe-calls={n_probe:5d}', flush=True)

    if a.smoke:
        plan = plan[:a.smoke]
    total = sum(len(c) for _, _, _, _, c in plan)
    print(f'\nTOTAL probe rows={len(plan)} calls={total} '
          f'est={total*0.13/60:.1f}min', flush=True)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding='utf-8'):
            try:
                r = json.loads(line)
                done.add((r['model'], r['sid'], r['variant']))
            except Exception:
                pass
        print(f'resume: {len(done)} already scored', flush=True)

    from s5_omni_filter import Omni
    model = Omni(a.omni, 'cuda:0')
    print('omni loaded', flush=True)

    fh = open(a.out, 'a', encoding='utf-8')
    t0, n = time.time(), 0
    for m, r, bb, q, comp in plan:
        sid = r.get('sample_id') or r.get('base_sample_id')
        ip = img_path(r)
        if not ip:
            continue
        for p in comp:
            key = (m, sid, p['edited_surface'])
            if key in done:
                continue
            try:
                z = model.score(ip, bb, p['query'])
                err = None
            except Exception as exc:
                z, err = None, f'{type(exc).__name__}: {exc}'
            fh.write(json.dumps(dict(
                model=m, sid=sid,
                htype=r.get('hallucination_type') or r.get('query_role'),
                query_role=r.get('query_role'),
                label_exists=bool(r.get('label_exists')),
                iou=float(r.get('iou') or 0.0),
                variant=p['edited_surface'], rule_id=p['rule_id'],
                query=p['query'], z=z, error=err), ensure_ascii=False) + '\n')
            fh.flush()
            n += 1
            if n % 300 == 0:
                el = time.time() - t0
                print(f'  {n}/{total} {el/n:.3f}s/it {el/60:.1f}min', flush=True)
    fh.close()
    print(f'ALLDONE scored={n} elapsed={(time.time()-t0)/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
