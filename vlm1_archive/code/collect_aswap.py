#!/usr/bin/env python3
"""Aswap control (plan v0.2 section 8 + section 18.6): does the EXTRA verification
call actually use THIS image?

Design constraint from the plan: hold the original verdict, the candidate set and
the control flow FIXED, and replace ONLY the probe's image input with an
independent mismatched image. Replacing the original verdict and the gate at the
same time would confound the cause.

So this script re-scores every cached competitive rival with OmniVerifier on a
WRONG image, keeping the query text and the box coordinates identical. Offline we
can then replay any selection rule (fixed / random / 2B-selected) against these
scores, which is why all rivals are scored rather than just one.

Wrong-image selection is deterministic, seeded, and LABEL-BLIND:
  * candidate pool = the sorted set of images in this subset
  * index = md5(model|sid) % N, walking forward until the image differs from the
    row's own image AND the original box fits inside the wrong image's real
    width/height
  * if no such image exists the row is SKIPPED and counted (never re-shaped to
    make a result look better -- plan section 8: "原框无法安全映射到错图尺寸时跳过并报告")

A mismatched image is NOT a correct negative label, and this arm is an ablation,
never a deployment action.
"""
import argparse
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES = os.path.join(HERE, 'probe_upstream.jsonl')
CANON = os.path.join(HERE, 'canon_roots.json')
IMG_ROOT = os.path.expanduser('~/models/LENS/data/refcoco/train2014')


def parse_bbox(v):
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


def img_name(sid):
    base = str(sid).split('__')[0]
    i = base.find('COCO_')
    return base[i:] + '.jpg' if i >= 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args()

    from PIL import Image

    # ---- cached rivals (query text + the Omni score on the REAL image)
    rivals = {}
    for line in open(PROBES, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('error') or r.get('z') is None:
            continue
        rivals.setdefault((r['model'], r['sid']), []).append(
            dict(variant=r['variant'], query=r['query'], z_real=r['z']))

    # ---- boxes from the canonical runs
    canon = json.load(open(CANON))
    box = {}
    for m in sorted(canon):
        rp = os.path.join(canon[m], m, 'records.jsonl')
        if not os.path.exists(rp):
            continue
        for line in open(rp, encoding='utf-8'):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if str(rec.get('task', '')).lower() not in ('t2', 't2_vqa_grounding'):
                continue
            sid = rec.get('sample_id') or rec.get('base_sample_id')
            bb = parse_bbox(rec.get('pred_bbox_xyxy'))
            if bb is not None:
                box[(m, sid)] = bb

    # ---- image pool + real dimensions (cached once; label-blind)
    pool = sorted({img_name(sid) for (_, sid) in rivals if img_name(sid)})
    dims = {}
    for nm in pool:
        p = os.path.join(IMG_ROOT, nm)
        try:
            with Image.open(p) as im:
                dims[nm] = im.size          # (W, H)
        except Exception:
            pass
    pool = [nm for nm in pool if nm in dims]
    print('image pool=%d' % len(pool), flush=True)

    plan, n_skip = [], 0
    for key, rv in sorted(rivals.items()):
        m, sid = key
        own = img_name(sid)
        bb = box.get(key)
        if bb is None or own not in dims:
            n_skip += 1
            continue
        h = int(hashlib.md5(('%s|%s' % (m, sid)).encode()).hexdigest(), 16)
        chosen = None
        for off in range(len(pool)):
            nm = pool[(h + off) % len(pool)]
            if nm == own:
                continue
            W, H = dims[nm]
            if bb[0] >= 0 and bb[1] >= 0 and bb[2] <= W and bb[3] <= H:
                chosen = nm
                break
        if chosen is None:
            n_skip += 1
            continue
        plan.append((m, sid, bb, own, chosen, rv))
    print('rows=%d skipped(no safe wrong image)=%d' % (len(plan), n_skip), flush=True)

    if a.nshards > 1:
        plan = [r for r in plan
                if int(hashlib.md5(('%s|%s' % (r[0], r[1])).encode()).hexdigest(), 16)
                % a.nshards == a.shard]
    if a.smoke:
        plan = plan[:a.smoke]
    calls = sum(len(r[5]) for r in plan)
    print('shard rows=%d calls=%d est=%.1fmin'
          % (len(plan), calls, calls * 0.13 / 60), flush=True)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding='utf-8'):
            try:
                d = json.loads(line)
                done.add((d['model'], d['sid'], d['variant']))
            except Exception:
                pass
        print('resume: %d already scored' % len(done), flush=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    import sys
    sys.path.insert(0, os.path.expanduser('~/SVD/grpo_verifier'))
    from s5_omni_filter import Omni
    model = Omni(os.path.expanduser('~/models/OmniVerifier-7B'), 'cuda:0')
    print('omni loaded', flush=True)

    fh = open(a.out, 'a', encoding='utf-8')
    t0, n = time.time(), 0
    for m, sid, bb, own, wrong, rv in plan:
        wp = os.path.join(IMG_ROOT, wrong)
        for c in rv:
            if (m, sid, c['variant']) in done:
                continue
            try:
                z = model.score(wp, bb, c['query'])
                err = None
            except Exception as exc:
                z, err = None, '%s: %s' % (type(exc).__name__, exc)
            fh.write(json.dumps(dict(
                model=m, sid=sid, variant=c['variant'], query=c['query'],
                own_image=own, wrong_image=wrong,
                z_real=c['z_real'], z_swap=z, error=err),
                ensure_ascii=False) + '\n')
            fh.flush()
            n += 1
            if n % 300 == 0:
                el = time.time() - t0
                print('  %d/%d %.3fs/it %.1fmin' % (n, calls, el / n, el / 60),
                      flush=True)
    fh.close()
    print('ALLDONE scored=%d elapsed=%.1fmin' % (n, (time.time() - t0) / 60),
          flush=True)


if __name__ == '__main__':
    main()
