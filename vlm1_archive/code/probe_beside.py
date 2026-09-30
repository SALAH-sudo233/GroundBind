#!/usr/bin/env python3
"""Test 'next to' (= beside / lateral adjacency) as a competitive rival.

Reading that motivates this (user's correction): 'next to' is NOT a distance
statement, it is a CONFIGURATION statement -- the target is alongside the
reference, on the same level. Under that reading 'next to' is mutually
exclusive with 'under' / 'on top of' / 'behind' / 'above', because those name a
different configuration, not a different distance. That is why 'far from'
failed (0.4369): it negates distance, which the negatives never violate.

Two directions are tested:

  (A) rows whose predicate is vertical/depth  ->  rival 'next to'
      Motivated by the paired-transition audit: next to -> behind 63,
      -> on top of 29, -> above 12, -> under 12. For those NEGATIVE rows the
      TRUE relation is 'next to', so probing 'next to' should score HIGH while
      the stated relation scores lower = contradiction evidence.
      These rows are mostly negatives, so this is where the signal should be.

  (B) rows whose predicate is 'next to'  ->  rivals = the vertical/depth set,
      aggregated by max. A single rival was already tested ('behind', 0.5860);
      the negative may use any of them, so max over the family is the correct
      aggregation when the true rival is unknown at probe time.

z0 (the original expression score) is REUSED from actions_relation.jsonl:
same box, same prompt, same deterministic forward -- no need to re-score.
"""
import argparse, json, os, re, sys, time

sys.path.insert(0, '/home/u2025141034/SVD/jev_scout')
sys.path.insert(0, '/home/u2025141034/SVD/agentic_probe')

import simple_relations as sr

DEV = "/home/u2025141034/SVD/候选池miou=0缓解/data/refcocog_500_dev.semantic_strict.json"
IMG_ROOT = "/home/u2025141034/models/LENS/data/refcoco/train2014"
PRIOR = "/home/u2025141034/SVD/agentic_probe/actions_relation.jsonl"

# configuration rivals of lateral adjacency
VERT = ['behind', 'in front of', 'under', 'on top of', 'above', 'below']
BESIDE = ['next to', 'beside']


def build():
    data = json.load(open(DEV))
    rows = []
    for rec in data:
        if rec["hallucination_type"] != "relation":
            continue
        box = rec.get("positive_bbox") or rec.get("gt_bbox_xyxy")
        img = os.path.join(IMG_ROOT, rec["image_filename"])
        for pol, text in (("pos", rec["positive_text"]),
                          ("neg", rec["negative_text"])):
            probes, meta = sr.build(text, max_candidates=2)
            if meta["semantic_status"] != "supported":
                continue
            surf = meta["predicate_surface"]
            m = re.search(r'\b' + re.escape(surf) + r'\b', text, re.IGNORECASE)
            if not m:
                continue
            head, tail = text[:m.start()], text[m.end():]
            if surf in VERT:
                variants = [(rv, head + rv + tail) for rv in BESIDE]
                direction = 'A_vert_to_beside'
            elif surf == 'next to':
                variants = [(rv, head + rv + tail) for rv in VERT]
                direction = 'B_beside_to_vert'
            else:
                continue
            rows.append(dict(sid=rec["sample_id"], polarity=pol,
                             label=1 if pol == "pos" else 0,
                             image=img, image_filename=rec["image_filename"],
                             bbox=box, q0=text, predicate=surf,
                             direction=direction, variants=variants))
    rows.sort(key=lambda r: (r["sid"], r["polarity"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/home/u2025141034/models/OmniVerifier-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    rows = build()
    if args.smoke:
        rows = rows[:args.smoke]
    na = [r for r in rows if r['direction'] == 'A_vert_to_beside']
    nb = [r for r in rows if r['direction'] == 'B_beside_to_vert']
    print(f"A (vertical/depth -> beside): {len(na)} rows "
          f"(pos {sum(r['label'] for r in na)}/neg {len(na)-sum(r['label'] for r in na)})",
          flush=True)
    print(f"B (next to -> vertical/depth): {len(nb)} rows "
          f"(pos {sum(r['label'] for r in nb)}/neg {len(nb)-sum(r['label'] for r in nb)})",
          flush=True)
    print(f"queries = {sum(len(r['variants']) for r in rows)}", flush=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["sid"], r["polarity"], r["variant"]))
            except Exception:
                pass
        print(f"resume {len(done)}", flush=True)

    from eval_openverifier import OmniVerifier
    ver = OmniVerifier(args.model, device=args.device)
    print("model loaded", flush=True)

    fh = open(args.out, "a")
    t0, n = time.time(), 0
    for r in rows:
        for name, q in r["variants"]:
            if (r["sid"], r["polarity"], name) in done:
                continue
            try:
                z = ver.score(dict(image=r["image"], bbox=r["bbox"], phrase=q))
                err = None
            except Exception as exc:
                z, err = None, f"{type(exc).__name__}: {exc}"
            fh.write(json.dumps(dict(
                sid=r["sid"], polarity=r["polarity"], label=r["label"],
                image_filename=r["image_filename"], predicate=r["predicate"],
                direction=r["direction"], variant=name, query=q,
                z=z, error=err), ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {n} scored {(time.time()-t0)/n:.3f}s/it", flush=True)
    fh.close()
    print(f"ALLDONE scored={n} elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
