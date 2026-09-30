#!/usr/bin/env python3
"""P1: offline collection of ALL legal probe actions on 500-dev relation rows.

Plan v0.2 sec 13/P1: before training any router, measure whether the extra
probe carries NEW discriminative information at all.

Protocol:
  - GT-box path (500-dev): the box is the annotated positive box, identical for
    the positive and the negative expression of a pair. So a probe can only
    change the TEXT condition -- exactly what we want to isolate.
  - Reuses the OmniVerifier red-rectangle + forced-'{"answer":' primitive from
    jev_scout/eval_openverifier.py. Does NOT reimplement scoring.
  - Scores, per row: the original expression q0 and every legal probe.
  - Writes one JSONL line per (row, query) so partial runs stay usable.

This is offline ACTION COLLECTION, not a policy. Selection/routing comes later
and must read only fold-A-fitted rules.
"""
import argparse, json, os, sys, time

sys.path.insert(0, '/home/u2025141034/SVD/jev_scout')
sys.path.insert(0, '/home/u2025141034/SVD/agentic_probe')

import simple_relations as sr

DEV = "/home/u2025141034/SVD/候选池miou=0缓解/data/refcocog_500_dev.semantic_strict.json"
IMG_ROOT = "/home/u2025141034/models/LENS/data/refcoco/train2014"


def build_rows(max_rows=None):
    """relation rows only; each (sample, polarity) becomes one row."""
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
            rows.append(dict(
                sid=rec["sample_id"], polarity=pol,
                label=1 if pol == "pos" else 0,
                image=img, image_filename=rec["image_filename"], bbox=box,
                q0=text,
                predicate=meta["predicate_surface"],
                family=meta["family"],
                n_legal=meta["n_legal_candidates"],
                probes=probes,
            ))
    rows.sort(key=lambda r: (r["sid"], r["polarity"]))
    if max_rows:
        rows = rows[:max_rows]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/home/u2025141034/models/OmniVerifier-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    rows = build_rows(args.max_rows or None)
    if args.smoke:
        rows = rows[:args.smoke]
    n_q = sum(1 + len(r["probes"]) for r in rows)
    print(f"rows={len(rows)}  total queries to score={n_q}", flush=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["sid"], r["polarity"], r["query"], r["probe_type"]))
            except Exception:
                pass
        print(f"resume: {len(done)} already scored", flush=True)

    from eval_openverifier import OmniVerifier
    ver = OmniVerifier(args.model, device=args.device)
    print("model loaded", flush=True)

    fh = open(args.out, "a")
    t0 = time.time()
    n = 0
    for r in rows:
        queries = [(r["q0"], "original", None)] + \
                  [(p["query"], p["probe_type"], p["rule_id"]) for p in r["probes"]]
        for q, ptype, rule in queries:
            key = (r["sid"], r["polarity"], q, ptype)
            if key in done:
                continue
            item = dict(image=r["image"], bbox=r["bbox"], phrase=q)
            try:
                z = ver.score(item)
                err = None
            except Exception as exc:
                z, err = None, f"{type(exc).__name__}: {exc}"
            fh.write(json.dumps(dict(
                sid=r["sid"], polarity=r["polarity"], label=r["label"],
                image_filename=r["image_filename"],
                predicate=r["predicate"], family=r["family"],
                n_legal=r["n_legal"],
                query=q, probe_type=ptype, rule_id=rule,
                z=z, error=err), ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 100 == 0:
                el = time.time() - t0
                print(f"  {n} scored  {el/n:.3f}s/it  elapsed {el/60:.1f}min",
                      flush=True)
    fh.close()
    print(f"ALLDONE scored={n} elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
