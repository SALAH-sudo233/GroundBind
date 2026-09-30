#!/usr/bin/env python3
"""Audit candidate competitive rivals for the predicate 'next to'.

Question: is 'far from' the right competitive probe for 'next to'?

'next to' is the single most frequent relation predicate in 500-dev relation
queries (177 positives) but currently has NO competitive rival, so it is the
largest coverage gap in the probe library.

Candidates tested (all as competitive probes on 'next to' rows):
  far from   - the semantic antonym (user hypothesis)
  away from  - antonym variant, guards against surface-form artefacts
  behind     - the empirically most common paired substitution (33.0%),
               included as a CONTRAST: it is not logically exclusive with
               'next to', so if it discriminates better than the antonym the
               mechanism is depth, not distance.

A rival is useful iff  -(z_rival - z_0)  separates positives from negatives.
Reuses OmniVerifier.score (red rectangle + forced '{"answer":' prefix).
"""
import argparse, json, os, re, sys, time

sys.path.insert(0, '/home/u2025141034/SVD/jev_scout')
sys.path.insert(0, '/home/u2025141034/SVD/agentic_probe')

DEV = "/home/u2025141034/SVD/候选池miou=0缓解/data/refcocog_500_dev.semantic_strict.json"
IMG_ROOT = "/home/u2025141034/models/LENS/data/refcoco/train2014"

RIVALS = ["far from", "away from", "behind"]
_NX = re.compile(r'\bnext to\b', re.IGNORECASE)


def build_rows():
    """Every (sample, polarity) row whose predicate surface is exactly 'next to'."""
    data = json.load(open(DEV))
    rows = []
    for rec in data:
        if rec["hallucination_type"] != "relation":
            continue
        box = rec.get("positive_bbox") or rec.get("gt_bbox_xyxy")
        img = os.path.join(IMG_ROOT, rec["image_filename"])
        for pol, text in (("pos", rec["positive_text"]),
                          ("neg", rec["negative_text"])):
            m = _NX.search(text or "")
            if not m:
                continue
            # single-predicate rows only: 'standing next to' / 'sitting next to'
            # are different surfaces and are excluded by requiring that the
            # match is not preceded by a participle.
            head = text[:m.start()].rstrip()
            if head.lower().endswith(("standing", "sitting", "lying", "leaning")):
                continue
            rows.append(dict(
                sid=rec["sample_id"], polarity=pol,
                label=1 if pol == "pos" else 0,
                image=img, image_filename=rec["image_filename"], bbox=box,
                q0=text, start=m.start(), end=m.end(),
            ))
    rows.sort(key=lambda r: (r["sid"], r["polarity"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/home/u2025141034/models/OmniVerifier-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    rows = build_rows()
    if args.smoke:
        rows = rows[:args.smoke]
    print(f"'next to' rows={len(rows)} "
          f"(pos {sum(r['label'] for r in rows)} / "
          f"neg {len(rows)-sum(r['label'] for r in rows)})", flush=True)
    print(f"queries to score = {len(rows)} x (1 + {len(RIVALS)}) = "
          f"{len(rows)*(1+len(RIVALS))}", flush=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["sid"], r["polarity"], r["variant"]))
            except Exception:
                pass
        print(f"resume: {len(done)} scored", flush=True)

    from eval_openverifier import OmniVerifier
    ver = OmniVerifier(args.model, device=args.device)
    print("model loaded", flush=True)

    fh = open(args.out, "a")
    t0, n = time.time(), 0
    for r in rows:
        variants = [("original", r["q0"])]
        for rv in RIVALS:
            variants.append((rv, r["q0"][:r["start"]] + rv + r["q0"][r["end"]:]))
        for name, q in variants:
            if (r["sid"], r["polarity"], name) in done:
                continue
            try:
                z = ver.score(dict(image=r["image"], bbox=r["bbox"], phrase=q))
                err = None
            except Exception as exc:
                z, err = None, f"{type(exc).__name__}: {exc}"
            fh.write(json.dumps(dict(
                sid=r["sid"], polarity=r["polarity"], label=r["label"],
                image_filename=r["image_filename"],
                variant=name, query=q, z=z, error=err), ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 100 == 0:
                print(f"  {n} scored {(time.time()-t0)/n:.3f}s/it", flush=True)
    fh.close()
    print(f"ALLDONE scored={n} elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
