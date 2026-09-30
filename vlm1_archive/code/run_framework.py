#!/usr/bin/env python3
"""End-to-end agentic competitive-probe framework: collection + evaluation.

Pipeline (plan v0.2 sec 0/5, competitive-only after the 2026-09-27 audit):

    query + GT box
        |
    Omni scores the original expression            -> z0
        |
    probe builder: legal competitive rivals?  --no--> baseline path (z0 only)
        |yes
    Omni scores each rival                          -> z_c1, z_c2
        |
    contradiction = max_i(z_ci) - z0                (max = strongest rival)
        |
    decision head fitted on fold A, read once on fold B

Arms reported (plan v0.2 sec 8):
    B1     Omni original expression only            (main reference)
    Blex   predicate log-likelihood ratio, NO image (shortcut control)
    BlexO  Omni + predicate prior                   (shortcut+vision control)
    A0     fixed first rival + Omni                 (does the probe help at all)
    A2     all legal rivals, max aggregation        (full framework)
    A2+lex A2 + predicate prior                     (deployment-realistic)

Every arm is fitted on fold A and read ONCE on fold B. Image-level
md5(image_filename) % 2 folds. AUROC is rank-sum (ties -> 0.5).
CIs are paired bootstrap clustered on source image.
"""
import argparse, json, math, os, random, sys, time
from collections import defaultdict, Counter

sys.path.insert(0, '/home/u2025141034/SVD/jev_scout')
sys.path.insert(0, '/home/u2025141034/SVD/agentic_probe')

import simple_relations as sr

DEV = "/home/u2025141034/SVD/候选池miou=0缓解/data/refcocog_500_dev.semantic_strict.json"
IMG_ROOT = "/home/u2025141034/models/LENS/data/refcoco/train2014"
MAX_RIVALS = 2


# ---------------------------------------------------------------- rows
def build_rows(htypes):
    data = json.load(open(DEV))
    rows = []
    for rec in data:
        ht = rec["hallucination_type"]
        if ht not in htypes:
            continue
        box = rec.get("positive_bbox") or rec.get("gt_bbox_xyxy")
        img = os.path.join(IMG_ROOT, rec["image_filename"])
        for pol, text in (("pos", rec["positive_text"]),
                          ("neg", rec["negative_text"])):
            probes, meta = sr.build(text, max_candidates=MAX_RIVALS)
            comp = [p for p in probes if p["probe_type"] == "competitive"]
            rows.append(dict(
                sid=rec["sample_id"], htype=ht, polarity=pol,
                label=1 if pol == "pos" else 0,
                image=img, image_filename=rec["image_filename"], bbox=box,
                q0=text,
                predicate=meta["predicate_surface"] or "<none>",
                family=meta["family"],
                supported=bool(comp),
                skip_reason=meta["skip_reason"],
                rivals=[(p["edited_surface"], p["query"]) for p in comp],
            ))
    rows.sort(key=lambda r: (r["sid"], r["polarity"]))
    return rows


# ---------------------------------------------------------------- collect
def collect(rows, out, model, device):
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
                done.add((r["sid"], r["polarity"], r["variant"]))
            except Exception:
                pass
        print(f"resume: {len(done)} scored", flush=True)

    todo = sum(1 + len(r["rivals"]) for r in rows)
    print(f"rows={len(rows)}  queries<= {todo}", flush=True)

    from eval_openverifier import OmniVerifier
    ver = OmniVerifier(model, device=device)
    print("model loaded", flush=True)

    fh = open(out, "a")
    t0, n = time.time(), 0
    for r in rows:
        variants = [("original", r["q0"])] + \
                   [(s, q) for (s, q) in r["rivals"]]
        for name, q in variants:
            if (r["sid"], r["polarity"], name) in done:
                continue
            try:
                z = ver.score(dict(image=r["image"], bbox=r["bbox"], phrase=q))
                err = None
            except Exception as exc:
                z, err = None, f"{type(exc).__name__}: {exc}"
            fh.write(json.dumps(dict(
                sid=r["sid"], htype=r["htype"], polarity=r["polarity"],
                label=r["label"], image_filename=r["image_filename"],
                predicate=r["predicate"], family=r["family"],
                supported=r["supported"], variant=name, query=q,
                z=z, error=err), ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {n} scored {(time.time()-t0)/n:.3f}s/it "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
    fh.close()
    print(f"ALLDONE scored={n} elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/home/u2025141034/models/OmniVerifier-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--htypes", default="relation")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    hts = set(a.htypes.split(","))
    rows = build_rows(hts)
    sup = [r for r in rows if r["supported"]]
    print(f"htypes={sorted(hts)} rows={len(rows)} supported={len(sup)} "
          f"({len(sup)/len(rows):.1%})", flush=True)
    if a.smoke:
        rows = rows[:a.smoke]
    collect(rows, a.out, a.model, a.device)
