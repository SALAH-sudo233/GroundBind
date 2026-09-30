# Snapshot audit

Tracks implementation/report differences and the limits of what has been validated.
Updated 2026-09-30 after the v0.60 review cycle; items 1, 2 and 5 are now **resolved
with receipts**, and one new item (6) records a difference against the paper PDF.

## Resolved

1. **Additional rejection versus replacement decision — RESOLVED: the figure is
   wrong, the code is right.** `eval_combo.py` `keep_a2()` returns the probe-head
   decision for covered, usable rows without requiring `keep_base(r)` first, so it
   can also rescue a rejected row. That behaviour **matches Eq. (10)**, the
   query-routed composite policy: the routed branch is not multiplied by a
   preceding support vote. The inconsistency is therefore in **Figure 4**, which
   still draws a prior support gate on the routed branch. Fix the figure, not the
   code. Previously this item warned against calling it a monotone rejection stage;
   that warning stands for the *composite* policy, but note the repo's reported
   tables now deliberately use a monotone gate (see item 6), which is a separate,
   explicitly labelled policy choice.

2. **Routing provenance — RESOLVED, with a caveat that changes a published claim.**
   Label-free query-only routing is now established: the text router reads only the
   query and the frozen predicate table. Receipt: `reviewer_v060/audit_routing_fixed.json`.
   **Caveat:** the original `audit_routing.py` never actually ran — its
   `records_for()` looked for `<root>/<model>/<task>/records.jsonl` while the real
   layout is `<root>/<model>/records.jsonl` with a `task` field, so every counter
   stayed 0 and `audit_routing.json` was all-empty dicts. The legality claim had
   **no receipt at all** before this was fixed. The corrected audit also shows the
   router is **not relation-only**: it fires on all four types (object 0.238,
   co_occurrence 0.326, attribute 0.349, relation 0.490 of rows). Consequently the
   old report line "the other three types change by exactly 0.00pp" was an artifact
   of an **illegal metadata gate** reading `hallucination_type`, not a property of
   the method. The broken script is kept as
   `vlm1_archive/code/audit_routing_BROKEN_readpath.py`.

5. **Evaluation limits — the named controls have now been run.** Text-prior,
   matched-retention and routing audits are done: `reviewer_v060/attribution.json`
   (text prior, equal-budget paraphrase, wrong-image), `matched_retention.json`
   (achieved-retention curves, Eq. (12) feasibility 26/26 folds, image-clustered CI
   over the 500 source images), `audit_routing_fixed.json` (routing). The fixed
   eligibility set, full FGR/mIoU denominators and image-level split discipline are
   preserved throughout. Single-arm controls remain necessary ablations.

## Open

3. **Training/scoring prompt provenance.** `jev_verifier.py` declares a shorter
   prompt than `collect_jevhead.py`'s required JSON-response template. Check the
   saved training run and checkpoint provenance before claiming byte-identical
   training and scoring prompts. Do not silently rewrite either template.

4. **Reproduction dependencies — scope corrected 2026-10-01.** This *checkout* lacks
   the per-model canonical upstream `records.jsonl`, verifier score caches, model
   weights, image files and the JEV adapter/head/temperature, so source availability
   plus imported result JSON is not a fresh reproduction **of this directory alone**.
   That is a statement about the repo, not about the work: **the full pipeline does
   exist on vlm1 and was verified element by element on 2026-10-01** — 13/13
   `records.jsonl` present, 82,783 images, JEV adapter + `head_final.pt` +
   `temperature.json` (T = 1.7765671239028529), OmniVerifier-7B and all 13 upstream
   checkpoints. Paths, the four reproduction levels (render / CPU recompute / GPU
   collection / training) and the acceptance anchors are recorded in `REPRODUCE.md`.
   An earlier version of this item was worded in a way that invited the reading
   "the pipeline cannot be reproduced"; that reading was wrong.

6. **NEW — the repo's mitigation row differs from the paper PDF's third row.**
   Recomputation reproduces Table 3's **first two rows** bit-for-bit (max deviation
   0.018pp, rounding). The **third row differs on every entry, each in our favour**:
   FGR 21.55 → 19.30, attribute CBR 10.21 → 8.50, relation CBR 19.29 → 18.56,
   retention 91.99 → 91.51, mIoU 0.3884 → 0.3860.
   Cause: the PDF's third row lets the probe head decide **every routed row**. With
   the legal text router that spans all four types, so the three non-relation types
   were handed to a head fitted **pooled across types**, whose gap weight has
   opposite signs per type (object +0.418, co_occurrence +0.334, attribute +0.283,
   relation −0.673). That is the source of the PDF's attribute CBR rebound from 8.55
   to 10.21. The repo instead uses a **monotone gate** (probe head may only turn
   keep→reject) **restricted to predicates with a genuine opposing configuration**;
   blocked rows keep the support decision bit-for-bit. Justification: object,
   co_occurrence and attribute are mitigated by a single Omni posterior and never
   enter predicate-table competitive scoring, so the full arm must be strictly
   additive on them. Receipts: `reviewer_v060/paper_tables.json`,
   `final_policy.json`, `type_gate.json`, and the retraction record in
   `reviewer_v060/CORRECTION.md`. **The paper text must be updated to this policy
   and these numbers, or the discrepancy explained.**

## Numbers that must not be cited

- ALL FGR 59.20% → 22.15%, relation CBR 40.1% → 18.7%, positive mIoU 0.3738 →
  0.3549 — inherited v0.51 values produced under the illegal metadata gate.
- "Mitigation fires only on relation; the other three types change by exactly
  0.00pp" — same illegal gate.
- "Specialist models fail to localize 94% of positives" — a coordinate-convention
  bug, not model capability (UniVG-R1 30 → 254 successes, mIoU 0.1122 → 0.4936;
  visual-rft 26 → 211, 0.1047 → 0.4114).
- Cross-scorer subtraction null (Pearson r = 0.32) and the alphabetical-rival
  control arm (AUROC 0.8651, leaks the predicate prior).
- The original-coordinate four-type CBR for the two corrected models
  (UniVG-R1 33.3/46.7/40.0/46.7 at n_c=30; visual-rft 0.0/11.5/19.2/11.5 at n_c=26)
  — version audit only; at that sample size the difficulty ladder does not hold.

## Validation scope

Validated on the Mac: preserved-source SHA-256 checks, Python syntax, local helper
imports, the relation-template tests. Validated on vlm1 (CPU only, no GPU, no model
re-inference): the recomputations listed above, each gated on reproducing a
published anchor — `eval_paper_tables.py` refuses to be trusted unless the
unfiltered CBR matches the independent `export_cf_cbr.py` output, which in turn
reproduces the published relation CBR 55.1 / 37.9 bit-for-bit.
Not claimed: GPU collection, training, and paper compilation.
