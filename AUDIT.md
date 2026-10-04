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
   (text prior, equal-budget paraphrase, wrong-image), 
   `matched_retention_current.json` (当前策略 `A_reject_opp` 同保留对照，13/13 模型，
   relation FGR −7.88pp CI[−9.43,−6.30] 零回退、ALL FGR +0.16pp CI 跨零不显著；
   旧产物 `matched_retention_WITHDRAWN_A_all.json` 已被 CORRECTION.md 撤回),
   `audit_routing_fixed.json` (routing). The fixed
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

8. **The coordinate correction never reached t4 — RESOLVED 2026-10-01.**
   `write_rescaled_run.py` filters on `T2 = ('t2', 't2_vqa_grounding')`, so every other
   task was copied through verbatim. Verified, not assumed: t4 boxes in
   `~/benchmark/coordfix_500` are byte-identical to the uncorrected run for both
   affected models (2500/2500), while t2 boxes differ on 2496/2500 and 974/2500.
   The published joint-grounding figures for UniVG-R1 and Visual-RFT therefore carried
   the same 0-1000-read-as-pixels defect the reviewer raised for t2, and the repo's t4
   script inherited it by reading the stored `iou`. Example: raw `[54,700,236,999]`
   stored as `[54,428,236,428]`, a zero-height box after clipping.
   `reviewer_v060/export_cf_joint.py` reuses the t2 correction unchanged and is gated
   on reproducing the published t2 CBR for both models (PASS). Receipt:
   `cf_joint_grounding.json`, writeup `JOINT_GROUNDING_CF.md`.
   **Now folded into the t4 panel.** `eval_tasks_t1_t3_t4.py` re-derives t4 boxes for
   these two models and emits both versions; a second gate requires bit-for-bit
   agreement with `export_cf_joint.py` (247/247 and 207/207, PASS). The correction
   moves the positive side only: 13-model positive success 29.49% -> 35.60%, mIoU
   0.3118 -> 0.3639, while FGR barely moves (48.26% -> 48.25%). Uncorrected, UniVG-R1's
   t4 reads 5.60% positive success at mIoU 0.1127 -- the same broken pair the reviewer
   objected to for t2.

7. **Task coverage — RESOLVED 2026-10-01, all four tasks now have receipts.**
   The repo previously backed only direct grounding (t2). Table 1 defines four tasks,
   and section 4.1 quotes a false-accept contrast that no repo file could produce.
   The records were on vlm1 all along (t1 and t4: 13 models x 2,500 rows; t3: 500 rows
   each) and no script had ever read them. `reviewer_v060/eval_tasks_t1_t3_t4.py`
   now evaluates expression verification, pure captioning and joint grounding, gated
   on reproducing the published 29.44% / 33.03% family false-accept averages
   (measured 29.4375% / 33.0278%, both PASS). Receipt:
   `reviewer_v060/tasks_t1_t3_t4.json`, including `run_provenance` per model per task.
   **Caveat worth carrying into the paper:** `canon_roots_paper.json` guarantees only
   the t2 canonical boxes. Qwen3-VL-8B's t1 lives in a different run, and t3 exists
   only in the 4tasks run. Resolving tasks naively against the canonical root drops
   Qwen3-VL-8B from t1 and yields 31.35% instead of the published 29.44%.

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

## 9. Mitigation was only ever measured on direct grounding — RESOLVED

Every mitigation number in the paper came from t2. The joint-grounding task (t4) had
benchmark numbers but no mitigation, so "the framework mitigates hallucination" rested
on one task out of four.

Closed by collecting verifier scores on t4 boxes and re-running the same policy:
`collect_t4_mitigation.py` + `eval_t4_mitigation.py` → `t4_mitigation.json`,
documented in `reviewer_v060/T4_MITIGATION.md`.

**t2 scores could not be reused.** `collect_jevhead.py` and
`collect_probe_textroute.py` both hard-filter `task == 't2_vqa_grounding'`. The
verifier scores one specific box, and the t4 box differs from the t2 box for the same
query because emitting a caption moves the box. Reusing t2 scores would be the same
error class as subtracting across two scorers. 59,025 fresh forward passes
(z0 18,553 / jev 18,553 / rival 21,919), 8 shards, 25.4 min, zero errors, counts
matching the plan item for item.

Result (13 models, target 0.95): FGR 48.24% → 19.70% → 18.45%,
correct retention 100% → 93.95% → 92.99%.

**Four types, zero regressions, 13/13 models improved** — object −0.231pp
CI[−0.415,−0.092], co_occurrence −0.723pp CI[−1.108,−0.385], attribute −1.046pp
CI[−1.554,−0.585], relation −2.985pp CI[−3.723,−2.246]; worse-model count 0/13 on
every type. This is the strict-additivity requirement holding on a second task: the
first three types are mitigated by a single Omni posterior and never enter predicate
competitive scoring, so any regression there would be an implementation bug. The
check is built into the script and prints REGRESSION if it ever fails.

The full arm costs *less* on t4 than on t2 (FGR 18.45% vs 19.16%, retention 92.99% vs
91.71%), so the framework is not specific to direct grounding.

Not matched-retention: this is a single operating point. On t2 the gain roughly halved
under matched retention; the t4 matched comparison has not been run.

## Validation scope

Validated on the Mac: preserved-source SHA-256 checks, Python syntax, local helper
imports, the relation-template tests. Validated on vlm1 (CPU only for items 1–8, no
model re-inference): the recomputations listed above, each gated on reproducing a
published anchor — `eval_paper_tables.py` refuses to be trusted unless the
unfiltered CBR matches the independent `export_cf_cbr.py` output, which in turn
reproduces the published relation CBR 55.1 / 37.9 bit-for-bit.

Item 9 **did** use the GPUs: 8×RTX4090 for 25.4 min of verifier scoring. It wrote only
to the new `t4mitig/` directory and left every L2 input untouched.
Not claimed: training, upstream candidate re-generation, and paper compilation.
