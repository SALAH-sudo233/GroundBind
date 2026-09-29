# v0.51 snapshot audit

The selection is based on the v0.51 delivery in “重构论文第一阶段” and
`MITIGATION_FRAMEWORK_REPORT.md` (2026-09-27), not on a numerical metric of 0.50.

## Unresolved implementation/report differences

1. **Additional rejection versus replacement decision.** The report says the
   relation stage can only reconsider rows retained by the original-support gate.
   In `eval_combo.py`, `fit()` → `keep_a2()` directly returns the probe-head
   decision for covered, usable rows without requiring `keep_base(r)` first.
   Thus the implementation can also rescue a rejected row. Do not describe it as
   a proven monotone rejection stage. Changing this now would change the method
   and invalidate the inherited numerical table; the cleanup does neither.
2. **Routing provenance.** `deploy_upstream.py` uses evaluation record types to
   choose relation rows. Label-free query-only deployment routing has not been
   established by this cleanup.
3. **Training/scoring prompt provenance.** The retrieved `jev_verifier.py`
   declares a shorter prompt than `collect_jevhead.py`'s required JSON-response
   template. Check the actual saved training run and checkpoint provenance before
   claiming byte-identical training and scoring prompts. Do not silently rewrite
   either template.
4. **Reproduction dependencies.** Local migration lacks the complete canonical
   upstream records and JEV score caches. Source availability and imported
   result JSON do not constitute a fresh reproduction.
5. **Evaluation limits.** Keep the fixed positive eligibility set for CBR,
   full denominators for FGR/mIoU, image-level split discipline and relevant
   single-arm/baseline controls. Text-prior, matched retention/cost and routing
   audits remain necessary. These controls are not obsolete material.

## Cleanup validation scope

Preserved-source SHA-256 checks, Python syntax, local helper imports and the
existing relation-template tests can be validated on the Mac. GPU collection,
training, numerical regeneration and paper compilation are not claimed.
