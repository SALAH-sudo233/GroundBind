# Right Region, Wrong Reference

Diagnosing and Mitigating Grounding Hallucinations — v0.51 paper method

This checkout retains **Relation-Contrastive Verification**, the method selected for the **v0.51 paper**: JEV-2B and
OmniVerifier-7B score the same upstream candidate, compare alternative relations
within each verifier, and combine their readings into a keep/reject decision.
The candidate coordinates remain fixed. V-SIGHT-Bench, FGR, conditional box reuse
(CBR), and positive localization retention remain part of the evaluation.

This is a curated research snapshot, not a newly validated release. Original
source files and reported results are retained byte-for-byte. See
[AUDIT.md](AUDIT.md) before rerunning or interpreting the results.

## Start here

- [Method and recorded results](evidence/MITIGATION_FRAMEWORK_REPORT.md)
- [Combined-arm results](evidence/combo.json): B0, B1 and A2, including single-arm controls
- `method/eval_combo.py`: four-feature fusion and cross-fitted evaluation
- `method/collect_jevhead.py`: JEV original/rival readings
- `method/deploy_upstream.py`, `method/s5_omni_filter.py`: Omni rival collection and scorer
- `method/simple_relations.py`: relation alternatives
- `training/jev_verifier.py`, `training/run_jev.sh`: recorded JEV training source
- `benchmark/refcocog_500_dev.semantic_strict.json`: preserved benchmark specification
- [File provenance](provenance.json) and [historical recovery](HISTORY.md)

## Method boundary

Both verifiers contribute an original-expression score and an internal relation
gap. Differences are **not** taken between different verifiers. The full arm
uses four features; single-arm controls are necessary ablations, not alternative
current methods. No relocalization, proposal switching, human-review loop, TRACE,
CCV/CABLE or agentic flywheel is presented as the current method.

The recorded combined arm reports ALL FGR 59.20% → 22.15%, relation CBR
40.1% → 18.7%, and positive mIoU 0.3738 → 0.3549 (~94.9% retained).
These are inherited results, **not experiments rerun during cleanup**. They do
not establish lossless filtering or superiority on every type; attribute CBR is
worse than the JEV-only control. See the report and audit for remaining issues.

## Verification and execution

CPU-only relation-template tests:

```sh
python3 -m unittest discover -s method -p 'test_*.py' -v
```

The preserved collection and evaluation scripts describe the original Linux
workspace (`~/SVD/agentic_probe`, `~/SVD/grpo_verifier`) and canonical run map.
They are **not portable standalone launchers**. Full reproduction additionally
requires the original per-model `records.jsonl`, verifier caches, model weights,
image files, the JEV adapter/head/temperature, and the matching GPU environment.
The Mac cleanup does not supply those weights or run training. Do not execute
`training/run_jev.sh` merely to browse this repository.

The v0.51 LaTeX/PDF artifacts are referenced in the chat “重构论文第一阶段” but
were not present in the local migration. This checkout does not pretend to
contain that paper source. The user selected v0.51 as the current authority. V-SIGHT-Bench remains the
benchmark name; numerical results are unchanged from the source report.
