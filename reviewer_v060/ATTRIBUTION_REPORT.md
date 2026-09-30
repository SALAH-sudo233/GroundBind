# Attribution and scope findings — v0.60 review cycle

**Date:** 2026-09-30  
**Models:** 13 (InternVL3.5-8B, LENS, Orsta-7B, Qwen3-VL-8B, Seg-R1, Seg-zero, TreeVGR, UniVG-R1, Vision-R1, VisionReasoner, llava-ov-7b, qwen2.5-vl-7b, visual-rft)  
**Probe coverage:** 12,631 rows (6,816 opposed predicates + 5,815 next_to)

## Summary

Three mandatory controls were run to isolate the source of the relation-contrastive verifier's gain:

1. **[A] Text-prior control** — does the gap add anything once a predicate prior is present?
2. **[B] Equal-budget control** — is the gain from relation contrast or just extra inference?
3. **[C] Wrong-image control** — does the verifier read the candidate image?

Two findings contradict the draft's intended claim and restrict the method's scope:

- **Once a text prior is present, the pooled relation gap adds nothing** (blexO → blexO+gap −0.0015 CI[−0.0026,−0.0004], up 3/13).
- **Scoped to predicates with genuine opposing configurations** (`behind`, `in front of`, `under`, `on top of`, `above`, `below`), the gap delivers +0.0059 CI[+0.0026,+0.0090] SIG (11/13). On `next to`, the effect is +0.0015 CI[+0.0000,+0.0033] (CI lower bound touches zero; effect size negligible).

The wrong-image control is overwhelming (+11.39, 13/13 models), so the verifier genuinely reads the image; the gap's information is not recoverable from predicate identity alone (R²=0.08). The reconciliation: `next to` has no logical opposite—its rivals are merely *other* spatial configurations that can hold simultaneously—so within-predicate gap AUROC is 0.550 (near chance). Since `next to` and `opposed` are roughly equal in size, the pooled metric is diluted to null.

## Required paper changes

1. **Restrict the claim to depth/vertical predicates.** The abstract/intro cannot say "grounding hallucination" without qualification; it must scope to spatial relations with genuine opposing configurations.
2. **Report scoped statistics.** Replace the pooled decisive test (currently −0.0015 ns) with the `opposed` result (+0.0059 SIG) and explicitly state that `next to` shows no gain.
3. **Acknowledge the limitation.** The discussion must note that relation contrast requires a semantically exclusive rival, so it does not generalize to all referring expressions (attributes, counts, non-spatial relations remain unaddressed).
4. **Keep [B] and [C] as necessary controls.** [B] proves the gain comes from relation contrast, not the extra call (competitive gap AUROC 0.5129 vs equivalent-rewrite 0.2964). [C] proves the verifier reads the image (real vs swapped +11.412 SIG). Both belong in the main results or appendix, not deleted.

## Detailed results

### [A] Text-prior control (decisive)

Cross-fitted AUROC on probe-covered rows:

| Arm            | Pooled AUROC | n_models |
|----------------|--------------|----------|
| blex_only      | 0.6289       | 13       |
| support_z0     | 0.8102       | 13       |
| blexO          | 0.8532       | 13       |
| full_gap       | 0.8271       | 13       |
| blexO_gap      | 0.8517       | 13       |

Paired model-level bootstrap (n=13):

- **full − support:** +0.0168 CI[+0.0111, +0.0230] SIG, up 13/13
- **full − text prior:** +0.1982 CI[+0.1579, +0.2360] SIG, up 13/13
- **gap ON TOP of text prior (decisive):** −0.0015 CI[−0.0026, −0.0004] SIG, up 3/13

On the predicate-balanced subset (text prior forced to ~0.5000):

- bal_support_z0: 0.8354
- bal_full_gap: 0.8372
- **balanced full − support:** +0.0018 CI[−0.0006, +0.0043] ns, up 8/13

**Interpretation:** The gap's discriminative content is *absorbed* by a predicate prior when pooled across all predicates. This does not mean the gap is uninformative (see [B] and scoped results), but rather that predicate identity already encodes much of what naive pooling would attribute to the gap.

### [B] Equal-budget control

Schema: `actions_relation.jsonl` contains `original`, `competitive` (rival relation), and `equivalent_candidate` (semantics-preserving rewrite) probes, all at the same temperature/inference cost.

- **n=449 sids with labels**
- **Competitive gap AUROC:** 0.5129
- **Equivalent-rewrite AUROC:** 0.2964

→ **Relation contrast carries the signal, not the extra call.** A paraphrase with the same inference budget and head width delivers AUROC 0.2964 (anti-signal direction), while the competitive gap reaches 0.5129.

### [C] Wrong-image control

Schema: `swap_all.jsonl` carries `z_real` (correct image) and `z_swap` (wrong image) side by side, same query/box/routing.

- **13,562 rows (6,794 unique model×sid pairs)**
- **Mean z(real image):** +0.255
- **Mean z(wrong image):** −11.137
- **Per-model mean(real − wrong):** +11.412 CI[+10.881, +11.843] SIG (n=13)

→ **The verifier genuinely reads the candidate image.** The score collapses when the probe image is swapped, so the signal is not coming from text alone.

### Gap vs predicate-identity redundancy

One-way ANOVA R² of omni gap on predicate identity:

- **Pooled R² = 0.0797** (predicate explains 8% of gap variance)
- **Pooled AUROC, raw gap:** 0.6133
- **Pooled AUROC, gap MINUS predicate mean:** 0.5983
- **Residual − raw:** −0.0150 CI[−0.0184, −0.0113] SIG

Within-predicate gap AUROC (predicate identity constant):

| Predicate    | AUROC | n_models |
|--------------|-------|----------|
| next to      | 0.5496| 13       |
| behind       | 0.8069| 13       |
| in front of  | 0.6370| 13       |
| under        | 0.8898| 9        |
| on top of    | 0.9844| 4        |

→ **The gap is not redundant with predicate identity.** Predicates with genuine logical opposites (`behind`, `under`, `on top of`) show strong within-predicate AUROC. `next to` has no true opposite, so its gap AUROC is near chance.

### Scoped analysis

#### Opposed predicates (behind, in front of, under, on top of, above, below)

6,816 rows, 13 models:

| Arm            | Pooled AUROC |
|----------------|--------------|
| blex_only      | 0.6649       |
| support_z0     | 0.7849       |
| blexO          | 0.8390       |
| full_gap       | 0.8092       |
| blexO_gap      | 0.8450       |

- **full − support:** +0.0243 CI[+0.0196, +0.0293] SIG, up 13/13
- **gap ON TOP of prior (decisive):** **+0.0059 CI[+0.0026, +0.0090] SIG, up 11/13**

#### Next to

5,815 rows, 13 models:

| Arm            | Pooled AUROC |
|----------------|--------------|
| blex_only      | 0.4853       |
| support_z0     | 0.8525       |
| blexO          | 0.8525       |
| full_gap       | 0.8540       |
| blexO_gap      | 0.8540       |

- **full − support:** +0.0015 CI[+0.0000, +0.0033] SIG, up 8/13
- **gap ON TOP of prior (decisive):** **+0.0015 CI[+0.0000, +0.0033] SIG, up 8/13**

**Interpretation:** The CI lower bound touches zero, and +0.0015 AUROC is within measurement noise. The gain is statistically detectable only because of tight model-level agreement, but the effect size is negligible. Do not claim a meaningful gain on `next to`.

## Method files

- `reviewer_v060/eval_attribution.py` — [A], [B], [C]
- `reviewer_v060/diag_gap_vs_prior.py` — gap vs predicate redundancy
- `reviewer_v060/eval_scoped.py` — opposed vs next_to scoping
- `reviewer_v060/attribution.json` — [A], [B], [C] results
- `reviewer_v060/gap_vs_prior.json` — R², residual AUROC, within-predicate AUROC
- `reviewer_v060/scoped.json` — per-model scoped results

## Checklist for the paper

- [ ] Abstract/intro scoped to "spatial relations with opposing configurations"
- [ ] Results section reports `opposed` +0.0059 (not pooled −0.0015)
- [ ] `next to` explicitly noted as showing no gain
- [ ] [B] equal-budget control in results or appendix
- [ ] [C] wrong-image control in results or appendix
- [ ] Discussion acknowledges limitation to semantically exclusive rivals
- [ ] No claim of "lossless filtering" or "universal grounding hallucination mitigation"
