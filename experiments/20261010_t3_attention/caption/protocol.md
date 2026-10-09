# T3 query-free caption false assertion rate: 2026-10-10 supplement

This supplement scores the unchanged, real query-free captions of all 13 evaluated models against all four types of existing benchmark negative semantic units. No negative query is supplied to the caption generation model. The old object-only caption scores remain preserved in their original files.

## Actual cohort and denominators

The source cohort has 500 distinct images, 500 aligned base groups and 2,000 annotated positive/negative pairs. Each model contributes exactly 500 cached captions, one per base group, for 6,500 real captions total. Each caption is assessed against its four aligned annotated negative units: object, association (source key `co_occurrence`), attribute and relation. This makes 26,000 new semantic judgments. Each model has 500 judgments per edit type and 1,000 negative units per EOH/ROH group. Reusing one caption for four assessment units does not create four new captions or expand the image cohort to 1,000.

## Semantic event

A false assertion occurs when the generated caption definitely asserts the changed, annotated unsupported semantic unit. The event does not require verbatim reproduction of the entire referring expression or its unchanged positive modifiers. A missing object explicitly asserted as present counts for EOH; the property for an attribute edit must bind to its target entity; the relation must bind to the intended subject and reference roles. Synonyms and faithful paraphrases count. Keyword co-occurrence, role reversal, properties of other entities, negation, uncertainty, quotation, hypotheticals and unspecified relations do not count.

When an existing annotation scopes object falsity to a particular target/location and the same object appears elsewhere in the scene, a mention of that other object does not assert the scoped false unit. Scope is resolved from the original annotated object inventory: if the introduced object is listed elsewhere, the scoped negative expression supplies its target/location; otherwise the edited presence unit is the hypothesis. This resolves proposition scope, never the judgment label. ROH receives the positive target reference for entity identity when multiple candidate entities are mentioned; unchanged contextual modifiers are not required as part of the edited proposition. Existing annotation metadata are preserved without fabricating new human-review flags; the final reviewed status is the status supplied by the user and manuscript.

## Evaluator

Independent local text semantic judge: Qwen/Qwen3.5-9B, snapshot `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, bfloat16, deterministic inference, thinking disabled. The judge receives the cached caption, positive target reference, edited negative expression and existing negative semantic units. It receives no image and does not guess whether the expression is visually true. The benchmark supplies the negative-unit truth; the judge assesses only what the caption asserts.

The exact system prompt is in `t3_semantic_eval.py` and is saved with its SHA in every judgment. The first generated token is selected by unconstrained vocabulary argmax. Output must be YES or NO; a nonbinary first token triggers an unconstrained deterministic retry up to 16 tokens. Any unresolved parse is retained as invalid and cannot be silently counted as a clean caption. Conditional binary probabilities are logged as diagnostics, without imposing a chosen confidence threshold on the reported event.

## Control validation

The predeclared 64 authored diagnostic controls cover synonyms, faithful inverse relation paraphrases, entity binding, reversed roles, negation, uncertainty, quotes, hypothetical statements, questions, EOH object mentions without full unchanged context, and a scoped object that also appears elsewhere. Expected labels follow the explicit logical content of these controls. They are diagnostic controls authored for this evaluation, not claims of newly human-labeled benchmark data. The complete control outputs and accuracy are saved. Development round 1 had 36/64 (56.25%) accuracy and was rejected; explicit premise/hypothesis projection yielded 64/64 in round 2; final entity binding context yielded 63/64 (98.4375%). A separate 64-control set independently authored by the parent agent was frozen before any of its judgments, with SHA-256 `1fc655415579282f2b64adfa78b9c4e3bc4f598b37646460e86867b01e4abd80`. Its gold, category and provenance never enter the judge prompt. Its accuracy is reported separately, and it is not used to tune the protocol. The full run is started only after evaluating the diagnostic results and resolving any material protocol failures.

## Aggregation and audit

For each type, FAR is the integer number of asserted edited negative units divided by the 500 real evaluated units. EOH averages object and association; ROH averages attribute and relation. Because each type has 500 units, each grouped FAR equals the integer total divided by 1,000. Family rows average model rates with equal model weight (general: 4; RL: 9). T3 FAR expands to false ASSERTION rate, while verification FAR expands to false ACCEPTANCE rate.

All source annotation and caption-record SHA-256 hashes are taken before preparation and checked after scoring. Outputs include all original cached caption strings, the original annotation units, all 26,000 per-unit judgment inputs/outputs, evaluator metadata, integer count summaries, and control diagnostics. Model weight blob references are recorded from the immutable local snapshot; no original model weights or evaluation records are edited.
