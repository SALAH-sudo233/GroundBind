#!/usr/bin/env python3
"""Deterministic relation-word substitution probe builder.

Plan v0.2 section 4: probes are produced by deterministic templates ONLY.
The 2B verifier scores them; it does NOT generate them.

Hard constraints enforced here (v0.2 sec 4.5 / 4.6):
  - target role must be preserved (never swap target <-> reference)
  - longest phrase first ('on top of' before 'on', 'in front of' before 'in')
  - no automatic antonym dictionary; only audited pairs in COMPETITORS
  - unparseable / ambiguous -> unsupported (fall back to Omni baseline)
  - probes are built from the CURRENT query only: never from the paired
    positive expression, htype, GT box, or group_id

Probe types (v0.2 sec 4.3):
  competitive          - swap the relation word for an audited rival
  equivalent_candidate - conservative paraphrase, target word untouched
  weakened             - drop the relational clause, keep the target head
"""
import re

# Predicates observed in refcocog 500-dev relation queries, longest first.
# Counts (pos/neg over 500 relation groups) from the distribution audit:
#   next to 177/21 | behind 23/168 | in front of 96/40 | holding 66/28
#   on 50/23 | under 3/44 | with 28/18 | in 23/20 | on top of 1/30
#   standing next to 5/22 | sitting on 13/12 | above 2/18 | wearing 2/11
PREDICATES = [
    'to the left of', 'to the right of', 'on the left side of',
    'on the right side of', 'in front of', 'on top of',
    'standing next to', 'sitting next to', 'standing on', 'sitting on',
    'lying on', 'leaning on', 'attached to', 'held by', 'worn by',
    'covered by', 'surrounded by', 'pointing at', 'looking at',
    'next to', 'close to', 'far from', 'left of', 'right of',
    'underneath', 'beneath', 'inside', 'beside', 'between', 'among',
    'around', 'behind', 'above', 'below', 'under', 'over', 'near',
    'onto', 'into', 'holding', 'carrying', 'wearing', 'riding',
    'eating', 'drinking', 'touching',
    'on', 'in', 'at', 'with', 'by',
]
PREDICATES = sorted(set(PREDICATES), key=len, reverse=True)

_RX = re.compile(r'\b(' + '|'.join(re.escape(p) for p in PREDICATES) + r')\b',
                 re.IGNORECASE)

# Semantic families (v0.2 sec 4.2). Used for template selection and logging,
# NOT fed into the final decision head.
FAMILY = {
    'next to': 'proximity', 'beside': 'proximity', 'near': 'proximity',
    'close to': 'proximity', 'standing next to': 'proximity',
    'sitting next to': 'proximity',
    'in front of': 'depth_order', 'behind': 'depth_order',
    'holding': 'holding', 'carrying': 'holding', 'held by': 'holding',
    'on': 'vertical_or_support', 'on top of': 'vertical_or_support',
    'under': 'vertical_or_support', 'underneath': 'vertical_or_support',
    'beneath': 'vertical_or_support', 'above': 'vertical_or_support',
    'below': 'vertical_or_support', 'sitting on': 'vertical_or_support',
    'standing on': 'vertical_or_support', 'lying on': 'vertical_or_support',
    'to the left of': 'simple_image_left_right',
    'to the right of': 'simple_image_left_right',
    'left of': 'simple_image_left_right',
    'right of': 'simple_image_left_right',
}

# Audited competitive rivals. Derived from the OFFLINE paired-transition audit
# (v0.2 sec 18.3) but used here only as a symmetric rival table -- the paired
# positive expression is never read at probe time.
# Only pairs where the two conditions are genuinely rival on the SAME reference.
#
# MEASURED (2026-09-27, 500-dev relation, GT box, fold B):
#   antonym rivals below            AUROC 0.7706  <- keep FIRST, strongest
#   beside-family rivals (BESIDE_*) AUROC 0.7600  <- fallback, redundant with
#                                                    the antonym on the same rows
#                                                    (new-old -0.0108, n.s.)
COMPETITORS = {
    'in front of': ['behind'],
    'behind': ['in front of'],
    'above': ['below'],
    'below': ['above'],
    'on top of': ['under'],
    'under': ['on top of'],
    'to the left of': ['to the right of'],
    'to the right of': ['to the left of'],
    'left of': ['right of'],
    'right of': ['left of'],
}

# Lateral-adjacency ('beside') <-> configuration rivals.
#
# 'next to' is a CONFIGURATION statement (target alongside the reference, same
# level), NOT a distance statement. So its rivals are the vertical/depth
# configurations, never a distance antonym.
#
# MEASURED, 'next to' rows (n=191, 173 pos / 18 neg):
#   far from    AUROC 0.4369   <- ANTI-signal; reject-ed on both polarities
#   away from   AUROC 0.4315   <- same
#   behind only AUROC 0.5860
#   VERTICAL_RIVALS max        AUROC 0.7453   <- adopted
# Corpus check: 'far from'/'away from' appear 0x in positives, 4x in negatives.
# Never add a distance antonym here (plan v0.2 sec 4.6 forbids it, now measured).
VERTICAL_RIVALS = ['behind', 'in front of', 'under', 'on top of',
                   'above', 'below']
BESIDE_SURFACES = ['next to', 'beside']

# 'next to' had NO competitive rival before this: +191 relation rows of coverage
# that previously could only fall back to the Omni baseline.
COMPETITORS['next to'] = list(VERTICAL_RIVALS)
COMPETITORS['beside'] = list(VERTICAL_RIVALS)

# Vertical/depth surfaces get the beside-family as a SECOND rival, after their
# antonym. Fallback only -- measured slightly weaker than the antonym.
for _s in VERTICAL_RIVALS:
    COMPETITORS.setdefault(_s, [])
    for _b in BESIDE_SURFACES:
        if _b not in COMPETITORS[_s]:
            COMPETITORS[_s].append(_b)

# Conservative paraphrases that keep the target role and the relation meaning.
EQUIVALENTS = {
    'in front of': 'positioned in front of',
    'behind': 'positioned behind',
    'next to': 'right next to',
    'holding': 'who is holding',
    'under': 'positioned under',
    'above': 'positioned above',
    'on top of': 'resting on top of',
}

# Ambiguity blocklist: these surfaces are too polysemous to edit safely.
# 'on' can be support ('cup on the table'), wearing ('logo on a shirt'),
# depiction ('picture on the wall') or idiom -> require audit, skip for now.
AMBIGUOUS = {'on', 'in', 'at', 'with', 'by', 'over', 'into', 'onto'}


def find_predicate(query):
    """Longest-phrase-first predicate match on the CURRENT query.

    Returns (surface, start, end) or (None, -1, -1).
    """
    if not query:
        return None, -1, -1
    m = _RX.search(query)
    if not m:
        return None, -1, -1
    return m.group(0).lower(), m.start(), m.end()


def build(query, max_candidates=2, enable_equivalent=False,
          enable_weakened=False):
    """Return (probes, meta). probes = list of dicts, never more than
    max_candidates. meta carries the parse contract fields of v0.2 sec 4.7.

    Defaults reflect the 2026-09-27 measurement: only `competitive` probes
    carry relational information. `equivalent_candidate` is an anti-signal
    (0.4398) and `weakened` is uninformative (0.6881 vs 0.6861 baseline);
    both stay available behind flags for the Apara / Aweak ablations.
    """
    meta = {'predicate_surface': None, 'family': None,
            'semantic_status': 'unsupported', 'skip_reason': None,
            'n_legal_candidates': 0, 'rule_id': None}
    surf, a, b = find_predicate(query)
    if surf is None:
        meta['skip_reason'] = 'no_predicate_found'
        return [], meta

    meta['predicate_surface'] = surf
    meta['family'] = FAMILY.get(surf)

    # multiple distinct predicates -> scope conflict, do not edit
    hits = [m.group(0).lower() for m in _RX.finditer(query)]
    if len(hits) > 1:
        meta['skip_reason'] = 'multi_relation_scope'
        return [], meta

    if surf in AMBIGUOUS:
        meta['skip_reason'] = f'ambiguous_surface:{surf}'
        return [], meta

    if meta['family'] is None:
        meta['skip_reason'] = f'no_family_for:{surf}'
        return [], meta

    probes = []
    head = query[:a]
    tail = query[b:]

    # Competitive rivals, in table order (antonym first, beside-family after).
    # The true rival is unknown at probe time, so several may be emitted and the
    # policy aggregates them with max (strongest contradiction).
    for rival in COMPETITORS.get(surf, []):
        if len(probes) >= max_candidates:
            break
        probes.append({'query': head + rival + tail,
                       'probe_type': 'competitive',
                       'rule_id': f'comp:{surf}->{rival}',
                       'edited_surface': rival})

    # equivalent_candidate is DISABLED by default: measured AUROC 0.4398 on
    # fold B (n=363) i.e. an ANTI-signal -- paraphrase instability correlates
    # with positives, not with hallucinations. Kept behind a flag for ablation.
    if enable_equivalent:
        eq = EQUIVALENTS.get(surf)
        if eq and len(probes) < max_candidates:
            probes.append({'query': head + eq + tail,
                           'probe_type': 'equivalent_candidate',
                           'rule_id': f'equiv:{surf}->{eq}',
                           'edited_surface': eq})

    # weakened: drop the relational clause, keep target head.
    # DISABLED by default: measured AUROC 0.6881 vs 0.6861 for the original
    # expression on the same subset (n=194) -- it only confirms the object
    # exists, it carries no relational information. Flag kept for ablation.
    if enable_weakened and len(probes) < max_candidates:
        h = head.strip()
        if h and len(h.split()) >= 2:
            probes.append({'query': h,
                           'probe_type': 'weakened',
                           'rule_id': f'weak:drop_{surf}_clause',
                           'edited_surface': None})

    probes = probes[:max_candidates]
    meta['n_legal_candidates'] = len(probes)
    meta['semantic_status'] = 'supported' if probes else 'unsupported'
    if not probes:
        meta['skip_reason'] = 'no_legal_template'
    meta['rule_id'] = probes[0]['rule_id'] if probes else None
    return probes, meta
