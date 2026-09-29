#!/usr/bin/env python3
"""Tests for the probe builder, mapped to plan v0.2 section 14 checklist."""
import unittest
import simple_relations as sr


class TestParse(unittest.TestCase):
    def test_longest_phrase_first_on_top_of(self):
        """sec 14: 'on top of' must win over bare 'on'."""
        surf, a, b = sr.find_predicate('the book on top of the box')
        self.assertEqual(surf, 'on top of')

    def test_longest_phrase_first_in_front_of(self):
        surf, _, _ = sr.find_predicate('the man in front of the bicycle')
        self.assertEqual(surf, 'in front of')

    def test_left_of_not_support_family(self):
        """sec 14: 'on the left of' must not fall into the support family."""
        surf, _, _ = sr.find_predicate('the cup to the left of the plate')
        self.assertEqual(sr.FAMILY[surf], 'simple_image_left_right')
        self.assertNotEqual(sr.FAMILY[surf], 'vertical_or_support')

    def test_no_predicate(self):
        probes, meta = sr.build('the red teddy bear')
        self.assertEqual(probes, [])
        self.assertEqual(meta['skip_reason'], 'no_predicate_found')
        self.assertEqual(meta['semantic_status'], 'unsupported')


class TestUnsupported(unittest.TestCase):
    def test_ambiguous_on_is_skipped(self):
        """Bare 'on' is polysemous -> unsupported, fall back to Omni."""
        probes, meta = sr.build('the cup on the table')
        self.assertEqual(probes, [])
        self.assertTrue(meta['skip_reason'].startswith('ambiguous_surface'))

    def test_multi_relation_scope_conflict(self):
        probes, meta = sr.build('the dog next to the man behind the car')
        self.assertEqual(probes, [])
        self.assertEqual(meta['skip_reason'], 'multi_relation_scope')

    def test_unsupported_never_emits_probe(self):
        for q in ['the cup on the table', 'the plain object',
                  'the dog next to the man behind the car']:
            probes, meta = sr.build(q)
            self.assertEqual(len(probes), 0)
            self.assertEqual(meta['n_legal_candidates'], 0)


class TestTargetRole(unittest.TestCase):
    def test_holding_keeps_target_as_holder(self):
        """sec 14/4.5: must NOT turn 'man holding a bag' into 'bag held by man'."""
        probes, meta = sr.build('the man holding a bag')
        for p in probes:
            self.assertNotIn('held by', p['query'])
            # target head 'the man' must survive at the front
            self.assertTrue(p['query'].startswith('the man'))

    def test_weakened_keeps_target_head(self):
        # weakened is off by default now; enable it to test the template itself
        probes, _ = sr.build('the man holding a bag', enable_weakened=True)
        weak = [p for p in probes if p['probe_type'] == 'weakened']
        self.assertTrue(weak)
        self.assertEqual(weak[0]['query'].strip(), 'the man')

    def test_competitive_keeps_both_noun_phrases(self):
        probes, _ = sr.build('the man in front of the bicycle')
        comp = [p for p in probes if p['probe_type'] == 'competitive'][0]
        self.assertEqual(comp['query'], 'the man behind the bicycle')
        self.assertTrue(comp['query'].startswith('the man'))
        self.assertIn('the bicycle', comp['query'])


class TestProbeTypes(unittest.TestCase):
    def test_competitive_is_bidirectional(self):
        """behind -> in front of must work too (no 'behind is always false' bias)."""
        probes, _ = sr.build('the man behind the bicycle')
        comp = [p for p in probes if p['probe_type'] == 'competitive'][0]
        self.assertEqual(comp['query'], 'the man in front of the bicycle')

    def test_weakened_not_confused_with_equivalent(self):
        """sec 14: weakened and equivalent_candidate must stay distinct."""
        probes, _ = sr.build('the man in front of the bicycle')
        types = {p['probe_type'] for p in probes}
        self.assertNotIn('weakened', types & {'equivalent_candidate'})
        for p in probes:
            if p['probe_type'] == 'equivalent_candidate':
                self.assertIn('the bicycle', p['query'])   # reference kept
            if p['probe_type'] == 'weakened':
                self.assertNotIn('the bicycle', p['query'])  # clause dropped

    def test_max_candidates_respected(self):
        for q in ['the man in front of the bicycle', 'the dog next to the man']:
            probes, meta = sr.build(q, max_candidates=2)
            self.assertLessEqual(len(probes), 2)
            self.assertEqual(meta['n_legal_candidates'], len(probes))

    def test_single_candidate_is_recorded(self):
        """sec 5.2: with one candidate there is no real selection."""
        probes, meta = sr.build('the cat under the table', max_candidates=1)
        self.assertEqual(len(probes), 1)
        self.assertEqual(meta['n_legal_candidates'], 1)

    def test_no_forbidden_shortcut_negation(self):
        """sec 4.6: never build probes by negation."""
        for q in ['the dog next to the man', 'the man holding a bag',
                  'the man in front of the bicycle']:
            probes, _ = sr.build(q)
            for p in probes:
                self.assertNotIn(' not ', ' ' + p['query'] + ' ')
                self.assertNotIn('far from', p['query'])

    def test_no_on_under_auto_swap(self):
        """sec 4.6: 'on' <-> 'under' must not be an automatic rule."""
        probes, meta = sr.build('the cup on the table')
        self.assertEqual(probes, [])   # bare 'on' is unsupported entirely
        # under's rivals are 'on top of' (antonym, first) then the beside family;
        # bare 'on' must never appear as a rival surface.
        rivals = [p['edited_surface'] for p in sr.build('the cat under the table')[0]
                  if p['probe_type'] == 'competitive']
        self.assertEqual(rivals[0], 'on top of')
        self.assertNotIn('on', rivals)


class TestBesideRivals(unittest.TestCase):
    """2026-09-27: 'next to' as a configuration predicate, not a distance one."""

    def test_next_to_has_configuration_rivals(self):
        probes, meta = sr.build('the chair next to the person', max_candidates=2)
        comp = [p for p in probes if p['probe_type'] == 'competitive']
        self.assertGreaterEqual(len(comp), 1)
        rivals = {p['edited_surface'] for p in comp}
        self.assertTrue(rivals <= set(sr.VERTICAL_RIVALS),
                        f"unexpected rivals for 'next to': {rivals}")

    def test_next_to_never_gets_distance_antonym(self):
        """far from / away from measured as ANTI-signals (0.4369 / 0.4315)."""
        probes, _ = sr.build('the chair next to the person', max_candidates=2)
        for p in probes:
            self.assertNotIn('far from', p['query'])
            self.assertNotIn('away from', p['query'])
        self.assertNotIn('far from', sr.COMPETITORS['next to'])
        self.assertNotIn('away from', sr.COMPETITORS['next to'])

    def test_vertical_keeps_antonym_first(self):
        """Antonym measured 0.7706 > beside-family 0.7600 -> must rank first."""
        for surf, antonym in (('behind', 'in front of'),
                              ('in front of', 'behind'),
                              ('under', 'on top of'),
                              ('on top of', 'under'),
                              ('above', 'below')):
            self.assertEqual(sr.COMPETITORS[surf][0], antonym,
                             f"{surf} must probe its antonym first")

    def test_vertical_has_beside_fallback(self):
        probes, _ = sr.build('the car behind the bird', max_candidates=2)
        rivals = [p['edited_surface'] for p in probes
                  if p['probe_type'] == 'competitive']
        self.assertEqual(rivals[0], 'in front of')
        self.assertIn(rivals[1], sr.BESIDE_SURFACES)

    def test_multiple_competitive_now_emitted(self):
        """Previously the builder broke after one rival; max is the aggregator."""
        probes, meta = sr.build('the chair next to the person', max_candidates=2)
        self.assertEqual(len([p for p in probes
                             if p['probe_type'] == 'competitive']), 2)
        self.assertEqual(meta['n_legal_candidates'], 2)

    def test_target_role_preserved_for_beside_rivals(self):
        probes, _ = sr.build('the chair next to the person', max_candidates=2)
        for p in probes:
            self.assertTrue(p['query'].startswith('the chair'))
            self.assertIn('the person', p['query'])


class TestDisabledByDefault(unittest.TestCase):
    """equivalent (0.4398 anti-signal) and weakened (uninformative) are off."""

    def test_equivalent_off_by_default(self):
        probes, _ = sr.build('the man in front of the bicycle', max_candidates=3)
        self.assertNotIn('equivalent_candidate',
                         {p['probe_type'] for p in probes})

    def test_weakened_off_by_default(self):
        probes, _ = sr.build('the man holding a bag', max_candidates=3)
        self.assertNotIn('weakened', {p['probe_type'] for p in probes})

    def test_flags_re_enable_for_ablation(self):
        # 'in front of' now has 3 competitive rivals (behind + beside family),
        # so the budget must exceed them for the ablation probes to appear.
        probes, _ = sr.build('the man in front of the bicycle',
                             max_candidates=6, enable_equivalent=True,
                             enable_weakened=True)
        types = {p['probe_type'] for p in probes}
        self.assertIn('equivalent_candidate', types)
        self.assertIn('weakened', types)
        self.assertIn('competitive', types)

    def test_holding_still_unsupported_for_competitive(self):
        """'holding' has no audited rival -> only ablation probes exist."""
        probes, meta = sr.build('the man holding a bag')
        self.assertEqual([p for p in probes
                          if p['probe_type'] == 'competitive'], [])
        self.assertEqual(meta['semantic_status'], 'unsupported')


class TestDeterminism(unittest.TestCase):
    def test_stable_across_calls(self):
        a = sr.build('the man in front of the bicycle')
        b = sr.build('the man in front of the bicycle')
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])

    def test_probe_differs_from_original(self):
        for q in ['the man in front of the bicycle', 'the dog next to the man',
                  'the cat under the table']:
            probes, _ = sr.build(q)
            for p in probes:
                self.assertNotEqual(p['query'].strip().lower(), q.strip().lower())


if __name__ == '__main__':
    unittest.main(verbosity=2)
