#!/usr/bin/env python3
"""Evaluate expression verification (t1), pure captioning (t3) and joint grounding (t4).

WHY THIS EXISTS. The repo could back only direct grounding (t2). Table 1 of the paper
defines four tasks, and section 4.1 quotes a false-accept contrast (29.44% vs 33.03%)
that no file in the repo could produce -- it existed only in the PDF. The records
were on vlm1 the whole time (t1: 12 models x 2500 rows, t4: 13 x 2500) with the
needed fields; nothing had ever read them. This script does.

FIELD SEMANTICS, read off the records rather than assumed:
  t1  label_exists  ground truth support of the expression
      pred_exists   the model's own support judgement (parsed from free text)
      parse_valid   whether the parse was clean; a model may answer "No." with
                    parse_valid False, so parse failures are reported, never
                    silently folded into a rejection
  t4  caption_target_hallucination  defined ONLY on negatives (None on all 6500
                                    positives) -- so its denominator is negatives,
                                    not all rows
      target_coverage / amber_cosine  present on all 32500 rows
  t3  one query-free caption per image, 500 rows per model

COORDINATE CORRECTION ON t4. UniVG-R1 and visual-rft emit boxes in a 0-1000
normalised frame. write_rescaled_run.py repaired that for t2 only (its filter is
T2 = ('t2','t2_vqa_grounding')), so t4 kept the defect: raw [54,700,236,999] stored as
[54,428,236,428], a zero-height box. Left uncorrected, UniVG-R1's t4 positive success
reads 5.6% with mIoU 0.1127 -- the same broken pair the reviewer objected to for t2.
This script therefore re-derives t4 boxes for those two models from raw_output_text
via export_cf_joint.rescale(), the same correction t2 uses, and reports both versions
so the swap is visible rather than silent.

TASK RECORDS LIVE IN DIFFERENT RUNS. canon_roots_paper.json points each model at the
run holding its canonical t2 boxes, but that run does not always carry the other
tasks: Qwen3-VL-8B has no t1 there (it is in refcocog_eval_11models_500_repaired) and
t3 lives only in refcocog_eval_13models_4tasks_500. Resolving each task separately is
what makes the t1 panel 13 models instead of 12, and that difference matters: with
Qwen3-VL-8B present the general family false-accept average is 29.44%, which is the
number the paper prints. Taking canon roots at face value silently drops it and gives
31.35%.

METRICS (paper section 3.3 wording):
  false accept  share of NEGATIVE queries judged supported
  balanced acc  mean of per-class accuracy over positives and negatives, so the
                4:1 negative:positive imbalance cannot flatter a model that simply
                says "no" to everything
  t4 adds caption hallucination on negatives, target coverage, and the same
  FGR / mIoU / positive-success definitions used for t2 when boxes are present

Equal model weight, matching every other table in this repo.
"""
import argparse
import collections
import json
import os
import statistics as st

PROBE = os.path.expanduser('~/SVD/agentic_probe')
HERE = os.path.dirname(os.path.abspath(__file__))
HT4 = ['object', 'co_occurrence', 'attribute', 'relation']
GENERAL = {'InternVL3.5-8B', 'Qwen3-VL-8B', 'llava-ov-7b', 'qwen2.5-vl-7b'}
# models whose raw boxes are 0-1000 normalised; t4 was never repaired for them
COORDFIX_T4 = {'UniVG-R1', 'visual-rft'}


# fallback runs searched, in order, when a task is absent from the canonical root
EXTRA_RUNS = [
    '~/benchmark/refcocog_eval_11models_500_repaired/run_500_semantic_strict',
    '~/benchmark/refcocog_eval_13models_4tasks_500/run_20260918_125802',
    '~/benchmark/qwen3vl_fixed_coords/run_20260919_043051',
]


def resolve(root, model, task):
    """Return (rows, run_used). Canonical root wins; otherwise search EXTRA_RUNS.

    Never falls back to a smoke run: those hold 40 rows and would quietly replace a
    real panel with a toy one, so any candidate under 500 rows is rejected.
    """
    rows = rows_for(root, model, task)
    if rows:
        return rows, root
    for cand in EXTRA_RUNS:
        cand = os.path.expanduser(cand)
        if os.path.abspath(cand) == os.path.abspath(root):
            continue
        r = rows_for(cand, model, task)
        if len(r) >= 500:
            return r, cand
    return [], None


def rows_for(root, model, task):
    p = os.path.join(root, model, 'records.jsonl')
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('task') == task:
                out.append(d)
    return out


def eval_t1(rows):
    """Expression verification: false accept on negatives, balanced accuracy."""
    pos = [r for r in rows if r.get('label_exists') is True]
    neg = [r for r in rows if r.get('label_exists') is False]
    if not pos or not neg:
        return None
    # pred_exists True == model says the expression is supported
    fa = sum(1 for r in neg if r.get('pred_exists') is True) / len(neg)
    tpr = sum(1 for r in pos if r.get('pred_exists') is True) / len(pos)
    bal = (tpr + (1.0 - fa)) / 2.0
    bad = sum(1 for r in rows if not r.get('parse_valid'))
    by = {}
    for h in HT4:
        sub = [r for r in neg if r.get('hallucination_type') == h]
        by[h] = (sum(1 for r in sub if r.get('pred_exists') is True) / len(sub)
                 if sub else None)
    return dict(n_pos=len(pos), n_neg=len(neg), false_accept=fa,
                true_accept=tpr, balanced_acc=bal,
                parse_invalid=bad, parse_invalid_rate=bad / len(rows),
                false_accept_by_type=by,
                boh=st.mean([by[h] for h in HT4[:2]]) if all(by[h] is not None for h in HT4[:2]) else None,
                roh=st.mean([by[h] for h in HT4[2:]]) if all(by[h] is not None for h in HT4[2:]) else None)


def eval_t4(rows, coordfix=False):
    """Joint grounding: caption hallucination, coverage, plus FGR/mIoU/pos-success.

    coordfix=True re-derives every box from raw_output_text with the t2 correction,
    then recomputes IoU against the untouched GT. Rows with no parseable box keep a
    zero IoU and count as a genuine no-output.
    """
    if coordfix:
        import export_cf_joint as CF
        cache = {}
        fixed = []
        for r in rows:
            r = dict(r)
            box = CF.rescale(r, cache)
            r['pred_bbox_xyxy'] = box
            r['iou'] = CF.iou(box, r.get('gt_bbox_xyxy')) if box else 0.0
            r['pred_found'] = bool(box)
            fixed.append(r)
        rows = fixed
    pos = [r for r in rows if r.get('label_exists') is True]
    neg = [r for r in rows if r.get('label_exists') is False]
    if not pos or not neg:
        return None
    drew = lambda r: bool(r.get('pred_found'))
    fgr = sum(1 for r in neg if drew(r)) / len(neg)
    iou = lambda r: (r.get('iou') or 0.0)
    miou = sum(iou(r) for r in pos) / len(pos)
    succ = sum(1 for r in pos if iou(r) >= 0.5) / len(pos)
    # caption_target_hallucination is defined on negatives only
    cth_rows = [r for r in neg if r.get('caption_target_hallucination') is not None]
    cth = (sum(1 for r in cth_rows if r['caption_target_hallucination']) / len(cth_rows)
           if cth_rows else None)
    cov = [r['target_coverage'] for r in rows if r.get('target_coverage') is not None]
    amb = [r['amber_cosine'] for r in rows if r.get('amber_cosine') is not None]
    by = {}
    for h in HT4:
        sub = [r for r in neg if r.get('hallucination_type') == h]
        by[h] = sum(1 for r in sub if drew(r)) / len(sub) if sub else None
    # support judgement also emitted in this task
    fa = None
    jn = [r for r in neg if r.get('pred_exists') is not None]
    if jn:
        fa = sum(1 for r in jn if r['pred_exists'] is True) / len(jn)
    return dict(n_pos=len(pos), n_neg=len(neg), fgr=fgr, pos_miou=miou,
                pos_success=succ, caption_hallu_neg=cth, n_cth=len(cth_rows),
                target_coverage=st.mean(cov) if cov else None,
                amber_cosine=st.mean(amb) if amb else None,
                false_accept=fa, fgr_by_type=by,
                boh=st.mean([by[h] for h in HT4[:2]]) if all(by[h] is not None for h in HT4[:2]) else None,
                roh=st.mean([by[h] for h in HT4[2:]]) if all(by[h] is not None for h in HT4[2:]) else None)


def eval_t3(rows):
    """Pure captioning: object hallucination and target coverage, no query."""
    if not rows:
        return None
    cth = [r for r in rows if r.get('caption_target_hallucination') is not None]
    hallu = (sum(1 for r in cth if r['caption_target_hallucination']) / len(cth)
             if cth else None)
    cov = [r['target_coverage'] for r in rows if r.get('target_coverage') is not None]
    amb = [r['amber_cosine'] for r in rows if r.get('amber_cosine') is not None]
    units = [r.get('caption_target_hallucination_unit_count') or 0 for r in rows]
    return dict(n=len(rows), caption_hallu=hallu, n_scored=len(cth),
                target_coverage=st.mean(cov) if cov else None,
                amber_cosine=st.mean(amb) if amb else None,
                hallu_units_mean=st.mean(units) if units else None)


def show(title, per, cols, fam=True):
    print()
    print('=' * 118)
    print(title)
    print('=' * 118)
    ms = sorted(per, key=lambda m: -(per[m].get(cols[0][1]) or 0))
    hdr = '%-16s %6s' % ('model', 'family')
    for label, _ in cols:
        hdr += ' %10s' % label
    print(hdr)
    print('-' * 118)
    for m in ms:
        line = '%-16s %6s' % (m, 'general' if m in GENERAL else 'RL')
        for _, key in cols:
            v = per[m].get(key)
            line += ' %10s' % ('n/a' if v is None else
                               ('%.4f' % v if key.endswith(('miou', 'cosine', 'coverage'))
                                else '%.2f%%' % (v * 100)))
        print(line)
    print('-' * 118)
    if not fam:
        return
    groups = [('general', [m for m in ms if m in GENERAL]),
              ('RL-adapted', [m for m in ms if m not in GENERAL]),
              ('all %d' % len(ms), ms)]
    for nm, sub in groups:
        if not sub:
            continue
        line = '%-16s %6s' % ('%s (%d)' % (nm, len(sub)), '')
        for _, key in cols:
            vals = [per[m][key] for m in sub if per[m].get(key) is not None]
            v = st.mean(vals) if vals else None
            line += ' %10s' % ('n/a' if v is None else
                               ('%.4f' % v if key.endswith(('miou', 'cosine', 'coverage'))
                                else '%.2f%%' % (v * 100)))
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-out', default='tasks_t1_t4.json')
    a = ap.parse_args()
    canon = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))

    t1, t3, t4, prov = {}, {}, {}, {}
    for m in sorted(canon):
        r1, u1 = resolve(canon[m], m, 't1_discriminative_vqa')
        r3, u3 = resolve(canon[m], m, 't3_pure_caption')
        r4, u4 = resolve(canon[m], m, 't4_caption_grounding')
        v1 = eval_t1(r1) if r1 else None
        v3 = eval_t3(r3) if r3 else None
        v4 = eval_t4(r4, coordfix=m in COORDFIX_T4) if r4 else None
        if r4 and m in COORDFIX_T4:
            raw4 = eval_t4(r4, coordfix=False)
            v4['uncorrected'] = {k: raw4[k] for k in
                                 ('fgr', 'pos_success', 'pos_miou', 'boh', 'roh')}
            v4['coordfix_applied'] = True
        if v1:
            t1[m] = v1
        if v3:
            t3[m] = v3
        if v4:
            t4[m] = v4
        prov[m] = {'t1': u1, 't3': u3, 't4': u4}
        note = '' if u1 == canon[m] else '  (t1 from %s)' % (
            os.path.basename(u1) if u1 else 'MISSING')
        print('[%s] t1=%d t3=%d t4=%d%s' % (m, len(r1), len(r3), len(r4), note),
              flush=True)

    show('TASK t1  expression verification (%d models)' % len(t1), t1,
         [('falseAcc', 'false_accept'), ('trueAcc', 'true_accept'),
          ('balancedAcc', 'balanced_acc'), ('BOH', 'boh'), ('ROH', 'roh'),
          ('parseBad', 'parse_invalid_rate')])
    show('TASK t3  pure captioning, no query (%d models)' % len(t3), t3,
         [('capHallu', 'caption_hallu'), ('coverage', 'target_coverage'),
          ('amberCos', 'amber_cosine')])
    cf = [m for m in t4 if t4[m].get('coordfix_applied')]
    if cf:
        print()
        print('=' * 118)
        print('t4 COORDINATE CORRECTION applied to %s' % ', '.join(sorted(cf)))
        print('=' * 118)
        print('%-14s %-14s %9s %9s %9s' % ('model', 'version', 'FGR', 'posSucc', 'mIoU'))
        for m in sorted(cf):
            u = t4[m]['uncorrected']
            print('%-14s %-14s %8.2f%% %8.2f%% %9.4f'
                  % (m, 'uncorrected', u['fgr'] * 100, u['pos_success'] * 100,
                     u['pos_miou']))
            print('%-14s %-14s %8.2f%% %8.2f%% %9.4f'
                  % ('', 'corrected', t4[m]['fgr'] * 100,
                     t4[m]['pos_success'] * 100, t4[m]['pos_miou']))
        print('\n  Leaving t4 uncorrected reproduces the defect the reviewer raised:')
        print('  a positive-success rate near 5%% with mIoU near 0.11.')

    show('TASK t4  joint grounding (%d models)' % len(t4), t4,
         [('FGR', 'fgr'), ('BOH', 'boh'), ('ROH', 'roh'),
          ('posSucc', 'pos_success'), ('mIoU', 'pos_miou'),
          ('capHallu', 'caption_hallu_neg'), ('coverage', 'target_coverage')])

    print()
    print('=' * 118)
    print('CROSS-TASK: false accept (t1) versus false grounding, paper section 4.1')
    print('=' * 118)
    for nm, sub in (('general', [m for m in t1 if m in GENERAL]),
                    ('RL-adapted', [m for m in t1 if m not in GENERAL])):
        if not sub:
            continue
        print('  %-12s falseAccept %.2f%%   (n=%d models)'
              % (nm, st.mean([t1[m]['false_accept'] for m in sub]) * 100, len(sub)))
    miss = sorted(set(canon) - set(t1))
    if miss:
        print('\n  NOTE: no t1 records anywhere for %s -> t1 panel is %d models.'
              % (', '.join(miss), len(t1)))

    print()
    print('=' * 118)
    print('ACCEPTANCE GATE: paper section 4.1 false-accept contrast')
    print('=' * 118)
    exp = {'general': 29.44, 'RL-adapted': 33.03}
    allok = True
    for nm, sub in (('general', [m for m in t1 if m in GENERAL]),
                    ('RL-adapted', [m for m in t1 if m not in GENERAL])):
        got = st.mean([t1[m]['false_accept'] for m in sub]) * 100
        ok = abs(got - exp[nm]) < 0.005
        allok &= ok
        print('  %-12s %d models  got %.4f%%  paper %.2f%%  %s'
              % (nm, len(sub), got, exp[nm], 'PASS' if ok else 'FAIL'))
    print('\n  %s' % ('both PASS -- t1 reproduces the paper'
                      if allok else 'MISMATCH -- check which run each model came from'))

    # second gate: the corrected t4 must match the independent export_cf_joint run
    t4gate = None
    ref = os.path.join(HERE, 'cf_joint_grounding.json')
    if cf and os.path.exists(ref):
        R = json.load(open(ref))['models']
        print()
        print('=' * 118)
        print('GATE: corrected t4 versus the independent export_cf_joint receipt')
        print('=' * 118)
        t4gate = True
        for m in sorted(cf):
            if m not in R:
                continue
            mine, ref_m = t4[m], R[m]['t4_joint']
            dn = abs(mine['pos_success'] * 500 - ref_m['n_correct'])
            di = abs(mine['pos_miou'] - ref_m['pos_miou'])
            ok = dn < 0.5 and di < 1e-9
            t4gate &= ok
            print('  %-12s n_correct %d vs %d, mIoU %.4f vs %.4f  %s'
                  % (m, round(mine['pos_success'] * 500), ref_m['n_correct'],
                     mine['pos_miou'], ref_m['pos_miou'], 'PASS' if ok else 'FAIL'))
        print('\n  %s' % ('two independent scripts agree bit for bit'
                          if t4gate else 'MISMATCH -- do not use these t4 numbers'))

    json.dump(dict(t1=t1, t3=t3, t4=t4, run_provenance=prov,
                   section41_gate_passed=bool(allok),
                   t4_coordfix_gate_passed=t4gate),
              open(os.path.join(HERE, a.json_out), 'w'),
              indent=2, ensure_ascii=False, default=str)
    print('\nwrote %s' % a.json_out)


if __name__ == '__main__':
    main()
