#!/usr/bin/env python3
"""Export the OTHER THREE unfiltered CBR values for the two coordinate-corrected models.

The paper's Table 2 / Figure 3 / Table A5 show relation CBR for UniVG-R1 (55.1) and
Visual-RFT (37.9) on corrected coordinates but leave object / co_occurrence /
attribute as em-dashes. This exports those three, on the SAME corrected candidates.

CONTRACT reused verbatim from eval_cbr_paper_aligned.py (read, not reinvented):
  elig  = {base | valid_box(pos.pred) and iou(pos.pred, pos.gt) >= 0.5}     theta=0.5
  r_i,t = 1[iou(neg.pred, POSITIVE PREDICTED box) >= 0.8]                   rho=0.8
  CBR_t = sum_i r_i,t / len(elig)   -- denominator is the shared eligible count,
          and the source asserts all four types share it
  unfiltered = no keepmap (idx=None): nothing is rejected, this is raw upstream

VERIFICATION GATE: relation CBR must reproduce the published 55.1 / 37.9 bit-for-bit.
If it does not, the export is wrong and nothing here may be used. No extra conditions
are added (no degenerate-box exclusion, no denominator trimming) -- that discipline
exists because inventing a "more rigorous" filter previously broke agreement with the
paper.
"""
import json
import os
import sys

PROBE = os.path.expanduser('~/SVD/agentic_probe')
sys.path.insert(0, PROBE)
import eval_cbr_paper_aligned as A

CF = ('UniVG-R1', 'visual-rft')
PUBLISHED_REL = {'UniVG-R1': 55.1, 'visual-rft': 37.9}   # Table 2, corrected coords


def main():
    cfroots = json.load(open(os.path.join(PROBE, 'canon_roots_coordfix.json')))
    paper = json.load(open(os.path.join(PROBE, 'canon_roots_paper.json')))

    out = {}
    print('CBR contract: theta=0.5, rho=0.8, denominator=len(elig), unfiltered')
    print('root(coordfix) =', cfroots.get(CF[0]))
    print()
    print('%-12s%7s%10s%14s%12s%12s%10s' %
          ('model', 'n_c', 'object', 'co_occurrence', 'attribute', 'relation', 'check'))

    for m in CF:
        root = cfroots[m]
        pos, neg = A.load_boxes(m, root)
        elig = [b for b, p in pos.items()
                if A.valid_box(p['pred']) and A.iou(p['pred'], p['gt']) >= 0.5]
        vals = A.cbr(elig, pos, neg, {}, None)      # idx=None -> unfiltered
        v = [x * 100 for x in vals]
        rel = v[3]
        ok = abs(rel - PUBLISHED_REL[m]) <= 0.06
        print('%-12s%7d%9.1f%%%13.1f%%%11.1f%%%11.1f%%%10s' %
              (m, len(elig), v[0], v[1], v[2], v[3],
               'OK' if ok else 'MISMATCH'))
        if not ok:
            print('   !! relation CBR %.2f != published %.1f -- export INVALID'
                  % (rel, PUBLISHED_REL[m]))
        out[m] = dict(n_c=len(elig), root=root,
                      cbr={'object': v[0], 'co_occurrence': v[1],
                           'attribute': v[2], 'relation': v[3]},
                      published_relation=PUBLISHED_REL[m],
                      reproduces_published_relation=bool(ok))

    # same numbers on the ORIGINAL-coordinate run, to show what changed and to prove
    # the corrected export is not silently reading the old root
    print('\nfor contrast, the SAME computation on the original-coordinate run:')
    print('%-12s%7s%10s%14s%12s%12s' %
          ('model', 'n_c', 'object', 'co_occurrence', 'attribute', 'relation'))
    for m in CF:
        pos, neg = A.load_boxes(m, paper[m])
        elig = [b for b, p in pos.items()
                if A.valid_box(p['pred']) and A.iou(p['pred'], p['gt']) >= 0.5]
        v = [x * 100 for x in A.cbr(elig, pos, neg, {}, None)]
        print('%-12s%7d%9.1f%%%13.1f%%%11.1f%%%11.1f%%'
              % (m, len(elig), v[0], v[1], v[2], v[3]))
        out[m]['original_coords'] = dict(
            n_c=len(elig), root=paper[m],
            cbr={'object': v[0], 'co_occurrence': v[1],
                 'attribute': v[2], 'relation': v[3]})

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cf_cbr_4types.json')
    json.dump(out, open(p, 'w'), indent=2, ensure_ascii=False)
    print('\nwrote', os.path.basename(p))


if __name__ == '__main__':
    main()
