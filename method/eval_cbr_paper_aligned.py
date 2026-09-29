#!/usr/bin/env python3
"""CBR aligned to the v0.48 paper口径, for BOTH mitigation lines.

WHAT WAS MISALIGNED. The scoring contract already matched the paper: theta=0.5 for
positive correctness, tau=0.8 for box similarity, IoU=0 for empty/invalid, every
paired negative (refusals included) stays in its type-specific denominator, and
c_i=0 drops the group. Verified numerically -- recomputing the UNFILTERED baseline
over all 500 groups reproduces v0.48 Table 4 (Direct grounding) to rounding on all
13 models, 13-model means object 15.78 / co_occ 26.13 / attribute 33.18 /
relation 40.10 and sum(n_c)=2423.

The divergence was the REPORTING DENOMINATOR. Table 4's n_c counts correctly
localized positives over all 500 groups (Seg-Zero 275, Seg-R1 261, LENS 248...),
while the mitigation run could only report fold B (151, 146, 133...) because the
decision threshold is fitted on fold A. Quoting a fold-B CBR next to the paper's
full-500 CBR silently compares different denominators.

WHAT THIS SCRIPT REPORTS. Three blocks, so nothing is conflated:

  [1] PAPER-ALIGNED, full 500 groups, B0 only.
      Reproduces Table 4 and prints the per-model delta vs the paper values as an
      alignment receipt. No mitigation number appears here: any arm with a fitted
      threshold cannot be evaluated on fold A without training on it.

  [2] PAPER-ALIGNED DENOMINATOR, full 500 groups, mitigated.
      Uses cross-fitting so no group is ever scored by a threshold fitted on
      itself: fold A rows are decided by the fold-B-fitted head and vice versa.
      This recovers the paper's n_c while keeping the out-of-sample discipline.
      Eligibility c_i and the positive reference box stay FROZEN on the
      unfiltered positive prediction, exactly as in the paper.

  [3] SINGLE-FOLD (fold B only), mitigated -- the statistics block.
      Same numbers as the earlier report, kept because the paired CIs and the
      "which models improved" counts were computed on it.

Blocks [2] and [3] answer different questions and must not be averaged together.
Both are printed with their n_c so the denominator is always visible.
"""
import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_upstream as E

S5 = os.path.expanduser('~/SVD/grpo_verifier/s5grpo')
HT4 = ['object', 'co_occurrence', 'attribute', 'relation']

# The three mitigation lines. Each must use ONE scorer for both z0 and the probe
# rivals; mixing scorers makes `gap` meaningless (measured: two different 2B
# readings correlate at only r=0.32, see collect_r1_probe.py docstring).
#   omni     OmniVerifier-7B, zero-training, red rectangle
#   r1       Qwen3.5-2B + GRPO ckpt-400, forced-decision logit margin (binary-ish)
#   jevhead  jev_verifier_v2 full-layer LoRA + scalar head, temperature-calibrated;
#            the only arm with a genuinely graded score (39.4% of p in .05-.95)
LINES = ('omni', 'r1', 'jevhead')

# v0.48 Table 4, Direct grounding block: model -> (n_c, object, co_occ, attr, rel)
PAPER_T4_DG = {
    'UniVG-R1': (30, 33.3, 46.7, 40.0, 46.7),
    'visual-rft': (26, 0.0, 11.5, 19.2, 11.5),
    'Seg-zero': (275, 41.5, 47.3, 60.7, 64.0),
    'Seg-R1': (261, 22.2, 36.4, 46.7, 58.6),
    'VisionReasoner': (253, 34.0, 48.2, 53.0, 63.2),
    'LENS': (248, 48.8, 51.6, 68.5, 67.3),
    'qwen2.5-vl-7b': (242, 4.5, 18.6, 26.9, 44.6),
    'Qwen3-VL-8B': (238, 2.1, 10.9, 19.3, 33.2),
    'InternVL3.5-8B': (234, 7.7, 23.5, 45.7, 43.6),
    'llava-ov-7b': (186, 4.3, 18.3, 18.3, 42.5),
    'Orsta-7B': (151, 0.7, 8.6, 11.3, 15.9),
    'Vision-R1': (148, 1.4, 7.4, 8.8, 14.2),
    'TreeVGR': (131, 4.6, 10.7, 13.0, 16.0),
}


def parse_bbox(v):
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        b = [float(x) for x in v]
    except Exception:
        return None
    if any(x != x for x in b):
        return None
    return b


def valid_box(b):
    if b is None:
        return False
    if all(abs(x) < 1e-9 for x in b):
        return False
    return (b[2] - b[0]) > 0 and (b[3] - b[1]) > 0


def iou(a, b):
    if not valid_box(a) or not valid_box(b):
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _selftest():
    assert abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert iou([0, 0, 10, 10], [5, 5, 5, 9]) == 0.0      # degenerate -> no reuse
    assert not valid_box([0, 0, 0, 0])
    assert abs(E.auroc([1, 1, 1, 1], [1, 1, 0, 0]) - 0.5) < 1e-12


def load_boxes(model, root):
    rp = os.path.join(root, model, 'records.jsonl')
    pos, neg = {}, defaultdict(dict)
    for line in open(rp, encoding='utf-8'):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if str(r.get('task', '')) != 't2_vqa_grounding':
            continue
        sid = r.get('sample_id') or ''
        base = str(sid).split('__')[0]
        bb = parse_bbox(r.get('pred_bbox_xyxy'))
        if r.get('query_role') == 'positive':
            pos[base] = dict(pred=bb, gt=parse_bbox(r.get('gt_bbox_xyxy')), sid=sid)
        else:
            ht = r.get('hallucination_type')
            if ht in HT4:
                neg[base][ht] = dict(pred=bb, sid=sid)
    return pos, neg


# Models re-collected against the paper's run. Their arm files carry a distinct tag
# so the original 13mod files are never mixed in, and their rivals come from the
# matching rival file.
REPROV = {'qwen2.5-vl-7b': dict(omni_arm='s5omni_omni11_%s.jsonl',
                                r1_arm='s5jevz11_q35_2b_%s.jsonl',
                                omni_probe='probe_11rep.jsonl',
                                r1_probe='r1probe_11rep.jsonl',
                                jevhead_probe='jevheadprobe_11rep.jsonl')}


def _jevhead_rows(models):
    """jevhead z0 comes from one combined file keyed by (model, sid)."""
    p = os.path.join(HERE, 'jevhead_all.jsonl')
    if not os.path.exists(p):
        sys.exit('missing %s -- run collect_jevhead.py first' % p)
    out = defaultdict(list)
    for l in open(p, encoding='utf-8'):
        d = json.loads(l)
        out[d['model']].append(d)
    return out


def load_line(line, models):
    """Rows with that line's own z0 AND its own probe scores (one scorer only)."""
    key = {'omni': 'z', 'r1': 'z_r1', 'jevhead': 'z_head'}[line]

    def read_probe(path, dest):
        if not os.path.exists(path):
            sys.exit('missing %s' % path)
        for l in open(path, encoding='utf-8'):
            d = json.loads(l)
            if d.get('error') or d.get(key) is None:
                continue
            dest[(d['model'], d['sid'])][d['variant']] = d[key]

    MAIN = {'omni': 'probe_upstream.jsonl', 'r1': 'r1probe_all.jsonl',
            'jevhead': 'jevheadprobe_all.jsonl'}
    RE = {'omni': 'omni_probe', 'r1': 'r1_probe', 'jevhead': 'jevhead_probe'}
    probe = defaultdict(dict)
    read_probe(os.path.join(HERE, MAIN[line]), probe)
    # re-provenanced models: drop the old-run rivals, load the matching ones
    for m, spec in REPROV.items():
        for k in [k for k in probe if k[0] == m]:
            del probe[k]
        read_probe(os.path.join(HERE, spec[RE[line]]), probe)

    jh = _jevhead_rows(models) if line == 'jevhead' else None

    data = {}
    for m in models:
        if line == 'jevhead':
            src = jh.get(m) or []
            if not src:
                print('[skip] %s line=jevhead: no rows' % m)
                continue
            rows = []
            for r in src:
                sid = r['sid']
                rv = probe.get((m, sid), {})
                zs = [v for v in rv.values() if v is not None]
                rows.append(dict(
                    model=m, sid=sid, htype=r.get('htype'),
                    is_pos=(r.get('htype') == 'positive'),
                    label=1 if r.get('htype') == 'positive' else 0,
                    iou=float(r.get('iou') or 0.0), drew=bool(r.get('drew')),
                    z0=r.get('z_head'), zmax=(max(zs) if zs else None),
                    fold=E.fold_of(sid)))
            data[m] = rows
            continue
        spec = REPROV.get(m)
        if spec:
            f = spec['omni_arm' if line == 'omni' else 'r1_arm'] % m
        else:
            f = ('s5omni_omni_%s.jsonl' if line == 'omni'
                 else 's5jevz_q35_2b_%s.jsonl') % m
        p = os.path.join(S5, f)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print('[skip] %s line=%s: missing/empty %s' % (m, line, f))
            continue
        rows = []
        for l in open(p, encoding='utf-8'):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            sid = r['sid']
            rv = probe.get((m, sid), {})
            zs = [v for v in rv.values() if v is not None]
            rows.append(dict(
                model=m, sid=sid, htype=r.get('htype'),
                is_pos=(r.get('htype') == 'positive'),
                label=1 if r.get('htype') == 'positive' else 0,
                iou=float(r.get('iou') or 0.0), drew=bool(r.get('drew')),
                z0=(r.get('z_omni') if line == 'omni' else r.get('z_jev')),
                zmax=(max(zs) if zs else None),
                fold=E.fold_of(sid)))
        data[m] = rows
    return data


def gap(r):
    if r['zmax'] is None or r['z0'] is None:
        return 0.0
    return r['zmax'] - r['z0']


def covered(r):
    return r['zmax'] is not None and r['z0'] is not None


def fit_on(rows, pos_keep):
    """Return (keep_b1, keep_a2) decided by a head fitted on THESE rows."""
    Ad = [r for r in rows if r['drew'] and r['z0'] is not None]
    if len(Ad) < 30 or len(set(r['label'] for r in Ad)) < 2:
        return (lambda r: bool(r['drew'])), (lambda r: bool(r['drew']))
    y = [r['label'] for r in Ad]
    f1, _ = E.logreg([[r['z0']] for r in Ad], y)
    s1 = lambda r: f1([r['z0']])
    p1 = sorted(s1(r) for r in Ad if r['is_pos'])
    t1 = p1[max(0, min(int(round((1 - pos_keep) * len(p1))), len(p1) - 1))] \
        if p1 else float('-inf')

    Ac = [r for r in Ad if covered(r)]
    s2 = t2 = None
    if len(Ac) >= 30 and len(set(r['label'] for r in Ac)) > 1:
        f2, _ = E.logreg([[r['z0'], gap(r)] for r in Ac],
                         [r['label'] for r in Ac])
        s2 = lambda r: f2([r['z0'], gap(r)])
        p2 = sorted(s2(r) for r in Ac if r['is_pos'])
        t2 = p2[max(0, min(int(round((1 - pos_keep) * len(p2))), len(p2) - 1))] \
            if p2 else float('-inf')

    def keep_b1(r):
        if not r['drew']:
            return False
        if r['z0'] is None:
            return True
        return s1(r) >= t1

    def keep_a2(r):
        # gate: only probe-covered rows switch heads; others keep the B1 decision
        if not covered(r) or s2 is None:
            return keep_b1(r)
        return s2(r) >= t2

    return keep_b1, keep_a2


def cbr(elig, pos, neg, keepmap, idx):
    """CBR per htype over the given eligible groups. Asserts one shared
    eligibility set across the four types (the paper's contract)."""
    vals, dens = [], {}
    for ht in HT4:
        num = den = 0
        for b in elig:
            nb = neg.get(b, {}).get(ht)
            if not nb:
                continue
            den += 1
            if idx is not None:
                kv = keepmap.get(nb['sid'])
                if kv is not None and not kv[idx]:
                    continue          # refusal -> zero reuse, stays in denominator
            if iou(nb['pred'], pos[b]['pred']) >= 0.8:
                num += 1
        dens[ht] = den
        vals.append(num / len(elig) if elig else float('nan'))
    assert len(set(dens.values())) == 1, 'eligibility differs across htypes: %s' % dens
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos-keep', type=float, default=0.95)
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()
    _selftest()
    print('geometry + AUROC self-tests PASSED\n')

    # PAPER-ALIGNED provenance. qwen2.5-vl-7b is the ONLY model whose canonical run
    # disagreed with v0.48 Table 4 (13mod n_c=247 vs paper 242); its verifier scores
    # were measured on 13mod boxes (500/500 IoU match vs 138/500 for 11rep), so the
    # root could not be repointed without re-scoring. Both arms and the rivals were
    # re-collected against 11rep under the *11 tags.
    canon = json.load(open(os.path.join(HERE, 'canon_roots_paper.json')))
    models = sorted(canon)
    lines = {ln: load_line(ln, models) for ln in LINES}
    common = [m for m in models if all(lines[ln].get(m) for ln in LINES)]
    print('models in BOTH lines: %d/%d' % (len(common), len(models)))
    if len(common) < len(models):
        print('  missing: %s' % ', '.join(m for m in models if m not in common))

    boxes = {m: load_boxes(m, canon[m]) for m in models}

    # ---------------- [1] paper alignment receipt, B0, full 500
    print('\n' + '=' * 104)
    print('[1] 论文口径对齐核验：B0 缓释前，全部 500 组（对应 v0.48 Table 4 Direct grounding）')
    print('=' * 104)
    print('%-17s%6s%6s   %-30s%-30s' % ('model', 'n_c', 'paper', 'recomputed', 'paper'))
    align, mine_all = {}, {}
    for m in models:
        pos, neg = boxes[m]
        elig = [b for b, p in pos.items()
                if valid_box(p['pred']) and iou(p['pred'], p['gt']) >= 0.5]
        v = cbr(elig, pos, neg, {}, None)
        mine_all[m] = (len(elig), v)
        pt = PAPER_T4_DG.get(m)
        align[m] = dict(n_c=len(elig), recomputed=[x * 100 for x in v],
                        paper=(list(pt[1:]) if pt else None),
                        paper_n_c=(pt[0] if pt else None))
        rec = ' '.join('%5.1f' % (x * 100) for x in v)
        pap = ' '.join('%5.1f' % x for x in pt[1:]) if pt else '   --'
        flag = '' if (pt and len(elig) == pt[0]) else '  <-- n_c MISMATCH'
        print('%-17s%6d%6s   %-30s%-30s%s'
              % (m, len(elig), (pt[0] if pt else '--'), rec, pap, flag))
    mm = [st.mean([mine_all[m][1][i] for m in models]) * 100 for i in range(4)]
    pp = [st.mean([PAPER_T4_DG[m][1 + i] for m in PAPER_T4_DG]) for i in range(4)]
    print('%-17s%6d%6d   %-30s%-30s'
          % ('MEAN(13)', sum(mine_all[m][0] for m in models),
             sum(v[0] for v in PAPER_T4_DG.values()),
             ' '.join('%5.2f' % x for x in mm),
             ' '.join('%5.2f' % x for x in pp)))
    print('列顺序：object / co_occurrence / attribute / relation')

    # ---------------- [2] paper denominator + cross-fitted mitigation
    print('\n' + '=' * 104)
    print('[2] 论文口径分母（全 500 组）+ 交叉拟合缓释：A 折用 B 折拟合的头判，反之亦然')
    print('    阈值永不在评价自身的组上拟合；c_i 与正例参考框缓释前冻结')
    print('=' * 104)
    per2 = {ln: {} for ln in LINES}
    for ln in LINES:
        print('--- %s 线 ---' % ln.upper())
        print('%-17s%6s  %s' % ('model', 'n_c',
                                ''.join('%11s' % h[:9] for h in HT4)))
        for m in common:
            rows = lines[ln][m]
            A = [r for r in rows if r['fold'] == 0]
            B = [r for r in rows if r['fold'] == 1]
            kA1, kA2 = fit_on(B, a.pos_keep)      # decide fold A with fold-B head
            kB1, kB2 = fit_on(A, a.pos_keep)      # decide fold B with fold-A head
            keepmap = {}
            for r in rows:
                k1, k2 = (kA1, kA2) if r['fold'] == 0 else (kB1, kB2)
                keepmap[r['sid']] = (k1(r), k2(r))
            pos, neg = boxes[m]
            elig = [b for b, p in pos.items()
                    if valid_box(p['pred']) and iou(p['pred'], p['gt']) >= 0.5]
            res = {}
            for arm, idx in (('B0', None), ('B1', 0), ('A2', 1)):
                res[arm] = cbr(elig, pos, neg, keepmap, idx)
            per2[ln][m] = dict(n_c=len(elig), **res)
            for arm in ('B0', 'B1', 'A2'):
                print('%-17s%6s  %s   %s'
                      % (m if arm == 'B0' else '',
                         (len(elig) if arm == 'B0' else ''),
                         ''.join('%10.1f%%' % (x * 100) for x in res[arm]), arm))
        print()

    print('池化（%d 模型等权，论文分母）' % len(common))
    print('%-12s%-6s%s' % ('line', 'arm', ''.join('%12s' % h[:9] for h in HT4)))
    for ln in LINES:
        for arm in ('B0', 'B1', 'A2'):
            print('%-12s%-6s%s'
                  % (ln.upper() if arm == 'B0' else '', arm,
                     ''.join('%11.1f%%' % (st.mean([per2[ln][m][arm][i]
                                                    for m in common]) * 100)
                             for i in range(4))))

    print('\n模型级配对检验（n=%d 等权，论文分母）' % len(common))
    stat = {}
    for ln in LINES:
        print('  %s 线  A2 − B1' % ln.upper())
        for i, ht in enumerate(HT4):
            d = [per2[ln][m]['A2'][i] - per2[ln][m]['B1'][i] for m in common]
            mu, lo, hi = E.model_level_paired(d)
            print('    %-16s %+7.2fpp CI[%+6.2f,%+6.2f] %-6s 下降 %d/%d'
                  % (ht, mu * 100, lo * 100, hi * 100,
                     '显著' if (lo > 0 or hi < 0) else '不显著',
                     sum(1 for x in d if x < 0), len(d)))
            stat['paperdenom_%s_A2-B1_%s' % (ln, ht)] = dict(mean=mu, ci=[lo, hi])
    print('  线间  R1 − OMNI（同 pos_keep 目标）')
    for arm in ('B1', 'A2'):
        for i, ht in enumerate(HT4):
            d = [per2['r1'][m][arm][i] - per2['omni'][m][arm][i] for m in common]
            mu, lo, hi = E.model_level_paired(d)
            print('    %-4s%-14s %+7.2fpp CI[%+6.2f,%+6.2f] %s'
                  % (arm, ht, mu * 100, lo * 100, hi * 100,
                     '显著' if (lo > 0 or hi < 0) else '不显著'))
            stat['paperdenom_r1-omni_%s_%s' % (arm, ht)] = dict(mean=mu, ci=[lo, hi])

    # ---------------- [3] single-fold block (statistics as previously reported)
    print('\n' + '=' * 104)
    print('[3] 单折口径（仅 fold B，阈值 fold A 拟合）——此前报告的统计块，分母更小')
    print('=' * 104)
    per3 = {ln: {} for ln in LINES}
    for ln in LINES:
        for m in common:
            rows = lines[ln][m]
            A = [r for r in rows if r['fold'] == 0]
            B = [r for r in rows if r['fold'] == 1]
            k1, k2 = fit_on(A, a.pos_keep)
            keepmap = {r['sid']: (k1(r), k2(r)) for r in rows}
            pos, neg = boxes[m]
            foldB = {r['sid'] for r in B}
            elig = [b for b, p in pos.items()
                    if p['sid'] in foldB and valid_box(p['pred'])
                    and iou(p['pred'], p['gt']) >= 0.5]
            per3[ln][m] = dict(n_c=len(elig),
                               **{arm: cbr(elig, pos, neg, keepmap, idx)
                                  for arm, idx in (('B0', None), ('B1', 0), ('A2', 1))})
    print('%-12s%-6s%8s%s' % ('line', 'arm', 'n_c',
                              ''.join('%12s' % h[:9] for h in HT4)))
    for ln in LINES:
        ncs = sum(per3[ln][m]['n_c'] for m in common)
        for arm in ('B0', 'B1', 'A2'):
            print('%-12s%-6s%8s%s'
                  % (ln.upper() if arm == 'B0' else '', arm,
                     (ncs if arm == 'B0' else ''),
                     ''.join('%11.1f%%' % (st.mean([per3[ln][m][arm][i]
                                                    for m in common]) * 100)
                             for i in range(4))))
    for ln in LINES:
        print('  %s 线  A2 − B1（单折）' % ln.upper())
        for i, ht in enumerate(HT4):
            d = [per3[ln][m]['A2'][i] - per3[ln][m]['B1'][i] for m in common]
            mu, lo, hi = E.model_level_paired(d)
            print('    %-16s %+7.2fpp CI[%+6.2f,%+6.2f] %-6s 下降 %d/%d'
                  % (ht, mu * 100, lo * 100, hi * 100,
                     '显著' if (lo > 0 or hi < 0) else '不显著',
                     sum(1 for x in d if x < 0), len(d)))
            stat['foldB_%s_A2-B1_%s' % (ln, ht)] = dict(mean=mu, ci=[lo, hi])

    print('\n注：[2] 与 [3] 回答不同问题、分母不同，不可平均或混排。')

    if a.json_out:
        json.dump(dict(alignment_B0_full500=align,
                       paper_denominator_crossfit=per2,
                       single_fold=per3, comparisons=stat,
                       models=common, pos_keep_target=a.pos_keep),
                  open(a.json_out, 'w'), indent=2, ensure_ascii=False)
        print('wrote %s' % a.json_out)


if __name__ == '__main__':
    main()
