#!/usr/bin/env python3
"""Paper-level evaluation of the competitive-probe framework on 13 upstream VLMs.

Metric contract (locked to the project's aligned口径, see vsight-experiments skill):
  FGR / FG@Neg = #{negatives whose box survives} / ALL 2000 negatives per model.
                 "has a box" is the OFFICIAL pred_found semantics: degenerate
                 boxes are NOT excluded (eval_11models L2712/L2722).
  pos_mIoU     = sum(IoU_i * keep_i) / 500. Rejected / no-box count as ZERO and
                 are never removed from the denominator.
  folds        = md5(image basename) % 2, image-level (adaptive_v2 convention).
                 Thresholds & weights fitted on fold A, fold B read ONCE.
  AUROC        = rank-sum (ties -> averaged ranks; constant input -> 0.5).
  CIs          = paired bootstrap clustered on SOURCE IMAGE (never on
                 model x sample, which would fake independence), plus a
                 model-level paired test over n=13. BOTH are reported; a pooled
                 effect with inconsistent model-level signs is not a claim.

Arms:
  B0        upstream unfiltered (reference point, no verifier call)
  B1        Omni single arm, threshold on z0                     [1 fwd/row]
  A2        B1 + competitive-probe gap on relation rows only     [+1.2 fwd on relation]
  A2lex     A2 + predicate prior (exploratory upper bound; the prior is a
            benchmark-distribution artefact and may not transfer)

Scope: probes fire ONLY on relation rows (per-htype audit showed the gap weight
flips sign on object/co_occurrence/attribute and A2-B1 is n.s. there).
"""
import argparse, hashlib, json, math, os, random, sys
from collections import Counter, defaultdict

S5 = '/home/u2025141034/SVD/grpo_verifier/s5grpo'
CANON = '/home/u2025141034/SVD/agentic_probe/canon_roots.json'
PROBES = '/home/u2025141034/SVD/agentic_probe/probe_upstream.jsonl'
BOH = {'object', 'co_occurrence'}
ROH = {'attribute', 'relation'}
HT4 = ['object', 'co_occurrence', 'attribute', 'relation']


# ------------------------------------------------------------------ metrics
def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    if n == 0:
        return float('nan')
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    npos = sum(l for _, l in pairs)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    spos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (spos - npos * (npos + 1) / 2.0) / (npos * nneg)


def sig(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logreg(X, y, l2=1.0, iters=300):
    k = len(X[0]); n = len(X)
    mu = [sum(r[j] for r in X) / n for j in range(k)]
    sd = [((sum((r[j] - mu[j]) ** 2 for r in X) / n) ** 0.5) or 1.0 for j in range(k)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(k)] for r in X]
    w = [0.0] * k; b = 0.0
    for _ in range(iters):
        g = [0.0] * k; gb = 0.0
        H = [[0.0] * k for _ in range(k)]; Hb = 0.0; Hxb = [0.0] * k
        for zi, yi in zip(Z, y):
            p = sig(sum(w[j] * zi[j] for j in range(k)) + b)
            e = p - yi; wt = max(p * (1 - p), 1e-6)
            for j in range(k):
                g[j] += e * zi[j]; Hxb[j] += wt * zi[j]
                for j2 in range(k):
                    H[j][j2] += wt * zi[j] * zi[j2]
            gb += e; Hb += wt
        for j in range(k):
            g[j] += l2 * w[j]; H[j][j] += l2
        M = [[H[i][j] for j in range(k)] + [Hxb[i]] for i in range(k)]
        M.append([Hxb[j] for j in range(k)] + [Hb + 1e-6])
        rhs = g + [gb]
        for i in range(k + 1):
            piv = M[i][i] if abs(M[i][i]) > 1e-12 else 1e-12
            for j in range(i + 1, k + 1):
                f = M[j][i] / piv
                for c in range(i, k + 1):
                    M[j][c] -= f * M[i][c]
                rhs[j] -= f * rhs[i]
        dl = [0.0] * (k + 1)
        for i in range(k, -1, -1):
            s = rhs[i] - sum(M[i][c] * dl[c] for c in range(i + 1, k + 1))
            dl[i] = s / (M[i][i] if abs(M[i][i]) > 1e-12 else 1e-12)
        st = 1.0
        while st > 1e-4 and max(abs(x * st) for x in dl) > 4:
            st *= 0.5
        for j in range(k):
            w[j] -= dl[j] * st
        b -= dl[k] * st
        if max(abs(x * st) for x in dl) < 1e-9:
            break

    def pred(r):
        return sig(sum(w[j] * ((r[j] - mu[j]) / sd[j]) for j in range(k)) + b)
    return pred, w


def img_of(sid):
    base = str(sid).split('__')[0]
    i = base.find('COCO_')
    return (base[i:] if i >= 0 else base) + '.jpg'


def fold_of(sid):
    return int(hashlib.md5(img_of(sid).encode()).hexdigest(), 16) % 2


def _selftest():
    assert abs(auroc([1, 1, 1, 1], [1, 1, 0, 0]) - 0.5) < 1e-12
    assert abs(auroc([2, 2, 1], [1, 0, 0]) - 0.75) < 1e-12
    assert img_of('hallu_1_COCO_train2014_000000310457__rel') == \
        'COCO_train2014_000000310457.jpg'


# ------------------------------------------------------------------ load
def load_all(models):
    """Join z0 (existing omni arm) with the new probe scores, per model."""
    probes = defaultdict(dict)
    nerr = 0
    if os.path.exists(PROBES):
        for line in open(PROBES, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('error') or r.get('z') is None:
                nerr += 1
                continue
            probes[(r['model'], r['sid'])][r['variant']] = r['z']

    data = {}
    for m in models:
        p = os.path.join(S5, f's5omni_omni_{m}.jsonl')
        if not os.path.exists(p):
            print(f'[skip] {m}: no omni arm')
            continue
        rows = []
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r['sid']
            ht = r.get('htype')
            is_pos = (ht == 'positive')
            rv = probes.get((m, sid), {})
            zs = [v for v in rv.values() if v is not None]
            rows.append(dict(
                model=m, sid=sid, htype=ht, is_pos=is_pos,
                label=1 if is_pos else 0,
                iou=float(r.get('iou') or 0.0),
                drew=bool(r.get('drew')),
                z0=r.get('z_omni'),
                zmax=max(zs) if zs else None,
                n_rivals=len(zs),
                fold=fold_of(sid),
                img=img_of(sid),
            ))
        data[m] = rows
    return data, nerr


def gap(r):
    if r['zmax'] is None or r['z0'] is None:
        return 0.0
    return r['zmax'] - r['z0']


def probed(r):
    return 1.0 if (r['zmax'] is not None and r['z0'] is not None) else 0.0


# ------------------------------------------------------------------ arms
def fit_arms(A):
    """Fit on fold A rows that HAVE a box (only those get a verifier score)."""
    Ad = [r for r in A if r['drew'] and r['z0'] is not None]
    y = [r['label'] for r in Ad]
    out = {}
    f, w = logreg([[r['z0']] for r in Ad], y)
    out['B1'] = (lambda r: f([r['z0']]), w)
    f2, w2 = logreg([[r['z0'], gap(r), probed(r)] for r in Ad], y)
    out['A2'] = (lambda r: f2([r['z0'], gap(r), probed(r)]), w2)
    return out


def pick_tau(A, scorer, target_pos_keep):
    """Threshold chosen on fold A to hit a target positive-keep rate."""
    pos = [scorer(r) for r in A
           if r['is_pos'] and r['drew'] and r['z0'] is not None]
    if not pos:
        return float('-inf')
    pos.sort()
    k = int(round((1.0 - target_pos_keep) * len(pos)))
    k = max(0, min(k, len(pos) - 1))
    return pos[k]


def evaluate(rows, scorer, tau):
    """Official FGR + pos_mIoU with locked denominators (500 / 2000)."""
    pos = [r for r in rows if r['is_pos']]
    neg = [r for r in rows if not r['is_pos']]
    npos, nneg = 500.0, 2000.0

    def keep(r):
        if not r['drew']:
            return False
        if r['z0'] is None:
            return True           # no verdict -> conservative keep
        return scorer(r) >= tau

    fg = sum(1 for r in neg if keep(r))
    base_fg = sum(1 for r in neg if r['drew'])
    miou = sum(r['iou'] for r in pos if keep(r)) / npos
    base_miou = sum(r['iou'] for r in pos) / npos
    pk = sum(1 for r in pos if keep(r)) / npos
    by_ht = {}
    for ht in HT4:
        sub = [r for r in neg if r['htype'] == ht]
        k = sum(1 for r in sub if keep(r))
        b = sum(1 for r in sub if r['drew'])
        by_ht[ht] = dict(fgr=k / 500.0, base=b / 500.0,
                         catch=(b - k) / b if b else float('nan'))
    return dict(fgr=fg / nneg, base_fgr=base_fg / nneg,
                pos_miou=miou, base_miou=base_miou,
                miou_loss=base_miou - miou, pos_keep=pk,
                boh_fgr=sum(by_ht[h]['fgr'] for h in BOH) / 2,
                roh_fgr=sum(by_ht[h]['fgr'] for h in ROH) / 2,
                by_htype=by_ht)


# ------------------------------------------------------------------ stats
def image_cluster_boot(pairs, n=2000, seed=0):
    """pairs = [(image, value)]; resample IMAGES, return mean + 95% CI."""
    rnd = random.Random(seed)
    byimg = defaultdict(list)
    for img, v in pairs:
        byimg[img].append(v)
    keys = list(byimg)
    if not keys:
        return float('nan'), float('nan'), float('nan')
    out = []
    for _ in range(n):
        acc = []
        for _ in range(len(keys)):
            acc += byimg[rnd.choice(keys)]
        out.append(sum(acc) / len(acc))
    out.sort()
    return (sum(out) / len(out), out[int(.025 * len(out))],
            out[int(.975 * len(out))])


def model_level_paired(deltas, n=2000, seed=0):
    """Paired test over models (n=13). Equal weight per upstream model."""
    rnd = random.Random(seed)
    if not deltas:
        return float('nan'), float('nan'), float('nan')
    out = []
    for _ in range(n):
        s = [rnd.choice(deltas) for _ in deltas]
        out.append(sum(s) / len(s))
    out.sort()
    return (sum(deltas) / len(deltas), out[int(.025 * len(out))],
            out[int(.975 * len(out))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='')
    ap.add_argument('--pos-keep', type=float, default=0.95,
                    help='target positive-keep rate, chosen on fold A')
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()
    _selftest()
    print('metric self-test PASSED (ties -> 0.5)\n')

    canon = json.load(open(CANON))
    models = [x.strip() for x in a.models.split(',') if x.strip()] or sorted(canon)
    data, nerr = load_all(models)
    print(f'models={len(data)}  dropped probe errors={nerr}')

    # ---- denominator audit
    print(f"\n{'model':18s}{'rows':>6}{'pos':>5}{'neg':>6}{'drew_neg':>9}"
          f"{'probed':>8}{'rel_probed':>11}")
    for m in models:
        rows = data.get(m)
        if not rows:
            continue
        pos = [r for r in rows if r['is_pos']]
        neg = [r for r in rows if not r['is_pos']]
        npr = sum(1 for r in rows if probed(r))
        nrel = sum(1 for r in rows if probed(r) and r['htype'] == 'relation')
        flag = '' if (len(pos) == 500 and len(neg) == 2000) else '  <-- OFF-SPEC'
        print(f'{m:18s}{len(rows):>6}{len(pos):>5}{len(neg):>6}'
              f"{sum(1 for r in neg if r['drew']):>9}{npr:>8}{nrel:>11}{flag}")

    # ---- per model: fit on fold A, read fold B once
    per = {}
    for m in models:
        rows = data.get(m)
        if not rows:
            continue
        A = [r for r in rows if r['fold'] == 0]
        B = [r for r in rows if r['fold'] == 1]
        arms = fit_arms(A)
        res = {}
        # B0 = upstream unfiltered, on fold B rows
        res['B0'] = evaluate(B, lambda r: 1e9, float('-inf'))
        for name in ('B1', 'A2'):
            scorer, w = arms[name]
            tau = pick_tau(A, scorer, a.pos_keep)
            res[name] = evaluate(B, scorer, tau)
            res[name]['tau'] = tau
            res[name]['weights'] = [round(x, 4) for x in w]
        per[m] = dict(res=res, B=B)

    # ---- main table
    print(f'\n{"="*104}')
    print(f'主表  fold B 样本外，阈值在 fold A 选 (pos_keep 目标 {a.pos_keep:.2f})')
    print(f'FGR 分母=2000 全部负例(官方 pred_found)  pos_mIoU 分母=500(拒绝计零)')
    print(f'{"="*104}')
    hdr = (f'{"model":18s}{"arm":6s}{"FGR":>8}{"BOH":>8}{"ROH":>8}'
           f'{"relFGR":>9}{"posmIoU":>9}{"ΔmIoU":>8}{"posKeep":>9}')
    print(hdr)
    for m in models:
        if m not in per:
            continue
        for arm in ('B0', 'B1', 'A2'):
            r = per[m]['res'][arm]
            print(f'{m if arm=="B0" else "":18s}{arm:6s}'
                  f'{r["fgr"]*100:>7.1f}%{r["boh_fgr"]*100:>7.1f}%'
                  f'{r["roh_fgr"]*100:>7.1f}%'
                  f'{r["by_htype"]["relation"]["fgr"]*100:>8.1f}%'
                  f'{r["pos_miou"]:>9.4f}{r["miou_loss"]:>8.4f}'
                  f'{r["pos_keep"]:>9.3f}')

    # ---- pooled means
    print(f'\n{"="*104}')
    print('池化(13 模型等权平均) + 两层统计')
    print(f'{"="*104}')
    def mean(arm, key, sub=None):
        vals = []
        for m in per:
            r = per[m]['res'][arm]
            vals.append(r['by_htype'][sub]['fgr'] if sub else r[key])
        return sum(vals) / len(vals)

    print(f'{"arm":8s}{"FGR":>9}{"BOH":>9}{"ROH":>9}{"relFGR":>9}'
          f'{"posmIoU":>10}{"ΔmIoU":>9}')
    for arm in ('B0', 'B1', 'A2'):
        print(f'{arm:8s}{mean(arm,"fgr")*100:>8.2f}%{mean(arm,"boh_fgr")*100:>8.2f}%'
              f'{mean(arm,"roh_fgr")*100:>8.2f}%'
              f'{mean(arm,None,"relation")*100:>8.2f}%'
              f'{mean(arm,"pos_miou"):>10.4f}{mean(arm,"miou_loss"):>9.4f}')

    # relation FGR: A2 vs B1, both levels
    d_model = [per[m]['res']['A2']['by_htype']['relation']['fgr'] -
               per[m]['res']['B1']['by_htype']['relation']['fgr'] for m in per]
    mm, mlo, mhi = model_level_paired(d_model)
    print(f'\nrelation FGR  A2 − B1')
    print(f'  模型级(n={len(d_model)}, 等权): {mm*100:+.2f}pp '
          f'CI[{mlo*100:+.2f},{mhi*100:+.2f}] '
          f'{"显著" if (mlo>0 or mhi<0) else "不显著"}')
    neg_signs = sum(1 for d in d_model if d < 0)
    print(f'  下降的模型数: {neg_signs}/{len(d_model)}')

    # image-clustered pooled: per (model,image) relation negatives
    pairs = []
    for m in per:
        arms = fit_arms([r for r in data[m] if r['fold'] == 0])
        sB1, _ = arms['B1']; sA2, _ = arms['A2']
        tB1 = per[m]['res']['B1']['tau']; tA2 = per[m]['res']['A2']['tau']
        for r in per[m]['B']:
            if r['is_pos'] or r['htype'] != 'relation':
                continue
            k1 = r['drew'] and (r['z0'] is None or sB1(r) >= tB1)
            k2 = r['drew'] and (r['z0'] is None or sA2(r) >= tA2)
            pairs.append((r['img'], (1 if k2 else 0) - (1 if k1 else 0)))
    pm, plo, phi = image_cluster_boot(pairs)
    print(f'  池化(图像聚类, n_obs={len(pairs)}): {pm*100:+.2f}pp '
          f'CI[{plo*100:+.2f},{phi*100:+.2f}] '
          f'{"显著" if (plo>0 or phi<0) else "不显著"}')

    # ---- cost
    print(f'\n{"="*104}')
    print('成本（每个已出框行的验证器前向次数）')
    print(f'{"="*104}')
    tot_drew = tot_probe = 0
    for m in per:
        for r in data[m]:
            if r['drew'] and r['z0'] is not None:
                tot_drew += 1
                tot_probe += r['n_rivals']
    print(f'  B1: 1.000 fwd/row')
    print(f'  A2: {1 + tot_probe/tot_drew:.3f} fwd/row '
          f'(+{tot_probe/tot_drew:.3f} 探针, 只在 relation 触发)')
    print(f'  探针调用总数 {tot_probe} / 已打分行 {tot_drew}')

    if a.json_out:
        out = {m: {k: {kk: vv for kk, vv in v.items() if kk != 'by_htype'}
                   | {'by_htype': v['by_htype']}
                   for k, v in per[m]['res'].items()} for m in per}
        json.dump({'per_model': out,
                   'relation_delta_model_level':
                       {'mean': mm, 'ci': [mlo, mhi], 'n': len(d_model),
                        'n_improved': neg_signs},
                   'relation_delta_pooled_image_clustered':
                       {'mean': pm, 'ci': [plo, phi], 'n_obs': len(pairs)},
                   'pos_keep_target': a.pos_keep},
                  open(a.json_out, 'w'), indent=2, ensure_ascii=False)
        print(f'\nwrote {a.json_out}')


if __name__ == '__main__':
    main()
