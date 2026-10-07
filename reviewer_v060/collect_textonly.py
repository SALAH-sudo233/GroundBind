#!/usr/bin/env python3
"""Text-only control (reviewer #4): 去掉真实图像信号，只保留文本 claim，
看 OmniVerifier 是否仍能分 positive/negative。若能，说明存在 lexical artifact。

口径与 collect_nobox.py 完全一致（同一候选 plan、同一 no-box 问句），
唯一区别：喂一张固定中性灰图代替真实图像——模型拿不到任何视觉信号，
只能靠 claim 文本本身。z = true-vs-false logit 差。

label：sid 后缀含 '__' = negative(0)，否则 positive(1)。与 eval_nobox 一致。
"""
import argparse, json, os, sys, time

PROBE = os.path.expanduser('~/SVD/agentic_probe')
IMG_ROOT = os.path.expanduser('~/models/LENS/data/refcoco/train2014')
PROBES = os.path.join(PROBE, 'probe_textroute_all.jsonl')
CANON = os.path.join(PROBE, 'canon_roots_paper.json')

sys.path.insert(0, PROBE)

import torch
from PIL import Image


def parse_bbox(v):
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return [float(x) for x in v]
        except Exception:
            return None
    return None


def img_name(sid):
    """sid 形如 'hallu_000249_COCO_train2014_000000310457' 或带 '__obj' 后缀。
    提取 COCO 文件名基部，补 .jpg（真实文件 COCO_train2014_000000310457.jpg）。"""
    if not sid:
        return None
    key = 'COCO_train2014_'
    i = str(sid).find(key)
    if i < 0:
        return None
    rest = str(sid)[i:]
    # 去掉幻觉类型后缀（__obj/__cooc/__attr/__rel 等）
    j = rest.find('__')
    if j > 0:
        rest = rest[:j]
    # 已带 .jpg 则原样，否则补
    if rest.endswith('.jpg'):
        return rest
    return rest + '.jpg'


class OmniTextOnly:
    """OmniVerifier-7B，no-box 问句 + 固定中性灰图(无视觉信号)。"""

    def __init__(self, path, device, max_side=896, blank=448):
        from transformers import (Qwen2_5_VLForConditionalGeneration,
                                   AutoProcessor)
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
        self.device = device
        # 固定中性灰图：无边、无内容、无框——纯粹占位让多模态管线成立
        self.blank = Image.new('RGB', (blank, blank), (127, 127, 127))
        tok = self.proc.tokenizer
        t, f = [], []
        for s in ('true', ' true', 'True', ' True'):
            t += tok.encode(s, add_special_tokens=False)[:1]
        for s in ('false', ' false', 'False', ' False'):
            f += tok.encode(s, add_special_tokens=False)[:1]
        self.ids_true, self.ids_false = sorted(set(t)), sorted(set(f))

    def question(self, phrase):
        # 与 collect_nobox no-box 问句逐字一致
        region = (f'Claim about the image: "{phrase}".\n')
        tail = 'by the image. '
        return (
            region +
            'Carefully analyze the image and determine whether the object, its '
            'attributes, and its spatial relationships stated in the claim are '
            'all correctly satisfied ' + tail +
            'If the claim is accurate, answer true; otherwise false.\n'
            'Respond strictly in JSON: {"answer": true/false}')

    @torch.no_grad()
    def score(self, phrase):
        msgs = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text',
                                'text': self.question(phrase)}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        text = text + '{"answer":'
        inputs = self.proc(text=[text], images=[self.blank],
                           return_tensors='pt').to(self.device)
        logits = self.model(**inputs).logits[0, -1].float()
        zt = torch.logsumexp(logits[self.ids_true], 0).item()
        zf = torch.logsumexp(logits[self.ids_false], 0).item()
        return zt - zf


def load_plan():
    """复用 collect_nobox 同一候选集，保证可比。"""
    query = {}
    for line in open(PROBES, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('error') or r.get('z') is None:
            continue
        key = (r['model'], r['sid'])
        if key not in query:
            query[key] = r['query']

    canon = json.load(open(CANON))
    box = {}
    for m in sorted(canon):
        rp = os.path.join(canon[m], m, 'records.jsonl')
        if not os.path.exists(rp):
            continue
        for line in open(rp, encoding='utf-8'):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if str(rec.get('task', '')).lower() not in ('t2', 't2_vqa_grounding'):
                continue
            sid = rec.get('sample_id') or rec.get('base_sample_id')
            bb = parse_bbox(rec.get('pred_bbox_xyxy'))
            if bb is not None:
                box[(m, sid)] = bb

    plan, skip = [], 0
    for key, q in sorted(query.items()):
        m, sid = key
        bb = box.get(key)
        own = img_name(sid)
        if bb is None or own is None:
            skip += 1
            continue
        # text-only 不读真实图像，但保持与 nobox 相同的候选筛选条件（图像须存在）
        p = os.path.join(IMG_ROOT, own)
        if not os.path.exists(p):
            skip += 1
            continue
        plan.append(dict(model=m, sid=sid, query=q, own_image=own))
    print(f'plan={len(plan)} skipped={skip}', flush=True)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=8)
    ap.add_argument('--omni_path',
                    default=os.path.expanduser('~/models/OmniVerifier-7B'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    plan = load_plan()
    plan = [p for i, p in enumerate(plan) if i % a.nshard == a.shard]
    if a.limit:
        plan = plan[:a.limit]
    print(f'shard {a.shard}/{a.nshard}: {len(plan)} rows', flush=True)

    dev = 'cuda:0'
    omni = OmniTextOnly(a.omni_path, dev)
    print('omni loaded', flush=True)

    out = a.out or os.path.join(PROBE, f'textonly_shard{a.shard}.jsonl')
    n = 0
    t0 = time.time()
    with open(out, 'w') as f:
        for p in plan:
            try:
                z_text = omni.score(p['query'])
            except Exception as e:
                f.write(json.dumps(dict(model=p['model'], sid=p['sid'],
                                        error=str(e))) + '\n')
                continue
            f.write(json.dumps(dict(
                model=p['model'], sid=p['sid'], query=p['query'],
                own_image=p['own_image'], z_text=z_text, error=None)) + '\n')
            n += 1
            if n % 200 == 0:
                f.flush()
                print(f'  {n} done, {time.time()-t0:.0f}s', flush=True)
    print(f'shard {a.shard} ALLDONE n={n} {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
