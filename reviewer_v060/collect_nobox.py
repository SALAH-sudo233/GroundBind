#!/usr/bin/env python3
"""Q5 补充消融：去掉红框，验证器还用候选区域吗？

机制(s5_omni_filter.draw_box): OmniVerifier 在候选 bbox 处画红框，问
"the red-boxed region" 是否满足 claim。

本消融保持**图像、查询文本完全不变**，只去掉红框(no_box)，并把问题里的
"red-boxed region" 改为中性的 "image"。若验证器真用候选区域，去框后它无法
定位到被指称的物体，分数应向 0/负漂移、判别力(AUROC)应塌向 0.5；若它只读
图像+文本先验，去框几乎无影响。

与 wrong-image 消融互补：
  wrong-image: 固定框+文本，换图   -> 验证器是否用**这张图**
  no-box     : 固定图+文本，去框   -> 验证器是否用**这个区域**

label-blind：选样、问句、打分全程不看标签；这是消融，不是部署动作。
不改 main、不覆盖任何已有结果，单 GPU，输出独立文件 nobox_all.jsonl。

用法 (vlm1):
    CUDA_VISIBLE_DEVICES=0 python3 collect_nobox.py --shard 0 --nshard 8
    (8 卡分片后 cat nobox_shard*.jsonl > nobox_all.jsonl)
"""
import argparse, hashlib, json, os, sys, time

PROBE = os.path.expanduser('~/SVD/agentic_probe')
VERIF = os.path.expanduser('~/SVD/grpo_verifier')
IMG_ROOT = os.path.expanduser('~/models/LENS/data/refcoco/train2014')
PROBES = os.path.join(PROBE, 'probe_textroute_all.jsonl')
CANON = os.path.join(PROBE, 'canon_roots_paper.json')

sys.path.insert(0, PROBE)
sys.path.insert(0, VERIF)

import torch
from PIL import Image, ImageDraw


def parse_bbox(v):
    if not v:
        return None
    try:
        b = [float(x) for x in v]
        if len(b) == 4 and b[2] > b[0] and b[3] > b[1]:
            return b
    except Exception:
        return None
    return None


def img_name(sid):
    base = str(sid).split('__')[0]
    i = base.find('COCO_')
    return base[i:] + '.jpg' if i >= 0 else None


class OmniNoBox:
    """复用 OmniVerifier-7B，但提供 no-box 问句与无框图像。"""

    def __init__(self, path, device, max_side=896):
        from transformers import (Qwen2_5_VLForConditionalGeneration,
                                   AutoProcessor)
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
        self.device = device
        self.max_side = max_side
        tok = self.proc.tokenizer
        t, f = [], []
        for s in ('true', ' true', 'True', ' True'):
            t += tok.encode(s, add_special_tokens=False)[:1]
        for s in ('false', ' false', 'False', ' False'):
            f += tok.encode(s, add_special_tokens=False)[:1]
        self.ids_true, self.ids_false = sorted(set(t)), sorted(set(f))

    def _img_nobox(self, path):
        img = Image.open(path).convert('RGB')
        if max(img.size) > self.max_side:
            s = self.max_side / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)),
                             Image.BICUBIC)
        return img

    def _img_box(self, path, bbox):
        img = Image.open(path).convert('RGB')
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = [float(v) for v in bbox]
        w = max(2, int(round(min(img.size) * 0.008)))
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=w)
        if max(img.size) > self.max_side:
            s = self.max_side / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)),
                             Image.BICUBIC)
        return img

    def question(self, phrase, boxed):
        if boxed:
            region = ('The red rectangle in the image marks a candidate region. '
                      f'Claim about the red-boxed region: "{phrase}".\n')
            tail = 'by the red-boxed region. '
        else:
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
    def score(self, image_path, bbox, phrase, boxed):
        img = self._img_box(image_path, bbox) if boxed else self._img_nobox(image_path)
        msgs = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text',
                                'text': self.question(phrase, boxed)}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        text = text + '{"answer":'
        inputs = self.proc(text=[text], images=[img],
                           return_tensors='pt').to(self.device)
        logits = self.model(**inputs).logits[0, -1].float()
        zt = torch.logsumexp(logits[self.ids_true], 0).item()
        zf = torch.logsumexp(logits[self.ids_false], 0).item()
        return zt - zf


def load_plan():
    """复用 wrong-image 消融的同一候选构造口径(collect_aswap)，保证两消融可比。
    每行取 (model, sid) 的 query(来自 PROBES) + 原框(records) + 图像(IMG_ROOT)。"""
    # rivals: query text per (model, sid)，取第一个合法 query
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
        p = os.path.join(IMG_ROOT, own)
        if not os.path.exists(p):
            skip += 1
            continue
        plan.append(dict(model=m, sid=sid, query=q, own_image=own,
                         bbox=bb, image_path=p))
    print(f'plan={len(plan)} skipped={skip}', flush=True)
    return plan


def resolve_image(plan_row):
    p = plan_row.get('image_path')
    return p if p and os.path.exists(p) else None


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
    omni = OmniNoBox(a.omni_path, dev)
    print('omni loaded', flush=True)

    out = a.out or os.path.join(PROBE, f'nobox_shard{a.shard}.jsonl')
    n = 0
    skip = 0
    t0 = time.time()
    with open(out, 'w') as f:
        for p in plan:
            img = resolve_image(p)
            if not img:
                skip += 1
                continue
            try:
                z_box = omni.score(img, p['bbox'], p['query'], boxed=True)
                z_nobox = omni.score(img, p['bbox'], p['query'], boxed=False)
            except Exception as e:
                f.write(json.dumps(dict(model=p['model'], sid=p['sid'],
                                        error=str(e))) + '\n')
                continue
            f.write(json.dumps(dict(
                model=p['model'], sid=p['sid'], query=p['query'],
                own_image=p['own_image'], z_box=z_box, z_nobox=z_nobox,
                error=None)) + '\n')
            n += 1
            if n % 200 == 0:
                f.flush()
                print(f'  {n} done, {time.time()-t0:.0f}s', flush=True)
    print(f'shard {a.shard} ALLDONE n={n} skip={skip} {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
